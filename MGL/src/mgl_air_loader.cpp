//------------------------------------------------------------------------------------------------
// mgl_air_loader.cpp — AIR metallib → MTL::Library → PSO（Metal-cpp 实现）
//
// 本 TU 不定义 NS_PRIVATE_IMPLEMENTATION / MTL_PRIVATE_IMPLEMENTATION
// （mgl_render_cpp.cpp 是唯一定义点）；仅 include mgl_metal_cpp.h 拿声明。
//
// Phase 1 范围（METALCPP_RENDERER_PLAN）：metallib 加载 + render/compute PSO
// 创建 + 简易 PSO 缓存。render PSO 的 finalDescriptor 等价装配目前覆盖
// color/depth/stencil/vertexDescriptor(attrib)/rasterization/ICB；blend 与
// 二进制归档 Phase 4 补齐。主路径接入见 MGLRenderer+RenderPass.m 的门控分支。
//------------------------------------------------------------------------------------------------
#include "mgl_metal_cpp.h"
#include "mgl_air_loader.h"

#include <dispatch/dispatch.h>
#include <map>
#include <string>

namespace {

// PSO 缓存：key = 描述符状态序列化摘要。进程退出时 C 的 auto-cleanup
// 可能晚于 C++ 静态析构；因此容器自身保持进程寿命，只在显式 shutdown
// 中释放其 Metal 对象并 clear，避免访问已经析构的 std::map。
using PSOCache = std::map<std::string, void*>;

PSOCache& psoCache() {
    static PSOCache* cache = new PSOCache();
    return *cache;
}

std::string pipelineKey(const void* vs, const void* fs,
                        const MGLPipelineDescriptorState* d) {
    std::string key;
    key.reserve(sizeof(vs) + sizeof(fs) + sizeof(*d));
    key.append(reinterpret_cast<const char*>(&vs), sizeof(vs));
    key.append(reinterpret_cast<const char*>(&fs), sizeof(fs));
    key.append(reinterpret_cast<const char*>(d), sizeof(*d));
    return key;
}

void copyError(NS::Error* e, char* err, size_t errcap) {
    if (!err || errcap == 0) return;
    if (e && e->localizedDescription()) {
        const char* s = e->localizedDescription()->utf8String();
        if (s) {
            snprintf(err, errcap, "%s", s);
            return;
        }
    }
    snprintf(err, errcap, "unknown Metal error");
}

} // namespace

extern "C" {

int mglAirLoadLibrary(const void* device, const unsigned char* bytes, size_t size,
                      void** library_out, char* err, size_t errcap) {
    if (!device || !bytes || size == 0 || !library_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    *library_out = nullptr;
    MTL::Device* dev = static_cast<MTL::Device*>(const_cast<void*>(device));

    // ObjC 侧同款：dispatch_data_create → newLibrary(dispatch_data)。
    dispatch_data_t data = dispatch_data_create(bytes, size, nullptr,
                                                DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (!data) {
        if (err && errcap) snprintf(err, errcap, "dispatch_data_create failed");
        return -1;
    }
    NS::Error* nsErr = nullptr;
    MTL::Library* lib = dev->newLibrary(data, &nsErr);
    dispatch_release(data);
    if (!lib) {
        copyError(nsErr, err, errcap);
        return -1;
    }
    *library_out = lib; // +1 retained，调用方拥有
    return 0;
}

int mglAirCreateRenderPipeline(const void* device, void* vs_function, void* fs_function,
                               const MGLPipelineDescriptorState* desc, void** pso_out,
                               char* err, size_t errcap) {
    if (!device || !vs_function || !desc || !pso_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    MTL::Device* dev = static_cast<MTL::Device*>(const_cast<void*>(device));
    MTL::Function* vsFn = static_cast<MTL::Function*>(vs_function);
    MTL::Function* fsFn = fs_function
        ? static_cast<MTL::Function*>(fs_function) : nullptr;

    *pso_out = nullptr;
    std::string key = pipelineKey(vs_function, fs_function, desc);
    PSOCache& cache = psoCache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        static_cast<MTL::RenderPipelineState*>(it->second)->retain();
        *pso_out = it->second;
        return 0;
    }

    MTL::RenderPipelineDescriptor* rpd = MTL::RenderPipelineDescriptor::alloc()->init();
    if (!rpd) {
        if (err && errcap) snprintf(err, errcap, "descriptor alloc failed");
        return -1;
    }

    rpd->setVertexFunction(vsFn);
    if (fsFn) {
        rpd->setFragmentFunction(fsFn);
    }

    rpd->setRasterizationEnabled(desc->rasterization_enabled ? true : false);
    rpd->setSupportIndirectCommandBuffers(desc->icb_enabled ? true : false);
    rpd->setAlphaToCoverageEnabled(desc->alpha_to_coverage_enabled ? true : false);
    rpd->setAlphaToOneEnabled(desc->alpha_to_one_enabled ? true : false);
    rpd->setInputPrimitiveTopology(
        (MTL::PrimitiveTopologyClass)desc->input_primitive_topology);
    if (desc->raster_sample_count > 0)
        rpd->setRasterSampleCount(desc->raster_sample_count);

    for (uint32_t i = 0; i < desc->color_count && i < 8; i++) {
        MTL::RenderPipelineColorAttachmentDescriptor* ca =
            rpd->colorAttachments()->object(i);
        ca->setPixelFormat((MTL::PixelFormat)desc->color_format[i]);
    }
    rpd->setDepthAttachmentPixelFormat((MTL::PixelFormat)desc->depth_format);
    rpd->setStencilAttachmentPixelFormat((MTL::PixelFormat)desc->stencil_format);

    if (desc->attrib_count > 0) {
        MTL::VertexDescriptor* vd = MTL::VertexDescriptor::alloc()->init();
        for (uint32_t i = 0; i < desc->attrib_count && i < 32; i++) {
            const uint32_t bufIdx = desc->attrib_buffer_index[i];
            vd->attributes()->object(i)->setFormat((MTL::VertexFormat)desc->attrib_format[i]);
            vd->attributes()->object(i)->setOffset(desc->attrib_offset[i]);
            vd->attributes()->object(i)->setBufferIndex(bufIdx);
            vd->layouts()->object(bufIdx)->setStride(desc->attrib_stride[i]);
            vd->layouts()->object(bufIdx)->setStepFunction(
                (MTL::VertexStepFunction)desc->attrib_step_function[i]);
            vd->layouts()->object(bufIdx)->setStepRate(desc->attrib_step_rate[i]);
        }
        rpd->setVertexDescriptor(vd);
        vd->release();
    }

    rpd->setTessellationPartitionMode(
        (MTL::TessellationPartitionMode)desc->tessellation_partition_mode);
    rpd->setMaxTessellationFactor(desc->max_tessellation_factor);
    rpd->setTessellationFactorScaleEnabled(
        desc->tessellation_factor_scale_enabled ? true : false);
    rpd->setTessellationFactorFormat(
        (MTL::TessellationFactorFormat)desc->tessellation_factor_format);
    rpd->setTessellationControlPointIndexType(
        (MTL::TessellationControlPointIndexType)
            desc->tessellation_control_point_index_type);
    rpd->setTessellationFactorStepFunction(
        (MTL::TessellationFactorStepFunction)
            desc->tessellation_factor_step_function);
    rpd->setTessellationOutputWindingOrder(
        (MTL::Winding)desc->tessellation_output_winding_order);

    for (uint32_t i = 0; i < desc->color_count && i < 8; i++) {
        MTL::RenderPipelineColorAttachmentDescriptor* ca =
            rpd->colorAttachments()->object(i);
        ca->setWriteMask((MTL::ColorWriteMask)desc->color_write_mask[i]);
        if (desc->blending_enabled_mask & (1u << i)) {
            ca->setBlendingEnabled(true);
            ca->setSourceRGBBlendFactor((MTL::BlendFactor)desc->source_rgb_blend_factor[i]);
            ca->setDestinationRGBBlendFactor((MTL::BlendFactor)desc->destination_rgb_blend_factor[i]);
            ca->setSourceAlphaBlendFactor((MTL::BlendFactor)desc->source_alpha_blend_factor[i]);
            ca->setDestinationAlphaBlendFactor((MTL::BlendFactor)desc->destination_alpha_blend_factor[i]);
            ca->setRgbBlendOperation((MTL::BlendOperation)desc->rgb_blend_operation[i]);
            ca->setAlphaBlendOperation((MTL::BlendOperation)desc->alpha_blend_operation[i]);
        }
    }

    NS::Error* nsErr = nullptr;
    MTL::RenderPipelineState* pso = dev->newRenderPipelineState(rpd, &nsErr);
    rpd->release();
    if (!pso) {
        copyError(nsErr, err, errcap);
        return -1;
    }

    pso->retain(); // 缓存长期持有
    cache[key] = pso;
    *pso_out = pso; // 调用方持有一份引用（mglAirRelease）
    return 0;
}

int mglAirCreateComputePipeline(const void* device, void* library,
                                void** pso_out, char* err, size_t errcap) {
    if (!device || !library || !pso_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    *pso_out = nullptr;
    MTL::Device* dev = static_cast<MTL::Device*>(const_cast<void*>(device));
    MTL::Library* lib = static_cast<MTL::Library*>(library);
    MTL::Function* fn = lib->newFunction(NS::String::string("main", NS::UTF8StringEncoding));
    if (!fn) {
        if (err && errcap) snprintf(err, errcap, "compute function 'main' not found");
        return -1;
    }
    NS::Error* nsErr = nullptr;
    MTL::ComputePipelineState* pso = dev->newComputePipelineState(fn, &nsErr);
    fn->release();
    if (!pso) {
        copyError(nsErr, err, errcap);
        return -1;
    }
    *pso_out = pso; // +1 retained（compute PSO 暂不进缓存，Phase 4 并入）
    return 0;
}

void mglAirRelease(void* obj) {
    if (obj) {
        static_cast<NS::Object*>(obj)->release();
    }
}

void mglAirLoaderShutdown(void) {
    PSOCache& cache = psoCache();
    for (auto &entry : cache) {
        if (entry.second) {
            static_cast<MTL::RenderPipelineState*>(entry.second)->release();
        }
    }
    cache.clear();
}

} // extern "C"
