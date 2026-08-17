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
#include "mgl_env_flag.h"

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
                        const MGLRenderCppPipelineDescriptorState* d) {
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

// MTL::PixelFormat packed depth-stencil predicate（镜像
// mgl_texture_compat.h 的 mglMetalPixelFormatIsPackedDepthStencil）。
bool isPackedDepthStencil(uint32_t format) {
    return format == static_cast<uint32_t>(MTL::PixelFormatDepth24Unorm_Stencil8) ||
           format == static_cast<uint32_t>(MTL::PixelFormatDepth32Float_Stencil8);
}

// P4.2: 在 value-state 上复刻 ObjC mglNormalizePipelineDepthStencilFormats：
// depth/stencil 各占独立 attachment 但其中之一是 packed 格式时，Metal 要求
// 两者使用同一个 packed 格式（depth 与 stencil 共享纹理）。
void normalizeDepthStencilFormats(MGLRenderCppPipelineDescriptorState* desc) {
    uint32_t depth = desc->depth_format;
    uint32_t stencil = desc->stencil_format;
    if (depth == static_cast<uint32_t>(MTL::PixelFormatInvalid) ||
        stencil == static_cast<uint32_t>(MTL::PixelFormatInvalid) ||
        depth == stencil) {
        return;
    }
    const bool depthPacked = isPackedDepthStencil(depth);
    const bool stencilPacked = isPackedDepthStencil(stencil);
    if (!depthPacked && !stencilPacked) {
        return;
    }
    const uint32_t packed = stencilPacked ? stencil : depth;
    desc->depth_format = packed;
    desc->stencil_format = packed;
}

// P4.2: 由 value-state 组装 MTL::RenderPipelineDescriptor（final/simple/safe
// 共用）。镜像 renderer pipeline descriptor state +
// bindBlendStateToPipelineStateDescriptor + mglEnableIndirectCommandBuffersForPipeline：
//   - label "GLSL Pipeline"
//   - color attachment 的 writeMask/blend 只在 pixelFormat 有效时设置
//     （未触碰的 attachment 保持 Metal 默认值，与 ObjC descriptor 一致）
//   - supportIndirectCommandBuffers 由 MGL_ENABLE_ICB_PIPELINES 显式 opt-in
// 调用方负责先 normalizeDepthStencilFormats。
MTL::RenderPipelineDescriptor* buildRenderPipelineDescriptor(
    const MGLRenderCppPipelineDescriptorState* desc) {
    MTL::RenderPipelineDescriptor* rpd =
        MTL::RenderPipelineDescriptor::alloc()->init();
    if (!rpd) {
        return nullptr;
    }
    rpd->setLabel(
        NS::String::string("GLSL Pipeline", NS::UTF8StringEncoding));

    rpd->setRasterizationEnabled(desc->rasterization_enabled ? true : false);
    if (mgl_env_flag_enabled("MGL_ENABLE_ICB_PIPELINES")) {
        rpd->setSupportIndirectCommandBuffers(true);
    }
    rpd->setAlphaToCoverageEnabled(desc->alpha_to_coverage_enabled ? true : false);
    rpd->setAlphaToOneEnabled(desc->alpha_to_one_enabled ? true : false);
    rpd->setInputPrimitiveTopology(
        (MTL::PrimitiveTopologyClass)desc->input_primitive_topology);
    if (desc->raster_sample_count > 0) {
        rpd->setRasterSampleCount(desc->raster_sample_count);
    }

    for (uint32_t i = 0; i < desc->color_count && i < 8; i++) {
        MTL::RenderPipelineColorAttachmentDescriptor* ca =
            rpd->colorAttachments()->object(i);
        ca->setPixelFormat((MTL::PixelFormat)desc->color_format[i]);
        /* Untouched (invalid-format) attachments keep Metal defaults —
         * writeMask All, blending off — exactly like the ObjC descriptor
         * that never touched them. */
        if (desc->color_format[i] !=
            static_cast<uint32_t>(MTL::PixelFormatInvalid)) {
            ca->setWriteMask((MTL::ColorWriteMask)desc->color_write_mask[i]);
            if (desc->blending_enabled_mask & (1u << i)) {
                ca->setBlendingEnabled(true);
                ca->setSourceRGBBlendFactor(
                    (MTL::BlendFactor)desc->source_rgb_blend_factor[i]);
                ca->setDestinationRGBBlendFactor(
                    (MTL::BlendFactor)desc->destination_rgb_blend_factor[i]);
                ca->setSourceAlphaBlendFactor(
                    (MTL::BlendFactor)desc->source_alpha_blend_factor[i]);
                ca->setDestinationAlphaBlendFactor(
                    (MTL::BlendFactor)desc->destination_alpha_blend_factor[i]);
                ca->setRgbBlendOperation(
                    (MTL::BlendOperation)desc->rgb_blend_operation[i]);
                ca->setAlphaBlendOperation(
                    (MTL::BlendOperation)desc->alpha_blend_operation[i]);
            }
        }
    }

    rpd->setDepthAttachmentPixelFormat((MTL::PixelFormat)desc->depth_format);
    rpd->setStencilAttachmentPixelFormat((MTL::PixelFormat)desc->stencil_format);

    if (desc->attrib_count > 0) {
        MTL::VertexDescriptor* vd = MTL::VertexDescriptor::alloc()->init();
        for (uint32_t i = 0; i < desc->attrib_count && i < 32; i++) {
            const uint32_t bufIdx = desc->attrib_buffer_index[i];
            vd->attributes()->object(i)->setFormat(
                (MTL::VertexFormat)desc->attrib_format[i]);
            vd->attributes()->object(i)->setOffset(desc->attrib_offset[i]);
            vd->attributes()->object(i)->setBufferIndex(bufIdx);
            /* 只有格式有效的 attribute 才写 layout —— 与 ObjC
             * generateVertexDescriptorState 一致：未使用的 attrib（Invalid
             * 格式、零值 buffer 索引）不得用 0 stride/stepRate 覆盖已写
             * 的 layout 状态。 */
            if (desc->attrib_format[i] !=
                static_cast<uint32_t>(MTL::VertexFormatInvalid)) {
                vd->layouts()->object(bufIdx)->setStride(desc->attrib_stride[i]);
                vd->layouts()->object(bufIdx)->setStepFunction(
                    (MTL::VertexStepFunction)desc->attrib_step_function[i]);
                vd->layouts()->object(bufIdx)->setStepRate(desc->attrib_step_rate[i]);
            }
        }
        rpd->setVertexDescriptor(vd);
        vd->release();
    }

    rpd->setTessellationPartitionMode(
        (MTL::TessellationPartitionMode)desc->tessellation_partition_mode);
    /* maxTessellationFactor 默认 64（ObjC descriptor 默认值）；0 表示
     * 未设置（safe/simple fallback 的零值 state），跳过以避免 Metal
     * "maxTessellationFactor must be >= 1 and <= 64" 断言。 */
    if (desc->max_tessellation_factor > 0) {
        rpd->setMaxTessellationFactor(desc->max_tessellation_factor);
    }
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

    return rpd;
}

// P4.2: 共享 PSO 创建。完整 pipeline 先用
// FailOnBinaryArchiveMiss 查询 archive；命中直接使用，miss 才普通编译
// 并 add。这使 archive 在多轮加载/保存中保持增量且不重复追加。
int createRenderPipelineInternal(
    MTL::Device* dev, MTL::Function* vsFn, MTL::Function* fsFn,
    const MGLRenderCppPipelineDescriptorState* desc, MTL::BinaryArchive* archive,
    void** pso_out, char* err, size_t errcap) {
    if (pso_out) *pso_out = nullptr;
    if (!dev || !vsFn || !desc || !pso_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }

    MGLRenderCppPipelineDescriptorState state = *desc;
    normalizeDepthStencilFormats(&state);

    std::string key = pipelineKey(vsFn, fsFn, &state);
    PSOCache& cache = psoCache();
    auto it = cache.find(key);
    const bool archiveEligible = archive && vsFn && fsFn;
    if (it != cache.end() && !archiveEligible) {
        static_cast<MTL::RenderPipelineState*>(it->second)->retain();
        *pso_out = it->second;
        return 0;
    }

    MTL::RenderPipelineDescriptor* rpd =
        buildRenderPipelineDescriptor(&state);
    if (!rpd) {
        if (err && errcap) snprintf(err, errcap, "descriptor alloc failed");
        return -1;
    }
    rpd->setVertexFunction(vsFn);
    if (fsFn) {
        rpd->setFragmentFunction(fsFn);
    }
    /* Metal accepts incomplete render pipelines used by capture/discard
     * paths, but MTLBinaryArchive rejects either missing stage when it later
     * serializes.  Match the ObjC archive gate and keep both vertex-only and
     * fragment-only PSOs out of the archive. */
    if (archiveEligible) {
        rpd->setBinaryArchives(NS::Array::array(archive));
    }

    NS::Error* nsErr = nullptr;
    MTL::RenderPipelineState* pso = nullptr;
    if (archiveEligible) {
        pso = dev->newRenderPipelineState(
            rpd, MTL::PipelineOptionFailOnBinaryArchiveMiss,
            nullptr, &nsErr);
        if (pso) {
            if (it != cache.end()) {
                pso->release();
                static_cast<MTL::RenderPipelineState*>(it->second)->retain();
                *pso_out = it->second;
                rpd->release();
                return 0;
            }
        } else if (it != cache.end()) {
            /* The PSO already exists in this process, so only teach the
             * persistent archive about the miss; recompiling the same PSO is
             * unnecessary. */
            NS::Error* addErr = nullptr;
            if (!archive->addRenderPipelineFunctions(rpd, &addErr)) {
                char addMessage[512] = {0};
                copyError(addErr, addMessage, sizeof(addMessage));
                fprintf(stderr,
                        "MGL BINARY ARCHIVE: addRenderPipeline warning: %s\n",
                        addMessage[0] ? addMessage : "unknown error");
            }
            static_cast<MTL::RenderPipelineState*>(it->second)->retain();
            *pso_out = it->second;
            rpd->release();
            return 0;
        }
    }
    const bool archiveMiss = archiveEligible && !pso;
    if (!pso) {
        nsErr = nullptr;
        pso = dev->newRenderPipelineState(rpd, &nsErr);
    }
    if (!pso) {
        copyError(nsErr, err, errcap);
        rpd->release();
        return -1;
    }
    if (archiveMiss) {
        NS::Error* addErr = nullptr;
        if (!archive->addRenderPipelineFunctions(rpd, &addErr)) {
            char addMessage[512] = {0};
            copyError(addErr, addMessage, sizeof(addMessage));
            fprintf(stderr,
                    "MGL BINARY ARCHIVE: addRenderPipeline warning: %s\n",
                    addMessage[0] ? addMessage : "unknown error");
        }
    }
    rpd->release();

    pso->retain(); // 缓存长期持有
    cache[key] = pso;
    *pso_out = pso; // 调用方持有一份引用（mglAirRelease）
    return 0;
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
    return createRenderPipelineInternal(
        static_cast<MTL::Device*>(const_cast<void*>(device)),
        static_cast<MTL::Function*>(vs_function),
        fs_function ? static_cast<MTL::Function*>(fs_function) : nullptr,
        desc, nullptr /* archive */, pso_out, err, errcap);
}

int mglAirCreateRenderPipelineWithArchive(
    const void* device, void* vs_function, void* fs_function,
    const MGLRenderCppPipelineDescriptorState* desc, void* binary_archive,
    void** pso_out, char* err, size_t errcap) {
    if (!device || !vs_function || !desc || !pso_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    return createRenderPipelineInternal(
        static_cast<MTL::Device*>(const_cast<void*>(device)),
        static_cast<MTL::Function*>(vs_function),
        fs_function ? static_cast<MTL::Function*>(fs_function) : nullptr,
        desc, static_cast<MTL::BinaryArchive*>(binary_archive),
        pso_out, err, errcap);
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
