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
#include <mutex>
#include <string>

namespace {

// PSO 缓存：key = 描述符状态序列化摘要。缓存长期 retain 一份，返回给调用方
// 的引用由调用方 mglAirRelease。Phase 4 并入 C++ 渲染器。
std::mutex g_psoMutex;
std::map<std::string, void*> g_psoCache;

std::string pipelineKey(const MGLPipelineDescriptorState* d) {
    char buf[1024];
    int n = snprintf(buf, sizeof buf,
                     "r%u|d%u|s%u|rz%d|icb%d|c%u", d->color_count,
                     d->depth_format, d->stencil_format, d->rasterization_enabled,
                     d->icb_enabled, d->color_count);
    size_t off = (size_t)n;
    for (uint32_t i = 0; i < d->color_count && off < sizeof buf - 2; i++) {
        n = snprintf(buf + off, sizeof buf - off, "|c%u:%u", i, d->color_format[i]);
        if (n < 0) break;
        off += (size_t)n;
    }
    n = snprintf(buf + off, sizeof buf - off, "|a%d", d->attrib_count);
    if (n > 0) off += (size_t)n;
    for (uint32_t i = 0; i < d->attrib_count && off < sizeof buf - 2; i++) {
        n = snprintf(buf + off, sizeof buf - off, "|a%u:%u@%u~%u", i,
                     d->attrib_format[i], d->attrib_offset[i], d->attrib_stride[i]);
        if (n < 0) break;
        off += (size_t)n;
    }
    return std::string(buf);
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
        nsErr->release();
        return -1;
    }
    *library_out = lib; // +1 retained，调用方拥有
    return 0;
}

int mglAirCreateRenderPipeline(const void* device, void* vs_library, void* fs_library,
                               const MGLPipelineDescriptorState* desc, void** pso_out,
                               char* err, size_t errcap) {
    if (!device || !vs_library || !desc || !pso_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    MTL::Device* dev = static_cast<MTL::Device*>(const_cast<void*>(device));
    MTL::Library* vs = static_cast<MTL::Library*>(vs_library);
    MTL::Library* fs = fs_library ? static_cast<MTL::Library*>(fs_library) : nullptr;

    std::string key = pipelineKey(desc);
    {
        std::lock_guard<std::mutex> lock(g_psoMutex);
        auto it = g_psoCache.find(key);
        if (it != g_psoCache.end()) {
            *pso_out = it->second;
            return 0;
        }
    }

    MTL::RenderPipelineDescriptor* rpd = MTL::RenderPipelineDescriptor::alloc()->init();
    if (!rpd) {
        if (err && errcap) snprintf(err, errcap, "descriptor alloc failed");
        return -1;
    }

    MTL::Function* vsFn = vs->newFunction(NS::String::string("main", NS::UTF8StringEncoding));
    if (!vsFn) {
        if (err && errcap) snprintf(err, errcap, "vertex function 'main' not found");
        rpd->release();
        return -1;
    }
    rpd->setVertexFunction(vsFn);
    vsFn->release();

    if (fs) {
        MTL::Function* fsFn =
            fs->newFunction(NS::String::string("main", NS::UTF8StringEncoding));
        if (!fsFn) {
            if (err && errcap) snprintf(err, errcap, "fragment function 'main' not found");
            rpd->release();
            return -1;
        }
        rpd->setFragmentFunction(fsFn);
        fsFn->release();
    }

    rpd->setRasterizationEnabled(desc->rasterization_enabled ? true : false);
    rpd->setSupportIndirectCommandBuffers(desc->icb_enabled ? true : false);

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
            const uint32_t bufIdx = 16 + i; // kMGLVertexAttribBufferBase
            vd->attributes()->object(i)->setFormat((MTL::VertexFormat)desc->attrib_format[i]);
            vd->attributes()->object(i)->setOffset(desc->attrib_offset[i]);
            vd->attributes()->object(i)->setBufferIndex(bufIdx);
            vd->layouts()->object(bufIdx)->setStride(desc->attrib_stride[i]);
        }
        rpd->setVertexDescriptor(vd);
        vd->release();
    }

    NS::Error* nsErr = nullptr;
    MTL::RenderPipelineState* pso = dev->newRenderPipelineState(rpd, &nsErr);
    rpd->release();
    if (!pso) {
        copyError(nsErr, err, errcap);
        nsErr->release();
        return -1;
    }

    pso->retain(); // 缓存长期持有
    {
        std::lock_guard<std::mutex> lock(g_psoMutex);
        g_psoCache[key] = pso;
    }
    *pso_out = pso; // 调用方持有一份引用（mglAirRelease）
    return 0;
}

int mglAirCreateComputePipeline(const void* device, void* library,
                                void** pso_out, char* err, size_t errcap) {
    if (!device || !library || !pso_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
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
        nsErr->release();
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

} // extern "C"
