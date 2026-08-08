//------------------------------------------------------------------------------------------------
// mgl_render_cpp.cpp — C++ 渲染门面骨架（Phase 0：无行为变化）
//
// 本 TU 是 NS_PRIVATE_IMPLEMENTATION / MTL_PRIVATE_IMPLEMENTATION 的唯一定义点
// （私有类/选择器符号由此产出）；其他 TU 仅 include mgl_metal_cpp.h 拿声明。
//
// Phase 0 范围：持有 MTL::Device*（桥接现有 ObjC device，不重建）+ 空 PSO 缓存
// map。Phase 1 起填充 mgl_air_loader 的 PSO 缓存与加载路径。
//------------------------------------------------------------------------------------------------
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mgl_metal_cpp.h"
#include "mgl_render_cpp.h"

#include <map>
#include <string>

namespace mgl {

MTL::Device* wrapDevice(void* objcDevice) {
    // MTL::Device* 与 id<MTLDevice> 指针同地址（metal-cpp 薄包装）。
    // C++ 侧 +1 retain，所有权由渲染器持有；ObjC 侧保留自己那份。
    MTL::Device* device = static_cast<MTL::Device*>(objcDevice);
    if (device) {
        device->retain();
    }
    return device;
}

namespace {

// C++ 渲染器单例。Phase 0 仅 device + 空 PSO 缓存 map；
// Phase 1 填充 PSO 缓存（key = program + 顶点布局 + RT 格式 + ...）。
struct RendererCpp {
    MTL::Device* device = nullptr;
    std::map<std::string, void*> pipelineCache; // void* = MTL::RenderPipelineState*
};

RendererCpp g_renderer;

} // namespace

} // namespace mgl

//------------------------------------------------------------------------------------------------
// 纯 C 入口（mgl_render_cpp.h）
//------------------------------------------------------------------------------------------------
extern "C" {

int mglRenderCppInit(void* objc_device) {
    if (!objc_device) {
        return -1;
    }
    if (mgl::g_renderer.device) {
        return 0; // 已初始化（幂等）
    }
    mgl::g_renderer.device = mgl::wrapDevice(objc_device);
    return mgl::g_renderer.device ? 0 : -1;
}

void mglRenderCppShutdown(void) {
    if (mgl::g_renderer.device) {
        mgl::g_renderer.device->release();
        mgl::g_renderer.device = nullptr;
    }
    mgl::g_renderer.pipelineCache.clear();
}

void* mglRenderCppGetDevice(void) {
    return mgl::g_renderer.device;
}

} // extern "C"
