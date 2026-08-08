//------------------------------------------------------------------------------------------------
// mgl_render_cpp.h — C++ 渲染门面的纯 C 入口
//
// 状态层（gl_core / glm_dispatch）与残留 ObjC 侧只通过本头接触 C++ 渲染层，
// 不看见任何 MTL::* 类型。Metal-cpp 私有实现宏唯一定义点在 mgl_render_cpp.cpp。
//------------------------------------------------------------------------------------------------
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* 初始化渲染层。objc_device 为现有 id<MTLDevice>（桥接 +1 retain，不转移所有权）。
 * 返回 0 = 成功；< 0 = 失败（参数为空 / 桥接失败）。 */
int mglRenderCppInit(void* objc_device);

/* 释放渲染层持有的 MTL::* 对象（含 device 的 C++ 侧 retain）。幂等。 */
void mglRenderCppShutdown(void);

/* 调试/测试用：返回 C++ 侧持有的 MTL::Device*（void* 形式），未初始化返回 NULL。 */
void* mglRenderCppGetDevice(void);

#ifdef __cplusplus
} // extern "C"
#endif
