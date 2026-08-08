//------------------------------------------------------------------------------------------------
// mgl_air_loader.h — AIR metallib → MTL::Library → PSO 的纯 C 接口
//
// 调用方（ObjC 渲染层）只通过本头接触 C++ AIR 加载器，不看见 MTL::* 类型。
// 所有权规则：
//   mglAirLoadLibrary 返回的 library（void* = MTL::Library*，+1 retained）由调用方
//   拥有，可直接 CFBridgingRetain 转移给 ObjC 侧或 mglAirRelease 释放。
//   mglAirCreateRenderPipeline / mglAirCreateComputePipeline 返回的 PSO 由 C++
//   侧 PSO 缓存额外 retain（缓存长期持有），调用方持有一份引用，用完 mglAirRelease。
//------------------------------------------------------------------------------------------------
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* finalDescriptor 等价装配的输入（Phase 1 最小集：color/depth/stencil +
 * rasterization + ICB + AIR 顶点布局）。blend/二进制归档 Phase 4 补齐。 */
typedef struct MGLPipelineDescriptorState {
    uint32_t color_count;
    uint32_t color_format[8];   /* MTLPixelFormat 以 uint 传 */
    uint32_t depth_format;      /* MTLPixelFormat 以 uint 传（MTLPixelFormatInvalid=0） */
    uint32_t stencil_format;    /* MTLPixelFormat 以 uint 传 */
    int      rasterization_enabled;
    int      icb_enabled;       /* indirect command buffers */
    /* AIR 顶点布局（由反射给出）：每个 attrib 一个 (format, offset, buffer stride)。
     * 简化：attrib i → buffer (kMGLVertexAttribBufferBase + i)，stride 为 attrib_stride[i]。 */
    uint32_t attrib_count;
    uint32_t attrib_format[32];
    uint32_t attrib_offset[32];
    uint32_t attrib_stride[32];
} MGLPipelineDescriptorState;

/* device: void* = MTL::Device*（mglRenderCppGetDevice() 取得）。
 * bytes/size: .metallib 字节块。成功返回 0 且 *library_out 非空。 */
int mglAirLoadLibrary(const void* device, const unsigned char* bytes, size_t size,
                      void** library_out, char* err, size_t errcap);

/* vs_library/fs_library: void* = MTL::Library*（mglAirLoadLibrary 产物）。
 * 内部 newFunction("main") 装配 PSO。成功返回 0 且 *pso_out 非空。 */
int mglAirCreateRenderPipeline(const void* device, void* vs_library, void* fs_library,
                               const MGLPipelineDescriptorState* desc, void** pso_out,
                               char* err, size_t errcap);

/* library: void* = MTL::Library*；compute function 名为 "main"。 */
int mglAirCreateComputePipeline(const void* device, void* library,
                                void** pso_out, char* err, size_t errcap);

/* MTL::Release 包装（对 loader 返回的 +1 引用）。 */
void mglAirRelease(void* obj);

#ifdef __cplusplus
} // extern "C"
#endif
