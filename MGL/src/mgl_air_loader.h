//------------------------------------------------------------------------------------------------
// mgl_air_loader.h — AIR metallib → MTL::Library → PSO 的纯 C 接口
//
// 调用方（ObjC 渲染层）只通过本头接触 C++ AIR 加载器，不看见 MTL::* 类型。
// 所有权规则：
//   mglAirLoadLibrary 返回的 library（void* = MTL::Library*，+1 retained）由调用方
//   拥有，可直接 CFBridgingRetain 转移给 ObjC 侧或 mglAirRelease 释放。
//   mglAirCreateRenderPipeline 返回的 PSO 由 loader 缓存额外 retain；调用方也
//   持有一份引用。mglAirCreateComputePipeline 不缓存，返回值仅由调用方持有。
//   两者的调用方引用都用 mglAirRelease 释放。
//------------------------------------------------------------------------------------------------
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* P4.2: final/simple/safe descriptor 等价装配的 value-state 输入。ObjC 只
 * 构造本结构（不再组装 MTLRenderPipelineDescriptor），render/vertex/
 * tessellation 状态逐字段传入 C++ builder；二进制归档由调用方以
 * MTL::BinaryArchive*（void*）传给 mglRenderCppCreateRenderPipelineFromState。 */
typedef struct MGLRenderCppPipelineDescriptorState {
    uint64_t vertex_program_instance;
    uint64_t vertex_program_generation;
    uint64_t fragment_program_instance;
    uint64_t fragment_program_generation;
    uint32_t color_count;
    uint32_t color_format[8];   /* MTLPixelFormat 以 uint 传 */
    uint32_t depth_format;      /* MTLPixelFormat 以 uint 传（MTLPixelFormatInvalid=0） */
    uint32_t stencil_format;    /* MTLPixelFormat 以 uint 传 */
    int      rasterization_enabled;
    int      icb_enabled;       /* indirect command buffers */
    int      alpha_to_coverage_enabled;
    int      alpha_to_one_enabled;
    uint32_t input_primitive_topology;
    /* AIR 顶点布局（由 finalDescriptor 给出）。 */
    uint32_t attrib_count;
    uint32_t attrib_format[32];
    uint32_t attrib_offset[32];
    uint32_t attrib_stride[32];
    uint32_t attrib_buffer_index[32];
    uint32_t attrib_step_function[32];
    uint32_t attrib_step_rate[32];
    uint32_t color_write_mask[8];
    uint32_t source_rgb_blend_factor[8];
    uint32_t destination_rgb_blend_factor[8];
    uint32_t source_alpha_blend_factor[8];
    uint32_t destination_alpha_blend_factor[8];
    uint32_t rgb_blend_operation[8];
    uint32_t alpha_blend_operation[8];
    uint32_t blending_enabled_mask;
    uint32_t raster_sample_count;
    uint32_t tessellation_partition_mode;
    uint32_t max_tessellation_factor;
    int      tessellation_factor_scale_enabled;
    uint32_t tessellation_factor_format;
    uint32_t tessellation_control_point_index_type;
    uint32_t tessellation_factor_step_function;
    uint32_t tessellation_output_winding_order;
} MGLRenderCppPipelineDescriptorState;

/* 旧名兼容别名（P3.4 backend-neutral 命名迁移期的过渡名）。 */
typedef MGLRenderCppPipelineDescriptorState MGLPipelineDescriptorState;

/* device: void* = MTL::Device*（mglRenderCppGetDevice() 取得）。
 * bytes/size: .metallib 字节块。成功返回 0 且 *library_out 非空。 */
int mglAirLoadLibrary(const void* device, const unsigned char* bytes, size_t size,
                      void** library_out, char* err, size_t errcap);

/* vs_function/fs_function: void* = finalDescriptor 已选定的 MTL::Function*。
 * 使用实际 function 可保留 capture/clip/tess 等 AIR 变体。成功返回 0 且
 * *pso_out 非空。 */
int mglAirCreateRenderPipeline(const void* device, void* vs_function, void* fs_function,
                               const MGLPipelineDescriptorState* desc, void** pso_out,
                               char* err, size_t errcap);

/* P4.2: mglAirCreateRenderPipeline + 二进制归档。binary_archive（+0 borrowed
 * MTL::BinaryArchive*，可为 NULL）仅用于同时具有 vertex/fragment function
 * 的完整 render pipeline；先查 archive hit，miss 才编译并 add。合法的
 * vertex-only capture/rasterizer-discard PSO 不进 archive，因为 Metal 会在
 * 序列化阶段拒绝该记录。 */
int mglAirCreateRenderPipelineWithArchive(
    const void* device, void* vs_function, void* fs_function,
    const MGLRenderCppPipelineDescriptorState* desc, void* binary_archive,
    void** pso_out, char* err, size_t errcap);

/* library: void* = MTL::Library*；compute function 名为 "main"。返回的
 * PSO 是未缓存的 +1 引用。Program/renderer compute 缓存在 mgl_render_cpp。 */
int mglAirCreateComputePipeline(const void* device, void* library,
                                void** pso_out, char* err, size_t errcap);

/* MTL::Release 包装（对 loader 返回的 +1 引用）。 */
void mglAirRelease(void* obj);

/* Release the loader-owned PSO cache. Called by the C++ renderer when its
 * final device user shuts down. Safe to call repeatedly. */
void mglAirLoaderShutdown(void);

#ifdef __cplusplus
} // extern "C"
#endif
