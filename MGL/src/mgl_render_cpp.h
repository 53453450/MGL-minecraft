//------------------------------------------------------------------------------------------------
// mgl_render_cpp.h — C++ 渲染门面的纯 C 入口
//
// 状态层（gl_core / glm_dispatch）与残留 ObjC 侧只通过本头接触 C++ 渲染层，
// 不看见任何 MTL::* 类型。Metal-cpp 私有实现宏唯一定义点在 mgl_render_cpp.cpp。
//------------------------------------------------------------------------------------------------
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Forward decl (mgl_types_texture.h pulls in GLMContext-typed state). */
typedef struct TextureLevel_t TextureLevel;

typedef struct GLMContextRec_t *GLMContext;
typedef struct Buffer_t Buffer;
typedef struct Texture_t Texture;
typedef struct TextureParameter_t TextureParameter;
typedef struct Program_t Program;
typedef struct __GLsync Sync;

/* P4.2: final/simple/safe pipeline descriptor 的 value-state。完整定义在
 * mgl_air_loader.h（MGLRenderCppPipelineDescriptorState）；此处只前向声明，
 * ObjC 侧构造 value-state，不再组装 MTLRenderPipelineDescriptor。 */
typedef struct MGLRenderCppPipelineDescriptorState
    MGLRenderCppPipelineDescriptorState;

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

/* Direct renderer entries. Objects passed here carry the +1 bridge reference
 * owned by the GL state. */
void mglRenderCppDeleteMTLObj(GLMContext glm_ctx, void *object);
void mglRenderCppReleaseBufferMetalData(GLMContext glm_ctx, Buffer *buffer);
void mglRenderCppReleaseBufferCowPool(Buffer *buffer);
void mglRenderCppBindBuffer(GLMContext glm_ctx, Buffer *buffer);
void mglRenderCppBufferSubData(GLMContext glm_ctx,
                               Buffer *buffer,
                               size_t offset,
                               size_t size,
                               const void *bytes);
void *mglRenderCppMapUnmapBuffer(GLMContext glm_ctx,
                                 Buffer *buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map);
void mglRenderCppReadBackBuffer(GLMContext glm_ctx,
                                Buffer *buffer,
                                size_t offset,
                                size_t size);
void mglRenderCppFlushBufferRange(GLMContext glm_ctx,
                                  Buffer *buffer,
                                  intptr_t offset,
                                  intptr_t length);
void mglRenderCppBindProgram(GLMContext glm_ctx, Program *program);
void mglRenderCppGetSync(GLMContext glm_ctx, Sync *sync);
void mglRenderCppWaitForSync(GLMContext glm_ctx, Sync *sync);
unsigned int mglRenderCppGetSyncStatus(GLMContext glm_ctx, Sync *sync);
void mglRenderCppReleaseSync(GLMContext glm_ctx, Sync *sync);
void mglRenderCppFlush(GLMContext glm_ctx, bool finish);
void mglRenderCppInvalidateRenderPass(GLMContext glm_ctx);
uint64_t mglRenderCppGetGPUTimestamp(GLMContext glm_ctx);
typedef struct MGLRenderCppDrawCallbackArgs_t MGLRenderCppDrawCallbackArgs;
typedef struct MGLRenderCppResourceCallbackArgs_t MGLRenderCppResourceCallbackArgs;
typedef struct MGLRenderCppLegacyCallbackArgs_t MGLRenderCppLegacyCallbackArgs;

typedef struct MGLRenderCppCallbackRuntimeOps_t {
    void (*release_context)(void *runtime_context);
    void (*dispatch_compute)(void *runtime_context,
                             GLMContext glm_ctx,
                             unsigned int groups_x,
                             unsigned int groups_y,
                             unsigned int groups_z);
    void (*dispatch_compute_indirect)(void *runtime_context,
                                      GLMContext glm_ctx,
                                      intptr_t indirect);
    void (*draw)(void *runtime_context,
                 GLMContext glm_ctx,
                 const MGLRenderCppDrawCallbackArgs *args);
    void (*bind_texture)(void *runtime_context,
                         GLMContext glm_ctx,
                         Texture *texture);
    void (*flush_draw_buffer)(void *runtime_context, GLMContext glm_ctx);
    void (*swap_buffers)(void *runtime_context, GLMContext glm_ctx);
    void (*clear_buffer)(void *runtime_context,
                         GLMContext glm_ctx,
                         unsigned int type,
                         unsigned int mask);
    void (*blit_framebuffer)(void *runtime_context,
                             GLMContext glm_ctx,
                             int src_x0,
                             int src_y0,
                             int src_x1,
                             int src_y1,
                             int dst_x0,
                             int dst_y0,
                             int dst_x1,
                             int dst_y1,
                             unsigned int mask,
                             unsigned int filter);
    int (*resource)(void *runtime_context,
                    GLMContext glm_ctx,
                    const MGLRenderCppResourceCallbackArgs *args);
    uint64_t (*legacy)(void *runtime_context,
                       GLMContext glm_ctx,
                       const MGLRenderCppLegacyCallbackArgs *args);
} MGLRenderCppCallbackRuntimeOps;

enum {
    MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS = 0,
    MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS = 1,
    MGL_RENDER_CPP_DRAW_CALLBACK_RANGE_ELEMENTS = 2,
    MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS_INSTANCED = 3,
    MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED = 4,
    MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_BASE_VERTEX = 5,
    MGL_RENDER_CPP_DRAW_CALLBACK_RANGE_ELEMENTS_BASE_VERTEX = 6,
    MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED_BASE_VERTEX = 7,
    MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS_INDIRECT = 8,
    MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INDIRECT = 9,
    MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS_INSTANCED_BASE_INSTANCE = 10,
    MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED_BASE_INSTANCE = 11,
    MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE = 12,
    MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ARRAYS = 13,
    MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ELEMENTS = 14,
    MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ELEMENTS_BASE_VERTEX = 15,
    MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ARRAYS_INDIRECT = 16,
    MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ELEMENTS_INDIRECT = 17,
};

struct MGLRenderCppDrawCallbackArgs_t {
    uint32_t kind;
    uint32_t mode;
    uint32_t type;
    uint32_t start;
    uint32_t end;
    int32_t first;
    int32_t count;
    int32_t instance_count;
    int32_t base_vertex;
    uint32_t base_instance;
    int32_t draw_count;
    int32_t stride;
    const void *indices_or_indirect;
    const int32_t *firsts;
    const int32_t *counts;
    const int32_t *base_vertices;
};

enum {
    MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_DRAWABLE = 0,
    MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_INTEGER_PIXELS = 1,
    MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_DEPTH_PIXELS = 2,
    MGL_RENDER_CPP_RESOURCE_CALLBACK_GET_TEX_IMAGE = 3,
    MGL_RENDER_CPP_RESOURCE_CALLBACK_GENERATE_MIPMAPS = 4,
    MGL_RENDER_CPP_RESOURCE_CALLBACK_TEX_SUB_IMAGE = 5,
    MGL_RENDER_CPP_RESOURCE_CALLBACK_TEX_SUB_IMAGE_BYTES = 6,
    MGL_RENDER_CPP_RESOURCE_CALLBACK_COPY_TEX_SUB_IMAGE = 7,
    MGL_RENDER_CPP_RESOURCE_CALLBACK_COPY_IMAGE_SUB_DATA = 8,
};

struct MGLRenderCppResourceCallbackArgs_t {
    uint32_t kind;
    Texture *texture;
    Texture *source_texture;
    Texture *destination_texture;
    Buffer *buffer;
    void *pixel_bytes;
    const void *bytes;
    size_t bytes_size;
    size_t source_offset;
    size_t source_pitch;
    size_t source_image_size;
    size_t source_size;
    size_t width;
    size_t height;
    size_t depth;
    size_t x_offset;
    size_t y_offset;
    size_t z_offset;
    uint32_t bytes_per_row;
    uint32_t bytes_per_image;
    uint32_t format;
    uint32_t type;
    uint32_t slice;
    uint32_t level;
    int32_t x;
    int32_t y;
    int32_t source_level;
    int32_t source_x;
    int32_t source_y;
    int32_t source_z;
    int32_t destination_level;
    int32_t destination_x;
    int32_t destination_y;
    int32_t destination_z;
};

enum {
    MGL_RENDER_CPP_LEGACY_CALLBACK_BIND_BUFFER = 0,
    MGL_RENDER_CPP_LEGACY_CALLBACK_BIND_PROGRAM = 1,
    MGL_RENDER_CPP_LEGACY_CALLBACK_DELETE_OBJECT = 2,
    MGL_RENDER_CPP_LEGACY_CALLBACK_GET_SYNC = 3,
    MGL_RENDER_CPP_LEGACY_CALLBACK_WAIT_SYNC = 4,
    MGL_RENDER_CPP_LEGACY_CALLBACK_GET_SYNC_STATUS = 5,
    MGL_RENDER_CPP_LEGACY_CALLBACK_RELEASE_SYNC = 6,
    MGL_RENDER_CPP_LEGACY_CALLBACK_FLUSH = 7,
    MGL_RENDER_CPP_LEGACY_CALLBACK_INVALIDATE_RENDER_PASS = 8,
    MGL_RENDER_CPP_LEGACY_CALLBACK_BUFFER_SUB_DATA = 9,
    MGL_RENDER_CPP_LEGACY_CALLBACK_MAP_UNMAP_BUFFER = 10,
    MGL_RENDER_CPP_LEGACY_CALLBACK_READ_BACK_BUFFER = 11,
    MGL_RENDER_CPP_LEGACY_CALLBACK_FLUSH_BUFFER_RANGE = 12,
    MGL_RENDER_CPP_LEGACY_CALLBACK_BEGIN_TIMER_QUERY = 13,
    MGL_RENDER_CPP_LEGACY_CALLBACK_END_TIMER_QUERY = 14,
    MGL_RENDER_CPP_LEGACY_CALLBACK_GET_GPU_TIMESTAMP = 15,
    MGL_RENDER_CPP_LEGACY_CALLBACK_BEGIN_SAMPLE_QUERY = 16,
    MGL_RENDER_CPP_LEGACY_CALLBACK_END_SAMPLE_QUERY = 17,
};

struct MGLRenderCppLegacyCallbackArgs_t {
    uint32_t kind;
    Buffer *buffer;
    Program *program;
    Sync *sync;
    void *object;
    const void *bytes;
    size_t offset;
    size_t size;
    intptr_t signed_offset;
    intptr_t signed_length;
    uint32_t value;
    bool flag;
};

int mglRenderCppCreateCallbackRuntime(
    void *runtime_context,
    const MGLRenderCppCallbackRuntimeOps *ops,
    void **runtime_out);
void mglRenderCppDestroyCallbackRuntime(void **runtime_io);

/* ObjC renderer category implementations registered in the runtime operation
 * table.  These are C-callable shared entry points, not selector-forwarding
 * bridge functions. */
void mglRendererCallbackDispatchCompute(void *runtime_context,
                                        GLMContext glm_ctx,
                                        unsigned int groups_x,
                                        unsigned int groups_y,
                                        unsigned int groups_z);
void mglRendererCallbackDispatchComputeIndirect(void *runtime_context,
                                                GLMContext glm_ctx,
                                                intptr_t indirect);
void mglRendererCallbackDraw(void *runtime_context,
                             GLMContext glm_ctx,
                             const MGLRenderCppDrawCallbackArgs *args);
void mglRendererCallbackBindTexture(void *runtime_context,
                                    GLMContext glm_ctx,
                                    Texture *texture);
void mglRendererCallbackFlushDrawBuffer(void *runtime_context,
                                        GLMContext glm_ctx);
void mglRendererCallbackSwapBuffers(void *runtime_context,
                                    GLMContext glm_ctx);
void mglRendererCallbackClearBuffer(void *runtime_context,
                                    GLMContext glm_ctx,
                                    unsigned int type,
                                    unsigned int mask);
void mglRendererCallbackBlitFramebuffer(void *runtime_context,
                                        GLMContext glm_ctx,
                                        int src_x0,
                                        int src_y0,
                                        int src_x1,
                                        int src_y1,
                                        int dst_x0,
                                        int dst_y0,
                                        int dst_x1,
                                        int dst_y1,
                                        unsigned int mask,
                                        unsigned int filter);
int mglRendererCallbackResource(void *runtime_context,
                                GLMContext glm_ctx,
                                const MGLRenderCppResourceCallbackArgs *args);
uint64_t mglRendererCallbackLegacy(
    void *runtime_context,
    GLMContext glm_ctx,
    const MGLRenderCppLegacyCallbackArgs *args);

uint64_t mglRenderCppInvokeLegacyCallback(
    GLMContext glm_ctx,
    const MGLRenderCppLegacyCallbackArgs *args);
void mglRenderCppInvokeBindTextureCallback(GLMContext glm_ctx,
                                           Texture *texture);
void mglRenderCppInvokeFlushDrawBufferCallback(GLMContext glm_ctx);
void mglRenderCppInvokeSwapBuffersCallback(GLMContext glm_ctx);
void mglRenderCppInvokeClearBufferCallback(GLMContext glm_ctx,
                                           unsigned int type,
                                           unsigned int mask);
void mglRenderCppInvokeBlitFramebufferCallback(
    GLMContext glm_ctx,
    int src_x0, int src_y0, int src_x1, int src_y1,
    int dst_x0, int dst_y0, int dst_x1, int dst_y1,
    unsigned int mask, unsigned int filter);
int mglRenderCppInvokeResourceCallback(
    GLMContext glm_ctx,
    const MGLRenderCppResourceCallbackArgs *args);
void mglRenderCppInvokeComputeCallback(GLMContext glm_ctx,
                                       unsigned int groups_x,
                                       unsigned int groups_y,
                                       unsigned int groups_z);
void mglRenderCppInvokeComputeIndirectCallback(GLMContext glm_ctx,
                                               intptr_t indirect);
void mglRenderCppInvokeDrawCallback(
    GLMContext glm_ctx,
    const MGLRenderCppDrawCallbackArgs *args);

/* Install every callback that currently enters the C++ renderer.  Strict C++
 * callbacks and callbacks dispatched through the opaque pure-C runtime table
 * are reported separately.  Wrappers that can still call a legacy bridge do
 * not count toward P4's completion criterion. */
/* Publish the borrowed runtime owner handles used by direct C++ callbacks. */
int mglRenderCppAttachRuntimeOwners(GLMContext glm_ctx,
                                    void *command_buffer_owner,
                                    void *render_encoder_owner,
                                    void *render_pass_state_owner);
void mglRenderCppDetachRuntimeOwners(GLMContext glm_ctx);
void mglRenderCppBeginTimerQueryCallback(GLMContext glm_ctx);
uint64_t mglRenderCppEndTimerQueryCallback(GLMContext glm_ctx);
void mglRenderCppBeginSampleQueryCallback(GLMContext glm_ctx,
                                          unsigned int target);
uint64_t mglRenderCppEndSampleQueryCallback(GLMContext glm_ctx);

enum {
    MGL_RENDER_CPP_AIR_PROGRAM_BOUND = 0,
    MGL_RENDER_CPP_AIR_PROGRAM_NOT_APPLICABLE = 1,
    MGL_RENDER_CPP_AIR_PROGRAM_ERROR = -1,
};

/* Load every AIR-backed stage in a linked Program and install the resulting
 * +1 library/function references directly in its MGLShaderModule slots.  Programs that
 * still contain a legacy MSL stage are left untouched and return
 * NOT_APPLICABLE so the ObjC baseline can bind the whole program. */
int mglRenderCppBindAIRProgram(Program *program,
                               int *failed_stage_out,
                               char *err,
                               size_t errcap);

enum {
    MGL_RENDER_CPP_BUFFER_BOUND = 0,
    MGL_RENDER_CPP_BUFFER_NOT_APPLICABLE = 1,
    MGL_RENDER_CPP_BUFFER_ERROR = -1,
};

/* Materialize shared, copy-backed, client-storage, and persistent no-copy
 * Buffer storage in Metal-cpp. No-copy buffers transfer VM-range cleanup to
 * the retained Metal object and set data.mtl_owns_buffer_data. */
int mglRenderCppBindBufferStorage(Buffer *buffer,
                                  char *err,
                                  size_t errcap);

enum {
    MGL_RENDER_CPP_BUFFER_OPERATION_HANDLED = 0,
    MGL_RENDER_CPP_BUFFER_OPERATION_NOT_APPLICABLE = 1,
    MGL_RENDER_CPP_BUFFER_OPERATION_ERROR = -1,
};

/* Update buffer storage and dirty state for an encoder bind. */
int mglRenderCppUpdateDirtyBuffer(Buffer *buffer,
                                  char *err,
                                  size_t errcap);
int mglRenderCppBufferSubDataStorage(Buffer *buffer,
                                     size_t offset,
                                     size_t size,
                                     const void *bytes,
                                     char *err,
                                     size_t errcap);
int mglRenderCppSnapshotSharedDirtyBuffer(Buffer *buffer,
                                          void **metal_buffer_out,
                                          char *err,
                                          size_t errcap);
int mglRenderCppSnapshotSharedBufferRange(Buffer *buffer,
                                          size_t offset,
                                          size_t length,
                                          void **metal_buffer_out,
                                          char *err,
                                          size_t errcap);
uint64_t mglRenderCppAdvanceBufferGeneration(void);
void mglRenderCppRecordBufferGenerationCompleted(uint64_t generation);
uint64_t mglRenderCppCompletedBufferGeneration(void);
void mglRenderCppNoteBufferEncoded(Buffer *buffer);
int mglRenderCppMapBufferStorage(Buffer *buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map,
                                 void **mapped_out,
                                 char *err,
                                 size_t errcap);
int mglRenderCppFlushBufferRangeStorage(Buffer *buffer,
                                         intptr_t offset,
                                         intptr_t length,
                                         char *err,
                                         size_t errcap);

typedef enum MGLRenderCppVertexConversionKind_t {
    MGL_RENDER_CPP_VERTEX_DOUBLE_TO_FLOAT = 0,
    MGL_RENDER_CPP_VERTEX_INT_TO_FLOAT = 1,
    MGL_RENDER_CPP_VERTEX_FIXED_TO_FLOAT = 2,
    MGL_RENDER_CPP_VERTEX_PACKED_1010102_TO_FLOAT = 3,
    MGL_RENDER_CPP_VERTEX_PACKED_10F11F11F_TO_FLOAT = 4,
    MGL_RENDER_CPP_VERTEX_INTEGER_TO_32 = 5,
} MGLRenderCppVertexConversionKind;

typedef struct MGLRenderCppVertexConversion_t {
    uint32_t kind;
    uint32_t component_count;
    uint32_t source_type;
    uint32_t normalized;
    uint32_t destination_signed;
    int64_t binding_offset;
    int64_t relative_offset;
    uint64_t stride;
} MGLRenderCppVertexConversion;

/* Convert unsupported GL vertex formats and return a +1 MTLBuffer as void*.
 * The caller must consume it with __bridge_transfer or release it through
 * mglRenderCppDeleteMTLObj. The renderer cache owns a separate reference. */
int mglRenderCppConvertVertexBuffer(
    Buffer *source_buffer,
    const MGLRenderCppVertexConversion *conversion,
    uint64_t *converted_stride_out,
    void **converted_buffer_out,
    char *err,
    size_t errcap);

/* Pack a plain-struct uniform into renderer-owned transient storage. The
 * returned Buffer wrapper remains owned by the renderer; its MTLBuffer
 * backing is replaced when the 128-slot ring wraps. */
Buffer *mglRenderCppAcquirePackedStructBuffer(const void *data,
                                               size_t size,
                                               char *err,
                                               size_t errcap);

/* Renderer-owned device utility facade. Newly created resources are +1 and
 * must be consumed by __bridge_transfer or released through the C++ facade. */
int mglRenderCppCreateBuffer(uint64_t length,
                             uint64_t resource_options,
                             const char *label,
                             void **buffer_out);
int mglRenderCppCreateBufferWithBytes(const void *bytes,
                                      uint64_t length,
                                      uint64_t resource_options,
                                      const char *label,
                                      void **buffer_out);
/* C++-owned transient upload buffer. The returned buffer is borrowed from
 * the opaque owner; Metal command encoders retain it when a copy command is
 * recorded, so the owner may be destroyed immediately after encoding. */
int mglRenderCppCreateTextureStagingOwner(
    const void *bytes,
    uint64_t length,
    uint64_t resource_options,
    void **owner_out,
    void **buffer_out);
void mglRenderCppDestroyTextureStagingOwner(void **owner);
/* Create a shared/no-copy buffer for VM-backed GL client or persistent
 * storage.  When deallocate_vm is non-zero Metal owns the VM range and
 * releases it with vm_deallocate after the last in-flight command buffer. */
int mglRenderCppCreateBufferWithBytesNoCopy(const void *bytes,
                                            uint64_t length,
                                            uint64_t resource_options,
                                            const char *label,
                                            int deallocate_vm,
                                            void **buffer_out);
typedef struct MGLRenderCppTextureDescriptorState_t {
    uint32_t texture_type;
    uint32_t pixel_format;
    uint64_t width;
    uint64_t height;
    uint64_t depth;
    uint64_t mipmap_level_count;
    uint64_t sample_count;
    uint64_t array_length;
    uint64_t resource_options;
    uint64_t usage;
    uint32_t cpu_cache_mode;
    uint32_t storage_mode;
    uint32_t hazard_tracking_mode;
    uint32_t compression_type;
    uint32_t placement_sparse_page_size;
    uint32_t allow_gpu_optimized_contents;
    uint32_t swizzle_red;
    uint32_t swizzle_green;
    uint32_t swizzle_blue;
    uint32_t swizzle_alpha;
} MGLRenderCppTextureDescriptorState;

/* C++ owns and releases the temporary MTL::TextureDescriptor. The C ABI
 * carries only descriptor values, never an Objective-C/MTL descriptor. */
int mglRenderCppCreateTextureFromState(
    const MGLRenderCppTextureDescriptorState *texture_descriptor,
    const char *label,
    void **texture_out);
int mglRenderCppCreateBufferTextureFromState(
    void *buffer,
    const MGLRenderCppTextureDescriptorState *texture_descriptor,
    uint64_t offset,
    uint64_t bytes_per_row,
    void **texture_out);
int mglRenderCppCreateTextureView(void *texture,
                                  uint32_t pixel_format,
                                  void **texture_view_out);
int mglRenderCppCreateTextureViewRange(
    void *texture,
    uint32_t pixel_format,
    uint32_t texture_type,
    uint64_t level_location,
    uint64_t level_length,
    uint64_t slice_location,
    uint64_t slice_length,
    int use_swizzle,
    uint32_t swizzle_red,
    uint32_t swizzle_green,
    uint32_t swizzle_blue,
    uint32_t swizzle_alpha,
    void **texture_view_out);
/* CPU-visible texture transfer facade. use_slice selects Metal's
 * slice/bytesPerImage overload; region values are passed explicitly so the
 * C ABI does not expose MTLRegion. */
int mglRenderCppTextureReplaceRegion(void *texture,
                                     uint64_t x,
                                     uint64_t y,
                                     uint64_t z,
                                     uint64_t width,
                                     uint64_t height,
                                     uint64_t depth,
                                     uint64_t level,
                                     uint64_t slice,
                                     const void *bytes,
                                     uint64_t bytes_per_row,
                                     uint64_t bytes_per_image,
                                     int use_slice);
int mglRenderCppTextureGetBytes(void *texture,
                                void *bytes,
                                uint64_t bytes_per_row,
                                uint64_t bytes_per_image,
                                uint64_t x,
                                uint64_t y,
                                uint64_t z,
                                uint64_t width,
                                uint64_t height,
                                uint64_t depth,
                                uint64_t level,
                                uint64_t slice,
                                int use_slice);

/* P4.5: GL texture creation target + sample count -> Metal descriptor shape.
 * The value result keeps Metal enums behind uint32_t and carries the legacy
 * upload/completeness flags that must stay consistent with the chosen type.
 * GL_TEXTURE_BUFFER is handled by its dedicated buffer-texture path before
 * this helper is called. */
typedef struct MGLRenderCppTextureTargetPlan_t {
    uint32_t texture_type;
    uint32_t num_faces;
    uint32_t is_array;
    uint32_t texture_1d_backed_by_2d;
    uint32_t texture_1d_array_backed_by_2d_array;
} MGLRenderCppTextureTargetPlan;

int mglRenderCppTextureTargetPlan(
    uint32_t gl_target,
    uint32_t sample_count,
    MGLRenderCppTextureTargetPlan *plan_out);

/* P4.4: GL subimage coordinates -> Metal upload subresource plan.  In
 * particular, GL_TEXTURE_1D_ARRAY stores its first layer/count in
 * yoffset/height, while the Metal 2D-array backing needs slice/arrayLength
 * with origin.y=0 and height=1.  The result is pure value state; no MTL type
 * crosses the C ABI. */
typedef struct MGLRenderCppTextureSubUploadPlan_t {
    uint64_t destination_base_slice;
    uint64_t destination_x;
    uint64_t destination_y;
    uint64_t destination_z;
    uint64_t copy_width;
    uint64_t copy_height;
    uint64_t copy_depth;
    uint64_t layer_count;
    uint64_t source_layer_stride;
} MGLRenderCppTextureSubUploadPlan;

int mglRenderCppTextureSubUploadPlan(
    uint32_t gl_target,
    uint32_t texture_type,
    uint64_t requested_slice,
    uint64_t xoffset,
    uint64_t yoffset,
    uint64_t zoffset,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    uint64_t source_bytes_per_row,
    uint64_t source_bytes_per_image,
    MGLRenderCppTextureSubUploadPlan *plan_out);

/* P4.5 (item 1014/887): reflected shader-resource image shape ->
 * MTLTextureType ABI value.  The C ABI stays backend-neutral: all inputs and
 * the result are uint32_t values, and has_resource preserves the historical
 * NULL-resource result.  Unsupported dimensions return 0. */
uint32_t mglRenderCppTextureTypeForShaderResource(
    uint32_t has_resource,
    uint32_t image_dim,
    uint32_t image_arrayed,
    uint32_t image_multisampled);

/* P4.5 (item 1116/887): MTLTextureType ABI value -> per-target OpenGL
 * texture-unit slot. Unsupported Metal texture types return -1. */
int32_t mglRenderCppTextureIndexForMetalType(uint32_t texture_type);

/* P4.5 (item 1014/887): MTLPixelFormat ABI value -> shader-visible texture
 * data kind.  Keep the C ABI backend-neutral; the numeric results mirror
 * MGLTextureDataKind without exposing that ObjC enum here. */
#define MGL_RENDER_CPP_TEXTURE_DATA_KIND_UNKNOWN 0u
#define MGL_RENDER_CPP_TEXTURE_DATA_KIND_FLOAT   1u
#define MGL_RENDER_CPP_TEXTURE_DATA_KIND_SINT    2u
#define MGL_RENDER_CPP_TEXTURE_DATA_KIND_UINT    3u
#define MGL_RENDER_CPP_TEXTURE_DATA_KIND_DEPTH   4u

uint32_t mglRenderCppTextureDataKindForPixelFormat(uint32_t pixel_format);
/* P4.5 (item 1111): pure pixel-format and GL internal-format predicates.
 * The C ABI carries only stable integer enum values; ObjC compatibility
 * headers remain thin wrappers around these C++ tables. */
int mglRenderCppMetalPixelFormatIsDepthOrStencil(uint32_t pixel_format);
int mglRenderCppMetalPixelFormatIsPackedDepthStencil(uint32_t pixel_format);
int mglRenderCppGLInternalFormatLooksDepthOrStencil(uint32_t internal_format);
int mglRenderCppTexturePixelFormatCompatibleWithExpectedDataKind(
    uint32_t pixel_format, uint32_t expected_kind);
/* P4.5 (item 1111): compressed upload row math.  Returns the block height
 * and rounded upload-row count using uint64_t so the C ABI is Foundation-free. */
uint64_t mglRenderCppMetalCompressedBlockHeight(uint32_t pixel_format);
uint64_t mglRenderCppMetalUploadRowsForPixelFormat(uint32_t pixel_format,
                                                   uint64_t pixel_height);
/* P4.5 (item 1111): data-kind → debug name string (static literals).
 * kind uses MGL_RENDER_CPP_TEXTURE_DATA_KIND_*. */
const char *mglRenderCppTextureDataKindName(uint32_t kind);
/* P4.5 (item 1111): min-filter → uses-mipmaps.  Returns 1/0. */
int mglRenderCppTextureMinFilterUsesMipmaps(uint32_t min_filter);

/* P4.5 (item 1171): readback bytes-per-pixel table (MTLPixelFormat ABI value
 * -> bytes).  Pure CPU table shared by both gates — mirrors the ObjC
 * mglMetalReadbackBytesPerPixel exactly (default 4 bytes for unlisted
 * formats).  The C ABI carries the pixel format as uint32_t (Apple stable
 * enum), matching mglRenderCppTextureDataKindForPixelFormat. */
uint32_t mglRenderCppReadbackBytesPerPixel(uint32_t pixel_format);

/* P4.5 (item 1171): readback pixel-format classification (MTLPixelFormat ABI
 * value -> boolean).  Pure CPU tables shared by both gates — mirror the ObjC
 * mglMetalReadbackFormatIsBGRA8Compatible / mglMetalPixelFormatIsIntegerColor /
 * mglMetalPixelFormatIsSignedIntegerColor exactly.  Returns 1/0. */
int mglRenderCppReadbackFormatIsBGRA8Compatible(uint32_t pixel_format);
int mglRenderCppPixelFormatIsIntegerColor(uint32_t pixel_format);
int mglRenderCppPixelFormatIsSignedIntegerColor(uint32_t pixel_format);

/* P4.5 (item 1111/887): layer / sRGB pixel-format tables.  Pixel format
 * is the Apple MTLPixelFormat numeric value.  Effective honors
 * GL_EXT_texture_sRGB_decode via the raw srgb_decode_ext enum. */
int mglRenderCppMetalLayerPixelFormatIsSupported(uint32_t pixel_format);
uint32_t mglRenderCppSRGBPixelFormat(uint32_t pixel_format);
uint32_t mglRenderCppLinearPixelFormat(uint32_t pixel_format);
uint32_t mglRenderCppEffectiveMTLPixelFormat(uint32_t pixel_format,
                                             uint32_t srgb_decode_ext);

/* P4.5 (item 1171): copy packed rows with optional Y-flip.  Pure CPU
 * memcpy of `row_bytes` per row — mirrors mglMetalCopyRows (void). */
void mglRenderCppCopyRows(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t row_bytes, uint64_t height, int flip_y);

/* P4.5 (item 1171): Depth16Unorm / unpacked depth-float rows -> GL
 * float rows with optional Y-flip.  Mirrors the CPU convert loop in
 * mglReadDepthTextureAsFloat (void; bad args are a no-op). */
void mglRenderCppCopyDepthTextureBytesToFloat(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint64_t src_depth_bytes, int is_depth16, int flip_y);

/* P4.5 (item 1171): copy GL BGRA8 rows into a BGRA8-compatible Metal pixel
 * format (RGBA8Unorm / BGRA8Unorm / RGB9E5Float / RGB10A2Unorm /
 * BGR10A2Unorm) with optional Y-flip.  Pure CPU data transform shared by
 * both gates — mirrors the ObjC
 * mglMetalCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes exactly (1 on
 * success, 0 on bad args / unsupported format). */
int mglRenderCppCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, int flip_y);

/* P4.5 (item 1171): copy Metal texture bytes into GL BGRA8 (source-format
 * decode: RGBA8/BGRA8, R/RG/RGBA 8/16/32 unorm/snorm/int/uint/float,
 * RGB9E5, RGB10A2/BGR10A2, BGR5A1, ABGR4, RG11B10, half/float variants)
 * with optional Y-flip.  Pure CPU data transform shared by both gates —
 * mirrors the ObjC mglMetalCopyTextureBytesToBGRA8 exactly (void). */
void mglRenderCppCopyTextureBytesToBGRA8(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, int flip_y);

/* P4.5 (item 1171): accepted GL pixel types for
 * mglMetalCopyBGRA8CompatibleTextureBytesToGL.  Returns 1/0. */
int mglRenderCppReadbackGLTypeAccepted(uint32_t type);

/* P4.5 (item 1171): SNORM8 texture bytes -> GL format/type, bypassing
 * the lossy BGRA8 UNORM intermediate.  Mirrors the ObjC sourceIsSnorm8
 * path (1 on success, 0 on bad args / unsupported format). */
int mglRenderCppCopySnorm8TextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* P4.5 (item 1171): RGB10A2Unorm texture bytes -> GL format/type,
 * bypassing the lossy BGRA8 UNORM intermediate.  Mirrors the ObjC
 * sourceIsRGB10A2Direct path (1 on success, 0 on bad args / unsupported). */
int mglRenderCppCopyRGB10A2TextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* P4.5 (item 1171): RG11B10Float texture bytes -> GL format/type,
 * bypassing the lossy BGRA8 UNORM intermediate.  Mirrors the ObjC
 * sourceIsRG11B10FloatDirect path (1 on success, 0 on bad args). */
int mglRenderCppCopyRG11B10TextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* P4.5 (item 1171): R16/RG16/RGBA16 Unorm/Snorm/Float and
 * R32/RG32/RGBA32 Float texture bytes -> GL format/type, bypassing
 * the lossy BGRA8 UNORM intermediate.  Mirrors the ObjC 16/32-bit
 * direct path (1 on success, 0 on bad args / unsupported). */
int mglRenderCppCopy16or32TextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* P4.5 (item 1171): BGRA8/RGBA8 UNORM texture bytes -> GL scalar
 * types (BYTE/SHORT/INT/UINT/USHORT/HALF/FLOAT).  Mirrors the ObjC
 * scalar integer/half/float readback path (1 on success, 0 on bad
 * args / unsupported). */
int mglRenderCppCopyUnorm8ScalarTextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* P4.5 (item 1171): BGRA8/RGBA8 UNORM texture bytes -> GL packed
 * types (3_3_2 / 5_6_5 / 4_4_4_4 / 5_5_5_1 / 8_8_8_8 /
 * 10_10_10_2 / 10F_11F_11F_REV / 5_9_9_9_REV and REV variants).
 * Mirrors the ObjC packed readback path (1 on success, 0 on bad
 * args / unsupported). */
int mglRenderCppCopyUnorm8PackedTextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* P4.5 (item 1171): BGRA8/RGBA8 UNORM texture bytes -> GL channel
 * swizzle tail (UNSIGNED_BYTE, plus the leftover RGBA FLOAT branch).
 * Mirrors the ObjC final format switch (1 on success, 0 on bad args /
 * unsupported). */
int mglRenderCppCopyUnorm8SwizzleTextureBytesToGL(
    const void *src, uint64_t src_bytes_per_row,
    void *dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y);

/* P4.4: CPU→GPU 上传路径选路。纯决策函数（无 Metal 对象参与），把
 * MGLRenderer+Texture.m uploadTextureSliceViaBlit 的「storage mode /
 * 纹理类型 / AGX 能力位 → replaceRegion 或 blit 或 reject」判定迁入 C++，
 * ObjC 只剩按返回路由执行对应分支体。texture_type / storage_mode 直接传
 * MTLTextureType / MTLStorageMode 的 ABI 数值（Apple 稳定枚举）。
 * 路由语义与既有内联判定完全一致：
 *   - 1D/1DArray 且非 Private → REPLACE_1D（低频率路径，replaceRegion 安全）
 *   - 3D 且 AGX copyFromBuffer slice OOB bug 生效 → Private 拒绝（REJECT），
 *     否则 REPLACE_3D（需紧凑重打包 + bytesPerImage）
 *   - 其余（2D/2DArray/Cube/1D-Private…）→ BLIT（dedicated CB 保 GPU 顺序） */
#define MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_BLIT          0
#define MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_REPLACE_1D    1
#define MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_REPLACE_3D    2
#define MGL_RENDER_CPP_TEXTURE_UPLOAD_ROUTE_REJECT        3

int mglRenderCppTextureUploadRoute(uint32_t texture_type,
                                   uint32_t storage_mode,
                                   int has_agx_3d_copy_bug);
/* Complete value-state plan for a full texture-level/slice upload.  This
 * centralizes the layout normalization that used to live around the ObjC
 * replaceRegion/blit branches.  A REJECT route is a valid plan; malformed
 * dimensions/strides and staging allocations above 512 MiB return -1. */
typedef struct MGLRenderCppTextureUploadPlan_t {
    uint32_t route;
    uint32_t replace_region_dimension; /* 1, 2, or 3; 0 for blit/reject */
    uint32_t replace_use_slice;
    uint32_t requires_repack;
    uint64_t normalized_height;
    uint64_t normalized_depth;
    uint64_t upload_rows;
    uint64_t expected_bytes_per_image;
    uint64_t normalized_bytes_per_image;
    uint64_t copy_depth;
    uint64_t buffer_size;
    uint64_t destination_slice;
    uint64_t destination_level;
} MGLRenderCppTextureUploadPlan;

int mglRenderCppBuildTextureUploadPlan(
    uint32_t gl_target,
    uint32_t texture_type,
    uint32_t storage_mode,
    uint32_t pixel_format,
    int has_agx_3d_copy_bug,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    uint64_t bytes_per_row,
    uint64_t bytes_per_image,
    uint64_t destination_level,
    uint64_t destination_slice,
    MGLRenderCppTextureUploadPlan *plan_out);
/* P4.4: 3D 纹理 depth-plane 重打包（tight image stride）。把 strided
 * (bytes_per_image > expected) 的 depth planes 压成 tight
 * (expected_bytes_per_image) 布局，供 replaceRegion 上传（Metal 要求
 * bytesPerImage = bpr*height，padded 的 plane stride 必须重打包）。
 * 返回 malloc 的 buffer（调用方 free）；参数非法 / 分配失败返回 NULL。 */
void *mglRenderCppTextureRepackDepthPlanes(const void *bytes,
                                           size_t bytes_per_image,
                                           size_t expected_bytes_per_image,
                                           size_t copy_depth);
/* P4.4: RGB→RGBA 通道扩展（texel buffer 2D fallback 的 CloudFaces 路径）。
 * src 每 texel = src_comp_bytes×3，dst 每 texel = dst_comp_bytes×4（输出
 * tex_width×tex_height 网格，行优先）；alpha 取 alpha_default 的低
 * dst_comp_bytes 字节；超出 texel_count 的尾 texel 置零。返回 0 成功，
 * 坏参返回 -1。dst 由调用方提供（ObjC 用 NSMutableData 管生命周期）。 */
int mglRenderCppTextureExpandRGBToRGBA(const void *src,
                                       void *dst,
                                       size_t texel_count,
                                       size_t tex_width,
                                       size_t tex_height,
                                       size_t src_comp_bytes,
                                       size_t dst_comp_bytes,
                                       uint64_t alpha_default);
/* P4.4: RGBA8 通道扩展（旧式 packed 格式 → RGBA8）。internal_format 为 GL
 * 枚举（R3_G3_B2 / RGB4/5/565 / RGB10/12 / RGBA2/4 / RGB5_A1 / RGB8 变体），
 * 按位展开为 8-bit RGBA（unorm 取整，snorm 1.0=0x7f，整型 a=1）。返回
 * malloc 的 dst（调用方 free），坏参 / 未知格式 / 尺寸超限返回 NULL。 */
/* P4.5 (item 1138): stage-binding copy-back entry (C-ABI mirror of the
 * ObjC MGLStageBindingCopyBack — the ObjC side bridges the buffer refs). */
typedef struct MGLRenderCppCopyBackEntry_t {
    const void *temporary;        /* MTL::Buffer* */
    const void *destination;      /* MTL::Buffer* */
    const void *destination_buffer; /* GL Buffer* (CPU prefix sync) */
    uint64_t destination_offset;
    uint64_t length;
} MGLRenderCppCopyBackEntry;

/* Validate every non-empty entry (bounds vs the Metal buffer lengths) and,
 * when blit_encoder is non-NULL, encode each copy via
 * mglRenderCppBlitCopyBuffer.  Returns 0 on success, -1 on the first
 * invalid entry / encode failure. */
int mglRenderCppEncodeStageBindingCopyBacks(
    const MGLRenderCppCopyBackEntry *entries,
    uint32_t count,
    void *blit_encoder);

/* Synchronize the written CPU prefix of each entry's GL destination buffer
 * (guards + memmove; the Metal contents pointer is read via the
 * destination buffer).  Returns 0, or -1 with *failed_index_out set. */
int mglRenderCppCopyBackCPUPrefix(
    const MGLRenderCppCopyBackEntry *entries,
    uint32_t count,
    uint32_t *failed_index_out);

/* P4.5 (item 1138): runtime-array-size SSBO sizing constants.  The AIR
 * backend emits code that reads uint32 byte-sizes from
 * MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX when a compute shader uses .length()
 * on an unsized SSBO array.  This fills `out_sizes[out_capacity]` from the
 * per-buffer {metal_slot, visible_size} pairs, skipping the runtime-size
 * buffer slot itself and any slot >= max_slot (the ordinary user-buffer table
 * cap, kMGLMaxMetalUserBufferCount=31).  `out_sizes` is expected to be
 * zero-initialized by the caller; only claimed slots are written.  Returns
 * 0 on success, -1 on bad args (NULL out, NULL entries with nonzero count,
 * out_capacity < max_slot). */
typedef struct MGLRenderCppBufferSizeEntry_t {
    uint32_t metal_slot;      /* Metal buffer argument index */
    uint64_t visible_size;    /* byte size, truncated to uint32 by the facade */
} MGLRenderCppBufferSizeEntry;

int mglRenderCppBuildRuntimeArraySizes(
    const MGLRenderCppBufferSizeEntry *entries,
    uint32_t entry_count,
    uint32_t runtime_buffer_index,
    uint32_t max_slot,
    uint32_t *out_sizes,
    uint32_t out_capacity);

/* P4.5 (item 1111): per-level CPU upload data preparation — pure CPU
 * transform shared by both gates (the expansion entries it calls are the
 * same both gates use).  Computes the copy geometry and applies any required
 * format expansion (RGBA8 / channel) to the level bytes.  Returns:
 *   0  success (*out filled; data may be owned — free when owns_data=1)
 *  -1  bad args / rejected level
 *  -2  short backing store (level data smaller than the image needs;
 *      *out still carries the computed geometry for diagnostics) */
typedef struct MGLRenderCppIntegerReadbackConvertParams_t {
    const uint8_t *src;
    uint64_t src_bytes_per_row;
    uint32_t source_component_count;
    uint32_t source_component_bytes;
    int source_signed;
    int source_rgb10a2_uint;
    uint32_t copy_w;
    uint32_t copy_h;
    uint8_t *dst;
    uint64_t dst_bytes_per_row;
    uint64_t dst_pixel_bytes;
    uint64_t dst_x;
    uint64_t dst_y;
    uint32_t output_components;
    const int *component_map;
    uint32_t output_component_bytes;
    uint32_t packed_type;
    int is_packed_type;
    const uint32_t *packed_bit_widths;
    const uint32_t *packed_shifts;
    uint32_t packed_output_bytes;
} MGLRenderCppIntegerReadbackConvertParams;

/* P4.5 (item 1171/1116): integer texture readback CPU conversion — the
 * per-pixel component extraction + GL_INTEGER packing/clamping loop of
 * mglReadIntegerTextureAsRGBA32, as a pure data transformation shared by
 * both gates.  Returns 0 on success, -1 on bad args. */
int mglRenderCppConvertIntegerReadback(
    const MGLRenderCppIntegerReadbackConvertParams *params);

/* P4.5 (item 1141/887): tess-factor buffer CPU transforms — the default
 * canonical factor fill (12B/patch: 4x outer + 2x inner __fp16), the
 * canonical->triangle repack (12B -> 8B/patch) and the native primitive
 * count (GL 4.6 11.2.2.2 ceil rules).  Pure data transforms shared by both
 * gates.
 * Return 0 on success, -1 on bad args (count entry returns 0). */
int mglRenderCppFillDefaultTessFactorBuffer(
    void *dst,
    uint64_t dst_bytes,
    const float *outer_levels,
    const float *inner_levels,
    uint32_t patch_count);
int mglRenderCppRepackTessFactorTriangles(
    const void *src,
    uint64_t src_bytes,
    void *dst,
    uint64_t dst_bytes,
    uint32_t patch_count);
uint64_t mglRenderCppTessPrimitiveCount(
    const void *factors,
    uint64_t bytes,
    uint32_t patch_count,
    uint32_t tess_gen_mode,
    uint32_t instance_count);

/* P4.5 (item 1141/887): GL 4.6 section 11.2.2.2 patch discard predicate.
 * Tests the applicable outer/inner tessellation levels before any clamp to
 * one; non-positive or NaN levels discard the patch.  NULL inputs are
 * conservatively treated as discarded.  Shared by both gates. */
bool mglRenderCppTessFactorsDiscardPatch(
    uint32_t gen_mode,
    const float *edge,
    const float *inside);

/* P4.5 (item 1141/887): per-patch expanded item count for the isolines /
 * point-mode TES kernel (lockstep with mgl_air_backend.cpp's u/v
 * decomposition) — returns 0 when the factor record is missing or the patch
 * is discarded (caller falls back to 1).  Pure data transform shared by
 * both gates. */
uint32_t mglRenderCppTessEvalItemsPerPatch(
    const void *factor_record,
    uint32_t gen_mode,
    uint32_t spacing,
    uint32_t point_mode);

/* P4.5 (item 1141/887): GL 4.6 §11.2.2.2 subdivision-count rounding —
 * fractional_even -> next even (min 2), fractional_odd -> next odd,
 * otherwise ceil(level).  Single source of truth shared by the TES
 * eval-item accounting and the ObjC native per-patch primitive counting
 * (mglTessRoundLevelForSpacing shell in MGLRenderer+Tessellation.m). */
uint32_t mglRenderCppTessRoundLevelForSpacing(
    uint32_t spacing,
    uint32_t ceil_level);

/* P4.5 (item 1141/887): TES XFB field byte size for a GL type (FLOAT/INT/
 * UINT + vec2/3/4; 0 for unsupported).  Matches mglTESXFBFieldByteSize and
 * the packed-write stride contract in mglFixMSLTesAsComputeKernel.  Shared
 * by both gates. */
uint64_t mglRenderCppTESXFBFieldByteSize(uint64_t gl_type);

/* P4.5 (item 1141/887): overflow-checked product (a * b) for tessellation
 * size math; matches the ObjC mglCheckedNSUIntegerProduct.  Returns 0 with
 * *result set, -1 on bad args / overflow.  Shared by both gates. */
int mglRenderCppCheckedProduct(uint64_t a, uint64_t b, uint64_t *result);

/* P4.5 (item 1141/887): unpack an 11-bit (6-bit mantissa) / 10-bit
 * (5-bit mantissa) unsigned float — CPU decode for
 * GL_UNSIGNED_INT_10F_11F_11F_REV vertex data.  5-bit exponent bias 15,
 * no sign bit; matches the ObjC mglFloat11ToFloat / mglFloat10ToFloat
 * exactly (denormal, inf, NaN and ldexpf paths).  Shared by both gates. */
float mglRenderCppFloat11ToFloat(uint32_t val);
float mglRenderCppFloat10ToFloat(uint32_t val);

/* P4.5 (item 1171): CPU pixel-format scalar converters shared by the
 * readback path (mgl_readback.m's mglMetalFloatToUnorm8 /
 * mglMetalSnorm16ToFloat / mglMetalSnorm8ToFloat — pure data transforms,
 * both gates).  Float->unorm8 rounds to nearest (0.5 rounds up); snorm
 * decode maps INT_MIN to -1.0 exactly. */
uint8_t mglRenderCppFloatToUnorm8(float value);
float mglRenderCppSnorm16ToFloat(int16_t value);
float mglRenderCppSnorm8ToFloat(int8_t value);

/* P4.5 (item 1141/887): GL type -> MTLVertexFormat ABI value for TES
 * control-point stage inputs (Float/Float2/3/4, Int/Int2/3/4,
 * UInt/UInt2/3/4, else 0 = MTLVertexFormatInvalid).  Values match the
 * macOS SDK enum (Float=28 ... UInt4=39).  Shared by both gates. */
uint32_t mglRenderCppTessControlPointFormat(uint64_t gl_type);

/* P4.5 (item 1141/887): TES XFB compact vertex stride — sum of the byte
 * sizes of the transform-feedback varyings resolved by name against the
 * TES stage-output resource list (lockstep with the packed writes injected
 * by mglFixMSLTesAsComputeKernel).  0 when the stride cannot be proven
 * (no varyings / unknown field type / overflow).  Matches the ObjC
 * mglTESXFBVertexStride.  Shared by both gates. */
uint64_t mglRenderCppTESXFBVertexStride(const void *program);

/* Overflow-checked tess capture size (records x stride, min_stride floor).
 * Returns 0 with size_out/offset_out set, -1 on bad args / overflow. */
int mglRenderCppCheckedTessCaptureSize(
    int64_t count,
    int64_t instance_count,
    uint64_t stride,
    uint64_t min_stride,
    uint64_t *size_out,
    uint64_t *offset_out);

/* P4.5 (item 1141/887): native TES interface support decision — module /
 * function presence, point-mode / XFB exclusion, TRI/QUADS gen-mode gate,
 * and the MTL::Function patchType + patchControlPointCount consistency
 * checks (zero control-point count = legacy encoding, tolerated).  Shared
 * by both gates; the ObjC caller passes __bridge'd MTL::Function pointers. */
int mglRenderCppNativeTESInterfaceSupported(
    void *tes_function,
    uint64_t tes_metallib_bytes,
    uint32_t tes_gen_point_mode,
    uint32_t tes_xfb_varying_count,
    uint32_t tes_gen_mode,
    void *tcs_function,
    uint64_t tcs_metallib_bytes,
    uint32_t tcs_output_vertices);

/* P4.5 (item 1141/887): pure viewport/scissor/framebuffer intersection
 * decision for the per-draw rasterization-empty early-out.  Returns 1 when
 * the draw cannot rasterize any pixel, 0 otherwise (a zero pass size is
 * "not empty" — the caller resolves the pass size first).  Shared by both
 * gates. */
int mglRenderCppRasterizationIsEmpty(
    int32_t vx,
    int32_t vy,
    int32_t vw,
    int32_t vh,
    uint32_t pass_width,
    uint32_t pass_height,
    int32_t scissor_enabled,
    int32_t sx,
    int32_t sy,
    int32_t sw,
    int32_t sh);

typedef struct MGLRenderCppIntegerReadbackClassify_t {
    int source_is_integer_texture;
    int output_is_integer_format;
    uint32_t output_components;
    int component_map[4];
    uint32_t output_component_bytes;
} MGLRenderCppIntegerReadbackClassify;

/* P4.5 (item 1171/1116): integer-readback classification — the 19-format
 * source-integer table, the GL_*_INTEGER output check, the per-format
 * component map (incl. BGR/BGRA orderings and the GREEN/BLUE/ALPHA
 * single-component compat enums) and the per-type output component bytes.
 * Pure classification shared by both gates.  Returns 0 on success, -1 on
 * bad args. */
int mglRenderCppIntegerReadbackClassify(
    uint32_t pixel_format,
    uint32_t gl_format,
    uint32_t gl_type,
    MGLRenderCppIntegerReadbackClassify *out);

typedef struct MGLRenderCppIntegerPackedType_t {
    int is_packed;
    uint32_t bit_widths[4];
    uint32_t shifts[4];
    uint32_t output_bytes;
    uint32_t output_components;
} MGLRenderCppIntegerPackedType;

/* P4.5 (item 1171/1116): integer-readback packed-type classification —
 * the 10-entry GL packed-type table (3_3_2 / 2_3_3_REV / 5_6_5(+REV) /
 * 4_4_4_4(+REV) / 5_5_5_1 / 1_5_5_5_REV / 8_8_8_8(+REV) /
 * 10_10_10_2 / 2_10_10_10_REV).  Pure classification shared by both
 * gates.  Returns 0 on success, -1 on bad args. */
int mglRenderCppIntegerReadbackPackedTypeClassify(
    uint32_t packed_type,
    MGLRenderCppIntegerPackedType *out);

typedef struct MGLRenderCppIntegerReadbackSource_t {
    uint32_t component_count;
    uint32_t component_bytes;
    int source_signed;
    int source_rgb10a2_uint;
    int recognized;
} MGLRenderCppIntegerReadbackSource;

/* P4.5 (item 1171/1116): integer-readback SOURCE format classification —
 * the 19-entry MTLPixelFormat -> {components, component bytes, signed,
 * RGB10A2} table.  Pure classification shared by both gates.  Returns 0
 * with recognized=1 on a known format, 0 with recognized=0 on unknown,
 * -1 on bad args. */
int mglRenderCppIntegerReadbackSourceClassify(
    uint32_t pixel_format,
    MGLRenderCppIntegerReadbackSource *out);

/* P4.5 (item 1141/887): shadow-upload range math — for gpu_write_target
 * buffers, clamps the recorded written_min/written_max span to the limit;
 * otherwise the whole limit.  Returns 0 with offset/length set, -1 when
 * there is nothing to upload (no written span / zero length).  Pure range
 * computation shared by both gates. */
int mglRenderCppBufferShadowUploadRange(
    int gpu_write_target,
    int64_t written_min,
    int64_t written_max,
    uint64_t limit,
    uint64_t *out_offset,
    uint64_t *out_length);

/* P4.5 (item 1141/887): GL draw mode -> MTLPrimitiveType numbering
 * (0=Point, 1=Line, 2=LineStrip, 3=Triangle, 4=TriangleStrip;
 * 0xFFFFFFFF for modes the renderer routes elsewhere).  Pure table shared
 * by both gates; the caller casts to MTLPrimitiveType. */
uint32_t mglRenderCppMTLPrimitiveTypeForGLMode(uint32_t mode);

/* P4.5 (item 1141/887): GL element index type -> MTLIndexType numbering
 * (0=UInt16, 1=UInt32; 0xFFFFFFFF otherwise).  Pure table shared by both
 * gates; the caller casts to MTLIndexType. */
uint32_t mglRenderCppMTLIndexTypeForGLType(uint32_t gl_type);

/* P4.5 (item 1141/887): Metal mipmap level dimension — the greatest
 * 2^(level) divisor of base (base>>level, clamped to 1).  Pure computation
 * shared by both gates (the ObjC mglMetalTextureLevelDimension keeps the
 * extern linkage its many callers use). */
uint64_t mglRenderCppMetalTextureLevelDimension(uint64_t base, uint64_t level);

/* P4.5 (item 1141/887): triangle-fan element emulation — expand a raw
 * element index stream into `(center, i+1, i+2)` triplets (count-2
 * triangles x 3, all uint32).  Pure CPU; caller frees the returned array.
 * Returns 0 on success with *out_count set, -1 on bad args / overflow. */
int mglRenderCppExpandTriangleFanIndices(
    const uint8_t *bytes,
    uint32_t elem_width,        /* 1, 2 or 4 */
    uint32_t source_count,
    uint32_t **out_indices,     /* malloc'd, count*3 entries */
    uint64_t *out_count);

/* P4.5 (item 1141/887): triangle-strip element emulation — expand a raw
 * element stream into `(first, second, tri+2)` triplets with alternating
 * first/second offset (tri strips), count-2 triangles, all uint32.
 * Pure CPU; caller frees.  Returns 0 with *out_count set, -1 on error. */
int mglRenderCppExpandTriangleStripIndices(
    const uint8_t *bytes, uint32_t elem_width, uint32_t source_count,
    uint32_t **out_indices, uint64_t *out_count);

/* P4.5 (item 1141/887): LINE_LOOP element emulation — copy the raw index
 * stream and append the first index to close the loop (count+1).  Pure CPU;
 * caller frees. */
int mglRenderCppExpandLineLoopIndices(
    const uint8_t *bytes, uint32_t elem_width, uint32_t source_count,
    uint32_t **out_indices, uint64_t *out_count);

/* P4.5 (item 1141/887): quad-array emulation — for each group of 4 array
 * vertices emit `(a,a+1,a+2,a,a+2,a+3)` (two triangles), quad_count*6
 * uint32 total.  Pure CPU; caller frees.  Returns 0 with *out_count, -1 on
 * bad args. */
int mglRenderCppExpandQuadArrayIndices(
    uint32_t quad_count, uint32_t **out_indices, uint64_t *out_count);

/* P4.5 (item 1141/887): quad-element emulation — read 4 source indexes per
 * quad from the raw stream and emit `(i0,i1,i2,i0,i2,i3)`.  Pure CPU;
 * caller frees. */
int mglRenderCppExpandQuadElementIndices(
    const uint8_t *bytes, uint32_t elem_width, uint32_t quad_count,
    uint32_t **out_indices, uint64_t *out_count);

/* P4.5 (item 1141/887): GL_UNSIGNED_BYTE element buffer -> UInt16
 * expansion — write each byte as uint16.  Pure CPU; caller frees. */
int mglRenderCppExpandUInt8ToUInt16(
    const uint8_t *bytes, uint32_t byte_count,
    uint16_t **out_indices, uint64_t *out_count);

/* P4.5 (item 1141/887): triangle-fan ARRAY emulation — vertexCount-2
 * triangles `(0, tri+1, tri+2)`, all uint32.  Pure CPU; caller frees. */
int mglRenderCppExpandTriangleFanArrayIndices(
    uint32_t vertex_count, uint32_t **out_indices, uint64_t *out_count);

/* P4.5 (item 1141/887): triangle-strip ARRAY emulation — vertexCount-2
 * triangles with alternating offset `(tri&1)`.  Pure CPU; caller frees. */
int mglRenderCppExpandTriangleStripArrayIndices(
    uint32_t vertex_count, uint32_t **out_indices, uint64_t *out_count);

/* P4.5 (item 1141/887): LINE_LOOP ARRAY emulation — copy `firstVertex+i`
 * for count vertices then append `firstVertex`.  Pure CPU; caller frees. */
int mglRenderCppExpandLineLoopArrayIndices(
    uint32_t first_vertex, uint32_t vertex_count,
    uint32_t **out_indices, uint64_t *out_count);

/* P4.5 (item 1141/887: quad-array LINE_LOOP emulation — for each group of
 * 4 array vertices emit `(a,a+1,a+1,a+2,a+2,a+3,a+3,a)` (a 4-edge closed
 * loop), quad_count*8 uint32 total.  Pure CPU; caller frees. */
int mglRenderCppExpandQuadArrayLineIndices(
    uint32_t quad_count, uint32_t **out_indices, uint64_t *out_count);

/* P4.5 (item 1141/887): quad-element LINE_LOOP emulation — read 4 source
 * indexes per quad and emit `(i0,i1,i1,i2,i2,i3,i3,i0)`.  Pure CPU;
 * caller frees. */
int mglRenderCppExpandQuadElementLineIndices(
    const uint8_t *bytes, uint32_t elem_width, uint32_t quad_count,
    uint32_t **out_indices, uint64_t *out_count);

/* P4.5 (item 1141/887): index-range scan ignoring primitive-restart markers
 * — computes min/max over the byte stream (BYTE/SHORT/INT width), skipping
 * the restart value.  Pure CPU; matches mglScanIndexRangeIgnoringRestart.
 * Returns 0 on success (with *out_valid = 1 if at least one non-restart
 * index was seen), -1 on bad args. */
int mglRenderCppScanIndexRangeIgnoringRestart(
    const uint8_t *bytes, uint32_t elem_width, uint32_t count,
    int restart_enabled, uint32_t restart_index,
    uint32_t *out_min, uint32_t *out_max, int *out_valid);

/* P4.5 (item 1141/887): prepared (Metal-side) byte offset for a GL element
 * buffer — GL_UNSIGNED_BYTE indices are expanded to UInt16 so the offset
 * doubles, other types pass through.  Matches mglComputePreparedIndexByteOffset.
 * Returns 0 on success, -1 on overflow / bad args. */
int mglRenderCppComputePreparedIndexByteOffset(uint64_t gl_index_type,
                                               uint64_t gl_byte_offset,
                                               uint64_t *out_prepared_offset);

/* P4.5 (item 1141/887): baseByteOffset + firstElement * indexStride with
 * overflow checks.  Matches mglComputeIndexByteOffset.  Returns 0 on success,
 * -1 on bad args / overflow. */
int mglRenderCppComputeIndexByteOffset(uint64_t base_byte_offset,
                                       uint64_t first_element,
                                       uint64_t index_stride,
                                       uint64_t *out_byte_offset);

/* P4.5 (item 1141/887): GL index element byte size (BYTE=1, SHORT=2, INT=4).
 * Matches mglGLIndexElementSize.  Returns 0 for unknown type. */
uint32_t mglRenderCppGLIndexElementSize(uint64_t gl_index_type);

/* P4.5 (item 1141/887): read a single GL index value from a byte buffer at
 * `element_index` (elem_width 1/2/4).  Matches mglReadGLIndexValue; returns 0
 * for NULL buffer or unknown width. */
uint32_t mglRenderCppReadGLIndexValue(const uint8_t *bytes, uint32_t elem_width,
                                      uint64_t element_index);

/* P4.5 (item 1141/887): GL vertex-attribute component size in bytes (1/2/4/8).
 * Matches mglVertexAttribComponentSize.  Returns 0 for unknown. */
uint32_t mglRenderCppVertexAttribComponentSize(uint64_t gl_type);

/* P4.5 (item 1141/887): total bytes for a vertex-attribute element (type x
 * size), with special handling for packed 10_10_10_2 formats.  Matches
 * mglVertexAttribElementBytes.  Returns 0 for unknown / zero size. */
uint64_t mglRenderCppVertexAttribElementBytes(uint64_t gl_type, uint32_t size);

/* P4.5 (item 1141/887): does GL primitive mode produce polygonal primitives
 * (triangles/quads) subject to glPolygonMode point/line emulation?  Matches
 * mglDrawModeProducesPolygons.  Returns 1/0. */
int mglRenderCppDrawModeProducesPolygons(uint64_t gl_mode);

/* P4.5 (item 1141/887): does `mode` with `indexCount` vertices produce at
 * least one drawable segment (point/line/triangle/quad)?  Matches
 * mglPrimitiveModeHasDrawableSegment.  Returns 1/0. */
int mglRenderCppPrimitiveModeHasDrawableSegment(uint64_t gl_mode,
                                                uint64_t index_count);

/* P4.5 (item 1141/887): total triangle index count for `source_vertex_count`
 * vertices arranged as quads (4/quad -> 6 indices).  Matches
 * mglQuadTriangleIndexCount; returns 0 on overflow. */
uint64_t mglRenderCppQuadTriangleIndexCount(uint64_t source_vertex_count);
/* Align vertex stride to 4; matches mglAlignVertexStrideForMetal. */
uint64_t mglRenderCppAlignVertexStrideForMetal(uint64_t stride);
/* double-attrib size -> MTLVertexFormat value; matches mglDoubleVertexAttribFloatFormat. */
uint32_t mglRenderCppDoubleVertexAttribFloatFormat(uint32_t size);
/* Integer attrib signedness mismatch -> Int/UInt MTLVertexFormat ABI value.
 * Returns MTLVertexFormatInvalid when no CPU conversion is required. */
uint32_t mglRenderCppIntegerAttribConversionFormat(
    uint64_t src_type,
    uint64_t shader_gl_type,
    uint32_t size);
/* FNV-1a single hash step; matches mglHashStepU64. */
uint64_t mglRenderCppHashStepU64(uint64_t hash, uint64_t value);
/* Fixed restart-index for a type; matches the fixed branch of
 * mglPrimitiveRestartIndexForType.  1 if defined; *out set. */
int mglRenderCppPrimitiveRestartFixedIndex(uint64_t gl_index_type, uint32_t *out);
/* GL uniform/attrib type -> element byte size; matches mglGLTypeElementByteSize. */
uint32_t mglRenderCppGLTypeElementByteSize(uint64_t gl_type);

typedef struct MGLRenderCppGeometryGatherResult_t {
    uint32_t *gather;          /* malloc'd raw gather (vertex_ids) */
    uint32_t gather_count;
    uint32_t primitive_count;
    uint32_t max_index;
} MGLRenderCppGeometryGatherResult;

/* P4.5 (item 1141/887): the indexed-PATCHES geometry gather — expand a raw
 * index stream (BYTE/SHORT/INT element size) into a flat vertex-id gather,
 * counting complete primitives of `last` vertices and dropping primitive
 * restarts / trailing incomplete groups.  Pure CPU; caller frees
 * result.gather.  Returns 0 on success, -1 on bad args / no valid gather. */
int mglRenderCppGeometryGatherIndices(
    const uint8_t *index_bytes,
    uint32_t index_type_byte_width,   /* 1, 2 or 4 */
    uint32_t count,
    int restart_enabled,
    uint32_t restart_index,
    uint32_t input_vertices,
    MGLRenderCppGeometryGatherResult *out);

typedef struct MGLRenderCppReadTextureRegionClip_t {
    int32_t copy_w;
    int32_t copy_h;
    int32_t dst_x;
    int32_t dst_y;
    int32_t metal_src_x;
    int32_t metal_src_y;
    int empty;   /* copyW <= 0 || copyH <= 0 (nothing to copy) */
} MGLRenderCppReadTextureRegionClip;

/* P4.5 (item 1141/887): readPixels region-vs-level clip — clamps a source
 * read region against the level extents and computes the destination
 * offset-origin for the clipped copy and the Metal source Y (flipped).
 * Pure computation shared by both gates; the empty flag matches the
 * original `copyW <= 0 || copyH <= 0`. */
int mglRenderCppReadTextureRegionClip(
    int64_t region_x, int64_t region_y,
    int64_t region_w, int64_t region_h,
    int64_t level_w, int64_t level_h,
    MGLRenderCppReadTextureRegionClip *out);

typedef struct MGLRenderCppThreadgroupSize_t {
    uint32_t x;   /* local workgroup size with 0 resolved to 1 */
    uint32_t y;
    uint32_t z;
} MGLRenderCppThreadgroupSize;

/* P4.5 (item 1147/887): compute dispatch threadgroup size — resolves a
 * zero local workgroup component to 1 (the `x ? x : 1` default used by the
 * ObjC dispatch fallback).  Pure computation shared by both gates. */
int mglRenderCppThreadgroupSize(
    uint32_t local_x, uint32_t local_y, uint32_t local_z,
    MGLRenderCppThreadgroupSize *out);

typedef struct MGLRenderCppVertexAttribResolve_t {
    int use_binding_table;   /* bindingIndex < limit && binding has buffer */
    int64_t binding_offset;  /* table offset, or attrib binding_offset */
    uint32_t stride;         /* table stride, or attrib stride */
    uint32_t divisor;
} MGLRenderCppVertexAttribResolve;

/* P4.5 (item 1141/887): ARB_vertex_attrib_binding resolve — the
 * binding-table override (offset/stride/divisor) vs the legacy per-attrib
 * values.  Pure decision shared by both gates; the GL buffer validation
 * stays on the ObjC side. */
int mglRenderCppResolveVertexAttribBinding(
    uint32_t binding_index,
    int binding_has_buffer,
    int64_t binding_offset,
    uint32_t binding_stride,
    int64_t attrib_binding_offset,
    uint32_t attrib_stride,
    uint32_t binding_divisor,
    uint32_t attrib_divisor,
    MGLRenderCppVertexAttribResolve *out);

typedef struct MGLRenderCppPolygonOffsetDecision_t {
    int triangle_fill_mode;      /* 0 = fill, 1 = lines */
    int needs_polygon_mode_repair;
    int enable_depth_bias;
} MGLRenderCppPolygonOffsetDecision;

/* P4.5 (item 1141/887): polygon-offset draw decision — the triangle fill
 * mode (GL_LINE -> lines), the invalid polygon-mode repair condition and
 * the depth-bias enablement per polygon mode with the three capability
 * flags.  Pure decision shared by both gates. */
int mglRenderCppPolygonOffsetDecision(
    uint32_t mode,
    int has_ctx,
    int produces_polygons,
    uint32_t polygon_mode,
    int cap_point,
    int cap_line,
    int cap_fill,
    MGLRenderCppPolygonOffsetDecision *out);

/* P4.5 (item 1141/887): GL draw mode -> primitive vertex count (for the
 * cull-distance emulation params; 1 for unknown modes).  Pure table shared
 * by both gates. */
uint32_t mglRenderCppPrimitiveVertexCountForMode(uint32_t mode);

typedef struct MGLRenderCppScaledBlitUVs_t {
    float uv_left;
    float uv_top;
    float uv_right;
    float uv_bottom;
} MGLRenderCppScaledBlitUVs;

typedef struct MGLRenderCppBlitScissorRect_t {
    int64_t x0;
    int64_t x1;
    int64_t y0;
    int64_t y1;
} MGLRenderCppBlitScissorRect;

/* P4.5 (item 1069/1141): scaled-blit UV computation (normalized source
 * rect with the Metal Y-flip, clamped, direction-swapped per the forward
 * flags).  Pure CPU, shared by both gates. */
int mglRenderCppScaledBlitUVs(
    uint32_t src_tex_w,
    uint32_t src_tex_h,
    double src_min_x,
    double src_max_x,
    double src_min_y,
    double src_max_y,
    int src_x_forward,
    int src_y_forward,
    int dst_x_forward,
    int dst_y_forward,
    MGLRenderCppScaledBlitUVs *out);

/* P4.5 (item 1069/1141): scaled-blit destination scissor base — floor/ceil
 * of the destination rect in Metal Y, clamped to the destination texture.
 * The caller intersects the GL scissor box on top.  Pure CPU, shared by
 * both gates. */
int mglRenderCppBlitScissorRect(
    double dst_min_x,
    double dst_max_x,
    double scaled_dst_metal_y,
    double dst_h,
    uint32_t dst_tex_w,
    uint32_t dst_tex_h,
    MGLRenderCppBlitScissorRect *out);

typedef struct MGLRenderCppBlitFramebufferPlan_t {
    int src_x_forward;
    int src_y_forward;
    int dst_x_forward;
    int dst_y_forward;
    int blit_needs_flip;
    int needs_scaled_blit;
    double src_min_x;
    double src_max_x;
    double src_min_y;
    double src_max_y;
    double dst_min_x;
    double dst_max_x;
    double dst_min_y;
    double dst_max_y;
    double src_w;
    double src_h;
    double dst_w;
    double dst_h;
    int64_t copy_src_x;
    int64_t copy_src_y;
    int64_t copy_dst_x;
    int64_t copy_dst_y;
    int64_t copy_w;
    int64_t copy_h;
    int64_t src_metal_y;
    int64_t dst_metal_y;
    double scaled_dst_metal_y;
} MGLRenderCppBlitFramebufferPlan;

/* P4.5 (item 1069/1141): glBlitFramebuffer region math + decisions after
 * the axis clip — direction/flip flags, min/max/abs extents, the scaled-
 * blit decision (format conversion / RT sync / scissor / flip / size
 * mismatch with the 1e-5 epsilon of mglNearlyEqual), the integer copy
 * rect, the Metal Y-flips and the scaled-path destination Y.  Pure CPU
 * plan shared by both gates.  Returns 0 with the plan filled, -1 when the
 * clipped region has zero extent (caller logs and skips). */
int mglRenderCppBlitFramebufferPlan(
    double src_x0,
    double src_x1,
    double src_y0,
    double src_y1,
    double dst_x0,
    double dst_x1,
    double dst_y0,
    double dst_y1,
    uint32_t src_tex_w,
    uint32_t src_tex_h,
    uint32_t dst_tex_w,
    uint32_t dst_tex_h,
    int needs_format_conversion_blit,
    int needs_render_target_sync_blit,
    int scissor_test_enabled,
    MGLRenderCppBlitFramebufferPlan *out);

typedef struct MGLRenderCppGetTexImagePlan_t {
    int direct_r32_float_read;
    int use_bgra8_conversion;
    int source_is_bgra8;
    uint64_t row_bytes;
    uint64_t image_bytes;
    uint64_t total_bytes;
} MGLRenderCppGetTexImagePlan;

/* P4.5 (item 1171/1116): mtlGetTexImage staging plan — direct R32F read
 * detection, the BGRA8 conversion eligibility (dst bytes + single depth
 * layer + non-direct + compatible source), the source-is-BGRA8-family
 * check, and the row/image/total byte computation (conversion pitch:
 * width*sourceBpp for non-BGRA8 sources, width*4 for BGRA8 sources;
 * otherwise bytesPerRow or width*max(dst,1); the depth>1 + bytesPerImage
 * case applies to private storage only).  Shared by both gates; the caller
 * resolves sizeForFormatType / readback bytes-per-pixel / format
 * compatibility through the existing C helpers. */
int mglRenderCppGetTexImagePlan(
    uint32_t pixel_format,
    uint32_t gl_format,
    uint32_t gl_type,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    uint32_t dst_pixel_bytes,
    uint32_t source_bpp,
    int bgra8_format_compatible,
    uint32_t bytes_per_row,
    uint32_t bytes_per_image,
    int storage_private,
    MGLRenderCppGetTexImagePlan *out);

typedef struct MGLRenderCppLevelUploadOp_t {
    uint32_t level;
    uint32_t kind;          /* 0 = upload op, 1 = short-backing (skip) */
    uint32_t width;
    uint32_t height;
    uint64_t bytes_per_row;
    uint64_t bytes_per_image;
    uint64_t copy_depth;
    uint64_t available_bytes; /* short-backing: bytes available */
    uint64_t needed_bytes;    /* short-backing: bytes_per_image * copy_depth */
    const void *data;         /* upload op: borrowed or owned (owns_data) */
    int owns_data;
} MGLRenderCppLevelUploadOp;

/* P4.5 (item 1116): build the level-upload op list for a single-face
 * (2D) CPU upload — inlines the has-uploadable CPU-data check, runs
 * mglRenderCppTexturePrepareLevelUpload per level and classifies each as
 * upload op / short-backing / bad.  levels must have level_count entries.
 * Returns 0 with *op_count_out ops (capacity must hold level_count), or -1
 * on bad args / capacity overflow.  short-backing ops carry kind=1 with the
 * have/need bytes; bad levels are counted in *bad_out (skipped silently,
 * matching the ObjC baseline). */
int mglRenderCppBuildLevelUploadOps(
    const TextureLevel *levels,
    uint32_t level_count,
    uint32_t texture_type,
    uint32_t internal_format,
    uint32_t pixel_format,
    MGLRenderCppLevelUploadOp *ops,
    uint32_t ops_capacity,
    uint32_t *op_count_out,
    uint32_t *short_backing_out,
    uint32_t *bad_out);

typedef struct MGLRenderCppLevelUploadPrep_t {
    const void *data;         /* borrowed or owned */
    uint64_t bytes_per_row;
    uint64_t bytes_per_image;
    uint64_t copy_depth;
    uint64_t available_bytes;
    int owns_data;            /* 1: caller must free((void *)data) */
} MGLRenderCppLevelUploadPrep;

int mglRenderCppTexturePrepareLevelUpload(
    const TextureLevel *level,
    uint32_t texture_type,
    uint32_t internal_format,
    uint32_t pixel_format,
    MGLRenderCppLevelUploadPrep *out);

/* P4.5 (item 1111): RGB → RGBA channel expansion (RGBA16/RGBA32 family
 * backed by RGBA variants) — the table + verification moved from
 * mgl_texture_compat.m; malloc'd result, NULL on bad args / unknown format. */
uint8_t *mglRenderCppCreateChannelExpandedUpload(uint32_t internal_format,
                                                 uint32_t pixel_format,
                                                 const void *src_data,
                                                 size_t width,
                                                 size_t height,
                                                 size_t src_bytes_per_row,
                                                 size_t *out_bytes_per_row,
                                                 size_t *out_bytes_per_image);
uint8_t *mglRenderCppCreateRGBA8ExpandedUpload(const void *src_data,
                                               size_t width,
                                               size_t height,
                                               size_t src_bytes_per_row,
                                               uint32_t internal_format,
                                               size_t *out_bytes_per_row,
                                               size_t *out_bytes_per_image);
/* P4.5 (item 1111): RGB-family → RGBA expansion gates.  Pixel format is
 * the Apple MTLPixelFormat numeric value.  Returns 1/0. */
int mglRenderCppTextureInternalFormatNeedsRGBA8Expansion(
    uint32_t internal_format, uint32_t pixel_format);
int mglRenderCppTextureNeedsChannelExpansion(uint32_t internal_format,
                                             uint32_t pixel_format);

/* P4.5 (item 1111): R8 swizzle component + single-channel upload expand.
 * Resolve mirrors mglResolveR8SwizzledComponent (tex unused).  Create
 * expands GL_R8 1B/px → RGBA8 via the four swizzle enums; malloc'd
 * result, NULL on bad args / non-R8 / size cap. */
uint8_t mglRenderCppResolveR8SwizzledComponent(uint32_t swizzle, uint8_t red);
/* P4.5 (item 1111): R-only upload-swizzle gate.  swizzled==0 → 0;
 * otherwise the GL_R* internal-format table.  Returns 1/0. */
int mglRenderCppTextureUploadNeedsSingleChannelSwizzle(uint32_t internal_format,
                                                       int swizzled);
/* P4.5 (item 1111): stored color-component count for an internal format.
 * Mirrors mglStoredColorComponentsForTexture after the null-tex check
 * (null stays in ObjC and returns 4).  Unknown formats → 4. */
uint32_t mglRenderCppStoredColorComponents(uint32_t internal_format);
/* P4.5 (item 1111): GL swizzle enum → Metal TextureSwizzle ABI value
 * (uint32_t).  components gates missing channels to Zero / One(for Alpha). */
uint32_t mglRenderCppMTLSwizzleForGLSwizzle(uint32_t gl_swizzle,
                                            uint32_t components);
uint8_t *mglRenderCppCreateSingleChannelSwizzledUpload(
    uint32_t internal_format,
    uint32_t swizzle_r, uint32_t swizzle_g,
    uint32_t swizzle_b, uint32_t swizzle_a,
    const void *src_data, size_t width, size_t height,
    size_t src_bytes_per_row,
    size_t *out_bytes_per_row, size_t *out_bytes_per_image);
int mglRenderCppCreateSampler(void *sampler_descriptor,
                              void **sampler_out);
/* Translate GL texture parameters into a Metal-cpp sampler descriptor and
 * create the sampler without exposing MTL::* through this C ABI. */
int mglRenderCppCreateSamplerForGL(const TextureParameter *params,
                                   uint32_t target,
                                   void **sampler_out,
                                   char *err,
                                   size_t errcap);
int mglRenderCppCreateDepthStencilState(void *depth_stencil_descriptor,
                                        void **depth_stencil_state_out);

enum {
    MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS = 7,
    MGL_RENDER_CPP_PIPELINE_COLOR_ATTACHMENTS = 8,
};

typedef struct MGLRenderCppStencilDescriptorState_t {
    uint32_t present;
    uint32_t compare_function;
    uint32_t read_mask;
    uint32_t write_mask;
    uint32_t stencil_failure_operation;
    uint32_t depth_failure_operation;
    uint32_t depth_stencil_pass_operation;
} MGLRenderCppStencilDescriptorState;

typedef struct MGLRenderCppDepthStencilDescriptorState_t {
    uint32_t depth_compare_function;
    uint32_t depth_write_enabled;
    MGLRenderCppStencilDescriptorState front;
    MGLRenderCppStencilDescriptorState back;
} MGLRenderCppDepthStencilDescriptorState;

int mglRenderCppCreateDepthStencilStateFromState(
    const MGLRenderCppDepthStencilDescriptorState *descriptor,
    void **depth_stencil_state_out);

typedef struct MGLRenderCppPipelineActiveState_t {
    void *pipeline_state;
    void *vertex_function;
    void *fragment_function;
    uint32_t color0_format;
    uint32_t depth_format;
    uint32_t stencil_format;
    uint32_t program_name;
} MGLRenderCppPipelineActiveState;

typedef struct MGLRenderCppPipelineBlendState_t {
    uint32_t source_rgb_factor;
    uint32_t destination_rgb_factor;
    uint32_t source_alpha_factor;
    uint32_t destination_alpha_factor;
    uint32_t rgb_operation;
    uint32_t alpha_operation;
    uint32_t color_write_mask;
} MGLRenderCppPipelineBlendState;

/* Per-renderer pipeline ownership. The opaque owner retains active objects,
 * cached PSOs/functions/descriptors, and depth-stencil states. All returned
 * object pointers are borrowed for the lifetime of the owner/cache entry. */
int mglRenderCppCreatePipelineCacheOwner(
    int pso_dedup_enabled,
    int depth_stencil_cache_enabled,
    int binary_archive_enabled,
    void **owner_out);
void mglRenderCppDestroyPipelineCacheOwner(void **owner);
void mglRenderCppResetPipelineCacheOwner(void *owner);
int mglRenderCppGetPipelineCacheFlags(
    void *owner,
    int *pso_dedup_enabled_out,
    int *depth_stencil_cache_enabled_out,
    int *binary_archive_enabled_out);
void mglRenderCppDisablePipelineBinaryArchive(void *owner);
int mglRenderCppGetPipelineBinaryArchiveState(
    void *owner, int *enabled_out, int *present_out);
int mglRenderCppLoadPipelineBinaryArchive(
    void *owner,
    const char *cache_key,
    void *url,
    int archive_exists,
    int *reused_out,
    char *err,
    size_t errcap);
int mglRenderCppSerializePipelineBinaryArchive(
    void *owner, void *url, char *err, size_t errcap);
void mglRenderCppDiscardPipelineBinaryArchive(
    void *owner, const char *cache_key);
int mglRenderCppGetPipelineActiveState(
    void *owner, MGLRenderCppPipelineActiveState *state_out);
int mglRenderCppInvalidatePipelineActiveState(void *owner);
int mglRenderCppSetPipelineActiveObject(void *owner, void *pipeline_state);
int mglRenderCppActivatePipelineState(
    void *owner, const MGLRenderCppPipelineActiveState *state);
int mglRenderCppSetPipelineBlendState(
    void *owner, uint32_t attachment,
    const MGLRenderCppPipelineBlendState *state);
int mglRenderCppGetPipelineBlendState(
    void *owner, uint32_t attachment,
    MGLRenderCppPipelineBlendState *state_out);
int mglRenderCppGetOrCreateDepthStencilState(
    void *owner,
    const MGLRenderCppDepthStencilDescriptorState *descriptor,
    void **depth_stencil_state_out,
    int *created_out);
int mglRenderCppLookupPipeline(
    void *owner,
    const uint64_t key_words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS],
    MGLRenderCppPipelineActiveState *state_out);
int mglRenderCppStorePipeline(
    void *owner,
    const uint64_t key_words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS],
    const MGLRenderCppPipelineActiveState *state,
    uint32_t *evicted_out);
/* P4.2: descriptor cache（value-state 版）。缓存 MGLRenderCppPipelineDescriptorState
 * 值，命中时 ObjC 无需重新组装 descriptor state。旧 pointer-based
 * LookupPipelineDescriptor / StorePipelineDescriptor 已删除。 */
int mglRenderCppLookupPipelineDescriptorState(
    void *owner,
    const uint64_t key_words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS],
    MGLRenderCppPipelineDescriptorState *state_out);
int mglRenderCppStorePipelineDescriptorState(
    void *owner,
    const uint64_t key_words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS],
    const MGLRenderCppPipelineDescriptorState *state);
int mglRenderCppCreateEvent(void **event_out);
int mglRenderCppCreateFunction(void *library,
                               const char *name,
                               void *function_constant_values,
                               void **function_out,
                               char *err,
                               size_t errcap);
int mglRenderCppCreateRenderPipelineState(
    void *render_pipeline_descriptor,
    void **pipeline_out,
    char *err,
    size_t errcap);
/* Descriptor-based archive-aware creation used by the temporary ObjC
 * descriptor paths. Complete VS+FS pipelines query the archive first and are
 * added only on a miss. archive_hit_out is optional and receives 1 only when
 * the returned PSO came directly from the archive. */
int mglRenderCppCreateRenderPipelineStateWithArchive(
    void *render_pipeline_descriptor,
    void *binary_archive,
    void **pipeline_out,
    int *archive_hit_out,
    char *err,
    size_t errcap);
int mglRenderCppCreateRenderPipelineStateWithArchiveOwner(
    void *owner,
    void *render_pipeline_descriptor,
    void **pipeline_out,
    int *archive_hit_out,
    char *err,
    size_t errcap);
/* P4.2: final/simple/safe descriptor builder 的 C ABI 入口 —— 从
 * MGLRenderCppPipelineDescriptorState value-state 直接创建 render PSO，
 * ObjC 不再组装 MTLRenderPipelineDescriptor。vs_function/fs_function 为 +0
 * borrowed MTL::Function*；binary_archive 为 +0 borrowed MTL::BinaryArchive*
 * （可为 NULL）。深度/模板 packed normalize 与 MGL_ENABLE_ICB_PIPELINES
 * opt-in 在 C++ builder 内完成。成功返回 0 且 *pipeline_out 为 +1 引用
 * （mglAirRelease 释放）。 */
int mglRenderCppCreateRenderPipelineFromState(
    void *vs_function,
    void *fs_function,
    const MGLRenderCppPipelineDescriptorState *state,
    void *binary_archive,
    void **pipeline_out,
    char *err,
    size_t errcap);
int mglRenderCppCreateRenderPipelineFromStateWithArchiveOwner(
    void *owner,
    void *vs_function,
    void *fs_function,
    const MGLRenderCppPipelineDescriptorState *state,
    void **pipeline_out,
    char *err,
    size_t errcap);
int mglRenderCppCreateComputePipelineState(void *function,
                                           void **pipeline_out,
                                           char *err,
                                           size_t errcap);
int mglRenderCppCreateBinaryArchive(void *binary_archive_descriptor,
                                    const char *label,
                                    void **binary_archive_out,
                                    char *err,
                                    size_t errcap);
int mglRenderCppSerializeBinaryArchive(void *binary_archive,
                                       void *url,
                                       char *err,
                                       size_t errcap);
int mglRenderCppSetVisibilityResultMode(void *render_encoder,
                                        uint32_t mode,
                                        uint64_t offset);
int mglRenderCppSetVisibilityResultModeForRenderEncoderOwner(
    void *render_encoder_owner,
    uint32_t mode,
    uint64_t offset);
int mglRenderCppSampleTimestamps(uint64_t *cpu_timestamp_out,
                                 uint64_t *gpu_timestamp_out);
int mglRenderCppCreateQueryStateOwner(uint32_t visibility_slot_count,
                                      void **owner_out);
int mglRenderCppBeginSampleQuery(void *owner,
                                 uint32_t counting,
                                 const char *buffer_label,
                                 void **visibility_buffer_out);
int mglRenderCppGetQueryVisibilityBuffer(void *owner,
                                         void **visibility_buffer_out);
void mglRenderCppEndSampleQuery(void *owner);
int mglRenderCppIsSampleQueryActive(void *owner, uint32_t *active_out);
int mglRenderCppAcquireSampleQuerySlot(void *owner,
                                       uint32_t *mode_out,
                                       uint64_t *offset_out);
int mglRenderCppGetSampleQueryResult(void *owner, uint64_t *result_out);
int mglRenderCppBeginTimerQuery(void *owner);
int mglRenderCppEndTimerQuery(void *owner, uint64_t *elapsed_out);
void mglRenderCppDestroyQueryStateOwner(void **owner);

/* Create or reuse a compute PSO owned by the C++ renderer. function is the
 * actual MTLFunction selected by the caller, preserving AIR stage variants.
 * On success *pipeline_out is a +1 MTLComputePipelineState reference that the
 * ObjC bridge may consume with __bridge_transfer. */
int mglRenderCppGetOrCreateComputePipeline(
    void *function,
    uint64_t program_instance,
    uint64_t program_generation,
    uint32_t stage,
    int cache_enabled,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Drop all C++ compute PSOs for a Program lifetime. Called before relink and
 * final Program destruction; safe before renderer initialization. */
void mglRenderCppInvalidateProgramPipelines(uint64_t program_instance);

enum {
    MGL_RENDER_CPP_AUX_COMPUTE_SCALED_BLIT = 1,
    MGL_RENDER_CPP_AUX_COMPUTE_MSAA_INTEGER_RESOLVE = 2,
    MGL_RENDER_CPP_AUX_RENDER_SCALED_BLIT = 3,
    MGL_RENDER_CPP_AUX_RENDER_SCALED_DEPTH_BLIT = 4,
    MGL_RENDER_CPP_AUX_RENDER_CLEAR_RECT = 5,
};

/* Lookup or create a renderer-lifetime auxiliary compute PSO. Passing a NULL
 * function performs lookup only and returns 1 on a cache miss. On success the
 * returned pipeline is an independent +1 reference. */
int mglRenderCppGetOrCreateAuxComputePipeline(
    void *function,
    uint32_t kind,
    uint64_t variant,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Lookup or create a renderer-lifetime auxiliary render PSO. Passing NULL for
 * both functions performs lookup only and returns 1 on a cache miss. The
 * descriptor contains the fixed-format blit/clear surface state; functions
 * are the actual MTLFunction objects compiled by the caller. */
int mglRenderCppGetOrCreateAuxRenderPipeline(
    void *vertex_function,
    void *fragment_function,
    uint32_t kind,
    uint64_t variant,
    uint32_t color_format,
    uint32_t depth_format,
    uint32_t stencil_format,
    uint32_t color_write_mask,
    int icb_enabled,
    uint32_t raster_sample_count,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Aux render PSO from the precompiled aux shader asset table
 * (see mgl_aux_assets.h). bytes/size/hash come from an embedded table row;
 * the C++ side validates size and the FNV-1a hash, loads MTL::Library from the
 * bytes, resolves the entry functions, and creates the PSO through the same
 * renderer-lifetime cache as the function-based path. vertex_entry is the
 * metallib entry name; fragment_entry may be NULL for fragment-less kinds.
 * On success *pipeline_out is a +1 MTL::RenderPipelineState reference. */
int mglRenderCppGetOrCreateAuxRenderPipelineFromMetallib(
    const unsigned char *bytes,
    size_t size,
    uint64_t asset_hash,
    const char *vertex_entry,
    const char *fragment_entry,
    uint32_t kind,
    uint64_t variant,
    uint32_t color_format,
    uint32_t depth_format,
    uint32_t stencil_format,
    uint32_t color_write_mask,
    int icb_enabled,
    uint32_t raster_sample_count,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Aux compute PSO from the precompiled aux shader asset table. entry_name is
 * the metallib kernel name. On success *pipeline_out is a +1
 * MTL::ComputePipelineState reference. */
int mglRenderCppGetOrCreateAuxComputePipelineFromMetallib(
    const unsigned char *bytes,
    size_t size,
    uint64_t asset_hash,
    const char *entry_name,
    uint32_t kind,
    uint64_t variant,
    void **pipeline_out,
    char *err,
    size_t errcap);

/* Resolve entry functions from a precompiled aux shader asset for descriptor
 * paths that keep ObjC descriptor assembly (e.g. the safe fallback branch).
 * vertex_out is always a +1 MTL::Function; fragment_out is +1 when
 * fragment_entry is non-NULL. The underlying library is cached by the C++
 * renderer and released at shutdown. */
int mglRenderCppCreateAuxFunctions(
    const unsigned char *bytes,
    size_t size,
    uint64_t asset_hash,
    const char *vertex_entry,
    const char *fragment_entry,
    void **vertex_out,
    void **fragment_out,
    char *err,
    size_t errcap);

enum {
    MGL_RENDER_CPP_BINDING_STAGE_VERTEX = 0,
    MGL_RENDER_CPP_BINDING_STAGE_FRAGMENT = 1,
};

enum {
    MGL_RENDER_CPP_BINDING_VERTEX_TEXTURE = 0,
    MGL_RENDER_CPP_BINDING_FRAGMENT_TEXTURE = 1,
    MGL_RENDER_CPP_BINDING_VERTEX_SAMPLER = 2,
    MGL_RENDER_CPP_BINDING_FRAGMENT_SAMPLER = 3,
    MGL_RENDER_CPP_BINDING_VIEWPORT = 4,
    MGL_RENDER_CPP_BINDING_SCISSOR = 5,
    MGL_RENDER_CPP_BINDING_TRIANGLE_FILL = 6,
    MGL_RENDER_CPP_BINDING_SETTER_COUNT = 7,
};

typedef struct MGLRenderCppBindingStats {
    uint64_t emitted[MGL_RENDER_CPP_BINDING_SETTER_COUNT];
    uint64_t skipped[MGL_RENDER_CPP_BINDING_SETTER_COUNT];
} MGLRenderCppBindingStats;

/* Per-renderer-context binding dedup state. Metal objects stored in this
 * handle are retained by C++ and released on replacement, invalidation, or
 * destroy. Setter calls return 1 when encoded, 0 when deduplicated, and -1
 * for invalid arguments. */
void *mglRenderCppBindingCreate(uint32_t max_texture_slots);
void mglRenderCppBindingDestroy(void *binding_state);
void mglRenderCppBindingInvalidate(void *binding_state);
void mglRenderCppBindingSetValid(void *binding_state, int valid);
int mglRenderCppBindingGetValid(void *binding_state, uint32_t *valid_out);
int mglRenderCppBindingGetTextureSlotMask(void *binding_state,
                                          uint64_t mask_out[2]);
int mglRenderCppBindingRecordVertexBuffer(void *binding_state,
                                          void *buffer,
                                          uint64_t offset,
                                          uint32_t index);
int mglRenderCppBindingRecordFragmentBuffer(void *binding_state,
                                            void *buffer,
                                            uint64_t offset,
                                            uint32_t index);
int mglRenderCppBindingInvalidateVertexBuffer(void *binding_state,
                                              uint32_t index);
int mglRenderCppBindingInvalidateFragmentBuffer(void *binding_state,
                                                uint32_t index);
int mglRenderCppBindingUpdateVertexBuffer(void *binding_state,
                                          void *buffer,
                                          uint64_t offset,
                                          uint32_t index);
int mglRenderCppBindingUpdateFragmentBuffer(void *binding_state,
                                            void *buffer,
                                            uint64_t offset,
                                            uint32_t index);
int mglRenderCppBindingClearVertexBuffer(void *binding_state,
                                         uint32_t index);
int mglRenderCppBindingClearFragmentBuffer(void *binding_state,
                                           uint32_t index);
int mglRenderCppBindingGetBuffer(void *binding_state,
                                 uint32_t stage,
                                 uint32_t index,
                                 void **buffer_out,
                                 uint64_t *offset_out);
void mglRenderCppBindingOrVertexBufferMask(void *binding_state,
                                           uint32_t mask);
void mglRenderCppBindingOrFragmentBufferMask(void *binding_state,
                                             uint32_t mask);
void mglRenderCppBindingSetPipelineState(void *binding_state,
                                         void *pipeline_state);
void mglRenderCppBindingSetDepthStencilState(void *binding_state,
                                             void *depth_stencil_state);
int mglRenderCppBindingGetPipelineState(void *binding_state,
                                        void **pipeline_state_out);
int mglRenderCppBindingGetDepthStencilState(
    void *binding_state, void **depth_stencil_state_out);
void mglRenderCppBindingSetCullMode(void *binding_state, uint32_t mode);
void mglRenderCppBindingSetWinding(void *binding_state, uint32_t winding);
void mglRenderCppBindingSetDepthBias(void *binding_state,
                                     float bias,
                                     float clamp,
                                     float slope_scale);
void mglRenderCppBindingSetBlendColor(void *binding_state,
                                      float red,
                                      float green,
                                      float blue,
                                      float alpha);
int mglRenderCppBindingSetPipelineIfNeeded(void *binding_state,
                                           void *render_encoder,
                                           void *pipeline_state);
int mglRenderCppBindingSetDepthStencilIfNeeded(void *binding_state,
                                               void *render_encoder,
                                               void *depth_stencil_state);
int mglRenderCppBindingSetCullIfNeeded(void *binding_state,
                                       void *render_encoder,
                                       uint32_t mode);
int mglRenderCppBindingSetWindingIfNeeded(void *binding_state,
                                          void *render_encoder,
                                          uint32_t winding);
int mglRenderCppBindingSetDepthBiasIfNeeded(void *binding_state,
                                            void *render_encoder,
                                            float bias,
                                            float clamp,
                                            float slope_scale);
int mglRenderCppBindingSetBlendColorIfNeeded(void *binding_state,
                                             void *render_encoder,
                                             float red,
                                             float green,
                                             float blue,
                                             float alpha);
int mglRenderCppBindingSetPipelineIfNeededForOwner(
    void *binding_state, void *render_encoder_owner, void *pipeline_state);
int mglRenderCppBindingSetDepthStencilIfNeededForOwner(
    void *binding_state, void *render_encoder_owner,
    void *depth_stencil_state);
int mglRenderCppBindingSetCullIfNeededForOwner(
    void *binding_state, void *render_encoder_owner, uint32_t mode);
int mglRenderCppBindingSetWindingIfNeededForOwner(
    void *binding_state, void *render_encoder_owner, uint32_t winding);
int mglRenderCppBindingSetBlendColorIfNeededForOwner(
    void *binding_state, void *render_encoder_owner,
    float red, float green, float blue, float alpha);
int mglRenderCppBindingSetTexture(void *binding_state,
                                 void *render_encoder,
                                 void *texture,
                                 uint32_t stage,
                                 uint32_t index);
int mglRenderCppBindingSetSampler(void *binding_state,
                                 void *render_encoder,
                                 void *sampler,
                                 uint32_t stage,
                                 uint32_t index);
int mglRenderCppBindingSetTextureForOwner(void *binding_state,
                                         void *render_encoder_owner,
                                         void *texture,
                                         uint32_t stage,
                                         uint32_t index);
int mglRenderCppBindingSetSamplerForOwner(void *binding_state,
                                         void *render_encoder_owner,
                                         void *sampler,
                                         uint32_t stage,
                                         uint32_t index);
int mglRenderCppBindingSetDepthBiasIfNeededForOwner(
    void *binding_state,
    void *render_encoder_owner,
    float depth_bias,
    float clamp,
    float slope_scale);
int mglRenderCppBindingGetTexture(void *binding_state,
                                  uint32_t stage,
                                  uint32_t index,
                                  void **texture_out);
int mglRenderCppBindingGetSampler(void *binding_state,
                                  uint32_t stage,
                                  uint32_t index,
                                  void **sampler_out);
int mglRenderCppBindingSetViewport(void *binding_state,
                                  void *render_encoder,
                                  double origin_x,
                                  double origin_y,
                                  double width,
                                  double height,
                                  double znear,
                                  double zfar);
/* Array viewport binding (gl_ViewportIndex): viewports carries count
 * interleaved {x, y, w, h, znear, zfar} tuples, count <= 16. */
int mglRenderCppBindingSetViewports(void *binding_state,
                                    void *render_encoder,
                                    const double *viewports,
                                    uint64_t count);
int mglRenderCppBindingSetViewportsForOwner(void *binding_state,
                                            void *render_encoder_owner,
                                            const double *viewports,
                                            uint64_t count);
int mglRenderCppBindingSetScissor(void *binding_state,
                                 void *render_encoder,
                                 uint64_t x,
                                 uint64_t y,
                                 uint64_t width,
                                 uint64_t height);
int mglRenderCppBindingSetTriangleFill(void *binding_state,
                                      void *render_encoder,
                                      uint32_t mode);
int mglRenderCppBindingSetViewportForOwner(void *binding_state,
                                          void *render_encoder_owner,
                                          double origin_x,
                                          double origin_y,
                                          double width,
                                          double height,
                                          double znear,
                                          double zfar);
int mglRenderCppBindingSetScissorForOwner(void *binding_state,
                                         void *render_encoder_owner,
                                         uint64_t x,
                                         uint64_t y,
                                         uint64_t width,
                                         uint64_t height);
int mglRenderCppBindingSetTriangleFillForOwner(void *binding_state,
                                              void *render_encoder_owner,
                                              uint32_t mode);
int mglRenderCppBindingGetStats(void *binding_state,
                               MGLRenderCppBindingStats *stats_out);

/* Compute encoder setter facade.  These entry points intentionally do not
 * retain resources: the command encoder owns the encoded references, matching
 * Objective-C Metal semantics.  Return 0 on success and -1 for bad inputs. */
int mglRenderCppSetComputePipelineState(void *compute_encoder,
                                        void *pipeline_state);
int mglRenderCppSetComputeBuffer(void *compute_encoder,
                                 void *buffer,
                                 uint64_t offset,
                                 uint32_t index);
int mglRenderCppSetComputeTexture(void *compute_encoder,
                                  void *texture,
                                  uint32_t index);
int mglRenderCppSetComputeSampler(void *compute_encoder,
                                  void *sampler,
                                  uint32_t index);
int mglRenderCppSetComputeBytes(void *compute_encoder,
                                const void *bytes,
                                size_t length,
                                uint32_t index);
int mglRenderCppSetComputeThreadgroupMemoryLength(void *compute_encoder,
                                                  uint64_t length,
                                                  uint32_t index);

/* P4.5 compute 绑定 snapshot：与 render binding snapshot 同构的 op 列表，
 * 专用于 compute encoder。kind 0 = setBuffer（NULL = 槽位清除）、
 * 1 = setBytes（对称性提供）、2 = setTexture、3 = setSamplerState；
 * texture/sampler op 的对象指针放 buffer 字段。契约与
 * mglRenderCppEncodeBindingSnapshot 一致：调用方预校验，本函数对坏 kind /
 * NULL bytes / 越界计数返回 -1。临时对象（__bridge_transfer 局部）必须在
 * emit 后立即 flush（编码器当场 retain），禁止悬垂进延迟重放。 */
#define MGL_RENDER_CPP_COMPUTE_BINDING_SNAPSHOT_MAX_OPS 32u

typedef struct MGLRenderCppComputeBindingOp_t {
    uint32_t kind;      /* 0 = buffer, 1 = bytes, 2 = texture, 3 = sampler */
    uint32_t index;     /* Metal slot */
    uint64_t offset;    /* kind 0: byte offset */
    void *buffer;       /* kind 0/2/3: borrowed MTL object (NULL = clear) */
    const void *bytes;  /* kind 1: borrowed byte pointer */
    uint32_t length;    /* kind 1: byte length */
} MGLRenderCppComputeBindingOp;

typedef struct MGLRenderCppComputeBindingSnapshot_t {
    uint32_t op_count;
    MGLRenderCppComputeBindingOp
        ops[MGL_RENDER_CPP_COMPUTE_BINDING_SNAPSHOT_MAX_OPS];
} MGLRenderCppComputeBindingSnapshot;

int mglRenderCppEncodeComputeBindingSnapshot(
    void *compute_encoder,
    const MGLRenderCppComputeBindingSnapshot *snapshot,
    char *err,
    size_t errcap);
int mglRenderCppDispatchCompute(void *compute_encoder,
                                uint32_t groups_x,
                                uint32_t groups_y,
                                uint32_t groups_z,
                                uint32_t threads_x,
                                uint32_t threads_y,
                                uint32_t threads_z);
int mglRenderCppDispatchComputeIndirect(void *compute_encoder,
                                        void *indirect_buffer,
                                        uint64_t indirect_offset,
                                        uint32_t threads_x,
                                        uint32_t threads_y,
                                        uint32_t threads_z);

/* P4.5 compute 首切片：dispatch 参数 value-state plan。ObjC 只组装纯值
 * （groups + 未解析的 local size），C++ 内把 local size 0 解析为 1（与
 * mtlDispatchCompute 的 `x ? x : 1` 默认一致）并一次完成
 * dispatchThreadgroups / dispatchThreadgroupsWithIndirectBuffer 编码。
 * 为 item 1138 的「ObjC 只传 MGLRenderCppComputePlan value-state」定型。 */
#define MGL_RENDER_CPP_COMPUTE_DISPATCH_DIRECT   0
#define MGL_RENDER_CPP_COMPUTE_DISPATCH_INDIRECT 1

typedef struct MGLRenderCppComputePlan_t {
    uint32_t dispatch_kind;   /* DIRECT / INDIRECT */
    uint32_t groups_x;
    uint32_t groups_y;
    uint32_t groups_z;
    uint32_t local_x;         /* 0 → C++ 解析为 1 */
    uint32_t local_y;
    uint32_t local_z;
    void *indirect_buffer;    /* INDIRECT: borrowed MTL::Buffer* */
    uint64_t indirect_offset; /* INDIRECT: 参数块字节偏移 */
} MGLRenderCppComputePlan;

int mglRenderCppDispatchComputePlan(
    void *compute_encoder,
    const MGLRenderCppComputePlan *plan,
    char *err,
    size_t errcap);

/* P4.5 compute execution plan: ObjC collects the ordered binding operations
 * and keeps temporary Metal objects alive until this call returns. C++ owns
 * encoder creation, pipeline/binding replay, dispatch, and endEncoding. */
#define MGL_RENDER_CPP_COMPUTE_EXECUTION_MAX_OPS 512u
#define MGL_RENDER_CPP_COMPUTE_EXECUTION_MAX_DISPATCHES 128u

typedef struct MGLRenderCppComputeDispatchEntry_t {
    /* Replay this dispatch after exactly binding_op_count binding operations. */
    uint32_t binding_op_count;
    MGLRenderCppComputePlan dispatch;
} MGLRenderCppComputeDispatchEntry;

typedef struct MGLRenderCppComputeExecutionPlan_t {
    void *pipeline; /* +0 borrowed MTL::ComputePipelineState* */
    uint32_t binding_op_count;
    MGLRenderCppComputeBindingOp
        binding_ops[MGL_RENDER_CPP_COMPUTE_EXECUTION_MAX_OPS];
    uint32_t dispatch_op_count;
    MGLRenderCppComputeDispatchEntry
        dispatch_ops[MGL_RENDER_CPP_COMPUTE_EXECUTION_MAX_DISPATCHES];
    /* Backward-compatible single-dispatch form used when dispatch_op_count=0. */
    MGLRenderCppComputePlan dispatch;
    uint32_t barrier_scope;
} MGLRenderCppComputeExecutionPlan;

/* Value-state barrier request. These values intentionally mirror Metal's
 * BarrierScope bit values without exposing MTL::* through the C ABI. */
enum {
    MGL_RENDER_CPP_COMPUTE_BARRIER_NONE = 0u,
    MGL_RENDER_CPP_COMPUTE_BARRIER_BUFFERS = 1u,
    MGL_RENDER_CPP_COMPUTE_BARRIER_TEXTURES = 2u,
    MGL_RENDER_CPP_COMPUTE_BARRIER_RENDER_TARGETS = 4u,
};

int mglRenderCppAppendComputeBindingSnapshotToPlan(
    MGLRenderCppComputeExecutionPlan *plan,
    const MGLRenderCppComputeBindingSnapshot *snapshot,
    char *err,
    size_t errcap);
int mglRenderCppAppendComputeDispatchToPlan(
    MGLRenderCppComputeExecutionPlan *plan,
    const MGLRenderCppComputePlan *dispatch,
    char *err,
    size_t errcap);

int mglRenderCppEncodeComputeExecutionPlanForCommandBufferOwner(
    void *command_buffer_owner,
    const MGLRenderCppComputeExecutionPlan *plan,
    char *err,
    size_t errcap);

/* P4.3e: GS/TES compute dispatch 编排的固定序列（建 encoder → pipeline →
 * ABI 槽位 buffer/bytes）一次交给 C++；GL 资源绑定（stage buffers/textures）
 * 在 begin/end 之间由 ObjC 完成（只经 C++ facade）。与逐条
 * mglRenderCppSetCompute* / DispatchCompute / EndComputeEncoder 完全等价。 */
#define MGL_RENDER_CPP_COMPUTE_DISPATCH_MAX_BUFFERS 16u
#define MGL_RENDER_CPP_COMPUTE_DISPATCH_MAX_BYTES 4u

typedef struct MGLRenderCppComputeBufferEntry_t {
    void *buffer;   /* +0 borrowed MTL::Buffer* */
    uint64_t offset;
    uint32_t index;
} MGLRenderCppComputeBufferEntry;

typedef struct MGLRenderCppComputeBytesEntry_t {
    const void *bytes;
    uint32_t length;
    uint32_t index;
} MGLRenderCppComputeBytesEntry;

typedef struct MGLRenderCppComputeDispatchSetup_t {
    void *pipeline;             /* +0 borrowed MTL::ComputePipelineState* */
    uint32_t buffer_count;
    MGLRenderCppComputeBufferEntry
        buffers[MGL_RENDER_CPP_COMPUTE_DISPATCH_MAX_BUFFERS];
    uint32_t bytes_count;
    MGLRenderCppComputeBytesEntry
        bytes[MGL_RENDER_CPP_COMPUTE_DISPATCH_MAX_BYTES];
} MGLRenderCppComputeDispatchSetup;

/* begin：创建 compute encoder（command_buffer 当前 CB）+ setComputePipelineState
 * + 绑定 setup 内全部 buffer/bytes。*compute_encoder_out 为 +0 borrowed
 * （command buffer 持有 encoder）。失败返回 -1。 */
int mglRenderCppBeginComputeDispatch(
    void *command_buffer,
    const MGLRenderCppComputeDispatchSetup *setup,
    void **compute_encoder_out,
    char *err,
    size_t errcap);
/* Owner-aware form. CommandBufferOwner.current remains inside C++. */
int mglRenderCppBeginComputeDispatchForCommandBufferOwner(
    void *command_buffer_owner,
    const MGLRenderCppComputeDispatchSetup *setup,
    void **compute_encoder_out,
    char *err,
    size_t errcap);

/* end：dispatchThreadgroups + endEncoding。encoder 为 begin 返回的同一句柄。 */
int mglRenderCppEndComputeDispatch(void *compute_encoder,
                                   const uint32_t groups[3],
                                   const uint32_t threads[3],
                                   char *err,
                                   size_t errcap);
int mglRenderCppDispatchComputeThreads(void *compute_encoder,
                                       uint32_t threads_x,
                                       uint32_t threads_y,
                                       uint32_t threads_z,
                                       uint32_t group_x,
                                       uint32_t group_y,
                                       uint32_t group_z);
int mglRenderCppCreateComputeEncoder(void *command_buffer,
                                     void **compute_encoder_out);
int mglRenderCppEndComputeEncoder(void *compute_encoder);

/* Command-buffer/render-pass lifecycle facade.  Returned Metal objects are
 * borrowed Objective-C-compatible pointers; the caller retains them through
 * its normal strong state field. */
int mglRenderCppCreateCommandBuffer(void *command_queue,
                                    void **command_buffer_out);
enum {
    MGL_RENDER_CPP_ERROR_DOMAIN_CAPACITY = 128,
    MGL_RENDER_CPP_ERROR_DESCRIPTION_CAPACITY = 512,
};

typedef struct MGLRenderCppCommandBufferState_t {
    uint32_t status;
    uint32_t has_error;
    int64_t error_code;
    char error_domain[MGL_RENDER_CPP_ERROR_DOMAIN_CAPACITY];
    char error_description[MGL_RENDER_CPP_ERROR_DESCRIPTION_CAPACITY];
} MGLRenderCppCommandBufferState;

typedef enum MGLRenderCppCommandBufferCommitAction_t {
    MGL_RENDER_CPP_COMMAND_BUFFER_COMMIT_PROCEED = 0,
    MGL_RENDER_CPP_COMMAND_BUFFER_COMMIT_SKIP_ALREADY_COMMITTED = 1,
} MGLRenderCppCommandBufferCommitAction;

typedef struct MGLRenderCppCommandBufferCommitDecision_t {
    uint32_t action;
} MGLRenderCppCommandBufferCommitDecision;

typedef enum MGLRenderCppCommandBufferTransactionResult_t {
    MGL_RENDER_CPP_COMMAND_BUFFER_TRANSACTION_COMMITTED = 0,
    MGL_RENDER_CPP_COMMAND_BUFFER_TRANSACTION_SKIPPED = 1,
    MGL_RENDER_CPP_COMMAND_BUFFER_TRANSACTION_NESTED = 2,
    MGL_RENDER_CPP_COMMAND_BUFFER_TRANSACTION_ERROR = 3,
} MGLRenderCppCommandBufferTransactionResult;

typedef struct MGLRenderCppCommandRecoverySnapshot_t {
    uint64_t consecutive_errors;
    uint64_t consecutive_successes;
    double last_error_time;
    uint32_t recovery_mode;
} MGLRenderCppCommandRecoverySnapshot;

/* Result of one owner-aware submit transaction.  State snapshots are value
 * copies; no command-buffer pointer is retained by the result. */
typedef struct MGLRenderCppCommandBufferTransaction_t {
    MGLRenderCppCommandBufferState before;
    MGLRenderCppCommandBufferState after;
    MGLRenderCppCommandBufferState completion;
    uint32_t result;
    uint32_t used_submission;
    uint32_t completion_registered;
    uint32_t waited;
    uint32_t has_error;
    uint32_t is_driver_rejection;
    uint32_t device_reset_requested;
    uint32_t recovery_error_recorded;
    MGLRenderCppCommandRecoverySnapshot recovery;
    uint32_t needs_new_command_buffer;
    /* Set when the C++ owner created the next current command buffer as part
     * of this transaction.  A zero value means the caller must retain its
     * legacy queue/reset adapter (for example an adopted ObjC buffer). */
    uint32_t current_command_buffer_created;
} MGLRenderCppCommandBufferTransaction;

/* Result of one owner-contained compute execution.  No submission or Metal
 * object pointer escapes the transaction.  When submitted is zero, the
 * encoded compute work remains in CommandBufferOwner.current for the normal
 * renderer flush. */
typedef struct MGLRenderCppComputeExecutionResult_t {
    MGLRenderCppCommandBufferTransaction transaction;
    uint32_t submitted;
    uint32_t cpu_prefix_synchronized;
    uint32_t failed_copy_back_index;
} MGLRenderCppComputeExecutionResult;

/* Validate and encode a complete compute plan, then (when copy-backs or CPU
 * visibility require a boundary) encode the copy-back blit and perform the
 * owner submit/wait transaction before synchronizing GL CPU prefixes. */
int mglRenderCppExecuteComputeExecutionPlan(
    void *command_buffer_owner,
    void *recovery_owner,
    const MGLRenderCppComputeExecutionPlan *plan,
    const MGLRenderCppCopyBackEntry *copy_backs,
    uint32_t copy_back_count,
    uint32_t require_cpu_visibility,
    MGLRenderCppComputeExecutionResult *result,
    char *err,
    size_t errcap);

typedef struct MGLRenderCppCommandBufferCompletionDecision_t {
    uint32_t has_error;
    uint32_t is_driver_rejection;
} MGLRenderCppCommandBufferCompletionDecision;

typedef struct MGLRenderCppCommandRecoverySuccess_t {
    MGLRenderCppCommandRecoverySnapshot state;
    uint32_t sustained_recovery;
    uint64_t recovered_successes;
    uint64_t previous_errors;
} MGLRenderCppCommandRecoverySuccess;

typedef struct MGLRenderCppCommandRecoverySkipDecision_t {
    MGLRenderCppCommandRecoverySnapshot state;
    uint32_t should_skip;
    uint32_t entered_recovery_mode;
    uint32_t recovery_timed_out;
    uint64_t previous_errors;
} MGLRenderCppCommandRecoverySkipDecision;

typedef struct MGLRenderCppCommandBufferCompletionResult_t {
    MGLRenderCppCommandBufferCompletionDecision decision;
    MGLRenderCppCommandRecoverySnapshot state;
    uint32_t sustained_recovery;
    uint32_t cleared_recovery_mode;
    uint64_t recovered_successes;
    uint64_t previous_errors;
} MGLRenderCppCommandBufferCompletionResult;

typedef void (*MGLRenderCppCommandBufferCompletion)(
    void *context,
    const MGLRenderCppCommandBufferState *state);
typedef void (*MGLRenderCppDestroyContext)(void *context);

/* Snapshot status/error data into caller-owned storage. The completion
 * registration keeps context alive until Metal completes the command buffer,
 * invokes callback once, then invokes destroy_context exactly once. The state
 * pointer passed to callback is valid only for the duration of that call. */
int mglRenderCppGetCommandBufferState(
    void *command_buffer,
    MGLRenderCppCommandBufferState *state_out);
/* Pure value-state classification used by the owner transaction and platform
 * log adapters. Commit classification preserves the legacy status ordering. */
int mglRenderCppClassifyCommandBufferCommit(
    const MGLRenderCppCommandBufferState *state,
    MGLRenderCppCommandBufferCommitDecision *decision_out);
/* Commit one detached/current command buffer through the C++ owner.  When
 * submission_handle points at a matching C++ submission, that ownership is
 * consumed; otherwise the borrowed command buffer is committed directly.
 * Recovery counting, driver-rejection classification, and reset decisions are
 * returned as value-state; the caller only publishes platform logging/reset. */
int mglRenderCppCommitCommandBufferTransaction(
    void *owner,
    void **submission_handle,
    void *command_buffer,
    void *recovery_owner,
    uint32_t wait_for_completion,
    MGLRenderCppCommandBufferTransaction *result_out);
int mglRenderCppClassifyCommandBufferCompletion(
    const MGLRenderCppCommandBufferState *state,
    MGLRenderCppCommandBufferCompletionDecision *decision_out);
/* Thread-safe owner for the renderer's command-completion error counters.
 * Timestamps are caller-provided seconds so policy remains independent of
 * Foundation and can be tested deterministically. */
int mglRenderCppCreateCommandRecoveryOwner(void **owner_out);
void mglRenderCppDestroyCommandRecoveryOwner(void **owner);
int mglRenderCppCommandRecoveryRecordError(
    void *owner,
    double now,
    MGLRenderCppCommandRecoverySnapshot *state_out);
/* Platform exception boundary for failures that cannot cross the C++ ABI.
 * Applies the recovery update at most once to transaction_inout. */
int mglRenderCppCommandRecoveryRecordTransactionFailure(
    void *owner,
    const MGLRenderCppCommandBufferState *state,
    MGLRenderCppCommandBufferTransaction *transaction_inout);
int mglRenderCppCommandRecoveryRecordSuccess(
    void *owner,
    double now,
    MGLRenderCppCommandRecoverySuccess *result_out);
/* Kept separate from RecordSuccess to preserve the legacy two-lock completion
 * sequence. Returns 1 when recovery mode was cleared, 0 when already clear. */
int mglRenderCppCommandRecoveryClearMode(void *owner);
int mglRenderCppCommandRecoveryShouldSkip(
    void *owner,
    double now,
    MGLRenderCppCommandRecoverySkipDecision *decision_out);
/* Classify one completed command buffer and apply the legacy recovery-owner
 * update sequence. Success intentionally performs RecordSuccess followed by
 * the separate ClearMode operation so the former two-lock ordering remains
 * observable; ObjC consumes the returned value state for logging/reset work. */
int mglRenderCppProcessCommandBufferCompletion(
    void *owner,
    const MGLRenderCppCommandBufferState *state,
    double now,
    MGLRenderCppCommandBufferCompletionResult *result_out);
/* Register the standard command-recovery completion handler without capturing
 * an Objective-C renderer.  The C++ recovery owner records error/success
 * counters and latches a deferred-reset request for the GL thread. */
int mglRenderCppAddCommandBufferRecoveryCompletion(
    void *command_buffer,
    void *recovery_owner);
/* Consume a reset request latched by a completion worker. Returns 1 when a
 * request was consumed, 0 when none is pending, and -1 for invalid owner. */
int mglRenderCppCommandRecoveryTakeResetRequest(void *recovery_owner);
int mglRenderCppAddCommandBufferCompletion(
    void *command_buffer,
    MGLRenderCppCommandBufferCompletion callback,
    void *context,
    MGLRenderCppDestroyContext destroy_context);
/* Register a completion on CommandBufferOwner.current without exposing the
 * borrowed command buffer through the C ABI. */
int mglRenderCppAddCommandBufferOwnerCompletion(
    void *owner,
    MGLRenderCppCommandBufferCompletion callback,
    void *context,
    MGLRenderCppDestroyContext destroy_context);
/* The current-buffer owner retains the autoreleased command buffer returned
 * by Metal. Detach moves that +1 reference into a submission handle; commit
 * consumes the submission only after Metal accepts it. Returned command
 * buffer pointers are borrowed. */
int mglRenderCppCreateCommandBufferOwner(void *command_queue,
                                         void **owner_out,
                                         void **command_buffer_out);
/* Adopt an existing (ObjC-created) command buffer as the owner's current —
 * gate-off fallback so the owner stays the single source on both gates.
 * Returns 0 with *owner_out set (the owner retains the buffer). */
int mglRenderCppCreateCommandBufferOwnerAdopt(void *command_buffer,
                                              void **owner_out);
/* Borrowed pointer to the owner's current command buffer (NULL when the
 * owner has none / owner is NULL). */
void *mglRenderCppCommandBufferOwnerGetCurrent(void *owner);
/* Returns 1 when current exists, 0 when empty, and -1 for a null owner. */
int mglRenderCppCommandBufferOwnerHasCurrent(void *owner);
/* Create the next current command buffer from the queue retained by the
 * owner.  Returns 0 on success, 1 when the owner has no queue (adopted
 * fallback), and -1 on allocation/argument failure. */
int mglRenderCppCommandBufferOwnerCreateNext(void *owner,
                                             void **command_buffer_out);
/* Snapshot the owner's current buffer without exposing it to the caller.
 * Returns -1 when the owner/current buffer/state output is missing. */
int mglRenderCppGetCommandBufferOwnerState(
    void *owner,
    MGLRenderCppCommandBufferState *state_out);
/* The owner retains the most recently accepted submission. These APIs keep
 * glFinish/readback synchronization in the lifecycle owner without exposing
 * a borrowed command-buffer pointer to Objective-C. */
int mglRenderCppCommandBufferOwnerHasLastSubmitted(void *owner);
/* Wait for one submitted command buffer and return a value-state snapshot.
 * Returns 0 on completed success, 1 when the buffer is still NotEnqueued,
 * and -1 for invalid arguments, wait failures, or command-buffer errors. */
int mglRenderCppWaitCommandBufferState(
    void *command_buffer,
    MGLRenderCppCommandBufferState *state_out);
int mglRenderCppWaitCommandBufferOwnerLastSubmitted(
    void *owner,
    MGLRenderCppCommandBufferState *state_out);
/* Encode presentation on the owner's current not-enqueued command buffer.
 * Returns 0 on success, 1 when the current buffer is already finalized, and
 * -1 for missing owner/current buffer/drawable. */
int mglRenderCppPresentDrawableForCommandBufferOwner(
    void *owner,
    void *drawable,
    MGLRenderCppCommandBufferState *state_out);
int mglRenderCppEncodeWaitForEventForCommandBufferOwner(
    void *owner, void *event, uint64_t value);
int mglRenderCppResetCommandBufferOwner(void *owner,
                                        void *command_queue,
                                        void **command_buffer_out);
void mglRenderCppDiscardCommandBufferOwnerCurrent(void *owner);
/* Reentrancy guard for command-buffer commit. Returns 1 when acquired, 0
 * when a commit is already in progress, and -1 for a missing owner. This
 * preserves the former MGLCommandState BOOL semantics; it is intentionally
 * not a cross-thread synchronization primitive. */
int mglRenderCppCommandBufferOwnerBeginCommit(void *owner);
void mglRenderCppCommandBufferOwnerEndCommit(void *owner);
/* Consume the marker for a current buffer created by the preceding submit
 * transaction. Returns 1 and a borrowed current buffer once, 0 when no such
 * buffer is pending, and -1 for invalid arguments. */
int mglRenderCppCommandBufferOwnerConsumeTransactionCurrent(
    void *owner,
    void **command_buffer_out);
int mglRenderCppTakeCommandBufferSubmission(void *owner,
                                             void **submission_out,
                                             void **command_buffer_out);
int mglRenderCppCommitCommandBufferSubmission(void **submission);
void mglRenderCppDestroyCommandBufferSubmission(void **submission);
void mglRenderCppDestroyCommandBufferOwner(void **owner);
/* The opaque owner holds the +1 Metal-cpp command-queue reference. The queue
 * pointer is borrowed and may be assigned to an ObjC strong field during the
 * migration. max_command_buffers=0 selects Metal's default configuration. */
int mglRenderCppCreateCommandQueueOwner(uint32_t max_command_buffers,
                                        void **owner_out,
                                        void **command_queue_out);
int mglRenderCppResetCommandQueueOwner(void *owner,
                                       uint32_t max_command_buffers,
                                       void **command_queue_out);
void mglRenderCppDestroyCommandQueueOwner(void **owner);
/* Per-command-buffer MDI argument arena. The opaque owner keeps the sole
 * persistent +1 reference; returned buffers are borrowed migration views. */
int mglRenderCppCreateMDIScratchOwner(void **owner_out);
int mglRenderCppAllocateMDIScratch(void *owner,
                                   uint64_t length,
                                   uint64_t alignment,
                                   void **buffer_out,
                                   uint64_t *offset_out,
                                   uint64_t *capacity_out);
void mglRenderCppResetMDIScratchOwner(void *owner);
void mglRenderCppDestroyMDIScratchOwner(void **owner);
int mglRenderCppCommitCommandBuffer(void *command_buffer);
int mglRenderCppWaitCommandBuffer(void *command_buffer);

/* P4.3b: per-draw binding snapshot。ObjC 侧保留「判定哪些绑定需要 emit」
 * 的 GL 逻辑（dedup 检查、统计、COW 记账），把通过判定的绑定序列收集进
 * snapshot，单次交给 mglRenderCppEncodeBindingSnapshot 在 C++ 内重放
 * （setter 序列在 C++；与直接 draw 路径的 mglRenderCppSetRenderBuffer 等价）。 */
#define MGL_RENDER_CPP_BINDING_SNAPSHOT_MAX_OPS 32u

/* One per-draw binding op: kind 0 = set buffer (buffer == NULL clears the
 * slot, matching mglRenderCppSetRenderBuffer with a nil resource), kind 1 =
 * set bytes (bytes borrowed — valid until EncodeBindingSnapshot returns).
 * The op list keeps the exact per-stage emit order, including interleaved
 * buffer/bytes/clear ops on the same slot. */
typedef struct MGLRenderCppBindingOp_t {
    uint32_t kind;      /* 0 = buffer, 1 = bytes */
    uint32_t index;     /* Metal slot */
    uint64_t offset;    /* kind 0: byte offset */
    void *buffer;       /* kind 0: borrowed MTL::Buffer* (NULL = clear) */
    const void *bytes;  /* kind 1: borrowed byte pointer */
    uint32_t length;    /* kind 1: byte length */
} MGLRenderCppBindingOp;

typedef struct MGLRenderCppBindingSnapshot_t {
    uint32_t vertex_op_count;
    MGLRenderCppBindingOp
        vertex_ops[MGL_RENDER_CPP_BINDING_SNAPSHOT_MAX_OPS];
    uint32_t fragment_op_count;
    MGLRenderCppBindingOp
        fragment_ops[MGL_RENDER_CPP_BINDING_SNAPSHOT_MAX_OPS];
} MGLRenderCppBindingSnapshot;

int mglRenderCppEncodeBindingSnapshot(
    void *render_encoder,
    const MGLRenderCppBindingSnapshot *snapshot,
    char *err,
    size_t errcap);
int mglRenderCppEncodeBindingSnapshotForRenderEncoderOwner(
    void *render_encoder_owner,
    const MGLRenderCppBindingSnapshot *snapshot,
    char *err,
    size_t errcap);

/* Texture/sampler bindings are collected separately from buffer/bytes ops.
 * Resource resolution may rotate the render encoder while uploading a
 * texture, so ObjC submits this snapshot in ordered segments at those
 * boundaries.  The C++ binding owner remains authoritative for dedup state
 * and retains every resource that becomes current. */
#define MGL_RENDER_CPP_RESOURCE_BINDING_SNAPSHOT_MAX_OPS 512u

enum {
    MGL_RENDER_CPP_RESOURCE_BINDING_TEXTURE = 0,
    MGL_RENDER_CPP_RESOURCE_BINDING_SAMPLER = 1,
};

typedef struct MGLRenderCppResourceBindingOp_t {
    uint32_t kind;
    uint32_t index;
    void *resource; /* borrowed MTL::Texture* or MTL::SamplerState* */
} MGLRenderCppResourceBindingOp;

typedef struct MGLRenderCppResourceBindingSnapshot_t {
    uint32_t vertex_op_count;
    MGLRenderCppResourceBindingOp
        vertex_ops[MGL_RENDER_CPP_RESOURCE_BINDING_SNAPSHOT_MAX_OPS];
    uint32_t fragment_op_count;
    MGLRenderCppResourceBindingOp
        fragment_ops[MGL_RENDER_CPP_RESOURCE_BINDING_SNAPSHOT_MAX_OPS];
} MGLRenderCppResourceBindingSnapshot;

int mglRenderCppEncodeResourceBindingSnapshot(
    void *binding_state,
    void *render_encoder,
    const MGLRenderCppResourceBindingSnapshot *snapshot,
    char *err,
    size_t errcap);
int mglRenderCppEncodeResourceBindingSnapshotForRenderEncoderOwner(
    void *binding_state,
    void *render_encoder_owner,
    const MGLRenderCppResourceBindingSnapshot *snapshot,
    char *err,
    size_t errcap);

/* P4.3c: whole-batch simple replay（最小 surgery 版）。满足「简单批」条件的
 * batch（无 dynamic binding / sampler 快照 / cull-distance / 多边形模拟 /
 * primitive restart，元素命令已 prepare 索引缓冲）由 ObjC 把命令解析成纯 C
 * 数组，一次交给 C++ 循环绘制 —— replay 执行 loop 在 C++，数据仍是 ObjC
 * batch arena 的只读快照。命令数超上限或任一条无法解析时 ObjC 回退原循环。 */
#define MGL_RENDER_CPP_REPLAY_BATCH_MAX_COMMANDS 128u

typedef struct MGLRenderCppReplayBatchCommand_t {
    uint32_t cmd_type;          /* MGLDrawCommandType 数值（draw_command.h） */
    int32_t first;
    uint32_t count;
    uint32_t instance_count;
    int32_t base_vertex;
    uint32_t base_instance;
    uint32_t index_type;        /* MTLIndexType（ObjC 已转换） */
    uint32_t index_buffer_offset;
    void *index_buffer;         /* +0 borrowed MTL::Buffer*（ObjC 已 prepare） */
} MGLRenderCppReplayBatchCommand;

typedef struct MGLRenderCppReplayBatch_t {
    uint32_t primitive_type;    /* MTLPrimitiveType（batch key） */
    uint32_t command_count;
    const MGLRenderCppReplayBatchCommand *commands;
} MGLRenderCppReplayBatch;

enum {
    MGL_RENDER_CPP_REPLAY_BATCH_OK = 0,          /* 全部命令已由 C++ 绘制 */
    MGL_RENDER_CPP_REPLAY_BATCH_NEEDS_OBJC = 1,  /* 有命令无法在 C++ 处理 */
    MGL_RENDER_CPP_REPLAY_BATCH_ERROR = -1,      /* 参数非法（不应发生） */
};

/* 契约：调用方必须已保证全部命令类型合法、元素命令 index_buffer 非空且
 * index_type != 0xFFFFFFFF、count 超限时直接回退 ObjC；因此本函数成功即
 * 全部绘制，失败（NEEDS_OBJC/ERROR）时 ObjC 必须整体回退原循环（不得部分
 * 重放）。 */
int mglRenderCppReplayBatchDraws(void *render_encoder,
                                 const MGLRenderCppReplayBatch *batch,
                                 char *err,
                                 size_t errcap);

enum {
    MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS = 8,
    MGL_RENDER_CPP_MAX_SAMPLE_POSITIONS = 32,
};

typedef struct MGLRenderCppRenderPassAttachmentState_t {
    void *texture;
    void *resolve_texture;
    uint64_t level;
    uint64_t slice;
    uint64_t depth_plane;
    uint64_t resolve_level;
    uint64_t resolve_slice;
    uint64_t resolve_depth_plane;
    uint32_t load_action;
    uint32_t store_action;
    uint64_t store_action_options;
} MGLRenderCppRenderPassAttachmentState;

typedef struct MGLRenderCppRenderPassColorState_t {
    MGLRenderCppRenderPassAttachmentState attachment;
    double clear_red;
    double clear_green;
    double clear_blue;
    double clear_alpha;
} MGLRenderCppRenderPassColorState;

typedef struct MGLRenderCppRenderPassDepthState_t {
    MGLRenderCppRenderPassAttachmentState attachment;
    double clear_depth;
    uint32_t resolve_filter;
} MGLRenderCppRenderPassDepthState;

typedef struct MGLRenderCppRenderPassStencilState_t {
    MGLRenderCppRenderPassAttachmentState attachment;
    uint32_t clear_stencil;
    uint32_t resolve_filter;
} MGLRenderCppRenderPassStencilState;

typedef struct MGLRenderCppSamplePosition_t {
    float x;
    float y;
} MGLRenderCppSamplePosition;

typedef struct MGLRenderCppRenderPassState_t {
    MGLRenderCppRenderPassColorState
        color[MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS];
    MGLRenderCppRenderPassDepthState depth;
    MGLRenderCppRenderPassStencilState stencil;
    void *visibility_result_buffer;
    void *rasterization_rate_map;
    uint64_t render_target_array_length;
    uint64_t render_target_width;
    uint64_t render_target_height;
    uint64_t default_raster_sample_count;
    uint64_t imageblock_sample_length;
    uint64_t threadgroup_memory_length;
    uint64_t tile_width;
    uint64_t tile_height;
    uint32_t visibility_result_type;
    uint32_t support_color_attachment_mapping;
    uint32_t sample_position_count;
    MGLRenderCppSamplePosition
        sample_positions[MGL_RENDER_CPP_MAX_SAMPLE_POSITIONS];
} MGLRenderCppRenderPassState;

/* Initialize a value state with Metal's render-pass descriptor defaults. */
void mglRenderCppInitDefaultRenderPassState(
    MGLRenderCppRenderPassState *state_out);

typedef enum MGLRenderCppRenderPassAttachmentKind_t {
    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_COLOR = 0,
    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_DEPTH = 1,
    MGL_RENDER_CPP_RENDER_PASS_ATTACHMENT_STENCIL = 2,
} MGLRenderCppRenderPassAttachmentKind;

typedef struct MGLRenderCppRenderPassIdentityState_t {
    void *framebuffer;
    uint32_t framebuffer_name;
    uint32_t draw_buffer;
    uint32_t draw_buffer_count;
    uint32_t draw_buffers[MGL_RENDER_CPP_MAX_COLOR_ATTACHMENTS];
} MGLRenderCppRenderPassIdentityState;

typedef struct MGLRenderCppFboMatchCacheState_t {
    uint32_t fbo_name;
    uint64_t generation;
    int result;
} MGLRenderCppFboMatchCacheState;

/* Persistent render-pass identity and FBO cache. The owner is authoritative
 * for Metal-cpp mode; ObjC fields remain a synchronized migration view. */
int mglRenderCppCreateRenderPassIdentityOwner(void **owner_out);
int mglRenderCppUpdateRenderPassIdentity(
    void *owner, const MGLRenderCppRenderPassIdentityState *state);
int mglRenderCppGetRenderPassIdentity(
    void *owner, MGLRenderCppRenderPassIdentityState *state_out);
int mglRenderCppSetFboMatchCache(
    void *owner, const MGLRenderCppFboMatchCacheState *cache);
int mglRenderCppGetFboMatchCache(
    void *owner, MGLRenderCppFboMatchCacheState *cache_out);
void mglRenderCppClearFboMatchCache(void *owner);
void mglRenderCppDestroyRenderPassIdentityOwner(void **owner);

/* Persistent value-state owner for render-pass attachment/dimension fields.
 * The owner retains every attachment/resolve/visibility/rate-map resource
 * referenced by the snapshot and releases replaced resources on update. */
int mglRenderCppCreateRenderPassStateOwner(
    const MGLRenderCppRenderPassState *state, void **owner_out);
int mglRenderCppCreateDefaultRenderPassStateOwner(void **owner_out);
int mglRenderCppSetRenderPassStateAttachment(
    void *owner,
    uint32_t attachment_kind,
    uint32_t color_index,
    const MGLRenderCppRenderPassAttachmentState *attachment);
int mglRenderCppSetRenderPassStateAttachmentTexture(
    void *owner,
    uint32_t attachment_kind,
    uint32_t color_index,
    void *texture,
    uint64_t level,
    uint64_t slice,
    uint64_t depth_plane,
    uint32_t layered);
int mglRenderCppSetRenderPassStateAttachmentActions(
    void *owner,
    uint32_t attachment_kind,
    uint32_t color_index,
    uint32_t load_action,
    uint32_t store_action,
    uint64_t store_action_options);
int mglRenderCppSetRenderPassStateColorClear(
    void *owner,
    uint32_t color_index,
    double red,
    double green,
    double blue,
    double alpha);
int mglRenderCppSetRenderPassStateDepthClear(
    void *owner, double clear_depth);
int mglRenderCppSetRenderPassStateStencilClear(
    void *owner, uint32_t clear_stencil);
int mglRenderCppSetRenderPassStateVisibility(
    void *owner, void *visibility_result_buffer,
    uint32_t visibility_result_type);
int mglRenderCppSetRenderPassStateDimensions(
    void *owner, uint64_t width, uint64_t height);
/* P4.5: mirror-fallback color-attachment query（ObjC
 * mglRenderPassUsesColorTexture 迁入）。descriptor 为 MTL::RenderPassDescriptor*，
 * texture 为 MTL::Texture*；命中返回 1 并写 attachment_index_out，未命中 0；
 * 坏参返回 -1。 */
int mglRenderCppRenderPassUsesColorTexture(void *render_pass_descriptor,
                                           void *texture,
                                           size_t *attachment_index_out);
/* P4.5 (item 1141): pending shared-event slot inside the C++ owner.
 * `int` in these decls is GLsizei (GL signed 32-bit) — the C ABI matches. */
int mglRenderCppCreatePendingEventOwner(void **owner_out);
int mglRenderCppPendingEventPrepare(void *owner_handle, int sync_name,
                                    void **event_out);
int mglRenderCppPendingEventDetach(void *owner_handle,
                                   int *sync_name_out, void **event_out);
void mglRenderCppPendingEventClear(void *owner_handle);
void mglRenderCppDestroyPendingEventOwner(void **owner_handle);
/* P4.5 (item 1141): detached-submission ownership guard. */
int mglRenderCppCommandBufferSubmissionMatchesBuffer(void *submission_handle,
                                                     void *command_buffer);
/* P4.5 (item 1141): current-CB sync tracking list inside the C++ owner. */
int mglRenderCppCommandBufferOwnerAppendSync(void *owner_handle, Sync *sync);
void mglRenderCppCommandBufferOwnerClearSyncs(void *owner_handle);
int mglRenderCppGetRenderPassStateOwner(
    void *owner, MGLRenderCppRenderPassState *state_out);
int mglRenderCppCreateRenderEncoderFromStateOwner(
    void *command_buffer, void *state_owner, void **render_encoder_out);
/* Owner-aware variant used by command-lifecycle callers. The command buffer
 * stays inside CommandBufferOwner; the returned encoder is borrowed. */
int mglRenderCppCreateRenderEncoderFromCommandBufferOwnerState(
    void *command_buffer_owner,
    const MGLRenderCppRenderPassState *render_pass,
    void **render_encoder_out);
/* Compatibility path for an already-materialized ObjC render-pass
 * descriptor. The descriptor pointer is borrowed and remains opaque at the
 * C ABI; CommandBufferOwner.current does not escape. */
int mglRenderCppCreateRenderEncoderFromCommandBufferOwnerDescriptor(
    void *command_buffer_owner,
    void *render_pass_descriptor,
    void **render_encoder_out);
void mglRenderCppDestroyRenderPassStateOwner(void **owner);

/* C++ owns the temporary MTL::RenderPassDescriptor used to create the
 * borrowed render encoder. Attachment resources remain caller-owned. */
int mglRenderCppCreateRenderEncoderFromState(
    void *command_buffer,
    const MGLRenderCppRenderPassState *render_pass,
    void **render_encoder_out);
int mglRenderCppEncodeColorClear(void *command_buffer,
                                 void *texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double red,
                                 double green,
                                 double blue,
                                 double alpha);
/* Owner-aware variant used by renderer clear paths. The current command
 * buffer remains inside CommandBufferOwner and is never borrowed through the
 * C ABI. */
int mglRenderCppEncodeColorClearForCommandBufferOwner(
    void *command_buffer_owner,
    void *texture,
    uint64_t level,
    uint64_t slice,
    uint64_t depth_plane,
    double red,
    double green,
    double blue,
    double alpha);
int mglRenderCppEncodeDepthClear(void *command_buffer,
                                 void *texture,
                                 uint64_t level,
                                 uint64_t slice,
                                 uint64_t depth_plane,
                                 double clear_depth);
int mglRenderCppEncodeDepthClearForCommandBufferOwner(
    void *command_buffer_owner,
    void *texture,
    uint64_t level,
    uint64_t slice,
    uint64_t depth_plane,
    double clear_depth);
int mglRenderCppEncodeMultisampleResolve(
    void *command_buffer,
    uint32_t attachment_kind,
    void *source_texture,
    uint64_t source_level,
    uint64_t source_slice,
    uint64_t source_depth_plane,
    void *resolve_texture,
    uint64_t resolve_level,
    uint64_t resolve_slice,
    uint64_t resolve_depth_plane,
    uint32_t resolve_filter);
int mglRenderCppEncodeMultisampleResolveForCommandBufferOwner(
    void *command_buffer_owner,
    uint32_t attachment_kind,
    void *source_texture,
    uint64_t source_level,
    uint64_t source_slice,
    uint64_t source_depth_plane,
    void *resolve_texture,
    uint64_t resolve_level,
    uint64_t resolve_slice,
    uint64_t resolve_depth_plane,
    uint32_t resolve_filter);
/* The owner retains the autoreleased encoder returned by Metal. End is
 * idempotent per owned encoder; destroy releases the retained reference. */
int mglRenderCppCreateRenderEncoderOwnerFromState(
    void *command_buffer,
    const MGLRenderCppRenderPassState *render_pass,
    void **owner_out,
    void **render_encoder_out);
int mglRenderCppResetRenderEncoderOwnerFromState(
    void *owner,
    void *command_buffer,
    const MGLRenderCppRenderPassState *render_pass,
    void **render_encoder_out);
int mglRenderCppCreateRenderEncoderOwner(
    void *render_encoder,
    void **owner_out);
int mglRenderCppResetRenderEncoderOwner(
    void *owner,
    void *render_encoder);
int mglRenderCppEndRenderEncoderOwner(void *owner);
int mglRenderCppSetRenderEncoderOwnerLabel(void *owner,
                                           const char *label);
int mglRenderCppRenderEncoderOwnerHasCurrent(void *owner);
void mglRenderCppDestroyRenderEncoderOwner(void **owner);
int mglRenderCppEndRenderEncoder(void *render_encoder);
int mglRenderCppCreateBlitEncoder(void *command_buffer,
                                  void **blit_encoder_out);
/* Creates a borrowed blit encoder from CommandBufferOwner.current without
 * exposing the current command buffer through the C ABI. */
int mglRenderCppCreateBlitEncoderFromCommandBufferOwner(
    void *command_buffer_owner,
    void **blit_encoder_out);
/* Encode a complete texture-to-texture preservation copy inside the owner.
 * Every common array slice and mip level is copied at its full mip extent;
 * encoder creation and endEncoding remain entirely in C++. */
int mglRenderCppCopyMatchingTextureSubresourcesForCommandBufferOwner(
    void *command_buffer_owner,
    void *source_texture,
    void *destination_texture);
typedef struct MGLRenderCppBufferCopyEntry_t {
    void *source_buffer;
    uint64_t source_offset;
    void *destination_buffer;
    uint64_t destination_offset;
    uint64_t length;
} MGLRenderCppBufferCopyEntry;
int mglRenderCppEncodeBufferCopiesForCommandBufferOwner(
    void *command_buffer_owner,
    const MGLRenderCppBufferCopyEntry *entries,
    uint32_t entry_count);
int mglRenderCppCreateComputeEncoderFromCommandBufferOwner(
    void *command_buffer_owner,
    void **compute_encoder_out);
int mglRenderCppEndBlitEncoder(void *blit_encoder);
/* Encode and end a complete buffer-to-texture upload blit in C++. The
 * command buffer retains the encoded resources after this function returns. */
int mglRenderCppEncodeTextureUpload(void *command_buffer,
                                    void *source_buffer,
                                    uint64_t source_offset,
                                    uint64_t source_bytes_per_row,
                                    uint64_t source_bytes_per_image,
                                    uint64_t source_width,
                                    uint64_t source_height,
                                    uint64_t source_depth,
                                    void *destination_texture,
                                    uint64_t destination_slice,
                                    uint64_t destination_level,
                                    uint64_t destination_x,
                                    uint64_t destination_y,
                                    uint64_t destination_z);
/* Multi-slice form used by array-texture subimages.  Arithmetic and resource
 * extents are validated before a single blit encoder is opened, so a bad
 * range cannot leave a partially encoded layer prefix. */
int mglRenderCppEncodeTextureUploadLayers(
    void *command_buffer,
    void *source_buffer,
    uint64_t source_offset,
    uint64_t source_bytes_per_row,
    uint64_t source_bytes_per_image,
    uint64_t source_layer_stride,
    uint64_t source_width,
    uint64_t source_height,
    uint64_t source_depth,
    void *destination_texture,
    uint64_t destination_base_slice,
    uint64_t layer_count,
    uint64_t destination_level,
    uint64_t destination_x,
    uint64_t destination_y,
    uint64_t destination_z);
int mglRenderCppEncodeTextureUploadLayersForCommandBufferOwner(
    void *command_buffer_owner,
    void *source_buffer,
    uint64_t source_offset,
    uint64_t source_bytes_per_row,
    uint64_t source_bytes_per_image,
    uint64_t source_layer_stride,
    uint64_t source_width,
    uint64_t source_height,
    uint64_t source_depth,
    void *destination_texture,
    uint64_t destination_base_slice,
    uint64_t layer_count,
    uint64_t destination_level,
    uint64_t destination_x,
    uint64_t destination_y,
    uint64_t destination_z);
int mglRenderCppBlitCopyBuffer(void *blit_encoder,
                               void *source_buffer,
                               uint64_t source_offset,
                               void *destination_buffer,
                               uint64_t destination_offset,
                               uint64_t size);
int mglRenderCppBlitCopyBufferToTexture(void *blit_encoder,
                                        void *source_buffer,
                                        uint64_t source_offset,
                                        uint64_t source_bytes_per_row,
                                        uint64_t source_bytes_per_image,
                                        uint64_t source_width,
                                        uint64_t source_height,
                                        uint64_t source_depth,
                                        void *destination_texture,
                                        uint64_t destination_slice,
                                        uint64_t destination_level,
                                        uint64_t destination_x,
                                        uint64_t destination_y,
                                        uint64_t destination_z);
int mglRenderCppBlitSynchronizeTexture(void *blit_encoder,
                                       void *texture,
                                       uint64_t slice,
                                       uint64_t level);
int mglRenderCppBlitCopyTexture(void *blit_encoder,
                                void *source_texture,
                                uint64_t source_slice,
                                uint64_t source_level,
                                uint64_t source_x,
                                uint64_t source_y,
                                uint64_t source_z,
                                uint64_t width,
                                uint64_t height,
                                uint64_t depth,
                                void *destination_texture,
                                uint64_t destination_slice,
                                uint64_t destination_level,
                                uint64_t destination_x,
                                uint64_t destination_y,
                                uint64_t destination_z);
int mglRenderCppBlitCopyTextureToBuffer(
    void *blit_encoder,
    void *source_texture,
    uint64_t source_slice,
    uint64_t source_level,
    uint64_t source_x,
    uint64_t source_y,
    uint64_t source_z,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    void *destination_buffer,
    uint64_t destination_offset,
    uint64_t destination_bytes_per_row,
    uint64_t destination_bytes_per_image);
int mglRenderCppBlitGenerateMipmaps(void *blit_encoder,
                                    void *texture);

/* Render draw command facade. Enum values are passed as uint32_t so the C ABI
 * remains independent of Metal headers. Resources are borrowed for encoding. */
int mglRenderCppDrawPrimitives(void *render_encoder,
                               uint32_t primitive_type,
                               uint64_t vertex_start,
                               uint64_t vertex_count,
                               uint64_t instance_count,
                               uint64_t base_instance);
int mglRenderCppDrawIndexedPrimitives(void *render_encoder,
                                      uint32_t primitive_type,
                                      uint64_t index_count,
                                      uint32_t index_type,
                                      void *index_buffer,
                                      uint64_t index_buffer_offset,
                                      uint64_t instance_count,
                                      int64_t base_vertex,
                                      uint64_t base_instance);
int mglRenderCppDrawPrimitivesIndirect(void *render_encoder,
                                       uint32_t primitive_type,
                                       void *indirect_buffer,
                                       uint64_t indirect_buffer_offset);
int mglRenderCppDrawIndexedPrimitivesIndirect(
    void *render_encoder,
    uint32_t primitive_type,
    uint32_t index_type,
    void *index_buffer,
    uint64_t index_buffer_offset,
    void *indirect_buffer,
    uint64_t indirect_buffer_offset);

/* P4.3a: draw 提交的统一 value-state plan。ObjC draw 入口（Draw/Batch/
 * BatchReplay/draw_encode/DrawSupport/Tessellation/Blit/swap-diagnostics）
 * 只构造 plan，然后单次调用 mglRenderCppEncodeDraw；最终 draw 提交全部由
 * C++ 完成。资源为 +0 borrowed。 */
typedef struct MGLRenderCppDrawPlan_t {
    uint32_t kind;              /* MGL_RENDER_CPP_DRAW_* */
    uint32_t primitive_type;    /* MTLPrimitiveType 以 uint 传 */
    /* ARRAY: */
    uint64_t vertex_start;
    uint64_t vertex_count;
    /* INDEXED: */
    uint64_t index_count;
    uint32_t index_type;        /* MTLIndexType 以 uint 传 */
    void *index_buffer;         /* +0 borrowed MTL::Buffer* */
    uint64_t index_buffer_offset;
    int64_t base_vertex;
    /* INDIRECT: */
    void *indirect_buffer;      /* +0 borrowed MTL::Buffer* */
    uint64_t indirect_buffer_offset;
    /* PATCHES（native TES）: */
    uint64_t control_point_count;
    uint64_t patch_start;
    uint64_t patch_count;
    void *patch_index_buffer;           /* +0 borrowed */
    uint64_t patch_index_buffer_offset;
    void *control_point_index_buffer;   /* +0 borrowed */
    uint64_t control_point_index_buffer_offset;
    /* 通用: */
    uint64_t instance_count;
    uint64_t base_instance;
} MGLRenderCppDrawPlan;

enum {
    MGL_RENDER_CPP_DRAW_ARRAY = 1,
    MGL_RENDER_CPP_DRAW_INDEXED = 2,
    MGL_RENDER_CPP_DRAW_ARRAY_INDIRECT = 3,
    MGL_RENDER_CPP_DRAW_INDEXED_INDIRECT = 4,
    MGL_RENDER_CPP_DRAW_PATCHES = 5,
    MGL_RENDER_CPP_DRAW_INDEXED_PATCHES = 6,
};

/* P4.3a: 单一 draw 提交入口。render_encoder 为 +0 borrowed
 * MTL::RenderCommandEncoder*。plan 校验失败（非法 kind/空 encoder/缺 buffer
 * 等）返回 -1 并写 err，调用方回退 ObjC 直接编码。 */
int mglRenderCppEncodeDraw(void *render_encoder,
                           const MGLRenderCppDrawPlan *plan,
                           char *err,
                           size_t errcap);
int mglRenderCppEncodeDrawForRenderEncoderOwner(
    void *render_encoder_owner,
    const MGLRenderCppDrawPlan *plan,
    char *err,
    size_t errcap);

typedef struct MGLRenderCppCullDistancePrimitive_t {
    uint32_t vertices[4];
    uint32_t vertex_count;
    uint32_t primitive_type;
    uint32_t index_count;
    uint64_t index_buffer_offset;
} MGLRenderCppCullDistancePrimitive;

/* Build a UInt32 index buffer whose records each represent one complete GL
 * primitive. The opaque owner retains the borrowed index buffer and the
 * per-primitive explicit vertex IDs used by exact gl_CullDistance emulation. */
int mglRenderCppCreateCullDistanceIndexPlan(
    void *device,
    const void *source_indices,
    uint32_t source_index_type,
    uint64_t source_index_count,
    uint32_t draw_mode,
    int primitive_restart_enabled,
    uint32_t primitive_restart_index,
    int64_t base_vertex,
    int polygon_line_mode,
    void **owner_out,
    void **index_buffer_out,
    uint64_t *primitive_count_out);
int mglRenderCppGetCullDistanceIndexPrimitive(
    void *owner,
    uint64_t primitive_index,
    MGLRenderCppCullDistancePrimitive *primitive_out);
void mglRenderCppDestroyCullDistanceIndexPlan(void **owner);

int mglRenderCppSetRenderBuffer(void *render_encoder,
                                void *buffer,
                                uint64_t offset,
                                uint32_t stage,
                                uint32_t index);
int mglRenderCppSetRenderBytes(void *render_encoder,
                               const void *bytes,
                               size_t length,
                               uint32_t stage,
                               uint32_t index);
int mglRenderCppSetRenderPipelineState(void *render_encoder,
                                       void *pipeline_state);
int mglRenderCppSetRenderDepthStencilState(void *render_encoder,
                                           void *depth_stencil_state);
int mglRenderCppSetRenderTexture(void *render_encoder,
                                 void *texture,
                                 uint32_t stage,
                                 uint32_t index);
int mglRenderCppSetRenderSampler(void *render_encoder,
                                 void *sampler,
                                 uint32_t stage,
                                 uint32_t index);
int mglRenderCppSetRenderViewport(void *render_encoder,
                                  double origin_x,
                                  double origin_y,
                                  double width,
                                  double height,
                                  double znear,
                                  double zfar);
int mglRenderCppSetRenderScissor(void *render_encoder,
                                 uint64_t x,
                                 uint64_t y,
                                 uint64_t width,
                                 uint64_t height);
int mglRenderCppSetDepthClipMode(void *render_encoder, uint32_t mode);
int mglRenderCppSetStencilReferenceValues(void *render_encoder,
                                          uint32_t front_reference,
                                          uint32_t back_reference);
int mglRenderCppSetTessellationFactorBuffer(void *render_encoder,
                                            void *buffer,
                                            uint64_t offset,
                                            uint64_t instance_stride);
int mglRenderCppSetRenderBufferForOwner(void *render_encoder_owner,
                                        void *buffer,
                                        uint64_t offset,
                                        uint32_t stage,
                                        uint32_t index);
int mglRenderCppSetRenderBytesForOwner(void *render_encoder_owner,
                                       const void *bytes,
                                       size_t length,
                                       uint32_t stage,
                                       uint32_t index);
int mglRenderCppSetRenderPipelineStateForOwner(void *render_encoder_owner,
                                               void *pipeline_state);
int mglRenderCppSetRenderDepthStencilStateForOwner(void *render_encoder_owner,
                                                   void *depth_stencil_state);
int mglRenderCppSetRenderTextureForOwner(void *render_encoder_owner,
                                         void *texture,
                                         uint32_t stage,
                                         uint32_t index);
int mglRenderCppSetRenderSamplerForOwner(void *render_encoder_owner,
                                         void *sampler,
                                         uint32_t stage,
                                         uint32_t index);
int mglRenderCppSetRenderViewportForOwner(void *render_encoder_owner,
                                          double origin_x,
                                          double origin_y,
                                          double width,
                                          double height,
                                          double znear,
                                          double zfar);
int mglRenderCppSetRenderScissorForOwner(void *render_encoder_owner,
                                         uint64_t x,
                                         uint64_t y,
                                         uint64_t width,
                                         uint64_t height);
int mglRenderCppSetDepthClipModeForOwner(void *render_encoder_owner,
                                         uint32_t mode);
int mglRenderCppSetStencilReferenceValuesForOwner(
    void *render_encoder_owner,
    uint32_t front_reference,
    uint32_t back_reference);
int mglRenderCppSetTessellationFactorBufferForOwner(
    void *render_encoder_owner,
    void *buffer,
    uint64_t offset,
    uint64_t instance_stride);
int mglRenderCppDrawPatches(void *render_encoder,
                            uint64_t control_point_count,
                            uint64_t patch_start,
                            uint64_t patch_count,
                            void *patch_index_buffer,
                            uint64_t patch_index_buffer_offset,
                            uint64_t instance_count,
                            uint64_t base_instance);
int mglRenderCppDrawIndexedPatches(void *render_encoder,
                                   uint64_t control_point_count,
                                   uint64_t patch_start,
                                   uint64_t patch_count,
                                   void *patch_index_buffer,
                                   uint64_t patch_index_buffer_offset,
                                   void *control_point_index_buffer,
                                   uint64_t control_point_index_buffer_offset,
                                   uint64_t instance_count,
                                   uint64_t base_instance);

int mglRenderCppCreateIndirectCommandBuffer(
    uint32_t command_types,
    int inherit_pipeline_state,
    int inherit_buffers,
    uint32_t max_vertex_buffer_bind_count,
    uint32_t max_fragment_buffer_bind_count,
    uint64_t max_command_count,
    uint64_t resource_options,
    void **indirect_buffer_out);
int mglRenderCppResetIndirectCommandBuffer(void *indirect_buffer,
                                           uint64_t location,
                                           uint64_t length);
int mglRenderCppGetIndirectRenderCommand(void *indirect_buffer,
                                         uint64_t command_index,
                                         void **command_out);
int mglRenderCppSetIndirectDrawIndexed(void *indirect_command,
                                       uint32_t primitive_type,
                                       uint64_t index_count,
                                       uint32_t index_type,
                                       void *index_buffer,
                                       uint64_t index_buffer_offset,
                                       uint64_t instance_count,
                                       int64_t base_vertex,
                                       uint64_t base_instance);
int mglRenderCppSetIndirectDraw(void *indirect_command,
                                uint32_t primitive_type,
                                uint64_t vertex_start,
                                uint64_t vertex_count,
                                uint64_t instance_count,
                                uint64_t base_instance);
int mglRenderCppUseRenderResource(void *render_encoder,
                                  void *resource,
                                  uint32_t usage,
                                  uint32_t stages);
int mglRenderCppExecuteIndirectCommands(void *render_encoder,
                                        void *indirect_buffer,
                                        uint64_t location,
                                        uint64_t length);
int mglRenderCppReplayBatchDrawsForRenderEncoderOwner(
    void *render_encoder_owner,
    const MGLRenderCppReplayBatch *batch,
    char *err,
    size_t errcap);
int mglRenderCppUseRenderResourceForOwner(void *render_encoder_owner,
                                          void *resource,
                                          uint32_t usage,
                                          uint32_t stages);
int mglRenderCppExecuteIndirectCommandsForOwner(void *render_encoder_owner,
                                                 void *indirect_buffer,
                                                 uint64_t location,
                                                 uint64_t length);

#ifdef __cplusplus
} // extern "C"
#endif
