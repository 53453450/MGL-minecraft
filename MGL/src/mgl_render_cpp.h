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

typedef struct GLMContextRec_t *GLMContext;
typedef struct Buffer_t Buffer;
typedef struct TextureParameter_t TextureParameter;
typedef struct Program_t Program;
typedef struct __GLsync Sync;

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

/* GLMMetalFuncs entries that do not require access to the ObjC renderer.
 * Objects passed here carry the +1 bridge reference owned by the GL state. */
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
void mglRenderCppWaitForSync(GLMContext glm_ctx, Sync *sync);
unsigned int mglRenderCppGetSyncStatus(GLMContext glm_ctx, Sync *sync);
void mglRenderCppReleaseSync(GLMContext glm_ctx, Sync *sync);

enum {
    MGL_RENDER_CPP_AIR_PROGRAM_BOUND = 0,
    MGL_RENDER_CPP_AIR_PROGRAM_NOT_APPLICABLE = 1,
    MGL_RENDER_CPP_AIR_PROGRAM_ERROR = -1,
};

/* Load every AIR-backed stage in a linked Program and install the resulting
 * +1 library/function references directly in its Spirv slots.  Programs that
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

/* Materialize ordinary shared/copy-backed Buffer storage in Metal-cpp.
 * Client-storage and persistent no-copy buffers return NOT_APPLICABLE because
 * their custom vm_deallocate ownership remains in the ObjC baseline. */
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
int mglRenderCppLookupPipelineDescriptor(
    void *owner,
    const uint64_t key_words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS],
    void **descriptor_out);
int mglRenderCppStorePipelineDescriptor(
    void *owner,
    const uint64_t key_words[MGL_RENDER_CPP_PIPELINE_CACHE_KEY_WORDS],
    void *descriptor);
int mglRenderCppCreateEvent(void **event_out);
int mglRenderCppCreateMetal4Compiler(const char *label,
                                     void **compiler_out,
                                     char *err,
                                     size_t errcap);
int mglRenderCppCompileLibrary(void *compiler,
                               void *source_string,
                               void *compile_options,
                               const char *label,
                               void **library_out,
                               char *err,
                               size_t errcap);
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
int mglRenderCppCreateComputePipelineState(void *function,
                                           void **pipeline_out,
                                           char *err,
                                           size_t errcap);
int mglRenderCppCreateBinaryArchive(void *binary_archive_descriptor,
                                    const char *label,
                                    void **binary_archive_out,
                                    char *err,
                                    size_t errcap);
int mglRenderCppSetRenderPipelineBinaryArchive(
    void *render_pipeline_descriptor,
    void *binary_archive);
int mglRenderCppAddRenderPipelineFunctionsToBinaryArchive(
    void *binary_archive,
    void *render_pipeline_descriptor,
    char *err,
    size_t errcap);
int mglRenderCppSerializeBinaryArchive(void *binary_archive,
                                       void *url,
                                       char *err,
                                       size_t errcap);
int mglRenderCppSetVisibilityResultMode(void *render_encoder,
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
int mglRenderCppBindingSetScissor(void *binding_state,
                                 void *render_encoder,
                                 uint64_t x,
                                 uint64_t y,
                                 uint64_t width,
                                 uint64_t height);
int mglRenderCppBindingSetTriangleFill(void *binding_state,
                                      void *render_encoder,
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
int mglRenderCppAddCommandBufferCompletion(
    void *command_buffer,
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
int mglRenderCppResetCommandBufferOwner(void *owner,
                                        void *command_queue,
                                        void **command_buffer_out);
void mglRenderCppDiscardCommandBufferOwnerCurrent(void *owner);
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
int mglRenderCppPresentDrawable(void *command_buffer, void *drawable);

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
    uint64_t depth_plane);
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
int mglRenderCppGetRenderPassStateOwner(
    void *owner, MGLRenderCppRenderPassState *state_out);
int mglRenderCppCreateRenderEncoderFromStateOwner(
    void *command_buffer, void *state_owner, void **render_encoder_out);
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
int mglRenderCppEncodeDepthClear(void *command_buffer,
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
void mglRenderCppDestroyRenderEncoderOwner(void **owner);
int mglRenderCppEndRenderEncoder(void *render_encoder);
int mglRenderCppCreateBlitEncoder(void *command_buffer,
                                  void **blit_encoder_out);
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

#ifdef __cplusplus
} // extern "C"
#endif
