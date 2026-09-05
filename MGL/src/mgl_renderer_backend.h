/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#ifndef MGL_RENDERER_BACKEND_H
#define MGL_RENDERER_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mgl_render.h"
#include "mgl_renderer_sync.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GLMContextRec_t *GLMContext;
typedef struct MGLRendererBackendHandle MGLRendererBackendHandle;
typedef struct MGLShaderResource_t MGLShaderResource;
typedef struct MGLSamplerSnapshotKey MGLSamplerSnapshotKey;

typedef enum MGLRendererBackendOwnerKind {
    MGL_RENDERER_BACKEND_OWNER_COMMAND_QUEUE = 0,
    MGL_RENDERER_BACKEND_OWNER_COMMAND_BUFFER = 1,
    MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER = 2,
    MGL_RENDERER_BACKEND_OWNER_RENDER_PASS = 3,
    MGL_RENDERER_BACKEND_OWNER_QUERY = 4,
    MGL_RENDERER_BACKEND_OWNER_RECOVERY = 5,
    MGL_RENDERER_BACKEND_OWNER_BINDING = 6,
} MGLRendererBackendOwnerKind;

typedef enum MGLRendererBackendBlitCacheKind {
    MGL_RENDERER_BACKEND_BLIT_CACHE_NEAREST_SAMPLER = 0,
    MGL_RENDERER_BACKEND_BLIT_CACHE_LINEAR_SAMPLER = 1,
    MGL_RENDERER_BACKEND_BLIT_CACHE_CLEAR_DEPTH_STATE = 2,
} MGLRendererBackendBlitCacheKind;

typedef enum MGLRendererBackendPassthroughKind {
    MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY = 0,
    MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION = 1,
} MGLRendererBackendPassthroughKind;

typedef enum MGLRendererBackendDefaultDrawBufferAttachmentKind {
    MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR = 0,
    MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH = 1,
    MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL = 2,
} MGLRendererBackendDefaultDrawBufferAttachmentKind;

typedef enum MGLRendererBackendSizeConstantsStage {
    MGL_RENDERER_BACKEND_SIZE_CONSTANTS_VERTEX = 0,
    MGL_RENDERER_BACKEND_SIZE_CONSTANTS_FRAGMENT = 1,
} MGLRendererBackendSizeConstantsStage;

typedef enum MGLRendererBackendFallbackResourceKind {
    MGL_RENDERER_BACKEND_FALLBACK_SAMPLED_TEXTURE = 0,
    MGL_RENDERER_BACKEND_FALLBACK_CUBE_SAMPLED_TEXTURE = 1,
    MGL_RENDERER_BACKEND_FALLBACK_TEXTURE_BUFFER_STORAGE = 2,
    MGL_RENDERER_BACKEND_FALLBACK_SINT_TEXTURE_BUFFER = 3,
    MGL_RENDERER_BACKEND_FALLBACK_SAMPLER = 4,
} MGLRendererBackendFallbackResourceKind;

typedef struct MGLRendererBackendCreateInfo {
    void *objc_device;
    GLMContext context;
    uint32_t binding_slot_count;
    uint32_t query_capacity;
} MGLRendererBackendCreateInfo;

typedef struct MGLRendererBackendShutdownResult {
    int32_t status;
    uint32_t waited_for_last_submission;
    uint32_t last_submission_has_error;
    int64_t last_submission_error_code;
} MGLRendererBackendShutdownResult;

int mglRendererBackendCreate(const MGLRendererBackendCreateInfo *info,
                             MGLRendererBackendHandle **backend_out);
int mglRendererBackendIsReady(const MGLRendererBackendHandle *backend);
/* Immutable after create; returns the borrowed device retained by the backend. */
void *mglRendererBackendGetDevice(
    const MGLRendererBackendHandle *backend);
int mglRendererBackendResetCommandQueue(MGLRendererBackendHandle *backend,
                                        uint32_t max_command_buffers,
                                        void **command_queue_out);
/* Returns the current borrowed queue owned by CommandQueueOwner. */
void *mglRendererBackendGetCommandQueue(
    const MGLRendererBackendHandle *backend);
int mglRendererBackendAttachRuntimeOwners(MGLRendererBackendHandle *backend,
                                          void *command_buffer_owner,
                                          void *render_encoder_owner,
                                          void *render_pass_state_owner);
int mglRendererBackendSetFallbackRenderTargetTexture(
    MGLRendererBackendHandle *backend, void *texture);
/* Texture getters return borrowed references owned by the backend. */
void *mglRendererBackendGetFallbackRenderTargetTexture(
    const MGLRendererBackendHandle *backend);
/* Shared fallback buffers are borrowed references owned by the backend. */
void *mglRendererBackendGetFallbackBindingBuffer(
    MGLRendererBackendHandle *backend, uint64_t minimum_length);
void *mglRendererBackendGetCullDistanceDummyBuffer(
    MGLRendererBackendHandle *backend);
int mglRendererBackendSetTransientDepthTexture(
    MGLRendererBackendHandle *backend, void *texture,
    uint64_t width, uint64_t height);
void *mglRendererBackendGetTransientDepthTexture(
    const MGLRendererBackendHandle *backend,
    uint64_t *width_out, uint64_t *height_out);
int mglRendererBackendSetDefaultDrawBufferAttachment(
    MGLRendererBackendHandle *backend, uint32_t draw_buffer_index,
    MGLRendererBackendDefaultDrawBufferAttachmentKind kind, void *texture);
/* Returns a borrowed default draw-buffer attachment owned by the backend. */
void *mglRendererBackendGetDefaultDrawBufferAttachment(
    const MGLRendererBackendHandle *backend, uint32_t draw_buffer_index,
    MGLRendererBackendDefaultDrawBufferAttachmentKind kind);
int mglRendererBackendClearDefaultDrawBuffer(
    MGLRendererBackendHandle *backend, uint32_t draw_buffer_index);
int mglRendererBackendSetStageCopyBackResources(
    MGLRendererBackendHandle *backend, const void *copy_back_list_key,
    uint32_t slot, void *temporary, void *destination);
int mglRendererBackendGetStageCopyBackResources(
    const MGLRendererBackendHandle *backend, const void *copy_back_list_key,
    uint32_t slot, void **temporary_out, void **destination_out);
int mglRendererBackendClearStageCopyBackSlot(
    MGLRendererBackendHandle *backend, const void *copy_back_list_key,
    uint32_t slot);
int mglRendererBackendClearStageCopyBackList(
    MGLRendererBackendHandle *backend, const void *copy_back_list_key);
/* Returns a borrowed cached buffer when the value key and stride match. */
void *mglRendererBackendGetCurrentAttribBuffer(
    const MGLRendererBackendHandle *backend, uint32_t attrib,
    const void *bytes, uint32_t byte_count, uint64_t stride);
int mglRendererBackendSetCurrentAttribBuffer(
    MGLRendererBackendHandle *backend, uint32_t attrib,
    const void *bytes, uint32_t byte_count, uint64_t stride, void *buffer);
/* Packed current-value attribute pool: ONE shared Metal buffer holding
 * `repeat_count` copies of a MAX_ATTRIBS×16B value block, laid out as
 * [attrib][iteration] so a vertex descriptor addresses attrib `a` at
 * offset a*repeat_count*16 with stride 16.  Returns a borrowed buffer
 * when every packed byte and the repeat count match the cached pool. */
void *mglRendererBackendGetPackedCurrentAttribBuffer(
    const MGLRendererBackendHandle *backend, const void *bytes,
    uint32_t byte_count, uint32_t repeat_count);
int mglRendererBackendSetPackedCurrentAttribBuffer(
    MGLRendererBackendHandle *backend, const void *bytes,
    uint32_t byte_count, uint32_t repeat_count, void *buffer);
/* Size-constant cache getters return borrowed buffers owned by the backend. */
void *mglRendererBackendGetSizeConstantsBuffer(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendSizeConstantsStage stage,
    const uint32_t *constants, uint32_t count);
int mglRendererBackendSetSizeConstantsBuffer(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendSizeConstantsStage stage,
    const uint32_t *constants, uint32_t count, void *buffer);
int mglRendererBackendSetBlitCachedObject(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendBlitCacheKind kind, void *object);
/* Blit cache getters return borrowed references owned by the backend. */
void *mglRendererBackendGetBlitCachedObject(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendBlitCacheKind kind);
int mglRendererBackendSetPassthroughFunction(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendPassthroughKind kind,
    void *library, void *function, uint64_t program_instance_id);
/* Returns 1 on an exact cache hit, 0 on miss, and -1 for invalid input.
 * function_out is a borrowed reference owned by the backend. */
int mglRendererBackendGetPassthroughFunction(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendPassthroughKind kind,
    uint64_t program_instance_id, void **function_out);
/* Sampler snapshot states are retained by the backend. Get returns 1 on an
 * exact key hit, 0 on miss, and -1 for invalid input. state_out is borrowed. */
int mglRendererBackendGetSamplerSnapshotState(
    const MGLRendererBackendHandle *backend,
    const MGLSamplerSnapshotKey *key, void **state_out);
int mglRendererBackendPutSamplerSnapshotState(
    MGLRendererBackendHandle *backend,
    const MGLSamplerSnapshotKey *key, void *state);
/* TES-only default factor buffers are cached by patch count and six levels. */
int mglRendererBackendGetTessFactorBuffer(
    const MGLRendererBackendHandle *backend, uint32_t patch_count,
    const float levels[6], void **buffer_out);
int mglRendererBackendPutTessFactorBuffer(
    MGLRendererBackendHandle *backend, uint32_t patch_count,
    const float levels[6], void *buffer);
int mglRendererBackendSetCurrentTessFactorBuffer(
    MGLRendererBackendHandle *backend, void *buffer);
/* Returns the borrowed factor buffer selected for the current tess draw. */
void *mglRendererBackendGetCurrentTessFactorBuffer(
    const MGLRendererBackendHandle *backend);
/* Returns a borrowed TES XFB dummy buffer when it meets minimum_length. */
int mglRendererBackendGetTessXfbDummyBuffer(
    const MGLRendererBackendHandle *backend, uint64_t minimum_length,
    void **buffer_out);
int mglRendererBackendPutTessXfbDummyBuffer(
    MGLRendererBackendHandle *backend, void *buffer);
int mglRendererBackendSetCullDistanceCaptureBuffer(
    MGLRendererBackendHandle *backend, void *buffer);
/* Returns the borrowed cull-distance capture buffer owned by the backend. */
void *mglRendererBackendGetCullDistanceCaptureBuffer(
    const MGLRendererBackendHandle *backend);
int mglRendererBackendSetTessControlPointIndexBuffer(
    MGLRendererBackendHandle *backend, void *buffer);
/* Returns the borrowed indexed-TES gather buffer owned by the backend. */
void *mglRendererBackendGetTessControlPointIndexBuffer(
    const MGLRendererBackendHandle *backend);
int mglRendererBackendSetTessVertexCaptureBuffer(
    MGLRendererBackendHandle *backend, void *buffer);
/* Returns the borrowed VS capture buffer owned by the backend. */
void *mglRendererBackendGetTessVertexCaptureBuffer(
    const MGLRendererBackendHandle *backend);
int mglRendererBackendSetTcsPatchOutBuffer(
    MGLRendererBackendHandle *backend, void *buffer);
/* Returns the borrowed TCS per-patch output buffer owned by the backend. */
void *mglRendererBackendGetTcsPatchOutBuffer(
    const MGLRendererBackendHandle *backend);
int mglRendererBackendSetTcsOutputBuffer(
    MGLRendererBackendHandle *backend, void *buffer);
/* Returns the borrowed TCS per-vertex output buffer owned by the backend. */
void *mglRendererBackendGetTcsOutputBuffer(
    const MGLRendererBackendHandle *backend);
int mglRendererBackendSetFallbackResource(
    MGLRendererBackendHandle *backend,
    MGLRendererBackendFallbackResourceKind kind, void *resource);
/* Fallback resource getters return borrowed references owned by the backend. */
void *mglRendererBackendGetFallbackResource(
    const MGLRendererBackendHandle *backend,
    MGLRendererBackendFallbackResourceKind kind);
int mglRendererBackendGetFallbackSampledTexture(
    const MGLRendererBackendHandle *backend,
    uint64_t key, void **texture_out);
int mglRendererBackendPutFallbackSampledTexture(
    MGLRendererBackendHandle *backend,
    uint64_t key, void *texture);
int mglRendererBackendRetainProactiveTexture(
    MGLRendererBackendHandle *backend, void *texture);
/* Create, upload, and retain the initialization texture entirely in C++. */
int mglRendererBackendCreateProactiveTexture(
    MGLRendererBackendHandle *backend);
int mglRendererBackendIsDestroying(
    const MGLRendererBackendHandle *backend);
void *mglRendererBackendGetOwner(const MGLRendererBackendHandle *backend,
                                 MGLRendererBackendOwnerKind kind);
int mglRendererBackendShutdown(MGLRendererBackendHandle *backend,
                               MGLRendererBackendShutdownResult *result_out);
void mglRendererBackendDestroy(MGLRendererBackendHandle **backend);

/* Fixed renderer ABI. GL state calls these directly; capability is determined
 * once by mglRendererBackendIsReady rather than by nullable function slots. */
void mglRendererBindBuffer(GLMContext context, Buffer *buffer);
void mglRendererBindTexture(GLMContext context, Texture *texture);
void mglRendererBindProgram(GLMContext context, Program *program);
void mglRendererDeleteMetalObject(GLMContext context, void *object);
void mglRendererReleaseBufferMetalData(GLMContext context, Buffer *buffer);
void mglRendererGetSync(GLMContext context, Sync *sync);
void mglRendererWaitForSync(GLMContext context, Sync *sync);
uint32_t mglRendererGetSyncStatus(GLMContext context, Sync *sync);
void mglRendererReleaseSync(GLMContext context, Sync *sync);
void mglRendererFlush(GLMContext context, bool finish);
void mglRendererSwapBuffers(GLMContext context);
void mglRendererFlushDrawBuffer(GLMContext context);
int mglRendererProcessGLState(GLMContext context, int draw_command);
void mglRendererInvalidateRenderPass(GLMContext context);
void mglRendererClearBuffer(GLMContext context, uint32_t type, uint32_t mask);
void mglRendererBlitFramebuffer(GLMContext context,
                                int32_t src_x0, int32_t src_y0,
                                int32_t src_x1, int32_t src_y1,
                                int32_t dst_x0, int32_t dst_y0,
                                int32_t dst_x1, int32_t dst_y1,
                                uint32_t mask, uint32_t filter);
void mglRendererBufferSubData(GLMContext context, Buffer *buffer,
                              size_t offset, size_t size, const void *bytes);
void *mglRendererMapUnmapBuffer(GLMContext context, Buffer *buffer,
                                size_t offset, size_t size,
                                uint32_t access, bool map);
void mglRendererReadBackBuffer(GLMContext context, Buffer *buffer,
                               size_t offset, size_t size);
/* imageStore on GL_TEXTURE_BUFFER writes a Metal texture2d fallback; copy
 * those texels back into the attached Buffer (CPU + Metal) so subsequent
 * glGetBufferSubData observes the shader writes. */
void mglRendererSyncTextureBufferFromImage(GLMContext context, Texture *texture);
/* Non-layered BindImage of GL_TEXTURE_3D: Metal cannot view a depth plane as
 * Type2D. Prepare a staging 2D texture (and Flush writes it back). */
void mglRendererPrepareImageUnitSlice(GLMContext context, uint32_t unit);
void mglRendererFlushImageUnitSlice(GLMContext context, uint32_t unit);
/* BindImageTexture <format>/level/slice → Metal texture (or cached view).
 * Borrowed pointer; view lifetime is owned by iu->mtl_image_view. */
typedef struct ImageUnit_t ImageUnit;
void *mglRendererStorageImageTexture(void *base_texture, ImageUnit *iu);
void mglRendererFlushBufferRange(GLMContext context, Buffer *buffer,
                                 intptr_t offset, intptr_t length);
void mglRendererReadDrawable(GLMContext context, void *pixel_bytes,
                             uint32_t bytes_per_row, uint32_t bytes_per_image,
                             int32_t x, int32_t y,
                             int32_t width, int32_t height);
void mglRendererReadIntegerPixels(GLMContext context, void *pixel_bytes,
                                  uint32_t bytes_per_row,
                                  uint32_t bytes_per_image,
                                  int32_t x, int32_t y,
                                  int32_t width, int32_t height,
                                  uint32_t format, uint32_t type);
void mglRendererReadDepthPixels(GLMContext context, void *pixel_bytes,
                                uint32_t bytes_per_row,
                                uint32_t bytes_per_image,
                                int32_t x, int32_t y,
                                int32_t width, int32_t height);
void mglRendererGetTexImage(GLMContext context, Texture *texture,
                            void *pixel_bytes,
                            uint32_t bytes_per_row, uint32_t bytes_per_image,
                            int32_t x, int32_t y,
                            int32_t width, int32_t height,
                            uint32_t format, uint32_t type,
                            uint32_t level, uint32_t slice);
void mglRendererGenerateMipmaps(GLMContext context, Texture *texture);
void mglRendererTexSubImage(GLMContext context, Texture *texture,
                            Buffer *buffer,
                            size_t source_offset, size_t source_pitch,
                            size_t source_image_size, size_t source_size,
                            uint32_t slice, uint32_t level,
                            size_t width, size_t height, size_t depth,
                            size_t x_offset, size_t y_offset, size_t z_offset);
bool mglRendererTexSubImageBytes(GLMContext context, Texture *texture,
                                 const void *bytes, size_t bytes_size,
                                 size_t source_offset, size_t source_pitch,
                                 size_t source_image_size,
                                 uint32_t slice, uint32_t level,
                                 size_t width, size_t height, size_t depth,
                                 size_t x_offset, size_t y_offset,
                                 size_t z_offset);
void mglRendererCopyTexSubImage(GLMContext context, Texture *texture,
                                uint32_t slice, int32_t level,
                                int32_t x_offset, int32_t y_offset,
                                int32_t x, int32_t y,
                                int32_t width, int32_t height);
void mglRendererCopyImageSubData(GLMContext context,
                                 Texture *source_texture,
                                 int32_t source_level,
                                 int32_t source_x, int32_t source_y,
                                 int32_t source_z,
                                 Texture *destination_texture,
                                 int32_t destination_level,
                                 int32_t destination_x, int32_t destination_y,
                                 int32_t destination_z,
                                 int32_t width, int32_t height, int32_t depth);
void mglRendererDrawArrays(GLMContext context, uint32_t mode,
                           int32_t first, int32_t count);
void mglRendererDrawElements(GLMContext context, uint32_t mode,
                             int32_t count, uint32_t type,
                             const void *indices);
void mglRendererDrawRangeElements(GLMContext context, uint32_t mode,
                                  uint32_t start, uint32_t end,
                                  int32_t count, uint32_t type,
                                  const void *indices);
void mglRendererDrawArraysInstanced(GLMContext context, uint32_t mode,
                                    int32_t first, int32_t count,
                                    int32_t instance_count);
void mglRendererDrawElementsInstanced(GLMContext context, uint32_t mode,
                                      int32_t count, uint32_t type,
                                      const void *indices,
                                      int32_t instance_count);
void mglRendererDrawElementsBaseVertex(GLMContext context, uint32_t mode,
                                       int32_t count, uint32_t type,
                                       const void *indices,
                                       int32_t base_vertex);
void mglRendererDrawRangeElementsBaseVertex(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex);
void mglRendererDrawElementsInstancedBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex);
void mglRendererDrawArraysIndirect(GLMContext context, uint32_t mode,
                                   const void *indirect);
void mglRendererDrawElementsIndirect(GLMContext context, uint32_t mode,
                                     uint32_t type, const void *indirect);
void mglRendererDrawArraysInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count, uint32_t base_instance);
void mglRendererDrawElementsInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, uint32_t base_instance);
void mglRendererDrawElementsInstancedBaseVertexBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex,
    uint32_t base_instance);
void mglRendererMultiDrawArrays(GLMContext context, uint32_t mode,
                                const int32_t *firsts,
                                const int32_t *counts,
                                int32_t draw_count);
void mglRendererMultiDrawElements(GLMContext context, uint32_t mode,
                                  const int32_t *counts, uint32_t type,
                                  const void *const *indices,
                                  int32_t draw_count);
void mglRendererMultiDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, const int32_t *counts, uint32_t type,
    const void *const *indices, int32_t draw_count,
    const int32_t *base_vertices);
void mglRendererMultiDrawArraysIndirect(GLMContext context, uint32_t mode,
                                        const void *indirect,
                                        int32_t draw_count, int32_t stride);
void mglRendererMultiDrawElementsIndirect(GLMContext context, uint32_t mode,
                                          uint32_t type,
                                          const void *indirect,
                                          int32_t draw_count,
                                          int32_t stride);
void mglRendererDispatchCompute(GLMContext context,
                                uint32_t groups_x,
                                uint32_t groups_y,
                                uint32_t groups_z);
void mglRendererDispatchComputeIndirect(GLMContext context,
                                        intptr_t indirect);
void mglRendererBeginSampleQuery(GLMContext context, uint32_t target);
uint64_t mglRendererEndSampleQuery(GLMContext context);
void mglRendererBeginTimerQuery(GLMContext context);
uint64_t mglRendererEndTimerQuery(GLMContext context);
uint64_t mglRendererGetGPUTimestamp(GLMContext context);

int32_t mglRendererGetProgramBindingCount(GLMContext context,
                                          int32_t stage,
                                          int32_t type);
int32_t mglRendererGetProgramBinding(GLMContext context,
                                     int32_t stage,
                                     int32_t type,
                                     int32_t index);
int32_t mglRendererGetProgramGLBinding(GLMContext context,
                                       int32_t stage,
                                       int32_t type,
                                       int32_t index);
int32_t mglRendererGetProgramLocation(GLMContext context,
                                      int32_t stage,
                                      int32_t type,
                                      int32_t index);
size_t mglRendererGetProgramBindingRequiredSize(GLMContext context,
                                                int32_t stage,
                                                int32_t type,
                                                int32_t index);
size_t mglRendererGetProgramBindingRequiredSizeForStage(
    GLMContext context, int32_t stage, uint32_t client_binding);
intptr_t mglRendererGetProgramMetalBufferIndexForStage(
    GLMContext context, int32_t stage, uint32_t client_binding);
uint32_t mglRendererGetProgramDeclaredTextureType(GLMContext context,
                                                  int32_t stage,
                                                  int32_t type,
                                                  int32_t index);
uint32_t mglRendererGetProgramExpectedTextureType(GLMContext context,
                                                  int32_t stage,
                                                  int32_t type,
                                                  int32_t index);
uint32_t mglRendererGetProgramExpectedTextureDataKind(GLMContext context,
                                                      int32_t stage,
                                                      int32_t type,
                                                      int32_t index);
uint32_t mglDeclaredTextureTypeFromResource(const MGLShaderResource *resource);
uint32_t mglExpectedTextureTypeForResource(Program *program,
                                           int32_t stage,
                                           MGLShaderResource *resource);
uint32_t mglExpectedTextureDataKindForResource(
    Program *program, int32_t stage, MGLShaderResource *resource);

#ifdef __cplusplus
}
#endif

#endif
