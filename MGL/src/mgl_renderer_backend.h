#ifndef MGL_RENDERER_BACKEND_H
#define MGL_RENDERER_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mgl_render_cpp.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GLMContextRec_t *GLMContext;
typedef struct MGLRendererBackendHandle MGLRendererBackendHandle;

typedef enum MGLRendererBackendOwnerKind {
    MGL_RENDERER_BACKEND_OWNER_COMMAND_QUEUE = 0,
    MGL_RENDERER_BACKEND_OWNER_COMMAND_BUFFER = 1,
    MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER = 2,
    MGL_RENDERER_BACKEND_OWNER_RENDER_PASS = 3,
    MGL_RENDERER_BACKEND_OWNER_QUERY = 4,
    MGL_RENDERER_BACKEND_OWNER_RECOVERY = 5,
    MGL_RENDERER_BACKEND_OWNER_BINDING = 6,
} MGLRendererBackendOwnerKind;

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
int mglRendererBackendResetCommandQueue(MGLRendererBackendHandle *backend,
                                        uint32_t max_command_buffers,
                                        void **command_queue_out);
int mglRendererBackendAttachRuntimeOwners(MGLRendererBackendHandle *backend,
                                          void *command_buffer_owner,
                                          void *render_encoder_owner,
                                          void *render_pass_state_owner);
int mglRendererBackendInstallCallbackRuntime(
    MGLRendererBackendHandle *backend,
    void *callback_runtime);
void *mglRendererBackendGetCallbackRuntime(
    const MGLRendererBackendHandle *backend);
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

#ifdef __cplusplus
}
#endif

#endif
