/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_BUFFER_H
#define MGL_RENDER_API_BUFFER_H

/* Declarations for the buffer slice of the renderer facade.
 * Value layouts live in mgl_render.h. Standalone include pulls
 * mgl_render_fwd.h (incomplete types) instead of the full facade. */

#ifndef MGL_RENDER_H
#include "mgl_render_fwd.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void mglRenderReleaseBufferMetalData(GLMContext glm_ctx, Buffer *buffer);

void mglRenderReleaseBufferCowPool(Buffer *buffer);

void mglRenderBufferSubData(GLMContext glm_ctx,
                               Buffer *buffer,
                               size_t offset,
                               size_t size,
                               const void *bytes);

void *mglRenderMapUnmapBuffer(GLMContext glm_ctx,
                                 Buffer *buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map);

void mglRenderReadBackBuffer(GLMContext glm_ctx,
                                Buffer *buffer,
                                size_t offset,
                                size_t size);

void mglRenderFlushBufferRange(GLMContext glm_ctx,
                                  Buffer *buffer,
                                  intptr_t offset,
                                  intptr_t length);

/* Update buffer storage and dirty state for an encoder bind. */
int mglRenderUpdateDirtyBuffer(Buffer *buffer,
                                  char *err,
                                  size_t errcap);

int mglRenderBufferSubDataStorage(Buffer *buffer,
                                     size_t offset,
                                     size_t size,
                                     const void *bytes,
                                     char *err,
                                     size_t errcap);

int mglRenderSnapshotSharedDirtyBuffer(Buffer *buffer,
                                          void **metal_buffer_out,
                                          char *err,
                                          size_t errcap);

int mglRenderSnapshotSharedBufferRange(Buffer *buffer,
                                          size_t offset,
                                          size_t length,
                                          void **metal_buffer_out,
                                          char *err,
                                          size_t errcap);

uint64_t mglRenderAdvanceBufferGeneration(void);

void mglRenderRecordBufferGenerationCompleted(uint64_t generation);

uint64_t mglRenderCompletedBufferGeneration(void);

void mglRenderNoteBufferEncoded(Buffer *buffer);

int mglRenderMapBufferStorage(Buffer *buffer,
                                 size_t offset,
                                 size_t size,
                                 unsigned int access,
                                 bool map,
                                 void **mapped_out,
                                 char *err,
                                 size_t errcap);

int mglRenderFlushBufferRangeStorage(Buffer *buffer,
                                         intptr_t offset,
                                         intptr_t length,
                                         char *err,
                                         size_t errcap);

/* Convert unsupported GL vertex formats and return a +1 MTLBuffer as void*.
 * The caller must consume it with __bridge_transfer or release it through
 * mglRenderDeleteMTLObj. The renderer cache owns a separate reference. */
int mglRenderConvertVertexBuffer(
    Buffer *source_buffer,
    const MGLRenderVertexConversion *conversion,
    uint64_t *converted_stride_out,
    void **converted_buffer_out,
    char *err,
    size_t errcap);

/* Pack a plain-struct uniform into renderer-owned transient storage. The
 * returned Buffer wrapper remains owned by the renderer; its MTLBuffer
 * backing is replaced when the 128-slot ring wraps. */
Buffer *mglRenderAcquirePackedStructBuffer(const void *data,
                                               size_t size,
                                               char *err,
                                               size_t errcap);

/* Renderer-owned device utility facade. Newly created resources are +1 and
 * must be consumed by __bridge_transfer or released through the C++ facade. */
int mglRenderCreateBuffer(uint64_t length,
                             uint64_t resource_options,
                             const char *label,
                             void **buffer_out);

int mglRenderCreateBufferWithBytes(const void *bytes,
                                      uint64_t length,
                                      uint64_t resource_options,
                                      const char *label,
                                      void **buffer_out);

/* Create a shared/no-copy buffer for VM-backed GL client or persistent
 * storage.  When deallocate_vm is non-zero Metal owns the VM range and
 * releases it with vm_deallocate after the last in-flight command buffer. */
int mglRenderCreateBufferWithBytesNoCopy(const void *bytes,
                                            uint64_t length,
                                            uint64_t resource_options,
                                            const char *label,
                                            int deallocate_vm,
                                            void **buffer_out);

int mglRenderGetBufferContents(void *buffer,
                                  void **contents_out,
                                  uint64_t *length_out);

int mglRenderGetBufferInfo(const void *buffer,
                              MGLRenderBufferInfo *info_out);

int mglRenderAddBufferDebugMarker(void *buffer,
                                     const char *marker,
                                     uint64_t location,
                                     uint64_t length);

/* tess-factor buffer CPU transforms — the default
 * canonical factor fill (RECORD_BYTES/patch: 12B half + 24B exact f32),
 * the canonical->triangle repack (RECORD -> 8B/patch halves) and the
 * primitive count from the shared domain engine (EQUAL spacing floor;
 * Program-aware callers use mglTessGeneratedPrimitiveCount).
 * Return 0 on success, -1 on bad args (count entry returns 0). */
int mglRenderFillDefaultTessFactorBuffer(
    void *dst,
    uint64_t dst_bytes,
    const float *outer_levels,
    const float *inner_levels,
    uint32_t patch_count);

/* shadow-upload range math — for gpu_write_target
 * buffers, clamps the recorded written_min/written_max span to the limit;
 * otherwise the whole limit.  Returns 0 with offset/length set, -1 when
 * there is nothing to upload (no written span / zero length).  Pure range
 * computation shared by both gates. */
int mglRenderBufferShadowUploadRange(
    int gpu_write_target,
    int64_t written_min,
    int64_t written_max,
    uint64_t limit,
    uint64_t *out_offset,
    uint64_t *out_length);

int mglRenderHasDirtyBufferBit(uint32_t dirty_bits);

int mglRenderDefaultReadBufferIndex(uint32_t read_buffer, uint32_t *out);

int mglRenderFBOReadBufferValid(uint32_t read_buffer, uint32_t max_color,
                                uint32_t max_attach);

/* C1: MSAAArrayLayerStride -> mgl_readback_policy.h (EncodeMultisampleResolve residual in cpp) */
int mglRenderDefaultDrawBufferIndex(uint32_t draw_buffer, uint32_t *out);

uint32_t mglRenderEmptyDrawBuffer(void);

uint32_t mglRenderDefaultFrontBuffer(void);

int mglRenderShouldPresentDrawBuffer(uint32_t draw_buffer);

int mglRenderDrawBufferIsColorAttachment(uint32_t draw_buffer, uint32_t max,
                                         uint32_t *out_index);

int mglRenderDrawBufferIsDefaultFBOCompat(uint32_t draw_buffer);

int mglRenderDefaultDrawBufferIsFront(uint32_t mgl_drawbuffer);

int mglRenderDefaultDrawBufferIsOffscreen(uint32_t mgl_drawbuffer,
                                          uint32_t max_draw_buffers);

int mglRenderBufferHasMapWriteBit(uint32_t access_flags);

int mglRenderBufferNeedsCPUUpload(int64_t size, uint32_t dirty_bits);

int mglRenderBufferHasCPUDirty(uint32_t dirty_bits);

int mglRenderBufferSlotInRange(int32_t slot, uint32_t max_slots);

int mglRenderResolveMappedBufferSlot(int has_metal_binding,
                                     int32_t metal_binding_index,
                                     int32_t buffer_base_index,
                                     uint32_t max_slots, uint32_t *out_slot);

int mglRenderBufferMapOffsetValid(int64_t offset);

int mglRenderBufferSizeValid(int64_t size);

uint64_t mglRenderBufferSizeOrZero(int64_t size);

int mglRenderBufferPlanIsStructPacked(uint32_t flags);

int mglRenderBufferPlanAllowFallback(int has_fallback, uint32_t flags);

int mglRenderAllowGlobalBufferFallback(int has_fallback, int spvc_type,
                                       uint32_t flags);

int mglRenderIsUniformBufferResource(int spvc_type);

int mglRenderGetQueryVisibilityBuffer(MGLQueryStateOwner *owner, void **visibility_buffer_out);

int mglRenderSetComputeBuffer(void *compute_encoder,
                                 void *buffer,
                                 uint64_t offset,
                                 uint32_t index);

void mglRenderMarkBufferCPUWrite(Buffer *buf, int64_t offset, int64_t size);

int mglRenderSetRenderBuffer(void *render_encoder,
                                void *buffer,
                                uint64_t offset,
                                uint32_t stage,
                                uint32_t index);

int mglRenderSetTessellationFactorBuffer(void *render_encoder,
                                            void *buffer,
                                            uint64_t offset,
                                            uint64_t instance_stride);

int mglRenderSetRenderBufferForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *buffer, uint64_t offset, uint32_t stage, uint32_t index);

int mglRenderSetTessellationFactorBufferForOwner(MGLRenderEncoderOwner *render_encoder_owner, void *buffer, uint64_t offset, uint64_t instance_stride);

#ifdef __cplusplus
}
#endif

#endif
