/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_BUFFER_H
#define MGL_RENDER_API_BUFFER_H

/* Declarations for the buffer slice of the renderer facade.
 * Value layouts are defined here; types of other subsystems that are only
 * passed by pointer are incomplete (mgl_render_fwd.h). */

#include "glcorearb.h"
#include "mgl_render_fwd.h"

/* Client-facing buffer vocabulary: what a map access allows and where the
 * storage lives.  Kept here so the backend asks a question, not an enum. */
static inline bool mglBufferAccessIsReadable(uint32_t access)
{
    return access == GL_READ_ONLY || access == GL_READ_WRITE ||
           (access & GL_MAP_READ_BIT) != 0;
}

static inline bool mglBufferMapIsWritable(uint32_t access_flags)
{
    return (access_flags & GL_MAP_WRITE_BIT) != 0;
}

static inline bool mglBufferMapIsCoherent(uint32_t access_flags)
{
    return (access_flags & GL_MAP_COHERENT_BIT) != 0;
}

static inline bool mglBufferStorageIsMapped(uint32_t storage_flags)
{
    return (storage_flags & GL_MAP_READ_BIT) != 0;
}

static inline bool mglBufferStorageIsPersistent(uint32_t storage_flags)
{
    return (storage_flags & GL_MAP_PERSISTENT_BIT) != 0;
}

static inline bool mglBufferStorageIsClient(uint32_t storage_flags)
{
    return (storage_flags & GL_CLIENT_STORAGE_BIT) != 0;
}

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MGL_RENDER_BUFFER_BOUND = 0,
    MGL_RENDER_BUFFER_NOT_APPLICABLE = 1,
    MGL_RENDER_BUFFER_ERROR = -1,
};

enum {
    MGL_RENDER_BUFFER_OPERATION_HANDLED = 0,
    MGL_RENDER_BUFFER_OPERATION_NOT_APPLICABLE = 1,
    MGL_RENDER_BUFFER_OPERATION_ERROR = -1,
};

typedef enum MGLRenderVertexConversionKind_t {
    MGL_RENDER_VERTEX_DOUBLE_TO_FLOAT = 0,
    MGL_RENDER_VERTEX_INT_TO_FLOAT = 1,
    MGL_RENDER_VERTEX_FIXED_TO_FLOAT = 2,
    MGL_RENDER_VERTEX_PACKED_1010102_TO_FLOAT = 3,
    MGL_RENDER_VERTEX_PACKED_10F11F11F_TO_FLOAT = 4,
    MGL_RENDER_VERTEX_INTEGER_TO_32 = 5,
} MGLRenderVertexConversionKind;

typedef struct MGLRenderVertexConversion_t {
    uint32_t kind;
    uint32_t component_count;
    uint32_t source_type;
    uint32_t normalized;
    uint32_t destination_signed;
    int64_t binding_offset;
    int64_t relative_offset;
    uint64_t stride;
} MGLRenderVertexConversion;

typedef struct MGLRenderBufferInfo_t {
    uint64_t length;
} MGLRenderBufferInfo;

/* runtime-array-size SSBO sizing constants.  The AIR
 * backend emits code that reads uint32 byte-sizes from
 * MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX when a compute shader uses .length()
 * on an unsized SSBO array.  This fills `out_sizes[out_capacity]` from the
 * per-buffer {metal_slot, visible_size} pairs, skipping the runtime-size
 * buffer slot itself and any slot >= max_slot (the ordinary user-buffer table
 * cap, kMGLMaxMetalUserBufferCount=31).  `out_sizes` is expected to be
 * zero-initialized by the caller; only claimed slots are written.  Returns
 * 0 on success, -1 on bad args (NULL out, NULL entries with nonzero count,
 * out_capacity < max_slot). */
typedef struct MGLRenderBufferSizeEntry_t {
    uint32_t metal_slot;      /* Metal buffer argument index */
    uint64_t visible_size;    /* byte size, truncated to uint32 by the facade */
} MGLRenderBufferSizeEntry;

int mglRenderBuildRuntimeArraySizes(
    const MGLRenderBufferSizeEntry *entries,
    uint32_t entry_count,
    uint32_t runtime_buffer_index,
    uint32_t max_slot,
    uint32_t *out_sizes,
    uint32_t out_capacity);

int mglRenderCPUPointerUsable(const void *p);

int mglRenderShaderResourceToGLBufferType(int spvc_type);

int mglRenderUsePlainUniformBuffers(int spvc_type);

int mglRenderStructMemberInElementRange(uint32_t member_loc_off,
                                        uint32_t loc_start, uint32_t loc_end);

int mglRenderBindableLocValid(int32_t loc, uint32_t max);

int mglRenderCPUShadowReadable(const void *cpu, int64_t size);

uint64_t mglRenderClampCopyToStruct(uint64_t dest_off, uint64_t copy_size,
                                    uint64_t struct_size);

int64_t mglRenderMappedUniformSize(int spvc_type, int64_t bound, int64_t buf_size,
                                   int64_t offset, uint64_t reflected);

int32_t mglRenderPlainUniformBaseLoc(int32_t uniform_location,
                                     uint32_t location);

uint32_t mglRenderMemberOffsetInElement(uint32_t offset,
                                        uint32_t elem_byte_start);

int mglRenderMemberOffsetInStruct(uint32_t offset, uint32_t struct_size);

void mglRenderPlainUniformArrayStrides(const char *name, uint32_t type_bytes,
                                       int32_t array_stride, uint32_t *src_out,
                                       uint32_t *elem_out);

int mglRenderMetalBackingTooSmall(int64_t gl_size, uint64_t metal_length);


void mglRenderReleaseBufferMetalData(GLMContext glm_ctx, Buffer *buffer);

/* Release the backend state kept for `buffer` (CoW snapshot pool, expanded
 * index copy) and free it; the buffer keeps no backend state afterwards. */
void mglRenderReleaseBufferBackend(Buffer *buffer);

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

/* Current (minted, not yet completed) command-buffer generation.  Tagging a
 * recycled scratch buffer with this value and reusing it only once
 * mglRenderCompletedBufferGeneration() reaches it fences in-flight reads. */
uint64_t mglRenderCurrentBufferGeneration(void);

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
