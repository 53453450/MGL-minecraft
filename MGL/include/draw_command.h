/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * draw_command.h
 * MGL
 *
 */

#ifndef draw_command_h
#define draw_command_h

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct GLMContextRec_t;
typedef struct GLMContextRec_t *GLMContext;

#define MGL_MAX_DRAWS_PER_BATCH   4096
#define MGL_MAX_BATCHES           128
#define MGL_MDI_MIN_BATCH_SIZE    2
#define MGL_MAX_PENDING_BUFFER_RANGES 4096
#define MGL_BUFFER_RANGE_BUCKET_SIZE  1024  /* power of two; chains per buffer-ptr hash */
#define MGL_BUFFER_RANGE_BUCKET_MASK  (MGL_BUFFER_RANGE_BUCKET_SIZE - 1)
#define MGL_MAX_PENDING_TEXTURE_WRITES 256
#define MGL_MAX_PENDING_TEXTURE_READS 512
#define MGL_MAX_DYNAMIC_UNIFORM_BINDINGS 8
#define MGL_MAX_DYNAMIC_TEXTURE_BINDINGS 8
#define MGL_MAX_DYNAMIC_VERTEX_BINDINGS 8
#define MGL_MAX_SAMPLER_SNAPSHOT_KEYS 128
#define MGL_MAX_SAMPLER_SNAPSHOT_SETS 256
#define MGL_MAX_SAMPLER_SNAPSHOT_ENTRIES 8
#define MGL_SAMPLER_SNAPSHOT_KEY_INDEX_SIZE 256
#define MGL_SAMPLER_SNAPSHOT_SET_INDEX_SIZE 512
#define MGL_INVALID_SAMPLER_SNAPSHOT_ID UINT16_MAX
#define MGL_FALLBACK_SAMPLER_KEY_INDEX UINT16_MAX

/* open-addressing hash-set index sizes (2× the array capacity, rounded
 * up to a power of two, so load factor stays ≤ 0.5 for O(1) probe length).
 * Each slot stores (array_index + 1); 0 means empty.  The backing object
 * arrays remain the source of truth for iteration — the index only accelerates
 * dedup (mglTrackPendingTexture*) and membership (mglPendingDrawsWrite/ReadTexture). */
#define MGL_TEX_WRITE_INDEX_SIZE  512   /* 2 × MGL_MAX_PENDING_TEXTURE_WRITES */
#define MGL_TEX_READ_INDEX_SIZE   1024  /* 2 × MGL_MAX_PENDING_TEXTURE_READS */
#define MGL_TEX_WRITE_INDEX_MASK  (MGL_TEX_WRITE_INDEX_SIZE - 1)
#define MGL_TEX_READ_INDEX_MASK   (MGL_TEX_READ_INDEX_SIZE - 1)

/* Bump-allocator arena for batch snapshot allocations (Task 4).
 * Gated by env var MGL_ARENA_SNAPSHOT (default ON; =0 disables).  When
 * enabled, state_snapshot, vao_snapshot, and the commands array are allocated
 * from this arena instead of individual malloc calls, and freed via arena
 * reset instead of individual free calls.  The arena is owned by MGLRenderer
 * and accessed from C via GLMContextRec_t::batch_arena. */
typedef struct MGLBatchArenaChunk MGLBatchArenaChunk;

typedef struct MGLBatchArena {
    MGLBatchArenaChunk *head;
    MGLBatchArenaChunk *current;
    size_t              initial_capacity;
    int                 enabled;
} MGLBatchArena;

typedef enum {
    MGL_CMD_DRAW_ARRAYS = 0,
    MGL_CMD_DRAW_ELEMENTS,
    MGL_CMD_DRAW_ARRAYS_INSTANCED,
    MGL_CMD_DRAW_ELEMENTS_INSTANCED,
    MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX,
    MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX,
    MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE,
    MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE,
    MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE,
} MGLDrawCommandType;

typedef struct {
    uint16_t binding_index;
    GLintptr offset;
    GLsizeiptr size;
} MGLDynamicUniformBinding;

typedef struct {
    uint8_t unit;
    uint8_t target_index;
    uint8_t is_active;
    void   *texture;
} MGLDynamicTextureBinding;

/* Per-draw vertex binding override for BindNoFlush merges.
 * `offset` is the absolute VERTEX_BINDING_OFFSET (GL 4.6 Table 23.4) for
 * this draw — not a delta vs the batch VAO snapshot. Storing absolute
 * offsets keeps BuildDynamicVertexArray and the direct Metal rebind path
 * consistent when the batch base slice is non-zero. */
typedef struct {
    void    *buffer;
    uint32_t offset;
    uint8_t  binding_index;
    uint8_t  reserved[3];
} MGLDynamicVertexBinding;

/* Immutable value key for the subset of TextureParameter represented by an
 * MTLSamplerState. Keeping this type independent of mgl_types_texture.h
 * avoids the glm_context.h / mgl_types_state.h include cycle. */
typedef struct MGLSamplerSnapshotKey {
    uint32_t target;
    uint32_t min_filter;
    uint32_t mag_filter;
    uint32_t wrap_s;
    uint32_t wrap_t;
    uint32_t wrap_r;
    uint32_t compare_mode;
    uint32_t compare_func;
    float    max_anisotropy;
    float    min_lod;
    float    max_lod;
    float    border_color[4];
} MGLSamplerSnapshotKey;

typedef struct {
    uint16_t key_index;
    uint8_t  stage;
    uint8_t  metal_slot;
    uint8_t  texture_unit;
    uint8_t  target_index;
} MGLSamplerSnapshotEntry;

typedef struct {
    uint8_t count;
    uint8_t reserved;
    MGLSamplerSnapshotEntry entries[MGL_MAX_SAMPLER_SNAPSHOT_ENTRIES];
} MGLSamplerSnapshotSet;

typedef struct {
    MGLDrawCommandType type;
    GLenum   mode;
    GLint    first;
    GLsizei  count;
    GLsizei  instanceCount;
    GLint    baseVertex;
    GLuint   baseInstance;
    GLenum   indexType;
    GLuint   indexBufferOffset;
    void    *elementBuffer;
    uint8_t  dynamic_vertex_binding_count;
    MGLDynamicVertexBinding
             dynamic_vertex_bindings[MGL_MAX_DYNAMIC_VERTEX_BINDINGS];
    uint8_t  dynamic_uniform_binding_count;
    MGLDynamicUniformBinding
             dynamic_uniform_bindings[MGL_MAX_DYNAMIC_UNIFORM_BINDINGS];
    uint8_t  dynamic_texture_binding_count;
    MGLDynamicTextureBinding
             dynamic_texture_bindings[MGL_MAX_DYNAMIC_TEXTURE_BINDINGS];
    uint16_t sampler_snapshot_id;
} MGLDrawCommand;

typedef enum {
    MGL_BATCH_PATH_DIRECT = 0,
    MGL_BATCH_PATH_MDI,
    MGL_BATCH_PATH_STREAM_MERGE,
    MGL_BATCH_PATH_ICB,
    MGL_BATCH_PATH_COUNT,
} MGLBatchPath;

typedef struct {
    uint32_t program_name;
    uint32_t program_pipeline_name;
    uint32_t vertex_program_name;
    uint32_t fragment_program_name;
    uint32_t vao_name;
    uint32_t fbo_name;
    int16_t  viewport[4];
    int16_t  scissor[4];
    uint8_t  scissor_enabled;
    uint8_t  primitive_type;
    uint16_t caps_flags;
    uint64_t texture_hash;
    uint64_t render_state_hash;
    uint64_t uniform_buffer_hash;
    uint64_t vertex_layout_hash;
} MGLStateKey;

typedef struct {
    MGLStateKey     key;
    uint32_t        command_count;
    uint32_t        command_capacity;
    MGLDrawCommand *commands;
    void           *state_snapshot;
    void           *vao_snapshot;
    void           *source_vao;
    void           *retained_program;
    void           *retained_vertex_program;
    void           *retained_fragment_program;
    void           *stream_vertex_buffer;
    void           *stream_index_buffer;
    size_t          stream_vertex_bytes;
    size_t          stream_index_bytes;
    size_t          stream_index_count;
    size_t          stream_vertex_stride;
    uint64_t        stream_layout_hash;
    uint16_t        sampler_snapshot_id;
    bool            mdi_compatible;
    bool            uses_elements;
    bool            stream_merged;
    bool            has_dynamic_uniform_bindings;
    bool            has_dynamic_vertex_bindings;
    bool            has_dynamic_texture_bindings;
    bool            has_sampler_snapshots;
    bool            sampler_snapshots_mixed;
    bool            arena_managed;  /* snapshot/commands allocated from arena */
} MGLDrawBatch;

typedef struct {
    void     *buffer;
    uint64_t  start;
    uint64_t  end;
} MGLBufferReadRange;

typedef struct {
    MGLDrawBatch batches[MGL_MAX_BATCHES];
    uint32_t     batch_count;
    uint32_t     total_commands;
    void        *mdi_scratch_buffer;
    size_t       mdi_scratch_capacity;
    uint32_t     array_cmd_count;
    uint32_t     element_cmd_count;
    bool         has_deferred_uniform_range_rebind;
    bool         sampler_snapshot_incomplete;
    MGLSamplerSnapshotKey sampler_snapshot_keys[MGL_MAX_SAMPLER_SNAPSHOT_KEYS];
    uint16_t     sampler_snapshot_key_count;
    uint16_t     sampler_snapshot_key_index[MGL_SAMPLER_SNAPSHOT_KEY_INDEX_SIZE];
    MGLSamplerSnapshotSet sampler_snapshot_sets[MGL_MAX_SAMPLER_SNAPSHOT_SETS];
    uint16_t     sampler_snapshot_set_count;
    uint16_t     sampler_snapshot_set_index[MGL_SAMPLER_SNAPSHOT_SET_INDEX_SIZE];
    MGLBufferReadRange buffer_read_ranges[MGL_MAX_PENDING_BUFFER_RANGES];
    uint32_t     buffer_read_range_count;
    bool         buffer_read_range_overflow;
    /* Hash index over buffer_read_ranges bucketed by buffer pointer so
     * insert/query walk only that buffer's ranges instead of all of them.
     * bucket holds (range_index + 1) of the newest entry, 0 = empty; entries
     * chain via buffer_read_range_next (same +1 encoding).  Zeroed by the
     * whole-struct memset in reset. */
    uint32_t     buffer_read_range_bucket[MGL_BUFFER_RANGE_BUCKET_SIZE];
    uint32_t     buffer_read_range_next[MGL_MAX_PENDING_BUFFER_RANGES];
    void        *texture_write_objects[MGL_MAX_PENDING_TEXTURE_WRITES];
    uint32_t     texture_write_count;
    bool         texture_write_overflow;
    /* hash-set index for O(1) dedup/membership on texture_write_objects.
     * Slot value is (array_index + 1); 0 = empty. Zeroed by memset in reset. */
    uint32_t     texture_write_index[MGL_TEX_WRITE_INDEX_SIZE];
    void        *texture_read_objects[MGL_MAX_PENDING_TEXTURE_READS];
    uint32_t     texture_read_count;
    bool         texture_read_overflow;
    /* hash-set index for O(1) dedup/membership on texture_read_objects. */
    uint32_t     texture_read_index[MGL_TEX_READ_INDEX_SIZE];
} MGLCommandBuffer;

/* GL API -> DrawCommand Recorder -> DrawCommandBuffer. */
void mglInitCommandBuffer(MGLCommandBuffer *cb);
void mglResetCommandBuffer(MGLCommandBuffer *cb);
void mglResetCommandBufferForContext(GLMContext ctx, MGLCommandBuffer *cb);
void mglComputeStateKey(GLMContext ctx, GLenum mode, bool uses_elements, MGLStateKey *out);
bool mglStateKeysEqual(const MGLStateKey *a, const MGLStateKey *b);
void mglRecordDrawCommand(GLMContext ctx, const MGLDrawCommand *cmd);
void mglAppendDrawCommand(GLMContext ctx, const MGLDrawCommand *cmd);
void mglFlushCommandBuffer(GLMContext ctx);
void mglFlushPendingDraws(GLMContext ctx);
bool mglPendingDrawsReadBufferRange(GLMContext ctx, void *buffer, int64_t offset, int64_t size);
bool mglPendingDrawsWriteTexture(GLMContext ctx, void *texture);
bool mglPendingDrawsReadTexture(GLMContext ctx, void *texture);
void mglFlushPendingDrawsForBuffer(GLMContext ctx, void *buffer);
void mglFlushPendingDrawsForBufferRange(GLMContext ctx, void *buffer, int64_t offset, int64_t size);
void mglFlushPendingDrawsForVertexArray(GLMContext ctx, void *vao);
void mglFlushPendingDrawsForTexture(GLMContext ctx, void *texture);
void mglFlushPendingDrawsBeforeTextureWrite(GLMContext ctx, void *texture);
void mglFlushPendingDrawsForActiveTextures(GLMContext ctx);

/* MGL_BIND_NO_FLUSH (default ON; =0 off): pure texture/buffer rebinds skip
 * unconditional full flush; content mutation paths still flush. */
int mglBindNoFlushEnabled(void);

/* Per-draw sampler snapshots are default-on and can be disabled with
 * MGL_DRAW_SAMPLER_SNAPSHOT=0. The parameter query returns true only when
 * every pending draw has a complete immutable sampler snapshot. */
int mglSamplerSnapshotEnabled(void);
bool mglSamplerSnapshotCanDeferParameter(GLMContext ctx, GLenum pname);

/* Initializes an address-stable, chunked batch arena. */
bool mglInitBatchArena(MGLBatchArena *arena, size_t initial_capacity);

/* Reset the batch arena (sets chunk offsets to 0, keeping all chunks).
 * Safe to call only when no worker/encoder is accessing snapshot data
 * (i.e. after teardownBatchReplayForContext).  No-op if arena is NULL. */
void mglResetBatchArena(MGLBatchArena *arena);

/* Releases every backing chunk and clears the arena. */
void mglDestroyBatchArena(MGLBatchArena *arena);

/* Draw command classification helpers (pure functions, no ctx dependency). */

/* Human-readable name for a draw command type, or "unknown" for
 * unrecognized values.  Used by diagnostic/trace logging. */
const char *mglDrawCommandTypeName(MGLDrawCommandType type);

/* Returns true if `cmd->type` is an indexed (glDrawElements*) variant,
 * false for array (glDrawArrays*) variants or NULL cmd. */
bool mglDrawCommandUsesElements(const MGLDrawCommand *cmd);

#ifdef __cplusplus
}
#endif

#endif /* draw_command_h */
