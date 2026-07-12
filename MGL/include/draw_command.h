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
#define MGL_MAX_PENDING_TEXTURE_WRITES 256
#define MGL_MAX_PENDING_TEXTURE_READS 512

/* Bump-allocator arena for batch snapshot allocations (Task 4).
 * Gated by env var MGL_ARENA_SNAPSHOT=1 (default OFF).  When enabled,
 * state_snapshot, vao_snapshot, and the commands array are allocated from
 * this arena instead of individual malloc calls, and freed via arena reset
 * instead of individual free calls.  The arena is owned by MGLRenderer and
 * accessed from C via GLMContextRec_t::batch_arena. */
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
    bool            mdi_compatible;
    bool            uses_elements;
    bool            stream_merged;
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
    MGLBufferReadRange buffer_read_ranges[MGL_MAX_PENDING_BUFFER_RANGES];
    uint32_t     buffer_read_range_count;
    bool         buffer_read_range_overflow;
    void        *texture_write_objects[MGL_MAX_PENDING_TEXTURE_WRITES];
    uint32_t     texture_write_count;
    bool         texture_write_overflow;
    void        *texture_read_objects[MGL_MAX_PENDING_TEXTURE_READS];
    uint32_t     texture_read_count;
    bool         texture_read_overflow;
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

/* Parallel-group planning (Stage 5.1).
 *
 * A parallel group is a maximal run of consecutive batches in the command
 * buffer that share the same FBO (key.fbo_name) and have no inter-batch
 * resource hazard. The Hazard Tracker already splits read-after-write hazards
 * into separate flushes, so within one command buffer two batches sharing an
 * FBO are candidate members of one group. This pure function fills `groups`
 * (start index + length for each group) without touching ctx state; the
 * renderer decides whether to actually parallelize.
 *
 * Pure data in/out — no Metal, no ctx side effects (core principle 3). */

#define MGL_MAX_PARALLEL_GROUPS  (MGL_MAX_BATCHES)

typedef struct {
    uint32_t start_batch;   /* index into cb->batches where the group starts */
    uint32_t batch_count;   /* number of consecutive batches in this group  */
} MGLParallelGroup;

/* Compute parallel groups for the batches currently in `cb`. Returns the
 * number of groups written to `out_groups` (<= max_groups). Batches with
 * command_count == 0 are skipped (they are not replayed). `out_groups` is
 * capped at max_groups entries; the caller passes at least
 * MGL_MAX_PARALLEL_GROUPS. */
uint32_t mglComputeParallelGroups(const MGLCommandBuffer *cb,
                                  MGLParallelGroup *out_groups,
                                  uint32_t max_groups);

#ifdef __cplusplus
}
#endif

#endif /* draw_command_h */
