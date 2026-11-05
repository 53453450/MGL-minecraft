/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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
 * draw_command.c
 * MGL
 *
 */

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <stdatomic.h>

#include "glm_context.h"
#include "draw_command.h"
#include "mgl_byte_hash.h"
#include "mgl_safety.h"
#include "mgl_frame_activity.h"
#include "mgl_trace_log.h"
#include "mgl_sampler_compat.h"
#include "mgl_batch_hazard.h"
#include "mgl_env_flag.h"
#include "mgl_render_api_buffer.h"
#include "mgl_buffer_backend.h"
#include "mgl_draw_command_shared.h"

extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);
extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern void mglSparseScrubBoundImages(GLMContext ctx);


/* Transient pool implemented in mgl_draw_stream_merge.c */

/* T12 peak occupancy watermarks (process lifetime). */
uint32_t s_peak_buffer_read_ranges;
uint32_t s_peak_texture_writes;
uint32_t s_peak_texture_reads;
uint32_t s_peak_sampler_snapshot_keys;
uint32_t s_peak_sampler_snapshot_sets;
uint32_t s_peak_pending_vao_refs;
uint32_t s_peak_pending_vao_names;


void mglGetCommandBufferOccupancyPeaks(MGLCommandBufferOccupancyPeaks *out)
{
    if (!out) return;
    out->buffer_read_ranges = s_peak_buffer_read_ranges;
    out->texture_writes = s_peak_texture_writes;
    out->texture_reads = s_peak_texture_reads;
    out->sampler_snapshot_keys = s_peak_sampler_snapshot_keys;
    out->sampler_snapshot_sets = s_peak_sampler_snapshot_sets;
    out->pending_vao_refs = s_peak_pending_vao_refs;
    out->pending_vao_names = s_peak_pending_vao_names;
}

void mglResetCommandBufferOccupancyPeaks(void)
{
    s_peak_buffer_read_ranges = 0;
    s_peak_texture_writes = 0;
    s_peak_texture_reads = 0;
    s_peak_sampler_snapshot_keys = 0;
    s_peak_sampler_snapshot_sets = 0;
    s_peak_pending_vao_refs = 0;
    s_peak_pending_vao_names = 0;
}

/* === Task 4: Snapshot Arena (bump allocator) === */

#define MGL_ARENA_INITIAL_CAPACITY  (4u * 1024u * 1024u)  /* 4 MB */

struct MGLBatchArenaChunk {
    struct MGLBatchArenaChunk *next;
    size_t                     offset;
    size_t                     capacity;
    unsigned char              data[];
};

static MGLBatchArenaChunk *mglNewBatchArenaChunk(size_t capacity)
{
    if (capacity == 0 || capacity > SIZE_MAX - sizeof(MGLBatchArenaChunk)) {
        return NULL;
    }

    MGLBatchArenaChunk *chunk = malloc(sizeof(*chunk) + capacity);
    if (!chunk) {
        return NULL;
    }

    chunk->next = NULL;
    chunk->offset = 0;
    chunk->capacity = capacity;
    return chunk;
}

bool mglInitBatchArena(MGLBatchArena *arena, size_t initial_capacity)
{
    if (!arena) {
        return false;
    }

    memset(arena, 0, sizeof(*arena));
    size_t capacity = initial_capacity ? initial_capacity : MGL_ARENA_INITIAL_CAPACITY;
    MGLBatchArenaChunk *chunk = mglNewBatchArenaChunk(capacity);
    if (!chunk) {
        return false;
    }

    arena->head = chunk;
    arena->current = chunk;
    arena->initial_capacity = capacity;
    arena->enabled = 1;
    return true;
}

void *arenaAlloc(MGLBatchArena *arena, size_t size)
{
    if (!arena || !arena->enabled || size == 0) {
        return NULL;
    }

    /* Align to 16 bytes */
    if (size > SIZE_MAX - 15u) {
        return NULL;
    }
    size_t aligned = (size + 15u) & ~(size_t)15u;

    MGLBatchArenaChunk *chunk = arena->current;
    if (!chunk || aligned > chunk->capacity - chunk->offset) {
        MGLBatchArenaChunk *next = chunk ? chunk->next : arena->head;
        while (next && aligned > next->capacity - next->offset) {
            next = next->next;
        }
        if (next) {
            chunk = next;
        } else {
            size_t capacity = chunk ? chunk->capacity : arena->initial_capacity;
            if (capacity == 0) {
                capacity = MGL_ARENA_INITIAL_CAPACITY;
            }
            while (capacity < aligned) {
                if (capacity > SIZE_MAX / 2u) {
                    capacity = aligned;
                    break;
                }
                capacity *= 2u;
            }
            next = mglNewBatchArenaChunk(capacity);
            if (!next) {
                return NULL;
            }
            if (chunk) {
                MGLBatchArenaChunk *tail = chunk;
                while (tail->next) {
                    tail = tail->next;
                }
                tail->next = next;
            } else if (!arena->head) {
                arena->head = next;
            }
            chunk = next;
        }
        arena->current = chunk;
    }

    void *ptr = chunk->data + chunk->offset;
    chunk->offset += aligned;
    return ptr;
}

void *mglBatchArenaAlloc(MGLBatchArena *arena, size_t size)
{
    return arenaAlloc(arena, size);
}

void mglResetBatchArena(MGLBatchArena *arena)
{
    if (!arena) return;
    for (MGLBatchArenaChunk *chunk = arena->head; chunk; chunk = chunk->next) {
        chunk->offset = 0;
    }
    arena->current = arena->head;
}

void mglDestroyBatchArena(MGLBatchArena *arena)
{
    if (!arena) return;

    MGLBatchArenaChunk *chunk = arena->head;
    while (chunk) {
        MGLBatchArenaChunk *next = chunk->next;
        free(chunk);
        chunk = next;
    }
    memset(arena, 0, sizeof(*arena));
}

void mglInitCommandBuffer(MGLCommandBuffer *cb)
{
    if (!cb) return;
    memset(cb, 0, sizeof(*cb));
}

void mglDestroyTransientBuffer(GLMContext ctx, Buffer *buffer)
{
    if (!buffer) return;

    if (ctx)
        mglRendererReleaseBufferMetalData(ctx, buffer);
    if (buffer->data.buffer_data) {
        free((void *)(uintptr_t)buffer->data.buffer_data);
        buffer->data.buffer_data = 0;
        buffer->data.buffer_size = 0;
    }

    free(buffer);
}

/* Buffer_base types that batch snapshots must retain/release.
 * Only the 5 "hot" types are in the snapshot (mglCopyHotStateFields Region 4);
 * the 11 cold types stay live in active_state through replay (never written).
 * Retaining cold types would dereference stale arena pointers → SIGSEGV.
 * MUST stay in sync with mglCopyHotStateFields Region 4. */
static const int kMGLSnapshotBufferBaseTypes[] = {
#define MGL_SNAPSHOT_LIST_HOT(_t_) _t_,
    MGL_SNAPSHOT_HOT_BUFFER_BASE_TYPES(MGL_SNAPSHOT_LIST_HOT)
#undef MGL_SNAPSHOT_LIST_HOT
};
static const size_t kMGLSnapshotBufferBaseTypeCount =
    sizeof(kMGLSnapshotBufferBaseTypes) / sizeof(kMGLSnapshotBufferBaseTypes[0]);

static void mglRetainVAOBufferReferences(const VertexArray *vao)
{
    if (!vao)
        return;
    for (size_t i = 0; i < MGL_MAX_VERTEX_ATTRIB_BINDINGS; i++) {
        if (vao->bindings[i].buffer)
            mglRetainBufferReference(vao->bindings[i].buffer);
    }
    for (size_t i = 0; i < MAX_ATTRIBS; i++) {
        if (vao->attrib[i].buffer)
            mglRetainBufferReference(vao->attrib[i].buffer);
    }
    if (vao->element_array.buffer)
        mglRetainBufferReference(vao->element_array.buffer);
}

static void mglReleaseVAOBufferReferences(GLMContext ctx, const VertexArray *vao)
{
    if (!vao)
        return;
    for (size_t i = 0; i < MGL_MAX_VERTEX_ATTRIB_BINDINGS; i++) {
        if (vao->bindings[i].buffer)
            mglReleaseBufferReference(ctx, vao->bindings[i].buffer);
    }
    for (size_t i = 0; i < MAX_ATTRIBS; i++) {
        if (vao->attrib[i].buffer)
            mglReleaseBufferReference(ctx, vao->attrib[i].buffer);
    }
    if (vao->element_array.buffer)
        mglReleaseBufferReference(ctx, vao->element_array.buffer);
}

void mglRetainBatchBufferReferences(MGLDrawBatch *batch)
{
    GLMState *snap = (GLMState *)batch->state_snapshot;
    if (!snap) return;
    for (size_t t = 0; t < kMGLSnapshotBufferBaseTypeCount; t++) {
        BufferBaseTarget *slots = snap->buffer_base[kMGLSnapshotBufferBaseTypes[t]].buffers;
        for (size_t i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            if (slots[i].buf) mglRetainBufferReference(slots[i].buf);
        }
    }
    /* VAO snapshot is a shallow copy — retain every Buffer it still points at
     * so DeleteBuffers cannot free shells before deferred replay (A14). */
    mglRetainVAOBufferReferences((const VertexArray *)batch->vao_snapshot);
}

/* Must be called BEFORE state_snapshot is freed, since it reads buf pointers
 * from the snapshot. */
static void mglReleaseBatchBufferReferences(GLMContext ctx, MGLDrawBatch *batch)
{
    GLMState *snap = (GLMState *)batch->state_snapshot;
    if (!snap) return;
    for (size_t t = 0; t < kMGLSnapshotBufferBaseTypeCount; t++) {
        BufferBaseTarget *slots = snap->buffer_base[kMGLSnapshotBufferBaseTypes[t]].buffers;
        for (size_t i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            if (slots[i].buf) mglReleaseBufferReference(ctx, slots[i].buf);
        }
    }
    mglReleaseVAOBufferReferences(ctx, (const VertexArray *)batch->vao_snapshot);
}

/* Phase A of batch release: drop the batch's object references.  Reads buf
 * pointers from the (possibly shared) snapshot, so in the command-buffer
 * reset loop every batch must run this phase BEFORE any batch frees a
 * snapshot (see mglResetCommandBufferForContext). */
static void mglReleaseBatchReferences(GLMContext ctx, MGLDrawBatch *batch)
{
    if (!batch) return;
    mglReleaseBatchBufferReferences(ctx, batch);
    if (batch->retained_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_program);
        batch->retained_program = NULL;
    }
    if (batch->retained_vertex_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_vertex_program);
        batch->retained_vertex_program = NULL;
    }
    if (batch->retained_fragment_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_fragment_program);
        batch->retained_fragment_program = NULL;
    }
    if (batch->retained_geometry_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_geometry_program);
        batch->retained_geometry_program = NULL;
    }
    if (batch->retained_tess_control_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_tess_control_program);
        batch->retained_tess_control_program = NULL;
    }
    if (batch->retained_tess_eval_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_tess_eval_program);
        batch->retained_tess_eval_program = NULL;
    }
    if (batch->retained_compute_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_compute_program);
        batch->retained_compute_program = NULL;
    }
    if (batch->stream_vertex_buffer) {
        mglRecycleTransientBatchBuffer(ctx, (Buffer *)batch->stream_vertex_buffer);
        batch->stream_vertex_buffer = NULL;
    }
    if (batch->stream_index_buffer) {
        mglRecycleTransientBatchBuffer(ctx, (Buffer *)batch->stream_index_buffer);
        batch->stream_index_buffer = NULL;
    }
}

/* Phase B of batch release: free the batch's own memory.  A batch sharing an
 * equal-key donor's snapshot (M2 dedup) must not free it - the donor owns
 * it; the reset loop only reaches here after every batch's reference phase
 * is done, so no snapshot is read after a free. */
static void mglReleaseBatchMemory(MGLDrawBatch *batch)
{
    if (!batch) return;

    /* Arena-managed allocations (commands, state_snapshot, vao_snapshot) are
     * freed collectively via arena reset (mglResetBatchArena), not
     * individually.  Only free them on the non-arena path.
     *
     * snapshot_shared only forgives the two SNAPSHOT pointers (the donor owns
     * them).  `commands` is always this batch's own realloc'd array
     * (draw_command.c ~4351), so a sharer must still free it - otherwise every
     * shared batch leaks its command array on the non-arena path. */
    if (batch->snapshot_shared) {
        batch->state_snapshot = NULL;
        batch->vao_snapshot = NULL;
        if (!batch->arena_managed && batch->commands) {
            free(batch->commands);
        }
        batch->commands = NULL;
    } else if (!batch->arena_managed) {
        /* Reached only when snapshot_shared is false AND not arena-managed,
         * so these frees are the owner's; state_machine_invariants.py I7
         * checks the enclosing condition for exactly those two terms. */
        if (batch->commands) {
            free(batch->commands);
            batch->commands = NULL;
        }
        if (batch->state_snapshot) {
            free(batch->state_snapshot);
            batch->state_snapshot = NULL;
        }
        if (batch->vao_snapshot) {
            free(batch->vao_snapshot);
            batch->vao_snapshot = NULL;
        }
    } else {
        batch->commands = NULL;
        batch->state_snapshot = NULL;
        batch->vao_snapshot = NULL;
    }
    batch->source_vao = NULL;
}

void mglReleaseBatch(GLMContext ctx, MGLDrawBatch *batch)
{
    mglReleaseBatchReferences(ctx, batch);
    mglReleaseBatchMemory(batch);
}

static Program *mglRetainBatchProgram(GLMContext ctx, MGLDrawBatch *batch, Program *program, GLuint expectedName, void **slot)
{
    if (!ctx || !batch || !program || !slot) {
        return NULL;
    }

    /* Callers pass the known name for every program they hand over (state
     * binding name or a retained pipeline stage's own name); name==0 with a
     * non-NULL program is an inconsistent state — fail closed (D6). */
    if (expectedName == 0u ||
        !mglProgramPointerUsableForName(ctx, program, expectedName)) {
        return NULL;
    }

    /* UsableForName just proved the pointer live and matching; retain
     * directly rather than re-validating via mglRetainProgramReference. */
    program->refcount++;
    *slot = program;
    return program;
}

/* Retain every pipeline stage program replay may dereference.  The set of
 * retained slots is generated from ProgramPipeline.stage_programs so adding
 * a replay consumer cannot silently skip retain. */
void mglRetainBatchProgramReferences(GLMContext ctx, MGLDrawBatch *batch)
{
    if (!ctx || !batch) {
        return;
    }

    (void)mglRetainBatchProgram(ctx,
                                batch,
                                ctx->active_state->program,
                                ctx->active_state->program_name,
                                &batch->retained_program);

    if (ctx->active_state->program_name != 0u || !ctx->active_state->program_pipeline) {
        return;
    }

    /* The state holds a reference on the pipeline (UseProgramStages retained
     * its stage programs), so the pointer is dereferenceable. */
    ProgramPipeline *pipeline = ctx->active_state->program_pipeline;
    if (!mglObjectPointerLooksPlausible(pipeline)) {
        return;
    }

    (void)mglRetainBatchProgram(ctx,
                                batch,
                                pipeline->stage_programs[_VERTEX_SHADER],
                                pipeline->stage_programs[_VERTEX_SHADER]
                                    ? pipeline->stage_programs[_VERTEX_SHADER]->name
                                    : 0u,
                                &batch->retained_vertex_program);
    (void)mglRetainBatchProgram(ctx,
                                batch,
                                pipeline->stage_programs[_FRAGMENT_SHADER],
                                pipeline->stage_programs[_FRAGMENT_SHADER]
                                    ? pipeline->stage_programs[_FRAGMENT_SHADER]->name
                                    : 0u,
                                &batch->retained_fragment_program);
    (void)mglRetainBatchProgram(ctx,
                                batch,
                                pipeline->stage_programs[_GEOMETRY_SHADER],
                                pipeline->stage_programs[_GEOMETRY_SHADER]
                                    ? pipeline->stage_programs[_GEOMETRY_SHADER]->name
                                    : 0u,
                                &batch->retained_geometry_program);
    (void)mglRetainBatchProgram(ctx,
                                batch,
                                pipeline->stage_programs[_TESS_CONTROL_SHADER],
                                pipeline->stage_programs[_TESS_CONTROL_SHADER]
                                    ? pipeline->stage_programs[_TESS_CONTROL_SHADER]->name
                                    : 0u,
                                &batch->retained_tess_control_program);
    (void)mglRetainBatchProgram(ctx,
                                batch,
                                pipeline->stage_programs[_TESS_EVALUATION_SHADER],
                                pipeline->stage_programs[_TESS_EVALUATION_SHADER]
                                    ? pipeline->stage_programs[_TESS_EVALUATION_SHADER]->name
                                    : 0u,
                                &batch->retained_tess_eval_program);
    (void)mglRetainBatchProgram(ctx,
                                batch,
                                pipeline->stage_programs[_COMPUTE_SHADER],
                                pipeline->stage_programs[_COMPUTE_SHADER]
                                    ? pipeline->stage_programs[_COMPUTE_SHADER]->name
                                    : 0u,
                                &batch->retained_compute_program);
}

bool mglInitializeBatchStateSnapshot(GLMContext ctx, MGLDrawBatch *batch)
{
    MGL_SIGNPOST_BEGIN(InitBatchSnapshot);
    if (!ctx || !batch) {
        MGL_SIGNPOST_END(InitBatchSnapshot);
        return false;
    }

    MGLBatchArena *arena = ctx->batch_arena;
    if (arena && arena->enabled) {
        batch->state_snapshot = arenaAlloc(arena, sizeof(GLMState));
        batch->vao_snapshot = arenaAlloc(arena, sizeof(VertexArray));
        batch->arena_managed = true;
    } else {
        batch->state_snapshot = malloc(sizeof(GLMState));
        batch->vao_snapshot = malloc(sizeof(VertexArray));
        batch->arena_managed = false;
    }
    if (!batch->state_snapshot || !batch->vao_snapshot) {
        MGL_SIGNPOST_END(InitBatchSnapshot);
        return false;
    }
    /* Selective snapshot: only copy hot fields (~61KB vs 92KB full).
     * Cold fields (HashTables + unused buffer_base types) are skipped;
     * they are restored from savedState at replay time. */
    mglCopyHotStateFields((GLMState *)batch->state_snapshot, ctx->active_state);
    batch->source_vao = ctx->active_state->vao;
    if (ctx->active_state->vao) {
        memcpy(batch->vao_snapshot, ctx->active_state->vao, sizeof(VertexArray));
        ((VertexArray *)batch->vao_snapshot)->transient_batch_vao = GL_TRUE;
        ((GLMState *)batch->state_snapshot)->vao = (VertexArray *)batch->vao_snapshot;
    } else {
        if (!batch->arena_managed) {
            free(batch->vao_snapshot);
        }
        batch->vao_snapshot = NULL;
    }

    MGL_PERF_INC(g_mglSnapshotAllocationCountSinceSwap);
    /* Report the bytes actually written (hot state copy + VAO freeze), not
     * the allocation size: this counter is the snapshot-cost metric the
     * state-machine migration tracks (STATE_MACHINE_REVIEW 4.3).  The
     * allocation itself remains sizeof(GLMState)+sizeof(VertexArray) per
     * batch regardless. */
    MGL_PERF_ADD(g_mglSnapshotBytesAllocatedSinceSwap,
                 mglSnapshotHotStateBytes() +
                 (batch->vao_snapshot ? sizeof(VertexArray) : 0));

    mglRetainBatchProgramReferences(ctx, batch);
    mglRetainBatchBufferReferences(batch);
    MGL_SIGNPOST_END(InitBatchSnapshot);
    return true;
}

/* T13-4: register VAO pointers/names referenced by a non-stream pending batch
 * so mglPendingDrawsReferenceVertexArray is O(1) instead of scanning all
 * batches and dereferencing snapshots. */
static void mglRegisterPendingVaoRef(MGLCommandBuffer *cb, void *vao)
{
    if (!cb || !vao || cb->pending_vao_ref_overflow) return;

    uint32_t slot = (uint32_t)(((uintptr_t)vao >> 4) & MGL_VAO_REF_INDEX_MASK);
    for (;;) {
        uint32_t entry = cb->pending_vao_ref_index[slot];
        if (entry == 0) break;
        if (cb->pending_vao_refs[entry - 1] == vao) return;
        slot = (slot + 1) & MGL_VAO_REF_INDEX_MASK;
    }

    if (cb->pending_vao_ref_count >= MGL_MAX_PENDING_VAO_REFS) {
        cb->pending_vao_ref_overflow = true;
        return;
    }

    uint32_t idx = cb->pending_vao_ref_count++;
    cb->pending_vao_refs[idx] = vao;
    cb->pending_vao_ref_index[slot] = idx + 1;
    mglNoteOccupancyPeak(&s_peak_pending_vao_refs, cb->pending_vao_ref_count);
}

static void mglRegisterPendingVaoName(MGLCommandBuffer *cb, uint32_t name)
{
    if (!cb || name == 0 || cb->pending_vao_name_overflow) return;

    uint32_t slot = (name * 2654435761u) & MGL_VAO_REF_INDEX_MASK;
    for (;;) {
        uint32_t entry = cb->pending_vao_name_index[slot];
        if (entry == 0) break;
        if (cb->pending_vao_names[entry - 1] == name) return;
        slot = (slot + 1) & MGL_VAO_REF_INDEX_MASK;
    }

    if (cb->pending_vao_name_count >= MGL_MAX_PENDING_VAO_REFS) {
        cb->pending_vao_name_overflow = true;
        return;
    }

    uint32_t idx = cb->pending_vao_name_count++;
    cb->pending_vao_names[idx] = name;
    cb->pending_vao_name_index[slot] = idx + 1;
    mglNoteOccupancyPeak(&s_peak_pending_vao_names, cb->pending_vao_name_count);
}

static void mglRegisterBatchVaoHazard(MGLCommandBuffer *cb, MGLDrawBatch *batch)
{
    if (!cb || !batch || batch->stream_merged) return;

    mglRegisterPendingVaoRef(cb, batch->source_vao);
    if (batch->state_snapshot) {
        GLMState *snapshot = (GLMState *)batch->state_snapshot;
        mglRegisterPendingVaoRef(cb, snapshot->vao);
    }
    mglRegisterPendingVaoName(cb, batch->key.vao_name);
}

/* M2 (STATE_MACHINE_REVIEW 3.1/M2): dedup snapshots by key.  Equal keys
 * already share one snapshot through last-batch reuse; this extends the same
 * invariant (equal key => equal captured state; hash collisions ~2^-51, the
 * same basis the batch merger relies on) to any earlier batch in the current
 * command buffer, so a draw sequence that revisits a state pays one capture
 * instead of one per batch.  Stream-merged snapshots are specialized
 * (patched buffer/element pointers) and never donate.  A sharer owns no
 * snapshot memory: it must not free the shared pointers (snapshot_shared),
 * and it re-retains program/buffer references for itself from the live state
 * and the shared snapshot.
 *
 * UBO ring offsets (MC BindBufferRange per draw) change the full key without
 * changing any frozen payload other than buffer_base ranges.  When first_cmd
 * is provided, a donor that matches ignoring uniform ranges can still share:
 * the command stores the live ranges as dynamic overrides (same path as the
 * adjacent-batch UBO merge). */
/* MC 1.21 frames land ~70–90 batches/CB; 64 left early equal-key donors
 * outside the window so M2 share stayed ~0 (PERF shared=0, ~4MiB snap/frame). */
enum { kMGLBatchSnapshotDonorWindow = 256u };

bool mglInitializeOrShareBatchStateSnapshot(GLMContext ctx,
                                                   MGLCommandBuffer *cb,
                                                   MGLDrawBatch *batch,
                                                   MGLDrawCommand *first_cmd)
{
    if (ctx && cb && batch) {
        /* Bounded donor window: the full cb scan was O(batches²) key memcmps
         * per flush.  MC frames typically sit under ~64 batches; a 16-wide
         * window missed equal-key revisits that land just outside it. */
        const uint32_t window =
            cb->batch_count > (uint32_t)kMGLBatchSnapshotDonorWindow
                ? cb->batch_count - (uint32_t)kMGLBatchSnapshotDonorWindow
                : 0u;
        MGLDrawBatch *ubo_donor = NULL;
        for (uint32_t i = window; i < cb->batch_count; i++) {
            MGLDrawBatch *donor = &cb->batches[i];
            if (donor == batch || donor->stream_merged || !donor->state_snapshot) {
                continue;
            }
            if (mglStateKeysEqual(&donor->key, &batch->key)) {
                batch->state_snapshot = donor->state_snapshot;
                batch->vao_snapshot = donor->vao_snapshot;
                /* Re-read the live VAO pointer instead of copying the donor's:
                 * a name-identical VAO could have been recreated between the
                 * two captures.  The frozen VAO content itself follows the
                 * donor's capture - the same semantics batch reuse already
                 * gives equal keys. */
                batch->source_vao = ctx->active_state->vao;
                batch->snapshot_shared = true;
                /* Match what a fresh capture would choose for commands
                 * allocation; the shared snapshot is never freed by this batch
                 * either way (snapshot_shared). */
                batch->arena_managed =
                    (ctx->batch_arena && ctx->batch_arena->enabled) ? true
                                                                   : false;
                MGL_PERF_INC(g_mglSnapshotShareHitsSinceSwap);
                mglRetainBatchProgramReferences(ctx, batch);
                mglRetainBatchBufferReferences(batch);
                mglRegisterBatchVaoHazard(cb, batch);
                return true;
            }
            if (first_cmd &&
                mglStateKeysEqualIgnoringUniformRanges(&donor->key,
                                                       &batch->key)) {
                /* Keep the newest match: its frozen UBO ranges are usually
                 * closer to the live ring position. */
                ubo_donor = donor;
            }
        }
        if (ubo_donor) {
            batch->state_snapshot = ubo_donor->state_snapshot;
            batch->vao_snapshot = ubo_donor->vao_snapshot;
            batch->source_vao = ctx->active_state->vao;
            batch->snapshot_shared = true;
            batch->has_dynamic_uniform_bindings = true;
            batch->mdi_compatible = false;
            batch->arena_managed =
                (ctx->batch_arena && ctx->batch_arena->enabled) ? true : false;
            if (mglCaptureDynamicUniformRanges(ctx, batch, first_cmd, true)) {
                MGL_PERF_INC(g_mglSnapshotShareHitsSinceSwap);
                mglRetainBatchProgramReferences(ctx, batch);
                mglRetainBatchBufferReferences(batch);
                mglRegisterBatchVaoHazard(cb, batch);
                return true;
            }
            /* Live UBO object set diverged from the donor snapshot — fall
             * through to a fresh capture. */
            batch->state_snapshot = NULL;
            batch->vao_snapshot = NULL;
            batch->snapshot_shared = false;
            batch->has_dynamic_uniform_bindings = false;
        }
    }
    if (!mglInitializeBatchStateSnapshot(ctx, batch)) {
        return false;
    }
    mglRegisterBatchVaoHazard(cb, batch);
    return true;
}

void mglResetCommandBufferForContext(GLMContext ctx, MGLCommandBuffer *cb)
{
    if (!cb) return;

    /* Two phases so equal-key shared snapshots (M2 dedup) cannot be freed
     * while another batch still reads them: every batch first drops its
     * object references (which read snapshot memory), then memory is freed.
     * Batches are released in index order, so a donor would otherwise free
     * its snapshot before a sharer's reference phase ran. */
    for (uint32_t i = 0; i < cb->batch_count; i++) {
        mglReleaseBatchReferences(ctx, &cb->batches[i]);
    }
    for (uint32_t i = 0; i < cb->batch_count; i++) {
        mglReleaseBatchMemory(&cb->batches[i]);
    }

    /* T12-1: free heap-backed hazard / sampler tables before memset. */
    free(cb->buffer_read_ranges);
    free(cb->buffer_read_range_next);
    free(cb->sampler_snapshot_keys);
    free(cb->sampler_snapshot_sets);
    free(cb->texture_write_objects);
    free(cb->texture_read_objects);

    /* The transient scratch pool intentionally survives this reset — it is
     * context-owned (ctx->transient_pool) and drains only at context
     * teardown (mglDrainTransientPool from destroyGLMContext). */
    memset(cb, 0, sizeof(*cb));
}

void mglResetCommandBuffer(MGLCommandBuffer *cb)
{
    mglResetCommandBufferForContext(NULL, cb);
}


void mglRecordDrawCommand(GLMContext ctx, const MGLDrawCommand *cmd)
{
    MGL_SIGNPOST_BEGIN(RecordDrawCommand);
    if (!ctx || !cmd) {
        MGL_SIGNPOST_END(RecordDrawCommand);
        return;
    }

    /* DrawCommand Recorder: keep the GL entry points shallow and capture a
     * validated draw into the deferred command buffer.  The Batch Builder and
     * Batch Queue stages below decide whether the command joins a direct batch
     * or a stream-merge batch. */
    mglFlushPendingDrawsForActiveTextures(ctx);
    mglFlushPendingDrawsBeforeFramebufferTextureWrites(ctx);

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    uint32_t maxDrawsPerBatch = mglRuntimeMaxDrawsPerBatch();
    uint32_t maxBatchCount = mglRuntimeMaxBatchCount();
    bool cmd_uses_elements =
        (cmd->type != MGL_CMD_DRAW_ARRAYS &&
         cmd->type != MGL_CMD_DRAW_ARRAYS_INSTANCED &&
         cmd->type != MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE);

    /* A short snapshot ID is only valid for the current command buffer.
     * Avoid a later capacity flush invalidating an ID captured below. */
    if (cb->batch_count >= maxBatchCount) {
        MGL_PERF_INC(g_mglFlushReasonCapacitySinceSwap);
        mglFlushCommandBuffer(ctx);
    }

    MGLDrawCommand stored_cmd = *cmd;
    stored_cmd.sampler_snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;
    if (mglSamplerSnapshotEnabled() && !cb->sampler_snapshot_incomplete) {
        bool captured =
            mglCaptureSamplerSnapshot(ctx, &stored_cmd.sampler_snapshot_id);
        if (!captured && cb->total_commands > 0) {
            mglFlushCommandBuffer(ctx);
            captured =
                mglCaptureSamplerSnapshot(ctx, &stored_cmd.sampler_snapshot_id);
        }
        if (!captured) {
            cb->sampler_snapshot_incomplete = true;
            stored_cmd.sampler_snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;
        }
    }

    MGLStateKey key;
    mglComputeStateKey(ctx, cmd->mode, cmd_uses_elements, &key);

    MGLStreamMergeCandidate streamCandidate;
    bool can_stream_merge = false;
    if (mglStreamMergeCannotMergeCached(ctx)) {
        MGL_PERF_INC(g_mglStreamMergePrepareNegHitsSinceSwap);
    } else {
        MGL_PERF_INC(g_mglStreamMergePrepareCallsSinceSwap);
        can_stream_merge =
            mglPrepareStreamMergeCandidate(ctx, cmd, cmd_uses_elements, &streamCandidate);
        if (can_stream_merge) {
            MGL_PERF_INC(g_mglStreamMergePrepareOkSinceSwap);
        }
    }
    if (can_stream_merge && cb->batch_count > 0) {
        MGLDrawBatch *last = &cb->batches[cb->batch_count - 1];
        bool last_keys_match = mglStateKeysEqual(&last->key, &key);
        if (last_keys_match &&
            (last->sampler_snapshots_mixed ||
             last->sampler_snapshot_id != stored_cmd.sampler_snapshot_id)) {
            /* Once sampler state changes, keep subsequent draws in one mixed
             * direct batch instead of consuming a stream batch per snapshot. */
            can_stream_merge = false;
        } else if (!last_keys_match) {
            const VertexArray *last_real_vao =
                (const VertexArray *)(last->stream_merged
                                          ? last->source_vao
                                          : last->vao_snapshot);
            const CurrentVertexAttrib *last_attribs =
                last->state_snapshot
                    ? (const CurrentVertexAttrib *)
                          &((const GLMState *)last->state_snapshot)
                               ->current_vertex_attrib[0]
                    : NULL;
            bool tolerant_equal =
                mglStateKeysEqualIgnoringUniformRanges(&last->key, &key) ||
                mglStateKeysEqualIgnoringUniformAndElementRanges(
                    &last->key, &key, last_real_vao,
                    ctx->active_state->vao, last_attribs,
                    ctx->active_state->current_vertex_attrib,
                    cmd_uses_elements);
            if (tolerant_equal) {
                /* Keys differ only in uniform-range identity or per-section
                 * data movement (per-draw UBO offsets / EBOs / vertex slices
                 * — MC 1.21.11 dynamic transforms and terrain sections).  A
                 * stream batch cannot express per-sub-draw overrides, so as
                 * a stream candidate this draw would open a single-draw
                 * stream batch and also block the tolerant merge branches
                 * below (they require !can_stream_merge).  Demote to a
                 * direct draw so the dynamic-capture merge path can absorb
                 * it instead. */
                can_stream_merge = false;
                MGL_PERF_INC(g_mglStreamDemotedToDirectSinceSwap);
            }
        }
    }
    /*
     * Batch reuse is decoupled from stream-merge.  When keys match, non-
     * stream-merged draws can share a batch — each draw keeps its own
     * MGLDrawCommand (first/count/baseVertex) but reuses the shared
     * state_snapshot.  Hazard detection (mglFlushPendingDrawsForBuffer/
     * VertexArray/Texture) flushes the batch before any state change that
     * would invalidate the snapshot, so reuse is safe.
     */
    bool can_reuse_batch = true;

    /* Find matching batch (check last first for spatial locality).
     *
     * Batch matching uses memcmp on the full MGLStateKey struct (via
     * mglStateKeysEqual).  The key contains hash fields for texture
     * bindings, render state (blend/depth/stencil/etc.), and vertex layout.
     * A hash collision would cause two different states to be falsely merged
     * into one batch, resulting in the second draw using the first draw's
     * render or vertex state during replay.
     *
     * Collision probability: for N=128 batches, P(collision) ≈ N²/2^65
     * ≈ 2^-51, which is negligible.  The hash incorporates pointer
     * addresses (for textures) and ~50 state fields (for render state),
     * making intentional collision infeasible.  If correctness is ever
     * questioned, enable MGL_DEBUG_STREAM_MERGE to log batch merges. */
    MGLDrawBatch *batch = NULL;
    if (can_reuse_batch && cb->batch_count > 0) {
        MGLDrawBatch *last = &cb->batches[cb->batch_count - 1];
        bool keys_match = mglStateKeysEqual(&last->key, &key);
        if (keys_match &&
            last->uses_elements == cmd_uses_elements &&
            last->stream_merged == can_stream_merge &&
            (!can_stream_merge ||
             (last->stream_layout_hash == streamCandidate.layout_hash &&
              !last->sampler_snapshots_mixed &&
              last->sampler_snapshot_id == stored_cmd.sampler_snapshot_id)) &&
            last->command_count < maxDrawsPerBatch &&
            (!last->has_dynamic_uniform_bindings ||
             mglCaptureDynamicUniformRanges(ctx, last, &stored_cmd, true)) &&
            (!last->has_dynamic_texture_bindings ||
             mglCaptureDynamicTextureBindings(ctx, last, &stored_cmd)) &&
            (!last->has_dynamic_vertex_bindings ||
             mglCaptureDynamicVertexBindings(ctx, last, &stored_cmd))) {
            batch = last;
        } else if (!keys_match && !can_stream_merge && !last->stream_merged &&
                   last->uses_elements == cmd_uses_elements &&
                   last->command_count < maxDrawsPerBatch &&
                   mglStateKeysEqualIgnoringUniformRanges(&last->key, &key) &&
                   mglCaptureDynamicUniformRanges(
                       ctx, last, &stored_cmd,
                       last->has_dynamic_uniform_bindings) &&
                   (!last->has_dynamic_texture_bindings ||
                    mglCaptureDynamicTextureBindings(ctx, last, &stored_cmd)) &&
                   (!last->has_dynamic_vertex_bindings ||
                    mglCaptureDynamicVertexBindings(ctx, last, &stored_cmd))) {
            batch = last;
            batch->has_dynamic_uniform_bindings = true;
            batch->mdi_compatible = false;
        } else if (!keys_match && mglBindNoFlushEnabled() &&
                   !can_stream_merge && !last->stream_merged &&
                   last->uses_elements == cmd_uses_elements &&
                   last->command_count < maxDrawsPerBatch &&
                   mglStateKeysEqualIgnoringDynamicBindings(&last->key, &key)) {
            if (mglCaptureDynamicDrawBindings(ctx, last, &key, &stored_cmd)) {
                batch = last;
                if (stored_cmd.dynamic_vertex_binding_count > 0) {
                    batch->has_dynamic_vertex_bindings = true;
                }
                if (stored_cmd.dynamic_uniform_binding_count > 0) {
                    batch->has_dynamic_uniform_bindings = true;
                }
                if (stored_cmd.dynamic_texture_binding_count > 0) {
                    batch->has_dynamic_texture_bindings = true;
                }
                batch->mdi_compatible = false;
            } else {
                /* Compatible except VAO/UBO/texture bindings, but capture
                 * could not materialize a safe per-draw override. */
                MGL_PERF_INC(g_mglMergeRejectDynamicCaptureSinceSwap);
            }
        } else if (!keys_match) {
            MGL_PERF_INC(g_mglMergeRejectStateDiffersSinceSwap);
        } else if (last->command_count >= maxDrawsPerBatch) {
            MGL_PERF_INC(g_mglMergeRejectAppendFailedSinceSwap);
        }
    }
    if (!batch) {
        if (cb->batch_count >= maxBatchCount) {
            MGL_PERF_INC(g_mglFlushReasonCapacitySinceSwap);
            mglFlushCommandBuffer(ctx);
            if (cb->batch_count >= maxBatchCount) {
                /* The flush above did not free a batch slot (it is a no-op when
                 * the renderer handle is gone), so this draw cannot be stored.
                 * Report it: a silently dropped draw is indistinguishable from
                 * missing geometry and never reaches any PERF counter. */
                fprintf(stderr, "MGL Error: mglAppendDrawCommand: batch buffer full after flush\n");
                mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                MGL_SIGNPOST_END(RecordDrawCommand);
                return;
            }
        }
        batch = &cb->batches[cb->batch_count];
        memset(batch, 0, sizeof(*batch));
        batch->sampler_snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;
        batch->key = key;
        batch->uses_elements = cmd_uses_elements;
        /* The frontend command is zero-initialized and does not own the
         * command-buffer-local snapshot ID.  Check the normalized copy so a
         * draw without a sampler snapshot is not mistaken for snapshot 0. */
        batch->mdi_compatible = mglBatchInitMDICompatible(batch, &stored_cmd);

        if (can_stream_merge) {
            if (!mglInitializeStreamMergedBatch(ctx, batch, &streamCandidate)) {
                static unsigned long long s_streamMergeInitFailCount = 0;
                unsigned long long smInitHit = ++s_streamMergeInitFailCount;
                if (smInitHit <= 64ull || (smInitHit % 512ull) == 0ull) {
                    fprintf(stderr, "MGL Warning: stream merged batch init failed; falling back to normal deferred draw (hit=%llu)\n",
                            smInitHit);
                }
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                batch->sampler_snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;
                batch->key = key;
                batch->mdi_compatible = false;
                batch->uses_elements = cmd_uses_elements;
                can_stream_merge = false;
            }
        }

        if (!can_stream_merge) {
            if (!mglInitializeOrShareBatchStateSnapshot(ctx, cb, batch,
                                                        &stored_cmd)) {
                fprintf(stderr, "MGL Error: mglAppendDrawCommand: state snapshot alloc failed\n");
                mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                MGL_SIGNPOST_END(RecordDrawCommand);
                return;
            }
        }

        cb->batch_count++;
    }

    if (can_stream_merge) {
        if (!mglAppendStreamMergedData(batch, &streamCandidate, cmd, &stored_cmd)) {
            MGL_PERF_INC(g_mglMergeRejectAppendFailedSinceSwap);
            static unsigned long long s_streamMergeAppendFailCount = 0;
            unsigned long long smAppendHit = ++s_streamMergeAppendFailCount;
            if (smAppendHit <= 64ull || (smAppendHit % 512ull) == 0ull) {
                fprintf(stderr, "MGL Warning: stream merged append failed; falling back to normal deferred draw (hit=%llu)\n",
                        smAppendHit);
            }
            if (batch->command_count == 0 &&
                cb->batch_count > 0 &&
                batch == &cb->batches[cb->batch_count - 1]) {
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                cb->batch_count--;
            }
            can_stream_merge = false;
            batch = NULL;
            if (!batch) {
                if (cb->batch_count >= maxBatchCount) {
                    MGL_PERF_INC(g_mglFlushReasonCapacitySinceSwap);
                    mglFlushCommandBuffer(ctx);
                    if (cb->batch_count >= maxBatchCount) {
                        fprintf(stderr, "MGL Error: mglAppendDrawCommand: fallback batch buffer full after flush\n");
                        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                        MGL_SIGNPOST_END(RecordDrawCommand);
                        return;
                    }
                }
                batch = &cb->batches[cb->batch_count];
                memset(batch, 0, sizeof(*batch));
                batch->sampler_snapshot_id = MGL_INVALID_SAMPLER_SNAPSHOT_ID;
                batch->key = key;
                batch->mdi_compatible = false;
                batch->uses_elements = cmd_uses_elements;
                if (!mglInitializeOrShareBatchStateSnapshot(ctx, cb, batch,
                                                            &stored_cmd)) {
                    fprintf(stderr, "MGL Error: mglAppendDrawCommand: fallback state snapshot alloc failed\n");
                    mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
                    mglReleaseBatch(ctx, batch);
                    memset(batch, 0, sizeof(*batch));
                    MGL_SIGNPOST_END(RecordDrawCommand);
                    return;
                }
                cb->batch_count++;
            }
            uint16_t sampler_snapshot_id = stored_cmd.sampler_snapshot_id;
            stored_cmd = *cmd;
            stored_cmd.sampler_snapshot_id = sampler_snapshot_id;
        }
    }

    if (batch->command_count >= batch->command_capacity) {
        uint32_t newCapacity = batch->command_capacity ? batch->command_capacity * 2u : 64u;
        if (newCapacity < batch->command_capacity || newCapacity > maxDrawsPerBatch) {
            newCapacity = maxDrawsPerBatch;
        }
        if (newCapacity <= batch->command_count) {
            fprintf(stderr, "MGL Error: mglAppendDrawCommand: command capacity exhausted\n");
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            MGL_SIGNPOST_END(RecordDrawCommand);
            return;
        }
        size_t newBytes = (size_t)newCapacity * sizeof(MGLDrawCommand);
        MGLDrawCommand *new_cmds;
        if (batch->arena_managed && ctx->batch_arena && ctx->batch_arena->enabled) {
            /* Arena path: allocate new block and copy; old block is
             * reclaimed on arena reset (no individual free). */
            new_cmds = (MGLDrawCommand *)arenaAlloc(ctx->batch_arena, newBytes);
            if (new_cmds && batch->commands && batch->command_count > 0) {
                memcpy(new_cmds, batch->commands,
                       (size_t)batch->command_count * sizeof(MGLDrawCommand));
            }
        } else {
            new_cmds = (MGLDrawCommand *)realloc(batch->commands, newBytes);
        }
        if (!new_cmds) {
            fprintf(stderr, "MGL Error: mglAppendDrawCommand: realloc failed\n");
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            if (batch->command_count == 0 &&
                cb->batch_count > 0 &&
                batch == &cb->batches[cb->batch_count - 1]) {
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                cb->batch_count--;
            }
            MGL_SIGNPOST_END(RecordDrawCommand);
            return;
        }
        batch->commands = new_cmds;
        batch->command_capacity = newCapacity;
    }
    mglUpdateBatchSamplerSnapshotState(batch, stored_cmd.sampler_snapshot_id);
    batch->commands[batch->command_count] = stored_cmd;
    batch->command_count++;
    cb->total_commands++;

    mglBatchUpdateMDIOnAppend(batch, &stored_cmd);

    if (batch->uses_elements) {
        cb->element_cmd_count++;
    } else {
        cb->array_cmd_count++;
    }

    if (can_stream_merge) {
        mglTrackPendingBaseBufferReads(ctx);
    } else {
        mglTrackPendingDrawBufferReads(ctx, cmd, cmd_uses_elements);
    }
    mglTrackPendingSampledTextureReads(ctx);
    mglTrackPendingFramebufferTextureWrites(ctx);
    MGL_SIGNPOST_END(RecordDrawCommand);
}

void mglAppendDrawCommand(GLMContext ctx, const MGLDrawCommand *cmd)
{
    mglRecordDrawCommand(ctx, cmd);
}

/*
 * mglFlushCommandBuffer — lowest-level flush, submitting the deferred draw
 * buffer to Metal
 *
 * Trigger: when batch_count > 0, submit the encoded deferred draws to the
 * Metal backend for execution via mglRendererFlushDrawBuffer(ctx).
 * Guarantee: final rendezvous of every hazard-detection flush; after the call
 * all per-CB read/write tracking arrays are cleared.
 * Overflow degradation: none (this function performs no hazard detection; it
 * only submits).
 */
void mglFlushCommandBuffer(GLMContext ctx)
{
    GLuint i, n;
    int sparse_images = 0;

    if (!ctx) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0) return;

    MGL_PERF_INC(g_mglFlushTotalSinceSwap);
    mglRendererFlushDrawBuffer(ctx);

    /* Draw-path sparse imageStore can remap uncommitted tiles the same way
     * compute does; scrub after the batch so later samples see zeros
     * (ARB_sparse_texture2).  Only pay the wait when a sparse image is bound. */
    n = STATE(var).max_image_units;
    if (n > (GLuint)TEXTURE_UNITS)
        n = (GLuint)TEXTURE_UNITS;
    for (i = 0; i < n; i++) {
        Texture *tex = STATE(image_units)[i].tex;
        if (tex && tex->params.sparse) {
            sparse_images = 1;
            break;
        }
    }
    if (sparse_images) {
        mglRendererFlush(ctx, true);
        mglSparseScrubBoundImages(ctx);
    }
}

/*
 * mglFlushPendingDraws — unconditional full flush (no hazard detection)
 *
 * Trigger: unconditionally call mglFlushCommandBuffer when draw_defer_enabled
 * is set; no hazard detection is performed.
 * Guarantee: used for any state change that could broadly invalidate pending
 * draws (e.g. broad GL state resets); conservatively guarantees all encoded
 * draws are submitted before later operations proceed.
 * Overflow degradation: this function is itself a full flush, equivalent to
 * the final behavior after overflow degradation.
 */
void mglFlushPendingDraws(GLMContext ctx)
{
    if (!ctx || !ctx->draw_defer_enabled) return;
    mglFlushCommandBuffer(ctx);
}

const char *mglDrawCommandTypeName(MGLDrawCommandType type)
{
    switch (type) {
        case MGL_CMD_DRAW_ARRAYS: return "draw_arrays";
        case MGL_CMD_DRAW_ELEMENTS: return "draw_elements";
        case MGL_CMD_DRAW_ARRAYS_INSTANCED: return "draw_arrays_instanced";
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED: return "draw_elements_instanced";
        case MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX: return "draw_elements_base_vertex";
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX: return "draw_elements_instanced_base_vertex";
        case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE: return "draw_arrays_instanced_base_instance";
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE: return "draw_elements_instanced_base_instance";
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE: return "draw_elements_instanced_base_vertex_base_instance";
        default: return "unknown";
    }
}

bool mglDrawCommandUsesElements(const MGLDrawCommand *cmd)
{
    if (!cmd) {
        return false;
    }

    switch (cmd->type) {
        case MGL_CMD_DRAW_ARRAYS:
        case MGL_CMD_DRAW_ARRAYS_INSTANCED:
        case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE:
            return false;
        default:
            return true;
    }
}

Buffer *mglDrawCommandElementBuffer(GLMContext ctx, const MGLDrawCommand *cmd)
{
    if (!ctx || !cmd || cmd->element_buffer_name == 0u)
        return NULL;
    return findBuffer(ctx, cmd->element_buffer_name);
}

Buffer *mglNamedBuffer(GLMContext ctx, GLuint name)
{
    if (!ctx || name == 0u)
        return NULL;
    return findBuffer(ctx, name);
}

Texture *mglNamedTexture(GLMContext ctx, GLuint name)
{
    if (!ctx || name == 0u)
        return NULL;
    return findTexture(ctx, name);
}
