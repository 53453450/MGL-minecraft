/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_batching_state.h — the batch-replay / batching switches and arena.
 *
 * The record used to be an Objective-C-only struct (MGLRenderer_State.h) with
 * BOOL fields, so every C driver had to reach each flag through its own shim
 * port.  It is C state now: the renderer still owns it as an ivar, one port
 * hands out its address, and C reads and writes the fields directly.  That
 * retired six wrapper ports.
 */

#ifndef MGL_BATCHING_STATE_H
#define MGL_BATCHING_STATE_H

#include "draw_command.h"   /* MGLBatchArena */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLBatchingState_t {
    MGLBatchArena batchArena;
    uint8_t arenaSnapshotEnabled;
    uint8_t skipSameKeyRestoreEnabled;
    uint8_t dirtyKeyDeltaEnabled;
    /* Replay of a BindNoFlush batch that captured per-draw BindVertexBuffer
     * overrides.  Descriptor bakes only relativeoffset; setVertexBuffer uses
     * the absolute VERTEX_BINDING_OFFSET so overrides are not double-counted. */
    uint8_t absoluteVertexBindingOffsets;
} MGLBatchingState;

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCHING_STATE_H */
