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
 * mgl_batch_hazard.h — pure-C hazard overflow policy (O2.2).
 *
 * Owns sticky vs flush-and-continue semantics for pending-draw tracking
 * tables (buffer ranges / texture read / texture write). ObjC must not
 * choose the strategy; draw_command.c only applies the action.
 *
 * No Metal. Safe for Linux unit tests.
 */

#ifndef MGL_BATCH_HAZARD_H
#define MGL_BATCH_HAZARD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MGLHazardOverflowPolicy {
    /* Capacity full → latch sticky overflow; later queries degrade to
     * unconditional hazard hits until the command buffer is flushed. */
    MGL_HAZARD_OVERFLOW_STICKY = 0,
    /* Capacity full with pending draws → flush now (resets tables) then
     * retry the insert so tracking stays precise. */
    MGL_HAZARD_OVERFLOW_FLUSH_AND_CONTINUE = 1
} MGLHazardOverflowPolicy;

typedef enum MGLHazardTrackAction {
    MGL_HAZARD_TRACK_INSERT = 0,
    MGL_HAZARD_TRACK_LATCH_OVERFLOW = 1,
    MGL_HAZARD_TRACK_FLUSH_THEN_RETRY = 2
} MGLHazardTrackAction;

/* Map an enable flag (e.g. MGL_HAZARD_OVERFLOW_FLUSH_CONTINUE) to policy.
 * Default (flag == 0) is sticky — preserves historical behavior. */
MGLHazardOverflowPolicy mgl_batch_hazard_overflow_policy(int flush_and_continue_enabled);

/* Decide what to do when a tracking table is at capacity.
 * has_pending_draws: non-zero when batch_count/total_commands imply a flush
 * can make progress. Without pending draws, flush-and-continue degrades to
 * latch (flush would be a no-op and retry would loop). */
MGLHazardTrackAction mgl_batch_hazard_on_capacity_full(
    MGLHazardOverflowPolicy policy, int has_pending_draws);

/* Query-side degradation: non-zero overflow latch ⇒ treat as hit-all. */
int mgl_batch_hazard_query_degraded(int overflow_latched);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BATCH_HAZARD_H */
