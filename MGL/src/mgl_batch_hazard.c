/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_batch_hazard.h"

MGLHazardOverflowPolicy mgl_batch_hazard_overflow_policy(int flush_and_continue_enabled)
{
    if (flush_and_continue_enabled) {
        return MGL_HAZARD_OVERFLOW_FLUSH_AND_CONTINUE;
    }
    return MGL_HAZARD_OVERFLOW_STICKY;
}

MGLHazardTrackAction mgl_batch_hazard_on_capacity_full(
    MGLHazardOverflowPolicy policy, int has_pending_draws)
{
    if (policy == MGL_HAZARD_OVERFLOW_FLUSH_AND_CONTINUE && has_pending_draws) {
        return MGL_HAZARD_TRACK_FLUSH_THEN_RETRY;
    }
    return MGL_HAZARD_TRACK_LATCH_OVERFLOW;
}

int mgl_batch_hazard_query_degraded(int overflow_latched)
{
    return overflow_latched ? 1 : 0;
}
