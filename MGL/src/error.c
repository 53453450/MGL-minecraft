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
 * error.h
 * MGL
 *
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "error.h"
#include "mgl_frame_activity.h"

/* Mirror the legacy single-slot error field onto the current active_state so
 * in-replay `STATE(error)` probes still see the latest push. The queue itself
 * stays on live state only (T0-2). */
static void mglMirrorLegacyError(GLMContext ctx, GLenum error)
{
    LIVE_STATE(error) = error;
    if (ctx->active_state && ctx->active_state != &ctx->state)
        ctx->active_state->error = error;
}

GLenum  mglGetError(GLMContext ctx)
{
    /* Per GL spec, glGetError on a NULL context is undefined, but we must not
     * crash.  CTS and well-behaved apps always pass a valid context. */
    if (!ctx)
        return GL_NO_ERROR;

    /* Drain the live error queue only — never the replay workspace. */
    if (LIVE_STATE(error_count) == 0)
    {
        /* Backwards compatibility: mglClearCurrentError is the only remaining
         * writer of the legacy single-slot error outside this file.  Surface
         * that single error so it is not silently lost. */
        GLenum legacy = LIVE_STATE(error);
        mglMirrorLegacyError(ctx, GL_NO_ERROR);
        return legacy;
    }

    GLenum err = LIVE_STATE(error_queue)[LIVE_STATE(error_head)];
    LIVE_STATE(error_head) =
        (LIVE_STATE(error_head) + 1u) % MGL_ERROR_QUEUE_SIZE;
    LIVE_STATE(error_count)--;

    /* Mirror the new head (or GL_NO_ERROR when empty) for legacy code that
     * reads STATE(error) directly. */
    mglMirrorLegacyError(ctx,
                         (LIVE_STATE(error_count) > 0)
                             ? LIVE_STATE(error_queue)[LIVE_STATE(error_head)]
                             : GL_NO_ERROR);

    return err;
}


void mglDispatchError(GLMContext ctx, const char *func, GLenum error)
{
    if (!ctx) {
        fprintf(stderr,
                "MGL ERROR: dispatch with NULL ctx in %s (0x%x)\n",
                func ? func : "(null)",
                error);
        return;
    }

    if (ctx->error_func) {
        ctx->error_func(ctx, func, error);
        return;
    }

    fprintf(stderr,
            "MGL WARNING: ctx->error_func is NULL in %s (0x%x), falling back to default handler\n",
            func ? func : "(null)",
            error);
    error_func(ctx, func, error);
}

void mglClearCurrentError(GLMContext ctx)
{
    if (!ctx)
        return;
    mglMirrorLegacyError(ctx, GL_NO_ERROR);
}

void error_func(GLMContext ctx, const char *func, GLenum error)
{
    fprintf(stderr, "MGL GL Error in %s: 0x%x (%d)\n", func, error, error);

    /* T0-3: count pushes that would previously have landed on the workspace. */
    if (mglCtxActiveIsReplayWorkspace(ctx))
        MGL_FRAME_INC(g_mglReplayErrorRedirectsSinceSwap);

    /* Push into the live queue only (T0-2). Per GL 4.6 §2.3.1 the queue holds
     * at least 16 errors; when full, the new error is dropped. */
    if (LIVE_STATE(error_count) < MGL_ERROR_QUEUE_SIZE)
    {
        GLuint tail =
            (LIVE_STATE(error_head) + LIVE_STATE(error_count)) %
            MGL_ERROR_QUEUE_SIZE;
        LIVE_STATE(error_queue)[tail] = error;
        LIVE_STATE(error_count)++;
        mglMirrorLegacyError(ctx,
                             LIVE_STATE(error_queue)[LIVE_STATE(error_head)]);
    }
    else
    {
        /* Queue full — drop the new error per spec; keep legacy head. */
        mglMirrorLegacyError(ctx,
                             LIVE_STATE(error_queue)[LIVE_STATE(error_head)]);
    }

    /* Temporarily disabled to allow QEMU to continue despite errors */
    // if (ctx->assert_on_error)
    //     assert(0);
}
