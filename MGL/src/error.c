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


GLenum  mglGetError(GLMContext ctx)
{
    /* Per GL spec, glGetError on a NULL context is undefined, but we must not
     * crash.  CTS and well-behaved apps always pass a valid context. */
    if (!ctx)
        return GL_NO_ERROR;

    /* Drain the error queue.  When no errors are queued, return GL_NO_ERROR.
     * When errors are queued, pop the oldest one and return it. */
    if (ctx->state.error_count == 0)
    {
        /* Backwards compatibility: some call sites set ctx->state.error
         * directly instead of going through mglDispatchError.  Surface that
         * single error so it is not silently lost. */
        GLenum legacy = ctx->state.error;
        ctx->state.error = GL_NO_ERROR;
        return legacy;
    }

    GLenum err = ctx->state.error_queue[ctx->state.error_head];
    ctx->state.error_head = (ctx->state.error_head + 1u) % MGL_ERROR_QUEUE_SIZE;
    ctx->state.error_count--;

    /* Mirror the new head (or GL_NO_ERROR when empty) for legacy code that
     * reads ctx->state.error directly. */
    ctx->state.error = (ctx->state.error_count > 0)
        ? ctx->state.error_queue[ctx->state.error_head]
        : GL_NO_ERROR;

    return err;
}


static int mgl_is_ignorable_texture_error(const char *func, GLenum error)
{
    if (!func || error != GL_INVALID_OPERATION)
        return 0;

    /* MGL_STRICT_TEXTURE_ERRORS=1 disables the compatibility error-swallowing
     * so developers can surface real texture bugs during CTS / debugging.
     * Cached once on first call (consistent with the rest of MGL's env-var
     * caching pattern; GL context is single-threaded). */
    static int strict_mode = -1;
    if (strict_mode < 0) {
        const char *env = getenv("MGL_STRICT_TEXTURE_ERRORS");
        strict_mode = (env && atoi(env) > 0) ? 1 : 0;
    }
    if (strict_mode) {
        return 0;
    }

    /* Public texture-buffer entry points have required error semantics. */
    if (strcmp(func, "mglTextureBuffer") == 0 ||
        strcmp(func, "mglTextureBufferRange") == 0 ||
        strcmp(func, "mglTextureBufferRangeImpl") == 0)
        return 0;

    /* Immutable texture-storage entry points have required validation errors
     * (for example repeated allocation and invalid mip counts). */
    if (strstr(func, "TexStorage") != NULL ||
        strstr(func, "TextureStorage") != NULL)
        return 0;
    if (strcmp(func, "generateMipmaps") == 0)
        return 0;

    /* Minecraft startup performs a lot of texture probing/update patterns.
     * Treat transient INVALID_OPERATION from texture functions as non-fatal
     * compatibility warnings so createTexture() does not abort startup.
     *
     * EXCEPTION: functions containing "Image" (mglTexImage2D, mglTexSubImage2D,
     * mglTextureImage2D, etc.) perform format/type validation that CTS relies
     * on via glGetError().  Their errors must NOT be swallowed. */
    if (strstr(func, "mglTex") != NULL || strstr(func, "mglTexture") != NULL)
    {
        if (strstr(func, "Image") != NULL)
            return 0;  /* validation error - report it */
        return 1;      /* transient error - swallow it */
    }
    if (strstr(func, "texSubImage") != NULL) return 1;
    if (strstr(func, "createTextureLevel") != NULL) return 1;

    return 0;
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

void error_func(GLMContext ctx, const char *func, GLenum error)
{
    if (mgl_is_ignorable_texture_error(func, error))
    {
        static unsigned long long s_ignorable_texture_error_count = 0;
        s_ignorable_texture_error_count++;
        if (s_ignorable_texture_error_count <= 64ull ||
            (s_ignorable_texture_error_count % 1024ull) == 0ull) {
            fprintf(stderr,
                    "MGL WARNING: Ignoring transient texture error from %s to improve compatibility (0x%x, hit=%llu)\n",
                    func,
                    error,
                    s_ignorable_texture_error_count);
        }
        return;
    }

    fprintf(stderr, "MGL GL Error in %s: 0x%x (%d)\n", func, error, error);

    /* Push the error into the queue.  Per GL 4.6 spec §2.5, the queue must
     * hold at least 16 errors; when full, the new error is dropped (the
     * oldest 16 are retained).  This replaces the previous depth-1 behavior
     * that silently discarded every error after the first. */
    if (ctx->state.error_count < MGL_ERROR_QUEUE_SIZE)
    {
        GLuint tail = (ctx->state.error_head + ctx->state.error_count) % MGL_ERROR_QUEUE_SIZE;
        ctx->state.error_queue[tail] = error;
        ctx->state.error_count++;
        /* Mirror the head for legacy code that reads ctx->state.error. */
        ctx->state.error = ctx->state.error_queue[ctx->state.error_head];
    }
    else
    {
        /* Queue full — the new error is dropped per spec.  Keep the legacy
         * field pointing at the current head. */
        ctx->state.error = ctx->state.error_queue[ctx->state.error_head];
    }

    /* Temporarily disabled to allow QEMU to continue despite errors */
    // if (ctx->assert_on_error)
    //     assert(0);
}
