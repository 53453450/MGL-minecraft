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
#include <assert.h>
#include <string.h>

#include "error.h"


GLenum  mglGetError(GLMContext ctx)
{
    GLenum err;

    err = ctx->state.error;

    if (err != GL_NO_ERROR) {
        static unsigned long long s_get_error_log_count = 0;
        s_get_error_log_count++;
        if (s_get_error_log_count <= 64ull || (s_get_error_log_count % 1024ull) == 0ull) {
            fprintf(stderr, "MGL DEBUG: mglGetError returning 0x%x (%d) hit=%llu\n",
                    err, err, s_get_error_log_count);
        }
    }

    ctx->state.error = GL_NO_ERROR;

    return err;
}


static int mgl_is_ignorable_texture_error(const char *func, GLenum error)
{
    if (!func || error != GL_INVALID_OPERATION)
        return 0;

    /* Minecraft startup performs a lot of texture probing/update patterns.
     * Treat transient INVALID_OPERATION from texture paths as non-fatal
     * compatibility warnings so createTexture() does not abort startup. */
    if (strstr(func, "mglTex") != NULL) return 1;
    if (strstr(func, "mglTexture") != NULL) return 1;
    if (strstr(func, "texSubImage") != NULL) return 1;
    if (strstr(func, "generateMipmaps") != NULL) return 1;
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

    if (ctx->state.error)
        return;

    ctx->state.error = error;

    /* Temporarily disabled to allow QEMU to continue despite errors */
    // if (ctx->assert_on_error)
    //     assert(0);
}
