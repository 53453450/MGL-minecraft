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
 * fence.c
 * MGL
 *
 */

#include <strings.h>
#include <stdlib.h>
#include <time.h>

#include "glm_context.h"
#include "draw_command.h"

Sync *newSync(GLMContext ctx)
{
    Sync *ptr;

    ptr = (Sync *)malloc(sizeof(Sync));
    // CRITICAL SECURITY FIX: Check malloc result instead of using assert()
    if (!ptr) {
        fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate memory for Sync\n");
        return NULL;
    }

    bzero(ptr, sizeof(Sync));

    ptr->name = STATE(sync_name)++;

    /* initial reference owned by the caller's GLsync handle. */
    atomic_store_explicit(&ptr->refcount, 1, memory_order_relaxed);
    ptr->delete_status = GL_FALSE;

    return ptr;
}

/* Sync reference counting.  Prevents use-after-free when glDeleteSync
 * races with an in-progress mglClientWaitSync/mglWaitSync on another thread. */
static void mglRetainSyncReference(Sync *sync)
{
    if (!sync) return;
    atomic_fetch_add_explicit(&sync->refcount, 1, memory_order_relaxed);
}

/* Release a reference.  When refcount hits zero and delete_status is set,
 * release Metal resources and free the shell.  Mirrors the Buffer/Program
 * refcount pattern. */
static void mglReleaseSyncReference(GLMContext ctx, Sync *sync)
{
    if (!sync) return;
    int prev = atomic_fetch_sub_explicit(&sync->refcount, 1, memory_order_acq_rel);
    if (prev == 1) {
        /* Last reference dropped. Check delete_status with acquire semantics to
         * synchronize with the store in glDeleteSync. */
        bool should_delete = atomic_load_explicit((_Atomic bool *)&sync->delete_status,
                                                   memory_order_acquire);
        if (should_delete) {
            /* glDeleteSync was called: release Metal resources and free. */
            if (ctx && ctx->mtl_funcs.mtlReleaseSync) {
                ctx->mtl_funcs.mtlReleaseSync(ctx, sync);
            } else if (ctx && ctx->mtl_funcs.mtlWaitForSync &&
                       (sync->mtl_command_buffer || sync->mtl_event)) {
                ctx->mtl_funcs.mtlWaitForSync(ctx, sync);
            }
            free(sync);
        }
    }
}

int isSync(GLMContext ctx, GLsync sync)
{
    if (sync->name < STATE(sync_name))
        return 1;

    return 0;
}

GLsync mglFenceSync(GLMContext ctx, GLenum condition, GLbitfield flags)
{
    Sync *ptr;

    /* GL 4.6 §5.3: condition must be GL_SYNC_GPU_COMMANDS_COMPLETE */
    if (condition != GL_SYNC_GPU_COMMANDS_COMPLETE)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return NULL;
    }

    /* GL 4.6 §5.3: flags must be zero */
    if (flags != 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return NULL;
    }

    ptr = newSync(ctx);

    ctx->mtl_funcs.mtlGetSync(ctx, ptr);

    /* register in sync_table so destroyGLMContext can release
     * Metal resources. mglDeleteSync removes the entry on explicit free. */
    insertHashElement(&STATE(sync_table), ptr->name, ptr);

    return ptr;
}


GLboolean mglIsSync(GLMContext ctx, GLsync sync)
{
    if (sync == NULL)
    {
        return false;
    }

    return isSync(ctx, sync);
}

void mglDeleteSync(GLMContext ctx, GLsync sync)
{
    if (isSync(ctx, sync) == GL_FALSE)
    {
        // CRITICAL FIX: Handle invalid sync gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Attempting to delete invalid sync object %p\n", sync);
        return;
    }

    /* remove from sync_table before releasing resources. */
    deleteHashElement(&STATE(sync_table), sync->name);

    /* mark for deletion and release the caller's reference. If a
     * concurrent mglClientWaitSync/mglWaitSync holds a reference, the shell
     * survives until the last release frees it. mtl_data is released only in
     * mglReleaseSyncReference (or mglDestroyContextSync for never-deleted
     * syncs at context destroy time). Use release semantics to synchronize
     * with the acquire load in mglReleaseSyncReference. */
    atomic_store_explicit((_Atomic bool *)&sync->delete_status, GL_TRUE,
                          memory_order_release);
    mglReleaseSyncReference(ctx, sync);
}

GLenum  mglClientWaitSync(GLMContext ctx, GLsync sync, GLbitfield flags, GLuint64 timeout)
{
    GLenum result = GL_INVALID_VALUE;

    if (flags & ~GL_SYNC_FLUSH_COMMANDS_BIT)
    {
        // CRITICAL FIX: Handle invalid flags gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Invalid sync flags 0x%x, only GL_SYNC_FLUSH_COMMANDS_BIT allowed\n", flags);
        return GL_INVALID_VALUE;
    }

    if (isSync(ctx, sync) == GL_FALSE)
    {
        // CRITICAL FIX: Handle invalid sync gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Invalid sync object %p passed to client wait sync\n", sync);
        return GL_INVALID_VALUE;
    }

    /* retain so a concurrent glDeleteSync cannot free the sync while
     * we access its mtl_command_buffer/mtl_event below. */
    mglRetainSyncReference(sync);

    /* GL_ALREADY_SIGNALED: the fence had already completed at call time, so no
     * wait is performed. mtlGetSyncStatus reports GL_SIGNALED when the retained
     * command buffer has completed or when there is no CB to wait on. */
    if (ctx->mtl_funcs.mtlGetSyncStatus &&
        ctx->mtl_funcs.mtlGetSyncStatus(ctx, sync) == GL_SIGNALED)
    {
        result = GL_ALREADY_SIGNALED;
        goto cleanup;
    }

    /* timeout == 0 is a non-blocking probe: return immediately without waiting. */
    if (timeout == 0)
    {
        result = GL_TIMEOUT_EXPIRED;
        goto cleanup;
    }

    /* Finite timeout: mtlWaitForSync blocks via waitUntilCompleted (which has no
     * timeout), so to honor a bounded timeout we poll the non-blocking status
     * with short sleeps up to the timeout, returning GL_TIMEOUT_EXPIRED if the
     * fence does not complete in time. */
    if (ctx->mtl_funcs.mtlGetSyncStatus)
    {
        const uint64_t poll_interval_ns = 500000; /* 0.5 ms */
        uint64_t elapsed_ns = 0;

        while (elapsed_ns < timeout)
        {
            if (ctx->mtl_funcs.mtlGetSyncStatus(ctx, sync) == GL_SIGNALED)
            {
                result = GL_CONDITION_SATISFIED;
                goto cleanup;
            }

            struct timespec ts;
            ts.tv_sec = 0;
            ts.tv_nsec = (long)poll_interval_ns;
            nanosleep(&ts, NULL);

            elapsed_ns += poll_interval_ns;
        }

        /* Final check after the timeout has elapsed. */
        if (ctx->mtl_funcs.mtlGetSyncStatus(ctx, sync) == GL_SIGNALED)
        {
            result = GL_CONDITION_SATISFIED;
            goto cleanup;
        }

        result = GL_TIMEOUT_EXPIRED;
        goto cleanup;
    }

    /* Fallback (no status query available): block until the fence completes.
     *
     * MGL_SYNC_STRICT: fence wait already performs conservative sync via
     * mtlWaitForSync (waitUntilCompleted); no extra strict branch needed. */
    ctx->mtl_funcs.mtlWaitForSync(ctx, sync);
    result = GL_CONDITION_SATISFIED;

cleanup:
    mglReleaseSyncReference(ctx, sync);
    return result;
}

void mglWaitSync(GLMContext ctx, GLsync sync, GLbitfield flags, GLuint64 timeout)
{
    if (isSync(ctx, sync) == GL_FALSE)
    {
        // CRITICAL FIX: Handle invalid sync gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Invalid sync object %p passed to wait sync\n", sync);
        return;
    }

    if (timeout != GL_TIMEOUT_IGNORED) {
        // CRITICAL FIX: Handle invalid timeout gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Server wait sync timeout must be GL_TIMEOUT_IGNORED, got 0x%llx\n", timeout);
        // Continue with GL_TIMEOUT_IGNORED behavior
    }

    /* retain so a concurrent glDeleteSync cannot free the sync while
     * mtlWaitForSync blocks on sync->mtl_command_buffer. */
    mglRetainSyncReference(sync);

    /* mtlWaitForSync now blocks via waitUntilCompleted on the retained command
     * buffer, satisfying the GL spec requirement that glWaitSync block until the
     * fence's insertion-point-prior commands have completed on the GPU.
     *
     * MGL_SYNC_STRICT: fence wait already performs conservative sync via
     * mtlWaitForSync (waitUntilCompleted); no extra strict branch needed. */
    ctx->mtl_funcs.mtlWaitForSync(ctx, sync);

    mglReleaseSyncReference(ctx, sync);
}

void mglGetSynciv(GLMContext ctx, GLsync sync, GLenum pname, GLsizei count, GLsizei *length, GLint *values)
{
    if (isSync(ctx, sync) == GL_FALSE)
    {
        // CRITICAL FIX: Handle invalid sync gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Invalid sync object %p passed to get sync iv\n", sync);
        if (length) *length = 0;
        return;
    }

    // CRITICAL FIX: count is the number of elements the caller allocated in values.
    // Per OpenGL spec, only one value is returned per pname. length is an OUTPUT parameter.
    if (!count || count < 0) {
        fprintf(stderr, "MGL ERROR: Invalid count %d in get sync iv\n", count);
        if (length) *length = 0;
        return;
    }
    if (!values) {
        fprintf(stderr, "MGL ERROR: NULL values pointer in get sync iv\n");
        if (length) *length = 0;
        return;
    }

    /* retain so a concurrent glDeleteSync cannot free the sync while
     * we read its mtl_command_buffer/mtl_event below. */
    mglRetainSyncReference(sync);

    // Only write one value per pname per the OpenGL spec
    switch(pname)
    {
        case GL_OBJECT_TYPE:
            *values = GL_SYNC_FENCE;
            break;

        case GL_SYNC_STATUS:
            /* Non-blocking status query. Report GL_SIGNALED when the retained
             * command buffer has completed (or there is no CB to wait on);
             * otherwise GL_UNSIGNALED. Does not block. Falls back to the
             * void*-null check if the backend status entry is unavailable. */
            if (ctx->mtl_funcs.mtlGetSyncStatus) {
                *values = ctx->mtl_funcs.mtlGetSyncStatus(ctx, sync);
            } else {
                *values = (sync->mtl_command_buffer == NULL && sync->mtl_event == NULL)
                          ? GL_SIGNALED : GL_UNSIGNALED;
            }
            break;

        case GL_SYNC_CONDITION:
            *values = GL_SYNC_GPU_COMMANDS_COMPLETE;
            break;

        case GL_SYNC_FLAGS:
            *values = 0;
            break;

        default:
            // CRITICAL FIX: Handle unknown sync parameters gracefully instead of crashing
            fprintf(stderr, "MGL ERROR: Unknown sync parameter 0x%x in get sync iv\n", pname);
            *values = 0;
            break;
    }

    if (length) *length = 1;

    mglReleaseSyncReference(ctx, sync);
}

void mglTextureBarrier(GLMContext ctx)
{
    if (!ctx) {
        return;
    }

    mglFlushCommandBuffer(ctx);
}

void mglMemoryBarrier(GLMContext ctx, GLbitfield barriers)
{
    const GLbitfield valid_barriers =
        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT |
        GL_ELEMENT_ARRAY_BARRIER_BIT |
        GL_UNIFORM_BARRIER_BIT |
        GL_TEXTURE_FETCH_BARRIER_BIT |
        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
        GL_COMMAND_BARRIER_BIT |
        GL_PIXEL_BUFFER_BARRIER_BIT |
        GL_TEXTURE_UPDATE_BARRIER_BIT |
        GL_BUFFER_UPDATE_BARRIER_BIT |
        GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT |
        GL_FRAMEBUFFER_BARRIER_BIT |
        GL_TRANSFORM_FEEDBACK_BARRIER_BIT |
        GL_ATOMIC_COUNTER_BARRIER_BIT |
        GL_SHADER_STORAGE_BARRIER_BIT;

    if (barriers != GL_ALL_BARRIER_BITS && (barriers & ~valid_barriers))
    {
        // extra bits...
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (!ctx) {
        return;
    }

    /*
     * Metal command buffers provide the actual visibility boundary for compute
     * writes consumed by later GL reads or draws. This conservative barrier
     * gives SSBO/image/texture updates GL ordering semantics until finer-grain
     * encoder hazards are implemented.
     *
     * Compute encoder coverage: MGL does NOT keep a long-lived compute encoder
     * across GL calls — every glDispatchCompute / tessellation dispatch creates
     * a local MTLComputeCommandEncoder via [_currentCommandBuffer
     * computeCommandEncoder] and calls endEncoding() before returning (see
     * mtlDispatchCompute and the TCS/TES dispatch paths in MGLRenderer.m). Thus
     * no open compute encoder exists when mglMemoryBarrier is reached, and the
     * flush path below (mglFlushCommandBuffer -> mtlFlush -> flushCommandBuffer:
     * -> endRenderEncoding + commit + waitUntilCompleted) is sufficient: it
     * commits the current CB (which already contains all encoded compute
     * dispatches) and waits for completion, making compute writes visible to
     * subsequent GL draws/reads. No explicit endComputeEncoding is needed here.
     */
    mglFlushCommandBuffer(ctx);
    if (ctx->mtl_funcs.mtlFlush) {
        ctx->mtl_funcs.mtlFlush(ctx, true);
    }
    /* MGL_SYNC_STRICT: mglFlushCommandBuffer + mtlFlush(ctx, true)
     * (commit + waitUntilCompleted) already ran here, a conservative path
     * that needs no extra strict branch. */

    /* Storage image (imageStore) writes go directly to the GPU Metal texture.
     * Without marking the texture/level as metal_data_authoritative, subsequent
     * glGetTexImage calls read stale CPU cached data (lvl->data) instead of
     * the GPU-written pixels.  Per the GL 4.6 spec, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
     * and GL_TEXTURE_UPDATE_BARRIER_BIT both guarantee that later texture reads
     * observe prior shader image writes, so flip the authoritative flag on
     * every currently-bound image unit's texture here.  The flag is cleared
     * again by any subsequent CPU-side texture upload (glTexSubImage/glTexImage). */
    GLbitfield image_relevant_bits =
        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT;
    if (barriers == GL_ALL_BARRIER_BITS || (barriers & image_relevant_bits))
    {
        GLuint max_units = ctx->state.var.max_image_units;
        for (GLuint i = 0; i < max_units && i < TEXTURE_UNITS; i++) {
            ImageUnit *iu = &ctx->state.image_units[i];
            Texture *tex = iu->tex;
            if (!tex || !tex->faces[0].levels) {
                continue;
            }
            if (iu->level >= (GLint)tex->num_levels) {
                continue;
            }
            tex->metal_data_authoritative = GL_TRUE;
            tex->faces[0].levels[iu->level].metal_data_authoritative = GL_TRUE;
        }
    }
}

void mglMemoryBarrierByRegion(GLMContext ctx, GLbitfield barriers)
{

    if (barriers & ~(GL_ATOMIC_COUNTER_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT))
    {
        // extra bits...
        ERROR_RETURN(GL_INVALID_VALUE);
    }
}
