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
 * fence.c
 * MGL
 *
 */

#include <strings.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

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

    /* Fence calls can run concurrently on one context.  Reserve the name
     * under sync_lock when the object is published, rather than racing on the
     * context-local counter during allocation. */
    (void)ctx;
    ptr->name = 0;

    /* initial reference owned by the caller's GLsync handle. */
    atomic_store_explicit(&ptr->refcount, 1, memory_order_relaxed);
    atomic_store_explicit(&ptr->delete_status, GL_FALSE, memory_order_relaxed);

    return ptr;
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
        GLboolean should_delete = atomic_load_explicit(&sync->delete_status,
                                                        memory_order_acquire);
        if (should_delete) {
            /* glDeleteSync was called: release Metal resources and free. */
            if (ctx) {
                mglRendererReleaseSync(ctx, sync);
            }
            free(sync);
        }
    }
}

typedef struct {
    const void *needle;
    Sync *found;
} MGLSyncLookup;

static void mglFindSyncByPointer(GLuint name, void *data, void *user)
{
    (void)name;
    MGLSyncLookup *lookup = (MGLSyncLookup *)user;
    if (lookup && data == lookup->needle) {
        lookup->found = (Sync *)data;
    }
}

/* Find by pointer identity without dereferencing the caller's handle.  A
 * stale GLsync may already have been freed, so reading sync->name before the
 * membership check is itself unsafe. */
static Sync *mglFindSyncLocked(GLMContext ctx, GLsync handle)
{
    if (!ctx || !handle) {
        return NULL;
    }

    MGLSyncLookup lookup = { handle, NULL };
    mglHashTableForEach(&STATE(sync_table), mglFindSyncByPointer, &lookup);
    return lookup.found;
}

/* Atomically acquire a sync handle with respect to glDeleteSync.  The table
 * lock covers both pointer lookup and the refcount increment; after this
 * function returns the caller owns one reference and may touch Metal state. */
static Sync *mglAcquireSync(GLMContext ctx, GLsync handle)
{
    if (!ctx || !handle || !ctx->sync_lock_initialized ||
        pthread_mutex_lock(&ctx->sync_lock) != 0) {
        return NULL;
    }

    Sync *sync = mglFindSyncLocked(ctx, handle);
    if (sync && atomic_load_explicit(&sync->refcount, memory_order_acquire) > 0 &&
        !atomic_load_explicit(&sync->delete_status, memory_order_acquire)) {
        atomic_fetch_add_explicit(&sync->refcount, 1, memory_order_relaxed);
    } else {
        sync = NULL;
    }

    (void)pthread_mutex_unlock(&ctx->sync_lock);
    return sync;
}

/* Remove a sync from the name table and mark it deleted while holding the
 * same lock used by mglAcquireSync.  The returned pointer still owns its GL
 * handle reference, which the caller must release. */
static Sync *mglDetachSyncForDelete(GLMContext ctx, GLsync handle)
{
    if (!ctx || !handle || !ctx->sync_lock_initialized ||
        pthread_mutex_lock(&ctx->sync_lock) != 0) {
        return NULL;
    }

    Sync *sync = mglFindSyncLocked(ctx, handle);
    if (sync) {
        deleteHashElement(&STATE(sync_table), sync->name);
        atomic_store_explicit(&sync->delete_status, GL_TRUE,
                              memory_order_release);
    }

    (void)pthread_mutex_unlock(&ctx->sync_lock);
    return sync;
}

/* Register a fence API operation before touching the context or backend.
 * destroyGLMContext closes this gate and waits for the count to reach zero
 * before freeing either object. */
static GLboolean mglSyncOperationEnter(GLMContext ctx)
{
    if (!ctx || !ctx->sync_lock_initialized ||
        pthread_mutex_lock(&ctx->sync_lock) != 0) {
        return GL_FALSE;
    }

    if (ctx->sync_destroying) {
        (void)pthread_mutex_unlock(&ctx->sync_lock);
        return GL_FALSE;
    }

    ctx->sync_active_ops++;
    (void)pthread_mutex_unlock(&ctx->sync_lock);
    return GL_TRUE;
}

static void mglSyncOperationLeave(GLMContext ctx)
{
    if (!ctx || !ctx->sync_lock_initialized ||
        pthread_mutex_lock(&ctx->sync_lock) != 0) {
        return;
    }

    if (ctx->sync_active_ops != 0) {
        ctx->sync_active_ops--;
    }
    if (ctx->sync_destroying && ctx->sync_active_ops == 0 &&
        ctx->sync_cond_initialized) {
        (void)pthread_cond_signal(&ctx->sync_cond);
    }
    (void)pthread_mutex_unlock(&ctx->sync_lock);
}

int isSync(GLMContext ctx, GLsync sync)
{
    /* Membership is checked by pointer identity under the same lock used by
     * mglAcquireSync/mglDetachSyncForDelete.  This is a validity probe only;
     * callers that dereference the handle must use mglAcquireSync instead. */
    if (!ctx || !sync || !ctx->sync_lock_initialized ||
        pthread_mutex_lock(&ctx->sync_lock) != 0) {
        return 0;
    }
    Sync *found = mglFindSyncLocked(ctx, sync);
    (void)pthread_mutex_unlock(&ctx->sync_lock);
    return found != NULL ? 1 : 0;
}

GLsync mglFenceSync(GLMContext ctx, GLenum condition, GLbitfield flags)
{
    Sync *ptr;

    if (!mglSyncOperationEnter(ctx)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return NULL;
    }

    /* GL 4.6 §5.3: condition must be GL_SYNC_GPU_COMMANDS_COMPLETE */
    if (condition != GL_SYNC_GPU_COMMANDS_COMPLETE)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        mglSyncOperationLeave(ctx);
        return NULL;
    }

    /* GL 4.6 §5.3: flags must be zero */
    if (flags != 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        mglSyncOperationLeave(ctx);
        return NULL;
    }

    ptr = newSync(ctx);
    if (!ptr) {
        mglSyncOperationLeave(ctx);
        return NULL;
    }

    /* The GL semantic layer drains deferred draws before the backend captures
     * the fence command buffer. The gate-on callback can then submit and
     * rotate the C++ owner without selector-forwarding into the renderer. */
    mglFlushPendingDraws(ctx);
    mglRendererGetSync(ctx, ptr);

    /* Register in sync_table so destroyGLMContext can release Metal resources.
     * Publication is locked so a concurrent wait cannot observe a half-built
     * entry. */
    if (!ctx->sync_lock_initialized || pthread_mutex_lock(&ctx->sync_lock) != 0) {
        mglRendererReleaseSync(ctx, ptr);
        free(ptr);
        ERROR_RETURN(GL_INVALID_OPERATION);
        mglSyncOperationLeave(ctx);
        return NULL;
    }
    ptr->name = STATE(sync_name)++;
    insertHashElement(&STATE(sync_table), ptr->name, ptr);
    (void)pthread_mutex_unlock(&ctx->sync_lock);

    mglSyncOperationLeave(ctx);
    return ptr;
}


GLboolean mglIsSync(GLMContext ctx, GLsync sync)
{
    if (sync == NULL)
    {
        return false;
    }

    if (!mglSyncOperationEnter(ctx)) {
        return GL_FALSE;
    }
    GLboolean result = isSync(ctx, sync);
    mglSyncOperationLeave(ctx);
    return result;
}

void mglDeleteSync(GLMContext ctx, GLsync sync)
{
    if (!mglSyncOperationEnter(ctx)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    Sync *owned = mglDetachSyncForDelete(ctx, sync);
    if (!owned)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        mglSyncOperationLeave(ctx);
        return;
    }

    /* mark for deletion and release the caller's reference. If a
     * concurrent mglClientWaitSync/mglWaitSync holds a reference, the shell
     * survives until the last release frees it. mtl_data is released only in
     * mglReleaseSyncReference (or mglDestroyContextSync for never-deleted
     * syncs at context destroy time). Use release semantics to synchronize
     * with the acquire load in mglReleaseSyncReference. */
    mglReleaseSyncReference(ctx, owned);
    mglSyncOperationLeave(ctx);
}

GLenum  mglClientWaitSync(GLMContext ctx, GLsync sync, GLbitfield flags, GLuint64 timeout)
{
    GLenum result = GL_INVALID_VALUE;

    if (!mglSyncOperationEnter(ctx)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return GL_WAIT_FAILED;
    }

    if (flags & ~GL_SYNC_FLUSH_COMMANDS_BIT)
    {
        // CRITICAL FIX: Handle invalid flags gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Invalid sync flags 0x%x, only GL_SYNC_FLUSH_COMMANDS_BIT allowed\n", flags);
        mglSyncOperationLeave(ctx);
        return GL_INVALID_VALUE;
    }

    Sync *owned = mglAcquireSync(ctx, sync);
    if (!owned)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        mglSyncOperationLeave(ctx);
        return GL_WAIT_FAILED;
    }

    /* GL_ALREADY_SIGNALED: the fence had already completed at call time, so no
     * wait is performed. The backend status query reports GL_SIGNALED when the retained
     * command buffer has completed or when there is no CB to wait on. */
    if (mglRendererGetSyncStatus(ctx, owned) == GL_SIGNALED)
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

    /* Finite timeout: the backend wait blocks until completion (which has no
     * timeout), so to honor a bounded timeout we poll the non-blocking status
     * with short sleeps up to the timeout, returning GL_TIMEOUT_EXPIRED if the
     * fence does not complete in time. */
    const uint64_t poll_interval_ns = 500000; /* 0.5 ms */
    uint64_t elapsed_ns = 0;

    while (elapsed_ns < timeout)
    {
        if (mglRendererGetSyncStatus(ctx, owned) == GL_SIGNALED)
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

    if (mglRendererGetSyncStatus(ctx, owned) == GL_SIGNALED)
    {
        result = GL_CONDITION_SATISFIED;
        goto cleanup;
    }

    result = GL_TIMEOUT_EXPIRED;

cleanup:
    mglReleaseSyncReference(ctx, owned);
    mglSyncOperationLeave(ctx);
    return result;
}

void mglWaitSync(GLMContext ctx, GLsync sync, GLbitfield flags, GLuint64 timeout)
{
    if (!mglSyncOperationEnter(ctx)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    /* GL 4.6 §5.3: flags must be zero. */
    if (flags != 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        mglSyncOperationLeave(ctx);
        return;
    }

    if (timeout != GL_TIMEOUT_IGNORED) {
        ERROR_RETURN(GL_INVALID_VALUE);
        mglSyncOperationLeave(ctx);
        return;
    }

    Sync *owned = mglAcquireSync(ctx, sync);
    if (!owned)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        mglSyncOperationLeave(ctx);
        return;
    }

    /* The backend blocks on the retained command
     * buffer, satisfying the GL spec requirement that glWaitSync block until the
     * fence's insertion-point-prior commands have completed on the GPU.
     *
     * MGL_SYNC_STRICT: fence wait already performs conservative sync via
     * the same completion wait; no extra strict branch is needed. */
    mglRendererWaitForSync(ctx, owned);

    mglReleaseSyncReference(ctx, owned);
    mglSyncOperationLeave(ctx);
}

void mglGetSynciv(GLMContext ctx, GLsync sync, GLenum pname, GLsizei count, GLsizei *length, GLint *values)
{
    if (!mglSyncOperationEnter(ctx)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        if (length) *length = 0;
        return;
    }
    // CRITICAL FIX: count is the number of elements the caller allocated in values.
    // Per OpenGL spec, only one value is returned per pname. length is an OUTPUT parameter.
    if (!count || count < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        if (length) *length = 0;
        mglSyncOperationLeave(ctx);
        return;
    }
    if (!values) {
        ERROR_RETURN(GL_INVALID_VALUE);
        if (length) *length = 0;
        mglSyncOperationLeave(ctx);
        return;
    }

    Sync *owned = mglAcquireSync(ctx, sync);
    if (!owned)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        if (length) *length = 0;
        mglSyncOperationLeave(ctx);
        return;
    }

    // Only write one value per pname per the OpenGL spec
    switch(pname)
    {
        case GL_OBJECT_TYPE:
            *values = GL_SYNC_FENCE;
            break;

        case GL_SYNC_STATUS:
            *values = mglRendererGetSyncStatus(ctx, owned);
            break;

        case GL_SYNC_CONDITION:
            *values = GL_SYNC_GPU_COMMANDS_COMPLETE;
            break;

        case GL_SYNC_FLAGS:
            *values = 0;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            if (length) *length = 0;
            mglReleaseSyncReference(ctx, owned);
            mglSyncOperationLeave(ctx);
            return;
    }

    if (length) *length = 1;

    mglReleaseSyncReference(ctx, owned);
    mglSyncOperationLeave(ctx);
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
    mglRendererFlush(ctx, true);
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
     * again by any subsequent CPU-side texture upload (glTexSubImage/glTexImage).
     *
     * TEXTURE_BUFFER has no mip faces; imageStores write a texture2d copy of the
     * buffer. Barriers that later consume that buffer as draw-indirect commands,
     * vertex/element arrays, or client/GPU buffer reads therefore also require
     * copying those texels back into the attached Buffer. */
    GLbitfield image_relevant_bits =
        GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
        GL_TEXTURE_UPDATE_BARRIER_BIT |
        GL_BUFFER_UPDATE_BARRIER_BIT |
        GL_COMMAND_BARRIER_BIT |
        GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT |
        GL_ELEMENT_ARRAY_BARRIER_BIT |
        GL_TEXTURE_FETCH_BARRIER_BIT;
    if (barriers == GL_ALL_BARRIER_BITS || (barriers & image_relevant_bits))
    {
        GLuint max_units = STATE(var).max_image_units;
        for (GLuint i = 0; i < max_units && i < TEXTURE_UNITS; i++) {
            ImageUnit *iu = &STATE(image_units)[i];
            Texture *tex = iu->tex;
            if (!tex) {
                continue;
            }
            if (tex->target == GL_TEXTURE_BUFFER) {
                tex->metal_data_authoritative = GL_TRUE;
                mglRendererSyncTextureBufferFromImage(ctx, tex);
                continue;
            }
            if (!tex->faces[0].levels) {
                continue;
            }
            if (iu->level >= (GLint)tex->num_levels) {
                continue;
            }
            tex->metal_data_authoritative = GL_TRUE;
            tex->faces[0].levels[iu->level].metal_data_authoritative = GL_TRUE;
            mglRendererFlushImageUnitSlice(ctx, i);
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
