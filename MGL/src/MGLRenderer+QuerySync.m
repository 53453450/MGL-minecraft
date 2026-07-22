// MGLRenderer+QuerySync.m
// Occlusion/timer query and GL sync (fence) methods extracted from MGLRenderer.m

#import "MGLRenderer_Private.h"
#import "MGLRenderer+QuerySync_Private.h"

@implementation MGLRenderer (QuerySync)
#pragma mark Metal visibility result (GL occlusion query)

/* Called from glBeginQuery for GL_SAMPLES_PASSED / GL_ANY_SAMPLES_PASSED.
 * Ends the current render encoder so the next draw creates a fresh encoder
 * with the visibility result buffer attached and boolean mode enabled. */
-(void)mtlBeginSampleQuery:(GLMContext)glm_ctx
{
    (void)glm_ctx;
    if (![_queryManager beginSampleQueryWithDevice:_device]) {
        NSLog(@"MGL ERROR: Failed to allocate visibility result buffer");
        return;
    }
    /* End the current render encoder so the next draw creates a new one with
     * the visibility result buffer attached. */
    [self endRenderEncoding];
}

/* Called from glEndQuery for GL_SAMPLES_PASSED / GL_ANY_SAMPLES_PASSED.
 * Ends the current render encoder (flushing the visibility result to the
 * buffer), waits for the GPU to complete, and reads back the result.
 * Returns 0 if no samples passed, or non-zero if any samples passed. */
-(GLuint64)mtlEndSampleQuery:(GLMContext)glm_ctx
{
    (void)glm_ctx;

    /* End the current render encoder so the GPU writes the visibility result
     * to the buffer. Do this before clearing the manager's active flag so that
     * any code triggered by endRenderEncoding sees a consistent state
     * (query still active while the visibility-mode encoder is being torn
     * down). */
    [self endRenderEncoding];

    /* Now that the encoder with visibility mode has been ended, clear the
     * flag so subsequent render encoders are created without it. */
    [_queryManager endSampleQuery];

    /* Flush and wait for the GPU to complete so the buffer is readable. */
    if ([_queryManager hasSampleQueryResultBuffer]) {
        [self flushCommandBuffer:YES];
    }
    return [_queryManager sampleQueryResult];
}

#pragma mark Metal GPU timer query (GL_TIME_ELAPSED / GL_TIMESTAMP)

/* Called from glBeginQuery(GL_TIME_ELAPSED).  Flushes all pending GPU
 * work so the GPU is idle, then samples the GPU timestamp.  The timestamp
 * is stored by MGLQueryManager and used by mtlEndTimerQuery to compute
 * the elapsed GPU time.
 *
 * The flush ensures the begin timestamp is taken before any commands
 * submitted between begin/end reach the GPU.  This gives an accurate
 * measurement of GPU execution time for the bracketed commands. */
-(void)mtlBeginTimerQuery:(GLMContext)glm_ctx
{
    (void)glm_ctx;
    /* Flush and wait for all pending GPU work to complete so the GPU
     * is idle when we sample the begin timestamp. */
    [self flushCommandBuffer:YES];
    [_queryManager beginTimerQueryWithDevice:_device];
}

/* Called from glEndQuery(GL_TIME_ELAPSED).  Flushes all pending GPU work
 * (including any draws submitted between begin and end), waits for the
 * GPU to complete, then samples the end timestamp.  Returns the elapsed
 * GPU nanoseconds (end - begin). */
-(GLuint64)mtlEndTimerQuery:(GLMContext)glm_ctx
{
    (void)glm_ctx;
    /* Flush and wait for all GPU work submitted between begin and end. */
    [self flushCommandBuffer:YES];
    return [_queryManager endTimerQueryWithDevice:_device];
}

/* Returns the current GPU timestamp in nanoseconds.  Used by
 * glQueryCounter(GL_TIMESTAMP).  Per the GL spec, the timestamp must be
 * recorded after all previously issued commands have completed, so we
 * flush the pending command buffer and wait for completion before
 * sampling the GPU timestamp. */
-(GLuint64)mtlGetGPUTimestamp:(GLMContext)glm_ctx
{
    (void)glm_ctx;
    [self flushCommandBuffer:YES];
    return [_queryManager gpuTimestampWithDevice:_device];
}

#pragma mark C interface to mtlGetSync
/*
 * mtlGetSync:sync: — fence insertion point command capture (CB-wait mechanism)
 *
 * Trigger: called when glFenceSync creates a fence sync object.
 * Implementation contract:
 *   1. processGLState:false flushes pending draws to the current command buffer;
 *   2. endRenderEncoding closes any open render encoder;
 *   3. retain the current command buffer into sync->mtl_command_buffer (this CB
 *      contains exactly all GL commands prior to the fence insertion point),
 *      then commit it to the GPU via commitCommandBufferWithAGXRecovery:;
 *   4. newCommandBuffer creates a fresh CB for subsequent GL command encoding
 *      (must not encode into the already-committed fence CB).
 * Guarantee: the fence reflects GPU completion status of commands before the
 *            insertion point. mtlWaitForSync blocks until GPU completion by
 *            calling waitUntilCompleted on that CB — no longer a no-op.
 * Degradation: if no submittable CB exists (e.g. no draw before the fence, or
 *              CB already finalized/errored), mtl_command_buffer is NULL and
 *              the fence is immediately in the signaled state.
 * kMGLDisableSharedEventSync only gates the legacy shared-event path (event
 * creation + SyncList); CB-wait is always active and is the real wait mechanism.
 */
-(void) mtlGetSync:(GLMContext) glm_ctx sync: (Sync *)sync
{
    METAL_LOCK();
    @try {
    // SAFETY: Check Metal objects before processing
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or queue is NULL in mtlGetSync");
        if (sync) {
            sync->mtl_event = NULL;
            sync->mtl_command_buffer = NULL;
        }
        return;
    }

    if (!sync) {
        NSLog(@"MGL ERROR: mtlGetSync - sync object is NULL");
        return;
    }

    // Flush pending draws into the current command buffer so the CB captures
    // all GL commands issued before the fence insertion point.
    if (![self processGLState: false]) {
        NSLog(@"MGL WARNING: processGLState failed in mtlGetSync");
    }

    // End any open render encoder so the command buffer can be committed.
    [self endRenderEncoding];

    // CB-wait mechanism: retain the current command buffer (which now contains
    // exactly the commands issued before the fence), commit it to the GPU, and
    // create a fresh command buffer for subsequent GL commands. The retained CB
    // is stored in sync->mtl_command_buffer so mtlWaitForSync can block on its
    // completion via waitUntilCompleted. This runs regardless of
    // kMGLDisableSharedEventSync (which only gates the legacy shared-event path).
    if (_renderPassManager.state->currentCommandBuffer &&
        _renderPassManager.state->currentCommandBuffer.status == MTLCommandBufferStatusNotEnqueued &&
        !_renderPassManager.state->currentCommandBuffer.error) {
        id<MTLCommandBuffer> cbToCommit =
            [_renderPassManager detachCurrentCommandBufferForSubmission];
        sync->mtl_command_buffer = (void *)CFBridgingRetain(cbToCommit);

        @try {
            [self commitCommandBufferWithAGXRecovery:cbToCommit];
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Failed to commit fence command buffer: %@", exception);
            [self recordGPUError];
        }
    } else {
        // No in-flight command buffer (e.g. no draws issued before the fence)
        // or the buffer is already finalized/errored: the fence is immediately
        // signaled.
        sync->mtl_command_buffer = NULL;
    }

    // Always provide a fresh command buffer for subsequent GL commands so they
    // are not encoded into the already-committed fence command buffer.
    [self newCommandBuffer];

    // Legacy shared-event path. Gated by kMGLDisableSharedEventSync; when
    // disabled, mtl_event stays NULL and the CB-wait above is the sole wait
    // mechanism. The event/SyncList code below only runs when shared-event
    // sync is explicitly enabled.
    if (kMGLDisableSharedEventSync) {
        sync->mtl_event = NULL;
        [_renderPassManager clearPendingEvent];
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: mtlGetSync captured CB=%p (shared event sync disabled)",
                  sync->mtl_command_buffer);
        }
        return;
    }

    id<MTLEvent> pendingEvent = nil;
    @try {
        pendingEvent = [_renderPassManager preparePendingEventWithDevice:_device
                                                                syncName:sync->name];
        if (!pendingEvent) {
            NSLog(@"MGL ERROR: Failed to create Metal event");
            return;
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating Metal event: %@", exception);
        return;
    }

    sync->mtl_event = (void *)CFBridgingRetain(pendingEvent);

    // Phase 3: Lock the sync list for the write path — newCommandBuffer
    // acquires the same lock on the read/clear path, so without this lock
    // mtlGetSync: would race against newCommandBuffer on multi-thread use.
    // Uses _syncListLock (independent from _metalStateLock) to avoid
    // deadlock if mtlGetSync: is ever called from within a Locked method.
    SYNC_LOCK();
    BOOL appended = [_renderPassManager appendSyncToCurrentCommandBuffer:sync];
    SYNC_UNLOCK();
    if (!appended) {
        return;
    }
    } @finally {
        METAL_UNLOCK();
    }
}

#pragma mark C interface to mtlWaitForSync
/*
 * mtlWaitForSync:sync: — fence blocking wait (CB-wait mechanism)
 *
 * Trigger: called on glClientWaitSync / glWaitSync / glDeleteSync paths.
 * Implementation contract: calls waitUntilCompleted on the command buffer
 *           captured and retained by mtlGetSync, blocking the current thread
 *           until the GPU finishes that CB (i.e. all commands before the fence
 *           insertion point). On completion, CFBridgingRelease drops the CB
 *           reference and clears sync->mtl_command_buffer.
 *           No longer a no-op (the old implementation only released mtl_event
 *           and returned immediately, violating GL semantics).
 * Degradation: if mtl_command_buffer is NULL (fence already completed or no CB
 *              at creation time), returns immediately. Also cleans up any
 *              legacy mtl_event reference if present.
 * kMGLDisableSharedEventSync does not gate this path — CB-wait is always active.
 */
-(void) mtlWaitForSync:(GLMContext) glm_ctx sync: (Sync *)sync
{
    // CRITICAL SAFETY: Validate sync object before processing
    if (!sync) {
        NSLog(@"MGL ERROR: mtlWaitForSync - sync object is NULL");
        return;
    }

    // CB-wait path: block until the command buffer captured at fence insertion
    // completes on the GPU. This is the real wait mechanism and runs regardless
    // of kMGLDisableSharedEventSync.
    if (sync->mtl_command_buffer) {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)sync->mtl_command_buffer;
        @try {
            if (cb.status != MTLCommandBufferStatusCompleted) {
                [cb waitUntilCompleted];
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Exception waiting on fence command buffer: %@", exception);
        }
        mglSafeReleaseMetalObj((void **)&sync->mtl_command_buffer);
    }

    // Legacy shared-event cleanup (harmless if mtl_event is NULL, which is the
    // case when kMGLDisableSharedEventSync is enabled).
    if (sync->mtl_event) {
        @try {
            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL INFO: Releasing Metal sync event");
            }
            mglSafeReleaseMetalObj((void **)&sync->mtl_event);
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Exception releasing sync event: %@", exception);
        }
    }

    if (kMGLDisableSharedEventSync && kMGLVerboseFrameLoopLogs) {
        NSLog(@"MGL INFO: mtlWaitForSync completed via CB-wait (shared event sync disabled)");
    }
}

#pragma mark C interface to mtlGetSyncStatus
/*
 * mtlGetSyncStatus:sync: — fence non-blocking status query
 *
 * Trigger: called by mglGetSynciv(GL_SYNC_STATUS) / mglClientWaitSync polling.
 * Guarantee: returns GL_SIGNALED if and only if the associated CB has completed
 *            (status == Completed) or there is no associated CB
 *            (mtl_command_buffer == NULL); otherwise returns GL_UNSIGNALED.
 *            Does not block.
 */
-(GLenum) mtlGetSyncStatus:(GLMContext) glm_ctx sync: (Sync *)sync
{
    if (!sync || !sync->mtl_command_buffer) {
        return GL_SIGNALED;
    }

    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)sync->mtl_command_buffer;
    @try {
        if (cb.status == MTLCommandBufferStatusCompleted) {
            return GL_SIGNALED;
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception querying fence command buffer status: %@", exception);
    }

    return GL_UNSIGNALED;
}

#pragma mark C interface to mtlReleaseSync
/*
 * mtlReleaseSync:sync: — fence resource non-blocking release
 *
 * Trigger: called by mglDeleteSync.
 * Guarantee: releases the retained command buffer and legacy event references
 *            associated with the fence. Does not wait for GPU completion (Metal
 *            internally retains in-flight CBs, so releasing our reference is
 *            safe and does not interrupt GPU execution). Prevents leaks, non-blocking.
 */
-(void) mtlReleaseSync:(GLMContext) glm_ctx sync: (Sync *)sync
{
    if (!sync) {
        return;
    }

    if (sync->mtl_command_buffer) {
        @try {
            mglSafeReleaseMetalObj((void **)&sync->mtl_command_buffer);
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Exception releasing fence command buffer: %@", exception);
        }
    }

    if (sync->mtl_event) {
        @try {
            mglSafeReleaseMetalObj((void **)&sync->mtl_event);
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Exception releasing sync event: %@", exception);
        }
    }
}

@end
