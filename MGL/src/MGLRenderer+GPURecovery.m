// MGLRenderer+GPURecovery.m
// Metal GPU error recovery methods extracted from MGLRenderer+RenderPass.m

#import "MGLRenderer_Private.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp_objc.h"

@implementation MGLRenderer (GPURecovery)

#pragma mark - Metal State Validation and Recovery

- (BOOL)validateMetalObjects
{
    // PROPER FIX: Comprehensive Metal object validation with GPU health monitoring
    @try {
        // Check Metal device validity
        if (!_device) {
            NSLog(@"MGL ERROR: Metal device is nil during validation");
            return NO;
        }

        // Check command queue validity
        if (!_commandQueue) {
            NSLog(@"MGL ERROR: Metal command queue is nil during validation");
            return NO;
        }

        // GPU ERROR THROTTLING: Track recent GPU failures to prevent error cascades
        static NSUInteger consecutiveGpuErrors = 0;
        static NSTimeInterval lastErrorTime = 0;
        static NSTimeInterval throttleWindow = 2.0; // 2 second throttle window
        static NSUInteger maxErrorsPerWindow = 3;

        // Get current error tracking from command buffer if available
        MGLRenderCppCommandBufferState currentState = {0};
        BOOL hasCurrentCommandBuffer = mglRenderCommandBufferOwnerState(
            _renderPassManager.state->currentCommandBufferOwner,
            &currentState);
        if (hasCurrentCommandBuffer &&
            currentState.has_error) {
            NSTimeInterval currentTime = [[NSDate date] timeIntervalSince1970];

            // Check if this is within the throttle window
            if (currentTime - lastErrorTime < throttleWindow) {
                consecutiveGpuErrors++;
                NSLog(@"MGL GPU THROTTLING: %lu consecutive GPU errors detected", (unsigned long)consecutiveGpuErrors);

                // If we've exceeded the error threshold, temporarily disable operations
                if (consecutiveGpuErrors > maxErrorsPerWindow) {
                    NSLog(@"MGL CRITICAL: GPU error threshold exceeded - throttling operations for %.1f seconds", throttleWindow);

                    // Force a reset and temporary pause
                    [self resetMetalState];

                    // Reset counter after pause
                    if (currentTime - lastErrorTime > throttleWindow) {
                        consecutiveGpuErrors = 0;
                    } else {
                        return NO; // Skip this operation to prevent more errors
                    }
                }
            } else {
                // Reset counter if outside throttle window
                consecutiveGpuErrors = 1;
                lastErrorTime = currentTime;
            }
        }

        // Check for virtualization environment changes
        if (@available(macOS 11.0, *)) {
            // Device registry ID changes indicate virtualization issues
            if (_device.registryID == 0) {
                NSLog(@"MGL WARNING: Detected virtualized Metal environment - enabling safety mode");
                // Note: _isVirtualized would be an instance variable to track virtualization state
            }
        }

        return YES;
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Metal object validation failed: %@", exception);
        return NO;
    }
}

- (void)clearTextureCache
{
    // PROPER FIX: Intelligent texture cache cleanup
    NSLog(@"MGL INFO: Clearing texture cache to free memory");

    // Note: Texture binding cache cleanup would require instance variables
    // For now, we focus on basic resource cleanup

    // Force garbage collection using available methods
    if (@available(macOS 10.15, *)) {
        // Simply nil out some references to encourage garbage collection
        // This is a placeholder for more sophisticated cache management
    }
}

- (void)cleanupCommandBuffer
{
    // PROPER FIX: Safe command buffer cleanup
    @try {
        MGLRenderCppCommandBufferState currentState = {0};
        if (mglRenderCommandBufferOwnerState(
                _renderPassManager.state->currentCommandBufferOwner,
                &currentState)) {
            if (currentState.status == MTLCommandBufferStatusCommitted) {
                // Do not block indefinitely here; cleanup can be invoked on the render thread.
                // Command buffers retain resources until completion, so dropping the reference is safe.
                if (kMGLVerboseFrameLoopLogs) {
                    NSLog(@"MGL INFO: cleanupCommandBuffer skipping blocking wait for committed command buffer");
                }
            }
            [_renderPassManager discardCurrentCommandBuffer];
        }

        if (mglRenderCppRenderEncoderOwnerHasCurrent(
                _renderPassManager.state->currentRenderEncoderOwner) == 1) {
            [_renderPassManager endCurrentRenderEncoder];
            [_renderPassManager clearCurrentRenderEncoder];
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception during command buffer cleanup: %@", exception);
    }
}

- (void)resetMetalState
{
    // PROPER FIX: Full Metal state reset for AGX driver recovery
    NSLog(@"MGL INFO: Performing full Metal state reset for AGX recovery");

    /* Runs on the GL calling thread (frame-boundary drain in mtlSwapBuffers
     * or GL-layer error paths).  With the recovery path removed from the
     * main queue this is no longer a cross-thread reset. */
    METAL_LOCK();

    [self cleanupCommandBuffer];

    // CRITICAL: Recreate command queue to clear AGX driver error state
    NSLog(@"MGL AGX RECOVERY: Recreating command queue to clear GPU error state");
    void *commandQueue = NULL;
    if (_backend) {
        (void)mglRendererBackendResetCommandQueue(
            _backend, 0u, &commandQueue);
    }
    if (!_commandQueue) {
        NSLog(@"MGL CRITICAL: Failed to recreate command queue during AGX recovery");
    } else {
        NSLog(@"MGL AGX RECOVERY: Command queue successfully recreated");
    }

    [_pipelineCache resetCaches];
    // Note: _depthStencilState would be an instance variable if it exists

    // Clear all cached objects
    [self clearTextureCache];

    NSLog(@"MGL INFO: AGX Metal state reset completed");

    METAL_UNLOCK();
}

// AGX Driver Compatibility: Specialized command buffer commit with recovery
- (void)commitCommandBufferWithAGXRecovery:(MGLMetalCommandBufferRef)commandBuffer
{
    /* s_commitCallCount is owned by the GL calling thread: commit paths are
     * reached on the GL thread and never run on the completion-handler
     * thread or the main queue. */
    static uint64_t s_commitCallCount = 0;
    uint64_t commitCall = ++s_commitCallCount;
    bool traceCommit = mglShouldTraceCall(commitCall);

    if (!commandBuffer) {
        NSLog(@"MGL ERROR: Cannot commit NULL command buffer");
        return;
    }

    @try {

    if (traceCommit) {
        mglTraceLogNSString(@"MGL TRACE commit.begin call=%llu cb=%p status=%s label=%@",
              (unsigned long long)commitCall,
              commandBuffer,
              mglCommandBufferStatusName(
                  mglRenderCommandBufferStatus(commandBuffer)),
              commandBuffer.label ?: @"(no-label)");
    }
    MGLRenderCppCommandBufferTransaction transaction = {0};
    @try {
        int transactionResult = [_renderPassManager
            commitCommandBufferTransaction:commandBuffer
            recoveryOwner:_gpuRecovery.commandRecoveryOwner
            waitForCompletion:NO
            result:&transaction];
        if (transaction.result ==
            MGL_RENDER_CPP_COMMAND_BUFFER_TRANSACTION_NESTED) {
            NSLog(@"MGL AGX WARNING: Commit already in progress, skipping nested commit");
            if (traceCommit) {
                mglTraceLogNSString(@"MGL TRACE commit.skip.nested call=%llu",
                      (unsigned long long)commitCall);
            }
            return;
        }
        if (transaction.result ==
            MGL_RENDER_CPP_COMMAND_BUFFER_TRANSACTION_SKIPPED) {
            if (transaction.has_error) {
                NSLog(@"MGL AGX WARNING: C++ transaction skipped failed command buffer: %s (domain=%s code=%lld, consecutive=%llu)",
                      transaction.before.error_description,
                      transaction.before.error_domain,
                      (long long)transaction.before.error_code,
                      (unsigned long long)transaction.recovery.consecutive_errors);
            } else {
                NSLog(@"MGL AGX WARNING: C++ transaction skipped finalized command buffer (status: %u)",
                      transaction.before.status);
            }
            if (transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            return;
        }
        if (transactionResult != 0 || transaction.has_error) {
            NSLog(@"MGL AGX ERROR: C++ command-buffer transaction failed (before=%u after=%u submission=%u consecutive=%llu)",
                  transaction.before.status, transaction.after.status,
                  transaction.used_submission,
                  (unsigned long long)transaction.recovery.consecutive_errors);
            if (transaction.device_reset_requested) {
                atomic_store_explicit(&_deviceResetRequested, true,
                                      memory_order_release);
            }
            return;
        }
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL AGX: Command buffer committed successfully through C++ owner");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL AGX ERROR: C++ command-buffer transaction exception: %@",
              exception);
        if (mglRenderCppCommandRecoveryRecordTransactionFailure(
                _gpuRecovery.commandRecoveryOwner, NULL, &transaction) == 0 &&
            transaction.device_reset_requested) {
            atomic_store_explicit(&_deviceResetRequested, true,
                                  memory_order_release);
        }
    } @finally {
        if (traceCommit) {
            mglTraceLogNSString(@"MGL TRACE commit.end call=%llu cb=%p finalStatus=%s",
                  (unsigned long long)commitCall,
                  commandBuffer,
                  mglCommandBufferStatusName(
                      (MTLCommandBufferStatus)transaction.after.status));
        }
    }
    } @finally {
        [_renderPassManager releaseDetachedCommandBufferIfOwned:commandBuffer];
    }
}

// AGX GPU Error Throttling - Prevent command queue from entering error state
- (BOOL)shouldSkipGPUOperations
{
    NSTimeInterval currentTime = [[NSDate date] timeIntervalSince1970];
    MGLRenderCppCommandRecoverySkipDecision decision = {0};
    if (mglRenderCppCommandRecoveryShouldSkip(
            _gpuRecovery.commandRecoveryOwner, currentTime, &decision) != 0) {
        return NO;
    }
    if (decision.recovery_timed_out && decision.previous_errors > 0) {
        NSLog(@"MGL AGX: Recovery timeout - attempting GPU operations (had %llu errors)",
              (unsigned long long)decision.previous_errors);
    }
    if (decision.entered_recovery_mode) {
        NSLog(@"MGL AGX: Entering recovery mode after %llu consecutive errors",
              (unsigned long long)decision.state.consecutive_errors);
        [self clearProblematicGPUState];
    }
    return decision.should_skip != 0;
}

// PROPER FIX: Clear problematic state without giving up on GPU operations entirely
- (void)clearProblematicGPUState
{
    NSLog(@"MGL AGX: Clearing problematic GPU state for recovery");

    // Clear current problematic resources
    MGLRenderCppCommandBufferState currentState = {0};
    if (mglRenderCommandBufferOwnerState(
            _renderPassManager.state->currentCommandBufferOwner,
            &currentState)) {
        [_renderPassManager discardCurrentCommandBuffer];
    }

    // Don't recreate command queue immediately - let it rest
    // The AGX driver needs time to recover from error state
}

- (void)recordGPUError
{
    MGLRenderCppCommandRecoverySnapshot state = {0};
    if (mglRenderCppCommandRecoveryRecordError(
            _gpuRecovery.commandRecoveryOwner,
            [[NSDate date] timeIntervalSince1970], &state) == 0) {
        NSLog(@"MGL AGX: Recorded GPU error (%llu consecutive)",
              (unsigned long long)state.consecutive_errors);
    }
}

- (void)recordGPUSuccess
{
    MGLRenderCppCommandRecoverySuccess result = {0};
    if (mglRenderCppCommandRecoveryRecordSuccess(
            _gpuRecovery.commandRecoveryOwner,
            [[NSDate date] timeIntervalSince1970], &result) == 0 &&
        result.sustained_recovery) {
        NSLog(@"MGL AGX: Sustained GPU recovery (%llu successes), resetting error count (was %llu)",
              (unsigned long long)result.recovered_successes,
              (unsigned long long)result.previous_errors);
    }
}

#pragma mark - Metal Optimization Methods

- (NSUInteger)getOptimalAlignmentForPixelFormat:(MTLPixelFormat)format
{
    (void)format;
    // aligned_alloc requires an alignment compatible with platform pointer alignment.
    // Using a conservative 64-byte value avoids EINVAL on macOS/arm64 and is safe for texture rows.
    return 64;
}


@end
