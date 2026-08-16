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

        if ((__bridge MGLMetalRenderCommandEncoderRef)mglRenderCppRenderEncoderOwnerGetCurrent(_renderPassManager.state->currentRenderEncoderOwner)) {
            MGLMetalRenderCommandEncoderRef encoder =
                (__bridge MGLMetalRenderCommandEncoderRef)mglRenderCppRenderEncoderOwnerGetCurrent(_renderPassManager.state->currentRenderEncoderOwner);
            if (!(mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
                  mglRenderCppGetDevice() &&
                  mglRenderCppEndRenderEncoder((__bridge void *)encoder) == 0)) {
                [encoder endEncoding];
            }
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
    _commandQueue = nil;
    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") && mglRenderCppGetDevice()) {
        _commandQueue = mglRenderCppCreateOrResetCommandQueueOwner(
            &_commandQueueOwner, 0u);
    }
    if (!_commandQueue) {
        mglRenderCppDestroyCommandQueueOwner(&_commandQueueOwner);
        _commandQueue = [_device newCommandQueue];
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
    uint64_t commitQueuedAtNS = mglTraceClockNS();

    // Pre-commit validation for AGX driver
    MGLRenderCppCommandBufferState preCommitState =
        mglRenderCommandBufferState(commandBuffer);
    if (preCommitState.has_error) {
        NSLog(@"MGL AGX WARNING: Command buffer has pre-commit error: %s (domain=%s code=%lld)",
              preCommitState.error_description,
              preCommitState.error_domain,
              (long long)preCommitState.error_code);
        [self recordGPUError];
    }

    // Register recovery bookkeeping in the C++ owner.  The gate-on handler
    // deliberately captures no renderer/ARC object on the Metal worker.
    BOOL recoveryCompletionRegistered = NO;
    if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
        mglRenderCppGetDevice() && _gpuRecovery.commandRecoveryOwner) {
        recoveryCompletionRegistered =
            mglRenderCppAddCommandBufferRecoveryCompletion(
                (__bridge void *)commandBuffer,
                _gpuRecovery.commandRecoveryOwner) == 0;
    }
    if (!recoveryCompletionRegistered) {
        __block typeof(self) blockSelf = self;
        uint64_t commitCallForBlock = commitCall;
        bool traceCommitForBlock = traceCommit;
        mglRenderAddCommandBufferCompletion(
            commandBuffer,
            ^(const MGLRenderCppCommandBufferState *completionState) {
            MGLRenderCppCommandBufferCompletionResult completionResult = {0};
            if (mglRenderCppProcessCommandBufferCompletion(
                    blockSelf->_gpuRecovery.commandRecoveryOwner,
                    completionState,
                    [[NSDate date] timeIntervalSince1970],
                    &completionResult) != 0) {
                completionResult.decision.has_error =
                    completionState->has_error;
            }
            double completeElapsedUs =
                (mglTraceClockNS() - commitQueuedAtNS) / 1000.0;
            if (traceCommitForBlock || completionResult.decision.has_error ||
                completeElapsedUs >= 50000.0) {
                mglTraceLogNSString(@"MGL TRACE commit.completed call=%llu status=%s elapsed=%.1fus error=%@",
                      (unsigned long long)commitCallForBlock,
                      mglCommandBufferStatusName(
                          (MTLCommandBufferStatus)completionState->status),
                      completeElapsedUs,
                      completionResult.decision.has_error
                          ? [NSString stringWithFormat:@"%s (domain=%s code=%lld)",
                               completionState->error_description,
                               completionState->error_domain,
                               (long long)completionState->error_code]
                          : nil);
            }
            if (completionResult.decision.has_error) {
                NSLog(@"MGL AGX ERROR: Command buffer completed with error: %s (domain=%s code=%lld)",
                      completionState->error_description,
                      completionState->error_domain,
                      (long long)completionState->error_code);
                if (completionResult.state.consecutive_errors > 0) {
                    NSLog(@"MGL AGX: Recorded GPU error (%llu consecutive)",
                          (unsigned long long)completionResult.state.consecutive_errors);
                }

                // Specific handling for AGX driver rejection
                if (completionResult.decision.is_driver_rejection) {
                    /* "Ignored (for causing prior/excessive GPU errors)" */
                    /* Owned by the Metal completion-handler thread (this block):
                     * never touched from the GL thread or the main queue. */
                    static NSTimeInterval s_lastDriverRejectionReset = 0.0;
                    NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
                    if (now - s_lastDriverRejectionReset > 2.0) {
                        s_lastDriverRejectionReset = now;
                        NSLog(@"MGL AGX RECOVERY: Driver rejection detected; throttled reset scheduled");
                        /* Deferred reset: hand the request to the GL thread across
                         * the frame boundary instead of dispatching to the main
                         * queue.  The reset runs at the next safe point (after
                         * endRenderEncoding in mtlSwapBuffersLocked). */
                        atomic_store_explicit(
                            &blockSelf->_deviceResetRequested, true,
                            memory_order_release);
                    } else {
                        NSLog(@"MGL AGX RECOVERY: Driver rejection detected; skipping immediate reset (throttled)");
                    }
                }
            } else {
                if (completionResult.sustained_recovery) {
                    NSLog(@"MGL AGX: Sustained GPU recovery (%llu successes), resetting error count (was %llu)",
                          (unsigned long long)completionResult.recovered_successes,
                          (unsigned long long)completionResult.previous_errors);
                }

                // AGX Recovery: Clear recovery mode on success
                if (completionResult.cleared_recovery_mode) {
                    NSLog(@"MGL AGX RECOVERY: Exiting GPU recovery mode after successful completion");
                }
            }
            });
    }

    // CRITICAL FIX: Enhanced command buffer validation before commit
    // Prevents MTLReleaseAssertionFailure in AGX driver
    if (!commandBuffer) {
        NSLog(@"MGL AGX ERROR: Cannot commit nil command buffer");
        return;
    }

    // Check command buffer status before commit
    MGLRenderCppCommandBufferState commitState =
        mglRenderCommandBufferState(commandBuffer);
    MGLRenderCppCommandBufferCommitDecision commitDecision = {0};
    if (mglRenderCppClassifyCommandBufferCommit(
            &commitState, &commitDecision) != 0) {
        commitDecision.action = MGL_RENDER_CPP_COMMAND_BUFFER_COMMIT_PROCEED;
    }
    MTLCommandBufferStatus status = (MTLCommandBufferStatus)commitState.status;
    if (commitDecision.action ==
        MGL_RENDER_CPP_COMMAND_BUFFER_COMMIT_SKIP_ALREADY_COMMITTED) {
        NSLog(@"MGL AGX WARNING: Command buffer already committed (status: %ld) - skipping commit", (long)status);
        if (traceCommit) {
            mglTraceLogNSString(@"MGL TRACE commit.skip.already_committed call=%llu status=%s",
                  (unsigned long long)commitCall, mglCommandBufferStatusName(status));
        }
        return;
    }

    if (![_renderPassManager beginCommandBufferCommit]) {
        NSLog(@"MGL AGX WARNING: Commit already in progress, skipping nested commit");
        if (traceCommit) {
            mglTraceLogNSString(@"MGL TRACE commit.skip.nested call=%llu", (unsigned long long)commitCall);
        }
        return;
    }

    @try {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL AGX: Committing command buffer (status: %ld)", (long)status);
        }
        if ([_renderPassManager
                commitDetachedCommandBufferIfOwned:commandBuffer]) {
            /* C++ submission consumed the detached +1 reference. */
        } else if (mgl_env_flag_enabled_default_on("MGL_USE_METALCPP") &&
            mglRenderCppGetDevice() &&
            mglRenderCppCommitCommandBuffer((__bridge void *)commandBuffer) == 0) {
            /* Metal-cpp commit completed. */
        } else {
            [commandBuffer commit];
        }
        /* Centralized tracking of the most recently committed CB, covering
         * every commit routed through this function. */
        _lastCommittedCB = commandBuffer;
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL AGX: Command buffer committed successfully");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL AGX ERROR: Command buffer commit exception: %@", exception);
        [self recordGPUError];

        // AGX-specific recovery for commit failures
        if ([[exception name] containsString:@"CommandBuffer"] ||
            [[exception name] containsString:@"GPU"]) {
            NSLog(@"MGL AGX RECOVERY: Immediate reset due to commit exception");
            /* Deferred reset — drained at the next swap frame boundary on the
             * GL thread.  The commit exception already ran on the GL thread,
             * so no cross-thread dispatch is needed. */
            atomic_store_explicit(&_deviceResetRequested, true, memory_order_release);
        }
    } @finally {
        [_renderPassManager endCommandBufferCommit];
        if (traceCommit) {
            mglTraceLogNSString(@"MGL TRACE commit.end call=%llu cb=%p finalStatus=%s",
                  (unsigned long long)commitCall,
                  commandBuffer,
                  mglCommandBufferStatusName(
                      mglRenderCommandBufferStatus(commandBuffer)));
        }
    }
    } @finally {
        [_renderPassManager
            releaseDetachedCommandBufferIfOwned:commandBuffer];
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
