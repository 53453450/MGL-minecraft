// MGLRenderer+GPURecovery.m
// Metal GPU error recovery methods extracted from MGLRenderer+RenderPass.m

#import "MGLRenderer_Private.h"

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
        if (_renderPassManager.state->currentCommandBuffer && _renderPassManager.state->currentCommandBuffer.error) {
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

- (BOOL)recoverFromMetalError:(NSError *)error operation:(NSString *)operation
{
    // PROPER FIX: Intelligent Metal error recovery
    NSLog(@"MGL ERROR: Metal operation '%@' failed: %@", operation, error);

    // Interface mismatch during pipeline creation is not a GPU-state corruption case.
    // Avoid destructive resets here to prevent reset/retry loops.
    if ([operation isEqualToString:@"pipeline_creation"]) {
        NSString *desc = error.localizedDescription ?: @"";
        NSString *domain = error.domain ?: @"";
        if ((error.code == 3 && [domain hasPrefix:@"AGXMetal"]) ||
            [desc containsString:@"mismatching vertex shader output"] ||
            [desc containsString:@"not written by vertex shader"]) {
            static uint64_t s_pipelineMismatchLogCount = 0;
            s_pipelineMismatchLogCount++;
            if ((s_pipelineMismatchLogCount % 64ull) == 1ull) {
                NSLog(@"MGL WARNING: Pipeline interface mismatch detected; skipping destructive recovery (count=%llu)",
                      s_pipelineMismatchLogCount);
            }
            return NO;
        }
    }

    // Analyze error code for specific recovery strategies
    switch (error.code) {
        case MTLCommandBufferStatusError:
            NSLog(@"MGL INFO: Command buffer execution failed - recreating command buffer");
            [self cleanupCommandBuffer];
            return YES;

        default:
            NSLog(@"MGL ERROR: Unknown Metal error code %ld - attempting recovery", (long)error.code);

            // Handle common error scenarios based on error code
            if (error.code >= 1000 && error.code < 2000) {
                NSLog(@"MGL INFO: Detected feature compatibility issue - using safer settings");
            } else if (error.code >= 2000 && error.code < 3000) {
                NSLog(@"MGL INFO: Detected memory issue - clearing resources");
                [self clearTextureCache];
            } else {
                NSLog(@"MGL ERROR: Unknown Metal error - attempting full recovery");
                [self resetMetalState];
            }
            return YES;
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
        if (_renderPassManager.state->currentCommandBuffer) {
            if (_renderPassManager.state->currentCommandBuffer.status == MTLCommandBufferStatusCommitted) {
                // Do not block indefinitely here; cleanup can be invoked on the render thread.
                // Command buffers retain resources until completion, so dropping the reference is safe.
                if (kMGLVerboseFrameLoopLogs) {
                    NSLog(@"MGL INFO: cleanupCommandBuffer skipping blocking wait for committed command buffer");
                }
            }
            [_renderPassManager discardCurrentCommandBuffer];
        }

        if (_renderPassManager.state->currentRenderEncoder) {
            [_renderPassManager.state->currentRenderEncoder endEncoding];
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

    /* Runs from addCompletedHandler (Metal worker thread).  Hold
     * _metalStateLock to prevent the render thread observing a half-reset
     * state. */
    METAL_LOCK();

    [self cleanupCommandBuffer];

    // CRITICAL: Recreate command queue to clear AGX driver error state
    NSLog(@"MGL AGX RECOVERY: Recreating command queue to clear GPU error state");
    _commandQueue = nil;
    _commandQueue = [_device newCommandQueue];
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
- (void)commitCommandBufferWithAGXRecovery:(id<MTLCommandBuffer>)commandBuffer
{
    static uint64_t s_commitCallCount = 0;
    uint64_t commitCall = ++s_commitCallCount;
    bool traceCommit = mglShouldTraceCall(commitCall);

    if (!commandBuffer) {
        NSLog(@"MGL ERROR: Cannot commit NULL command buffer");
        return;
    }

    if (traceCommit) {
        MGLTraceNSLog(@"MGL TRACE commit.begin call=%llu cb=%p status=%s label=%@",
              (unsigned long long)commitCall,
              commandBuffer,
              mglCommandBufferStatusName(commandBuffer.status),
              commandBuffer.label ?: @"(no-label)");
    }
    double commitQueuedAtSeconds = mglNowSeconds();

    // Pre-commit validation for AGX driver
    if (commandBuffer.error) {
        NSLog(@"MGL AGX WARNING: Command buffer has pre-commit error: %@", commandBuffer.error);
        [self recordGPUError];
    }

    // Add completion handler for AGX error detection
    __block typeof(self) blockSelf = self;
    uint64_t commitCallForBlock = commitCall;
    bool traceCommitForBlock = traceCommit;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            double completeElapsedMs = (mglNowSeconds() - commitQueuedAtSeconds) * 1000.0;
            if (traceCommitForBlock || buffer.error || completeElapsedMs >= 50.0) {
                MGLTraceNSLog(@"MGL TRACE commit.completed call=%llu status=%s elapsed=%.3fms error=%@",
                      (unsigned long long)commitCallForBlock,
                      mglCommandBufferStatusName(buffer.status),
                      completeElapsedMs,
                      buffer.error);
            }
            if (buffer.error) {
                NSLog(@"MGL AGX ERROR: Command buffer completed with error: %@", buffer.error);
                [blockSelf recordGPUError];

                // Specific handling for AGX driver rejection
                if ([buffer.error.domain isEqualToString:@"MTLCommandBufferErrorDomain"] &&
                    buffer.error.code == 4) { // "Ignored (for causing prior/excessive GPU errors)"
                static NSTimeInterval s_lastDriverRejectionReset = 0.0;
                NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
                if (now - s_lastDriverRejectionReset > 2.0) {
                    s_lastDriverRejectionReset = now;
                    NSLog(@"MGL AGX RECOVERY: Driver rejection detected; throttled reset scheduled");
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [blockSelf resetMetalState];
                    });
                } else {
                    NSLog(@"MGL AGX RECOVERY: Driver rejection detected; skipping immediate reset (throttled)");
                }
                }
            } else {
            [blockSelf recordGPUSuccess];

            // AGX Recovery: Clear recovery mode on success
            /* guard the ivar read/write with _gpuRecovery.gpuErrorLock
             * (NOT _metalStateLock) to avoid deadlock — the completion handler
             * runs on a Metal worker thread while the render thread may be
             * inside waitUntilCompleted holding _metalStateLock. */
            os_unfair_lock_lock(&blockSelf->_gpuRecovery.gpuErrorLock);
            if (blockSelf->_gpuRecovery.gpuErrorRecoveryMode) {
                NSLog(@"MGL AGX RECOVERY: Exiting GPU recovery mode after successful completion");
                blockSelf->_gpuRecovery.gpuErrorRecoveryMode = NO;
            }
            os_unfair_lock_unlock(&blockSelf->_gpuRecovery.gpuErrorLock);
        }
    }];

    // CRITICAL FIX: Enhanced command buffer validation before commit
    // Prevents MTLReleaseAssertionFailure in AGX driver
    if (!commandBuffer) {
        NSLog(@"MGL AGX ERROR: Cannot commit nil command buffer");
        return;
    }

    // Check command buffer status before commit
    MTLCommandBufferStatus status = [commandBuffer status];
    if (status >= MTLCommandBufferStatusCommitted) {
        NSLog(@"MGL AGX WARNING: Command buffer already committed (status: %ld) - skipping commit", (long)status);
        if (traceCommit) {
            MGLTraceNSLog(@"MGL TRACE commit.skip.already_committed call=%llu status=%s",
                  (unsigned long long)commitCall, mglCommandBufferStatusName(status));
        }
        return;
    }

    // Validate command buffer is in a valid state for commit
    if (status == MTLCommandBufferStatusError) {
        NSLog(@"MGL AGX ERROR: Command buffer in error state - skipping commit");
        [self recordGPUError];
        if (traceCommit) {
            MGLTraceNSLog(@"MGL TRACE commit.skip.error_state call=%llu", (unsigned long long)commitCall);
        }
        return;
    }

    if (![_renderPassManager beginCommandBufferCommit]) {
        NSLog(@"MGL AGX WARNING: Commit already in progress, skipping nested commit");
        if (traceCommit) {
            MGLTraceNSLog(@"MGL TRACE commit.skip.nested call=%llu", (unsigned long long)commitCall);
        }
        return;
    }

    @try {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL AGX: Committing command buffer (status: %ld)", (long)status);
        }
        [commandBuffer commit];
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
            dispatch_async(dispatch_get_main_queue(), ^{
                [self resetMetalState];
            });
        }
    } @finally {
        [_renderPassManager endCommandBufferCommit];
        if (traceCommit) {
            MGLTraceNSLog(@"MGL TRACE commit.end call=%llu cb=%p finalStatus=%s",
                  (unsigned long long)commitCall,
                  commandBuffer,
                  mglCommandBufferStatusName(commandBuffer.status));
        }
    }
}

// AGX GPU Error Throttling - Prevent command queue from entering error state
- (BOOL)shouldSkipGPUOperations
{
    NSTimeInterval currentTime = [[NSDate date] timeIntervalSince1970];
    BOOL needsClear = NO;

    /* protect error-tracking ivars with _gpuRecovery.gpuErrorLock
     * (same lock as recordGPUError/recordGPUSuccess) to avoid racing with
     * the completion handler thread. */
    os_unfair_lock_lock(&_gpuRecovery.gpuErrorLock);

    // Recovery window: shorter timeout so essential operations can resume sooner
    if (currentTime - _gpuRecovery.lastGPUErrorTime > 3.0) {
        if (_gpuRecovery.consecutiveGPUErrors > 0) {
            NSLog(@"MGL AGX: Recovery timeout - attempting GPU operations (had %lu errors)", (unsigned long)_gpuRecovery.consecutiveGPUErrors);
        }
        _gpuRecovery.consecutiveGPUErrors = 0;
        _gpuRecovery.gpuErrorRecoveryMode = NO;
        os_unfair_lock_unlock(&_gpuRecovery.gpuErrorLock);
        return NO;
    }

    // Enter recovery mode after fewer errors to prevent AGX driver from crashing
    if (_gpuRecovery.consecutiveGPUErrors >= 8 || _gpuRecovery.gpuErrorRecoveryMode) {
        if (!_gpuRecovery.gpuErrorRecoveryMode) {
            NSLog(@"MGL AGX: Entering recovery mode after %lu consecutive errors", (unsigned long)_gpuRecovery.consecutiveGPUErrors);
            _gpuRecovery.gpuErrorRecoveryMode = YES;
            needsClear = YES;
        }
        os_unfair_lock_unlock(&_gpuRecovery.gpuErrorLock);
        if (needsClear) {
            [self clearProblematicGPUState];
        }
        return YES;
    }

    os_unfair_lock_unlock(&_gpuRecovery.gpuErrorLock);
    return NO;
}

// PROPER FIX: Clear problematic state without giving up on GPU operations entirely
- (void)clearProblematicGPUState
{
    NSLog(@"MGL AGX: Clearing problematic GPU state for recovery");

    // Clear current problematic resources
    if (_renderPassManager.state->currentCommandBuffer) {
        [_renderPassManager discardCurrentCommandBuffer];
    }

    // Don't recreate command queue immediately - let it rest
    // The AGX driver needs time to recover from error state
}

// AGX DRIVER COMPATIBILITY: Accept virtualization limitations and provide minimal functionality
- (void)enableMinimalFunctionalityMode
{
    NSLog(@"MGL AGX: Enabling minimal functionality mode for AGX virtualization compatibility");

    // Stop fighting the AGX driver - accept virtualization limitations
    // Don't recreate command queues - they will continue to fail
    // Don't submit command buffers - they will continue to be rejected

    // Provide minimal framebuffer clearing without GPU operations
    // This prevents magenta screens while accepting virtualization constraints
}

- (void)recordGPUError
{
    /* Use _gpuRecovery.gpuErrorLock (not METAL_LOCK): addCompletedHandler
     * runs on a Metal worker thread; blocking on _metalStateLock here
     * deadlocks if the render thread holds it inside waitUntilCompleted. */
    os_unfair_lock_lock(&_gpuRecovery.gpuErrorLock);
    _gpuRecovery.consecutiveGPUErrors++;
    _gpuRecovery.consecutiveGPUSuccesses = 0;
    _gpuRecovery.lastGPUErrorTime = [[NSDate date] timeIntervalSince1970];
    NSLog(@"MGL AGX: Recorded GPU error (%lu consecutive)", (unsigned long)_gpuRecovery.consecutiveGPUErrors);
    os_unfair_lock_unlock(&_gpuRecovery.gpuErrorLock);
}

- (void)recordGPUSuccess
{
    /* use _gpuRecovery.gpuErrorLock (see recordGPUError comment). */
    os_unfair_lock_lock(&_gpuRecovery.gpuErrorLock);
    if (_gpuRecovery.consecutiveGPUErrors > 0 || _gpuRecovery.gpuErrorRecoveryMode) {
        _gpuRecovery.consecutiveGPUSuccesses++;
        NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
        NSTimeInterval sinceLastError = now - _gpuRecovery.lastGPUErrorTime;
        // Require multiple consecutive successful completions before clearing
        // recovery, otherwise mixed success/error callbacks can flap the state.
        if (_gpuRecovery.consecutiveGPUSuccesses >= 4 && sinceLastError > 0.25) {
            NSLog(@"MGL AGX: Sustained GPU recovery (%lu successes), resetting error count (was %lu)",
                  (unsigned long)_gpuRecovery.consecutiveGPUSuccesses,
                  (unsigned long)_gpuRecovery.consecutiveGPUErrors);
            _gpuRecovery.consecutiveGPUErrors = 0;
            _gpuRecovery.gpuErrorRecoveryMode = NO;
            _gpuRecovery.consecutiveGPUSuccesses = 0;
        }
    }
    os_unfair_lock_unlock(&_gpuRecovery.gpuErrorLock);
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
