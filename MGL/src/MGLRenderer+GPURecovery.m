/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+GPURecovery.m
// Metal GPU error recovery methods extracted from MGLRenderer+RenderPass.m

#import "MGLRenderer_Private.h"
#include "mgl_env_flag.h"
#include "mgl_gpu_recovery.h"

@implementation MGLRenderer (GPURecovery)

#pragma mark - Metal State Validation and Recovery





// AGX Driver Compatibility: Specialized command buffer commit with recovery
- (void)commitCommandBufferWithAGXRecovery:(id)commandBuffer
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
        char commandBufferLabel[128];
        (void)mglRenderGetCommandBufferLabel(
            (__bridge const void *)commandBuffer,
            commandBufferLabel, sizeof(commandBufferLabel));
        mglTraceLog("MGL TRACE commit.begin call=%llu cb=%p status=%s label=%s",
              (unsigned long long)commitCall,
              commandBuffer,
              mglCommandBufferStatusName(
                  mglRenderCommandBufferStatus(
                      (__bridge void *)commandBuffer)),
              commandBufferLabel);
    }
    MGLRenderCommandBufferTransaction transaction = {0};
    @try {
        int transactionResult = [_renderPassManager
            commitCommandBufferTransaction:(__bridge void *)commandBuffer
            recoveryOwner:_gpuRecovery.commandRecoveryOwner
            waitForCompletion:NO
            result:&transaction];
        if (transaction.result ==
            MGL_RENDER_COMMAND_BUFFER_TRANSACTION_NESTED) {
            NSLog(@"MGL AGX WARNING: Commit already in progress, skipping nested commit");
            if (traceCommit) {
                mglTraceLog("MGL TRACE commit.skip.nested call=%llu",
                      (unsigned long long)commitCall);
            }
            return;
        }
        if (transaction.result ==
            MGL_RENDER_COMMAND_BUFFER_TRANSACTION_SKIPPED) {
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
        if (mglRenderCommandRecoveryRecordTransactionFailure(
                _gpuRecovery.commandRecoveryOwner, NULL, &transaction) == 0 &&
            transaction.device_reset_requested) {
            atomic_store_explicit(&_deviceResetRequested, true,
                                  memory_order_release);
        }
    } @finally {
        if (traceCommit) {
            mglTraceLog("MGL TRACE commit.end call=%llu cb=%p finalStatus=%s",
                  (unsigned long long)commitCall,
                  commandBuffer,
                  mglCommandBufferStatusName(
                      transaction.after.status));
        }
    }
    } @finally {
        [_renderPassManager releaseDetachedCommandBufferIfOwned:(__bridge void *)commandBuffer];
    }
}

// AGX GPU Error Throttling - Prevent command queue from entering error state

// PROPER FIX: Clear problematic state without giving up on GPU operations entirely



#pragma mark - Metal Optimization Methods



@end
