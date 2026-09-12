/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * MGLRenderer+BatchPorts_Private.h
 * MGL
 *
 * Cross-port method surface for the A3 Batch-cluster encode ports
 * (mgl_batch_flush_restore_encode.m / mgl_batch_dyn_bind_encode.m /
 * mgl_batch_issue_encode.m / mgl_batch_icb_mdi_encode.m /
 * mgl_batch_replay_trace.m / mgl_batch_rt_mark_port.m).
 *
 * The port .m files are separate translation units; ObjC requires a
 * visible @interface before a selector can be sent across them.
 */

#ifndef MGLRenderer_BatchPorts_Private_h
#define MGLRenderer_BatchPorts_Private_h

#import "MGLRenderer_Private.h"
#import "draw_command.h"
#import "mgl_batch_replay.h"

@interface MGLRenderer (BatchPorts)

/* restore-from-key host (impl: MGLRenderer+Batch.m) */
- (void)restoreStateFromKey:(const MGLStateKey *)key context:(GLMContext)glm_ctx;

/* replay tracing (impl: mgl_batch_replay_trace.m) */
/* traceReplayBatch:... is now the C driver mglBatchTraceReplayBatch
 * (mgl_batch_rt_mark.h, implemented in mgl_batch_replay_trace.c). */

/* dyn-bind / simple replay (impl: mgl_batch_dyn_bind_encode.m) */
- (bool)applyDynamicBindingsForCommand:(const MGLDrawCommand *)cmd
                                context:(GLMContext)glm_ctx
                          encodeContext:(MGLEncodeContext *)encCtx;
- (bool)applySamplerSnapshotForCommand:(const MGLDrawCommand *)cmd
                                context:(GLMContext)glm_ctx
                          encodeContext:(const MGLEncodeContext *)encCtx;
- (BOOL)tryReplaySimpleBatch:(MGLDrawBatch *)batch
                     context:(GLMContext)glm_ctx
               encodeContext:(const MGLEncodeContext *)encCtx;

@end

#endif /* MGLRenderer_BatchPorts_Private_h */
