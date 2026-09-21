/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RENDER_API_COMMAND_H
#define MGL_RENDER_API_COMMAND_H

/* Declarations for the command slice of the renderer facade.
 * Value layouts live in mgl_render.h. Standalone include pulls
 * mgl_render_fwd.h (incomplete types) instead of the full facade. */

#ifndef MGL_RENDER_H
#include "mgl_render_fwd.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Pure synchronization helpers. Metal descriptor inspection is confined to
 * the Metal-cpp implementation TU; the C ABI carries only opaque handles and
 * integer enum values. */
bool mglRenderPassAttachmentMatchesSubresource(
    const void *descriptor,
    const MGLMetalAttachmentSubresource *subresource);

const char *mglRenderCommandBufferStatusName(uint32_t status);

void mglRenderInvalidateRenderPass(GLMContext glm_ctx);

/* C1: PixelFormatIsPackedDepthStencil -> mgl_pso_format_class.h */
uint32_t mglRenderDepthBlitStencilFormat(uint32_t pixel_format);

int mglRenderFBOBlitAttachmentKnown(uint32_t attachment, int is_color);

uint32_t mglRenderFBOBlitAttachmentOrColor0(uint32_t attachment, int is_color);

int mglRenderBlitIsRGBA8BGRA8Pair(uint32_t src_format, uint32_t dst_format);

/* scaled-blit UV computation (normalized source
 * rect with the Metal Y-flip, clamped, direction-swapped per the forward
 * flags).  Pure CPU, shared by both gates. */
int mglRenderScaledBlitUVs(
    uint32_t src_tex_w,
    uint32_t src_tex_h,
    double src_min_x,
    double src_max_x,
    double src_min_y,
    double src_max_y,
    int src_x_forward,
    int src_y_forward,
    int dst_x_forward,
    int dst_y_forward,
    MGLRenderScaledBlitUVs *out);

/* scaled-blit destination scissor base — floor/ceil
 * of the destination rect in Metal Y, clamped to the destination texture.
 * The caller intersects the GL scissor box on top.  Pure CPU, shared by
 * both gates. */
int mglRenderBlitScissorRect(
    double dst_min_x,
    double dst_max_x,
    double scaled_dst_metal_y,
    double dst_h,
    uint32_t dst_tex_w,
    uint32_t dst_tex_h,
    MGLRenderBlitScissorRect *out);

/* glBlitFramebuffer region math + decisions after
 * the axis clip — direction/flip flags, min/max/abs extents, the scaled-
 * blit decision (format conversion / RT sync / scissor / flip / size
 * mismatch with the 1e-5 epsilon of mglNearlyEqual), the integer copy
 * rect, the Metal Y-flips and the scaled-path destination Y.  Pure CPU
 * plan shared by both gates.  Returns 0 with the plan filled, -1 when the
 * clipped region has zero extent (caller logs and skips). */
int mglRenderBlitFramebufferPlan(
    double src_x0,
    double src_x1,
    double src_y0,
    double src_y1,
    double dst_x0,
    double dst_x1,
    double dst_y0,
    double dst_y1,
    uint32_t src_tex_w,
    uint32_t src_tex_h,
    uint32_t dst_tex_w,
    uint32_t dst_tex_h,
    int needs_format_conversion_blit,
    int needs_render_target_sync_blit,
    int scissor_test_enabled,
    MGLRenderBlitFramebufferPlan *out);

int mglRenderDispatchCompute(void *compute_encoder,
                                uint32_t groups_x,
                                uint32_t groups_y,
                                uint32_t groups_z,
                                uint32_t threads_x,
                                uint32_t threads_y,
                                uint32_t threads_z);

int mglRenderDispatchComputeIndirect(void *compute_encoder,
                                        void *indirect_buffer,
                                        uint64_t indirect_offset,
                                        uint32_t threads_x,
                                        uint32_t threads_y,
                                        uint32_t threads_z);

int mglRenderDispatchComputePlan(
    void *compute_encoder,
    const MGLRenderComputePlan *plan,
    char *err,
    size_t errcap);

int mglRenderAppendComputeDispatchToPlan(
    MGLRenderComputeExecutionPlan *plan,
    const MGLRenderComputePlan *dispatch,
    char *err,
    size_t errcap);

int mglRenderEncodeComputeExecutionPlanForCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, const MGLRenderComputeExecutionPlan *plan, char *err, size_t errcap);

/* Creates a compute encoder, sets its pipeline and setup bindings, and returns
 * a borrowed encoder owned by the command buffer. Returns -1 on failure. */
int mglRenderBeginComputeDispatch(
    void *command_buffer,
    const MGLRenderComputeDispatchSetup *setup,
    void **compute_encoder_out,
    char *err,
    size_t errcap);

/* Owner-aware form. CommandBufferOwner.current remains inside C++. */
int mglRenderBeginComputeDispatchForCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, const MGLRenderComputeDispatchSetup *setup, void **compute_encoder_out, char *err, size_t errcap);

/* Dispatches and ends the encoder returned by mglRenderBeginComputeDispatch. */
int mglRenderEndComputeDispatch(void *compute_encoder,
                                   const uint32_t groups[3],
                                   const uint32_t threads[3],
                                   char *err,
                                   size_t errcap);

int mglRenderDispatchComputeThreads(void *compute_encoder,
                                       uint32_t threads_x,
                                       uint32_t threads_y,
                                       uint32_t threads_z,
                                       uint32_t group_x,
                                       uint32_t group_y,
                                       uint32_t group_z);

int mglRenderCreateComputeEncoder(void *command_buffer,
                                     void **compute_encoder_out);

int mglRenderEndComputeEncoder(void *compute_encoder);

/* Command-buffer/render-pass lifecycle facade.  Returned Metal objects are
 * borrowed Objective-C-compatible pointers; the caller retains them through
 * its normal strong state field. */
int mglRenderCreateCommandBuffer(void *command_queue,
                                    void **command_buffer_out);

/* Snapshot status/error data into caller-owned storage. The completion
 * registration keeps context alive until Metal completes the command buffer,
 * invokes callback once, then invokes destroy_context exactly once. The state
 * pointer passed to callback is valid only for the duration of that call. */
int mglRenderGetCommandBufferState(
    void *command_buffer,
    MGLRenderCommandBufferState *state_out);

const char *mglRenderCommandBufferErrorDescription(
    const MGLRenderCommandBufferState *state);

uint32_t mglRenderCommandBufferStatus(void *command_buffer);

int mglRenderGetCommandBufferLabel(const void *command_buffer,
                                      char *label_out,
                                      size_t label_capacity);

int mglRenderSetCommandBufferLabel(void *command_buffer,
                                      const char *label);

/* Pure value-state classification used by the owner transaction and platform
 * log adapters. Commit classification preserves the legacy status ordering. */
int mglRenderClassifyCommandBufferCommit(
    const MGLRenderCommandBufferState *state,
    MGLRenderCommandBufferCommitDecision *decision_out);

/* Commit one detached/current command buffer through the C++ owner.  When
 * submission_handle points at a matching C++ submission, that ownership is
 * consumed; otherwise the borrowed command buffer is committed directly.
 * Recovery counting, driver-rejection classification, and reset decisions are
 * returned as value-state; the caller only publishes platform logging/reset. */
int mglRenderCommitCommandBufferTransaction(MGLCommandBufferOwner *owner, void **submission_handle, void *command_buffer, MGLCommandBufferRecoveryOwner *recovery_owner, uint32_t wait_for_completion, MGLRenderCommandBufferTransaction *result_out);

int mglRenderClassifyCommandBufferCompletion(
    const MGLRenderCommandBufferState *state,
    MGLRenderCommandBufferCompletionDecision *decision_out);

/* Thread-safe owner for the renderer's command-completion error counters.
 * Timestamps are caller-provided seconds so policy remains independent of
 * Foundation and can be tested deterministically. */
int mglRenderCreateCommandRecoveryOwner(MGLCommandBufferRecoveryOwner **owner_out);

void mglRenderDestroyCommandRecoveryOwner(MGLCommandBufferRecoveryOwner **owner);

int mglRenderCommandRecoveryRecordError(MGLCommandBufferRecoveryOwner *owner, double now, MGLRenderCommandRecoverySnapshot *state_out);

/* Platform exception boundary for failures that cannot cross the C++ ABI.
 * Applies the recovery update at most once to transaction_inout. */
int mglRenderCommandRecoveryRecordTransactionFailure(MGLCommandBufferRecoveryOwner *owner, const MGLRenderCommandBufferState *state, MGLRenderCommandBufferTransaction *transaction_inout);

int mglRenderCommandRecoveryRecordSuccess(MGLCommandBufferRecoveryOwner *owner, double now, MGLRenderCommandRecoverySuccess *result_out);

int mglRenderCommandRecoveryShouldSkip(MGLCommandBufferRecoveryOwner *owner, double now, MGLRenderCommandRecoverySkipDecision *decision_out);

/* Classify one completed command buffer and apply the legacy recovery-owner
 * update sequence. Success intentionally performs RecordSuccess followed by
 * the separate ClearMode operation so the former two-lock ordering remains
 * observable; ObjC consumes the returned value state for logging/reset work. */
int mglRenderProcessCommandBufferCompletion(
    MGLCommandBufferRecoveryOwner *owner,
    const MGLRenderCommandBufferState *state,
    double now,
    MGLRenderCommandBufferCompletionResult *result_out);

/* Register the standard command-recovery completion handler without capturing
 * an Objective-C renderer.  The C++ recovery owner records error/success
 * counters and latches a deferred-reset request for the GL thread. */
int mglRenderAddCommandBufferRecoveryCompletion(void *command_buffer, MGLCommandBufferRecoveryOwner *recovery_owner);

/* Consume a reset request latched by a completion worker. Returns 1 when a
 * request was consumed, 0 when none is pending, and -1 for invalid owner. */
int mglRenderCommandRecoveryTakeResetRequest(MGLCommandBufferRecoveryOwner *recovery_owner);

int mglRenderAddCommandBufferCompletion(
    void *command_buffer,
    MGLRenderCommandBufferCompletion callback,
    void *context,
    MGLRenderDestroyContext destroy_context);

/* Register a completion on CommandBufferOwner.current without exposing the
 * borrowed command buffer through the C ABI. */
int mglRenderAddCommandBufferOwnerCompletion(MGLCommandBufferOwner *owner, MGLRenderCommandBufferCompletion callback, void *context, MGLRenderDestroyContext destroy_context);

/* The current-buffer owner retains the autoreleased command buffer returned
 * by Metal. Detach moves that +1 reference into a submission handle; commit
 * consumes the submission only after Metal accepts it. Returned command
 * buffer pointers are borrowed. */
int mglRenderCreateCommandBufferOwner(void *command_queue, MGLCommandBufferOwner **owner_out, void **command_buffer_out);

/* Adopt an existing (ObjC-created) command buffer as the owner's current —
 * gate-off fallback so the owner stays the single source on both gates.
 * Returns 0 with *owner_out set (the owner retains the buffer). */
int mglRenderCreateCommandBufferOwnerAdopt(void *command_buffer, MGLCommandBufferOwner **owner_out);

/* Borrowed pointer to the owner's current command buffer (NULL when the
 * owner has none / owner is NULL). */
void *mglRenderCommandBufferOwnerGetCurrent(MGLCommandBufferOwner *owner);

/* Returns 1 when current exists, 0 when empty, and -1 for a null owner. */
int mglRenderCommandBufferOwnerHasCurrent(MGLCommandBufferOwner *owner);

/* Create the next current command buffer from the queue retained by the
 * owner.  Returns 0 on success, 1 when the owner has no queue (adopted
 * fallback), and -1 on allocation/argument failure. */
int mglRenderCommandBufferOwnerCreateNext(MGLCommandBufferOwner *owner, void **command_buffer_out);

/* Snapshot the owner's current buffer without exposing it to the caller.
 * Returns -1 when the owner/current buffer/state output is missing. */
int mglRenderGetCommandBufferOwnerState(MGLCommandBufferOwner *owner, MGLRenderCommandBufferState *state_out);

/* Boolean convenience form: returns 1 when a snapshot was produced. */
int mglRenderCommandBufferOwnerHasState(
    MGLCommandBufferOwner *owner,
    MGLRenderCommandBufferState *state_out);

/* The owner retains the most recently accepted submission. These APIs keep
 * glFinish/readback synchronization in the lifecycle owner without exposing
 * a borrowed command-buffer pointer to Objective-C. */
int mglRenderCommandBufferOwnerHasLastSubmitted(MGLCommandBufferOwner *owner);

/* Wait for one submitted command buffer and return a value-state snapshot.
 * Returns 0 on completed success, 1 when the buffer is still NotEnqueued,
 * and -1 for invalid arguments, wait failures, or command-buffer errors. */
int mglRenderWaitCommandBufferState(
    void *command_buffer,
    MGLRenderCommandBufferState *state_out);

int mglRenderWaitCommandBufferOwnerLastSubmitted(MGLCommandBufferOwner *owner, MGLRenderCommandBufferState *state_out);

/* Encode presentation on the owner's current not-enqueued command buffer.
 * Returns 0 on success, 1 when the current buffer is already finalized, and
 * -1 for missing owner/current buffer/drawable. */
int mglRenderPresentDrawableForCommandBufferOwner(MGLCommandBufferOwner *owner, void *drawable, MGLRenderCommandBufferState *state_out);

int mglRenderEncodeWaitForEventForCommandBufferOwner(MGLCommandBufferOwner *owner, void *event, uint64_t value);

int mglRenderResetCommandBufferOwner(MGLCommandBufferOwner *owner, void *command_queue, void **command_buffer_out);

void mglRenderDiscardCommandBufferOwnerCurrent(MGLCommandBufferOwner *owner);

/* Reentrancy guard for command-buffer commit. Returns 1 when acquired, 0
 * when a commit is already in progress, and -1 for a missing owner. This
 * preserves the former MGLCommandState BOOL semantics; it is intentionally
 * not a cross-thread synchronization primitive. */
int mglRenderCommandBufferOwnerBeginCommit(MGLCommandBufferOwner *owner);

void mglRenderCommandBufferOwnerEndCommit(MGLCommandBufferOwner *owner);

/* Consume the marker for a current buffer created by the preceding submit
 * transaction. Returns 1 and a borrowed current buffer once, 0 when no such
 * buffer is pending, and -1 for invalid arguments. */
int mglRenderCommandBufferOwnerConsumeTransactionCurrent(MGLCommandBufferOwner *owner, void **command_buffer_out);

int mglRenderTakeCommandBufferSubmission(MGLCommandBufferOwner *owner, void **submission_out, void **command_buffer_out);

int mglRenderCommitCommandBufferSubmission(void **submission);

void mglRenderDestroyCommandBufferSubmission(void **submission);

void mglRenderDestroyCommandBufferOwner(MGLCommandBufferOwner **owner);

/* The opaque owner holds the +1 Metal-cpp command-queue reference. The queue
 * pointer is borrowed and may be assigned to an ObjC strong field during the
 * migration. max_command_buffers=0 selects Metal's default configuration. */
int mglRenderCreateCommandQueueOwner(uint32_t max_command_buffers, MGLCommandQueueOwner **owner_out, void **command_queue_out);

int mglRenderResetCommandQueueOwner(MGLCommandQueueOwner *owner, uint32_t max_command_buffers, void **command_queue_out);

void mglRenderDestroyCommandQueueOwner(MGLCommandQueueOwner **owner);

int mglRenderCommitCommandBuffer(void *command_buffer);

int mglRenderWaitCommandBuffer(void *command_buffer);

/* Initialize a value state with Metal's render-pass descriptor defaults. */
void mglRenderInitDefaultRenderPassState(
    MGLRenderPassState *state_out);

/* Persistent render-pass identity and FBO cache. The owner is authoritative
 * for Metal-cpp mode; ObjC fields remain a synchronized migration view. */
int mglRenderCreateRenderPassIdentityOwner(MGLRenderPassIdentityOwner **owner_out);

int mglRenderUpdateRenderPassIdentity(MGLRenderPassIdentityOwner *owner, const MGLRenderPassIdentityState *state);

int mglRenderGetRenderPassIdentity(MGLRenderPassIdentityOwner *owner, MGLRenderPassIdentityState *state_out);

void mglRenderDestroyRenderPassIdentityOwner(MGLRenderPassIdentityOwner **owner);

/* Persistent value-state owner for render-pass attachment/dimension fields.
 * The owner retains every attachment/resolve/visibility/rate-map resource
 * referenced by the snapshot and releases replaced resources on update. */
int mglRenderCreateRenderPassStateOwner(const MGLRenderPassState *state, MGLRenderPassStateOwner **owner_out);

int mglRenderCreateDefaultRenderPassStateOwner(MGLRenderPassStateOwner **owner_out);

int mglRenderSetRenderPassStateAttachment(MGLRenderPassStateOwner *owner, uint32_t attachment_kind, uint32_t color_index, const MGLRenderPassAttachmentState *attachment);

int mglRenderSetRenderPassStateAttachmentTexture(MGLRenderPassStateOwner *owner, uint32_t attachment_kind, uint32_t color_index, void *texture, uint64_t level, uint64_t slice, uint64_t depth_plane, uint32_t layered);

int mglRenderSetRenderPassStateAttachmentActions(MGLRenderPassStateOwner *owner, uint32_t attachment_kind, uint32_t color_index, uint32_t load_action, uint32_t store_action, uint64_t store_action_options);

int mglRenderSetRenderPassStateColorClear(MGLRenderPassStateOwner *owner, uint32_t color_index, double red, double green, double blue, double alpha);

int mglRenderSetRenderPassStateDepthClear(MGLRenderPassStateOwner *owner, double clear_depth);

int mglRenderSetRenderPassStateStencilClear(MGLRenderPassStateOwner *owner, uint32_t clear_stencil);

int mglRenderSetRenderPassStateVisibility(MGLRenderPassStateOwner *owner, void *visibility_result_buffer, uint32_t visibility_result_type);

int mglRenderSetRenderPassStateDimensions(MGLRenderPassStateOwner *owner, uint64_t width, uint64_t height);

/* detached-submission ownership guard. */
int mglRenderCommandBufferSubmissionMatchesBuffer(void *submission_handle,
                                                     void *command_buffer);

/* current-CB sync tracking list inside the C++ owner. */
int mglRenderCommandBufferOwnerAppendSync(MGLCommandBufferOwner *owner_handle, Sync *sync);

int mglRenderGetRenderPassStateOwner(MGLRenderPassStateOwner *owner, MGLRenderPassState *state_out);

/* Returns a borrowed attachment snapshot. Object pointers remain owned by the
 * render-pass owner and are valid only while that owner keeps the state. */
int mglRenderGetRenderPassAttachmentStateOwner(MGLRenderPassStateOwner *owner, uint32_t attachment_kind, uint32_t color_index, MGLRenderPassAttachmentState *attachment_out);

int mglRenderCreateRenderEncoderFromStateOwner(void *command_buffer, MGLRenderPassStateOwner *state_owner, void **render_encoder_out);

/* Owner-aware variant used by command-lifecycle callers. The command buffer
 * stays inside CommandBufferOwner; the returned encoder is borrowed. */
int mglRenderCreateRenderEncoderFromCommandBufferOwnerState(MGLCommandBufferOwner *command_buffer_owner, const MGLRenderPassState *render_pass, void **render_encoder_out);

/* Borrowed-object convenience forms for Objective-C callers.  The return
 * value is an opaque Metal object owned by the command-buffer owner. */
void *mglRenderCreateRenderEncoderBorrowed(MGLCommandBufferOwner *command_buffer_owner, const MGLRenderPassState *render_pass);

void *mglRenderCreateBlitEncoderBorrowed(MGLCommandBufferOwner *command_buffer_owner);

void *mglRenderCreateComputeEncoderBorrowed(MGLCommandBufferOwner *command_buffer_owner);

void mglRenderDestroyRenderPassStateOwner(MGLRenderPassStateOwner **owner);

/* C++ owns the temporary MTL::RenderPassDescriptor used to create the
 * borrowed render encoder. Attachment resources remain caller-owned. */
int mglRenderCreateRenderEncoderFromState(
    void *command_buffer,
    const MGLRenderPassState *render_pass,
    void **render_encoder_out);

int mglRenderGetRenderPassAttachmentSubresourceOwner(MGLRenderPassStateOwner *owner, uint32_t attachment_kind, uint32_t color_index, uint64_t *level_out, uint64_t *slice_out, uint64_t *depth_plane_out);

int mglRenderGetRenderPassAttachmentActionsOwner(MGLRenderPassStateOwner *owner, uint32_t attachment_kind, uint32_t color_index, uint32_t *load_action_out, uint32_t *store_action_out, uint64_t *store_action_options_out);

uint32_t mglRenderPassLoadActionForTrace(
    MGLRenderPassStateOwner *owner, uint32_t attachment_kind, uint32_t color_index,
    uint32_t default_load_action);

uint32_t mglRenderPassStoreActionForTrace(
    MGLRenderPassStateOwner *owner, uint32_t attachment_kind, uint32_t color_index,
    uint32_t default_store_action);

int mglRenderEncodeMultisampleResolve(
    void *command_buffer,
    uint32_t attachment_kind,
    void *source_texture,
    uint64_t source_level,
    uint64_t source_slice,
    uint64_t source_depth_plane,
    void *resolve_texture,
    uint64_t resolve_level,
    uint64_t resolve_slice,
    uint64_t resolve_depth_plane,
    uint32_t resolve_filter);

int mglRenderEncodeMultisampleResolveForCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, uint32_t attachment_kind, void *source_texture, uint64_t source_level, uint64_t source_slice, uint64_t source_depth_plane, void *resolve_texture, uint64_t resolve_level, uint64_t resolve_slice, uint64_t resolve_depth_plane, uint32_t resolve_filter);

/* The owner retains the autoreleased encoder returned by Metal. End is
 * idempotent per owned encoder; destroy releases the retained reference. */
int mglRenderCreateRenderEncoderOwnerFromState(void *command_buffer, const MGLRenderPassState *render_pass, MGLRenderEncoderOwner **owner_out, void **render_encoder_out);

int mglRenderResetRenderEncoderOwnerFromState(MGLRenderEncoderOwner *owner, void *command_buffer, const MGLRenderPassState *render_pass, void **render_encoder_out);

int mglRenderCreateRenderEncoderOwner(void *render_encoder, MGLRenderEncoderOwner **owner_out);

int mglRenderResetRenderEncoderOwner(MGLRenderEncoderOwner *owner, void *render_encoder);

int mglRenderEndRenderEncoderOwner(MGLRenderEncoderOwner *owner);

int mglRenderSetRenderEncoderOwnerLabel(MGLRenderEncoderOwner *owner, const char *label);

int mglRenderEncoderOwnerHasCurrent(MGLRenderEncoderOwner *owner);

void mglRenderDestroyRenderEncoderOwner(MGLRenderEncoderOwner **owner);

int mglRenderEndRenderEncoder(void *render_encoder);

int mglRenderCreateBlitEncoder(void *command_buffer,
                                  void **blit_encoder_out);

/* Creates a borrowed blit encoder from CommandBufferOwner.current without
 * exposing the current command buffer through the C ABI. */
int mglRenderCreateBlitEncoderFromCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, void **blit_encoder_out);

int mglRenderEncodeBufferCopiesForCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, const MGLRenderBufferCopyEntry *entries, uint32_t entry_count);

int mglRenderCreateComputeEncoderFromCommandBufferOwner(MGLCommandBufferOwner *command_buffer_owner, void **compute_encoder_out);

int mglRenderEndBlitEncoder(void *blit_encoder);

int mglRenderBlitCopyBuffer(void *blit_encoder,
                               void *source_buffer,
                               uint64_t source_offset,
                               void *destination_buffer,
                               uint64_t destination_offset,
                               uint64_t size);

int mglRenderBlitGenerateMipmaps(void *blit_encoder,
                                    void *texture);

/* Encodes one draw. render_encoder is borrowed. Invalid plans return -1 and
 * populate err without encoding a partial draw. */
int mglRenderEncodeDraw(void *render_encoder,
                           const MGLRenderDrawPlan *plan,
                           char *err,
                           size_t errcap);

int mglRenderEncodeDrawForRenderEncoderOwner(MGLRenderEncoderOwner *render_encoder_owner, const MGLRenderDrawPlan *plan, char *err, size_t errcap);

int mglRenderCreateIndirectCommandBuffer(
    uint32_t command_types,
    int inherit_pipeline_state,
    int inherit_buffers,
    uint32_t max_vertex_buffer_bind_count,
    uint32_t max_fragment_buffer_bind_count,
    uint64_t max_command_count,
    uint64_t resource_options,
    void **indirect_buffer_out);

int mglRenderResetIndirectCommandBuffer(void *indirect_buffer,
                                           uint64_t location,
                                           uint64_t length);

int mglRenderGetIndirectRenderCommand(void *indirect_buffer,
                                         uint64_t command_index,
                                         void **command_out);

#ifdef __cplusplus
}
#endif

#endif
