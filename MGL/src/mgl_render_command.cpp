/* SPDX-License-Identifier: LGPL-3.0-only */
#include "mgl_metal.h"
#include "mgl_render.h"
#include "mgl_render_pixel.h"
extern "C" {
#include "pixel_utils.h"
}
#include "mgl_program_resource.h"
#include "mgl_renderer_backend.h"
#include "mgl_air_loader.h"
#include "mgl_air_tess_abi.h"
#include "mgl_aux_assets.h"
#include "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_program_reflection.h"
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_program.h"
#include "mgl_types_state.h"
#include "mgl_types_sync.h"
#include "glm_context.h"
#include "mgl_render_internal.h"
#include "mgl_capability.h"
#include "mgl_sync.h"
#include "glm_limits.h"
#include "mgl_shader_abi.h"
#include "mgl_buffer_slots.h"
#include "mgl_buffer_plan.h"
#include "mgl_tess_domain.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <chrono>
#include <tuple>
#include <utility>
#include <vector>

#include <mach/mach.h>
#include <Block.h>
#include <objc/runtime.h>


uint32_t mglRenderDepthBlitStencilFormat(uint32_t pixel_format) {
    return mglRenderPixelFormatIsPackedDepthStencil(pixel_format)
               ? pixel_format
               : 0u;
}

int mglRenderFBOBlitAttachmentKnown(uint32_t attachment, int is_color) {
    if (is_color) {
        return 1;
    }
    return attachment == GL_DEPTH_ATTACHMENT ||
                   attachment == GL_STENCIL_ATTACHMENT ||
                   attachment == GL_DEPTH_STENCIL_ATTACHMENT
               ? 1
               : 0;
}

uint32_t mglRenderFBOBlitAttachmentOrColor0(uint32_t attachment, int is_color) {
    if (mglRenderFBOBlitAttachmentKnown(attachment, is_color)) {
        return attachment;
    }
    return GL_COLOR_ATTACHMENT0;
}

int mglRenderBlitIsRGBA8BGRA8Pair(uint32_t src_format, uint32_t dst_format) {
    int src_rgba = src_format == 70u /* RGBA8Unorm */;
    int src_bgra = src_format == 80u /* BGRA8Unorm */;
    int dst_rgba = dst_format == 70u;
    int dst_bgra = dst_format == 80u;
    return (src_rgba && dst_bgra) || (src_bgra && dst_rgba) ? 1 : 0;
}

extern "C"
int mglRenderScaledBlitUVs(
    uint32_t src_tex_w, uint32_t src_tex_h,
    double src_min_x, double src_max_x, double src_min_y, double src_max_y,
    int src_x_forward, int src_y_forward,
    int dst_x_forward, int dst_y_forward,
    MGLRenderScaledBlitUVs* out) {
    if (!out) return -1;
    const float invSrcW = src_tex_w ? (1.0f / (float)src_tex_w) : 0.0f;
    const float invSrcH = src_tex_h ? (1.0f / (float)src_tex_h) : 0.0f;
    float uvLeft = fmaxf(0.0f, fminf(1.0f, (float)src_min_x * invSrcW));
    float uvRight = fmaxf(0.0f, fminf(1.0f, (float)src_max_x * invSrcW));
    float uvTop = fmaxf(0.0f, fminf(1.0f, (float)((double)src_tex_h - src_max_y) * invSrcH));
    float uvBottom = fmaxf(0.0f, fminf(1.0f, (float)((double)src_tex_h - src_min_y) * invSrcH));
    if (src_x_forward != dst_x_forward) {
        const float tmp = uvLeft;
        uvLeft = uvRight;
        uvRight = tmp;
    }
    if (src_y_forward != dst_y_forward) {
        const float tmp = uvTop;
        uvTop = uvBottom;
        uvBottom = tmp;
    }
    out->uv_left = uvLeft;
    out->uv_top = uvTop;
    out->uv_right = uvRight;
    out->uv_bottom = uvBottom;
    return 0;
}

extern "C"
int mglRenderBlitScissorRect(
    double dst_min_x, double dst_max_x,
    double scaled_dst_metal_y, double dst_h,
    uint32_t dst_tex_w, uint32_t dst_tex_h,
    MGLRenderBlitScissorRect* out) {
    if (!out) return -1;
    const double scaledDstMetalBottom = scaled_dst_metal_y + dst_h;
    int64_t x0 = (int64_t)floor(dst_min_x + 0.00001);
    int64_t x1 = (int64_t)ceil(dst_max_x - 0.00001);
    int64_t y0 = (int64_t)floor(scaled_dst_metal_y + 0.00001);
    int64_t y1 = (int64_t)ceil(scaledDstMetalBottom - 0.00001);
    x0 = fmax((int64_t)0, fmin(x0, (int64_t)dst_tex_w));
    x1 = fmax((int64_t)0, fmin(x1, (int64_t)dst_tex_w));
    y0 = fmax((int64_t)0, fmin(y0, (int64_t)dst_tex_h));
    y1 = fmax((int64_t)0, fmin(y1, (int64_t)dst_tex_h));
    out->x0 = x0;
    out->x1 = x1;
    out->y0 = y0;
    out->y1 = y1;
    return 0;
}

extern "C"
int mglRenderBlitFramebufferPlan(
    double src_x0, double src_x1, double src_y0, double src_y1,
    double dst_x0, double dst_x1, double dst_y0, double dst_y1,
    uint32_t src_tex_w, uint32_t src_tex_h,
    uint32_t dst_tex_w, uint32_t dst_tex_h,
    int needs_format_conversion_blit, int needs_render_target_sync_blit,
    int scissor_test_enabled,
    MGLRenderBlitFramebufferPlan* out) {
    if (!out) return -1;
    out->src_x_forward = src_x1 >= src_x0 ? 1 : 0;
    out->src_y_forward = src_y1 >= src_y0 ? 1 : 0;
    out->dst_x_forward = dst_x1 >= dst_x0 ? 1 : 0;
    out->dst_y_forward = dst_y1 >= dst_y0 ? 1 : 0;
    out->blit_needs_flip =
        (out->src_x_forward != out->dst_x_forward ||
         out->src_y_forward != out->dst_y_forward) ? 1 : 0;
    out->src_min_x = fmin(src_x0, src_x1);
    out->src_max_x = fmax(src_x0, src_x1);
    out->src_min_y = fmin(src_y0, src_y1);
    out->src_max_y = fmax(src_y0, src_y1);
    out->dst_min_x = fmin(dst_x0, dst_x1);
    out->dst_max_x = fmax(dst_x0, dst_x1);
    out->dst_min_y = fmin(dst_y0, dst_y1);
    out->dst_max_y = fmax(dst_y0, dst_y1);
    out->src_w = fabs(src_x1 - src_x0);
    out->src_h = fabs(src_y1 - src_y0);
    out->dst_w = fabs(dst_x1 - dst_x0);
    out->dst_h = fabs(dst_y1 - dst_y0);
    if (out->src_w <= 0.0 || out->src_h <= 0.0 ||
        out->dst_w <= 0.0 || out->dst_h <= 0.0) {
        return -1;
    }
    out->needs_scaled_blit =
        (needs_format_conversion_blit || needs_render_target_sync_blit ||
         scissor_test_enabled || out->blit_needs_flip ||
         fabs(out->src_w - out->dst_w) > 0.00001 ||
         fabs(out->src_h - out->dst_h) > 0.00001) ? 1 : 0;
    out->copy_src_x = (int64_t)floor(out->src_min_x + 0.00001);
    out->copy_src_y = (int64_t)floor(out->src_min_y + 0.00001);
    out->copy_dst_x = (int64_t)floor(out->dst_min_x + 0.00001);
    out->copy_dst_y = (int64_t)floor(out->dst_min_y + 0.00001);
    out->copy_w = (int64_t)ceil(out->src_max_x - 0.00001) - out->copy_src_x;
    out->copy_h = (int64_t)ceil(out->src_max_y - 0.00001) - out->copy_src_y;
    out->src_metal_y = (int64_t)src_tex_h - (out->copy_src_y + out->copy_h);
    out->dst_metal_y = (int64_t)dst_tex_h - (out->copy_dst_y + out->copy_h);
    out->scaled_dst_metal_y = (double)dst_tex_h - out->dst_max_y;
    return 0;
}

int mglRenderDispatchCompute(void* compute_encoder,
                                uint32_t groups_x,
                                uint32_t groups_y,
                                uint32_t groups_z,
                                uint32_t threads_x,
                                uint32_t threads_y,
                                uint32_t threads_z) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder || groups_x == 0 || groups_y == 0 || groups_z == 0 ||
        threads_x == 0 || threads_y == 0 || threads_z == 0) {
        return -1;
    }
    MTL::Size groups = MTL::Size(groups_x, groups_y, groups_z);
    MTL::Size threads = MTL::Size(threads_x, threads_y, threads_z);
    encoder->dispatchThreadgroups(groups, threads);
    return 0;
}

int mglRenderDispatchComputeIndirect(void* compute_encoder,
                                        void* indirect_buffer,
                                        uint64_t indirect_offset,
                                        uint32_t threads_x,
                                        uint32_t threads_y,
                                        uint32_t threads_z) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    MTL::Buffer* buffer = static_cast<MTL::Buffer*>(indirect_buffer);
    if (!encoder || !buffer || threads_x == 0 || threads_y == 0 ||
        threads_z == 0) {
        return -1;
    }
    MTL::Size threads = MTL::Size(threads_x, threads_y, threads_z);
    encoder->dispatchThreadgroups(buffer,
                                  static_cast<NS::UInteger>(indirect_offset),
                                  threads);
    return 0;
}

int mglRenderDispatchComputePlan(
    void* compute_encoder,
    const MGLRenderComputePlan* plan,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!compute_encoder || !plan) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    const uint32_t local_x = plan->local_x ? plan->local_x : 1u;
    const uint32_t local_y = plan->local_y ? plan->local_y : 1u;
    const uint32_t local_z = plan->local_z ? plan->local_z : 1u;
    MTL::Size threads = MTL::Size(local_x, local_y, local_z);

    if (plan->dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_DIRECT) {
        encoder->dispatchThreadgroups(
            MTL::Size(plan->groups_x, plan->groups_y, plan->groups_z),
            threads);
        return 0;
    }
    if (plan->dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_INDIRECT) {
        MTL::Buffer* buffer = static_cast<MTL::Buffer*>(plan->indirect_buffer);
        if (!buffer) {
            if (err && errcap) {
                snprintf(err, errcap, "null indirect buffer");
            }
            return -1;
        }
        encoder->dispatchThreadgroups(
            buffer, static_cast<NS::UInteger>(plan->indirect_offset), threads);
        return 0;
    }
    if (err && errcap) snprintf(err, errcap, "bad dispatch kind %u",
                                (unsigned)plan->dispatch_kind);
    return -1;
}

int mglRenderAppendComputeDispatchToPlan(
    MGLRenderComputeExecutionPlan* plan,
    const MGLRenderComputePlan* dispatch,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!plan || !dispatch ||
        plan->dispatch_op_count >=
            MGL_RENDER_COMPUTE_EXECUTION_MAX_DISPATCHES) {
        if (err && errcap) snprintf(err, errcap, "compute dispatch sequence overflow");
        return -1;
    }
    if (dispatch->dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_DIRECT) {
        if (!dispatch->groups_x || !dispatch->groups_y || !dispatch->groups_z) {
            if (err && errcap) snprintf(err, errcap, "zero compute dispatch groups");
            return -1;
        }
    } else if (dispatch->dispatch_kind ==
               MGL_RENDER_COMPUTE_DISPATCH_INDIRECT) {
        if (!dispatch->indirect_buffer) {
            if (err && errcap) snprintf(err, errcap, "null indirect buffer");
            return -1;
        }
    } else {
        if (err && errcap) snprintf(err, errcap, "bad dispatch kind %u",
                                    dispatch->dispatch_kind);
        return -1;
    }
    MGLRenderComputeDispatchEntry* entry =
        &plan->dispatch_ops[plan->dispatch_op_count++];
    entry->binding_op_count = plan->binding_op_count;
    entry->dispatch = *dispatch;
    return 0;
}

int mglRenderDispatchComputeThreads(void* compute_encoder,
                                       uint32_t threads_x,
                                       uint32_t threads_y,
                                       uint32_t threads_z,
                                       uint32_t group_x,
                                       uint32_t group_y,
                                       uint32_t group_z) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder || threads_x == 0 || threads_y == 0 || threads_z == 0 ||
        group_x == 0 || group_y == 0 || group_z == 0) {
        return -1;
    }
    MTL::Size threads = MTL::Size(threads_x, threads_y, threads_z);
    MTL::Size threadgroup = MTL::Size(group_x, group_y, group_z);
    encoder->dispatchThreads(threads, threadgroup);
    return 0;
}

int mglRenderBeginComputeDispatch(
    void* command_buffer,
    const MGLRenderComputeDispatchSetup* setup,
    void** compute_encoder_out,
    char* err,
    size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (compute_encoder_out) *compute_encoder_out = nullptr;
    if (!command_buffer || !setup || !compute_encoder_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    if (setup->buffer_count > MGL_RENDER_COMPUTE_DISPATCH_MAX_BUFFERS ||
        setup->bytes_count > MGL_RENDER_COMPUTE_DISPATCH_MAX_BYTES) {
        if (err && errcap) snprintf(err, errcap, "setup count overflow");
        return -1;
    }
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    MTL::ComputeCommandEncoder* encoder = command->computeCommandEncoder();
    if (!encoder) {
        if (err && errcap) snprintf(err, errcap, "compute encoder failed");
        return -1;
    }
    if (setup->pipeline) {
        encoder->setComputePipelineState(
            static_cast<MTL::ComputePipelineState*>(setup->pipeline));
    }
    for (uint32_t i = 0; i < setup->buffer_count; i++) {
        const MGLRenderComputeBufferEntry* entry = &setup->buffers[i];
        if (!entry->buffer) {
            encoder->endEncoding();
            if (err && errcap) {
                snprintf(err, errcap, "null compute buffer entry %u", i);
            }
            return -1;
        }
        encoder->setBuffer(
            static_cast<MTL::Buffer*>(entry->buffer),
            static_cast<NS::UInteger>(entry->offset), entry->index);
    }
    for (uint32_t i = 0; i < setup->bytes_count; i++) {
        const MGLRenderComputeBytesEntry* entry = &setup->bytes[i];
        if (!entry->bytes || entry->length == 0) {
            encoder->endEncoding();
            if (err && errcap) {
                snprintf(err, errcap, "null compute bytes entry %u", i);
            }
            return -1;
        }
        encoder->setBytes(entry->bytes, entry->length, entry->index);
    }
    *compute_encoder_out = encoder;
    return 0;
}

int mglRenderEndComputeDispatch(void* compute_encoder,
                                   const uint32_t groups[3],
                                   const uint32_t threads[3],
                                   char* err,
                                   size_t errcap) {
    if (err && errcap) err[0] = '\0';
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder || !groups || !threads) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    MTL::Size groupsSize =
        MTL::Size(groups[0], groups[1], groups[2]);
    MTL::Size threadsSize =
        MTL::Size(threads[0], threads[1], threads[2]);
    encoder->dispatchThreadgroups(groupsSize, threadsSize);
    encoder->endEncoding();
    return 0;
}

int mglRenderCreateComputeEncoder(void* command_buffer,
                                     void** compute_encoder_out) {
    if (compute_encoder_out) *compute_encoder_out = nullptr;
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command || !compute_encoder_out) return -1;
    MTL::ComputeCommandEncoder* encoder = command->computeCommandEncoder();
    if (!encoder) return -1;
    *compute_encoder_out = encoder;
    return 0;
}

int mglRenderEndComputeEncoder(void* compute_encoder) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->endEncoding();
    return 0;
}

int mglRenderCreateCommandBuffer(void* command_queue,
                                    void** command_buffer_out) {
    if (command_buffer_out) *command_buffer_out = nullptr;
    MTL::CommandQueue* queue =
        static_cast<MTL::CommandQueue*>(command_queue);
    if (!queue || !command_buffer_out) return -1;
    MTL::CommandBuffer* commandBuffer = queue->commandBuffer();
    if (!commandBuffer) return -1;
    *command_buffer_out = commandBuffer;
    return 0;
}

const char *mglRenderCommandBufferErrorDescription(
    const MGLRenderCommandBufferState* state) {
    return state && state->has_error && state->error_description[0]
        ? state->error_description : "unknown command-buffer error";
}

uint32_t mglRenderCommandBufferStatus(void* command_buffer) {
    MGLRenderCommandBufferState state = {};
    return mglRenderGetCommandBufferState(command_buffer, &state) == 0
        ? state.status : static_cast<uint32_t>(MTL::CommandBufferStatusError);
}

int mglRenderGetCommandBufferLabel(const void *command_buffer,
                                      char *label_out,
                                      size_t label_capacity) {
    if (label_out && label_capacity) label_out[0] = '\0';
    const MTL::CommandBuffer *cb =
        static_cast<const MTL::CommandBuffer *>(command_buffer);
    if (!cb || !label_out || !label_capacity) return -1;
    NS::String *label = cb->label();
    const char *utf8 = label ? label->utf8String() : nullptr;
    std::snprintf(label_out, label_capacity, "%s",
                  utf8 && utf8[0] ? utf8 : "(no-label)");
    return 0;
}

int mglRenderSetCommandBufferLabel(void *command_buffer,
                                      const char *label) {
    MTL::CommandBuffer* object = static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!object || !label) return -1;
    object->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
    return 0;
}

int mglRenderClassifyCommandBufferCommit(
    const MGLRenderCommandBufferState* state,
    MGLRenderCommandBufferCommitDecision* decision_out) {
    if (decision_out) memset(decision_out, 0, sizeof(*decision_out));
    if (!state || !decision_out) return -1;

    /* Preserve commitCommandBufferWithAGXRecovery's original ordering. Since
     * Error follows Committed numerically, Error is classified as the legacy
     * already-committed skip rather than changing recovery behavior here. */
    if (state->status >=
        static_cast<uint32_t>(MTL::CommandBufferStatusCommitted)) {
        decision_out->action =
            MGL_RENDER_COMMAND_BUFFER_COMMIT_SKIP_ALREADY_COMMITTED;
    } else {
        decision_out->action =
            MGL_RENDER_COMMAND_BUFFER_COMMIT_PROCEED;
    }
    return 0;
}

int mglRenderClassifyCommandBufferCompletion(
    const MGLRenderCommandBufferState* state,
    MGLRenderCommandBufferCompletionDecision* decision_out) {
    if (decision_out) memset(decision_out, 0, sizeof(*decision_out));
    if (!state || !decision_out) return -1;

    decision_out->has_error = state->has_error != 0;
    decision_out->is_driver_rejection =
        decision_out->has_error &&
        strncmp(state->error_domain, "MTLCommandBufferErrorDomain",
                sizeof(state->error_domain)) == 0 &&
        state->error_code == 4;
    return 0;
}

int mglRenderProcessCommandBufferCompletion(
    MGLCommandBufferRecoveryOwner* owner_handle,
    const MGLRenderCommandBufferState* state,
    double now,
    MGLRenderCommandBufferCompletionResult* result_out) {
    if (result_out) memset(result_out, 0, sizeof(*result_out));
    if (!owner_handle || !state || !result_out) return -1;
    if (mglRenderClassifyCommandBufferCompletion(
            state, &result_out->decision) != 0) {
        return -1;
    }

    if (result_out->decision.has_error) {
        return mglRenderCommandRecoveryRecordError(
            owner_handle, now, &result_out->state);
    }

    MGLRenderCommandRecoverySuccess success = {};
    if (mglRenderCommandRecoveryRecordSuccess(
            owner_handle, now, &success) != 0) {
        return -1;
    }
    result_out->state = success.state;
    result_out->sustained_recovery = success.sustained_recovery;
    result_out->recovered_successes = success.recovered_successes;
    result_out->previous_errors = success.previous_errors;

    /* Keep this as a distinct owner operation. The legacy completion path
     * acquired its recovery lock once in recordGPUSuccess and once again to
     * clear recovery mode on the first successful completion. */
    int cleared = mglRenderCommandRecoveryClearMode(owner_handle);
    if (cleared < 0) return -1;
    result_out->cleared_recovery_mode = (uint32_t)cleared;
    if (cleared == 1) result_out->state.recovery_mode = 0;
    return 0;
}


int mglRenderCommitCommandBuffer(void* command_buffer) {
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command) return -1;
    command->commit();
    return 0;
}

int mglRenderWaitCommandBuffer(void* command_buffer) {
    MGLRenderCommandBufferState state = {};
    return mglRenderWaitCommandBufferState(command_buffer, &state);
}

int mglRenderCommandBufferOwnerHasState(
    MGLCommandBufferOwner* owner_handle,
    MGLRenderCommandBufferState* state_out) {
    return owner_handle && state_out &&
               mglRenderGetCommandBufferOwnerState(owner_handle, state_out) == 0
        ? 1 : 0;
}

void* mglRenderCreateRenderEncoderBorrowed(MGLCommandBufferOwner * command_buffer_owner, const MGLRenderPassState* render_pass) {
    void* encoder = nullptr;
    return mglRenderCreateRenderEncoderFromCommandBufferOwnerState(
               command_buffer_owner, render_pass, &encoder) == 0
        ? encoder : nullptr;
}

void* mglRenderCreateBlitEncoderBorrowed(MGLCommandBufferOwner * command_buffer_owner) {
    void* encoder = nullptr;
    return mglRenderCreateBlitEncoderFromCommandBufferOwner(
               command_buffer_owner, &encoder) == 0
        ? encoder : nullptr;
}

void* mglRenderCreateComputeEncoderBorrowed(MGLCommandBufferOwner * command_buffer_owner) {
    void* encoder = nullptr;
    return mglRenderCreateComputeEncoderFromCommandBufferOwner(
               command_buffer_owner, &encoder) == 0
        ? encoder : nullptr;
}

uint32_t mglRenderPassLoadActionForTrace(
    MGLRenderPassStateOwner* owner_handle, uint32_t attachment_kind, uint32_t color_index,
    uint32_t default_load_action) {
    uint32_t load_action = 0;
    return mglRenderGetRenderPassAttachmentActionsOwner(
               owner_handle, attachment_kind, color_index,
               &load_action, nullptr, nullptr) == 0
        ? load_action : default_load_action;
}

uint32_t mglRenderPassStoreActionForTrace(
    MGLRenderPassStateOwner* owner_handle, uint32_t attachment_kind, uint32_t color_index,
    uint32_t default_store_action) {
    uint32_t store_action = 0;
    return mglRenderGetRenderPassAttachmentActionsOwner(
               owner_handle, attachment_kind, color_index,
               nullptr, &store_action, nullptr) == 0
        ? store_action : default_store_action;
}

int mglRenderEndRenderEncoder(void* render_encoder) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->endEncoding();
    return 0;
}

int mglRenderCreateBlitEncoder(void* command_buffer,
                                  void** blit_encoder_out) {
    if (blit_encoder_out) *blit_encoder_out = nullptr;
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command || !blit_encoder_out) return -1;
    MTL::BlitCommandEncoder* encoder = command->blitCommandEncoder();
    if (!encoder) return -1;
    *blit_encoder_out = encoder;
    return 0;
}

int mglRenderEndBlitEncoder(void* blit_encoder) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    if (!encoder) return -1;
    encoder->endEncoding();
    return 0;
}

int mglRenderBlitCopyBuffer(void* blit_encoder,
                               void* source_buffer,
                               uint64_t source_offset,
                               void* destination_buffer,
                               uint64_t destination_offset,
                               uint64_t size) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Buffer* source = static_cast<MTL::Buffer*>(source_buffer);
    MTL::Buffer* destination =
        static_cast<MTL::Buffer*>(destination_buffer);
    if (!encoder || !source || !destination || size == 0) return -1;
    encoder->copyFromBuffer(source, static_cast<NS::UInteger>(source_offset),
                            destination,
                            static_cast<NS::UInteger>(destination_offset),
                            static_cast<NS::UInteger>(size));
    return 0;
}

int mglRenderBlitGenerateMipmaps(void* blit_encoder,
                                    void* texture) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!encoder || !source) return -1;
    encoder->generateMipmaps(source);
    return 0;
}

int mglRenderEncodeDraw(void* render_encoder,
                           const MGLRenderDrawPlan* plan,
                           char* err,
                           size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!render_encoder || !plan) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    switch (plan->kind) {
        case MGL_RENDER_DRAW_ARRAY:
            return mglRenderDrawPrimitives(
                render_encoder, plan->primitive_type,
                plan->vertex_start, plan->vertex_count,
                plan->instance_count, plan->base_instance);
        case MGL_RENDER_DRAW_INDEXED:
            return mglRenderDrawIndexedPrimitives(
                render_encoder, plan->primitive_type,
                plan->index_count, plan->index_type, plan->index_buffer,
                plan->index_buffer_offset, plan->instance_count,
                plan->base_vertex, plan->base_instance);
        case MGL_RENDER_DRAW_ARRAY_INDIRECT:
            return mglRenderDrawPrimitivesIndirect(
                render_encoder, plan->primitive_type,
                plan->indirect_buffer, plan->indirect_buffer_offset);
        case MGL_RENDER_DRAW_INDEXED_INDIRECT:
            return mglRenderDrawIndexedPrimitivesIndirect(
                render_encoder, plan->primitive_type, plan->index_type,
                plan->index_buffer, plan->index_buffer_offset,
                plan->indirect_buffer, plan->indirect_buffer_offset);
        case MGL_RENDER_DRAW_PATCHES:
            return mglRenderDrawPatches(
                render_encoder, plan->control_point_count, plan->patch_start,
                plan->patch_count, plan->patch_index_buffer,
                plan->patch_index_buffer_offset, plan->instance_count,
                plan->base_instance);
        case MGL_RENDER_DRAW_INDEXED_PATCHES:
            return mglRenderDrawIndexedPatches(
                render_encoder, plan->control_point_count, plan->patch_start,
                plan->patch_count, plan->patch_index_buffer,
                plan->patch_index_buffer_offset,
                plan->control_point_index_buffer,
                plan->control_point_index_buffer_offset,
                plan->instance_count, plan->base_instance);
        default:
            if (err && errcap) {
                snprintf(err, errcap, "unknown draw plan kind %u",
                         (unsigned)plan->kind);
            }
            return -1;
    }
}

int mglRenderResetIndirectCommandBuffer(void* indirect_buffer,
                                           uint64_t location,
                                           uint64_t length) {
    MTL::IndirectCommandBuffer* buffer =
        static_cast<MTL::IndirectCommandBuffer*>(indirect_buffer);
    if (!buffer || length == 0) return -1;
    buffer->reset(NS::Range(location, length));
    return 0;
}

int mglRenderGetIndirectRenderCommand(void* indirect_buffer,
                                         uint64_t command_index,
                                         void** command_out) {
    if (command_out) *command_out = nullptr;
    MTL::IndirectCommandBuffer* buffer =
        static_cast<MTL::IndirectCommandBuffer*>(indirect_buffer);
    if (!buffer || !command_out) return -1;
    MTL::IndirectRenderCommand* command =
        buffer->indirectRenderCommand(command_index);
    if (!command) return -1;
    *command_out = command;
    return 0;
}

bool mglRenderPassAttachmentMatchesSubresource(
    const void *descriptor,
    const MGLMetalAttachmentSubresource *subresource) {
    const MTL::RenderPassAttachmentDescriptor *attachment =
        static_cast<const MTL::RenderPassAttachmentDescriptor *>(descriptor);
    if (!attachment || !subresource) return false;
    return static_cast<uint64_t>(attachment->level()) == subresource->level &&
           static_cast<uint64_t>(attachment->slice()) == subresource->slice &&
           static_cast<uint64_t>(attachment->depthPlane()) == subresource->depthPlane;
}

const char *mglRenderCommandBufferStatusName(uint32_t status) {
    switch (static_cast<MTL::CommandBufferStatus>(status)) {
        case MTL::CommandBufferStatusNotEnqueued: return "NotEnqueued";
        case MTL::CommandBufferStatusEnqueued: return "Enqueued";
        case MTL::CommandBufferStatusCommitted: return "Committed";
        case MTL::CommandBufferStatusScheduled: return "Scheduled";
        case MTL::CommandBufferStatusCompleted: return "Completed";
        case MTL::CommandBufferStatusError: return "Error";
        default: return "Unknown";
    }
}

void mglRenderInitDefaultRenderPassState(
    MGLRenderPassState* state_out) {
    if (!state_out) return;
    *state_out = mgl::defaultRenderPassState();
}

int mglRenderBeginComputeDispatchForCommandBufferOwner(MGLCommandBufferOwner * command_buffer_owner, const MGLRenderComputeDispatchSetup* setup, void** compute_encoder_out, char* err, size_t errcap) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    if (!owner || !owner->current) {
        if (err && errcap) snprintf(err, errcap, "missing current command buffer");
        if (compute_encoder_out) *compute_encoder_out = nullptr;
        return -1;
    }
    return mglRenderBeginComputeDispatch(
        owner->current, setup, compute_encoder_out, err, errcap);
}

int mglRenderGetCommandBufferState(
    void* command_buffer,
    MGLRenderCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    return mgl::snapshotCommandBufferState(
        static_cast<MTL::CommandBuffer*>(command_buffer), state_out);
}

int mglRenderCreateCommandRecoveryOwner(MGLCommandBufferRecoveryOwner ** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    mgl::CommandBufferRecoveryOwner* owner =
        new (std::nothrow) mgl::CommandBufferRecoveryOwner();
    if (!owner) return -1;
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    return 0;
}

void mglRenderDestroyCommandRecoveryOwner(MGLCommandBufferRecoveryOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::CommandBufferRecoveryOwner* owner =
        reinterpret_cast<mgl::CommandBufferRecoveryOwner*>(*owner_handle);
    *owner_handle = nullptr;
    releaseCommandRecoveryOwner(owner);
}

int mglRenderCommandRecoveryRecordError(MGLCommandBufferRecoveryOwner * owner_handle, double now, MGLRenderCommandRecoverySnapshot* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    mgl::CommandBufferRecoveryOwner* owner =
        reinterpret_cast<mgl::CommandBufferRecoveryOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !state_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->consecutiveErrors++;
    owner->consecutiveSuccesses = 0;
    owner->lastErrorTime = now;
    mgl::snapshotCommandRecovery(*owner, state_out);
    return 0;
}

int mglRenderCommandRecoveryRecordSuccess(MGLCommandBufferRecoveryOwner * owner_handle, double now, MGLRenderCommandRecoverySuccess* result_out) {
    if (result_out) memset(result_out, 0, sizeof(*result_out));
    mgl::CommandBufferRecoveryOwner* owner =
        reinterpret_cast<mgl::CommandBufferRecoveryOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !result_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (owner->consecutiveErrors > 0 || owner->recoveryMode) {
        owner->consecutiveSuccesses++;
        if (owner->consecutiveSuccesses >= 4 &&
            now - owner->lastErrorTime > 0.25) {
            result_out->sustained_recovery = 1;
            result_out->recovered_successes = owner->consecutiveSuccesses;
            result_out->previous_errors = owner->consecutiveErrors;
            owner->consecutiveErrors = 0;
            owner->recoveryMode = false;
            owner->consecutiveSuccesses = 0;
        }
    }
    mgl::snapshotCommandRecovery(*owner, &result_out->state);
    return 0;
}

int mglRenderCommandRecoveryShouldSkip(MGLCommandBufferRecoveryOwner * owner_handle, double now, MGLRenderCommandRecoverySkipDecision* decision_out) {
    if (decision_out) memset(decision_out, 0, sizeof(*decision_out));
    mgl::CommandBufferRecoveryOwner* owner =
        reinterpret_cast<mgl::CommandBufferRecoveryOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !decision_out) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (now - owner->lastErrorTime > 3.0) {
        decision_out->recovery_timed_out = 1;
        decision_out->previous_errors = owner->consecutiveErrors;
        owner->consecutiveErrors = 0;
        owner->recoveryMode = false;
    } else if (owner->consecutiveErrors >= 8 || owner->recoveryMode) {
        decision_out->should_skip = 1;
        if (!owner->recoveryMode) {
            owner->recoveryMode = true;
            decision_out->entered_recovery_mode = 1;
        }
    }
    mgl::snapshotCommandRecovery(*owner, &decision_out->state);
    return 0;
}

int mglRenderCommandRecoveryTakeResetRequest(MGLCommandBufferRecoveryOwner * recovery_owner) {
    mgl::CommandBufferRecoveryOwner* owner =
        reinterpret_cast<mgl::CommandBufferRecoveryOwner*>(static_cast<void*>(recovery_owner));
    if (!owner) return -1;
    std::lock_guard<std::mutex> lock(owner->mutex);
    if (!owner->resetRequested) return 0;
    owner->resetRequested = false;
    return 1;
}

int mglRenderAddCommandBufferCompletion(
    void* command_buffer,
    MGLRenderCommandBufferCompletion callback,
    void* context,
    MGLRenderDestroyContext destroy_context) {
    MTL::CommandBuffer* commandBuffer =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!commandBuffer || !callback) return -1;

    mgl::CommandBufferCompletionContext* completion =
        new (std::nothrow) mgl::CommandBufferCompletionContext();
    if (!completion) return -1;
    completion->configure(callback, context, destroy_context);
    /* The block captures only a raw pointer. A separate reference belongs to
     * the handler, so a completion that runs before addCompletedHandler
     * returns cannot race a block copy helper or destroy this context early. */
    completion->retain();
    MTL::CommandBufferHandler stackHandler =
        ^(MTL::CommandBuffer* completedBuffer) {
            completion->complete(completedBuffer);
            completion->release();
        };
#ifdef __OBJC__
    MTL::CommandBufferHandler handler = [stackHandler copy];
#else
    MTL::CommandBufferHandler handler = Block_copy(stackHandler);
#endif
    if (!handler) {
        completion->abandonCallerContext();
        completion->release();
        completion->release();
        return -1;
    }
    try {
        commandBuffer->addCompletedHandler(handler);
    } catch (...) {
#ifndef __OBJC__
        Block_release(handler);
#endif
        completion->abandonCallerContext();
        completion->release();
        completion->release();
        return -1;
    }
#ifndef __OBJC__
    Block_release(handler);
#endif
    completion->release();
    return 0;
}

int mglRenderAddCommandBufferOwnerCompletion(MGLCommandBufferOwner * owner_handle, MGLRenderCommandBufferCompletion callback, void* context, MGLRenderDestroyContext destroy_context) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !owner->current) return -1;
    return mglRenderAddCommandBufferCompletion(
        owner->current, callback, context, destroy_context);
}

int mglRenderCreateCommandBufferOwner(void* command_queue, MGLCommandBufferOwner ** owner_out, void** command_buffer_out) {
    if (owner_out) *owner_out = nullptr;
    if (command_buffer_out) *command_buffer_out = nullptr;
    MTL::CommandQueue* queue =
        static_cast<MTL::CommandQueue*>(command_queue);
    if (!queue || !owner_out || !command_buffer_out) return -1;
    mgl::CommandBufferOwner* owner =
        new (std::nothrow) mgl::CommandBufferOwner();
    if (!owner) return -1;
    queue->retain();
    owner->queue = queue;
    MTL::CommandBuffer* commandBuffer = queue->commandBuffer();
    if (!commandBuffer) {
        delete owner;
        return -1;
    }
    commandBuffer->retain();
    owner->current = commandBuffer;
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    *command_buffer_out = commandBuffer;
    return 0;
}

int mglRenderResetCommandBufferOwner(MGLCommandBufferOwner * owner_handle, void* command_queue, void** command_buffer_out) {
    if (command_buffer_out) *command_buffer_out = nullptr;
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    MTL::CommandQueue* queue =
        static_cast<MTL::CommandQueue*>(command_queue);
    if (!owner || !queue || !command_buffer_out) return -1;
    if (owner->queue != queue) {
        queue->retain();
        if (owner->queue) owner->queue->release();
        owner->queue = queue;
    }
    MTL::CommandBuffer* commandBuffer = queue->commandBuffer();
    if (!commandBuffer) return -1;
    commandBuffer->retain();
    if (owner->current) owner->current->release();
    owner->current = commandBuffer;
    owner->transaction_created_current = false;
    owner->syncs.reset();
    *command_buffer_out = commandBuffer;
    return 0;
}

extern "C"
int mglRenderCreateCommandBufferOwnerAdopt(void* command_buffer, MGLCommandBufferOwner ** owner_out) {
    if (owner_out) *owner_out = nullptr;
    MTL::CommandBuffer* commandBuffer =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!commandBuffer || !owner_out) return -1;
    mgl::CommandBufferOwner* owner = new (std::nothrow) mgl::CommandBufferOwner();
    if (!owner) return -1;
    commandBuffer->retain();
    owner->current = commandBuffer;
    owner->transaction_created_current = false;
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    return 0;
}

extern "C"
void* mglRenderCommandBufferOwnerGetCurrent(MGLCommandBufferOwner * owner_handle) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    return owner ? static_cast<void*>(owner->current) : nullptr;
}

int mglRenderCommandBufferOwnerHasCurrent(MGLCommandBufferOwner * owner_handle) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    return owner ? (owner->current ? 1 : 0) : -1;
}

int mglRenderCommandBufferOwnerCreateNext(MGLCommandBufferOwner * owner_handle, void** command_buffer_out) {
    if (command_buffer_out) *command_buffer_out = nullptr;
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !command_buffer_out) return -1;
    if (!owner->queue) return 1;
    if (owner->current) {
        /* A direct commit path can leave the committed object in the owner
         * instead of transferring a submission handle.  Drop that owner
         * reference before rotating; an unfinalized current buffer must stay
         * untouched because callers may still be encoding into it. */
        MTL::CommandBufferStatus status = owner->current->status();
        if (status < MTL::CommandBufferStatusCommitted) {
            owner->transaction_created_current = false;
            *command_buffer_out = owner->current;
            return 0;
        }
        owner->current->release();
        owner->current = nullptr;
    }
    MTL::CommandBuffer* commandBuffer = owner->queue->commandBuffer();
    if (!commandBuffer) return -1;
    commandBuffer->retain();
    owner->current = commandBuffer;
    owner->transaction_created_current = false;
    owner->syncs.reset();
    *command_buffer_out = commandBuffer;
    return 0;
}

int mglRenderGetCommandBufferOwnerState(MGLCommandBufferOwner * owner_handle, MGLRenderCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !owner->current || !state_out) return -1;
    return mgl::snapshotCommandBufferState(owner->current, state_out);
}

int mglRenderCommandBufferOwnerHasLastSubmitted(MGLCommandBufferOwner * owner_handle) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    return owner ? (owner->lastSubmitted ? 1 : 0) : -1;
}

int mglRenderWaitCommandBufferState(
    void* command_buffer,
    MGLRenderCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command || !state_out) return -1;

    MGLRenderCommandBufferState before = {};
    if (mgl::snapshotCommandBufferState(command, &before) != 0) return -1;
    if (before.status ==
        static_cast<uint32_t>(MTL::CommandBufferStatusNotEnqueued)) {
        *state_out = before;
        return 1;
    }
    try {
        if (before.status !=
            static_cast<uint32_t>(MTL::CommandBufferStatusCompleted)) {
            command->waitUntilCompleted();
        }
    } catch (...) {
        (void)mgl::snapshotCommandBufferState(command, state_out);
        return -1;
    }
    if (mgl::snapshotCommandBufferState(command, state_out) != 0) return -1;
    return state_out->has_error ? -1 : 0;
}

int mglRenderWaitCommandBufferOwnerLastSubmitted(MGLCommandBufferOwner * owner_handle, MGLRenderCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !state_out) return -1;
    MTL::CommandBuffer* commandBuffer = owner->lastSubmitted;
    if (!commandBuffer) return 1;
    return mglRenderWaitCommandBufferState(commandBuffer, state_out);
}

int mglRenderPresentDrawableForCommandBufferOwner(MGLCommandBufferOwner * owner_handle, void* drawable, MGLRenderCommandBufferState* state_out) {
    if (state_out) memset(state_out, 0, sizeof(*state_out));
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    MTL::Drawable* surface = static_cast<MTL::Drawable*>(drawable);
    if (!owner || !owner->current || !surface) return -1;

    MGLRenderCommandBufferState state = {};
    if (mgl::snapshotCommandBufferState(owner->current, &state) != 0) {
        return -1;
    }
    if (state_out) *state_out = state;
    if (state.status !=
        static_cast<uint32_t>(MTL::CommandBufferStatusNotEnqueued)) {
        return 1;
    }
    owner->current->presentDrawable(surface);
    return 0;
}

int mglRenderEncodeWaitForEventForCommandBufferOwner(MGLCommandBufferOwner * owner_handle, void* event, uint64_t value) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    MTL::Event* metal_event = static_cast<MTL::Event*>(event);
    if (!owner || !owner->current || !metal_event || value == 0u) return -1;
    MGLRenderCommandBufferState state = {};
    if (mgl::snapshotCommandBufferState(owner->current, &state) != 0 ||
        state.status !=
            static_cast<uint32_t>(MTL::CommandBufferStatusNotEnqueued)) {
        return -1;
    }
    owner->current->encodeWait(metal_event, value);
    return 0;
}

void mglRenderDiscardCommandBufferOwnerCurrent(MGLCommandBufferOwner * owner_handle) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !owner->current) return;
    owner->current->release();
    owner->current = nullptr;
    owner->transaction_created_current = false;
    owner->syncs.reset();
}

int mglRenderTakeCommandBufferSubmission(MGLCommandBufferOwner * owner_handle, void** submission_out, void** command_buffer_out) {
    if (submission_out) *submission_out = nullptr;
    if (command_buffer_out) *command_buffer_out = nullptr;
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !owner->current || !submission_out ||
        !command_buffer_out) {
        return -1;
    }
    mgl::CommandBufferSubmission* submission =
        new (std::nothrow) mgl::CommandBufferSubmission();
    if (!submission) return -1;
    submission->buffer = owner->current;
    owner->current = nullptr;
    owner->transaction_created_current = false;
    *submission_out = submission;
    *command_buffer_out = submission->buffer;
    return 0;
}

int mglRenderCommitCommandBufferSubmission(void** submission_handle) {
    if (!submission_handle || !*submission_handle) return -1;
    mgl::CommandBufferSubmission* submission =
        static_cast<mgl::CommandBufferSubmission*>(*submission_handle);
    if (!submission->buffer) return -1;
    submission->buffer->commit();
    *submission_handle = nullptr;
    delete submission;
    return 0;
}

int mglRenderCommandBufferSubmissionMatchesBuffer(
    void* submission_handle, void* command_buffer) {
    mgl::CommandBufferSubmission* submission =
        static_cast<mgl::CommandBufferSubmission*>(submission_handle);
    if (!submission || !command_buffer) return -1;
    return submission->buffer ==
                   static_cast<MTL::CommandBuffer*>(command_buffer)
               ? 1
               : 0;
}

void mglRenderDestroyCommandBufferSubmission(void** submission_handle) {
    if (!submission_handle || !*submission_handle) return;
    mgl::CommandBufferSubmission* submission =
        static_cast<mgl::CommandBufferSubmission*>(*submission_handle);
    *submission_handle = nullptr;
    delete submission;
}

int mglRenderCommandBufferOwnerAppendSync(MGLCommandBufferOwner * owner_handle, Sync* sync) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !sync) return -1;
    mgl::CommandBufferSyncList& list = owner->syncs;
    if (list.count >= list.size) {
        const uint32_t old_size = list.size;
        const uint32_t new_size =
            old_size ? (old_size > (UINT32_MAX / 2) ? 0u : old_size * 2u) : 8u;
        if (new_size == 0u ||
            new_size > (UINT32_MAX / sizeof(Sync*))) {
            return -1;
        }
        Sync** new_list = (Sync**)realloc(
            list.list, sizeof(Sync*) * (size_t)new_size);
        if (!new_list) return -1;
        list.list = new_list;
        list.size = new_size;
    }
    list.list[list.count++] = sync;
    return 0;
}

int mglRenderCommandBufferOwnerBeginCommit(MGLCommandBufferOwner * owner_handle) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    if (owner->commit_in_progress) return 0;
    owner->commit_in_progress = true;
    return 1;
}

void mglRenderCommandBufferOwnerEndCommit(MGLCommandBufferOwner * owner_handle) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return;
    owner->commit_in_progress = false;
}

extern "C"
int mglRenderCommandBufferOwnerConsumeTransactionCurrent(MGLCommandBufferOwner * owner_handle, void** command_buffer_out) {
    if (command_buffer_out) *command_buffer_out = nullptr;
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !command_buffer_out) return -1;
    if (!owner->transaction_created_current || !owner->current) return 0;
    MGLRenderCommandBufferState state = {};
    if (mgl::snapshotCommandBufferState(owner->current, &state) != 0 ||
        state.status != static_cast<uint32_t>(MTL::CommandBufferStatusNotEnqueued)) {
        owner->transaction_created_current = false;
        return 0;
    }
    owner->transaction_created_current = false;
    owner->syncs.reset();
    *command_buffer_out = owner->current;
    return 1;
}

void mglRenderDestroyCommandBufferOwner(MGLCommandBufferOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateCommandQueueOwner(uint32_t max_command_buffers, MGLCommandQueueOwner ** owner_out, void** command_queue_out) {
    if (owner_out) *owner_out = nullptr;
    if (command_queue_out) *command_queue_out = nullptr;
    if (!owner_out || !command_queue_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    mgl::CommandQueueOwner* owner =
        new (std::nothrow) mgl::CommandQueueOwner();
    if (!owner) return -1;
    MTL::CommandQueue* queue = max_command_buffers
        ? renderer.device->newCommandQueue(max_command_buffers)
        : renderer.device->newCommandQueue();
    if (!queue) {
        delete owner;
        return -1;
    }
    owner->queue = queue;
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    *command_queue_out = queue;
    return 0;
}

int mglRenderResetCommandQueueOwner(MGLCommandQueueOwner * owner_handle, uint32_t max_command_buffers, void** command_queue_out) {
    if (command_queue_out) *command_queue_out = nullptr;
    mgl::CommandQueueOwner* owner =
        reinterpret_cast<mgl::CommandQueueOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !command_queue_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::CommandQueue* queue = max_command_buffers
        ? renderer.device->newCommandQueue(max_command_buffers)
        : renderer.device->newCommandQueue();
    if (!queue) return -1;
    if (owner->queue) owner->queue->release();
    owner->queue = queue;
    *command_queue_out = queue;
    return 0;
}

void mglRenderDestroyCommandQueueOwner(MGLCommandQueueOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::CommandQueueOwner* owner =
        reinterpret_cast<mgl::CommandQueueOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateRenderEncoderFromState(
    void* command_buffer,
    const MGLRenderPassState* render_pass,
    void** render_encoder_out) {
    if (render_encoder_out) *render_encoder_out = nullptr;
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!command || !render_pass || !render_encoder_out) return -1;
    MTL::RenderPassDescriptor* descriptor =
        mgl::newRenderPassDescriptor(render_pass);
    if (!descriptor) return -1;
    MTL::RenderCommandEncoder* encoder =
        command->renderCommandEncoder(descriptor);
    descriptor->release();
    if (!encoder) return -1;
    *render_encoder_out = encoder;
    return 0;
}

int mglRenderEncodeMultisampleResolve(
    void* command_buffer,
    uint32_t attachment_kind,
    void* source_texture,
    uint64_t source_level,
    uint64_t source_slice,
    uint64_t source_depth_plane,
    void* resolve_texture,
    uint64_t resolve_level,
    uint64_t resolve_slice,
    uint64_t resolve_depth_plane,
    uint32_t resolve_filter) {
    if (!command_buffer || !source_texture || !resolve_texture) return -1;
    MGLRenderPassState state = mgl::defaultRenderPassState();
    MGLRenderPassAttachmentState* attachment = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
        attachment = &state.color[0].attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
        attachment = &state.depth.attachment;
        state.depth.resolve_filter = resolve_filter;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
        attachment = &state.stencil.attachment;
        state.stencil.resolve_filter = resolve_filter;
        break;
    default:
        return -1;
    }
    attachment->texture = source_texture;
    attachment->level = source_level;
    attachment->slice = source_slice;
    attachment->depth_plane = source_depth_plane;
    attachment->resolve_texture = resolve_texture;
    attachment->resolve_level = resolve_level;
    attachment->resolve_slice = resolve_slice;
    attachment->resolve_depth_plane = resolve_depth_plane;
    attachment->load_action = static_cast<uint32_t>(MTL::LoadActionLoad);
    attachment->store_action =
        static_cast<uint32_t>(MTL::StoreActionMultisampleResolve);
    void* encoder_handle = nullptr;
    if (mglRenderCreateRenderEncoderFromState(
            command_buffer, &state, &encoder_handle) != 0 ||
        !encoder_handle) {
        return -1;
    }
    static_cast<MTL::RenderCommandEncoder*>(encoder_handle)->endEncoding();
    return 0;
}

int mglRenderEncodeMultisampleResolveForCommandBufferOwner(MGLCommandBufferOwner * command_buffer_owner, uint32_t attachment_kind, void* source_texture, uint64_t source_level, uint64_t source_slice, uint64_t source_depth_plane, void* resolve_texture, uint64_t resolve_level, uint64_t resolve_slice, uint64_t resolve_depth_plane, uint32_t resolve_filter) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    if (!owner || !owner->current) return -1;
    return mglRenderEncodeMultisampleResolve(
        owner->current, attachment_kind, source_texture, source_level,
        source_slice, source_depth_plane, resolve_texture, resolve_level,
        resolve_slice, resolve_depth_plane, resolve_filter);
}

int mglRenderCreateRenderEncoderOwner(void* render_encoder, MGLRenderEncoderOwner ** owner_out) {
    if (owner_out) *owner_out = nullptr;
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || !owner_out) return -1;
    mgl::RenderEncoderOwner* owner =
        new (std::nothrow) mgl::RenderEncoderOwner();
    if (!owner) return -1;
    encoder->retain();
    owner->encoder = encoder;
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    return 0;
}

int mglRenderResetRenderEncoderOwner(MGLRenderEncoderOwner * owner_handle, void* render_encoder) {
    mgl::RenderEncoderOwner* owner =
        reinterpret_cast<mgl::RenderEncoderOwner*>(static_cast<void*>(owner_handle));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!owner || !encoder) return -1;
    encoder->retain();
    if (owner->encoder) owner->encoder->release();
    owner->encoder = encoder;
    owner->ended = false;
    return 0;
}

int mglRenderEndRenderEncoderOwner(MGLRenderEncoderOwner * owner_handle) {
    mgl::RenderEncoderOwner* owner =
        reinterpret_cast<mgl::RenderEncoderOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    if (!owner->encoder) return owner->ended ? 0 : -1;
    if (!owner->ended) {
        owner->encoder->endEncoding();
        owner->ended = true;
    }
    owner->encoder->release();
    owner->encoder = nullptr;
    return 0;
}

int mglRenderEncoderOwnerHasCurrent(MGLRenderEncoderOwner * owner_handle) {
    mgl::RenderEncoderOwner* owner =
        reinterpret_cast<mgl::RenderEncoderOwner*>(static_cast<void*>(owner_handle));
    return owner && owner->encoder && !owner->ended ? 1 : 0;
}

int mglRenderSetRenderEncoderOwnerLabel(MGLRenderEncoderOwner * owner_handle, const char* label) {
    mgl::RenderEncoderOwner* owner =
        reinterpret_cast<mgl::RenderEncoderOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !owner->encoder || owner->ended || !label) return -1;
    owner->encoder->setLabel(
        NS::String::string(label, NS::UTF8StringEncoding));
    return 0;
}

void mglRenderDestroyRenderEncoderOwner(MGLRenderEncoderOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::RenderEncoderOwner* owner =
        reinterpret_cast<mgl::RenderEncoderOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateRenderPassIdentityOwner(MGLRenderPassIdentityOwner ** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    mgl::RenderPassIdentityOwner* owner =
        new (std::nothrow) mgl::RenderPassIdentityOwner();
    if (!owner) return -1;
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    return 0;
}

int mglRenderUpdateRenderPassIdentity(MGLRenderPassIdentityOwner * owner_handle, const MGLRenderPassIdentityState* state) {
    mgl::RenderPassIdentityOwner* owner =
        reinterpret_cast<mgl::RenderPassIdentityOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !state ||
        state->draw_buffer_count > MGL_RENDER_MAX_COLOR_ATTACHMENTS) {
        return -1;
    }
    owner->state = *state;
    owner->cache = {};
    owner->cache_valid = false;
    return 0;
}

int mglRenderGetRenderPassIdentity(MGLRenderPassIdentityOwner * owner_handle, MGLRenderPassIdentityState* state_out) {
    mgl::RenderPassIdentityOwner* owner =
        reinterpret_cast<mgl::RenderPassIdentityOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !state_out) return -1;
    *state_out = owner->state;
    return 0;
}

void mglRenderDestroyRenderPassIdentityOwner(MGLRenderPassIdentityOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::RenderPassIdentityOwner* owner =
        reinterpret_cast<mgl::RenderPassIdentityOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateRenderPassStateOwner(const MGLRenderPassState* state, MGLRenderPassStateOwner ** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!state || !owner_out ||
        state->sample_position_count > MGL_RENDER_MAX_SAMPLE_POSITIONS) {
        return -1;
    }
    mgl::RenderPassStateOwner* owner =
        new (std::nothrow) mgl::RenderPassStateOwner();
    if (!owner) return -1;
    owner->state = *state;
    mgl::retainRenderPassStateResources(owner->state);
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    return 0;
}

int mglRenderCreateDefaultRenderPassStateOwner(MGLRenderPassStateOwner ** owner_out) {
    MGLRenderPassState state = mgl::defaultRenderPassState();
    return mglRenderCreateRenderPassStateOwner(&state, owner_out);
}

int mglRenderSetRenderPassStateAttachment(MGLRenderPassStateOwner * owner_handle, uint32_t attachment_kind, uint32_t color_index, const MGLRenderPassAttachmentState* attachment) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !attachment) return -1;

    MGLRenderPassAttachmentState* destination = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
        if (color_index >= MGL_RENDER_MAX_COLOR_ATTACHMENTS) return -1;
        destination = &owner->state.color[color_index].attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
        destination = &owner->state.depth.attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
        destination = &owner->state.stencil.attachment;
        break;
    default:
        return -1;
    }

    MGLRenderPassAttachmentState next = *attachment;
    mgl::retainRenderPassObject(next.texture);
    mgl::retainRenderPassObject(next.resolve_texture);
    mgl::releaseRenderPassObject(destination->texture);
    mgl::releaseRenderPassObject(destination->resolve_texture);
    *destination = next;
    return 0;
}

int mglRenderSetRenderPassStateAttachmentActions(MGLRenderPassStateOwner * owner_handle, uint32_t attachment_kind, uint32_t color_index, uint32_t load_action, uint32_t store_action, uint64_t store_action_options) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;

    MGLRenderPassAttachmentState* attachment = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
        if (color_index >= MGL_RENDER_MAX_COLOR_ATTACHMENTS) return -1;
        attachment = &owner->state.color[color_index].attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
        attachment = &owner->state.depth.attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
        attachment = &owner->state.stencil.attachment;
        break;
    default:
        return -1;
    }

    attachment->load_action = load_action;
    attachment->store_action = store_action;
    attachment->store_action_options = store_action_options;
    return 0;
}

int mglRenderSetRenderPassStateColorClear(MGLRenderPassStateOwner * owner_handle, uint32_t color_index, double red, double green, double blue, double alpha) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || color_index >= MGL_RENDER_MAX_COLOR_ATTACHMENTS) {
        return -1;
    }
    MGLRenderPassColorState& color = owner->state.color[color_index];
    color.clear_red = red;
    color.clear_green = green;
    color.clear_blue = blue;
    color.clear_alpha = alpha;
    return 0;
}

int mglRenderSetRenderPassStateDepthClear(MGLRenderPassStateOwner * owner_handle, double clear_depth) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    owner->state.depth.clear_depth = clear_depth;
    return 0;
}

int mglRenderSetRenderPassStateStencilClear(MGLRenderPassStateOwner * owner_handle, uint32_t clear_stencil) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    owner->state.stencil.clear_stencil = clear_stencil;
    return 0;
}

int mglRenderSetRenderPassStateVisibility(MGLRenderPassStateOwner * owner_handle, void* visibility_result_buffer, uint32_t visibility_result_type) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    mgl::retainRenderPassObject(visibility_result_buffer);
    mgl::releaseRenderPassObject(owner->state.visibility_result_buffer);
    owner->state.visibility_result_buffer = visibility_result_buffer;
    owner->state.visibility_result_type = visibility_result_type;
    return 0;
}

int mglRenderSetRenderPassStateDimensions(MGLRenderPassStateOwner * owner_handle, uint64_t width, uint64_t height) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;
    owner->state.render_target_width = width;
    owner->state.render_target_height = height;
    return 0;
}

int mglRenderGetRenderPassStateOwner(MGLRenderPassStateOwner * owner_handle, MGLRenderPassState* state_out) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !state_out) return -1;
    *state_out = owner->state;
    return 0;
}

int mglRenderGetRenderPassAttachmentStateOwner(MGLRenderPassStateOwner * owner_handle, uint32_t attachment_kind, uint32_t color_index, MGLRenderPassAttachmentState* attachment_out) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !attachment_out) return -1;

    const MGLRenderPassAttachmentState* attachment = nullptr;
    switch (attachment_kind) {
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
            if (color_index >= MGL_RENDER_MAX_COLOR_ATTACHMENTS) return -1;
            attachment = &owner->state.color[color_index].attachment;
            break;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
            attachment = &owner->state.depth.attachment;
            break;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
            attachment = &owner->state.stencil.attachment;
            break;
        default:
            return -1;
    }
    *attachment_out = *attachment;
    return 0;
}

int mglRenderCreateRenderEncoderFromStateOwner(void* command_buffer, MGLRenderPassStateOwner * owner_handle, void** render_encoder_out) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner) {
        if (render_encoder_out) *render_encoder_out = nullptr;
        return -1;
    }
    return mglRenderCreateRenderEncoderFromState(
        command_buffer, &owner->state, render_encoder_out);
}

int mglRenderCreateRenderEncoderFromCommandBufferOwnerState(MGLCommandBufferOwner * command_buffer_owner, const MGLRenderPassState* render_pass, void** render_encoder_out) {
    if (render_encoder_out) *render_encoder_out = nullptr;
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    if (!owner || !owner->current) return -1;
    return mglRenderCreateRenderEncoderFromState(
        owner->current, render_pass, render_encoder_out);
}

void mglRenderDestroyRenderPassStateOwner(MGLRenderPassStateOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateBlitEncoderFromCommandBufferOwner(MGLCommandBufferOwner * command_buffer_owner, void** blit_encoder_out) {
    if (blit_encoder_out) *blit_encoder_out = nullptr;
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    if (!owner || !owner->current) return -1;
    return mglRenderCreateBlitEncoder(
        owner->current, blit_encoder_out);
}

int mglRenderEncodeBufferCopiesForCommandBufferOwner(MGLCommandBufferOwner * command_buffer_owner, const MGLRenderBufferCopyEntry* entries, uint32_t entry_count) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    if (!owner || !owner->current || !entries || entry_count == 0u) {
        return -1;
    }
    for (uint32_t i = 0; i < entry_count; ++i) {
        const MGLRenderBufferCopyEntry& entry = entries[i];
        MTL::Buffer* source = static_cast<MTL::Buffer*>(entry.source_buffer);
        MTL::Buffer* destination =
            static_cast<MTL::Buffer*>(entry.destination_buffer);
        if (!source || !destination || entry.length == 0u ||
            entry.source_offset > source->length() ||
            entry.length > source->length() - entry.source_offset ||
            entry.destination_offset > destination->length() ||
            entry.length > destination->length() - entry.destination_offset) {
            return -1;
        }
    }
    MTL::BlitCommandEncoder* encoder =
        owner->current->blitCommandEncoder();
    if (!encoder) return -1;
    for (uint32_t i = 0; i < entry_count; ++i) {
        const MGLRenderBufferCopyEntry& entry = entries[i];
        encoder->copyFromBuffer(
            static_cast<MTL::Buffer*>(entry.source_buffer),
            static_cast<NS::UInteger>(entry.source_offset),
            static_cast<MTL::Buffer*>(entry.destination_buffer),
            static_cast<NS::UInteger>(entry.destination_offset),
            static_cast<NS::UInteger>(entry.length));
    }
    encoder->endEncoding();
    return 0;
}

int mglRenderCreateComputeEncoderFromCommandBufferOwner(MGLCommandBufferOwner * command_buffer_owner, void** compute_encoder_out) {
    if (compute_encoder_out) *compute_encoder_out = nullptr;
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    if (!owner || !owner->current) return -1;
    return mglRenderCreateComputeEncoder(
        owner->current, compute_encoder_out);
}

int mglRenderCreateIndirectCommandBuffer(
    uint32_t command_types,
    int inherit_pipeline_state,
    int inherit_buffers,
    uint32_t max_vertex_buffer_bind_count,
    uint32_t max_fragment_buffer_bind_count,
    uint64_t max_command_count,
    uint64_t resource_options,
    void** indirect_buffer_out) {
    if (indirect_buffer_out) *indirect_buffer_out = nullptr;
    if (!indirect_buffer_out || max_command_count == 0) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::IndirectCommandBufferDescriptor* allocated =
        MTL::IndirectCommandBufferDescriptor::alloc();
    if (!allocated) return -1;
    MTL::IndirectCommandBufferDescriptor* descriptor = allocated->init();
    if (!descriptor) return -1;
    descriptor->setCommandTypes(
        static_cast<MTL::IndirectCommandType>(command_types));
    descriptor->setInheritPipelineState(inherit_pipeline_state != 0);
    descriptor->setInheritBuffers(inherit_buffers != 0);
    descriptor->setMaxVertexBufferBindCount(max_vertex_buffer_bind_count);
    descriptor->setMaxFragmentBufferBindCount(max_fragment_buffer_bind_count);
    MTL::IndirectCommandBuffer* buffer = renderer.device->newIndirectCommandBuffer(
        descriptor, static_cast<NS::UInteger>(max_command_count),
        static_cast<MTL::ResourceOptions>(resource_options));
    descriptor->release();
    if (!buffer) return -1;
    *indirect_buffer_out = buffer;
    return 0;
}



void mglRenderInvalidateRenderPass(GLMContext glm_ctx) {
    BackendLeaseScope lease(glm_ctx);
    MGLRenderEncoderOwner* render_owner =
        static_cast<MGLRenderEncoderOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_RENDER_ENCODER));
    if (!render_owner) return;
    if (mglRenderEncoderOwnerHasCurrent(render_owner) == 1) {
        (void)mglRenderEndRenderEncoderOwner(render_owner);
    }
}

static int mglRenderEncodeComputeDispatchOnEncoder(
    MTL::ComputeCommandEncoder* encoder,
    const MGLRenderComputePlan* dispatch,
    char* err,
    size_t errcap) {
    if (!encoder || !dispatch) return -1;
    const uint32_t local_x = dispatch->local_x ? dispatch->local_x : 1u;
    const uint32_t local_y = dispatch->local_y ? dispatch->local_y : 1u;
    const uint32_t local_z = dispatch->local_z ? dispatch->local_z : 1u;
    const MTL::Size threads(local_x, local_y, local_z);
    if (dispatch->dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_DIRECT) {
        if (!dispatch->groups_x || !dispatch->groups_y || !dispatch->groups_z) {
            if (err && errcap) snprintf(err, errcap, "zero compute dispatch groups");
            return -1;
        }
        encoder->dispatchThreadgroups(
            MTL::Size(dispatch->groups_x, dispatch->groups_y,
                      dispatch->groups_z),
            threads);
        return 0;
    }
    if (dispatch->dispatch_kind == MGL_RENDER_COMPUTE_DISPATCH_INDIRECT) {
        MTL::Buffer* indirect =
            static_cast<MTL::Buffer*>(dispatch->indirect_buffer);
        if (!indirect) {
            if (err && errcap) snprintf(err, errcap, "null indirect buffer");
            return -1;
        }
        encoder->dispatchThreadgroups(
            indirect, static_cast<NS::UInteger>(dispatch->indirect_offset),
            threads);
        return 0;
    }
    if (err && errcap) snprintf(err, errcap, "bad dispatch kind %u",
                                dispatch->dispatch_kind);
    return -1;
}

int mglRenderEncodeComputeExecutionPlanForCommandBufferOwner(MGLCommandBufferOwner * command_buffer_owner, const MGLRenderComputeExecutionPlan* plan, char* err, size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!command_buffer_owner || !plan || !plan->pipeline) {
        if (err && errcap) snprintf(err, errcap, "bad compute execution plan");
        return -1;
    }
    if (plan->binding_op_count > MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS) {
        if (err && errcap) snprintf(err, errcap, "compute execution op overflow");
        return -1;
    }
    if (plan->dispatch_op_count >
        MGL_RENDER_COMPUTE_EXECUTION_MAX_DISPATCHES) {
        if (err && errcap) snprintf(err, errcap, "compute dispatch sequence overflow");
        return -1;
    }
    const uint32_t validBarrierScope =
        MGL_RENDER_COMPUTE_BARRIER_BUFFERS |
        MGL_RENDER_COMPUTE_BARRIER_TEXTURES |
        MGL_RENDER_COMPUTE_BARRIER_RENDER_TARGETS;
    if (plan->barrier_scope & ~validBarrierScope) {
        if (err && errcap) snprintf(err, errcap, "invalid compute barrier scope");
        return -1;
    }
    if (plan->dispatch_barrier_scope & ~validBarrierScope) {
        if (err && errcap) {
            snprintf(err, errcap, "invalid inter-dispatch barrier scope");
        }
        return -1;
    }
    for (uint32_t i = 0; i < plan->binding_op_count; i++) {
        const MGLRenderComputeBindingOp* op = &plan->binding_ops[i];
        if (op->kind > 3u || (op->kind == 1u && !op->bytes)) {
            if (err && errcap) {
                snprintf(err, errcap, "invalid compute binding op %u", i);
            }
            return -1;
        }
    }
    if (plan->dispatch_op_count == 0 &&
        plan->dispatch.dispatch_kind ==
            MGL_RENDER_COMPUTE_DISPATCH_DIRECT &&
        (!plan->dispatch.groups_x || !plan->dispatch.groups_y ||
         !plan->dispatch.groups_z)) {
        if (err && errcap) snprintf(err, errcap, "zero compute dispatch groups");
        return -1;
    }
    if (plan->dispatch_op_count == 0 &&
        plan->dispatch.dispatch_kind ==
            MGL_RENDER_COMPUTE_DISPATCH_INDIRECT &&
        !plan->dispatch.indirect_buffer) {
        if (err && errcap) snprintf(err, errcap, "null indirect buffer");
        return -1;
    }
    if (plan->dispatch_op_count == 0 &&
        plan->dispatch.dispatch_kind !=
            MGL_RENDER_COMPUTE_DISPATCH_DIRECT &&
        plan->dispatch.dispatch_kind !=
            MGL_RENDER_COMPUTE_DISPATCH_INDIRECT) {
        if (err && errcap) {
            snprintf(err, errcap, "bad dispatch kind %u",
                     plan->dispatch.dispatch_kind);
        }
        return -1;
    }
    uint32_t previousDispatchBindingCount = 0u;
    for (uint32_t i = 0; i < plan->dispatch_op_count; i++) {
        const MGLRenderComputeDispatchEntry* entry = &plan->dispatch_ops[i];
        if (entry->binding_op_count < previousDispatchBindingCount ||
            entry->binding_op_count > plan->binding_op_count) {
            if (err && errcap) snprintf(err, errcap, "invalid dispatch ordering %u", i);
            return -1;
        }
        if (entry->dispatch.dispatch_kind ==
                MGL_RENDER_COMPUTE_DISPATCH_DIRECT &&
            (!entry->dispatch.groups_x || !entry->dispatch.groups_y ||
             !entry->dispatch.groups_z)) {
            if (err && errcap) snprintf(err, errcap, "zero compute dispatch groups");
            return -1;
        }
        if (entry->dispatch.dispatch_kind ==
                MGL_RENDER_COMPUTE_DISPATCH_INDIRECT &&
            !entry->dispatch.indirect_buffer) {
            if (err && errcap) snprintf(err, errcap, "null indirect buffer");
            return -1;
        }
        if (entry->dispatch.dispatch_kind !=
                MGL_RENDER_COMPUTE_DISPATCH_DIRECT &&
            entry->dispatch.dispatch_kind !=
                MGL_RENDER_COMPUTE_DISPATCH_INDIRECT) {
            if (err && errcap) snprintf(err, errcap, "bad dispatch kind %u",
                                        entry->dispatch.dispatch_kind);
            return -1;
        }
        previousDispatchBindingCount = entry->binding_op_count;
    }

    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    MTL::CommandBuffer* command_buffer = owner->current;
    if (!command_buffer) {
        if (err && errcap) snprintf(err, errcap, "no current command buffer");
        return -1;
    }
    MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
    if (!encoder) {
        if (err && errcap) snprintf(err, errcap, "compute encoder failed");
        return -1;
    }

    encoder->setComputePipelineState(
        static_cast<MTL::ComputePipelineState*>(plan->pipeline));
    uint32_t nextDispatch = 0u;
    for (uint32_t i = 0; i <= plan->binding_op_count; i++) {
        while (nextDispatch < plan->dispatch_op_count &&
               plan->dispatch_ops[nextDispatch].binding_op_count == i) {
            /* Dispatch boundaries of one encoder are only execution-ordered;
             * memory written by an earlier dispatch reaches a later one only
             * through this fence.  Plans that assert memory independence
             * leave dispatch_barrier_scope at NONE (see mgl_render.h). */
            if (nextDispatch > 0u &&
                plan->dispatch_barrier_scope !=
                    MGL_RENDER_COMPUTE_BARRIER_NONE) {
                encoder->memoryBarrier(static_cast<MTL::BarrierScope>(
                    plan->dispatch_barrier_scope));
            }
            if (mglRenderEncodeComputeDispatchOnEncoder(
                    encoder, &plan->dispatch_ops[nextDispatch].dispatch,
                    err, errcap) != 0) {
                encoder->endEncoding();
                return -1;
            }
            nextDispatch++;
        }
        if (i == plan->binding_op_count) break;
        const MGLRenderComputeBindingOp* op = &plan->binding_ops[i];
        switch (op->kind) {
            case 0u:
                encoder->setBuffer(static_cast<MTL::Buffer*>(op->buffer),
                                   static_cast<NS::UInteger>(op->offset),
                                   op->index);
                break;
            case 1u:
                if (!op->bytes) {
                    encoder->endEncoding();
                    if (err && errcap) snprintf(err, errcap,
                                                "null compute bytes op %u", i);
                    return -1;
                }
                encoder->setBytes(op->bytes, op->length, op->index);
                break;
            case 2u:
                encoder->setTexture(static_cast<MTL::Texture*>(op->buffer),
                                    op->index);
                break;
            case 3u:
                encoder->setSamplerState(
                    static_cast<MTL::SamplerState*>(op->buffer), op->index);
                break;
            default:
                encoder->endEncoding();
                if (err && errcap) snprintf(err, errcap,
                                            "bad compute op kind %u", op->kind);
                return -1;
        }
    }

    if (plan->dispatch_op_count == 0 &&
        mglRenderEncodeComputeDispatchOnEncoder(
            encoder, &plan->dispatch, err, errcap) != 0) {
        encoder->endEncoding();
        return -1;
    }
    if (plan->barrier_scope != MGL_RENDER_COMPUTE_BARRIER_NONE) {
        encoder->memoryBarrier(static_cast<MTL::BarrierScope>(
            plan->barrier_scope));
    }
    encoder->endEncoding();
    return 0;
}

namespace {

double commandRecoveryNowSeconds() {
    using Clock = std::chrono::system_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

void snapshotCommandRecoveryOwner(
    mgl::CommandBufferRecoveryOwner* owner,
    MGLRenderCommandRecoverySnapshot* state) {
    if (!owner || !state) return;
    std::lock_guard<std::mutex> lock(owner->mutex);
    mgl::snapshotCommandRecovery(*owner, state);
}

int applyCommandRecoveryFailure(
    mgl::CommandBufferRecoveryOwner* owner,
    const MGLRenderCommandBufferState* state,
    bool request_reset,
    MGLRenderCommandBufferTransaction* transaction) {
    if (!owner || !transaction) return -1;
    if (transaction->recovery_error_recorded) {
        snapshotCommandRecoveryOwner(owner, &transaction->recovery);
        return 0;
    }

    MGLRenderCommandBufferCompletionDecision decision = {};
    if (state && mglRenderClassifyCommandBufferCompletion(
                     state, &decision) != 0) {
        return -1;
    }
    const bool driver_rejection = decision.is_driver_rejection != 0;
    {
        std::lock_guard<std::mutex> lock(owner->mutex);
        owner->consecutiveErrors++;
        owner->consecutiveSuccesses = 0;
        owner->lastErrorTime = commandRecoveryNowSeconds();
        mgl::snapshotCommandRecovery(*owner, &transaction->recovery);
    }
    transaction->has_error = 1u;
    transaction->is_driver_rejection = driver_rejection ? 1u : 0u;
    transaction->device_reset_requested =
        (request_reset || driver_rejection) ? 1u : 0u;
    transaction->recovery_error_recorded = 1u;
    return 0;
}

struct CommandRecoveryCompletionContext {
    ~CommandRecoveryCompletionContext() {
        mgl::releaseCommandRecoveryOwner(owner);
    }

    void retain() {
        references.fetch_add(1u, std::memory_order_relaxed);
    }

    void release() {
        if (references.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
            delete this;
        }
    }

    std::atomic<uint32_t> references{1u};
    std::mutex applyMutex;
    mgl::CommandBufferRecoveryOwner* owner = nullptr;
    bool completionApplied = false;
    bool completionHadError = false;
    bool completionWasDriverRejection = false;
    bool transactionFailureApplied = false;
};

void processCommandRecoveryCompletionLocked(
    CommandRecoveryCompletionContext* completion,
    const MGLRenderCommandBufferState* state,
    MGLRenderCommandBufferCompletionResult* result_out) {
    if (!completion || !completion->owner || !state) return;
    MGLRenderCommandBufferCompletionDecision decision = {};
    if (mglRenderClassifyCommandBufferCompletion(state, &decision) != 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(completion->applyMutex);
    if (completion->transactionFailureApplied ||
        completion->completionApplied) {
        if (result_out) {
            result_out->decision = decision;
            snapshotCommandRecoveryOwner(completion->owner,
                                         &result_out->state);
        }
        return;
    }

    MGLRenderCommandBufferCompletionResult result = {};
    if (mglRenderProcessCommandBufferCompletion(
            reinterpret_cast<MGLCommandBufferRecoveryOwner*>(completion->owner),
            state, commandRecoveryNowSeconds(), &result) != 0) {
        return;
    }
    completion->completionApplied = true;
    completion->completionHadError = result.decision.has_error != 0;
    completion->completionWasDriverRejection =
        result.decision.is_driver_rejection != 0;
    if (result.decision.is_driver_rejection) {
        std::lock_guard<std::mutex> ownerLock(completion->owner->mutex);
        completion->owner->resetRequested = true;
    }
    if (result_out) *result_out = result;
}

void commandRecoveryCompletion(void* context,
                               const MGLRenderCommandBufferState* state) {
    processCommandRecoveryCompletionLocked(
        static_cast<CommandRecoveryCompletionContext*>(context), state,
        nullptr);
}

void destroyCommandRecoveryCompletionContext(void* context) {
    CommandRecoveryCompletionContext* completion =
        static_cast<CommandRecoveryCompletionContext*>(context);
    if (completion) completion->release();
}

int addCommandBufferRecoveryCompletion(
    void* command_buffer,
    void* recovery_owner,
    CommandRecoveryCompletionContext** context_out) {
    if (context_out) *context_out = nullptr;
    if (!command_buffer || !recovery_owner) return -1;
    CommandRecoveryCompletionContext* context =
        new (std::nothrow) CommandRecoveryCompletionContext();
    if (!context) return -1;
    context->owner =
        reinterpret_cast<mgl::CommandBufferRecoveryOwner*>(recovery_owner);
    mgl::retainCommandRecoveryOwner(context->owner);
    if (context_out) {
        context->retain();
        *context_out = context;
    }
    int result = mglRenderAddCommandBufferCompletion(
        command_buffer, commandRecoveryCompletion, context,
        destroyCommandRecoveryCompletionContext);
    if (result != 0) {
        if (context_out) {
            *context_out = nullptr;
            context->release();
        }
        context->release();
    }
    return result;
}

int applyCommandRecoveryTransactionFailure(
    CommandRecoveryCompletionContext* completion,
    mgl::CommandBufferRecoveryOwner* owner,
    const MGLRenderCommandBufferState* state,
    MGLRenderCommandBufferTransaction* transaction) {
    if (!owner || !transaction) return 0;
    if (!completion) {
        return applyCommandRecoveryFailure(owner, state, true, transaction);
    }

    std::lock_guard<std::mutex> lock(completion->applyMutex);
    if (completion->transactionFailureApplied ||
        (completion->completionApplied && completion->completionHadError)) {
        snapshotCommandRecoveryOwner(owner, &transaction->recovery);
        transaction->has_error = 1u;
        transaction->is_driver_rejection =
            completion->completionWasDriverRejection ? 1u : 0u;
        transaction->device_reset_requested = 1u;
        transaction->recovery_error_recorded = 1u;
        return 0;
    }
    int result = applyCommandRecoveryFailure(owner, state, true, transaction);
    if (result == 0) completion->transactionFailureApplied = true;
    return result;
}

struct ScopedRecoveryCompletionContext {
    ~ScopedRecoveryCompletionContext() {
        if (context) context->release();
    }
    CommandRecoveryCompletionContext* context = nullptr;
};

}  // namespace

extern "C"
int mglRenderCommitCommandBufferTransaction(MGLCommandBufferOwner * owner_handle, void** submission_handle, void* command_buffer, MGLCommandBufferRecoveryOwner * recovery_owner, uint32_t wait_for_completion, MGLRenderCommandBufferTransaction* result_out) {
    if (result_out) memset(result_out, 0, sizeof(*result_out));
    if (!command_buffer || !result_out) return -1;
    result_out->result = MGL_RENDER_COMMAND_BUFFER_TRANSACTION_ERROR;

    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    mgl::CommandBufferRecoveryOwner* recovery =
        reinterpret_cast<mgl::CommandBufferRecoveryOwner*>(static_cast<void*>(recovery_owner));
    ScopedRecoveryCompletionContext recovery_completion;
    if (mgl::snapshotCommandBufferState(command, &result_out->before) != 0) {
        applyCommandRecoveryTransactionFailure(
            nullptr, recovery, nullptr, result_out);
        return -1;
    }

    MGLRenderCommandBufferCommitDecision decision = {};
    if (mglRenderClassifyCommandBufferCommit(
            &result_out->before, &decision) != 0) {
        applyCommandRecoveryTransactionFailure(
            nullptr, recovery, &result_out->before, result_out);
        return -1;
    }
    if (decision.action ==
        MGL_RENDER_COMMAND_BUFFER_COMMIT_SKIP_ALREADY_COMMITTED) {
        result_out->result =
            MGL_RENDER_COMMAND_BUFFER_TRANSACTION_SKIPPED;
        result_out->after = result_out->before;
        if (result_out->before.has_error && recovery) {
            applyCommandRecoveryFailure(
                recovery, &result_out->before, false, result_out);
        } else if (recovery) {
            snapshotCommandRecoveryOwner(recovery, &result_out->recovery);
        }
        return 0;
    }

    bool commit_guard_acquired = false;
    if (owner_handle) {
        int guard = mglRenderCommandBufferOwnerBeginCommit(owner_handle);
        if (guard < 0) {
            applyCommandRecoveryTransactionFailure(
                nullptr, recovery, &result_out->before, result_out);
            return -1;
        }
        if (guard == 0) {
            result_out->result =
                MGL_RENDER_COMMAND_BUFFER_TRANSACTION_NESTED;
            result_out->after = result_out->before;
            if (recovery) {
                snapshotCommandRecoveryOwner(recovery,
                                             &result_out->recovery);
            }
            return 0;
        }
        commit_guard_acquired = true;
    }

    struct CommitGuard {
        MGLCommandBufferOwner* owner = nullptr;
        bool* acquired = nullptr;
        ~CommitGuard() {
            if (owner && acquired && *acquired) {
                mglRenderCommandBufferOwnerEndCommit(owner);
                *acquired = false;
            }
        }
    } commit_guard{owner_handle, &commit_guard_acquired};

    if (submission_handle && *submission_handle &&
        mglRenderCommandBufferSubmissionMatchesBuffer(
            *submission_handle, command_buffer) != 1) {
        result_out->after = result_out->before;
        applyCommandRecoveryTransactionFailure(
            nullptr, recovery, &result_out->before, result_out);
        if (!recovery) result_out->has_error = 1u;
        return -1;
    }

    int commit_result = -1;
    bool committed = false;
    if (recovery &&
        addCommandBufferRecoveryCompletion(
            command_buffer, recovery_owner,
            &recovery_completion.context) != 0) {
        applyCommandRecoveryTransactionFailure(
            nullptr, recovery, &result_out->before, result_out);
        return -1;
    }
    result_out->completion_registered = recovery ? 1u : 0u;
    try {
        if (submission_handle && *submission_handle &&
            mglRenderCommandBufferSubmissionMatchesBuffer(
                *submission_handle, command_buffer) == 1) {
            result_out->used_submission = 1u;
            commit_result = mglRenderCommitCommandBufferSubmission(
                submission_handle);
        } else {
            command->commit();
            commit_result = 0;
        }
        committed = commit_result == 0;
    } catch (...) {
        commit_result = -1;
        applyCommandRecoveryTransactionFailure(
            recovery_completion.context, recovery, &result_out->before,
            result_out);
        if (!recovery) result_out->has_error = 1u;
    }

    if (mgl::snapshotCommandBufferState(command, &result_out->after) != 0) {
        applyCommandRecoveryTransactionFailure(
            recovery_completion.context, recovery, nullptr, result_out);
        if (!recovery) result_out->has_error = 1u;
    }
    if (!committed) {
        applyCommandRecoveryTransactionFailure(
            recovery_completion.context, recovery, &result_out->after,
            result_out);
        if (!recovery) result_out->has_error = 1u;
        return -1;
    }
    if (owner_handle) {
        mgl::setLastSubmitted(
            reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle)), command);
    }
    result_out->result =
        MGL_RENDER_COMMAND_BUFFER_TRANSACTION_COMMITTED;
    result_out->needs_new_command_buffer = 1u;
    if (wait_for_completion) {
        result_out->waited = 1u;
        try {
            command->waitUntilCompleted();
        } catch (...) {
            applyCommandRecoveryTransactionFailure(
                recovery_completion.context, recovery, nullptr, result_out);
            if (!recovery) result_out->has_error = 1u;
            return -1;
        }
        if (mgl::snapshotCommandBufferState(
                command, &result_out->completion) != 0) {
            applyCommandRecoveryTransactionFailure(
                recovery_completion.context, recovery, nullptr, result_out);
            if (!recovery) result_out->has_error = 1u;
            return -1;
        }
        MGLRenderCommandBufferCompletionDecision completionDecision = {};
        if (mglRenderClassifyCommandBufferCompletion(
                &result_out->completion, &completionDecision) != 0) {
            applyCommandRecoveryTransactionFailure(
                recovery_completion.context, recovery,
                &result_out->completion, result_out);
            if (!recovery) result_out->has_error = 1u;
            return -1;
        }
        result_out->has_error = completionDecision.has_error;
        result_out->is_driver_rejection =
            completionDecision.is_driver_rejection;
        if (recovery_completion.context) {
            MGLRenderCommandBufferCompletionResult completionResult = {};
            processCommandRecoveryCompletionLocked(
                recovery_completion.context, &result_out->completion,
                &completionResult);
            result_out->recovery = completionResult.state;
            result_out->recovery_error_recorded =
                completionDecision.has_error ? 1u : 0u;
        }
        if (recovery) {
            result_out->device_reset_requested =
                mglRenderCommandRecoveryTakeResetRequest(
                    recovery_owner) == 1 ? 1u : 0u;
        }
        if (result_out->has_error) return -1;
    }
    /* Taking a submission leaves the owner without a current command buffer.
     * Owners created from the C++ queue rotate the next buffer here so the
     * lifecycle transaction owns creation as well as submission. Adopted ObjC
     * buffers have no queue and keep the legacy reset adapter. */
    if (owner_handle) {
        void* next = nullptr;
        int nextResult = mglRenderCommandBufferOwnerCreateNext(
            owner_handle, &next);
        if (nextResult == 0 && next) {
            result_out->current_command_buffer_created = 1u;
            reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(owner_handle))
                ->transaction_created_current = true;
        } else if (nextResult < 0) {
            applyCommandRecoveryTransactionFailure(
                recovery_completion.context, recovery, nullptr, result_out);
            if (!recovery) result_out->has_error = 1u;
            return -1;
        }
    }
    if (recovery) {
        snapshotCommandRecoveryOwner(recovery, &result_out->recovery);
    }
    return 0;
}

int mglRenderCommandRecoveryRecordTransactionFailure(MGLCommandBufferRecoveryOwner * owner_handle, const MGLRenderCommandBufferState* state, MGLRenderCommandBufferTransaction* transaction_inout) {
    mgl::CommandBufferRecoveryOwner* owner =
        reinterpret_cast<mgl::CommandBufferRecoveryOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !transaction_inout) return -1;
    transaction_inout->result =
        MGL_RENDER_COMMAND_BUFFER_TRANSACTION_ERROR;
    return applyCommandRecoveryFailure(
        owner, state, true, transaction_inout);
}

static int mglRenderResetRenderEncoderOwnerImpl(
    mgl::RenderEncoderOwner* owner,
    void* command_buffer,
    const MGLRenderPassState* render_pass,
    void** render_encoder_out) {
    if (render_encoder_out) *render_encoder_out = nullptr;
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    if (!owner || !command || !render_pass || !render_encoder_out) return -1;
    MTL::RenderPassDescriptor* descriptor =
        mgl::newRenderPassDescriptor(render_pass);
    if (!descriptor) return -1;
    MTL::RenderCommandEncoder* encoder =
        command->renderCommandEncoder(descriptor);
    descriptor->release();
    if (!encoder) return -1;
    encoder->retain();
    if (owner->encoder) owner->encoder->release();
    owner->encoder = encoder;
    owner->ended = false;
    *render_encoder_out = encoder;
    return 0;
}

int mglRenderCreateRenderEncoderOwnerFromState(void* command_buffer, const MGLRenderPassState* render_pass, MGLRenderEncoderOwner ** owner_out, void** render_encoder_out) {
    if (owner_out) *owner_out = nullptr;
    if (render_encoder_out) *render_encoder_out = nullptr;
    if (!owner_out || !render_encoder_out) return -1;
    mgl::RenderEncoderOwner* owner =
        new (std::nothrow) mgl::RenderEncoderOwner();
    if (!owner) return -1;
    if (mglRenderResetRenderEncoderOwnerImpl(
            owner, command_buffer, render_pass, render_encoder_out) != 0) {
        delete owner;
        return -1;
    }
    *owner_out = reinterpret_cast<MGLRenderEncoderOwner*>(owner);
    return 0;
}

int mglRenderResetRenderEncoderOwnerFromState(MGLRenderEncoderOwner * owner_handle, void* command_buffer, const MGLRenderPassState* render_pass, void** render_encoder_out) {
    return mglRenderResetRenderEncoderOwnerImpl(
        reinterpret_cast<mgl::RenderEncoderOwner*>(static_cast<void*>(owner_handle)),
        command_buffer, render_pass, render_encoder_out);
}

void* mglRenderActiveRenderEncoder(MGLRenderEncoderOwner * owner_handle) {
    mgl::RenderEncoderOwner* owner =
        reinterpret_cast<mgl::RenderEncoderOwner*>(static_cast<void*>(owner_handle));
    return owner && owner->encoder && !owner->ended
        ? static_cast<void*>(owner->encoder)
        : nullptr;
}

static uint64_t mglRenderTargetLayerCount(
    MTL::Texture* texture,
    uint64_t level) {
    if (!texture) return 0u;
    switch (texture->textureType()) {
    case MTL::TextureType1DArray:
    case MTL::TextureType2DArray:
    case MTL::TextureType2DMultisampleArray:
        return static_cast<uint64_t>(texture->arrayLength());
    case MTL::TextureTypeCube:
        return 6u;
    case MTL::TextureTypeCubeArray:
        return static_cast<uint64_t>(texture->arrayLength()) * 6u;
    case MTL::TextureType3D:
        return mglRenderMetalTextureLevelDimension(
            static_cast<uint64_t>(texture->depth()), level);
    default:
        return 1u;
    }
}

static uint64_t mglRenderPassArrayLength(
    const MGLRenderPassState& state) {
    uint64_t commonArrayLength = 0u;
    bool hasLayeredAttachment = false;
    auto accumulate = [&commonArrayLength, &hasLayeredAttachment](
                          const MGLRenderPassAttachmentState& attachment) {
        MTL::Texture* texture = static_cast<MTL::Texture*>(attachment.texture);
        if (!texture) return;
        if (!attachment.layered) return;
        uint64_t layerCount =
            mglRenderTargetLayerCount(texture, attachment.level);
        if (layerCount == 0u) return;
        hasLayeredAttachment = true;
        commonArrayLength = commonArrayLength == 0u
            ? layerCount
            : std::min(commonArrayLength, layerCount);
    };
    for (uint32_t i = 0u; i < MGL_RENDER_MAX_COLOR_ATTACHMENTS; ++i) {
        accumulate(state.color[i].attachment);
    }
    accumulate(state.depth.attachment);
    accumulate(state.stencil.attachment);
    return hasLayeredAttachment ? commonArrayLength : 0u;
}

int mglRenderSetRenderPassStateAttachmentTexture(MGLRenderPassStateOwner * owner_handle, uint32_t attachment_kind, uint32_t color_index, void* texture, uint64_t level, uint64_t slice, uint64_t depth_plane, uint32_t layered) {
    mgl::RenderPassStateOwner* owner =
        reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return -1;

    MGLRenderPassAttachmentState* destination = nullptr;
    switch (attachment_kind) {
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
        if (color_index >= MGL_RENDER_MAX_COLOR_ATTACHMENTS) return -1;
        destination = &owner->state.color[color_index].attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
        destination = &owner->state.depth.attachment;
        break;
    case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
        destination = &owner->state.stencil.attachment;
        break;
    default:
        return -1;
    }

    mgl::retainRenderPassObject(texture);
    mgl::releaseRenderPassObject(destination->texture);
    destination->texture = texture;
    destination->level = level;
    destination->slice = slice;
    destination->depth_plane = depth_plane;
    destination->layered = layered != 0u;

    owner->state.render_target_array_length =
        mglRenderPassArrayLength(owner->state);
    if (layered) {
        destination->slice = 0u;
        destination->depth_plane = 0u;
    }
    return 0;
}

int mglRenderEncodeDrawForRenderEncoderOwner(MGLRenderEncoderOwner * render_encoder_owner, const MGLRenderDrawPlan* plan, char* err, size_t errcap) {
    return mglRenderEncodeDraw(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        plan, err, errcap);
}

int mglRenderAddCommandBufferRecoveryCompletion(void* command_buffer, MGLCommandBufferRecoveryOwner * recovery_owner) {
    return addCommandBufferRecoveryCompletion(
        command_buffer, recovery_owner, nullptr);
}

MGLRenderPassAttachmentState*
mglRenderAttachmentForOwner(
    mgl::RenderPassStateOwner* owner,
    uint32_t attachment_kind,
    uint32_t color_index) {
    if (!owner) return nullptr;
    switch (attachment_kind) {
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR:
            return color_index < MGL_RENDER_MAX_COLOR_ATTACHMENTS
                ? &owner->state.color[color_index].attachment : nullptr;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH:
            return &owner->state.depth.attachment;
        case MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL:
            return &owner->state.stencil.attachment;
        default:
            return nullptr;
    }
}

int mglRenderGetRenderPassAttachmentSubresourceOwner(MGLRenderPassStateOwner * owner_handle, uint32_t attachment_kind, uint32_t color_index, uint64_t* level_out, uint64_t* slice_out, uint64_t* depth_plane_out) {
    auto* owner = reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    auto* attachment = mglRenderAttachmentForOwner(
        owner, attachment_kind, color_index);
    if (!attachment) return -1;
    if (level_out) *level_out = attachment->level;
    if (slice_out) *slice_out = attachment->slice;
    if (depth_plane_out) *depth_plane_out = attachment->depth_plane;
    return 0;
}

int mglRenderGetRenderPassAttachmentActionsOwner(MGLRenderPassStateOwner * owner_handle, uint32_t attachment_kind, uint32_t color_index, uint32_t* load_action_out, uint32_t* store_action_out, uint64_t* store_action_options_out) {
    auto* owner = reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    auto* attachment = mglRenderAttachmentForOwner(
        owner, attachment_kind, color_index);
    if (!attachment) return -1;
    if (load_action_out) *load_action_out = attachment->load_action;
    if (store_action_out) *store_action_out = attachment->store_action;
    if (store_action_options_out) {
        *store_action_options_out = attachment->store_action_options;
    }
    return 0;
}
