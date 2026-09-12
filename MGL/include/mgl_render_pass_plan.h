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
 * mgl_render_pass_plan.h — O1.1 processGLState planning (pure C, no Metal).
 *
 * Two-phase plan: (1) early classify / CB lifecycle / dirty dispatch,
 * (2) after dirty domains, encoder/pipeline/bind materialization flags.
 * ObjC only applies MTL* actions the plan requests.
 */

#ifndef MGL_RENDER_PASS_PLAN_H
#define MGL_RENDER_PASS_PLAN_H

#include <stdint.h>

#include "mgl_render_values.h" /* MGLLoadAction* / MGLStoreAction* (pure-C ABI) */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MGL_PROCESS_GL_ABORT = 0,
    MGL_PROCESS_GL_NON_DRAW = 1,
    MGL_PROCESS_GL_NO_VAO_CLEAR = 2,
    MGL_PROCESS_GL_CONTINUE = 3
} MGLProcessGLStateClass;

typedef enum {
    MGL_PGL_RESULT_ABORT = 0,
    MGL_PGL_RESULT_EARLY_OK = 1,
    MGL_PGL_RESULT_CONTINUE = 2
} MGLProcessGLStateResult;

/* Match MGLCommandBufferStatusCommitted in mgl_render_values.h. */
enum { MGL_PGL_CB_STATUS_COMMITTED = 2 };

typedef struct MGLProcessGLStateInputs {
    uint8_t has_ctx;
    uint8_t draw_command;
    uint8_t has_vao;
    uint8_t dirty_state;
    uint8_t ctx_ptr_sane;
    uint8_t device_ok;
    uint8_t queue_ok;
    uint8_t quarantine_blocks_draw;
    uint8_t has_command_buffer;
    uint8_t encoder_has_current;
    uint32_t command_buffer_status;
} MGLProcessGLStateInputs;

typedef struct MGLProcessGLStatePlan {
    MGLProcessGLStateResult result;
    MGLProcessGLStateClass process_class;
    uint8_t clear_rt_sampled_copy;
    uint8_t rotate_finalized_command_buffer;
    uint8_t create_initial_command_buffer;
    uint8_t process_dirty_domains;
    uint8_t non_draw_end_pass_if_fbo_changed;
    uint8_t no_vao_clear_path;
} MGLProcessGLStatePlan;

typedef struct MGLProcessGLStateAfterInputs {
    uint8_t draw_command;
    uint8_t encoder_has_current;
    uint8_t has_pipeline_state;
    uint8_t frag_needs_fragcoord;
    uint8_t frag_needs_sample;
    uint8_t frag_needs_lod_bias;
    uint8_t fragment_trace_uses_rt_sampled_copy;
} MGLProcessGLStateAfterInputs;

typedef struct MGLProcessGLStateAfterPlan {
    MGLProcessGLStateResult result;
    uint8_t recover_nil_encoder;
    uint8_t ensure_pass_matches_fbo;
    uint8_t fail_nil_pipeline;
    uint8_t validate_attachments;
    uint8_t set_pipeline;
    uint8_t sync_resources;
    uint8_t bind_frag_coord_slot;
    uint8_t bind_lod_bias_slot;
    uint8_t maybe_mark_rt_sampled_copy;
} MGLProcessGLStateAfterPlan;

MGLProcessGLStateClass mglRenderClassifyProcessGLState(int has_ctx,
                                                       int draw_command,
                                                       int has_vao,
                                                       int dirty_state);

int mglRenderProcessGLState(const MGLProcessGLStateInputs *in,
                            MGLProcessGLStatePlan *out);

int mglRenderProcessGLStateAfterDirty(const MGLProcessGLStateAfterInputs *in,
                                      MGLProcessGLStateAfterPlan *out);

/* ---- O3.1: load / store actions, stale clear bits, attachment match ---- */

enum {
    MGL_RP_ATTACHMENT_COLOR = 0,
    MGL_RP_ATTACHMENT_DEPTH = 1,
    MGL_RP_ATTACHMENT_STENCIL = 2
};

typedef struct MGLRenderPassLoadStoreInput {
    uint32_t attachment_kind;  /* MGL_RP_ATTACHMENT_* */
    int attachment_present;    /* color: the draw slot resolved to an attached
                                * attachment; depth/stencil: unused */
    int has_clear_pending;     /* the framebuffer's clear bit for it */
    int texture_present;       /* depth/stencil: an attachment texture exists */
    int dontcare_enabled;      /* MGL_ENABLE_DONTCARE_LOAD */
    int first_use_this_frame;  /* color: the frame generation said first use */
    int blend_enabled;         /* color: blending forbids DontCare */
} MGLRenderPassLoadStoreInput;

typedef struct MGLRenderPassLoadStorePlan {
    uint32_t load_action;      /* MGLLoadAction* */
    /* The clear arm also stores; every other arm leaves the store action the
     * pass state already carries (allocation-time default: Store). */
    int set_store_action;
    uint32_t store_action;     /* MGLStoreAction* when set_store_action */
} MGLRenderPassLoadStorePlan;

int mglRenderPassPlanLoadStore(const MGLRenderPassLoadStoreInput *in,
                               MGLRenderPassLoadStorePlan *out);

/* A pending color clear for an attachment that is not attached cannot be
 * consumed by this pass and must be dropped. */
int mglRenderPassDropsStaleColorClear(uint32_t clear_mask,
                                      uint32_t attached_bitfield,
                                      uint32_t attachment_index);

/* One attachment of the pass match: a color draw slot, or the depth / stencil
 * attachment.  Rules, in order: an entry that does not compare is skipped; the
 * texture pointers must be equal; a required attachment must have a texture;
 * and when asked, the pass-state subresource must line up with the
 * framebuffer's (level / slice / depth plane). */
typedef struct MGLRenderPassSubresource {
    uint32_t level;
    uint32_t slice;
    uint32_t depth_plane;
} MGLRenderPassSubresource;

typedef struct MGLRenderPassAttachmentMatchEntry {
    int compare;
    const void *actual_texture;   /* texture the pass state carries */
    const void *expected_texture; /* texture the framebuffer wants */
    int required;                 /* a required attachment must be there */
    int compare_subresource;
    MGLRenderPassSubresource actual_sub;   /* pass-state side */
    MGLRenderPassSubresource expected_sub; /* framebuffer side */
} MGLRenderPassAttachmentMatchEntry;

/* Fill one entry from what the caller resolved: the two textures, whether the
 * attachment is required, whether the subresource must line up (only for a
 * framebuffer side that has one) and the two subresources. */
void mglRenderPassFillMatchEntry(MGLRenderPassAttachmentMatchEntry *entry,
                                 const void *actual_texture,
                                 const void *expected_texture, int required,
                                 int compare_subresource,
                                 MGLRenderPassSubresource actual_sub,
                                 MGLRenderPassSubresource expected_sub);

typedef struct MGLRenderPassAttachmentMatchInput {
    int identity_ok; /* framebuffer pointer / name / draw buffer / count */
    const MGLRenderPassAttachmentMatchEntry *entries;
    uint32_t entry_count;
} MGLRenderPassAttachmentMatchInput;

/* 1 = the persistent pass covers exactly this framebuffer. */
int mglRenderPassAttachmentsMatch(const MGLRenderPassAttachmentMatchInput *in);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDER_PASS_PLAN_H */
