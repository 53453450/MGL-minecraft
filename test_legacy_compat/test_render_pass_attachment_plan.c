/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Render-pass attachment planning harness (audit P4).
 *
 * Golden for the three pure decisions that used to be spelled out inline in
 * mgl_render_pass_manager_ops.c:
 *
 *   format compatibility   a cached pipeline against the pass it runs in
 *                          (mglRenderPassPlanFormatCompat);
 *   draw-slot attachment   which framebuffer color attachment feeds a draw
 *                          buffer, including the GL_NONE hole that still
 *                          occupies its Metal slot
 *                          (mglRenderPassPlanDrawSlotAttachment);
 *   subresource entry      the pass-match entry for a user-framebuffer
 *                          attachment (mglRenderPassPlanFillSubresourceEntry).
 *
 * The "reference" functions below are the original inline expressions; each
 * test sweeps a small input space and requires the plan to agree with them.
 *
 * Build: see `make test-render-pass-attachment-plan`.
 */

#include "mgl_render_pass_plan.h"

#include <stdio.h>
#include <string.h>

static int tests_run;
static int tests_passed;

#define CHECK(cond, label)                                       \
    do {                                                         \
        tests_run++;                                             \
        if (cond) {                                              \
            tests_passed++;                                      \
        } else {                                                 \
            printf("  [FAIL] %s\n", (label));                    \
        }                                                        \
    } while (0)

/* ---- format compatibility ------------------------------------------------ */

/* mglEnsureRasterEncoderForDraw's inline test. */
static void ref_format_inline(const MGLRenderPassFormatCompatInput *in,
                              int *color, int *depth, int *stencil)
{
    *color = in->pipeline_color0 != 0u && in->pass_color0 != 0u &&
             in->pipeline_color0 != in->pass_color0;
    *depth = in->pipeline_depth != in->pass_depth;
    *stencil = in->pipeline_stencil != in->pass_stencil;
}

/* The validate path's helpers (retired mglRenderPipelinePassColorMismatch /
 * mglRenderPipelinePassAttachmentMismatch). */
static int ref_color_mismatch(uint32_t pipeline, uint32_t pass)
{
    return pipeline != 0u && pass != 0u && pipeline != pass;
}

static int ref_attachment_mismatch(uint32_t pipeline, uint32_t pass)
{
    if (pipeline == 0u && pass == 0u) {
        return 0;
    }
    return pipeline != pass;
}

static void test_format_compat(void)
{
    static const uint32_t formats[] = {0u, 70u, 80u, 252u, 253u, 260u};
    const unsigned n = sizeof(formats) / sizeof(formats[0]);
    int agree = 1;

    for (unsigned a = 0; a < n; a++)
    for (unsigned b = 0; b < n; b++)
    for (unsigned c = 0; c < n; c++)
    for (unsigned d = 0; d < n; d++)
    for (unsigned e = 0; e < n; e++)
    for (unsigned f = 0; f < n; f++) {
        MGLRenderPassFormatCompatInput in = {
            formats[a], formats[b], formats[c],
            formats[d], formats[e], formats[f]};
        MGLRenderPassFormatCompat out;
        int ic, id, is;
        if (mglRenderPassPlanFormatCompat(&in, &out) != 0) {
            agree = 0;
            continue;
        }
        ref_format_inline(&in, &ic, &id, &is);
        if (!!out.color_mismatch != !!ic || !!out.depth_mismatch != !!id ||
            !!out.stencil_mismatch != !!is) {
            agree = 0;
        }
        if (!!out.color_mismatch !=
                !!ref_color_mismatch(in.pipeline_color0, in.pass_color0) ||
            !!out.depth_mismatch !=
                !!ref_attachment_mismatch(in.pipeline_depth, in.pass_depth) ||
            !!out.stencil_mismatch !=
                !!ref_attachment_mismatch(in.pipeline_stencil,
                                          in.pass_stencil)) {
            agree = 0;
        }
    }
    CHECK(agree, "format compat matches both retired spellings on 6^6 inputs");

    MGLRenderPassFormatCompatInput in;
    MGLRenderPassFormatCompat out;
    memset(&in, 0, sizeof(in));
    CHECK(mglRenderPassPlanFormatCompat(&in, &out) == 0 &&
              !out.color_mismatch && !out.depth_mismatch &&
              !out.stencil_mismatch,
          "no attachments on either side agree");

    in.pipeline_color0 = 80u;
    CHECK(mglRenderPassPlanFormatCompat(&in, &out) == 0 &&
              !out.color_mismatch,
          "a pass without a color attachment accepts a color pipeline");

    in.pass_color0 = 70u;
    CHECK(mglRenderPassPlanFormatCompat(&in, &out) == 0 &&
              out.color_mismatch,
          "two different color formats mismatch");

    memset(&in, 0, sizeof(in));
    in.pipeline_depth = 252u;
    CHECK(mglRenderPassPlanFormatCompat(&in, &out) == 0 &&
              out.depth_mismatch,
          "a depth pipeline in a pass without depth mismatches");

    memset(&in, 0, sizeof(in));
    in.pass_stencil = 253u;
    CHECK(mglRenderPassPlanFormatCompat(&in, &out) == 0 &&
              out.stencil_mismatch,
          "a pass with stencil under a pipeline without it mismatches");

    CHECK(mglRenderPassPlanFormatCompat(NULL, &out) == -1 &&
              mglRenderPassPlanFormatCompat(&in, NULL) == -1,
          "NULL arguments are refused");
}

/* ---- draw-slot attachment ------------------------------------------------ */

/* The inline selection from mglRenderPassConfigureUserFBOAttachments (needs the
 * texture) and mglRenderPassMatchesFramebufferImpl (bitfield only). */
static int ref_draw_slot(const MGLRenderPassDrawSlotInput *in, uint32_t *index)
{
    uint32_t attachment = in->resolved_index;
    int present;
    if (in->require_texture) {
        if (in->draw_buffer_resolved && attachment < in->max_attachments &&
            (in->attached_bitfield & (1u << attachment)) &&
            (in->texture_bitfield & (1u << attachment))) {
            present = 1;
        } else if (in->draw_buffer_is_none && in->slot < in->max_attachments &&
                   (in->attached_bitfield & (1u << in->slot)) &&
                   (in->texture_bitfield & (1u << in->slot))) {
            attachment = in->slot;
            present = 1;
        } else {
            present = 0;
        }
    } else {
        present = in->draw_buffer_resolved && attachment < in->max_attachments &&
                  ((in->attached_bitfield >> attachment) & 1u) != 0u;
        if (!present && in->draw_buffer_is_none &&
            ((in->attached_bitfield >> in->slot) & 1u) != 0u) {
            attachment = in->slot;
            present = 1;
        }
    }
    *index = attachment;
    return present;
}

static void test_draw_slot(void)
{
    int agree = 1;
    for (uint32_t bits = 0; bits < 16u; bits++)
    for (uint32_t tex = 0; tex < 16u; tex++)
    for (uint32_t resolved = 0; resolved < 2u; resolved++)
    for (uint32_t index = 0; index < 6u; index++)
    for (uint32_t none = 0; none < 2u; none++)
    for (uint32_t slot = 0; slot < 4u; slot++)
    for (uint32_t require = 0; require < 2u; require++) {
        MGLRenderPassDrawSlotInput in;
        MGLRenderPassDrawSlotChoice out;
        memset(&in, 0, sizeof(in));
        in.draw_buffer_resolved = (int)resolved;
        in.resolved_index = index;
        in.draw_buffer_is_none = (int)none;
        in.slot = slot;
        in.max_attachments = 4u; /* index 4 and 5 are out of range */
        in.attached_bitfield = bits;
        in.require_texture = (int)require;
        in.texture_bitfield = tex;
        uint32_t want_index;
        const int want_present = ref_draw_slot(&in, &want_index);
        if (mglRenderPassPlanDrawSlotAttachment(&in, &out) != 0 ||
            !!out.present != !!want_present ||
            (want_present && out.attachment_index != want_index)) {
            agree = 0;
        }
    }
    CHECK(agree, "draw slot matches the retired inline selections exhaustively");

    MGLRenderPassDrawSlotInput in;
    MGLRenderPassDrawSlotChoice out;
    memset(&in, 0, sizeof(in));
    in.max_attachments = 8u;
    in.attached_bitfield = 0x5u; /* attachments 0 and 2 */
    in.draw_buffer_resolved = 1;
    in.resolved_index = 2u;
    in.slot = 0u;
    CHECK(mglRenderPassPlanDrawSlotAttachment(&in, &out) == 0 && out.present &&
              out.attachment_index == 2u,
          "COLOR_ATTACHMENT2 feeds draw buffer 0");

    in.draw_buffer_resolved = 0;
    in.draw_buffer_is_none = 1;
    in.slot = 2u;
    CHECK(mglRenderPassPlanDrawSlotAttachment(&in, &out) == 0 && out.present &&
              out.attachment_index == 2u,
          "a GL_NONE draw buffer keeps its own slot occupied");

    in.slot = 1u;
    CHECK(mglRenderPassPlanDrawSlotAttachment(&in, &out) == 0 && !out.present,
          "a GL_NONE hole over an unattached slot is empty");

    in.slot = 40u;
    in.max_attachments = 64u;
    in.attached_bitfield = 0xffffffffu;
    CHECK(mglRenderPassPlanDrawSlotAttachment(&in, &out) == 0 && !out.present,
          "a slot beyond the 32-bit bitfield is never present");

    CHECK(mglRenderPassPlanDrawSlotAttachment(NULL, &out) == -1 &&
              mglRenderPassPlanDrawSlotAttachment(&in, NULL) == -1,
          "NULL arguments are refused");
}

/* ---- subresource entry --------------------------------------------------- */

static void test_subresource_entry(void)
{
    int tex_a = 1, tex_b = 2;
    const MGLRenderPassSubresource snapshot = {1u, 2u, 3u};
    const MGLRenderPassSubresource same = {1u, 2u, 3u};
    const MGLRenderPassSubresource other = {1u, 4u, 3u};
    MGLRenderPassAttachmentMatchEntry entry;
    MGLRenderPassAttachmentMatchInput match = {0};
    match.identity_ok = 1;
    match.entries = &entry;
    match.entry_count = 1;

    mglRenderPassPlanFillSubresourceEntry(&entry, &tex_a, &tex_a, 1, 1,
                                          snapshot, same);
    CHECK(entry.compare && !entry.required && entry.compare_subresource &&
              mglRenderPassAttachmentsMatch(&match),
          "same texture and subresource match");

    mglRenderPassPlanFillSubresourceEntry(&entry, &tex_a, &tex_a, 1, 1,
                                          snapshot, other);
    CHECK(entry.compare_subresource && !mglRenderPassAttachmentsMatch(&match),
          "a different slice does not match");

    mglRenderPassPlanFillSubresourceEntry(&entry, &tex_a, &tex_a, 1, 0,
                                          snapshot, other);
    CHECK(!entry.compare_subresource && mglRenderPassAttachmentsMatch(&match),
          "without a pass-state snapshot the subresource is not compared");

    mglRenderPassPlanFillSubresourceEntry(&entry, &tex_a, &tex_a, 0, 1,
                                          snapshot, other);
    CHECK(!entry.compare_subresource &&
              entry.actual_sub.level == 0u && entry.expected_sub.slice == 0u &&
              mglRenderPassAttachmentsMatch(&match),
          "a closed gate compares textures only and zeroes both sides");

    mglRenderPassPlanFillSubresourceEntry(&entry, &tex_a, &tex_b, 1, 1,
                                          snapshot, same);
    CHECK(!mglRenderPassAttachmentsMatch(&match),
          "different textures never match");

    mglRenderPassPlanFillSubresourceEntry(&entry, NULL, NULL, 1, 1,
                                          snapshot, other);
    CHECK(mglRenderPassAttachmentsMatch(&match),
          "an absent attachment on both sides matches");

    mglRenderPassPlanFillSubresourceEntry(NULL, &tex_a, &tex_a, 1, 1,
                                          snapshot, same);
    CHECK(1, "a NULL entry is tolerated");
}

int main(void)
{
    printf("render-pass attachment plan harness (audit P4)\n");
    test_format_compat();
    test_draw_slot();
    test_subresource_entry();
    printf("test_render_pass_attachment_plan: %d/%d passed\n", tests_passed,
           tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
