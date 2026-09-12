/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Render-pass load/store + attachment-match harness (O3.1).
 *
 * Characterisation oracle for the O3.1 slice: the load / store action a
 * render-pass attachment gets from the framebuffer's pending clear state, the
 * "stale clear bit" rule for an attachment that is not attached, and the
 * attachment match that decides whether the persistent pass may be reused for
 * the current framebuffer.
 *
 * The goldens below were read off the Objective-C code before it was ported
 * (+RenderPass.m: configureUserFBOLoadStoreActionsLocked: and
 * mglRenderPassMatchesFramebufferImpl:name:):
 *   - a pending clear always wins and also stores;
 *   - DontCare needs the env flag, a texture, the first use this frame and no
 *     blending, and it only replaces the load action (the store stays);
 *   - depth / stencil load and store whenever an attachment texture exists;
 *   - a draw slot that does not resolve to an attached attachment is left
 *     alone (Load);
 *   - a pass matches only when every compared slot, the depth/stencil pair and
 *     the identity agree - and a required attachment that is missing never
 *     matches.
 *
 * Build: see `make test-render-pass-load-store`.
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
            printf("  [PASS] %s\n", (label));                    \
        } else {                                                 \
            printf("  [FAIL] %s\n", (label));                    \
        }                                                        \
    } while (0)

static MGLRenderPassLoadStoreInput color_input(void)
{
    MGLRenderPassLoadStoreInput in;
    memset(&in, 0, sizeof(in));
    in.attachment_kind = MGL_RP_ATTACHMENT_COLOR;
    in.attachment_present = 1;
    in.texture_present = 1;
    return in;
}

static MGLRenderPassLoadStoreInput depth_input(void)
{
    MGLRenderPassLoadStoreInput in;
    memset(&in, 0, sizeof(in));
    in.attachment_kind = MGL_RP_ATTACHMENT_DEPTH;
    in.texture_present = 1;
    return in;
}

static void test_color_load_store(void)
{
    MGLRenderPassLoadStorePlan plan;

    MGLRenderPassLoadStoreInput in = color_input();
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "color plan ok");
    CHECK(plan.load_action == MGLLoadActionLoad, "color loads by default");
    CHECK(plan.set_store_action == 0, "color keeps its store action");

    in = color_input();
    in.has_clear_pending = 1;
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "color clear plan");
    CHECK(plan.load_action == MGLLoadActionClear, "pending clear wins");
    CHECK(plan.set_store_action == 1 &&
              plan.store_action == MGLStoreActionStore,
          "pending clear also stores");

    in = color_input();
    in.dontcare_enabled = 1;
    in.first_use_this_frame = 1;
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "color dontcare plan");
    CHECK(plan.load_action == MGLLoadActionDontCare,
          "first use with a texture may discard");
    CHECK(plan.set_store_action == 0, "dontcare does not touch the store");

    in = color_input();
    in.dontcare_enabled = 1;
    in.first_use_this_frame = 1;
    in.blend_enabled = 1;
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "color blend plan");
    CHECK(plan.load_action == MGLLoadActionLoad,
          "blending forbids discarding the contents");

    in = color_input();
    in.dontcare_enabled = 1;
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "color reuse plan");
    CHECK(plan.load_action == MGLLoadActionLoad,
          "a later use this frame keeps the contents");

    in = color_input();
    in.first_use_this_frame = 1;
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "color flag-off plan");
    CHECK(plan.load_action == MGLLoadActionLoad,
          "discarding needs the env flag");

    in = color_input();
    in.dontcare_enabled = 1;
    in.first_use_this_frame = 1;
    in.texture_present = 0;
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "color no-texture plan");
    CHECK(plan.load_action == MGLLoadActionLoad,
          "a slot without a texture loads");

    in = color_input();
    in.attachment_present = 0;
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "color unattached plan");
    CHECK(plan.load_action == MGLLoadActionLoad,
          "an unresolved draw slot loads");
}

static void test_depth_stencil_load_store(void)
{
    MGLRenderPassLoadStorePlan plan;

    MGLRenderPassLoadStoreInput in = depth_input();
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "depth plan ok");
    CHECK(plan.load_action == MGLLoadActionLoad, "depth loads by default");
    CHECK(plan.set_store_action == 1 &&
              plan.store_action == MGLStoreActionStore,
          "depth with a texture stores");

    in = depth_input();
    in.texture_present = 0;
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "depth empty plan");
    CHECK(plan.load_action == MGLLoadActionLoad, "depth without a texture loads");
    CHECK(plan.set_store_action == 0,
          "depth without a texture leaves the store action");

    in = depth_input();
    in.has_clear_pending = 1;
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "depth clear plan");
    CHECK(plan.load_action == MGLLoadActionClear &&
              plan.set_store_action == 1 &&
              plan.store_action == MGLStoreActionStore,
          "depth clear wins over everything");

    in = depth_input();
    in.attachment_kind = MGL_RP_ATTACHMENT_STENCIL;
    CHECK(mglRenderPassPlanLoadStore(&in, &plan) == 0, "stencil plan");
    CHECK(plan.load_action == MGLLoadActionLoad &&
              plan.set_store_action == 1,
          "stencil follows the depth rules");

    CHECK(mglRenderPassPlanLoadStore(NULL, &plan) == -1,
          "NULL input is refused");
    CHECK(mglRenderPassPlanLoadStore(&in, NULL) == -1, "NULL output is refused");
}

static void test_stale_clear_bits(void)
{
    /* Attachment 1 has a pending clear but is not attached. */
    CHECK(mglRenderPassDropsStaleColorClear(0x2u, 0x1u, 1u) == 1,
          "an unattached pending clear is dropped");
    CHECK(mglRenderPassDropsStaleColorClear(0x3u, 0x3u, 0u) == 0,
          "an attached pending clear is kept");
    CHECK(mglRenderPassDropsStaleColorClear(0x0u, 0x0u, 0u) == 0,
          "no pending clear stays clear");
    CHECK(mglRenderPassDropsStaleColorClear(0x2u, 0x1u, 33u) == 0,
          "out-of-range index is not dropped");
}

static void test_attachment_match(void)
{
    const void *tex_a = (const void *)(uintptr_t)0x1000;
    const void *tex_b = (const void *)(uintptr_t)0x2000;

    MGLRenderPassAttachmentMatchEntry entries[3];
    memset(entries, 0, sizeof(entries));
    entries[0].compare = 1; /* a color draw slot */
    entries[0].actual_texture = tex_a;
    entries[0].expected_texture = tex_a;
    entries[0].compare_subresource = 1;
    entries[0].compare_subresource = 1;
    entries[0].actual_sub.level = 1;
    entries[0].expected_sub.level = 1;
    entries[1].compare = 0; /* not a draw slot of this framebuffer */
    entries[1].actual_texture = tex_b; /* stale texture the pass still carries */
    entries[2].compare = 1; /* the depth attachment */
    entries[2].actual_texture = tex_b;
    entries[2].expected_texture = tex_b;

    MGLRenderPassAttachmentMatchInput in;
    memset(&in, 0, sizeof(in));
    in.identity_ok = 1;
    in.entries = entries;
    in.entry_count = 3;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 1,
          "matching attachments match, uncompared ones are ignored");

    in.identity_ok = 0;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a different framebuffer never matches");
    in.identity_ok = 1;

    entries[0].expected_texture = tex_b;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a different color texture does not match");
    entries[0].expected_texture = tex_a;

    /* The subresource must line up with the framebuffer's. */
    entries[0].expected_sub.level = 2;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a different mip level does not match");
    entries[0].expected_sub.level = 1;
    entries[0].actual_sub.slice = 3;
    entries[0].expected_sub.slice = 4;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a different array slice does not match");
    entries[0].actual_sub.slice = 0;
    entries[0].expected_sub.slice = 0;
    entries[0].actual_sub.depth_plane = 1;
    entries[0].expected_sub.depth_plane = 0;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a different depth plane does not match");
    entries[0].actual_sub.depth_plane = 0;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 1,
          "matching subresources match");

    /* A subresource is only meaningful when both sides have a texture. */
    entries[0].actual_texture = NULL;
    entries[0].expected_texture = NULL;
    entries[0].expected_sub.level = 7;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 1,
          "absent textures do not compare subresources");
    entries[0].actual_texture = tex_a;
    entries[0].expected_texture = tex_a;
    entries[0].expected_sub.level = 1;

    /* A required attachment that is missing never matches. */
    entries[2].expected_texture = NULL;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a stale depth attachment does not match");
    entries[2].actual_texture = NULL;
    entries[2].required = 1;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a required but missing depth attachment does not match");
    entries[2].required = 0;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 1,
          "an optional depth attachment matches when it is absent");

    /* Everything absent matches. */
    in.entry_count = 0;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 1,
          "nothing to compare matches");
    in.entry_count = 3;

    CHECK(mglRenderPassAttachmentsMatch(NULL) == 0, "NULL input does not match");

    /* The fill helper takes what the caller resolved. */
    {
        MGLRenderPassAttachmentMatchEntry filled;
        const MGLRenderPassSubresource actual_sub = {2u, 3u, 4u};
        const MGLRenderPassSubresource expected_sub = {2u, 3u, 4u};
        mglRenderPassFillMatchEntry(&filled, tex_a, tex_a, 1, 1, actual_sub,
                                    expected_sub);
        CHECK(filled.compare == 1 && filled.required == 1 &&
                  filled.compare_subresource == 1,
              "fill sets the flags");
        CHECK(filled.actual_texture == tex_a &&
                  filled.expected_texture == tex_a,
              "fill copies the textures");
        MGLRenderPassAttachmentMatchInput filledIn = {0};
        filledIn.identity_ok = 1;
        filledIn.entries = &filled;
        filledIn.entry_count = 1;
        CHECK(mglRenderPassAttachmentsMatch(&filledIn) == 1,
              "a filled entry matches");
        mglRenderPassFillMatchEntry(&filled, tex_a, tex_b, 0, 0, actual_sub,
                                    expected_sub);
        CHECK(mglRenderPassAttachmentsMatch(&filledIn) == 0,
              "a filled mismatching entry does not match");
        mglRenderPassFillMatchEntry(NULL, tex_a, tex_a, 0, 0, actual_sub,
                                    expected_sub);
        CHECK(1, "fill tolerates a NULL entry");
    }
}

int main(void)
{
    printf("render-pass load/store harness (O3.1)\n");
    test_color_load_store();
    test_depth_stencil_load_store();
    test_stale_clear_bits();
    test_attachment_match();
    printf("test_render_pass_load_store_plan: %d/%d passed\n", tests_passed,
           tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
