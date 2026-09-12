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

    MGLRenderPassSlotMatch slots[2];
    memset(slots, 0, sizeof(slots));
    slots[0].compare = 1;
    slots[0].actual = tex_a;
    slots[0].expected = tex_a;
    slots[1].compare = 0; /* not a draw slot of this framebuffer */
    slots[1].actual = tex_b; /* stale texture the pass still carries */
    slots[1].expected = NULL;

    MGLRenderPassAttachmentMatchInput in;
    memset(&in, 0, sizeof(in));
    in.identity_ok = 1;
    in.slots = slots;
    in.slot_count = 2;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 1,
          "matching slots match, uncompared slots are ignored");

    in.identity_ok = 0;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a different framebuffer never matches");
    in.identity_ok = 1;

    slots[0].expected = tex_b;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a different color texture does not match");
    slots[0].expected = tex_a;

    in.actual_depth = tex_a;
    in.expected_depth = NULL;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a stale depth attachment does not match");

    in.actual_depth = NULL;
    in.expected_depth = tex_a;
    in.depth_required = 1;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a required but missing depth attachment does not match");

    in.actual_depth = tex_a;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 1,
          "an optional depth attachment matches when it is there");

    in.depth_required = 0;
    in.actual_depth = NULL;
    in.expected_depth = NULL;
    in.actual_stencil = NULL;
    in.expected_stencil = tex_b;
    in.stencil_required = 1;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 0,
          "a required but missing stencil attachment does not match");

    in.expected_stencil = NULL;
    in.stencil_required = 0;
    CHECK(mglRenderPassAttachmentsMatch(&in) == 1,
          "everything absent matches");

    CHECK(mglRenderPassAttachmentsMatch(NULL) == 0, "NULL input does not match");
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
