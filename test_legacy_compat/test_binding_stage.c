/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Unit tests for mgl_binding_stage (O3.3). No Metal required.
 */

#include "mgl_binding_stage.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int g_fails;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

static MGLStageBufferBindInput base_ubo(void)
{
    MGLStageBufferBindInput in;
    memset(&in, 0, sizeof(in));
    in.phase = MGL_SB_PHASE_PRE_MTL;
    in.is_fragment = 0;
    in.is_base_binding = 1;
    in.has_metal_binding = 1;
    in.metal_binding_index = 3;
    in.gl_binding_index = 1;
    in.resource_type = 1u; /* UBO */
    in.offset = 0;
    in.buffer_size = 256;
    in.has_buffer = 1;
    in.has_cpu_data = 1;
    in.cpu_ptr = (const void *)(uintptr_t)0x200000000ULL;
    in.max_metal_slots = 31u;
    in.max_gl_bindings = 84u;
    in.reflected_required = 256u;
    in.min_stage_bytes = 256u;
    in.scratch_cap = 4096u;
    in.visible_cpu = 256u;
    in.visible_range = 256;
    return in;
}

static void test_helpers(void)
{
    expect(mglRenderUseInlineFragmentBytes(0, 100) == 1, "fs small inline");
    expect(mglRenderUseInlineFragmentBytes(1, 100) == 0, "base not fs inline");
    expect(mglRenderUseUniformConstantInline(1, 2, 1, 0, 64, 4096) == 1,
           "uc inline");
    expect(mglRenderWritableStorageNeedsGPUAuthoritative(3) == 1, "ssbo auth");
    expect(mglRenderWritableStorageNeedsGPUAuthoritative(9) == 1, "atomic auth");
    expect(mglRenderWritableStorageNeedsGPUAuthoritative(1) == 0, "ubo not auth");
    expect(mglRenderRequiredBindingBytesForMap(1, 128, 64, 256) == 64u,
           "ubo visible clamp");
    expect(mglRenderNeedsIsolatedStageBinding(1, 0, 100, 50, 256) == 1,
           "isolate undersized");
}

static void test_fallback_table(void)
{
    uint32_t types[8];
    uint32_t n = mglBindingStageFallbackResourceTypes(types, 8);
    uint8_t present[8] = {1, 0, 0, 1, 0, 0, 0, 1};
    uint8_t pad[64];
    const char *src = "abcd";
    const void *out;
    expect(n == 4u, "fallback count");
    expect(types[0] == 1u && types[1] == 2u && types[2] == 3u && types[3] == 9u,
           "fallback types UBO/UC/SSBO/atomic");
    expect(mglBindingStageFallbackNeedsBind(0, 1) == 1, "needs fallback");
    expect(mglBindingStageFallbackNeedsBind(1, 1) == 0, "already present");
    expect(mglBindingStagePlanFallbackSlot(1, 1, 1, 0) == MGL_FB_SLOT_SKIP,
           "present skip");
    expect(mglBindingStagePlanFallbackSlot(0, 0, 1, 0) == MGL_FB_SLOT_SKIP,
           "no buf skip");
    expect(mglBindingStagePlanFallbackSlot(0, 1, 1, 1) == MGL_FB_SLOT_MATCHED,
           "matched");
    expect(mglBindingStagePlanFallbackSlot(0, 1, 1, 0) == MGL_FB_SLOT_EMIT,
           "emit");
    expect(mglBindingStagePlanFallbackSlot(0, 1, 0, 0) == MGL_FB_SLOT_EMIT,
           "invalid emit");
    expect(mglBindingStageBuildPresentMask(present, 8) == 0x89u, "present mask");
    expect(mglBindingStageCountPresent(present, 8) == 3u, "present count");
    out = mglBindingStageInlineBytesSrc(pad, 64, src, 4, 4);
    expect(out == src, "inline no pad");
    out = mglBindingStageInlineBytesSrc(pad, 64, src, 2, 8);
    expect(out == pad && pad[0] == 'a' && pad[1] == 'b' && pad[7] == 0, "inline pad");
    {
        int overflow = 0;
        expect(mglBindingStageClampMapCount(3, 8, &overflow) == 3u && overflow == 0,
               "clamp ok");
        expect(mglBindingStageClampMapCount(12, 8, &overflow) == 8u && overflow == 1,
               "clamp overflow");
        expect(mglBindingStagePostMtlUsable(0, (const void *)(uintptr_t)0x10000u, 0) == 1,
               "vs usable");
        expect(mglBindingStagePostMtlUsable(0, (const void *)(uintptr_t)0x100u, 0) == 0,
               "vs unusable");
        expect(mglBindingStagePostMtlUsable(
                   1, (const void *)(uintptr_t)0x100000000ULL, 0) == 1,
               "fs high usable");
        expect(mglBindingStagePostMtlUsable(
                   1, (const void *)(uintptr_t)0x10000u, 1) == 0,
               "fs inline suppresses mid");
        expect(mglBindingStagePostMtlUsable(
                   1, (const void *)(uintptr_t)0x10000u, 0) == 1,
               "fs mid usable when not inline");
    }
}

static void test_plan_skip_and_clear(void)
{
    MGLStageBufferBindInput in = base_ubo();
    MGLStageBufferBindPlan plan = {0};
    in.is_base_binding = 0;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "plan ok");
    expect(plan.action == MGL_SB_ACTION_SKIP, "vertex non-base skip");

    in = base_ubo();
    in.has_buffer = 0;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "plan ok");
    expect(plan.action == MGL_SB_ACTION_CLEAR, "null buffer clear");
    expect(plan.mark_base_present == 1, "base present marked");

    in = base_ubo();
    in.attrib_slot_reserved = 1;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "plan ok");
    expect(plan.action == MGL_SB_ACTION_SKIP, "attrib reserved skip");
}

static void test_plan_uc_inline_and_need_mtl(void)
{
    MGLStageBufferBindInput in = base_ubo();
    MGLStageBufferBindPlan plan = {0};
    in.resource_type = 2u; /* uniform constant */
    in.reflected_required = 64u;
    in.visible_cpu = 64u;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "uc plan");
    expect(plan.action == MGL_SB_ACTION_INLINE_BYTES, "uc inline");
    /* required floors to min_stage(256) for non-UBO; inline pads to required */
    expect(plan.inline_length == 256u, "uc len padded to min_stage");
    expect(plan.inline_visible == 64u, "uc visible");

    in = base_ubo();
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "ubo pre");
    expect(plan.action == MGL_SB_ACTION_NEED_MTL, "ubo needs mtl");
}

static void test_plan_post_isolate_and_bind(void)
{
    MGLStageBufferBindInput in = base_ubo();
    MGLStageBufferBindPlan plan = {0};
    in.phase = MGL_SB_PHASE_POST_MTL;
    in.has_mtl_data = 1;
    in.mtl_usable = 1;
    in.mtl_ptr = (const void *)(uintptr_t)0x200000000ULL;
    in.metal_len = 64u;
    in.visible_mtl = 64u;
    in.reflected_required = 256u;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "post isolate");
    expect(plan.action == MGL_SB_ACTION_ISOLATE, "isolate undersized mtl");
    expect(plan.needs_flush_snapshot == 1, "isolate flush");
    /* An isolated copy is bound at 0, so bind_offset is NOT the caller's map
     * offset: a copy-back destination must come from the map entry. */
    expect(plan.bind_offset == 0u, "isolate binds at offset 0");

    in.visible_mtl = 256u;
    in.metal_len = 256u;
    in.binding_state_valid = 1;
    in.buffer_matches = 1;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "post match");
    expect(plan.action == MGL_SB_ACTION_SKIP_MATCHED, "matched skip");

    in.buffer_matches = 0;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "post bind");
    expect(plan.action == MGL_SB_ACTION_BIND_BUFFER, "bind buffer");
}

static void test_plan_fs_small(void)
{
    MGLStageBufferBindInput in;
    memset(&in, 0, sizeof(in));
    in.phase = MGL_SB_PHASE_PRE_MTL;
    in.is_fragment = 1;
    in.is_base_binding = 0;
    in.gl_binding_index = 2;
    in.buffer_size = 128;
    in.has_buffer = 1;
    in.has_cpu_data = 1;
    in.cpu_ptr = (const void *)(uintptr_t)0x200000000ULL;
    in.offset = 16;
    in.max_metal_slots = 84u;
    in.max_gl_bindings = 84u;
    in.min_stage_bytes = 256u;
    in.scratch_cap = 4096u;
    MGLStageBufferBindPlan plan = {0};
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "fs small");
    expect(plan.action == MGL_SB_ACTION_INLINE_BYTES, "fs inline bytes");
    expect(plan.inline_src_offset == 16u, "fs offset");
    expect(plan.inline_length == 112u, "fs length");
}


static void test_fill_helpers(void)
{
    MGLStageBufferBindInput bin;
    mglBindingStageFillMapEntryInput(
        &bin, 0, MGL_SB_PHASE_PRE_MTL, 1, 1, 3, 1, 1u, 0, 256, 1, 1, 0,
        (const void *)(uintptr_t)0x200000000ULL, NULL, 0, 0, 0, 0, 31u, 84u,
        256u, 256u, 4096u, 256u, 256);
    expect(bin.phase == MGL_SB_PHASE_PRE_MTL, "fill phase");
    expect(bin.is_base_binding == 1, "fill base");
    expect(bin.metal_binding_index == 3, "fill metal");
    expect(bin.has_cpu_data == 1 && bin.mtl_usable == 0, "fill cpu/mtl");

    MGLAttribBindInput ain;
    mglBindingStageFillAttribSelectInput(&ain, 1, 0, 1, 2, 31u, 1, 0, 0, 0,
                                         16u, 1);
    expect(ain.phase == MGL_ATTR_PHASE_SELECT, "attr select phase");
    expect(ain.mapped_index == 2 && ain.binding_offset == 16u, "attr select");
    mglBindingStageFillAttribPostMtlInput(&ain, 1, 1, 1024u, 1, 0);
    expect(ain.phase == MGL_ATTR_PHASE_POST_MTL, "attr post phase");
    expect(ain.metal_len == 1024u && ain.buffer_matches == 0, "attr post");
    expect(mglBindingStageAttribNeedsEmit(1, 1) == 0, "no emit matched");
    expect(mglBindingStageAttribNeedsEmit(1, 0) == 1, "emit mismatch");
    expect(mglBindingStageAttribNeedsEmit(0, 1) == 1, "emit invalid");

    mglBindingStageFillMapEntryPostMtl(&bin, 1, (const void *)(uintptr_t)0x200000000ULL,
                                       1, 512u, 256u, 1, 0);
    expect(bin.phase == MGL_SB_PHASE_POST_MTL, "post phase");
    expect(bin.metal_len == 512u && bin.visible_mtl == 256u, "post lens");
    expect(bin.buffer_matches == 0 && bin.mtl_usable == 1, "post flags");
}

/* The reason table a materializing caller (compute) consumes: skip / refuse /
 * carry out, plus the isolate fallback floor. */
static void test_plan_disposition(void)
{
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_NOT_BASE) == 1,
           "non-base entry is skipped");
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_SLOT_OOR) == 1,
           "out-of-range slot is skipped");
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_ATTRIB_RESERVED) == 1,
           "attrib-reserved entry is skipped");
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_NEED_ENSURE) == 0,
           "ensure is carried out");
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_ISOLATE) == 0,
           "isolate is carried out");
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_BIND) == 0,
           "bind is carried out");
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_MATCHED) == 0,
           "matched is carried out (perf skip handled by the caller)");
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_NULL_BUFFER) == -1,
           "no backing is refused");
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_BAD_OFFSET) == -1,
           "bad offset is refused");
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_BAD_SIZE) == -1,
           "bad size is refused");
    expect(mglBindingStageMapEntryDisposition(MGL_SB_REASON_INLINE_UC) == -1,
           "an inline arm a caller cannot encode is refused");

    expect(strcmp(mglBindingStagePlanReasonName(MGL_SB_REASON_ISOLATE),
                  "isolate") == 0,
           "reason name isolate");
    expect(strcmp(mglBindingStagePlanReasonName(MGL_SB_REASON_SLOT_OOR),
                  "metal-slot-out-of-range") == 0,
           "reason name slot");
    expect(strcmp(mglBindingStagePlanReasonName(9999u), "unknown") == 0,
           "unknown reason name");

    expect(mglBindingStageIsolateFallbackLength(0u) == 4u,
           "isolate fallback floors at one uint32");
    expect(mglBindingStageIsolateFallbackLength(4u) == 4u,
           "isolate fallback keeps 4");
    expect(mglBindingStageIsolateFallbackLength(256u) == 256u,
           "isolate fallback keeps the required size");
}

/* The compute stage-binding path (O5.2) ports onto this plan with three
 * opt-in switches; they must leave the vertex/fragment plan untouched and
 * reproduce exactly what the compute loop decided by hand before. */
static void test_plan_compute_switches(void)
{
    MGLStageBufferBindPlan plan = {0};

    /* no_inline: a plain-uniform slot that the fragment plan inlines must
     * stay a real Metal binding (the compute encoder path has no set*Bytes). */
    MGLStageBufferBindInput in = base_ubo();
    in.resource_type = 2u; /* uniform constant */
    in.reflected_required = 64u;
    in.visible_cpu = 64u;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "uc plan");
    expect(plan.action == MGL_SB_ACTION_INLINE_BYTES, "uc inline by default");
    in.no_inline = 1;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "uc no-inline plan");
    expect(plan.action == MGL_SB_ACTION_NEED_MTL, "no_inline skips uc inline");

    /* Same switch on the fragment small path. */
    MGLStageBufferBindInput fs;
    memset(&fs, 0, sizeof(fs));
    fs.phase = MGL_SB_PHASE_PRE_MTL;
    fs.is_fragment = 1;
    fs.is_base_binding = 0;
    fs.gl_binding_index = 2;
    fs.buffer_size = 128;
    fs.has_buffer = 1;
    fs.has_cpu_data = 1;
    fs.cpu_ptr = (const void *)(uintptr_t)0x200000000ULL;
    fs.offset = 16;
    fs.max_metal_slots = 84u;
    fs.max_gl_bindings = 84u;
    fs.min_stage_bytes = 256u;
    fs.scratch_cap = 4096u;
    expect(mglBindingStagePlanMapEntry(&fs, &plan) == 0, "fs plan");
    expect(plan.action == MGL_SB_ACTION_INLINE_BYTES, "fs inline by default");
    fs.no_inline = 1;
    expect(mglBindingStagePlanMapEntry(&fs, &plan) == 0, "fs no-inline plan");
    expect(plan.action == MGL_SB_ACTION_NEED_MTL, "no_inline skips fs inline");

    /* An exhausted GL storage range isolates even with an adequate Metal
     * backing, but only when the caller opts in. */
    in = base_ubo();
    in.phase = MGL_SB_PHASE_POST_MTL;
    in.has_mtl_data = 1;
    in.mtl_usable = 1;
    in.mtl_ptr = (const void *)(uintptr_t)0x200000000ULL;
    in.metal_len = 256u;
    in.visible_mtl = 256u;
    in.reflected_required = 256u;
    in.min_stage_bytes = 0u;
    in.visible_range = 0;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "post plan");
    expect(plan.action == MGL_SB_ACTION_BIND_BUFFER, "bind by default");
    in.iso_storage_exhausted = 1;
    in.storage_remaining = 0;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "storage plan");
    expect(plan.action == MGL_SB_ACTION_ISOLATE, "exhausted storage isolates");
    in.storage_remaining = 64;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "storage ok plan");
    expect(plan.action == MGL_SB_ACTION_BIND_BUFFER,
           "storage left binds again");

    /* An empty visible backing isolates even when nothing is required. */
    in = base_ubo();
    in.phase = MGL_SB_PHASE_POST_MTL;
    in.has_mtl_data = 1;
    in.mtl_usable = 1;
    in.mtl_ptr = (const void *)(uintptr_t)0x200000000ULL;
    in.metal_len = 256u;
    in.visible_mtl = 0u;
    in.reflected_required = 0u;
    in.min_stage_bytes = 0u;
    in.visible_range = 0;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "empty plan");
    expect(plan.action == MGL_SB_ACTION_BIND_BUFFER, "empty binds by default");
    in.iso_empty_visible = 1;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "empty iso plan");
    expect(plan.action == MGL_SB_ACTION_ISOLATE, "empty visible isolates");

    /* Compute isolates a GPU write target too (it is the writer) and asks for
     * the post-dispatch copy back of a writable SSBO. */
    in = base_ubo();
    in.phase = MGL_SB_PHASE_POST_MTL;
    in.resource_type = 3u; /* storage buffer */
    in.has_mtl_data = 1;
    in.mtl_usable = 1;
    in.mtl_ptr = (const void *)(uintptr_t)0x200000000ULL;
    in.metal_len = 64u;
    in.visible_mtl = 64u;
    in.reflected_required = 256u;
    in.min_stage_bytes = 0u;
    in.visible_range = 0;
    in.gpu_write_target = 1;
    in.allow_isolate_when_gpu = 0;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "gpu plan");
    expect(plan.action == MGL_SB_ACTION_BIND_BUFFER,
           "gpu write target binds unless isolation is allowed");
    in.allow_isolate_when_gpu = 1;
    expect(mglBindingStagePlanMapEntry(&in, &plan) == 0, "gpu iso plan");
    expect(plan.action == MGL_SB_ACTION_ISOLATE, "compute isolates the writer");
    expect(plan.needs_copy_back == 1, "writable isolate copies back");
    expect(plan.needs_flush_snapshot == 1, "isolate flushes the snapshot");
}

int main(void)
{
    test_helpers();
    test_fill_helpers();
    test_fallback_table();
    test_plan_skip_and_clear();
    test_plan_uc_inline_and_need_mtl();
    test_plan_post_isolate_and_bind();
    test_plan_compute_switches();
    test_plan_disposition();
    test_plan_fs_small();
    if (g_fails) {
        fprintf(stderr, "%d failure(s)\n", g_fails);
        return 1;
    }
    puts("test_binding_stage: ok");
    return 0;
}
