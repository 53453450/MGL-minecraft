/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Unit tests for mgl_batch_same_key_skip_decision /
 * mgl_batch_compute_key_delta_dirty_bits (A3 / O2.5). No Metal.
 */

#include "mgl_batch_restore.h"

#include <stdio.h>
#include <string.h>

static int g_fails;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

static MGLBatchSameKeySkipIn base_skip(void)
{
    MGLBatchSameKeySkipIn in;
    memset(&in, 0, sizeof(in));
    in.skip_enabled = 1u;
    in.last_key_valid = 1u;
    in.last_execute_ok = 1u;
    in.has_encoder = 1u;
    in.bind_valid = 1u;
    in.keys_equal = 1u;
    in.absolute_offsets_match = 1u;
    in.pass_matches = 1u;
    return in;
}

static void test_same_key_skip(void)
{
    MGLBatchSameKeySkipIn in = base_skip();
    expect(mgl_batch_same_key_skip_decision(&in) == MGL_BATCH_SAME_KEY_SKIP,
           "all gates → SKIP");
    in.last_was_stream = 1u;
    expect(mgl_batch_same_key_skip_decision(&in) == MGL_BATCH_SAME_KEY_NO_SKIP,
           "stream batch → NO_SKIP");
    in = base_skip();
    in.has_encoder = 0u;
    expect(mgl_batch_same_key_skip_decision(&in) ==
               MGL_BATCH_SAME_KEY_FAIL_NO_ENCODER,
           "no encoder → FAIL_NO_ENCODER");
    in = base_skip();
    in.bind_valid = 0u;
    expect(mgl_batch_same_key_skip_decision(&in) == MGL_BATCH_SAME_KEY_FAIL_BIND,
           "bind invalid → FAIL_BIND");
    in = base_skip();
    in.keys_equal = 0u;
    expect(mgl_batch_same_key_skip_decision(&in) == MGL_BATCH_SAME_KEY_FAIL_KEY,
           "key differ → FAIL_KEY");
    in = base_skip();
    in.pass_matches = 0u;
    expect(mgl_batch_same_key_skip_decision(&in) == MGL_BATCH_SAME_KEY_FAIL_PASS,
           "pass mismatch → FAIL_PASS");
    in = base_skip();
    in.skip_enabled = 0u;
    expect(mgl_batch_same_key_skip_decision(&in) == MGL_BATCH_SAME_KEY_NO_SKIP,
           "disabled → NO_SKIP");
}

static void test_dirty_delta(void)
{
    MGLBatchDirtyDomainMasks masks = {
        .program = 0x1u,
        .vao = 0x2u,
        .texture = 0x4u,
        .render_state = 0x8u,
    };
    const uint32_t full = 0xFu;
    MGLBatchStateKeyView a;
    MGLBatchStateKeyView b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    MGLBatchDirtyDeltaFlags flags;
    uint32_t bits = mgl_batch_compute_key_delta_dirty_bits(0, &a, &b, full,
                                                           &masks, &flags);
    expect(bits == full && flags.narrowed == 0u, "can_delta=0 → full");

    bits = mgl_batch_compute_key_delta_dirty_bits(1, &a, &b, full, &masks,
                                                  &flags);
    expect(bits == 0u && flags.narrowed == 1u, "identical keys → empty narrow");

    b.program_name = 7u;
    bits = mgl_batch_compute_key_delta_dirty_bits(1, &a, &b, full, &masks,
                                                  &flags);
    expect(bits == masks.program && flags.domain_program == 1u,
           "program change");

    memset(&b, 0, sizeof(b));
    b.texture_hash = 99u;
    bits = mgl_batch_compute_key_delta_dirty_bits(1, &a, &b, full, &masks,
                                                  &flags);
    expect(bits == masks.texture && flags.domain_texture == 1u,
           "texture hash change");

    memset(&b, 0, sizeof(b));
    b.render_state_hash = 0xABCD0000u;
    b.uniform_buffer_hash = 0xABCD0000u; /* XOR equal to a (0^0 vs ABCD^ABCD) */
    a.render_state_hash = 0u;
    a.uniform_buffer_hash = 0u;
    /* Make XOR of (rs ^ ubo) match while rs differs: a: 1^1=0, b: 2^2=0 */
    a.render_state_hash = 1u;
    a.uniform_buffer_hash = 1u;
    b.render_state_hash = 2u;
    b.uniform_buffer_hash = 2u;
    bits = mgl_batch_compute_key_delta_dirty_bits(1, &a, &b, full, &masks,
                                                  &flags);
    expect(bits == masks.render_state &&
               flags.domain_render_state_ubo_only == 1u &&
               flags.domain_render_state == 0u,
           "ubo-only render_state_hash noise");
}

static void test_fbo_fold(void)
{
    const uint32_t full = 0xFFu;
    const uint32_t dirty_fbo = 0x80u;
    MGLBatchRestoreFboIn in;
    memset(&in, 0, sizeof(in));
    in.has_encoder = 1u;
    in.bind_valid = 1u;
    in.pass_matches = 1u;
    uint32_t bits = mgl_batch_restore_fold_fbo_dirty(0x1u, full, dirty_fbo, &in);
    expect(bits == 0x1u, "no fbo pressure → unchanged");
    in.prev_fbo_differs = 1u;
    bits = mgl_batch_restore_fold_fbo_dirty(0x1u, full, dirty_fbo, &in);
    expect(bits == (0x1u | dirty_fbo), "prev fbo differ → OR DIRTY_FBO");
    in.prev_fbo_differs = 0u;
    in.has_encoder = 0u;
    bits = mgl_batch_restore_fold_fbo_dirty(0x1u, full, dirty_fbo, &in);
    expect(bits == full, "empty encoder → full domains");
    in.has_encoder = 1u;
    in.bind_valid = 0u;
    in.fbo_binding_dirty = 1u;
    bits = mgl_batch_restore_fold_fbo_dirty(0u, full, dirty_fbo, &in);
    expect((bits & dirty_fbo) != 0u && (bits & full) == full,
           "invalid bind + fbo dirty → full|FBO");
    expect(mgl_batch_restore_absolute_contract_dirty(1, 0, 0x3u) == 0x3u,
           "absolute contract flip");
    expect(mgl_batch_restore_absolute_contract_dirty(1, 1, 0x3u) == 0u,
           "absolute contract same");
}

static void test_restore_encode_fold(void)
{
    expect(mgl_batch_restore_full_dirty_bits() != 0u, "full dirty nonzero");
    MGLBatchDirtyDomainMasks masks;
    mgl_batch_restore_default_domain_masks(&masks);
    expect(masks.program != 0u && masks.vao != 0u, "domain masks");
    expect(mgl_batch_restore_can_delta(1, 1, 1, 1) == 1, "can delta");
    expect(mgl_batch_restore_can_delta(1, 0, 1, 1) == 0, "no prev");
    expect(mgl_batch_restore_can_delta(0, 1, 1, 1) == 0, "disabled");
}


static int g_key_steps;
static void key_prog(void *ctx, uint32_t program, uint32_t pipeline)
{
    (void)ctx;
    (void)program;
    (void)pipeline;
    g_key_steps += 1;
}
static void key_vao(void *ctx, uint32_t name)
{
    (void)ctx;
    (void)name;
    g_key_steps += 2;
}
static void key_fbo(void *ctx, uint32_t name)
{
    (void)ctx;
    (void)name;
    g_key_steps += 4;
}
static void key_sync(void *ctx)
{
    (void)ctx;
    g_key_steps += 8;
}
static void key_vpsc(void *ctx, const int32_t viewport[4], int scissor_enabled,
                     const int32_t scissor[4])
{
    (void)ctx;
    (void)viewport;
    (void)scissor;
    g_key_steps += scissor_enabled ? 16 : 32;
}

static void test_restore_from_key(void)
{
    g_key_steps = 0;
    MGLBatchRestoreFromKeyOps ops;
    memset(&ops, 0, sizeof(ops));
    ops.program_name = 1;
    ops.program_pipeline_name = 2;
    ops.vao_name = 3;
    ops.fbo_name = 4;
    ops.scissor_enabled = 1;
    ops.viewport[0] = 1; ops.viewport[1] = 2; ops.viewport[2] = 3; ops.viewport[3] = 4;
    ops.scissor[0] = 5; ops.scissor[1] = 6; ops.scissor[2] = 7; ops.scissor[3] = 8;
    ops.restore_program = key_prog;
    ops.set_vao = key_vao;
    ops.set_fbo = key_fbo;
    ops.sync_fbo_names = key_sync;
    ops.apply_viewport_scissor = key_vpsc;
    mgl_batch_restore_apply_from_key(&ops);
    expect(g_key_steps == (1 + 2 + 4 + 8 + 16), "restore from key steps");
}

static void test_restore_residual(void)
{
    expect(mgl_batch_restore_oracle_would_skip(0, 1, 1, 1) == 1, "oracle on");
    expect(mgl_batch_restore_oracle_would_skip(1, 1, 1, 1) == 0, "oracle off when skip on");
    expect(mgl_batch_restore_oracle_would_skip(0, 0, 1, 1) == 0, "oracle needs last key");
    MGLBatchStateKeyView a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    b.program_name = 1u;
    MGLBatchDirtyDeltaFlags flags;
    uint32_t bits = mgl_batch_restore_plan_delta_dirty(1, &a, &b, 0xFu, &flags);
    expect(bits != 0u && flags.domain_program == 1u, "plan delta dirty");
    MGLBatchRestoreFboIn fbo;
    memset(&fbo, 0, sizeof(fbo));
    fbo.has_encoder = 1u;
    fbo.bind_valid = 1u;
    fbo.pass_matches = 1u;
    expect(mgl_batch_restore_finish_dirty(0x1u, 0x2u, 0xFu, 0x80u, &fbo) == 0x3u,
           "finish dirty ors forced");
}


static int g_rest_steps;
static void rest_snap(void *v) { (void)v; g_rest_steps |= 1; }
static void rest_key(void *v) { (void)v; g_rest_steps |= 2; }
static void rest_after(void *v) { (void)v; g_rest_steps |= 4; }
static int rest_can(void *v) { (void)v; return 0; }
static void rest_fbo(void *v, MGLBatchRestoreFboIn *f)
{
    (void)v;
    memset(f, 0, sizeof(*f));
    f->has_encoder = 1u;
    f->bind_valid = 1u;
    f->pass_matches = 1u;
}
static void rest_mark(void *v, uint32_t bits)
{
    uint32_t *out = (uint32_t *)v;
    *out = bits;
}

static void test_restore_run(void)
{
    uint32_t marked = 0;
    g_rest_steps = 0;
    MGLBatchRestoreForBatchOps ops = {
        .ctx = &marked,
        .has_snapshot = 1,
        .forced_bits = 0x2u,
        .dirty_fbo_mask = 0x80u,
        .apply_snapshot = rest_snap,
        .apply_from_key = rest_key,
        .after_apply = rest_after,
        .can_delta = rest_can,
        .fill_fbo = rest_fbo,
        .mark_dirty = rest_mark,
    };
    mgl_batch_restore_run_for_batch(&ops);
    expect((g_rest_steps & 1) && (g_rest_steps & 4) && !(g_rest_steps & 2),
           "snapshot path");
    expect(marked != 0u, "mark dirty called");
    g_rest_steps = 0;
    ops.has_snapshot = 0;
    mgl_batch_restore_run_for_batch(&ops);
    expect((g_rest_steps & 2) && (g_rest_steps & 4), "key path");
}

static int g_td_steps;
static void td_step(void *v)
{
    int *p = (int *)v;
    (*p)++;
    g_td_steps++;
}
static void test_teardown_run(void)
{
    int n = 0;
    g_td_steps = 0;
    MGLBatchTeardownOps ops = {
        .ctx = &n,
        .used_replay_workspace = 1,
        .arena_snapshot_enabled = 1,
        .sync_hash_from_replay = td_step,
        .restore_live_active = td_step,
        .clear_absolute_offsets = td_step,
        .reset_command_buffer = td_step,
        .reset_arena = td_step,
        .restore_saved_state = td_step,
        .clear_dirty_preserve_hash = td_step,
        .restore_program_pair = td_step,
        .propagate_replay_error = td_step,
    };
    mgl_batch_teardown_run(&ops);
    expect(g_td_steps == 8, "teardown skips restore_saved when replay ws");
    g_td_steps = 0;
    ops.used_replay_workspace = 0;
    mgl_batch_teardown_run(&ops);
    expect(g_td_steps == 8, "teardown live path uses restore_saved not sync");
}

int main(void)
{
    test_same_key_skip();
    test_dirty_delta();
    test_restore_encode_fold();
    test_fbo_fold();
    test_restore_residual();
    test_restore_from_key();
    test_restore_run();
    test_teardown_run();
    if (g_fails) {
        fprintf(stderr, "test_batch_restore: %d fail(s)\n", g_fails);
        return 1;
    }
    printf("test_batch_restore: ok\n");
    return 0;
}
