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

int main(void)
{
    test_same_key_skip();
    test_dirty_delta();
    if (g_fails) {
        fprintf(stderr, "test_batch_restore: %d fail(s)\n", g_fails);
        return 1;
    }
    printf("test_batch_restore: ok\n");
    return 0;
}
