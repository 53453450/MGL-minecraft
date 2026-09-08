/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * Unit tests for mglRenderProcessGLState (O1.1). No Metal required.
 */

#include "mgl_render_pass_plan.h"

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

static MGLProcessGLStateInputs base_draw(void)
{
    MGLProcessGLStateInputs in;
    memset(&in, 0, sizeof(in));
    in.has_ctx = 1u;
    in.draw_command = 1u;
    in.has_vao = 1u;
    in.ctx_ptr_sane = 1u;
    in.device_ok = 1u;
    in.queue_ok = 1u;
    in.has_command_buffer = 1u;
    in.encoder_has_current = 1u;
    return in;
}

static void test_classify_and_abort(void)
{
    expect(mglRenderClassifyProcessGLState(0, 1, 1, 0) == MGL_PROCESS_GL_ABORT,
           "no ctx → ABORT");
    expect(mglRenderClassifyProcessGLState(1, 1, 0, 0) == MGL_PROCESS_GL_ABORT,
           "draw without VAO → ABORT");
    expect(mglRenderClassifyProcessGLState(1, 0, 0, 1) ==
               MGL_PROCESS_GL_NO_VAO_CLEAR,
           "non-draw dirty no VAO → CLEAR");
    expect(mglRenderClassifyProcessGLState(1, 0, 1, 0) == MGL_PROCESS_GL_NON_DRAW,
           "non-draw → NON_DRAW");
    expect(mglRenderClassifyProcessGLState(1, 1, 1, 0) == MGL_PROCESS_GL_CONTINUE,
           "draw → CONTINUE");
}

static void test_early_paths(void)
{
    MGLProcessGLStateInputs in = base_draw();
    MGLProcessGLStatePlan plan = {0};
    in.has_vao = 0u;
    expect(mglRenderProcessGLState(&in, &plan) == 0, "plan ok");
    expect(plan.result == MGL_PGL_RESULT_ABORT, "draw no VAO aborts");

    in = base_draw();
    in.draw_command = 0u;
    expect(mglRenderProcessGLState(&in, &plan) == 0, "nondraw plan");
    expect(plan.result == MGL_PGL_RESULT_EARLY_OK, "nondraw early ok");
    expect(plan.non_draw_end_pass_if_fbo_changed == 1u, "nondraw ends pass");
    expect(plan.process_dirty_domains == 0u, "nondraw skips dirty");

    in = base_draw();
    in.quarantine_blocks_draw = 1u;
    expect(mglRenderProcessGLState(&in, &plan) == 0, "quarantine plan");
    expect(plan.result == MGL_PGL_RESULT_ABORT, "quarantine aborts");
}

static void test_continue_cb_and_after(void)
{
    MGLProcessGLStateInputs in = base_draw();
    MGLProcessGLStatePlan plan = {0};
    in.encoder_has_current = 0u;
    in.command_buffer_status = MGL_PGL_CB_STATUS_COMMITTED;
    expect(mglRenderProcessGLState(&in, &plan) == 0, "continue plan");
    expect(plan.result == MGL_PGL_RESULT_CONTINUE, "continue");
    expect(plan.clear_rt_sampled_copy == 1u, "clear RT flag");
    expect(plan.rotate_finalized_command_buffer == 1u, "rotate finalized CB");
    expect(plan.process_dirty_domains == 1u, "dirty domains");

    in = base_draw();
    in.has_command_buffer = 0u;
    expect(mglRenderProcessGLState(&in, &plan) == 0, "create CB plan");
    expect(plan.create_initial_command_buffer == 1u, "create initial CB");

    MGLProcessGLStateAfterInputs after_in = {0};
    MGLProcessGLStateAfterPlan after = {0};
    after_in.draw_command = 1u;
    after_in.encoder_has_current = 0u;
    after_in.has_pipeline_state = 0u;
    expect(mglRenderProcessGLStateAfterDirty(&after_in, &after) == 0, "after");
    expect(after.recover_nil_encoder == 1u, "recover encoder");
    expect(after.fail_nil_pipeline == 1u, "fail nil pipeline");
    expect(after.result == MGL_PGL_RESULT_ABORT, "after abort");

    after_in.has_pipeline_state = 1u;
    after_in.encoder_has_current = 1u;
    after_in.frag_needs_fragcoord = 1u;
    after_in.frag_needs_lod_bias = 1u;
    after_in.fragment_trace_uses_rt_sampled_copy = 1u;
    expect(mglRenderProcessGLStateAfterDirty(&after_in, &after) == 0, "after ok");
    expect(after.result == MGL_PGL_RESULT_CONTINUE, "after continue");
    expect(after.validate_attachments == 1u, "validate");
    expect(after.set_pipeline == 1u, "set pipeline");
    expect(after.bind_frag_coord_slot == 1u, "fragcoord");
    expect(after.bind_lod_bias_slot == 1u, "lod bias");
    expect(after.maybe_mark_rt_sampled_copy == 1u, "rt sampled");
}

int main(void)
{
    test_classify_and_abort();
    test_early_paths();
    test_continue_cb_and_after();
    if (g_fails) {
        fprintf(stderr, "test_process_gl_state_plan: %d failure(s)\n", g_fails);
        return 1;
    }
    printf("test_process_gl_state_plan: ok\n");
    return 0;
}
