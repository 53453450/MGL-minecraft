#include <gtest/gtest.h>

extern "C" {
#include "mgl_sync_domains.h"
#include "mgl_types_state.h"
}

TEST(StateDirty, MarkStateInvalidatesHashCaches)
{
    GLMState state = {};
    state.cached_texture_hash = 0x1111;
    state.cached_vertex_layout_hash = 0x2222;
    state.cached_render_state_hash = 0x3333;

    mglMarkStateDirtyBits(&state, DIRTY_TEX_BINDING | DIRTY_VAO | DIRTY_RENDER_STATE);

    EXPECT_EQ(1, state.texture_dirty);
    EXPECT_EQ(1, state.vertex_layout_dirty);
    EXPECT_EQ(1, state.render_state_dirty);
    EXPECT_EQ(1, state.uniform_buffer_dirty);
    EXPECT_NE(0u, state.dirty_bits & (DIRTY_TEX_BINDING | DIRTY_VAO | DIRTY_RENDER_STATE));
}

TEST(StateDirty, MarkRendererDoesNotInvalidateHashCaches)
{
    GLMState state = {};
    state.cached_texture_hash = 0x1111;
    state.texture_dirty = 0;

    mglMarkRendererDirtyBits(&state, DIRTY_TEX_BINDING);

    EXPECT_EQ(0, state.texture_dirty);
    EXPECT_NE(0u, state.dirty_bits & DIRTY_TEX_BINDING);
}

TEST(StateDirty, ProgramDirtyClearsSampledMaskValid)
{
    GLMState state = {};
    state.active_sampled_texture_unit_mask_valid = 1u;

    mglMarkStateDirtyBits(&state, DIRTY_PROGRAM);

    EXPECT_EQ(0u, state.active_sampled_texture_unit_mask_valid);
}

TEST(SyncDomains, ClassifyFboAndPipeline)
{
    uint32_t domains = mglRenderClassifyDirtySyncDomains(DIRTY_FBO | DIRTY_PROGRAM);
    EXPECT_NE(0u, domains & MGL_SYNC_DOMAIN_FBO);
    EXPECT_NE(0u, domains & MGL_SYNC_DOMAIN_PROGRAM_VAO);
    EXPECT_NE(0u, domains & MGL_SYNC_DOMAIN_PIPELINE);
}

TEST(SyncDomains, ClassifyTextureDomain)
{
    uint32_t domains = mglRenderClassifyDirtySyncDomains(DIRTY_TEX_BINDING);
    EXPECT_NE(0u, domains & MGL_SYNC_DOMAIN_TEX);
    EXPECT_EQ(0u, domains & MGL_SYNC_DOMAIN_FBO);
}
