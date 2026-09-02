#include <gtest/gtest.h>

extern "C" {
#include "mgl_sync_domains.h"
#include "mgl_pipeline_cache_key.h"
#include "mgl_pipeline_recovery.h"
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

TEST(PipelineCacheKey, PrimaryKeyPacksProgramAndClipState)
{
    MGLPipelineCacheKeyInputs inputs = {
        .program_name = 42u,
        .clip_origin = 0xAu,
        .clip_depth_mode = 0x5u,
        .tess_flags = 0u,
    };
    uint64_t primary = mglPipelineCacheBuildPrimaryKey(&inputs);
    EXPECT_EQ(42ull << 32, primary & (0xFFFFFFFFull << 32));
    EXPECT_EQ(0xAull << 28, primary & (0xFull << 28));
    EXPECT_EQ(0x5ull << 24, primary & (0xFull << 24));
}

TEST(PipelineCacheKey, TessFlagsOccupyPrimaryKeyBits)
{
    uint32_t flags = mglPipelineCachePackTessFlags(true, true, false, true, true);
    MGLPipelineCacheKeyInputs inputs = {
        .program_name = 1u,
        .tess_flags = flags,
    };
    uint64_t primary = mglPipelineCacheBuildPrimaryKey(&inputs);
    EXPECT_NE(0ull, primary & (1ull << 23));
    EXPECT_NE(0ull, primary & (1ull << 22));
    EXPECT_EQ(0ull, primary & (1ull << 21));
    EXPECT_NE(0ull, primary & (1ull << 20));
    EXPECT_NE(0ull, primary & (1ull << 19));
}

TEST(PipelineCacheKey, LookupKeyWordsRoundTrip)
{
    MGLPipelineCacheKeyInputs inputs = {
        .program_name = 7u,
        .clip_origin = 1u,
        .clip_depth_mode = 2u,
        .tess_flags = MGL_PIPELINE_TESS_FLAG_GEOMETRY_EXPANSION,
        .vertex_instance_id = 11u,
        .vertex_generation = 12u,
        .fragment_instance_id = 13u,
        .fragment_generation = 14u,
        .pipeline_sig = 0xAAAu,
        .vertex_sig = 0xBBBu,
    };
    uint64_t words[MGL_PIPELINE_CACHE_KEY_WORDS] = {};
    mglPipelineCacheBuildLookupKeyWords(&inputs, words);
    EXPECT_EQ(mglPipelineCacheBuildPrimaryKey(&inputs), words[0]);
    EXPECT_EQ(11u, words[1]);
    EXPECT_EQ(12u, words[2]);
    EXPECT_EQ(13u, words[3]);
    EXPECT_EQ(14u, words[4]);
    EXPECT_EQ(0xAAAu, words[5]);
    EXPECT_EQ(0xBBBu, words[6]);
}

TEST(PipelineRecovery, ProgramMismatchBreakerRequiresExistingPipeline)
{
    MGLPipelineRecoveryState recovery = {
        .program_mismatch_retry_after = 100.0,
        .program_mismatch_program_name = 9u,
    };
    EXPECT_FALSE(mglPipelineRecoveryShouldAbortForProgramMismatch(
        &recovery, 50.0, 9u, nullptr));
    EXPECT_TRUE(mglPipelineRecoveryShouldAbortForProgramMismatch(
        &recovery, 50.0, 9u, reinterpret_cast<const void *>(0x1)));
}

TEST(PipelineRecovery, PipelineRetrySkipsBuildWhenPipelineExists)
{
    MGLPipelineRecoveryState recovery = {
        .pipeline_retry_after = 100.0,
        .interface_mismatch_program_name = 3u,
    };
    bool skip_build = false;
    EXPECT_TRUE(mglPipelineRecoveryEvaluatePipelineRetry(
        &recovery, 50.0, 3u, reinterpret_cast<const void *>(0x1), &skip_build));
    EXPECT_TRUE(skip_build);
}

TEST(PipelineRecovery, InterfaceMismatchAbortMatchesSignature)
{
    MGLPipelineRecoveryState recovery = {
        .interface_mismatch_retry_after = 100.0,
        .interface_mismatch_program_name = 5u,
        .interface_mismatch_color0_format = 10u,
        .interface_mismatch_depth_format = 20u,
        .interface_mismatch_stencil_format = 30u,
    };
    EXPECT_TRUE(mglPipelineRecoveryShouldAbortForInterfaceMismatch(
        &recovery, 50.0, 5u, 10u, 20u, 30u));
    EXPECT_FALSE(mglPipelineRecoveryShouldAbortForInterfaceMismatch(
        &recovery, 50.0, 5u, 11u, 20u, 30u));
}

TEST(PipelineRecovery, ReusePreviousRequiresMatchingFunctionsAndFormats)
{
    const void *prev = reinterpret_cast<const void *>(0x1000);
    const void *vs = reinterpret_cast<const void *>(0x2000);
    const void *fs = reinterpret_cast<const void *>(0x3000);
    MGLPipelineRecoveryReuseInput ok = {
        .previous_pipeline_state = prev,
        .current_program_name = 4u,
        .cached_program_name = 4u,
        .cached_vertex_function = vs,
        .cached_fragment_function = fs,
        .vertex_function = vs,
        .fragment_function = fs,
        .cached_color0_format = 10u,
        .built_color0_format = 10u,
        .cached_depth_format = 20u,
        .built_depth_format = 20u,
        .cached_stencil_format = 30u,
        .built_stencil_format = 30u,
        .invalid_pixel_format = 0xFFFFFFFFu,
    };
    EXPECT_TRUE(mglPipelineRecoveryCanReusePreviousOnInterfaceMismatch(&ok));

    MGLPipelineRecoveryReuseInput bad_vs = ok;
    bad_vs.vertex_function = reinterpret_cast<const void *>(0x9999);
    EXPECT_FALSE(mglPipelineRecoveryCanReusePreviousOnInterfaceMismatch(&bad_vs));
}

TEST(PipelineRecovery, InterfaceMismatchFailureBackoffEscalates)
{
    MGLPipelineRecoveryState recovery = {
        .interface_mismatch_program_name = 8u,
        .interface_mismatch_color0_format = 1u,
        .interface_mismatch_depth_format = 2u,
        .interface_mismatch_stencil_format = 3u,
        .interface_mismatch_streak = 3u,
    };
    MGLPipelineRecoveryMismatchDelays delays = {};
    mglPipelineRecoveryRecordInterfaceMismatchFailure(
        &recovery, 1000.0, 8u, 1u, 2u, 3u, &delays);
    EXPECT_EQ(4u, recovery.interface_mismatch_streak);
    EXPECT_DOUBLE_EQ(0.40, delays.interface_retry_delay);
    EXPECT_GT(recovery.program_mismatch_retry_after, 1000.0);
    EXPECT_GT(recovery.interface_mismatch_blocked_until, 1000.0);
}
