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
 * mgl_coordinate.m
 * MGL
 *
 * Implementation of the Coordinate Compatibility Subsystem.
 *
 * See mgl_coordinate.h for the architectural rationale.  This module owns:
 *   - The Y-Flip Authority decision matrix (render_yflip × sample_yflip).
 *   - Helper queries over the authority field packed into
 *     Texture::mtl_render_yflip_authority.
 *
 * It does NOT own:
 *   - Setting the authority (done in MGLRenderer.m at RT write time, so that
 *     the authority is updated synchronously with mtl_render_target_write_version
 *     in the same code path).
 *   - Generating or releasing Y-flipped copies (owned by RT Sync).
 *   - VS/FS Y-flip injection (owned by the Shader Translation Layer in
 *     program.c, which sets Program::modules[_VERTEX_SHADER].mgl_injected_framebuffer_yflip).
 *
 * Keeping the decision logic here lets the VS/FS sampler-binding paths in
 * MGLRenderer.m call a single unified query instead of duplicating the
 * 4-quadrant matrix at each binding site.
 */

#import "mgl_coordinate.h"

bool mglProgramHasExistingFramebufferSampleYFlip(Program *program)
{
    if (!program) {
        return false;
    }

    /* Use the flag set during MSL post-processing rather than fragile string
     * matching.  The flag is set in program.c when MGL injects the texCoord
     * Y-flip for fullscreen sampled-framebuffer shaders.  This avoids false
     * negatives when the shader uses a different Y-flip expression that
     * wasn't in the hardcoded string list. */
    return program->modules[_VERTEX_SHADER].mgl_injected_framebuffer_yflip == GL_TRUE;
}

MGLYFlipDecision mglDecideYFlipForSampledRT(Texture *tex, Program *samplingProgram)
{
    if (!tex || !tex->is_render_target) {
        return MGL_YFLIP_USE_ORIGINAL;
    }

    GLuint rt_ver = tex->mtl_render_target_write_version;
    GLuint auth_packed = tex->mtl_render_yflip_authority;
    GLuint auth_ver = auth_packed >> 1;
    bool render_uses_original = (auth_packed & 1u) != 0u;

    /* Authority version mismatch: defensively downgrade to "not injected",
     * which will use the Y-flipped copy (the safe pre-fix behavior). */
    if (auth_ver != rt_ver) {
        render_uses_original = false;
    }

    bool sample_yflip_injected = samplingProgram &&
        samplingProgram->modules[_VERTEX_SHADER].mgl_injected_framebuffer_yflip == GL_TRUE;

    if (render_uses_original && !sample_yflip_injected) {
        return MGL_YFLIP_USE_ORIGINAL;
    }
    if (!render_uses_original && sample_yflip_injected) {
        return MGL_YFLIP_USE_ORIGINAL_AND_INJECT;
    }
    if (render_uses_original && sample_yflip_injected) {
        return MGL_YFLIP_USE_ORIGINAL_AND_INJECT;
    }
    return MGL_YFLIP_USE_SAMPLED_COPY;
}

bool mglRTWriteAuthorityIsCurrentAndUsesOriginal(Texture *tex)
{
    if (!tex || !tex->is_render_target) {
        return false;
    }

    GLuint auth_packed = tex->mtl_render_yflip_authority;
    GLuint auth_ver = auth_packed >> 1;
    bool render_uses_original = (auth_packed & 1u) != 0u;

    return (auth_ver == tex->mtl_render_target_write_version) && render_uses_original;
}
