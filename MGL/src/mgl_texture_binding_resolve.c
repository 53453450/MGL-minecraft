/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_binding_resolve.c — the sampled-resource texture resolution,
 * moved out of MGLRenderer+Texture.m (P0-1 / T3 sink).  Two NSLog diagnostics
 * became stderr writes; everything else is the original logic.
 */

#include "mgl_texture_binding_resolve.h"
#include "mgl_texture_compat.h"    /* mglTextureUnitForSampledResource */
#include "mgl_program_resource.h"  /* mglShaderStageName */
#include "mgl_render.h"
#include "glm_limits.h"            /* TEX_OBJ_RES_NAME */

#include <stdio.h>

Texture *mglTextureForSampledResource(GLMContext ctx, MGLShaderResource *sampledResource,
                                       GLuint metalBinding, int stage,
                                       uint32_t expectedType, GLuint textureUnit)
{
    if (!ctx || mglRenderMetalBindingPastUnits(metalBinding, TEXTURE_UNITS)) {
        return NULL;
    }

    if (mglRenderMetalBindingPastUnits(textureUnit, TEXTURE_UNITS)) {
        return NULL;
    }

    if (mglRenderExpectedTypeUnset(expectedType)) {
        return ctx->active_state->active_textures[textureUnit];
    }

    /* AIR lowers sampler1D / samplerBuffer to texture2d, so expectedType is
     * often MGLTextureType2D. Prefer the GL target slot that matches the
     * reflected image_dim before trusting a leftover _TEXTURE_2D binding. */
    if (sampledResource &&
        mglRenderPrefer1DSampler(sampledResource->image_dim,
                                 sampledResource->image_arrayed ? 1 : 0)) {
        Texture *tex1D =
            ctx->active_state->texture_units[textureUnit].textures[_TEXTURE_1D];
        if (tex1D && tex1D->name != TEX_OBJ_RES_NAME) {
            return tex1D;
        }
        Texture *activeTexture = ctx->active_state->active_textures[textureUnit];
        if (activeTexture &&
            mglRenderTextureTargetIs1D((uint32_t)activeTexture->target) &&
            activeTexture->name != TEX_OBJ_RES_NAME) {
            return activeTexture;
        }
    }
    if (sampledResource &&
        mglRenderImageDimIsBuffer(sampledResource->image_dim)) {
        Texture *bufferTexture =
            ctx->active_state->texture_units[textureUnit].textures[_TEXTURE_BUFFER];
        if (bufferTexture &&
            !mglRenderTextureNameIsDefault(bufferTexture->name)) {
            return bufferTexture;
        }
        Texture *activeTexture = ctx->active_state->active_textures[textureUnit];
        if (activeTexture &&
            mglRenderIsTextureBufferTarget(activeTexture->target) &&
            !mglRenderTextureNameIsDefault(activeTexture->name)) {
            return activeTexture;
        }
    }
    /* AIR lowers sampler2DMS* to texture2d_array. Prefer the MS typed
     * binding even if BindTextureUnit later pointed the unit's "active"
     * texture at a non-MS object (CTS StorageMultisampleTest does
     * bindTexture(MS) then bindTextureUnit(unit, nonMS)). */
    if (sampledResource && sampledResource->image_multisampled) {
        GLuint msIndex =
            (GLuint)mglRenderMSTextureUnitIndex(sampledResource->image_arrayed
                                                    ? 1
                                                    : 0);
        Texture *msTexture =
            ctx->active_state->texture_units[textureUnit].textures[msIndex];
        if (msTexture && !mglRenderTextureNameIsDefault(msTexture->name)) {
            return msTexture;
        }
        Texture *activeTexture = ctx->active_state->active_textures[textureUnit];
        if (activeTexture &&
            mglRenderIsMultisampleTextureTarget(activeTexture->target) &&
            !mglRenderTextureNameIsDefault(activeTexture->name)) {
            return activeTexture;
        }
    }

    int textureIndex = (int)mglRenderTextureIndexForMetalType(expectedType);
    if (textureIndex >= 0 && textureIndex < _MAX_TEXTURE_TYPES) {
        Texture *typedTexture = ctx->active_state->texture_units[textureUnit].textures[textureIndex];
        /* The AIR backend lowers sampler1D to texture2d, so expectedType is
         * MGLTextureType2D even for GL_TEXTURE_1D bindings. If the _TEXTURE_2D
         * slot only contains an auto-created default texture (name ==
         * TEX_OBJ_RES_NAME) while the unit's active texture is a real
         * GL_TEXTURE_1D, prefer the 1D texture. Otherwise the default 2D
         * texture leaks across test cases and masks the real 1D binding. */
        Texture *activeTyped = ctx->active_state->active_textures[textureUnit];
        if (typedTexture &&
            mglRenderRejectDefaultTypedTexture(
                mglRenderTextureNameIsDefault(typedTexture->name),
                activeTyped &&
                        !mglRenderTextureNameIsDefault(activeTyped->name)
                    ? 1
                    : 0)) {
            typedTexture = NULL;
        }
        if (typedTexture) {
            return typedTexture;
        }

        if (mglRenderPrefer1DOverDefault2D(
                expectedType, activeTyped ? activeTyped->target : 0u)) {
            return activeTyped;
        }
        if (expectedType == MGLTextureType2D) {
            Texture *activeTexture = activeTyped;
            /* AIR packs samplerBuffer as texture2d; the GL binding lives in
             * the TEXTURE_BUFFER slot. Prefer that over a missing 2D binding
             * when the reflected resource is a buffer sampler/image. */
            if (sampledResource &&
                mglRenderImageDimIsBuffer(sampledResource->image_dim)) {
                Texture *bufferTexture =
                    ctx->active_state->texture_units[textureUnit].textures[_TEXTURE_BUFFER];
                if (bufferTexture &&
                    !mglRenderTextureNameIsDefault(bufferTexture->name)) {
                    return bufferTexture;
                }
                if (activeTexture &&
                    mglRenderIsTextureBufferTarget(activeTexture->target) &&
                    !mglRenderTextureNameIsDefault(activeTexture->name)) {
                    return activeTexture;
                }
            }
        }

        /* AIR lowers sampler1DArray / sampler2DMS* to texture2d_array, while GL
         * binds into _TEXTURE_1D_ARRAY / _TEXTURE_2D_MULTISAMPLE[_ARRAY]. */
        if (mglRenderPreferMSOr1DArrayOver2DArray(
                expectedType, activeTyped ? activeTyped->target : 0u)) {
            return activeTyped;
        }

        // Texel-buffer resources must not silently fall back to GL_TEXTURE_2D.
        // Minecraft's CloudFaces is declared with buffer image_dim, and the lowerer
        // lowers it to a 1-row texture2d<int> in MSL. If no GL_TEXTURE_BUFFER
        // is bound, using the active 2D atlas here feeds float/RGBA data into a
        // signed integer vertex resource and corrupts the whole frame.
        if (mglRenderExpectedTypeIsTextureBuffer(expectedType)) {
            static uint64_t s_missingTextureBufferBindingLogs = 0;
            uint64_t hit = ++s_missingTextureBufferBindingLogs;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                Texture *activeTexture = ctx->active_state->active_textures[textureUnit];
                fprintf(stderr, "MGL TEXBUFFER BIND MISSING binding=%u unit=%u activeTex=%u activeTarget=0x%x hit=%llu" "\n", (unsigned)metalBinding,
                      (unsigned)textureUnit,
                      activeTexture ? (unsigned)activeTexture->name : 0u,
                      activeTexture ? (unsigned)activeTexture->target : 0u,
                      (unsigned long long)hit);
            }
            return NULL;
        }

        /*
         * OpenGL texture units keep one binding per texture target. A sampler2D
         * samples the GL_TEXTURE_2D slot for its unit, even if a cubemap or texel
         * buffer was bound more recently on that same unit. Falling back to the
         * unit's "active" texture here lets sky cubemaps and buffer textures bleed
         * into item/entity shaders when Minecraft switches pipelines.
         */
        static uint64_t s_missingTypedTextureBindingLogs = 0;
        uint64_t hit = ++s_missingTypedTextureBindingLogs;
        if (hit <= 64ull || (hit % 512ull) == 0ull) {
            Texture *activeTexture = ctx->active_state->active_textures[textureUnit];
            fprintf(stderr, "MGL TEX TYPED BIND MISSING binding=%u stage=%s unit=%u expectedType=%lu expectedIndex=%d activeTex=%u activeTarget=0x%x hit=%llu" "\n", (unsigned)metalBinding,
                  mglShaderStageName(stage),
                  (unsigned)textureUnit,
                  (unsigned long)expectedType,
                  textureIndex,
                  activeTexture ? (unsigned)activeTexture->name : 0u,
                  activeTexture ? (unsigned)activeTexture->target : 0u,
                  (unsigned long long)hit);
        }
        return NULL;
    }

    return ctx->active_state->active_textures[textureUnit];
}

Texture *mglTextureForSampledResourceForStage(GLMContext ctx, MGLShaderResource *sampledResource,
                                             GLuint metalBinding, int stage,
                                             uint32_t expectedType)
{
    if (!ctx || mglRenderMetalBindingPastUnits(metalBinding, TEXTURE_UNITS)) {
        return NULL;
    }
    GLuint textureUnit = mglTextureUnitForSampledResource(sampledResource, mglResolveProgramForStageFromState(ctx, stage), metalBinding, stage);
    return mglTextureForSampledResource(ctx, sampledResource, metalBinding, stage, expectedType, textureUnit);
}
