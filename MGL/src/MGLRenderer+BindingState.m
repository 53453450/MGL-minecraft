/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+BindingState.m — V/F buffer, attrib, texture bind ports

#import "MGLRenderer_Private.h"
#include "mgl_blit_drivers.h"   /* sampled RT copy repair (log 150) */
#include "mgl_sampled_sampler.h"   /* sampler materialize (log 151) */
#include "mgl_stage_buffer_bind.h"  /* stage-buffer binding drivers (log 129) */
#include "mgl_storage_image_bind.h" /* storage-image driver (log 130) */
#include "mgl_sampled_fallback.h" /* sampled-texture fallback chain (log 132) */
#include "mgl_texture_sampler.h"
#include "mgl_renderer_ports.h"
#include "mgl_buffer_map.h"  /* buffer mapping entries (was MGLRenderer+Buffer.m) */
#include "mgl_binding_state_ops.h"
#include "mgl_texture_binding_resolve.h"
#import "MGLRenderer+Draw_Private.h"
#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "pixel_utils.h"

enum {
    MGL_BINDING_RESOURCE_STORAGE_SHARED = 0u,
    MGL_BINDING_VERTEX_FORMAT_INVALID = 0u,
    MGL_BINDING_PIXEL_FORMAT_INVALID = 0u,
    /* Match MGLTextureType / MTLTextureType values from mgl_render_values.h.
     * (A prior local enum used 3D=4, which is actually 2DMultisample.) */
    MGL_BINDING_TEXTURE_TYPE_1D = MGLTextureType1D,
    MGL_BINDING_TEXTURE_TYPE_1D_ARRAY = MGLTextureType1DArray,
    MGL_BINDING_TEXTURE_TYPE_2D = MGLTextureType2D,
    MGL_BINDING_TEXTURE_TYPE_2D_ARRAY = MGLTextureType2DArray,
    MGL_BINDING_TEXTURE_TYPE_3D = MGLTextureType3D,
    MGL_BINDING_TEXTURE_TYPE_CUBE = MGLTextureTypeCube,
    MGL_BINDING_TEXTURE_TYPE_CUBE_ARRAY = MGLTextureTypeCubeArray,
};


static MGLRenderTextureInfo mglBindingStateTextureInfo(id texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo((__bridge void *)texture, &info);
    }
    return info;
}

static uint32_t mglBindingStateTexturePixelFormat(id texture)
{
    return mglBindingStateTextureInfo(texture).pixel_format;
}

static uint32_t mglBindingStateTextureType(id texture)
{
    return mglBindingStateTextureInfo(texture).texture_type;
}

static uint64_t mglBindingStateTextureWidth(id texture)
{
    return mglBindingStateTextureInfo(texture).width;
}

static uint64_t mglBindingStateTextureHeight(id texture)
{
    return mglBindingStateTextureInfo(texture).height;
}

static uint64_t mglBindingStateTextureArrayLength(id texture)
{
    return mglBindingStateTextureInfo(texture).array_length;
}

static uint64_t mglBindingStateTextureMipmapLevelCount(id texture)
{
    return mglBindingStateTextureInfo(texture).mipmap_level_count;
}

static BOOL mglBindingStateRenderPassUsesColorTexture(
    void *owner,
    void *texture,
    NSUInteger *attachmentIndexOut)
{
    uint32_t attachmentIndex = MAX_COLOR_ATTACHMENTS;
    const BOOL found = mglRenderPassUsesColorTextureOwner(
        owner, texture, &attachmentIndex);
    if (attachmentIndexOut) {
        *attachmentIndexOut = attachmentIndex;
    }
    return found;
}

static BOOL mglBindingStateHasActiveEncoder(const MGLEncodeContext *encCtx)
{
    if (!encCtx) {
        return NO;
    }
    return mglRenderEncoderOwnerHasCurrent(
        encCtx->render_encoder_owner) != 0;
}


static id mglBindingStateCacheImageUnitView(ImageUnit *iu, id fallback, void *view)
{
    if (!view) {
        return fallback;
    }
    if (iu->mtl_image_view) {
        mglRenderReleaseMetalObject(iu->mtl_image_view);
        iu->mtl_image_view = NULL;
    }
    iu->mtl_image_view = view; /* +1 from newTextureView */
    return (__bridge id)iu->mtl_image_view;
}

/* BindImageTexture <format> → PixelFormatView (CTS advanced-cast). */
static uint32_t mglBindingStateImageBindPixelFormat(const ImageUnit *iu,
                                                    uint32_t native_format)
{
    if (!iu) {
        return native_format;
    }
    const uint32_t bind_format =
        mtlFormatForGLInternalFormat(iu->internalformat);
    return mglRenderImageBindPixelFormat(iu->internalformat, native_format,
                                         bind_format);
}

/* Storage-image view: non-layered slice + format/mip PixelFormatView (CTS). */
void *mglRendererStorageImageTexture(void *base_texture, ImageUnit *iu)
{
    id texture = (__bridge id)base_texture;
    if (!texture || !iu) {
        return base_texture;
    }
    if (iu->mtl_image_view) {
        return iu->mtl_image_view;
    }
    const MGLRenderTextureInfo info = mglBindingStateTextureInfo(texture);
    if (info.width == 0u) {
        return base_texture;
    }
    const NSUInteger level = (NSUInteger)iu->level;
    /* CTS incomplete_textures: mip past mipmapLevelCount → unbound. */
    if (!mglRenderImageLevelInRange((uint32_t)level,
                                    (uint32_t)info.mipmap_level_count)) {
        return NULL;
    }
    const uint32_t srcType = info.texture_type;
    const uint32_t bindFormat =
        mglBindingStateImageBindPixelFormat(iu, info.pixel_format);
    const GLenum glTarget = iu->tex ? iu->tex->target : (GLenum)0;
    const int isMsTarget = mglRenderImageTargetIsMultisample((uint32_t)glTarget);
    uint32_t dstType = 0u;
    if (mglRenderImageNeedsNonLayeredSlice(iu->layered ? 1 : 0, isMsTarget,
                                           srcType, &dstType)) {
            void *view = NULL;
            int rc = mglRenderCreateTextureViewRange(
                    base_texture, bindFormat, dstType,
                    level, 1u, (uint64_t)iu->layer, 1u,
                    0, 0, 0, 0, 0, &view);
            if (rc == 0 && view) {
                return (__bridge void *)mglBindingStateCacheImageUnitView(
                    iu, texture, view);
            }
    }

    if (mglRenderImageNeedsFormatOrMipView(level, bindFormat,
                                           info.pixel_format)) {
        NSUInteger sliceCount = (NSUInteger)mglRenderImageViewSliceCount(
            srcType, mglBindingStateTextureArrayLength(texture));
        void *view = NULL;
        if (mglRenderCreateTextureViewRange(
                base_texture, bindFormat,
                info.texture_type, level, 1u, 0u, sliceCount,
                0, 0, 0, 0, 0, &view) == 0 && view) {
            return (__bridge void *)mglBindingStateCacheImageUnitView(
                iu, texture, view);
        }
    }
    return base_texture;
}


/* O3.3: resolve shader-resource list ordinal → resource (+ optional element). */
static MGLShaderResource *mglBindingStateResourceAtOrdinal(
    Program *program, int stage, int resType, GLuint ordinal, GLuint *elementOut)
{
    if (elementOut) {
        *elementOut = 0u;
    }
    if (!program || stage < 0 || resType < 0) {
        return NULL;
    }
    MGLShaderResourceList *list =
        &program->shader_resources_list[stage][resType];
    GLuint rem = ordinal;
    for (GLuint ri = 0; ri < list->count; ri++) {
        GLuint elements = mglRenderShaderResourceElementCount(
            (uint32_t)list->list[ri].gl_array_size);
        if (rem < elements) {
            if (elementOut) {
                *elementOut = rem;
            }
            return &list->list[ri];
        }
        rem -= elements;
    }
    return NULL;
}


@implementation MGLRenderer (Draw)






static const NSUInteger kMaxFragmentSamplerSlots = 16;

#define MGL_ABORT_TBIND_IF_ENCODER_CLOSED() do { \
    if (mglRenderEncoderOwnerHasCurrent(_renderPassManager->state->currentRenderEncoderOwner) == 0) { \
        if (ctx) { \
            mglMarkRendererDirtyBits(ctx->active_state, (DIRTY_TEX | DIRTY_TEX_BINDING | DIRTY_RENDER_STATE)); \
        } \
        return false; \
    } \
} while (0)

- (bool) bindTexturesToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx
{
    static uint64_t s_bindTexturesCallCount = 0;
    uint64_t bindCall = ++s_bindTexturesCallCount;
    bool traceBind = mglShouldTraceCall(bindCall);
    GLuint vertexSampledCount = 0;
    GLuint vertexBoundTextures = 0;
    GLuint vertexFallbackTextures = 0;
    GLuint boundSampledTextures = 0;
    GLuint nilSampledTextures = 0;
    GLuint fallbackSampledTextures = 0;
    GLuint boundSampledSamplers = 0;
    Program *vertexProgram = NULL;
    Program *fragmentProgram = NULL;
    GLuint vertexProgramName = 0u;
    GLuint fragmentProgramName = 0u;
    const BOOL useResourceSnapshot = YES;
    MGLRenderResourceBindingSnapshot resourceSnapshot = {0};

    if (!mglBindingStateHasActiveEncoder(encCtx)) {
        // No active render encoder yet (or it was rotated). Texture/sampler binding
        // can be deferred until the next encoder is created.
        return true;
    }

    /* Per-draw sampler snapshot for replay / RT-copy cull; clear stale slots. */
    if (mglTraceLogIsEnabled()) {
        mglTraceFragmentTextureTraceBindings("CLEAR",
                                             "bind_textures_begin",
                                             _resourceFallback.fragmentTextureTraceBindings,
                                             TEXTURE_UNITS,
                                             ctx ? mglCurrentRenderProgramKey(ctx) : 0u,
                                             _pipelineCache.state->pipelineProgramName);

        memset(_resourceFallback.fragmentTextureTraceBindings, 0,
               sizeof(_resourceFallback.fragmentTextureTraceBindings));
    } else {
        mglClearFragmentTextureTraceFunctionalFlags(
            _resourceFallback.fragmentTextureTraceBindings, TEXTURE_UNITS);
    }

    const int vertexResourceStage = _tessellation.nativeTESActive
        ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    vertexProgram = _tessellation.nativeTESActive
        ? _tessellation.nativeTESProgram
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    fragmentProgram = mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    vertexProgramName = vertexProgram ? vertexProgram->name : mglCurrentRenderProgramKey(ctx);
    fragmentProgramName = fragmentProgram ? fragmentProgram->name : mglCurrentRenderProgramKey(ctx);

    id defaultSampler = (__bridge id)mglTextureFallbackSamplerState((__bridge void *)self);
    if (defaultSampler) {
        if (vertexProgram) {
            (void)mglProgramSamplesTextureUnit(vertexProgram, 0);
        }
        if (fragmentProgram && fragmentProgram != vertexProgram) {
            (void)mglProgramSamplesTextureUnit(fragmentProgram, 0);
        }
        MGLSamplerWarmupPlan warm = {0};
        mglBindingTexturePlanSamplerWarmup(
            1, vertexProgram ? 1 : 0, fragmentProgram ? 1 : 0,
            vertexProgram ? vertexProgram->sampled_texture_unit_mask : NULL,
            (fragmentProgram && fragmentProgram != vertexProgram)
                ? fragmentProgram->sampled_texture_unit_mask
                : NULL,
            (uint32_t)TEXTURE_UNITS, (uint32_t)kMaxFragmentSamplerSlots, &warm);
        for (uint32_t s = 0; s < warm.warmup_count; s++) {
            if (warm.mode == MGL_SW_MODE_MASK &&
                !mglBindingTextureSamplerWarmupSlotActive(warm.mask, s)) {
                continue;
            }
            if (!mglBindingStateQueueResourceBinding(
                    useResourceSnapshot, _bindingStateOwner,
                    _renderPassManager->state->currentRenderEncoderOwner,
                    &resourceSnapshot, MGL_RENDER_BINDING_STAGE_VERTEX,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                    (__bridge void *)defaultSampler, s) ||
                !mglBindingStateQueueResourceBinding(
                    useResourceSnapshot, _bindingStateOwner,
                    _renderPassManager->state->currentRenderEncoderOwner,
                    &resourceSnapshot, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                    (__bridge void *)defaultSampler, s)) {
                return false;
            }
        }
    }

    if (useResourceSnapshot &&
        !mglBindingStateFlushResourceBindings(
            _bindingStateOwner,
            _renderPassManager->state->currentRenderEncoderOwner,
            &resourceSnapshot)) {
        return false;
    }

    GLuint sampledCount = 0;
    GLuint separateSamplerCount = 0;
    GLuint boundSeparateSamplers = 0;

    /* Bind VS+FS sampled images (Metal validates every active stage). */
    if (!mglSampledBindTexturesForStage(
            (__bridge void *)self, vertexResourceStage, 0, vertexProgram,
            vertexProgramName, vertexProgramName, 0u,
            (__bridge void *)defaultSampler, bindCall, traceBind ? 1 : 0,
            &vertexBoundTextures, &vertexFallbackTextures, NULL, NULL,
            &vertexSampledCount)) {
        return false;
    }
    if (!mglSampledBindTexturesForStage(
            (__bridge void *)self, _FRAGMENT_SHADER, 1, fragmentProgram,
            fragmentProgramName, vertexProgramName, fragmentProgramName,
            (__bridge void *)defaultSampler, bindCall, traceBind ? 1 : 0,
            &boundSampledTextures, &fallbackSampledTextures,
            &nilSampledTextures, &boundSampledSamplers, &sampledCount)) {
        return false;
    }

    /* The storage-image driver is C now (log 130). */
    if (!mglBindingStateBindStorageImagesForVertexProgram(
            (__bridge void *)self, vertexProgram, fragmentProgram)) {
        return false;
    }

    if (!mglSampledBindSeparateSamplersAndArrayTextures(
            (__bridge void *)self, vertexProgram, fragmentProgram,
            fragmentProgramName, vertexProgramName,
            (__bridge void *)defaultSampler, bindCall, traceBind ? 1 : 0,
            &separateSamplerCount, &boundSeparateSamplers)) {
        return false;
    }

    BOOL interesting = (sampledCount > 0 && boundSampledTextures == 0) ||
                       fallbackSampledTextures > 0 || vertexFallbackTextures > 0;
    static uint64_t s_interestingTextureSummaryCount = 0;
    if (traceBind ||
        (interesting && mglBindingTextureRateLogHit(
                            &s_interestingTextureSummaryCount, 64ull, 512ull))) {
        mglTraceLog(
            "texbind.summary call=%llu program=%u vertexSampled=%u "
            "vertexBoundTex=%u vertexFallback=%u sampled=%u boundTex=%u "
            "nilTex=%u fallbackTex=%u sampledSamplers=%u separateSamplers=%u "
            "boundSeparate=%u",
            (unsigned long long)bindCall,
            (unsigned)mglCurrentRenderProgramKey(ctx),
            (unsigned)vertexSampledCount, (unsigned)vertexBoundTextures,
            (unsigned)vertexFallbackTextures, (unsigned)sampledCount,
            (unsigned)boundSampledTextures, (unsigned)nilSampledTextures,
            (unsigned)fallbackSampledTextures, (unsigned)boundSampledSamplers,
            (unsigned)separateSamplerCount, (unsigned)boundSeparateSamplers);
    }

    return true;
}










@end
