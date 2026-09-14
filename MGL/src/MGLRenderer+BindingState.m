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
#include "mgl_stage_buffer_bind.h"  /* stage-buffer binding drivers (log 129) */
#include "mgl_storage_image_bind.h" /* storage-image driver (log 130) */
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
    if (![self bindSampledTexturesForStage:vertexResourceStage
                           isFragmentStage:NO
                                   program:vertexProgram
                               programName:vertexProgramName
                          vertexProgramName:vertexProgramName
                        fragmentProgramName:0u
                            defaultSampler:defaultSampler
                                   bindCall:bindCall
                                 traceBind:traceBind
                                boundCount:&vertexBoundTextures
                             fallbackCount:&vertexFallbackTextures
                                  nilCount:NULL
                               samplerCount:NULL
                               sampledCount:&vertexSampledCount]) {
        return false;
    }
    if (![self bindSampledTexturesForStage:_FRAGMENT_SHADER
                           isFragmentStage:YES
                                   program:fragmentProgram
                               programName:fragmentProgramName
                          vertexProgramName:vertexProgramName
                        fragmentProgramName:fragmentProgramName
                            defaultSampler:defaultSampler
                                   bindCall:bindCall
                                 traceBind:traceBind
                                boundCount:&boundSampledTextures
                             fallbackCount:&fallbackSampledTextures
                                  nilCount:&nilSampledTextures
                               samplerCount:&boundSampledSamplers
                               sampledCount:&sampledCount]) {
        return false;
    }

    /* The storage-image driver is C now (log 130). */
    if (!mglBindingStateBindStorageImagesForVertexProgram(
            (__bridge void *)self, vertexProgram, fragmentProgram)) {
        return false;
    }

    if (![self bindSeparateSamplersAndArrayTextures:vertexProgram
                                      fragmentProgram:fragmentProgram
                                fragmentProgramName:fragmentProgramName
                                  vertexProgramName:vertexProgramName
                                     defaultSampler:defaultSampler
                                            bindCall:bindCall
                                          traceBind:traceBind
                                 separateSamplerCount:&separateSamplerCount
                                   boundSeparateSamplers:&boundSeparateSamplers]) {
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

- (bool)bindSampledTexturesForStage:(int)shaderStage
                    isFragmentStage:(BOOL)isFragment
                            program:(Program *)program
                        programName:(GLuint)programName
                   vertexProgramName:(GLuint)vertexProgramName
                 fragmentProgramName:(GLuint)fragmentProgramName
                     defaultSampler:(id)defaultSampler
                            bindCall:(uint64_t)bindCall
                          traceBind:(bool)traceBind
                         boundCount:(GLuint *)boundCount
                      fallbackCount:(GLuint *)fallbackCount
                           nilCount:(GLuint *)nilCount
                        samplerCount:(GLuint *)samplerCount
                        sampledCount:(GLuint *)sampledCountOut
{
    GLuint bound = boundCount ? *boundCount : 0u;
    GLuint fallback = fallbackCount ? *fallbackCount : 0u;
    GLuint nilTex = nilCount ? *nilCount : 0u;
    GLuint boundSamplers = samplerCount ? *samplerCount : 0u;
    const BOOL useResourceSnapshot = YES;
    MGLRenderResourceBindingSnapshot resourceSnapshot = {0};
    const uint32_t metalStage = isFragment ? MGL_RENDER_BINDING_STAGE_FRAGMENT
                                           : MGL_RENDER_BINDING_STAGE_VERTEX;
    const char *stageTag = isFragment ? "fragment" : "vertex";
    GLuint count = mglRendererGetProgramBindingCount(ctx, shaderStage,
                                                     _SAMPLED_IMAGE_RES);
    if (sampledCountOut) {
        *sampledCountOut = count;
    }
    static uint64_t s_focusedSampledLogs[2] = {0, 0};
    static uint64_t s_traceFileSampledLogs[2] = {0, 0};

    for (GLuint i = 0; i < count; i++) {
        Program *sampleProgram = program;
        MGLShaderResource *sampledResource = mglBindingStateResourceAtOrdinal(
            sampleProgram, shaderStage, _SAMPLED_IMAGE_RES, i, NULL);
        const char *sampledName = sampledResource ? sampledResource->name : "";
        GLuint spirvBinding = sampledResource
            ? (GLuint)mglRendererGetProgramBinding(ctx, shaderStage,
                                                   _SAMPLED_IMAGE_RES, (int32_t)i)
            : 0u;
        GLuint glBinding = sampledResource
            ? (GLuint)mglRendererGetProgramGLBinding(ctx, shaderStage,
                                                     _SAMPLED_IMAGE_RES, (int32_t)i)
            : 0u;
        MGLSampledTextureBindInput sin = {0};
        mglBindingTextureFillSampledGateInput(
            &sin, spirvBinding, glBinding, TEXTURE_UNITS,
            /* No skip recipe: the SPIRV-era resource-skip heuristics are
             * gone, so a sampler-like resource always reaches the plan. */
            0,
            sampledResource ? 1 : 0);
        MGLSampledTextureBindPlan splan = {0};
        if (mglBindingTexturePlanSampled(&sin, &splan) != 0 ||
            splan.action == MGL_ST_ACTION_SKIP) {
            continue;
        }
        GLuint textureUnit = mglTextureUnitForSampledResource(sampledResource, sampleProgram, spirvBinding, shaderStage);
        uint32_t expectedType = (uint32_t)mglExpectedTextureTypeForResource(
            sampleProgram, shaderStage, sampledResource);
        uint32_t lookupType =
            (uint32_t)mglDeclaredTextureTypeFromResource(sampledResource);
        MGLTextureDataKind expectedKind =
            (MGLTextureDataKind)mglExpectedTextureDataKindForResource(
                sampleProgram, shaderStage, sampledResource);
        Texture *ptr = mglTextureForSampledResource(
            ctx, sampledResource, spirvBinding, shaderStage,
            (lookupType ? lookupType : expectedType), textureUnit);
        id texture = nil;
        id sampler = isFragment ? nil : defaultSampler;
        id directTextureForTrace = nil;
        id sampledCopyForTrace = nil;
        BOOL usedFallback = NO;
        BOOL suppressMissing = NO;
        BOOL usedSampledCopy = NO;

        if (ptr) {
            if (isFragment) {
                if (![self recoverFragmentSampledDepthTexture:&ptr
                                                       texture:&texture
                                                   sampledName:sampledName
                                                  spirvBinding:spirvBinding
                                                    textureUnit:textureUnit
                                                   expectedType:expectedType
                                                   expectedKind:expectedKind
                                           fragmentProgramName:fragmentProgramName
                                suppressMissingTextureFallback:&suppressMissing
                                          usedFallbackTexture:&usedFallback]) {
                    return false;
                }
                if (![self applySampledRenderTargetCopyPlan:ptr
                                                    texture:&texture
                                              sampleProgram:sampleProgram
                                               expectedType:expectedType
                                               expectedKind:expectedKind
                                          usedTypeFallback:usedFallback
                                                    stage:stageTag
                                               programName:programName
                                               spirvBinding:spirvBinding
                                                 textureUnit:textureUnit
                                                 sampledName:sampledName
                                        usedSampledCopyOut:&usedSampledCopy
                                      directTextureForTrace:&directTextureForTrace
                                      sampledCopyForTrace:&sampledCopyForTrace]) {
                    return false;
                }
                texture = [self applySampledCompatFallbackPlan:ptr
                                                       texture:texture
                                                  expectedType:expectedType
                                                  expectedKind:expectedKind
                                                         stage:stageTag
                                                   programName:programName
                                                  spirvBinding:spirvBinding
                                                 sampleProgram:sampleProgram
                                              usedFallbackOut:&usedFallback];
                if (usedFallback) {
                    usedSampledCopy = NO;
                }
                sampler = [self materializeSampledSamplerForTexture:ptr
                                                        textureUnit:textureUnit
                                                    defaultSampler:sampler
                                                      forceDefault:NO
                                                     samplerTarget:ptr ? ptr->target
                                                                       : 0u
                                                       programName:programName
                                                      spirvBinding:spirvBinding
                                                             stage:stageTag
                                                           texture:texture];
                if (mglMipDiagEnabled() && textureUnit < TEXTURE_UNITS) {
                    Sampler *glSampler =
                        MGL_STATE(ctx)->texture_samplers[textureUnit];
                    const TextureParameter *effective =
                        glSampler ? &glSampler->params : &ptr->params;
                    static uint64_t s_fragSamplerState[TEXTURE_UNITS];
                    (void)mglBindingTextureEmitMipDiagFragIfChanged(
                        &s_fragSamplerState[textureUnit],
                        mglBindingTextureMipDiagSignature(
                            ptr->name, effective->min_filter,
                            effective->mag_filter, ptr->params.base_level,
                            ptr->params.max_level,
                            texture ? mglBindingStateTextureMipmapLevelCount(texture)
                                    : 0u,
                            (uint64_t)(uintptr_t)texture, usedSampledCopy ? 1 : 0,
                            ptr->mtl_gl_sampled_levels,
                            ptr->mtl_gl_sampled_dirty_mip_mask,
                            ptr->mtl_gl_sampled_write_version !=
                                ptr->mtl_render_target_write_version),
                        textureUnit, spirvBinding, fragmentProgramName, ptr->name,
                        glSampler ? "glSampler" : "texParams",
                        effective->min_filter, effective->mag_filter,
                        effective->min_lod, effective->max_lod,
                        effective->max_anisotropy, ptr->params.base_level,
                        ptr->params.max_level, ptr->num_levels,
                        texture ? mglBindingStateTextureMipmapLevelCount(texture)
                                : 0u,
                        texture ? mglBindingStateTextureWidth(texture) : 0u,
                        texture ? mglBindingStateTextureHeight(texture) : 0u,
                        (__bridge const void *)texture,
                        ptr->is_render_target ? 1 : 0, usedSampledCopy ? 1 : 0,
                        ptr->mtl_gl_sampled_levels,
                        ptr->mtl_gl_sampled_dirty_mip_mask,
                        ptr->mtl_render_target_write_version,
                        ptr->mtl_gl_sampled_write_version);
                }
            } else {
                RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
                MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
                if (ptr->mtl_data) {
                    texture = (__bridge id)(ptr->mtl_data);
                    texture = (__bridge id)mglSampledTextureViewForBaseLevel(
                        ptr, (__bridge void *)texture);
                }
                texture = [self applySampledCompatFallbackPlan:ptr
                                                       texture:texture
                                                  expectedType:expectedType
                                                  expectedKind:expectedKind
                                                         stage:stageTag
                                                   programName:programName
                                                  spirvBinding:spirvBinding
                                                 sampleProgram:sampleProgram
                                              usedFallbackOut:&usedFallback];
                sampler = [self materializeSampledSamplerForTexture:ptr
                                                        textureUnit:textureUnit
                                                    defaultSampler:defaultSampler
                                                      forceDefault:NO
                                                     samplerTarget:ptr ? ptr->target
                                                                       : 0u
                                                       programName:programName
                                                      spirvBinding:spirvBinding
                                                             stage:stageTag
                                                           texture:texture];
                if (![self applySampledRenderTargetCopyPlan:ptr
                                                    texture:&texture
                                              sampleProgram:sampleProgram
                                               expectedType:expectedType
                                               expectedKind:expectedKind
                                          usedTypeFallback:usedFallback
                                                    stage:stageTag
                                               programName:programName
                                               spirvBinding:spirvBinding
                                                 textureUnit:textureUnit
                                                 sampledName:sampledName
                                        usedSampledCopyOut:NULL
                                      directTextureForTrace:NULL
                                      sampledCopyForTrace:NULL]) {
                    return false;
                }
            }
        }

        GLuint samplerBinding =
            sampledResource && sampledResource->has_combined_sampler
                ? mglMetalCombinedSamplerSlot(sampledResource)
                : spirvBinding;
        mglBindingTextureFillSampledFinalInput(
            &sin, texture ? 1 : 0, suppressMissing ? 1 : 0, usedFallback ? 1 : 0,
            sampledResource && sampledResource->has_combined_sampler ? 1 : 0,
            samplerBinding, (uint32_t)kMaxFragmentSamplerSlots, sampler ? 1 : 0,
            isFragment ? mglBindingTextureForceDefaultSampler(
                             usedFallback ? 1 : 0,
                             expectedKind == MGLTextureDataKindDepth ? 1 : 0)
                       : 0);

        if (!texture && !suppressMissing) {
            texture = [self fallbackSampledTextureForExpectedType:expectedType
                                                         dataKind:expectedKind];
            if (texture) {
                sin.has_bound_texture = 1;
                fallback++;
                if (isFragment) {
                    usedFallback = YES;
                    usedSampledCopy = NO;
                    mglFocusLoadingProgram(fragmentProgramName, "sample-fallback",
                                           bindCall);
                    static uint64_t s_fragFbLog = 0;
                    if (mglBindingTextureRateLogHit(&s_fragFbLog, 32ull, 512ull)) {
                        mglBindingLogTexFallbackEx(
                            s_fragFbLog, spirvBinding, fragmentProgramName,
                            ptr ? ptr->name : 0u, 0, NULL, 0u);
                    }
                }
            }
        } else if (isFragment && !texture && suppressMissing) {
            static uint64_t s_fragFbSup = 0;
            if (mglBindingTextureRateLogHit(&s_fragFbSup, 64ull, 512ull)) {
                mglBindingLogTexFallbackEx(
                    s_fragFbSup, spirvBinding, fragmentProgramName,
                    ptr ? ptr->name : 0u, 1, sampledName, textureUnit);
            }
        }

        if (isFragment) {
            MGLSamplerMaterializeInput fin = {0};
            mglBindingTextureFillSamplerMaterializeInput(
                &fin, sin.force_default_sampler, 0, 0, 0, 0, 0, 1);
            MGLSamplerMaterializePlan fplan = {0};
            (void)mglBindingTexturePlanSamplerMaterialize(&fin, &fplan);
            if (fplan.action == MGL_SM_ACTION_USE_DEFAULT || !sampler) {
                sampler = defaultSampler;
            }
            sin.has_sampler = sampler ? 1 : 0;
            sin.used_type_fallback = usedFallback ? 1 : 0;
            sin.force_default_sampler = mglBindingTextureForceDefaultSampler(
                usedFallback ? 1 : 0,
                expectedKind == MGLTextureDataKindDepth ? 1 : 0);
        }

        if (mglBindingTexturePlanSampled(&sin, &splan) != 0) {
            continue;
        }
        if (!isFragment && splan.action != MGL_ST_ACTION_QUEUE) {
            continue;
        }

        if ((isFragment || splan.queue_texture) &&
            !mglBindingStateQueueResourceBinding(
                useResourceSnapshot, _bindingStateOwner,
                _renderPassManager->state->currentRenderEncoderOwner,
                &resourceSnapshot, metalStage,
                MGL_RENDER_RESOURCE_BINDING_TEXTURE, (__bridge void *)texture,
                isFragment ? spirvBinding : splan.texture_slot)) {
            return false;
        }
        if (!isFragment && splan.queue_sampler &&
            !mglBindingStateQueueResourceBinding(
                useResourceSnapshot, _bindingStateOwner,
                _renderPassManager->state->currentRenderEncoderOwner,
                &resourceSnapshot, metalStage,
                MGL_RENDER_RESOURCE_BINDING_SAMPLER, (__bridge void *)sampler,
                splan.sampler_slot)) {
            return false;
        }

        GLuint sampleProgramName =
            sampleProgram ? sampleProgram->name : programName;
        if (isFragment && spirvBinding < TEXTURE_UNITS) {
            mglBindingTextureWriteFragTrace(
                &_resourceFallback.fragmentTextureTraceBindings[spirvBinding],
                ptr ? ptr->name : 0u, textureUnit, spirvBinding, sampleProgramName,
                ptr ? ptr->mtl_render_target_write_version : 0u,
                ptr ? ptr->mtl_gl_sampled_write_version : 0u, ptr,
                (__bridge void *)texture,
                (__bridge void *)(directTextureForTrace ? directTextureForTrace
                                                       : texture),
                (__bridge void *)sampledCopyForTrace,
                texture ? mglBindingStateTextureWidth(texture) : 0u,
                texture ? mglBindingStateTextureHeight(texture) : 0u,
                texture ? mglBindingStateTexturePixelFormat(texture)
                        : MGL_BINDING_PIXEL_FORMAT_INVALID,
                texture ? mglBindingStateTextureType(texture) : 0u,
                usedSampledCopy ? 1 : 0, usedFallback ? 1 : 0);
        }

        [self emitSampledDiagPortsForProgram:sampleProgram
                                       stage:stageTag
                             stageIsFragment:isFragment
                                 sampledName:sampledName
                                spirvBinding:spirvBinding
                                 textureUnit:textureUnit
                            sampledResource:sampledResource
                                         ptr:ptr
                                     texture:texture
                                     sampler:sampler
                               usedFallback:usedFallback
                              expectedType:expectedType
                                lookupType:lookupType
                                   bindCall:bindCall
                                programName:sampleProgramName
                           vertexProgramName:vertexProgramName
                         fragmentProgramName:isFragment ? fragmentProgramName
                                                        : 0u
                        usedSampledCopyTrace:usedSampledCopy
                       directTextureForTrace:directTextureForTrace
                       sampledCopyForTrace:sampledCopyForTrace
                           focusedCounter:&s_focusedSampledLogs[isFragment ? 1
                                                                          : 0]
                         traceFileCounter:&s_traceFileSampledLogs[isFragment ? 1
                                                                            : 0]];

        if (isFragment) {
            switch (mglBindingTextureSampledMarkKind(texture ? 1 : 0,
                                                     usedFallback ? 1 : 0)) {
            case MGL_ST_MARK_BOUND:
                bound++;
                break;
            case MGL_ST_MARK_FALLBACK:
                /* Keep nilTex as original GL failure count; Metal gets fallback. */
                nilTex++;
                break;
            default:
                nilTex++;
                break;
            }
            if (sampler &&
                (!sampledResource || sampledResource->has_combined_sampler) &&
                samplerBinding < kMaxFragmentSamplerSlots) {
                if (!mglBindingStateQueueResourceBinding(
                        useResourceSnapshot, _bindingStateOwner,
                        _renderPassManager->state->currentRenderEncoderOwner,
                        &resourceSnapshot, metalStage,
                        MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                        (__bridge void *)sampler, samplerBinding)) {
                    return false;
                }
                boundSamplers++;
            }
            if (traceBind && i < 6) {
                TextureLevel *level0 =
                    (ptr && ptr->faces[0].levels) ? &ptr->faces[0].levels[0]
                                                  : NULL;
                uint32_t cpuFirst = 0u;
                int cpuOk = level0 && level0->data && level0->data_size >= 4 &&
                            (uintptr_t)level0->data >= 0x1000ull;
                if (cpuOk) {
                    memcpy(&cpuFirst, (const void *)level0->data, 4);
                }
                mglTraceLog(
                    "texbind.sampled call=%llu idx=%u binding=%u glTex=%u "
                    "target=0x%x internal=0x%x l0=%ux%ux%u l0bytes=%lu "
                    "l0first=0x%08x(valid=%d) source=%u upload=%lu src=%p "
                    "hash=0x%016llx ever=%u full=%u zero=%u mtl=%p "
                    "size=%lux%lu sampler=%p fallback=%d",
                    (unsigned long long)bindCall, (unsigned)i,
                    (unsigned)spirvBinding, ptr ? (unsigned)ptr->name : 0u,
                    ptr ? (unsigned)ptr->target : 0u,
                    ptr ? (unsigned)ptr->internalformat : 0u,
                    level0 ? (unsigned)level0->width : 0u,
                    level0 ? (unsigned)level0->height : 0u,
                    level0 ? (unsigned)level0->depth : 0u,
                    (unsigned long)(level0 ? level0->data_size : 0u),
                    (unsigned)cpuFirst, cpuOk ? 1 : 0,
                    (unsigned)(level0 ? level0->last_init_source : 0u),
                    (unsigned long)(level0 ? level0->last_upload_size : 0u),
                    (void *)(level0 ? level0->last_src_ptr : NULL),
                    (unsigned long long)(level0 ? level0->last_src_hash : 0ull),
                    (unsigned)(level0 ? level0->ever_written : 0u),
                    (unsigned)(level0 ? level0->has_initialized_data : 0u),
                    (unsigned)(level0 ? level0->suspicious_zero_upload : 0u),
                    (__bridge void *)texture,
                    (unsigned long)(texture ? mglBindingStateTextureWidth(texture)
                                            : 0),
                    (unsigned long)(texture ? mglBindingStateTextureHeight(texture)
                                            : 0),
                    (__bridge void *)sampler, usedFallback ? 1 : 0);
            }
        } else if (texture) {
            bound++;
            if (usedFallback) {
                fallback++;
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
    if (boundCount) {
        *boundCount = bound;
    }
    if (fallbackCount) {
        *fallbackCount = fallback;
    }
    if (nilCount) {
        *nilCount = nilTex;
    }
    if (samplerCount) {
        *samplerCount = boundSamplers;
    }
    return true;
}

- (bool)recoverFragmentSampledDepthTexture:(Texture **)ptrPtr
                                    texture:(id *)texturePtr
                                sampledName:(const char *)sampledName
                                spirvBinding:(GLuint)spirvBinding
                                  textureUnit:(GLuint)textureUnit
                                 expectedType:(uint32_t)expectedType
                                 expectedKind:(MGLTextureDataKind)expectedKind
                         fragmentProgramName:(GLuint)fragmentProgramName
                  suppressMissingTextureFallback:(BOOL *)suppressMissingTextureFallbackPtr
                            usedFallbackTexture:(BOOL *)usedFallbackTexturePtr
{
    Texture *ptr = *ptrPtr;
    id texture = *texturePtr;
    BOOL suppressMissingTextureFallback = *suppressMissingTextureFallbackPtr;
    BOOL usedFallbackTexture = *usedFallbackTexturePtr;

    RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
    MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
    if (ptr->mtl_data) {
        texture = (__bridge id)(ptr->mtl_data);
        /* Defer RT base-level views until Y-flip (avoids MRT view races). */
        if (!ptr->is_render_target) {
            texture = (__bridge id)mglSampledTextureViewForBaseLevel(
                ptr, (__bridge void *)texture);
        }
    }

    TextureLevel *depthSampleLevel0 = mglTraceTextureBaseLevel(ptr);
    MGLDepthRecoverInput gin = {0};
    mglBindingTextureFillDepthRecoverGateInput(
        &gin, texture ? 1 : 0,
        mglBindingTextureSampledNameIsInSampler(sampledName),
        texture && mglMetalPixelFormatIsDepthOrStencil(
                       mglBindingStateTexturePixelFormat(texture)),
        ptr && ptr->is_render_target ? 1 : 0,
        depthSampleLevel0 && depthSampleLevel0->ever_written,
        depthSampleLevel0 && depthSampleLevel0->has_initialized_data);
    MGLDepthRecoverPlan gplan = {0};
    if (mglBindingTexturePlanDepthRecover(&gin, &gplan) != 0 ||
        gplan.action == MGL_DR_ACTION_KEEP) {
        goto done;
    }

    if (gplan.action == MGL_DR_ACTION_ENTER_INSAMPLER) {
        GLuint pairedFboName = 0u;
        Texture *pairedColor =
            mglFindFramebufferColorTexturePairedWithDepth(ctx, ptr, &pairedFboName);
        Texture *recoverTexture = NULL;
        id recoverMTL = nil;
        const char *recoverReason = "none";
        BOOL recoveredFromSampledCopy = NO, recoveredFromPreviousVersion = NO;
        NSUInteger recoverAtt = MAX_COLOR_ATTACHMENTS;
        NSUInteger curAtt = MAX_COLOR_ATTACHMENTS;
        BOOL pairedCur = mglCurrentDrawFramebufferUsesColorTexture(
            ctx, pairedColor, pairedFboName, &curAtt);
        id pairedMTL = nil;
        if (pairedColor) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:pairedColor]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            pairedMTL = pairedColor->mtl_data ? (__bridge id)(pairedColor->mtl_data) : nil;
            if (!pairedCur && pairedMTL) {
                pairedCur = mglBindingStateRenderPassUsesColorTexture(
                    _renderPassManager->state->renderPassStateOwner,
                    (__bridge void *)pairedMTL, &curAtt);
            }
        }
        MGLDepthRecoverInput iin = {0};
        mglBindingTextureFillDepthRecoverInSamplerInput(
            &iin, pairedColor ? 1 : 0, pairedCur ? 1 : 0, pairedMTL ? 1 : 0,
            pairedMTL && mglMetalPixelFormatIsDepthOrStencil(
                             mglBindingStateTexturePixelFormat(pairedMTL)),
            textureUnit < TEXTURE_UNITS ? 1 : 0);
        MGLDepthRecoverPlan iplan = {0};
        (void)mglBindingTexturePlanDepthRecover(&iin, &iplan);

        if (iplan.action == MGL_DR_ACTION_PROBE_PAIRED_COPY) {
            static uint64_t s_histSup = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_histSup)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_HIST_SUPPRESSED, .hit = s_histSup, .program = fragmentProgramName, .binding = spirvBinding, .unit = textureUnit, .fbo = pairedFboName, .color_att = curAtt, .depth_tex = ptr ? ptr->name : 0u, .paired_color = pairedColor ? pairedColor->name : 0u);
            }
            id pairedCopy = nil;
            BOOL usedPrev = NO;
            int usable = pairedColor && mglRendererGLSampledCopyLooksUsable(
                pairedColor, expectedType, expectedKind, YES, &pairedCopy, &usedPrev);
            MGLDepthRecoverInput cin = {0};
            mglBindingTextureFillDepthRecoverCopyInput(&cin, usable ? 1 : 0);
            MGLDepthRecoverPlan cplan = {0};
            (void)mglBindingTexturePlanDepthRecover(&cin, &cplan);
            if (cplan.action == MGL_DR_ACTION_USE_RECOVER) {
                recoverTexture = pairedColor;
                recoverMTL = pairedCopy;
                recoverReason = cplan.reason_tag ? cplan.reason_tag : "paired-current-copy";
                recoveredFromSampledCopy = YES;
                recoveredFromPreviousVersion = usedPrev;
                recoverAtt = curAtt;
            } else {
                static uint64_t s_noCopy = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_noCopy)) {
                    MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_NO_COPY, .hit = s_noCopy, .program = fragmentProgramName, .binding = spirvBinding, .unit = textureUnit, .fbo = pairedFboName, .color_att = curAtt, .depth_tex = ptr ? ptr->name : 0u, .color_tex = pairedColor ? pairedColor->name : 0u, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .sampled_ver = pairedColor ? pairedColor->mtl_gl_sampled_write_version : 0u, .rt_ver = pairedColor ? pairedColor->mtl_render_target_write_version : 0u);
                }
                texture = nil;
                suppressMissingTextureFallback = YES;
            }
        } else if (iplan.action == MGL_DR_ACTION_USE_PAIRED_DIRECT) {
            static uint64_t s_rec = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_rec)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_PAIRED_DIRECT, .hit = s_rec, .program = fragmentProgramName, .binding = spirvBinding, .unit = textureUnit, .fbo = pairedFboName, .depth_tex = ptr ? ptr->name : 0u, .color_tex = pairedColor->name, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .color_fmt = mglBindingStateTexturePixelFormat(pairedMTL), .w = mglBindingStateTextureWidth(pairedMTL), .h = mglBindingStateTextureHeight(pairedMTL));
            }
            ptr = pairedColor;
            texture = pairedMTL;
        } else if (iplan.action == MGL_DR_ACTION_SCAN_HISTORY) {
            for (GLuint hi = 0; hi < MGL_RECENT_SAMPLED_2D_HISTORY; hi++) {
                Texture *cand =
                    MGL_STATE(ctx)->recent_sampled_2d_textures[textureUnit][hi];
                if (!cand || cand == ptr || cand == pairedColor ||
                    !mglRendererTextureLooksLikeSampledColor2D(ctx, cand))
                    continue;
                id candMTL = cand->mtl_data ? (__bridge id)(cand->mtl_data) : nil;
                NSUInteger candAtt = MAX_COLOR_ATTACHMENTS;
                BOOL candCur =
                    mglCurrentDrawFramebufferUsesColorTexture(ctx, cand, 0u, &candAtt) ||
                    mglBindingStateRenderPassUsesColorTexture(
                        _renderPassManager->state->renderPassStateOwner,
                        (__bridge void *)candMTL, &candAtt);
                if (!candCur && (!cand->mtl_data || cand->dirty_bits)) {
                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture:cand]);
                    MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
                    candMTL = cand->mtl_data ? (__bridge id)(cand->mtl_data) : nil;
                    candAtt = MAX_COLOR_ATTACHMENTS;
                    candCur =
                        mglCurrentDrawFramebufferUsesColorTexture(ctx, cand, 0u, &candAtt) ||
                        mglBindingStateRenderPassUsesColorTexture(
                            _renderPassManager->state->renderPassStateOwner,
                            (__bridge void *)candMTL, &candAtt);
                }
                id candCopy = nil;
                BOOL usedPrev = NO;
                int copyOk = cand->is_render_target &&
                    mglRendererGLSampledCopyLooksUsable(cand, expectedType, expectedKind,
                                                        candCur, &candCopy, &usedPrev);
                MGLDepthRecoverInput hin = {0};
                mglBindingTextureFillDepthRecoverHistoryInput(
                    &hin, 1, cand->is_render_target ? 1 : 0, candCur ? 1 : 0,
                    candMTL ? 1 : 0, copyOk ? 1 : 0,
                    candMTL && mglMetalPixelFormatIsDepthOrStencil(
                                   mglBindingStateTexturePixelFormat(candMTL)),
                    !candMTL || expectedType == 0 ||
                        mglBindingStateTextureType(candMTL) == expectedType,
                    !candMTL ||
                        mglTexturePixelFormatCompatibleWithExpectedDataKind(
                            mglBindingStateTexturePixelFormat(candMTL), expectedKind));
                MGLDepthRecoverPlan hplan = {0};
                (void)mglBindingTexturePlanDepthRecover(&hin, &hplan);
                if (hplan.action == MGL_DR_ACTION_HISTORY_USE_COPY) {
                    recoverTexture = cand;
                    recoverMTL = candCopy;
                    recoverReason = hplan.reason_tag ? hplan.reason_tag : "history-copy";
                    recoveredFromSampledCopy = YES;
                    recoveredFromPreviousVersion = usedPrev;
                    recoverAtt = candAtt;
                    break;
                }
                if (hplan.action == MGL_DR_ACTION_HISTORY_USE_DIRECT) {
                    recoverTexture = cand;
                    recoverMTL = candMTL;
                    recoverReason = hplan.reason_tag ? hplan.reason_tag : "history-direct";
                    recoverAtt = candAtt;
                    break;
                }
            }
        } else if (iplan.action == MGL_DR_ACTION_LOG_UNPAIRED) {
            static uint64_t s_unp = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_unp)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_UNPAIRED, .hit = s_unp, .program = fragmentProgramName, .binding = spirvBinding, .unit = textureUnit, .depth_tex = ptr ? ptr->name : 0u, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .w = mglBindingStateTextureWidth(texture), .h = mglBindingStateTextureHeight(texture));
            }
        }

        if (recoverTexture && recoverMTL) {
            static uint64_t s_histRec = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_histRec)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_HISTORY_RECOVERY, .hit = s_histRec, .reason = recoverReason, .program = fragmentProgramName, .binding = spirvBinding, .unit = textureUnit, .fbo = pairedFboName, .color_att = recoverAtt, .depth_tex = ptr ? ptr->name : 0u, .recover_tex = recoverTexture ? recoverTexture->name : 0u, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .recover_fmt = mglBindingStateTexturePixelFormat(recoverMTL), .w = mglBindingStateTextureWidth(recoverMTL), .h = mglBindingStateTextureHeight(recoverMTL), .copy = recoveredFromSampledCopy ? 1 : 0, .prev_ver = recoveredFromPreviousVersion ? 1 : 0, .sampled_ver = recoverTexture ? recoverTexture->mtl_gl_sampled_write_version : 0u, .rt_ver = recoverTexture ? recoverTexture->mtl_render_target_write_version : 0u, .paired_color = pairedColor ? pairedColor->name : 0u, .paired_current = pairedCur ? 1 : 0);
            }
            ptr = recoverTexture;
            texture = recoverMTL;
        }
        goto done;
    }

    /* ENTER_RT — plan@C rt_sub 0/1/2 + thin bind/fallback ports. */
    {
        Texture *unitActive =
            textureUnit < TEXTURE_UNITS ? MGL_STATE(ctx)->active_textures[textureUnit] : NULL;
        Texture *unit2D = textureUnit < TEXTURE_UNITS
            ? MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D] : NULL;
        Texture *last2D = textureUnit < TEXTURE_UNITS
            ? MGL_STATE(ctx)->last_sampled_2d_textures[textureUnit] : NULL;
        Texture *recoverTexture = NULL;
        const char *recoverReason = "none";
        GLuint recoverFboName = 0u;
        Texture *pairedColor =
            mglFindFramebufferColorTexturePairedWithDepth(ctx, ptr, &recoverFboName);
        NSUInteger drawAtt = MAX_COLOR_ATTACHMENTS;
        if (pairedColor) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:pairedColor]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            id pairedMTL = pairedColor->mtl_data ? (__bridge id)(pairedColor->mtl_data) : nil;
            BOOL pairedCur = mglBindingStateRenderPassUsesColorTexture(
                _renderPassManager->state->renderPassStateOwner, (__bridge void *)pairedMTL,
                &drawAtt);
            MGLDepthRecoverInput rin = {0};
            mglBindingTextureFillDepthRecoverRTInput(
                &rin, 0, 1, pairedMTL ? 1 : 0, pairedCur ? 1 : 0,
                pairedMTL && mglMetalPixelFormatIsDepthOrStencil(
                                 mglBindingStateTexturePixelFormat(pairedMTL)),
                !pairedMTL || expectedType == 0 ||
                    mglBindingStateTextureType(pairedMTL) == expectedType,
                !pairedMTL ||
                    mglTexturePixelFormatCompatibleWithExpectedDataKind(
                        mglBindingStateTexturePixelFormat(pairedMTL), expectedKind),
                0, 0, 0, 0);
            MGLDepthRecoverPlan rplan = {0};
            (void)mglBindingTexturePlanDepthRecover(&rin, &rplan);
            if (rplan.action == MGL_DR_ACTION_RT_USE_PAIRED) {
                recoverTexture = pairedColor;
                recoverReason = rplan.reason_tag ? rplan.reason_tag : "paired-color";
            } else if (rplan.action == MGL_DR_ACTION_RT_SKIP_CURRENT) {
                static uint64_t s_skip = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_skip)) {
                    MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_RT_SKIP, .hit = s_skip, .program = fragmentProgramName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .fbo = recoverFboName, .color_att = drawAtt, .depth_tex = ptr ? ptr->name : 0u, .color_tex = pairedColor ? pairedColor->name : 0u);
                }
            }
        }
        int stillDepth = texture && mglMetalPixelFormatIsDepthOrStencil(
                                        mglBindingStateTexturePixelFormat(texture));
        MGLDepthRecoverInput r1 = {0};
        mglBindingTextureFillDepthRecoverRTInput(
            &r1, 1, 0, 0, 0, 0, 0, 0, recoverTexture ? 1 : 0,
            (!recoverTexture &&
             mglRendererTextureLooksRecoverableSampled2D(ctx, last2D, expectedType,
                                                         expectedKind))
                ? 1
                : 0,
            stillDepth ? 1 : 0, 0);
        MGLDepthRecoverPlan p1 = {0};
        (void)mglBindingTexturePlanDepthRecover(&r1, &p1);
        if (p1.action == MGL_DR_ACTION_RT_SUPPRESS_LAST2D) {
            static uint64_t s_sup = 0;
            if (mglBindingTextureDepthRecoverLogHit(&s_sup)) {
                MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_RT_SUPPRESS_LAST2D, .hit = s_sup, .program = fragmentProgramName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .depth_tex = ptr ? ptr->name : 0u, .last2d = last2D->name);
            }
            mglBindingTextureFillDepthRecoverRTInput(
                &r1, 1, 0, 0, 0, 0, 0, 0, recoverTexture ? 1 : 0, 0,
                stillDepth ? 1 : 0, 0);
            (void)mglBindingTexturePlanDepthRecover(&r1, &p1);
        }
        if (p1.action == MGL_DR_ACTION_RT_APPLY && recoverTexture) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:recoverTexture]);
            MGL_ABORT_TBIND_IF_ENCODER_CLOSED();
            id recoverMTL =
                recoverTexture->mtl_data ? (__bridge id)(recoverTexture->mtl_data) : nil;
            int recoverOk =
                recoverMTL &&
                !mglMetalPixelFormatIsDepthOrStencil(
                    mglBindingStateTexturePixelFormat(recoverMTL)) &&
                (expectedType == 0 ||
                 mglBindingStateTextureType(recoverMTL) == expectedType) &&
                mglTexturePixelFormatCompatibleWithExpectedDataKind(
                    mglBindingStateTexturePixelFormat(recoverMTL), expectedKind);
            MGLDepthRecoverInput r2 = {0};
            mglBindingTextureFillDepthRecoverRTInput(
                &r2, 2, 0, 0, 0, 0, 0, 0, 0, 0, stillDepth ? 1 : 0,
                recoverOk ? 1 : 0);
            MGLDepthRecoverPlan p2 = {0};
            (void)mglBindingTexturePlanDepthRecover(&r2, &p2);
            if (p2.action == MGL_DR_ACTION_USE_RECOVER) {
                Framebuffer *currentFbo = ctx ? MGL_STATE(ctx)->framebuffer : NULL;
                GLuint colorTexName = 0u, depthTexName = 0u;
                if (currentFbo && mglRendererObjectPointerLikelyValid(currentFbo) &&
                    mglPointerRangeIsReadable(currentFbo, sizeof(*currentFbo))) {
                    colorTexName = currentFbo->color_attachments[0].texture;
                    depthTexName = currentFbo->depth.texture;
                }
                static uint64_t s_rt = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_rt)) {
                    MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_RT_RECOVER, .hit = s_rt, .reason = recoverReason, .program = fragmentProgramName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .depth_tex = ptr ? ptr->name : 0u, .recover_tex = recoverTexture->name, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .recover_fmt = mglBindingStateTexturePixelFormat(recoverMTL), .w = mglBindingStateTextureWidth(texture), .h = mglBindingStateTextureHeight(texture), .level = depthSampleLevel0, .ever = depthSampleLevel0 ? depthSampleLevel0->ever_written : 0u, .init = depthSampleLevel0 ? depthSampleLevel0->has_initialized_data : 0u, .unit_active = mglTraceTextureName(unitActive), .unit_tex2d = mglTraceTextureName(unit2D), .unit_last2d = mglTraceTextureName(last2D), .recover_fbo = recoverFboName, .current_fbo = currentFbo ? currentFbo->name : 0u, .color_tex = colorTexName, .fbo_depth_tex = depthTexName);
                }
                ptr = recoverTexture;
                texture = recoverMTL;
                stillDepth = 0;
            }
        }
        if (stillDepth ||
            (texture && mglMetalPixelFormatIsDepthOrStencil(
                            mglBindingStateTexturePixelFormat(texture)))) {
            id fallbackTexture =
                [self fallbackSampledTextureForExpectedType:expectedType
                                                   dataKind:expectedKind];
            if (fallbackTexture) {
                static uint64_t s_fb = 0;
                if (mglBindingTextureDepthRecoverLogHit(&s_fb)) {
                    MGL_EMIT_DR_LOG(.kind = MGL_DR_LOG_RT_FALLBACK, .hit = s_fb, .program = fragmentProgramName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .depth_tex = ptr ? ptr->name : 0u, .depth_fmt = mglBindingStateTexturePixelFormat(texture), .w = mglBindingStateTextureWidth(texture), .h = mglBindingStateTextureHeight(texture), .level = depthSampleLevel0, .ever = depthSampleLevel0 ? depthSampleLevel0->ever_written : 0u, .init = depthSampleLevel0 ? depthSampleLevel0->has_initialized_data : 0u, .unit_active = mglTraceTextureName(unitActive), .unit_tex2d = mglTraceTextureName(unit2D), .unit_last2d = mglTraceTextureName(last2D));
                }
                texture = fallbackTexture;
                usedFallbackTexture = YES;
            }
        }
    }

done:
    *ptrPtr = ptr;
    *texturePtr = texture;
    *suppressMissingTextureFallbackPtr = suppressMissingTextureFallback;
    *usedFallbackTexturePtr = usedFallbackTexture;
    return true;
}

- (void)emitSampledDiagPortsForProgram:(Program *)program
                                 stage:(const char *)stage
                       stageIsFragment:(BOOL)stageIsFragment
                           sampledName:(const char *)sampledName
                          spirvBinding:(GLuint)spirvBinding
                           textureUnit:(GLuint)textureUnit
                      sampledResource:(MGLShaderResource *)sampledResource
                                   ptr:(Texture *)ptr
                               texture:(id)texture
                               sampler:(id)sampler
                         usedFallback:(BOOL)usedFallback
                        expectedType:(uint32_t)expectedType
                          lookupType:(uint32_t)lookupType
                             bindCall:(uint64_t)bindCall
                          programName:(GLuint)programName
                     vertexProgramName:(GLuint)vertexProgramName
                   fragmentProgramName:(GLuint)fragmentProgramName
                  usedSampledCopyTrace:(BOOL)usedSampledCopyTrace
                 directTextureForTrace:(id)directTextureForTrace
                 sampledCopyForTrace:(id)sampledCopyForTrace
                     focusedCounter:(uint64_t *)focusedCounter
                   traceFileCounter:(uint64_t *)traceFileCounter
{
    TextureLevel *level0 = mglTraceTextureBaseLevel(ptr);
    int expectedIndex =
        (int)mglRenderTextureIndexForMetalType((lookupType ? lookupType : expectedType));
    Texture *unitActive = NULL, *unitExpected = NULL, *unit2D = NULL, *unitCube = NULL;
    if (textureUnit < TEXTURE_UNITS) {
        unitActive = MGL_STATE(ctx)->active_textures[textureUnit];
        unit2D = MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_2D];
        unitCube = MGL_STATE(ctx)->texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP];
        if (expectedIndex >= 0 && expectedIndex < _MAX_TEXTURE_TYPES) {
            unitExpected =
                MGL_STATE(ctx)->texture_units[textureUnit].textures[expectedIndex];
        }
    }
    MGLSampledDiagEmitInput ein = {0};
    mglBindingTextureFillSampledDiagEmitCore(
        &ein, stage, programName, vertexProgramName, fragmentProgramName,
        sampledName, spirvBinding, textureUnit,
        sampledResource ? (int)sampledResource->sampler_unit : -1,
        (sampledResource && sampledResource->sampler_unit_explicit) ? 1 : 0,
        ptr ? ptr->name : 0u, ptr ? ptr->target : 0u,
        (usedFallback || (stageIsFragment && ptr && ptr->name == 13u)) ? 1 : 0,
        expectedType, lookupType, expectedIndex, mglTraceTextureName(unitActive),
        mglTraceTextureName(unitExpected), mglTraceTextureName(unit2D),
        mglTraceTextureName(unitCube),
        texture ? mglBindingStateTextureType(texture) : 0,
        texture ? mglBindingStateTextureWidth(texture) : 0,
        texture ? mglBindingStateTextureHeight(texture) : 0,
        texture ? mglBindingStateTexturePixelFormat(texture)
                : MGL_BINDING_PIXEL_FORMAT_INVALID,
        level0 ? level0->width : 0u, level0 ? level0->height : 0u,
        level0 ? level0->depth : 0u, level0 ? level0->data_size : 0u,
        level0 ? level0->ever_written : 0u,
        level0 ? level0->has_initialized_data : 0u,
        level0 ? level0->suspicious_zero_upload : 0u,
        level0 ? level0->last_init_source : 0u,
        level0 ? level0->last_upload_size : 0u,
        level0 ? level0->last_src_hash : 0ull,
        (level0 && level0->data && level0->data_size > 0)
            ? mglTraceHashBytes((const void *)(uintptr_t)level0->data,
                                level0->data_size)
            : 0ull,
        ptr ? ptr->name : 0u, stageIsFragment ? 1 : 0,
        stageIsFragment && ptr && mglTextureCanUseGLSampledRenderTargetCopy(ptr)
            ? 1
            : 0,
        stageIsFragment && mglIsFocusedLoadingProgram(programName) &&
                (bindCall <= 2048ull || ((bindCall % 512ull) == 0ull))
            ? 1
            : 0,
        !stageIsFragment &&
                ((program && program->name == 34u) ||
                 (!program && programName == 34u))
            ? 1
            : 0,
        ptr && mglRenderTextureTargetIsBuffer((uint32_t)ptr->target) ? 1 : 0,
        level0 && level0->suspicious_zero_upload,
        level0 && !level0->ever_written, level0 && !level0->has_initialized_data,
        texture ? 1 : 0, bindCall, usedSampledCopyTrace ? 1 : 0,
        ctx && MGL_STATE(ctx)->framebuffer ? MGL_STATE(ctx)->framebuffer->name
                                           : 0u,
        _renderPassManager->state->renderPassFramebufferName,
        textureUnit < TEXTURE_UNITS
            ? mglTraceTextureName(MGL_STATE(ctx)
                                      ->texture_units[textureUnit]
                                      .textures[_TEXTURE_BUFFER_TARGET])
            : 0u);
    ein.mtl = (__bridge const void *)texture;
    ein.ptr = ptr;
    ein.sampler = (__bridge const void *)sampler;
    ein.l0_src = level0 ? (const void *)(uintptr_t)level0->last_src_ptr : NULL;
    ein.do_focused = mglProgramNeedsBindingTrace(program) &&
                     mglShouldLogFocusedBinding(focusedCounter);
    ein.do_trace_file =
        mglProgramNeedsTraceLog(program) &&
        mglShouldLogTraceFileBindingForProgram(program, traceFileCounter);
    ein.direct_for_trace = (__bridge const void *)directTextureForTrace;
    ein.copy_for_trace = (__bridge const void *)sampledCopyForTrace;
    ein.rt_label = mglTraceTextureLabel(ptr);
    ein.rp_color = mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager->state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
    ein.rp_depth = mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager->state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    MGLSampledDiagEmitResult eres = {0};
    mglBindingTextureEmitSampledDiagPorts(&ein, &eres);
    if (eres.want_readback && texture && level0) {
        [self traceSampledTextureReadback:texture
                                    glTex:ptr
                                    level:level0
                                  program:programName
                                  binding:spirvBinding
                                    stage:stageIsFragment ? @"fragment"
                                                          : @"vertex"
                                   reason:(eres.readback_reason
                                               ? [NSString stringWithUTF8String:
                                                      eres.readback_reason]
                                               : @"")
                                      hit:eres.readback_hit];
    }
}

- (id)applySampledCompatFallbackPlan:(Texture *)ptr
                             texture:(id)texture
                        expectedType:(uint32_t)expectedType
                        expectedKind:(MGLTextureDataKind)expectedKind
                               stage:(const char *)stage
                         programName:(GLuint)programName
                        spirvBinding:(GLuint)spirvBinding
                       sampleProgram:(Program *)sampleProgram
                    usedFallbackOut:(BOOL *)usedFallbackOut
{
    MGLSampledTextureBindInput cin = {0};
    mglBindingTextureFillSampledCompatInput(
        &cin, texture ? 1 : 0,
        texture ? mglBindingStateTextureType(texture) : 0u, expectedType,
        !texture ||
                mglTexturePixelFormatCompatibleWithExpectedDataKind(
                    mglBindingStateTexturePixelFormat(texture), expectedKind)
            ? 1
            : 0);
    MGLSampledTextureBindPlan cplan = {0};
    if (mglBindingTexturePlanSampled(&cin, &cplan) != 0 ||
        (cplan.action != MGL_ST_ACTION_TYPE_FALLBACK &&
         cplan.action != MGL_ST_ACTION_KIND_FALLBACK)) {
        return texture;
    }
    static uint64_t s_compatMismatchLogCount = 0;
    if (mglBindingTextureRateLogHit(&s_compatMismatchLogCount, 32ull, 512ull)) {
        mglBindingLogTexCompatMismatch(
            cplan.action == MGL_ST_ACTION_TYPE_FALLBACK ? "TYPE" : "DATA",
            stage, spirvBinding, programName, ptr ? ptr->name : 0u, cin.mtl_type,
            expectedType, s_compatMismatchLogCount);
    }
    if (sampleProgram) {
        char dumpReason[128];
        snprintf(dumpReason, sizeof(dumpReason), "tex-%s-mismatch-%s-binding-%u",
                 cplan.action == MGL_ST_ACTION_TYPE_FALLBACK ? "type" : "data",
                 stage ? stage : "x", spirvBinding);
        mglWriteProgramMSLDump(sampleProgram, dumpReason);
    }
    texture = [self fallbackSampledTextureForExpectedType:expectedType
                                                 dataKind:expectedKind];
    if (usedFallbackOut) {
        *usedFallbackOut = YES;
    }
    return texture;
}

- (id)materializeSampledSamplerForTexture:(Texture *)ptr
                              textureUnit:(GLuint)textureUnit
                          defaultSampler:(id)defaultSampler
                            forceDefault:(BOOL)forceDefault
                           samplerTarget:(GLuint)samplerTarget
                             programName:(GLuint)programName
                            spirvBinding:(GLuint)spirvBinding
                                   stage:(const char *)stage
                                 texture:(id)texture
{
    Sampler *glSampler = (textureUnit < TEXTURE_UNITS)
                             ? MGL_STATE(ctx)->texture_samplers[textureUnit]
                             : NULL;
    MGLSamplerMaterializeInput in = {0};
    mglBindingTextureFillSamplerMaterializeInput(
        &in, forceDefault ? 1 : 0, textureUnit < TEXTURE_UNITS ? 1 : 0,
        glSampler ? 1 : 0, glSampler && glSampler->dirty_bits ? 1 : 0,
        glSampler && glSampler->mtl_data ? 1 : 0,
        ptr && ptr->params.mtl_data ? 1 : 0,
        (stage && stage[0] == 'v') ? 1 : 0);
    MGLSamplerMaterializePlan plan = {0};
    if (mglBindingTexturePlanSamplerMaterialize(&in, &plan) != 0) {
        return defaultSampler;
    }
    if (plan.action == MGL_SM_ACTION_USE_DEFAULT) {
        return defaultSampler;
    }
    id sampler = defaultSampler;
    const TextureParameter *params = NULL;
    GLuint samplerName = 0u;
    if (plan.action == MGL_SM_ACTION_USE_GL_SAMPLER && glSampler) {
        if (plan.recreate_gl_sampler_mtl) {
            if (glSampler->mtl_data) {
                mglSafeReleaseMetalObj((void **)&glSampler->mtl_data);
            }
            GLuint target = samplerTarget
                                ? samplerTarget
                                : (ptr ? ptr->target : GL_TEXTURE_2D);
            glSampler->mtl_data = (void *)CFBridgingRetain((__bridge id)mglTextureCreateSamplerForTexParam(&glSampler->params, target));
        }
        if (plan.clear_gl_sampler_dirty) {
            glSampler->dirty_bits = 0;
        }
        sampler = (__bridge id)(glSampler->mtl_data);
        params = &glSampler->params;
        samplerName = glSampler->name;
    } else if (plan.action == MGL_SM_ACTION_USE_TEX_PARAMS && ptr) {
        sampler = (__bridge id)(ptr->params.mtl_data);
        params = &ptr->params;
    } else {
        return defaultSampler;
    }
    if (params && mglTraceLogIsEnabled()) {
        mglBindingLogSamplerResolve(
            mglBindingTextureSamplerStageTag(stage), programName,
            spirvBinding, textureUnit, plan.source_tag ? plan.source_tag : "?",
            samplerName, params->min_filter, params->mag_filter, params->wrap_s,
            params->wrap_t, params->min_lod, params->max_lod,
            ptr ? ptr->name : 0u, ptr ? ptr->params.base_level : 0u,
            ptr ? ptr->params.max_level : 0u, ptr ? ptr->width : 0u,
            ptr ? ptr->height : 0u,
            texture ? mglBindingStateTextureWidth(texture) : 0u,
            texture ? mglBindingStateTextureHeight(texture) : 0u,
            texture ? mglBindingStateTextureMipmapLevelCount(texture) : 0u);
    }
    return sampler;
}

- (bool)applySampledRenderTargetCopyPlan:(Texture *)ptr
                                 texture:(id *)texturePtr
                             sampleProgram:(Program *)sampleProgram
                              expectedType:(uint32_t)expectedType
                              expectedKind:(MGLTextureDataKind)expectedKind
                         usedTypeFallback:(BOOL)usedTypeFallback
                                   stage:(const char *)stage
                            programName:(GLuint)programName
                            spirvBinding:(GLuint)spirvBinding
                              textureUnit:(GLuint)textureUnit
                              sampledName:(const char *)sampledName
                     usedSampledCopyOut:(BOOL *)usedSampledCopyOut
                   directTextureForTrace:(id *)directTextureForTrace
                   sampledCopyForTrace:(id *)sampledCopyForTrace
{
    if (!texturePtr || usedTypeFallback || !ptr || !ptr->is_render_target) {
        return true;
    }
    id texture = *texturePtr;
    MGLYFlipDecision yflip = mglDecideYFlipForSampledRT(ptr, sampleProgram);
    if (mglTraceRTYFlipDiagnosticsEnabled()) {
        mglBindingLogRTYFlipDecision(
            stage, programName, sampledName, spirvBinding, textureUnit, ptr->name,
            mglTraceTextureLabel(ptr), mglYFlipDecisionName(yflip), (int)yflip,
            ptr->mtl_render_yflip_authority, ptr->mtl_render_target_write_version,
            ptr->mtl_gl_sampled_write_version, ptr->mtl_gl_sampled_data ? 1 : 0,
            mglProgramHasExistingFramebufferSampleYFlip(sampleProgram) ? 1 : 0);
    }

    id sampledCopy = ptr->mtl_gl_sampled_data
                         ? (__bridge id)(ptr->mtl_gl_sampled_data)
                         : nil;
    MGLSampledTextureBindInput in = {0};
    mglBindingTextureFillSampledRTInput(
        &in, usedTypeFallback ? 1 : 0, 1, (int)yflip,
        ptr->mtl_gl_sampled_data ? 1 : 0,
        mglGLSampledCopyContentFresh(ptr) ? 1 : 0,
        mglTextureCanUseGLSampledRenderTargetCopy(ptr) ? 1 : 0,
        (stage && stage[0] == 'f') ? 1 : 0,
        sampledCopy &&
                (expectedType == 0 ||
                 mglBindingStateTextureType(sampledCopy) == expectedType)
            ? 1
            : 0,
        sampledCopy &&
                mglTexturePixelFormatCompatibleWithExpectedDataKind(
                    mglBindingStateTexturePixelFormat(sampledCopy), expectedKind)
            ? 1
            : 0);

    MGLSampledTextureBindPlan plan = {0};
    if (mglBindingTexturePlanSampled(&in, &plan) != 0) {
        return true;
    }

    if (plan.action == MGL_ST_ACTION_RT_USE_COPY && sampledCopy) {
        if (directTextureForTrace) {
            *directTextureForTrace = texture;
        }
        if (sampledCopyForTrace) {
            *sampledCopyForTrace = sampledCopy;
        }
        if (mglTraceLogIsEnabled()) {
            MGL_EMIT_RT_LOG(.kind = MGL_RT_LOG_BIND, .stage = stage, .program = programName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .tex = ptr->name, .label = mglTraceTextureLabel(ptr), .original = (__bridge const void *)texture, .copy = (__bridge const void *)sampledCopy);
        }
        id chosen = sampledCopy;
        if (plan.apply_base_level_view) {
            chosen = (__bridge id)mglSampledTextureViewForBaseLevel(
                ptr, (__bridge void *)sampledCopy);
        }
        *texturePtr = chosen;
        if (usedSampledCopyOut) {
            *usedSampledCopyOut = YES;
        }
        return true;
    }

    if (plan.action == MGL_ST_ACTION_RT_REPAIR) {
        id repairedCopy =
            [self freshGLSampledRenderTargetCopyForSampling:ptr
                                                      source:texture
                                                       stage:stage
                                                     program:programName
                                                     binding:spirvBinding
                                                        unit:textureUnit
                                                expectedType:expectedType
                                                expectedKind:expectedKind];
        if (!repairedCopy) {
            return true;
        }
        in.repaired_available = 1;
        in.repaired_fresh = mglGLSampledCopyContentFresh(ptr) ? 1 : 0;
        if (mglBindingTexturePlanSampled(&in, &plan) != 0) {
            return true;
        }
        if (plan.action == MGL_ST_ACTION_RT_RETRY) {
            return false;
        }
        if (plan.action == MGL_ST_ACTION_RT_USE_COPY) {
            id chosen = repairedCopy;
            if (plan.apply_base_level_view) {
                chosen = (__bridge id)mglSampledTextureViewForBaseLevel(
                    ptr, (__bridge void *)repairedCopy);
            }
            *texturePtr = chosen;
            if (usedSampledCopyOut) {
                *usedSampledCopyOut = YES;
            }
        }
        return true;
    }

    if (plan.action == MGL_ST_ACTION_RT_GATE_MISS && mglTraceLogIsEnabled()) {
        MGL_EMIT_RT_LOG(.kind = MGL_RT_LOG_GATE_MISS, .stage = stage, .program = programName, .name = sampledName, .binding = spirvBinding, .unit = textureUnit, .tex = ptr->name, .label = mglTraceTextureLabel(ptr), .is_rt = 1, .has_copy = ptr->mtl_gl_sampled_data ? 1 : 0, .can_use = in.can_use_rt_copy, .expected_type = expectedType);
    } else if (plan.action == MGL_ST_ACTION_RT_ORIGINAL) {
        static uint64_t s_rtSampleCopySkipExistingFlipLogCount = 0;
        if (mglTraceLogIsEnabled() &&
            mglBindingTextureRateLogHit(&s_rtSampleCopySkipExistingFlipLogCount,
                                        32ull, 512ull)) {
            MGL_EMIT_RT_LOG(.kind = MGL_RT_LOG_SKIP_YFLIP, .hit = s_rtSampleCopySkipExistingFlipLogCount, .stage = stage, .program = programName, .name = sampledName, .binding = spirvBinding, .tex = ptr ? ptr->name : 0u, .decision_name = mglYFlipDecisionName(yflip), .decision = (int)yflip);
        }
        if (plan.apply_base_level_view && texture) {
            *texturePtr = (__bridge id)mglSampledTextureViewForBaseLevel(
                ptr, (__bridge void *)texture);
        }
    }
    return true;
}



- (bool)bindSeparateSamplersAndArrayTextures:(Program *)vertexProgram
                              fragmentProgram:(Program *)fragmentProgram
                        fragmentProgramName:(GLuint)fragmentProgramName
                          vertexProgramName:(GLuint)vertexProgramName
                             defaultSampler:(id)defaultSampler
                                    bindCall:(uint64_t)bindCall
                                  traceBind:(bool)traceBind
                         separateSamplerCount:(GLuint *)separateSamplerCount
                           boundSeparateSamplers:(GLuint *)boundSeparateSamplers
{
    const BOOL useResourceSnapshot = YES;
    MGLRenderResourceBindingSnapshot resourceSnapshot = {0};
    *separateSamplerCount = mglRendererGetProgramBindingCount(ctx, _FRAGMENT_SHADER, _SEPARATE_SAMPLERS_RES);
    *boundSeparateSamplers = 0;
    for (GLuint i = 0; i < *separateSamplerCount; i++)
    {
        GLuint spirvBinding = mglRendererGetProgramBinding(ctx, _FRAGMENT_SHADER, _SEPARATE_SAMPLERS_RES, (int)i);
        GLuint glBinding = mglRendererGetProgramGLBinding(ctx, _FRAGMENT_SHADER, _SEPARATE_SAMPLERS_RES, (int)i);
        if (!mglBindingTextureSeparateSamplerInRange(spirvBinding, glBinding,
                                                     TEXTURE_UNITS)) {
            continue;
        }
        Program *sampleProgram = fragmentProgram;
        MGLShaderResource *samplerResource = NULL;
        if (sampleProgram &&
            i < sampleProgram->shader_resources_list[_FRAGMENT_SHADER][_SEPARATE_SAMPLERS_RES].count) {
            samplerResource = &sampleProgram->shader_resources_list[_FRAGMENT_SHADER][_SEPARATE_SAMPLERS_RES].list[i];
        }
        GLuint textureUnit = mglTextureUnitForSampledResource(samplerResource, mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER), spirvBinding, _FRAGMENT_SHADER);

        id sampler = [self materializeSampledSamplerForTexture:NULL
                                                   textureUnit:textureUnit
                                               defaultSampler:defaultSampler
                                                 forceDefault:NO
                                                samplerTarget:(GLuint)mglRenderSamplerObjectTarget()
                                                  programName:fragmentProgramName
                                                 spirvBinding:spirvBinding
                                                        stage:"fragment"
                                                      texture:nil];
        if (sampler && spirvBinding < kMaxFragmentSamplerSlots) {
            if (!mglBindingStateQueueResourceBinding(
                    useResourceSnapshot, _bindingStateOwner,
                    _renderPassManager->state->currentRenderEncoderOwner,
                    &resourceSnapshot, MGL_RENDER_BINDING_STAGE_FRAGMENT,
                    MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                    (__bridge void *)sampler, spirvBinding)) {
                return false;
            }
            boundSeparateSamplers++;
        }

        if (traceBind && i < 6) {
            mglTraceLog("texbind.separateSampler call=%llu idx=%u binding=%u "
                        "unit=%u sampler=%p",
                        (unsigned long long)bindCall, (unsigned)i,
                        (unsigned)spirvBinding, (unsigned)textureUnit,
                        (__bridge void *)sampler);
        }
    }

    Program *arrayPrograms[] = { vertexProgram, fragmentProgram };
    int arrayStages[] = {
        _tessellation.nativeTESActive
            ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER,
        _FRAGMENT_SHADER
    };
    for (NSUInteger programIndex = 0; programIndex < 2; programIndex++) {
        Program *arrayProgram = arrayPrograms[programIndex];
        int arrayStage = arrayStages[programIndex];
        if (!arrayProgram) {
            continue;
        }

        MGLShaderResourceList *arrayResources =
            &arrayProgram->shader_resources_list[arrayStage][_SAMPLED_IMAGE_RES];
        for (GLuint resourceIndex = 0; arrayResources->list && resourceIndex < arrayResources->count; resourceIndex++) {
            MGLShaderResource *resource = &arrayResources->list[resourceIndex];
            if (resource->gl_array_size <= 1) {
                continue;
            }

            uint32_t expectedType = (uint32_t)
                mglRendererGetProgramExpectedTextureType(ctx, arrayStage, _SAMPLED_IMAGE_RES, (int)resourceIndex);
            for (GLint element = 1; element < resource->gl_array_size; element++) {
                GLuint metalSlot = resource->binding + (GLuint)element;
                GLuint samplerSlot =
                    mglMetalCombinedSamplerSlotForElement(resource,
                                                          (GLuint)element);
                if (!mglBindingTextureArrayElementSlotOk(metalSlot, TEXTURE_UNITS)) {
                    break;
                }

                GLuint textureUnit = mglTextureUnitForSampledResource(NULL, mglResolveProgramForStageFromState(ctx, arrayStage), metalSlot, arrayStage);
                Texture *arrayTexture = mglTextureForSampledResourceForStage(ctx, NULL, metalSlot, arrayStage, expectedType);
                id metalTexture = nil;
                id metalSampler = defaultSampler;
                if (arrayTexture && [self bindMTLTexture:arrayTexture]) {
                    metalTexture = (__bridge id)(arrayTexture->mtl_data);
                    metalSampler = [self materializeSampledSamplerForTexture:arrayTexture
                                                                 textureUnit:textureUnit
                                                             defaultSampler:defaultSampler
                                                               forceDefault:NO
                                                             samplerTarget:arrayTexture->target
                                                                programName:arrayProgram->name
                                                               spirvBinding:metalSlot
                                                                      stage:"vertex"
                                                                    texture:metalTexture];
                }
                if (!metalTexture) {
                    metalTexture = [self fallbackSampledTextureForExpectedType:expectedType
                                                                      dataKind:MGLTextureDataKindFloat];
                }

                uint32_t bindStage =
                    mglRenderTextureBindingStageForShader(arrayStage);
                if (!mglBindingStateQueueResourceBinding(
                        useResourceSnapshot, _bindingStateOwner,
                        _renderPassManager->state->currentRenderEncoderOwner,
                        &resourceSnapshot, bindStage,
                        MGL_RENDER_RESOURCE_BINDING_TEXTURE,
                        (__bridge void *)metalTexture, metalSlot)) {
                    return false;
                }
                if (mglBindingTextureShouldBindCombinedSampler(
                        resource->has_combined_sampler ? 1 : 0,
                        metalSampler ? 1 : 0, samplerSlot,
                        kMaxFragmentSamplerSlots)) {
                    if (!mglBindingStateQueueResourceBinding(
                            useResourceSnapshot, _bindingStateOwner,
                            _renderPassManager->state->currentRenderEncoderOwner,
                            &resourceSnapshot, bindStage,
                            MGL_RENDER_RESOURCE_BINDING_SAMPLER,
                            (__bridge void *)metalSampler, samplerSlot)) {
                        return false;
                    }
                }
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
    return true;
}

@end
