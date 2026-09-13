/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */



#include "mgl_texture_compat.h"
#include "mgl_render.h"
#include "mgl_sampler_compat.h"   /* mglFindSamplerResourceForMetalBinding */
#include "mgl_metal_ref.h"         /* mglSafeReleaseMetalObj */

MGLTextureDataKind mglTextureDataKindForPixelFormat(uint32_t pixelFormat)
{
    return (MGLTextureDataKind)mglRenderTextureDataKindForPixelFormat(
        (uint32_t)pixelFormat);
}

const char *mglTextureDataKindName(MGLTextureDataKind kind)
{
    return mglRenderTextureDataKindName((uint32_t)kind);
}

size_t mglMetalTextureLevelDimension(size_t base, size_t level)
{

    return (size_t)mglRenderMetalTextureLevelDimension(
        (uint64_t)base, (uint64_t)level);
}

void *mglSampledTextureViewForBaseLevel(Texture *ptr, void *texture)
{
    (void)mglRenderSampledTextureViewForBaseLevel(ptr, texture, &texture);
    return texture;
}

size_t mglStoredColorComponentsForTexture(Texture *tex)
{
    if (!tex) {
        return 4;
    }
    return (size_t)mglRenderStoredColorComponents(
        (uint32_t)tex->internalformat);
}

uint32_t mglMTLSwizzleForGLSwizzle(Texture *tex, GLenum swizzle)
{
    size_t components = mglStoredColorComponentsForTexture(tex);
    return mglRenderMTLSwizzleForGLSwizzle(
        (uint32_t)swizzle, (uint32_t)components);
}

bool mglTextureUploadNeedsSingleChannelSwizzle(Texture *tex)
{
    if (!tex || !tex->params.swizzled) {
        return false;
    }
    return mglRenderTextureUploadNeedsSingleChannelSwizzle(
        (uint32_t)tex->internalformat, 1) != 0;
}

bool mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(Texture *tex)
{
    if (!tex || !tex->params.swizzled) {
        return false;
    }
    return mglRenderTextureUploadNeedsIntegerMultiChannelSwizzleBake(
        (uint32_t)tex->internalformat, 1) != 0;
}

bool mglTextureUploadNeedsSingleChannelSwizzleBake(Texture *tex)
{
    if (!tex || !tex->params.swizzled) {
        return false;
    }
    return mglRenderTextureUploadNeedsSingleChannelSwizzleBake(
        (uint32_t)tex->internalformat, 1) != 0;
}

bool mglTextureUploadNeedsStencilSwizzleBake(Texture *tex)
{
    if (!tex || !tex->params.swizzled) {
        return false;
    }
    return mglRenderTextureUploadNeedsStencilSwizzleBake(
        (uint32_t)tex->internalformat, 1,
        (uint32_t)tex->params.depth_stencil_mode) != 0;
}

bool mglTextureUploadNeedsDepthStencilDepthSwizzleBake(Texture *tex)
{
    if (!tex || !tex->params.swizzled) {
        return false;
    }
    return mglRenderTextureUploadNeedsDepthStencilDepthSwizzleBake(
        (uint32_t)tex->internalformat, 1,
        (uint32_t)tex->params.depth_stencil_mode) != 0;
}

bool mglTextureUploadNeedsSwizzleBake(Texture *tex)
{
    return mglTextureUploadNeedsSingleChannelSwizzleBake(tex) ||
           mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) ||
           mglTextureUploadNeedsStencilSwizzleBake(tex) ||
           mglTextureUploadNeedsDepthStencilDepthSwizzleBake(tex);
}

uint8_t mglResolveR8SwizzledComponent(Texture *tex, GLenum swizzle, uint8_t red)
{
    (void)tex;

    return mglRenderResolveR8SwizzledComponent((uint32_t)swizzle, red);
}

uint8_t *mglCreateSingleChannelSwizzledUpload(Texture *tex,
                                              const uint8_t *srcData,
                                              size_t width,
                                              size_t height,
                                              size_t srcBytesPerRow,
                                              size_t *outBytesPerRow,
                                              size_t *outBytesPerImage)
{
    if (!tex || !srcData || width == 0 || height == 0 || !outBytesPerRow || !outBytesPerImage) {
        return NULL;
    }


    size_t outBPR = 0;
    size_t outBPI = 0;
    uint8_t *result = mglRenderCreateSingleChannelSwizzledUpload(
        (uint32_t)tex->internalformat,
        (uint32_t)tex->params.swizzle_r,
        (uint32_t)tex->params.swizzle_g,
        (uint32_t)tex->params.swizzle_b,
        (uint32_t)tex->params.swizzle_a,
        srcData, (size_t)width, (size_t)height, (size_t)srcBytesPerRow,
        &outBPR, &outBPI);
    if (result) {
        *outBytesPerRow = outBPR;
        *outBytesPerImage = outBPI;
    }
    return result;
}

uint8_t *mglCreateIntegerMultiChannelSwizzledUpload(Texture *tex,
                                                    const uint8_t *srcData,
                                                    size_t width,
                                                    size_t height,
                                                    size_t srcBytesPerRow,
                                                    size_t *outBytesPerRow,
                                                    size_t *outBytesPerImage)
{
    if (!tex || !srcData || width == 0 || height == 0 || !outBytesPerRow || !outBytesPerImage) {
        return NULL;
    }

    size_t outBPR = 0;
    size_t outBPI = 0;
    uint8_t *result = mglRenderCreateIntegerMultiChannelSwizzledUpload(
        (uint32_t)tex->internalformat,
        (uint32_t)tex->params.swizzle_r,
        (uint32_t)tex->params.swizzle_g,
        (uint32_t)tex->params.swizzle_b,
        (uint32_t)tex->params.swizzle_a,
        srcData, (size_t)width, (size_t)height, (size_t)srcBytesPerRow,
        &outBPR, &outBPI);
    if (result) {
        *outBytesPerRow = outBPR;
        *outBytesPerImage = outBPI;
    }
    return result;
}

uint8_t *mglCreateStencilSwizzledUpload(Texture *tex,
                                        const uint8_t *srcData,
                                        size_t width,
                                        size_t height,
                                        size_t srcBytesPerRow,
                                        size_t *outBytesPerRow,
                                        size_t *outBytesPerImage)
{
    if (!tex || !srcData || width == 0 || height == 0 || !outBytesPerRow || !outBytesPerImage) {
        return NULL;
    }

    size_t outBPR = 0;
    size_t outBPI = 0;
    uint8_t *result = mglRenderCreateStencilSwizzledUpload(
        (uint32_t)tex->internalformat,
        (uint32_t)tex->params.swizzle_r,
        (uint32_t)tex->params.swizzle_g,
        (uint32_t)tex->params.swizzle_b,
        (uint32_t)tex->params.swizzle_a,
        srcData, (size_t)width, (size_t)height, (size_t)srcBytesPerRow,
        &outBPR, &outBPI);
    if (result) {
        *outBytesPerRow = outBPR;
        *outBytesPerImage = outBPI;
    }
    return result;
}

uint8_t *mglCreateSwizzledUpload(Texture *tex,
                                 const uint8_t *srcData,
                                 size_t width,
                                 size_t height,
                                 size_t srcBytesPerRow,
                                 size_t *outBytesPerRow,
                                 size_t *outBytesPerImage)
{
    uint8_t *result = NULL;
    if (mglTextureUploadNeedsStencilSwizzleBake(tex)) {
        result = mglCreateStencilSwizzledUpload(
            tex, srcData, width, height, srcBytesPerRow,
            outBytesPerRow, outBytesPerImage);
    }
    if (result) {
        return result;
    }
    if (mglTextureUploadNeedsDepthStencilDepthSwizzleBake(tex)) {
        result = mglCreateSingleChannelSwizzledUpload(
            tex, srcData, width, height, srcBytesPerRow,
            outBytesPerRow, outBytesPerImage);
    }
    if (result) {
        return result;
    }
    if (mglTextureUploadNeedsSingleChannelSwizzleBake(tex)) {
        result = mglCreateSingleChannelSwizzledUpload(
            tex, srcData, width, height, srcBytesPerRow,
            outBytesPerRow, outBytesPerImage);
    }
    if (result) {
        return result;
    }
    return mglCreateIntegerMultiChannelSwizzledUpload(
        tex, srcData, width, height, srcBytesPerRow,
        outBytesPerRow, outBytesPerImage);
}

bool mglTextureInternalFormatNeedsRGBA8Expansion(GLenum internalformat,
                                                 uint32_t pixelFormat)
{
    return mglRenderTextureInternalFormatNeedsRGBA8Expansion(
        (uint32_t)internalformat, pixelFormat) != 0;
}

bool mglTextureNeedsChannelExpansion(GLenum internalformat,
                                     uint32_t pixelFormat)
{
    return mglRenderTextureNeedsChannelExpansion(
        (uint32_t)internalformat, pixelFormat) != 0;
}

uint8_t *mglCreateChannelExpandedUpload(Texture *tex,
                                        uint32_t pixelFormat,
                                        const uint8_t *srcData,
                                        size_t width,
                                        size_t height,
                                        size_t srcBytesPerRow,
                                        size_t *outBytesPerRow,
                                        size_t *outBytesPerImage)
{
    if (!tex || !srcData || width == 0 || height == 0 ||
        srcBytesPerRow == 0 || !outBytesPerRow || !outBytesPerImage ||
        !mglTextureNeedsChannelExpansion(tex->internalformat,
                                         (uint32_t)pixelFormat)) {
        return NULL;
    }


    size_t outBPR = 0;
    size_t outBPI = 0;
    uint8_t *result = mglRenderCreateChannelExpandedUpload(
        (uint32_t)tex->internalformat, (uint32_t)pixelFormat,
        srcData, (size_t)width, (size_t)height, (size_t)srcBytesPerRow,
        &outBPR, &outBPI);
    if (result) {
        *outBytesPerRow = outBPR;
        *outBytesPerImage = outBPI;
    }
    return result;
}

uint8_t *mglCreateRGBA8ExpandedUpload(Texture *tex,
                                      const uint8_t *srcData,
                                      size_t width,
                                      size_t height,
                                      size_t srcBytesPerRow,
                                      size_t *outBytesPerRow,
                                      size_t *outBytesPerImage)
{
    if (!tex || !srcData || width == 0 || height == 0 ||
        srcBytesPerRow == 0 || !outBytesPerRow || !outBytesPerImage ||
        !mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, 70u)) {
        return NULL;
    }


    return mglRenderCreateRGBA8ExpandedUpload(
        srcData, width, height, srcBytesPerRow,
        (uint32_t)tex->internalformat, outBytesPerRow, outBytesPerImage);
}


/* === Layer pixel format helpers === */

bool mglMetalLayerPixelFormatIsSupported(uint32_t pixelFormat)
{
    return mglRenderMetalLayerPixelFormatIsSupported(pixelFormat) != 0;
}

uint32_t mglSRGBPixelFormat(uint32_t fmt)
{
    return mglRenderSRGBPixelFormat(fmt);
}

uint32_t mglLinearPixelFormat(uint32_t fmt)
{
    return mglRenderLinearPixelFormat(fmt);
}

uint32_t mglEffectiveMTLPixelFormatForTexture(uint32_t fmt, Texture *tex)
{
    uint32_t decode = tex ? (uint32_t)tex->params.srgb_decode_ext : 0u;
    return mglRenderEffectiveMTLPixelFormat(fmt, decode);
}

/* === Moved out of MGLRenderer+Texture.m (P0-1 / T3 sink) ===
 * The three entry points below were Objective-C methods whose bodies were
 * already plain C; C callers (the batch drivers, the compute binding loop) call
 * them directly now, which also retired a shim port. */

GLuint mglTextureUnitForSampledResource(MGLShaderResource *sampled_resource,
                                        Program *program, GLuint metal_binding,
                                        int stage)
{
    if (!program) {
        GLuint candidate = sampled_resource &&
                           sampled_resource->sampler_unit >= 0 &&
                           sampled_resource->sampler_unit < TEXTURE_UNITS
            ? (GLuint)sampled_resource->sampler_unit
            : metal_binding;
        return candidate;
    }

    MGLShaderResource *res = sampled_resource;
    const char *sampledName = NULL;
    if (!res && metal_binding < TEXTURE_UNITS) {
        res = mglFindSamplerResourceForMetalBinding(program, stage, metal_binding);
    }
    if (res) {
        sampledName = res->name;
    }
    (void)sampledName;

    /*
     * Minecraft usually assigns sampler texture units from the RenderPipeline
     * sampler list, not from numeric suffixes like Sampler2. For example, chunk
     * rendering declares Sampler0 and Sampler2, so Sampler2 can be uploaded
     * through glUniform1i(..., 1). Keep sampler units on the exact reflected
     * resource instead of only the Metal binding: vertex and fragment resources
     * commonly share binding numbers, and binding-level state can make entity,
     * hand, and text textures bleed into each other.
     */
    if (res) {
        uint32_t explicitUnit = mglRenderSampledResourceUnit(
            res->sampler_unit_explicit ? 1 : 0,
            res->sampler_unit, metal_binding,
            res->binding, TEXTURE_UNITS);
        if (explicitUnit != UINT32_MAX) {
            return explicitUnit;
        }
    }

    if (mglRenderMetalBindingPastUnits(metal_binding, TEXTURE_UNITS)) {
        return metal_binding;
    }

    bool stageExplicit = mglRenderShaderStageValid(stage)
        ? mglRenderSamplerUnitExplicit(
              (uint32_t)program->sampler_units_explicit_by_stage[stage][metal_binding]) != 0
        : false;
    bool globalExplicit = mglRenderSamplerUnitExplicit(
                              (uint32_t)program->sampler_units_explicit[metal_binding]) != 0;

    GLint unit = mglRenderShaderStageValid(stage)
        ? program->sampler_units_by_stage[stage][metal_binding]
        : program->sampler_units[metal_binding];

    if (stageExplicit && mglRenderSamplerUnitValid(unit, TEXTURE_UNITS)) {
        return (GLuint)unit;
    }

    unit = program->sampler_units[metal_binding];
    if (globalExplicit && mglRenderSamplerUnitValid(unit, TEXTURE_UNITS)) {
        return (GLuint)unit;
    }

    GLint defaultUnit = mglRenderShaderStageValid(stage)
        ? program->sampler_units_by_stage[stage][metal_binding]
        : program->sampler_units[metal_binding];
    if (!mglRenderSamplerUnitValid(defaultUnit, TEXTURE_UNITS)) {
        defaultUnit = program->sampler_units[metal_binding];
    }

    if (res && !res->sampler_unit_explicit) {
        uint32_t implicitUnit = mglRenderSampledResourceUnit(
            1, res->sampler_unit, metal_binding, res->binding, TEXTURE_UNITS);
        if (implicitUnit != UINT32_MAX) {
            return implicitUnit;
        }
    }

    return mglRenderDefaultSamplerUnit(defaultUnit, TEXTURE_UNITS);
}

void mglTextureSwizzleDescriptor(MGLRenderTextureDescriptorState *tex_desc,
                                 Texture *tex)
{
    if (!tex_desc || !tex) {
        return;
    }
    tex_desc->swizzle_red = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_r);
    tex_desc->swizzle_green = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_g);
    tex_desc->swizzle_blue = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_b);
    tex_desc->swizzle_alpha = mglMTLSwizzleForGLSwizzle(tex, tex->params.swizzle_a);
    tex_desc->has_swizzle = 1u;
}

void mglTextureReleaseGLSampledCopy(Texture *tex)
{
    if (!tex) {
        return;
    }
    if (tex->mtl_gl_sampled_data) {
        mglSafeReleaseMetalObj((void **)&tex->mtl_gl_sampled_data);
    }
    tex->mtl_gl_sampled_width = 0u;
    tex->mtl_gl_sampled_height = 0u;
    tex->mtl_gl_sampled_format = 0u;
    tex->mtl_gl_sampled_levels = 0u;
    tex->mtl_gl_sampled_write_version = 0u;
    tex->mtl_gl_sampled_dirty_mip_mask = 0u;
}
