/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_renderer_host.c — the exported C symbols moved out of MGLRenderer.m
 * (P0-1, log 160).  Mechanical translation: `__bridge` casts become plain
 * pointer conversions and BOOL/NSUInteger/nil become int/size_t/NULL.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mgl_renderer_host.h"
#include "mgl_renderer_ports.h"
#include "mgl_render.h"
#include "mgl_texture_compat.h"
#include "mgl_rt_sync.h"
#include "mgl_trace_log.h"
#include "mgl_types_state.h"
#include "mgl_safety.h"       /* mglPointerRangeIsReadable */
#include "pixel_utils.h"     /* mtlFormatForGLInternalFormat */
#include "mgl_frame_activity.h" /* MGL_FRAME_LOAD / draw-since-swap */
#include "mgl_index_buffer.h"    /* mglGLIndexElementSize */

/* Restated from the ObjC private headers (rules 7 / 26). */
extern void mglMarkGLSampledCopyLevelDirty(Texture *tex, GLuint level);
extern signed char mglEnvFlagEnabled(const char *name);

/* The .m's file-local constants this TU needs (values copied verbatim). */
enum {
    MGL_RENDERER_PIXEL_FORMAT_INVALID = 0u,
    MGL_RENDERER_CB_NOT_ENQUEUED = 0u,
    MGL_RENDERER_LOAD_DONT_CARE = 0u,
    MGL_RENDERER_STORE_DONT_CARE = 0u,
};

/* The .m's texture-info statics this TU needs. */
static MGLRenderTextureInfo mglRendererTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

static uint32_t mglRendererTextureFieldFormat(void *texture)
{
    return mglRendererTextureInfo(texture).pixel_format;
}

static uint32_t mglRendererTextureFieldType(void *texture)
{
    return mglRendererTextureInfo(texture).texture_type;
}

static uint64_t mglRendererTextureFieldWidth(void *texture)
{
    return mglRendererTextureInfo(texture).width;
}

static uint64_t mglRendererTextureFieldHeight(void *texture)
{
    return mglRendererTextureInfo(texture).height;
}


/* MGLRenderer+Draw_Private.h's static inline predicate, in C (rule 7). */
static int mglRendererObjectPointerLikelyValid(const void *ptr)
{
    return mglObjectPointerLooksPlausible(ptr) ? 1 : 0;
}

/* Declared in MGLRenderer+RenderPass_Private.h; a real C function in the .m. */
extern int mglRendererPointerInHashTable(HashTable *table, const void *ptr);

signed char mglRendererTextureLooksRecoverableSampled2D(GLMContext glctx,
                                                  Texture *tex,
                                                  uint32_t expected_type,
                                                  MGLTextureDataKind expected_kind)
{
    if (!glctx || !tex) {
        return 0;
    }
    if (expected_type != 0 && expected_type != MGLTextureType2D) {
        return 0;
    }
    if (!mglRendererObjectPointerLikelyValid(tex) ||
        !mglRendererPointerInHashTable(&glctx->active_state->texture_table, tex) ||
        !mglPointerRangeIsReadable(tex, sizeof(*tex))) {
        return 0;
    }
    if (!mglRenderTextureTargetIs2D((uint32_t)tex->target) ||
        tex->index != _TEXTURE_2D ||
        tex->is_render_target ||
        mglRendererGLInternalFormatLooksDepthOrStencil(tex->internalformat)) {
        return 0;
    }

    TextureLevel *level0 = mglTraceTextureBaseLevel(tex);
    if (!level0 ||
        !level0->complete ||
        (!level0->ever_written && !level0->has_initialized_data)) {
        return 0;
    }

    void *mtlTexture = tex->mtl_data ? (tex->mtl_data) : NULL;
    if (mtlTexture) {
        if (mglMetalPixelFormatIsDepthOrStencil(mglRendererTextureFieldFormat(mtlTexture)) ||
            !mglTexturePixelFormatCompatibleWithExpectedDataKind(mglRendererTextureFieldFormat(mtlTexture), expected_kind)) {
            return 0;
        }
        if (expected_type != 0 && mglRendererTextureFieldType(mtlTexture) != expected_type) {
            return 0;
        }
    }

    return 1;
}

signed char mglRendererGLSampledCopyLooksUsable(Texture *tex,
                                         uint32_t expected_type,
                                         MGLTextureDataKind expected_kind,
                                         int allow_previous_write_version,
                                         void **copy_out,
                                                signed char *used_previous_out)
{
    if (copy_out) {
        *copy_out = NULL;
    }
    if (used_previous_out) {
        *used_previous_out = 0;
    }
    if (!tex || !tex->mtl_gl_sampled_data) {
        return 0;
    }

    void *sampledCopy = (tex->mtl_gl_sampled_data);
    if (!sampledCopy ||
        mglMetalPixelFormatIsDepthOrStencil(mglRendererTextureFieldFormat(sampledCopy)) ||
        !mglTexturePixelFormatCompatibleWithExpectedDataKind(mglRendererTextureFieldFormat(sampledCopy), expected_kind) ||
        (expected_type != 0 && mglRendererTextureFieldType(sampledCopy) != expected_type)) {
        return 0;
    }
    if (tex->mtl_gl_sampled_width != (GLuint)mglRendererTextureFieldWidth(sampledCopy) ||
        tex->mtl_gl_sampled_height != (GLuint)mglRendererTextureFieldHeight(sampledCopy) ||
        tex->mtl_gl_sampled_format != (GLuint)mglRendererTextureFieldFormat(sampledCopy)) {
        return 0;
    }

    if (tex->mtl_gl_sampled_dirty_mip_mask != 0u) {
        return 0;
    }

    int exactVersion =
        tex->mtl_gl_sampled_write_version != 0u &&
        tex->mtl_gl_sampled_write_version == tex->mtl_render_target_write_version;
    int previousVersion =
        allow_previous_write_version &&
        tex->mtl_gl_sampled_write_version != 0u &&
        tex->mtl_render_target_write_version != 0u &&
        tex->mtl_gl_sampled_write_version + 1u == tex->mtl_render_target_write_version;
    if (!exactVersion && !previousVersion) {
        return 0;
    }

    if (copy_out) {
        *copy_out = sampledCopy;
    }
    if (used_previous_out) {
        *used_previous_out = previousVersion;
    }
    return 1;
}






void mglMarkTextureLevelRenderTargetWrittenImpl(Texture *tex,
                                                 GLuint level,
                                                 const char *caller,
                                                 int line)
{
    TextureLevel *texLevel = mglTextureAttachmentLevel(tex, level);
    if (!texLevel) {
        return;
    }

    GLuint oldRenderTargetWriteVersion = tex->mtl_render_target_write_version;

    mglRenderMarkTextureLevelWritten(&texLevel->ever_written,
                                     &texLevel->has_initialized_data,
                                     &texLevel->suspicious_zero_upload);
    texLevel->last_init_source = kTexRenderTargetWrite;
    texLevel->last_upload_size = 0u;
    texLevel->last_src_ptr = NULL;
    texLevel->last_src_hash = 0ull;

    tex->mtl_render_target_write_version++;
    mglMarkGLSampledCopyLevelDirty(tex, level);


    tex->mtl_render_yflip_authority = (tex->mtl_render_target_write_version << 1);

    if (tex->name == 8u && mglEnvFlagEnabled("MGL_TRACE_RT_WRITE_MARKS")) {
        void *mtlTexture = tex->mtl_data ? (tex->mtl_data) : NULL;
        mglTraceLog("RT_WRITE_MARK tex=%u level=%u oldRtVer=%u newRtVer=%u caller=%s:%d mtl=%p fmt=%lu size=%lux%lu dirty=0x%x sampledVer=%u copy=%p",
                    (unsigned)tex->name,
                    (unsigned)level,
                    (unsigned)oldRenderTargetWriteVersion,
                    (unsigned)tex->mtl_render_target_write_version,
                    caller ? caller : "(unknown)",
                    line,
                    mtlTexture,
                    (unsigned long)(mtlTexture ? mglRendererTextureFieldFormat(mtlTexture) : MGL_RENDERER_PIXEL_FORMAT_INVALID),
                    (unsigned long)(mtlTexture ? mglRendererTextureFieldWidth(mtlTexture) : 0),
                    (unsigned long)(mtlTexture ? mglRendererTextureFieldHeight(mtlTexture) : 0),
                    (unsigned)tex->dirty_bits,
                    (unsigned)tex->mtl_gl_sampled_write_version,
                    tex->mtl_gl_sampled_data);
    }

    /*
     * Once Metal has rendered into a texture, the CPU-side backing copy is stale.
     * Keeping DIRTY_TEXTURE_DATA set lets a later sampler bind recreate the Metal
     * texture and upload old all-zero or placeholder bytes over the rendered
     * contents. Minecraft 1.21.8's item atlas and post-chain render targets hit
     * this path frequently.
     */
    tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
}

signed char mglRendererTextureLooksLikeSampledColor2D(GLMContext glctx,
                                                      Texture *tex)
{
    if (!glctx || !tex) {
        return 0;
    }
    if (!mglRendererObjectPointerLikelyValid(tex) ||
        !mglRendererPointerInHashTable(&glctx->active_state->texture_table, tex) ||
        !mglPointerRangeIsReadable(tex, sizeof(*tex))) {
        return 0;
    }
    if (!mglRenderTextureTargetIs2D((uint32_t)tex->target) ||
        tex->index != _TEXTURE_2D ||
        mglRendererGLInternalFormatLooksDepthOrStencil(tex->internalformat)) {
        return 0;
    }

    return 1;
}

Texture *mglFindFramebufferColorTexturePairedWithDepth(GLMContext glctx,
                                                              Texture *depthTexture,
                                                              GLuint *fboNameOut)
{
    if (fboNameOut) {
        *fboNameOut = 0u;
    }
    if (!glctx || !depthTexture) {
        return NULL;
    }

    Framebuffer *currentFbo = glctx->active_state->framebuffer;
    if (currentFbo &&
        mglRendererObjectPointerLikelyValid(currentFbo) &&
        mglPointerRangeIsReadable(currentFbo, sizeof(*currentFbo))) {
        int depthMatches =
            currentFbo->depth.buf.tex == depthTexture ||
            currentFbo->stencil.buf.tex == depthTexture ||
            currentFbo->depth.texture == depthTexture->name ||
            currentFbo->stencil.texture == depthTexture->name;
        if (depthMatches && (currentFbo->color_attachment_bitfield & 1u) != 0u) {
            FBOAttachment *colorAttachment = &currentFbo->color_attachments[0];
            Texture *colorTexture = colorAttachment->buf.tex;
            if (!colorTexture && colorAttachment->texture != 0u) {
                colorTexture = (Texture *)searchHashTable(&glctx->active_state->texture_table,
                                                          colorAttachment->texture);
            }
            /* Validate raw pointer is still registered (see table-scan path). */
            if (colorTexture) {
                Texture *verified = (Texture *)searchHashTable(&glctx->active_state->texture_table,
                                                                colorTexture->name);
                if (verified != colorTexture) {
                    colorAttachment->buf.tex = NULL;
                    colorAttachment->texture = 0u;
                    colorTexture = NULL;
                }
            }
            if (colorTexture &&
                colorTexture != depthTexture &&
                mglRendererObjectPointerLikelyValid(colorTexture) &&
                mglPointerRangeIsReadable(colorTexture, sizeof(*colorTexture)) &&
                (!colorTexture->mtl_data ||
                 !mglMetalPixelFormatIsDepthOrStencil(
                     mglRendererTextureFieldFormat(colorTexture->mtl_data)))) {
                if (fboNameOut) {
                    *fboNameOut = currentFbo->name;
                }
                return colorTexture;
            }
        }
    }

    HashTable *table = &glctx->active_state->framebuffer_table;
    if (!mglHashTableValidateStorage(table, "findPairedFramebufferColor") ||
        !table->keys || !table->states || table->size == 0u) {
        return NULL;
    }

    for (size_t slot = 0; slot < table->size; slot++) {
        if (table->states[slot] != 1u || !table->keys[slot].data) {
            continue;
        }

        Framebuffer *fbo = (Framebuffer *)table->keys[slot].data;
        if (!mglRendererObjectPointerLikelyValid(fbo) ||
            !mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
            continue;
        }

        int depthMatches =
            fbo->depth.buf.tex == depthTexture ||
            fbo->stencil.buf.tex == depthTexture ||
            fbo->depth.texture == depthTexture->name ||
            fbo->stencil.texture == depthTexture->name;
        if (!depthMatches) {
            continue;
        }

        FBOAttachment *colorAttachment = &fbo->color_attachments[0];
        Texture *colorTexture = colorAttachment->buf.tex;
        if (!colorTexture && colorAttachment->texture != 0u) {
            colorTexture = (Texture *)searchHashTable(&glctx->active_state->texture_table,
                                                      colorAttachment->texture);
        }

        /* Validate that the raw pointer is still registered in the texture
         * table.  glDeleteTextures frees the Texture struct but stale raw
         * pointers can survive in FBO attachments (and mglPointerRangeIsReadable
         * cannot reliably detect freed-but-mapped malloc memory). */
        if (colorTexture) {
            Texture *verified = (Texture *)searchHashTable(&glctx->active_state->texture_table,
                                                            colorTexture->name);
            if (verified != colorTexture) {

                colorAttachment->buf.tex = NULL;
                colorAttachment->texture = 0u;
                continue;
            }
        }

        if (!colorTexture ||
            colorTexture == depthTexture ||
            !mglRendererObjectPointerLikelyValid(colorTexture) ||
            !mglPointerRangeIsReadable(colorTexture, sizeof(*colorTexture))) {
            continue;
        }

        if (colorTexture->mtl_data &&
            mglMetalPixelFormatIsDepthOrStencil(
                mglRendererTextureFieldFormat(colorTexture->mtl_data))) {
            continue;
        }

        if (fboNameOut) {
            *fboNameOut = fbo->name;
        }
        return colorTexture;
    }

    return NULL;
}
