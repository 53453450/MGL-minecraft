/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_blit_sampled_copy.c — the sampled-copy cluster moved out of
 * MGLRenderer+Blit.m.  The first entry point (the eligibility predicate) is
 * here; the refresh/update pair follows once its render-encoder dependency is
 * available to C.
 */

#include "mgl_blit_sampled_copy.h"
#include "mgl_render.h"
#include "mgl_texture_compat.h"
#include "mgl_rt_sync.h"        /* mglTextureCanUseGLSampledRenderTargetCopy */

/* Local twin of the file-static mglBlitTextureInfo in MGLRenderer+Blit.m
 * (same three lines: fetch the Metal texture info, zero when absent). */
static MGLRenderTextureInfo mglBlitSampledCopyTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

int mglBlitTextureCanUseGLSampledRenderTargetCopy(Texture *tex, void *source)
{
    if (!tex || !source || !tex->is_render_target) {
        return 0;
    }

    if (!mglRenderTextureTargetIs2D((uint32_t)tex->target) ||
        tex->width == 0u ||
        tex->height == 0u ||
        mglBlitSampledCopyTextureInfo(source).texture_type != MGLTextureType2D ||
        mglBlitSampledCopyTextureInfo(source).mipmap_level_count == 0u ||
        mglBlitSampledCopyTextureInfo(source).width == 0u ||
        mglBlitSampledCopyTextureInfo(source).height == 0u ||
        mglMetalPixelFormatIsDepthOrStencil(mglBlitSampledCopyTextureInfo(source).pixel_format)) {
        return 0;
    }

    /* Float + integer color RTs need a GL-sampled copy for FBO feedback
     * (same texture as attachment and sampler).  Depth/stencil stay out. */
    MGLTextureDataKind kind =
        mglTextureDataKindForPixelFormat(mglBlitSampledCopyTextureInfo(source).pixel_format);
    if (kind != MGLTextureDataKindFloat &&
        kind != MGLTextureDataKindUint &&
        kind != MGLTextureDataKindSint) {
        return 0;
    }

    /* Apply sampled-copy protection to all 2D float render targets
     * regardless of size.  The previous size-based gating was a
     * Minecraft-specific heuristic that broke on larger render targets. */
    if (!mglTextureCanUseGLSampledRenderTargetCopy(tex)) {
        return 0;
    }

    return 1;
}
