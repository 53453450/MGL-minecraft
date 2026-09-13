/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+Binding.m
// Buffer/texture Metal object binding methods extracted from MGLRenderer+RenderPass.m

#import "MGLRenderer_Private.h"
#include "mgl_texture_sampler.h"
#include "mgl_render_pass_manager_ops.h"
#import "MGLRenderer+Blit_Private.h"
#include "mgl_render.h"
#include "mgl_batch_issue.h"
#include "mgl_texture_bind.h"

void mglRendererBindTexture(GLMContext glm_ctx,
                                  Texture *texture)
{
    MGLRendererBackendLease _backend_lease = {};
    if (mglRendererEnterBackendLease(glm_ctx, &_backend_lease) != 0) return;
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (renderer && glm_ctx && texture) {
        (void)[renderer bindMTLTexture:texture];
    }
    mglRendererBackendEnd(&_backend_lease);
}

@implementation MGLRenderer (Binding)



- (bool)bindMTLTexture:(Texture *)tex
{
    METAL_LOCK();
    const bool result = mglRendererBindMTLTexture((__bridge void *)self, tex);
    METAL_UNLOCK();
    return result;
}

- (bool)syncResourceBindingsForContext:(GLMContext)glm_ctx
                           alreadyDone:(const MGLResourceSyncWork *)done
{
    GLMState *state = MGL_STATE(glm_ctx);
    if (!done || !done->mappedBuffers) {
        RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
    }
    if (!done || !done->updatedBaseLists) {
        RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList:&state->vertex_buffer_map_list]);
        RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList:&state->fragment_buffer_map_list]);
    }
    MGLEncodeContext encCtx = {
        .render_encoder_owner = _renderPassManager.state->currentRenderEncoderOwner,
    };
    RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder:&encCtx]);
    RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder:&encCtx]);
    RETURN_FALSE_ON_FAILURE([self bindBufferSizeConstantsForRenderEncoder]);
    if (!done || !done->boundActiveTextures) {
        RETURN_FALSE_ON_FAILURE(mglBatchBindActiveTexturesToMTL((__bridge void *)self, ctx));
    }
    RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"final-active-texture-bind"]);
    if (![self bindTexturesToCurrentRenderEncoder:&encCtx]) {
        RETURN_FALSE_ON_FAILURE([self restoreRenderEncoderAfterTextureUploadForDraw:"final-sampled-texture-bind"]);
        RETURN_FALSE_ON_FAILURE([self bindTexturesToCurrentRenderEncoder:&encCtx]);
    }
    return true;
}


@end
