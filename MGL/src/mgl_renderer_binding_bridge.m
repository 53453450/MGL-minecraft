/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ObjC binding dispatch bridge — entry from C++ facade to MGLRenderer (Binding).
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#include "mgl_renderer_sync.h"

bool mglRendererTextureBindLocked(MGLRenderer *self, Texture *tex);

void mglRendererObjCBindTexture(GLMContext glm_ctx, Texture *texture)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx || !texture) return;
    (void)mglRendererTextureBindLocked(renderer, texture);
}

bool mglRendererObjCSyncResourceBindings(GLMContext glm_ctx,
                                         const MGLResourceSyncWork *done)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer) {
        return false;
    }
    return [renderer syncResourceBindingsForContext:glm_ctx alreadyDone:done];
}
