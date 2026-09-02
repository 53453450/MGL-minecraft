/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Thin buffer/texture bind wrappers extracted from MGLRenderer+Binding.m.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Binding_Private.h"
#include "mgl_render.h"

bool mglRendererTextureBindLocked(MGLRenderer *self, Texture *tex);

@implementation MGLRenderer (BindingImplBridge)

- (void) bindMTLBuffer:(Buffer *) ptr
{
    METAL_LOCK();
    [self bindMTLBufferLocked:ptr];
    METAL_UNLOCK();
}

- (void) bindMTLBufferLocked:(Buffer *)ptr
{
    char bindError[256] = {0};
    int bindResult = mglRenderBindBufferStorage(
        ptr, bindError, sizeof(bindError));
    if (bindResult != MGL_RENDER_BUFFER_BOUND) {
        NSLog(@"MGL ERROR: Metal-cpp buffer bind failed buffer=%u: %s",
              ptr ? (unsigned)ptr->name : 0u,
              bindError[0] ? bindError : "?");
    }
}

- (bool)bindMTLTexture:(Texture *)tex
{
    METAL_LOCK();
    bool result = mglRendererTextureBindLocked(self, tex);
    METAL_UNLOCK();
    return result;
}

- (bool)bindMTLTextureLocked:(Texture *)tex
{
    return mglRendererTextureBindLocked(self, tex);
}

@end
