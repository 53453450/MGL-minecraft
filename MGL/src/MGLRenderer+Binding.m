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
// Buffer/texture Metal object binding — implementation in bridge modules.

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Binding_Private.h"
#include "mgl_render.h"

bool mglRendererTextureBindLocked(MGLRenderer *self, Texture *tex);

@implementation MGLRenderer (Binding)

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
