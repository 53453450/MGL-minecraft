/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ObjC batch dispatch bridge — entry from C++ facade to MGLRenderer (Batch).
 */

#import "MGLRenderer_Private.h"

void mglRendererObjCFlushDrawBuffer(GLMContext glm_ctx)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    @autoreleasepool {
        @try {
            [renderer flushDrawBuffer:glm_ctx];
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: callback flushDrawBuffer exception: %@", exception);
        }
    }
}
