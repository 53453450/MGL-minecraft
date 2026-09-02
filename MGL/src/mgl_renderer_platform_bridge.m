/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ObjC platform dispatch bridge — swap/clear entry from C++ facade.
 */

#import "MGLRenderer_Private.h"

@interface MGLRenderer (PlatformBridge)
- (void)mtlSwapBuffers:(GLMContext)glm_ctx;
- (void)mtlClearBuffer:(GLMContext)glm_ctx type:(GLuint)type mask:(GLbitfield)mask;
@end

void mglRendererObjCSwapBuffers(GLMContext glm_ctx)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    @autoreleasepool {
        @try {
            [renderer mtlSwapBuffers:glm_ctx];
        } @catch (NSException *exception) {
            NSLog(@"MGL CRITICAL: callback swap exception: %@", exception);
        }
    }
}

void mglRendererObjCClearBuffer(GLMContext glm_ctx, unsigned int type,
                                unsigned int mask)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlClearBuffer:glm_ctx type:type mask:mask];
}
