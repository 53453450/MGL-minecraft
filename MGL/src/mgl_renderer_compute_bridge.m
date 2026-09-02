/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ObjC compute dispatch bridge — entry from C++ facade to MGLRenderer (Compute).
 */

#import "MGLRenderer_Private.h"

@interface MGLRenderer (ComputeBridge)
- (void)mtlDispatchComputeLocked:(GLMContext)glm_ctx groupsX:(GLuint)groups_x
                         groupsY:(GLuint)groups_y groupsZ:(GLuint)groups_z;
- (void)mtlDispatchComputeIndirectLocked:(GLMContext)glm_ctx
                                indirect:(GLintptr)indirect;
@end

void mglRendererObjCDispatchCompute(GLMContext glm_ctx, unsigned int groups_x,
                                    unsigned int groups_y, unsigned int groups_z)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    METAL_LOCK();
    [renderer mtlDispatchComputeLocked:glm_ctx groupsX:groups_x groupsY:groups_y
                                 groupsZ:groups_z];
    METAL_UNLOCK();
}

void mglRendererObjCDispatchComputeIndirect(GLMContext glm_ctx,
                                            intptr_t indirect)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    METAL_LOCK();
    [renderer mtlDispatchComputeIndirectLocked:glm_ctx indirect:indirect];
    METAL_UNLOCK();
}
