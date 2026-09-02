/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ObjC blit dispatch bridge — entry from C++ facade to MGLRenderer (Blit).
 */

#import "MGLRenderer_Private.h"

@interface MGLRenderer (BlitBridge)
- (void)mtlBlitFramebuffer:(GLMContext)glm_ctx srcX0:(GLint)srcX0
                     srcY0:(GLint)srcY0 srcX1:(GLint)srcX1 srcY1:(GLint)srcY1
                     dstX0:(GLint)dstX0 dstY0:(GLint)dstY0 dstX1:(GLint)dstX1
                     dstY1:(GLint)dstY1 mask:(GLbitfield)mask filter:(GLenum)filter;
@end

void mglRendererObjCBlitFramebuffer(GLMContext glm_ctx, int src_x0, int src_y0,
                                    int src_x1, int src_y1, int dst_x0,
                                    int dst_y0, int dst_x1, int dst_y1,
                                    unsigned int mask, unsigned int filter)
{
    MGLRenderer *renderer = mglRendererForContext(glm_ctx);
    if (!renderer || !glm_ctx) return;
    [renderer mtlBlitFramebuffer:glm_ctx srcX0:src_x0 srcY0:src_y0 srcX1:src_x1
                          srcY1:src_y1 dstX0:dst_x0 dstY0:dst_y0 dstX1:dst_x1
                          dstY1:dst_y1 mask:mask filter:filter];
}
