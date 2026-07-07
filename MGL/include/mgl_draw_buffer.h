/*
 * mgl_draw_buffer.h
 * MGL
 *
 * Draw Buffer Mapping Subsystem.
 *
 * Maps GL draw-buffer enums (GL_FRONT, GL_BACK, GL_COLOR_ATTACHMENT0+n, …)
 * to Metal color attachment slots.  Used by render-pass setup and
 * clear/framebuffer-query paths.
 *
 * All functions take GLMContext as a parameter (reading ctx->state.* only).
 *
 * Dependencies: glm_context.h (GLMContext, MAX_COLOR_ATTACHMENTS) +
 * glcorearb.h (GL enums).
 */

#ifndef MGL_DRAW_BUFFER_H
#define MGL_DRAW_BUFFER_H

#include "glcorearb.h"

#include <objc/objc.h>   /* BOOL */

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#endif

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

GLuint mglDefaultDrawBufferIndexForGL(GLenum drawBuffer);
GLsizei mglMetalDrawBufferCount(GLMContext drawCtx);
GLenum mglMetalDrawBufferAt(GLMContext drawCtx, GLuint slot);
BOOL mglMetalResolveFboDrawAttachmentIndex(GLMContext drawCtx,
                                           GLenum drawBuffer,
                                           GLuint *attachmentIndex);
GLuint mglMetalColorSlotForDrawBuffer(GLMContext drawCtx, GLuint drawBufferSlot);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_BUFFER_H */
