#include <stdbool.h>
#include <stddef.h>
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

#ifdef __OBJC__
#endif

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The GL-side vocabulary for draw/read buffers, as small inline helpers: the
 * backend asks "which attachment is this?" and "is this the default buffer?"
 * without doing GL enum arithmetic of its own. */
static inline bool mglDrawBufferIsNone(uint32_t e) { return e == GL_NONE; }
static inline bool mglDrawBufferIsFrontAndBack(uint32_t e)
{
    return e == GL_FRONT_AND_BACK;
}
static inline uint32_t mglDrawBufferNoneEnum(void) { return GL_NONE; }
static inline uint32_t mglDrawBufferFrontEnum(void) { return GL_FRONT; }
static inline uint32_t mglDrawBufferColorAttachmentEnum(uint32_t index)
{
    return GL_COLOR_ATTACHMENT0 + index;
}
/* Index of a GL_COLOR_ATTACHMENTi enum; false when `e` is below the base. */
static inline bool mglDrawBufferColorAttachmentIndex(uint32_t e, uint32_t *index)
{
    if (e < GL_COLOR_ATTACHMENT0) {
        return false;
    }
    if (index) {
        *index = e - GL_COLOR_ATTACHMENT0;
    }
    return true;
}
/* Read buffer of the default framebuffer -> canonical slot: FRONT/BACK -> 0,
 * *LEFT -> 2, *RIGHT -> 3.  False for anything else. */
static inline bool mglDrawBufferDefaultSlot(uint32_t e, uint32_t *out)
{
    switch (e) {
        case GL_FRONT:
        case GL_BACK: *out = 0u; return true;
        case GL_FRONT_LEFT:
        case GL_BACK_LEFT:
        case GL_LEFT: *out = 2u; return true;
        case GL_FRONT_RIGHT:
        case GL_BACK_RIGHT:
        case GL_RIGHT: *out = 3u; return true;
        default: *out = 0u; return false;
    }
}
GLuint mglDefaultDrawBufferIndexForGL(GLenum drawBuffer);
GLsizei mglMetalDrawBufferCount(GLMContext drawCtx);
GLenum mglMetalDrawBufferAt(GLMContext drawCtx, GLuint slot);
bool mglMetalResolveFboDrawAttachmentIndex(GLMContext drawCtx,
                                           GLenum drawBuffer,
                                           GLuint *attachmentIndex);
GLuint mglMetalColorSlotForDrawBuffer(GLMContext drawCtx, GLuint drawBufferSlot);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DRAW_BUFFER_H */
