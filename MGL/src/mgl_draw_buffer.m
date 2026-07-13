/*
 * mgl_draw_buffer.m
 * MGL
 *
 * Implementation of the Draw Buffer Mapping Subsystem.
 * See mgl_draw_buffer.h for the API contract.
 *
 * Function bodies are preserved verbatim from MGLRenderer.m; only the
 * "static" storage-class qualifier was removed to make the symbols
 * externally visible.
 */

#import "mgl_draw_buffer.h"

/* Local draw-buffer slot indices (mirrors the enum in MGLRenderer.m used by
 * the default-draw-buffer lookup).  Enumerators have no linkage, so defining
 * them here does not conflict with the definition in MGLRenderer.m. */
enum {
    _FRONT,
    _BACK,
    _FRONT_LEFT,
    _FRONT_RIGHT,
    _BACK_LEFT,
    _BACK_RIGHT,
    _MAX_DRAW_BUFFERS
};

GLuint mglDefaultDrawBufferIndexForGL(GLenum drawBuffer)
{
    switch (drawBuffer)
    {
        case GL_FRONT: return _FRONT;
        case GL_BACK: return _FRONT;
        case GL_FRONT_LEFT: return _FRONT_LEFT;
        case GL_FRONT_RIGHT: return _FRONT_RIGHT;
        case GL_BACK_LEFT: return _FRONT_LEFT;
        case GL_BACK_RIGHT: return _FRONT_RIGHT;
        case GL_LEFT: return _FRONT_LEFT;
        case GL_RIGHT: return _FRONT_RIGHT;
        case GL_FRONT_AND_BACK: return _FRONT;
        case GL_COLOR_ATTACHMENT0: return _FRONT;
        case GL_NONE: return _FRONT;
        default: return _FRONT;
    }
}

GLsizei mglMetalDrawBufferCount(GLMContext drawCtx)
{
    if (!drawCtx || drawCtx->active_state->draw_buffer_count <= 0) {
        return 0;
    }
    if (drawCtx->active_state->draw_buffer_count > (GLsizei)MAX_COLOR_ATTACHMENTS) {
        return MAX_COLOR_ATTACHMENTS;
    }
    return drawCtx->active_state->draw_buffer_count;
}

GLenum mglMetalDrawBufferAt(GLMContext drawCtx, GLuint slot)
{
    if (!drawCtx) {
        return GL_NONE;
    }

    GLsizei count = mglMetalDrawBufferCount(drawCtx);
    if (slot < (GLuint)count) {
        return drawCtx->active_state->draw_buffers[slot];
    }

    return GL_NONE;
}

BOOL mglMetalResolveFboDrawAttachmentIndex(GLMContext drawCtx,
                                                  GLenum drawBuffer,
                                                  GLuint *attachmentIndex)
{
    if (!drawCtx || drawBuffer == GL_NONE) {
        return NO;
    }

    if (drawBuffer >= GL_COLOR_ATTACHMENT0 &&
        drawBuffer < (GL_COLOR_ATTACHMENT0 + drawCtx->active_state->max_color_attachments) &&
        drawBuffer < (GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS)) {
        if (attachmentIndex) {
            *attachmentIndex = (GLuint)(drawBuffer - GL_COLOR_ATTACHMENT0);
        }
        return YES;
    }

    switch (drawBuffer) {
        case GL_FRONT:
        case GL_BACK:
        case GL_FRONT_LEFT:
        case GL_FRONT_RIGHT:
        case GL_BACK_LEFT:
        case GL_BACK_RIGHT:
        case GL_LEFT:
        case GL_RIGHT:
        case GL_FRONT_AND_BACK:
            if (attachmentIndex) {
                *attachmentIndex = 0u;
            }
            return YES;

        default:
            return NO;
    }
}

GLuint mglMetalColorSlotForDrawBuffer(GLMContext drawCtx, GLuint drawBufferSlot)
{
    if (!drawCtx || drawCtx->active_state->draw_buffer_count == 1u) {
        return 0u;
    }
    return drawBufferSlot;
}
