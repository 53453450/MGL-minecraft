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
#include "mgl_render.h"

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
    uint32_t idx = 0u;
    (void)mglRenderDefaultDrawBufferIndex((uint32_t)drawBuffer, &idx);
    return (GLuint)idx;
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
        return (GLenum)mglRenderEmptyDrawBuffer();
    }

    GLsizei count = mglMetalDrawBufferCount(drawCtx);
    if (slot < (GLuint)count) {
        return drawCtx->active_state->draw_buffers[slot];
    }

    return (GLenum)mglRenderEmptyDrawBuffer();
}

BOOL mglMetalResolveFboDrawAttachmentIndex(GLMContext drawCtx,
                                                  GLenum drawBuffer,
                                                  GLuint *attachmentIndex)
{
    if (!drawCtx || mglRenderDrawBufferIsNone((uint32_t)drawBuffer)) {
        return NO;
    }

    uint32_t att = 0u;
    uint32_t maxAtt = (uint32_t)drawCtx->active_state->max_color_attachments;
    if (mglRenderDrawBufferIsColorAttachment((uint32_t)drawBuffer, maxAtt, &att)) {
        if (attachmentIndex) {
            *attachmentIndex = att;
        }
        return YES;
    }

    if (mglRenderDrawBufferIsDefaultFBOCompat((uint32_t)drawBuffer)) {
        if (attachmentIndex) {
            *attachmentIndex = 0u;
        }
        return YES;
    }
    return NO;
}

GLuint mglMetalColorSlotForDrawBuffer(GLMContext drawCtx, GLuint drawBufferSlot)
{
    if (!drawCtx || drawCtx->active_state->draw_buffer_count == 1u) {
        return 0u;
    }
    return drawBufferSlot;
}
