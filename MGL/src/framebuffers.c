/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * framebuffers.c
 * MGL
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "glm_context.h"
#include "draw_command.h"
#include "mgl_safety.h"
#include "pixel_utils.h"
#include "utils.h"

#define RENDBUF_STATE(_val_)    ctx->state.renderbuffer->_val_

#ifndef MGL_VERBOSE_FBO_LOGS
#define MGL_VERBOSE_FBO_LOGS 0
#endif

extern GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);
extern Texture *newTexObj(GLMContext ctx, GLenum target);
extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern Texture *newTexture(GLMContext ctx, GLenum target, GLuint texture);
extern void mglClearLastSampled2DTextureIfMatches(GLMContext ctx, Texture *tex);
bool isCubeMapTarget(GLMContext ctx, GLuint textarget);
void mglNamedFramebufferDrawBuffers(GLMContext ctx, GLuint framebuffer, GLsizei n, const GLenum *bufs);
extern void mglClearBufferiv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLint *value);
extern void mglClearBufferuiv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLuint *value);
extern void mglClearBufferfv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLfloat *value);
extern void mglClearBufferfi(GLMContext ctx, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil);
extern void mglTraceLogExternal(const char *fmt, ...);

static GLubyte mglClampClearComponentToByte(GLfloat value)
{
    if (value != value || value <= 0.0f) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    return (GLubyte)(value * 255.0f + 0.5f);
}

static GLuint mglTraceSafeFramebufferName(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx || !fbo ||
        !mglObjectPointerLooksPlausible(fbo) ||
        !mglHashTableContainsData(&ctx->state.framebuffer_table, fbo) ||
        !mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
        return 0u;
    }
    return fbo->name;
}

static Texture *mglTraceAttachmentTexture(GLMContext ctx, FBOAttachment *att)
{
    if (!ctx || !att) {
        return NULL;
    }
    if (att->textarget == GL_RENDERBUFFER) {
        return (att->buf.rbo && att->buf.rbo->tex) ? att->buf.rbo->tex : NULL;
    }
    if (att->buf.tex) {
        return att->buf.tex;
    }
    if (att->texture != 0u) {
        return findTexture(ctx, att->texture);
    }
    return NULL;
}

static void mglTraceTextureInitSummary(Texture *tex,
                                       GLuint level,
                                       GLuint *ever,
                                       GLuint *full,
                                       GLuint *source)
{
    TextureLevel *tex_level = NULL;
    if (tex && tex->num_levels > 0u && tex->faces[0].levels && level < tex->num_levels) {
        tex_level = &tex->faces[0].levels[level];
    }
    if (ever) {
        *ever = tex_level ? (GLuint)tex_level->ever_written : 0u;
    }
    if (full) {
        *full = tex_level ? (GLuint)tex_level->has_initialized_data : 0u;
    }
    if (source) {
        *source = tex_level ? (GLuint)tex_level->last_init_source : 0u;
    }
}

static Texture *mglPendingClearAttachmentTexture(FBOAttachment *att)
{
    if (!att) {
        return NULL;
    }
    if (att->textarget == GL_RENDERBUFFER) {
        return (att->buf.rbo && att->buf.rbo->tex) ? att->buf.rbo->tex : NULL;
    }
    return att->buf.tex;
}

static GLboolean mglAttachmentChangeNeedsPendingClearFlush(FBOAttachment *att,
                                                           GLuint texture,
                                                           GLenum textarget,
                                                           GLint level,
                                                           GLint layer)
{
    if (!att ||
        !(att->clear_bitmask & (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) ||
        !mglPendingClearAttachmentTexture(att)) {
        return GL_FALSE;
    }

    return att->texture != texture ||
           att->textarget != textarget ||
           att->level != (GLuint)level ||
           att->layer != (GLuint)layer;
}

static GLboolean mglEnsureTextureLevelStorage(TextureLevel *level, size_t required_size, size_t row_pitch)
{
    if (!level || required_size == 0u || row_pitch == 0u) {
        return GL_FALSE;
    }

    if (level->data && level->data_size >= required_size && level->pitch >= row_pitch) {
        return GL_TRUE;
    }

    void *storage = calloc(1u, required_size);
    if (!storage) {
        return GL_FALSE;
    }

    if (level->data) {
        free((void *)level->data);
    }

    level->data = (vm_address_t)storage;
    level->data_size = required_size;
    level->pitch = row_pitch;
    return GL_TRUE;
}

static GLboolean mglFlushPendingColorClearToTexture(GLMContext ctx, FBOAttachment *att, const char *reason)
{
    Texture *tex;
    TextureLevel *level;
    GLuint face;
    size_t row_bytes;
    size_t row_pitch;
    size_t required_size;
    MTLPixelFormat mtl_format;
    GLubyte r, g, b, a;
    GLubyte pixel[4];
    GLboolean bgra;

    GLbitfield pendingMask =
        att ? (att->clear_bitmask & (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) : 0u;
    Texture *attachedTexture = mglPendingClearAttachmentTexture(att);
    if (!ctx || !att || pendingMask == 0u || !attachedTexture) {
        return GL_FALSE;
    }

    if (!ctx->state.caps.scissor_test && ctx->mtl_funcs.mtlClearBuffer) {
        static unsigned s_immediate_flush_logs = 0u;
        if (s_immediate_flush_logs < 64u) {
            fprintf(stderr,
                    "MGL TRACE fbo.clear.flush.immediate tex=%u mask=0x%x attachment-change reason=%s\n",
                    attachedTexture ? attachedTexture->name : 0u,
                    pendingMask,
                    reason ? reason : "unknown");
            s_immediate_flush_logs++;
        }

        ctx->mtl_funcs.mtlClearBuffer(ctx, 0, pendingMask);
        if (!(att->clear_bitmask & pendingMask)) {
            return GL_TRUE;
        }
    }

    if (!(att->clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        return GL_FALSE;
    }

    tex = attachedTexture;
    face = isCubeMapTarget(ctx, att->textarget) ? textureIndexFromTarget(ctx, att->textarget) : 0u;
    if (face >= _CUBE_MAP_MAX_FACE ||
        att->level >= tex->mipmap_levels ||
        !tex->faces[face].levels) {
        fprintf(stderr,
                "MGL WARNING: fbo.clear.flush skipped invalid target tex=%u face=%u level=%u mipLevels=%u reason=%s\n",
                tex->name,
                face,
                att->level,
                tex->mipmap_levels,
                reason ? reason : "unknown");
        return GL_FALSE;
    }

    level = &tex->faces[face].levels[att->level];
    if (!level->complete || level->width == 0u || level->height == 0u) {
        fprintf(stderr,
                "MGL WARNING: fbo.clear.flush skipped incomplete level tex=%u face=%u level=%u complete=%d size=%ux%u reason=%s\n",
                tex->name,
                face,
                att->level,
                level->complete,
                level->width,
                level->height,
                reason ? reason : "unknown");
        return GL_FALSE;
    }

    row_bytes = (size_t)level->width * 4u;
    row_pitch = level->pitch >= row_bytes ? level->pitch : row_bytes;
    required_size = row_pitch * (size_t)level->height;
    if (!mglEnsureTextureLevelStorage(level, required_size, row_pitch)) {
        fprintf(stderr,
                "MGL WARNING: fbo.clear.flush could not allocate tex=%u face=%u level=%u bytes=%zu reason=%s\n",
                tex->name,
                face,
                att->level,
                required_size,
                reason ? reason : "unknown");
        return GL_FALSE;
    }

    r = mglClampClearComponentToByte(att->clear_color[0]);
    g = mglClampClearComponentToByte(att->clear_color[1]);
    b = mglClampClearComponentToByte(att->clear_color[2]);
    a = mglClampClearComponentToByte(att->clear_color[3]);

    mtl_format = mtlFormatForGLInternalFormat(tex->internalformat);
    bgra = (mtl_format == MTLPixelFormatBGRA8Unorm ||
            mtl_format == MTLPixelFormatBGRA8Unorm_sRGB);

    pixel[0] = bgra ? b : r;
    pixel[1] = g;
    pixel[2] = bgra ? r : b;
    pixel[3] = a;

    for (GLuint y = 0; y < level->height; y++) {
        GLubyte *row = (GLubyte *)level->data + ((size_t)y * level->pitch);
        for (GLuint x = 0; x < level->width; x++) {
            memcpy(row + ((size_t)x * 4u), pixel, sizeof(pixel));
        }
    }

    level->has_initialized_data = GL_TRUE;
    level->ever_written = GL_TRUE;
    level->suspicious_zero_upload = GL_FALSE;
    level->last_init_source = kTexMetalFill;
    level->last_upload_size = required_size;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;

    tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    STATE(dirty_bits) |= DIRTY_TEX | DIRTY_FBO;
    att->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;

    static unsigned s_flush_logs = 0u;
    if (s_flush_logs < 64u) {
        fprintf(stderr,
                "MGL TRACE fbo.clear.flush tex=%u face=%u level=%u size=%ux%u rgba=(%.3f,%.3f,%.3f,%.3f) byteOrder=%s reason=%s\n",
                tex->name,
                face,
                att->level,
                level->width,
                level->height,
                att->clear_color[0],
                att->clear_color[1],
                att->clear_color[2],
                att->clear_color[3],
                bgra ? "BGRA" : "RGBA",
                reason ? reason : "unknown");
        s_flush_logs++;
    }

    return GL_TRUE;
}

#pragma mark renderbuffer logic

static Renderbuffer *newRenderbuffer(GLMContext ctx, GLuint renderbuffer)
{
    Renderbuffer *ptr;

    ptr = (Renderbuffer *)malloc(sizeof(Renderbuffer));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate renderbuffer %u\n", renderbuffer);
        return NULL;
    }

    bzero(ptr, sizeof(Renderbuffer));

    ptr->name = renderbuffer;

    return ptr;
}

static Renderbuffer *getRenderbuffer(GLMContext ctx, GLuint renderbuffer)
{
    Renderbuffer *ptr;

    ptr = (Renderbuffer *)searchHashTable(&STATE(renderbuffer_table), renderbuffer);

    if (!ptr)
    {
        ptr = newRenderbuffer(ctx, renderbuffer);
        if (!ptr)
            return NULL;

        insertHashElement(&STATE(renderbuffer_table), renderbuffer, ptr);
    }

    return ptr;
}

static int isRenderBuffer(GLMContext ctx, GLuint renderbuffer)
{
    Renderbuffer *ptr;

    ptr = (Renderbuffer *)searchHashTable(&STATE(renderbuffer_table), renderbuffer);

    if (ptr)
        return 1;

    return 0;
}

Renderbuffer *findRenderbuffer(GLMContext ctx, GLuint renderbuffer)
{
    Renderbuffer *ptr;

    ptr = (Renderbuffer *)searchHashTable(&STATE(renderbuffer_table), renderbuffer);

    return ptr;
}

Framebuffer *currentFBOForType(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_FRAMEBUFFER:
        case GL_DRAW_FRAMEBUFFER:
            return ctx->state.framebuffer;
            break;

        case GL_READ_FRAMEBUFFER:
            return ctx->state.readbuffer;
            break;

        default:
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, NULL);
    }
}

#pragma mark framebuffer logic
static Framebuffer *newFramebuffer(GLMContext ctx, GLuint framebuffer)
{
    Framebuffer *ptr;

    ptr = (Framebuffer *)malloc(sizeof(Framebuffer));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate framebuffer %u\n", framebuffer);
        return NULL;
    }

    bzero(ptr, sizeof(Framebuffer));

    ptr->name = framebuffer;
    ptr->draw_buffer = GL_COLOR_ATTACHMENT0;
    ptr->draw_buffer_count = 1;
    ptr->draw_buffers[0] = GL_COLOR_ATTACHMENT0;
    for (GLuint i = 1; i < MAX_COLOR_ATTACHMENTS; ++i) {
        ptr->draw_buffers[i] = GL_NONE;
    }
    ptr->read_buffer = GL_COLOR_ATTACHMENT0;

    return ptr;
}

static Framebuffer *getFramebuffer(GLMContext ctx, GLuint framebuffer)
{
    Framebuffer *ptr;

    ptr = (Framebuffer *)searchHashTable(&STATE(framebuffer_table), framebuffer);

    if (!ptr)
    {
        ptr = newFramebuffer(ctx, framebuffer);
        if (!ptr)
            return NULL;

        insertHashElement(&STATE(framebuffer_table), framebuffer, ptr);
    }

    return ptr;
}

static int isFramebuffer(GLMContext ctx, GLuint framebuffer)
{
    Framebuffer *ptr;

    ptr = (Framebuffer *)searchHashTable(&STATE(framebuffer_table), framebuffer);

    if (ptr)
        return 1;

    return 0;
}

Framebuffer *findFrameBuffer(GLMContext ctx, GLuint framebuffer)
{
    Framebuffer *ptr;

    ptr = (Framebuffer *)searchHashTable(&STATE(framebuffer_table), framebuffer);

    return ptr;
}

GLboolean mglFramebufferPrimaryColorSize(GLMContext ctx, Framebuffer *fbo, GLuint *outWidth, GLuint *outHeight)
{
    FBOAttachment *color0;
    Texture *tex;

    if (outWidth) {
        *outWidth = 0u;
    }
    if (outHeight) {
        *outHeight = 0u;
    }
    if (!ctx || !fbo || !outWidth || !outHeight) {
        return GL_FALSE;
    }

    if ((fbo->color_attachment_bitfield & 1u) == 0u) {
        return GL_FALSE;
    }

    color0 = &fbo->color_attachments[0];
    tex = NULL;
    if (color0->textarget == GL_RENDERBUFFER && color0->buf.rbo) {
        tex = color0->buf.rbo->tex;
    } else {
        tex = color0->buf.tex;
        if (!tex && color0->texture != 0u) {
            tex = findTexture(ctx, color0->texture);
        }
    }

    if (!tex || tex->width == 0u || tex->height == 0u) {
        return GL_FALSE;
    }

    *outWidth = tex->width;
    *outHeight = tex->height;
    return GL_TRUE;
}

void mglSetViewportToFramebufferSize(GLMContext ctx, Framebuffer *fbo)
{
    GLuint width = 0u;
    GLuint height = 0u;

    if (!ctx) {
        return;
    }

    if (fbo && mglFramebufferPrimaryColorSize(ctx, fbo, &width, &height)) {
        ctx->state.viewport[0] = 0;
        ctx->state.viewport[1] = 0;
        ctx->state.viewport[2] = (GLint)width;
        ctx->state.viewport[3] = (GLint)height;
        ctx->state.viewport_array[0][0] = 0.0f;
        ctx->state.viewport_array[0][1] = 0.0f;
        ctx->state.viewport_array[0][2] = (GLfloat)width;
        ctx->state.viewport_array[0][3] = (GLfloat)height;
    }

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

static GLboolean mglFramebufferBufferIsColorAttachment(GLMContext ctx, GLenum buffer)
{
    return ctx &&
           buffer >= GL_COLOR_ATTACHMENT0 &&
           buffer < (GL_COLOR_ATTACHMENT0 + ctx->state.max_color_attachments) &&
           buffer < (GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS);
}

static GLuint mglFramebufferMaxDrawBuffers(GLMContext ctx)
{
    GLuint maxDrawBuffers = ctx ? ctx->state.var.max_draw_buffers : 0u;
    if (maxDrawBuffers == 0u || maxDrawBuffers > MAX_COLOR_ATTACHMENTS) {
        maxDrawBuffers = MAX_COLOR_ATTACHMENTS;
    }
    return maxDrawBuffers;
}

static void mglFramebufferSetSingleDrawBuffer(GLMContext ctx, GLenum buffer)
{
    ctx->state.draw_buffer = buffer;
    ctx->state.draw_buffer_count = 1;
    ctx->state.draw_buffers[0] = buffer;
    for (GLuint i = 1; i < MAX_COLOR_ATTACHMENTS; ++i) {
        ctx->state.draw_buffers[i] = GL_NONE;
    }
}

static void mglFramebufferStoreCurrentDrawBuffer(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx) {
        return;
    }

    GLenum *drawBuffers = fbo ? fbo->draw_buffers : ctx->state.default_draw_buffers;
    GLsizei *drawBufferCount = fbo ? &fbo->draw_buffer_count : &ctx->state.default_draw_buffer_count;
    GLuint *drawBuffer = fbo ? &fbo->draw_buffer : &ctx->state.default_draw_buffer;

    *drawBuffer = ctx->state.draw_buffer;
    *drawBufferCount = ctx->state.draw_buffer_count;
    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; ++i) {
        drawBuffers[i] = ctx->state.draw_buffers[i];
    }
}

static void mglFramebufferStoreCurrentReadBuffer(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx) {
        return;
    }

    if (fbo) {
        fbo->read_buffer = ctx->state.read_buffer;
    } else {
        ctx->state.default_read_buffer = ctx->state.read_buffer;
    }
}

static void mglFramebufferLoadDrawBuffer(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx) {
        return;
    }

    const GLenum *drawBuffers = fbo ? fbo->draw_buffers : ctx->state.default_draw_buffers;
    GLsizei drawBufferCount = fbo ? fbo->draw_buffer_count : ctx->state.default_draw_buffer_count;
    GLenum drawBuffer = fbo ? fbo->draw_buffer : ctx->state.default_draw_buffer;

    if (drawBufferCount <= 0 || drawBufferCount > (GLsizei)MAX_COLOR_ATTACHMENTS) {
        drawBufferCount = 1;
        drawBuffer = fbo ? GL_COLOR_ATTACHMENT0 : GL_FRONT;
    }

    ctx->state.draw_buffer = drawBuffer;
    ctx->state.draw_buffer_count = drawBufferCount;
    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; ++i) {
        ctx->state.draw_buffers[i] = (i < (GLuint)drawBufferCount) ? drawBuffers[i] : GL_NONE;
    }
}

static void mglFramebufferLoadReadBuffer(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx) {
        return;
    }

    GLenum readBuffer = fbo ? fbo->read_buffer : ctx->state.default_read_buffer;
    if (readBuffer == 0u) {
        readBuffer = fbo ? GL_COLOR_ATTACHMENT0 : GL_FRONT;
    }
    ctx->state.read_buffer = readBuffer;
}

static void mglFramebufferSyncBindingNames(GLMContext ctx)
{
    if (!ctx) {
        return;
    }

    ctx->state.var.draw_framebuffer_binding =
        ctx->state.framebuffer ? ctx->state.framebuffer->name : 0u;
    ctx->state.var.read_framebuffer_binding =
        ctx->state.readbuffer ? ctx->state.readbuffer->name : 0u;
}

static void mglFramebufferUseTemporaryDrawBinding(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx) {
        return;
    }

    ctx->state.framebuffer = fbo;
    mglFramebufferSyncBindingNames(ctx);
    mglFramebufferLoadDrawBuffer(ctx, fbo);
}

static void mglFramebufferUseTemporaryReadDrawBinding(GLMContext ctx,
                                                      Framebuffer *readFbo,
                                                      Framebuffer *drawFbo)
{
    if (!ctx) {
        return;
    }

    ctx->state.readbuffer = readFbo;
    ctx->state.framebuffer = drawFbo;
    mglFramebufferSyncBindingNames(ctx);
    mglFramebufferLoadReadBuffer(ctx, readFbo);
    mglFramebufferLoadDrawBuffer(ctx, drawFbo);
}

void mglAssignDrawFramebuffer(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx) {
        return;
    }

    ctx->state.framebuffer = fbo;
    mglFramebufferSyncBindingNames(ctx);
    mglFramebufferLoadDrawBuffer(ctx, fbo);
    if (fbo) {
        fbo->dirty_bits |= DIRTY_FBO_BINDING;
    }
    ctx->state.dirty_bits |= DIRTY_FBO | DIRTY_STATE | DIRTY_RENDER_STATE;
}

typedef struct MGLFramebufferBindingSnapshot_t {
    Framebuffer *draw_fbo;
    Framebuffer *read_fbo;
    GLuint draw_buffer;
    GLsizei draw_buffer_count;
    GLenum draw_buffers[MAX_COLOR_ATTACHMENTS];
    GLuint read_buffer;
} MGLFramebufferBindingSnapshot;

static void mglFramebufferCaptureBindingSnapshot(GLMContext ctx, MGLFramebufferBindingSnapshot *snapshot)
{
    if (!ctx || !snapshot) {
        return;
    }

    snapshot->draw_fbo = ctx->state.framebuffer;
    snapshot->read_fbo = ctx->state.readbuffer;
    snapshot->draw_buffer = ctx->state.draw_buffer;
    snapshot->draw_buffer_count = ctx->state.draw_buffer_count;
    snapshot->read_buffer = ctx->state.read_buffer;
    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; ++i) {
        snapshot->draw_buffers[i] = ctx->state.draw_buffers[i];
    }
}

static void mglFramebufferRestoreBindingSnapshot(GLMContext ctx, const MGLFramebufferBindingSnapshot *snapshot)
{
    if (!ctx || !snapshot) {
        return;
    }

    ctx->state.framebuffer = snapshot->draw_fbo;
    ctx->state.readbuffer = snapshot->read_fbo;
    mglFramebufferSyncBindingNames(ctx);
    ctx->state.draw_buffer = snapshot->draw_buffer;
    ctx->state.draw_buffer_count = snapshot->draw_buffer_count;
    ctx->state.read_buffer = snapshot->read_buffer;
    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; ++i) {
        ctx->state.draw_buffers[i] = snapshot->draw_buffers[i];
    }
}

static void mglNormalizeDrawBufferForFramebufferBinding(GLMContext ctx, Framebuffer *fbo)
{
    GLboolean drawBufferIsColorAttachment =
        mglFramebufferBufferIsColorAttachment(ctx, ctx->state.draw_buffer);

    if (fbo && !drawBufferIsColorAttachment && ctx->state.draw_buffer != GL_NONE) {
        mglFramebufferSetSingleDrawBuffer(ctx, GL_COLOR_ATTACHMENT0);
    } else if (!fbo && drawBufferIsColorAttachment) {
        mglFramebufferSetSingleDrawBuffer(ctx, GL_FRONT);
    }

    mglFramebufferStoreCurrentDrawBuffer(ctx, fbo);
}

static void mglNormalizeReadBufferForFramebufferBinding(GLMContext ctx, Framebuffer *fbo)
{
    GLboolean readBufferIsColorAttachment =
        mglFramebufferBufferIsColorAttachment(ctx, ctx->state.read_buffer);

    if (fbo && !readBufferIsColorAttachment && ctx->state.read_buffer != GL_NONE) {
        ctx->state.read_buffer = GL_COLOR_ATTACHMENT0;
    } else if (!fbo && readBufferIsColorAttachment) {
        ctx->state.read_buffer = GL_FRONT;
    }

    mglFramebufferStoreCurrentReadBuffer(ctx, fbo);
}

#pragma mark Framebuffer calls
GLboolean mglIsFramebuffer(GLMContext ctx, GLuint framebuffer)
{
    return isFramebuffer(ctx, framebuffer);
}

void mglGenFramebuffers(GLMContext ctx, GLsizei n, GLuint *framebuffers)
{
    if (!ctx)
        return;

    if (n < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (n == 0)
        return;

    if (!framebuffers) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    while(n--)
    {
        *framebuffers++ = getNewName(&STATE(framebuffer_table));
    }
}

void mglBindFramebuffer(GLMContext ctx, GLenum target, GLuint framebuffer)
{
    Framebuffer *ptr;
    Framebuffer *oldDrawFbo;
    Framebuffer *oldReadFbo;
    GLboolean drawTargetChanged = GL_FALSE;
    GLboolean readTargetChanged = GL_FALSE;

    if (!ctx)
        return;

    switch(target) {
        case GL_DRAW_FRAMEBUFFER:
        case GL_READ_FRAMEBUFFER:
        case GL_FRAMEBUFFER:
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    oldDrawFbo = ctx->state.framebuffer;
    oldReadFbo = ctx->state.readbuffer;

    VertexArray *currentVAO = ctx->state.vao;
    if (currentVAO &&
        (!mglObjectPointerLooksPlausible(currentVAO) ||
         !mglHashTableContainsData(&ctx->state.vao_table, currentVAO) ||
         !mglPointerRangeIsReadable(currentVAO, sizeof(*currentVAO))))
    {
        fprintf(stderr, "MGL WARNING: VAO pointer polluted before BindFramebuffer: vao=%p not in sane VAO table, resetting\n",
                (void *)currentVAO);
        ctx->state.vao = NULL;
        ctx->state.buffers[_ELEMENT_ARRAY_BUFFER] = ctx->state.default_vao_element_array_buffer;
        ctx->state.var.element_array_buffer_binding =
            ctx->state.default_vao_element_array_buffer ? ctx->state.default_vao_element_array_buffer->name : 0;
        ctx->state.dirty_bits |= DIRTY_VAO;
        currentVAO = NULL;
    }
    else if (currentVAO && currentVAO->magic != MGL_VAO_MAGIC)
    {
        fprintf(stderr, "MGL WARNING: VAO pointer polluted before BindFramebuffer: vao=%p magic=0x%x, resetting\n",
                (void *)currentVAO,
                currentVAO->magic);
        ctx->state.vao = NULL;
        ctx->state.buffers[_ELEMENT_ARRAY_BUFFER] = ctx->state.default_vao_element_array_buffer;
        ctx->state.var.element_array_buffer_binding =
            ctx->state.default_vao_element_array_buffer ? ctx->state.default_vao_element_array_buffer->name : 0;
        ctx->state.dirty_bits |= DIRTY_VAO;
    }

    if(framebuffer)
    {
        ptr = getFramebuffer(ctx, framebuffer);
        if (!ptr)
            return;
        if (MGL_VERBOSE_FBO_LOGS) {
            fprintf(stderr, "MGL: glBindFramebuffer target=%x fbo=%u ptr=%p\n", target, framebuffer, ptr);
        }
    }
    else
    {
        ptr = NULL;
        if (MGL_VERBOSE_FBO_LOGS) {
            fprintf(stderr, "MGL: glBindFramebuffer target=%x fbo=0 (default framebuffer)\n", target);
        }
    }

    if ((target == GL_DRAW_FRAMEBUFFER || target == GL_FRAMEBUFFER) &&
        oldDrawFbo != ptr)
    {
        mglFlushPendingDraws(ctx);
    }

    if ((target == GL_DRAW_FRAMEBUFFER || target == GL_FRAMEBUFFER) &&
        oldDrawFbo != ptr &&
        ctx->mtl_funcs.mtlInvalidateRenderPass)
    {
        ctx->mtl_funcs.mtlInvalidateRenderPass(ctx);
    }

    switch(target) {
        case GL_DRAW_FRAMEBUFFER:
            mglFramebufferStoreCurrentDrawBuffer(ctx, oldDrawFbo);
            ctx->state.framebuffer = ptr;
            mglFramebufferSyncBindingNames(ctx);
            mglFramebufferLoadDrawBuffer(ctx, ptr);
            drawTargetChanged = GL_TRUE;
            break;

        case GL_READ_FRAMEBUFFER:
            mglFramebufferStoreCurrentReadBuffer(ctx, oldReadFbo);
            ctx->state.readbuffer = ptr;
            mglFramebufferSyncBindingNames(ctx);
            mglFramebufferLoadReadBuffer(ctx, ptr);
            readTargetChanged = GL_TRUE;
            break;

        case GL_FRAMEBUFFER:
            mglFramebufferStoreCurrentDrawBuffer(ctx, oldDrawFbo);
            mglFramebufferStoreCurrentReadBuffer(ctx, oldReadFbo);
            ctx->state.framebuffer = ptr;
            ctx->state.readbuffer = ptr;
            mglFramebufferSyncBindingNames(ctx);
            mglFramebufferLoadDrawBuffer(ctx, ptr);
            mglFramebufferLoadReadBuffer(ctx, ptr);
            drawTargetChanged = GL_TRUE;
            readTargetChanged = GL_TRUE;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    if (drawTargetChanged)
    {
        mglNormalizeDrawBufferForFramebufferBinding(ctx, ptr);
    }

    if (readTargetChanged)
    {
        mglNormalizeReadBufferForFramebufferBinding(ctx, ptr);
    }

    if (drawTargetChanged || readTargetChanged)
    {
        static uint64_t s_fbindTraceCount = 0;
        uint64_t hit = ++s_fbindTraceCount;
        FBOAttachment *color0 = (ctx->state.framebuffer &&
                                 (ctx->state.framebuffer->color_attachment_bitfield & 1u))
            ? &ctx->state.framebuffer->color_attachments[0]
            : NULL;
        FBOAttachment *depth = ctx->state.framebuffer ? &ctx->state.framebuffer->depth : NULL;
        Texture *color_tex = mglTraceAttachmentTexture(ctx, color0);
        Texture *depth_tex = mglTraceAttachmentTexture(ctx, depth);
        GLuint c_ever = 0u, c_full = 0u, c_source = 0u;
        GLuint d_ever = 0u, d_full = 0u, d_source = 0u;
        mglTraceTextureInitSummary(color_tex,
                                   color0 ? color0->level : 0u,
                                   &c_ever,
                                   &c_full,
                                   &c_source);
        mglTraceTextureInitSummary(depth_tex,
                                   depth ? depth->level : 0u,
                                   &d_ever,
                                   &d_full,
                                   &d_source);
        GLboolean small_box = ctx->state.var.scissor_box[2] > 0 &&
                              ctx->state.var.scissor_box[3] > 0 &&
                              ctx->state.var.scissor_box[2] <= 128 &&
                              ctx->state.var.scissor_box[3] <= 128;
        if (hit <= 256ull ||
            (hit % 512ull) == 0ull ||
            small_box ||
            (color_tex && color_tex->width <= 1024u && color_tex->height <= 1024u)) {
            mglTraceLogExternal("FBIND_GL hit=%llu target=0x%x new=%u oldDraw=%u oldRead=%u "
                                "drawChanged=%d readChanged=%d state(draw=%u read=%u drawBuf=0x%x readBuf=0x%x drawCount=%d pendingBatches=%u pendingCmds=%u dirty=0x%x) "
                                "color0(tex=%u target=0x%x level=%u ptr=%p size=%ux%u isRT=%d init=%u/%u/%u rtVer=%u sampledVer=%u clearMask=0x%x) "
                                "depth(tex=%u target=0x%x level=%u ptr=%p size=%ux%u isRT=%d init=%u/%u/%u rtVer=%u sampledVer=%u clearMask=0x%x) "
                                "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d)",
                                (unsigned long long)hit,
                                (unsigned)target,
                                (unsigned)framebuffer,
                                (unsigned)mglTraceSafeFramebufferName(ctx, oldDrawFbo),
                                (unsigned)mglTraceSafeFramebufferName(ctx, oldReadFbo),
                                drawTargetChanged ? 1 : 0,
                                readTargetChanged ? 1 : 0,
                                (unsigned)mglTraceSafeFramebufferName(ctx, ctx->state.framebuffer),
                                (unsigned)mglTraceSafeFramebufferName(ctx, ctx->state.readbuffer),
                                (unsigned)ctx->state.draw_buffer,
                                (unsigned)ctx->state.read_buffer,
                                (int)ctx->state.draw_buffer_count,
                                (unsigned)ctx->draw_command_buffer.batch_count,
                                (unsigned)ctx->draw_command_buffer.total_commands,
                                (unsigned)ctx->state.dirty_bits,
                                color0 ? (unsigned)color0->texture : 0u,
                                color0 ? (unsigned)color0->textarget : 0u,
                                color0 ? (unsigned)color0->level : 0u,
                                color_tex,
                                color_tex ? (unsigned)color_tex->width : 0u,
                                color_tex ? (unsigned)color_tex->height : 0u,
                                color_tex ? (int)color_tex->is_render_target : 0,
                                (unsigned)c_ever,
                                (unsigned)c_full,
                                (unsigned)c_source,
                                color_tex ? (unsigned)color_tex->mtl_render_target_write_version : 0u,
                                color_tex ? (unsigned)color_tex->mtl_gl_sampled_write_version : 0u,
                                color0 ? (unsigned)color0->clear_bitmask : 0u,
                                depth ? (unsigned)depth->texture : 0u,
                                depth ? (unsigned)depth->textarget : 0u,
                                depth ? (unsigned)depth->level : 0u,
                                depth_tex,
                                depth_tex ? (unsigned)depth_tex->width : 0u,
                                depth_tex ? (unsigned)depth_tex->height : 0u,
                                depth_tex ? (int)depth_tex->is_render_target : 0,
                                (unsigned)d_ever,
                                (unsigned)d_full,
                                (unsigned)d_source,
                                depth_tex ? (unsigned)depth_tex->mtl_render_target_write_version : 0u,
                                depth_tex ? (unsigned)depth_tex->mtl_gl_sampled_write_version : 0u,
                                depth ? (unsigned)depth->clear_bitmask : 0u,
                                (int)ctx->state.viewport[0],
                                (int)ctx->state.viewport[1],
                                (int)ctx->state.viewport[2],
                                (int)ctx->state.viewport[3],
                                ctx->state.caps.scissor_test ? 1 : 0,
                                (int)ctx->state.var.scissor_box[0],
                                (int)ctx->state.var.scissor_box[1],
                                (int)ctx->state.var.scissor_box[2],
                                (int)ctx->state.var.scissor_box[3]);
        }
    }
    
    if (drawTargetChanged)
    {
        if (ptr)
        {
            ptr->dirty_bits |= DIRTY_FBO_BINDING;
        }
        STATE(dirty_bits) |= DIRTY_FBO | DIRTY_STATE | DIRTY_RENDER_STATE | DIRTY_PROGRAM;
    }
}

void mglDeleteFramebuffers(GLMContext ctx, GLsizei n, const GLuint *framebuffers)
{
    mglFlushPendingDraws(ctx);

    for (GLsizei i = 0; i < n; i++)
    {
        if (framebuffers[i] == 0)
            continue;
            
        Framebuffer *fbo = findFrameBuffer(ctx, framebuffers[i]);
        if (!fbo)
            continue;
            
        // Unbind if currently bound
        if (ctx->state.framebuffer == fbo)
            ctx->state.framebuffer = NULL;
        if (ctx->state.readbuffer == fbo)
            ctx->state.readbuffer = NULL;
        mglFramebufferSyncBindingNames(ctx);
            
        // Remove from hash table
        deleteHashElement(&STATE(framebuffer_table), framebuffers[i]);
        
        // Free the framebuffer
        free(fbo);
    }
    
    STATE(dirty_bits) |= DIRTY_FBO;
}

static Texture *mglFramebufferAttachmentTextureObject(GLMContext ctx, FBOAttachment *att)
{
    if (!ctx || !att || att->texture == 0u) {
        return NULL;
    }

    if (att->textarget == GL_RENDERBUFFER) {
        return (att->buf.rbo != NULL) ? att->buf.rbo->tex : NULL;
    }

    if (!att->buf.tex) {
        att->buf.tex = findTexture(ctx, att->texture);
    }
    return att->buf.tex;
}

static GLboolean mglFramebufferAttachmentHasStorage(GLMContext ctx, FBOAttachment *att)
{
    Texture *tex = mglFramebufferAttachmentTextureObject(ctx, att);
    GLuint face = 0u;

    if (!tex) {
        return GL_FALSE;
    }

    if (isCubeMapTarget(ctx, att->textarget)) {
        face = textureIndexFromTarget(ctx, att->textarget);
    }

    if (face >= _CUBE_MAP_MAX_FACE ||
        !tex->faces[face].levels ||
        att->level >= tex->mipmap_levels ||
        att->level >= tex->num_levels) {
        return GL_FALSE;
    }

    TextureLevel *level = &tex->faces[face].levels[att->level];
    return level->complete &&
           level->width > 0u &&
           level->height > 0u &&
           level->depth > 0u;
}

static GLboolean mglFramebufferHasAnyAttachment(Framebuffer *fbo)
{
    if (!fbo) {
        return GL_FALSE;
    }

    if (fbo->color_attachment_bitfield != 0u ||
        fbo->depth.texture != 0u ||
        fbo->stencil.texture != 0u) {
        return GL_TRUE;
    }

    return GL_FALSE;
}

static GLboolean mglFramebufferDrawBufferReferencesMissingAttachment(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx || !fbo) {
        return GL_FALSE;
    }

    GLsizei count = fbo->draw_buffer_count;
    if (count < 0 || count > (GLsizei)MAX_COLOR_ATTACHMENTS) {
        return GL_TRUE;
    }

    for (GLsizei i = 0; i < count; ++i) {
        GLenum drawBuffer = fbo->draw_buffers[i];
        if (drawBuffer == GL_NONE) {
            continue;
        }
        if (!mglFramebufferBufferIsColorAttachment(ctx, drawBuffer)) {
            return GL_TRUE;
        }

        GLuint attachmentIndex = (GLuint)(drawBuffer - GL_COLOR_ATTACHMENT0);
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static GLboolean mglFramebufferReadBufferReferencesMissingAttachment(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx || !fbo || fbo->read_buffer == GL_NONE) {
        return GL_FALSE;
    }

    if (!mglFramebufferBufferIsColorAttachment(ctx, fbo->read_buffer)) {
        return GL_TRUE;
    }

    GLuint attachmentIndex = (GLuint)(fbo->read_buffer - GL_COLOR_ATTACHMENT0);
    return ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u;
}

static GLenum mglCheckFramebufferStatusForObject(GLMContext ctx, Framebuffer *fbo)
{
    if (!fbo) {
        return GL_FRAMEBUFFER_COMPLETE;
    }

    if (!mglFramebufferHasAnyAttachment(fbo)) {
        if (fbo->default_width > 0 && fbo->default_height > 0) {
            return GL_FRAMEBUFFER_COMPLETE;
        }
        return GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
    }

    for (GLuint i = 0; i < STATE(max_color_attachments) && i < MAX_COLOR_ATTACHMENTS; ++i) {
        if (((fbo->color_attachment_bitfield >> i) & 1u) == 0u) {
            continue;
        }
        if (!mglFramebufferAttachmentHasStorage(ctx, &fbo->color_attachments[i])) {
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }
    }

    if (fbo->depth.texture != 0u &&
        !mglFramebufferAttachmentHasStorage(ctx, &fbo->depth)) {
        return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    }

    if (fbo->stencil.texture != 0u &&
        !mglFramebufferAttachmentHasStorage(ctx, &fbo->stencil)) {
        return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    }

    if (mglFramebufferDrawBufferReferencesMissingAttachment(ctx, fbo)) {
        return GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER;
    }

    if (mglFramebufferReadBufferReferencesMissingAttachment(ctx, fbo)) {
        return GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER;
    }

    return GL_FRAMEBUFFER_COMPLETE;
}

GLenum  mglCheckFramebufferStatus(GLMContext ctx, GLenum target)
{
    Framebuffer *fbo = NULL;

    switch (target) {
        case GL_FRAMEBUFFER:
        case GL_DRAW_FRAMEBUFFER:
            fbo = ctx->state.framebuffer;
            break;

        case GL_READ_FRAMEBUFFER:
            fbo = ctx->state.readbuffer;
            break;

        default:
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, 0);
    }

    return mglCheckFramebufferStatusForObject(ctx, fbo);
}

#pragma mark Renderbuffer calls
GLboolean mglIsRenderbuffer(GLMContext ctx, GLuint renderbuffer)
{
    return isRenderBuffer(ctx, renderbuffer);
}

void mglGenRenderbuffers(GLMContext ctx, GLsizei n, GLuint *renderbuffers)
{
    if (!ctx)
        return;

    if (n < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (n == 0)
        return;

    if (!renderbuffers) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    while(n--)
    {
        *renderbuffers++ = getNewName(&STATE(renderbuffer_table));
    }
}

void mglBindRenderbuffer(GLMContext ctx, GLenum target, GLuint renderbuffer)
{
    Renderbuffer    *ptr;
    GLuint index;

    // if (ctx->state.framebuffer == NULL)
    // {
    //     // no fbo bound..
    //     assert(0);
    // }

    if (target != GL_RENDERBUFFER) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (renderbuffer)
    {
        ptr = getRenderbuffer(ctx, renderbuffer);
        if (!ptr)
            return;
    }
    else
    {
        ptr = NULL;
    }

    index = textureIndexFromTarget(ctx, target);
    if (index == _MAX_TEXTURE_TYPES)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    ctx->state.renderbuffer = ptr;
    // no dirty state
}

void mglDeleteRenderbuffers(GLMContext ctx, GLsizei n, const GLuint *renderbuffers)
{
    if (n < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (n > 0 && !renderbuffers) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglFlushPendingDraws(ctx);

    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = renderbuffers[i];
        if (name == 0u) {
            continue;
        }

        Renderbuffer *rbo = findRenderbuffer(ctx, name);
        if (!rbo) {
            continue;
        }

        if (ctx->state.renderbuffer == rbo) {
            ctx->state.renderbuffer = NULL;
        }

        deleteHashElement(&STATE(renderbuffer_table), name);
        free(rbo);
    }

    STATE(dirty_bits) |= DIRTY_FBO;
}

void mglRenderbufferStorage(GLMContext ctx, GLenum target, GLenum internalformat, GLsizei width, GLsizei height)
{
    Texture *tex;

    if (target != GL_RENDERBUFFER) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if(ctx->state.renderbuffer == NULL)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    mglFlushPendingDraws(ctx);

    tex = newTexObj(ctx, target);
    if (!tex) {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }

    //bool createTextureLevel(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLboolean is_array, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels, GLboolean proxy)
    if (!createTextureLevel(ctx, tex, 0, 0, false, internalformat, width, height, 1, 0, 0, NULL, false))
        return;

    tex->access = GL_READ_WRITE;
    tex->is_render_target = true;
    
    ctx->state.renderbuffer->tex = tex;
}

void mglGetRenderbufferParameteriv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    if (target != GL_RENDERBUFFER) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!ctx->state.renderbuffer) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    switch(pname)
    {
        case GL_RENDERBUFFER_WIDTH:
            *params = RENDBUF_STATE(tex) ? (GLint)RENDBUF_STATE(tex->width) : 0; break;

        case GL_RENDERBUFFER_HEIGHT:
            *params = RENDBUF_STATE(tex) ? (GLint)RENDBUF_STATE(tex->height) : 0; break;

        case GL_RENDERBUFFER_INTERNAL_FORMAT:
            *params = RENDBUF_STATE(tex) ? (GLint)RENDBUF_STATE(tex->internalformat) : GL_RGBA; break;

        // for now renderbuffers inherit the pixel format from the context..
        case GL_RENDERBUFFER_RED_SIZE:
            *params = bicountForFormatType(ctx->pixel_format.format, ctx->pixel_format.type, GL_RED); break;

        case GL_RENDERBUFFER_GREEN_SIZE:
            *params = bicountForFormatType(ctx->pixel_format.format, ctx->pixel_format.type, GL_GREEN); break;

        case GL_RENDERBUFFER_BLUE_SIZE:
            *params = bicountForFormatType(ctx->pixel_format.format, ctx->pixel_format.type, GL_BLUE); break;

        case GL_RENDERBUFFER_ALPHA_SIZE:
            *params = bicountForFormatType(ctx->pixel_format.format, ctx->pixel_format.type, GL_ALPHA); break;

        case GL_RENDERBUFFER_DEPTH_SIZE:
            *params = RENDBUF_STATE(tex) ? (GLint)bitcountForInternalFormat(RENDBUF_STATE(tex->internalformat), GL_DEPTH) : 0; break;

        case GL_RENDERBUFFER_STENCIL_SIZE:
            *params = RENDBUF_STATE(tex) ? (GLint)bitcountForInternalFormat(RENDBUF_STATE(tex->internalformat), GL_STENCIL) : 0; break;

        case GL_RENDERBUFFER_SAMPLES:
            *params = 0; break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }

}

#pragma mark Framebuffer Texture Bind calls
static bool mglColorAttachmentIndex(GLMContext ctx, GLenum attachment, GLuint *index_out);

FBOAttachment *getFBOAttachment(GLMContext ctx, Framebuffer *fbo, GLenum attachment)
{
    if (!ctx || !fbo) {
        return NULL;
    }

    switch(attachment)
    {
        case GL_DEPTH_ATTACHMENT:
        case GL_DEPTH_STENCIL_ATTACHMENT:
            return &fbo->depth;
            break;

        case GL_STENCIL_ATTACHMENT:
            return &fbo->stencil;
            break;

        default:
        {
            GLuint index;
            if (!mglColorAttachmentIndex(ctx, attachment, &index)) {
                return NULL;
            }
            return &fbo->color_attachments[index];
        }
    }
}

bool isColorAttachment(GLMContext ctx, GLuint attachment)
{
    return ((attachment >= GL_COLOR_ATTACHMENT0) &&
            (attachment < (GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments))));
}

static bool mglColorAttachmentIndex(GLMContext ctx, GLenum attachment, GLuint *index_out)
{
    if (!ctx || attachment < GL_COLOR_ATTACHMENT0) {
        return false;
    }

    GLuint index = attachment - GL_COLOR_ATTACHMENT0;
    if (index >= STATE(max_color_attachments) || index >= MAX_COLOR_ATTACHMENTS) {
        return false;
    }

    if (index_out) {
        *index_out = index;
    }
    return true;
}

bool isCubeMapTarget(GLMContext ctx, GLuint textarget)
{
    return ((textarget >= GL_TEXTURE_CUBE_MAP_POSITIVE_X) &&
            (textarget <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z));
}

static GLboolean mglFramebufferWholeTextureAttachmentIsLayered(GLenum textarget)
{
    switch (textarget) {
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

void framebufferTexture(GLMContext ctx, GLenum target, GLenum attachment_type, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint layer)
{
    Framebuffer *fbo;
    Texture *tex;
    FBOAttachment *fbo_attachment_ptr;
    GLenum effective_textarget = textarget;
    GLboolean is_color_attachment = GL_FALSE;
    GLuint color_attachment_index = 0u;

    fbo = currentFBOForType(ctx, target);
    mglFlushPendingDraws(ctx);
    
    // Log FBO texture attachments for large textures (framebuffer size)
    if (MGL_VERBOSE_FBO_LOGS && texture != 0) {
        Texture *t = findTexture(ctx, texture);
        if (t && t->width >= 640 && t->height >= 400) {
            fprintf(stderr, "MGL DEBUG: FBO attach tex %u (%dx%d) to FBO %u attachment 0x%x\n",
                    texture, t->width, t->height, fbo ? fbo->name : 0, attachment);
        }
    }

    switch(attachment)
    {
        case GL_DEPTH_ATTACHMENT:
        case GL_STENCIL_ATTACHMENT:
        case GL_DEPTH_STENCIL_ATTACHMENT:
            break;

        default:
        {
            GLuint index;
            if (mglColorAttachmentIndex(ctx, attachment, &index))
            {
                is_color_attachment = GL_TRUE;
                color_attachment_index = index;
                break;
            }

            fprintf(stderr,
                    "MGL ERROR: framebufferTexture invalid attachment=0x%x maxColor=%u target=0x%x texture=%u textarget=0x%x level=%d\n",
                    attachment,
                    STATE(max_color_attachments),
                    target,
                    texture,
                    textarget,
                    level);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
        }
    }

    if (texture)
    {
        tex = findTexture(ctx, texture);

        // Some apps attach by name before the texture object is fully realized in MGL.
        // Create a placeholder object so later render/blit paths can resolve it.
        if (!tex && textarget != GL_NONE)
        {
            tex = newTexture(ctx, textarget, texture);
            if (tex)
            {
                insertHashElement(&STATE(texture_table), texture, tex);
                fprintf(stderr, "MGL INFO: framebufferTexture created missing texture object %u target=0x%x\n",
                        texture, textarget);
            }
        }

        if (effective_textarget == GL_NONE && tex)
        {
            effective_textarget = tex->target;
        }

        switch(effective_textarget)
        {
            case GL_TEXTURE_BUFFER:
            case GL_TEXTURE_1D:
            case GL_TEXTURE_2D:
            case GL_TEXTURE_3D:
            case GL_TEXTURE_RECTANGLE:
            case GL_TEXTURE_1D_ARRAY:
            case GL_TEXTURE_2D_ARRAY:
            case GL_TEXTURE_2D_MULTISAMPLE:
            case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
                break;

            default:
                if (!tex)
                {
                    // Keep attachment name and defer completeness validation.
                    break;
                }

                if (tex->target == GL_TEXTURE_CUBE_MAP)
                {
                    if (isCubeMapTarget(ctx, effective_textarget) ||
                        effective_textarget == GL_TEXTURE_CUBE_MAP)
                    {
                        break;
                    }
                }

                STATE(error) = GL_INVALID_OPERATION;
                return;

                break;
        }

        if (level < 0)
        {
            STATE(error) = GL_INVALID_VALUE;
            return;
        }

        // A texture may legally be attached before it has storage allocated; that should
        // make the FBO incomplete, not raise GL_INVALID_VALUE.
        if (tex && tex->mipmap_levels != 0 && level >= (GLint)tex->mipmap_levels)
        {
            STATE(error) = GL_INVALID_VALUE;
            return;
        }

        if (level > 0)
        {
            switch(effective_textarget)
            {
                // If textarget is GL_TEXTURE_RECTANGLE, GL_TEXTURE_2D_MULTISAMPLE, or GL_TEXTURE_2D_MULTISAMPLE_ARRAY, then level must be zero.
                case GL_TEXTURE_RECTANGLE:
                case GL_TEXTURE_2D_MULTISAMPLE:
                case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
                    STATE(error) = GL_INVALID_VALUE;
                    return;

                // if textarget is GL_TEXTURE_3D, then level must be greater than or equal to zero and less than or equal to $log_2$ of the value of GL_MAX_3D_TEXTURE_SIZE.
                case GL_TEXTURE_3D:
                    if (level >= ilog2(STATE_VAR(max_texture_size)))
                    {
                        fprintf(stderr,
                                "MGL ERROR: framebufferTexture invalid 3D mip level=%d maxTex=%u target=0x%x attachment=0x%x texture=%u textarget=0x%x\n",
                                level,
                                STATE_VAR(max_texture_size),
                                target,
                                attachment,
                                texture,
                                effective_textarget);
                        STATE(error) = GL_INVALID_VALUE;
                        return;
                    }
                    break;

                default:
                    if (tex && tex->target == GL_TEXTURE_CUBE_MAP)
                    {
                        // if textarget is one of GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, or GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, then level must be greater than or equal to zero and less than or equal to $log_2$ of the value of GL_MAX_CUBE_MAP_TEXTURE_SIZE.

                        if (isCubeMapTarget(ctx, effective_textarget) ||
                            effective_textarget == GL_TEXTURE_CUBE_MAP)
                        {
                            if (level >=0 && level <= ilog2(STATE_VAR(max_texture_size)))
                            {
                                break;
                            }

                            fprintf(stderr,
                                    "MGL ERROR: framebufferTexture invalid cube mip level=%d maxTex=%u target=0x%x attachment=0x%x texture=%u textarget=0x%x\n",
                                    level,
                                    STATE_VAR(max_texture_size),
                                    target,
                                    attachment,
                                    texture,
                                    effective_textarget);
                            STATE(error) = GL_INVALID_VALUE;
                            return;
                        }
                    }
                    else if (level >= ilog2(STATE_VAR(max_texture_size)))
                    {
                        // For all other values of textarget, level must be greater than or equal to zero and less than or equal to $log_2$ of the value of GL_MAX_TEXTURE_SIZE.


                        fprintf(stderr,
                                "MGL ERROR: framebufferTexture invalid mip level=%d maxTex=%u target=0x%x attachment=0x%x texture=%u textarget=0x%x texTarget=0x%x texLevels=%u\n",
                                level,
                                STATE_VAR(max_texture_size),
                                target,
                                attachment,
                                texture,
                                effective_textarget,
                                tex ? tex->target : 0u,
                                tex ? tex->mipmap_levels : 0u);
                        STATE(error) = GL_INVALID_VALUE;
                        return;
                    }
                    break;
            }
        }
    }
    else
    {
        // ignore all error checking
        tex = NULL;
    }

    fbo_attachment_ptr = getFBOAttachment(ctx, fbo, attachment);
    if (!fbo_attachment_ptr) {
        fprintf(stderr,
                "MGL ERROR: framebufferTexture could not resolve attachment=0x%x fbo=%p\n",
                attachment,
                fbo);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (mglAttachmentChangeNeedsPendingClearFlush(fbo_attachment_ptr, texture, effective_textarget, level, layer)) {
        mglFlushPendingColorClearToTexture(ctx, fbo_attachment_ptr, "framebufferTexture attachment change");
    }

    if (is_color_attachment) {
        if (texture) {
            fbo->color_attachment_bitfield |= (0x1u << color_attachment_index);
        } else {
            fbo->color_attachment_bitfield &= ~(0x1u << color_attachment_index);
        }
    }

    fbo_attachment_ptr->texture = texture;
    fbo_attachment_ptr->textarget = effective_textarget;
    fbo_attachment_ptr->level = level;
    fbo_attachment_ptr->layer = layer;
    fbo_attachment_ptr->layered = (texture != 0u &&
                                   attachment_type == GL_NONE &&
                                   mglFramebufferWholeTextureAttachmentIsLayered(effective_textarget));
    fbo_attachment_ptr->clear_bitmask = 0;
    fbo_attachment_ptr->clear_color[0] = 0.f;
    fbo_attachment_ptr->clear_color[1] = 0.f;
    fbo_attachment_ptr->clear_color[2] = 0.f;
    fbo_attachment_ptr->clear_color[3] = 0.f;
    fbo_attachment_ptr->buf.tex = tex;
    if (tex) {
        mglClearLastSampled2DTextureIfMatches(ctx, tex);
    }

    if (attachment == GL_DEPTH_STENCIL_ATTACHMENT)
    {
        fbo->stencil = fbo->depth;
    }

    fbo->dirty_bits |= DIRTY_FBO_BINDING;
    STATE(dirty_bits) |= DIRTY_FBO;
}

/*
 target
 Specifies the target to which the framebuffer is bound for all commands except glNamedFramebufferTexture.

 framebuffer
 Specifies the name of the framebuffer object for glNamedFramebufferTexture.

 attachment
 Specifies the attachment point of the framebuffer.

 textarget
 For glFramebufferTexture1D, glFramebufferTexture2D and glFramebufferTexture3D, specifies what type of texture is expected in the texture parameter, or for cube map textures, which face is to be attached.

 texture
 Specifies the name of an existing texture object to attach.

 level
 Specifies the mipmap level of the texture object to attach.
 */

void mglFramebufferTexture(GLMContext ctx, GLenum target, GLenum attachment, GLuint texture, GLint level)
{
    framebufferTexture(ctx, target, GL_NONE, attachment, GL_NONE, texture, level, 0);
}


void mglFramebufferTexture1D(GLMContext ctx, GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level)
{
    if (textarget != GL_TEXTURE_1D) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    framebufferTexture(ctx, target, GL_TEXTURE_1D, attachment, textarget, texture, level, 0);
}

void mglFramebufferTexture2D(GLMContext ctx, GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level)
{
    switch(textarget)
    {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_2D_MULTISAMPLE:
            break;

        default:
            if (isCubeMapTarget(ctx, textarget))
            {
                break;
            }

            fprintf(stderr,
                    "MGL ERROR: mglFramebufferTexture2D invalid textarget=0x%x texture=%u attachment=0x%x level=%d\n",
                    textarget,
                    texture,
                    attachment,
                    level);
            ERROR_RETURN(GL_INVALID_ENUM);

            return;
    }

    framebufferTexture(ctx, target, GL_TEXTURE_2D, attachment, textarget, texture, level, 0);
}

void mglFramebufferTexture3D(GLMContext ctx, GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset)
{
    if (textarget != GL_TEXTURE_3D) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    framebufferTexture(ctx, target, GL_TEXTURE_3D, attachment, textarget, texture, level, zoffset);
}

void mglFramebufferTextureLayer(GLMContext ctx, GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer)
{
    Texture *tex = texture ? findTexture(ctx, texture) : NULL;
    GLenum textarget = tex ? tex->target : GL_TEXTURE_3D;
    framebufferTexture(ctx, target, GL_TEXTURE_3D, attachment, textarget, texture, level, layer);
}


void mglFramebufferRenderbuffer(GLMContext ctx, GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)
{
    Framebuffer *fbo;
    Renderbuffer *rbo;
    FBOAttachment *fbo_attachment_ptr;

    fbo = currentFBOForType(ctx, target);
    mglFlushPendingDraws(ctx);

    switch(attachment)
    {
        case GL_DEPTH_ATTACHMENT:
        case GL_DEPTH_STENCIL_ATTACHMENT:
        case GL_STENCIL_ATTACHMENT:
            break;

        default:
        {
            GLuint index;
            if (mglColorAttachmentIndex(ctx, attachment, &index))
            {
                if (renderbuffer)
                {
                    fbo->color_attachment_bitfield |= (0x1 << index);
                }
                else
                {
                    fbo->color_attachment_bitfield &= ~(0x1 << index);
                }
                break;
            }

            fprintf(stderr,
                    "MGL ERROR: mglFramebufferRenderbuffer invalid attachment=0x%x maxColor=%u target=0x%x renderbuffer=%u\n",
                    attachment,
                    STATE(max_color_attachments),
                    target,
                    renderbuffer);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
        }
    }

    if (renderbuffer)
    {
        rbo = findRenderbuffer(ctx, renderbuffer);
        if (!rbo) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }
    else
    {
        rbo = NULL;
    }

    fbo_attachment_ptr = getFBOAttachment(ctx, fbo, attachment);

    fbo_attachment_ptr->textarget = GL_RENDERBUFFER;
    fbo_attachment_ptr->texture = renderbuffer;
    fbo_attachment_ptr->level = 0;
    fbo_attachment_ptr->layer = 0;
    fbo_attachment_ptr->layered = GL_FALSE;
    fbo_attachment_ptr->buf.rbo = rbo;
    if (fbo_attachment_ptr->buf.rbo)
    {
        fbo_attachment_ptr->buf.rbo->is_draw_buffer = GL_FALSE;
    }

    if (attachment == GL_DEPTH_STENCIL_ATTACHMENT)
    {
        fbo->stencil = fbo->depth;
    }

    fbo->dirty_bits |= DIRTY_FBO_BINDING;
    STATE(dirty_bits) |= DIRTY_FBO;
}

#pragma mark =====

static GLuint mglFramebufferAttachmentBits(GLenum internalformat, GLenum component)
{
    if (internalformat == 0u) {
        return 0u;
    }

    switch (internalformat) {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R8I:
        case GL_R8UI:
            return component == GL_RED ? 8u : 0u;
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R16I:
        case GL_R16UI:
            return component == GL_RED ? 16u : 0u;
        case GL_R32F:
        case GL_R32I:
        case GL_R32UI:
            return component == GL_RED ? 32u : 0u;

        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RG8I:
        case GL_RG8UI:
            return (component == GL_RED || component == GL_GREEN) ? 8u : 0u;
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_RG16F:
        case GL_RG16I:
        case GL_RG16UI:
            return (component == GL_RED || component == GL_GREEN) ? 16u : 0u;
        case GL_RG32F:
        case GL_RG32I:
        case GL_RG32UI:
            return (component == GL_RED || component == GL_GREEN) ? 32u : 0u;

        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB8_SNORM:
        case GL_RGB8I:
        case GL_RGB8UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 8u : 0u;
        case GL_RGB16:
        case GL_RGB16_SNORM:
        case GL_RGB16F:
        case GL_RGB16I:
        case GL_RGB16UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 16u : 0u;
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 32u : 0u;

        case GL_RGBA8:
        case GL_SRGB8_ALPHA8:
        case GL_RGBA8_SNORM:
        case GL_RGBA8I:
        case GL_RGBA8UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE || component == GL_ALPHA) ? 8u : 0u;
        case GL_RGBA16:
        case GL_RGBA16_SNORM:
        case GL_RGBA16F:
        case GL_RGBA16I:
        case GL_RGBA16UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE || component == GL_ALPHA) ? 16u : 0u;
        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE || component == GL_ALPHA) ? 32u : 0u;

        case GL_RGB565:
            switch (component) {
                case GL_RED: return 5u;
                case GL_GREEN: return 6u;
                case GL_BLUE: return 5u;
                default: return 0u;
            }
        case GL_RGB5_A1:
            switch (component) {
                case GL_RED:
                case GL_GREEN:
                case GL_BLUE: return 5u;
                case GL_ALPHA: return 1u;
                default: return 0u;
            }
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
            switch (component) {
                case GL_RED:
                case GL_GREEN:
                case GL_BLUE: return 10u;
                case GL_ALPHA: return 2u;
                default: return 0u;
            }

        case GL_DEPTH_COMPONENT16:
            return component == GL_DEPTH ? 16u : 0u;
        case GL_DEPTH_COMPONENT24:
            return component == GL_DEPTH ? 24u : 0u;
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            return component == GL_DEPTH ? 32u : 0u;
        case GL_DEPTH24_STENCIL8:
            return component == GL_DEPTH ? 24u : (component == GL_STENCIL ? 8u : 0u);
        case GL_DEPTH32F_STENCIL8:
            return component == GL_DEPTH ? 32u : (component == GL_STENCIL ? 8u : 0u);
        case GL_STENCIL_INDEX1:
            return component == GL_STENCIL ? 1u : 0u;
        case GL_STENCIL_INDEX4:
            return component == GL_STENCIL ? 4u : 0u;
        case GL_STENCIL_INDEX8:
            return component == GL_STENCIL ? 8u : 0u;
        case GL_STENCIL_INDEX16:
            return component == GL_STENCIL ? 16u : 0u;

        default:
            return bitcountForInternalFormat(internalformat, component);
    }
}

static GLenum mglFramebufferAttachmentComponentType(GLenum internalformat)
{
    switch (internalformat) {
        case GL_R8I:
        case GL_R16I:
        case GL_R32I:
        case GL_RG8I:
        case GL_RG16I:
        case GL_RG32I:
        case GL_RGB8I:
        case GL_RGB16I:
        case GL_RGB32I:
        case GL_RGBA8I:
        case GL_RGBA16I:
        case GL_RGBA32I:
        case GL_STENCIL_INDEX1:
        case GL_STENCIL_INDEX4:
        case GL_STENCIL_INDEX8:
        case GL_STENCIL_INDEX16:
            return GL_INT;

        case GL_R8UI:
        case GL_R16UI:
        case GL_R32UI:
        case GL_RG8UI:
        case GL_RG16UI:
        case GL_RG32UI:
        case GL_RGB8UI:
        case GL_RGB16UI:
        case GL_RGB32UI:
        case GL_RGBA8UI:
        case GL_RGBA16UI:
        case GL_RGBA32UI:
        case GL_RGB10_A2UI:
            return GL_UNSIGNED_INT;

        case GL_R16F:
        case GL_R32F:
        case GL_RG16F:
        case GL_RG32F:
        case GL_RGB16F:
        case GL_RGB32F:
        case GL_RGBA16F:
        case GL_RGBA32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH32F_STENCIL8:
            return GL_FLOAT;

        case GL_R8_SNORM:
        case GL_R16_SNORM:
        case GL_RG8_SNORM:
        case GL_RG16_SNORM:
        case GL_RGB8_SNORM:
        case GL_RGB16_SNORM:
        case GL_RGBA8_SNORM:
        case GL_RGBA16_SNORM:
            return GL_SIGNED_NORMALIZED;

        default:
            return GL_UNSIGNED_NORMALIZED;
    }
}

static GLenum mglFramebufferAttachmentColorEncoding(GLenum internalformat)
{
    switch (internalformat) {
        case GL_SRGB:
        case GL_SRGB8:
        case GL_SRGB_ALPHA:
        case GL_SRGB8_ALPHA8:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB8_ETC2:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
            return GL_SRGB;
        default:
            return GL_LINEAR;
    }
}

static GLboolean mglDefaultFramebufferAttachmentIsColor(GLenum attachment)
{
    switch (attachment) {
        case GL_FRONT:
        case GL_BACK:
        case GL_FRONT_LEFT:
        case GL_FRONT_RIGHT:
        case GL_BACK_LEFT:
        case GL_BACK_RIGHT:
        case GL_LEFT:
        case GL_RIGHT:
        case GL_COLOR:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean mglDefaultFramebufferAttachmentIsValid(GLenum attachment)
{
    return mglDefaultFramebufferAttachmentIsColor(attachment) ||
           attachment == GL_DEPTH ||
           attachment == GL_STENCIL ||
           attachment == GL_DEPTH_ATTACHMENT ||
           attachment == GL_STENCIL_ATTACHMENT;
}

static GLuint mglDefaultFramebufferColorBits(GLMContext ctx, GLenum component)
{
    GLenum format = ctx ? ctx->pixel_format.format : GL_RGBA;
    GLenum type = ctx ? ctx->pixel_format.type : GL_UNSIGNED_BYTE;

    switch (format) {
        case GL_RED:
            if (component != GL_RED) return 0u;
            break;
        case GL_RG:
            if (component != GL_RED && component != GL_GREEN) return 0u;
            break;
        case GL_RGB:
        case GL_BGR:
            if (component == GL_ALPHA) return 0u;
            break;
        default:
            break;
    }

    return bicountForFormatType(format, type, component);
}

static void mglGetDefaultFramebufferAttachmentParameteriv(GLMContext ctx, GLenum attachment, GLenum pname, GLint *params)
{
    if (!mglDefaultFramebufferAttachmentIsValid(attachment)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    GLboolean colorAttachment = mglDefaultFramebufferAttachmentIsColor(attachment);
    GLboolean depthAttachment = (attachment == GL_DEPTH || attachment == GL_DEPTH_ATTACHMENT);
    GLboolean stencilAttachment = (attachment == GL_STENCIL || attachment == GL_STENCIL_ATTACHMENT);

    switch (pname) {
        case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
            *params = GL_FRAMEBUFFER_DEFAULT;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
        case GL_FRAMEBUFFER_ATTACHMENT_LAYERED:
            *params = 0;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
            *params = GL_NONE;
            return;

        case GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE:
            *params = colorAttachment ? (GLint)mglDefaultFramebufferColorBits(ctx, GL_RED) : 0;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE:
            *params = colorAttachment ? (GLint)mglDefaultFramebufferColorBits(ctx, GL_GREEN) : 0;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE:
            *params = colorAttachment ? (GLint)mglDefaultFramebufferColorBits(ctx, GL_BLUE) : 0;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE:
            *params = colorAttachment ? (GLint)mglDefaultFramebufferColorBits(ctx, GL_ALPHA) : 0;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE:
            *params = depthAttachment ? (GLint)mglFramebufferAttachmentBits(ctx->depth_format.format, GL_DEPTH) : 0;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE:
            *params = stencilAttachment ? (GLint)mglFramebufferAttachmentBits(ctx->stencil_format.format, GL_STENCIL) : 0;
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
            if (colorAttachment) {
                *params = GL_UNSIGNED_NORMALIZED;
            } else if (depthAttachment) {
                *params = mglFramebufferAttachmentComponentType(ctx->depth_format.format);
            } else if (stencilAttachment) {
                *params = GL_INT;
            } else {
                *params = GL_NONE;
            }
            return;
        case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
            *params = (colorAttachment && ctx && ctx->default_framebuffer_srgb_capable)
                ? GL_SRGB
                : GL_LINEAR;
            return;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}

void getFramebufferAttachmentParameteriv(GLMContext ctx, GLuint framebuffer, GLenum target, GLenum attachment, GLenum pname, GLint *params)
{
    Framebuffer *fbo = NULL;
    FBOAttachment *fbo_attachment_ptr = NULL;

    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch(target)
    {
        case GL_DRAW_FRAMEBUFFER:
        case GL_FRAMEBUFFER:
            fbo = ctx->state.framebuffer;
            break;

        case GL_READ_FRAMEBUFFER:
            fbo = ctx->state.readbuffer;
            break;

        default:
            if (target != 0u) {
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
            }
            if (framebuffer != 0u) {
                fbo = findFrameBuffer(ctx, framebuffer);
                if (!fbo) {
                    ERROR_RETURN(GL_INVALID_OPERATION);
                    return;
                }
            }
            break;
    }

    if (fbo)
    {
        Texture *tex = NULL;
        GLenum object_type = GL_NONE;
        GLenum attachment_target;

        fbo_attachment_ptr = getFBOAttachment(ctx, fbo, attachment);

        if (fbo_attachment_ptr == NULL)
        {
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
        }

        attachment_target = fbo_attachment_ptr->textarget;
        if (fbo_attachment_ptr->texture != 0u) {
            if (attachment_target == GL_RENDERBUFFER) {
                object_type = GL_RENDERBUFFER;
                tex = (fbo_attachment_ptr->buf.rbo != NULL) ? fbo_attachment_ptr->buf.rbo->tex : NULL;
            } else {
                object_type = GL_TEXTURE;
                tex = fbo_attachment_ptr->buf.tex;
                if (!tex) {
                    tex = findTexture(ctx, fbo_attachment_ptr->texture);
                    fbo_attachment_ptr->buf.tex = tex;
                }
            }
        }

        GLenum internalformat = tex ? tex->internalformat : 0u;

        switch(pname)
        {
            case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
                *params = object_type;
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
                *params = (GLint)fbo_attachment_ptr->texture;
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE:
                *params = (GLint)mglFramebufferAttachmentBits(internalformat, GL_RED);
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE:
                *params = (GLint)mglFramebufferAttachmentBits(internalformat, GL_GREEN);
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE:
                *params = (GLint)mglFramebufferAttachmentBits(internalformat, GL_BLUE);
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE:
                *params = (GLint)mglFramebufferAttachmentBits(internalformat, GL_ALPHA);
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE:
                *params = (GLint)mglFramebufferAttachmentBits(internalformat, GL_DEPTH);
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE:
                *params = (GLint)mglFramebufferAttachmentBits(internalformat, GL_STENCIL);
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
                *params = (object_type == GL_NONE) ? GL_NONE : mglFramebufferAttachmentComponentType(internalformat);
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
                *params = (object_type == GL_NONE) ? GL_LINEAR : mglFramebufferAttachmentColorEncoding(internalformat);
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
                *params = (object_type == GL_TEXTURE) ? (GLint)fbo_attachment_ptr->level : 0;
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
                *params = (object_type == GL_TEXTURE && isCubeMapTarget(ctx, attachment_target)) ? (GLint)attachment_target : GL_NONE;
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
                *params = (object_type == GL_TEXTURE &&
                           (fbo_attachment_ptr->layered ||
                            attachment_target == GL_TEXTURE_3D ||
                            attachment_target == GL_TEXTURE_1D_ARRAY ||
                            attachment_target == GL_TEXTURE_2D_ARRAY ||
                            attachment_target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY ||
                            attachment_target == GL_TEXTURE_CUBE_MAP_ARRAY))
                    ? (GLint)fbo_attachment_ptr->layer
                    : 0;
                return;

            case GL_FRAMEBUFFER_ATTACHMENT_LAYERED:
                *params = (object_type == GL_TEXTURE && fbo_attachment_ptr->layered) ? GL_TRUE : GL_FALSE;
                return;

            default:
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
        }
    }
    else
    {
        mglGetDefaultFramebufferAttachmentParameteriv(ctx, attachment, pname, params);
    }
}

void mglGetFramebufferAttachmentParameteriv(GLMContext ctx, GLenum target, GLenum attachment, GLenum pname, GLint *params)
{
    getFramebufferAttachmentParameteriv(ctx, 0, target, attachment, pname, params);
}

void mglGetNamedFramebufferAttachmentParameteriv(GLMContext ctx, GLuint framebuffer, GLenum attachment, GLenum pname, GLint *params)
{
    getFramebufferAttachmentParameteriv(ctx, framebuffer, 0, attachment, pname, params);
}


void mglBlitFramebuffer(GLMContext ctx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
{
    static uint64_t s_blitTraceCount = 0;
    uint64_t blitHit = ++s_blitTraceCount;
    if (blitHit <= 128ull ||
        (mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) != 0u ||
        (blitHit % 512ull) == 0ull) {
        Framebuffer *drawFbo = ctx ? ctx->state.framebuffer : NULL;
        Framebuffer *readFbo = ctx ? ctx->state.readbuffer : NULL;
        mglTraceLogExternal("BLIT_FRAMEBUFFER call=%llu src=(%d,%d)-(%d,%d) dst=(%d,%d)-(%d,%d) mask=0x%x filter=0x%x drawFbo=%u readFbo=%u drawBuf=0x%x readBuf=0x%x depthStencil=%d",
                            (unsigned long long)blitHit,
                            srcX0, srcY0, srcX1, srcY1,
                            dstX0, dstY0, dstX1, dstY1,
                            (unsigned)mask,
                            (unsigned)filter,
                            (unsigned)mglTraceSafeFramebufferName(ctx, drawFbo),
                            (unsigned)mglTraceSafeFramebufferName(ctx, readFbo),
                            (unsigned)(ctx ? ctx->state.draw_buffer : 0u),
                            (unsigned)(ctx ? ctx->state.read_buffer : 0u),
                            (mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) != 0u ? 1 : 0);
    }

    if (MGL_VERBOSE_FBO_LOGS) {
        fprintf(stderr, "MGL: glBlitFramebuffer src(%d,%d)-(%d,%d) dst(%d,%d)-(%d,%d) mask=0x%x\n",
                srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask);
    }

    const GLbitfield validMask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
    if ((mask & ~validMask) != 0u) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (filter != GL_NEAREST && filter != GL_LINEAR) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if ((mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) != 0u && filter != GL_NEAREST) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if ((mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) != 0u) {
        fprintf(stderr,
                "MGL WARNING: glBlitFramebuffer depth/stencil mask=0x%x is unsupported by the Metal blit path\n",
                mask);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ctx->mtl_funcs.mtlBlitFramebuffer(ctx, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
}

void mglRenderbufferStorageMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height)
{
    if (target != GL_RENDERBUFFER) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (samples < 0 || width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (samples > 0) {
        fprintf(stderr,
                "MGL WARNING: glRenderbufferStorageMultisample unsupported samples=%d internalformat=0x%x %dx%d\n",
                samples, internalformat, width, height);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    mglRenderbufferStorage(ctx, target, internalformat, width, height);
}

void mglFramebufferParameteri(GLMContext ctx, GLenum target, GLenum pname, GLint param)
{
    Framebuffer *fbo = NULL;
    
    // Get the appropriate framebuffer based on target
    switch(target) {
        case GL_FRAMEBUFFER:
        case GL_DRAW_FRAMEBUFFER:
            fbo = STATE(framebuffer);
            break;
        case GL_READ_FRAMEBUFFER:
            fbo = STATE(readbuffer);
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
    
    if (!fbo) {
        // Default framebuffer - these parameters don't apply
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    switch(pname) {
        case GL_FRAMEBUFFER_DEFAULT_WIDTH:
            if (param < 0) {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
            fbo->default_width = param;
            break;
        case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
            if (param < 0) {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
            fbo->default_height = param;
            break;
        case GL_FRAMEBUFFER_DEFAULT_LAYERS:
            if (param < 0) {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
            fbo->default_layers = param;
            break;
        case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
            if (param < 0) {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
            fbo->default_samples = param;
            break;
        case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
            fbo->default_fixed_sample_locations = param ? GL_TRUE : GL_FALSE;
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }
}

void mglGetFramebufferParameteriv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    Framebuffer *fbo = NULL;

    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch(target) {
        case GL_FRAMEBUFFER:
        case GL_DRAW_FRAMEBUFFER:
            fbo = STATE(framebuffer);
            break;
        case GL_READ_FRAMEBUFFER:
            fbo = STATE(readbuffer);
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch (pname) {
        case GL_FRAMEBUFFER_DEFAULT_WIDTH:
            *params = fbo ? fbo->default_width : 0;
            break;
        case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
            *params = fbo ? fbo->default_height : 0;
            break;
        case GL_FRAMEBUFFER_DEFAULT_LAYERS:
            *params = fbo ? fbo->default_layers : 0;
            break;
        case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
            *params = fbo ? fbo->default_samples : 0;
            break;
        case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
            *params = fbo ? fbo->default_fixed_sample_locations : GL_TRUE;
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}

void mglInvalidateFramebuffer(GLMContext ctx, GLenum target, GLsizei numAttachments, const GLenum *attachments)
{
    switch (target) {
        case GL_FRAMEBUFFER:
        case GL_DRAW_FRAMEBUFFER:
        case GL_READ_FRAMEBUFFER:
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    if (numAttachments < 0 || (numAttachments > 0 && !attachments)) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
}

void mglInvalidateSubFramebuffer(GLMContext ctx, GLenum target, GLsizei numAttachments, const GLenum *attachments, GLint x, GLint y, GLsizei width, GLsizei height)
{
    switch (target) {
        case GL_FRAMEBUFFER:
        case GL_DRAW_FRAMEBUFFER:
        case GL_READ_FRAMEBUFFER:
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    if (numAttachments < 0 || (numAttachments > 0 && !attachments) || width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    (void)x;
    (void)y;
}

void mglCreateFramebuffers(GLMContext ctx, GLsizei n, GLuint *framebuffers)
{
    if (n < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (n > 0 && !framebuffers) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = getNewName(&STATE(framebuffer_table));
        Framebuffer *fbo = newFramebuffer(ctx, name);
        insertHashElement(&STATE(framebuffer_table), name, fbo);
        framebuffers[i] = name;
    }
}

void mglNamedFramebufferRenderbuffer(GLMContext ctx, GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)
{
    Framebuffer *fbo = findFrameBuffer(ctx, framebuffer);

    if (framebuffer == 0u || !fbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    MGLFramebufferBindingSnapshot snapshot;
    mglFramebufferCaptureBindingSnapshot(ctx, &snapshot);
    mglFlushPendingDraws(ctx);
    mglFramebufferUseTemporaryDrawBinding(ctx, fbo);
    mglFramebufferRenderbuffer(ctx, GL_DRAW_FRAMEBUFFER, attachment, renderbuffertarget, renderbuffer);
    mglFramebufferRestoreBindingSnapshot(ctx, &snapshot);
}

void mglNamedFramebufferParameteri(GLMContext ctx, GLuint framebuffer, GLenum pname, GLint param)
{
    Framebuffer *fbo = findFrameBuffer(ctx, framebuffer);

    if (framebuffer == 0u || !fbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    switch(pname) {
        case GL_FRAMEBUFFER_DEFAULT_WIDTH:
            if (param < 0) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            fbo->default_width = param;
            break;
        case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
            if (param < 0) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            fbo->default_height = param;
            break;
        case GL_FRAMEBUFFER_DEFAULT_LAYERS:
            if (param < 0) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            fbo->default_layers = param;
            break;
        case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
            if (param < 0) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            fbo->default_samples = param;
            break;
        case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
            fbo->default_fixed_sample_locations = param ? GL_TRUE : GL_FALSE;
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    fbo->dirty_bits |= DIRTY_FBO_BINDING;
    STATE(dirty_bits) |= DIRTY_FBO;
}

void mglNamedFramebufferTexture(GLMContext ctx, GLuint framebuffer, GLenum attachment, GLuint texture, GLint level)
{
    Framebuffer *fbo = findFrameBuffer(ctx, framebuffer);

    if (framebuffer == 0u || !fbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    MGLFramebufferBindingSnapshot snapshot;
    mglFramebufferCaptureBindingSnapshot(ctx, &snapshot);
    mglFlushPendingDraws(ctx);
    mglFramebufferUseTemporaryDrawBinding(ctx, fbo);
    framebufferTexture(ctx, GL_DRAW_FRAMEBUFFER, GL_NONE, attachment, GL_NONE, texture, level, 0);
    mglFramebufferRestoreBindingSnapshot(ctx, &snapshot);
}

void mglNamedFramebufferTextureLayer(GLMContext ctx, GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLint layer)
{
    Framebuffer *fbo = findFrameBuffer(ctx, framebuffer);

    if (framebuffer == 0u || !fbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    MGLFramebufferBindingSnapshot snapshot;
    mglFramebufferCaptureBindingSnapshot(ctx, &snapshot);
    mglFlushPendingDraws(ctx);
    mglFramebufferUseTemporaryDrawBinding(ctx, fbo);
    framebufferTexture(ctx, GL_DRAW_FRAMEBUFFER, GL_NONE, attachment, GL_NONE, texture, level, layer);
    mglFramebufferRestoreBindingSnapshot(ctx, &snapshot);
}

void mglNamedFramebufferDrawBuffer(GLMContext ctx, GLuint framebuffer, GLenum buf)
{
    mglNamedFramebufferDrawBuffers(ctx, framebuffer, 1, &buf);
}

void mglNamedFramebufferDrawBuffers(GLMContext ctx, GLuint framebuffer, GLsizei n, const GLenum *bufs)
{
    Framebuffer *fbo = framebuffer ? findFrameBuffer(ctx, framebuffer) : NULL;

    if (framebuffer != 0u && !fbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (n < 0 || n > (GLsizei)mglFramebufferMaxDrawBuffers(ctx)) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (n > 0 && !bufs) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    GLbitfield seenColorAttachments = 0u;
    for (GLsizei i = 0; i < n; ++i) {
        GLenum buf = bufs[i];
        if (buf == GL_NONE) {
            continue;
        }

        if (fbo) {
            if (!mglFramebufferBufferIsColorAttachment(ctx, buf)) {
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
            }
            GLuint attachmentIndex = (GLuint)(buf - GL_COLOR_ATTACHMENT0);
            GLbitfield attachmentBit = (GLbitfield)(1u << attachmentIndex);
            if (seenColorAttachments & attachmentBit) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            seenColorAttachments |= attachmentBit;
            continue;
        }

        switch (buf) {
            case GL_FRONT:
            case GL_BACK:
            case GL_FRONT_LEFT:
            case GL_FRONT_RIGHT:
            case GL_BACK_LEFT:
            case GL_BACK_RIGHT:
            case GL_LEFT:
            case GL_RIGHT:
            case GL_FRONT_AND_BACK:
                break;
            default:
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
        }
    }

    GLenum *drawBuffers = fbo ? fbo->draw_buffers : ctx->state.default_draw_buffers;
    GLsizei *drawBufferCount = fbo ? &fbo->draw_buffer_count : &ctx->state.default_draw_buffer_count;
    GLuint *drawBuffer = fbo ? &fbo->draw_buffer : &ctx->state.default_draw_buffer;
    GLboolean changed = (*drawBufferCount != n);

    for (GLsizei i = 0; !changed && i < n; ++i) {
        if (drawBuffers[i] != bufs[i]) {
            changed = GL_TRUE;
        }
    }
    for (GLsizei i = n; !changed && i < (GLsizei)MAX_COLOR_ATTACHMENTS; ++i) {
        if (drawBuffers[i] != GL_NONE) {
            changed = GL_TRUE;
        }
    }

    if (changed) {
        mglFlushPendingDraws(ctx);
    }

    for (GLsizei i = 0; i < n; ++i) {
        drawBuffers[i] = bufs[i];
    }
    for (GLsizei i = n; i < (GLsizei)MAX_COLOR_ATTACHMENTS; ++i) {
        drawBuffers[i] = GL_NONE;
    }
    *drawBufferCount = n;
    *drawBuffer = (n > 0) ? bufs[0] : GL_NONE;

    if (fbo == ctx->state.framebuffer) {
        ctx->state.draw_buffer = *drawBuffer;
        ctx->state.draw_buffer_count = *drawBufferCount;
        for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; ++i) {
            ctx->state.draw_buffers[i] = drawBuffers[i];
        }
    }

    if (fbo) {
        fbo->dirty_bits |= DIRTY_FBO_BINDING;
    }
    STATE(dirty_bits) |= DIRTY_FBO | DIRTY_STATE | DIRTY_RENDER_STATE;
}

void mglNamedFramebufferReadBuffer(GLMContext ctx, GLuint framebuffer, GLenum src)
{
    Framebuffer *fbo = framebuffer ? findFrameBuffer(ctx, framebuffer) : NULL;
    GLenum *readBuffer = NULL;

    if (framebuffer != 0u && !fbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (fbo) {
        if (src != GL_NONE && !mglFramebufferBufferIsColorAttachment(ctx, src)) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
        readBuffer = &fbo->read_buffer;
    } else {
        switch (src) {
            case GL_FRONT:
            case GL_BACK:
            case GL_NONE:
            case GL_FRONT_LEFT:
            case GL_FRONT_RIGHT:
            case GL_BACK_LEFT:
            case GL_BACK_RIGHT:
            case GL_LEFT:
            case GL_RIGHT:
                readBuffer = &ctx->state.default_read_buffer;
                break;
            default:
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
        }
    }

    if (readBuffer && *readBuffer != src) {
        *readBuffer = src;
    }

    if (fbo == ctx->state.readbuffer) {
        ctx->state.read_buffer = src;
    } else if (!fbo && !ctx->state.readbuffer) {
        ctx->state.read_buffer = src;
    }

}

void mglInvalidateNamedFramebufferData(GLMContext ctx, GLuint framebuffer, GLsizei numAttachments, const GLenum *attachments)
{
    if (framebuffer != 0u && !findFrameBuffer(ctx, framebuffer)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (numAttachments < 0 || (numAttachments > 0 && !attachments)) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
}

void mglInvalidateNamedFramebufferSubData(GLMContext ctx, GLuint framebuffer, GLsizei numAttachments, const GLenum *attachments, GLint x, GLint y, GLsizei width, GLsizei height)
{
    if (framebuffer != 0u && !findFrameBuffer(ctx, framebuffer)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (numAttachments < 0 || (numAttachments > 0 && !attachments) || width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    (void)x;
    (void)y;
}

void mglClearNamedFramebufferiv(GLMContext ctx, GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint *value)
{
    Framebuffer *fbo = framebuffer ? findFrameBuffer(ctx, framebuffer) : NULL;
    if (framebuffer != 0u && !fbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    MGLFramebufferBindingSnapshot snapshot;
    mglFramebufferCaptureBindingSnapshot(ctx, &snapshot);
    mglFlushPendingDraws(ctx);
    mglFramebufferUseTemporaryDrawBinding(ctx, fbo);
    mglClearBufferiv(ctx, buffer, drawbuffer, value);
    mglFramebufferRestoreBindingSnapshot(ctx, &snapshot);
}

void mglClearNamedFramebufferuiv(GLMContext ctx, GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint *value)
{
    Framebuffer *fbo = framebuffer ? findFrameBuffer(ctx, framebuffer) : NULL;
    if (framebuffer != 0u && !fbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    MGLFramebufferBindingSnapshot snapshot;
    mglFramebufferCaptureBindingSnapshot(ctx, &snapshot);
    mglFlushPendingDraws(ctx);
    mglFramebufferUseTemporaryDrawBinding(ctx, fbo);
    mglClearBufferuiv(ctx, buffer, drawbuffer, value);
    mglFramebufferRestoreBindingSnapshot(ctx, &snapshot);
}

void mglClearNamedFramebufferfv(GLMContext ctx, GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat *value)
{
    Framebuffer *fbo = framebuffer ? findFrameBuffer(ctx, framebuffer) : NULL;
    if (framebuffer != 0u && !fbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    MGLFramebufferBindingSnapshot snapshot;
    mglFramebufferCaptureBindingSnapshot(ctx, &snapshot);
    mglFlushPendingDraws(ctx);
    mglFramebufferUseTemporaryDrawBinding(ctx, fbo);
    mglClearBufferfv(ctx, buffer, drawbuffer, value);
    mglFramebufferRestoreBindingSnapshot(ctx, &snapshot);
}

void mglClearNamedFramebufferfi(GLMContext ctx, GLuint framebuffer, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil)
{
    Framebuffer *fbo = framebuffer ? findFrameBuffer(ctx, framebuffer) : NULL;
    if (framebuffer != 0u && !fbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    MGLFramebufferBindingSnapshot snapshot;
    mglFramebufferCaptureBindingSnapshot(ctx, &snapshot);
    mglFlushPendingDraws(ctx);
    mglFramebufferUseTemporaryDrawBinding(ctx, fbo);
    mglClearBufferfi(ctx, buffer, drawbuffer, depth, stencil);
    mglFramebufferRestoreBindingSnapshot(ctx, &snapshot);
}

void mglBlitNamedFramebuffer(GLMContext ctx, GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
{
    Framebuffer *readFbo = readFramebuffer ? findFrameBuffer(ctx, readFramebuffer) : NULL;
    Framebuffer *drawFbo = drawFramebuffer ? findFrameBuffer(ctx, drawFramebuffer) : NULL;

    if ((readFramebuffer != 0u && !readFbo) ||
        (drawFramebuffer != 0u && !drawFbo)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    MGLFramebufferBindingSnapshot snapshot;
    mglFramebufferCaptureBindingSnapshot(ctx, &snapshot);
    mglFlushPendingDraws(ctx);
    mglFramebufferUseTemporaryReadDrawBinding(ctx, readFbo, drawFbo);
    mglBlitFramebuffer(ctx, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
    mglFramebufferRestoreBindingSnapshot(ctx, &snapshot);
}

GLenum  mglCheckNamedFramebufferStatus(GLMContext ctx, GLuint framebuffer, GLenum target)
{
    Framebuffer *fbo = NULL;

    switch (target) {
        case GL_FRAMEBUFFER:
        case GL_DRAW_FRAMEBUFFER:
        case GL_READ_FRAMEBUFFER:
            break;
        default:
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, 0);
    }

    if (framebuffer != 0u) {
        fbo = findFrameBuffer(ctx, framebuffer);
        if (!fbo) {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, 0);
        }
    }

    return mglCheckFramebufferStatusForObject(ctx, fbo);
}

void mglGetNamedFramebufferParameteriv(GLMContext ctx, GLuint framebuffer, GLenum pname, GLint *param)
{
    Framebuffer *fbo = NULL;

    if (!param) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (framebuffer != 0u) {
        fbo = findFrameBuffer(ctx, framebuffer);
        if (!fbo) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    switch (pname) {
        case GL_FRAMEBUFFER_DEFAULT_WIDTH:
            *param = fbo ? fbo->default_width : 0;
            break;
        case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
            *param = fbo ? fbo->default_height : 0;
            break;
        case GL_FRAMEBUFFER_DEFAULT_LAYERS:
            *param = fbo ? fbo->default_layers : 0;
            break;
        case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
            *param = fbo ? fbo->default_samples : 0;
            break;
        case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
            *param = fbo ? fbo->default_fixed_sample_locations : GL_TRUE;
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}

void mglCreateRenderbuffers(GLMContext ctx, GLsizei n, GLuint *renderbuffers)
{
    if (n < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (n > 0 && !renderbuffers) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < n; ++i) {
        GLuint name = getNewName(&STATE(renderbuffer_table));
        Renderbuffer *rbo = newRenderbuffer(ctx, name);
        insertHashElement(&STATE(renderbuffer_table), name, rbo);
        renderbuffers[i] = name;
    }
}

void mglNamedRenderbufferStorage(GLMContext ctx, GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height)
{
    Renderbuffer *rbo = findRenderbuffer(ctx, renderbuffer);
    if (renderbuffer == 0u || !rbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = newTexObj(ctx, GL_RENDERBUFFER);
    if (!tex) {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }

    createTextureLevel(ctx, tex, 0, 0, false, internalformat, width, height, 1, 0, 0, NULL, false);
    tex->access = GL_READ_WRITE;
    tex->is_render_target = true;
    rbo->tex = tex;
    rbo->dirty_bits |= DIRTY_RENDBUF_TEX;
    STATE(dirty_bits) |= DIRTY_FBO;
}

void mglNamedRenderbufferStorageMultisample(GLMContext ctx, GLuint renderbuffer, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height)
{
    if (samples < 0 || width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (renderbuffer == 0u || !findRenderbuffer(ctx, renderbuffer)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (samples > 0) {
        fprintf(stderr,
                "MGL WARNING: glNamedRenderbufferStorageMultisample unsupported renderbuffer=%u samples=%d internalformat=0x%x %dx%d\n",
                renderbuffer, samples, internalformat, width, height);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    mglNamedRenderbufferStorage(ctx, renderbuffer, internalformat, width, height);
}

void mglGetNamedRenderbufferParameteriv(GLMContext ctx, GLuint renderbuffer, GLenum pname, GLint *params)
{
    Renderbuffer *saved = ctx->state.renderbuffer;
    Renderbuffer *rbo = findRenderbuffer(ctx, renderbuffer);

    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (renderbuffer == 0u || !rbo) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ctx->state.renderbuffer = rbo;
    mglGetRenderbufferParameteriv(ctx, GL_RENDERBUFFER, pname, params);
    ctx->state.renderbuffer = saved;
}
