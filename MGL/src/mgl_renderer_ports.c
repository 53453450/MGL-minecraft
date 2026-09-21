/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_renderer_ports.c — C ports of renderer accessors (ObjC-zeroing T4).
 *
 * mglRendererAttachmentTextureFor() is the exact behaviour of the Objective-C
 * -[MGLRenderer framebufferAttachmentTexture:] it replaced: a renderbuffer
 * attachment resolves through rbo->tex, a texture attachment through the cached
 * buf.tex or a findTexture() lookup that is then cached, and both a NULL
 * attachment and an attachment without a texture are reported on stderr.
 * -[MGLRenderer framebufferAttachmentTexture:] forwards here, so the ~27 ObjC
 * call sites keep their behaviour.
 */

#include "mgl_renderer_ports.h"
#include "mgl_render.h"           /* command-buffer snapshot, MDI scratch owner */
#include "mgl_renderer_backend.h" /* mglRendererBackendGetDevice, sampler cache */
#include "mgl_texture_sampler.h"  /* mglTextureCreateSamplerForTexParam */
#include "mgl_batch_public.h"     /* mgl_batch_replay_fill_sampler_params */
#include "mgl_metal_ref.h"        /* mglReleaseMetalObjNoNull */
#include "mgl_draw_encode.h"      /* mglDrawCommandElementBuffer */
#include "draw_command.h"         /* MGLDrawCommand */
#include "error.h"                /* mglDispatchError */

#include <stdint.h>
#include <stdio.h>

/* Defined in textures.c (C); declared here like framebuffers.c does. */
extern Texture *findTexture(GLMContext ctx, GLuint texture);

Texture *mglRendererAttachmentTextureFor(GLMContext ctx, FBOAttachment *att)
{
    Texture *tex = NULL;

    if (!att) {
        fprintf(stderr,
                "MGL ERROR: framebufferAttachmentTexture called with NULL attachment\n");
        return NULL;
    }

    if (mglRenderTargetIsRenderbuffer((uint32_t)att->textarget)) {
        if (att->buf.rbo) {
            tex = att->buf.rbo->tex;
        }
    } else {
        tex = att->buf.tex;
        if (!tex && att->texture != 0 && ctx) {
            tex = findTexture(ctx, att->texture);
            if (tex) {
                att->buf.tex = tex;
            }
        }
    }
    if (!tex) {
        fprintf(stderr,
                "MGL WARN: framebuffer attachment has no texture (target=0x%x)\n",
                (unsigned)att->textarget);
    }

    return tex;
}

/* The manager's command state, reached through the state areas.  Replaces the
 * former mglRendererCommandStatePort wrapper (the areas already carry the
 * pointer, so the shim does not need an entry point of its own). */
const MGLCommandState *mglRendererCommandStateFor(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    return areas.command;
}

/* Body of the former -[MGLRenderPassManager mdiArgumentScratchBufferWithDevice:
 * length:offset:].  The arena itself is C++ (mglRenderAllocateMDIScratch) and
 * the owner pointer is a field of the command state, so no Objective-C message
 * is involved: the returned buffer is borrowed, exactly as before. */
void *mglRendererMdiScratchBuffer(void *renderer, uint64_t length,
                                  uint64_t *offset_out)
{
    if (offset_out) {
        *offset_out = 0;
    }
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs || length == 0 ||
        !mglRendererBackendGetDevice(areas.backend)) {
        return NULL;
    }

    MGLRenderCommandBufferState commandBufferState = {0};
    if (!mglRenderCommandBufferOwnerHasState(cs->currentCommandBufferOwner,
                                             &commandBufferState)) {
        return NULL;
    }

    if (!cs->mdiArgsScratchOwner &&
        mglRenderCreateMDIScratchOwner(&cs->mdiArgsScratchOwner) != 0) {
        return NULL;
    }
    void *buffer = NULL;
    uint64_t offset = 0;
    uint64_t capacity = 0;
    if (mglRenderAllocateMDIScratch(cs->mdiArgsScratchOwner, length, 256u,
                                    &buffer, &offset, &capacity) != 0 ||
        !buffer) {
        return NULL;
    }
    if (offset_out) {
        *offset_out = offset;
    }
    return buffer;
}

/* === element-buffer resolve / processBuffer / sampler-snapshot ============
 *
 * Bodies of the former -[MGLRenderer resolveElementBuffer*:], -processBuffer:
 * and -samplerStateForSnapshotKey:.  Every dependency was already C
 * (getElementBuffer, mglRendererGetValidatedBuffer,
 * mglRenderUpdateDirtyBaseBufferList, mglRenderUpdateDirtyBuffer,
 * mglRenderBindBufferStorage, the sampler creation and the backend snapshot
 * cache); only the Objective-C `id` handles were in the way, and those are
 * `void *` here.
 *
 * Each of these functions is declared next to its definition in
 * MGLRenderer+Draw_Private.h / MGLRenderer+DrawSupportUtil.h, which are
 * Objective-C headers, so the C prototypes are repeated here the way
 * framebuffers.c already does for findTexture(). */

/* NSUInteger is `unsigned long` on the 64-bit targets MGL builds for. */
extern Buffer *getElementBuffer(GLMContext ctx);
extern Buffer *mglRendererGetValidatedBuffer(GLMContext ctx, Buffer *candidate,
                                             const char *where,
                                             unsigned long slot);
extern int mglRenderUpdateDirtyBuffer(Buffer *ptr, char *err, size_t errcap);
extern int mglRenderBindBufferStorage(Buffer *buffer, char *err, size_t errcap);

int mglRendererProcessBuffer(void *renderer, Buffer *buffer)
{
    if (!buffer) {
        fprintf(stderr, "MGL Error: processBuffer failed\n");
        return 0;
    }

    if (buffer->data.mtl_data == NULL) {
        /* The bind takes METAL_LOCK, so it stays the shim port. */
        mglRendererBindMTLBuffer(renderer, buffer);
        if (buffer->data.mtl_data == NULL) {
            return 0;
        }
    }

    if (buffer->data.dirty_bits) {
        char error[256] = {0};
        int result = mglRenderUpdateDirtyBuffer(buffer, error, sizeof(error));
        if (result != MGL_RENDER_BUFFER_OPERATION_HANDLED) {
            fprintf(stderr,
                    "MGL BUFFER ERROR: Metal-cpp dirty update failed buffer=%u: %s\n",
                    buffer ? (unsigned)buffer->name : 0u,
                    error[0] ? error : "?");
            return 0;
        }
    }

    return 1;
}

int mglRendererResolveElementBufferForDraw(void *renderer, const char *label,
                                           GLMContext ctx, Buffer **gl_out,
                                           void **mtl_out)
{
    Buffer *gl_element_buffer = getElementBuffer(ctx);
    return mglRendererResolveElementBuffer(renderer, gl_element_buffer, label,
                                           ctx, gl_out, mtl_out);
}

int mglRendererResolveElementBufferForCommand(void *renderer, const void *command,
                                              const char *label, GLMContext ctx,
                                              Buffer **gl_out, void **mtl_out)
{
    const MGLDrawCommand *cmd = (const MGLDrawCommand *)command;
    Buffer *gl_element_buffer = NULL;
    if (cmd && cmd->element_buffer_name) {
        gl_element_buffer = mglRendererGetValidatedBuffer(
            ctx, mglDrawCommandElementBuffer(ctx, cmd),
            label ? label : "deferred indexed draw", 0);
        if (!gl_element_buffer) {
            return 0;
        }
    } else {
        gl_element_buffer = getElementBuffer(ctx);
    }

    return mglRendererResolveElementBuffer(renderer, gl_element_buffer, label,
                                           ctx, gl_out, mtl_out);
}

int mglRendererResolveElementBuffer(void *renderer, Buffer *gl_element_buffer,
                                    const char *label, GLMContext ctx,
                                    Buffer **gl_out, void **mtl_out)
{
    (void)renderer;
    if (!gl_element_buffer) {
        fprintf(stderr,
                "MGL WARNING: %s skipped because no element array buffer is bound\n",
                label ? label : "indexed draw");
        if (ctx) {
            mglDispatchError(ctx, label ? label : "resolveElementBuffer",
                             (GLenum)mglRenderErrorInvalidOperation());
        }
        return 0;
    }

    if (!mglRendererProcessBuffer(renderer, gl_element_buffer)) {
        return 0;
    }

    void *indexBuffer = gl_element_buffer->data.mtl_data;
    if (!indexBuffer) {
        fprintf(stderr,
                "MGL WARNING: %s skipped because element buffer %u has no Metal buffer\n",
                label ? label : "indexed draw",
                gl_element_buffer->name);
        return 0;
    }

    if (gl_out) {
        *gl_out = gl_element_buffer;
    }
    if (mtl_out) {
        *mtl_out = indexBuffer;
    }
    return 1;
}

/* Body of the former -[MGLRenderer samplerStateForSnapshotKey:], moved from the
 * shim: the backend handle comes from the state areas and every other step was
 * already C.  The return stays unretained -- the backend snapshot cache owns the
 * state it hands back. */
void *mglRendererSamplerStateForSnapshotKey(void *renderer, const void *key)
{
    if (!renderer || !key) {
        return NULL;
    }
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);

    void *cachedState = NULL;
    int cacheResult = mglRendererBackendGetSamplerSnapshotState(
        areas.backend, (const MGLSamplerSnapshotKey *)key, &cachedState);
    if (cacheResult == 1) {
        return cachedState;
    }
    if (cacheResult < 0) {
        return NULL;
    }

    TextureParameter params;
    mgl_batch_replay_fill_sampler_params((const MGLSamplerSnapshotKey *)key,
                                         &params);
    /* +1 from the C sampler creation; the backend cache below takes ownership
     * through the Put call, so release our reference again. */
    void *state = mglTextureCreateSamplerForTexParam(
        &params, ((const MGLSamplerSnapshotKey *)key)->target);
    if (!state) {
        return NULL;
    }
    if (mglRendererBackendPutSamplerSnapshotState(
            areas.backend, (const MGLSamplerSnapshotKey *)key, state) != 0) {
        mglReleaseMetalObjNoNull(state);
        return NULL;
    }
    mglReleaseMetalObjNoNull(state);   /* the backend snapshot cache retains it */
    return state;
}

/* Body of the former -[MGLRenderer bindMTLBuffer:] (+ its Locked half).
 * METAL_LOCK() is only MGL_ASSERT_GL_THREAD() (the mutex was removed long ago),
 * so the whole bind is C: mglRenderBindBufferStorage plus the same diagnostic. */
void mglRendererBindMTLBuffer(void *renderer, Buffer *buffer)
{
    (void)renderer;
    char bindError[256] = {0};
    int bindResult = mglRenderBindBufferStorage(buffer, bindError, sizeof(bindError));
    if (bindResult != MGL_RENDER_BUFFER_BOUND) {
        fprintf(stderr, "MGL ERROR: Metal-cpp buffer bind failed buffer=%u: %s\n",
                buffer ? (unsigned)buffer->name : 0u,
                bindError[0] ? bindError : "?");
    }
}
