/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Metal DrawExecutor adapter (ARCHITECTURE_AUDIT R4).
 * Resource-owner backend routes present/transfer/compute through this VTable
 * instead of calling ObjC Compat selectors directly.
 */

#include "mgl_backend_handles.h"
#include "mgl_renderer_compat_bridge.h"

#include <stdlib.h>

typedef struct MGLMetalDrawExecutor {
    MGLDrawExecutorVTable vtable;
} MGLMetalDrawExecutor;

static void mglMetalFlushDrawBuffer(void *executor, void *context)
{
    (void)executor;
    if (context)
        mglRendererCompatFlushDrawBuffer((GLMContext)context);
}

static void mglMetalSwapBuffers(void *executor, void *context)
{
    (void)executor;
    if (context)
        mglRendererCompatSwapBuffers((GLMContext)context);
}

static void mglMetalClearBuffer(void *executor, void *context, uint32_t type,
                                uint32_t mask)
{
    (void)executor;
    if (context)
        mglRendererCompatClearBuffer((GLMContext)context, type, mask);
}

static void mglMetalDrawArrays(void *executor, void *context, uint32_t mode,
                               int32_t first, int32_t count)
{
    (void)executor;
    if (context)
        mglRendererCompatDrawArrays((GLMContext)context, mode, first, count);
}

static void mglMetalDrawElements(void *executor, void *context, uint32_t mode,
                                 int32_t count, uint32_t type,
                                 const void *indices)
{
    (void)executor;
    if (context)
        mglRendererCompatDrawElements((GLMContext)context, mode, count, type,
                                      indices);
}

static void mglMetalBlitFramebuffer(void *executor, void *context,
                                    int src_x0, int src_y0, int src_x1,
                                    int src_y1, int dst_x0, int dst_y0,
                                    int dst_x1, int dst_y1, uint32_t mask,
                                    uint32_t filter)
{
    (void)executor;
    if (context)
        mglRendererCompatBlitFramebuffer((GLMContext)context, src_x0, src_y0,
                                         src_x1, src_y1, dst_x0, dst_y0,
                                         dst_x1, dst_y1, mask, filter);
}

static void mglMetalDispatchCompute(void *executor, void *context,
                                    uint32_t groups_x, uint32_t groups_y,
                                    uint32_t groups_z)
{
    (void)executor;
    if (context)
        mglRendererCompatDispatchCompute((GLMContext)context, groups_x,
                                         groups_y, groups_z);
}

static void mglMetalGenerateMipmaps(void *executor, void *context,
                                    void *texture)
{
    (void)executor;
    if (context && texture)
        mglRendererCompatGenerateMipmaps((GLMContext)context,
                                         (Texture *)texture);
}

static int mglMetalEncodeIndexed(void *executor, const void *draw_state,
                                 MGLBufferHandle vb, MGLBufferHandle ib,
                                 uint32_t index_count)
{
    /* Indexed encode remains on the ObjC draw path; this stub exists so the
     * VTable is complete for CPU tests that only exercise flush/swap. */
    (void)executor;
    (void)draw_state;
    (void)vb;
    (void)ib;
    (void)index_count;
    return -1;
}

static void mglMetalDestroy(void *executor)
{
    free(executor);
}

void *mglMetalDrawExecutorCreate(void)
{
    MGLMetalDrawExecutor *ex =
        (MGLMetalDrawExecutor *)calloc(1, sizeof(MGLMetalDrawExecutor));
    if (!ex)
        return NULL;
    ex->vtable.encode_indexed = mglMetalEncodeIndexed;
    ex->vtable.flush_draw_buffer = mglMetalFlushDrawBuffer;
    ex->vtable.swap_buffers = mglMetalSwapBuffers;
    ex->vtable.clear_buffer = mglMetalClearBuffer;
    ex->vtable.draw_arrays = mglMetalDrawArrays;
    ex->vtable.draw_elements = mglMetalDrawElements;
    ex->vtable.blit_framebuffer = mglMetalBlitFramebuffer;
    ex->vtable.dispatch_compute = mglMetalDispatchCompute;
    ex->vtable.generate_mipmaps = mglMetalGenerateMipmaps;
    ex->vtable.destroy = mglMetalDestroy;
    return ex;
}

const MGLDrawExecutorVTable *mglMetalDrawExecutorVTable(void *executor)
{
    MGLMetalDrawExecutor *ex = (MGLMetalDrawExecutor *)executor;
    return ex ? &ex->vtable : NULL;
}
