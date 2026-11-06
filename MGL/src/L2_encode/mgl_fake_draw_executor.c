/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * CPU-side fake DrawExecutor for ARCHITECTURE_AUDIT R4.  Encodes nothing to
 * Metal; used by arch-correctness to prove DrawState + typed handles can be
 * consumed without the ObjC/Metal owner graph.
 */

#include "mgl_backend_handles.h"

#include <stdlib.h>

typedef struct MGLFakeDrawExecutor {
    MGLDrawExecutorVTable vtable;
    unsigned encode_count;
    unsigned last_index_count;
} MGLFakeDrawExecutor;

static int mglFakeEncodeIndexed(void *executor, const void *draw_state,
                                MGLBufferHandle vb, MGLBufferHandle ib,
                                uint32_t index_count)
{
    MGLFakeDrawExecutor *ex = (MGLFakeDrawExecutor *)executor;
    if (!ex || !draw_state || index_count == 0u) {
        return -1;
    }
    if (!mglHandleIsLive(vb.obj, vb.generation, vb.generation) ||
        !mglHandleIsLive(ib.obj, ib.generation, ib.generation)) {
        return -1;
    }
    ex->encode_count++;
    ex->last_index_count = index_count;
    return 0;
}

static void mglFakeDestroy(void *executor)
{
    free(executor);
}

void *mglFakeDrawExecutorCreate(void)
{
    MGLFakeDrawExecutor *ex =
        (MGLFakeDrawExecutor *)calloc(1, sizeof(MGLFakeDrawExecutor));
    if (!ex) {
        return NULL;
    }
    ex->vtable.encode_indexed = mglFakeEncodeIndexed;
    ex->vtable.flush_draw_buffer = NULL;
    ex->vtable.swap_buffers = NULL;
    ex->vtable.clear_buffer = NULL;
    ex->vtable.draw_arrays = NULL;
    ex->vtable.draw_elements = NULL;
    ex->vtable.blit_framebuffer = NULL;
    ex->vtable.dispatch_compute = NULL;
    ex->vtable.generate_mipmaps = NULL;
    ex->vtable.destroy = mglFakeDestroy;
    return ex;
}

const MGLDrawExecutorVTable *mglFakeDrawExecutorVTable(void *executor)
{
    MGLFakeDrawExecutor *ex = (MGLFakeDrawExecutor *)executor;
    return ex ? &ex->vtable : NULL;
}

unsigned mglFakeDrawExecutorEncodeCount(void *executor)
{
    MGLFakeDrawExecutor *ex = (MGLFakeDrawExecutor *)executor;
    return ex ? ex->encode_count : 0u;
}
