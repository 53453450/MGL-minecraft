/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Typed backend handles (ARCHITECTURE_AUDIT R4).
 * Opaque generation-tagged references so owners can detect use-after-delete
 * without exposing Metal/ObjC types across the C GL facade.
 */

#ifndef MGL_BACKEND_HANDLES_H
#define MGL_BACKEND_HANDLES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLBufferHandle {
    void *obj;           /* backend-owned; NULL if unbound */
    uint64_t generation; /* bumped on delete/recreate */
} MGLBufferHandle;

typedef struct MGLTextureHandle {
    void *obj;
    uint64_t generation;
} MGLTextureHandle;

typedef struct MGLPipelineHandle {
    void *obj;
    uint64_t generation;
} MGLPipelineHandle;

/* Executor entry used by fake/CPU tests and the Metal path. */
typedef struct MGLDrawExecutorVTable {
    int (*encode_indexed)(void *executor, const void *draw_state,
                          MGLBufferHandle vb, MGLBufferHandle ib,
                          uint32_t index_count);
    /* Present / transfer entry points (Metal or fake). Context is GLMContext. */
    void (*flush_draw_buffer)(void *executor, void *context);
    void (*swap_buffers)(void *executor, void *context);
    void (*clear_buffer)(void *executor, void *context, uint32_t type,
                         uint32_t mask);
    void (*draw_arrays)(void *executor, void *context, uint32_t mode,
                        int32_t first, int32_t count);
    void (*draw_elements)(void *executor, void *context, uint32_t mode,
                          int32_t count, uint32_t type, const void *indices);
    void (*blit_framebuffer)(void *executor, void *context,
                             int src_x0, int src_y0, int src_x1, int src_y1,
                             int dst_x0, int dst_y0, int dst_x1, int dst_y1,
                             uint32_t mask, uint32_t filter);
    void (*dispatch_compute)(void *executor, void *context,
                             uint32_t groups_x, uint32_t groups_y,
                             uint32_t groups_z);
    void (*generate_mipmaps)(void *executor, void *context, void *texture);
    void (*destroy)(void *executor);
} MGLDrawExecutorVTable;

static inline int mglHandleIsLive(const void *obj, uint64_t gen,
                                  uint64_t expected_gen)
{
    return obj != NULL && gen == expected_gen;
}

/* CPU fake executor (mgl_fake_draw_executor.c) — no Metal dependency. */
void *mglFakeDrawExecutorCreate(void);
const MGLDrawExecutorVTable *mglFakeDrawExecutorVTable(void *executor);
unsigned mglFakeDrawExecutorEncodeCount(void *executor);

/* Metal adapter (mgl_metal_draw_executor.c) — routes via Compat bridge. */
void *mglMetalDrawExecutorCreate(void);
const MGLDrawExecutorVTable *mglMetalDrawExecutorVTable(void *executor);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BACKEND_HANDLES_H */
