/* SPDX-License-Identifier: LGPL-3.0-only */

#ifndef MGL_BATCH_ARENA_H
#define MGL_BATCH_ARENA_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLBatchArenaChunk MGLBatchArenaChunk;

typedef struct MGLBatchArena {
    MGLBatchArenaChunk *head;
    MGLBatchArenaChunk *current;
    size_t initial_capacity;
    int enabled;
} MGLBatchArena;

bool mglInitBatchArena(MGLBatchArena *arena, size_t initial_capacity);
void *mglBatchArenaAlloc(MGLBatchArena *arena, size_t size);
void mglResetBatchArena(MGLBatchArena *arena);
void mglDestroyBatchArena(MGLBatchArena *arena);

#ifdef __cplusplus
}
#endif

#endif
