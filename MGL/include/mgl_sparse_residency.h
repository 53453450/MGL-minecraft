#ifndef MGL_SPARSE_RESIDENCY_H
#define MGL_SPARSE_RESIDENCY_H

#include <stddef.h>
#include <stdint.h>

#include "mgl_types_texture.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLSparseResidencySlot {
    uint32_t first_rect;
    uint32_t rect_count;
    int32_t level_origin;
    int32_t layer_origin;
    int32_t level_max;
    uint32_t flags;
    float min_lod;
    float max_lod;
    uint32_t min_filter;
    uint32_t mag_filter;
    uint32_t wrap_s;
    uint32_t wrap_t;
    uint32_t wrap_r;
    uint32_t sample_count;
} MGLSparseResidencySlot;

typedef struct MGLSparseResidencyRect {
    int32_t level;
    int32_t x, y, z;
    int32_t w, h, d;
} MGLSparseResidencyRect;

enum { MGL_SPARSE_RESIDENCY_ACTIVE = 1u << 0 };

_Static_assert(sizeof(MGLSparseResidencySlot) == 56u,
               "sparse residency slot ABI must remain 14 words");
_Static_assert(sizeof(MGLSparseResidencyRect) == 28u,
               "sparse residency rect ABI must remain seven words");

void mglSparseResidencyResetSlots(MGLSparseResidencySlot *slots,
                                  size_t slot_count);

/* Serialize the owner/root commit list into the variable-length rect tail.
 * The slot and rect array use independent storage supplied by the caller. */
int mglSparseResidencyBuildSlot(const Texture *tex, const ImageUnit *image,
                                MGLSparseResidencySlot *slot,
                                MGLSparseResidencyRect **rects,
                                size_t *rect_count, size_t *rect_cap);

int mglSparseResidencyRectContains(const MGLSparseResidencyRect *rect,
                                   int32_t level, int32_t x, int32_t y,
                                   int32_t z);

#ifdef __cplusplus
}
#endif

#endif
