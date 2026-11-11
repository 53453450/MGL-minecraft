#include "mgl_sparse_residency.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

void mglSparseResidencyResetSlots(MGLSparseResidencySlot *slots,
                                  size_t slot_count)
{
    if (slots) memset(slots, 0, slot_count * sizeof(*slots));
}

int mglSparseResidencyRectContains(const MGLSparseResidencyRect *rect,
                                   int32_t level, int32_t x, int32_t y,
                                   int32_t z)
{
    if (!rect || rect->level != level || rect->w <= 0 || rect->h <= 0 ||
        rect->d <= 0) return 0;
    return x >= rect->x && (int64_t)x < (int64_t)rect->x + rect->w &&
           y >= rect->y && (int64_t)y < (int64_t)rect->y + rect->h &&
           z >= rect->z && (int64_t)z < (int64_t)rect->z + rect->d;
}

int mglSparseResidencyBuildSlot(const Texture *tex, const ImageUnit *image,
                                MGLSparseResidencySlot *slot,
                                MGLSparseResidencyRect **rects,
                                size_t *rect_count, size_t *rect_cap)
{
    if (!slot || !rects || !rect_count || !rect_cap) return -1;
    memset(slot, 0, sizeof(*slot));
    if (!tex) return 0;

    const Texture *owner = tex->view_root ? tex->view_root : tex;
    if (!owner) return -1;
    slot->level_max = owner->num_levels > 0u ? (int32_t)owner->num_levels - 1 : -1;
    slot->min_lod = tex->params.min_lod;
    slot->max_lod = tex->params.max_lod;
    slot->min_filter = tex->params.min_filter;
    slot->mag_filter = tex->params.mag_filter;
    slot->wrap_s = tex->params.wrap_s;
    slot->wrap_t = tex->params.wrap_t;
    slot->wrap_r = tex->params.wrap_r;
    slot->sample_count = tex->samples > 0u ? tex->samples : 1u;

    const uint64_t view_level = tex->view_min_level;
    const uint64_t view_layer = tex->view_min_layer;
    const uint64_t relative_level = image ? image->level : tex->params.base_level;
    const int64_t relative_layer = image && !image->layered ? image->layer : 0;
    if (view_level + relative_level > INT32_MAX ||
        view_layer > INT32_MAX || relative_layer > INT32_MAX ||
        view_layer + (uint64_t)relative_layer > INT32_MAX) return -1;
    slot->level_origin = (int32_t)(view_level + relative_level);
    slot->layer_origin = (int32_t)(view_layer + (uint64_t)relative_layer);

    if (!owner->params.sparse) return 0;
    slot->flags |= MGL_SPARSE_RESIDENCY_ACTIVE;
    const size_t owner_count = owner->sparse_commit_count;
    if (owner_count == 0u) return 0;
    if (owner_count > SIZE_MAX - *rect_count) return -1;
    const size_t needed = *rect_count + owner_count;
    if (*rect_count > UINT32_MAX || needed > UINT32_MAX) return -1;
    if (needed > *rect_cap) {
        size_t cap = *rect_cap ? *rect_cap : 8u;
        while (cap < needed) {
            if (cap > SIZE_MAX / 2u) return -1;
            cap *= 2u;
        }
        if (cap > SIZE_MAX / sizeof(**rects)) return -1;
        MGLSparseResidencyRect *grown = realloc(
            *rects, cap * sizeof(**rects));
        if (!grown) return -1;
        *rects = grown;
        *rect_cap = cap;
    }

    slot->first_rect = (uint32_t)*rect_count;
    slot->rect_count = (uint32_t)owner_count;
    for (size_t i = 0; i < owner_count; ++i) {
        const MGLSparseCommit *src = &owner->sparse_commits[i];
        MGLSparseResidencyRect *dst = &(*rects)[*rect_count + i];
        *dst = (MGLSparseResidencyRect){
            src->level, src->x, src->y, src->z, src->w, src->h, src->d
        };
    }
    *rect_count = needed;
    return 0;
}
