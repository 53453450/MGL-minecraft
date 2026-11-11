#include "mgl_sparse_residency.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "CHECK failed: %s:%d\n", __FILE__, __LINE__); return 1; } } while (0)

static Texture make_texture(void) {
    Texture tex;
    memset(&tex, 0, sizeof(tex));
    tex.num_levels = 4;
    tex.params.min_filter = GL_NEAREST;
    tex.params.mag_filter = GL_NEAREST;
    tex.params.max_lod = 3.0f;
    return tex;
}

int main(void) {
    MGLSparseResidencySlot slots[4];
    mglSparseResidencyResetSlots(slots, 4);
    CHECK(slots[2].flags == 0u && slots[2].rect_count == 0u);

    MGLSparseResidencyRect rect = {1, 10, 20, 30, 4, 5, 6};
    CHECK(mglSparseResidencyRectContains(&rect, 1, 10, 20, 30));
    CHECK(mglSparseResidencyRectContains(&rect, 1, 13, 24, 35));
    CHECK(!mglSparseResidencyRectContains(&rect, 0, 10, 20, 30));
    CHECK(!mglSparseResidencyRectContains(&rect, 1, 14, 20, 30));
    CHECK(!mglSparseResidencyRectContains(&rect, 1, 10, 25, 30));
    CHECK(!mglSparseResidencyRectContains(&rect, 1, 10, 20, 36));

    Texture owner = make_texture();
    owner.params.sparse = 1;
    MGLSparseCommit commits[2] = {
        {0, 0, 0, 0, 64, 64, 1},
        {2, 64, 0, 3, 64, 64, 2},
    };
    owner.sparse_commits = commits;
    owner.sparse_commit_count = 2;

    Texture view = make_texture();
    view.view_root = &owner;
    view.view_min_level = 1;
    view.view_min_layer = 2;
    view.params.base_level = 1;
    view.params.sparse = 1;

    MGLSparseResidencyRect *rects = NULL;
    size_t count = 0, cap = 0;
    CHECK(mglSparseResidencyBuildSlot(&view, NULL, &slots[1], &rects, &count, &cap) == 0);
    CHECK(slots[1].flags == MGL_SPARSE_RESIDENCY_ACTIVE);
    CHECK(slots[1].level_origin == 2 && slots[1].layer_origin == 2);
    CHECK(slots[1].level_max == 3);
    CHECK(slots[1].first_rect == 0 && slots[1].rect_count == 2);
    CHECK(count == 2 && rects[1].level == 2 && rects[1].z == 3);

    ImageUnit image;
    memset(&image, 0, sizeof(image));
    image.level = 2;
    image.layer = 4;
    image.layered = 0;
    CHECK(mglSparseResidencyBuildSlot(&view, &image, &slots[2], &rects, &count, &cap) == 0);
    CHECK(slots[2].level_origin == 3 && slots[2].layer_origin == 6);
    CHECK(slots[2].first_rect == 2 && count == 4);

    owner.sparse_commit_count = 0;
    CHECK(mglSparseResidencyBuildSlot(&view, NULL, &slots[3], &rects, &count, &cap) == 0);
    CHECK(slots[3].flags == MGL_SPARSE_RESIDENCY_ACTIVE && slots[3].rect_count == 0);

    free(rects);
    puts("test_sparse_residency: ok");
    return 0;
}
