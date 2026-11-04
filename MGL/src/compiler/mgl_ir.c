/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

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
 * mgl_ir.c
 * MGL - shader intermediate IR: type system + std140/std430 layout.
 */

#include "mgl_ir.h"

#include <stdlib.h>
#include <string.h>

/* Structural recursion cap (defensive; GLSL type graphs are trees). */
#define MGL_IR_MAX_DEPTH 64

static uint32_t scalar_bytes(MGLIRScalar s)
{
    switch (s) {
    case MGLIR_SCALAR_BOOL:
    case MGLIR_SCALAR_INT:
    case MGLIR_SCALAR_UINT:
    case MGLIR_SCALAR_FLOAT:
        return 4;
    case MGLIR_SCALAR_DOUBLE:
        return 8;
    case MGLIR_SCALAR_HALF:
        return 2;
    default:
        return 0;
    }
}

/* Layout arithmetic is part of the shader ABI.  Do it in 64 bits and fail
 * closed when a user supplied array or nested struct cannot be represented by
 * the 32-bit offsets consumed by the AIR backend. */
static int align_up_checked(uint32_t v, uint32_t a, uint32_t *out)
{
    if (!out || a == 0) {
        return -1;
    }
    if (a <= 1) {
        *out = v;
        return 0;
    }
    uint64_t aligned = ((uint64_t)v + (uint64_t)a - 1u) &
                       ~((uint64_t)a - 1u);
    if (aligned > UINT32_MAX) {
        return -1;
    }
    *out = (uint32_t)aligned;
    return 0;
}

static int add_checked(uint32_t a, uint32_t b, uint32_t *out)
{
    uint64_t value = (uint64_t)a + (uint64_t)b;
    if (!out || value > UINT32_MAX) {
        return -1;
    }
    *out = (uint32_t)value;
    return 0;
}

static int mul_checked(uint32_t a, uint32_t b, uint32_t *out)
{
    uint64_t value = (uint64_t)a * (uint64_t)b;
    if (!out || value > UINT32_MAX) {
        return -1;
    }
    *out = (uint32_t)value;
    return 0;
}

/* Base alignment of an N-component vector of scalar size `s`.
 * std140/std430 both use: 2*N for 2-comp, 4*N for 3/4-comp. */
static uint32_t vector_align(uint32_t comps, uint32_t s)
{
    if (comps <= 1) {
        return s;
    }
    if (comps == 2) {
        return 2 * s;
    }
    return 4 * s;
}

static int is_std140(MGLIRLayoutStd l)
{
    return l == MGLIR_LAYOUT_STD140 || l == MGLIR_LAYOUT_SHARED;
}

static MGLIRLayoutCache *layout_cache_for(MGLIRType *type,
                                          MGLIRLayoutStd layout)
{
    if (!type || layout < MGLIR_LAYOUT_NONE ||
        layout >= MGLIR_LAYOUT_CACHE_COUNT) {
        return NULL;
    }
    return &type->layout_cache[(unsigned)layout];
}

/* Keep the historical fields usable by existing backend code while the
 * per-standard caches remain the ownership boundary for immutable results. */
static void activate_layout_cache(MGLIRType *type, MGLIRLayoutStd layout,
                                  const MGLIRLayoutCache *cache)
{
    if (!type || !cache) {
        return;
    }
    type->layout = cache->info;
    type->layout_valid = cache->valid;
    type->layout_standard = layout;
    type->member_offsets = cache->member_offsets;
}

static int layout_type(MGLIRType *type, MGLIRLayoutStd layout, uint32_t depth,
                       MGLIRLayoutInfo *info)
{
    if (!type || depth > MGL_IR_MAX_DEPTH) {
        return -1;
    }
    MGLIRLayoutCache *cache = layout_cache_for(type, layout);
    if (!cache) {
        return -1;
    }
    if (cache->valid) {
        activate_layout_cache(type, layout, cache);
        if (info) {
            *info = cache->info;
        }
        return 0;
    }

    MGLIRLayoutInfo r;
    memset(&r, 0, sizeof(r));
    uint32_t *computed_member_offsets = NULL;

    switch (type->kind) {
    case MGLIR_TYPE_SCALAR: {
        uint32_t s = scalar_bytes(type->scalar);
        if (s == 0) {
            return -1;
        }
        r.size = s;
        r.alignment = s;
        break;
    }
    case MGLIR_TYPE_VECTOR: {
        uint32_t s = scalar_bytes(type->scalar);
        if (s == 0 || type->cols > 4) {
            return -1;
        }
        /* std140 vector base align is 16 for 3/4 comp, 8 for 2. */
        if (mul_checked(s, type->cols, &r.size) != 0) {
            return -1;
        }
        r.alignment = vector_align(type->cols, s);
        break;
    }
    case MGLIR_TYPE_MATRIX: {
        /* A matrix behaves as an array of vectors: N columns (column-major,
         * GL default) or rows (row-major flagged). */
        uint32_t s = scalar_bytes(type->scalar);
        uint32_t vec_comps = type->row_major ? type->cols : type->rows;
        uint32_t count = type->row_major ? type->rows : type->cols;
        if (s == 0 || vec_comps > 4 || count == 0) {
            return -1;
        }
        uint32_t base = vector_align(vec_comps, s);
        uint32_t stride = base;
        if (is_std140(layout) && align_up_checked(base, 16, &stride) != 0) {
            return -1;
        }
        /* The layout pass stores every vector at the full stride in both
         * std140 and std430: mat3 std430 = 48.
         * std140/shared: matrix base alignment follows the array-of-vector
         * rule (GL 4.6 §7.6.2.2 (3)+(4)) — round vector align up to 16. */
        if (mul_checked(count, stride, &r.size) != 0) {
            return -1;
        }
        r.alignment = is_std140(layout) ? stride : base;
        r.matrix_stride = stride;
        break;
    }
    case MGLIR_TYPE_ARRAY: {
        MGLIRLayoutInfo e;
        if (!type->elem_type ||
            layout_type(type->elem_type, layout, depth + 1, &e) != 0) {
            return -1;
        }
        /* std140: array element alignment/stride rounds up to 16 for
         * scalar/vector elements.  Struct elements carry their own alignment
         * (already potentially >16); round only if below 16 in std140. */
        uint32_t align = e.alignment;
        if (is_std140(layout)) {
            if (align_up_checked(align, 16, &align) != 0) {
                return -1;
            }
        }
        uint32_t stride = 0;
        if (align_up_checked(e.size, align, &stride) != 0) {
            return -1;
        }
        if (is_std140(layout)) {
            /* std140 requires array element stride be 16-aligned. */
            if (align_up_checked(stride, 16, &stride) != 0) {
                return -1;
            }
        }
        r.alignment = align;
        r.array_stride = stride;
        if (type->array_size == 0) {
            /* runtime array (SSBO tail): it does not contribute space to the
             * enclosing struct; callers must size it via array_stride */
            r.size = 0;
        } else {
            /* The layout pass stores every element at the full stride:
             * float[3] std140 = 48 (=3*16), vec3[2] std430 = 32. */
            if (mul_checked(type->array_size, stride, &r.size) != 0) {
                return -1;
            }
        }
        break;
    }
    case MGLIR_TYPE_STRUCT: {
        uint32_t *member_offsets = NULL;
        if (type->member_count > 0) {
            if ((size_t)type->member_count >
                SIZE_MAX / sizeof(*member_offsets)) {
                return -1;
            }
            member_offsets = (uint32_t *)calloc(type->member_count,
                                                sizeof(uint32_t));
            if (!member_offsets) {
                return -1;
            }
        }
        uint32_t offset = 0;
        uint32_t max_align = 1;
        for (uint32_t i = 0; i < type->member_count; i++) {
            MGLIRLayoutInfo m;
            if (layout_type(type->members[i], layout, depth + 1, &m) != 0) {
                free(member_offsets);
                return -1;
            }
            if (align_up_checked(offset, m.alignment, &offset) != 0 ||
                add_checked(offset, m.size, &offset) != 0) {
                free(member_offsets);
                return -1;
            }
            member_offsets[i] = offset - m.size;
            if (m.alignment > max_align) {
                max_align = m.alignment;
            }
        }
        r.alignment = max_align;
        if (is_std140(layout) &&
            align_up_checked(max_align, 16, &r.alignment) != 0) {
            free(member_offsets);
            return -1;
        }
        if (align_up_checked(offset, r.alignment, &r.size) != 0) {
            free(member_offsets);
            return -1;
        }
        computed_member_offsets = member_offsets;
        break;
    }
    case MGLIR_TYPE_SAMPLER:
    case MGLIR_TYPE_IMAGE:
    case MGLIR_TYPE_ATOMIC_COUNTER:
        r.size = 8; /* opaque handle size in a uniform block */
        r.alignment = 8;
        break;
    default:
        return -1;
    }

    /* Publish only after the complete subtree succeeds.  Child caches may
     * already contain a valid result, but this node never exposes a partial
     * member-offset array. */
    cache->info = r;
    cache->member_offsets = computed_member_offsets;
    cache->valid = 1;
    activate_layout_cache(type, layout, cache);
    if (info) {
        *info = r;
    }
    return 0;
}

MGLIRType *mglIRTypeScalar(MGLIRScalar s)
{
    MGLIRType *t = (MGLIRType *)calloc(1, sizeof(*t));
    if (!t) {
        return NULL;
    }
    t->kind = MGLIR_TYPE_SCALAR;
    t->scalar = s;
    t->rows = 1;
    t->cols = 1;
    return t;
}

MGLIRType *mglIRTypeVector(MGLIRScalar s, uint32_t comps)
{
    if (comps < 1 || comps > 4 || s == MGLIR_SCALAR_VOID) {
        return NULL;
    }
    MGLIRType *t = mglIRTypeScalar(s);
    if (!t) {
        return NULL;
    }
    t->kind = MGLIR_TYPE_VECTOR;
    t->cols = comps;
    return t;
}

MGLIRType *mglIRTypeMatrix(MGLIRScalar s, uint32_t cols, uint32_t rows)
{
    if (cols < 1 || cols > 4 || rows < 1 || rows > 4 || s == MGLIR_SCALAR_VOID) {
        return NULL;
    }
    MGLIRType *t = mglIRTypeScalar(s);
    if (!t) {
        return NULL;
    }
    t->kind = MGLIR_TYPE_MATRIX;
    t->cols = cols;
    t->rows = rows;
    return t;
}

MGLIRType *mglIRTypeArray(MGLIRType *elem, uint32_t size)
{
    if (!elem) {
        return NULL;
    }
    MGLIRType *t = (MGLIRType *)calloc(1, sizeof(*t));
    if (!t) {
        return NULL;
    }
    t->kind = MGLIR_TYPE_ARRAY;
    t->elem_type = elem;
    t->array_size = size;
    return t;
}

MGLIRType *mglIRTypeRuntimeArray(MGLIRType *elem)
{
    return mglIRTypeArray(elem, 0);
}

MGLIRType *mglIRTypeStruct(MGLIRType *const *members, const char *const *names,
                           uint32_t count, const char *name)
{
    if (!members || count == 0) {
        return NULL;
    }
    MGLIRType *t = (MGLIRType *)calloc(1, sizeof(*t));
    if (!t) {
        return NULL;
    }
    t->kind = MGLIR_TYPE_STRUCT;
    t->member_count = count;
    t->members = (MGLIRType **)calloc(count, sizeof(MGLIRType *));
    t->member_names = (char **)calloc(count, sizeof(char *));
    if (!t->members || !t->member_names) {
        free(t->members);
        free(t->member_names);
        free(t);
        return NULL;
    }
    for (uint32_t i = 0; i < count; i++) {
        t->members[i] = members[i];
        t->member_names[i] = names[i] ? strdup(names[i]) : NULL;
        if (names[i] && !t->member_names[i]) {
            goto oom;
        }
    }
    t->name = name ? strdup(name) : NULL;
    if (name && !t->name) {
        goto oom;
    }
    return t;
oom:
    for (uint32_t j = 0; j < t->member_count; j++) {
        free(t->member_names[j]);
    }
    for (uint32_t j = 0; j < t->member_count; j++) {
        mglIRTypeDestroy(t->members[j]);
    }
    free(t->member_names);
    free(t->members);
    free(t);
    return NULL;
}

MGLIRType *mglIRTypeSampler(MGLIRTexKind kind, MGLIRScalar storage, int depth)
{
    MGLIRType *t = (MGLIRType *)calloc(1, sizeof(*t));
    if (!t) {
        return NULL;
    }
    t->kind = MGLIR_TYPE_SAMPLER;
    t->tex_kind = kind;
    t->tex_depth = depth ? 1 : 0;
    /* For sampler/image the scalar type records the texel type. */
    t->tex_storage = storage;
    return t;
}

MGLIRType *mglIRTypeImage(MGLIRTexKind kind, MGLIRScalar storage, uint32_t gl_format)
{
    MGLIRType *t = mglIRTypeSampler(kind, storage, 0);
    if (!t) {
        return NULL;
    }
    t->kind = MGLIR_TYPE_IMAGE;
    t->tex_format = gl_format;
    return t;
}

MGLIRType *mglIRTypeAtomicCounter(void)
{
    MGLIRType *t = (MGLIRType *)calloc(1, sizeof(*t));
    if (!t) {
        return NULL;
    }
    t->kind = MGLIR_TYPE_ATOMIC_COUNTER;
    t->scalar = MGLIR_SCALAR_UINT;
    return t;
}

int mglIRComputeLayout(MGLIRType *type, MGLIRLayoutStd layout, uint32_t *size)
{
    if (!type) {
        return -1;
    }
    if (layout != MGLIR_LAYOUT_NONE &&
        layout != MGLIR_LAYOUT_STD140 && layout != MGLIR_LAYOUT_STD430 &&
        layout != MGLIR_LAYOUT_SHARED && layout != MGLIR_LAYOUT_PACKED) {
        return -1;
    }
    MGLIRLayoutInfo info;
    if (layout_type(type, layout, 0, &info) != 0) {
        return -1;
    }
    if (size) {
        *size = info.size;
    }
    return 0;
}

void mglIRTypeDestroy(MGLIRType *t)
{
    if (!t) {
        return;
    }
    if (t->kind == MGLIR_TYPE_ARRAY) {
        mglIRTypeDestroy(t->elem_type);
    } else if (t->kind == MGLIR_TYPE_STRUCT) {
        for (uint32_t i = 0; i < t->member_count; i++) {
            mglIRTypeDestroy(t->members[i]);
        }
        for (uint32_t i = 0; i < t->member_count; i++) {
            free(t->member_names[i]);
        }
    }
    free((void *)t->name);
    free(t->members);
    free(t->member_names);
    for (unsigned i = 0; i < MGLIR_LAYOUT_CACHE_COUNT; i++) {
        free(t->layout_cache[i].member_offsets);
    }
    free(t);
}
