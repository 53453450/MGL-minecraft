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
 * mgl_ir.h
 * MGL - Self-owned shader intermediate IR.
 *
 * M0 scope: type system + std140/std430 layout computation for the
 * self-written GLSL frontend (see docs/AIR_SHADER_BACKEND_DESIGN.md).
 * Pure C, no LLVM dependency.
 */

#ifndef MGL_IR_H
#define MGL_IR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Uniform layout standards (GLSL 4.60 §4.3.9 / ES 3.x §4.3.8). */
typedef enum MGLIRLayoutStd {
    MGLIR_LAYOUT_NONE = 0,
    MGLIR_LAYOUT_STD140,
    MGLIR_LAYOUT_STD430,
    MGLIR_LAYOUT_SHARED,   /* compat: computed as std140 */
    MGLIR_LAYOUT_PACKED,   /* compat: computed as std430 */
} MGLIRLayoutStd;

typedef enum MGLIRScalar {
    MGLIR_SCALAR_VOID = 0,
    MGLIR_SCALAR_BOOL,
    MGLIR_SCALAR_INT,
    MGLIR_SCALAR_UINT,
    MGLIR_SCALAR_FLOAT,
    MGLIR_SCALAR_DOUBLE,
    MGLIR_SCALAR_HALF,     /* GLSL ES mediump float / fp16 */
} MGLIRScalar;

typedef enum MGLIRTypeKind {
    MGLIR_TYPE_INVALID = 0,
    MGLIR_TYPE_SCALAR,
    MGLIR_TYPE_VECTOR,
    MGLIR_TYPE_MATRIX,
    MGLIR_TYPE_ARRAY,
    MGLIR_TYPE_STRUCT,
    MGLIR_TYPE_SAMPLER,
    MGLIR_TYPE_IMAGE,
    MGLIR_TYPE_ATOMIC_COUNTER,
} MGLIRTypeKind;

typedef enum MGLIRTexKind {
    MGLIR_TEX_1D = 0,
    MGLIR_TEX_2D,
    MGLIR_TEX_3D,
    MGLIR_TEX_CUBE,
    MGLIR_TEX_2D_RECT,
    MGLIR_TEX_1D_ARRAY,
    MGLIR_TEX_2D_ARRAY,
    MGLIR_TEX_CUBE_ARRAY,
    MGLIR_TEX_2D_MS,
    MGLIR_TEX_2D_MS_ARRAY,
    MGLIR_TEX_BUFFER,
    MGLIR_TEX_SUBPASS,
    MGLIR_TEX_SUBPASS_MS,
} MGLIRTexKind;

/* Type flags. */
enum {
    MGLIR_TYPE_ROW_MAJOR = 0x1,   /* matrices stored row-major */
    MGLIR_TYPE_COL_MAJOR = 0x2,   /* matrices stored column-major (GL default) */
};

/* Cached layout result. */
typedef struct MGLIRLayoutInfo {
    uint32_t size;                /* aligned size in bytes */
    uint32_t alignment;           /* required alignment in bytes (1/2/4/8/16) */
    uint32_t offset;              /* relative offset for struct members */
    uint32_t array_stride;        /* stride between array elements */
    uint32_t matrix_stride;       /* stride between matrix columns/rows */
} MGLIRLayoutInfo;

typedef struct MGLIRType MGLIRType;

struct MGLIRType {
    MGLIRTypeKind kind;
    MGLIRScalar scalar;
    uint32_t rows;              /* scalars per column (vector) / rows (matrix) */
    uint32_t cols;              /* vector width / matrix column count */
    MGLIRType *elem_type;       /* array element type (owned) */
    uint32_t array_size;        /* element count (0 = runtime array) */
    uint32_t member_count;      /* struct member count */
    MGLIRType **members;        /* struct members (owned) */
    char **member_names;        /* owned (dup'd at construction) */
    uint32_t *member_offsets;   /* computed offsets (valid after layout) */

    /* Sampler / image. */
    MGLIRTexKind tex_kind;      /* MGLIR_TEX_* */
    MGLIRScalar tex_storage;    /* texel scalar (float/int/uint) */
    uint32_t tex_depth;         /* depth texture */
    uint32_t tex_format;        /* image format enum (GL enum value) */

    uint32_t row_major;         /* matrix memory order flag */

    const char *name;           /* owned (dup'd at construction) */

    MGLIRLayoutInfo layout;     /* cached layout results */
    uint32_t layout_valid;      /* 1 after mglIRComputeLayout */
};

/* Constructors.  Returned structure is heap-owned; destroy with mglIRTypeDestroy. */
MGLIRType *mglIRTypeScalar(MGLIRScalar s);
MGLIRType *mglIRTypeVector(MGLIRScalar s, uint32_t components);
MGLIRType *mglIRTypeMatrix(MGLIRScalar s, uint32_t cols, uint32_t rows);
MGLIRType *mglIRTypeArray(MGLIRType *elem, uint32_t size);
MGLIRType *mglIRTypeRuntimeArray(MGLIRType *elem);
MGLIRType *mglIRTypeStruct(MGLIRType *const *members, const char *const *names,
                           uint32_t count, const char *name);
MGLIRType *mglIRTypeSampler(MGLIRTexKind kind, MGLIRScalar storage, int depth);
MGLIRType *mglIRTypeImage(MGLIRTexKind kind, MGLIRScalar storage, uint32_t gl_format);

/* Compute layout in place (fills size/alignment/array_stride/member offsets).
 * Returns MGLIR_LAYOUT_* on success or -1 on error.  mtl satisfies
 * row-major GL matrices. */
int mglIRComputeLayout(MGLIRType *type, MGLIRLayoutStd layout, uint32_t *size);

void mglIRTypeDestroy(MGLIRType *type);

#ifdef __cplusplus
}
#endif

#endif /* MGL_IR_H */