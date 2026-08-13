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
 * mgl_glsl_ast.h
 * MGL - self-written GLSL frontend AST definitions (see
 * docs/AIR_SHADER_BACKEND_DESIGN.md).  Pure C, no LLVM dependency.
 */

#ifndef MGL_GLSL_AST_H
#define MGL_GLSL_AST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Variable storage/interpolation qualifier bits (GLSL 4.60 §4.3). */
enum {
    MGL_AST_Q_NONE          = 0,
    MGL_AST_Q_CONST         = 0x0001,
    MGL_AST_Q_IN            = 0x0002,
    MGL_AST_Q_OUT           = 0x0004,
    MGL_AST_Q_INOUT         = 0x0008,
    MGL_AST_Q_UNIFORM       = 0x0010,
    MGL_AST_Q_BUFFER        = 0x0020, /* buffer block or variable */
    MGL_AST_Q_SHARED        = 0x0040,
    MGL_AST_Q_PATCH         = 0x0080,
    MGL_AST_Q_CENTROID      = 0x0100,
    MGL_AST_Q_SAMPLE        = 0x0200,
    MGL_AST_Q_INVARIANT     = 0x0400,
    MGL_AST_Q_PRECISE       = 0x0800,
    MGL_AST_Q_FLAT          = 0x1000,
    MGL_AST_Q_SMOOTH        = 0x2000,
    MGL_AST_Q_NOPERSPECTIVE = 0x4000,
};

/* Block layout qualifiers (GLSL 4.60 §4.3.9). */
enum {
    MGL_AST_LAYOUT_DEFAULT = 0,
    MGL_AST_LAYOUT_STD140  = 1,
    MGL_AST_LAYOUT_STD430  = 2,
    MGL_AST_LAYOUT_SHARED  = 3,
    MGL_AST_LAYOUT_PACKED  = 4,
};

/* Matrix memory-order qualifiers. */
enum {
    MGL_AST_MATRIX_DEFAULT = 0,
    MGL_AST_MATRIX_COL_MAJOR = 1,
    MGL_AST_MATRIX_ROW_MAJOR = 2,
};

/* Tessellation primitive modes (TES) and geometry input topologies (GS).
 * `triangles` is shared. */
enum {
    MGL_AST_TES_DEFAULT    = 0,
    MGL_AST_TES_TRIANGLES  = 1,
    MGL_AST_TES_QUADS      = 2,
    MGL_AST_TES_ISOLINES   = 3,
};

/* Geometry shader input topologies (GLSL 4.60 4.3.8.1). */
enum {
    MGL_AST_GS_IN_DEFAULT           = 0,
    MGL_AST_GS_IN_POINTS            = 16,
    MGL_AST_GS_IN_LINES             = 17,
    MGL_AST_GS_IN_LINES_ADJACENCY   = 18,
    MGL_AST_GS_IN_TRIANGLES         = 19,
    MGL_AST_GS_IN_TRIANGLES_ADJACENCY = 20,
};

/* Geometry shader output topologies (GLSL 4.60 4.3.8.2). */
enum {
    MGL_AST_GS_OUT_DEFAULT      = 0,
    MGL_AST_GS_OUT_POINTS       = 1,
    MGL_AST_GS_OUT_LINE_STRIP   = 2,
    MGL_AST_GS_OUT_TRIANGLE_STRIP = 3,
};

/* Tessellation spacing (TES, GLSL 4.60 4.3.8.2). */
enum {
    MGL_AST_SPACING_DEFAULT            = 0,
    MGL_AST_SPACING_EQUAL              = 1,
    MGL_AST_SPACING_FRACTIONAL_EVEN    = 2,
    MGL_AST_SPACING_FRACTIONAL_ODD     = 3,
};

/* Tessellation winding order (TES). */
enum {
    MGL_AST_WINDING_DEFAULT = 0,
    MGL_AST_WINDING_CW      = 1,
    MGL_AST_WINDING_CCW     = 2,
};

/* Precision qualifiers (GLSL ES / desktop 4.30+). */
enum {
    MGL_AST_PRECISION_NONE = 0,
    MGL_AST_PRECISION_LOWP = 1,
    MGL_AST_PRECISION_MEDIUMP = 2,
    MGL_AST_PRECISION_HIGHP = 3,
};

/* Basic scalar/opaque types.  User struct names resolve via the symbol
 * table during sema; the parser records them as MGL_AST_TYPE_STRUCT_NAME. */
typedef enum MGLGLSLBaseType {
    MGL_AST_TYPE_VOID = 0,
    MGL_AST_TYPE_BOOL,
    MGL_AST_TYPE_INT,
    MGL_AST_TYPE_UINT,
    MGL_AST_TYPE_FLOAT,
    MGL_AST_TYPE_DOUBLE,
    MGL_AST_TYPE_STRUCT,      /* struct definition or named user struct */
    MGL_AST_TYPE_SAMPLER,     /* any sampler2D/.../samplerCubeShadow */
    MGL_AST_TYPE_IMAGE,
    MGL_AST_TYPE_ATOMIC_UINT,
} MGLGLSLBaseType;

/* Expression node. */
typedef struct MGLExpr MGLExpr;

typedef enum MGLExprKind {
    MGL_EXPR_LITERAL = 0,   /* numeric/boolean literal */
    MGL_EXPR_VAR_REF,       /* identifier */
    MGL_EXPR_MEMBER,        /* obj.field */
    MGL_EXPR_INDEX,         /* arr[i] */
    MGL_EXPR_CALL,          /* f(a, b) - callee is identifier */
    MGL_EXPR_UNARY,         /* -x, !x, ++x, --x, ~x */
    MGL_EXPR_BINARY,        /* x op y */
    MGL_EXPR_ASSIGN,        /* x = y, x += y, ... */
    MGL_EXPR_TERNARY,       /* c ? a : b */
} MGLExprKind;

typedef enum MGLExprOp {
    MGL_OP_NONE = 0,
    MGL_OP_ADD, MGL_OP_SUB, MGL_OP_MUL, MGL_OP_DIV, MGL_OP_MOD,
    MGL_OP_SHL, MGL_OP_SHR,
    MGL_OP_AND, MGL_OP_OR, MGL_OP_XOR,       /* bitwise */
    MGL_OP_LAND, MGL_OP_LOR,                 /* logical */
    MGL_OP_EQ, MGL_OP_NE, MGL_OP_LT, MGL_OP_LE, MGL_OP_GT, MGL_OP_GE,
    MGL_OP_NOT, MGL_OP_BNOT,                 /* unary */
    MGL_OP_INC, MGL_OP_DEC,                  /* pre/post ++/-- */
    MGL_OP_ASSIGN, MGL_OP_ADD_ASSIGN, MGL_OP_SUB_ASSIGN, MGL_OP_MUL_ASSIGN,
    MGL_OP_DIV_ASSIGN, MGL_OP_MOD_ASSIGN, MGL_OP_SHL_ASSIGN, MGL_OP_SHR_ASSIGN,
    MGL_OP_AND_ASSIGN, MGL_OP_OR_ASSIGN, MGL_OP_XOR_ASSIGN,
} MGLExprOp;

struct MGLExpr {
    uint32_t kind;      /* MGLExprKind */
    uint32_t line;
    union {
        struct {
            uint32_t base;   /* MGLGLSLBaseType (bool/int/uint/float/double) */
            double value;    /* numeric value */
        } literal;
        struct {
            char *name;      /* owned */
        } var_ref;
        struct {
            MGLExpr *object;
            char *field;     /* owned */
        } member;
        struct {
            MGLExpr *object;
            MGLExpr *index;
        } index;
        struct {
            char *name;      /* owned */
            MGLExpr **args;
            uint32_t arg_count;
            int is_array_ctor;   /* 1 = vecN[](...) array constructor */
        } call;
        struct {
            uint32_t op;     /* MGLExprOp */
            MGLExpr *operand;
            int prefix;      /* 1 = prefix, 0 = postfix (++/-- only) */
        } unary;
        struct {
            uint32_t op;     /* MGLExprOp */
            MGLExpr *lhs, *rhs;
        } binary;
        struct {
            uint32_t op;     /* MGLExprOp assignment ops */
            MGLExpr *lhs, *rhs;
        } assign;
        struct {
            MGLExpr *cond, *then, *else_;
        } ternary;
    } u;
};

/* Statement node. */
typedef struct MGLStmt MGLStmt;
typedef struct MGLDecl MGLDecl;

typedef enum MGLStmtKind {
    MGL_STMT_COMPOUND = 0,  /* { ... } */
    MGL_STMT_EXPR,          /* expr ; */
    MGL_STMT_DECL,          /* type name = init ; */
    MGL_STMT_IF,            /* if (c) t [else e] */
    MGL_STMT_FOR,           /* for (init; cond; incr) body */
    MGL_STMT_WHILE,
    MGL_STMT_DO_WHILE,
    MGL_STMT_SWITCH,
    MGL_STMT_CASE,
    MGL_STMT_DEFAULT,
    MGL_STMT_BREAK,
    MGL_STMT_CONTINUE,
    MGL_STMT_RETURN,
    MGL_STMT_DISCARD,
} MGLStmtKind;

struct MGLStmt {
    uint32_t kind;      /* MGLStmtKind */
    uint32_t line;
    union {
        struct {
            MGLStmt **stmts;
            uint32_t count;
        } compound;
        struct {
            MGLExpr *expr;
        } expr;
        struct {
            MGLDecl *decl;
        } decl;
        struct {
            MGLExpr *cond;
            MGLStmt *then;
            MGLStmt *else_; /* may be NULL */
        } ifs;
        struct {
            MGLStmt *init;  /* may be NULL; decl or expr stmt */
            MGLExpr *cond;  /* may be NULL */
            MGLExpr *incr;  /* may be NULL */
            MGLStmt *body;
        } loop;
        struct {
            MGLExpr *cond;
            MGLStmt *body;
        } whilex;
        struct {
            MGLStmt *body;
        } body;
        struct {
            MGLExpr *cond;
            MGLStmt *body;
        } switchx;
        struct {
            MGLExpr *value;
        } casex;
        struct {
            MGLExpr *value; /* may be NULL for bare return */
        } ret;
    } u;
};

/* Variable / function declaration. */

/* Type specifier as written in source.  Struct member names or struct types
 * are resolved by sema; parser stores the raw text. */
typedef struct MGLTypeSpec {
    uint32_t base;         /* MGLGLSLBaseType */
    uint32_t precision;    /* MGL_AST_PRECISION_* */
    int vec_size;          /* vecN: N, else 0 */
    int mat_cols, mat_rows;/* matCxR */
    char *name;            /* struct name or NULL (owned) */
    MGLDecl *struct_def;   /* inline struct definition or NULL */
} MGLTypeSpec;

struct MGLDecl {
    char *name;            /* owned */
    MGLTypeSpec *type;     /* owned */
    uint32_t qualifiers;   /* MGL_AST_Q_* */
    uint32_t layout;       /* MGL_AST_LAYOUT_* */
    uint32_t matrix_major; /* MGL_AST_MATRIX_* */
    int32_t layout_location; /* layout(location=N), -1 if unspecified */
    int32_t layout_binding;  /* layout(binding=N), -1 if unspecified */
    /* Tessellation/geometry layout (M3).  -1/0 = unspecified. */
    int32_t  layout_vertices;       /* TCS: layout(vertices=N) */
    uint32_t layout_primitive;      /* TES: MGL_AST_TES_* ; GS in: MGL_AST_GS_IN_* */
    uint32_t layout_primitive_out;  /* GS out: MGL_AST_GS_OUT_* */
    int32_t  layout_max_vertices;   /* GS: layout(max_vertices=N) */
    int32_t  layout_invocations;    /* GS: layout(invocations=N), 1 if absent */
    int32_t  layout_stream;         /* GS output stream, -1 if unspecified (0) */
    uint32_t layout_spacing;        /* TES: MGL_AST_SPACING_* */
    uint32_t layout_winding;        /* TES: MGL_AST_WINDING_* */
    uint32_t layout_point_mode;     /* TES: point_mode flag */
    uint32_t *array_dims;  /* element counts; NULL = not an array */
    uint32_t array_count;
    MGLExpr *init;         /* initializer or NULL */
    MGLStmt *body;         /* function body; NULL = prototype/variable */
    MGLDecl **params;      /* function parameters */
    uint32_t param_count;
    MGLTypeSpec *return_type; /* function return type (body/params non-NULL) */
    MGLDecl **struct_members; /* struct/block members or NULL */
    uint32_t struct_member_count;
    uint32_t line;
};

/* Entire shader translation unit. */
typedef struct MGLTranslationUnit {
    uint32_t version;          /* #version number, 0 if absent */
    char *version_profile;     /* "core"/"compatibility"/"es"/NULL (owned) */
    MGLDecl **decls;
    uint32_t decl_count;
    char *error;               /* first parse error message or NULL (owned) */
    uint32_t error_line;
    /* Stage-level tessellation/geometry layout (M3): set by layout-only
     * declarations like `layout(vertices = 3) out;`.  -1/0 = unspecified. */
    int32_t  layout_vertices;      /* TCS: patch vertex count */
    int32_t  layout_max_vertices;  /* GS: max emitted vertices */
    int32_t  layout_invocations;   /* GS: invocation count (1 default) */
    int32_t  layout_stream;        /* GS default output stream (-1 = 0) */
    uint32_t layout_primitive;     /* TES: MGL_AST_TES_* ; GS in: MGL_AST_GS_IN_* */
    uint32_t layout_primitive_out; /* GS out: MGL_AST_GS_OUT_* */
    uint32_t layout_spacing;       /* TES: MGL_AST_SPACING_* */
    uint32_t layout_winding;       /* TES: MGL_AST_WINDING_* */
    uint32_t layout_point_mode;    /* TES: point_mode flag */
} MGLTranslationUnit;

#ifdef __cplusplus
}
#endif

#endif /* MGL_GLSL_AST_H */
