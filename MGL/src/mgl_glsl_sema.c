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
 * mgl_glsl_sema.c
 * MGL - GLSL semantic analysis skeleton: symbol tables, type resolution
 * (MGLTypeSpec -> MGLIRType), expression type checking with implicit
 * conversions, uniform/buffer block layout.
 */

#include "mgl_glsl_sema.h"
#include "mgl_glsl_ast.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Diagnostics                                                         */
/* ------------------------------------------------------------------ */

typedef struct Sema {
    const MGLTranslationUnit *tu;
    MGLIRModule *module;
    MGLSemaError *errors;
    uint32_t error_count;
    uint32_t error_cap;
} Sema;

static void sema_error(Sema *s, uint32_t line, const char *fmt, ...)
{
    if (s->error_count == s->error_cap) {
        uint32_t ncap = s->error_cap ? s->error_cap * 2 : 8;
        MGLSemaError *n = (MGLSemaError *)realloc(
            s->errors, ncap * sizeof(MGLSemaError));
        if (!n) {
            return;
        }
        s->errors = n;
        s->error_cap = ncap;
    }
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    s->errors[s->error_count].message = strdup(buf);
    s->errors[s->error_count].line = line;
    s->error_count++;
}

/* ------------------------------------------------------------------ */
/* Symbol table                                                        */
/* ------------------------------------------------------------------ */

/* Kind of a named entry. */
typedef enum SymKind {
    SYM_VARIABLE,
    SYM_FUNCTION,
    SYM_STRUCT,
} SymKind;

typedef struct Sym {
    char *name;             /* owned */
    SymKind kind;
    MGLIRType *type;        /* variable/parameter type */
    int type_owned;         /* 1 = this Sym owns `type` */
    MGLIRType *ret_type;    /* function return type (owned) */
    uint32_t param_count;
    MGLIRType **param_types; /* owned copies */
    uint32_t qualifiers;     /* MGL_AST_Q_* */
    struct Sym *next;        /* scope chain link */
    struct Sym *next_all;    /* global teardown chain link */
} Sym;

/* Flat scope stack: entries push/pop at block boundaries.  `top` points at
 * the innermost scope; lookup walks the chain. */
typedef struct Scope {
    Sym *sym;
    struct Scope *parent;
} Scope;

typedef struct SymTab {
    Scope *top;
    struct Sym *all;        /* all declared names (leak-free teardown) */
} SymTab;

static Sym *sym_new(const char *name)
{
    Sym *s = (Sym *)calloc(1, sizeof(*s));
    if (s) {
        s->name = strdup(name);
    }
    return s;
}

static int symtab_push(SymTab *t)
{
    Scope *sc = (Scope *)calloc(1, sizeof(*sc));
    if (!sc) {
        return -1;
    }
    sc->parent = t->top;
    t->top = sc;
    return 0;
}

static void symtab_pop(SymTab *t)
{
    if (t->top) {
        Scope *sc = t->top;
        t->top = sc->parent;
        free(sc);
    }
}

/* Look up a name in the current scope chain. */
static Sym *symtab_lookup(SymTab *t, const char *name)
{
    for (Scope *sc = t->top; sc; sc = sc->parent) {
        for (Sym *s = sc->sym; s; s = s->next) {
            if (strcmp(s->name, name) == 0) {
                return s;
            }
        }
    }
    return NULL;
}

/* Look up only in the innermost scope (redeclaration check). */
static Sym *symtab_lookup_local(SymTab *t, const char *name)
{
    if (!t->top) {
        return NULL;
    }
    for (Sym *s = t->top->sym; s; s = s->next) {
        if (strcmp(s->name, name) == 0) {
            return s;
        }
    }
    return NULL;
}

static int symtab_insert(SymTab *t, Sym *s)
{
    if (!t->top) {
        return -1;
    }
    /* keep scope chain order: append */
    if (t->top->sym == NULL) {
        t->top->sym = s;
    } else {
        Sym *last = t->top->sym;
        while (last->next) {
            last = last->next;
        }
        last->next = s;
    }
    /* register in `all` for teardown (prepend) */
    s->next_all = t->all;
    t->all = s;
    return 0;
}

/* Move a symbol to the current scope (used when a prototype is later
 * defined; keeps symbol identity). */
static void symtab_destroy(SymTab *t)
{
    Sym *s = t->all;
    while (s) {
        Sym *n = s->next_all;
        free(s->name);
        if (s->type && s->type_owned) {
            mglIRTypeDestroy(s->type);
        }
        if (s->ret_type) {
            mglIRTypeDestroy(s->ret_type);
        }
        for (uint32_t i = 0; i < s->param_count; i++) {
            if (s->param_types && s->param_types[i]) {
                mglIRTypeDestroy(s->param_types[i]);
            }
        }
        free(s->param_types);
        free(s);
        s = n;
    }
    while (t->top) {
        Scope *sc = t->top;
        t->top = sc->parent;
        free(sc);
    }
    t->all = NULL;
}

/* ------------------------------------------------------------------ */
/* Type resolution: MGLTypeSpec -> MGLIRType                           */
/* ------------------------------------------------------------------ */

static MGLIRScalar ast_base_to_ir(uint32_t base)
{
    switch (base) {
    case MGL_AST_TYPE_BOOL:   return MGLIR_SCALAR_BOOL;
    case MGL_AST_TYPE_INT:    return MGLIR_SCALAR_INT;
    case MGL_AST_TYPE_UINT:   return MGLIR_SCALAR_UINT;
    case MGL_AST_TYPE_FLOAT:  return MGLIR_SCALAR_FLOAT;
    case MGL_AST_TYPE_DOUBLE: return MGLIR_SCALAR_DOUBLE;
    case MGL_AST_TYPE_VOID:   return MGLIR_SCALAR_VOID;
    default:                  return MGLIR_SCALAR_VOID;
    }
}

/* Convert a sampler/image typename into MGLIRTexKind + storage + depth.
 * Returns 0 on success. */
static int parse_opaque_name(const char *name, size_t n, int is_sampler,
                             MGLIRTexKind *kind, MGLIRScalar *storage,
                             int *depth)
{
    /* Normalize "sampler2D" -> prefix "2d", suffix handled below. */
    const char *p = name;
    size_t len = n;
    if (len > 0 && p[0] == 's' && is_sampler && len >= 7 &&
        memcmp(p, "sampler", 7) == 0) {
        p += 7;
        len -= 7;
    } else if (len > 0 && p[0] == 'i' && !is_sampler && len >= 5 &&
               memcmp(p, "image", 5) == 0) {
        p += 5;
        len -= 5;
    } else {
        return -1;
    }
    int is_unsigned = 0;
    int is_signed = 0;
    int is_shadow = 0;
    if (len > 0 && p[0] == 'u') {
        is_unsigned = 1;
        p++;
        len--;
    } else if (len > 0 && p[0] == 'i') {
        is_signed = 1;
        p++;
        len--;
    }
    int dims = 0;
    /* Optional '2D'|'3D'|'Cube'|'1D' */
    if (len >= 2 && (p[0] == '2' && p[1] == 'D')) {
        dims = 2;
        p += 2;
        len -= 2;
    } else if (len >= 2 && (p[0] == '3' && p[1] == 'D')) {
        dims = 3;
        p += 2;
        len -= 2;
    } else if (len >= 4 && memcmp(p, "Cube", 4) == 0) {
        dims = 4;
        p += 4;
        len -= 4;
    } else if (len >= 2 && (p[0] == '1' && p[1] == 'D')) {
        dims = 1;
        p += 2;
        len -= 2;
    } else if (len >= 6 && memcmp(p, "Buffer", 6) == 0) {
        dims = 5;
        p += 6;
        len -= 6;
    }
    int is_array = 0;
    if (len >= 5 && memcmp(p, "Array", 5) == 0) {
        is_array = 1;
        p += 5;
        len -= 5;
    }
    int is_ms = 0;
    if (len >= 2 && memcmp(p, "MS", 2) == 0) {
        is_ms = 1;
        p += 2;
        len -= 2;
    }
    if (len >= 6 && memcmp(p, "Shadow", 6) == 0) {
        is_shadow = 1;
        p += 6;
        len -= 6;
    }
    switch (dims) {
    case 1:  *kind = is_array ? MGLIR_TEX_1D_ARRAY : MGLIR_TEX_1D; break;
    case 2:  *kind = is_array ? MGLIR_TEX_2D_ARRAY
                               : (is_ms ? MGLIR_TEX_2D_MS : MGLIR_TEX_2D); break;
    case 3:  *kind = MGLIR_TEX_3D; break;
    case 4:  *kind = is_array ? MGLIR_TEX_CUBE_ARRAY : MGLIR_TEX_CUBE; break;
    case 5:  *kind = MGLIR_TEX_BUFFER; break;
    default: return -1;
    }
    if (is_signed) {
        *storage = MGLIR_SCALAR_INT;
    } else if (is_unsigned) {
        *storage = MGLIR_SCALAR_UINT;
    } else {
        *storage = MGLIR_SCALAR_FLOAT;
    }
    *depth = is_shadow;
    return 0;
}

static MGLIRType *ir_type_clone(const MGLIRType *src)
{
    if (!src) {
        return NULL;
    }
    MGLIRType *t = (MGLIRType *)calloc(1, sizeof(*t));
    if (!t) {
        return NULL;
    }
    *t = *src;
    t->elem_type = NULL;
    t->members = NULL;
    t->member_names = NULL;
    t->member_offsets = NULL;
    switch (src->kind) {
    case MGLIR_TYPE_ARRAY:
        t->elem_type = ir_type_clone(src->elem_type);
        if (!t->elem_type) {
            free(t);
            return NULL;
        }
        break;
    case MGLIR_TYPE_STRUCT: {
        t->member_names = NULL;
        t->members = (MGLIRType **)calloc(src->member_count, sizeof(MGLIRType *));
        t->member_names = (char **)calloc(src->member_count, sizeof(char *));
        if (!t->members || !t->member_names) {
            free(t->members);
            free(t->member_names);
            free(t);
            return NULL;
        }
        for (uint32_t i = 0; i < src->member_count; i++) {
            t->members[i] = ir_type_clone(src->members[i]);
            t->member_names[i] = src->member_names[i]; /* borrowed */
            if (!t->members[i]) {
                for (uint32_t j = 0; j < i; j++) {
                    mglIRTypeDestroy(t->members[j]);
                }
                free(t->members);
                free(t->member_names);
                free(t);
                return NULL;
            }
        }
        break;
    }
    default:
        break;
    }
    return t;
}

static MGLIRType *resolve_type_spec(Sema *s, SymTab *tab, const MGLTypeSpec *ts);

/* Resolve a single declarator (type + array dims) into an IR type. */
static MGLIRType *resolve_decl_type(Sema *s, SymTab *tab, const MGLDecl *d)
{
    MGLIRType *t = resolve_type_spec(s, tab, d->type);
    if (!t) {
        return NULL;
    }
    if (d->matrix_major == MGL_AST_MATRIX_ROW_MAJOR &&
        t->kind == MGLIR_TYPE_MATRIX) {
        t->row_major = 1;
    }
    for (uint32_t i = d->array_count; i > 0; i--) {
        uint32_t sz = d->array_dims[i - 1];
        MGLIRType *arr = mglIRTypeArray(t, sz);
        if (!arr) {
            mglIRTypeDestroy(t);
            return NULL;
        }
        t = arr;
    }
    return t;
}

static MGLIRType *resolve_type_spec(Sema *s, SymTab *tab, const MGLTypeSpec *ts)
{
    if (!ts) {
        return NULL;
    }
    MGLIRScalar sc = ast_base_to_ir(ts->base);
    switch (ts->base) {
    case MGL_AST_TYPE_VOID:
        return mglIRTypeScalar(MGLIR_SCALAR_VOID);
    case MGL_AST_TYPE_BOOL:
    case MGL_AST_TYPE_INT:
    case MGL_AST_TYPE_UINT:
    case MGL_AST_TYPE_FLOAT:
    case MGL_AST_TYPE_DOUBLE: {
        MGLIRType *t = mglIRTypeScalar(sc);
        if (!t) {
            return NULL;
        }
        if (ts->vec_size) {
            MGLIRType *v = mglIRTypeVector(sc, (uint32_t)ts->vec_size);
            mglIRTypeDestroy(t);
            return v;
        }
        if (ts->mat_cols) {
            MGLIRType *m = mglIRTypeMatrix(sc, (uint32_t)ts->mat_cols,
                                           (uint32_t)ts->mat_rows);
            mglIRTypeDestroy(t);
            return m;
        }
        return t;
    }
    case MGL_AST_TYPE_STRUCT: {
        if (ts->struct_def) {
            /* inline struct definition */
            uint32_t n = ts->struct_def->struct_member_count;
            MGLIRType **members = (MGLIRType **)calloc(n, sizeof(MGLIRType *));
            const char **names = (const char **)calloc(n, sizeof(char *));
            if (!members || !names) {
                free(members);
                free(names);
                return NULL;
            }
            for (uint32_t i = 0; i < n; i++) {
                MGLDecl *m = ts->struct_def->struct_members[i];
                members[i] = resolve_decl_type(s, tab, m);
                names[i] = m->name;
                if (!members[i]) {
                    for (uint32_t j = 0; j < i; j++) {
                        mglIRTypeDestroy(members[j]);
                    }
                    free(members);
                    free(names);
                    return NULL;
                }
            }
            MGLIRType *t = mglIRTypeStruct(members, names, n, ts->name);
            free(members);
            free(names);
            return t;
        }
        if (ts->name) {
            Sym *st = symtab_lookup(tab, ts->name);
            if (!st || st->kind != SYM_STRUCT) {
                sema_error(s, ts->struct_def ? ts->struct_def->line : 0,
                           "unknown struct type '%s'", ts->name);
                return NULL;
            }
            /* clone the struct type (member layouts are per-instance) */
            return ir_type_clone(st->type);
        }
        return NULL;
    }
    case MGL_AST_TYPE_SAMPLER:
    case MGL_AST_TYPE_IMAGE: {
        if (!ts->name) {
            return NULL;
        }
        MGLIRTexKind kind;
        MGLIRScalar storage;
        int depth;
        if (parse_opaque_name(ts->name, strlen(ts->name),
                              ts->base == MGL_AST_TYPE_SAMPLER,
                              &kind, &storage, &depth) != 0) {
            sema_error(s, 0, "unsupported opaque type '%s'", ts->name);
            return NULL;
        }
        if (ts->base == MGL_AST_TYPE_SAMPLER) {
            return mglIRTypeSampler(kind, storage, depth);
        }
        return mglIRTypeImage(kind, storage, 0);
    }
    case MGL_AST_TYPE_ATOMIC_UINT:
        return mglIRTypeVector(MGLIR_SCALAR_UINT, 1); /* placeholder */
    default:
        return NULL;
    }
}

/* ------------------------------------------------------------------ */
/* Expression type checking                                            */
/* ------------------------------------------------------------------ */

static MGLIRType *check_expr(Sema *s, SymTab *tab, const MGLExpr *e);

static int ir_type_equal(const MGLIRType *a, const MGLIRType *b)
{
    if (a == b) {
        return 1;
    }
    if (!a || !b) {
        return 0;
    }
    if (a->kind != b->kind || a->scalar != b->scalar) {
        return 0;
    }
    switch (a->kind) {
    case MGLIR_TYPE_SCALAR:
        return 1;
    case MGLIR_TYPE_VECTOR:
        return a->cols == b->cols;
    case MGLIR_TYPE_MATRIX:
        return a->cols == b->cols && a->rows == b->rows;
    case MGLIR_TYPE_ARRAY:
        return a->array_size == b->array_size && ir_type_equal(a->elem_type, b->elem_type);
    case MGLIR_TYPE_STRUCT:
        return a->member_count == b->member_count;
    case MGLIR_TYPE_SAMPLER:
    case MGLIR_TYPE_IMAGE:
        return a->tex_kind == b->tex_kind && a->tex_depth == b->tex_depth;
    default:
        return 0;
    }
}

/* GLSL 4.60 §4.1.10 implicit conversion ranking:
 *   int -> uint -> float -> double
 *   int -> float -> double
 * Boolean conversions only via explicit constructors. */
static int implicit_convert(MGLIRType *from, MGLIRType *to)
{
    if (!from || !to || from->kind != MGLIR_TYPE_SCALAR ||
        to->kind != MGLIR_TYPE_SCALAR) {
        return 0;
    }
    static const MGLIRScalar rank[5] = {
        MGLIR_SCALAR_INT, MGLIR_SCALAR_UINT, MGLIR_SCALAR_FLOAT, MGLIR_SCALAR_DOUBLE,
        MGLIR_SCALAR_HALF,
    };
    int fi = -1, ti = -1;
    for (int i = 0; i < 5; i++) {
        if (from->scalar == rank[i]) {
            fi = i;
        }
        if (to->scalar == rank[i]) {
            ti = i;
        }
        if (fi >= 0 && ti >= 0) {
            break;
        }
    }
    if (fi < 0 || ti < 0) {
        return 0;
    }
    /* int -> uint -> float -> double chain; half only as target */
    if (to->scalar == MGLIR_SCALAR_HALF) {
        return from->scalar == MGLIR_SCALAR_FLOAT;
    }
    return fi <= ti;
}

static const char *ir_type_str(const MGLIRType *t, char *buf, size_t cap)
{
    if (!t) {
        snprintf(buf, cap, "<none>");
        return buf;
    }
    static const char *const scal[] = { "void", "bool", "int", "uint",
                                        "float", "double", "half" };
    const char *sc = (t->scalar < 7) ? scal[t->scalar] : "?";
    switch (t->kind) {
    case MGLIR_TYPE_SCALAR:
        snprintf(buf, cap, "%s", sc);
        break;
    case MGLIR_TYPE_VECTOR:
        snprintf(buf, cap, "%s%d", sc, (int)t->cols);
        break;
    case MGLIR_TYPE_MATRIX:
        snprintf(buf, cap, "mat%d%d", (int)t->cols, (int)t->rows);
        break;
    case MGLIR_TYPE_ARRAY: {
        char inner[64];
        snprintf(buf, cap, "%s[%u]", ir_type_str(t->elem_type, inner, sizeof(inner)),
                 t->array_size);
        break;
    }
    case MGLIR_TYPE_STRUCT:
        snprintf(buf, cap, "struct %s", t->name ? t->name : "?");
        break;
    case MGLIR_TYPE_SAMPLER:
        snprintf(buf, cap, "sampler(%d)", (int)t->tex_kind);
        break;
    case MGLIR_TYPE_IMAGE:
        snprintf(buf, cap, "image(%d)", (int)t->tex_kind);
        break;
    default:
        snprintf(buf, cap, "?");
        break;
    }
    return buf;
}

static const char *op_name(uint32_t op)
{
    switch (op) {
    case MGL_OP_ADD: return "+";   case MGL_OP_SUB: return "-";
    case MGL_OP_MUL: return "*";   case MGL_OP_DIV: return "/";
    case MGL_OP_MOD: return "%";   case MGL_OP_SHL: return "<<";
    case MGL_OP_SHR: return ">>";  case MGL_OP_AND: return "&";
    case MGL_OP_OR: return "|";    case MGL_OP_XOR: return "^";
    case MGL_OP_LAND: return "&&"; case MGL_OP_LOR: return "||";
    case MGL_OP_EQ: return "==";   case MGL_OP_NE: return "!=";
    case MGL_OP_LT: return "<";    case MGL_OP_LE: return "<=";
    case MGL_OP_GT: return ">";    case MGL_OP_GE: return ">=";
    default: return "?";
    }
}

static int is_numeric(const MGLIRType *t)
{
    return t && t->kind == MGLIR_TYPE_SCALAR &&
           (t->scalar == MGLIR_SCALAR_INT || t->scalar == MGLIR_SCALAR_UINT ||
            t->scalar == MGLIR_SCALAR_FLOAT || t->scalar == MGLIR_SCALAR_DOUBLE);
}

static MGLIRType *result_numeric(MGLIRType *a, MGLIRType *b, int is_mat_operand)
{
    (void)is_mat_operand;
    /* promote the common scalar of a/b */
    MGLIRScalar sc = a->scalar;
    if (b->scalar == MGLIR_SCALAR_DOUBLE ||
        (b->scalar == MGLIR_SCALAR_FLOAT && sc != MGLIR_SCALAR_DOUBLE) ||
        (b->scalar == MGLIR_SCALAR_UINT && sc == MGLIR_SCALAR_INT) ||
        (b->scalar == MGLIR_SCALAR_HALF && sc != MGLIR_SCALAR_HALF &&
         sc != MGLIR_SCALAR_DOUBLE && sc != MGLIR_SCALAR_FLOAT)) {
        sc = b->scalar;
    }
    MGLIRType *base = mglIRTypeScalar(sc);
    if (!base) {
        return NULL;
    }
    uint32_t cols = a->kind == MGLIR_TYPE_VECTOR ? a->cols :
                    (b->kind == MGLIR_TYPE_VECTOR ? b->cols : 1);
    if (cols > 1) {
        MGLIRType *v = mglIRTypeVector(sc, cols);
        mglIRTypeDestroy(base);
        return v;
    }
    return base;
}

static int check_assign_op(MGLIRType *dst, MGLIRType *src)
{
    if (!dst || !src) {
        return 0;
    }
    if (dst->kind == MGLIR_TYPE_SCALAR && src->kind == MGLIR_TYPE_SCALAR) {
        return implicit_convert(src, dst);
    }
    if (dst->kind == MGLIR_TYPE_VECTOR && src->kind == MGLIR_TYPE_VECTOR) {
        return dst->cols == src->cols &&
               (src->scalar == dst->scalar || implicit_convert(src, dst));
    }
    if (dst->kind == MGLIR_TYPE_MATRIX && src->kind == MGLIR_TYPE_MATRIX) {
        return dst->cols == src->cols && dst->rows == src->rows;
    }
    return ir_type_equal(dst, src);
}

/* Struct member lookup; returns member type and index. */
static const MGLIRType *struct_member(const MGLIRType *st, const char *name,
                                      uint32_t *idx)
{
    if (!st || st->kind != MGLIR_TYPE_STRUCT) {
        return NULL;
    }
    for (uint32_t i = 0; i < st->member_count; i++) {
        if (st->member_names && st->member_names[i] &&
            strcmp(st->member_names[i], name) == 0) {
            if (idx) {
                *idx = i;
            }
            return st->members[i];
        }
    }
    return NULL;
}

static MGLIRType *check_expr(Sema *s, SymTab *tab, const MGLExpr *e)
{
    if (!e) {
        return NULL;
    }
    char ta[64], tb[64];
    switch (e->kind) {
    case MGL_EXPR_LITERAL: {
        MGLIRScalar sc = ast_base_to_ir(e->u.literal.base);
        if (e->u.literal.base == MGL_AST_TYPE_FLOAT) {
            sc = MGLIR_SCALAR_FLOAT;
        } else if (e->u.literal.base == MGL_AST_TYPE_INT) {
            sc = MGLIR_SCALAR_INT;
        }
        MGLIRType *t = mglIRTypeScalar(sc);
        /* Literal type is cached on the node for the backend. */
        return t;
    }
    case MGL_EXPR_VAR_REF: {
        Sym *sym = symtab_lookup(tab, e->u.var_ref.name);
        if (!sym) {
            sema_error(s, e->line, "undeclared identifier '%s'",
                       e->u.var_ref.name);
            return NULL;
        }
        if (sym->kind == SYM_FUNCTION) {
            sema_error(s, e->line, "'%s' is a function, not a variable",
                       e->u.var_ref.name);
            return NULL;
        }
        return mglIRTypeScalar(sym->type->scalar) ? sym->type : NULL;
    }
    case MGL_EXPR_MEMBER: {
        MGLIRType *obj = check_expr(s, tab, e->u.member.object);
        if (!obj) {
            return NULL;
        }
        const MGLIRType *m = struct_member(obj, e->u.member.field, NULL);
        if (!m) {
            sema_error(s, e->line, "no member named '%s' in struct",
                       e->u.member.field);
            return NULL;
        }
        return (MGLIRType *)m;
    }
    case MGL_EXPR_INDEX: {
        MGLIRType *obj = check_expr(s, tab, e->u.index.object);
        if (!obj) {
            return NULL;
        }
        MGLIRType *idx = check_expr(s, tab, e->u.index.index);
        if (idx && !is_numeric(idx)) {
            sema_error(s, e->line, "array index must be an integer");
            return NULL;
        }
        if (obj->kind == MGLIR_TYPE_ARRAY) {
            return obj->elem_type;
        }
        if (obj->kind == MGLIR_TYPE_MATRIX) {
            /* matrix[i] yields a column vector */
            return mglIRTypeVector(obj->scalar, obj->rows);
        }
        sema_error(s, e->line, "indexing a non-array type");
        return NULL;
    }
    case MGL_EXPR_CALL: {
        /* Look up the function; single-definition match at skeleton stage. */
        Sym *sym = symtab_lookup(tab, e->u.call.name);
        if (!sym || sym->kind != SYM_FUNCTION) {
            /* builtin constructors: T(x) forms are handled by the backend;
             * here we accept a type-name call and return its type. */
            MGLTypeSpec fake;
            memset(&fake, 0, sizeof(fake));
            if (strcmp(e->u.call.name, "vec2") == 0) {
                fake.base = MGL_AST_TYPE_FLOAT; fake.vec_size = 2;
            } else if (strcmp(e->u.call.name, "vec3") == 0) {
                fake.base = MGL_AST_TYPE_FLOAT; fake.vec_size = 3;
            } else if (strcmp(e->u.call.name, "vec4") == 0) {
                fake.base = MGL_AST_TYPE_FLOAT; fake.vec_size = 4;
            } else if (strcmp(e->u.call.name, "mat4") == 0) {
                fake.base = MGL_AST_TYPE_FLOAT; fake.mat_cols = 4; fake.mat_rows = 4;
            } else if (strcmp(e->u.call.name, "mat3") == 0) {
                fake.base = MGL_AST_TYPE_FLOAT; fake.mat_cols = 3; fake.mat_rows = 3;
            } else if (strcmp(e->u.call.name, "mat2") == 0) {
                fake.base = MGL_AST_TYPE_FLOAT; fake.mat_cols = 2; fake.mat_rows = 2;
            } else {
                sema_error(s, e->line, "call to undeclared function '%s'",
                           e->u.call.name);
                return NULL;
            }
            MGLIRType *t = resolve_type_spec(s, tab, &fake);
            if (t) {
                /* constructors have no real function body; argument count is
                 * checked loosely here, exact checking with the builtin table */
                for (uint32_t i = 0; i < e->u.call.arg_count; i++) {
                    check_expr(s, tab, e->u.call.args[i]);
                }
            }
            return t;
        }
        if (e->u.call.arg_count != sym->param_count) {
            sema_error(s, e->line, "function '%s' expects %u argument(s), got %u",
                       e->u.call.name, sym->param_count, e->u.call.arg_count);
            return NULL;
        }
        for (uint32_t i = 0; i < e->u.call.arg_count; i++) {
            MGLIRType *at = check_expr(s, tab, e->u.call.args[i]);
            if (at && sym->param_types && sym->param_types[i]) {
                if (!check_assign_op(sym->param_types[i], at)) {
                    sema_error(s, e->line, "argument %u of '%s' expects %s, got %s",
                               i + 1, e->u.call.name,
                               ir_type_str(sym->param_types[i], ta, sizeof(ta)),
                               ir_type_str(at, tb, sizeof(tb)));
                }
            }
        }
        return sym->ret_type;
    }
    case MGL_EXPR_UNARY: {
        MGLIRType *o = check_expr(s, tab, e->u.unary.operand);
        if (!o) {
            return NULL;
        }
        switch (e->u.unary.op) {
        case MGL_OP_NOT:
            if (o->kind != MGLIR_TYPE_SCALAR || o->scalar != MGLIR_SCALAR_BOOL) {
                sema_error(s, e->line, "logical not requires bool");
                return NULL;
            }
            return mglIRTypeScalar(MGLIR_SCALAR_BOOL);
        case MGL_OP_BNOT:
            if (!is_numeric(o) || o->scalar == MGLIR_SCALAR_FLOAT ||
                o->scalar == MGLIR_SCALAR_DOUBLE) {
                sema_error(s, e->line, "bitwise not requires integer operand");
                return NULL;
            }
            return o;
        case MGL_OP_ADD:
        case MGL_OP_SUB:
            if (!is_numeric(o)) {
                sema_error(s, e->line, "unary +/- requires numeric operand");
                return NULL;
            }
            return o;
        case MGL_OP_INC:
        case MGL_OP_DEC:
            if (!is_numeric(o)) {
                sema_error(s, e->line, "++/-- requires numeric lvalue");
                return NULL;
            }
            return o;
        default:
            return o;
        }
    }
    case MGL_EXPR_BINARY: {
        MGLIRType *l = check_expr(s, tab, e->u.binary.lhs);
        MGLIRType *r = check_expr(s, tab, e->u.binary.rhs);
        if (!l || !r) {
            return NULL;
        }
        switch (e->u.binary.op) {
        case MGL_OP_EQ:
        case MGL_OP_NE:
            if (!ir_type_equal(l, r)) {
                sema_error(s, e->line, "operands of '%s' must have identical types (%s vs %s)",
                           op_name(e->u.binary.op),
                           ir_type_str(l, ta, sizeof(ta)),
                           ir_type_str(r, tb, sizeof(tb)));
                return NULL;
            }
            return mglIRTypeScalar(MGLIR_SCALAR_BOOL);
        case MGL_OP_LT: case MGL_OP_LE: case MGL_OP_GT: case MGL_OP_GE:
            if (!is_numeric(l) || !is_numeric(r)) {
                sema_error(s, e->line, "relational '%s' requires numeric operands",
                           op_name(e->u.binary.op));
                return NULL;
            }
            return mglIRTypeScalar(MGLIR_SCALAR_BOOL);
        case MGL_OP_LAND: case MGL_OP_LOR:
            if (l->scalar != MGLIR_SCALAR_BOOL || r->scalar != MGLIR_SCALAR_BOOL) {
                sema_error(s, e->line, "logical '%s' requires bool operands",
                           op_name(e->u.binary.op));
                return NULL;
            }
            return mglIRTypeScalar(MGLIR_SCALAR_BOOL);
        case MGL_OP_AND: case MGL_OP_OR: case MGL_OP_XOR:
        case MGL_OP_SHL: case MGL_OP_SHR:
            if (!is_numeric(l) || !is_numeric(r) ||
                l->scalar == MGLIR_SCALAR_FLOAT || l->scalar == MGLIR_SCALAR_DOUBLE ||
                r->scalar == MGLIR_SCALAR_FLOAT || r->scalar == MGLIR_SCALAR_DOUBLE) {
                sema_error(s, e->line, "bitwise '%s' requires integer operands",
                           op_name(e->u.binary.op));
                return NULL;
            }
            return l;
        case MGL_OP_ADD: case MGL_OP_SUB:
            if (l->kind == MGLIR_TYPE_MATRIX || r->kind == MGLIR_TYPE_MATRIX) {
                if (l->kind == MGLIR_TYPE_MATRIX && r->kind == MGLIR_TYPE_MATRIX &&
                    (l->cols != r->cols || l->rows != r->rows)) {
                    sema_error(s, e->line, "matrix add/sub requires matching dimensions");
                    return NULL;
                }
                return l;
            }
            /* fallthrough */
        case MGL_OP_MUL: case MGL_OP_DIV: case MGL_OP_MOD:
            if (!is_numeric(l) || !is_numeric(r)) {
                sema_error(s, e->line, "arithmetic '%s' requires numeric operands",
                           op_name(e->u.binary.op));
                return NULL;
            }
            if (e->u.binary.op == MGL_OP_MOD &&
                (l->scalar == MGLIR_SCALAR_FLOAT || l->scalar == MGLIR_SCALAR_DOUBLE)) {
                sema_error(s, e->line, "'%%' requires integer operands");
                return NULL;
            }
            return result_numeric(l, r, 0);
        default:
            return l;
        }
    }
    case MGL_EXPR_ASSIGN: {
        MGLIRType *l = check_expr(s, tab, e->u.assign.lhs);
        MGLIRType *r = check_expr(s, tab, e->u.assign.rhs);
        if (!l || !r) {
            return NULL;
        }
        if (!check_assign_op(l, r)) {
            sema_error(s, e->line, "cannot assign %s to %s",
                       ir_type_str(r, ta, sizeof(ta)),
                       ir_type_str(l, tb, sizeof(tb)));
            return NULL;
        }
        return l;
    }
    case MGL_EXPR_TERNARY: {
        MGLIRType *c = check_expr(s, tab, e->u.ternary.cond);
        MGLIRType *a = check_expr(s, tab, e->u.ternary.then);
        MGLIRType *b = check_expr(s, tab, e->u.ternary.else_);
        if (!c || !a || !b) {
            return NULL;
        }
        if (c->scalar != MGLIR_SCALAR_BOOL) {
            sema_error(s, e->line, "ternary condition must be bool");
            return NULL;
        }
        if (!ir_type_equal(a, b)) {
            sema_error(s, e->line, "ternary branches must have identical types (%s vs %s)",
                       ir_type_str(a, ta, sizeof(ta)),
                       ir_type_str(b, tb, sizeof(tb)));
            return NULL;
        }
        return a;
    }
    default:
        return NULL;
    }
}

/* ------------------------------------------------------------------ */
/* Block layout computation                                            */
/* ------------------------------------------------------------------ */

static void layout_block(Sema *s, const MGLDecl *d, MGLIRType *block_type)
{
    MGLIRLayoutStd std = MGLIR_LAYOUT_NONE;
    switch (d->layout) {
    case MGL_AST_LAYOUT_STD140: std = MGLIR_LAYOUT_STD140; break;
    case MGL_AST_LAYOUT_STD430: std = MGLIR_LAYOUT_STD430; break;
    case MGL_AST_LAYOUT_SHARED: std = MGLIR_LAYOUT_SHARED; break;
    case MGL_AST_LAYOUT_PACKED: std = MGLIR_LAYOUT_PACKED; break;
    default: std = MGLIR_LAYOUT_STD140; break;
    }
    uint32_t size = 0;
    if (mglIRComputeLayout(block_type, std, &size) != 0) {
        sema_error(s, d->line, "failed to compute layout for block '%s'",
                   d->name ? d->name : "?");
    }
    (void)size;
}

/* ------------------------------------------------------------------ */
/* Declarations                                                        */
/* ------------------------------------------------------------------ */

static void analyze_decl(Sema *s, SymTab *tab, const MGLDecl *d, int global);
static void analyze_stmt(Sema *s, SymTab *tab, const MGLStmt *st);

static void analyze_function(Sema *s, SymTab *tab, const MGLDecl *d)
{
    Sym *sym = sym_new(d->name);
    if (!sym) {
        return;
    }
    sym->kind = SYM_FUNCTION;
    sym->param_count = d->param_count;
    if (d->param_count) {
        sym->param_types = (MGLIRType **)calloc(d->param_count, sizeof(MGLIRType *));
        if (!sym->param_types) {
            free(sym->name);
            free(sym);
            return;
        }
    }
    sym->ret_type = resolve_type_spec(s, tab, d->return_type);
    for (uint32_t i = 0; i < d->param_count; i++) {
        MGLDecl *pd = d->params[i];
        sym->param_types[i] = resolve_decl_type(s, tab, pd);
        if (!sym->param_types[i]) {
            return;
        }
    }
    if (symtab_lookup_local(tab, d->name) != NULL) {
        sema_error(s, d->line, "redeclaration of '%s'", d->name);
        free(sym->name);
        free(sym->param_types);
        if (sym->ret_type) {
            mglIRTypeDestroy(sym->ret_type);
        }
        return;
    }
    symtab_insert(tab, sym);

    /* Add the function to the module (prototypes and definitions alike). */
    MGLIRSymbol *isym = (MGLIRSymbol *)calloc(1, sizeof(*isym));
    if (isym) {
        isym->name = strdup(d->name);
        isym->is_function = 1;
        isym->return_type = ir_type_clone(sym->ret_type);
        isym->param_count = d->param_count;
        if (d->param_count) {
            isym->param_types = (MGLIRType **)calloc(d->param_count,
                                                     sizeof(MGLIRType *));
            if (isym->param_types) {
                for (uint32_t i = 0; i < d->param_count; i++) {
                    isym->param_types[i] = ir_type_clone(sym->param_types[i]);
                }
            }
        }
        s->module->symbols = (MGLIRSymbol **)realloc(
            s->module->symbols,
            (s->module->symbol_count + 1) * sizeof(MGLIRSymbol *));
        if (s->module->symbols) {
            s->module->symbols[s->module->symbol_count++] = isym;
        } else {
            free(isym->name);
            free(isym);
        }
    }

    /* Function body: push a scope containing parameters. */
    if (d->body) {
        symtab_push(tab);
        for (uint32_t i = 0; i < d->param_count; i++) {
            Sym *ps = sym_new(d->params[i]->name ? d->params[i]->name : "");
            if (ps) {
                ps->kind = SYM_VARIABLE;
                ps->type = sym->param_types[i];
                symtab_insert(tab, ps);
            }
        }
        analyze_stmt(s, tab, d->body);
        symtab_pop(tab);
    }
}

static void analyze_variable(Sema *s, SymTab *tab, const MGLDecl *d, int global)
{
    MGLIRType *t = resolve_decl_type(s, tab, d);
    if (!t) {
        return;
    }
    if (symtab_lookup_local(tab, d->name) != NULL) {
        sema_error(s, d->line, "redeclaration of '%s'", d->name);
        mglIRTypeDestroy(t);
        return;
    }
    Sym *sym = sym_new(d->name);
    if (!sym) {
        mglIRTypeDestroy(t);
        return;
    }
    sym->kind = SYM_VARIABLE;
    sym->type = t;
    sym->type_owned = 1;
    sym->qualifiers = d->qualifiers;
    symtab_insert(tab, sym);

    if (global) {
        MGLIRSymbol *isym = (MGLIRSymbol *)calloc(1, sizeof(*isym));
        if (isym) {
            isym->name = strdup(d->name);
            isym->type = t;              /* owned by module now */
            isym->qualifiers = d->qualifiers;
            isym->layout = d->layout;
            isym->matrix_major = d->matrix_major;
            isym->offset = UINT32_MAX;
            isym->binding = UINT32_MAX;
            isym->location = UINT32_MAX;
            /* layout block: compute offsets on the block type */
            if (d->struct_members && d->struct_member_count > 0) {
                layout_block(s, d, t);
            }
            s->module->symbols = (MGLIRSymbol **)realloc(
                s->module->symbols,
                (s->module->symbol_count + 1) * sizeof(MGLIRSymbol *));
            if (s->module->symbols) {
                s->module->symbols[s->module->symbol_count++] = isym;
                /* ownership of `t` moved to the module; sym borrows it so the
                 * type stays resolvable inside function bodies */
                sym->type_owned = 0;
            } else {
                free(isym->name);
                free(isym);
                mglIRTypeDestroy(t);
                sym->type_owned = 0;
            }
        }
    }
    if (d->init) {
        MGLIRType *it = check_expr(s, tab, d->init);
        if (it && !check_assign_op(t, it)) {
            sema_error(s, d->line, "initializer type mismatch in declaration of '%s'",
                       d->name);
        }
    }
}

static void analyze_decl(Sema *s, SymTab *tab, const MGLDecl *d, int global)
{
    if (!d) {
        return;
    }
    if (d->body || d->params) {
        analyze_function(s, tab, d);
    } else {
        analyze_variable(s, tab, d, global);
    }
}

/* ------------------------------------------------------------------ */
/* Statements                                                          */
/* ------------------------------------------------------------------ */

static void analyze_stmt(Sema *s, SymTab *tab, const MGLStmt *st)
{
    if (!st) {
        return;
    }
    switch (st->kind) {
    case MGL_STMT_COMPOUND: {
        symtab_push(tab);
        for (uint32_t i = 0; i < st->u.compound.count; i++) {
            analyze_stmt(s, tab, st->u.compound.stmts[i]);
        }
        symtab_pop(tab);
        break;
    }
    case MGL_STMT_EXPR:
        check_expr(s, tab, st->u.expr.expr);
        break;
    case MGL_STMT_DECL:
        analyze_decl(s, tab, st->u.decl.decl, 0);
        break;
    case MGL_STMT_IF:
        check_expr(s, tab, st->u.ifs.cond);
        analyze_stmt(s, tab, st->u.ifs.then);
        if (st->u.ifs.else_) {
            analyze_stmt(s, tab, st->u.ifs.else_);
        }
        break;
    case MGL_STMT_FOR: {
        symtab_push(tab);
        if (st->u.loop.init) {
            analyze_stmt(s, tab, st->u.loop.init);
        }
        if (st->u.loop.cond) {
            check_expr(s, tab, st->u.loop.cond);
        }
        if (st->u.loop.incr) {
            check_expr(s, tab, st->u.loop.incr);
        }
        if (st->u.loop.body) {
            analyze_stmt(s, tab, st->u.loop.body);
        }
        symtab_pop(tab);
        break;
    }
    case MGL_STMT_WHILE:
        check_expr(s, tab, st->u.whilex.cond);
        analyze_stmt(s, tab, st->u.whilex.body);
        break;
    case MGL_STMT_DO_WHILE:
        analyze_stmt(s, tab, st->u.body.body);
        check_expr(s, tab, st->u.whilex.cond);
        break;
    case MGL_STMT_SWITCH:
        check_expr(s, tab, st->u.switchx.cond);
        analyze_stmt(s, tab, st->u.switchx.body);
        break;
    case MGL_STMT_CASE:
        check_expr(s, tab, st->u.casex.value);
        break;
    case MGL_STMT_DEFAULT:
    case MGL_STMT_BREAK:
    case MGL_STMT_CONTINUE:
    case MGL_STMT_DISCARD:
        break;
    case MGL_STMT_RETURN:
        if (st->u.ret.value) {
            check_expr(s, tab, st->u.ret.value);
        }
        break;
    default:
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

int mglGLSLSemanticCheck(const MGLTranslationUnit *tu, MGLIRModule *module,
                         MGLSemaError **errors, uint32_t *error_count)
{
    if (!tu || !module) {
        return -1;
    }
    memset(module, 0, sizeof(*module));

    Sema s;
    memset(&s, 0, sizeof(s));
    s.tu = tu;
    s.module = module;

    SymTab tab;
    memset(&tab, 0, sizeof(tab));
    if (symtab_push(&tab) != 0) {
        mglGLSLSemanticCheckDestroy(s.errors, s.error_count);
        return -1;
    }

    /* struct declarations first (they may be referenced by later decls) */
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        MGLDecl *d = tu->decls[i];
        if (d->type && d->type->base == MGL_AST_TYPE_STRUCT &&
            d->struct_members && d->struct_member_count > 0) {
            /* register struct name */
            Sym *sym = sym_new(d->type->name ? d->type->name : d->name);
            if (sym) {
                sym->kind = SYM_STRUCT;
                /* build IR struct type */
                uint32_t n = d->struct_member_count;
                MGLIRType **members = (MGLIRType **)calloc(n, sizeof(MGLIRType *));
                const char **names = (const char **)calloc(n, sizeof(char *));
                if (members && names) {
                    int ok = 1;
                    for (uint32_t j = 0; j < n; j++) {
                        MGLDecl *m = d->struct_members[j];
                        members[j] = resolve_decl_type(&s, &tab, m);
                        names[j] = m->name;
                        if (!members[j]) {
                            ok = 0;
                            break;
                        }
                    }
                    if (ok) {
                        MGLIRType *st = mglIRTypeStruct(
                            members, names, n,
                            d->type->name ? d->type->name : d->name);
                        if (st) {
                            sym->type = st;
                            sym->type_owned = 1;
                            symtab_insert(&tab, sym);
                        } else {
                            free(sym->name);
                            free(sym);
                            sym = NULL;
                        }
                    } else {
                        for (uint32_t j = 0; j < n; j++) {
                            if (members[j]) {
                                mglIRTypeDestroy(members[j]);
                            }
                        }
                        free(sym->name);
                        free(sym);
                        sym = NULL;
                    }
                    free(members);
                    free(names);
                } else {
                    free(members);
                    free(names);
                    free(sym->name);
                    free(sym);
                    sym = NULL;
                }
            }
        }
    }

    for (uint32_t i = 0; i < tu->decl_count; i++) {
        analyze_decl(&s, &tab, tu->decls[i], 1);
    }

    if (errors) {
        *errors = s.errors;
    }
    if (error_count) {
        *error_count = s.error_count;
    }
    symtab_destroy(&tab);
    return (int)s.error_count;
}

MGLIRSymbol *mglIRSymbolNew(const char *name, MGLIRType *type)
{
    MGLIRSymbol *is = (MGLIRSymbol *)calloc(1, sizeof(*is));
    if (!is) {
        return NULL;
    }
    is->name = strdup(name);
    is->type = type;
    is->binding = UINT32_MAX;
    is->location = UINT32_MAX;
    is->offset = UINT32_MAX;
    return is;
}

void mglGLSLSemanticCheckDestroy(MGLSemaError *errors, uint32_t count)
{
    for (uint32_t i = 0; i < count; i++) {
        free(errors[i].message);
    }
    free(errors);
}

void mglIRModuleDestroy(MGLIRModule *module)
{
    if (!module) {
        return;
    }
    for (uint32_t i = 0; i < module->symbol_count; i++) {
        MGLIRSymbol *is = module->symbols[i];
        if (!is) {
            continue;
        }
        free(is->name);
        if (is->type) {
            mglIRTypeDestroy(is->type);
        }
        if (is->return_type) {
            mglIRTypeDestroy(is->return_type);
        }
        for (uint32_t j = 0; j < is->param_count; j++) {
            if (is->param_types && is->param_types[j]) {
                mglIRTypeDestroy(is->param_types[j]);
            }
        }
        free(is->param_types);
        free(is);
    }
    free(module->symbols);
    module->symbols = NULL;
    module->symbol_count = 0;
}
