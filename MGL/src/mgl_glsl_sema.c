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
    /* scratch types created by expression typing (check_expr etc.); the M0
     * module does not hold typed expression IR, so these have no owner.
     * Collected here and destroyed together at the end of the check. */
    MGLIRType **tmp_types;
    uint32_t tmp_count;
    uint32_t tmp_cap;
} Sema;

static MGLIRType *scratch_type(Sema *s, MGLIRType *t)
{
    if (!t || !s) {
        return t;
    }
    if (s->tmp_count == s->tmp_cap) {
        uint32_t ncap = s->tmp_cap ? s->tmp_cap * 2 : 16;
        MGLIRType **n = (MGLIRType **)realloc(
            s->tmp_types, ncap * sizeof(MGLIRType *));
        if (!n) {
            return t;
        }
        s->tmp_types = n;
        s->tmp_cap = ncap;
    }
    s->tmp_types[s->tmp_count++] = t;
    return t;
}

static void scratch_destroy(Sema *s)
{
    for (uint32_t i = 0; i < s->tmp_count; i++) {
        mglIRTypeDestroy(s->tmp_types[i]);
    }
    free(s->tmp_types);
    s->tmp_types = NULL;
    s->tmp_count = s->tmp_cap = 0;
}

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

/* Destroy a Sym that was never inserted into a symbol table (all members
 * are owned).  Inserted syms are reclaimed via symtab_destroy. */
static void sym_free(Sym *sym)
{
    if (!sym) {
        return;
    }
    free(sym->name);
    if (sym->ret_type) {
        mglIRTypeDestroy(sym->ret_type);
    }
    if (sym->param_types) {
        for (uint32_t i = 0; i < sym->param_count; i++) {
            if (sym->param_types[i]) {
                mglIRTypeDestroy(sym->param_types[i]);
            }
        }
        free(sym->param_types);
    }
    if (sym->type && sym->type_owned) {
        mglIRTypeDestroy(sym->type);
    }
    free(sym);
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
    t->name = src->name ? strdup(src->name) : NULL;
    if (src->name && !t->name) {
        free(t);
        return NULL;
    }
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
            t->member_names[i] =
                src->member_names[i] ? strdup(src->member_names[i]) : NULL;
            if (src->member_names[i] && !t->member_names[i]) {
                for (uint32_t j = 0; j <= i; j++) {
                    mglIRTypeDestroy(t->members[j]);
                }
                free(t->member_names);
                free(t->members);
                free(t);
                return NULL;
            }
            if (!t->members[i]) {
                for (uint32_t j = 0; j < i; j++) {
                    mglIRTypeDestroy(t->members[j]);
                }
                free(t->member_names);
                free(t->members);
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
static int builtin_type_spec(const char *name, MGLTypeSpec *ts);

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

/* Strict interface matching (GLSL 4.60 §4.3.9.5): structs compare
 * member names, types and order; arrays compare dimensions recursively. */
static int ir_type_interface_equal(const MGLIRType *a, const MGLIRType *b)
{
    if (a == b) {
        return 1;
    }
    if (!a || !b || a->kind != b->kind || a->scalar != b->scalar) {
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
        return a->array_size == b->array_size &&
               ir_type_interface_equal(a->elem_type, b->elem_type);
    case MGLIR_TYPE_STRUCT:
        if (a->member_count != b->member_count) {
            return 0;
        }
        for (uint32_t i = 0; i < a->member_count; i++) {
            if (strcmp(a->member_names ? a->member_names[i] : "",
                       b->member_names ? b->member_names[i] : "") != 0) {
                return 0;
            }
            if (!ir_type_interface_equal(a->members[i], b->members[i])) {
                return 0;
            }
        }
        return 1;
    case MGLIR_TYPE_SAMPLER:
    case MGLIR_TYPE_IMAGE:
        return a->tex_kind == b->tex_kind && a->tex_depth == b->tex_depth;
    default:
        return 0;
    }
}

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
static int implicit_convert(const MGLIRType *from, const MGLIRType *to)
{
    if (!from || !to || from->kind != MGLIR_TYPE_SCALAR ||
        to->kind != MGLIR_TYPE_SCALAR) {
        return 0;
    }
    if (from->scalar == to->scalar) {
        return 1; /* identical types (incl. bool/void) always convert */
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
    if (!t) {
        return 0;
    }
    switch (t->kind) {
    case MGLIR_TYPE_SCALAR:
    case MGLIR_TYPE_VECTOR:
    case MGLIR_TYPE_MATRIX:
        return t->scalar == MGLIR_SCALAR_INT ||
               t->scalar == MGLIR_SCALAR_UINT ||
               t->scalar == MGLIR_SCALAR_FLOAT ||
               t->scalar == MGLIR_SCALAR_DOUBLE;
    default:
        return 0;
    }
}

static MGLIRType *result_numeric(Sema *s, MGLIRType *a, MGLIRType *b)
{
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
        return scratch_type(s, v);
    }
    return scratch_type(s, base);
}

/* GLSL 4.60 matrix multiplication typing:
 *   mat * vec  = vec (len = mat.rows), mat.cols == vec.len
 *   vec * mat  = vec (len = mat.cols), vec.len == mat.rows
 *   mat * mat  = mat, a.cols == b.rows, result (b.cols x a.rows) */
static MGLIRType *matrix_mul_result(Sema *s, const MGLExpr *e, MGLIRType *l,
                                    MGLIRType *r)
{
    MGLIRScalar sc = l->scalar;
    if (r->scalar == MGLIR_SCALAR_DOUBLE ||
        (r->scalar == MGLIR_SCALAR_FLOAT && sc != MGLIR_SCALAR_DOUBLE)) {
        sc = r->scalar;
    }
    if (l->kind == MGLIR_TYPE_MATRIX && r->kind == MGLIR_TYPE_MATRIX) {
        if (l->cols != r->rows) {
            sema_error(s, e->line, "matrix multiply dimension mismatch (%ux%u * %ux%u)",
                       l->cols, l->rows, r->cols, r->rows);
            return NULL;
        }
        return scratch_type(s, mglIRTypeMatrix(sc, r->cols, l->rows));
    }
    if (l->kind == MGLIR_TYPE_MATRIX && r->kind == MGLIR_TYPE_VECTOR) {
        if (l->cols != r->cols) {
            sema_error(s, e->line,
                       "matrix %ux%u must be multiplied by a vector of length %u",
                       l->cols, l->rows, l->cols);
            return NULL;
        }
        return scratch_type(s, mglIRTypeVector(sc, l->rows));
    }
    if (l->kind == MGLIR_TYPE_VECTOR && r->kind == MGLIR_TYPE_MATRIX) {
        if (l->cols != r->rows) {
            sema_error(s, e->line,
                       "vector of length %u must be multiplied by a matrix with %u rows",
                       l->cols, r->rows);
            return NULL;
        }
        return scratch_type(s, mglIRTypeVector(sc, r->cols));
    }
    /* matrix * scalar: same shape */
    return l->kind == MGLIR_TYPE_MATRIX ? l : r;
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

/* ------------------------------------------------------------------ */
/* Builtin functions (first-wave table)                               */
/* ------------------------------------------------------------------ */

typedef enum {
    BI_ARG_GENF,    /* float genType: float/vec2/vec3/vec4, all gen args
                     * must share the same dimensionality */
    BI_ARG_GENI,    /* int/uint genType: int/ivec/uint/uvec (same sharing
                     * rule); return type follows the signedness */
    BI_ARG_FLOAT,   /* scalar float (int/uint scalar implicitly ok) */
    BI_ARG_INT,     /* scalar int or uint */
    BI_ARG_VEC2,    /* vec2 */
    BI_ARG_VEC3,    /* vec3 */
    BI_ARG_VEC4,    /* vec4 */
    BI_ARG_MAT2,    /* mat2 */
    BI_ARG_MAT3,    /* mat3 */
    BI_ARG_MAT4,    /* mat4 */
    BI_ARG_S2D,     /* sampler2D */
    BI_ARG_S3D,     /* sampler3D */
    BI_ARG_SCUBE,   /* samplerCube */
} BiArgKind;

typedef enum {
    BI_RET_FLOAT,   /* scalar float */
    BI_RET_UINT,    /* scalar uint */
    BI_RET_GENF,    /* float genType matching the gen args */
    BI_RET_GENI,    /* int/uint genType matching the gen args */
    BI_RET_VEC2,    /* vec2 */
    BI_RET_VEC3,    /* vec3 */
    BI_RET_VEC4,    /* vec4 */
    BI_RET_MAT2,    /* mat2 */
    BI_RET_MAT3,    /* mat3 */
    BI_RET_MAT4,    /* mat4 */
} BiRetKind;

typedef struct {
    const char *name;
    uint32_t argc;
    const BiArgKind args[4];
    BiRetKind ret;
} BiFn;

static const BiFn kBuiltins[] = {
    { "texture",    3, { BI_ARG_S2D,   BI_ARG_VEC2, BI_ARG_FLOAT }, BI_RET_VEC4 },
    { "texture",    3, { BI_ARG_S3D,   BI_ARG_VEC3, BI_ARG_FLOAT }, BI_RET_VEC4 },
    { "texture",    3, { BI_ARG_SCUBE, BI_ARG_VEC3, BI_ARG_FLOAT }, BI_RET_VEC4 },
    { "texture",    2, { BI_ARG_S2D,   BI_ARG_VEC2 }, BI_RET_VEC4 },
    { "texture",    2, { BI_ARG_S3D,   BI_ARG_VEC3 }, BI_RET_VEC4 },
    { "texture",    2, { BI_ARG_SCUBE, BI_ARG_VEC3 }, BI_RET_VEC4 },
    { "textureLod", 3, { BI_ARG_S2D,   BI_ARG_VEC2, BI_ARG_FLOAT }, BI_RET_VEC4 },
    { "textureLod", 3, { BI_ARG_S3D,   BI_ARG_VEC3, BI_ARG_FLOAT }, BI_RET_VEC4 },
    { "textureLod", 3, { BI_ARG_SCUBE, BI_ARG_VEC3, BI_ARG_FLOAT }, BI_RET_VEC4 },
    { "textureSize", 2, { BI_ARG_S2D,   BI_ARG_FLOAT }, BI_RET_VEC2 },
    { "textureSize", 2, { BI_ARG_S3D,   BI_ARG_FLOAT }, BI_RET_VEC3 },
    { "textureSize", 2, { BI_ARG_SCUBE, BI_ARG_FLOAT }, BI_RET_VEC2 },
    { "normalize", 1, { BI_ARG_GENF }, BI_RET_GENF },
    { "length",    1, { BI_ARG_GENF }, BI_RET_FLOAT },
    { "distance",  2, { BI_ARG_GENF, BI_ARG_GENF }, BI_RET_FLOAT },
    { "dot",       2, { BI_ARG_GENF, BI_ARG_GENF }, BI_RET_FLOAT },
    { "abs",       1, { BI_ARG_GENI }, BI_RET_GENI },
    { "abs",       1, { BI_ARG_GENF }, BI_RET_GENF },
    { "min",       2, { BI_ARG_GENI, BI_ARG_GENI }, BI_RET_GENI },
    { "min",       2, { BI_ARG_GENI, BI_ARG_INT }, BI_RET_GENI },
    { "min",       2, { BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    { "min",       2, { BI_ARG_GENF, BI_ARG_FLOAT }, BI_RET_GENF },
    { "max",       2, { BI_ARG_GENI, BI_ARG_GENI }, BI_RET_GENI },
    { "max",       2, { BI_ARG_GENI, BI_ARG_INT }, BI_RET_GENI },
    { "max",       2, { BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    { "max",       2, { BI_ARG_GENF, BI_ARG_FLOAT }, BI_RET_GENF },
    { "clamp",     3, { BI_ARG_GENI, BI_ARG_GENI, BI_ARG_GENI }, BI_RET_GENI },
    { "clamp",     3, { BI_ARG_GENI, BI_ARG_INT, BI_ARG_INT }, BI_RET_GENI },
    { "clamp",     3, { BI_ARG_GENF, BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    { "clamp",     3, { BI_ARG_GENF, BI_ARG_FLOAT, BI_ARG_FLOAT }, BI_RET_GENF },
    { "mix",       3, { BI_ARG_GENF, BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    { "mix",       3, { BI_ARG_GENF, BI_ARG_GENF, BI_ARG_FLOAT }, BI_RET_GENF },
    /* trigonometric */
    { "sin",  1, { BI_ARG_GENF }, BI_RET_GENF },
    { "cos",  1, { BI_ARG_GENF }, BI_RET_GENF },
    { "tan",  1, { BI_ARG_GENF }, BI_RET_GENF },
    /* exponential */
    { "exp",  1, { BI_ARG_GENF }, BI_RET_GENF },
    { "exp2", 1, { BI_ARG_GENF }, BI_RET_GENF },
    { "log",  1, { BI_ARG_GENF }, BI_RET_GENF },
    { "log2", 1, { BI_ARG_GENF }, BI_RET_GENF },
    { "pow",  2, { BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    { "sqrt", 1, { BI_ARG_GENF }, BI_RET_GENF },
    { "inversesqrt", 1, { BI_ARG_GENF }, BI_RET_GENF },
    /* common */
    { "floor",     1, { BI_ARG_GENF }, BI_RET_GENF },
    { "ceil",      1, { BI_ARG_GENF }, BI_RET_GENF },
    { "trunc",     1, { BI_ARG_GENF }, BI_RET_GENF },
    { "round",     1, { BI_ARG_GENF }, BI_RET_GENF },
    { "roundEven", 1, { BI_ARG_GENF }, BI_RET_GENF },
    { "fract",     1, { BI_ARG_GENF }, BI_RET_GENF },
    { "sign",      1, { BI_ARG_GENF }, BI_RET_GENF },
    { "mod",       2, { BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    { "mod",       2, { BI_ARG_GENF, BI_ARG_FLOAT }, BI_RET_GENF },
    { "step",      2, { BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    { "step",      2, { BI_ARG_FLOAT, BI_ARG_GENF }, BI_RET_GENF },
    { "smoothstep", 3, { BI_ARG_GENF, BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    { "smoothstep", 3, { BI_ARG_FLOAT, BI_ARG_FLOAT, BI_ARG_GENF }, BI_RET_GENF },
    /* geometric */
    { "reflect",    2, { BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    { "refract",    3, { BI_ARG_GENF, BI_ARG_GENF, BI_ARG_FLOAT }, BI_RET_GENF },
    { "faceforward", 3, { BI_ARG_GENF, BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    /* angle conversion */
    { "radians", 1, { BI_ARG_GENF }, BI_RET_GENF },
    { "degrees", 1, { BI_ARG_GENF }, BI_RET_GENF },
    /* matrix */
    { "transpose", 1, { BI_ARG_MAT2 }, BI_RET_MAT2 },
    { "transpose", 1, { BI_ARG_MAT3 }, BI_RET_MAT3 },
    { "transpose", 1, { BI_ARG_MAT4 }, BI_RET_MAT4 },
    { "matrixCompMult", 2, { BI_ARG_MAT2, BI_ARG_MAT2 }, BI_RET_MAT2 },
    { "matrixCompMult", 2, { BI_ARG_MAT3, BI_ARG_MAT3 }, BI_RET_MAT3 },
    { "matrixCompMult", 2, { BI_ARG_MAT4, BI_ARG_MAT4 }, BI_RET_MAT4 },
    { "determinant", 1, { BI_ARG_MAT2 }, BI_RET_FLOAT },
    { "determinant", 1, { BI_ARG_MAT3 }, BI_RET_FLOAT },
    { "determinant", 1, { BI_ARG_MAT4 }, BI_RET_FLOAT },
    { "inverse", 1, { BI_ARG_MAT2 }, BI_RET_MAT2 },
    { "inverse", 1, { BI_ARG_MAT3 }, BI_RET_MAT3 },
    { "inverse", 1, { BI_ARG_MAT4 }, BI_RET_MAT4 },
    { "outerProduct", 2, { BI_ARG_VEC2, BI_ARG_VEC2 }, BI_RET_MAT2 },
    { "outerProduct", 2, { BI_ARG_VEC3, BI_ARG_VEC3 }, BI_RET_MAT3 },
    { "outerProduct", 2, { BI_ARG_VEC4, BI_ARG_VEC4 }, BI_RET_MAT4 },
    /* geometric */
    { "cross", 2, { BI_ARG_VEC3, BI_ARG_VEC3 }, BI_RET_VEC3 },
    /* inverse trigonometric */
    { "asin", 1, { BI_ARG_GENF }, BI_RET_GENF },
    { "acos", 1, { BI_ARG_GENF }, BI_RET_GENF },
    { "atan", 1, { BI_ARG_GENF }, BI_RET_GENF },
    { "atan", 2, { BI_ARG_GENF, BI_ARG_GENF }, BI_RET_GENF },
    /* pack / unpack (GLSL 4.60 8.4) */
    { "packUnorm2x16",   1, { BI_ARG_VEC2 }, BI_RET_UINT },
    { "unpackUnorm2x16", 1, { BI_ARG_INT }, BI_RET_VEC2 },
    { "packSnorm2x16",   1, { BI_ARG_VEC2 }, BI_RET_UINT },
    { "unpackSnorm2x16", 1, { BI_ARG_INT }, BI_RET_VEC2 },
    { "packHalf2x16",    1, { BI_ARG_VEC2 }, BI_RET_UINT },
    { "unpackHalf2x16",  1, { BI_ARG_INT }, BI_RET_VEC2 },
    /* atomic (compute) */
    { "atomicAdd", 2, { BI_ARG_GENI, BI_ARG_GENI }, BI_RET_GENI },
};

/* Does `t` satisfy a BI_ARG_GENF parameter?  Sets *gen_dim to the matched
 * dimensionality on success.  Numeric int/uint scalars implicitly convert
 * to float here, mirroring GLSL argument conversion. */
static int bif_gen_matches(const MGLIRType *t, uint32_t *gen_dim)
{
    if (!t) {
        return 0;
    }
    if (t->scalar != MGLIR_SCALAR_FLOAT &&
        !(t->kind == MGLIR_TYPE_SCALAR &&
          (t->scalar == MGLIR_SCALAR_INT || t->scalar == MGLIR_SCALAR_UINT))) {
        return 0;
    }
    if (t->kind == MGLIR_TYPE_VECTOR) {
        *gen_dim = t->cols;
        return 1;
    }
    if (t->kind == MGLIR_TYPE_SCALAR) {
        *gen_dim = 1;
        return 1;
    }
    return 0;
}

/* Does `t` satisfy a BI_ARG_GENI parameter (int/uint genType)?  Sets
 * *gen_dim and *gen_unsigned (signedness of the matched scalar). */
static int bif_geni_matches(const MGLIRType *t, uint32_t *gen_dim,
                            uint32_t *gen_unsigned)
{
    if (!t || (t->scalar != MGLIR_SCALAR_INT && t->scalar != MGLIR_SCALAR_UINT)) {
        return 0;
    }
    if (t->kind == MGLIR_TYPE_VECTOR) {
        *gen_dim = t->cols;
        *gen_unsigned = (t->scalar == MGLIR_SCALAR_UINT);
        return 1;
    }
    if (t->kind == MGLIR_TYPE_SCALAR) {
        *gen_dim = 1;
        *gen_unsigned = (t->scalar == MGLIR_SCALAR_UINT);
        return 1;
    }
    return 0;
}

/* Signedness carried out of the last matched BI_ARG_GENI (set per
 * signature by the arg-matching loop in builtin_call_type). */
static uint32_t bif_geni_unsigned;

static int bif_arg_matches(const MGLIRType *t, BiArgKind k, uint32_t *gen_dim)
{
    if (!t) {
        return 0;
    }
    switch (k) {
    case BI_ARG_GENF:
        return bif_gen_matches(t, gen_dim);
    case BI_ARG_GENI:
        return bif_geni_matches(t, gen_dim, &bif_geni_unsigned);
    case BI_ARG_FLOAT:
        return t->kind == MGLIR_TYPE_SCALAR &&
               (t->scalar == MGLIR_SCALAR_FLOAT || t->scalar == MGLIR_SCALAR_INT ||
                t->scalar == MGLIR_SCALAR_UINT);
    case BI_ARG_INT:
        return t->kind == MGLIR_TYPE_SCALAR &&
               (t->scalar == MGLIR_SCALAR_INT || t->scalar == MGLIR_SCALAR_UINT);
    case BI_ARG_VEC2:
        return t->kind == MGLIR_TYPE_VECTOR && t->cols == 2 &&
               t->scalar == MGLIR_SCALAR_FLOAT;
    case BI_ARG_VEC3:
        return t->kind == MGLIR_TYPE_VECTOR && t->cols == 3 &&
               t->scalar == MGLIR_SCALAR_FLOAT;
    case BI_ARG_VEC4:
        return t->kind == MGLIR_TYPE_VECTOR && t->cols == 4 &&
               t->scalar == MGLIR_SCALAR_FLOAT;
    case BI_ARG_MAT2:
        return t->kind == MGLIR_TYPE_MATRIX && t->cols == 2 && t->rows == 2 &&
               t->scalar == MGLIR_SCALAR_FLOAT;
    case BI_ARG_MAT3:
        return t->kind == MGLIR_TYPE_MATRIX && t->cols == 3 && t->rows == 3 &&
               t->scalar == MGLIR_SCALAR_FLOAT;
    case BI_ARG_MAT4:
        return t->kind == MGLIR_TYPE_MATRIX && t->cols == 4 && t->rows == 4 &&
               t->scalar == MGLIR_SCALAR_FLOAT;
    case BI_ARG_S2D:
        return t->kind == MGLIR_TYPE_SAMPLER && t->tex_kind == MGLIR_TEX_2D &&
               !t->tex_depth;
    case BI_ARG_S3D:
        return t->kind == MGLIR_TYPE_SAMPLER && t->tex_kind == MGLIR_TEX_3D &&
               !t->tex_depth;
    case BI_ARG_SCUBE:
        return t->kind == MGLIR_TYPE_SAMPLER && t->tex_kind == MGLIR_TEX_CUBE &&
               !t->tex_depth;
    default:
        return 0;
    }
}

/* Match a builtin call.  On success returns a new MGLIRType for the result
 * (caller owns it), on failure returns NULL and sets *known (1 = the name is
 * a builtin but no signature matched, 0 = unknown name). */
static MGLIRType *builtin_call_type(const char *name,
                                    const MGLIRType *const *arg_types,
                                    uint32_t argc, int *known)
{
    *known = 0;
    for (size_t i = 0; i < sizeof(kBuiltins) / sizeof(kBuiltins[0]); i++) {
        const BiFn *f = &kBuiltins[i];
        if (strcmp(f->name, name) != 0) {
            continue;
        }
        *known = 1;
        if (f->argc != argc) {
            continue;
        }
        uint32_t gen_dim = 0;
        bif_geni_unsigned = 0;
        int ok = 1;
        for (uint32_t j = 0; j < argc; j++) {
            uint32_t d = 0;
            if (!bif_arg_matches(arg_types[j], f->args[j], &d)) {
                ok = 0;
                break;
            }
            if (f->args[j] == BI_ARG_GENF || f->args[j] == BI_ARG_GENI) {
                if (gen_dim == 0) {
                    gen_dim = d;
                } else if (gen_dim != d) {
                    ok = 0;
                    break;
                }
            }
        }
        if (!ok) {
            continue;
        }
        switch (f->ret) {
        case BI_RET_GENF:
            return gen_dim > 1 ? mglIRTypeVector(MGLIR_SCALAR_FLOAT, gen_dim)
                               : mglIRTypeScalar(MGLIR_SCALAR_FLOAT);
        case BI_RET_GENI:
            return gen_dim > 1
                ? mglIRTypeVector(bif_geni_unsigned ? MGLIR_SCALAR_UINT
                                                    : MGLIR_SCALAR_INT,
                                  gen_dim)
                : mglIRTypeScalar(bif_geni_unsigned ? MGLIR_SCALAR_UINT
                                                    : MGLIR_SCALAR_INT);
        case BI_RET_FLOAT:
            return mglIRTypeScalar(MGLIR_SCALAR_FLOAT);
        case BI_RET_UINT:
            return mglIRTypeScalar(MGLIR_SCALAR_UINT);
        case BI_RET_VEC2:
            return mglIRTypeVector(MGLIR_SCALAR_FLOAT, 2);
        case BI_RET_VEC3:
            return mglIRTypeVector(MGLIR_SCALAR_FLOAT, 3);
        case BI_RET_VEC4:
            return mglIRTypeVector(MGLIR_SCALAR_FLOAT, 4);
        case BI_RET_MAT2:
            return mglIRTypeMatrix(MGLIR_SCALAR_FLOAT, 2, 2);
        case BI_RET_MAT3:
            return mglIRTypeMatrix(MGLIR_SCALAR_FLOAT, 3, 3);
        case BI_RET_MAT4:
            return mglIRTypeMatrix(MGLIR_SCALAR_FLOAT, 4, 4);
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Type constructors                                                   */
/* ------------------------------------------------------------------ */

/* Component count contributed by an argument to a vector constructor;
 * returns -1 for arguments that cannot appear in a vector constructor. */
static int constructor_components(const MGLIRType *at)
{
    if (!at) {
        return -1;
    }
    if (at->kind == MGLIR_TYPE_SCALAR) {
        return 1;
    }
    if (at->kind == MGLIR_TYPE_VECTOR) {
        return (int)at->cols;
    }
    return -1;
}

/* Exact argument checking for T(...) type constructors / conversions
 * (GLSL 4.60 §5.4.1/§5.4.2).  `t` is the target type, owned by caller. */

/* Can a value of type `from` be converted component-wise to scalar `to_sc`
 * for constructor purposes?  Constructors perform explicit conversions:
 * any non-void scalar base converts to any other (GLSL §4.1.10), and
 * vector arguments convert component-wise. */
static int constructor_scalar_convert(const MGLIRType *from, MGLIRScalar to_sc)
{
    if (!from || (from->kind != MGLIR_TYPE_SCALAR &&
                  from->kind != MGLIR_TYPE_VECTOR)) {
        return 0;
    }
    return from->scalar != MGLIR_SCALAR_VOID && to_sc != MGLIR_SCALAR_VOID;
}

static int check_constructor(Sema *s, uint32_t line, const char *tname,
                             MGLIRType *t, const MGLIRType *const *ats,
                             uint32_t argc)
{
    int ok = 1;
    if (t->kind == MGLIR_TYPE_SCALAR) {
        if (argc != 1 || !ats[0] || ats[0]->kind != MGLIR_TYPE_SCALAR) {
            ok = 0;
        } else if (!constructor_scalar_convert(ats[0], t->scalar)) {
            ok = 0;
        }
        if (!ok) {
            sema_error(s, line, "constructor '%s' takes one scalar, got %u",
                       tname, argc);
        }
        return ok;
    }
    if (t->kind == MGLIR_TYPE_VECTOR) {
        uint32_t n = t->cols;
        if (argc == 1 && ats[0]) {
            int c = constructor_components(ats[0]);
            if (ats[0]->kind == MGLIR_TYPE_SCALAR && c == 1) {
                /* broadcast: vec2(1.0) */
                return constructor_scalar_convert(ats[0], t->scalar);
            }
            if (c == (int)n) {
                /* single vector of the right size */
                return constructor_scalar_convert(ats[0], t->scalar);
            }
            sema_error(s, line,
                       "constructor 'vec%u' cannot take a single %s argument",
                       n, ir_type_str(ats[0], (char[64]){0}, 64));
            return 0;
        }
        uint32_t total = 0;
        for (uint32_t i = 0; i < argc; i++) {
            int c = ats[i] ? constructor_components(ats[i]) : -1;
            if (c < 0) {
                sema_error(s, line,
                           "cannot construct 'vec%u' from a %s argument",
                           n, ats[i] ? ir_type_str(ats[i], (char[64]){0}, 64) : "?");
                return 0;
            }
            total += (uint32_t)c;
        }
        if (total != n) {
            sema_error(s, line,
                       "constructor 'vec%u' from %u component(s), expected %u",
                       n, total, n);
            return 0;
        }
        for (uint32_t i = 0; i < argc; i++) {
            if (!constructor_scalar_convert(ats[i], t->scalar)) {
                char sa[64], sb[64];
                sema_error(s, line,
                           "constructor 'vec%u' argument %u: %s not "
                           "convertible to %s",
                           n, i + 1, ir_type_str(ats[i], sa, sizeof(sa)),
                           ir_type_str(t, sb, sizeof(sb)));
                return 0;
            }
        }
        return 1;
    }
    if (t->kind == MGLIR_TYPE_MATRIX) {
        uint32_t c = t->cols, r = t->rows;
        if (argc == 1 && ats[0]) {
            if (ats[0]->kind == MGLIR_TYPE_SCALAR) {
                return constructor_scalar_convert(ats[0], t->scalar); /* diagonal */
            }
            if (ats[0]->kind == MGLIR_TYPE_MATRIX && ats[0]->cols == c &&
                ats[0]->rows == r) {
                return 1;
            }
            sema_error(s, line, "constructor 'mat%ux%u' cannot take a single "
                       "%s argument", c, r,
                       ir_type_str(ats[0], (char[64]){0}, 64));
            return 0;
        }
        if (argc == c * r) {
            /* Scalar list: column-major fill (GLSL 4.60 5.4.2). */
            for (uint32_t i = 0; i < argc; i++) {
                if (!ats[i] || ats[i]->kind != MGLIR_TYPE_SCALAR ||
                    !constructor_scalar_convert(ats[i], t->scalar)) {
                    sema_error(s, line, "constructor 'mat%ux%u' argument %u "
                               "must be a scalar convertible to %s", c, r,
                               i + 1, ir_type_str(t, (char[64]){0}, 64));
                    return 0;
                }
            }
            return 1;
        }
        if (argc != c) {
            sema_error(s, line, "constructor 'mat%ux%u' expects %u column "
                       "vector(s), got %u", c, r, c, argc);
            return 0;
        }
        for (uint32_t i = 0; i < argc; i++) {
            if (!ats[i] || ats[i]->kind != MGLIR_TYPE_VECTOR ||
                ats[i]->cols != r) {
                sema_error(s, line,
                           "constructor 'mat%ux%u' column %u must be a vec%u",
                           c, r, i + 1, r);
                return 0;
            }
        }
        return 1;
    }
    return 1;
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
        MGLIRType *t = scratch_type(s, mglIRTypeScalar(sc));
        /* Literal type is cached on the node for the backend. */
        return t;
    }
    case MGL_EXPR_VAR_REF: {
        Sym *sym = symtab_lookup(tab, e->u.var_ref.name);
        if (!sym) {
            if (strcmp(e->u.var_ref.name, "gl_Position") == 0) {
                /* Vertex-stage built-in output; the AIR backend maps it to
                 * the air.position output entry. */
                return scratch_type(s,
                                    mglIRTypeVector(MGLIR_SCALAR_FLOAT, 4));
            }
            if (strcmp(e->u.var_ref.name, "gl_GlobalInvocationID") == 0) {
                /* Compute built-in; the AIR backend maps it to the
                 * thread_position_in_grid kernel argument. */
                return scratch_type(s,
                                    mglIRTypeVector(MGLIR_SCALAR_UINT, 3));
            }
            if (strcmp(e->u.var_ref.name, "gl_VertexID") == 0) {
                /* Vertex built-in; the AIR backend maps it to a
                 * vertex_id argument (capture variants). */
                return scratch_type(s, mglIRTypeScalar(MGLIR_SCALAR_INT));
            }
            sema_error(s, e->line, "undeclared identifier '%s'",
                       e->u.var_ref.name);
            return NULL;
        }
        if (sym->kind == SYM_FUNCTION) {
            sema_error(s, e->line, "'%s' is a function, not a variable",
                       e->u.var_ref.name);
            return NULL;
        }
        return sym->type;
    }
    case MGL_EXPR_MEMBER: {
        MGLIRType *obj = check_expr(s, tab, e->u.member.object);
        if (!obj) {
            return NULL;
        }
        if (obj->kind == MGLIR_TYPE_VECTOR && obj->rows <= 4) {
            /* Swizzle: xyzw/rgba component selection.  All components must
             * come from the same namespace and stay in range (GLSL 4.60
             * 5.4.2). */
            const char *f = e->u.member.field;
            size_t n = 0;
            const char *set = NULL;
            for (const char *p = f; *p; p++) {
                const char *which = strchr("xyzw", *p)
                    ? "xyzw" : strchr("rgba", *p) ? "rgba" : NULL;
                if (!which) {
                    sema_error(s, e->line, "no member named '%s' in struct",
                               e->u.member.field);
                    return NULL;
                }
                if (!set) set = which;
                else if (set != which) {
                    sema_error(s, e->line, "invalid swizzle '%s'",
                               e->u.member.field);
                    return NULL;
                }
                if ((size_t)(strchr(set, *p) - set) >= obj->cols) {
                    sema_error(s, e->line, "invalid swizzle '%s'",
                               e->u.member.field);
                    return NULL;
                }
                n++;
                if (n > 4) {
                    sema_error(s, e->line, "invalid swizzle '%s'",
                               e->u.member.field);
                    return NULL;
                }
            }
            if (n == 0) {
                sema_error(s, e->line, "invalid swizzle '%s'",
                           e->u.member.field);
                return NULL;
            }
            if (n == 1) {
                /* GLSL 4.60 5.4.3: single-component swizzle is a scalar. */
                return scratch_type(s, mglIRTypeScalar(obj->scalar));
            }
            return scratch_type(s, mglIRTypeVector(obj->scalar, (uint32_t)n));
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
            return scratch_type(s, mglIRTypeVector(obj->scalar, obj->rows));
        }
        if (obj->kind == MGLIR_TYPE_VECTOR) {
            /* vector[i] yields a scalar component */
            return scratch_type(s, mglIRTypeScalar(obj->scalar));
        }
        sema_error(s, e->line, "indexing a non-array type");
        return NULL;
    }
    case MGL_EXPR_CALL: {
        /* Look up the function; single-definition match at skeleton stage. */
        Sym *sym = symtab_lookup(tab, e->u.call.name);
        if (!sym || sym->kind != SYM_FUNCTION) {
            /* builtin constructors / type conversions: T(...) yields type T.
             * Recognise every builtin type name plus user struct types. */
            MGLTypeSpec fake;
            memset(&fake, 0, sizeof(fake));
            int known = builtin_type_spec(e->u.call.name, &fake);
            int is_struct_ctor = 0;
            if (known != 0) {
                Sym *st = symtab_lookup(tab, e->u.call.name);
                if (st && st->kind == SYM_STRUCT) {
                    fake.base = MGL_AST_TYPE_STRUCT;
                    fake.name = (char *)e->u.call.name;
                    known = 0;
                    is_struct_ctor = 1;
                }
            }
            if (known == 0) {
                MGLIRType *t = scratch_type(s, resolve_type_spec(s, tab, &fake));
                if (t) {
                    MGLIRType **ats = (MGLIRType **)calloc(
                        e->u.call.arg_count, sizeof(MGLIRType *));
                    for (uint32_t i = 0; i < e->u.call.arg_count; i++) {
                        ats[i] = check_expr(s, tab, e->u.call.args[i]);
                    }
                    if (!is_struct_ctor) {
                        check_constructor(s, e->line, e->u.call.name, t,
                                          (const MGLIRType *const *)ats,
                                          e->u.call.arg_count);
                    }
                    free(ats);
                }
                return t;
            }
            /* builtin functions (first-wave table) */
            MGLIRType **atb = (MGLIRType **)calloc(
                e->u.call.arg_count, sizeof(MGLIRType *));
            for (uint32_t i = 0; i < e->u.call.arg_count; i++) {
                atb[i] = check_expr(s, tab, e->u.call.args[i]);
            }
            int bknown = 0;
            MGLIRType *bt = builtin_call_type(e->u.call.name,
                                              (const MGLIRType *const *)atb,
                                              e->u.call.arg_count, &bknown);
            free(atb);
            if (bknown) {
                if (bt) {
                    return scratch_type(s, bt);
                }
                sema_error(s, e->line,
                           "no matching overload of builtin '%s' for the "
                           "given argument types", e->u.call.name);
                return NULL;
            }
            sema_error(s, e->line, "call to undeclared function '%s'",
                       e->u.call.name);
            return NULL;
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
            return scratch_type(s, mglIRTypeScalar(MGLIR_SCALAR_BOOL));
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
            return scratch_type(s, mglIRTypeScalar(MGLIR_SCALAR_BOOL));
        case MGL_OP_LT: case MGL_OP_LE: case MGL_OP_GT: case MGL_OP_GE:
            if (!is_numeric(l) || !is_numeric(r)) {
                sema_error(s, e->line, "relational '%s' requires numeric operands",
                           op_name(e->u.binary.op));
                return NULL;
            }
            return scratch_type(s, mglIRTypeScalar(MGLIR_SCALAR_BOOL));
        case MGL_OP_LAND: case MGL_OP_LOR:
            if (l->scalar != MGLIR_SCALAR_BOOL || r->scalar != MGLIR_SCALAR_BOOL) {
                sema_error(s, e->line, "logical '%s' requires bool operands",
                           op_name(e->u.binary.op));
                return NULL;
            }
            return scratch_type(s, mglIRTypeScalar(MGLIR_SCALAR_BOOL));
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
                /* matrix +/- scalar, or matrix +/- matrix: result is the
                 * matrix operand */
                return l->kind == MGLIR_TYPE_MATRIX ? l : r;
            }
            if (!is_numeric(l) || !is_numeric(r)) {
                sema_error(s, e->line, "arithmetic '%s' requires numeric operands",
                           op_name(e->u.binary.op));
                return NULL;
            }
            return result_numeric(s, l, r);
        case MGL_OP_MUL:
            if (!is_numeric(l) || !is_numeric(r)) {
                sema_error(s, e->line, "arithmetic '*' requires numeric operands",
                           op_name(e->u.binary.op));
                return NULL;
            }
            if (l->kind == MGLIR_TYPE_MATRIX || r->kind == MGLIR_TYPE_MATRIX) {
                return matrix_mul_result(s, e, l, r);
            }
            return result_numeric(s, l, r);
        case MGL_OP_DIV: case MGL_OP_MOD:
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
            return result_numeric(s, l, r);
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
            sym_free(sym);
            return;
        }
    }
    if (symtab_lookup_local(tab, d->name) != NULL) {
        sema_error(s, d->line, "redeclaration of '%s'", d->name);
        sym_free(sym);
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
    case MGL_STMT_IF: {
        MGLIRType *ct = check_expr(s, tab, st->u.ifs.cond);
        if (ct && (ct->kind != MGLIR_TYPE_SCALAR ||
                   ct->scalar != MGLIR_SCALAR_BOOL)) {
            sema_error(s, st->line, "if condition must be a scalar bool");
        }
        analyze_stmt(s, tab, st->u.ifs.then);
        if (st->u.ifs.else_) {
            analyze_stmt(s, tab, st->u.ifs.else_);
        }
        break;
    }
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
    case MGL_STMT_WHILE: {
        MGLIRType *ct = check_expr(s, tab, st->u.whilex.cond);
        if (ct && (ct->kind != MGLIR_TYPE_SCALAR ||
                   ct->scalar != MGLIR_SCALAR_BOOL)) {
            sema_error(s, st->line, "while condition must be a scalar bool");
        }
        analyze_stmt(s, tab, st->u.whilex.body);
        break;
    }
    case MGL_STMT_DO_WHILE: {
        analyze_stmt(s, tab, st->u.whilex.body);
        MGLIRType *ct = check_expr(s, tab, st->u.whilex.cond);
        if (ct && (ct->kind != MGLIR_TYPE_SCALAR ||
                   ct->scalar != MGLIR_SCALAR_BOOL)) {
            sema_error(s, st->line,
                       "do-while condition must be a scalar bool");
        }
        break;
    }
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
    scratch_destroy(&s);
    return (int)s.error_count;
}

/* ------------------------------------------------------------------ */
/* Cross-stage interface matching (GLSL 4.60 §4.3.9.5)                 */
/* ------------------------------------------------------------------ */

static int sym_is_interface_block(const MGLIRSymbol *is)
{
    return is->type && is->type->kind == MGLIR_TYPE_STRUCT &&
           is->layout != MGL_AST_LAYOUT_DEFAULT;
}

/* Decode a builtin type name into a MGLTypeSpec, mirroring the parser's
 * parse_type_spec keyword decoding.  Returns 0 on success, -1 if `name`
 * is not a builtin type name (callers then fall back to struct lookup). */
static int builtin_type_spec(const char *name, MGLTypeSpec *ts)
{
    memset(ts, 0, sizeof(*ts));
    ts->base = MGL_AST_TYPE_FLOAT;
    if (strcmp(name, "void") == 0) {
        ts->base = MGL_AST_TYPE_VOID;
        return 0;
    } else if (strcmp(name, "bool") == 0) {
        ts->base = MGL_AST_TYPE_BOOL;
        return 0;
    } else if (strcmp(name, "int") == 0) {
        ts->base = MGL_AST_TYPE_INT;
        return 0;
    } else if (strcmp(name, "uint") == 0) {
        ts->base = MGL_AST_TYPE_UINT;
        return 0;
    } else if (strcmp(name, "float") == 0) {
        ts->base = MGL_AST_TYPE_FLOAT;
        return 0;
    } else if (strcmp(name, "double") == 0) {
        ts->base = MGL_AST_TYPE_DOUBLE;
        return 0;
    } else if (strcmp(name, "atomic_uint") == 0) {
        ts->base = MGL_AST_TYPE_ATOMIC_UINT;
        return 0;
    }
    size_t n = strlen(name);
    if (n == 5 && (name[0] == 'i' || name[0] == 'u' || name[0] == 'b' ||
                   name[0] == 'd') &&
        name[1] == 'v' && name[2] == 'e' && name[3] == 'c' &&
        name[4] >= '1' && name[4] <= '4') {
        if (name[0] == 'i') {
            ts->base = MGL_AST_TYPE_INT;
        } else if (name[0] == 'u') {
            ts->base = MGL_AST_TYPE_UINT;
        } else if (name[0] == 'b') {
            ts->base = MGL_AST_TYPE_BOOL;
        } else {
            ts->base = MGL_AST_TYPE_DOUBLE;
        }
        ts->vec_size = name[4] - '0';
        return 0;
    }
    if (n == 4 && strncmp(name, "vec", 3) == 0 &&
        name[3] >= '1' && name[3] <= '4') {
        ts->vec_size = name[3] - '0';
        return 0;
    }
    if (n == 4 && strncmp(name, "mat", 3) == 0 &&
        name[3] >= '2' && name[3] <= '4') {
        ts->mat_cols = ts->mat_rows = name[3] - '0';
        return 0;
    }
    if (n == 6 && strncmp(name, "mat", 3) == 0 &&
        name[3] >= '2' && name[3] <= '4' && name[4] == 'x' &&
        name[5] >= '2' && name[5] <= '4') {
        ts->mat_cols = name[3] - '0';
        ts->mat_rows = name[5] - '0';
        return 0;
    }
    if (n == 5 && strncmp(name, "dmat", 4) == 0 &&
        name[4] >= '2' && name[4] <= '4') {
        ts->base = MGL_AST_TYPE_DOUBLE;
        ts->mat_cols = ts->mat_rows = name[4] - '0';
        return 0;
    }
    if (n == 7 && strncmp(name, "dmat", 4) == 0 &&
        name[4] >= '2' && name[4] <= '4' && name[5] == 'x' &&
        name[6] >= '2' && name[6] <= '4') {
        ts->base = MGL_AST_TYPE_DOUBLE;
        ts->mat_cols = name[4] - '0';
        ts->mat_rows = name[6] - '0';
        return 0;
    }
    return -1;
}

/* Link-time check between two compiled stages (e.g. vertex and fragment).
 * For every ordinary in/out variable declared on both sides the type must
 * match exactly; interface blocks match by block name (instance name may
 * differ) and require identical member lists and layout.  Variables present
 * on only one side are legal.  Returns the number of hard errors. */
int mglGLSLInterfaceCheck(const MGLIRModule *a, const MGLIRModule *b,
                          MGLSemaError **errors, uint32_t *error_count)
{
    Sema s;
    memset(&s, 0, sizeof(s));

    for (uint32_t i = 0; i < a->symbol_count; i++) {
        MGLIRSymbol *sa = a->symbols[i];
        if (!sa || sa->is_function || !sa->name || !sa->type) {
            continue;
        }
        for (uint32_t j = 0; j < b->symbol_count; j++) {
            MGLIRSymbol *sb = b->symbols[j];
            if (!sb || sb->is_function || !sb->type) {
                continue;
            }
            if (sym_is_interface_block(sa)) {
                if (!sym_is_interface_block(sb)) {
                    continue;
                }
                /* Block: match by block type name; instance names may
                 * differ. */
                if (!sa->type->name || !sb->type->name ||
                    strcmp(sa->type->name, sb->type->name) != 0) {
                    continue;
                }
                if (sa->layout != sb->layout ||
                    !ir_type_interface_equal(sa->type, sb->type)) {
                    sema_error(&s, 0,
                               "interface block '%s' does not match across stages",
                               sa->type->name);
                }
                continue;
            }
            if (sym_is_interface_block(sb)) {
                continue;
            }
            /* Ordinary in/out variables. */
            if (strcmp(sa->name, sb->name) != 0) {
                continue;
            }
            uint32_t both_ways =
                ((sa->qualifiers & MGL_AST_Q_OUT) &&
                 (sb->qualifiers & MGL_AST_Q_IN)) ||
                ((sa->qualifiers & MGL_AST_Q_IN) &&
                 (sb->qualifiers & MGL_AST_Q_OUT));
            if (!both_ways) {
                continue;
            }
            if (!ir_type_interface_equal(sa->type, sb->type)) {
                char ta[64], tb[64];
                sema_error(&s, 0,
                           "interface variable '%s' type mismatch across stages "
                           "(%s vs %s)",
                           sa->name, ir_type_str(sa->type, ta, sizeof(ta)),
                           ir_type_str(sb->type, tb, sizeof(tb)));
            }
        }
    }

    if (errors) {
        *errors = s.errors;
    }
    if (error_count) {
        *error_count = s.error_count;
    }
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
