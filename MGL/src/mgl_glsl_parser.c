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
 * mgl_glsl_parser.c
 * MGL - self-written GLSL frontend recursive-descent parser (skeleton).
 *
 * The lexer emits single-character PUNCT tokens; multi-character
 * operators (==, ++, += ...) are therefore sequences of consecutive
 * single-char tokens.  All operator matching below works on that basis.
 */

#include "mgl_glsl_parser.h"
#include "mgl_glsl_lexer.h"
#include "mgl_glsl_cpp.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MGL_MAX_TOKENS 131072
#define MGL_MAX_DIMS 8


/* ----------------------------------------------------------------- */
/* Token stream                                                       */
/* ----------------------------------------------------------------- */

typedef struct MGLTokenStream {
    MGLGLSLToken *tok;
    int count;
    int cap;
    char *src;              /* owned copy, NUL-terminated */
    size_t src_len;
} MGLTokenStream;

static int tokenize(MGLTokenStream *ts, const char *src, size_t len)
{
    ts->cap = 4096;
    ts->tok = (MGLGLSLToken *)malloc((size_t)ts->cap * sizeof(MGLGLSLToken));
    ts->src = (char *)malloc(len + 1);
    if (!ts->tok || !ts->src) {
        return -1;
    }
    memcpy(ts->src, src, len);
    ts->src[len] = '\0';
    ts->src_len = strlen(ts->src);
    ts->count = 0;

    MGLGLSLexer lx;
    mglGLSLexerInit(&lx, ts->src, ts->src_len);
    for (;;) {
        MGLGLSLToken t;
        if (mglGLSLexerNext(&lx, &t) != 0) {
            break;
        }
        if (ts->count >= ts->cap) {
            int nc = ts->cap * 2;
            MGLGLSLToken *nt = (MGLGLSLToken *)realloc(
                ts->tok, (size_t)nc * sizeof(MGLGLSLToken));
            if (!nt) {
                return -1;
            }
            ts->tok = nt;
            ts->cap = nc;
        }
        ts->tok[ts->count++] = t;
        if (t.kind == MGLGLSL_TOK_END) {
            break;
        }
    }
    return 0;
}

static void token_stream_free(MGLTokenStream *ts)
{
    free(ts->tok);
    free(ts->src);
    ts->tok = NULL;
    ts->src = NULL;
    ts->count = 0;
    ts->cap = 0;
}

/* ----------------------------------------------------------------- */
/* Parser                                                             */
/* ----------------------------------------------------------------- */

typedef struct MGLParser {
    MGLTokenStream *ts;
    int pos;
    MGLTranslationUnit *tu;
    char errbuf[256];
    uint32_t decl_precision;   /* pending precision qualifier */
    /* Const-int / array-length tables for folding array extents while
     * parsing (`float[a]`, `float[a+b]`, `float[arr.length()]`). */
    char const_names[64][64];
    int64_t const_vals[64];
    uint32_t const_count;
    char array_names[64][64];
    uint32_t array_lens[64];
    uint32_t array_count;
} MGLParser;

static unsigned int tk_line(MGLParser *p);
static const MGLGLSLToken *tk(MGLParser *p, int offset);

/* Image formats are layout qualifiers, not declaration qualifiers.  The
 * AIR type carries the image's element scalar kind; Metal obtains the actual
 * pixel format from the bound texture, so no extra AST field is needed. */
static int is_image_format_layout(const char *s, size_t n)
{
    static const char *const formats[] = {
        "r8", "r16", "r32f", "rg8", "rg16", "rg32f",
        "rgba8", "rgba16", "rgba32f", "rgba8_snorm",
        "rgba16_snorm", "rg8_snorm", "rg16_snorm", "r8_snorm",
        "r16_snorm", "r11f_g11f_b10f", "rgb10_a2",
        "r8i", "r16i", "r32i", "rg8i", "rg16i", "rg32i",
        "rgba8i", "rgba16i", "rgba32i", "r8ui", "r16ui",
        "r32ui", "rg8ui", "rg16ui", "rg32ui", "rgba8ui",
        "rgba16ui", "rgba32ui", "rgb10_a2ui",
    };
    for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); i++) {
        if (strlen(formats[i]) == n && memcmp(s, formats[i], n) == 0) {
            return 1;
        }
    }
    return 0;
}

static void parse_error(MGLParser *p, const char *fmt, ...)
{
    if (p->tu->error) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(p->errbuf, sizeof(p->errbuf), fmt, ap);
    va_end(ap);
    p->tu->error = strdup(p->errbuf);
    p->tu->error_line = tk_line(p);
}

static const MGLGLSLToken *tk(MGLParser *p, int offset)
{
    int i = p->pos + offset;
    if (i < 0 || i >= p->ts->count) {
        i = p->ts->count - 1; /* END token */
    }
    return &p->ts->tok[i];
}

static unsigned int tk_line(MGLParser *p)
{
    return tk(p, 0)->line;
}

static void advance(MGLParser *p)
{
    if (p->pos < p->ts->count - 1) {
        p->pos++;
    }
}
static int ops_at(MGLParser *p, const char *s)
{
    size_t len = strlen(s);
    if (len < 1 || len > 3) {
        return 0;
    }
    size_t i;
    for (i = 0; i < len; i++) {
        const MGLGLSLToken *t = tk(p, (int)i);
        if (t->kind != MGLGLSL_TOK_PUNCT) {
            return 0;
        }
        if (t->end - t->start != 1 ||
            p->ts->src[t->start] != s[i]) {
            return 0;
        }
    }
    return 1;
}

static int at_punct(MGLParser *p, const char *s)
{
    return ops_at(p, s);
}

static int at_any_ident(MGLParser *p)
{
    return tk(p, 0)->kind == MGLGLSL_TOK_IDENT;
}

static int at_peek_punct(MGLParser *p, int offset, const char *s)
{
    const MGLGLSLToken *t = tk(p, offset);
    if (t->kind != MGLGLSL_TOK_PUNCT || t->end - t->start != 1) {
        return 0;
    }
    return p->ts->src[t->start] == s[0];
}

static int at_ident(MGLParser *p, const char *s)
{
    size_t n = strlen(s);
    const MGLGLSLToken *t = tk(p, 0);
    return t->kind == MGLGLSL_TOK_IDENT &&
           n == (size_t)(t->end - t->start) &&
           memcmp(p->ts->src + t->start, s, n) == 0;
}

static int at_num(MGLParser *p)
{
    MGLGLSLTokenKind k = (MGLGLSLTokenKind)tk(p, 0)->kind;
    return k == MGLGLSL_TOK_INT || k == MGLGLSL_TOK_UINT ||
           k == MGLGLSL_TOK_FLOAT;
}

static int eat_punct(MGLParser *p, const char *s)
{
    if (ops_at(p, s)) {
        int i;
        for (i = 0; i < (int)strlen(s); i++) {
            advance(p);
        }
        return 1;
    }
    return 0;
}

static int eat_ident(MGLParser *p, const char *s)
{
    if (at_ident(p, s)) {
        advance(p);
        return 1;
    }
    return 0;
}

static uint32_t eat_precision_qualifier(MGLParser *p)
{
    if (eat_ident(p, "lowp")) {
        return MGL_AST_PRECISION_LOWP;
    }
    if (eat_ident(p, "mediump")) {
        return MGL_AST_PRECISION_MEDIUMP;
    }
    if (eat_ident(p, "highp")) {
        return MGL_AST_PRECISION_HIGHP;
    }
    return MGL_AST_PRECISION_NONE;
}

#define expect_punct(p, s) (expect_punct_impl(p, s, __LINE__))
static int expect_punct_impl(MGLParser *p, const char *s, int caller)
{
    (void)caller;
    if (eat_punct(p, s)) {
        return 1;
    }
    parse_error(p, "expected '%s' at line %u", s, tk_line(p));
    return 0;
}

static char *dup_token_text(MGLParser *p, const MGLGLSLToken *t)
{
    size_t n = t->end - t->start;
    char *s = (char *)malloc(n + 1);
    if (s) {
        memcpy(s, p->ts->src + t->start, n);
        s[n] = '\0';
    }
    return s;
}

static char *dup_current(MGLParser *p)
{
    return dup_token_text(p, tk(p, 0));
}

static double cur_double(MGLParser *p)
{
    MGLGLSLexer lx;
    mglGLSLexerInit(&lx, p->ts->src, p->ts->src_len);
    double v = 0;
    mglGLSLexerLiteral(&lx, tk(p, 0), &v);
    return v;
}

/* ----------------------------------------------------------------- */
/* Forward declarations                                               */
/* ----------------------------------------------------------------- */

static MGLExpr *parse_expression(MGLParser *p);
static MGLExpr *parse_assignment(MGLParser *p);
static MGLStmt *parse_statement(MGLParser *p);
static MGLDecl *parse_declaration(MGLParser *p);
static int at_decl_start(MGLParser *p);
static void free_expr(MGLExpr *e);
static void free_stmt(MGLStmt *s);
static void free_decl(MGLDecl *d);

/* ----------------------------------------------------------------- */
/* Types                                                              */
/* ----------------------------------------------------------------- */

static MGLTypeSpec *parse_type_spec(MGLParser *p)
{
    if (!at_any_ident(p)) {
        parse_error(p, "expected type at line %u", tk_line(p));
        return NULL;
    }
    const MGLGLSLToken *t = tk(p, 0);
    size_t n = (size_t)(t->end - t->start);
    const char *s = p->ts->src + t->start;

    MGLTypeSpec *ts = (MGLTypeSpec *)calloc(1, sizeof(*ts));
    if (!ts) {
        return NULL;
    }
    ts->base = MGL_AST_TYPE_FLOAT;

#define TY(k) ((n == strlen(k)) && memcmp(s, k, n) == 0)
    if (TY("void")) {
        ts->base = MGL_AST_TYPE_VOID;
    } else if (TY("bool")) {
        ts->base = MGL_AST_TYPE_BOOL;
    } else if (TY("int")) {
        ts->base = MGL_AST_TYPE_INT;
    } else if (TY("uint")) {
        ts->base = MGL_AST_TYPE_UINT;
    } else if (TY("float")) {
        ts->base = MGL_AST_TYPE_FLOAT;
    } else if (TY("double")) {
        ts->base = MGL_AST_TYPE_DOUBLE;
    } else if (TY("atomic_uint")) {
        ts->base = MGL_AST_TYPE_ATOMIC_UINT;
    } else if (n >= 7 && memcmp(s, "sampler", 7) == 0) {
        ts->base = MGL_AST_TYPE_SAMPLER;
        ts->name = dup_token_text(p, t);
    } else if (n >= 8 && memcmp(s, "isampler", 8) == 0) {
        ts->base = MGL_AST_TYPE_SAMPLER;
        ts->name = dup_token_text(p, t);
    } else if (n >= 8 && memcmp(s, "usampler", 8) == 0) {
        ts->base = MGL_AST_TYPE_SAMPLER;
        ts->name = dup_token_text(p, t);
    } else if (n >= 6 && memcmp(s, "iimage", 6) == 0) {
        ts->base = MGL_AST_TYPE_IMAGE;
        ts->name = dup_token_text(p, t);
    } else if (n >= 6 && memcmp(s, "uimage", 6) == 0) {
        ts->base = MGL_AST_TYPE_IMAGE;
        ts->name = dup_token_text(p, t);
    } else if (n >= 5 && memcmp(s, "image", 5) == 0) {
        ts->base = MGL_AST_TYPE_IMAGE;
        ts->name = dup_token_text(p, t);
    } else if (n == 4 && s[0] == 'v' && s[1] == 'e' && s[2] == 'c' &&
               s[3] >= '1' && s[3] <= '4') {
        ts->vec_size = s[3] - '0';
    } else if (n == 5 && (s[0] == 'i' || s[0] == 'u' || s[0] == 'b' ||
                          s[0] == 'd') &&
               s[1] == 'v' && s[2] == 'e' && s[3] == 'c' &&
               s[4] >= '1' && s[4] <= '4') {
        if (s[0] == 'i') {
            ts->base = MGL_AST_TYPE_INT;
        } else if (s[0] == 'u') {
            ts->base = MGL_AST_TYPE_UINT;
        } else if (s[0] == 'b') {
            ts->base = MGL_AST_TYPE_BOOL;
        } else {
            ts->base = MGL_AST_TYPE_DOUBLE;
        }
        ts->vec_size = s[4] - '0';
    } else if (n == 4 && s[0] == 'm' && s[1] == 'a' && s[2] == 't' &&
               s[3] >= '2' && s[3] <= '4') {
        ts->mat_cols = ts->mat_rows = s[3] - '0';
    } else if (n == 6 && s[0] == 'm' && s[1] == 'a' && s[2] == 't' &&
               s[3] >= '2' && s[3] <= '4' && s[4] == 'x' &&
               s[5] >= '2' && s[5] <= '4') {
        ts->mat_cols = s[3] - '0';
        ts->mat_rows = s[5] - '0';
    } else if (n == 5 && s[0] == 'd' && s[1] == 'm' && s[2] == 'a' &&
               s[3] == 't' && s[4] >= '2' && s[4] <= '4') {
        ts->base = MGL_AST_TYPE_DOUBLE;
        ts->mat_cols = ts->mat_rows = s[4] - '0';
    } else if (n == 7 && s[0] == 'd' && s[1] == 'm' && s[2] == 'a' &&
               s[3] == 't' && s[4] >= '2' && s[4] <= '4' && s[5] == 'x' &&
               s[6] >= '2' && s[6] <= '4') {
        ts->base = MGL_AST_TYPE_DOUBLE;
        ts->mat_cols = s[4] - '0';
        ts->mat_rows = s[6] - '0';
    } else {
        ts->base = MGL_AST_TYPE_STRUCT;
        ts->name = dup_token_text(p, t);
    }
    advance(p);
    return ts;
}

/* ------------------------------------------------------------------ */
/* Expressions                                                         */
/* ------------------------------------------------------------------ */

static int eval_const_int(MGLParser *p, const MGLExpr *e, int64_t *value);
static MGLExpr *parse_expression(MGLParser *p);

static void record_const_int(MGLParser *p, const char *name, int64_t value)
{
    if (!p || !name || p->const_count >= 64) return;
    size_t n = strlen(name);
    if (n == 0 || n >= 64) return;
    for (uint32_t i = 0; i < p->const_count; i++) {
        if (strcmp(p->const_names[i], name) == 0) {
            p->const_vals[i] = value;
            return;
        }
    }
    memcpy(p->const_names[p->const_count], name, n + 1);
    p->const_vals[p->const_count++] = value;
}

static void record_array_len(MGLParser *p, const char *name, uint32_t len)
{
    if (!p || !name || len == 0 || p->array_count >= 64) return;
    size_t n = strlen(name);
    if (n == 0 || n >= 64) return;
    for (uint32_t i = 0; i < p->array_count; i++) {
        if (strcmp(p->array_names[i], name) == 0) {
            p->array_lens[i] = len;
            return;
        }
    }
    memcpy(p->array_names[p->array_count], name, n + 1);
    p->array_lens[p->array_count++] = len;
}

static int lookup_const_int(const MGLParser *p, const char *name, int64_t *value)
{
    if (!p || !name) return 0;
    for (uint32_t i = 0; i < p->const_count; i++) {
        if (strcmp(p->const_names[i], name) == 0) {
            *value = p->const_vals[i];
            return 1;
        }
    }
    return 0;
}

static int lookup_array_len(const MGLParser *p, const char *name, uint32_t *len)
{
    if (!p || !name) return 0;
    for (uint32_t i = 0; i < p->array_count; i++) {
        if (strcmp(p->array_names[i], name) == 0) {
            *len = p->array_lens[i];
            return 1;
        }
    }
    return 0;
}

static void record_decl_constants(MGLParser *p, const MGLDecl *d)
{
    for (; d; d = d->next_declarator) {
        if (!d->name) continue;
        if (d->array_count > 0 && d->array_dims) {
            /* 1-D size used by `.length()`; multi-dim rejected elsewhere. */
            uint32_t sz = d->array_dims[0];
            if (sz > 0) record_array_len(p, d->name, sz);
        }
        if (!(d->qualifiers & MGL_AST_Q_CONST) || !d->type || !d->init)
            continue;
        if (d->type->vec_size || d->type->mat_cols) continue;
        if (d->type->base != MGL_AST_TYPE_INT &&
            d->type->base != MGL_AST_TYPE_UINT &&
            d->type->base != MGL_AST_TYPE_BOOL)
            continue;
        int64_t v = 0;
        if (eval_const_int(p, d->init, &v))
            record_const_int(p, d->name, v);
    }
}

static MGLExpr *expr_alloc(MGLParser *p, uint32_t kind, uint32_t line)
{
    (void)p;
    MGLExpr *e = (MGLExpr *)calloc(1, sizeof(*e));
    if (e) {
        e->kind = kind;
        e->line = line;
    }
    return e;
}

static MGLExpr *parse_primary(MGLParser *p)
{
    uint32_t line = tk_line(p);
    const MGLGLSLToken *t = tk(p, 0);

    if (t->kind == MGLGLSL_TOK_INT || t->kind == MGLGLSL_TOK_UINT ||
        t->kind == MGLGLSL_TOK_FLOAT) {
        MGLExpr *e = expr_alloc(p, MGL_EXPR_LITERAL, line);
        if (e) {
        e->u.literal.base = (t->kind == MGLGLSL_TOK_FLOAT)
                                ? MGL_AST_TYPE_FLOAT
                                : (t->kind == MGLGLSL_TOK_UINT)
                                      ? MGL_AST_TYPE_UINT
                                      : MGL_AST_TYPE_INT;
            e->u.literal.value = cur_double(p);
        }
        advance(p);
        return e;
    }
    if (at_punct(p, "(")) {
        eat_punct(p, "(");
        MGLExpr *inner = parse_expression(p);
        expect_punct(p, ")");
        return inner;
    }
    if (at_any_ident(p)) {
        char *name = dup_current(p);
        advance(p);
        if (strcmp(name, "true") == 0 || strcmp(name, "false") == 0) {
            MGLExpr *e = expr_alloc(p, MGL_EXPR_LITERAL, line);
            if (e) {
                e->u.literal.base = MGL_AST_TYPE_BOOL;
                e->u.literal.value = (name[0] == 't') ? 1.0 : 0.0;
            }
            free(name);
            return e;
        }
        if (ops_at(p, "(") || ops_at(p, "[")) {
            /* Distinguish `T[N](...)` array ctor from `arr[i]` indexing:
             * only treat brackets as an array ctor when `(` follows `]`. */
            int is_arr_ctor = 0;
            uint32_t ctor_size = 0;
            if (ops_at(p, "[")) {
                int saved = p->pos;
                advance(p); /* [ */
                if (ops_at(p, "]")) {
                    ctor_size = 0;
                } else {
                    /* Constant extent only; restore on failure so
                     * postfix indexing can re-parse `arr[expr]`. */
                    MGLExpr *ext = parse_expression(p);
                    int64_t value = 0;
                    int valid = eval_const_int(p, ext, &value);
                    free_expr(ext);
                    if (!valid || value < 0 ||
                        (uint64_t)value > UINT32_MAX || !ops_at(p, "]")) {
                        p->pos = saved;
                        /* Fall through to VAR_REF; postfix handles `[`. */
                        MGLExpr *e = expr_alloc(p, MGL_EXPR_VAR_REF, line);
                        if (e) {
                            e->u.var_ref.name = name;
                        }
                        return e;
                    }
                    ctor_size = (uint32_t)value;
                }
                advance(p); /* ] */
                if (!ops_at(p, "(")) {
                    p->pos = saved;
                    MGLExpr *e = expr_alloc(p, MGL_EXPR_VAR_REF, line);
                    if (e) {
                        e->u.var_ref.name = name;
                    }
                    return e;
                }
                is_arr_ctor = 1;
            }
            MGLExpr *e = expr_alloc(p, MGL_EXPR_CALL, line);
            if (e) {
                e->u.call.name = name;
                if (is_arr_ctor) {
                    e->u.call.is_array_ctor = 1;
                    e->u.call.array_ctor_size = ctor_size;
                }
                eat_punct(p, "(");
                uint32_t argc = 0;
                if (!ops_at(p, ")")) {
                    for (;;) {
                        MGLExpr *arg = parse_assignment(p);
                        if (!arg) {
                            break;
                        }
                        e->u.call.args = (MGLExpr **)realloc(
                            e->u.call.args, (argc + 1) * sizeof(MGLExpr *));
                        e->u.call.args[argc++] = arg;
                        if (!eat_punct(p, ",")) {
                            break;
                        }
                        if (ops_at(p, ")")) {
                            break;
                        }
                    }
                }
                e->u.call.arg_count = argc;
                expect_punct(p, ")");
            }
            return e;
        }
        MGLExpr *e = expr_alloc(p, MGL_EXPR_VAR_REF, line);
        if (e) {
            e->u.var_ref.name = name;
        }
        return e;
    }
    parse_error(p, "unexpected token at line %u", line);
    return NULL;
}

static MGLExpr *parse_postfix(MGLParser *p)
{
    MGLExpr *e = parse_primary(p);
    if (!e) {
        return NULL;
    }
    for (;;) {
        if (ops_at(p, ".")) {
            advance(p);
            if (!at_any_ident(p)) {
                parse_error(p, "expected field name at line %u", tk_line(p));
                break;
            }
            char *field = dup_current(p);
            advance(p);
            if (field && strcmp(field, "length") == 0 && ops_at(p, "(")) {
                MGLExpr *call = expr_alloc(p, MGL_EXPR_CALL, e->line);
                if (!call) {
                    free(field);
                    break;
                }
                call->u.call.name = strdup("__mgl_array_length");
                call->u.call.args = (MGLExpr **)calloc(1, sizeof(MGLExpr *));
                if (!call->u.call.name || !call->u.call.args) {
                    free(call->u.call.name);
                    free(call->u.call.args);
                    free(call);
                    free(field);
                    break;
                }
                call->u.call.args[0] = e;
                call->u.call.arg_count = 1;
                eat_punct(p, "(");
                if (!ops_at(p, ")")) {
                    parse_error(p, "array length() takes no arguments at line %u",
                                tk_line(p));
                }
                expect_punct(p, ")");
                free(field);
                e = call;
                continue;
            }
            MGLExpr *m = expr_alloc(p, MGL_EXPR_MEMBER, e->line);
            if (m) {
                m->u.member.object = e;
                m->u.member.field = field;
            } else {
                free(field);
            }
            e = m;
        } else if (ops_at(p, "[")) {
            advance(p);
            MGLExpr *idx = parse_expression(p);
            expect_punct(p, "]");
            MGLExpr *ix = expr_alloc(p, MGL_EXPR_INDEX, e->line);
            if (ix) {
                ix->u.index.object = e;
                ix->u.index.index = idx;
            }
            e = ix;
        } else if (ops_at(p, "++") || ops_at(p, "--")) {
            int is_inc = (p->ts->src[tk(p, 0)->start] == '+');
            advance(p);
            advance(p);
            MGLExpr *u = expr_alloc(p, MGL_EXPR_UNARY, e->line);
            if (u) {
                u->u.unary.op = is_inc ? MGL_OP_INC : MGL_OP_DEC;
                u->u.unary.operand = e;
                u->u.unary.prefix = 0;
            }
            e = u;
        } else {
            break;
        }
    }
    return e;
}

static int prefix_op(MGLParser *p, uint32_t *op)
{
    if (ops_at(p, "++")) {
        *op = MGL_OP_INC;
        return 1;
    }
    if (ops_at(p, "--")) {
        *op = MGL_OP_DEC;
        return 1;
    }
    if (ops_at(p, "-")) {
        *op = MGL_OP_SUB;
        return 1;
    }
    if (ops_at(p, "+")) {
        *op = MGL_OP_ADD;
        return 1;
    }
    if (ops_at(p, "!")) {
        *op = MGL_OP_NOT;
        return 1;
    }
    if (ops_at(p, "~")) {
        *op = MGL_OP_BNOT;
        return 1;
    }
    return 0;
}

static MGLExpr *parse_unary(MGLParser *p)
{
    uint32_t op = 0;
    if (prefix_op(p, &op) != 0) {
        int nchars = 1;
        if (op == MGL_OP_INC || op == MGL_OP_DEC) {
            nchars = 2;
        }
        advance(p);
        if (nchars == 2) {
            advance(p);
        }
        MGLExpr *operand = parse_unary(p);
        if (!operand) {
            return NULL;
        }
        MGLExpr *u = expr_alloc(p, MGL_EXPR_UNARY, operand->line);
        if (u) {
            u->u.unary.op = op;
            u->u.unary.operand = operand;
            u->u.unary.prefix = 1;
        }
        return u;
    }
    return parse_postfix(p);
}

/* Binary operators: precedence table. */
static const struct {
    const char *tok;
    uint32_t op;
    int prec;
} MGL_BINOPS[] = {
    { "||", MGL_OP_LOR, 1 },
    { "&&", MGL_OP_LAND, 2 },
    { "|", MGL_OP_OR, 3 },
    { "^", MGL_OP_XOR, 4 },
    { "&", MGL_OP_AND, 5 },
    { "==", MGL_OP_EQ, 6 },
    { "!=", MGL_OP_NE, 6 },
    { "<<", MGL_OP_SHL, 8 },
    { ">>", MGL_OP_SHR, 8 },
    { "<=", MGL_OP_LE, 7 },
    { "<", MGL_OP_LT, 7 },
    { ">=", MGL_OP_GE, 7 },
    { ">", MGL_OP_GT, 7 },
    { "+", MGL_OP_ADD, 9 },
    { "-", MGL_OP_SUB, 9 },
    { "*", MGL_OP_MUL, 10 },
    { "/", MGL_OP_DIV, 10 },
    { "%", MGL_OP_MOD, 10 },
};
#define MGL_BINOP_COUNT (sizeof(MGL_BINOPS) / sizeof(MGL_BINOPS[0]))

static int lookup_binop(MGLParser *p, uint32_t *op, int *prec)
{
    unsigned i;
    for (i = 0; i < MGL_BINOP_COUNT; i++) {
        if (ops_at(p, MGL_BINOPS[i].tok)) {
            *op = MGL_BINOPS[i].op;
            *prec = MGL_BINOPS[i].prec;
            return 1;
        }
    }
    return 0;
}

static MGLExpr *parse_binary(MGLParser *p, int min_prec)
{
    MGLExpr *lhs = parse_unary(p);
    if (!lhs) {
        return NULL;
    }
    for (;;) {
        uint32_t op;
        int prec;
        if (!lookup_binop(p, &op, &prec) || prec < min_prec) {
            break;
        }
        /* don't eat the leading char of a compound-assignment like "+=":
         * those are matched by lookup_assign one level up */
        int nchars = 1;
        if (op == MGL_OP_LOR || op == MGL_OP_LAND || op == MGL_OP_EQ ||
            op == MGL_OP_NE || op == MGL_OP_LE || op == MGL_OP_GE ||
            op == MGL_OP_SHL || op == MGL_OP_SHR) {
            nchars = 2;
        }
        if ((op == MGL_OP_ADD || op == MGL_OP_SUB || op == MGL_OP_MUL ||
             op == MGL_OP_DIV || op == MGL_OP_MOD || op == MGL_OP_AND ||
             op == MGL_OP_OR || op == MGL_OP_XOR || op == MGL_OP_SHL ||
             op == MGL_OP_SHR) &&
            at_peek_punct(p, nchars, "=")) {
            break;
        }
        /* consume operator (1 or 2 chars); every two-char operator is one
         * of these, so the length follows from the matched op */
        advance(p);
        if (nchars > 1) {
            advance(p);
        }
        MGLExpr *rhs = parse_binary(p, prec + 1);
        if (!rhs) {
            return NULL;
        }
        MGLExpr *b = expr_alloc(p, MGL_EXPR_BINARY, lhs->line);
        if (b) {
            b->u.binary.op = op;
            b->u.binary.lhs = lhs;
            b->u.binary.rhs = rhs;
        }
        lhs = b;
    }
    return lhs;
}

/* Assignment operators. */
static const struct {
    const char *tok;
    uint32_t op;
} MGL_ASSOPS[] = {
    { "=", MGL_OP_ASSIGN },
    { "+=", MGL_OP_ADD_ASSIGN },
    { "-=", MGL_OP_SUB_ASSIGN },
    { "*=", MGL_OP_MUL_ASSIGN },
    { "/=", MGL_OP_DIV_ASSIGN },
    { "%=", MGL_OP_MOD_ASSIGN },
    { "<<=", MGL_OP_SHL_ASSIGN },
    { ">>=", MGL_OP_SHR_ASSIGN },
    { "&=", MGL_OP_AND_ASSIGN },
    { "|=", MGL_OP_OR_ASSIGN },
    { "^=", MGL_OP_XOR_ASSIGN },
};
#define MGL_ASSOP_COUNT (sizeof(MGL_ASSOPS) / sizeof(MGL_ASSOPS[0]))

static int lookup_assign(MGLParser *p, uint32_t *op, int *len)
{
    unsigned i;
    for (i = 0; i < MGL_ASSOP_COUNT; i++) {
        if (ops_at(p, MGL_ASSOPS[i].tok)) {
            *op = MGL_ASSOPS[i].op;
            *len = (int)strlen(MGL_ASSOPS[i].tok);
            return 1;
        }
    }
    return 0;
}

static MGLExpr *parse_assignment(MGLParser *p)
{
    MGLExpr *lhs = parse_binary(p, 1);
    if (!lhs) {
        return NULL;
    }
    uint32_t op;
    int len;
    if (lookup_assign(p, &op, &len)) {
        int i;
        for (i = 0; i < len; i++) {
            advance(p);
        }
        MGLExpr *rhs = parse_assignment(p);
        MGLExpr *a = expr_alloc(p, MGL_EXPR_ASSIGN, lhs->line);
        if (a) {
            a->u.assign.op = op;
            a->u.assign.lhs = lhs;
            a->u.assign.rhs = rhs;
        }
        return a;
    }
    if (ops_at(p, "?")) {
        advance(p);
        MGLExpr *then = parse_expression(p);
        expect_punct(p, ":");
        MGLExpr *els = parse_assignment(p);
        MGLExpr *t = expr_alloc(p, MGL_EXPR_TERNARY, lhs->line);
        if (t) {
            t->u.ternary.cond = lhs;
            t->u.ternary.then = then;
            t->u.ternary.else_ = els;
        }
        return t;
    }
    return lhs;
}

static MGLExpr *parse_expression(MGLParser *p)
{
    MGLExpr *lhs = parse_assignment(p);
    if (!lhs) {
        return NULL;
    }
    while (ops_at(p, ",")) {
        uint32_t line = lhs->line;
        advance(p);
        MGLExpr *rhs = parse_assignment(p);
        if (!rhs) {
            return NULL;
        }
        MGLExpr *b = expr_alloc(p, MGL_EXPR_BINARY, line);
        if (!b) {
            return NULL;
        }
        b->u.binary.op = MGL_OP_COMMA;
        b->u.binary.lhs = lhs;
        b->u.binary.rhs = rhs;
        lhs = b;
    }
    return lhs;
}

/* Array extents are integral constant expressions in GLSL.  The AST stores
 * only the resulting extent, so evaluate the constant subset while parsing
 * declarations and reject expressions that depend on runtime values. */

typedef struct MGLConstVal {
    uint32_t base; /* MGL_AST_TYPE_INT/UINT/BOOL/FLOAT */
    uint32_t size; /* 1 = scalar, 2-4 = vector */
    double v[4];
} MGLConstVal;

static void const_val_scalar(MGLConstVal *cv, uint32_t base, double x)
{
    cv->base = base;
    cv->size = 1;
    cv->v[0] = x;
    cv->v[1] = cv->v[2] = cv->v[3] = 0.0;
}

static void const_val_broadcast(MGLConstVal *cv, uint32_t base, uint32_t size, double x)
{
    cv->base = base;
    cv->size = size;
    for (uint32_t i = 0; i < 4; i++) {
        cv->v[i] = (i < size) ? x : 0.0;
    }
}

static int const_val_from_int(MGLConstVal *cv, uint32_t base, int64_t x)
{
    if (base == MGL_AST_TYPE_BOOL) {
        const_val_scalar(cv, base, x ? 1.0 : 0.0);
        return 1;
    }
    if (base == MGL_AST_TYPE_UINT) {
        const_val_scalar(cv, base, (double)(uint32_t)x);
        return 1;
    }
    const_val_scalar(cv, base, (double)x);
    return 1;
}

static int64_t const_val_to_int(const MGLConstVal *cv)
{
    double x = cv->v[0];
    if (cv->base == MGL_AST_TYPE_BOOL) {
        return x != 0.0 ? 1 : 0;
    }
    if (cv->base == MGL_AST_TYPE_UINT) {
        return (int64_t)(uint32_t)x;
    }
    return (int64_t)x; /* truncate toward zero */
}

static int const_type_name(const char *name, uint32_t *base, uint32_t *size)
{
    if (!name) {
        return 0;
    }
    struct {
        const char *n;
        uint32_t base;
        uint32_t size;
    } table[] = {
        {"bool", MGL_AST_TYPE_BOOL, 1}, {"bvec2", MGL_AST_TYPE_BOOL, 2},
        {"bvec3", MGL_AST_TYPE_BOOL, 3}, {"bvec4", MGL_AST_TYPE_BOOL, 4},
        {"int", MGL_AST_TYPE_INT, 1}, {"ivec2", MGL_AST_TYPE_INT, 2},
        {"ivec3", MGL_AST_TYPE_INT, 3}, {"ivec4", MGL_AST_TYPE_INT, 4},
        {"uint", MGL_AST_TYPE_UINT, 1}, {"uvec2", MGL_AST_TYPE_UINT, 2},
        {"uvec3", MGL_AST_TYPE_UINT, 3}, {"uvec4", MGL_AST_TYPE_UINT, 4},
        {"float", MGL_AST_TYPE_FLOAT, 1}, {"vec2", MGL_AST_TYPE_FLOAT, 2},
        {"vec3", MGL_AST_TYPE_FLOAT, 3}, {"vec4", MGL_AST_TYPE_FLOAT, 4},
    };
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
        if (strcmp(name, table[i].n) == 0) {
            *base = table[i].base;
            *size = table[i].size;
            return 1;
        }
    }
    return 0;
}

static int member_index(const char *field, uint32_t *idx)
{
    if (!field || !field[0] || field[1] != '\0') {
        return 0;
    }
    switch (field[0]) {
    case 'x':
    case 'r':
    case 's': *idx = 0; return 1;
    case 'y':
    case 'g':
    case 't': *idx = 1; return 1;
    case 'z':
    case 'b':
    case 'p': *idx = 2; return 1;
    case 'w':
    case 'a':
    case 'q': *idx = 3; return 1;
    default: return 0;
    }
}

static int eval_const_val(MGLParser *p, const MGLExpr *e, MGLConstVal *out);

static int eval_const_args(MGLParser *p, const MGLExpr *call, MGLConstVal *args, uint32_t max)
{
    if (!call || call->kind != MGL_EXPR_CALL || call->u.call.arg_count > max) {
        return 0;
    }
    for (uint32_t i = 0; i < call->u.call.arg_count; i++) {
        if (!eval_const_val(p, call->u.call.args[i], &args[i])) {
            return 0;
        }
    }
    return (int)call->u.call.arg_count;
}

static int eval_const_promote(MGLConstVal *a, MGLConstVal *b)
{
    if (a->size == b->size) {
        return 1;
    }
    if (a->size == 1 && b->size > 1) {
        double x = a->v[0];
        const_val_broadcast(a, a->base, b->size, x);
        return 1;
    }
    if (b->size == 1 && a->size > 1) {
        double x = b->v[0];
        const_val_broadcast(b, b->base, a->size, x);
        return 1;
    }
    return 0;
}

static int eval_const_binary_op(uint32_t op, const MGLConstVal *a,
                              const MGLConstVal *b, MGLConstVal *out)
{
    MGLConstVal lhs = *a;
    MGLConstVal rhs = *b;
    if (!eval_const_promote(&lhs, &rhs)) {
        return 0;
    }
    out->base = lhs.base;
    out->size = lhs.size;
    for (uint32_t i = 0; i < lhs.size; i++) {
        double x = lhs.v[i];
        double y = rhs.v[i];
        double r = 0.0;
        switch (op) {
        case MGL_OP_ADD: r = x + y; break;
        case MGL_OP_SUB: r = x - y; break;
        case MGL_OP_MUL: r = x * y; break;
        case MGL_OP_DIV: if (y == 0.0) return 0; r = x / y; break;
        case MGL_OP_MOD:
            if (y == 0.0) return 0;
            r = x - y * floor(x / y);
            break;
        case MGL_OP_EQ: r = (x == y) ? 1.0 : 0.0; break;
        case MGL_OP_NE: r = (x != y) ? 1.0 : 0.0; break;
        case MGL_OP_LT: r = (x < y) ? 1.0 : 0.0; break;
        case MGL_OP_LE: r = (x <= y) ? 1.0 : 0.0; break;
        case MGL_OP_GT: r = (x > y) ? 1.0 : 0.0; break;
        case MGL_OP_GE: r = (x >= y) ? 1.0 : 0.0; break;
        case MGL_OP_AND: r = ((int64_t)x & (int64_t)y); break;
        case MGL_OP_OR: r = ((int64_t)x | (int64_t)y); break;
        case MGL_OP_XOR: r = ((int64_t)x ^ (int64_t)y); break;
        case MGL_OP_SHL:
            if (y < 0.0 || y >= 64.0) return 0;
            r = (double)((int64_t)x << (int)y);
            break;
        case MGL_OP_SHR:
            if (y < 0.0 || y >= 64.0) return 0;
            r = (double)((int64_t)x >> (int)y);
            break;
        default: return 0;
        }
        out->v[i] = r;
    }
    return 1;
}

static int eval_const_unary_op(uint32_t op, const MGLConstVal *in, MGLConstVal *out)
{
    *out = *in;
    for (uint32_t i = 0; i < in->size; i++) {
        double x = in->v[i];
        switch (op) {
        case MGL_OP_ADD: out->v[i] = x; break;
        case MGL_OP_SUB: out->v[i] = -x; break;
        case MGL_OP_NOT: out->v[i] = x ? 0.0 : 1.0; break;
        case MGL_OP_BNOT: out->v[i] = (double)(~(int64_t)x); break;
        default: return 0;
        }
    }
    return 1;
}

static int eval_const_builtin(const char *name, uint32_t argc, MGLConstVal *args,
                              MGLConstVal *out)
{
    if (!name) {
        return 0;
    }

#define ARG(i) (args[i])
#define SCAL(i) (ARG(i).v[0])

    if (strcmp(name, "radians") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = ARG(0).v[i] * (M_PI / 180.0);
        }
        return 1;
    }
    if (strcmp(name, "degrees") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = ARG(0).v[i] * (180.0 / M_PI);
        }
        return 1;
    }
    if (strcmp(name, "sin") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = sin(ARG(0).v[i]);
        }
        return 1;
    }
    if (strcmp(name, "cos") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = cos(ARG(0).v[i]);
        }
        return 1;
    }
    if (strcmp(name, "asin") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = asin(ARG(0).v[i]);
        }
        return 1;
    }
    if (strcmp(name, "acos") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = acos(ARG(0).v[i]);
        }
        return 1;
    }
    if (strcmp(name, "pow") == 0 && argc == 2) {
        if (!eval_const_promote(&args[0], &args[1])) return 0;
        *out = args[0];
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = pow(args[0].v[i], args[1].v[i]);
        }
        return 1;
    }
    if (strcmp(name, "exp") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) out->v[i] = exp(ARG(0).v[i]);
        return 1;
    }
    if (strcmp(name, "log") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) out->v[i] = log(ARG(0).v[i]);
        return 1;
    }
    if (strcmp(name, "exp2") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) out->v[i] = pow(2.0, ARG(0).v[i]);
        return 1;
    }
    if (strcmp(name, "log2") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) out->v[i] = log2(ARG(0).v[i]);
        return 1;
    }
    if (strcmp(name, "sqrt") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) out->v[i] = sqrt(ARG(0).v[i]);
        return 1;
    }
    if (strcmp(name, "inversesqrt") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = 1.0 / sqrt(ARG(0).v[i]);
        }
        return 1;
    }
    if (strcmp(name, "abs") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) {
            double x = ARG(0).v[i];
            if (ARG(0).base == MGL_AST_TYPE_FLOAT) {
                out->v[i] = fabs(x);
            } else {
                int64_t iv = (int64_t)x;
                out->v[i] = (double)(iv < 0 ? -iv : iv);
            }
        }
        return 1;
    }
    if (strcmp(name, "sign") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) {
            double x = ARG(0).v[i];
            out->v[i] = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
        }
        return 1;
    }
    if (strcmp(name, "floor") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) out->v[i] = floor(ARG(0).v[i]);
        return 1;
    }
    if (strcmp(name, "trunc") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) out->v[i] = trunc(ARG(0).v[i]);
        return 1;
    }
    if (strcmp(name, "round") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) out->v[i] = round(ARG(0).v[i]);
        return 1;
    }
    if (strcmp(name, "ceil") == 0 && argc == 1) {
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) out->v[i] = ceil(ARG(0).v[i]);
        return 1;
    }
    if (strcmp(name, "mod") == 0 && argc == 2) {
        if (!eval_const_promote(&args[0], &args[1])) return 0;
        *out = args[0];
        for (uint32_t i = 0; i < out->size; i++) {
            double x = args[0].v[i];
            double y = args[1].v[i];
            if (y == 0.0) return 0;
            out->v[i] = x - y * floor(x / y);
        }
        return 1;
    }
    if (strcmp(name, "min") == 0 && argc == 2) {
        if (!eval_const_promote(&args[0], &args[1])) return 0;
        *out = args[0];
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = fmin(args[0].v[i], args[1].v[i]);
        }
        return 1;
    }
    if (strcmp(name, "max") == 0 && argc == 2) {
        if (!eval_const_promote(&args[0], &args[1])) return 0;
        *out = args[0];
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = fmax(args[0].v[i], args[1].v[i]);
        }
        return 1;
    }
    if (strcmp(name, "clamp") == 0 && argc == 3) {
        MGLConstVal tmp = args[0];
        if (!eval_const_promote(&tmp, &args[1])) return 0;
        args[0] = tmp;
        if (!eval_const_promote(&args[0], &args[2])) return 0;
        *out = args[0];
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] = fmin(fmax(args[0].v[i], args[1].v[i]), args[2].v[i]);
        }
        return 1;
    }
    if (strcmp(name, "length") == 0 && argc == 1) {
        double sum = 0.0;
        for (uint32_t i = 0; i < ARG(0).size; i++) {
            sum += ARG(0).v[i] * ARG(0).v[i];
        }
        const_val_scalar(out, MGL_AST_TYPE_FLOAT, sqrt(sum));
        return 1;
    }
    if (strcmp(name, "dot") == 0 && argc == 2) {
        if (!eval_const_promote(&args[0], &args[1])) return 0;
        double sum = 0.0;
        for (uint32_t i = 0; i < args[0].size; i++) {
            sum += args[0].v[i] * args[1].v[i];
        }
        const_val_scalar(out, MGL_AST_TYPE_FLOAT, sum);
        return 1;
    }
    if (strcmp(name, "normalize") == 0 && argc == 1) {
        double sum = 0.0;
        for (uint32_t i = 0; i < ARG(0).size; i++) {
            sum += ARG(0).v[i] * ARG(0).v[i];
        }
        if (sum == 0.0) {
            return 0;
        }
        double inv = 1.0 / sqrt(sum);
        *out = ARG(0);
        for (uint32_t i = 0; i < out->size; i++) {
            out->v[i] *= inv;
        }
        return 1;
    }

#undef ARG
#undef SCAL
    return 0;
}

static int eval_const_ctor(const char *name, uint32_t argc, MGLConstVal *args,
                           MGLConstVal *out)
{
    uint32_t base = 0;
    uint32_t size = 0;
    if (!const_type_name(name, &base, &size)) {
        return 0;
    }
    if (argc == 1 && args[0].size > 1 && size > 1) {
        if (args[0].size != size) {
            return 0;
        }
        *out = args[0];
        out->base = base;
        return 1;
    }
    if (argc == 1 && size >= 1) {
        const_val_broadcast(out, base, size, args[0].v[0]);
        return 1;
    }
    if ((int)argc == (int)size) {
        out->base = base;
        out->size = size;
        for (uint32_t i = 0; i < size; i++) {
            out->v[i] = args[i].v[0];
        }
        return 1;
    }
    return 0;
}

static int eval_const_val(MGLParser *p, const MGLExpr *e, MGLConstVal *out)
{
    if (!e || !out) {
        return 0;
    }
    switch (e->kind) {
    case MGL_EXPR_LITERAL:
        if (e->u.literal.base == MGL_AST_TYPE_FLOAT) {
            const_val_scalar(out, MGL_AST_TYPE_FLOAT, e->u.literal.value);
            return 1;
        }
        return const_val_from_int(out, e->u.literal.base, (int64_t)e->u.literal.value);
    case MGL_EXPR_VAR_REF: {
        int64_t v = 0;
        if (!lookup_const_int(p, e->u.var_ref.name, &v)) {
            return 0;
        }
        return const_val_from_int(out, MGL_AST_TYPE_INT, v);
    }
    case MGL_EXPR_MEMBER: {
        MGLConstVal obj;
        uint32_t idx = 0;
        if (!eval_const_val(p, e->u.member.object, &obj) ||
            !member_index(e->u.member.field, &idx) || idx >= obj.size) {
            return 0;
        }
        const_val_scalar(out, obj.base, obj.v[idx]);
        return 1;
    }
    case MGL_EXPR_CALL: {
        MGLConstVal args[4];
        int argc = eval_const_args(p, e, args, 4);
        if (argc < 0) {
            return 0;
        }
        if (e->u.call.name &&
            strcmp(e->u.call.name, "__mgl_array_length") == 0 &&
            argc == 1 && e->u.call.args[0] &&
            e->u.call.args[0]->kind == MGL_EXPR_VAR_REF) {
            uint32_t len = 0;
            if (!lookup_array_len(p, e->u.call.args[0]->u.var_ref.name, &len)) {
                return 0;
            }
            return const_val_from_int(out, MGL_AST_TYPE_INT, (int64_t)len);
        }
        if (eval_const_ctor(e->u.call.name, (uint32_t)argc, args, out)) {
            return 1;
        }
        return eval_const_builtin(e->u.call.name, (uint32_t)argc, args, out);
    }
    case MGL_EXPR_UNARY: {
        MGLConstVal in;
        if (!eval_const_val(p, e->u.unary.operand, &in)) {
            return 0;
        }
        return eval_const_unary_op(e->u.unary.op, &in, out);
    }
    case MGL_EXPR_BINARY: {
        MGLConstVal lhs, rhs;
        if (!eval_const_val(p, e->u.binary.lhs, &lhs) ||
            !eval_const_val(p, e->u.binary.rhs, &rhs)) {
            return 0;
        }
        if (e->u.binary.op == MGL_OP_LAND || e->u.binary.op == MGL_OP_LOR) {
            int64_t lv = const_val_to_int(&lhs);
            if (e->u.binary.op == MGL_OP_LAND && !lv) {
                return const_val_from_int(out, MGL_AST_TYPE_BOOL, 0);
            }
            if (e->u.binary.op == MGL_OP_LOR && lv) {
                return const_val_from_int(out, MGL_AST_TYPE_BOOL, 1);
            }
            MGLConstVal rhs_val;
            if (!eval_const_val(p, e->u.binary.rhs, &rhs_val)) {
                return 0;
            }
            return const_val_from_int(out, MGL_AST_TYPE_BOOL,
                                      const_val_to_int(&rhs_val));
        }
        return eval_const_binary_op(e->u.binary.op, &lhs, &rhs, out);
    }
    case MGL_EXPR_TERNARY: {
        MGLConstVal cond;
        if (!eval_const_val(p, e->u.ternary.cond, &cond)) {
            return 0;
        }
        return eval_const_val(p, const_val_to_int(&cond) ? e->u.ternary.then
                                                        : e->u.ternary.else_,
                              out);
    }
    default:
        return 0;
    }
}

static int eval_const_int(MGLParser *p, const MGLExpr *e, int64_t *value)
{
    MGLConstVal cv;
    if (!eval_const_val(p, e, &cv) || cv.size != 1) {
        return 0;
    }
    *value = const_val_to_int(&cv);
    return 1;
}

static uint32_t parse_array_extent(MGLParser *p)
{
    if (ops_at(p, "]")) {
        return 0; /* unsized array */
    }
    MGLExpr *expr = parse_expression(p);
    int64_t value = 0;
    int valid = eval_const_int(p, expr, &value);
    free_expr(expr);
    if (!valid || value < 0 || (uint64_t)value > UINT32_MAX) {
        parse_error(p, "array extent must be a non-negative constant at line %u",
                    tk_line(p));
        return 0;
    }
    return (uint32_t)value;
}

/* Append one array dimension to a declarator.  GLSL arrays-of-arrays require
 * version >= 430 (or ES 3.10); earlier versions reject a second dimension. */
static void append_array_dim(MGLParser *p, MGLDecl *d, uint32_t sz)
{
    if (d->array_count >= 1) {
        uint32_t ver = p->tu ? p->tu->version : 0;
        if (ver > 0 && ver < 430) {
            parse_error(p,
                        "arrays of arrays require GLSL 430+ at line %u",
                        tk_line(p));
            return;
        }
    }
    d->array_dims = (uint32_t *)realloc(
        d->array_dims, (d->array_count + 1) * sizeof(uint32_t));
    if (!d->array_dims) {
        d->array_count = 0;
        return;
    }
    d->array_dims[d->array_count++] = sz;
}

/* Parse zero or more `[N]` / `[]` array_specifier suffixes onto `d`. */
static void parse_array_specifier_list(MGLParser *p, MGLDecl *d)
{
    while (ops_at(p, "[")) {
        advance(p);
        uint32_t sz = parse_array_extent(p);
        expect_punct(p, "]");
        append_array_dim(p, d, sz);
    }
}

/* ------------------------------------------------------------------ */
/* Statements                                                          */
/* ------------------------------------------------------------------ */

static MGLStmt *stmt_alloc(MGLParser *p, uint32_t kind, uint32_t line);

static MGLStmt *parse_block(MGLParser *p)
{
    uint32_t line = tk_line(p);
    if (!expect_punct(p, "{")) {
        return NULL;
    }
    MGLStmt *s = stmt_alloc(p, MGL_STMT_COMPOUND, line);
    if (!s) {
        return NULL;
    }
    while (!ops_at(p, "}") && tk(p, 0)->kind != MGLGLSL_TOK_END) {
        if (tk(p, 0)->kind == MGLGLSL_TOK_DIRECTIVE) {
            advance(p);
            continue;
        }
        MGLStmt *sub = parse_statement(p);
        if (!sub) {
            break;
        }
        s->u.compound.stmts = (MGLStmt **)realloc(
            s->u.compound.stmts, (s->u.compound.count + 1) * sizeof(MGLStmt *));
        s->u.compound.stmts[s->u.compound.count++] = sub;
    }
    expect_punct(p, "}");
    return s;
}

static MGLStmt *parse_decl_stmt(MGLParser *p, uint32_t line)
{
    MGLDecl *d = parse_declaration(p);
    if (!d) {
        return NULL;
    }
    record_decl_constants(p, d);
    MGLStmt *s = stmt_alloc(p, MGL_STMT_DECL, line);
    if (s) {
        s->u.decl.decl = d;
    }
    return s;
}

static MGLStmt *parse_statement(MGLParser *p)
{
    uint32_t line = tk_line(p);
    if (ops_at(p, "{")) {
        return parse_block(p);
    }
    if (eat_ident(p, "if")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_IF, line);
        if (!s) {
            return NULL;
        }
        expect_punct(p, "(");
        s->u.ifs.cond = parse_expression(p);
        expect_punct(p, ")");
        s->u.ifs.then = parse_statement(p);
        if (eat_ident(p, "else")) {
            s->u.ifs.else_ = parse_statement(p);
        }
        return s;
    }
    if (eat_ident(p, "for")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_FOR, line);
        if (!s) {
            return NULL;
        }
        expect_punct(p, "(");
        if (ops_at(p, ";")) {
            advance(p);
        } else if (at_decl_start(p)) {
            s->u.loop.init = parse_decl_stmt(p, line);
        } else {
            MGLExpr *e = parse_expression(p);
            expect_punct(p, ";");
            MGLStmt *init = stmt_alloc(p, MGL_STMT_EXPR, line);
            if (init) {
                init->u.expr.expr = e;
            }
            s->u.loop.init = init;
        }
        if (!ops_at(p, ";")) {
            s->u.loop.cond = parse_expression(p);
        }
        expect_punct(p, ";");
        if (!ops_at(p, ")")) {
            if (!ops_at(p, ";")) {
                s->u.loop.incr = parse_expression(p);
            }
            if (ops_at(p, ";")) {
                advance(p);
            }
        }
        expect_punct(p, ")");
        s->u.loop.body = parse_statement(p);
        return s;
    }
    if (eat_ident(p, "while")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_WHILE, line);
        if (!s) {
            return NULL;
        }
        expect_punct(p, "(");
        s->u.whilex.cond = parse_expression(p);
        expect_punct(p, ")");
        s->u.whilex.body = parse_statement(p);
        return s;
    }
    if (eat_ident(p, "do")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_DO_WHILE, line);
        if (!s) {
            return NULL;
        }
        s->u.whilex.body = parse_statement(p);
        if (!eat_ident(p, "while")) {
            parse_error(p, "expected 'while' at line %u", tk_line(p));
        }
        expect_punct(p, "(");
        s->u.whilex.cond = parse_expression(p);
        expect_punct(p, ")");
        expect_punct(p, ";");
        return s;
    }
    if (eat_ident(p, "switch")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_SWITCH, line);
        if (!s) {
            return NULL;
        }
        expect_punct(p, "(");
        s->u.switchx.cond = parse_expression(p);
        expect_punct(p, ")");
        s->u.switchx.body = parse_statement(p);
        return s;
    }
    if (eat_ident(p, "case")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_CASE, line);
        if (!s) {
            return NULL;
        }
        s->u.casex.value = parse_expression(p);
        expect_punct(p, ":");
        return s;
    }
    if (eat_ident(p, "default")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_DEFAULT, line);
        expect_punct(p, ":");
        return s;
    }
    if (eat_ident(p, "break")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_BREAK, line);
        expect_punct(p, ";");
        return s;
    }
    if (eat_ident(p, "continue")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_CONTINUE, line);
        expect_punct(p, ";");
        return s;
    }
    if (eat_ident(p, "return")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_RETURN, line);
        if (!s) {
            return NULL;
        }
        if (!ops_at(p, ";")) {
            s->u.ret.value = parse_expression(p);
        }
        expect_punct(p, ";");
        return s;
    }
    if (eat_ident(p, "discard")) {
        MGLStmt *s = stmt_alloc(p, MGL_STMT_DISCARD, line);
        expect_punct(p, ";");
        return s;
    }

    /* null / empty statement (GLSL expression_statement with omitted expr) */
    if (ops_at(p, ";")) {
        advance(p);
        return stmt_alloc(p, MGL_STMT_EXPR, line);
    }

    /* Is this a declaration (qualifier or type keyword first)? */
    if (at_decl_start(p)) {
        return parse_decl_stmt(p, line);
    }

    MGLExpr *e = parse_expression(p);
    if (!e) {
        return NULL;
    }
    expect_punct(p, ";");
    MGLStmt *s = stmt_alloc(p, MGL_STMT_EXPR, line);
    if (s) {
        s->u.expr.expr = e;
    }
    return s;
}

static int at_decl_start(MGLParser *p)
{
    static const char *kw[] = {
        "float", "int", "uint", "bool", "double", "void",
        "vec2", "vec3", "vec4", "mat2", "mat3", "mat4",
        "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4",
        "mat4x2", "mat4x3", "mat4x4",
        "ivec2", "ivec3", "ivec4", "uvec2", "uvec3", "uvec4",
        "bvec2", "bvec3", "bvec4", "dvec2", "dvec3", "dvec4",
        "dmat2", "dmat3", "dmat4", "dmat2x2", "dmat2x3", "dmat2x4",
        "dmat3x2", "dmat3x3", "dmat3x4", "dmat4x2", "dmat4x3", "dmat4x4",
        "const", "uniform", "struct", "sampler2D", "samplerCube",
        "flat", "smooth", "invariant", "lowp", "mediump", "highp",
        "in", "out", "inout",
    };
    size_t i;
    for (i = 0; i < sizeof(kw) / sizeof(kw[0]); i++) {
        if (at_ident(p, kw[i])) {
            return 1;
        }
    }
    return 0;
}

static MGLStmt *stmt_alloc(MGLParser *p, uint32_t kind, uint32_t line)
{
    (void)p;
    MGLStmt *s = (MGLStmt *)calloc(1, sizeof(*s));
    if (s) {
        s->kind = kind;
        s->line = line;
    }
    return s;
}

/* ------------------------------------------------------------------ */
/* Declarations                                                        */
/* ------------------------------------------------------------------ */

static MGLDecl *parse_declaration(MGLParser *p)
{
    uint32_t line = tk_line(p);
    MGLDecl *d = (MGLDecl *)calloc(1, sizeof(*d));
    if (!d) {
        return NULL;
    }
    d->line = line;
    d->layout_location = -1;   /* "unspecified", per mgl_glsl_ast.h */
    d->layout_binding = -1;    /* "unspecified", per mgl_glsl_ast.h */
    d->layout_offset = -1;     /* atomic-counter offset, -1 = unspecified */
    d->layout_vertices = -1;   /* TCS: layout(vertices=N), unspecified */
    d->layout_max_vertices = -1; /* GS: layout(max_vertices=N), unspecified */
    /* -1 = unspecified; a later stage-level declaration must not
     * overwrite an earlier explicit invocations value with this default. */
    d->layout_invocations = -1;
    d->layout_stream = -1;   /* GS output stream, -1 = unspecified (0) */
    d->layout_primitive = MGL_AST_TES_DEFAULT;      /* TES mode / GS in topology */
    d->layout_primitive_out = MGL_AST_GS_OUT_DEFAULT;
    d->layout_spacing = MGL_AST_SPACING_DEFAULT;
    d->layout_winding = MGL_AST_WINDING_DEFAULT;
    d->layout_point_mode = 0;

    /* qualifiers and storage */
more_qualifiers:
    for (;;) {
        if (eat_ident(p, "const")) {
            d->qualifiers |= MGL_AST_Q_CONST;
        } else if (eat_ident(p, "in")) {
            d->qualifiers |= MGL_AST_Q_IN;
        } else if (eat_ident(p, "out")) {
            d->qualifiers |= MGL_AST_Q_OUT;
        } else if (eat_ident(p, "inout")) {
            d->qualifiers |= MGL_AST_Q_IN | MGL_AST_Q_OUT;
        } else if (eat_ident(p, "uniform")) {
            d->qualifiers |= MGL_AST_Q_UNIFORM;
        } else if (eat_ident(p, "buffer")) {
            d->qualifiers |= MGL_AST_Q_BUFFER;
        } else if (eat_ident(p, "shared")) {
            d->qualifiers |= MGL_AST_Q_SHARED;
        } else if (eat_ident(p, "flat")) {
            d->qualifiers |= MGL_AST_Q_FLAT;
        } else if (eat_ident(p, "smooth")) {
            d->qualifiers |= MGL_AST_Q_SMOOTH;
        } else if (eat_ident(p, "noperspective")) {
            d->qualifiers |= MGL_AST_Q_NOPERSPECTIVE;
        } else if (eat_ident(p, "centroid")) {
            d->qualifiers |= MGL_AST_Q_CENTROID;
        } else if (eat_ident(p, "sample")) {
            d->qualifiers |= MGL_AST_Q_SAMPLE;
        } else if (eat_ident(p, "patch")) {
            d->qualifiers |= MGL_AST_Q_PATCH;
        } else if (eat_ident(p, "invariant")) {
            d->qualifiers |= MGL_AST_Q_INVARIANT;
        } else if (eat_ident(p, "precise")) {
            d->qualifiers |= MGL_AST_Q_PRECISE;
        } else if (eat_ident(p, "readonly") ||
                   eat_ident(p, "writeonly") ||
                   eat_ident(p, "coherent") ||
                   eat_ident(p, "volatile") ||
                   eat_ident(p, "restrict")) {
            /* Image memory qualifiers constrain legal shader access but do
             * not change the declaration's storage or interface shape. */
        } else if (at_ident(p, "lowp") || at_ident(p, "mediump") ||
                   at_ident(p, "highp")) {
            /* precision qualifier consumed; recorded on the type later */
            uint32_t prec = eat_precision_qualifier(p);
            if (p->decl_precision == 0) {
                p->decl_precision = prec;
            }
        } else {
            break;
        }
    }

    /* layout(...) */
    if (eat_ident(p, "layout")) {
        if (!expect_punct(p, "(")) {
            free(d);
            return NULL;
        }
        while (tk(p, 0)->kind != MGLGLSL_TOK_END && !ops_at(p, ")")) {
            if (!at_any_ident(p)) {
                parse_error(p, "expected layout qualifier at line %u",
                            tk_line(p));
                break;
            }
            const MGLGLSLToken *t = tk(p, 0);
            size_t n = (size_t)(t->end - t->start);
            const char *s = p->ts->src + t->start;

            /* classification: layout flag or key = value */
            int is_flag =
                (n == 6 && memcmp(s, "std140", 6) == 0) ||
                (n == 6 && memcmp(s, "std430", 6) == 0) ||
                (n == 6 && memcmp(s, "shared", 6) == 0) ||
                (n == 6 && memcmp(s, "packed", 6) == 0) ||
                (n == 9 && memcmp(s, "row_major", 9) == 0) ||
                (n == 12 && memcmp(s, "column_major", 12) == 0) ||
                (n == 8 && memcmp(s, "invariant", 8) == 0) ||
                (n == 13 && memcmp(s, "push_constant", 13) == 0) ||
                (n == 16 && memcmp(s, "origin_upper_left", 16) == 0) ||
                (n == 15 && memcmp(s, "local_size_x_id", 15) == 0) ||
                /* tessellation/geometry layout flags (M3) */
                (n == 8 && memcmp(s, "isolines", 8) == 0) ||
                (n == 5 && memcmp(s, "quads", 5) == 0) ||
                (n == 9 && memcmp(s, "triangles", 9) == 0) ||
                (n == 13 && memcmp(s, "equal_spacing", 13) == 0) ||
                (n == 23 && memcmp(s, "fractional_even_spacing", 23) == 0) ||
                (n == 22 && memcmp(s, "fractional_odd_spacing", 22) == 0) ||
                (n == 2 && memcmp(s, "cw", 2) == 0) ||
                (n == 3 && memcmp(s, "ccw", 3) == 0) ||
                (n == 10 && memcmp(s, "point_mode", 10) == 0) ||
                (n == 6 && memcmp(s, "points", 6) == 0) ||
                (n == 5 && memcmp(s, "lines", 5) == 0) ||
                (n == 10 && memcmp(s, "line_strip", 10) == 0) ||
                (n == 15 && memcmp(s, "lines_adjacency", 15) == 0) ||
                (n == 14 && memcmp(s, "triangle_strip", 14) == 0) ||
                (n == 19 && memcmp(s, "triangles_adjacency", 19) == 0) ||
                is_image_format_layout(s, n);
            int has_value = at_peek_punct(p, 1, "=");

            if (!is_flag && !has_value) {
                /* e.g. `in`, `out` after `)` — leave it for the caller */
                break;
            }
            advance(p); /* consume the layout identifier */

            if (n == 6 && memcmp(s, "std140", 6) == 0) {
                d->layout = MGL_AST_LAYOUT_STD140;
            } else if (n == 6 && memcmp(s, "std430", 6) == 0) {
                d->layout = MGL_AST_LAYOUT_STD430;
            } else if (n == 6 && memcmp(s, "shared", 6) == 0) {
                d->layout = MGL_AST_LAYOUT_SHARED;
            } else if (n == 6 && memcmp(s, "packed", 6) == 0) {
                d->layout = MGL_AST_LAYOUT_PACKED;
            } else if (n == 9 && memcmp(s, "row_major", 9) == 0) {
                d->matrix_major = MGL_AST_MATRIX_ROW_MAJOR;
            } else if (n == 12 && memcmp(s, "column_major", 12) == 0) {
                d->matrix_major = MGL_AST_MATRIX_COL_MAJOR;
            } else if (n == 8 && memcmp(s, "isolines", 8) == 0) {
                d->layout_primitive = MGL_AST_TES_ISOLINES;
            } else if (n == 5 && memcmp(s, "quads", 5) == 0) {
                d->layout_primitive = MGL_AST_TES_QUADS;
            } else if (n == 9 && memcmp(s, "triangles", 9) == 0) {
                d->layout_primitive = MGL_AST_TES_TRIANGLES;
            } else if (n == 13 && memcmp(s, "equal_spacing", 13) == 0) {
                d->layout_spacing = MGL_AST_SPACING_EQUAL;
            } else if (n == 23 && memcmp(s, "fractional_even_spacing", 23) == 0) {
                d->layout_spacing = MGL_AST_SPACING_FRACTIONAL_EVEN;
            } else if (n == 22 && memcmp(s, "fractional_odd_spacing", 22) == 0) {
                d->layout_spacing = MGL_AST_SPACING_FRACTIONAL_ODD;
            } else if (n == 2 && memcmp(s, "cw", 2) == 0) {
                d->layout_winding = MGL_AST_WINDING_CW;
            } else if (n == 3 && memcmp(s, "ccw", 3) == 0) {
                d->layout_winding = MGL_AST_WINDING_CCW;
            } else if (n == 10 && memcmp(s, "point_mode", 10) == 0) {
                d->layout_point_mode = 1;
            } else if (n == 6 && memcmp(s, "points", 6) == 0) {
                d->layout_primitive = MGL_AST_GS_IN_POINTS;
            } else if (n == 5 && memcmp(s, "lines", 5) == 0) {
                d->layout_primitive = MGL_AST_GS_IN_LINES;
            } else if (n == 10 && memcmp(s, "line_strip", 10) == 0) {
                d->layout_primitive_out = MGL_AST_GS_OUT_LINE_STRIP;
            } else if (n == 15 && memcmp(s, "lines_adjacency", 15) == 0) {
                d->layout_primitive = MGL_AST_GS_IN_LINES_ADJACENCY;
            } else if (n == 14 && memcmp(s, "triangle_strip", 14) == 0) {
                d->layout_primitive_out = MGL_AST_GS_OUT_TRIANGLE_STRIP;
            } else if (n == 19 && memcmp(s, "triangles_adjacency", 19) == 0) {
                d->layout_primitive = MGL_AST_GS_IN_TRIANGLES_ADJACENCY;
            }

            if (has_value) {
                expect_punct(p, "=");
                if (at_num(p)) {
                    if (n == 8 && memcmp(s, "location", 8) == 0) {
                        d->layout_location = (int32_t)cur_double(p);
                    } else if (n == 7 && memcmp(s, "binding", 7) == 0) {
                        d->layout_binding = (int32_t)cur_double(p);
                    } else if (n == 6 && memcmp(s, "offset", 6) == 0) {
                        /* GLSL 4.60 §4.4.2.3: explicit atomic-counter
                         * buffer offset. */
                        d->layout_offset = (int32_t)cur_double(p);
                    } else if (n == 8 && memcmp(s, "vertices", 8) == 0) {
                        d->layout_vertices = (int32_t)cur_double(p);
                    } else if (n == 12 && memcmp(s, "max_vertices", 12) == 0) {
                        d->layout_max_vertices = (int32_t)cur_double(p);
                        if (d->layout_max_vertices < 0) {
                            parse_error(p,
                                "layout(max_vertices) must be >= 0 at line %u",
                                tk_line(p));
                        }
                    } else if (n == 11 && memcmp(s, "invocations", 11) == 0) {
                        d->layout_invocations = (int32_t)cur_double(p);
                        /* GLSL 4.60 §4.4.1.2: invocations <= 0 is a
                         * compile-time error (not silently defaulted to 1). */
                        if (d->layout_invocations <= 0) {
                            parse_error(p,
                                "layout(invocations) must be greater than zero at line %u",
                                tk_line(p));
                        }
                    } else if (n == 6 && memcmp(s, "stream", 6) == 0) {
                        d->layout_stream = (int32_t)cur_double(p);
                    }
                    advance(p);
                } else if (at_any_ident(p)) {
                    advance(p);
                } else {
                    parse_error(p, "expected value in layout() at line %u",
                                tk_line(p));
                }
            }

            if (!eat_punct(p, ",")) {
                break;
            }
        }
        expect_punct(p, ")");
        /* storage qualifiers may follow layout(...): e.g. "layout(...) in ..." */
        goto more_qualifiers;
    }

    /* `layout(...) in/out;` stage-level declarations (compute workgroup
     * size, TCS vertices, TES mode, GS topologies) have no type or
     * variable; the caller advances past the `;`.  Tessellation/geometry
     * layout is recorded on the translation unit for sema + backend. */
    if (ops_at(p, ";")) {
        MGLTranslationUnit *tu = p->tu;
        /* `points` is valid on both sides of a geometry stage declaration.
         * The layout parser sees the token before it sees the trailing
         * storage qualifier, so classify it here once `in`/`out` is known. */
        if ((d->qualifiers & MGL_AST_Q_OUT) &&
            d->layout_primitive == MGL_AST_GS_IN_POINTS) {
            d->layout_primitive = MGL_AST_TES_DEFAULT;
            d->layout_primitive_out = MGL_AST_GS_OUT_POINTS;
        }
        /* GL 4.6 §11.1.2gs: a second geometry output-primitive or
         * max_vertices declaration with a different value is a link-time
         * error; reject it at parse time so the program fails to build. */
        /* TU layout_max_vertices is -1 until declared (explicit 0 is a
         * degenerate no-output program). */
        if (d->layout_max_vertices >= 0 &&
            tu->layout_max_vertices >= 0 &&
            tu->layout_max_vertices != d->layout_max_vertices) {
            parse_error(p, "conflicting max_vertices declarations (%d vs %d)",
                        tu->layout_max_vertices, d->layout_max_vertices);
        }
        if (d->layout_primitive_out != MGL_AST_GS_OUT_DEFAULT &&
            tu->layout_primitive_out != MGL_AST_GS_OUT_DEFAULT &&
            tu->layout_primitive_out != d->layout_primitive_out) {
            parse_error(p, "conflicting geometry output primitive declarations");
        }
        if (d->layout_primitive != MGL_AST_TES_DEFAULT &&
            tu->layout_primitive != MGL_AST_TES_DEFAULT &&
            tu->layout_primitive != d->layout_primitive) {
            parse_error(p, "conflicting geometry input primitive declarations");
        }
        if (d->layout_vertices >= 0)          tu->layout_vertices = d->layout_vertices;
        if (d->layout_max_vertices >= 0)      tu->layout_max_vertices = d->layout_max_vertices;
        if (d->layout_invocations >= 1)       tu->layout_invocations = d->layout_invocations;
        if (d->layout_stream >= 0)            tu->layout_stream = d->layout_stream;
        if (d->layout_primitive != MGL_AST_TES_DEFAULT)
            tu->layout_primitive = d->layout_primitive;
        if (d->layout_primitive_out != MGL_AST_GS_OUT_DEFAULT)
            tu->layout_primitive_out = d->layout_primitive_out;
        if (d->layout_spacing != MGL_AST_SPACING_DEFAULT)
            tu->layout_spacing = d->layout_spacing;
        if (d->layout_winding != MGL_AST_WINDING_DEFAULT)
            tu->layout_winding = d->layout_winding;
        if (d->layout_point_mode)             tu->layout_point_mode = 1;
        free(d);
        return NULL;
    }

    /* struct definition?  "struct" keyword or a type followed by '{' */
    if (eat_ident(p, "struct")) {
        /* struct name */
        if (at_any_ident(p)) {
            d->type = (MGLTypeSpec *)calloc(1, sizeof(MGLTypeSpec));
            if (!d->type) {
                free(d);
                return NULL;
            }
            d->type->base = MGL_AST_TYPE_STRUCT;
            d->type->name = dup_current(p);
            advance(p);
        }
        if (ops_at(p, "{")) {
            advance(p);
            MGLDecl **members = NULL;
            uint32_t mcount = 0;
            while (!ops_at(p, "}") && tk(p, 0)->kind != MGLGLSL_TOK_END) {
                MGLDecl *m = parse_declaration(p);
                if (!m) {
                    break;
                }
                members = (MGLDecl **)realloc(
                    members, (mcount + 1) * sizeof(MGLDecl *));
                members[mcount++] = m;
            }
            expect_punct(p, "}");
            d->struct_members = members;
            d->struct_member_count = mcount;
            return d;
        }
        if (!d->type) {
            free(d);
            return NULL;
        }
        return d;
    }

    /* ordinary type */
    d->type = parse_type_spec(p);
    if (!d->type) {
        free(d);
        return NULL;
    }
    if (p->decl_precision) {
        d->type->precision = p->decl_precision;
        p->decl_precision = 0;
    }

    /* block definition:  type { ... } instance;  (UBO/SSBO) */
    if (ops_at(p, "{")) {
        advance(p);
        MGLDecl **members = NULL;
        uint32_t mcount = 0;
        while (!ops_at(p, "}") && tk(p, 0)->kind != MGLGLSL_TOK_END) {
            MGLDecl *m = parse_declaration(p);
            if (!m) {
                break;
            }
            members = (MGLDecl **)realloc(
                members, (mcount + 1) * sizeof(MGLDecl *));
            members[mcount++] = m;
        }
        expect_punct(p, "}");
        d->struct_members = members;
        d->struct_member_count = mcount;
        /* instance name */
        if (at_any_ident(p)) {
            d->name = dup_current(p);
            advance(p);
        }
        if (!ops_at(p, ";")) {
            if (at_any_ident(p)) {
                d->name = dup_current(p);
                advance(p);
            }
        }
        while (ops_at(p, "[")) {
            advance(p);
            uint32_t sz = parse_array_extent(p);
            expect_punct(p, "]");
            append_array_dim(p, d, sz);
        }
        expect_punct(p, ";");
        return d;
    }

    /* Array specifier on the type before the declarator name, e.g.
     * `float[3] x`, `float[] y`, `float[3] func(...)`. */
    parse_array_specifier_list(p, d);
    uint32_t type_prefix_dims = d->array_count;

    if (!at_any_ident(p)) {
        parse_error(p, "expected identifier at line %u", tk_line(p));
        free_decl(d);
        return NULL;
    }
    d->name = dup_current(p);
    advance(p);

    /* declarator postfix array dims: `float x[3]` / `float[2] x[3]` */
    parse_array_specifier_list(p, d);

    /* function? */
    if (ops_at(p, "(")) {
        advance(p);
        while (!ops_at(p, ")") && tk(p, 0)->kind != MGLGLSL_TOK_END) {
            MGLDecl *param = (MGLDecl *)calloc(1, sizeof(*param));
            if (!param) {
                break;
            }
            /* parameter_qualifier + precision_qualifier in either order
             * (GLSL 4.60 §6.1): `in highp float a` / `highp in float a`. */
            uint32_t param_prec = MGL_AST_PRECISION_NONE;
            for (;;) {
                if (eat_ident(p, "in")) {
                    param->qualifiers |= MGL_AST_Q_IN;
                } else if (eat_ident(p, "out")) {
                    param->qualifiers |= MGL_AST_Q_OUT;
                } else if (eat_ident(p, "inout")) {
                    param->qualifiers |= MGL_AST_Q_IN | MGL_AST_Q_OUT;
                } else if (at_ident(p, "lowp") || at_ident(p, "mediump") ||
                           at_ident(p, "highp")) {
                    if (param_prec == MGL_AST_PRECISION_NONE) {
                        param_prec = eat_precision_qualifier(p);
                    } else {
                        (void)eat_precision_qualifier(p);
                    }
                } else {
                    break;
                }
            }
            param->type = parse_type_spec(p);
            if (!param->type) {
                free(param);
                break;
            }
            if (param_prec != MGL_AST_PRECISION_NONE) {
                param->type->precision = param_prec;
            }
            /* Type-prefix arrays: `float[3]` / `float[3] a`. */
            parse_array_specifier_list(p, param);
            if (at_any_ident(p)) {
                param->name = dup_current(p);
                advance(p);
            }
            /* Declarator postfix: `float a[3]`. */
            parse_array_specifier_list(p, param);
            d->params = (MGLDecl **)realloc(
                d->params, (d->param_count + 1) * sizeof(MGLDecl *));
            d->params[d->param_count++] = param;
            if (!eat_punct(p, ",")) {
                break;
            }
        }
        expect_punct(p, ")");
        if (ops_at(p, "{")) {
            MGLStmt *body = parse_block(p);
            if (!body) {
                free(d);
                return NULL;
            }
            d->body = body;
        } else {
            expect_punct(p, ";");
        }
        /* Mark as function even with zero parameters / no body so sema
         * does not treat `float f();` as a variable. */
        d->return_type = d->type;
        return d;
    }

    /* Comma-separated additional declarators share the declaration's
     * type spec, qualifiers and layout header (GLSL 4.60 §4.1: `int a,
     * b[2], c = 3;`).  Each node owns its name, array dims and
     * initializer; the type spec itself stays owned by the first
     * declarator. */
    {
        MGLDecl *tail = d;
        for (;;) {
            if (eat_punct(p, "=")) {
                tail->init = parse_expression(p);
            }
            if (!eat_punct(p, ",")) {
                break;
            }
            MGLDecl *nd = (MGLDecl *)calloc(1, sizeof(*nd));
            if (!nd) {
                break;
            }
            *nd = *tail;
            nd->next_declarator = NULL;
            nd->type_shared = 1;       /* type owned by the first node */
            nd->name = NULL;
            nd->array_dims = NULL;
            nd->array_count = 0;
            nd->init = NULL;
            nd->body = NULL;
            nd->params = NULL;
            nd->param_count = 0;
            nd->return_type = NULL;
            nd->struct_members = NULL;
            nd->struct_member_count = 0;
            nd->line = tk_line(p);
            if (!at_any_ident(p)) {
                parse_error(p, "expected identifier at line %u", tk_line(p));
                free_decl(nd);
                break;
            }
            nd->name = dup_current(p);
            advance(p);
            /* Inherit type-prefix dims (`float[3] a, b`) then allow
             * per-declarator postfix dims. */
            for (uint32_t i = 0; i < type_prefix_dims; i++) {
                append_array_dim(p, nd, d->array_dims[i]);
            }
            parse_array_specifier_list(p, nd);
            tail->next_declarator = nd;
            tail = nd;
        }
    }

    expect_punct(p, ";");
    return d;
}

/* Desktop GLSL accepts ES-style default precision statements as no-ops.
 * Consume them at translation-unit scope without creating an AST declaration. */
static int parse_precision_statement(MGLParser *p)
{
    if (!eat_ident(p, "precision")) {
        return 0;
    }
    if (eat_precision_qualifier(p) == MGL_AST_PRECISION_NONE) {
        parse_error(p, "expected precision qualifier at line %u", tk_line(p));
        return 1;
    }
    if (!at_any_ident(p)) {
        parse_error(p, "expected type at line %u", tk_line(p));
        return 1;
    }
    advance(p); /* scalar or opaque type; the default has no desktop effect */
    expect_punct(p, ";");
    return 1;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

/* Preprocess conditional directives (#ifdef/#ifndef/#else/#endif) by
 * dropping tokens of inactive branches; #define names are recorded so
 * #ifdef evaluates them (macro bodies are not expanded here).  #version/
 * #extension/#pragma directives are kept. */
/* Extract the identifier after a directive keyword, bounded by the
 * directive token (one source line): scanning past the token would run
 * into following lines and swallow the whole file as the "name". */
static size_t pp_directive_name(const char *d, size_t n, size_t kw,
                                const char **nm_out)
{
    const char *p = d + kw;
    const char *end = d + n;
    while (p < end && (*p == ' ' || *p == '\t')) p++;
    const char *start = p;
    while (p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
        p++;
    *nm_out = start;
    return (size_t)(p - start);
}

static void preprocess_tokens(MGLTokenStream *ts)
{
    enum { MAX_DEPTH = 32, MAX_DEFS = 64, MAX_NAME = 63 };
    int active[MAX_DEPTH];
    int parent_active[MAX_DEPTH];
    int depth = 0;
    char defs[MAX_DEFS][MAX_NAME + 1];
    uint32_t def_count = 0;

    uint32_t out = 0;
    for (uint32_t i = 0; i < (uint32_t)ts->count; i++) {
        MGLGLSLToken *t = &ts->tok[i];
        int cur_active = (depth == 0) ? 1 : active[depth - 1];
        if (t->kind != MGLGLSL_TOK_DIRECTIVE) {
            if (cur_active) ts->tok[out++] = *t;
            continue;
        }
        const char *d = ts->src + t->start;
        size_t n = (size_t)(t->end - t->start);
        int is_kw = (n >= 6 && memcmp(d, "#ifdef", 6) == 0 &&
                     (n == 6 || d[6] == ' '));
        if (is_kw) {
            if (depth >= MAX_DEPTH) break;
            int cond = 0;
            if (cur_active) {
                const char *nm;
                size_t nl = pp_directive_name(d, n, 6, &nm);
                for (uint32_t k = 0; k < def_count; k++) {
                    if (strlen(defs[k]) == nl && memcmp(defs[k], nm, nl) == 0) {
                        cond = 1;
                        break;
                    }
                }
            }
            parent_active[depth] = cur_active;
            active[depth] = cur_active && cond;
            depth++;
            continue;
        }
        if (n >= 7 && memcmp(d, "#ifndef", 7) == 0 &&
            (n == 7 || d[7] == ' ')) {
            if (depth >= MAX_DEPTH) break;
            int cond = 1;
            if (cur_active) {
                const char *nm;
                size_t nl = pp_directive_name(d, n, 7, &nm);
                for (uint32_t k = 0; k < def_count; k++) {
                    if (strlen(defs[k]) == nl && memcmp(defs[k], nm, nl) == 0) {
                        cond = 0;
                        break;
                    }
                }
            }
            parent_active[depth] = cur_active;
            active[depth] = cur_active && cond;
            depth++;
            continue;
        }
        if (n >= 5 && memcmp(d, "#else", 5) == 0 &&
            (n == 5 || d[5] == ' ')) {
            if (depth > 0) {
                active[depth - 1] =
                    parent_active[depth - 1] && !active[depth - 1];
            }
            continue;
        }
        if (n >= 6 && memcmp(d, "#endif", 6) == 0 &&
            (n == 6 || d[6] == ' ')) {
            if (depth > 0) depth--;
            continue;
        }
        if (n >= 7 && memcmp(d, "#define", 7) == 0 &&
            (n == 7 || d[7] == ' ' || d[7] == '\t')) {
            if (cur_active) {
                const char *nm;
                size_t nl = pp_directive_name(d, n, 7, &nm);
                if (nl > 0 && nl <= MAX_NAME && def_count < MAX_DEFS) {
                    memcpy(defs[def_count], nm, nl);
                    defs[def_count][nl] = 0;
                    def_count++;
                }
            }
            continue;
        }
        if (n >= 6 && memcmp(d, "#undef", 6) == 0 &&
            (n == 6 || d[6] == ' ')) {
            const char *nm;
            size_t nl = pp_directive_name(d, n, 6, &nm);
            for (uint32_t k = 0; k < def_count; k++) {
                if (strlen(defs[k]) == nl && memcmp(defs[k], nm, nl) == 0) {
                    memmove(&defs[k], &defs[k + 1],
                            (def_count - k - 1) * sizeof(defs[0]));
                    def_count--;
                    break;
                }
            }
            continue;
        }
        if (cur_active) ts->tok[out++] = *t;
    }
    ts->count = out;
}

static int pp_is_ident_start(char c)
{
    return isalpha((unsigned char)c) || c == '_';
}

static int pp_is_ident_char(char c)
{
    return isalnum((unsigned char)c) || c == '_';
}

static int pp_is_digit(char c)
{
    return c >= '0' && c <= '9';
}

/* Object-like macro expansion: collect `#define NAME value` lines and
 * replace NAME occurrences outside directives with the value text.  Runs
 * before tokenizing so macro values lex as normal tokens. */
static __attribute__((unused)) char *preprocess_macros(const char *src, size_t len)
{
    enum { MAX_MACROS = 64, MAX_NAME = 64, MAX_VAL = 256 };
    char names[MAX_MACROS][MAX_NAME];
    char vals[MAX_MACROS][MAX_VAL];
    uint32_t mcount = 0;

    const char *p = src;
    const char *end = src + len;
    while (p < end) {
        const char *eol = (const char *)memchr(p, '\n', (size_t)(end - p));
        if (!eol) eol = end;
        const char *line = p;
        size_t llen = (size_t)(eol - p);
        const char *c = line;
        while (c < eol && (*c == ' ' || *c == '\t')) c++;
        if (c + 7 <= eol && memcmp(c, "#define", 7) == 0 &&
            (c + 7 == eol || c[7] == ' ' || c[7] == '\t')) {
            c += 7;
            while (c < eol && (*c == ' ' || *c == '\t')) c++;
            size_t nlen = 0;
            while (c + nlen < eol &&
                   (pp_is_ident_char(c[nlen]) || (nlen > 0 && pp_is_digit(c[nlen])))) {
                nlen++;
            }
            if (nlen > 0 && nlen < MAX_NAME) {
                size_t vstart = nlen;
                while (vstart < llen && (c[vstart] == ' ' || c[vstart] == '\t')) {
                    vstart++;
                }
                size_t vlen = (size_t)(eol - (c + vstart));
                if (vlen > MAX_VAL - 1) vlen = MAX_VAL - 1;
                if (mcount < MAX_MACROS) {
                    memcpy(names[mcount], c, nlen);
                    names[mcount][nlen] = 0;
                    memcpy(vals[mcount], c + vstart, vlen);
                    vals[mcount][vlen] = 0;
                    mcount++;
                }
            }
        }
        p = eol + 1;
    }

    /* Expand into a fresh buffer. */
    size_t cap = len + 1024;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t o = 0;
    p = src;
    int line_start = 1;
    while (p < end) {
        if (line_start && *p == '#') {
            const char *eol = (const char *)memchr(p, '\n', (size_t)(end - p));
            size_t n = eol ? (size_t)(eol - p) + 1 : (size_t)(end - p);
            if (o + n + 1 > cap) {
                cap = cap * 2 + n;
                char *no = (char *)realloc(out, cap);
                if (!no) { free(out); return NULL; }
                out = no;
            }
            memcpy(out + o, p, n);
            o += n;
            p += n;
            line_start = 1;
            continue;
        }
        if (pp_is_ident_start(*p)) {
            const char *s = p;
            size_t ilen = 0;
            while (p < end && (pp_is_ident_char(*p) || pp_is_digit(*p))) {
                p++;
                ilen++;
            }
            const char *val = NULL;
            for (uint32_t i = 0; i < mcount; i++) {
                if (strlen(names[i]) == ilen &&
                    memcmp(names[i], s, ilen) == 0) {
                    val = vals[i];
                    break;
                }
            }
            if (val) {
                size_t vlen = strlen(val);
                if (o + vlen + 1 > cap) {
                    cap = cap * 2 + vlen;
                    char *no = (char *)realloc(out, cap);
                    if (!no) { free(out); return NULL; }
                    out = no;
                }
                memcpy(out + o, val, vlen);
                o += vlen;
            } else {
                if (o + ilen + 1 > cap) {
                    cap = cap * 2 + ilen;
                    char *no = (char *)realloc(out, cap);
                    if (!no) { free(out); return NULL; }
                    out = no;
                }
                memcpy(out + o, s, ilen);
                o += ilen;
            }
            continue;
        }
        if (o + 1 >= cap) {
            cap *= 2;
            char *no = (char *)realloc(out, cap);
            if (!no) { free(out); return NULL; }
            out = no;
        }
        out[o++] = *p;
        if (*p == '\n') line_start = 1;
        else line_start = 0;
        p++;
    }
    out[o] = 0;
    return out;
}

MGLTranslationUnit *mglGLSLParse(const char *src, size_t len)
{
    char pperr[256];
    char *ppsrc;
    MGLTokenStream ts;
    memset(&ts, 0, sizeof(ts));
    pperr[0] = 0;
    ppsrc = mglGLSLPreprocess(src, len, pperr, sizeof(pperr));
    if (!ppsrc) {
        MGLTranslationUnit *etu = (MGLTranslationUnit *)calloc(1, sizeof(*etu));
        if (!etu) {
            return NULL;
        }
        etu->layout_stream = -1;
        etu->layout_max_vertices = -1;
        etu->error = strdup(pperr[0] ? pperr : "preprocessor error");
        return etu;
    }
    if (tokenize(&ts, ppsrc, strlen(ppsrc)) != 0) {
        free(ppsrc);
        return NULL;
    }
    free(ppsrc);
    preprocess_tokens(&ts);

    MGLTranslationUnit *tu = (MGLTranslationUnit *)calloc(1, sizeof(*tu));
    if (!tu) {
        token_stream_free(&ts);
        return NULL;
    }
    tu->layout_stream = -1; /* GS default output stream unspecified (0) */
    tu->layout_max_vertices = -1; /* GS: unspecified until layout() */

    MGLParser p;
    memset(&p, 0, sizeof(p));
    p.ts = &ts;
    p.tu = tu;

    /* extract #version */
    for (int i = 0; i < ts.count; i++) {
        const MGLGLSLToken *t = &ts.tok[i];
        if (t->kind == MGLGLSL_TOK_DIRECTIVE) {
            const char *d = ts.src + t->start;
            size_t n = (size_t)(t->end - t->start);
            if (n >= 8 && memcmp(d, "#version", 8) == 0) {
                unsigned int v = 0;
                char prof[32] = { 0 };
                if (sscanf(d + 8, "%u %31s", &v, prof) >= 1) {
                    tu->version = v;
                    if (prof[0]) {
                        tu->version_profile = strdup(prof);
                    }
                }
            }
        }
    }

    /* top-level directives skipped; parse declarations */
    while (tk(&p, 0)->kind != MGLGLSL_TOK_END) {
        if (tk(&p, 0)->kind == MGLGLSL_TOK_DIRECTIVE) {
            advance(&p);
            continue;
        }
        if (parse_precision_statement(&p)) {
            continue;
        }
        MGLDecl *d = parse_declaration(&p);
        if (!d) {
            if (!tu->error && !ops_at(&p, ";")) {
                parse_error(&p, "unexpected token at line %u", tk_line(&p));
            }
            advance(&p);
            continue;
        }
        record_decl_constants(&p, d);
        tu->decls = (MGLDecl **)realloc(
            tu->decls, (tu->decl_count + 1) * sizeof(MGLDecl *));
        tu->decls[tu->decl_count++] = d;
    }

    token_stream_free(&ts);
    return tu;
}

void mglGLSLTranslationUnitDestroy(MGLTranslationUnit *tu)
{
    if (!tu) {
        return;
    }
    unsigned i;
    for (i = 0; i < tu->decl_count; i++) {
        free_decl(tu->decls[i]);
    }
    free(tu->decls);
    free(tu->version_profile);
    free(tu->error);
    free(tu);
}

/* ------------------------------------------------------------------ */
/* AST destruction                                                     */
/* ------------------------------------------------------------------ */

static void free_type_spec(MGLTypeSpec *ts)
{
    if (!ts) {
        return;
    }
    free(ts->name);
    free(ts->struct_def);
    free(ts);
}

static void free_expr(MGLExpr *e);

static void free_expr(MGLExpr *e)
{
    if (!e) {
        return;
    }
    switch (e->kind) {
    case MGL_EXPR_VAR_REF:
        free(e->u.var_ref.name);
        break;
    case MGL_EXPR_MEMBER:
        free_expr(e->u.member.object);
        free(e->u.member.field);
        break;
    case MGL_EXPR_INDEX:
        free_expr(e->u.index.object);
        free_expr(e->u.index.index);
        break;
    case MGL_EXPR_CALL: {
        unsigned i;
        for (i = 0; i < e->u.call.arg_count; i++) {
            free_expr(e->u.call.args[i]);
        }
        free(e->u.call.args);
        free(e->u.call.name);
        break;
    }
    case MGL_EXPR_UNARY:
        free_expr(e->u.unary.operand);
        break;
    case MGL_EXPR_BINARY:
        free_expr(e->u.binary.lhs);
        free_expr(e->u.binary.rhs);
        break;
    case MGL_EXPR_ASSIGN:
        free_expr(e->u.assign.lhs);
        free_expr(e->u.assign.rhs);
        break;
    case MGL_EXPR_TERNARY:
        free_expr(e->u.ternary.cond);
        free_expr(e->u.ternary.then);
        free_expr(e->u.ternary.else_);
        break;
    case MGL_EXPR_LITERAL:
    default:
        break;
    }
    free(e);
}

static void free_stmt(MGLStmt *s);

static void free_stmt(MGLStmt *s)
{
    if (!s) {
        return;
    }
    switch (s->kind) {
    case MGL_STMT_COMPOUND: {
        unsigned i;
        for (i = 0; i < s->u.compound.count; i++) {
            free_stmt(s->u.compound.stmts[i]);
        }
        free(s->u.compound.stmts);
        break;
    }
    case MGL_STMT_EXPR:
        free_expr(s->u.expr.expr);
        break;
    case MGL_STMT_DECL:
        free_decl(s->u.decl.decl);
        break;
    case MGL_STMT_IF:
        free_expr(s->u.ifs.cond);
        free_stmt(s->u.ifs.then);
        free_stmt(s->u.ifs.else_);
        break;
    case MGL_STMT_FOR:
        free_stmt(s->u.loop.init);
        free_expr(s->u.loop.cond);
        free_expr(s->u.loop.incr);
        free_stmt(s->u.loop.body);
        break;
    case MGL_STMT_WHILE:
    case MGL_STMT_DO_WHILE:
        free_expr(s->u.whilex.cond);
        free_stmt(s->u.whilex.body);
        break;
    case MGL_STMT_SWITCH:
        free_expr(s->u.switchx.cond);
        free_stmt(s->u.switchx.body);
        break;
    case MGL_STMT_CASE:
        free_expr(s->u.casex.value);
        break;
    case MGL_STMT_DEFAULT:
    case MGL_STMT_BREAK:
    case MGL_STMT_CONTINUE:
    case MGL_STMT_DISCARD:
        break;
    case MGL_STMT_RETURN:
        free_expr(s->u.ret.value);
        break;
    default:
        break;
    }
    free(s);
}

static void free_decl(MGLDecl *d)
{
    if (!d) {
        return;
    }
    free(d->name);
    if (!d->type_shared) {
        free_type_spec(d->type);
    }
    free(d->array_dims);
    free_expr(d->init);
    free_stmt(d->body);
    unsigned i;
    for (i = 0; i < d->param_count; i++) {
        free_decl(d->params[i]);
    }
    free(d->params);
    for (i = 0; i < d->struct_member_count; i++) {
        free_decl(d->struct_members[i]);
    }
    free(d->struct_members);
    /* Comma-separated sibling declarators form a singly linked chain. */
    MGLDecl *next = d->next_declarator;
    free(d);
    free_decl(next);
}
