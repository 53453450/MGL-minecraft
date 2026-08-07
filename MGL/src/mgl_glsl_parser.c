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
    ts->src_len = len;
    ts->count = 0;

    MGLGLSLexer lx;
    mglGLSLexerInit(&lx, src, len);
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
} MGLParser;

static unsigned int tk_line(MGLParser *p);
static const MGLGLSLToken *tk(MGLParser *p, int offset);

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
        if (ops_at(p, "(")) {
            MGLExpr *e = expr_alloc(p, MGL_EXPR_CALL, line);
            if (e) {
                e->u.call.name = name;
                eat_punct(p, "(");
                uint32_t argc = 0;
                if (!ops_at(p, ")")) {
                    for (;;) {
                        MGLExpr *arg = parse_expression(p);
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
            MGLExpr *m = expr_alloc(p, MGL_EXPR_MEMBER, e->line);
            if (m) {
                m->u.member.object = e;
                m->u.member.field = dup_current(p);
                advance(p);
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
    return parse_assignment(p);
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
        if (!ops_at(p, ";")) {
            if (at_decl_start(p)) {
                s->u.loop.init = parse_decl_stmt(p, line);
            } else {
                s->u.loop.init = parse_statement(p);
            }
        }
        /* skip cond only when the ";" directly precedes ")" */
        if (!ops_at(p, ";")) {
            s->u.loop.cond = parse_expression(p);
        }
        expect_punct(p, ";");
        if (!ops_at(p, ")") && !ops_at(p, ";")) {
            s->u.loop.incr = parse_expression(p);
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
        s->u.body.body = parse_statement(p);
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
        } else if (eat_ident(p, "lowp") || eat_ident(p, "mediump") ||
                   eat_ident(p, "highp")) {
            /* precision qualifier consumed; recorded on the type later */
            uint32_t prec = 0;
            if (at_ident(p, "lowp")) {
                prec = MGL_AST_PRECISION_LOWP;
            } else if (at_ident(p, "mediump")) {
                prec = MGL_AST_PRECISION_MEDIUMP;
            } else {
                prec = MGL_AST_PRECISION_HIGHP;
            }
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
                (n == 11 && memcmp(s, "column_major", 11) == 0) ||
                (n == 8 && memcmp(s, "invariant", 8) == 0) ||
                (n == 13 && memcmp(s, "push_constant", 13) == 0) ||
                (n == 17 && memcmp(s, "origin_upper_left", 17) == 0) ||
                (n == 15 && memcmp(s, "local_size_x_id", 15) == 0);
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
            } else if (n == 11 && memcmp(s, "column_major", 11) == 0) {
                d->matrix_major = MGL_AST_MATRIX_COL_MAJOR;
            }

            if (has_value) {
                expect_punct(p, "=");
                if (at_num(p) || at_any_ident(p)) {
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
            uint32_t sz = 0;
            if (at_num(p)) {
                sz = (uint32_t)cur_double(p);
                advance(p);
            }
            expect_punct(p, "]");
            d->array_dims = (uint32_t *)realloc(
                d->array_dims, (d->array_count + 1) * sizeof(uint32_t));
            d->array_dims[d->array_count++] = sz;
        }
        expect_punct(p, ";");
        return d;
    }

    /* declarator name */
    if (!at_any_ident(p)) {
        parse_error(p, "expected identifier at line %u", tk_line(p));
        free_decl(d);
        return NULL;
    }
    d->name = dup_current(p);
    advance(p);

    /* array dims */
    while (ops_at(p, "[")) {
        advance(p);
        uint32_t sz = 0;
        if (at_num(p)) {
            sz = (uint32_t)cur_double(p);
            advance(p);
        }
        expect_punct(p, "]");
        d->array_dims = (uint32_t *)realloc(
            d->array_dims, (d->array_count + 1) * sizeof(uint32_t));
        d->array_dims[d->array_count++] = sz;
    }

    /* function? */
    if (ops_at(p, "(")) {
        advance(p);
        while (!ops_at(p, ")") && tk(p, 0)->kind != MGLGLSL_TOK_END) {
            MGLDecl *param = (MGLDecl *)calloc(1, sizeof(*param));
            if (!param) {
                break;
            }
            if (eat_ident(p, "in")) {
                param->qualifiers |= MGL_AST_Q_IN;
            } else if (eat_ident(p, "out")) {
                param->qualifiers |= MGL_AST_Q_OUT;
            } else if (eat_ident(p, "inout")) {
                param->qualifiers |= MGL_AST_Q_IN | MGL_AST_Q_OUT;
            }
            param->type = parse_type_spec(p);
            if (!param->type) {
                free(param);
                break;
            }
            if (at_any_ident(p)) {
                param->name = dup_current(p);
                advance(p);
            }
            while (ops_at(p, "[")) {
                advance(p);
                if (!ops_at(p, "]") && at_num(p)) {
                    advance(p);
                }
                expect_punct(p, "]");
                param->array_count++;
            }
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
        return d;
    }

    /* initializer */
    if (eat_punct(p, "=")) {
        d->init = parse_expression(p);
    }

    expect_punct(p, ";");
    return d;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

MGLTranslationUnit *mglGLSLParse(const char *src, size_t len)
{
    MGLTokenStream ts;
    memset(&ts, 0, sizeof(ts));
    if (tokenize(&ts, src, len) != 0) {
        return NULL;
    }

    MGLTranslationUnit *tu = (MGLTranslationUnit *)calloc(1, sizeof(*tu));
    if (!tu) {
        token_stream_free(&ts);
        return NULL;
    }

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
        MGLDecl *d = parse_declaration(&p);
        if (!d) {
            if (!tu->error) {
                parse_error(&p, "unexpected token at line %u", tk_line(&p));
            }
            advance(&p);
            continue;
        }
        d->return_type = d->type != NULL && d->body != NULL ? d->type : NULL;
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
    free_type_spec(d->type);
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
    free(d);
}