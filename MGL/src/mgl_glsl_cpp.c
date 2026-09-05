/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * GLSL 4.60 §3.3 preprocessor: object- and function-like macros, ##,
 * #if/#elif integer expressions, and diagnostics for ill-formed directives.
 */

#include "mgl_glsl_cpp.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TK_ID = 1, TK_NUM, TK_PUNCT };

typedef struct {
    int kind;
    char *s;
    int spaced;
} Tok;

typedef struct {
    Tok *t;
    size_t n, cap;
} TokList;

typedef struct {
    char *name;
    int function_like;
    int nparams;
    char **params;
    TokList body;
    int special; /* 1=__LINE__ 2=__FILE__ 3=__VERSION__ */
    int reserved_gl;
} Macro;

enum { COND_MAX = 64, HIDE_MAX = 64, EXPAND_MAX = 256 };

typedef struct {
    int parent;
    int taking;
    int else_seen;
    int taken;
} Cond;

typedef struct {
    Macro *macros;
    size_t nmac, capmac;
    Cond cond[COND_MAX];
    int depth;
    int version;
    int file_no;
    long line;
    int failed;
    int if_mode;
    int saw_content;
    int expand_depth; /* nest depth of expand_macro/expand_list */
    char err[256];
    char *out;
    size_t olen, ocap;
} PP;

static int is_ident_s(char c)
{
    return isalpha((unsigned char)c) || c == '_';
}

static int is_ident_c(char c)
{
    return isalnum((unsigned char)c) || c == '_';
}

static void pp_fail(PP *pp, const char *msg)
{
    if (pp->failed) {
        return;
    }
    pp->failed = 1;
    snprintf(pp->err, sizeof(pp->err), "%s", msg);
}

static char *xstrndup(const char *s, size_t n)
{
    char *p = (char *)malloc(n + 1);
    if (!p) {
        return NULL;
    }
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}

static char *xstrdup(const char *s)
{
    return xstrndup(s, strlen(s));
}

static void tok_free(Tok *t)
{
    free(t->s);
    t->s = NULL;
}

static void toklist_free(TokList *tl)
{
    size_t i;
    for (i = 0; i < tl->n; i++) {
        tok_free(&tl->t[i]);
    }
    free(tl->t);
    tl->t = NULL;
    tl->n = tl->cap = 0;
}

static int toklist_push(TokList *tl, int kind, const char *s)
{
    if (tl->n == tl->cap) {
        size_t nc = tl->cap ? tl->cap * 2 : 8;
        Tok *nt = (Tok *)realloc(tl->t, nc * sizeof(Tok));
        if (!nt) {
            return -1;
        }
        tl->t = nt;
        tl->cap = nc;
    }
    tl->t[tl->n].kind = kind;
    tl->t[tl->n].s = xstrdup(s);
    tl->t[tl->n].spaced = 0;
    if (!tl->t[tl->n].s) {
        return -1;
    }
    tl->n++;
    return 0;
}

static int toklist_push_dup(TokList *tl, const Tok *t)
{
    if (toklist_push(tl, t->kind, t->s) != 0) {
        return -1;
    }
    tl->t[tl->n - 1].spaced = t->spaced;
    return 0;
}

static void out_ch(PP *pp, char c)
{
    if (pp->olen + 2 >= pp->ocap) {
        size_t nc = pp->ocap ? pp->ocap * 2 : 256;
        char *n = (char *)realloc(pp->out, nc);
        if (!n) {
            pp_fail(pp, "preprocessor: out of memory");
            return;
        }
        pp->out = n;
        pp->ocap = nc;
    }
    pp->out[pp->olen++] = c;
}

static void out_str(PP *pp, const char *s)
{
    while (*s) {
        out_ch(pp, *s++);
    }
}

static void out_nls(PP *pp, int n)
{
    while (n-- > 0) {
        out_ch(pp, '\n');
    }
}

static int taking(const PP *pp)
{
    return pp->depth == 0 || pp->cond[pp->depth - 1].taking;
}

static char *collapse_continuations(const char *src, size_t len, size_t *out_len)
{
    char *o = (char *)malloc(len + 1);
    size_t n = 0;
    size_t i = 0;
    if (!o) {
        return NULL;
    }
    while (i < len) {
        if (src[i] == '\\' && i + 1 < len &&
            (src[i + 1] == '\n' ||
             (src[i + 1] == '\r' && i + 2 < len && src[i + 2] == '\n'))) {
            i += (src[i + 1] == '\r') ? 3 : 2;
            continue;
        }
        o[n++] = src[i++];
    }
    o[n] = 0;
    *out_len = n;
    return o;
}

/* Read until a newline outside comments.  Comments become a single space.
 * *phys gets the number of physical newlines consumed (including terminator). */
static char *read_logical_line(PP *pp, const char *src, size_t len, size_t *pos,
                               int *phys)
{
    size_t cap = 64, n = 0;
    char *o = (char *)malloc(cap);
    int in_block = 0;
    int nl = 0;
    if (!o) {
        return NULL;
    }
    while (*pos < len) {
        char c = src[*pos];
        if (in_block) {
            if (c == '*' && *pos + 1 < len && src[*pos + 1] == '/') {
                in_block = 0;
                *pos += 2;
                if (n + 2 >= cap) {
                    cap *= 2;
                    o = (char *)realloc(o, cap);
                    if (!o) {
                        return NULL;
                    }
                }
                o[n++] = ' ';
                continue;
            }
            if (c == '\n') {
                nl++;
            }
            (*pos)++;
            continue;
        }
        if (c == '/' && *pos + 1 < len && src[*pos + 1] == '/') {
            *pos += 2;
            while (*pos < len && src[*pos] != '\n') {
                (*pos)++;
            }
            continue;
        }
        if (c == '/' && *pos + 1 < len && src[*pos + 1] == '*') {
            in_block = 1;
            *pos += 2;
            continue;
        }
        if (c == '\n') {
            (*pos)++;
            nl++;
            break;
        }
        if (n + 2 >= cap) {
            cap *= 2;
            o = (char *)realloc(o, cap);
            if (!o) {
                return NULL;
            }
        }
        o[n++] = c;
        (*pos)++;
    }
    o[n] = 0;
    *phys = nl ? nl : 0;
    if (*pos >= len && nl == 0) {
        *phys = 0;
    }
    if (in_block) {
        pp_fail(pp, "preprocessor: unterminated comment");
        free(o);
        return NULL;
    }
    return o;
}

static int punct_at(const char *s, size_t n, size_t i, const char *op)
{
    size_t k = strlen(op);
    return i + k <= n && memcmp(s + i, op, k) == 0;
}

static void tokenize_text(const char *s, TokList *tl)
{
    size_t n = strlen(s), i = 0;
    static const char *const ops[] = {
        "##", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "++", "--",
        NULL
    };
    int spaced = 0;
    while (i < n) {
        if (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\v' ||
            s[i] == '\f') {
            spaced = 1;
            i++;
            continue;
        }
        if (is_ident_s(s[i])) {
            size_t j = i + 1;
            while (j < n && is_ident_c(s[j])) {
                j++;
            }
            char *id = xstrndup(s + i, j - i);
            if (id) {
                toklist_push(tl, TK_ID, id);
                if (tl->n) {
                    tl->t[tl->n - 1].spaced = spaced;
                }
                free(id);
            }
            spaced = 0;
            i = j;
            continue;
        }
        if (isdigit((unsigned char)s[i]) ||
            (s[i] == '.' && i + 1 < n && isdigit((unsigned char)s[i + 1]))) {
            size_t j = i;
            if (s[j] == '0' && j + 1 < n && (s[j + 1] == 'x' || s[j + 1] == 'X')) {
                j += 2;
                while (j < n && isxdigit((unsigned char)s[j])) {
                    j++;
                }
            } else {
                while (j < n && (isalnum((unsigned char)s[j]) || s[j] == '.' ||
                                 s[j] == '+' || s[j] == '-')) {
                    if ((s[j] == '+' || s[j] == '-') &&
                        !(j > i && (s[j - 1] == 'e' || s[j - 1] == 'E' ||
                                    s[j - 1] == 'p' || s[j - 1] == 'P'))) {
                        break;
                    }
                    j++;
                }
            }
            while (j < n && (s[j] == 'u' || s[j] == 'U' || s[j] == 'l' ||
                             s[j] == 'L')) {
                j++;
            }
            char *num = xstrndup(s + i, j - i);
            if (num) {
                toklist_push(tl, TK_NUM, num);
                if (tl->n) {
                    tl->t[tl->n - 1].spaced = spaced;
                }
                free(num);
            }
            spaced = 0;
            i = j;
            continue;
        }
        {
            int k;
            int hit = 0;
            for (k = 0; ops[k]; k++) {
                if (punct_at(s, n, i, ops[k])) {
                    toklist_push(tl, TK_PUNCT, ops[k]);
                    if (tl->n) {
                        tl->t[tl->n - 1].spaced = spaced;
                    }
                    spaced = 0;
                    i += strlen(ops[k]);
                    hit = 1;
                    break;
                }
            }
            if (hit) {
                continue;
            }
        }
        {
            char tmp[2] = { s[i], 0 };
            toklist_push(tl, TK_PUNCT, tmp);
            if (tl->n) {
                tl->t[tl->n - 1].spaced = spaced;
            }
            spaced = 0;
            i++;
        }
    }
}

static Macro *find_macro(PP *pp, const char *name)
{
    size_t i;
    for (i = 0; i < pp->nmac; i++) {
        if (strcmp(pp->macros[i].name, name) == 0) {
            return &pp->macros[i];
        }
    }
    return NULL;
}

static int hidden(char *const *hide, int nh, const char *name)
{
    int i;
    for (i = 0; i < nh; i++) {
        if (strcmp(hide[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

static int canon_repl(const TokList *tl, char *buf, size_t cap)
{
    size_t i, o = 0;
    if (!buf || cap == 0) {
        return -1;
    }
    buf[0] = 0;
    for (i = 0; i < tl->n; i++) {
        size_t k = strlen(tl->t[i].s);
        if (o + k + 2 >= cap) {
            /* Keep a terminated prefix so callers may still strcmp safely. */
            buf[o < cap ? o : cap - 1] = 0;
            return -1;
        }
        if (i && tl->t[i].spaced) {
            buf[o++] = ' ';
        }
        memcpy(buf + o, tl->t[i].s, k);
        o += k;
        buf[o] = 0;
    }
    return 0;
}

static void macro_clear(Macro *m)
{
    int i;
    free(m->name);
    for (i = 0; i < m->nparams; i++) {
        free(m->params[i]);
    }
    free(m->params);
    toklist_free(&m->body);
    memset(m, 0, sizeof(*m));
}

static Macro *macro_add(PP *pp, const char *name)
{
    if (pp->nmac == pp->capmac) {
        size_t nc = pp->capmac ? pp->capmac * 2 : 16;
        Macro *nm = (Macro *)realloc(pp->macros, nc * sizeof(Macro));
        if (!nm) {
            pp_fail(pp, "preprocessor: out of memory");
            return NULL;
        }
        pp->macros = nm;
        pp->capmac = nc;
    }
    memset(&pp->macros[pp->nmac], 0, sizeof(Macro));
    pp->macros[pp->nmac].name = xstrdup(name);
    if (!pp->macros[pp->nmac].name) {
        pp_fail(pp, "preprocessor: out of memory");
        return NULL;
    }
    pp->nmac++;
    return &pp->macros[pp->nmac - 1];
}

static int is_float_num(const char *s)
{
    return strchr(s, '.') != NULL || strchr(s, 'e') != NULL ||
           strchr(s, 'E') != NULL || strchr(s, 'p') != NULL ||
           strchr(s, 'P') != NULL;
}

static int parse_int_num(const char *s, long long *v)
{
    char *end = NULL;
    const char *p = s;
    int neg = 0;
    if (*p == '+' || *p == '-') {
        neg = (*p == '-');
        p++;
    }
    if (is_float_num(p)) {
        return -1;
    }
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        *v = strtoll(p, &end, 16);
    } else if (p[0] == '0' && isdigit((unsigned char)p[1])) {
        *v = strtoll(p, &end, 8);
    } else {
        *v = strtoll(p, &end, 10);
    }
    while (end && (*end == 'u' || *end == 'U' || *end == 'l' || *end == 'L')) {
        end++;
    }
    if (!end || *end != 0) {
        return -1;
    }
    if (neg) {
        *v = -*v;
    }
    return 0;
}

static int expand_list(PP *pp, const TokList *in, char **hide, int nh,
                       TokList *out, long line);

static int collect_args(const TokList *in, size_t *i, TokList **args, int *na,
                        int nparams)
{
    int depth = 1;
    TokList cur;
    memset(&cur, 0, sizeof(cur));
    *args = NULL;
    *na = 0;
    (*i)++; /* skip '(' */
    if (*i < in->n && in->t[*i].kind == TK_PUNCT &&
        strcmp(in->t[*i].s, ")") == 0) {
        (*i)++;
        /* F() with one parameter is a single empty argument. */
        if (nparams == 1) {
            TokList *naa = (TokList *)calloc(1, sizeof(TokList));
            if (!naa) {
                return -1;
            }
            *args = naa;
            *na = 1;
        }
        return 0;
    }
    while (*i < in->n) {
        Tok *t = &in->t[*i];
        if (t->kind == TK_PUNCT && strcmp(t->s, "(") == 0) {
            depth++;
            toklist_push_dup(&cur, t);
        } else if (t->kind == TK_PUNCT && strcmp(t->s, ")") == 0) {
            depth--;
            if (depth == 0) {
                TokList *naa =
                    (TokList *)realloc(*args, (size_t)(*na + 1) * sizeof(TokList));
                if (!naa) {
                    toklist_free(&cur);
                    return -1;
                }
                *args = naa;
                (*args)[*na] = cur;
                (*na)++;
                memset(&cur, 0, sizeof(cur));
                (*i)++;
                return 0;
            }
            toklist_push_dup(&cur, t);
        } else if (t->kind == TK_PUNCT && strcmp(t->s, ",") == 0 && depth == 1) {
            TokList *naa =
                (TokList *)realloc(*args, (size_t)(*na + 1) * sizeof(TokList));
            if (!naa) {
                toklist_free(&cur);
                return -1;
            }
            *args = naa;
            (*args)[*na] = cur;
            (*na)++;
            memset(&cur, 0, sizeof(cur));
        } else {
            toklist_push_dup(&cur, t);
        }
        (*i)++;
    }
    toklist_free(&cur);
    return -1;
}

static int paste_list(PP *pp, TokList *tl)
{
    size_t i = 0;
    while (i < tl->n) {
        if (tl->t[i].kind == TK_PUNCT && strcmp(tl->t[i].s, "##") == 0) {
            if (i == 0 || i + 1 >= tl->n) {
                pp_fail(pp, "preprocessor: ## missing operand");
                return -1;
            }
            {
                size_t la = strlen(tl->t[i - 1].s);
                size_t lb = strlen(tl->t[i + 1].s);
                char *cat = (char *)malloc(la + lb + 1);
                Tok nt;
                size_t k;
                if (!cat) {
                    pp_fail(pp, "preprocessor: out of memory");
                    return -1;
                }
                memcpy(cat, tl->t[i - 1].s, la);
                memcpy(cat + la, tl->t[i + 1].s, lb);
                cat[la + lb] = 0;
                nt.s = cat;
                if (is_ident_s(cat[0])) {
                    nt.kind = TK_ID;
                } else if (isdigit((unsigned char)cat[0])) {
                    nt.kind = TK_NUM;
                } else {
                    nt.kind = TK_PUNCT;
                }
                tok_free(&tl->t[i - 1]);
                tok_free(&tl->t[i]);
                tok_free(&tl->t[i + 1]);
                for (k = i + 2; k < tl->n; k++) {
                    tl->t[k - 2] = tl->t[k];
                }
                tl->n -= 2;
                tl->t[i - 1] = nt;
                i = i - 1;
                continue;
            }
        }
        i++;
    }
    return 0;
}

static int expand_macro(PP *pp, Macro *m, TokList *args, int nargs,
                        char **hide, int nh, TokList *out, long line)
{
    TokList subst;
    int i, j;
    char *nhide[HIDE_MAX];
    int nnh;
    int rc;
    memset(&subst, 0, sizeof(subst));
    if (pp->expand_depth >= EXPAND_MAX) {
        pp_fail(pp, "preprocessor: macro expansion nesting too deep");
        return -1;
    }
    pp->expand_depth++;
    if (m->special == 1) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%ld", line);
        rc = toklist_push(out, TK_NUM, buf);
        pp->expand_depth--;
        return rc;
    }
    if (m->special == 2) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", pp->file_no);
        rc = toklist_push(out, TK_NUM, buf);
        pp->expand_depth--;
        return rc;
    }
    if (m->special == 3) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", pp->version);
        rc = toklist_push(out, TK_NUM, buf);
        pp->expand_depth--;
        return rc;
    }
    if (m->function_like && nargs != m->nparams) {
        pp_fail(pp, "preprocessor: macro argument count mismatch");
        pp->expand_depth--;
        return -1;
    }
    for (i = 0; i < (int)m->body.n; i++) {
        Tok *t = &m->body.t[i];
        int param = -1;
        int adj_paste = 0;
        if (t->kind == TK_ID) {
            for (j = 0; j < m->nparams; j++) {
                if (strcmp(t->s, m->params[j]) == 0) {
                    param = j;
                    break;
                }
            }
        }
        if (i > 0 && m->body.t[i - 1].kind == TK_PUNCT &&
            strcmp(m->body.t[i - 1].s, "##") == 0) {
            adj_paste = 1;
        }
        if (i + 1 < (int)m->body.n && m->body.t[i + 1].kind == TK_PUNCT &&
            strcmp(m->body.t[i + 1].s, "##") == 0) {
            adj_paste = 1;
        }
        if (t->kind == TK_PUNCT && strcmp(t->s, "#") == 0) {
            pp_fail(pp, "preprocessor: stringification is not supported");
            toklist_free(&subst);
            pp->expand_depth--;
            return -1;
        }
        if (param >= 0) {
            if (adj_paste) {
                size_t k;
                if (args[param].n == 0) {
                    toklist_push(&subst, TK_ID, "");
                } else {
                    for (k = 0; k < args[param].n; k++) {
                        toklist_push_dup(&subst, &args[param].t[k]);
                    }
                }
            } else {
                TokList exp;
                memset(&exp, 0, sizeof(exp));
                if (expand_list(pp, &args[param], hide, nh, &exp, line) != 0) {
                    toklist_free(&exp);
                    toklist_free(&subst);
                    pp->expand_depth--;
                    return -1;
                }
                {
                    size_t k;
                    for (k = 0; k < exp.n; k++) {
                        toklist_push_dup(&subst, &exp.t[k]);
                    }
                }
                toklist_free(&exp);
            }
        } else {
            toklist_push_dup(&subst, t);
        }
    }
    if (paste_list(pp, &subst) != 0) {
        toklist_free(&subst);
        pp->expand_depth--;
        return -1;
    }
    nnh = nh;
    for (i = 0; i < nh; i++) {
        nhide[i] = hide[i];
    }
    if (nnh >= HIDE_MAX) {
        /* Hide-set overflow would drop this macro from the hide set and allow
         * self-referential macros to recurse without bound (C1). */
        pp_fail(pp, "preprocessor: macro hide-set overflow");
        toklist_free(&subst);
        pp->expand_depth--;
        return -1;
    }
    nhide[nnh++] = m->name;
    rc = expand_list(pp, &subst, nhide, nnh, out, line);
    toklist_free(&subst);
    pp->expand_depth--;
    return rc;
}

static int expand_list(PP *pp, const TokList *in, char **hide, int nh,
                       TokList *out, long line)
{
    size_t i = 0;
    int rc = 0;
    if (pp->expand_depth >= EXPAND_MAX) {
        pp_fail(pp, "preprocessor: macro expansion nesting too deep");
        return -1;
    }
    pp->expand_depth++;
    while (i < in->n) {
        Tok *t = &in->t[i];
        Macro *m;
        if (pp->failed) {
            rc = -1;
            break;
        }
        if (pp->if_mode && t->kind == TK_ID && strcmp(t->s, "defined") == 0) {
            const char *id = NULL;
            int on;
            i++;
            if (i < in->n && in->t[i].kind == TK_PUNCT &&
                strcmp(in->t[i].s, "(") == 0) {
                i++;
                if (i >= in->n || in->t[i].kind != TK_ID) {
                    pp_fail(pp, "preprocessor: defined() expects identifier");
                    rc = -1;
                    break;
                }
                id = in->t[i].s;
                i++;
                if (i >= in->n || in->t[i].kind != TK_PUNCT ||
                    strcmp(in->t[i].s, ")") != 0) {
                    pp_fail(pp, "preprocessor: defined() missing ')'");
                    rc = -1;
                    break;
                }
                i++;
            } else if (i < in->n && in->t[i].kind == TK_ID) {
                id = in->t[i].s;
                i++;
            } else {
                pp_fail(pp, "preprocessor: defined expects identifier");
                rc = -1;
                break;
            }
            on = find_macro(pp, id) != NULL;
            if (toklist_push(out, TK_NUM, on ? "1" : "0") != 0) {
                rc = -1;
                break;
            }
            continue;
        }
        if (t->kind != TK_ID || hidden(hide, nh, t->s) ||
            (m = find_macro(pp, t->s)) == NULL) {
            if (toklist_push_dup(out, t) != 0) {
                rc = -1;
                break;
            }
            i++;
            continue;
        }
        if (m->function_like) {
            size_t j = i + 1;
            TokList *args = NULL;
            int nargs = 0;
            int k;
            if (j >= in->n || in->t[j].kind != TK_PUNCT ||
                strcmp(in->t[j].s, "(") != 0) {
                if (toklist_push_dup(out, t) != 0) {
                    rc = -1;
                    break;
                }
                i++;
                continue;
            }
            i = j;
            if (collect_args(in, &i, &args, &nargs, m->nparams) != 0) {
                pp_fail(pp, "preprocessor: unterminated macro invocation");
                rc = -1;
                break;
            }
            if (expand_macro(pp, m, args, nargs, hide, nh, out, line) != 0) {
                for (k = 0; k < nargs; k++) {
                    toklist_free(&args[k]);
                }
                free(args);
                rc = -1;
                break;
            }
            for (k = 0; k < nargs; k++) {
                toklist_free(&args[k]);
            }
            free(args);
            continue;
        }
        if (expand_macro(pp, m, NULL, 0, hide, nh, out, line) != 0) {
            rc = -1;
            break;
        }
        i++;
    }
    pp->expand_depth--;
    return rc;
}

typedef struct {
    const TokList *tl;
    size_t i;
    PP *pp;
    int eval;
} Ex;

static int expr_parse(Ex *e, long long *v);

static int expr_peek_op(Ex *e, const char *op)
{
    if (e->i >= e->tl->n) {
        return 0;
    }
    return e->tl->t[e->i].kind == TK_PUNCT && strcmp(e->tl->t[e->i].s, op) == 0;
}

static int expr_unary(Ex *e, long long *v)
{
    if (expr_peek_op(e, "+")) {
        e->i++;
        return expr_unary(e, v);
    }
    if (expr_peek_op(e, "-")) {
        e->i++;
        if (expr_unary(e, v) != 0) {
            return -1;
        }
        *v = -*v;
        return 0;
    }
    if (expr_peek_op(e, "~")) {
        e->i++;
        if (expr_unary(e, v) != 0) {
            return -1;
        }
        *v = ~*v;
        return 0;
    }
    if (expr_peek_op(e, "!")) {
        e->i++;
        if (expr_unary(e, v) != 0) {
            return -1;
        }
        *v = !*v;
        return 0;
    }
    if (expr_peek_op(e, "(")) {
        e->i++;
        if (expr_parse(e, v) != 0) {
            return -1;
        }
        if (!expr_peek_op(e, ")")) {
            pp_fail(e->pp, "preprocessor: missing ')' in #if");
            return -1;
        }
        e->i++;
        return 0;
    }
    if (e->i >= e->tl->n) {
        pp_fail(e->pp, "preprocessor: empty #if expression");
        return -1;
    }
    if (e->tl->t[e->i].kind == TK_NUM) {
        if (parse_int_num(e->tl->t[e->i].s, v) != 0) {
            pp_fail(e->pp, "preprocessor: #if requires integer constants");
            return -1;
        }
        e->i++;
        return 0;
    }
    if (e->tl->t[e->i].kind == TK_ID) {
        if (e->eval) {
            pp_fail(e->pp, "preprocessor: undefined identifier in #if");
            return -1;
        }
        *v = 0;
        e->i++;
        return 0;
    }
    pp_fail(e->pp, "preprocessor: invalid #if token");
    return -1;
}

static int expr_binop(Ex *e, long long *v, int prec);

static int op_prec(const char *op)
{
    if (!strcmp(op, "*") || !strcmp(op, "/") || !strcmp(op, "%")) {
        return 3;
    }
    if (!strcmp(op, "+") || !strcmp(op, "-")) {
        return 4;
    }
    if (!strcmp(op, "<<") || !strcmp(op, ">>")) {
        return 5;
    }
    if (!strcmp(op, "<") || !strcmp(op, ">") || !strcmp(op, "<=") ||
        !strcmp(op, ">=")) {
        return 6;
    }
    if (!strcmp(op, "==") || !strcmp(op, "!=")) {
        return 7;
    }
    if (!strcmp(op, "&")) {
        return 8;
    }
    if (!strcmp(op, "^")) {
        return 9;
    }
    if (!strcmp(op, "|")) {
        return 10;
    }
    if (!strcmp(op, "&&")) {
        return 11;
    }
    if (!strcmp(op, "||")) {
        return 12;
    }
    return 0;
}

static int apply_op(PP *pp, const char *op, long long a, long long b,
                    long long *r)
{
    if ((!strcmp(op, "/") || !strcmp(op, "%")) && b == 0) {
        pp_fail(pp, "preprocessor: division by zero in #if");
        return -1;
    }
    if (!strcmp(op, "*")) {
        *r = a * b;
    } else if (!strcmp(op, "/")) {
        *r = a / b;
    } else if (!strcmp(op, "%")) {
        *r = a % b;
    } else if (!strcmp(op, "+")) {
        *r = a + b;
    } else if (!strcmp(op, "-")) {
        *r = a - b;
    } else if (!strcmp(op, "<<")) {
        *r = a << (b & 63);
    } else if (!strcmp(op, ">>")) {
        *r = a >> (b & 63);
    } else if (!strcmp(op, "<")) {
        *r = a < b;
    } else if (!strcmp(op, ">")) {
        *r = a > b;
    } else if (!strcmp(op, "<=")) {
        *r = a <= b;
    } else if (!strcmp(op, ">=")) {
        *r = a >= b;
    } else if (!strcmp(op, "==")) {
        *r = a == b;
    } else if (!strcmp(op, "!=")) {
        *r = a != b;
    } else if (!strcmp(op, "&")) {
        *r = a & b;
    } else if (!strcmp(op, "^")) {
        *r = a ^ b;
    } else if (!strcmp(op, "|")) {
        *r = a | b;
    } else if (!strcmp(op, "&&")) {
        *r = (a != 0) && (b != 0);
    } else if (!strcmp(op, "||")) {
        *r = (a != 0) || (b != 0);
    } else {
        return -1;
    }
    return 0;
}

static int expr_binop(Ex *e, long long *v, int min_prec)
{
    if (expr_unary(e, v) != 0) {
        return -1;
    }
    for (;;) {
        int p;
        long long rhs, r;
        const char *op;
        if (e->i >= e->tl->n || e->tl->t[e->i].kind != TK_PUNCT) {
            return 0;
        }
        op = e->tl->t[e->i].s;
        p = op_prec(op);
        /* op_prec uses lower numbers for tighter binding (GLSL 4.60 §3.3). */
        if (p == 0 || p >= min_prec) {
            return 0;
        }
        e->i++;
        {
            int saved = e->eval;
            if (e->eval && !strcmp(op, "&&") && *v == 0) {
                e->eval = 0;
            } else if (e->eval && !strcmp(op, "||") && *v != 0) {
                e->eval = 0;
            }
            if (expr_binop(e, &rhs, p) != 0) {
                return -1;
            }
            e->eval = saved;
        }
        if (!e->eval) {
            *v = 0;
            continue;
        }
        if (apply_op(e->pp, op, *v, rhs, &r) != 0) {
            return -1;
        }
        *v = r;
    }
}

static int expr_parse(Ex *e, long long *v)
{
    return expr_binop(e, v, 13);
}

static int eval_if_expr(PP *pp, const TokList *raw, long long *v, long line)
{
    TokList t, exp;
    size_t i;
    Ex e;
    char *hide[1];
    memset(&t, 0, sizeof(t));
    memset(&exp, 0, sizeof(exp));
    for (i = 0; i < raw->n; i++) {
        toklist_push_dup(&t, &raw->t[i]);
    }
    hide[0] = NULL;
    pp->if_mode = 1;
    if (expand_list(pp, &t, hide, 0, &exp, line) != 0) {
        pp->if_mode = 0;
        toklist_free(&t);
        toklist_free(&exp);
        return -1;
    }
    pp->if_mode = 0;
    toklist_free(&t);
    {
        TokList z;
        size_t j;
        memset(&z, 0, sizeof(z));
        for (j = 0; j < exp.n; j++) {
            if (exp.t[j].kind == TK_ID) {
                toklist_push(&z, TK_NUM, "0");
            } else {
                toklist_push_dup(&z, &exp.t[j]);
            }
        }
        toklist_free(&exp);
        exp = z;
    }
    if (exp.n == 0) {
        pp_fail(pp, "preprocessor: empty #if expression");
        toklist_free(&exp);
        return -1;
    }
    e.tl = &exp;
    e.i = 0;
    e.pp = pp;
    e.eval = 1;
    if (expr_parse(&e, v) != 0) {
        toklist_free(&exp);
        return -1;
    }
    if (e.i != exp.n) {
        pp_fail(pp, "preprocessor: extra tokens in #if");
        toklist_free(&exp);
        return -1;
    }
    toklist_free(&exp);
    return 0;
}

static int ident_eq(const Tok *t, const char *s)
{
    return t->kind == TK_ID && strcmp(t->s, s) == 0;
}

static int require_ident(PP *pp, const TokList *tl, const char **name)
{
    if (tl->n < 1 || tl->t[0].kind != TK_ID) {
        pp_fail(pp, "preprocessor: expected identifier");
        return -1;
    }
    *name = tl->t[0].s;
    if (tl->n > 1) {
        pp_fail(pp, "preprocessor: extra tokens after identifier");
        return -1;
    }
    return 0;
}

static int push_if(PP *pp, int cond)
{
    int parent = taking(pp);
    if (pp->depth >= COND_MAX) {
        pp_fail(pp, "preprocessor: #if nesting too deep");
        return -1;
    }
    pp->cond[pp->depth].parent = parent;
    pp->cond[pp->depth].taking = parent && cond;
    pp->cond[pp->depth].else_seen = 0;
    pp->cond[pp->depth].taken = parent && cond;
    pp->depth++;
    return 0;
}

static int handle_else(PP *pp)
{
    Cond *c;
    if (pp->depth == 0) {
        pp_fail(pp, "preprocessor: #else without #if");
        return -1;
    }
    c = &pp->cond[pp->depth - 1];
    if (c->else_seen) {
        pp_fail(pp, "preprocessor: #else after #else");
        return -1;
    }
    c->else_seen = 1;
    c->taking = c->parent && !c->taken;
    if (c->taking) {
        c->taken = 1;
    }
    return 0;
}

static int handle_elif(PP *pp, const TokList *expr, long line)
{
    Cond *c;
    long long v = 0;
    if (pp->depth == 0) {
        pp_fail(pp, "preprocessor: #elif without #if");
        return -1;
    }
    c = &pp->cond[pp->depth - 1];
    if (c->else_seen) {
        pp_fail(pp, "preprocessor: #elif after #else");
        return -1;
    }
    if (!c->parent) {
        c->taking = 0;
        return 0;
    }
    if (c->taken) {
        c->taking = 0;
        return 0;
    }
    if (eval_if_expr(pp, expr, &v, line) != 0) {
        return -1;
    }
    c->taking = v != 0;
    if (c->taking) {
        c->taken = 1;
    }
    return 0;
}

static int handle_endif(PP *pp)
{
    if (pp->depth == 0) {
        pp_fail(pp, "preprocessor: #endif without #if");
        return -1;
    }
    pp->depth--;
    return 0;
}

static int skip_ws_text(const char *s, size_t n, size_t i)
{
    while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r')) {
        i++;
    }
    return (int)i;
}

static int parse_directive_name(const char *line, size_t *name_off,
                                size_t *name_len, size_t *rest_off)
{
    size_t n = strlen(line);
    size_t i = (size_t)skip_ws_text(line, n, 0);
    if (i >= n || line[i] != '#') {
        return 0;
    }
    i = (size_t)skip_ws_text(line, n, i + 1);
    *name_off = i;
    while (i < n && is_ident_c(line[i])) {
        i++;
    }
    *name_len = i - *name_off;
    *rest_off = (size_t)skip_ws_text(line, n, i);
    return 1;
}

static int add_predef(PP *pp, const char *name, int special, const char *val,
                      int reserved_gl)
{
    Macro *m = macro_add(pp, name);
    if (!m) {
        return -1;
    }
    m->special = special;
    m->reserved_gl = reserved_gl;
    if (val && toklist_push(&m->body, TK_NUM, val) != 0) {
        return -1;
    }
    return 0;
}

static int define_from_raw(PP *pp, const char *rest)
{
    size_t n = strlen(rest), i = 0;
    size_t name_s, name_e;
    int function_like = 0;
    Macro tmp, *exist, *m;
    char canon_new[1024], canon_old[1024];
    memset(&tmp, 0, sizeof(tmp));
    i = (size_t)skip_ws_text(rest, n, 0);
    if (i >= n || !is_ident_s(rest[i])) {
        pp_fail(pp, "preprocessor: #define requires an identifier");
        return -1;
    }
    name_s = i;
    i++;
    while (i < n && is_ident_c(rest[i])) {
        i++;
    }
    name_e = i;
    tmp.name = xstrndup(rest + name_s, name_e - name_s);
    if (!tmp.name) {
        pp_fail(pp, "preprocessor: out of memory");
        return -1;
    }
    if (!strncmp(tmp.name, "GL_", 3)) {
        pp_fail(pp, "preprocessor: cannot define reserved GL_ name");
        free(tmp.name);
        return -1;
    }
    if (i < n && rest[i] == '(') {
        function_like = 1;
        i++;
        tmp.function_like = 1;
        if (i < n && rest[i] == ')') {
            i++;
        } else {
            while (i < n) {
                size_t ps, pe;
                i = (size_t)skip_ws_text(rest, n, i);
                if (i < n && rest[i] == ')') {
                    i++;
                    break;
                }
                if (i >= n || !is_ident_s(rest[i])) {
                    pp_fail(pp, "preprocessor: expected macro parameter");
                    macro_clear(&tmp);
                    return -1;
                }
                ps = i;
                i++;
                while (i < n && is_ident_c(rest[i])) {
                    i++;
                }
                pe = i;
                {
                    char **np = (char **)realloc(
                        tmp.params, (size_t)(tmp.nparams + 1) * sizeof(char *));
                    if (!np) {
                        macro_clear(&tmp);
                        pp_fail(pp, "preprocessor: out of memory");
                        return -1;
                    }
                    tmp.params = np;
                    tmp.params[tmp.nparams] = xstrndup(rest + ps, pe - ps);
                    {
                        int di;
                        for (di = 0; di < tmp.nparams; di++) {
                            if (strcmp(tmp.params[di],
                                       tmp.params[tmp.nparams]) == 0) {
                                pp_fail(pp, "preprocessor: duplicate macro parameter");
                                tmp.nparams++;
                                macro_clear(&tmp);
                                return -1;
                            }
                        }
                    }
                    tmp.nparams++;
                }
                i = (size_t)skip_ws_text(rest, n, i);
                if (i < n && rest[i] == ',') {
                    i++;
                    continue;
                }
                if (i < n && rest[i] == ')') {
                    i++;
                    break;
                }
                pp_fail(pp, "preprocessor: invalid macro parameter list");
                macro_clear(&tmp);
                return -1;
            }
        }
    }
    i = (size_t)skip_ws_text(rest, n, i);
    tokenize_text(rest + i, &tmp.body);
    if (tmp.body.n && tmp.body.t[0].kind == TK_PUNCT &&
        strcmp(tmp.body.t[0].s, "##") == 0) {
        pp_fail(pp, "preprocessor: ## at start of replacement");
        macro_clear(&tmp);
        return -1;
    }
    if (tmp.body.n && tmp.body.t[tmp.body.n - 1].kind == TK_PUNCT &&
        strcmp(tmp.body.t[tmp.body.n - 1].s, "##") == 0) {
        pp_fail(pp, "preprocessor: ## at end of replacement");
        macro_clear(&tmp);
        return -1;
    }
    {
        size_t bi;
        for (bi = 0; bi < tmp.body.n; bi++) {
            if (tmp.body.t[bi].kind == TK_PUNCT &&
                strcmp(tmp.body.t[bi].s, "#") == 0) {
                pp_fail(pp, "preprocessor: stringification is not supported");
                macro_clear(&tmp);
                return -1;
            }
        }
    }
    exist = find_macro(pp, tmp.name);
    if (exist && exist->reserved_gl) {
        pp_fail(pp, "preprocessor: cannot define reserved GL_ name");
        macro_clear(&tmp);
        return -1;
    }
    if (canon_repl(&tmp.body, canon_new, sizeof(canon_new)) != 0) {
        pp_fail(pp, "preprocessor: macro replacement too long");
        macro_clear(&tmp);
        return -1;
    }
    if (exist) {
        if (canon_repl(&exist->body, canon_old, sizeof(canon_old)) != 0) {
            pp_fail(pp, "preprocessor: macro replacement too long");
            macro_clear(&tmp);
            return -1;
        }
        if (exist->function_like != tmp.function_like ||
            exist->nparams != tmp.nparams || strcmp(canon_new, canon_old) != 0) {
            pp_fail(pp, "preprocessor: illegal macro redefinition");
            macro_clear(&tmp);
            return -1;
        }
        if (exist->function_like) {
            int pi;
            for (pi = 0; pi < exist->nparams; pi++) {
                if (strcmp(exist->params[pi], tmp.params[pi]) != 0) {
                    pp_fail(pp, "preprocessor: illegal macro redefinition");
                    macro_clear(&tmp);
                    return -1;
                }
            }
        }
        /* identical redefinition is allowed */
        macro_clear(&tmp);
        return 0;
    }
    m = macro_add(pp, tmp.name);
    if (!m) {
        macro_clear(&tmp);
        return -1;
    }
    free(m->name);
    *m = tmp;
    (void)function_like;
    return 0;
}

static int handle_undef(PP *pp, const TokList *tl)
{
    const char *name;
    Macro *m;
    size_t idx;
    if (require_ident(pp, tl, &name) != 0) {
        return -1;
    }
    if (!strncmp(name, "GL_", 3)) {
        pp_fail(pp, "preprocessor: cannot undefine reserved GL_ name");
        return -1;
    }
    m = find_macro(pp, name);
    if (!m) {
        return 0;
    }
    idx = (size_t)(m - pp->macros);
    macro_clear(m);
    if (idx + 1 < pp->nmac) {
        memmove(&pp->macros[idx], &pp->macros[idx + 1],
                (pp->nmac - idx - 1) * sizeof(Macro));
    }
    pp->nmac--;
    return 0;
}

static int version_supported(int v)
{
    switch (v) {
    case 100:
    case 110:
    case 120:
    case 130:
    case 140:
    case 150:
    case 300:
    case 310:
    case 320:
    case 330:
    case 400:
    case 410:
    case 420:
    case 430:
    case 440:
    case 450:
    case 460:
        return 1;
    default:
        return 0;
    }
}

static int handle_version(PP *pp, const TokList *tl, const char *raw_line)
{
    long long v = 0;
    const char *prof = NULL;
    if (pp->saw_content) {
        pp_fail(pp, "preprocessor: #version must be first");
        return -1;
    }
    if (tl->n < 1 || tl->t[0].kind != TK_NUM ||
        parse_int_num(tl->t[0].s, &v) != 0) {
        pp_fail(pp, "preprocessor: invalid #version");
        return -1;
    }
    if (!version_supported((int)v)) {
        pp_fail(pp, "preprocessor: unsupported #version");
        return -1;
    }
    if (tl->n >= 2) {
        if (tl->t[1].kind != TK_ID) {
            pp_fail(pp, "preprocessor: invalid #version profile");
            return -1;
        }
        prof = tl->t[1].s;
        if (strcmp(prof, "core") && strcmp(prof, "compatibility") &&
            strcmp(prof, "es")) {
            pp_fail(pp, "preprocessor: invalid #version profile");
            return -1;
        }
    }
    if (tl->n > 2) {
        pp_fail(pp, "preprocessor: extra tokens after #version");
        return -1;
    }
    pp->version = (int)v;
    pp->saw_content = 1;
    out_str(pp, raw_line);
    out_ch(pp, '\n');
    return 0;
}

static int handle_extension(PP *pp, const TokList *tl, const char *raw_line)
{
    const char *beh;
    if (tl->n < 3) {
        pp_fail(pp, "preprocessor: invalid #extension");
        return -1;
    }
    if (tl->t[0].kind != TK_ID) {
        pp_fail(pp, "preprocessor: invalid #extension name");
        return -1;
    }
    if (tl->t[1].kind != TK_PUNCT || strcmp(tl->t[1].s, ":") != 0) {
        pp_fail(pp, "preprocessor: #extension expected ':'");
        return -1;
    }
    if (tl->t[2].kind != TK_ID) {
        pp_fail(pp, "preprocessor: invalid #extension behavior");
        return -1;
    }
    beh = tl->t[2].s;
    if (strcmp(beh, "require") && strcmp(beh, "enable") && strcmp(beh, "warn") &&
        strcmp(beh, "disable")) {
        pp_fail(pp, "preprocessor: invalid #extension behavior");
        return -1;
    }
    if (ident_eq(&tl->t[0], "all") &&
        (strcmp(beh, "require") == 0 || strcmp(beh, "enable") == 0)) {
        pp_fail(pp, "preprocessor: #extension all : require/enable is illegal");
        return -1;
    }
    if (tl->n > 3) {
        pp_fail(pp, "preprocessor: extra tokens after #extension");
        return -1;
    }
    /* GLSL 4.60 §3.3: enabling an extension defines a macro with the
     * extension's name (value 1).  CTS and apps probe this with #ifndef. */
    if (!ident_eq(&tl->t[0], "all") &&
        (strcmp(beh, "require") == 0 || strcmp(beh, "enable") == 0 ||
         strcmp(beh, "warn") == 0)) {
        if (add_predef(pp, tl->t[0].s, 0, "1", 0) != 0)
            return -1;
    }
    out_str(pp, raw_line);
    out_ch(pp, '\n');
    return 0;
}

static int handle_line_dir(PP *pp, const TokList *tl, const char *raw_line)
{
    long long ln = 0, fn = 0;
    TokList exp;
    char *hide[1];
    memset(&exp, 0, sizeof(exp));
    hide[0] = NULL;
    if (expand_list(pp, tl, hide, 0, &exp, pp->line) != 0) {
        toklist_free(&exp);
        return -1;
    }
    if (exp.n < 1 || parse_int_num(exp.t[0].s, &ln) != 0) {
        pp_fail(pp, "preprocessor: invalid #line");
        toklist_free(&exp);
        return -1;
    }
    if (exp.n >= 2) {
        if (parse_int_num(exp.t[1].s, &fn) != 0) {
            pp_fail(pp, "preprocessor: invalid #line file");
            toklist_free(&exp);
            return -1;
        }
        pp->file_no = (int)fn;
    }
    if (exp.n > 2) {
        pp_fail(pp, "preprocessor: extra tokens after #line");
        toklist_free(&exp);
        return -1;
    }
    /* Next physical line will be numbered ln.  We increment after the
     * directive, so set line so that after +phys it becomes ln. */
    pp->line = ln - 1;
    toklist_free(&exp);
    out_nls(pp, 1);
    (void)raw_line;
    return 0;
}

static void emit_expanded_line(PP *pp, const char *line, long lineno)
{
    TokList raw, exp;
    size_t i;
    char *hide[1];
    memset(&raw, 0, sizeof(raw));
    memset(&exp, 0, sizeof(exp));
    tokenize_text(line, &raw);
    for (i = 0; i < raw.n; i++) {
        if (raw.t[i].kind == TK_PUNCT && strcmp(raw.t[i].s, "#") == 0) {
            pp_fail(pp, "preprocessor: '#' is not a stringification operator");
            toklist_free(&raw);
            return;
        }
    }
    hide[0] = NULL;
    if (expand_list(pp, &raw, hide, 0, &exp, lineno) != 0) {
        toklist_free(&raw);
        toklist_free(&exp);
        return;
    }
    for (i = 0; i < exp.n; i++) {
        if (i) {
            out_ch(pp, ' ');
        }
        out_str(pp, exp.t[i].s);
    }
    out_ch(pp, '\n');
    toklist_free(&raw);
    toklist_free(&exp);
}

static int process_directive(PP *pp, const char *line, int phys)
{
    size_t name_off = 0, name_len = 0, rest_off = 0;
    char name[32];
    const char *rest;
    TokList args;
    int is_dir;
    long lineno = pp->line;
    memset(&args, 0, sizeof(args));
    is_dir = parse_directive_name(line, &name_off, &name_len, &rest_off);
    if (!is_dir) {
        if (taking(pp)) {
            size_t k = 0;
            while (line[k] == ' ' || line[k] == '\t' || line[k] == '\r') {
                k++;
            }
            if (line[k]) {
                pp->saw_content = 1;
            }
            emit_expanded_line(pp, line, lineno);
            if (phys > 1) {
                out_nls(pp, phys - 1);
            }
        } else {
            out_nls(pp, phys ? phys : 1);
        }
        return 0;
    }
    if (name_len == 0) {
        /* null directive */
        out_nls(pp, phys ? phys : 1);
        return 0;
    }
    if (name_len >= sizeof(name)) {
        pp_fail(pp, "preprocessor: unknown directive");
        return -1;
    }
    memcpy(name, line + name_off, name_len);
    name[name_len] = 0;
    rest = line + rest_off;
    tokenize_text(rest, &args);
    if (strcmp(name, "version") != 0) {
        pp->saw_content = 1;
    }

    if (!strcmp(name, "if") || !strcmp(name, "ifdef") ||
        !strcmp(name, "ifndef") || !strcmp(name, "else") ||
        !strcmp(name, "elif") || !strcmp(name, "endif")) {
        int rc = 0;
        if (!strcmp(name, "if")) {
            long long v = 0;
            if (!taking(pp)) {
                rc = push_if(pp, 0);
            } else {
                rc = eval_if_expr(pp, &args, &v, lineno);
                if (rc == 0) {
                    rc = push_if(pp, v != 0);
                }
            }
        } else if (!strcmp(name, "ifdef") || !strcmp(name, "ifndef")) {
            const char *id;
            int on = 0;
            if (taking(pp)) {
                if (require_ident(pp, &args, &id) != 0) {
                    toklist_free(&args);
                    return -1;
                }
                on = find_macro(pp, id) != NULL;
                if (!strcmp(name, "ifndef")) {
                    on = !on;
                }
            }
            rc = push_if(pp, on);
        } else if (!strcmp(name, "else")) {
            if (args.n) {
                pp_fail(pp, "preprocessor: extra tokens after #else");
                toklist_free(&args);
                return -1;
            }
            rc = handle_else(pp);
        } else if (!strcmp(name, "elif")) {
            rc = handle_elif(pp, &args, lineno);
        } else {
            if (args.n) {
                pp_fail(pp, "preprocessor: extra tokens after #endif");
                toklist_free(&args);
                return -1;
            }
            rc = handle_endif(pp);
        }
        toklist_free(&args);
        out_nls(pp, phys ? phys : 1);
        return rc;
    }

    if (!taking(pp)) {
        toklist_free(&args);
        out_nls(pp, phys ? phys : 1);
        return 0;
    }

    if (!strcmp(name, "define")) {
        int rc = define_from_raw(pp, rest);
        toklist_free(&args);
        out_nls(pp, phys ? phys : 1);
        return rc;
    }
    if (!strcmp(name, "undef")) {
        int rc = handle_undef(pp, &args);
        toklist_free(&args);
        out_nls(pp, phys ? phys : 1);
        return rc;
    }
    if (!strcmp(name, "error")) {
        pp_fail(pp, rest[0] ? rest : "#error");
        toklist_free(&args);
        return -1;
    }
    if (!strcmp(name, "pragma")) {
        out_str(pp, line);
        out_ch(pp, '\n');
        toklist_free(&args);
        return 0;
    }
    if (!strcmp(name, "extension")) {
        int rc = handle_extension(pp, &args, line);
        toklist_free(&args);
        if (phys > 1) {
            out_nls(pp, phys - 1);
        }
        return rc;
    }
    if (!strcmp(name, "version")) {
        int rc = handle_version(pp, &args, line);
        toklist_free(&args);
        if (phys > 1) {
            out_nls(pp, phys - 1);
        }
        return rc;
    }
    if (!strcmp(name, "line")) {
        int rc;
        long saved = pp->line;
        rc = handle_line_dir(pp, &args, line);
        toklist_free(&args);
        /* handle_line_dir set line to ln-1; caller adds phys.  We want the
         * next line to be ln, so after +phys the value should be ln. */
        if (rc == 0) {
            /* pp->line is ln-1; adding phys would overshoot if phys!=1. */
            long ln = pp->line + 1;
            pp->line = ln - phys;
            (void)saved;
        }
        if (phys > 1) {
            out_nls(pp, phys - 1);
        }
        return rc;
    }
    pp_fail(pp, "preprocessor: unknown directive");
    toklist_free(&args);
    return -1;
}

char *mglGLSLPreprocess(const char *src, size_t len, char *err, size_t err_cap)
{
    PP pp;
    size_t clen = 0, pos = 0;
    char *collapsed;
    memset(&pp, 0, sizeof(pp));
    pp.line = 1;
    pp.file_no = 0;
    pp.version = 0;
    if (!src) {
        src = "";
        len = 0;
    }
    collapsed = collapse_continuations(src, len, &clen);
    if (!collapsed) {
        if (err && err_cap) {
            snprintf(err, err_cap, "preprocessor: out of memory");
        }
        return NULL;
    }
    if (add_predef(&pp, "__LINE__", 1, "0", 0) != 0 ||
        add_predef(&pp, "__FILE__", 2, "0", 0) != 0 ||
        add_predef(&pp, "__VERSION__", 3, "0", 0) != 0 ||
        add_predef(&pp, "GL_core_profile", 0, "1", 1) != 0) {
        free(collapsed);
        if (err && err_cap) {
            snprintf(err, err_cap, "%s", pp.err);
        }
        return NULL;
    }
    while (pos < clen && !pp.failed) {
        int phys = 0;
        char *line = read_logical_line(&pp, collapsed, clen, &pos, &phys);
        if (!line) {
            pp_fail(&pp, "preprocessor: out of memory");
            break;
        }
        if (phys == 0 && pos >= clen && line[0] == 0) {
            free(line);
            break;
        }
        if (process_directive(&pp, line, phys ? phys : 1) != 0 && !pp.failed) {
            pp_fail(&pp, "preprocessor error");
        }
        pp.line += (phys ? phys : 1);
        free(line);
    }
    free(collapsed);
    if (!pp.failed && pp.depth != 0) {
        pp_fail(&pp, "preprocessor: unterminated #if");
    }
    if (pp.failed) {
        size_t i;
        if (err && err_cap) {
            snprintf(err, err_cap, "%s", pp.err);
        }
        free(pp.out);
        for (i = 0; i < pp.nmac; i++) {
            macro_clear(&pp.macros[i]);
        }
        free(pp.macros);
        return NULL;
    }
    out_ch(&pp, 0);
    {
        size_t i;
        for (i = 0; i < pp.nmac; i++) {
            macro_clear(&pp.macros[i]);
        }
        free(pp.macros);
    }
    return pp.out;
}
