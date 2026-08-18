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
 * mgl_glsl_lexer.c
 * MGL - self-written GLSL frontend tokenizer.
 */

#include "mgl_glsl_lexer.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

static int is_ident_start(char c)
{
    return isalpha((unsigned char)c) || c == '_';
}

static int is_ident_char(char c)
{
    return isalnum((unsigned char)c) || c == '_';
}

static void skip_line_comment(MGLGLSLexer *lx)
{
    while (lx->pos < lx->src_len && lx->src[lx->pos] != '\n') {
        lx->pos++;
    }
}

static void skip_block_comment(MGLGLSLexer *lx)
{
    while (lx->pos + 1 < lx->src_len) {
        if (lx->src[lx->pos] == '*' && lx->src[lx->pos + 1] == '/') {
            lx->pos += 2;
            return;
        }
        if (lx->src[lx->pos] == '\n') {
            lx->line++;
        }
        lx->pos++;
    }
    lx->pos = lx->src_len; /* unterminated; tokenizer reports END */
}

static void skip_ws(MGLGLSLexer *lx)
{
    for (;;) {
        while (lx->pos < lx->src_len) {
            char c = lx->src[lx->pos];
            if (c == '\n') {
                lx->line++;
                lx->pos++;
            } else if (c == ' ' || c == '\t' || c == '\r' || c == '\v' ||
                       c == '\f') {
                lx->pos++;
            } else {
                break;
            }
        }
        if (lx->pos >= lx->src_len) {
            return;
        }
        if (lx->src[lx->pos] == '/' && lx->pos + 1 < lx->src_len) {
            char n = lx->src[lx->pos + 1];
            if (n == '/') {
                lx->pos += 2;
                skip_line_comment(lx);
                continue;
            }
            if (n == '*') {
                lx->pos += 2;
                skip_block_comment(lx);
                continue;
            }
        }
        return;
    }
}

/* Capture one directive line including the '#'. */
static void scan_directive(MGLGLSLexer *lx, MGLGLSLToken *tok)
{
    tok->start = (uint32_t)lx->pos;
    while (lx->pos < lx->src_len && lx->src[lx->pos] != '\n') {
        lx->pos++;
    }
    tok->end = (uint32_t)lx->pos;
    tok->line = lx->line;
    tok->kind = MGLGLSL_TOK_DIRECTIVE;
}

static void scan_ident(MGLGLSLexer *lx, MGLGLSLToken *tok)
{
    tok->start = (uint32_t)lx->pos;
    while (lx->pos < lx->src_len && is_ident_char(lx->src[lx->pos])) {
        lx->pos++;
    }
    tok->end = (uint32_t)lx->pos;
    tok->line = lx->line;
    tok->kind = MGLGLSL_TOK_IDENT;
}

static int is_hex_digit(char c)
{
    return isdigit((unsigned char)c) ||
           (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

/* Scan the body of a numeric literal (no leading digit consumed).
 * Handles: 123, 123u/U, 0x1F, 0x1Fu, 1.5, 1., .5, 1e-3, 1.5E+2, 1f/F.
 * Returns 1 if the token is floating-point, 0 for integer (set *is_uint). */
static int scan_number(MGLGLSLexer *lx, MGLGLSLToken *tok, int *is_uint)
{
    tok->start = (uint32_t)lx->pos;
    int is_float = 0;
    *is_uint = 0;

    /* 0x/0X prefix. */
    if (lx->src[lx->pos] == '0' && lx->pos + 1 < lx->src_len &&
        (lx->src[lx->pos + 1] == 'x' || lx->src[lx->pos + 1] == 'X')) {
        lx->pos += 2;
        while (lx->pos < lx->src_len && is_hex_digit(lx->src[lx->pos])) {
            lx->pos++;
        }
    } else {
        while (lx->pos < lx->src_len && isdigit((unsigned char)lx->src[lx->pos])) {
            lx->pos++;
        }
        /* Fraction. */
        if (lx->pos < lx->src_len && lx->src[lx->pos] == '.') {
            is_float = 1;
            lx->pos++;
            while (lx->pos < lx->src_len && isdigit((unsigned char)lx->src[lx->pos])) {
                lx->pos++;
            }
        }
        /* Exponent. */
        if (lx->pos < lx->src_len &&
            (lx->src[lx->pos] == 'e' || lx->src[lx->pos] == 'E')) {
            size_t save = lx->pos;
            lx->pos++;
            if (lx->pos < lx->src_len &&
                (lx->src[lx->pos] == '+' || lx->src[lx->pos] == '-')) {
                lx->pos++;
            }
            if (lx->pos < lx->src_len &&
                isdigit((unsigned char)lx->src[lx->pos])) {
                is_float = 1;
                while (lx->pos < lx->src_len &&
                       isdigit((unsigned char)lx->src[lx->pos])) {
                    lx->pos++;
                }
            } else {
                lx->pos = save; /* not an exponent; rewind */
            }
        }
    }

    /* Suffixes. */
    if (lx->pos < lx->src_len) {
        char c = lx->src[lx->pos];
        if (c == 'u' || c == 'U') {
            *is_uint = 1;
            lx->pos++;
        } else if (c == 'f' || c == 'F') {
            is_float = 1;
            lx->pos++;
        }
    }

    tok->end = (uint32_t)lx->pos;
    tok->line = lx->line;
    tok->kind = (is_float || *is_uint) ? (is_float ? MGLGLSL_TOK_FLOAT : MGLGLSL_TOK_UINT)
                                       : MGLGLSL_TOK_INT;
    return is_float;
}

static int scan_punct(MGLGLSLexer *lx, MGLGLSLToken *tok)
{
    static const char *const single[] = {
        "(", ")", "[", "]", "{", "}", ",", ";", ":", "?", "~",
        "+", "-", "*", "/", "%", "<", ">", "=", "!", "&", "|",
        "^", ".",
    };
    size_t i;
    for (i = 0; i < sizeof(single) / sizeof(single[0]); i++) {
        size_t len = strlen(single[i]);
        if (lx->pos + len <= lx->src_len &&
            memcmp(lx->src + lx->pos, single[i], len) == 0) {
            tok->start = (uint32_t)lx->pos;
            lx->pos += len;
            tok->end = (uint32_t)lx->pos;
            tok->line = lx->line;
            tok->kind = MGLGLSL_TOK_PUNCT;
            return 0;
        }
    }
    return -1;
}

void mglGLSLexerInit(MGLGLSLexer *lx, const char *src, size_t len)
{
    lx->src = src;
    lx->src_len = len;
    lx->pos = 0;
    lx->line = 1;
}

int mglGLSLexerNext(MGLGLSLexer *lx, MGLGLSLToken *out)
{
    if (!lx || !out) {
        return -1;
    }
    skip_ws(lx);
    if (lx->pos >= lx->src_len) {
        out->kind = MGLGLSL_TOK_END;
        out->start = (uint32_t)lx->pos;
        out->end = (uint32_t)lx->pos;
        out->line = lx->line;
        return 0;
    }

    char c = lx->src[lx->pos];
    if (c == '#') {
        scan_directive(lx, out);
        return 0;
    }
    if (is_ident_start(c)) {
        scan_ident(lx, out);
        return 0;
    }
    if (isdigit((unsigned char)c) ||
        (c == '.' && lx->pos + 1 < lx->src_len &&
         isdigit((unsigned char)lx->src[lx->pos + 1]))) {
        int is_uint = 0;
        scan_number(lx, out, &is_uint);
        return 0;
    }
    if (scan_punct(lx, out) == 0) {
        return 0;
    }

    /* Unknown character. */
    out->kind = MGLGLSL_TOK_ERROR;
    out->start = (uint32_t)lx->pos;
    out->end = (uint32_t)(lx->pos + 1);
    out->line = lx->line;
    lx->pos++;
    return 0;
}

int mglGLSLexerLiteral(const MGLGLSLexer *lx, const MGLGLSLToken *tok,
                       double *out)
{
    if (!lx || !tok || !out || (tok->kind != MGLGLSL_TOK_INT &&
                                tok->kind != MGLGLSL_TOK_UINT &&
                                tok->kind != MGLGLSL_TOK_FLOAT)) {
        return -1;
    }
    size_t n = tok->end - tok->start;
    size_t cap = n + 1;
    char *buf = (char *)malloc(cap);
    if (!buf) {
        return -1;
    }
    memcpy(buf, lx->src + tok->start, n);
    buf[n] = '\0';

    /* Hex literal: parse directly.  strtod's hex grammar is not the C99
     * "0x1F" form only without an exponent, and the trailing 'F' of e.g.
     * 0x1F must not be mistaken for a float suffix.  Strip only u/U here;
     * f/F is not a valid hex suffix. */
    if (buf[0] == '0' && (buf[1] == 'x' || buf[1] == 'X')) {
        unsigned long v = 0;
        char *p = buf + 2;
        int any = 0;
        while (*p && *p != 'u' && *p != 'U') {
            unsigned long d;
            if (*p >= '0' && *p <= '9') {
                d = (unsigned long)(*p - '0');
            } else if (*p >= 'a' && *p <= 'f') {
                d = (unsigned long)(*p - 'a' + 10);
            } else if (*p >= 'A' && *p <= 'F') {
                d = (unsigned long)(*p - 'A' + 10);
            } else {
                break;
            }
            v = v * 16 + d;
            p++;
            any = 1;
        }
        free(buf);
        if (any) {
            *out = (double)v;
            return 0;
        }
        return -1;
    }

    /* Strip the u/U and f/F suffix char before strtod. */
    if (n > 0) {
        char last = buf[n - 1];
        if (last == 'u' || last == 'U' || last == 'f' || last == 'F') {
            buf[n - 1] = '\0';
        }
    }

    char *endptr = NULL;
    *out = strtod(buf, &endptr);
    int rc = (endptr && *endptr == '\0') ? 0 : -1;
    free(buf);
    return rc;
}
