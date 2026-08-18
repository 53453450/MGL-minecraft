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
 * mgl_glsl_lexer.h
 * MGL - self-written GLSL frontend tokenizer (see docs/AIR_SHADER_BACKEND_DESIGN.md).
 *
 * M0 scope: tokenizer + #version/#extension/#pragma directive capture.
 * Pure C, no LLVM dependency. Not yet integrated into the build chain;
 * the parser consumes this API in a later milestone.
 */

#ifndef MGL_GLSL_LEXER_H
#define MGL_GLSL_LEXER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MGLGLSLTokenKind {
    MGLGLSL_TOK_END = 0,   /* end of input */

    /* Literals. */
    MGLGLSL_TOK_IDENT,
    MGLGLSL_TOK_INT,       /* decimal/octal/hex literal */
    MGLGLSL_TOK_UINT,      /* unsigned literal (u/U suffix) */
    MGLGLSL_TOK_FLOAT,     /* floating point literal */

    /* Punctuation / operators. */
    MGLGLSL_TOK_PUNCT,

    /* Directives: the token body is the raw directive line text. */
    MGLGLSL_TOK_DIRECTIVE,

    /* Error. */
    MGLGLSL_TOK_ERROR,
} MGLGLSLTokenKind;

typedef struct MGLGLSLToken {
    MGLGLSLTokenKind kind;
    uint32_t start;     /* byte offset into source */
    uint32_t end;       /* byte offset one past the token */
    uint32_t line;      /* 1-based source line */
} MGLGLSLToken;

/* Lexer context.  Holds source range + line/column state so the tokenizer can
 * be driven incrementally. */
typedef struct MGLGLSLexer {
    const char *src;    /* not owned */
    size_t src_len;
    size_t pos;
    uint32_t line;
} MGLGLSLexer;

/* Initialise a lexer over a source buffer. */
void mglGLSLexerInit(MGLGLSLexer *lx, const char *src, size_t len);

/* Get the next token. Return the token kind in *out (or MGLGLSL_TOK_END when
 * input is exhausted).  On an unterminated literal/comment the token is
 * MGLGLSL_TOK_ERROR.  Alignment of tokens is char-based; the caller performs
 * no decoding of UTF-8 (identifiers are ASCII-only in GLSL). */
int mglGLSLexerNext(MGLGLSLexer *lx, MGLGLSLToken *out);

/* Decode a numeric-literal token's value (the lexer owns the source buffer).
 * Returns 0 and writes the parsed double, or -1 if the token is not an
 * INT/UINT/FLOAT token. */
int mglGLSLexerLiteral(const MGLGLSLexer *lx, const MGLGLSLToken *tok,
                       double *out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_GLSL_LEXER_H */