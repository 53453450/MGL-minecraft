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
 * mgl_glsl_parser.h
 * MGL - self-written GLSL frontend recursive-descent parser (see
 * docs/AIR_SHADER_BACKEND_DESIGN.md).  Pure C, no LLVM dependency.
 *
 * M0.5/M1 scope: statement/expression/declaration subset sufficient for
 * common vertex/fragment/compute shaders.  The translation unit AST is
 * consumed by mgl_glsl_sema in a later milestone.
 */

#ifndef MGL_GLSL_PARSER_H
#define MGL_GLSL_PARSER_H

#include "mgl_glsl_ast.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Parse a complete GLSL source buffer into a translation unit.  The
 * returned MGLTranslationUnit is heap-owned (including all nested nodes);
 * free with mglGLSLTranslationUnitDestroy.  On any parse error the function
 * still returns a valid MGLTranslationUnit with `error` set, or NULL on
 * allocation failure. */
MGLTranslationUnit *mglGLSLParse(const char *src, size_t len);

/* Free a translation unit and all nested AST nodes. */
void mglGLSLTranslationUnitDestroy(MGLTranslationUnit *tu);

#ifdef __cplusplus
}
#endif

#endif /* MGL_GLSL_PARSER_H */