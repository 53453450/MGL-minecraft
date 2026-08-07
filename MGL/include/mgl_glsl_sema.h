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
 * mgl_glsl_sema.h
 * MGL - self-written GLSL frontend semantic analysis (see
 * docs/AIR_SHADER_BACKEND_DESIGN.md).  Pure C, depends on mgl_glsl_ast.h
 * and mgl_ir.h.
 *
 * M0.5/M1 scope: symbol tables, type resolution (MGLTypeSpec -> MGLIRType),
 * expression type checking with implicit conversions, and uniform/buffer
 * block layout computation.  Function overload resolution is limited to
 * a single definition match (generalized overloads land with the builtin
 * table in a later milestone).
 */

#ifndef MGL_GLSL_SEMA_H
#define MGL_GLSL_SEMA_H

#include "mgl_glsl_ast.h"
#include "mgl_ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Error severity. */
typedef enum MGL_SemaErrorKind {
    MGL_SEMA_OK = 0,
    MGL_SEMA_ERR,              /* hard error: analysis aborted path */
    MGL_SEMA_WARN,             /* diagnostic only */
} MGLSemaErrorKind;

/* One semantic error, filled by mglGLSLSemanticCheck. */
typedef struct MGLSemaError {
    char *message;             /* heap-allocated */
    uint32_t line;
} MGLSemaError;

/* A compiled shader declaration (variable or function) as it will be
 * consumed by the M1 AIR backend.  This is the MGLIR "module" minimal
 * skeleton: one entry per global variable or function declaration. */
typedef struct MGLIRSymbol {
    char *name;                 /* owned */
    MGLIRType *type;            /* owned: variable/layout type */
    uint32_t qualifiers;        /* MGL_AST_Q_* */
    uint32_t layout;            /* MGL_AST_LAYOUT_* (interface block) */
    uint32_t matrix_major;      /* MGL_AST_MATRIX_* */
    uint32_t binding;           /* -1 if unspecified */
    uint32_t location;          /* -1 if unspecified */
    uint32_t offset;            /* block member offset / -1 */
    char *block_name;           /* owning anonymous block, or NULL */
    uint32_t block_member_index;/* member index within the block */
    int is_function;            /* 1 = function declaration */
    MGLIRType *return_type;     /* function return type */
    uint32_t param_count;
    MGLIRType **param_types;    /* owned array of copies */
} MGLIRSymbol;

/* Shader-level module produced by semantic analysis.  Layouts on
 * interface-block symbols carry computed member offsets. */
typedef struct MGLIRModule {
    MGLIRSymbol **symbols;
    uint32_t symbol_count;
} MGLIRModule;

/* Analyze a parsed translation unit.  Fills `module` with resolved/typed
 * global symbols (caller frees with mglIRModuleDestroy), and appends any
 * diagnostics to `errors` (caller frees each string).  Returns number of
 * hard errors (0 = clean). */
int mglGLSLSemanticCheck(const MGLTranslationUnit *tu,
                         MGLIRModule *module,
                         MGLSemaError **errors,
                         uint32_t *error_count);

/* Make a variable symbol without a type check (entry/helper). */
MGLIRSymbol *mglIRSymbolNew(const char *name, MGLIRType *type);

/* Link-time interface matching between two compiled stages (GLSL 4.60
 * §4.3.9.5): ordinary in/out variables declared on both sides must have
 * identical types; interface blocks match by block name and require
 * identical member lists and layout.  Variables on one side only are
 * legal.  Returns the number of hard errors. */
int mglGLSLInterfaceCheck(const MGLIRModule *a, const MGLIRModule *b,
                          MGLSemaError **errors, uint32_t *error_count);

void mglGLSLSemanticCheckDestroy(MGLSemaError *errors, uint32_t count);

void mglIRModuleDestroy(MGLIRModule *module);

#ifdef __cplusplus
}
#endif

#endif /* MGL_GLSL_SEMA_H */