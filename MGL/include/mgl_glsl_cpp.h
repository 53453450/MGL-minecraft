/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_glsl_cpp.h — GLSL 4.60 §3.3 preprocessor.
 */

#ifndef MGL_GLSL_CPP_H
#define MGL_GLSL_CPP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Expand macros and resolve conditionals.  Returns a heap buffer the
 * caller must free.  On a preprocessor diagnostic, returns NULL and
 * writes a message into err (if err_cap > 0). */
char *mglGLSLPreprocess(const char *src, size_t len, char *err, size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif /* MGL_GLSL_CPP_H */
