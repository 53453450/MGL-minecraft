/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * One parse + sema session shared by reflect, AIR codegen, and uniform seed.
 */

#ifndef MGL_FRONTEND_SESSION_H
#define MGL_FRONTEND_SESSION_H

#include <stddef.h>
#include <stdint.h>

#include "mgl_glsl_sema.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLFrontendSession {
    char *legacy_src; /* owned rewritten source, or NULL */
    const char *src;  /* points at legacy_src or the caller buffer */
    MGLTranslationUnit *tu;
    MGLIRModule mod;
    int stage;
    int ready;
} MGLFrontendSession;

void mglFrontendSessionInit(MGLFrontendSession *s);
void mglFrontendSessionDestroy(MGLFrontendSession *s);
int mglFrontendSessionBuild(MGLFrontendSession *s, const char *src, int stage,
                            char *err, size_t err_cap);
int mglFrontendRewriteLegacy(const char *src, int stage, char **out,
                             char *err, size_t err_cap);
uint32_t mglFrontendIRBuiltinArrayCount(const MGLIRModule *mod,
                                        const char *name);
uint32_t mglFrontendBuiltinArrayCount(const MGLIRModule *mod,
                                      const MGLTranslationUnit *tu,
                                      const char *name);

#ifdef __cplusplus
}
#endif

#endif /* MGL_FRONTEND_SESSION_H */
