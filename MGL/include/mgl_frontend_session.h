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
MGLTranslationUnit *mglFrontendSessionStealTU(MGLFrontendSession *s);
int mglFrontendRewriteLegacy(const char *src, int stage, char **out,
                             char *err, size_t err_cap);
uint32_t mglFrontendIRBuiltinArrayCount(const MGLIRModule *mod,
                                        const char *name);
/* Non-zero when this stage declares a `sample`-qualified interface (the
 * `sample in` case that used to be found by scanning the source text). */
int mglFrontendStageUsesSampleInterpolation(const MGLIRModule *mod,
                                            const MGLTranslationUnit *tu);

/* Reference queries over a stage's parsed TU bodies (no source-text scan):
 * `name` may be qualified (the leaf is matched), `member` is matched as a
 * member field, optionally required to be rooted at `instance`. */
int mglFrontendStageReferencesName(const MGLTranslationUnit *tu,
                                   const char *name);
int mglFrontendStageReferencesMember(const MGLTranslationUnit *tu,
                                     const char *instance,
                                     const char *member);

/* Non-zero when this stage's IR or TU references builtin `name` (exact
 * replacement for source-text scans; see MGL_AIR_BUILTIN_* in
 * mgl_shader_abi.h). */
int mglFrontendBuiltinUsed(const MGLIRModule *mod,
                           const MGLTranslationUnit *tu,
                           const char *name);

uint32_t mglFrontendBuiltinArrayCount(const MGLIRModule *mod,
                                      const MGLTranslationUnit *tu,
                                      const char *name);

#ifdef __cplusplus
}
#endif

#endif /* MGL_FRONTEND_SESSION_H */
