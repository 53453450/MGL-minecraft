/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_env_flag.h
 * MGL
 *
 * Single-source environment-flag truthiness parser.
 *
 * Previously the truthiness logic lived in three places: a private copy in
 * mgl_trace_log.m (mglTraceEnvFlag), the renderer's mglEnvFlagEnabledCached
 * in MGLRenderer.m, and ad-hoc getenv() checks in plain-C translation units
 * (tex_param.c, framebuffers.c).  This header consolidates the parse so the
 * semantics (empty/0/false/no/off => disabled) are defined exactly once.
 *
 * Returns 1 for a truthy value (non-empty, not 0/false/no/off) and 0
 * otherwise.  It does NOT implement "unset => default ON" — callers that
 * need that (e.g. the renderer's mglEnvFlagEnabledCached) must check for the
 * empty/unset case separately.
 *
 * Pure C (no Foundation/BOOL) so it is includable from .c and .m alike.
 */

#ifndef MGL_ENV_FLAG_H
#define MGL_ENV_FLAG_H

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Drop process-lifetime getenv memo entries (STATE_DATAFLOW T13-3).
 * Tests that setenv/unsetenv mid-process must call this after mutating env. */
void mgl_env_flag_cache_invalidate(void);

/* Uncached parse of an already-fetched getenv string. */
static inline int mgl_env_flag_value_enabled(const char *value)
{
    if (!value || value[0] == '\0') {
        return 0;
    }
    if (strcmp(value, "0") == 0 ||
        strcasecmp(value, "false") == 0 ||
        strcasecmp(value, "no") == 0 ||
        strcasecmp(value, "off") == 0) {
        return 0;
    }
    return 1;
}

/* Memoized getenv + parse. Defined in mgl_env_flag.c. */
int mgl_env_flag_enabled(const char *name);
int mgl_env_flag_enabled_default_on(const char *name);

#ifdef __cplusplus
}
#endif

#endif /* MGL_ENV_FLAG_H */
