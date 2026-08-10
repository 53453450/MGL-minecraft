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

static inline int mgl_env_flag_enabled(const char *name)
{
    const char *value = name ? getenv(name) : NULL;
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

static inline int mgl_env_flag_enabled_default_on(const char *name)
{
    const char *value = name ? getenv(name) : NULL;
    return (!value || value[0] == '\0') ? 1 : mgl_env_flag_enabled(name);
}

#endif /* MGL_ENV_FLAG_H */
