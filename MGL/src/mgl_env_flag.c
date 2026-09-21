/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_env_flag.h"

#include <pthread.h>

enum { MGL_ENV_FLAG_CACHE_CAP = 32 };

typedef struct {
    const char *name;
    int value;
    int default_on;
    int valid;
} MGLEnvFlagCacheEntry;

static MGLEnvFlagCacheEntry s_env_cache[MGL_ENV_FLAG_CACHE_CAP];
static pthread_mutex_t s_env_cache_mu = PTHREAD_MUTEX_INITIALIZER;

void mgl_env_flag_cache_invalidate(void)
{
    pthread_mutex_lock(&s_env_cache_mu);
    for (int i = 0; i < MGL_ENV_FLAG_CACHE_CAP; i++) {
        s_env_cache[i].valid = 0;
        s_env_cache[i].name = NULL;
        s_env_cache[i].value = 0;
        s_env_cache[i].default_on = 0;
    }
    pthread_mutex_unlock(&s_env_cache_mu);
}

static int mgl_env_flag_lookup(const char *name, int default_on)
{
    if (!name) {
        return default_on;
    }

    pthread_mutex_lock(&s_env_cache_mu);
    for (int i = 0; i < MGL_ENV_FLAG_CACHE_CAP; i++) {
        if (s_env_cache[i].valid &&
            s_env_cache[i].name == name &&
            s_env_cache[i].default_on == default_on) {
            int hit = s_env_cache[i].value;
            pthread_mutex_unlock(&s_env_cache_mu);
            return hit;
        }
    }

    const char *value = getenv(name);
    int result;
    if (!value || value[0] == '\0') {
        result = default_on;
    } else {
        result = mgl_env_flag_value_enabled(value);
    }

    for (int i = 0; i < MGL_ENV_FLAG_CACHE_CAP; i++) {
        if (!s_env_cache[i].valid) {
            s_env_cache[i].name = name;
            s_env_cache[i].value = result;
            s_env_cache[i].default_on = default_on;
            s_env_cache[i].valid = 1;
            break;
        }
    }
    pthread_mutex_unlock(&s_env_cache_mu);
    return result;
}

int mgl_env_flag_enabled(const char *name)
{
    return mgl_env_flag_lookup(name, 0);
}

int mgl_env_flag_enabled_default_on(const char *name)
{
    return mgl_env_flag_lookup(name, 1);
}
