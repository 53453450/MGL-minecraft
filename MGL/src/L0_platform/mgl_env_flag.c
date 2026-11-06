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
#include <stdatomic.h>

enum { MGL_ENV_FLAG_CACHE_CAP = 32 };

typedef struct {
    const char *name;
    atomic_int value;
    atomic_int valid;
    int default_on;
} MGLEnvFlagCacheEntry;

static MGLEnvFlagCacheEntry s_env_cache[MGL_ENV_FLAG_CACHE_CAP];
static pthread_mutex_t s_env_cache_mu = PTHREAD_MUTEX_INITIALIZER;
/* Seqlock generation: readers scan the slot array without the mutex and only
 * trust what they read when the generation is unchanged across the scan.
 * Writers (insert under the mutex, invalidate) bump it with release, so a
 * reader either sees a fully written entry or a changed generation.  Lookups
 * sit on per-batch paths (MDI/ICB selection), where the mutex round trip
 * dominated the cost. */
static atomic_uint_fast64_t s_env_cache_gen;

static uint64_t mgl_env_flag_bump_generation(void)
{
    return atomic_fetch_add_explicit(&s_env_cache_gen, 1,
                                     memory_order_acq_rel) + 1;
}

void mgl_env_flag_cache_invalidate(void)
{
    pthread_mutex_lock(&s_env_cache_mu);
    for (int i = 0; i < MGL_ENV_FLAG_CACHE_CAP; i++) {
        atomic_store_explicit(&s_env_cache[i].valid, 0, memory_order_relaxed);
        s_env_cache[i].name = NULL;
        atomic_store_explicit(&s_env_cache[i].value, 0, memory_order_relaxed);
        s_env_cache[i].default_on = 0;
    }
    mgl_env_flag_bump_generation();
    pthread_mutex_unlock(&s_env_cache_mu);
}

static int mgl_env_flag_lookup(const char *name, int default_on)
{
    if (!name) {
        return default_on;
    }

    /* Lock-free hit: stable-generation read of the slot array. */
    const uint64_t gen_before = atomic_load_explicit(&s_env_cache_gen,
                                                     memory_order_acquire);
    for (int i = 0; i < MGL_ENV_FLAG_CACHE_CAP; i++) {
        if (s_env_cache[i].name == name &&
            atomic_load_explicit(&s_env_cache[i].valid, memory_order_relaxed) &&
            s_env_cache[i].default_on == default_on) {
            const int hit = atomic_load_explicit(&s_env_cache[i].value,
                                                 memory_order_relaxed);
            if (atomic_load_explicit(&s_env_cache_gen,
                                     memory_order_acquire) == gen_before) {
                return hit;
            }
            break; /* writer raced us; take the locked path */
        }
    }

    pthread_mutex_lock(&s_env_cache_mu);
    for (int i = 0; i < MGL_ENV_FLAG_CACHE_CAP; i++) {
        if (atomic_load_explicit(&s_env_cache[i].valid, memory_order_relaxed) &&
            s_env_cache[i].name == name &&
            s_env_cache[i].default_on == default_on) {
            const int hit = atomic_load_explicit(&s_env_cache[i].value,
                                                 memory_order_relaxed);
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
        if (!atomic_load_explicit(&s_env_cache[i].valid, memory_order_relaxed)) {
            s_env_cache[i].name = name;
            atomic_store_explicit(&s_env_cache[i].value, result,
                                  memory_order_relaxed);
            s_env_cache[i].default_on = default_on;
            /* valid last, then publish via the generation bump. */
            atomic_store_explicit(&s_env_cache[i].valid, 1,
                                  memory_order_relaxed);
            mgl_env_flag_bump_generation();
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
