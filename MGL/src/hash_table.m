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
 * hash_table.c
 * MGL
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <assert.h>
#include <stdint.h>  // For SIZE_MAX and UINT_MAX
#include <limits.h>  // For UINT_MAX fallback
#include <string.h>  // For memcpy
#include <stdbool.h>

#ifdef __APPLE__
#include <Metal/Metal.h>
#endif

#include "hash_table.h"
#include "glm_context.h"

#define MGL_HASH_TABLE_MAX_CAPACITY (1u << 24)
#define MGL_HASH_LOAD_FACTOR_NUM 7u
#define MGL_HASH_LOAD_FACTOR_DEN 10u
#define MGL_HASH_STATE_EMPTY 0u
#define MGL_HASH_STATE_OCCUPIED 1u
#define MGL_HASH_STATE_DELETED 2u

#ifndef MGL_VERBOSE_HASH_LOGS
#define MGL_VERBOSE_HASH_LOGS 0
#endif
#define MGL_HASH_MIN_CAPACITY 64u
#define MGL_HASH_COOKIE_KEYS 0x4d474c5f48415348ULL
#define MGL_HASH_COOKIE_STATES 0x53544154455f4d47ULL

static int mglRehash(HashTable *table, size_t new_capacity);

static inline uintptr_t mglMakeCookie(const void *ptr, uintptr_t salt)
{
    return ptr ? (((uintptr_t)ptr ^ salt) + 0x9e3779b97f4a7c15ULL) : 0u;
}

static inline int mglCookieMatches(const void *ptr, uintptr_t cookie, uintptr_t salt)
{
    return ptr && cookie && (cookie == mglMakeCookie(ptr, salt));
}

static inline int mglIsPow2Size(size_t value)
{
    return value != 0u && ((value & (value - 1u)) == 0u);
}

static int mglHashTableLooksSane(const HashTable *table)
{
    if (!table) {
        return 0;
    }

    if (table->size == 0u) {
        return table->count == 0u &&
               table->keys == NULL &&
               table->states == NULL;
    }

    if (!mglIsPow2Size(table->size) ||
        table->size > MGL_HASH_TABLE_MAX_CAPACITY ||
        table->count > table->size ||
        !table->keys ||
        !table->states) {
        return 0;
    }

    if (!mglCookieMatches(table->keys, table->keys_cookie, MGL_HASH_COOKIE_KEYS) ||
        !mglCookieMatches(table->states, table->states_cookie, MGL_HASH_COOKIE_STATES)) {
        return 0;
    }

    return 1;
}

static int mglRepairHashTableIfNeeded(HashTable *table, const char *where)
{
    GLuint saved_name;

    if (!table) {
        return 0;
    }

    if (mglHashTableLooksSane(table)) {
        return 1;
    }

    saved_name = table->current_name;
    fprintf(stderr,
            "MGL WARNING: repairing corrupt hash table at %s table=%p size=%zu count=%zu current=%u keys=%p states=%p keyCookie=0x%llx stateCookie=0x%llx\n",
            where ? where : "unknown",
            (void *)table,
            table->size,
            table->count,
            saved_name,
            (void *)table->keys,
            (void *)table->states,
            (unsigned long long)table->keys_cookie,
            (unsigned long long)table->states_cookie);

    table->keys = NULL;
    table->states = NULL;
    table->keys_cookie = 0u;
    table->states_cookie = 0u;
    table->size = 0u;
    table->count = 0u;
    table->current_name = saved_name;

    return mglRehash(table, MGL_HASH_MIN_CAPACITY);
}

static inline void mglSetStorage(HashTable *table, HashObj *keys, unsigned char *states)
{
    if (!table) {
        return;
    }
    table->keys = keys;
    table->states = states;
    table->keys_cookie = mglMakeCookie(keys, MGL_HASH_COOKIE_KEYS);
    table->states_cookie = mglMakeCookie(states, MGL_HASH_COOKIE_STATES);
}

static void mglFreeStorageIfOwned(HashTable *table,
                                  HashObj *keys,
                                  unsigned char *states,
                                  uintptr_t keys_cookie,
                                  uintptr_t states_cookie,
                                  const char *reason)
{
    (void)table;

    if (keys) {
        if (mglCookieMatches(keys, keys_cookie, MGL_HASH_COOKIE_KEYS)) {
            free(keys);
        } else {
            fprintf(stderr,
                    "MGL WARNING: hash storage keys pointer ownership mismatch (%s) table=%p keys=%p cookie=0x%llx; skipping free\n",
                    reason ? reason : "unknown",
                    (void *)table,
                    (void *)keys,
                    (unsigned long long)keys_cookie);
        }
    }

    if (states) {
        if (mglCookieMatches(states, states_cookie, MGL_HASH_COOKIE_STATES)) {
            free(states);
        } else {
            fprintf(stderr,
                    "MGL WARNING: hash storage states pointer ownership mismatch (%s) table=%p states=%p cookie=0x%llx; skipping free\n",
                    reason ? reason : "unknown",
                    (void *)table,
                    (void *)states,
                    (unsigned long long)states_cookie);
        }
    }
}

static inline uint32_t mglHashName(GLuint name)
{
    uint32_t x = (uint32_t)name;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static size_t mglNextPow2(size_t v)
{
    if (v <= 1u) {
        return 1u;
    }

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if SIZE_MAX > UINT32_MAX
    v |= v >> 32;
#endif
    v++;
    return v;
}

static int mglAllocHashStorage(size_t capacity, HashObj **keys_out, unsigned char **states_out)
{
    HashObj *keys = NULL;
    unsigned char *states = NULL;

    if (!keys_out || !states_out || capacity == 0u) {
        return 0;
    }

    if (capacity > (SIZE_MAX / sizeof(HashObj))) {
        return 0;
    }

    keys = (HashObj *)calloc(capacity, sizeof(HashObj));
    if (!keys) {
        return 0;
    }

    states = (unsigned char *)calloc(capacity, sizeof(unsigned char));
    if (!states) {
        free(keys);
        return 0;
    }

    *keys_out = keys;
    *states_out = states;
    return 1;
}

static size_t mglFindSlot(const HashTable *table, GLuint name, int for_insert, int *found)
{
    size_t mask;
    size_t index;
    size_t first_deleted = SIZE_MAX;

    if (found) {
        *found = 0;
    }

    if (!table || !table->keys || !table->states || table->size == 0u) {
        return SIZE_MAX;
    }

    mask = table->size - 1u;
    index = (size_t)mglHashName(name) & mask;

    for (size_t probe = 0; probe < table->size; probe++) {
        unsigned char state = table->states[index];

        if (state == MGL_HASH_STATE_EMPTY) {
            if (for_insert) {
                return (first_deleted != SIZE_MAX) ? first_deleted : index;
            }
            return SIZE_MAX;
        }

        if (state == MGL_HASH_STATE_OCCUPIED && table->keys[index].name == name) {
            if (found) {
                *found = 1;
            }
            return index;
        }

        if (for_insert && state == MGL_HASH_STATE_DELETED && first_deleted == SIZE_MAX) {
            first_deleted = index;
        }

        index = (index + 1u) & mask;
    }

    return (for_insert && first_deleted != SIZE_MAX) ? first_deleted : SIZE_MAX;
}

static int mglRehash(HashTable *table, size_t new_capacity)
{
    HashObj *new_keys = NULL;
    unsigned char *new_states = NULL;
    HashObj *old_keys = NULL;
    unsigned char *old_states = NULL;
    uintptr_t old_keys_cookie = 0u;
    uintptr_t old_states_cookie = 0u;
    size_t moved = 0u;

    if (!table) {
        return 0;
    }

    if (new_capacity < MGL_HASH_MIN_CAPACITY) {
        new_capacity = MGL_HASH_MIN_CAPACITY;
    }

    new_capacity = mglNextPow2(new_capacity);
    if (new_capacity > MGL_HASH_TABLE_MAX_CAPACITY) {
        new_capacity = MGL_HASH_TABLE_MAX_CAPACITY;
    }

    if (!mglAllocHashStorage(new_capacity, &new_keys, &new_states)) {
        fprintf(stderr, "MGL ERROR: failed to allocate hash storage new_capacity=%zu\n", new_capacity);
        return 0;
    }

    old_keys = table->keys;
    old_states = table->states;
    old_keys_cookie = table->keys_cookie;
    old_states_cookie = table->states_cookie;

    if (table->keys && table->states && table->size > 0u) {
        for (size_t i = 0; i < table->size; i++) {
            if (table->states[i] != MGL_HASH_STATE_OCCUPIED || table->keys[i].data == NULL) {
                continue;
            }

            HashObj obj = table->keys[i];
            size_t mask = new_capacity - 1u;
            size_t idx = (size_t)mglHashName(obj.name) & mask;

            for (size_t probe = 0; probe < new_capacity; probe++) {
                if (new_states[idx] != MGL_HASH_STATE_OCCUPIED) {
                    new_keys[idx] = obj;
                    new_states[idx] = MGL_HASH_STATE_OCCUPIED;
                    moved++;
                    break;
                }
                idx = (idx + 1u) & mask;
            }
        }
    }

    mglSetStorage(table, new_keys, new_states);
    table->size = new_capacity;
    table->count = moved;
    mglFreeStorageIfOwned(table,
                          old_keys,
                          old_states,
                          old_keys_cookie,
                          old_states_cookie,
                          "rehash");

    return 1;
}

static int ensureHashTableCapacity(HashTable *table, GLuint name)
{
    size_t old_cap;
    size_t new_cap;

    if (!table) {
        return 0;
    }

    if (!mglRepairHashTableIfNeeded(table, "ensure")) {
        return 0;
    }

    if (!table->keys || !table->states || table->size == 0u) {
        return mglRehash(table, MGL_HASH_MIN_CAPACITY);
    }

    if (table->count + 1u < (table->size * MGL_HASH_LOAD_FACTOR_NUM) / MGL_HASH_LOAD_FACTOR_DEN) {
        return 1;
    }

    old_cap = table->size;
    new_cap = old_cap;
    while (new_cap < MGL_HASH_TABLE_MAX_CAPACITY &&
           (table->count + 1u) >= (new_cap * MGL_HASH_LOAD_FACTOR_NUM) / MGL_HASH_LOAD_FACTOR_DEN) {
        new_cap *= 2u;
    }

    if (new_cap == old_cap) {
        fprintf(stderr, "MGL ERROR: hash table cannot grow further table=%p key=%u count=%zu cap=%zu\n",
                (void *)table,
                name,
                table->count,
                table->size);
        return 0;
    }

    fprintf(stderr,
            "MGL HASH grow table=%p oldCap=%zu newCap=%zu count=%zu key=%u load=%.2f\n",
            (void *)table,
            old_cap,
            new_cap,
            table->count,
            name,
            old_cap ? ((double)table->count / (double)old_cap) : 0.0);

    return mglRehash(table, new_cap);
}

void initHashTable(HashTable *ptr, GLuint size)
{
    if (!ptr)
    {
        return;
    }

    ptr->keys = NULL;
    ptr->states = NULL;
    ptr->keys_cookie = 0u;
    ptr->states_cookie = 0u;
    ptr->current_name = 0;
    ptr->size = 0;
    ptr->count = 0;

    if (size > 0) {
        size_t desired = (size_t)size * 2u;
        if (desired < MGL_HASH_MIN_CAPACITY) {
            desired = MGL_HASH_MIN_CAPACITY;
        }
        if (!mglRehash(ptr, desired)) {
            fprintf(stderr, "MGL ERROR: initHashTable failed to allocate initial capacity %u\n", size);
        }
    }
}

HashTable *createHashTable(GLuint size)
{
    HashTable *table = (HashTable *)calloc(1, sizeof(HashTable));
    if (!table) {
        return NULL;
    }
    initHashTable(table, size);
    return table;
}

void destroyHashTable(HashTable *ptr)
{
    HashObj *keys;
    unsigned char *states;
    uintptr_t keys_cookie;
    uintptr_t states_cookie;

    if (!ptr) {
        return;
    }

    keys = ptr->keys;
    states = ptr->states;
    keys_cookie = ptr->keys_cookie;
    states_cookie = ptr->states_cookie;

    mglFreeStorageIfOwned(ptr, keys, states, keys_cookie, states_cookie, "destroy");

    ptr->keys = NULL;
    ptr->states = NULL;
    ptr->keys_cookie = 0u;
    ptr->states_cookie = 0u;
    ptr->size = 0u;
    ptr->count = 0u;
    ptr->current_name = 0u;
}

GLuint getNewName(HashTable *table)
{
    GLuint name;

    if (!table)
    {
        return 0;
    }

    if (table->current_name == UINT_MAX)
    {
        fprintf(stderr, "MGL ERROR: hash table name space exhausted\n");
        return 0;
    }

    name = ++table->current_name;

    // Pre-grow the table so callers using generated names never hit fixed-size limits.
    if (!ensureHashTableCapacity(table, name))
    {
        table->current_name--;
        return 0;
    }

    return name;
}

void *searchHashTable(HashTable *table, GLuint name)
{
    int found = 0;
    size_t slot;

    if (!table)
    {
        return NULL;
    }

    if (!mglRepairHashTableIfNeeded(table, "search")) {
        return NULL;
    }

    if (!table->keys || !table->states || table->size == 0)
    {
        return NULL;
    }

    slot = mglFindSlot(table, name, 0, &found);
    if (!found || slot == SIZE_MAX) {
        return NULL;
    }

    return table->keys[slot].data;
}

void insertHashElement(HashTable *table, GLuint name, void *data)
{
    if (!ensureHashTableCapacity(table, name))
    {
        fprintf(stderr, "MGL ERROR: insertHashElement failed grow table=%p name=%u data=%p\n",
                (void *)table, name, data);
        return;
    }

    int found = 0;
    size_t slot = mglFindSlot(table, name, 1, &found);
    if (slot == SIZE_MAX) {
        fprintf(stderr, "MGL ERROR: insertHashElement failed slot lookup table=%p name=%u data=%p\n",
                (void *)table, name, data);
        return;
    }

    if (!found) {
        table->count++;
    }

    table->keys[slot].name = name;
    table->keys[slot].data = data;
    table->states[slot] = MGL_HASH_STATE_OCCUPIED;

    if (MGL_VERBOSE_HASH_LOGS) {
        fprintf(stderr,
                "MGL HASH insert table=%p name=%u slot=%zu data=%p found=%d count=%zu cap=%zu\n",
                (void *)table,
                name,
                slot,
                data,
                found,
                table->count,
                table->size);
    }
}

void deleteHashElement(HashTable *table, GLuint name)
{
    int found = 0;
    size_t slot;

    if (!table || !table->keys || !table->states || table->size == 0) {
        return;
    }

    slot = mglFindSlot(table, name, 0, &found);
    if (!found || slot == SIZE_MAX) {
        return;
    }

    void *obj_data = table->keys[slot].data;

    // Perform Metal cleanup for different object types
    if (obj_data) {
        extern GLMContext _ctx;

        // Check if this is a shader object
        if (table == &_ctx->state.shader_table) {
            // Shader-specific Metal cleanup
            Shader *shader = (Shader *)obj_data;
            if (shader->mtl_data.function || shader->mtl_data.library) {
                fprintf(stderr, "MGL: Metal cleanup for shader object %u\n", name);
                // In ARC mode, we just need to set the pointers to nil
                // The memory will be automatically released
                shader->mtl_data.function = NULL;
                shader->mtl_data.library = NULL;
            }
        }
        // Check if this is a program object
        else if (table == &_ctx->state.program_table) {
            // Program-specific Metal cleanup
            Program *program = (Program *)obj_data;
            if (program->mtl_data) {
                fprintf(stderr, "MGL: Metal cleanup for program object %u\n", name);
                // In ARC mode, we just need to set the pointer to nil
                // The memory will be automatically released
                program->mtl_data = NULL;
            }
        }
        // Check if this is a texture object
        else if (table == &_ctx->state.texture_table) {
            // Texture-specific Metal cleanup
            Texture *texture = (Texture *)obj_data;
            if (texture->mtl_data) {
                fprintf(stderr, "MGL: Metal cleanup for texture object %u\n", name);
                // In ARC mode, we just need to set the pointer to nil
                // The memory will be automatically released
                texture->mtl_data = NULL;
            }
        }
        // Check if this is a buffer object
        else if (table == &_ctx->state.buffer_table) {
            // Buffer-specific Metal cleanup
            Buffer *buffer = (Buffer *)obj_data;
            if (buffer->data.mtl_data) {
                fprintf(stderr, "MGL: Metal cleanup for buffer object %u\n", name);
                // In ARC mode, we just need to set the pointer to nil
                // The memory will be automatically released
                buffer->data.mtl_data = NULL;
            }
        }
        else {
            // Generic cleanup for unknown object types
            fprintf(stderr, "MGL: deleteHashElement called for object %u (Metal cleanup implemented)\n", name);
        }
    }

    table->keys[slot].name = 0;
    table->keys[slot].data = NULL;
    table->states[slot] = MGL_HASH_STATE_DELETED;
    if (table->count > 0u) {
        table->count--;
    }

    if (table->count == 0u && table->states) {
        memset(table->states, MGL_HASH_STATE_EMPTY, table->size * sizeof(unsigned char));
    }
}
