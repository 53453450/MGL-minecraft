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

#ifdef __APPLE__
#include <Metal/Metal.h>
#endif

#include "hash_table.h"
#include "glm_context.h"

#define MGL_HASH_TABLE_MAX_CAPACITY (1u << 24)

static int ensureHashTableCapacity(HashTable *table, GLuint name)
{
    size_t old_size = 0;
    size_t new_size;
    HashObj *new_keys;
    int has_valid_old_storage = 0;

    assert(table);

    if (table->keys != NULL &&
        table->size > 0 &&
        table->size < MGL_HASH_TABLE_MAX_CAPACITY)
    {
        has_valid_old_storage = 1;
        old_size = table->size;
    }

    if (has_valid_old_storage && name < table->size)
    {
        return 1;
    }

    new_size = old_size ? old_size : 64;

    while (new_size <= name)
    {
        if (new_size > (size_t)(MGL_HASH_TABLE_MAX_CAPACITY / 2))
        {
            new_size = MGL_HASH_TABLE_MAX_CAPACITY;
            break;
        }

        new_size *= 2;
    }

    if (new_size <= name)
    {
        fprintf(stderr, "MGL ERROR: hash table cannot grow to hold name %u\n", name);
        return 0;
    }

    if (new_size > MGL_HASH_TABLE_MAX_CAPACITY)
    {
        fprintf(stderr, "MGL ERROR: hash table growth exceeds max capacity (%u)\n",
                MGL_HASH_TABLE_MAX_CAPACITY);
        return 0;
    }

    if (new_size > SIZE_MAX / sizeof(HashObj))
    {
        fprintf(stderr, "MGL SECURITY ERROR: Hash table allocation would overflow size_t, preventing\n");
        return 0;
    }

    new_keys = (HashObj *)calloc(new_size, sizeof(HashObj));
    if (!new_keys)
    {
        fprintf(stderr, "MGL SECURITY ERROR: Hash table allocation failed\n");
        return 0;
    }

    fprintf(stderr,
            "MGL HASH grow table=%p keys=%p oldCap=%zu required=%u newCap=%zu\n",
            (void *)table,
            (void *)table->keys,
            has_valid_old_storage ? table->size : (size_t)0,
            name,
            new_size);

    if (has_valid_old_storage && table->keys && old_size > 0)
    {
        memcpy(new_keys, table->keys, old_size * sizeof(HashObj));
        free(table->keys);
    }

    table->keys = new_keys;
    table->size = new_size;

    return 1;
}

void initHashTable(HashTable *ptr, GLuint size)
{
    if (!ptr)
    {
        return;
    }

    ptr->keys = NULL;
    ptr->current_name = 0;
    ptr->size = 0;

    if (size == 0)
    {
        return;
    }

    if (!ensureHashTableCapacity(ptr, size - 1))
    {
        fprintf(stderr, "MGL ERROR: initHashTable failed to allocate initial capacity %u\n", size);
    }
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
    assert(table);
    
    if (name >= table->size)
    {
        return NULL;
    }

    return table->keys[name].data;
}

void insertHashElement(HashTable *table, GLuint name, void *data)
{
    assert(table);

    if (!ensureHashTableCapacity(table, name))
    {
        return;
    }

    assert(table->keys[name].data == NULL);
    table->keys[name].data = data;
}

void deleteHashElement(HashTable *table, GLuint name)
{
    assert(table);

    if (name >= table->size) {
        fprintf(stderr, "MGL: deleteHashElement - name %u exceeds table size %zu\n", name, table->size);
        return;
    }

    void *obj_data = table->keys[name].data;

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

    table->keys[name].data = NULL;
}
