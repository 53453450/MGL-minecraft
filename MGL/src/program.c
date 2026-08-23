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
 * program.c
 * MGL
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <stdatomic.h>
#include <ctype.h>
#include <malloc/malloc.h>
#include <CoreFoundation/CoreFoundation.h>
#include "mgl_shader_abi.h"
#include "mgl.h"

#include "glm_context.h"
#include "shaders.h"
#include "buffers.h"
#include "mgl_safety.h"
#include "mgl_buffer_slots.h"
#include "mgl_metal_ref.h"
#include "mgl_uniform_reflection.h"
#include "mgl_program_reflection.h"
#include "mgl_program_resource.h"
#include "mgl_sampler_compat.h"
#include "mgl_buffer_plan.h"
#include "mgl_shader_resource.h"
#include "mgl_render.h"


static _Atomic uint64_t mglNextMSLTextureCacheInstanceID = 1u;

static GLboolean mglPointerLooksMallocOwned(const void *ptr)
{
    uintptr_t value = (uintptr_t)ptr;
    if (!ptr || value < 0x10000u) {
        return GL_FALSE;
    }

    return malloc_size(ptr) > 0 ? GL_TRUE : GL_FALSE;
}

static void mglFreeProgramAttribName(Program *program, GLuint index, const char *reason)
{
    if (!program || index >= MAX_ATTRIBS) {
        return;
    }

    char *name = program->attrib_location_names[index];
    GLboolean owned = program->attrib_location_name_owned[index];
    program->attrib_location_names[index] = NULL;
    program->attrib_location_name_owned[index] = GL_FALSE;

    if (!name) {
        return;
    }

    if (owned && mglPointerLooksMallocOwned(name)) {
        free(name);
        return;
    }

    fprintf(stderr,
            "MGL WARNING: skipped invalid attrib name free program=%u index=%u ptr=%p owned=%d reason=%s\n",
            program->name,
            index,
            (void *)name,
            owned,
            reason ? reason : "(unknown)");
}

static GLboolean mglSetProgramAttribName(Program *program, GLuint index, const char *name)
{
    if (!program || index >= MAX_ATTRIBS || !name) {
        return GL_FALSE;
    }

    mglFreeProgramAttribName(program, index, "replace");
    program->attrib_location_names[index] = strdup(name);
    if (!program->attrib_location_names[index]) {
        return GL_FALSE;
    }
    program->attrib_location_name_owned[index] = GL_TRUE;
    return GL_TRUE;
}




// Program Pipeline management
ProgramPipeline *newProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr;

    ptr = (ProgramPipeline *)malloc(sizeof(ProgramPipeline));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate program pipeline %u\n", pipeline);
        return NULL;
    }

    bzero(ptr, sizeof(ProgramPipeline));
    ptr->name = pipeline;

    return ptr;
}

ProgramPipeline *findProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    return (ProgramPipeline *)searchHashTable(&STATE(program_pipeline_table), pipeline);
}

ProgramPipeline *getProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr = findProgramPipeline(ctx, pipeline);

    if (!ptr)
    {
        ptr = newProgramPipeline(ctx, pipeline);
        if (!ptr)
            return NULL;
        insertHashElement(&STATE(program_pipeline_table), pipeline, ptr);
    }

    return ptr;
}

// Transform Feedback management
TransformFeedback *newTransformFeedback(GLMContext ctx, GLuint name)
{
    TransformFeedback *ptr;

    ptr = (TransformFeedback *)malloc(sizeof(TransformFeedback));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate transform feedback %u\n", name);
        return NULL;
    }

    bzero(ptr, sizeof(TransformFeedback));
    ptr->name = name;
    ptr->target = GL_TRANSFORM_FEEDBACK;
    ptr->created = (name == 0) ? GL_TRUE : GL_FALSE;
    ptr->active = GL_FALSE;
    ptr->paused = GL_FALSE;
    ptr->primitive_mode = GL_NONE;

    return ptr;
}

TransformFeedback *findTransformFeedback(GLMContext ctx, GLuint name)
{
    return (TransformFeedback *)searchHashTable(&STATE(transform_feedback_table), name);
}

TransformFeedback *getTransformFeedback(GLMContext ctx, GLuint name)
{
    TransformFeedback *ptr = findTransformFeedback(ctx, name);

    if (!ptr)
    {
        ptr = newTransformFeedback(ctx, name);
        if (!ptr)
            return NULL;
        insertHashElement(&STATE(transform_feedback_table), name, ptr);
    }

    return ptr;
}

Program *newProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    ptr = (Program *)malloc(sizeof(Program));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate program %u\n", program);
        return NULL;
    }

    bzero(ptr, sizeof(Program));

    ptr->name = program;
    ptr->pipeline_cache_instance_id =
        atomic_fetch_add_explicit(&mglNextMSLTextureCacheInstanceID,
                                  1u,
                                  memory_order_relaxed);
    ptr->legacy_clip_plane_loc = -1;
    ptr->legacy_clip_plane_enabled_loc = -1;
    for (GLuint i = 0; i < TEXTURE_UNITS; i++) {
        ptr->sampler_units[i] = -1;
    }
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        for (GLuint i = 0; i < TEXTURE_UNITS; i++) {
            ptr->sampler_units_by_stage[stage][i] = -1;
        }
    }

    return ptr;
}

Program *getProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    if (!ctx || program == 0u)
    {
        return NULL;
    }

    ptr = (Program *)searchHashTable(&STATE(program_table), program);

    if (!ptr)
    {
        ptr = newProgram(ctx, program);
        if (!ptr)
            return NULL;

        insertHashElement(&STATE(program_table), program, ptr);
    }

    return ptr;
}

int isProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    if (!ctx || program == 0u)
    {
        return 0;
    }

    ptr = (Program *)searchHashTable(&STATE(program_table), program);

    if (ptr)
        return 1;

    return 0;
}

Program *findProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    if (!ctx || program == 0u)
    {
        return NULL;
    }

    ptr = (Program *)searchHashTable(&STATE(program_table), program);

    return ptr;
}

GLuint mglCreateProgram(GLMContext ctx)
{
    GLuint program;

    program = getNewName(&STATE(program_table));

    if (!getProgram(ctx, program))
        return 0;

    return program;
}

void mglFreeProgram(GLMContext ctx, Program *ptr)
{
    /* Diagnostic: detect use-after-free via hash table — if the program
     * is still in the hash table when we free it, a later findProgram()
     * will return a dangling pointer.  This should never happen; all
     * callers must deleteHashElement() before mglFreeProgram(). */
    if (ctx && ptr && mglHashTableContainsData(&STATE(program_table), ptr)) {
        fprintf(stderr,
                "MGL BUG: mglFreeProgram name=%u ptr=%p is STILL in hash table — "
                "will cause dangling pointer (refcount=%d delete_status=%d)\n",
                ptr->name, (void *)ptr, ptr->refcount, ptr->delete_status);
        /* Context teardown frees programs that were never deleted through
         * glDeleteProgram; drop the table entry so the table's own teardown
         * does not chase the dangling pointer. */
        deleteHashElement(&STATE(program_table), ptr->name);
    }

    ptr->link_success = GL_FALSE;

    /* Free the buffer binding plan cache before releasing the spirv
     * resources it was built from.  The plan copies reflection values
     * (not pointers to MGLShaderResource), so order doesn't strictly matter,
     * but freeing here keeps the cleanup grouped with other caches. */
    mglBufferBindingPlanDestroy(ptr);
    /* Free the active-uniform cache (pointers into shader_resources_list,
     * so no owned allocations beyond the array itself). */
    mglFreeActiveUniformCache(ptr);

    mglRenderInvalidateProgramPipelines(
        ptr->pipeline_cache_instance_id);

    mglSafeReleaseMetalObj((void **)&ptr->mtl_data);

    for(int i=0; i<_MAX_SHADER_TYPES; i++)
    {
        // CRITICAL FIX: Add NULL checks before all free/release operations to prevent double-frees
        if (ptr->modules[i].metallib_bytes) {
            free(ptr->modules[i].metallib_bytes);
            ptr->modules[i].metallib_bytes = NULL;
            ptr->modules[i].metallib_size = 0;
        }
        if (ptr->modules[i].metallib_tess_capture_bytes) {
            free(ptr->modules[i].metallib_tess_capture_bytes);
            ptr->modules[i].metallib_tess_capture_bytes = NULL;
            ptr->modules[i].metallib_tess_capture_size = 0;
        }
        if (ptr->modules[i].metallib_cull_capture_bytes) {
            free(ptr->modules[i].metallib_cull_capture_bytes);
            ptr->modules[i].metallib_cull_capture_bytes = NULL;
            ptr->modules[i].metallib_cull_capture_size = 0;
        }
        if (ptr->modules[i].entry_point) {
            free(ptr->modules[i].entry_point);
            ptr->modules[i].entry_point = NULL;
        }
        mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_compute_pipeline);
        mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_function);
        mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_library);
        mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_tess_capture_function);
        mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_tess_capture_library);
        mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_cull_capture_function);
        mglSafeReleaseMetalObj((void **)&ptr->modules[i].mtl_cull_capture_library);
        
        for(int j=0; j<MGL_MAX_SHADER_RESOURCES; j++)
        {
            // CRITICAL FIX: Add NULL checks and clear pointers to prevent double-frees
            MGLShaderResourceList *rl = &ptr->shader_resources_list[i][j];
            if (rl->list) {
                for (GLuint k = 0; k < rl->count; k++) {
                    mglFreeMGLShaderResourceOwnedFields(&rl->list[k]);
                }
                free(rl->list);
                rl->list = NULL;
                rl->count = 0;
            }
        }
        
        if (ptr->attached_shader_counts[i] > 0) {
            for (GLuint attached = 0;
                 attached < ptr->attached_shader_counts[i] &&
                 attached < MAX_ATTACHED_SHADERS_PER_STAGE;
                 attached++) {
                Shader *sptr = ptr->attached_shader_slots[i][attached];
                if (!sptr) {
                    continue;
                }
                sptr->refcount--;
                if (sptr->refcount == 0 && sptr->delete_status)
                {
                    deleteHashElement(&STATE(shader_table), sptr->name);
                    mglFreeShader(ctx, sptr);
                }
                ptr->attached_shader_slots[i][attached] = NULL;
            }
            ptr->attached_shader_counts[i] = 0;
            ptr->shader_slots[i] = NULL;
        }
        else if (ptr->shader_slots[i])
        {
                Shader *sptr = ptr->shader_slots[i];
                sptr->refcount--;
                if (sptr->refcount == 0 && sptr->delete_status)
                {
                    deleteHashElement(&STATE(shader_table), sptr->name);
                    mglFreeShader(ctx, sptr);
                }
        }
    }

    for (int i = 0; i < MAX_ATTRIBS; i++) {
        mglFreeProgramAttribName(ptr, (GLuint)i, "program delete");
    }

    for (GLuint i = 0; i < ptr->frag_data_location_count && i < MAX_ATTRIBS; i++) {
        if (ptr->frag_data_location_names[i]) {
            free(ptr->frag_data_location_names[i]);
            ptr->frag_data_location_names[i] = NULL;
        }
    }
    ptr->frag_data_location_count = 0;

    for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
        for (GLuint i = 0; i < ptr->builtin_program_input_count[s] && i < 16; i++) {
            if (ptr->builtin_program_inputs[s][i].name) {
                free(ptr->builtin_program_inputs[s][i].name);
                ptr->builtin_program_inputs[s][i].name = NULL;
            }
        }
        ptr->builtin_program_input_count[s] = 0;
    }

    for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
        for (GLuint i = 0; i < ptr->builtin_program_output_count[s] && i < 16; i++) {
            if (ptr->builtin_program_outputs[s][i].name) {
                free(ptr->builtin_program_outputs[s][i].name);
                ptr->builtin_program_outputs[s][i].name = NULL;
            }
        }
        ptr->builtin_program_output_count[s] = 0;
    }

    free(ptr);
}

GLboolean mglProgramPointerUsableForName(GLMContext ctx, Program *program, GLuint expectedName)
{
    if (!ctx || !program || expectedName == 0u) {
        return GL_FALSE;
    }

    /* Fast path: hashtable membership implies the object is live (programs
     * are removed from the table before free), so the memory is readable and
     * the name can be compared without the vm_region_64 syscall. */
    if (mglHashTableContainsData(&STATE(program_table), program)) {
        return program->name == expectedName;
    }

    if (!mglObjectPointerLooksPlausible(program) ||
        !mglPointerRangeIsReadable(program, sizeof(*program)) ||
        program->name != expectedName) {
        return GL_FALSE;
    }

    /*
     * glDeleteProgram removes the name immediately, but the current program
     * and any deferred draws that captured it must keep using the object until
     * their references are released.
     */
    if (program->delete_status &&
        program->refcount > 0 &&
        program->link_success) {
        return GL_TRUE;
    }

    return GL_FALSE;
}

void mglRetainProgramReference(GLMContext ctx, Program *program)
{
    if (!ctx || !program) {
        return;
    }

    GLuint programName = 0u;
    if (mglObjectPointerLooksPlausible(program) &&
        mglPointerRangeIsReadable(program, sizeof(*program))) {
        programName = program->name;
    }

    if (programName == 0u ||
        !mglProgramPointerUsableForName(ctx, program, programName)) {
        return;
    }

    program->refcount++;
}

void mglReleaseProgramReference(GLMContext ctx, Program *program)
{
    if (!ctx || !program ||
        !mglObjectPointerLooksPlausible(program) ||
        !mglPointerRangeIsReadable(program, sizeof(*program))) {
        return;
    }

    if (program->refcount > 0) {
        program->refcount--;
    }
    if (program->refcount == 0 && program->delete_status) {
        mglFreeProgram(ctx, program);
    }
}

void mglDeleteProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    ptr = findProgram(ctx, program);

    if (!ptr)
    {
        // Per GL spec, glDeleteProgram silently ignores names that do not
        // correspond to an existing program object (including names that were
        // already deleted). Do NOT set a sticky GL error here — leaving
        // GL_INVALID_OPERATION set would poison the next API call's
        // glGetError() check (e.g. glDeleteFramebuffers), surfacing as a
        // spurious CTS crash.
        return;
    }

    mglFlushPendingDraws(ctx);

    deleteHashElement(&STATE(program_table), program);
    
    ptr->delete_status = GL_TRUE;
    
    if (ptr->refcount == 0)
    {
        mglFreeProgram(ctx, ptr);
    }
}

GLboolean mglIsProgram(GLMContext ctx, GLuint program)
{
    if (isProgram(ctx, program))
        return GL_TRUE;

    return GL_FALSE;
}

static GLboolean mglProgramHasAttachedShader(Program *program, GLuint stage, Shader *shader)
{
    if (!program || stage >= _MAX_SHADER_TYPES || !shader) {
        return GL_FALSE;
    }

    for (GLuint i = 0;
         i < program->attached_shader_counts[stage] &&
         i < MAX_ATTACHED_SHADERS_PER_STAGE;
         i++) {
        if (program->attached_shader_slots[stage][i] == shader) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

GLuint mglProgramAttachedShaderCount(Program *program, GLuint stage)
{
    if (!program || stage >= _MAX_SHADER_TYPES) {
        return 0u;
    }

    if (program->attached_shader_counts[stage] > 0u) {
        return program->attached_shader_counts[stage];
    }

    return ((program->attached_shader_mask & (1u << stage)) != 0u &&
            program->shader_slots[stage]) ? 1u : 0u;
}

void mglAttachShader(GLMContext ctx, GLuint program, GLuint shader)
{
    Program *pptr;
    Shader *sptr;
    GLuint index;

    sptr = findShader(ctx, shader);

    if (!sptr)
    {
        // CRITICAL FIX: Handle missing shader gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Shader %u not found in attach shader\n", shader);
        STATE(error) = GL_INVALID_VALUE;
        return;
    }

    pptr = findProgram(ctx, program);

    if (!pptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;

        return;
    }

    index = sptr->glm_type;

    mglFlushPendingDraws(ctx);

    if (mglProgramHasAttachedShader(pptr, index, sptr)) {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    if (pptr->attached_shader_counts[index] >= MAX_ATTACHED_SHADERS_PER_STAGE) {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    if (!pptr->shader_slots[index]) {
        pptr->shader_slots[index] = sptr;
    }

    pptr->attached_shader_slots[index][pptr->attached_shader_counts[index]++] = sptr;
    pptr->attached_shader_mask |= (1u << index);
    sptr->refcount++;
    pptr->dirty_bits |= DIRTY_PROGRAM;
}

void mglDetachShader(GLMContext ctx, GLuint program, GLuint shader)
{
    Program *pptr;
    Shader *sptr;
    GLuint index;

    pptr = findProgram(ctx, program);
    if (!pptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    sptr = findShader(ctx, shader);

    if (!sptr)
    {
        // If not found in hash table, check if it is attached to the program
        for (int i=0; i<_MAX_SHADER_TYPES; i++) {
            for (GLuint attached = 0;
                 attached < pptr->attached_shader_counts[i] &&
                 attached < MAX_ATTACHED_SHADERS_PER_STAGE;
                 attached++) {
                if (pptr->attached_shader_slots[i][attached] &&
                    pptr->attached_shader_slots[i][attached]->name == shader) {
                    sptr = pptr->attached_shader_slots[i][attached];
                    break;
                }
            }
            if (sptr) {
                break;
            }
            if (pptr->shader_slots[i] && pptr->shader_slots[i]->name == shader) {
                sptr = pptr->shader_slots[i];
                break;
            }
        }
    }

    if (!sptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    index = sptr->glm_type;

    GLuint detach_index = MAX_ATTACHED_SHADERS_PER_STAGE;
    for (GLuint attached = 0;
         attached < pptr->attached_shader_counts[index] &&
         attached < MAX_ATTACHED_SHADERS_PER_STAGE;
         attached++) {
        if (pptr->attached_shader_slots[index][attached] == sptr) {
            detach_index = attached;
            break;
        }
    }

    if (detach_index == MAX_ATTACHED_SHADERS_PER_STAGE ||
        (pptr->attached_shader_mask & (1u << index)) == 0u)
    {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    mglFlushPendingDraws(ctx);

    for (GLuint attached = detach_index + 1u;
         attached < pptr->attached_shader_counts[index] &&
         attached < MAX_ATTACHED_SHADERS_PER_STAGE;
         attached++) {
        pptr->attached_shader_slots[index][attached - 1u] =
            pptr->attached_shader_slots[index][attached];
    }
    if (pptr->attached_shader_counts[index] > 0u) {
        pptr->attached_shader_counts[index]--;
        pptr->attached_shader_slots[index][pptr->attached_shader_counts[index]] = NULL;
    }

    if (pptr->attached_shader_counts[index] == 0u) {
        pptr->attached_shader_mask &= ~(1u << index);
        if (!pptr->link_success) {
            pptr->shader_slots[index] = NULL;
        }
    } else if (pptr->shader_slots[index] == sptr) {
        pptr->shader_slots[index] = pptr->attached_shader_slots[index][0];
    }

    /*
     * A successful link creates an executable that survives shader detach and
     * deletion. Keep the shader object as the executable's backing storage;
     * it is released when replaced or when the program is destroyed.
     */
    if (!pptr->link_success) {
        sptr->refcount--;

        if (sptr->refcount == 0 && sptr->delete_status)
        {
            deleteHashElement(&STATE(shader_table), sptr->name);
            mglFreeShader(ctx, sptr);
        }

        pptr->dirty_bits |= DIRTY_PROGRAM;
    }
}


















static GLuint mglTransformFeedbackTypeComponents(GLuint type)
{
    switch (type) {
        case GL_FLOAT: case GL_INT: case GL_UNSIGNED_INT: case GL_BOOL:
        case GL_DOUBLE:
            return 1u;
        case GL_FLOAT_VEC2: case GL_INT_VEC2: case GL_UNSIGNED_INT_VEC2:
        case GL_BOOL_VEC2: case GL_DOUBLE_VEC2:
            return 2u;
        case GL_FLOAT_VEC3: case GL_INT_VEC3: case GL_UNSIGNED_INT_VEC3:
        case GL_BOOL_VEC3: case GL_DOUBLE_VEC3:
            return 3u;
        case GL_FLOAT_VEC4: case GL_INT_VEC4: case GL_UNSIGNED_INT_VEC4:
        case GL_BOOL_VEC4: case GL_DOUBLE_VEC4:
            return 4u;
        case GL_FLOAT_MAT2: case GL_DOUBLE_MAT2:
            return 4u;
        case GL_FLOAT_MAT3: case GL_DOUBLE_MAT3:
            return 9u;
        case GL_FLOAT_MAT4: case GL_DOUBLE_MAT4:
            return 16u;
        case GL_FLOAT_MAT2x3: case GL_FLOAT_MAT3x2:
        case GL_DOUBLE_MAT2x3: case GL_DOUBLE_MAT3x2:
            return 6u;
        case GL_FLOAT_MAT2x4: case GL_FLOAT_MAT4x2:
        case GL_DOUBLE_MAT2x4: case GL_DOUBLE_MAT4x2:
            return 8u;
        case GL_FLOAT_MAT3x4: case GL_FLOAT_MAT4x3:
        case GL_DOUBLE_MAT3x4: case GL_DOUBLE_MAT4x3:
            return 12u;
        default:
            return 0u;
    }
}

static bool mglTransformFeedbackBaseName(const char *name,
                                         char out[96])
{
    if (!name || !name[0] || !out)
        return false;
    const char *bracket = strchr(name, '[');
    size_t baseLength = bracket ? (size_t)(bracket - name) : strlen(name);
    if (baseLength == 0u || baseLength > 95u)
        return false;
    if (!bracket) {
        if (strchr(name, ']') != NULL)
            return false;
    } else {
        char *end = NULL;
        unsigned long element = strtoul(bracket + 1, &end, 10);
        if (end == bracket + 1 || !end || *end != ']' || end[1] != '\0' ||
            element > UINT_MAX)
            return false;
    }
    memcpy(out, name, baseLength);
    out[baseLength] = '\0';
    return true;
}

static bool mglTransformFeedbackArrayElement(const char *name,
                                             GLboolean *isElement,
                                             GLuint *elementIndex)
{
    if (!name || !isElement || !elementIndex)
        return false;
    const char *bracket = strchr(name, '[');
    if (!bracket) {
        *isElement = GL_FALSE;
        *elementIndex = 0u;
        return strchr(name, ']') == NULL;
    }
    char *end = NULL;
    unsigned long element = strtoul(bracket + 1, &end, 10);
    if (end == bracket + 1 || !end || *end != ']' || end[1] != '\0' ||
        element > UINT_MAX)
        return false;
    *isElement = GL_TRUE;
    *elementIndex = (GLuint)element;
    return true;
}

/* Build the link-time XFB scatter plan and validate the ARB_transform_feedback3
 * control entries.  Execution still deliberately gates GS SEPARATE_ATTRIBS
 * elsewhere, but every accepted program now has one authoritative binding,
 * component-offset, and stream assignment plan ready for that route. */
static bool mglValidateTransformFeedbackVaryings(GLMContext ctx, Program *pptr)
{
    if (!pptr)
        return false;

    memset(pptr->transform_feedback_layout, 0,
           sizeof(pptr->transform_feedback_layout));
    pptr->transform_feedback_layout_buffer_count = 0u;
    pptr->transform_feedback_layout_component_count = 0u;
    pptr->transform_feedback_layout_valid = GL_FALSE;
    if (pptr->transform_feedback_varying_count <= 0) {
        pptr->transform_feedback_layout_valid = GL_TRUE;
        return true;
    }

    /* Determine the last active stage before fragment (the stage that
     * provides transform feedback outputs). */
    int feedback_stage = -1;
    if (pptr->attached_shader_mask & GEOMETRY_SHADER_MASK_BIT)
        feedback_stage = _GEOMETRY_SHADER;
    else if (pptr->attached_shader_mask & TESS_EVALUATION_SHADER_MASK_BIT)
        feedback_stage = _TESS_EVALUATION_SHADER;
    else if (pptr->attached_shader_mask & VERTEX_SHADER_MASK_BIT)
        feedback_stage = _VERTEX_SHADER;
    if (feedback_stage < 0)
        return false;

    MGLShaderResourceList *outputs =
        &pptr->shader_resources_list[feedback_stage][_STAGE_OUTPUT_RES];
    const GLuint maxInterleaved = ctx
        ? ctx->state.var.max_transform_feedback_interleaved_components : 64u;
    const GLuint maxSeparateComponents = ctx
        ? ctx->state.var.max_transform_feedback_separate_components : 4u;
    const GLuint maxBuffers = ctx
        ? ctx->state.var.max_transform_feedback_buffers
        : MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS;
    const GLuint maxSeparateAttribs = ctx
        ? ctx->state.var.max_transform_feedback_separate_attribs
        : MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS;
    if (maxBuffers == 0u || maxBuffers > MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS ||
        maxSeparateAttribs == 0u ||
        maxSeparateAttribs > MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS)
        return false;

    GLint bufferStream[MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS];
    GLuint bufferOffsets[MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS] = {0};
    for (GLuint i = 0u; i < MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS; i++)
        bufferStream[i] = -1;
    GLuint buffer = 0u;
    GLuint bufferCount = pptr->transform_feedback_buffer_mode ==
        GL_SEPARATE_ATTRIBS
        ? (GLuint)pptr->transform_feedback_varying_count : 1u;
    if (pptr->transform_feedback_buffer_mode == GL_SEPARATE_ATTRIBS &&
        bufferCount > maxSeparateAttribs)
        return false;

    for (GLsizei i = 0; i < pptr->transform_feedback_varying_count; i++) {
        const char *name = pptr->transform_feedback_varying_names[i];
        if (!name || !name[0])
            return false;

        if (strcmp(name, "gl_NextBuffer") == 0) {
            if (pptr->transform_feedback_buffer_mode != GL_INTERLEAVED_ATTRIBS ||
                buffer + 1u >= maxBuffers)
                return false;
            buffer++;
            bufferOffsets[buffer] = 0u;
            bufferCount = buffer + 1u;
            pptr->transform_feedback_layout[i].buffer_index = buffer;
            pptr->transform_feedback_layout[i].component_offset = 0u;
            pptr->transform_feedback_layout[i].component_count = 0u;
            pptr->transform_feedback_layout[i].stream = -1;
            pptr->transform_feedback_layout[i].builtin = GL_TRUE;
            continue;
        }
        if (strncmp(name, "gl_SkipComponents", 17) == 0) {
            if (pptr->transform_feedback_buffer_mode != GL_INTERLEAVED_ATTRIBS ||
                name[17] < '1' || name[17] > '4' || name[18] != '\0')
                return false;
            GLuint skip = (GLuint)(name[17] - '0');
            GLuint componentOffset = bufferOffsets[buffer];
            if (componentOffset > maxInterleaved ||
                skip > maxInterleaved - componentOffset)
                return false;
            pptr->transform_feedback_layout[i].buffer_index = buffer;
            pptr->transform_feedback_layout[i].component_offset =
                componentOffset;
            pptr->transform_feedback_layout[i].component_count = skip;
            pptr->transform_feedback_layout[i].stream = -1;
            pptr->transform_feedback_layout[i].builtin = GL_TRUE;
            bufferOffsets[buffer] += skip;
            continue;
        }

        char baseName[96];
        if (!mglTransformFeedbackBaseName(name, baseName))
            return false;
        GLboolean isArrayElement = GL_FALSE;
        GLuint arrayElement = 0u;
        if (!mglTransformFeedbackArrayElement(name, &isArrayElement,
                                              &arrayElement))
            return false;
        for (GLsizei prior = 0; prior < i; prior++) {
            char priorName[96];
            const char *priorVarying =
                pptr->transform_feedback_varying_names[prior];
            if (mglTransformFeedbackBaseName(priorVarying, priorName) &&
                strcmp(baseName, priorName) == 0) {
                GLboolean priorIsElement = GL_FALSE;
                GLuint priorElement = 0u;
                if (!mglTransformFeedbackArrayElement(priorVarying,
                                                      &priorIsElement,
                                                      &priorElement))
                    return false;
                if (!isArrayElement || !priorIsElement ||
                    arrayElement == priorElement)
                    return false;
            }
        }

        /* Built-in per-vertex outputs are capturable (GL 4.6 §13.2.4)
         * even though the reflection lists only user varyings. */
        if (strcmp(baseName, "gl_Position") == 0 ||
            strcmp(baseName, "gl_PointSize") == 0) {
            GLuint components = strcmp(baseName, "gl_Position") == 0 ? 4u : 1u;
            if (pptr->transform_feedback_buffer_mode == GL_SEPARATE_ATTRIBS &&
                components > maxSeparateComponents)
                return false;
            pptr->transform_feedback_layout[i].buffer_index = buffer;
            pptr->transform_feedback_layout[i].component_offset =
                bufferOffsets[buffer];
            pptr->transform_feedback_layout[i].component_count = components;
            /* Built-in per-vertex outputs feed stream 0 like any other
             * captured varying. */
            pptr->transform_feedback_layout[i].stream = 0;
            pptr->transform_feedback_layout[i].builtin = GL_TRUE;
            if (pptr->transform_feedback_buffer_mode ==
                GL_INTERLEAVED_ATTRIBS)
                bufferOffsets[buffer] += components;
            continue;
        }

        MGLShaderResource *output = NULL;
        for (GLuint j = 0u; j < outputs->count; j++) {
            if (outputs->list[j].name &&
                strcmp(outputs->list[j].name, baseName) == 0) {
                output = &outputs->list[j];
                break;
            }
        }
        if (!output)
            return false;

        GLuint components = mglTransformFeedbackTypeComponents(output->gl_type);
        GLuint arraySize = output->gl_array_size > 0
            ? (GLuint)output->gl_array_size : 1u;
        if (isArrayElement) {
            if (!output->is_array || arrayElement >= arraySize)
                return false;
            arraySize = 1u;
        }
        if (components == 0u ||
            arraySize > UINT_MAX / components)
            return false;
        components *= arraySize;

        if (pptr->transform_feedback_buffer_mode == GL_SEPARATE_ATTRIBS) {
            buffer = (GLuint)i;
            if (buffer >= maxSeparateAttribs)
                return false;
        }
        GLuint componentOffset = bufferOffsets[buffer];
        if (pptr->transform_feedback_buffer_mode == GL_SEPARATE_ATTRIBS) {
            if (components > maxSeparateComponents)
                return false;
        } else if (componentOffset > maxInterleaved ||
                   components > maxInterleaved - componentOffset) {
            return false;
        }

        GLint stream = output->stream >= 0 ? output->stream : 0;
        if (stream < 0 || stream >= 4)
            return false;
        if (bufferStream[buffer] < 0)
            bufferStream[buffer] = stream;
        else if (bufferStream[buffer] != stream)
            return false;

        MGLTransformFeedbackVaryingPlan *entry =
            &pptr->transform_feedback_layout[i];
        entry->buffer_index = buffer;
        entry->component_offset = componentOffset;
        entry->component_count = components;
        entry->stream = stream;
        entry->builtin = GL_FALSE;
        bufferOffsets[buffer] += components;
    }

    pptr->transform_feedback_layout_buffer_count = bufferCount;
    pptr->transform_feedback_layout_component_count = 0u;
    for (GLuint i = 0u; i < bufferCount; i++) {
        if (bufferOffsets[i] > pptr->transform_feedback_layout_component_count)
            pptr->transform_feedback_layout_component_count = bufferOffsets[i];
    }
    pptr->transform_feedback_layout_valid = GL_TRUE;
    return true;
}

/* AIR path stage compiler: self-hosted frontend + LLVM -> metallib, plus
 * resource reflection.  Returns 1 on success; a failed stage is
 * non-fatal at link time (the stage is simply not renderable), matching
 * the legacy path's behaviour. */
static int mglAirCompileStage(GLMContext ctx, Program *pptr, int stage)
{
    Shader *shader = pptr->shader_slots[stage];
    if (!shader || !shader->src) {
        return 1;
    }
    /* Map the legacy stage numbering (VS=0, TCS=1, TES=2, GS=3, FS=4,
     * CS=5) to the AIR ABI stages (VS=0, FS=1, CS=2, TCS/TES/GS M3). */
    int air_stage;
    switch (stage) {
    case _VERTEX_SHADER:        air_stage = MGL_STAGE_VERTEX; break;
    case _FRAGMENT_SHADER:      air_stage = MGL_STAGE_FRAGMENT; break;
    case _COMPUTE_SHADER:       air_stage = MGL_STAGE_COMPUTE; break;
    case _TESS_CONTROL_SHADER:  air_stage = MGL_STAGE_TESS_CONTROL; break;
    case _TESS_EVALUATION_SHADER: air_stage = MGL_STAGE_TESS_EVALUATION; break;
    case _GEOMETRY_SHADER:      air_stage = MGL_STAGE_GEOMETRY; break;
    default:
        return 1;   /* unsupported stage: skip (link continues) */
    }
    clearStageCompileState(pptr, stage);
    unsigned char *bytes = NULL;
    size_t size = 0;
    char err[512] = {0};
    /* Snapshot the attribute bindings: the source strings are owned by
     * this program and could be released between stage compiles. */
    const char *attrib_snapshot[MAX_ATTRIBS] = {NULL};
    for (int ai = 0; ai < MAX_ATTRIBS; ai++) {
        if (pptr->attrib_location_names[ai]) {
            attrib_snapshot[ai] = strdup(pptr->attrib_location_names[ai]);
        }
    }
    MGLAIRStageInfo stage_info = {0};
    if (stage == _GEOMETRY_SHADER &&
        mglAirReflectGLSLStageInfo(shader->src, air_stage, &stage_info,
                                   err, sizeof err) == 0) {
        pptr->geometry_input_type = stage_info.geometry_input_type;
        pptr->geometry_output_type = stage_info.geometry_output_type;
        pptr->geometry_vertices_out = stage_info.geometry_vertices_out;
        pptr->geometry_invocations = stage_info.geometry_invocations;
        pptr->geometry_stream_count = stage_info.gs_stream_count;
        for (uint32_t si = 0; si < 4u; si++) {
            pptr->geometry_stream_varying_count[si] =
                stage_info.gs_stream_varying_count[si];
            pptr->geometry_stream_xfb_stride[si] =
                stage_info.gs_stream_xfb_stride[si];
        }
    }
    if (stage == _TESS_EVALUATION_SHADER) {
        /* The TES metallib TESS tag must carry the control points per
         * patch (4*cpc + patchKind); Metal uses that compile-time value to
         * compute the per-patch control-point offset, so a 0 here makes
         * every patch read record 0.  Prefer the TCS output vertices when
         * present, else the GL default patch size (3). */
        stage_info.tess_patch_vertices =
            pptr->tess_control_output_vertices > 0u
                ? pptr->tess_control_output_vertices
                : 3u;
    }
    int air_rc = mglAirCompileGLSLWithReflectInfo(
        shader->src, air_stage, attrib_snapshot, &bytes, &size,
        pptr->shader_resources_list[stage], &stage_info, err, sizeof err);
    for (int ai = 0; ai < MAX_ATTRIBS; ai++) {
        free((void *)attrib_snapshot[ai]);
    }
    if (air_rc != 0) {
        fprintf(stderr,
                "MGL WARNING: AIR compile failed program %u stage %d: %s\n",
                pptr->name, stage, err);
        return 0;
    }
    pptr->modules[stage].metallib_bytes = bytes;
    pptr->modules[stage].metallib_size = size;
    if (getenv("MGL_DUMP_AIR") && stage == _GEOMETRY_SHADER) {
        FILE *f = fopen("/tmp/poison_gs.air", "wb");
        if (f) {
            fwrite(bytes, 1, size, f);
            fclose(f);
            fprintf(stderr, "MGL DUMP: gs.air %zu bytes\n", size);
        }
    }
    if (getenv("MGL_DUMP_AIR") && stage == _VERTEX_SHADER) {
        FILE *f = fopen("/tmp/poison_vs.air", "wb");
        if (f) {
            fwrite(bytes, 1, size, f);
            fclose(f);
            fprintf(stderr, "MGL DUMP: vs.air %zu bytes\n", size);
        }
    }
    if (getenv("MGL_DUMP_AIR") && stage == _FRAGMENT_SHADER) {
        FILE *f = fopen("/tmp/poison_fs.air", "wb");
        if (f) {
            fwrite(bytes, 1, size, f);
            fclose(f);
            fprintf(stderr, "MGL DUMP: fs.air %zu bytes\n", size);
        }
    }
    pptr->modules[stage].needs_runtime_array_size_buffer =
        stage_info.needs_runtime_array_size_buffer ? GL_TRUE : GL_FALSE;
    if (stage == _VERTEX_SHADER) {
        pptr->uses_cull_distance = stage_info.uses_cull_distance
            ? GL_TRUE : GL_FALSE;
        pptr->cull_distance_count = stage_info.cull_distance_count;
        pptr->ir_uses_cull_distance = pptr->uses_cull_distance;
        unsigned char *capture_bytes = NULL;
        size_t capture_size = 0;
        char capture_err[512] = {0};
        if (mglShaderCompileGLSLTessCapture(
                shader->src, &capture_bytes, &capture_size,
                capture_err, sizeof capture_err) == 0) {
            pptr->modules[stage].metallib_tess_capture_bytes = capture_bytes;
            pptr->modules[stage].metallib_tess_capture_size = capture_size;
            if (getenv("MGL_DUMP_AIR")) {
                FILE *f = fopen("/tmp/poison_capvar.air", "wb");
                if (f) {
                    fwrite(capture_bytes, 1, capture_size, f);
                    fclose(f);
                    FILE *fsrc =
                        fopen("/tmp/poison_capvar_src.glsl", "wb");
                    if (fsrc) {
                        fwrite(shader->src, 1, strlen(shader->src), fsrc);
                        fclose(fsrc);
                    }
                    fprintf(stderr,
                            "MGL DUMP: capvar.air %zu bytes\n",
                            capture_size);
                }
            }
        } else {
            fprintf(stderr,
                    "MGL WARNING: AIR tess VS capture compile failed "
                    "program %u: %s\n",
                    pptr->name, capture_err);
        }
        if (stage_info.uses_cull_distance) {
            unsigned char *cull_capture_bytes = NULL;
            size_t cull_capture_size = 0;
            char cull_capture_err[512] = {0};
            if (mglShaderCompileGLSLCullDistanceCapture(
                    shader->src, &cull_capture_bytes, &cull_capture_size,
                    cull_capture_err, sizeof cull_capture_err) == 0) {
                pptr->modules[stage].metallib_cull_capture_bytes =
                    cull_capture_bytes;
                pptr->modules[stage].metallib_cull_capture_size =
                    cull_capture_size;
            } else {
                fprintf(stderr,
                        "MGL WARNING: AIR cull-distance capture compile failed "
                        "program %u: %s\n",
                        pptr->name, cull_capture_err);
            }
        }
    }
    if (pptr->modules[stage].entry_point) {
        free(pptr->modules[stage].entry_point);
    }
    pptr->modules[stage].entry_point = strdup("main");
    if (stage == _TESS_CONTROL_SHADER) {
        pptr->tess_control_output_vertices =
            stage_info.tess_control_output_vertices;
    } else if (stage == _TESS_EVALUATION_SHADER) {
        pptr->tess_gen_mode = stage_info.tess_gen_mode;
        pptr->tess_gen_spacing = stage_info.tess_gen_spacing;
        pptr->tess_gen_vertex_order = stage_info.tess_gen_vertex_order;
        pptr->tess_gen_point_mode =
            stage_info.tess_gen_point_mode ? GL_TRUE : GL_FALSE;
        pptr->tess_uses_cull_distance =
            stage_info.uses_cull_distance ? GL_TRUE : GL_FALSE;
        pptr->tess_cull_distance_count = stage_info.cull_distance_count;
    } else if (stage == _GEOMETRY_SHADER) {
        pptr->geometry_input_type = stage_info.geometry_input_type;
        pptr->geometry_output_type = stage_info.geometry_output_type;
        pptr->geometry_vertices_out = stage_info.geometry_vertices_out;
        pptr->geometry_invocations = stage_info.geometry_invocations;
        pptr->geometry_stream_count = stage_info.gs_stream_count;
        for (uint32_t si = 0; si < 4u; si++) {
            pptr->geometry_stream_varying_count[si] =
                stage_info.gs_stream_varying_count[si];
            pptr->geometry_stream_xfb_stride[si] =
                stage_info.gs_stream_xfb_stride[si];
        }
    }
    return 1;
}

/* GL 4.6 §7.4.1 / §11.3.1: the vertex-output and geometry-input
 * interfaces must agree on type, interpolation qualifier and (for
 * explicitly sized arrays) array size.  A GS input with no matching VS
 * output is legal and reads undefined values. */
static GLboolean mglValidateGeometryInterface(Program *pptr)
{
    if (!pptr ||
        (pptr->attached_shader_mask & (1u << _GEOMETRY_SHADER)) == 0u) {
        return GL_TRUE;
    }

    GLuint input_verts;
    switch (pptr->geometry_input_type) {
    case GL_POINTS:               input_verts = 1u; break;
    case GL_LINES:                input_verts = 2u; break;
    case GL_LINE_STRIP:           input_verts = 2u; break;
    case GL_LINES_ADJACENCY:      input_verts = 4u; break;
    case GL_LINE_STRIP_ADJACENCY: input_verts = 4u; break;
    case GL_TRIANGLES:            input_verts = 3u; break;
    case GL_TRIANGLE_STRIP:       input_verts = 3u; break;
    case GL_TRIANGLES_ADJACENCY:  input_verts = 6u; break;
    default:                      input_verts = 3u; break;
    }

    const MGLShaderResourceList *gsIn =
        &pptr->shader_resources_list[_GEOMETRY_SHADER][_STAGE_INPUT_RES];
    const MGLShaderResourceList *vsOut =
        &pptr->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES];
    for (GLuint gi = 0u; gi < gsIn->count; gi++) {
        const MGLShaderResource *in = &gsIn->list[gi];
        if (!in->name || in->name[0] == '\0') continue;
        if (strncmp(in->name, "gl_", 3) == 0) continue;
        if (in->is_array && in->gl_array_size > 0 &&
            (GLuint)in->gl_array_size != input_verts) {
            return GL_FALSE;
        }
        for (GLuint vo = 0u; vo < vsOut->count; vo++) {
            const MGLShaderResource *out = &vsOut->list[vo];
            if (!out->name || strcmp(out->name, in->name) != 0) continue;
            /* Interpolation qualifiers do not participate in the
             * VS->GS interface match (they are only enforced for FS
             * inputs), so only the type is compared here. */
            if (out->gl_type != in->gl_type) {
                return GL_FALSE;
            }
            break;
        }
    }
    /* GL 4.6 §13.8.x: two outputs sharing one explicit location alias,
     * which is only legal for dual-source blending (distinct index).
     * Reject same-location different-name pairs in the GS output list. */
    const MGLShaderResourceList *gsOut =
        &pptr->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES];
    for (GLuint a = 0u; a < gsOut->count; a++) {
        const MGLShaderResource *ra = &gsOut->list[a];
        if (!ra->name || ra->name[0] == '\0') continue;
        if (strncmp(ra->name, "gl_", 3) == 0) continue;
        for (GLuint b = a + 1u; b < gsOut->count; b++) {
            const MGLShaderResource *rb = &gsOut->list[b];
            if (!rb->name || rb->name[0] == '\0') continue;
            if (strncmp(rb->name, "gl_", 3) == 0) continue;
            if (strcmp(ra->name, rb->name) == 0) continue;
            /* Different streams have independent per-stream outputs, so
             * they may reuse the same location. */
            if (ra->stream == rb->stream &&
                ra->location == rb->location &&
                ra->location_index == rb->location_index) {
                return GL_FALSE;
            }
        }
    }
    return GL_TRUE;
}

void mglLinkProgram(GLMContext ctx, GLuint program)
{
    Program *pptr;
    bool link_ok = true;
    bool has_any_shader = false;

    pptr = findProgram(ctx, program);

    if (!pptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;

        return;
    }

    pptr->link_success = GL_FALSE;

    mglFlushPendingDraws(ctx);

    /* C++ compute PSOs retain functions from the previous link generation.
     * Drop them before stage objects and metallib libraries are replaced. */
    mglRenderInvalidateProgramPipelines(
        pptr->pipeline_cache_instance_id);

    pptr->uses_vertex_id = GL_FALSE;
    pptr->uses_primitive_id = GL_FALSE;
    /* Invalidate MSL query result cache; repopulated from the freshly generated MSL
     * after the stage compile loop succeeds. */
    pptr->usesFragCoordParams = GL_FALSE;
    pptr->vertexAttribUsageMask = 0u;
    pptr->uses_point_size_params = GL_FALSE;
    pptr->uses_cull_distance = GL_FALSE;
    pptr->cull_distance_count = 0u;
    pptr->tess_uses_cull_distance = GL_FALSE;
    pptr->tess_cull_distance_count = 0u;
    memset(pptr->validated_resource_lists, 0, sizeof(pptr->validated_resource_lists));
    memset(pptr->validated_resource_list_storage, 0, sizeof(pptr->validated_resource_list_storage));
    memset(pptr->validated_resource_list_counts, 0, sizeof(pptr->validated_resource_list_counts));
    /* Invalidate the buffer binding plan cache: the old plan reflects the
     * pre-link resource list and must not be reused.  The new plan is built
     * from the freshly reflected shader_resources_list at the end of a
     * successful link.  Destroy (rather than just mark invalid) so the
     * stale entries don't linger through a failed link. */
    mglBufferBindingPlanDestroy(pptr);
    /* Invalidate the sampled-texture-unit bitmap: the shader resource list
     * is rebuilt below, so the cached mapping from texture units to sampled
     * resources is stale until the next query rebuilds it. */
    pptr->sampled_texture_unit_mask_valid = 0u;
    /* Invalidate the sampler-binding-shared table; rebuilt from the
     * freshly reflected shader_resources_list after a successful link. */
    pptr->sampler_binding_shared_valid = 0u;
    memset(pptr->sampler_binding_shared, 0, sizeof(pptr->sampler_binding_shared));
    /* Invalidate the sampler-location bitmap; rebuilt at link end. */
    pptr->sampler_location_bitmap_valid = 0u;
    pptr->sampler_location_bitmap[0] = 0u;
    pptr->sampler_location_bitmap[1] = 0u;
    /* Invalidate the active-uniform cache; rebuilt from the freshly
     * reflected shader_resources_list after a successful link. */
    mglFreeActiveUniformCache(pptr);
    /* Invalidate the IR-level reflection cache for buffer slot
     * conflict detection.  Lazily recomputed on first
     * mglBufferSlotConflictsForProgram call during resource binding. */
    pptr->ir_cache_valid = GL_FALSE;
    /* Bump the per-Program cache generation so renderer pipeline keys cannot
     * reuse objects from the previous linked executable. */
    pptr->pipeline_cache_generation++;
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if (mglProgramAttachedShaderCount(pptr, (GLuint)stage) > 0u) {
            has_any_shader = true;
            break;
        }
    }

    if (!has_any_shader) {
        fprintf(stderr, "MGL WARNING: mglLinkProgram called with no attached shaders\n");
        return;
    }

    if ((pptr->attached_shader_mask & COMPUTE_SHADER_MASK_BIT) &&
        (pptr->attached_shader_mask & ~COMPUTE_SHADER_MASK_BIT)) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: compute shaders cannot be linked with non-compute stages\n",
                pptr->name);
        return;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if ((pptr->attached_shader_mask & (1u << stage)) == 0u) {
            continue;
        }

        GLuint attached_count = mglProgramAttachedShaderCount(pptr, (GLuint)stage);
        for (GLuint attached = 0u; attached < attached_count; attached++) {
            Shader *shader = (pptr->attached_shader_counts[stage] > 0u)
                ? pptr->attached_shader_slots[stage][attached]
                : pptr->shader_slots[stage];
            if (!shader || !shader->compile_success) {
                fprintf(stderr,
                        "MGL WARNING: mglLinkProgram failed program %u: shader stage %d is not compiled\n",
                        pptr->name,
                        stage);
                return;
            }
        }
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if (!mglAirCompileStage(ctx, pptr, stage)) {
            link_ok = false;
            break;
        }
    }

    if (!link_ok) {
        return;
    }

    if (!mglValidateGeometryInterface(pptr)) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: vertex "
                "output / geometry input interface mismatch\n",
                pptr->name);
        return;
    }

    /* Validate transform feedback varyings: the link must fail if any
     * captured varying is not an active output of the program. */
    if (!mglValidateTransformFeedbackVaryings(ctx, pptr)) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: transform feedback "
                "varying not found in program outputs\n",
                pptr->name);
        return;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        GLuint attached_count = mglProgramAttachedShaderCount(pptr, (GLuint)stage);
        for (GLuint attached = 0u; attached < attached_count; attached++) {
            Shader *shader = (pptr->attached_shader_counts[stage] > 0u)
                ? pptr->attached_shader_slots[stage][attached]
                : pptr->shader_slots[stage];
            if (!shader || !shader->src) {
                continue;
            }
            if (strstr(shader->src, "gl_VertexID") ||
                strstr(shader->src, "gl_VertexIndex")) {
                pptr->uses_vertex_id = GL_TRUE;
            }
            if (strstr(shader->src, "gl_PrimitiveID") ||
                strstr(shader->src, "gl_PrimitiveIndex")) {
                pptr->uses_primitive_id = GL_TRUE;
            }
        }
    }

    applyVertexInputLocations(pptr);
    applyFragmentOutputLocationIndices(pptr);
    applyMultiDimArrayUniformNames(pptr);
    alignFragmentInputLocationsToVertexOutputs(pptr);
    mglBridgeSkippedGeometryShaderVaryings(pptr);
    mglAssignPlainUniformLocations(pptr);
    if (pptr->modules[_VERTEX_SHADER].metallib_bytes) {
        mglAssignAggregateMemberLocations(pptr);
    }
    mglUnifySamplerUniformLocations(pptr);

    /* The AIR compute expansion consumes fixed 32-byte input records
     * (position + point size).  It accepts every core GS input topology and
     * expands point/line/triangle strips to Metal list primitives. */
    if (pptr->attached_shader_mask & GEOMETRY_SHADER_MASK_BIT) {
        bool computeRoute =
            pptr->modules[_GEOMETRY_SHADER].metallib_bytes != NULL &&
            (pptr->geometry_input_type == GL_POINTS ||
             pptr->geometry_input_type == GL_LINES ||
             pptr->geometry_input_type == GL_LINES_ADJACENCY ||
             pptr->geometry_input_type == GL_TRIANGLES ||
             pptr->geometry_input_type == GL_TRIANGLES_ADJACENCY) &&
            (pptr->geometry_output_type == GL_POINTS ||
             pptr->geometry_output_type == GL_LINE_STRIP ||
             pptr->geometry_output_type == GL_TRIANGLE_STRIP) &&
            pptr->geometry_vertices_out <= 1024u &&
            pptr->geometry_invocations > 0u &&
            pptr->geometry_invocations <= 32u;
        /* GS expansion uses the shared compute-stage binders for UBOs,
         * SSBOs, atomics, sampled textures and storage images.  Resource
         * presence is therefore no longer a route restriction.  XFB runs
         * the GL4 ordered 2-pass path (mgl_air_gs_abi.h §5b): pass 1 packs
         * per-stream records into the stage-out run, pass 2 scatters them
         * by the link-time plan; INTERLEAVED and SEPARATE modes are the
         * same per-buffer packing, so both route here. */
        pptr->gs_route = computeRoute
            ? MGL_GS_ROUTE_COMPUTE : MGL_GS_ROUTE_UNSUPPORTED;
        if (!computeRoute) {
            static uint64_t s_gsUnsupportedNotice = 0;
            uint64_t hit = ++s_gsUnsupportedNotice;
            if (hit <= 4ull) {
                fprintf(stderr,
                        "MGL WARNING: program %u geometry shader is outside "
                        "the AIR compute-expansion subset "
                        "(air=%d in=0x%x out=0x%x max=%u inv=%u xfb=%d)\n",
                        pptr->name,
                        pptr->modules[_GEOMETRY_SHADER].metallib_bytes != NULL,
                        pptr->geometry_input_type,
                        pptr->geometry_output_type,
                        pptr->geometry_vertices_out,
                        pptr->geometry_invocations,
                        pptr->transform_feedback_varying_count);
            }
        }
    } else {
        pptr->gs_route = MGL_GS_ROUTE_NONE;
    }

    if (pptr->program_separable &&
        (pptr->attached_shader_mask & (pptr->attached_shader_mask - 1u)) != 0u &&
        !mglLinkedProgramPerVertexCompatible(pptr)) {
        fprintf(stderr,
                "MGL WARNING: separable program %u has incompatible gl_PerVertex redeclarations\n",
                pptr->name);
        return;
    }

    /* Validate layout(binding=N) values against GL implementation limits.
     * The GL spec requires that the link fail if a binding point exceeds
     * the corresponding MAX_*_BINDINGS limit.  the compiler frontend does not know the
     * implementation limits, so MGL enforces them here. */
    {
        bool binding_error = false;
        for (int stage = 0; stage < _MAX_SHADER_TYPES && !binding_error; stage++) {
            if ((pptr->attached_shader_mask & (1u << stage)) == 0u) {
                continue;
            }

            /* Uniform blocks: GL_MAX_UNIFORM_BUFFER_BINDINGS */
            {
                MGLShaderResourceList *rl =
                    &pptr->shader_resources_list[stage][_UNIFORM_BUFFER_RES];
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint b = rl->list[i].gl_binding;
                    GLuint n = rl->list[i].ubo_array_size > 0
                                   ? rl->list[i].ubo_array_size : 1;
                    if (b + n > ctx->state.var.max_uniform_buffer_bindings) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d UBO '%s' binding %u (array size %u) "
                                "exceeds GL_MAX_UNIFORM_BUFFER_BINDINGS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b, n,
                                ctx->state.var.max_uniform_buffer_bindings);
                        binding_error = true;
                        break;
                    }
                }
            }

            /* Shader storage buffers: GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS */
            if (!binding_error) {
                MGLShaderResourceList *rl =
                    &pptr->shader_resources_list[stage][_STORAGE_BUFFER_RES];
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint b = rl->list[i].gl_binding;
                    if (b >= ctx->state.var.max_shader_storage_buffer_bindings) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d SSBO '%s' binding %u "
                                "exceeds GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b,
                                ctx->state.var.max_shader_storage_buffer_bindings);
                        binding_error = true;
                        break;
                    }
                }
            }

            /* Storage images: GL_MAX_IMAGE_UNITS */
            if (!binding_error) {
                MGLShaderResourceList *rl =
                    &pptr->shader_resources_list[stage][_STORAGE_IMAGE_RES];
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint b = rl->list[i].gl_binding;
                    if (b >= ctx->state.var.max_image_units) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d image '%s' binding %u "
                                "exceeds GL_MAX_IMAGE_UNITS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b,
                                ctx->state.var.max_image_units);
                        binding_error = true;
                        break;
                    }
                }
            }

            /* Atomic counters: GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS */
            if (!binding_error) {
                MGLShaderResourceList *rl =
                    &pptr->shader_resources_list[stage][_ATOMIC_COUNTER_RES];
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint b = rl->list[i].gl_binding;
                    if (b >= ctx->state.var.max_atomic_counter_buffer_bindings) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d atomic counter '%s' binding %u "
                                "exceeds GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b,
                                ctx->state.var.max_atomic_counter_buffer_bindings);
                        binding_error = true;
                        break;
                    }
                }
            }
        }

        /* The AIR backend assigns Metal buffer locations independently from
         * GL layout(binding=N).  Validate those reflected locations against
         * the internal ABI slots used by the program's actual execution
         * paths before marking the link successful.  Delaying this until
         * draw/dispatch would let user buffers silently overwrite GS/TES or
         * emulation data already bound on the same encoder. */
        if (!binding_error) {
            static const int buffer_resource_types[] = {
                _UNIFORM_BUFFER_RES,
                _UNIFORM_CONSTANT_RES,
                _STORAGE_BUFFER_RES,
                _ATOMIC_COUNTER_RES
            };
            for (int stage = 0;
                 stage < _MAX_SHADER_TYPES && !binding_error;
                 stage++) {
                if ((pptr->attached_shader_mask & (1u << stage)) == 0u) {
                    continue;
                }
                for (size_t type_index = 0;
                     type_index < sizeof(buffer_resource_types) /
                                      sizeof(buffer_resource_types[0]) &&
                     !binding_error;
                     type_index++) {
                    int resource_type = buffer_resource_types[type_index];
                    MGLShaderResourceList *resources =
                        &pptr->shader_resources_list[stage][resource_type];
                    for (GLuint resource_index = 0;
                         resource_index < resources->count && !binding_error;
                         resource_index++) {
                        MGLShaderResource *resource =
                            &resources->list[resource_index];
                        if (mglShouldSkipStageBufferResource(
                                pptr, stage, resource_type, resource)) {
                            continue;
                        }
                        GLuint element_count =
                            mglStageBufferResourceElementCount(resource_type,
                                                               resource);
                        if (element_count == 0u) {
                            element_count = 1u;
                        }
                        for (GLuint element = 0; element < element_count;
                             element++) {
                            uint64_t slot64 = (uint64_t)resource->binding +
                                              (uint64_t)element;
                            if (slot64 > UINT32_MAX) {
                                fprintf(stderr,
                                        "MGL LINK ERROR: program %u %s %s '%s' "
                                        "Metal buffer slot overflows uint32\n",
                                        pptr->name,
                                        mglShaderStageName(stage),
                                        mglMGLShaderResourceTypeName(resource_type),
                                        resource->name ? resource->name : "(null)");
                                binding_error = true;
                                break;
                            }
                            GLuint slot = (GLuint)slot64;
                            if (slot >= kMGLMaxMetalUserBufferCount) {
                                fprintf(stderr,
                                        "MGL LINK ERROR: program %u %s %s '%s' "
                                        "Metal buffer slot %u exceeds user slot "
                                        "limit [0, %u)\n",
                                        pptr->name,
                                        mglShaderStageName(stage),
                                        mglMGLShaderResourceTypeName(resource_type),
                                        resource->name ? resource->name : "(null)",
                                        slot,
                                        (unsigned)kMGLMaxMetalUserBufferCount);
                                binding_error = true;
                                break;
                            }
                            if (!mglBufferSlotConflictsForProgram(pptr, slot,
                                                                 stage)) {
                                continue;
                            }
                            const char *reserved =
                                mglBufferSlotReservedName(slot);
                            fprintf(stderr,
                                    "MGL LINK ERROR: program %u %s %s '%s' "
                                    "Metal buffer slot %u conflicts with %s\n",
                                    pptr->name,
                                    mglShaderStageName(stage),
                                    mglMGLShaderResourceTypeName(resource_type),
                                    resource->name ? resource->name : "(null)",
                                    slot,
                                    reserved ? reserved : "an internal buffer");
                            binding_error = true;
                            break;
                        }
                    }
                }
            }
        }

        if (binding_error) {
            return;
        }
    }

    pptr->link_success = GL_TRUE;
    pptr->dirty_bits |= DIRTY_PROGRAM;
    /* Cache the legacy clip-plane uniform locations (the translator injects
     * them only for VS stages that use gl_ClipVertex). */
    pptr->legacy_clip_plane_loc =
        mglGetUniformLocation(ctx, pptr->name, "_mglClipPlane");
    pptr->legacy_clip_plane_enabled_loc =
        mglGetUniformLocation(ctx, pptr->name, "_mglClipPlaneEnabled");
    /* Relink rebuilds the shader resource list, so the program-level
     * sampled-texture-unit bitmap is stale (mglLinkProgram invalidated it
     * earlier) and the merged state-level mask must follow.  glLinkProgram
     * only sets the renderer dirty bit directly, so the state-level cache
     * would otherwise stay valid while the active program changed what it
     * samples. */
    if (ctx->active_state && ctx->active_state->program == pptr) {
        ctx->active_state->active_sampled_texture_unit_mask_valid = 0u;
    }

    /* Populate renderer feature caches from AIR reflection. */
    {
        uint32_t attr_mask = 0u;
        if (pptr->modules[_VERTEX_SHADER].metallib_bytes) {
            MGLShaderResourceList *ins =
                &pptr->shader_resources_list[_VERTEX_SHADER]
                                           [_STAGE_INPUT_RES];
            for (GLuint i = 0; i < ins->count; i++) {
                GLuint loc = ins->list[i].location;
                if (loc < MAX_ATTRIBS) {
                    attr_mask |= (1u << loc);
                }
            }
        }
        pptr->vertexAttribUsageMask = attr_mask;
        pptr->usesFragCoordParams = GL_FALSE;
        pptr->uses_point_size_params = GL_FALSE;
        pptr->uses_lod_bias = GL_FALSE;
    }

    /* Precompute the sampler-binding-shared table.
     * For each Metal binding slot, count how many sampler-like resources
     * (across all stages and the 5 sampler resource types) map to it.  If
     * more than one resource shares a slot, mark sampler_binding_shared[slot]
     * = 1 so mglMetalSamplerSlotSharedAcrossResources can answer in O(1). */
    {
        static const int sampler_res_types[] = {
            _UNIFORM_CONSTANT_RES,
            _SAMPLED_IMAGE_RES,
            _SEPARATE_IMAGE_RES,
            _SEPARATE_SAMPLERS_RES,
            _STORAGE_IMAGE_RES
        };
        unsigned slot_hits[TEXTURE_UNITS];
        memset(slot_hits, 0, sizeof(slot_hits));
        uint64_t loc_bitmap[2] = {0u, 0u};

        for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
            for (size_t rt = 0; rt < sizeof(sampler_res_types) / sizeof(sampler_res_types[0]); rt++) {
                int res_type = sampler_res_types[rt];
                if (res_type < 0 || res_type >= MGL_MAX_SHADER_RESOURCES) {
                    continue;
                }
                MGLShaderResourceList *resources = &pptr->shader_resources_list[stage][res_type];
                if (!resources) {
                    continue;
                }
                for (GLuint i = 0; i < resources->count; i++) {
                    MGLShaderResource *res = &resources->list[i];
                    if (!mglRendererResourceLooksSamplerLike(res, res_type)) {
                        continue;
                    }
                    if (res->binding < TEXTURE_UNITS) {
                        slot_hits[res->binding]++;
                    }
                    /* Set bitmap bits for [uniform_location, uniform_location+array_size). */
                    if (res->uniform_location >= 0) {
                        GLint array_size = res->gl_array_size > 0 ? res->gl_array_size : 1;
                        for (GLint off = 0; off < array_size; off++) {
                            GLint loc = res->uniform_location + off;
                            if (loc >= 0 && loc < 128) {
                                loc_bitmap[(GLuint)loc >> 6] |= (1ull << ((GLuint)loc & 63u));
                            }
                        }
                    }
                }
            }
        }

        for (GLuint slot = 0; slot < TEXTURE_UNITS; slot++) {
            pptr->sampler_binding_shared[slot] = (slot_hits[slot] > 1u) ? 1u : 0u;
        }
        pptr->sampler_binding_shared_valid = 1u;
        pptr->sampler_location_bitmap[0] = loc_bitmap[0];
        pptr->sampler_location_bitmap[1] = loc_bitmap[1];
        pptr->sampler_location_bitmap_valid = 1u;
    }

    /* Build the buffer binding plan from the finalized shader_resources_list.
     * The plan caches reflection-derived data (metal slots, client bindings,
     * struct packing metadata) so per-draw mapGLBuffersToMTLBufferMap paths
     * can skip repeated name lookups and program resolution.  See
     * mgl_buffer_plan.h for the full cache contract. */
    mglBufferBindingPlanBuild(pptr);

    /* Build the deduplicated active-uniform cache from the finalized
     * shader_resources_list.  Eliminates O(N^3) dedup-on-every-query in
     * mglProgramActiveUniformCount / At / IndexByName / MaxNameLength. */
    mglBuildActiveUniformCache(pptr);

    mglRendererBindProgram(ctx, pptr);

    //ERROR_CHECK_RETURN(pptr->mtl_data, GL_INVALID_OPERATION);
}

void mglUseProgram(GLMContext ctx, GLuint program)
{
    Program *pptr = NULL;
    static GLuint s_last_unlinked_program = 0;
    static unsigned int s_unlinked_program_hits = 0;

    if (!ctx) {
        return;
    }

    if (ctx->state.program_name == program &&
        ((program == 0u && ctx->state.program == NULL) ||
         (program != 0u && ctx->state.program != NULL))) {
        return;
    }

    if (program)
    {
        pptr = findProgram(ctx, program);

        if (!pptr) {
            fprintf(stderr, "MGL DIAG mglUseProgram program=%u FAIL findProgram=NULL table_count=%zu table_size=%zu\n",
                    program, STATE(program_table).count, STATE(program_table).size);
        } else if (!mglObjectPointerLooksPlausible(pptr)) {
            fprintf(stderr, "MGL DIAG mglUseProgram program=%u FAIL looksPlausible pptr=%p\n",
                    program, (void*)pptr);
        } else if (!mglHashTableContainsData(&STATE(program_table), pptr)) {
            fprintf(stderr, "MGL DIAG mglUseProgram program=%u FAIL containsData pptr=%p table_count=%zu\n",
                    program, (void*)pptr, STATE(program_table).count);
        } else if (!mglPointerRangeIsReadable(pptr, sizeof(*pptr))) {
            /* Get vm_region_64 details to distinguish ASan mprotect from
             * true use-after-free (munmap).  Do NOT dereference pptr — it
             * may point to freed/poisoned memory. */
            vm_address_t raddr = (vm_address_t)pptr;
            vm_size_t rsize = 0;
            vm_region_basic_info_data_64_t rinfo;
            mach_msg_type_number_t rcount = VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t robj = MACH_PORT_NULL;
            kern_return_t rkr = vm_region_64(mach_task_self(), &raddr, &rsize,
                                             VM_REGION_BASIC_INFO_64,
                                             (vm_region_info_t)&rinfo, &rcount, &robj);
            if (robj != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), robj);
            fprintf(stderr, "MGL DIAG mglUseProgram program=%u FAIL notReadable pptr=%p size=%zu "
                    "kr=%d region_addr=0x%lx region_size=%lu prot=%u max_prot=%u\n",
                    program, (void*)pptr, sizeof(*pptr),
                    rkr, (unsigned long)raddr, (unsigned long)rsize,
                    rinfo.protection, rinfo.max_protection);
        }

        if (!pptr ||
            !mglObjectPointerLooksPlausible(pptr) ||
            !mglHashTableContainsData(&STATE(program_table), pptr) ||
            !mglPointerRangeIsReadable(pptr, sizeof(*pptr)))
        {
            fprintf(stderr, "MGL Error: mglUseProgram program %u not found or invalid\n", program);
            // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;

            return;
        }

        if (!pptr->link_success)
        {
            // Compatibility fallback: some pipelines can probe/use programs before
            // link is completed/available in this backend. Skip instead of poisoning
            // global GL error state every frame.
            s_unlinked_program_hits++;
            if (s_last_unlinked_program != program || (s_unlinked_program_hits % 128u) == 1u) {
                fprintf(stderr, "MGL WARNING: mglUseProgram skipping unlinked program %u (hit=%u)\n",
                        program, s_unlinked_program_hits);
                s_last_unlinked_program = program;
            }
            return;
        }
    }
    else
    {
        pptr = NULL;
    }

    bool bindingChanged =
        ctx->state.program != pptr ||
        ctx->state.program_name != program;

    if (bindingChanged)
    {
        Program *oldProgram = ctx->state.program;
        if (oldProgram &&
            !mglProgramPointerUsableForName(ctx,
                                            oldProgram,
                                            oldProgram->name ? oldProgram->name : ctx->state.program_name))
        {
            fprintf(stderr, "MGL WARNING: mglUseProgram dropping invalid cached program pointer %p\n",
                    (void *)oldProgram);
            oldProgram = NULL;
            ctx->state.program = NULL;
        }

        if (oldProgram)
        {
            oldProgram->refcount--;
            if (oldProgram->refcount == 0 && oldProgram->delete_status)
            {
                mglFreeProgram(ctx, oldProgram);
            }
        }

        ctx->state.program = pptr;

        if (pptr)
        {
            pptr->refcount++;
        }
        mglMarkStateDirtyBits(&ctx->state, DIRTY_PROGRAM);
    }

    /*
     * Keep program name and pointer state in sync so renderer-side recovery can
     * re-resolve by name if the cached pointer is lost.
     */
    ctx->state.program_name = program;

}

void mglBindAttribLocation(GLMContext ctx, GLuint program, GLuint index, const GLchar *name)
{
    if (index >= MAX_ATTRIBS || !name) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!mglSetProgramAttribName(ptr, index, name)) {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }

    ptr->dirty_bits |= DIRTY_PROGRAM;
}

void mglGetActiveAttrib(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    if (length) {
        *length = 0;
    }
    if (size) {
        *size = 0;
    }
    if (type) {
        *type = 0;
    }
    if (name && bufSize > 0) {
        name[0] = '\0';
    }

    if (!ctx) {
        return;
    }
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!ptr->link_success) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    MGLShaderResource *res = mglProgramActiveAttribAt(ptr, index);
    if (!res) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (size) {
        *size = 1;
    }
    if (type) {
        *type = mglProgramActiveAttribType(res);
    }

    const char *src = res->name ? res->name : "";
    GLsizei src_len = (GLsizei)strlen(src);
    if (length) {
        *length = src_len;
    }
    if (name && bufSize > 0) {
        GLsizei copy_len = src_len < (bufSize - 1) ? src_len : (bufSize - 1);
        if (copy_len > 0) {
            memcpy(name, src, (size_t)copy_len);
        }
        name[copy_len] = '\0';
    }
}

void mglGetActiveUniform(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    if (length) {
        *length = 0;
    }
    if (size) {
        *size = 0;
    }
    if (type) {
        *type = 0;
    }
    if (name && bufSize > 0) {
        name[0] = '\0';
    }

    if (!ctx) {
        return;
    }
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!ptr->link_success) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    int stage = -1;
    int res_type = -1;
    MGLShaderResource *res = mglProgramActiveUniformAt(ptr, index, &stage, &res_type);
    (void)stage;
    if (!res) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (size) {
        *size = res->ubo_member ? res->ubo_member->size
                                : mglProgramActiveUniformSize(res, res_type);
    }
    if (type) {
        *type = res->ubo_member ? (GLenum)res->ubo_member->gl_type
                                : (GLenum)mglProgramActiveUniformGLType(res, res_type);
    }
    mglProgramCopyActiveUniformName(res, bufSize, length, name);
}

void mglGetAttachedShaders(GLMContext ctx, GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders)
{
    if (count) {
        *count = 0;
    }
    if (!ctx) {
        return;
    }
    if (maxCount < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    GLsizei written = 0;
    for (int i = 0; i < _MAX_SHADER_TYPES; i++) {
        GLuint attached_count = mglProgramAttachedShaderCount(ptr, (GLuint)i);
        for (GLuint attached = 0u; attached < attached_count; attached++) {
            Shader *shader = (ptr->attached_shader_counts[i] > 0u)
                ? ptr->attached_shader_slots[i][attached]
                : ptr->shader_slots[i];
            if (!shader) {
                continue;
            }
            if (written < maxCount) {
                if (shaders) {
                    shaders[written] = shader->name;
                }
                written++;
            }
        }
    }

    if (count) {
        *count = written;
    }
}

GLint  mglGetAttribLocation(GLMContext ctx, GLuint program, const GLchar *name)
{
	if (isProgram(ctx, program) == GL_FALSE)
	{
		ERROR_RETURN(GL_INVALID_OPERATION); // also may be GL_INVALID_VALUE ????

		return -1;
	}

	Program *ptr;

	ptr = getProgram(ctx, program);
	if (!ptr)
	{
		ERROR_RETURN(GL_INVALID_OPERATION);
		return -1;
	}

	if (!ptr->link_success)
	{
		ERROR_RETURN(GL_INVALID_OPERATION);

		return -1;
	}

    MGLShaderResourceList *vertex_inputs =
        &ptr->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];

    /* Fast path: exact name match. */
    for (GLuint i = 0; vertex_inputs->list && i < vertex_inputs->count; i++)
    {
        const char *str = vertex_inputs->list[i].name;

        if (str && !strcmp(str, name))
        {
            return (GLint)vertex_inputs->list[i].location;
        }
    }

    /* Array element query: "foo[N]" should resolve to location(foo) + N.
     * Per the GL spec, glGetAttribLocation accepts "foo[0]" (equivalent to
     * "foo") and "foo[N]" for the Nth element of an array attribute. */
    const char *bracket = strrchr(name, '[');
    if (bracket && bracket[1] != ']')
    {
        size_t base_len = (size_t)(bracket - name);
        char *endp = NULL;
        long idx = strtol(bracket + 1, &endp, 10);
        if (endp && *endp == ']' && idx >= 0)
        {
            for (GLuint i = 0; vertex_inputs->list && i < vertex_inputs->count; i++)
            {
                const char *str = vertex_inputs->list[i].name;
                if (str && strlen(str) == base_len &&
                    !strncmp(str, name, base_len))
                {
                    return (GLint)vertex_inputs->list[i].location + (GLint)idx;
                }
            }
        }
    }

	return -1;
}

void mglGetProgramiv(GLMContext ctx, GLuint program, GLenum pname, GLint *params)
{
    Program *pptr = findProgram(ctx, program);
    ERROR_CHECK_RETURN(pptr, GL_INVALID_VALUE);
    
    switch (pname) {
        case GL_LINK_STATUS:
            *params = pptr->link_success ? GL_TRUE : GL_FALSE;
            break;
        case GL_DELETE_STATUS:
            *params = GL_FALSE;  /* Programs are not deleted by default */
            break;
        case GL_VALIDATE_STATUS:
            *params = GL_TRUE;  /* Assume valid */
            break;
        case GL_INFO_LOG_LENGTH:
            *params = 0;  /* No info log for now */
            break;
        case GL_ATTACHED_SHADERS:
            {
                int count = 0;
                for (int i = 0; i < _MAX_SHADER_TYPES; i++) {
                    count += (int)mglProgramAttachedShaderCount(pptr, (GLuint)i);
                }
                *params = count;
            }
            break;
        case GL_ACTIVE_ATTRIBUTES:
            *params = mglProgramActiveAttribCount(pptr);
            break;
        case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH:
            *params = mglProgramActiveAttribMaxNameLength(pptr);
            break;
        case GL_ACTIVE_UNIFORMS:
            *params = mglProgramActiveUniformCount(pptr);
            break;
        case GL_ACTIVE_UNIFORM_MAX_LENGTH:
            *params = mglProgramActiveUniformMaxNameLength(pptr);
            break;
        case GL_ACTIVE_UNIFORM_BLOCKS:
            *params = mglActiveUniformBlockCount(pptr);
            break;
        case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH:
            *params = mglActiveUniformBlockMaxNameLength(pptr);
            break;
        case GL_COMPUTE_WORK_GROUP_SIZE:
            /*
             * Per the spec, querying GL_COMPUTE_WORK_GROUP_SIZE on a program
             * with no linked compute stage must return {0,0,0}; it does NOT
             * generate an error.  GL_INVALID_OPERATION is raised when the
             * program itself is not linked.
             */
            if (!pptr->link_success) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            if (pptr->shader_slots[_COMPUTE_SHADER]) {
                params[0] = pptr->local_workgroup_size.x;
                params[1] = pptr->local_workgroup_size.y;
                params[2] = pptr->local_workgroup_size.z;
            } else {
                params[0] = 0;
                params[1] = 0;
                params[2] = 0;
            }
            break;
        case GL_ACTIVE_ATOMIC_COUNTER_BUFFERS:
            *params = mglActiveAtomicCounterBufferCount(pptr);
            break;
        case GL_GEOMETRY_INPUT_TYPE:        /* 0x8917 */
        case GL_GEOMETRY_OUTPUT_TYPE:       /* 0x8918 */
        case GL_GEOMETRY_VERTICES_OUT:      /* 0x8916 */
        case GL_GEOMETRY_SHADER_INVOCATIONS:/* 0x887F */
            if (!pptr->link_success) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            if (!pptr->shader_slots[_GEOMETRY_SHADER]) {
                *params = 0;
            } else if (pname == GL_GEOMETRY_INPUT_TYPE) {
                *params = (GLint)pptr->geometry_input_type;
            } else if (pname == GL_GEOMETRY_OUTPUT_TYPE) {
                *params = (GLint)pptr->geometry_output_type;
            } else if (pname == GL_GEOMETRY_VERTICES_OUT) {
                *params = (GLint)pptr->geometry_vertices_out;
            } else {
                *params = (GLint)pptr->geometry_invocations;
            }
            break;
        case GL_TESS_CONTROL_OUTPUT_VERTICES:  /* 0x8E75 */
            if (!pptr->link_success) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            *params = pptr->shader_slots[_TESS_CONTROL_SHADER]
                ? (GLint)pptr->tess_control_output_vertices : 0;
            break;
        case GL_TESS_GEN_MODE:             /* 0x8E76 */
        case GL_TESS_GEN_SPACING:          /* 0x8E77 */
        case GL_TESS_GEN_VERTEX_ORDER:     /* 0x8E78 */
        case GL_TESS_GEN_POINT_MODE:       /* 0x8E79 */
            /* TES execution-mode reflection.  Returns the layout(...) values
             * captured from AIR tessellation metadata at link time.  0 when no
             * TES is attached. */
            if (!pptr->link_success) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            if (!pptr->shader_slots[_TESS_EVALUATION_SHADER]) {
                *params = 0;
            } else if (pname == GL_TESS_GEN_MODE) {
                *params = (GLint)pptr->tess_gen_mode;
            } else if (pname == GL_TESS_GEN_SPACING) {
                *params = (GLint)pptr->tess_gen_spacing;
            } else if (pname == GL_TESS_GEN_VERTEX_ORDER) {
                *params = (GLint)pptr->tess_gen_vertex_order;
            } else {
                *params = (GLint)pptr->tess_gen_point_mode;
            }
            break;
        case GL_COMPLETION_STATUS_KHR: /* GL_ARB/KHR_parallel_shader_compile */
            /* MGL links synchronously, so every program is always complete by
             * the time this query is issued. */
            *params = GL_TRUE;
            break;
        case GL_TRANSFORM_FEEDBACK_VARYINGS:        /* 0x8C83 */
        case GL_TRANSFORM_FEEDBACK_BUFFER_MODE:     /* 0x8C7F */
        case GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH: /* 0x8C76 */
            /* ARB_transform_feedback3 reflection.  Special names
             * (gl_NextBuffer, gl_SkipComponentsN) count as varyings. */
            if (!pptr->link_success) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            if (pname == GL_TRANSFORM_FEEDBACK_VARYINGS) {
                *params = pptr->transform_feedback_varying_count;
            } else if (pname == GL_TRANSFORM_FEEDBACK_BUFFER_MODE) {
                /* 0 means never specified with glTransformFeedbackVaryings;
                 * the spec default is GL_INTERLEAVED_ATTRIBS. */
                *params = pptr->transform_feedback_buffer_mode != 0
                    ? (GLint)pptr->transform_feedback_buffer_mode
                    : (GLint)GL_INTERLEAVED_ATTRIBS;
            } else {
                GLint maxLen = 0;
                for (GLsizei vi = 0;
                     vi < pptr->transform_feedback_varying_count; vi++) {
                    const char *vname =
                        pptr->transform_feedback_varying_names[vi];
                    GLint len = vname ? (GLint)strlen(vname) + 1 : 0;
                    if (len > maxLen) maxLen = len;
                }
                *params = maxLen;
            }
            break;
        default:
            fprintf(stderr, "mglGetProgramiv: unhandled pname 0x%x\n", pname);
            *params = 0;
            break;
    }
}

void mglGetProgramInfoLog(GLMContext ctx, GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    Program *pptr = findProgram(ctx, program);
    ERROR_CHECK_RETURN(pptr, GL_INVALID_VALUE);
    
    /* For now, always return an empty info log */
    if (bufSize > 0 && infoLog) {
        infoLog[0] = '\0';
        if (length) {
            *length = 0;
        }
    }
}



#pragma mark program pipelines
void mglGenProgramPipelines(GLMContext ctx, GLsizei n, GLuint *pipelines)
{
    for (GLsizei i = 0; i < n; i++)
    {
        pipelines[i] = getNewName(&STATE(program_pipeline_table));
        getProgramPipeline(ctx, pipelines[i]);
    }
}

GLboolean mglIsProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr = findProgramPipeline(ctx, pipeline);
    return ptr ? GL_TRUE : GL_FALSE;
}

void mglDeleteProgramPipelines(GLMContext ctx, GLsizei n, const GLuint *pipelines)
{
    mglFlushPendingDraws(ctx);

    for (GLsizei i = 0; i < n; i++)
    {
        if (pipelines[i] == 0)
            continue;
            
        ProgramPipeline *ptr = findProgramPipeline(ctx, pipelines[i]);
        if (!ptr)
            continue;
            
        // If deleting currently bound pipeline, unbind it
        if (STATE(program_pipeline) && STATE(program_pipeline)->name == pipelines[i])
        {
            STATE(program_pipeline) = NULL;
            STATE(var.program_pipeline_binding) = 0;
            mglMarkStateDirtyBits(ctx->active_state, DIRTY_PROGRAM);
        }
        
        /* Release every retained stage program reference before freeing
         * the pipeline struct (matches the per-slot retain in
         * mglUseProgramStages). */
        for (int s = 0; s < _MAX_SHADER_TYPES; s++)
        {
            Program *stage_prog = ptr->stage_programs[s];
            ptr->stage_programs[s] = NULL;
            if (stage_prog)
                mglReleaseProgramReference(ctx, stage_prog);
        }

        // Remove from hash table and free
        deleteHashElement(&STATE(program_pipeline_table), pipelines[i]);
        free(ptr);
    }
}

void mglBindProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    if (pipeline == 0)
    {
        STATE(program_pipeline) = NULL;
        STATE(var.program_pipeline_binding) = 0;
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_PROGRAM);
        return;
    }
    
    ProgramPipeline *ptr = getProgramPipeline(ctx, pipeline);
    STATE(program_pipeline) = ptr;
    STATE(var.program_pipeline_binding) = ptr ? pipeline : 0;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_PROGRAM);
}

void mglUseProgramStages(GLMContext ctx, GLuint pipeline, GLbitfield stages, GLuint program)
{
    ProgramPipeline *pipe_ptr = findProgramPipeline(ctx, pipeline);
    if (!pipe_ptr)
    {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    mglFlushPendingDraws(ctx);

    Program *prog_ptr = NULL;
    if (program != 0)
    {
        prog_ptr = findProgram(ctx, program);
        if (!prog_ptr)
        {
            STATE(error) = GL_INVALID_VALUE;
            return;
        }
    }
    
    /* Attach program to specified stages.  Each stage slot owns an
     * independent reference: retain the new program BEFORE releasing the
     * old one so re-attaching a program that is already in a slot (or is
     * the only reference keeping it alive) does not free it mid-op. */
#define MGL_REPLACE_STAGE_SLOT(slot)                                         \
    do {                                                                     \
        Program *_old = pipe_ptr->stage_programs[(slot)];                    \
        if (prog_ptr)                                                        \
            mglRetainProgramReference(ctx, prog_ptr);                        \
        pipe_ptr->stage_programs[(slot)] = prog_ptr;                         \
        if (_old)                                                            \
            mglReleaseProgramReference(ctx, _old);                           \
    } while (0)

    if (stages & GL_VERTEX_SHADER_BIT)
        MGL_REPLACE_STAGE_SLOT(_VERTEX_SHADER);
    if (stages & GL_FRAGMENT_SHADER_BIT)
        MGL_REPLACE_STAGE_SLOT(_FRAGMENT_SHADER);
    if (stages & GL_GEOMETRY_SHADER_BIT)
        MGL_REPLACE_STAGE_SLOT(_GEOMETRY_SHADER);
    if (stages & GL_TESS_CONTROL_SHADER_BIT)
        MGL_REPLACE_STAGE_SLOT(_TESS_CONTROL_SHADER);
    if (stages & GL_TESS_EVALUATION_SHADER_BIT)
        MGL_REPLACE_STAGE_SLOT(_TESS_EVALUATION_SHADER);
    if (stages & GL_COMPUTE_SHADER_BIT)
        MGL_REPLACE_STAGE_SLOT(_COMPUTE_SHADER);

#undef MGL_REPLACE_STAGE_SLOT

    pipe_ptr->validated = GL_FALSE;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_PROGRAM);
}
