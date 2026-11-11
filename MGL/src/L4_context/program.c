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
#include "mgl_uniform_reflection.h"
#include "mgl_program_reflection.h"
#include "mgl_program_resource.h"
#include "mgl_sampler_compat.h"
#include "mgl_buffer_plan.h"
#include "mgl_shader_resource.h"
#include "mgl_compile_artifact.h"
#include "mgl_render_api_binding.h"
#include "mgl_env_flag.h"
#include "mgl_render_api_program.h"





static _Atomic uint64_t mglNextMSLTextureCacheInstanceID = 1u;

/* Drop CPU-side plain-uniform buffers that cache glUniform* writes. */

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
    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        if (i == index || !program->attrib_location_names[i]) {
            continue;
        }
        if (strcmp(program->attrib_location_names[i], name) == 0) {
            mglFreeProgramAttribName(program, i, "rebind");
        }
    }
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
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        fprintf(stderr, "MGL ERROR: failed to allocate program pipeline %u\n", pipeline);
        return NULL;
    }

    bzero(ptr, sizeof(ProgramPipeline));
    ptr->name = pipeline;

    return ptr;
}

ProgramPipeline *findProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr =
        (ProgramPipeline *)searchHashTable(&STATE(program_pipeline_table), pipeline);
    if (!ptr || !ptr->created)
        return NULL;
    return ptr;
}

ProgramPipeline *getProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr = (ProgramPipeline *)searchHashTable(&STATE(program_pipeline_table), pipeline);

    if (!ptr)
    {
        ptr = newProgramPipeline(ctx, pipeline);
        if (!ptr)
            return NULL;
        insertHashElement(&STATE(program_pipeline_table), pipeline, ptr);
    }

    return ptr;
}

static ProgramPipeline *ensureProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr =
        (ProgramPipeline *)searchHashTable(&STATE(program_pipeline_table), pipeline);
    if (!ptr)
        return NULL;
    ptr->created = GL_TRUE;
    return ptr;
}

// Transform Feedback management
TransformFeedback *newTransformFeedback(GLMContext ctx, GLuint name)
{
    TransformFeedback *ptr;

    ptr = (TransformFeedback *)malloc(sizeof(TransformFeedback));
    if (!ptr) {
        if (ctx)
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
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
            mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        fprintf(stderr, "MGL ERROR: failed to allocate program %u\n", program);
        return NULL;
    }

    bzero(ptr, sizeof(Program));

    ptr->name = program;
    mglRenderProgramAssignPipelineCacheInstanceId(
        ptr,
        atomic_fetch_add_explicit(&mglNextMSLTextureCacheInstanceID, 1u,
                                  memory_order_relaxed));
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
    /* GL §7.1: shader and program objects share one name space. */
    while (program != 0 &&
           searchHashTable(&STATE(shader_table), program) != NULL) {
        program = getNewName(&STATE(program_table));
    }

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
    ptr->link_state = MGL_PROGRAM_LINK_NONE;

    /* Free the buffer binding plan cache before releasing the spirv
     * resources it was built from.  The plan copies reflection values
     * (not pointers to MGLShaderResource), so order doesn't strictly matter,
     * but freeing here keeps the cleanup grouped with other caches. */
    mglBufferBindingPlanDestroy(ptr);
    /* Free the active-uniform cache (pointers into shader_resources_list,
     * so no owned allocations beyond the array itself). */
    mglFreeActiveUniformCache(ptr);

    mglRenderInvalidateProgramPipelines(
        mglRenderProgramPipelineCacheInstanceId(ptr));

    mglRenderProgramReleaseBackend(ptr);
    for(int i=0; i<_MAX_SHADER_TYPES; i++)
    {
        if (ptr->modules[i].entry_point) {
            free(ptr->modules[i].entry_point);
            ptr->modules[i].entry_point = NULL;
        }

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

    free(ptr->debug_label);
    free(ptr);
}

GLboolean mglProgramPointerUsableForName(GLMContext ctx, Program *program, GLuint expectedName)
{
    if (!ctx || !program || expectedName == 0u) {
        return GL_FALSE;
    }

    /* Fast path: hashtable membership implies the object is live (programs
     * are removed from the table before free), so the name can be compared
     * directly. */
    if (mglHashTableContainsData(&STATE(program_table), program)) {
        return program->name == expectedName;
    }

    /* Slow path: not a table member.  Only a plausible pointer whose own
     * name matches the caller-supplied known name may still be a deleted
     * program whose references have not drained yet. */
    if (!mglObjectPointerLooksPlausible(program) ||
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

    /* Callers pass programs they already hold (state bindings, pipeline
     * stage slots), so the pointer is dereferenceable and its own name is
     * the known name (decision D6). */
    if (!mglProgramPointerUsableForName(ctx, program, program->name)) {
        return;
    }

    program->refcount++;
}

void mglReleaseProgramReference(GLMContext ctx, Program *program)
{
    if (!ctx || !program) {
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

GLboolean mglProgramHasAttachedShader(Program *program, GLuint stage, Shader *shader)
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
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_VALUE);
        return;
    }

    pptr = findProgram(ctx, program);

    if (!pptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);

        return;
    }

    index = sptr->glm_type;

    mglFlushPendingDraws(ctx);

    if (mglProgramHasAttachedShader(pptr, index, sptr)) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (pptr->attached_shader_counts[index] >= MAX_ATTACHED_SHADERS_PER_STAGE) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
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
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
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
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
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
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
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


















GLuint mglTransformFeedbackTypeComponents(GLuint type)
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

bool mglTransformFeedbackBaseName(const char *name,
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

GLboolean mglTransformFeedbackArrayElement(const char *name,
                                           GLboolean *isElement,
                                           GLuint *elementIndex)
{
    if (!name || !isElement || !elementIndex)
        return GL_FALSE;
    const char *bracket = strchr(name, '[');
    if (!bracket) {
        *isElement = GL_FALSE;
        *elementIndex = 0u;
        return strchr(name, ']') == NULL ? GL_TRUE : GL_FALSE;
    }
    char *end = NULL;
    unsigned long element = strtoul(bracket + 1, &end, 10);
    if (end == bracket + 1 || !end || *end != ']' || end[1] != '\0' ||
        element > UINT_MAX)
        return false;
    *isElement = GL_TRUE;
    *elementIndex = (GLuint)element;
    return GL_TRUE;
}

MGLShaderResource *mglProgramFindStageOutputForXFBName(Program *program,
                                                       int stage,
                                                       const char *xfb_name)
{
    if (!program || !xfb_name || !xfb_name[0] ||
        stage < 0 || stage >= _MAX_SHADER_TYPES)
        return NULL;
    char baseName[96];
    if (!mglTransformFeedbackBaseName(xfb_name, baseName))
        return NULL;
    MGLShaderResourceList *outputs =
        &program->shader_resources_list[stage][_STAGE_OUTPUT_RES];
    for (GLuint j = 0u; j < outputs->count; j++) {
        if (outputs->list[j].name &&
            strcmp(outputs->list[j].name, baseName) == 0)
            return &outputs->list[j];
    }
    /* Named interface-block members: XFB name is BlockType.member while
     * reflection stores the bare member for Metal identifiers. */
    const char *dot = strrchr(baseName, '.');
    if (!dot || dot == baseName || dot[1] == '\0' ||
        strchr(dot + 1, '.') != NULL)
        return NULL;
    const char *member = dot + 1;
    MGLShaderResource *match = NULL;
    GLuint matches = 0u;
    for (GLuint j = 0u; j < outputs->count; j++) {
        MGLShaderResource *cand = &outputs->list[j];
        if (!cand->name || strcmp(cand->name, member) != 0)
            continue;
        match = cand;
        matches++;
    }
    return matches == 1u ? match : NULL;
}

/* Build the link-time XFB scatter plan and validate the ARB_transform_feedback3
 * control entries.  Execution still deliberately gates GS SEPARATE_ATTRIBS
 * elsewhere, but every accepted program now has one authoritative binding,
 * component-offset, and stream assignment plan ready for that route. */

int mglCompileCaptureVariant(const char *src,
                                    const char *const *attrib_names,
                                    uint32_t flags,
                                    unsigned char **bytes, size_t *size,
                                    char *err, size_t err_cap)
{
    MGLCompileArtifact art;
    mglCompileArtifactInit(&art);
    int rc = mglCompileArtifactFromGLSLEx(src, MGL_STAGE_VERTEX, attrib_names,
                                          flags, NULL, 0u,
                                          kMGLSparseResidencyDefaultBufferIndex,
                                          &art, err, err_cap);
    if (rc == 0 && art.complete && art.metallib_bytes && art.metallib_size) {
        *bytes = art.metallib_bytes;
        *size = art.metallib_size;
        art.metallib_bytes = NULL;
        art.metallib_size = 0;
        mglCompileArtifactDestroy(&art);
        return 0;
    }
    mglCompileArtifactDestroy(&art);
    return -1;
}

const char *mglSkipGLSLVersionDirective(const char *s)
{
    if (!s) {
        return s;
    }
    const char *p = s;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
        p++;
    }
    if (strncmp(p, "#version", 8) != 0) {
        return s;
    }
    p += 8;
    while (*p && *p != '\n') {
        p++;
    }
    if (*p == '\n') {
        p++;
    }
    return p;
}

char *mglConcatSameStageSources(Program *pptr, int stage)
{
    GLuint n = mglProgramAttachedShaderCount(pptr, (GLuint)stage);
    size_t total = 1u;
    GLuint i;
    for (i = 0u; i < n; i++) {
        Shader *sh = (pptr->attached_shader_counts[stage] > 0u)
            ? pptr->attached_shader_slots[stage][i]
            : pptr->shader_slots[stage];
        if (!sh || !sh->src) {
            continue;
        }
        const char *body = (i == 0u) ? sh->src
                                     : mglSkipGLSLVersionDirective(sh->src);
        total += strlen(body) + 1u;
    }
    char *out = (char *)malloc(total);
    if (!out) {
        return NULL;
    }
    size_t used = 0u;
    out[0] = '\0';
    for (i = 0u; i < n; i++) {
        Shader *sh = (pptr->attached_shader_counts[stage] > 0u)
            ? pptr->attached_shader_slots[stage][i]
            : pptr->shader_slots[stage];
        if (!sh || !sh->src) {
            continue;
        }
        const char *body = (i == 0u) ? sh->src
                                     : mglSkipGLSLVersionDirective(sh->src);
        size_t bl = strlen(body);
        if (used && used + 1u < total && out[used - 1u] != '\n') {
            out[used++] = '\n';
        }
        if (used + bl >= total) {
            break;
        }
        memcpy(out + used, body, bl);
        used += bl;
        out[used] = '\0';
    }
    return out;
}


/* AIR path stage compiler: self-hosted frontend + LLVM -> metallib, plus
 * resource reflection.  Returns 1 on success; a failed stage is
 * non-fatal at link time (the stage is simply not renderable), matching
 * the legacy path's behaviour. */

/* GL 4.6 §7.4.1 / §11.3.1: the vertex-output and geometry-input
 * interfaces must agree on type, interpolation qualifier and (for
 * explicitly sized arrays) array size.  A GS input with no matching VS
 * output is legal and reads undefined values. */

/* GL 4.6 §7.7.2: two distinct atomic counter uniforms in the same program
 * that share an atomic counter buffer binding must not have overlapping
 * [offset, offset + size) byte ranges.  The same counter name declared in
 * several stages refers to one counter and is exempt.  Reflection stores
 * the counter's byte offset in resource->location and its byte size in
 * resource->required_size. */

void mglCopyExplicitBinding(MGLShaderResource *dst,
                                   const MGLShaderResource *src)
{
    dst->gl_binding = src->gl_binding;
    dst->binding_explicit = GL_TRUE;
    if (src->sampler_unit >= 0)
        dst->sampler_unit = src->sampler_unit;
    if (dst->ubo_array_bindings && src->ubo_array_bindings &&
        dst->ubo_array_size == src->ubo_array_size) {
        memcpy(dst->ubo_array_bindings, src->ubo_array_bindings,
               dst->ubo_array_size * sizeof(GLuint));
    }
}

/* GLSL 4.60 §4.4.5/§4.4.6: declarations of the same uniform in different
 * stages must not specify different bindings; a binding given on only some
 * declarations applies to all of them. */

void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count,
                                          const char *name, GLuint gl_type,
                                          GLint array_size, GLboolean is_array)
{
    MGLShaderResource *r;
    if (!slots || !count || !name || *count >= 16)
        return;
    r = &slots[*count];
    memset(r, 0, sizeof(*r));
    r->name = strdup(name);
    if (!r->name)
        return;
    r->gl_type = gl_type;
    r->gl_array_size = array_size;
    r->is_array = is_array;
    r->location = 0xffffffffu;
    r->location_index = 0xffffffffu;
    r->uniform_location = -1;
    (*count)++;
}


void mglUseProgram(GLMContext ctx, GLuint program)
{
    Program *pptr = NULL;

    if (!ctx) {
        return;
    }

    /* GL 4.6 §7.3: UseProgram while transform feedback is active and not
     * paused.  PauseTransformFeedback allows program switches. */
    if (STATE(transform_feedback) && STATE(transform_feedback)->active &&
        !STATE(transform_feedback)->paused) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (STATE(program_name) == program &&
        ((program == 0u && STATE(program) == NULL) ||
         (program != 0u && STATE(program) != NULL))) {
        /* ARB_shader_subroutine: the selection reset happens on every
         * UseProgram call, including a same-program rebind (the selections
         * are context state, not program state).  No-op for programs
         * without subroutine uniforms. */
        if (program != 0u) {
            mglUniformResetSubroutineSelections(ctx, program);
        }
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
        }

        if (!pptr ||
            !mglObjectPointerLooksPlausible(pptr) ||
            !mglHashTableContainsData(&STATE(program_table), pptr))
        {
            fprintf(stderr, "MGL Error: mglUseProgram program %u not found or invalid\n", program);
            // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);

            return;
        }

        if (!pptr->link_success)
        {
            /* GL 4.6 §7.3: INVALID_OPERATION and keep the prior program. */
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
    }
    else
    {
        pptr = NULL;
    }

    bool bindingChanged =
        STATE(program) != pptr ||
        STATE(program_name) != program;

    if (bindingChanged)
    {
        Program *oldProgram = STATE(program);
        if (oldProgram &&
            !mglProgramPointerUsableForName(ctx,
                                            oldProgram,
                                            /* The state's own name is the known name for the
                                             * program we are unbinding (decision D6). */
                                            STATE(program_name)))
        {
            fprintf(stderr, "MGL WARNING: mglUseProgram dropping invalid cached program pointer %p\n",
                    (void *)oldProgram);
            oldProgram = NULL;
            STATE(program) = NULL;
        }

        if (oldProgram)
        {
            oldProgram->refcount--;
            if (oldProgram->refcount == 0 && oldProgram->delete_status)
            {
                mglFreeProgram(ctx, oldProgram);
            }
        }

        STATE(program) = pptr;

        if (pptr)
        {
            pptr->refcount++;
        }
        mglMarkStateDirtyBits(ctx->active_state, DIRTY_PROGRAM);
    }

    /*
     * Keep program name and pointer state in sync so renderer-side recovery can
     * re-resolve by name if the cached pointer is lost.
     */
    STATE(program_name) = program;

    /* ARB_shader_subroutine: subroutine selections are context state reset
     * to default functions whenever UseProgram runs, so they never survive
     * unbind/rebind of the same program.  No-op for programs without
     * subroutine uniforms. */
    if (pptr) {
        mglUniformResetSubroutineSelections(ctx, pptr->name);
    }
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
        case GL_PROGRAM_SEPARABLE:
            *params = pptr->linked_program_separable ? GL_TRUE : GL_FALSE;
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
            *params = pptr->spirv_nameless
                ? 1
                : mglProgramActiveAttribMaxNameLength(pptr);
            break;
        case GL_ACTIVE_UNIFORMS:
            *params = mglProgramActiveUniformCount(pptr);
            break;
        case GL_ACTIVE_UNIFORM_MAX_LENGTH:
            *params = pptr->spirv_nameless
                ? 1
                : mglProgramActiveUniformMaxNameLength(pptr);
            break;
        case GL_ACTIVE_UNIFORM_BLOCKS:
            *params = mglActiveUniformBlockCount(pptr);
            break;
        case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH:
            *params = pptr->spirv_nameless
                ? 1
                : mglActiveUniformBlockMaxNameLength(pptr);
            break;
        case GL_COMPUTE_WORK_GROUP_SIZE:
            /* GL 4.6 §7.14: INVALID_OPERATION unless linked with a compute
             * shader. */
            if (!pptr->link_success || !pptr->shader_slots[_COMPUTE_SHADER]) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            params[0] = pptr->local_workgroup_size.x;
            params[1] = pptr->local_workgroup_size.y;
            params[2] = pptr->local_workgroup_size.z;
            break;
        case GL_ACTIVE_ATOMIC_COUNTER_BUFFERS:
            *params = mglActiveAtomicCounterBufferCount(pptr);
            break;
        case GL_GEOMETRY_INPUT_TYPE:        /* 0x8917 */
        case GL_GEOMETRY_OUTPUT_TYPE:       /* 0x8918 */
        case GL_GEOMETRY_VERTICES_OUT:      /* 0x8916 */
        case GL_GEOMETRY_SHADER_INVOCATIONS:/* 0x887F */
            if (!pptr->link_success ||
                !pptr->shader_slots[_GEOMETRY_SHADER]) {
                /* GL 4.6 §22.2: these pnames are geometry-shader state;
                 * querying them on a program with no linked geometry
                 * shader generates INVALID_OPERATION (not zero). */
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            if (pname == GL_GEOMETRY_INPUT_TYPE) {
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
            if (!pptr->link_success ||
                !pptr->shader_slots[_TESS_CONTROL_SHADER]) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            *params = (GLint)pptr->tess_control_output_vertices;
            break;
        case GL_TESS_GEN_MODE:             /* 0x8E76 */
        case GL_TESS_GEN_SPACING:          /* 0x8E77 */
        case GL_TESS_GEN_VERTEX_ORDER:     /* 0x8E78 */
        case GL_TESS_GEN_POINT_MODE:       /* 0x8E79 */
            /* TES execution-mode reflection.  Returns the layout(...) values
             * captured from AIR tessellation metadata at link time. */
            if (!pptr->link_success ||
                !pptr->shader_slots[_TESS_EVALUATION_SHADER]) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            if (pname == GL_TESS_GEN_MODE) {
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
            } else if (pptr->spirv_nameless) {
                *params = 1;
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
    if (n < 0)
    {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_VALUE);
        return;
    }
    if (!pipelines)
        return;
    /* Reserve names only. IsProgramPipeline stays FALSE until Bind. */
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
            
        ProgramPipeline *ptr =
            (ProgramPipeline *)searchHashTable(&STATE(program_pipeline_table),
                                               pipelines[i]);
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
        if (ptr->active_program)
        {
            Program *active = ptr->active_program;
            ptr->active_program = NULL;
            mglReleaseProgramReference(ctx, active);
        }

        // Remove from hash table and free
        deleteHashElement(&STATE(program_pipeline_table), pipelines[i]);
        free(ptr->debug_label);
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

    ProgramPipeline *ptr = ensureProgramPipeline(ctx, pipeline);
    if (!ptr)
    {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    STATE(program_pipeline) = ptr;
    STATE(var.program_pipeline_binding) = pipeline;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_PROGRAM);
    /* ARB_shader_subroutine + SSO: binding a pipeline re-currentizes its
     * stage programs, so their subroutine uniforms reset to defaults
     * (CTS shader_subroutine.subroutine_uniform_reset). */
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        Program *sp = ptr->stage_programs[stage];
        if (sp && sp->link_success)
            mglUniformResetSubroutineSelections(ctx, sp->name);
    }
}

/* GL 4.6 §7.4: the pipeline's active program receives Uniform* updates when
 * no program is bound with UseProgram (§7.6.1). */
void mglActiveShaderProgram(GLMContext ctx, GLuint pipeline, GLuint program)
{
    ProgramPipeline *pipe_ptr = ensureProgramPipeline(ctx, pipeline);
    if (!pipe_ptr)
    {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    Program *prog_ptr = NULL;
    if (program != 0)
    {
        prog_ptr = findProgram(ctx, program);
        if (!prog_ptr)
        {
            mglDispatchError(ctx, __FUNCTION__,
                             mglIsShader(ctx, program) ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
            return;
        }
        if (!prog_ptr->link_success)
        {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
        mglRetainProgramReference(ctx, prog_ptr);
    }

    Program *old = pipe_ptr->active_program;
    pipe_ptr->active_program = prog_ptr;
    if (old)
        mglReleaseProgramReference(ctx, old);
}

void mglUseProgramStages(GLMContext ctx, GLuint pipeline, GLbitfield stages, GLuint program)
{
    ProgramPipeline *pipe_ptr = ensureProgramPipeline(ctx, pipeline);
    if (!pipe_ptr)
    {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    mglFlushPendingDraws(ctx);

    Program *prog_ptr = NULL;
    if (program != 0)
    {
        prog_ptr = findProgram(ctx, program);
        if (!prog_ptr)
        {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_VALUE);
            return;
        }
        if (!prog_ptr->link_success || !prog_ptr->linked_program_separable)
        {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
    }

    {
        const GLbitfield stage_bits =
            GL_VERTEX_SHADER_BIT | GL_FRAGMENT_SHADER_BIT |
            GL_GEOMETRY_SHADER_BIT | GL_TESS_CONTROL_SHADER_BIT |
            GL_TESS_EVALUATION_SHADER_BIT | GL_COMPUTE_SHADER_BIT;
        if (stages != GL_ALL_SHADER_BITS && (stages & ~stage_bits) != 0u)
        {
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_VALUE);
            return;
        }
        if (stages == GL_ALL_SHADER_BITS)
            stages = stage_bits;
    } 
    /* Attach program to specified stages.  Each stage slot owns an
     * independent reference: retain the new program BEFORE releasing the
     * old one so re-attaching a program that is already in a slot (or is
     * the only reference keeping it alive) does not free it mid-op. */
    /* GL 4.6 §7.4: a specified stage with no shader in program is
     * disabled (unbound), not attached to a program that lacks it. */
#define MGL_REPLACE_STAGE_SLOT(slot)                                         \
    do {                                                                     \
        Program *_new = prog_ptr;                                            \
        if (_new && !_new->shader_slots[(slot)])                             \
            _new = NULL;                                                     \
        Program *_old = pipe_ptr->stage_programs[(slot)];                    \
        if (_new)                                                            \
            mglRetainProgramReference(ctx, _new);                            \
        pipe_ptr->stage_programs[(slot)] = _new;                             \
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
    /* GL 4.6 §7.10 / ARB_shader_subroutine: UseProgramStages resets the
     * subroutine uniforms of every stage it touches. */
    if (prog_ptr) {
        mglUniformResetSubroutineSelections(ctx, prog_ptr->name);
    }
}
