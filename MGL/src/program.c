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
#include <ctype.h>
#include <malloc/malloc.h>
#include <CoreFoundation/CoreFoundation.h>
#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>
#include "spirv-tools/libspirv.h"
#include "spirv_cross_c.h"
#include "spirv.h"

#include "glm_context.h"
#include "shaders.h"
#include "buffers.h"
#include "mgl_safety.h"
#include "mgl_buffer_slots.h"
#include "mgl_ir_postprocess.h"
#include "msl_patch_pipeline.h"
#include "mgl_metal_ref.h"
#include "mgl_uniform_reflection.h"
#include "mgl_spirv_compile.h"


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
    }

    /* linked_glsl_program is a non-NULL linked-state marker (set to
     * (glslang_program_t*)ptr on successful link), NOT a real glslang
     * program.  The real glslang_program_t is deleted at the end of
     * mglLinkProgram.  Do NOT call glslang_program_delete here — it would
     * dereference the marker as if it were a glslang object. */
    ptr->linked_glsl_program = NULL;

    mglSafeReleaseMetalObj((void **)&ptr->mtl_data);

    for(int i=0; i<_MAX_SHADER_TYPES; i++)
    {
        // CRITICAL FIX: Add NULL checks before all free/release operations to prevent double-frees
        if (ptr->spirv[i].ir) {
            free(ptr->spirv[i].ir);
            ptr->spirv[i].ir = NULL;
        }
        if (ptr->spirv[i].msl_str) {
            free(ptr->spirv[i].msl_str);
            ptr->spirv[i].msl_str = NULL;
        }
        if (ptr->spirv[i].msl_str_capture) {
            free(ptr->spirv[i].msl_str_capture);
            ptr->spirv[i].msl_str_capture = NULL;
        }
        if (ptr->spirv[i].entry_point) {
            free(ptr->spirv[i].entry_point);
            ptr->spirv[i].entry_point = NULL;
        }
        mglSafeReleaseMetalObj((void **)&ptr->spirv[i].mtl_function);
        mglSafeReleaseMetalObj((void **)&ptr->spirv[i].mtl_library);
        
        for(int j=0; j<_MAX_SPIRV_RES; j++)
        {
            // CRITICAL FIX: Add NULL checks and clear pointers to prevent double-frees
            SpirvResourceList *rl = &ptr->spirv_resources_list[i][j];
            if (rl->list) {
                for (GLuint k = 0; k < rl->count; k++) {
                    mglFreeSpirvResourceOwnedFields(&rl->list[k]);
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

    if (!mglObjectPointerLooksPlausible(program) ||
        !mglPointerRangeIsReadable(program, sizeof(*program)) ||
        program->name != expectedName) {
        return GL_FALSE;
    }

    if (mglHashTableContainsData(&STATE(program_table), program)) {
        return GL_TRUE;
    }

    /*
     * glDeleteProgram removes the name immediately, but the current program
     * and any deferred draws that captured it must keep using the object until
     * their references are released.
     */
    if (program->delete_status &&
        program->refcount > 0 &&
        program->linked_glsl_program != NULL) {
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
        if (!pptr->linked_glsl_program) {
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
    if (!pptr->linked_glsl_program) {
        sptr->refcount--;

        if (sptr->refcount == 0 && sptr->delete_status)
        {
            deleteHashElement(&STATE(shader_table), sptr->name);
            mglFreeShader(ctx, sptr);
        }

        pptr->dirty_bits |= DIRTY_PROGRAM;
    }
}


















/* Validate that all transform feedback varying names are active outputs
 * of the program.  Per the GL spec, LinkProgram must fail if any captured
 * varying is not an output of the last pre-rasterization stage. */
static bool mglValidateTransformFeedbackVaryings(Program *pptr)
{
    if (!pptr || pptr->transform_feedback_varying_count <= 0)
        return true;

    /* Determine the last active stage before fragment (the stage that
     * provides transform feedback outputs).  Priority: geometry > tess
     * eval > vertex. */
    int feedback_stage = -1;
    if (pptr->attached_shader_mask & GEOMETRY_SHADER_MASK_BIT)
        feedback_stage = _GEOMETRY_SHADER;
    else if (pptr->attached_shader_mask & TESS_EVALUATION_SHADER_MASK_BIT)
        feedback_stage = _TESS_EVALUATION_SHADER;
    else if (pptr->attached_shader_mask & VERTEX_SHADER_MASK_BIT)
        feedback_stage = _VERTEX_SHADER;

    if (feedback_stage < 0)
    {
        /* No stage to capture from — all varyings are invalid. */
        return false;
    }

    SpirvResourceList *outputs =
        &pptr->spirv_resources_list[feedback_stage][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];

    for (GLsizei i = 0; i < pptr->transform_feedback_varying_count; i++)
    {
        const char *varying = pptr->transform_feedback_varying_names[i];
        if (!varying || varying[0] == '\0')
            continue;

        /* Strip array subscript for comparison (e.g. "foo[0]" -> "foo"). */
        char base_name[96];
        strncpy(base_name, varying, sizeof(base_name) - 1);
        base_name[sizeof(base_name) - 1] = '\0';
        char *bracket = strchr(base_name, '[');
        if (bracket)
            *bracket = '\0';

        bool found = false;
        for (GLuint j = 0; j < outputs->count; j++)
        {
            if (outputs->list[j].name &&
                strcmp(outputs->list[j].name, base_name) == 0)
            {
                found = true;
                break;
            }
        }

        if (!found)
            return false;
    }

    return true;
}

void mglLinkProgram(GLMContext ctx, GLuint program)
{
    Program *pptr;
    glslang_program_t *glsl_program;
    int err;
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

    mglFlushPendingDraws(ctx);

    pptr->uses_vertex_id = GL_FALSE;
    pptr->uses_primitive_id = GL_FALSE;
    /* Invalidate MSL query cache; repopulated from the freshly generated MSL
     * after the stage compile loop succeeds. */
    pptr->mslCacheValid = GL_FALSE;
    pptr->usesFragCoordParams = GL_FALSE;
    pptr->vertexAttribUsageMask = 0u;
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if (mglProgramAttachedShaderCount(pptr, (GLuint)stage) > 0u) {
            has_any_shader = true;
            break;
        }
    }

    if (!has_any_shader) {
        fprintf(stderr, "MGL WARNING: mglLinkProgram called with no attached shaders\n");
        pptr->linked_glsl_program = NULL;
        return;
    }

    if ((pptr->attached_shader_mask & COMPUTE_SHADER_MASK_BIT) &&
        (pptr->attached_shader_mask & ~COMPUTE_SHADER_MASK_BIT)) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: compute shaders cannot be linked with non-compute stages\n",
                pptr->name);
        pptr->linked_glsl_program = NULL;
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
            if (!shader || !shader->compiled_glsl_shader) {
                fprintf(stderr,
                        "MGL WARNING: mglLinkProgram failed program %u: shader stage %d is not compiled\n",
                        pptr->name,
                        stage);
                pptr->linked_glsl_program = NULL;
                return;
            }
        }
    }

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Creating glslang program for full-link\n");
    }
    glsl_program = glslang_program_create();
    if (!glsl_program) {
        fprintf(stderr, "MGL Error: glslang_program_create failed\n");
        pptr->linked_glsl_program = NULL;
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Adding shaders to program\n");
    }
    addShadersToProgram(ctx, pptr, glsl_program);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Shaders added\n");
    }

    err = glslang_program_link(glsl_program, GLSLANG_MSG_DEFAULT_BIT);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Program link returned %d\n", err);
    }
    if (!err)
    {
        fprintf(stderr, "MGL Error: glslang_program_link failed err: %d\n", err);
        fprintf(stderr, "MGL Error: glslang_program_SPIRV_get_messages:\n%s\n", glslang_program_SPIRV_get_messages(glsl_program));
        fprintf(stderr, "MGL Error: glslang_program_get_info_log:\n%s\n", glslang_program_get_info_log(glsl_program));
        fprintf(stderr, "MGL Error: glslang_program_get_info_debug_log:\n%s\n", glslang_program_get_info_debug_log(glsl_program));
        glslang_program_delete(glsl_program);
        pptr->linked_glsl_program = NULL;
        return;
    }

    err = glslang_program_map_io(glsl_program);
    if (!err)
    {
        fprintf(stderr, "MGL WARNING: glslang_program_map_io failed; continuing with linked program\n");
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
    {
        if (!compileStageFromLinkedProgram(ctx, pptr, glsl_program, stage)) {
            link_ok = false;
            break;
        }
    }

    if (!link_ok) {
        glslang_program_delete(glsl_program);
        pptr->linked_glsl_program = NULL;
        return;
    }

    /* Validate transform feedback varyings: the link must fail if any
     * captured varying is not an active output of the program. */
    if (!mglValidateTransformFeedbackVaryings(pptr)) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: transform feedback "
                "varying not found in program outputs\n",
                pptr->name);
        glslang_program_delete(glsl_program);
        pptr->linked_glsl_program = NULL;
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
    mglUnifySamplerUniformLocations(pptr);

    if (pptr->program_separable &&
        (pptr->attached_shader_mask & (pptr->attached_shader_mask - 1u)) != 0u &&
        !mglLinkedProgramPerVertexCompatible(pptr)) {
        fprintf(stderr,
                "MGL WARNING: separable program %u has incompatible gl_PerVertex redeclarations\n",
                pptr->name);
        glslang_program_delete(glsl_program);
        pptr->linked_glsl_program = NULL;
        return;
    }

    /* Validate layout(binding=N) values against GL implementation limits.
     * The GL spec requires that the link fail if a binding point exceeds
     * the corresponding MAX_*_BINDINGS limit.  glslang does not know the
     * implementation limits, so MGL must enforce them here. */
    {
        bool binding_error = false;
        for (int stage = 0; stage < _MAX_SHADER_TYPES && !binding_error; stage++) {
            if ((pptr->attached_shader_mask & (1u << stage)) == 0u) {
                continue;
            }

            /* Uniform blocks: GL_MAX_UNIFORM_BUFFER_BINDINGS */
            {
                SpirvResourceList *rl =
                    &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
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
                SpirvResourceList *rl =
                    &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_BUFFER];
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
                SpirvResourceList *rl =
                    &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
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
                SpirvResourceList *rl =
                    &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_ATOMIC_COUNTER];
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

        if (binding_error) {
            glslang_program_delete(glsl_program);
            pptr->linked_glsl_program = NULL;
            return;
        }
    }

    /* The glslang_program_t was only needed during linking for SPIR-V
     * generation and reflection.  Release it now; pptr->linked_glsl_program
     * is reused as a non-NULL linked-state marker (see callers that test
     * `if (ptr->linked_glsl_program)` to check link status). */
    glslang_program_delete(glsl_program);
    /* linked_glsl_program is used as a linked-state marker only. */
    pptr->linked_glsl_program = (glslang_program_t *)pptr;
    pptr->dirty_bits |= DIRTY_PROGRAM;

    /* Populate the MSL query result cache from the finalized per-stage MSL.
     * Scanning once here lets the per-draw paths skip repeated strstr() over
     * the source strings when MGL_MSL_CACHE=1.  mslCacheValid gates readers;
     * it is only set after both fields below reflect the current MSL. */
    {
        const char *fs_msl = pptr->spirv[_FRAGMENT_SHADER].msl_str;
        pptr->usesFragCoordParams =
            (fs_msl && strstr(fs_msl, MGL_FRAG_COORD_PARAMS_MSL_NAME))
                ? GL_TRUE : GL_FALSE;

        const char *vs_msl = pptr->spirv[_VERTEX_SHADER].msl_str;
        uint32_t attr_mask = 0u;
        if (vs_msl) {
            for (GLuint a = 0; a < MAX_ATTRIBS; a++) {
                char attr_pattern[32];
                snprintf(attr_pattern, sizeof(attr_pattern),
                         "[[attribute(%u)]]", a);
                if (strstr(vs_msl, attr_pattern)) {
                    attr_mask |= (1u << a);
                }
            }
        }
        pptr->vertexAttribUsageMask = attr_mask;
        pptr->mslCacheValid = GL_TRUE;
    }

    /* Only call mtlBindProgram if Metal functions are initialized */
    if (ctx->mtl_funcs.mtlBindProgram) {
        ctx->mtl_funcs.mtlBindProgram(ctx, pptr);
    } else {
        fprintf(stderr, "WARNING: Metal functions not initialized, skipping mtlBindProgram\n");
    }

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
        ctx->state.var.current_program == program &&
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

        if (!pptr->linked_glsl_program)
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
        ctx->state.program_name != program ||
        ctx->state.var.current_program != program;

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
        ctx->state.dirty_bits |= DIRTY_PROGRAM;
    }

    /*
     * Keep program name and pointer state in sync so renderer-side recovery can
     * re-resolve by name if the cached pointer is lost.
     */
    ctx->state.program_name = program;
    ctx->state.var.current_program = program;

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL UseProgram program=%u resolved=%p\n",
                program, (void *)ctx->state.program);
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
    if (ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    SpirvResource *res = mglProgramActiveAttribAt(ptr, index);
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
    if (ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    int stage = -1;
    int res_type = -1;
    SpirvResource *res = mglProgramActiveUniformAt(ptr, index, &stage, &res_type);
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

	if (ptr->linked_glsl_program == NULL)
	{
		ERROR_RETURN(GL_INVALID_OPERATION);

		return -1;
	}

    SpirvResourceList *vertex_inputs =
        &ptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];

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
            *params = pptr->linked_glsl_program ? GL_TRUE : GL_FALSE;
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
            if (!pptr->linked_glsl_program) {
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
            /* Geometry-shader reflection queries.  MGL does not execute GS on
             * Metal, but the program still carries the GS layout metadata
             * captured at compile/link time.  Return 0 when no GS is
             * attached so the queries are at least well-defined. */
            if (!pptr->linked_glsl_program) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            *params = 0;
            break;
        case GL_TESS_CONTROL_OUTPUT_VERTICES:  /* 0x8E75 */
            if (!pptr->linked_glsl_program) {
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
             * captured from SPIR-V OpExecutionMode at link time.  0 when no
             * TES is attached. */
            if (!pptr->linked_glsl_program) {
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
            STATE(dirty_bits) |= DIRTY_PROGRAM;
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
        STATE(dirty_bits) |= DIRTY_PROGRAM;
        return;
    }
    
    ProgramPipeline *ptr = getProgramPipeline(ctx, pipeline);
    STATE(program_pipeline) = ptr;
    STATE(var.program_pipeline_binding) = ptr ? pipeline : 0;
    STATE(dirty_bits) |= DIRTY_PROGRAM;
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
    STATE(dirty_bits) |= DIRTY_PROGRAM;
}
