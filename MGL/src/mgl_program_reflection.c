/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_program_reflection.h"
#include "mgl_glsl_ast.h" /* MGLTranslationUnit / MGLDecl: gl_PerVertex redeclarations */
#include "mgl_metal_ref.h"
#include "mgl_uniform_reflection.h"

void clearStageCompileState(Program *program, int stage)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }

    MGLShaderModule *compiled = &program->modules[stage];
    free(compiled->metallib_bytes);
    compiled->metallib_bytes = NULL;
    compiled->metallib_size = 0;
    free(compiled->metallib_tess_capture_bytes);
    compiled->metallib_tess_capture_bytes = NULL;
    compiled->metallib_tess_capture_size = 0;
    free(compiled->metallib_cull_capture_bytes);
    compiled->metallib_cull_capture_bytes = NULL;
    compiled->metallib_cull_capture_size = 0;
    free(compiled->entry_point);
    compiled->entry_point = NULL;
    compiled->needs_runtime_array_size_buffer = GL_FALSE;

    mglSafeReleaseMetalObj(&compiled->mtl_compute_pipeline);
    mglSafeReleaseMetalObj(&compiled->mtl_function);
    mglSafeReleaseMetalObj(&compiled->mtl_library);
    mglSafeReleaseMetalObj(&compiled->mtl_tess_capture_function);
    mglSafeReleaseMetalObj(&compiled->mtl_tess_capture_library);
    mglSafeReleaseMetalObj(&compiled->mtl_cull_capture_function);
    mglSafeReleaseMetalObj(&compiled->mtl_cull_capture_library);

    for (int type = 0; type < MGL_MAX_SHADER_RESOURCES; type++) {
        MGLShaderResourceList *list = &program->shader_resources_list[stage][type];
        for (GLuint i = 0; i < list->count; i++) {
            mglFreeMGLShaderResourceOwnedFields(&list->list[i]);
        }
        free(list->list);
        list->list = NULL;
        list->count = 0;
    }
}

GLboolean mglProgramPerVertexSignature(Program *program, int stage,
                                       unsigned *signature)
{
    if (signature) {
        *signature = 0;
    }
    if (!program || !signature || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return GL_FALSE;
    }

    /* A gl_PerVertex redeclaration is a normal block declaration in the parsed
     * TU (type name "gl_PerVertex", members = the redeclared builtins), so the
     * signature is read from the declaration instead of scanning the source
     * text for the block and its member tokens: a commented-out or similarly
     * named struct is no longer mistaken for a redeclaration, and a stage
     * without a TU simply has no signature.  The instance name is irrelevant -
     * `out gl_PerVertex { ... } vs_out;` declares the same interface. */
    Shader *shader = program->shader_slots[stage];
    MGLTranslationUnit *tu = shader ? shader->frontend_tu : NULL;
    if (!tu) {
        return GL_FALSE;
    }

    unsigned result = 0;
    GLboolean found = GL_FALSE;
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        for (MGLDecl *decl = tu->decls[i]; decl; decl = decl->next_declarator) {
            if (!decl->type || !decl->type->name ||
                strcmp(decl->type->name, "gl_PerVertex") != 0) {
                continue;
            }
            for (uint32_t m = 0; m < decl->struct_member_count; m++) {
                const MGLDecl *member = decl->struct_members[m];
                const char *name = member ? member->name : NULL;
                if (!name) {
                    continue;
                }
                if (strcmp(name, "gl_Position") == 0) {
                    result |= 1u << 0;
                } else if (strcmp(name, "gl_PointSize") == 0) {
                    result |= 1u << 1;
                } else if (strcmp(name, "gl_ClipDistance") == 0) {
                    result |= 1u << 2;
                } else if (strcmp(name, "gl_CullDistance") == 0) {
                    result |= 1u << 3;
                }
            }
            found = GL_TRUE;
        }
    }

    if (found) {
        *signature = result;
    }
    return found;
}

GLboolean mglProgramPipelinePerVertexCompatible(
    Program *const *stage_programs)
{
    unsigned reference = 0;
    GLboolean have_reference = GL_FALSE;
    if (!stage_programs) {
        return GL_TRUE;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        Program *program = stage_programs[stage];
        unsigned signature = 0;
        if (!program || !program->shader_slots[stage] ||
            !mglProgramPerVertexSignature(program, stage, &signature)) {
            continue;
        }
        if (!have_reference) {
            reference = signature;
            have_reference = GL_TRUE;
        } else if (signature != reference) {
            return GL_FALSE;
        }
    }
    return GL_TRUE;
}

GLboolean mglLinkedProgramPerVertexCompatible(Program *program)
{
    Program *stages[_MAX_SHADER_TYPES] = {0};
    if (!program) {
        return GL_TRUE;
    }
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if ((program->attached_shader_mask & (1u << stage)) &&
            program->shader_slots[stage]) {
            stages[stage] = program;
        }
    }
    return mglProgramPipelinePerVertexCompatible(stages);
}

GLint mglDefaultAttribLocationForName(const char *name)
{
    if (!name) return -1;
    if (strcmp(name, "Position") == 0) return 0;
    if (strcmp(name, "Color") == 0) return 1;
    if (strcmp(name, "UV0") == 0) return 2;
    if (strcmp(name, "UV1") == 0) return 3;
    if (strcmp(name, "UV2") == 0) return 4;
    if (strcmp(name, "Normal") == 0) return 5;
    return -1;
}

GLint mglProgramVertexInputOrdinal(Program *program, const char *name)
{
    if (!program || !name) return -1;
    MGLShaderResourceList *inputs =
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
    for (GLuint i = 0; i < inputs->count; i++) {
        if (inputs->list[i].name && strcmp(inputs->list[i].name, name) == 0) {
            return (GLint)i;
        }
    }
    return -1;
}

GLboolean mglProgramHasVertexInputNamed(Program *program, const char *name)
{
    return mglProgramVertexInputOrdinal(program, name) >= 0;
}

GLint mglContextualDefaultAttribLocationForName(Program *program,
                                                const char *name)
{
    if (!program || !name) return -1;
    GLboolean has_color = mglProgramHasVertexInputNamed(program, "Color");
    GLboolean has_uv0 = mglProgramHasVertexInputNamed(program, "UV0");
    GLboolean has_uv1 = mglProgramHasVertexInputNamed(program, "UV1");
    GLboolean has_uv2 = mglProgramHasVertexInputNamed(program, "UV2");

    if (strcmp(name, "UV2") == 0) {
        if (!has_uv0 && !has_uv1) return 2;
        if (has_uv0 && !has_uv1) return 3;
        return 4;
    }
    if (strcmp(name, "Normal") == 0) {
        if (has_uv2 && !has_uv1) return has_uv0 ? 4 : 3;
        return 5;
    }
    if (has_color && has_uv0 && !has_uv1 && !has_uv2) {
        GLint color = mglProgramVertexInputOrdinal(program, "Color");
        GLint uv0 = mglProgramVertexInputOrdinal(program, "UV0");
        if (uv0 >= 0 && color >= 0 && uv0 < color) {
            if (strcmp(name, "UV0") == 0) return 1;
            if (strcmp(name, "Color") == 0) return 2;
        }
    }
    return mglDefaultAttribLocationForName(name);
}

GLint mglDesiredAttribLocationForName(Program *program, const char *name)
{
    if (!program || !name) return -1;
    for (int index = 0; index < MAX_ATTRIBS; index++) {
        if (program->attrib_location_names[index] &&
            strcmp(program->attrib_location_names[index], name) == 0) {
            return index;
        }
    }
    return mglContextualDefaultAttribLocationForName(program, name);
}

void applyVertexInputLocations(Program *program)
{
    if (!program) return;
    MGLShaderResourceList *inputs =
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
    for (GLuint i = 0; i < inputs->count; i++) {
        GLint desired = mglDesiredAttribLocationForName(program,
                                                        inputs->list[i].name);
        if (desired >= 0 && desired < MAX_ATTRIBS) {
            inputs->list[i].location = (GLuint)desired;
        }
    }
}

void applyMultiDimArrayUniformNames(Program *program)
{
    (void)program;
}

void applyFragmentOutputLocationIndices(Program *program)
{
    if (!program || program->frag_data_location_count == 0) return;
    MGLShaderResourceList *outputs =
        &program->shader_resources_list[_FRAGMENT_SHADER][_STAGE_OUTPUT_RES];
    for (GLuint i = 0; i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (!output->name) continue;
        for (GLuint j = 0; j < program->frag_data_location_count; j++) {
            if (program->frag_data_location_names[j] &&
                strcmp(program->frag_data_location_names[j], output->name) == 0) {
                output->location = program->frag_data_color_numbers[j];
                output->location_index = program->frag_data_indices[j];
                break;
            }
        }
    }
}

GLboolean mglProgramVaryingTypesCompatible(const MGLShaderResource *a,
                                           const MGLShaderResource *b)
{
    if (!a || !b) return GL_FALSE;
    if (a->gl_type && b->gl_type && a->gl_type != b->gl_type) {
        return GL_FALSE;
    }
    if (a->gl_array_size > 0 && b->gl_array_size > 0 &&
        a->gl_array_size != b->gl_array_size) {
        return GL_FALSE;
    }
    return GL_TRUE;
}

MGLShaderResource *mglFindVaryingByName(MGLShaderResourceList *list,
                                    const char *name,
                                    const MGLShaderResource *type_peer)
{
    if (!list || !name) return NULL;
    for (GLuint i = 0; i < list->count; i++) {
        MGLShaderResource *candidate = &list->list[i];
        if (candidate->name && strcmp(candidate->name, name) == 0 &&
            (!type_peer || mglProgramVaryingTypesCompatible(candidate,
                                                            type_peer))) {
            return candidate;
        }
    }
    return NULL;
}

MGLShaderResource *mglFindVaryingByLocation(MGLShaderResourceList *list,
                                        GLuint location,
                                        const MGLShaderResource *type_peer)
{
    if (!list) return NULL;
    for (GLuint i = 0; i < list->count; i++) {
        MGLShaderResource *candidate = &list->list[i];
        if (candidate->location == location &&
            (!type_peer || mglProgramVaryingTypesCompatible(candidate,
                                                            type_peer))) {
            return candidate;
        }
    }
    return NULL;
}

static void mglAlignInputsToOutputs(MGLShaderResourceList *outputs,
                                    MGLShaderResourceList *inputs)
{
    if (!outputs || !inputs) return;
    for (GLuint i = 0; i < inputs->count; i++) {
        MGLShaderResource *input = &inputs->list[i];
        MGLShaderResource *output = mglFindVaryingByName(outputs, input->name,
                                                     input);
        if (!output) {
            output = mglFindVaryingByLocation(outputs, input->location, input);
        }
        if (output) {
            input->location = output->location;
        }
    }
}

void alignFragmentInputLocationsToVertexOutputs(Program *program)
{
    if (!program) return;
    mglAlignInputsToOutputs(
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES],
        &program->shader_resources_list[_FRAGMENT_SHADER][_STAGE_INPUT_RES]);
}

void mglBridgeSkippedGeometryShaderVaryings(Program *program)
{
    if (!program || !program->shader_slots[_GEOMETRY_SHADER]) return;
    mglAlignInputsToOutputs(
        &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES],
        &program->shader_resources_list[_FRAGMENT_SHADER][_STAGE_INPUT_RES]);
}
