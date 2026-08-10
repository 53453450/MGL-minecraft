#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_program_reflection.h"
#include "mgl_metal_ref.h"
#include "mgl_uniform_reflection.h"

static GLboolean mglShaderSourceHasToken(const char *start,
                                         const char *end,
                                         const char *token)
{
    if (!start || !end || !token || start > end) {
        return GL_FALSE;
    }

    size_t token_len = strlen(token);
    for (const char *p = start; p + token_len <= end; p++) {
        if (strncmp(p, token, token_len) != 0) {
            continue;
        }
        int before = p != start &&
            (isalnum((unsigned char)p[-1]) || p[-1] == '_');
        int after = p[token_len] != '\0' &&
            (isalnum((unsigned char)p[token_len]) || p[token_len] == '_');
        if (!before && !after) {
            return GL_TRUE;
        }
    }
    return GL_FALSE;
}

void clearStageCompileState(Program *program, int stage)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }

    Spirv *compiled = &program->spirv[stage];
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
    compiled->needs_buffer_size_buffer = GL_FALSE;

    mglSafeReleaseMetalObj(&compiled->mtl_compute_pipeline);
    mglSafeReleaseMetalObj(&compiled->mtl_function);
    mglSafeReleaseMetalObj(&compiled->mtl_library);
    mglSafeReleaseMetalObj(&compiled->mtl_tess_capture_function);
    mglSafeReleaseMetalObj(&compiled->mtl_tess_capture_library);
    mglSafeReleaseMetalObj(&compiled->mtl_cull_capture_function);
    mglSafeReleaseMetalObj(&compiled->mtl_cull_capture_library);

    for (int type = 0; type < _MAX_SPIRV_RES; type++) {
        SpirvResourceList *list = &program->spirv_resources_list[stage][type];
        for (GLuint i = 0; i < list->count; i++) {
            mglFreeSpirvResourceOwnedFields(&list->list[i]);
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

    Shader *shader = program->shader_slots[stage];
    const char *src = shader ? shader->src : NULL;
    if (!src) {
        return GL_FALSE;
    }

    unsigned result = 0;
    GLboolean found = GL_FALSE;
    const char *cursor = src;
    while ((cursor = strstr(cursor, "gl_PerVertex")) != NULL) {
        const char *open = strchr(cursor, '{');
        const char *close = open ? strchr(open + 1, '}') : NULL;
        if (!open || !close) {
            cursor += strlen("gl_PerVertex");
            continue;
        }
        if (mglShaderSourceHasToken(open, close, "gl_Position")) {
            result |= 1u << 0;
        }
        if (mglShaderSourceHasToken(open, close, "gl_PointSize")) {
            result |= 1u << 1;
        }
        if (mglShaderSourceHasToken(open, close, "gl_ClipDistance")) {
            result |= 1u << 2;
        }
        if (mglShaderSourceHasToken(open, close, "gl_CullDistance")) {
            result |= 1u << 3;
        }
        found = GL_TRUE;
        cursor = close + 1;
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
    SpirvResourceList *inputs =
        &program->spirv_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
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
    SpirvResourceList *inputs =
        &program->spirv_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
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
    SpirvResourceList *outputs =
        &program->spirv_resources_list[_FRAGMENT_SHADER][_STAGE_OUTPUT_RES];
    for (GLuint i = 0; i < outputs->count; i++) {
        SpirvResource *output = &outputs->list[i];
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

GLboolean mglProgramVaryingTypesCompatible(const SpirvResource *a,
                                           const SpirvResource *b)
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

SpirvResource *mglFindVaryingByName(SpirvResourceList *list,
                                    const char *name,
                                    const SpirvResource *type_peer)
{
    if (!list || !name) return NULL;
    for (GLuint i = 0; i < list->count; i++) {
        SpirvResource *candidate = &list->list[i];
        if (candidate->name && strcmp(candidate->name, name) == 0 &&
            (!type_peer || mglProgramVaryingTypesCompatible(candidate,
                                                            type_peer))) {
            return candidate;
        }
    }
    return NULL;
}

SpirvResource *mglFindVaryingByLocation(SpirvResourceList *list,
                                        GLuint location,
                                        const SpirvResource *type_peer)
{
    if (!list) return NULL;
    for (GLuint i = 0; i < list->count; i++) {
        SpirvResource *candidate = &list->list[i];
        if (candidate->location == location &&
            (!type_peer || mglProgramVaryingTypesCompatible(candidate,
                                                            type_peer))) {
            return candidate;
        }
    }
    return NULL;
}

static void mglAlignInputsToOutputs(SpirvResourceList *outputs,
                                    SpirvResourceList *inputs)
{
    if (!outputs || !inputs) return;
    for (GLuint i = 0; i < inputs->count; i++) {
        SpirvResource *input = &inputs->list[i];
        SpirvResource *output = mglFindVaryingByName(outputs, input->name,
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
        &program->spirv_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES],
        &program->spirv_resources_list[_FRAGMENT_SHADER][_STAGE_INPUT_RES]);
}

void mglBridgeSkippedGeometryShaderVaryings(Program *program)
{
    if (!program || !program->shader_slots[_GEOMETRY_SHADER]) return;
    mglAlignInputsToOutputs(
        &program->spirv_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES],
        &program->spirv_resources_list[_FRAGMENT_SHADER][_STAGE_INPUT_RES]);
}

GLboolean mglProgramHasPassthroughGeometryShader(Program *program)
{
    const char *src = program && program->shader_slots[_GEOMETRY_SHADER]
        ? program->shader_slots[_GEOMETRY_SHADER]->src : NULL;
    if (!src) return GL_FALSE;
    return strstr(src, "EmitVertex()") &&
           strstr(src, "EndPrimitive()") &&
           strstr(src, "gl_Position = gl_in[n_vertex_index].gl_Position") &&
           !strstr(src, "gl_PrimitiveID") &&
           !strstr(src, "gl_Layer") &&
           !strstr(src, "gl_ViewportIndex");
}
