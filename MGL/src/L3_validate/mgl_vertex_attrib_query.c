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
 * mgl_vertex_attrib_query.m
 * MGL
 *
 * Implementation of the Vertex Attrib Query Subsystem.
 * See mgl_vertex_attrib_query.h for the API contract.
 *
 * Function bodies are preserved verbatim from MGLRenderer.m; only the
 * "static" storage-class qualifier was removed to make the symbols
 * externally visible.
 */

#include "mgl_vertex_attrib_query.h"
#include <stdbool.h>

#include "mgl_shader_abi.h"
#include <strings.h>        /* strcasecmp */

static GLuint mglAttribLocationSpan(const MGLShaderResource *res)
{
    if (!res) return 1u;
    return mglAIRVaryingLocationSpan(res->gl_type, res->gl_array_size);
}

bool mglRendererProgramUsesVertexAttrib(Program *program, GLuint attribute)
{
    if (attribute >= MAX_ATTRIBS) {
        return false;
    }
    if (!program) {
        return false;
    }

    MGLShaderResourceList *inputs =
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
    if (!inputs->list || inputs->count == 0) {
        return false;
    }

    for (GLuint i = 0; i < inputs->count; i++) {
        GLuint location = inputs->list[i].location;
        if (location == attribute) {
            return true;
        }

        /* Array / matrix stage inputs occupy consecutive locations.  The
         * AIR frontend reflects only the base location, but the generated
         * metallib flattens them into individual [[attribute(N)]] inputs
         * at [base, base + span - 1] (GL 4.6 §4.4.1). */
        GLuint span = mglAttribLocationSpan(&inputs->list[i]);
        if (span > 1u &&
            attribute >= location &&
            attribute < location + span) {
            return true;
        }

        /* No `location == UINT32_MAX` declaration-order fallback: the AIR
         * linker assigns every vertex stage input a location (explicit
         * layout(location), glBindAttribLocation, or declaration order) in
         * assignStageVarSymLocations, and applyVertexInputLocations only
         * overrides bound names -- measured with a probe: neither the local
         * suite nor the 1328-case GL46 hotspot list ever saw an input without
         * one. */
    }

    return false;
}

MGLShaderResource *mglRendererProgramVertexAttribResource(Program *program, GLuint attribute)
{
    if (!program || attribute >= MAX_ATTRIBS) {
        return NULL;
    }

    MGLShaderResourceList *inputs =
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
    if (!inputs->list || inputs->count == 0) {
        return NULL;
    }

    for (GLuint i = 0; i < inputs->count; i++) {
        GLuint location = inputs->list[i].location;
        if (location == attribute) {
            return &inputs->list[i];
        }

        GLuint span = mglAttribLocationSpan(&inputs->list[i]);
        if (span > 1u &&
            attribute >= location &&
            attribute < location + span) {
            return &inputs->list[i];
        }

        /* See mglRendererProgramUsesVertexAttrib: inputs always carry a
         * location on the IR chain, so there is no declaration-order fallback. */
    }

    return NULL;
}

bool mglRendererVertexAttribIsColorInput(Program *program, GLuint attribute)
{
    MGLShaderResource *resource = mglRendererProgramVertexAttribResource(program, attribute);
    const char *name = resource ? resource->name : NULL;
    return name &&
           (strcasecmp(name, "Color") == 0 ||
            strcasecmp(name, "a_Color") == 0 ||
            strcasecmp(name, "in_Color") == 0 ||
            strcasecmp(name, "vertColor") == 0 ||
           strcasecmp(name, "vertexColor") == 0 ||
           strcasecmp(name, "VertColor") == 0);
}

bool mglRendererVertexAttribUsesCurrentValue(VertexArray *vao, GLuint attribute)
{
    /* GL: a disabled generic attribute (including when no arrays are
     * enabled at all) feeds the current vertex attrib value.  An empty
     * VAO (enabled_attribs==0) used with DrawArrays is the attribless-
     * looking CTS path that still has `in` attributes in the VS. */
    return vao &&
           attribute < MAX_ATTRIBS &&
           (vao->enabled_attribs & (0x1u << attribute)) == 0u;
}
