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

#import "mgl_vertex_attrib_query.h"


#include <strings.h>        /* strcasecmp */

BOOL mglRendererProgramUsesVertexAttrib(Program *program, GLuint attribute)
{
    if (attribute >= MAX_ATTRIBS) {
        return NO;
    }
    if (!program) {
        return NO;
    }

    MGLShaderResourceList *inputs =
        &program->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
    if (!inputs->list || inputs->count == 0) {
        return NO;
    }

    for (GLuint i = 0; i < inputs->count; i++) {
        GLuint location = inputs->list[i].location;
        if (location == attribute) {
            return YES;
        }

        /* Array stage inputs occupy consecutive locations.  The AIR frontend
         * reflects only the base location, but the generated MSL flattens
         * the array into individual [[attribute(N)]] inputs at locations
         * [base, base + array_size - 1]. */
        if (inputs->list[i].gl_array_size > 1 &&
            attribute >= location &&
            attribute < location + (GLuint)inputs->list[i].gl_array_size) {
            return YES;
        }

        if (location == 0xffffffffu && i == attribute) {
            return YES;
        }
    }

    return NO;
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

        /* Array stage inputs span consecutive locations (see
         * mglRendererProgramUsesVertexAttrib). */
        if (inputs->list[i].gl_array_size > 1 &&
            attribute >= location &&
            attribute < location + (GLuint)inputs->list[i].gl_array_size) {
            return &inputs->list[i];
        }

        if (location == 0xffffffffu && i == attribute) {
            return &inputs->list[i];
        }
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

BOOL mglRendererVertexAttribUsesCurrentValue(VertexArray *vao, GLuint attribute)
{
    /* GL: a disabled generic attribute (including when no arrays are
     * enabled at all) feeds the current vertex attrib value.  An empty
     * VAO (enabled_attribs==0) used with DrawArrays is the attribless-
     * looking CTS path that still has `in` attributes in the VS. */
    return vao &&
           attribute < MAX_ATTRIBS &&
           (vao->enabled_attribs & (0x1u << attribute)) == 0u;
}
