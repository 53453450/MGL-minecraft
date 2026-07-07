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

#import "spirv_cross_c.h"   /* SPVC_RESOURCE_TYPE_STAGE_INPUT */

#include <strings.h>        /* strcasecmp */

BOOL mglRendererProgramUsesVertexAttrib(Program *program, GLuint attribute)
{
    if (attribute >= MAX_ATTRIBS) {
        return NO;
    }
    if (!program) {
        return NO;
    }

    SpirvResourceList *inputs =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    if (!inputs->list || inputs->count == 0) {
        return NO;
    }

    for (GLuint i = 0; i < inputs->count; i++) {
        GLuint location = inputs->list[i].location;
        if (location == attribute) {
            return YES;
        }

        /* Array stage inputs occupy consecutive locations.  SPIRV-Cross
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

SpirvResource *mglRendererProgramVertexAttribResource(Program *program, GLuint attribute)
{
    if (!program || attribute >= MAX_ATTRIBS) {
        return NULL;
    }

    SpirvResourceList *inputs =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
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
    SpirvResource *resource = mglRendererProgramVertexAttribResource(program, attribute);
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
    return vao &&
           attribute < MAX_ATTRIBS &&
           vao->enabled_attribs != 0u &&
           (vao->enabled_attribs & (0x1u << attribute)) == 0u;
}
