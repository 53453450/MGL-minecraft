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
 * mgl_vertex_attrib_plan.c
 * MGL
 *
 * Vertex-attribute buffer map plan.  See mgl_vertex_attrib_plan.h for the API
 * contract; the logic is a direct port of
 * -[MGLRenderer(Buffer) mapVertexAttributeBuffersToBufferMap:vao:stageInputCount:stage:]
 * with the per-attribute resolution moved behind MGLVertexAttribResolveFn.
 */

#include "mgl_vertex_attrib_plan.h"

#include "mgl_buffer_slots.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Vertex attribute buffer map                                         */
/* ------------------------------------------------------------------ */

int mglRenderMappedBufferCountOK(uint32_t count, uint32_t max)
{
    return count < max ? 1 : 0;
}

/* Worst-case diagnostic rate: the mismatch warning fires on the first hit and
 * then every 64th, like the ObjC loop it replaces. */
#define MGL_VATTR_MAP_MISMATCH_LOG_INTERVAL 64ull

int mglRenderPlanVertexAttribBuffers(BufferMapList *buffer_map,
                                     const MGLVertexAttribBufferPlanInput *in)
{
    int vao_buffer_start;
    int mapped_buffers = 0;
    GLuint next_vertex_binding_index = (GLuint)kMGLVertexAttribBufferBase;

    if (!buffer_map || !in) {
        return -1;
    }

    /* The caller gated on mglRenderStageMapsVertexAttribs(stage) and owns the
     * VAO validation (a NULL VAO maps nothing and never reaches here). */

    if (kMGLVertexAttribBufferBase >= kMGLMaxMetalVertexBufferCount) {
        fprintf(stderr,
                "MGL ERROR: invalid vertex attrib base index=%lu "
                "(max valid=%lu)\n",
                (unsigned long)kMGLVertexAttribBufferBase,
                (unsigned long)kMGLMaxMetalVertexBufferIndex);
        return -1;
    }

    /* vao buffers start after the uniforms and shader buffers */
    vao_buffer_start = buffer_map->count;
    if (!mglRenderMappedBufferCountOK((uint32_t)buffer_map->count,
                                      in->map_capacity)) {
        fprintf(stderr,
                "MGL SECURITY ERROR: buffer_map count %d exceeds "
                "MAX_MAPPED_BUFFERS %u\n",
                buffer_map->count, in->map_capacity);
        return -1;
    }
    memset(&buffer_map->buffers[vao_buffer_start], 0,
           sizeof(buffer_map->buffers[vao_buffer_start]));
    buffer_map->buffers[vao_buffer_start].buffer_base_index =
        (GLuint)kMGLVertexAttribBufferBase;
    buffer_map->buffers[vao_buffer_start].has_metal_binding =
        (GLboolean)GL_FALSE;

    for (int att = 0; att < MAX_ATTRIBS; att++) {
        MGLResolvedVertexAttribBinding resolved;
        Buffer *gl_buffer = NULL;

        if ((in->candidate_mask & (1u << att)) == 0u) {
            continue;
        }
        memset(&resolved, 0, sizeof(resolved));
        if (!in->resolve ||
            in->resolve(in->resolve_user, (GLuint)att, &resolved) != 0 ||
            !resolved.buffer) {
            fprintf(stderr,
                    "MGL WARNING: mapGLBuffersToMTLBufferMap: enabled attrib "
                    "%d has invalid/NULL buffer, skipping attrib\n",
                    att);
            continue;
        }
        gl_buffer = resolved.buffer;

        /* Empty slot maps here; only works on the first buffer. */
        if (buffer_map->buffers[vao_buffer_start].buf == NULL) {
            if (next_vertex_binding_index >= kMGLMaxMetalVertexBufferCount) {
                fprintf(stderr,
                        "MGL WARNING: vertex binding index overflow "
                        "(next=%u maxValid=%lu), skipping attrib %d\n",
                        next_vertex_binding_index,
                        (unsigned long)kMGLMaxMetalVertexBufferIndex, att);
                continue;
            }
            if (!mglRenderMappedBufferCountOK((uint32_t)buffer_map->count,
                                              in->map_capacity)) {
                fprintf(stderr,
                        "MGL WARNING: vertex buffer map is full (count=%u "
                        "max=%u), skipping attrib %d\n",
                        buffer_map->count, in->map_capacity, att);
                continue;
            }
            buffer_map->buffers[vao_buffer_start].attribute_mask |=
                (0x1u << att);
            buffer_map->buffers[vao_buffer_start].buf = gl_buffer;
            buffer_map->buffers[vao_buffer_start].buffer_base_index =
                next_vertex_binding_index++;
            buffer_map->buffers[vao_buffer_start].has_metal_binding =
                (GLboolean)GL_FALSE;
            buffer_map->buffers[vao_buffer_start].offset =
                resolved.binding_offset;
            buffer_map->buffers[vao_buffer_start].size = 0;
            buffer_map->count++;

            mapped_buffers++;
            continue;
        }

        {
            bool found_buffer = false;

            /* Find an already-mapped entry with the same buffer.  Name and
             * target are compared (not pointers), and offset is intentionally
             * NOT compared: attributes sharing the same VBO/stride/divisor are
             * grouped into one Metal buffer slot with per-attribute offsets
             * expressed through the vertex descriptor. */
            for (int map = vao_buffer_start;
                 (found_buffer == false) && map < (int)buffer_map->count;
                 map++) {
                Buffer *map_buffer = buffer_map->buffers[map].buf;
                if (!map_buffer) {
                    continue;
                }
                if ((map_buffer->name == gl_buffer->name) &&
                    (map_buffer->target == gl_buffer->target)) {
                    bool compatibleStream = true;
                    for (GLuint prevAttrib = 0; prevAttrib < MAX_ATTRIBS;
                         prevAttrib++) {
                        MGLResolvedVertexAttribBinding prevResolved;
                        if ((buffer_map->buffers[map].attribute_mask &
                             (0x1u << prevAttrib)) == 0u) {
                            continue;
                        }
                        memset(&prevResolved, 0, sizeof(prevResolved));
                        if (!in->resolve ||
                            in->resolve(in->resolve_user, prevAttrib,
                                        &prevResolved) != 0) {
                            continue;
                        }
                        if (prevResolved.stride != resolved.stride ||
                            prevResolved.divisor != resolved.divisor) {
                            compatibleStream = false;
                            break;
                        }
                    }
                    if (compatibleStream) {
                        /* include it in the list of attributes */
                        buffer_map->buffers[map].attribute_mask |=
                            (0x1u << att);
                        found_buffer = true;
                        mapped_buffers++;
                        break;
                    }
                }
            }

            if (found_buffer == false) {
                if (next_vertex_binding_index >= kMGLMaxMetalVertexBufferCount) {
                    fprintf(stderr,
                            "MGL WARNING: vertex binding index overflow "
                            "(next=%u maxValid=%lu), cannot append attrib %d\n",
                            next_vertex_binding_index,
                            (unsigned long)kMGLMaxMetalVertexBufferIndex, att);
                    continue;
                }
                if (!mglRenderMappedBufferCountOK((uint32_t)buffer_map->count,
                                                  in->map_capacity)) {
                    fprintf(stderr,
                            "MGL WARNING: vertex buffer map is full (count=%u "
                            "max=%u), cannot append attrib %d\n",
                            buffer_map->count, in->map_capacity, att);
                    continue;
                }
                buffer_map->buffers[buffer_map->count].attribute_mask =
                    (0x1u << att);
                buffer_map->buffers[buffer_map->count].buffer_base_index =
                    next_vertex_binding_index++;
                buffer_map->buffers[buffer_map->count].resource_type = 0;
                buffer_map->buffers[buffer_map->count].resource_index = 0;
                buffer_map->buffers[buffer_map->count].metal_binding_index = 0;
                buffer_map->buffers[buffer_map->count].has_metal_binding =
                    (GLboolean)GL_FALSE;
                buffer_map->buffers[buffer_map->count].buf = gl_buffer;
                buffer_map->buffers[buffer_map->count].offset =
                    resolved.binding_offset;
                buffer_map->buffers[buffer_map->count].size = 0;
                buffer_map->count++;

                mapped_buffers++;
            }
        }
    }

    if (mapped_buffers != in->stage_input_count) {
        static unsigned long long s_map_mismatch_hits = 0;
        s_map_mismatch_hits++;
        if ((s_map_mismatch_hits % MGL_VATTR_MAP_MISMATCH_LOG_INTERVAL) == 1ull) {
            fprintf(stderr,
                    "MGL WARNING: mapGLBuffersToMTLBufferMap mismatch "
                    "(pipeline=%p mapped=%u expected=%d stage=%d hit=%llu "
                    "indexBuffer=%p vao=%p)\n",
                    in->pipeline_state, mapped_buffers, in->stage_input_count,
                    in->stage, s_map_mismatch_hits, in->index_buffer_metal,
                    in->vao);
        }
    }

    return 0;
}
