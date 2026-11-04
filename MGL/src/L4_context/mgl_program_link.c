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

#include "mgl_trace_log.h"
#include <stdbool.h>

Program *findProgram(GLMContext ctx, GLuint program);
GLuint mglProgramAttachedShaderCount(Program *program, GLuint stage);
GLboolean mglProgramHasAttachedShader(Program *program, GLuint stage, Shader *shader);

GLuint mglTransformFeedbackTypeComponents(GLuint type);

bool mglTransformFeedbackBaseName(const char *name,                                         char out[96]);

int mglCompileCaptureVariant(const char *src,                                    const char *const *attrib_names,                                    uint32_t flags,                                    unsigned char **bytes, size_t *size,                                    char *err, size_t err_cap);

const char *mglSkipGLSLVersionDirective(const char *s);

char *mglConcatSameStageSources(Program *pptr, int stage);

void mglCopyExplicitBinding(MGLShaderResource *dst,                                   const MGLShaderResource *src);

void mglProgramPushBuiltinResource(MGLShaderResource *slots, GLuint *count,                                          const char *name, GLuint gl_type,                                          GLint array_size, GLboolean is_array);


/* A16: keep the last successful executable until a new link commits. */
typedef struct MGLSavedLinkExecutable {
    MGLShaderModule modules[_MAX_SHADER_TYPES];
    unsigned char module_backends[MGL_RENDER_PROGRAM_MODULE_BACKEND_BLOB];
    MGLShaderResourceList shader_resources_list[_MAX_SHADER_TYPES][MGL_MAX_SHADER_RESOURCES];
    GLboolean uses_vertex_id;
    GLboolean uses_primitive_id;
    GLboolean usesFragCoordParams;
    GLboolean uses_sample_params;
    uint32_t vertexAttribUsageMask;
    GLboolean uses_point_size_params;
    GLboolean uses_shader_draw_params;
    GLboolean uses_cull_distance;
    uint32_t cull_distance_count;
    uint32_t clip_distance_count;
    GLboolean tess_uses_cull_distance;
    uint32_t tess_cull_distance_count;
    GLboolean uses_lod_bias;
    MGLGSRoute gs_route;
    GLenum geometry_input_type;
    GLenum geometry_output_type;
    GLuint geometry_vertices_out;
    GLboolean geometry_max_vertices_specified;
    GLuint geometry_invocations;
    GLuint geometry_stream_count;
    GLuint geometry_stream_varying_count[4];
    GLuint geometry_stream_xfb_stride[4];
    GLuint tess_control_output_vertices;
    GLenum tess_gen_mode;
    GLenum tess_gen_spacing;
    GLenum tess_gen_vertex_order;
    GLboolean tess_gen_point_mode;
    GLboolean tess_eval_compute;
    GLboolean tess_eval_render_vertex;
    GLboolean tess_gen_mode_specified;
    struct { unsigned x, y, z; } local_workgroup_size;
    GLuint shared_memory_bytes;
    GLint legacy_clip_plane_loc;
    GLint legacy_clip_plane_enabled_loc;
    uint8_t sampler_binding_shared[TEXTURE_UNITS];
    uint8_t sampler_binding_shared_valid;
    uint64_t sampler_location_bitmap[2];
    uint8_t sampler_location_bitmap_valid;
    uint32_t sampled_texture_unit_mask[4];
    uint8_t sampled_texture_unit_mask_valid;
} MGLSavedLinkExecutable;

static void mglCaptureLinkExecutable(Program *pptr, MGLSavedLinkExecutable *out)
{
    memset(out, 0, sizeof(*out));
    memcpy(out->modules, pptr->modules, sizeof(out->modules));
    mglRenderProgramCaptureModuleBackends(pptr, out->module_backends,
                                          sizeof(out->module_backends));
    memcpy(out->shader_resources_list, pptr->shader_resources_list,
           sizeof(out->shader_resources_list));
    out->uses_vertex_id = pptr->uses_vertex_id;
    out->uses_primitive_id = pptr->uses_primitive_id;
    out->usesFragCoordParams = pptr->usesFragCoordParams;
    out->uses_sample_params = pptr->uses_sample_params;
    out->vertexAttribUsageMask = pptr->vertexAttribUsageMask;
    out->uses_point_size_params = pptr->uses_point_size_params;
    out->uses_shader_draw_params = pptr->uses_shader_draw_params;
    out->uses_cull_distance = pptr->uses_cull_distance;
    out->cull_distance_count = pptr->cull_distance_count;
    out->clip_distance_count = pptr->clip_distance_count;
    out->tess_uses_cull_distance = pptr->tess_uses_cull_distance;
    out->tess_cull_distance_count = pptr->tess_cull_distance_count;
    out->uses_lod_bias = pptr->uses_lod_bias;
    out->gs_route = pptr->gs_route;
    out->geometry_input_type = pptr->geometry_input_type;
    out->geometry_output_type = pptr->geometry_output_type;
    out->geometry_vertices_out = pptr->geometry_vertices_out;
    out->geometry_max_vertices_specified = pptr->geometry_max_vertices_specified;
    out->geometry_invocations = pptr->geometry_invocations;
    out->geometry_stream_count = pptr->geometry_stream_count;
    memcpy(out->geometry_stream_varying_count, pptr->geometry_stream_varying_count,
           sizeof(out->geometry_stream_varying_count));
    memcpy(out->geometry_stream_xfb_stride, pptr->geometry_stream_xfb_stride,
           sizeof(out->geometry_stream_xfb_stride));
    out->tess_control_output_vertices = pptr->tess_control_output_vertices;
    out->tess_gen_mode = pptr->tess_gen_mode;
    out->tess_gen_spacing = pptr->tess_gen_spacing;
    out->tess_gen_vertex_order = pptr->tess_gen_vertex_order;
    out->tess_gen_point_mode = pptr->tess_gen_point_mode;
    out->tess_eval_compute = pptr->tess_eval_compute;
    out->tess_eval_render_vertex = pptr->tess_eval_render_vertex;
    out->tess_gen_mode_specified = pptr->tess_gen_mode_specified;
    out->local_workgroup_size.x = pptr->local_workgroup_size.x;
    out->local_workgroup_size.y = pptr->local_workgroup_size.y;
    out->local_workgroup_size.z = pptr->local_workgroup_size.z;
    out->shared_memory_bytes = pptr->shared_memory_bytes;
    out->legacy_clip_plane_loc = pptr->legacy_clip_plane_loc;
    out->legacy_clip_plane_enabled_loc = pptr->legacy_clip_plane_enabled_loc;
    memcpy(out->sampler_binding_shared, pptr->sampler_binding_shared,
           sizeof(out->sampler_binding_shared));
    out->sampler_binding_shared_valid = pptr->sampler_binding_shared_valid;
    out->sampler_location_bitmap[0] = pptr->sampler_location_bitmap[0];
    out->sampler_location_bitmap[1] = pptr->sampler_location_bitmap[1];
    out->sampler_location_bitmap_valid = pptr->sampler_location_bitmap_valid;
    memcpy(out->sampled_texture_unit_mask, pptr->sampled_texture_unit_mask,
           sizeof(out->sampled_texture_unit_mask));
    out->sampled_texture_unit_mask_valid = pptr->sampled_texture_unit_mask_valid;

    memset(pptr->modules, 0, sizeof(pptr->modules));
    mglRenderProgramClearModuleBackends(pptr);
    memset(pptr->shader_resources_list, 0, sizeof(pptr->shader_resources_list));
}

static void mglDiscardSavedLinkExecutable(MGLSavedLinkExecutable *saved)
{
    Program tmp;
    memset(&tmp, 0, sizeof(tmp));
    memcpy(tmp.modules, saved->modules, sizeof(tmp.modules));
    memcpy(tmp.shader_resources_list, saved->shader_resources_list,
           sizeof(tmp.shader_resources_list));
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        clearStageCompileState(&tmp, stage);
        mglRenderProgramClearCapturedModuleBackend(saved->module_backends,
                                                   sizeof(saved->module_backends),
                                                   stage);
    }
    memset(saved, 0, sizeof(*saved));
}

static void mglRestoreSavedLinkExecutable(Program *pptr, MGLSavedLinkExecutable *saved)
{
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        clearStageCompileState(pptr, stage);
    }
    memcpy(pptr->modules, saved->modules, sizeof(pptr->modules));
    mglRenderProgramRestoreModuleBackends(pptr, saved->module_backends,
                                          sizeof(saved->module_backends));
    memcpy(pptr->shader_resources_list, saved->shader_resources_list,
           sizeof(pptr->shader_resources_list));
    pptr->uses_vertex_id = saved->uses_vertex_id;
    pptr->uses_primitive_id = saved->uses_primitive_id;
    pptr->usesFragCoordParams = saved->usesFragCoordParams;
    pptr->uses_sample_params = saved->uses_sample_params;
    pptr->vertexAttribUsageMask = saved->vertexAttribUsageMask;
    pptr->uses_point_size_params = saved->uses_point_size_params;
    pptr->uses_shader_draw_params = saved->uses_shader_draw_params;
    pptr->uses_cull_distance = saved->uses_cull_distance;
    pptr->cull_distance_count = saved->cull_distance_count;
    pptr->clip_distance_count = saved->clip_distance_count;
    pptr->tess_uses_cull_distance = saved->tess_uses_cull_distance;
    pptr->tess_cull_distance_count = saved->tess_cull_distance_count;
    pptr->uses_lod_bias = saved->uses_lod_bias;
    pptr->gs_route = saved->gs_route;
    pptr->geometry_input_type = saved->geometry_input_type;
    pptr->geometry_output_type = saved->geometry_output_type;
    pptr->geometry_vertices_out = saved->geometry_vertices_out;
    pptr->geometry_max_vertices_specified = saved->geometry_max_vertices_specified;
    pptr->geometry_invocations = saved->geometry_invocations;
    pptr->geometry_stream_count = saved->geometry_stream_count;
    memcpy(pptr->geometry_stream_varying_count, saved->geometry_stream_varying_count,
           sizeof(pptr->geometry_stream_varying_count));
    memcpy(pptr->geometry_stream_xfb_stride, saved->geometry_stream_xfb_stride,
           sizeof(pptr->geometry_stream_xfb_stride));
    pptr->tess_control_output_vertices = saved->tess_control_output_vertices;
    pptr->tess_gen_mode = saved->tess_gen_mode;
    pptr->tess_gen_spacing = saved->tess_gen_spacing;
    pptr->tess_gen_vertex_order = saved->tess_gen_vertex_order;
    pptr->tess_gen_point_mode = saved->tess_gen_point_mode;
    pptr->tess_eval_compute = saved->tess_eval_compute;
    pptr->tess_eval_render_vertex = saved->tess_eval_render_vertex;
    pptr->tess_gen_mode_specified = saved->tess_gen_mode_specified;
    pptr->local_workgroup_size.x = saved->local_workgroup_size.x;
    pptr->local_workgroup_size.y = saved->local_workgroup_size.y;
    pptr->local_workgroup_size.z = saved->local_workgroup_size.z;
    pptr->shared_memory_bytes = saved->shared_memory_bytes;
    pptr->legacy_clip_plane_loc = saved->legacy_clip_plane_loc;
    pptr->legacy_clip_plane_enabled_loc = saved->legacy_clip_plane_enabled_loc;
    memcpy(pptr->sampler_binding_shared, saved->sampler_binding_shared,
           sizeof(pptr->sampler_binding_shared));
    pptr->sampler_binding_shared_valid = saved->sampler_binding_shared_valid;
    pptr->sampler_location_bitmap[0] = saved->sampler_location_bitmap[0];
    pptr->sampler_location_bitmap[1] = saved->sampler_location_bitmap[1];
    pptr->sampler_location_bitmap_valid = saved->sampler_location_bitmap_valid;
    memcpy(pptr->sampled_texture_unit_mask, saved->sampled_texture_unit_mask,
           sizeof(pptr->sampled_texture_unit_mask));
    pptr->sampled_texture_unit_mask_valid = saved->sampled_texture_unit_mask_valid;
    memset(saved, 0, sizeof(*saved));
    /* Failed relink keeps the prior executable for currently-bound draw
     * (GL 4.6 glLinkProgram), but LINK_STATUS must stay FALSE. */
    pptr->link_success = GL_FALSE;
    pptr->link_state = MGL_PROGRAM_LINK_FAILED;
    pptr->dirty_bits |= DIRTY_PROGRAM;
    mglBuildActiveUniformCache(pptr);
}

static void mglProgramClearPlainUniformStorage(GLMContext ctx, Program *pptr)
{
    if (!ctx || !pptr) {
        return;
    }
    for (GLuint w = 0u; w < MGL_PLAIN_UNIFORM_ACTIVE_WORDS; w++) {
        uint64_t bits = pptr->plain_uniform_active_mask[w];
        while (bits) {
            unsigned b = (unsigned)__builtin_ctzll(bits);
            bits &= bits - 1ull;
            GLuint i = w * 64u + (GLuint)b;
            if (i >= MAX_PLAIN_UNIFORM_LOCATIONS) {
                continue;
            }
            BufferBaseTarget *slot = &pptr->plain_uniform_buffers[i];
            Buffer *buf = slot->buf;
            if (buf) {
                deleteHashElement(&STATE(buffer_table), buf->name);
                buf->delete_status = GL_TRUE;
                mglReleaseBufferReference(ctx, buf);
            }
            memset(slot, 0, sizeof(*slot));
        }
    }
    memset(pptr->plain_uniform_active_mask, 0,
           sizeof(pptr->plain_uniform_active_mask));
}

/* gl_-prefixed uniforms can only come from the legacy matrix injection.
 * Compatibility profile §12.1.1: every fixed-function matrix starts as
 * identity, as do its inverse/transpose forms. */
static void mglSeedLegacyMatrixIdentity(GLMContext ctx, Program *pptr)
{
    enum { kMaxElems = 8 }; /* gl_TextureMatrix[gl_MaxTextureCoords] */
    GLfloat ident[kMaxElems * 16];
    Program *prev_prog = STATE(program);
    GLuint prev_name = STATE(program_name);
    STATE(program) = pptr;
    STATE(program_name) = pptr->name;

    GLint count = mglProgramActiveUniformCount(pptr);
    for (GLint i = 0; i < count; i++) {
        GLchar name[64];
        GLint size = 0;
        GLenum type = 0;
        mglGetActiveUniform(ctx, pptr->name, (GLuint)i, (GLsizei)sizeof name,
                            NULL, &size, &type, name);
        if (strncmp(name, "gl_", 3) != 0 ||
            (type != GL_FLOAT_MAT4 && type != GL_FLOAT_MAT3)) {
            continue;
        }
        GLint loc = mglGetUniformLocation(ctx, pptr->name, name);
        if (loc < 0 || size < 1 || size > kMaxElems) {
            continue;
        }
        GLuint n = (type == GL_FLOAT_MAT4) ? 4u : 3u;
        memset(ident, 0, sizeof ident);
        for (GLint e = 0; e < size; e++) {
            for (GLuint d = 0; d < n; d++) {
                ident[(GLuint)e * n * n + d * n + d] = 1.0f;
            }
        }
        if (type == GL_FLOAT_MAT4) {
            mglUniformMatrix4fv(ctx, loc, size, GL_FALSE, ident);
        } else {
            mglUniformMatrix3fv(ctx, loc, size, GL_FALSE, ident);
        }
    }

    STATE(program) = prev_prog;
    STATE(program_name) = prev_name;
}

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
     * provides transform feedback outputs).  GL desktop also allows
     * capturing from a tessellation control–only separable program. */
    int feedback_stage = -1;
    if (pptr->attached_shader_mask & GEOMETRY_SHADER_MASK_BIT)
        feedback_stage = _GEOMETRY_SHADER;
    else if (pptr->attached_shader_mask & TESS_EVALUATION_SHADER_MASK_BIT)
        feedback_stage = _TESS_EVALUATION_SHADER;
    else if (pptr->attached_shader_mask & TESS_CONTROL_SHADER_MASK_BIT)
        feedback_stage = _TESS_CONTROL_SHADER;
    else if (pptr->attached_shader_mask & VERTEX_SHADER_MASK_BIT)
        feedback_stage = _VERTEX_SHADER;
    if (feedback_stage < 0)
        return false;

    const GLuint maxInterleaved = ctx
        ? STATE(var).max_transform_feedback_interleaved_components : 64u;
    const GLuint maxSeparateComponents = ctx
        ? STATE(var).max_transform_feedback_separate_components : 4u;
    const GLuint maxBuffers = ctx
        ? STATE(var).max_transform_feedback_buffers
        : MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS;
    const GLuint maxSeparateAttribs = ctx
        ? STATE(var).max_transform_feedback_separate_attribs
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
            if (pptr->transform_feedback_buffer_mode == GL_SEPARATE_ATTRIBS) {
                /* Same buffer assignment as user varyings: one buffer per
                 * varying index. Without this, gl_Position kept the previous
                 * buffer index and collided with the prior SEPARATE attrib
                 * (CTS capture_*_separate packs Position in the last slot). */
                buffer = (GLuint)i;
                if (buffer >= maxSeparateAttribs ||
                    components > maxSeparateComponents)
                    return false;
            }
            pptr->transform_feedback_layout[i].buffer_index = buffer;
            pptr->transform_feedback_layout[i].component_offset =
                bufferOffsets[buffer];
            pptr->transform_feedback_layout[i].component_count = components;
            /* Built-in per-vertex outputs feed stream 0 like any other
             * captured varying. */
            pptr->transform_feedback_layout[i].stream = 0;
            pptr->transform_feedback_layout[i].builtin = GL_TRUE;
            bufferOffsets[buffer] += components;
            continue;
        }

        MGLShaderResource *output =
            mglProgramFindStageOutputForXFBName(pptr, feedback_stage, name);
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
    /* MGL_TRACE_XFB=1 prints the link-time scatter plan: which varying lands
     * in which buffer at which component offset, and how many components it
     * claims.  A matrix varying is several locations wide, so the component
     * count is the first thing to check when a capture reads zeros. */
    if (getenv("MGL_TRACE_XFB")) {
        fprintf(stderr, "MGL XFB PLAN: program=%u varyings=%d mode=0x%x\n",
                (unsigned)pptr->name, (int)pptr->transform_feedback_varying_count,
                (unsigned)pptr->transform_feedback_buffer_mode);
        for (GLsizei i = 0; i < pptr->transform_feedback_varying_count; i++) {
            const MGLTransformFeedbackVaryingPlan *pl =
                &pptr->transform_feedback_layout[i];
            fprintf(stderr,
                    "MGL XFB PLAN   [%d] '%s' comps=%u off=%u buf=%d stream=%d\n",
                    (int)i,
                    pptr->transform_feedback_varying_names[i]
                        ? pptr->transform_feedback_varying_names[i] : "?",
                    (unsigned)pl->component_count, (unsigned)pl->component_offset,
                    (int)pl->buffer_index, (int)pl->stream);
        }
    }
    pptr->transform_feedback_layout_valid = GL_TRUE;
    return true;
}

static int mglComputeLocalSizeLinkOk(Program *pptr)
{
    int32_t lx = -1, ly = -1, lz = -1;
    int declared = 0;
    GLuint n = mglProgramAttachedShaderCount(pptr, _COMPUTE_SHADER);
    GLuint i;
    for (i = 0u; i < n; i++) {
        Shader *sh = (pptr->attached_shader_counts[_COMPUTE_SHADER] > 0u)
            ? pptr->attached_shader_slots[_COMPUTE_SHADER][i]
            : pptr->shader_slots[_COMPUTE_SHADER];
        const MGLTranslationUnit *tu = sh ? sh->frontend_tu : NULL;
        if (!tu) {
            continue;
        }
        int32_t tx = -1, ty = -1, tz = -1;
        mglCompileTUGetComputeLocalSize(tu, &tx, &ty, &tz);
        if (tx < 0 && ty < 0 && tz < 0) {
            continue;
        }
        declared = 1;
        if (tx >= 0) {
            if (lx >= 0 && lx != tx) {
                return 0;
            }
            lx = tx;
        }
        if (ty >= 0) {
            if (ly >= 0 && ly != ty) {
                return 0;
            }
            ly = ty;
        }
        if (tz >= 0) {
            if (lz >= 0 && lz != tz) {
                return 0;
            }
            lz = tz;
        }
    }
    return declared;
}

/* mglAirCompileStage → mgl_program_air_compile.c */
int mglAirCompileStage(GLMContext ctx, Program *pptr, int stage);

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
        /* GL 4.6 §11.1.3.9: geometry shader user inputs must be declared
         * as arrays (per-input-vertex); a scalar input is a link error.
         * Interface-block members are exempt — their instance carries
         * the per-vertex dimension. */
        if (!in->block_member && !in->is_array) {
            return GL_FALSE;
        }
        /* Interface-block members carry their own ordinary array
         * dimension; only the block instance itself is indexed by the
         * input-vertex count. */
        if (!in->block_member && in->is_array && in->gl_array_size > 0 &&
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

static bool mglValidateAtomicCounterOffsetOverlap(Program *pptr)
{
    if (!pptr) {
        return false;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        const MGLShaderResourceList *acs =
            &pptr->shader_resources_list[stage][_ATOMIC_COUNTER_RES];
        for (GLuint i = 0u; i < acs->count; i++) {
            const MGLShaderResource *a = &acs->list[i];
            GLuint a_size = (GLuint)a->required_size;
            if (a_size == 0u) {
                a_size = (GLuint)sizeof(GLuint);
            }
            for (int stage2 = stage; stage2 < _MAX_SHADER_TYPES; stage2++) {
                const MGLShaderResourceList *bcs =
                    &pptr->shader_resources_list[stage2][_ATOMIC_COUNTER_RES];
                for (GLuint j = (stage2 == stage) ? i + 1u : 0u;
                     j < bcs->count; j++) {
                    const MGLShaderResource *b = &bcs->list[j];
                    if (a->gl_binding != b->gl_binding) {
                        continue;
                    }
                    if (a->name && b->name && strcmp(a->name, b->name) == 0) {
                        continue;
                    }
                    GLuint b_size = (GLuint)b->required_size;
                    if (b_size == 0u) {
                        b_size = (GLuint)sizeof(GLuint);
                    }
                    if (a->location < b->location + b_size &&
                        b->location < a->location + a_size) {
                        fprintf(stderr,
                                "MGL WARNING: mglLinkProgram failed program %u: "
                                "atomic counters '%s' (offset %u) and '%s' (offset %u) "
                                "overlap on binding %u\n",
                                pptr->name,
                                a->name ? a->name : "(null)",
                                a->location,
                                b->name ? b->name : "(null)",
                                b->location,
                                a->gl_binding);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static bool mglValidateCombinedClipAndCullDistances(GLMContext ctx,
                                                    Program *pptr)
{
    GLuint maxCombined =
        ctx && STATE(var).max_combined_clip_and_cull_distances
            ? STATE(var).max_combined_clip_and_cull_distances
            : 8u;
    GLuint clipSize = pptr->clip_distance_count;
    GLuint cullSize = pptr->cull_distance_count;
    if (pptr->tess_cull_distance_count > cullSize)
        cullSize = pptr->tess_cull_distance_count;

    /* Only sizes proven by IR symbols participate. */
    if (clipSize == 0u || cullSize == 0u) return true;
    if (clipSize + cullSize <= maxCombined) return true;

    fprintf(stderr,
            "MGL WARNING: mglLinkProgram failed program %u: "
            "gl_ClipDistance[%u] + gl_CullDistance[%u] exceeds "
            "MAX_COMBINED_CLIP_AND_CULL_DISTANCES (%u)\n",
            pptr->name, clipSize, cullSize, maxCombined);
    return false;
}

static bool mglValidateCrossStageLayoutBindings(Program *pptr)
{
    static const int types[] = {
        _SAMPLED_IMAGE_RES,
        _STORAGE_IMAGE_RES,
        _ATOMIC_COUNTER_RES,
        _UNIFORM_BUFFER_RES,
        _STORAGE_BUFFER_RES
    };
    if (!pptr) {
        return false;
    }
    for (size_t ti = 0; ti < sizeof(types) / sizeof(types[0]); ti++) {
        int rt = types[ti];
        for (int s1 = 0; s1 < _MAX_SHADER_TYPES; s1++) {
            MGLShaderResourceList *a =
                &pptr->shader_resources_list[s1][rt];
            for (GLuint i = 0u; i < a->count; i++) {
                const char *name = a->list[i].name;
                if (!name || name[0] == '\0') {
                    continue;
                }
                MGLShaderResource *ra = &a->list[i];
                for (int s2 = s1 + 1; s2 < _MAX_SHADER_TYPES; s2++) {
                    MGLShaderResourceList *b =
                        &pptr->shader_resources_list[s2][rt];
                    for (GLuint j = 0u; j < b->count; j++) {
                        MGLShaderResource *rb = &b->list[j];
                        if (!rb->name || strcmp(name, rb->name) != 0) {
                            continue;
                        }
                        if (ra->binding_explicit && !rb->binding_explicit) {
                            mglCopyExplicitBinding(rb, ra);
                        } else if (rb->binding_explicit &&
                                   !ra->binding_explicit) {
                            mglCopyExplicitBinding(ra, rb);
                        } else if (ra->binding_explicit &&
                                   ra->gl_binding != rb->gl_binding) {
                            fprintf(stderr,
                                    "MGL LINK ERROR: program %u uniform '%s' "
                                    "binding %u vs %u across stages\n",
                                    pptr->name, name, ra->gl_binding,
                                    rb->gl_binding);
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

static void mglProgramClearBuiltinInterface(Program *pptr)
{
    if (!pptr)
        return;
    for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
        for (GLuint i = 0; i < pptr->builtin_program_input_count[s] && i < 16; i++) {
            free((void *)pptr->builtin_program_inputs[s][i].name);
            pptr->builtin_program_inputs[s][i].name = NULL;
        }
        memset(pptr->builtin_program_inputs[s], 0,
               sizeof(pptr->builtin_program_inputs[s]));
        pptr->builtin_program_input_count[s] = 0;
        for (GLuint i = 0; i < pptr->builtin_program_output_count[s] && i < 16; i++) {
            free((void *)pptr->builtin_program_outputs[s][i].name);
            pptr->builtin_program_outputs[s][i].name = NULL;
        }
        memset(pptr->builtin_program_outputs[s], 0,
               sizeof(pptr->builtin_program_outputs[s]));
        pptr->builtin_program_output_count[s] = 0;
    }
}

static void mglProgramApplyBuiltinInterface(Program *pptr)
{
    if (!pptr)
        return;
    mglProgramClearBuiltinInterface(pptr);
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        Shader *sh = pptr->shader_slots[stage];
        MGLTranslationUnit *tu = sh ? sh->frontend_tu : NULL;
        if (!tu)
            continue;
        if (stage == _VERTEX_SHADER) {
            if (mglCompileTUBuiltinUsed(tu, "gl_VertexID"))
                mglProgramPushBuiltinResource(
                    pptr->builtin_program_inputs[stage],
                    &pptr->builtin_program_input_count[stage],
                    "gl_VertexID", GL_INT, 1, GL_FALSE);
            if (mglCompileTUBuiltinUsed(tu, "gl_InstanceID"))
                mglProgramPushBuiltinResource(
                    pptr->builtin_program_inputs[stage],
                    &pptr->builtin_program_input_count[stage],
                    "gl_InstanceID", GL_INT, 1, GL_FALSE);
        }
        if (stage == _FRAGMENT_SHADER) {
            if (mglCompileTUBuiltinUsed(tu, "gl_FragDepth"))
                mglProgramPushBuiltinResource(
                    pptr->builtin_program_outputs[stage],
                    &pptr->builtin_program_output_count[stage],
                    "gl_FragDepth", GL_FLOAT, 1, GL_FALSE);
            if (mglCompileTUBuiltinUsed(tu, "gl_SampleMask"))
                mglProgramPushBuiltinResource(
                    pptr->builtin_program_outputs[stage],
                    &pptr->builtin_program_output_count[stage],
                    "gl_SampleMask", GL_INT, 1, GL_TRUE);
        }
        if (stage == _TESS_CONTROL_SHADER) {
            if (mglCompileTUBuiltinUsed(tu, "gl_InvocationID"))
                mglProgramPushBuiltinResource(
                    pptr->builtin_program_inputs[stage],
                    &pptr->builtin_program_input_count[stage],
                    "gl_InvocationID", GL_INT, 1, GL_FALSE);
        }
    }
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
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);

        return;
    }

    /* GL 4.6 §13.2.2 / ARB_transform_feedback2: cannot relink a program
     * that is in use by an active (including paused) transform feedback
     * object. */
    if (STATE(program_name) == program &&
        STATE(transform_feedback) && STATE(transform_feedback)->active) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    mglFlushPendingDraws(ctx);

    /* Validate attachments before tearing down a prior executable (A16).
     * GL keeps the last successful link when a new link fails. */
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

    MGLSavedLinkExecutable saved_executable;
    GLboolean have_saved_executable = GL_FALSE;
    if (pptr->link_success ||
        mglRenderProgramModuleMetallibBytes(pptr, _VERTEX_SHADER) ||
        mglRenderProgramModuleMetallibBytes(pptr, _FRAGMENT_SHADER) ||
        mglRenderProgramModuleMetallibBytes(pptr, _COMPUTE_SHADER) ||
        mglRenderProgramModuleMetallibBytes(pptr, _GEOMETRY_SHADER) ||
        mglRenderProgramModuleMetallibBytes(pptr, _TESS_CONTROL_SHADER) ||
        mglRenderProgramModuleMetallibBytes(pptr, _TESS_EVALUATION_SHADER)) {
        mglCaptureLinkExecutable(pptr, &saved_executable);
        have_saved_executable = GL_TRUE;
    }

    pptr->link_success = GL_FALSE;
    pptr->spirv_nameless = GL_FALSE;
    pptr->link_state = MGL_PROGRAM_LINK_FAILED;
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        Shader *old_slot = pptr->shader_slots[stage];
        Shader *new_slot = NULL;
        if (pptr->attached_shader_counts[stage] > 0u) {
            new_slot = pptr->attached_shader_slots[stage][0];
        } else if (!have_saved_executable) {
            clearStageCompileState(pptr, stage);
        }
        if (old_slot && old_slot != new_slot &&
            !mglProgramHasAttachedShader(pptr, stage, old_slot)) {
            old_slot->refcount--;
            if (old_slot->refcount == 0 && old_slot->delete_status) {
                deleteHashElement(&STATE(shader_table), old_slot->name);
                mglFreeShader(ctx, old_slot);
            }
        }
        pptr->shader_slots[stage] = new_slot;
    }

    /* C++ compute PSOs retain functions from the previous link generation.
     * Drop them before stage objects and metallib libraries are replaced. */
    mglRenderInvalidateProgramPipelines(
        mglRenderProgramPipelineCacheInstanceId(pptr));

    pptr->uses_vertex_id = GL_FALSE;
    pptr->uses_primitive_id = GL_FALSE;
    /* Invalidate MSL query result cache; repopulated from the freshly generated MSL
     * after the stage compile loop succeeds. */
    pptr->usesFragCoordParams = GL_FALSE;
    pptr->uses_sample_params = GL_FALSE;
    pptr->vertexAttribUsageMask = 0u;
    pptr->uses_point_size_params = GL_FALSE;
    pptr->uses_shader_draw_params = GL_FALSE;
    pptr->uses_cull_distance = GL_FALSE;
    pptr->cull_distance_count = 0u;
    pptr->clip_distance_count = 0u;
    memset(pptr->air_builtin_mask, 0, sizeof(pptr->air_builtin_mask));
    mglProgramClearBuiltinInterface(pptr);
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
    /* Sampler/plain-uniform reset is deferred until a successful commit when
     * a prior executable is parked (A16+A18).  First-time links still clear. */
    if (!have_saved_executable) {
        for (GLuint i = 0; i < TEXTURE_UNITS; i++) {
            pptr->sampler_units[i] = -1;
            pptr->sampler_units_explicit[i] = GL_FALSE;
        }
        for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
            for (GLuint i = 0; i < TEXTURE_UNITS; i++) {
                pptr->sampler_units_by_stage[stage][i] = -1;
                pptr->sampler_units_explicit_by_stage[stage][i] = GL_FALSE;
            }
        }
        memset(pptr->plain_uniform_active_mask, 0, sizeof(pptr->plain_uniform_active_mask));
    }
    /* Invalidate the IR-level reflection cache for buffer slot
     * conflict detection.  Lazily recomputed on first
     * mglBufferSlotConflictsForProgram call during resource binding. */
    pptr->ir_cache_valid = GL_FALSE;
    /* Bump the per-Program cache generation so renderer pipeline keys cannot
     * reuse objects from the previous linked executable. */
    mglRenderProgramBumpPipelineCacheGeneration(pptr);

    {
        GLboolean saw_spirv = GL_FALSE;
        GLboolean saw_glsl = GL_FALSE;
        GLboolean spirv_nameless = GL_TRUE;
        GLboolean saw_any = GL_FALSE;

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
                    goto link_fail;
                }
                saw_any = GL_TRUE;
                if (shader->spir_v_binary) {
                    saw_spirv = GL_TRUE;
                    if (!shader->spir_v_nameless)
                        spirv_nameless = GL_FALSE;
                } else {
                    saw_glsl = GL_TRUE;
                }
            }
        }

        /* ARB_gl_spirv: all attached shaders must share SPIR_V_BINARY state. */
        if (saw_spirv && saw_glsl) {
            fprintf(stderr,
                    "MGL WARNING: mglLinkProgram failed program %u: mixed SPIR-V and GLSL shaders\n",
                    pptr->name);
            goto link_fail;
        }
        pptr->spirv_nameless = (saw_any && saw_spirv && spirv_nameless)
            ? GL_TRUE : GL_FALSE;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if ((pptr->attached_shader_mask & (1u << stage)) == 0u)
            continue;
        GLboolean have_main = GL_FALSE;
        GLboolean saw_tu = GL_FALSE;
        GLuint attached_count = mglProgramAttachedShaderCount(pptr, (GLuint)stage);
        for (GLuint attached = 0u; attached < attached_count; attached++) {
            Shader *shader = (pptr->attached_shader_counts[stage] > 0u)
                ? pptr->attached_shader_slots[stage][attached]
                : pptr->shader_slots[stage];
            const MGLTranslationUnit *tu = shader ? shader->frontend_tu : NULL;
            if (!tu)
                continue;
            saw_tu = GL_TRUE;
            if (mglCompileTUHasNamedFunction(tu, "main")) {
                have_main = GL_TRUE;
                break;
            }
        }
        if (saw_tu && !have_main) {
            fprintf(stderr,
                    "MGL WARNING: mglLinkProgram failed program %u: "
                    "shader stage %d has no main\n",
                    pptr->name, stage);
            goto link_fail;
        }
    }

    if ((pptr->attached_shader_mask & (1u << _COMPUTE_SHADER)) != 0u &&
        !mglComputeLocalSizeLinkOk(pptr)) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: "
                "compute local_size missing or conflicting\n",
                pptr->name);
        goto link_fail;
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if (!mglAirCompileStage(ctx, pptr, stage)) {
            link_ok = false;
            break;
        }
    }

    if (!link_ok) {
        goto link_fail;
    }

    if (!mglValidateCombinedClipAndCullDistances(ctx, pptr)) {
        goto link_fail;
    }

    /* GL 4.6 §11.4.3: a non-separable program that contains any of
     * TCS/TES/GS/FS must also contain a vertex shader to link. */
    {
        const GLuint graphics_without_vs =
            (1u << _TESS_CONTROL_SHADER) | (1u << _TESS_EVALUATION_SHADER) |
            (1u << _GEOMETRY_SHADER) | (1u << _FRAGMENT_SHADER);
        if (!pptr->program_separable &&
            (pptr->attached_shader_mask & graphics_without_vs) &&
            !(pptr->attached_shader_mask & (1u << _VERTEX_SHADER))) {
            fprintf(stderr,
                    "MGL WARNING: mglLinkProgram failed program %u: "
                    "no vertex shader attached for a non-separable program\n",
                    pptr->name);
            goto link_fail;
        }
    }

    if (!mglValidateGeometryInterface(pptr)) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: vertex "
                "output / geometry input interface mismatch\n",
                pptr->name);
        goto link_fail;
    }

    /* GL 4.6 §11.3.2: a program fails to link if max_vertices is not
     * specified in any geometry compilation unit. */
    if ((pptr->attached_shader_mask & GEOMETRY_SHADER_MASK_BIT) &&
        !pptr->geometry_max_vertices_specified) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: "
                "geometry shader missing layout(max_vertices)\n",
                pptr->name);
        goto link_fail;
    }

    /* GLSL 4.60 §4.4.1 / GL 4.6 §11.2.1.2: TES must declare an input
     * primitive mode; missing mode is a link failure (compile may still
     * succeed — CTS te_lacking_primitive_mode_declaration). */
    if ((pptr->attached_shader_mask & (1u << _TESS_EVALUATION_SHADER)) &&
        !pptr->tess_gen_mode_specified) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: "
                "tessellation evaluation shader missing input primitive mode\n",
                pptr->name);
        goto link_fail;
    }

    /* GL 4.6 §7.4/§11.1.3.9: a program fails to link if a geometry shader
     * references more atomic counters, or atomic counter buffer binding
     * points, than the per-stage limits advertise. */
    if ((pptr->attached_shader_mask & GEOMETRY_SHADER_MASK_BIT)) {
        const MGLShaderResourceList *acs =
            &pptr->shader_resources_list[_GEOMETRY_SHADER]
                                        [_ATOMIC_COUNTER_RES];
        GLuint total_counters = 0u;
        GLuint distinct_bindings = 0u;
        GLboolean binding_seen[MAX_BINDABLE_BUFFERS] = {GL_FALSE};
        for (GLuint i = 0u; i < acs->count; i++) {
            const MGLShaderResource *ac = &acs->list[i];
            GLuint elems = ac->gl_array_size > 0u ? ac->gl_array_size : 1u;
            total_counters += elems;
            GLuint b = ac->gl_binding;
            if (b < MAX_BINDABLE_BUFFERS && !binding_seen[b]) {
                binding_seen[b] = GL_TRUE;
                distinct_bindings++;
            }
        }
        if (total_counters >
            (GLuint)STATE(var).max_geometry_atomic_counters) {
            fprintf(stderr,
                    "MGL WARNING: mglLinkProgram failed program %u: "
                    "geometry shader uses %u atomic counters, limit %d\n",
                    pptr->name, total_counters,
                    STATE(var).max_geometry_atomic_counters);
            goto link_fail;
        }
        if (distinct_bindings >
            (GLuint)STATE(var).max_geometry_atomic_counter_buffers) {
            fprintf(stderr,
                    "MGL WARNING: mglLinkProgram failed program %u: "
                    "geometry shader uses %u atomic counter buffers, "
                    "limit %d\n",
                    pptr->name, distinct_bindings,
                    STATE(var).max_geometry_atomic_counter_buffers);
            goto link_fail;
        }
    }

    /* Validate transform feedback varyings: the link must fail if any
     * captured varying is not an active output of the program. */
    if (!mglValidateTransformFeedbackVaryings(ctx, pptr)) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: transform feedback "
                "varying not found in program outputs\n",
                pptr->name);
        goto link_fail;
    }

    /* Builtin usage comes from the per-stage masks the frontend published at
     * compile time (exact), not from scanning the shader sources. */
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        const GLuint bits = pptr->air_builtin_mask[stage];
        if (bits & MGL_AIR_BUILTIN_VERTEX_ID) {
            pptr->uses_vertex_id = GL_TRUE;
        }
        if (bits & MGL_AIR_BUILTIN_PRIMITIVE_ID) {
            pptr->uses_primitive_id = GL_TRUE;
        }
        if (stage == _VERTEX_SHADER &&
            (bits & (MGL_AIR_BUILTIN_DRAW_ID | MGL_AIR_BUILTIN_BASE_VERTEX))) {
            pptr->uses_shader_draw_params = GL_TRUE;
        }
    }

    applyFragmentOutputLocationIndices(pptr);
    applyMultiDimArrayUniformNames(pptr);
    mglProgramApplyBuiltinInterface(pptr);
    alignFragmentInputLocationsToVertexOutputs(pptr);
    mglBridgeSkippedGeometryShaderVaryings(pptr);
    mglAssignPlainUniformLocations(pptr);
    if (mglAssignAggregateMemberLocations(pptr) != 0) {
        fprintf(stderr,
                "MGL WARNING: mglLinkProgram failed program %u: "
                "overlapping or out-of-range explicit uniform locations\n",
                pptr->name);
        goto link_fail;
    }
    mglAssignSamplerUniformLocations(pptr);
    mglUnifySamplerUniformLocations(pptr);

    /* The AIR compute expansion consumes fixed 32-byte input records
     * (position + point size).  It accepts every core GS input topology and
     * expands point/line/triangle strips to Metal list primitives. */
    if (pptr->attached_shader_mask & GEOMETRY_SHADER_MASK_BIT) {
        bool computeRoute =
            mglRenderProgramModuleMetallibBytes(pptr, _GEOMETRY_SHADER) != NULL &&
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
            fprintf(stderr,
                    "MGL WARNING: mglLinkProgram failed program %u: "
                    "geometry shader is outside the AIR compute-expansion "
                    "subset (air=%d in=0x%x out=0x%x max=%u inv=%u)\n",
                    pptr->name,
                    mglRenderProgramModuleMetallibBytes(pptr, _GEOMETRY_SHADER) !=
                        NULL,
                    pptr->geometry_input_type,
                    pptr->geometry_output_type,
                    pptr->geometry_vertices_out,
                    pptr->geometry_invocations);
            goto link_fail;
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
        goto link_fail;
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
                    if (b + n > STATE(var).max_uniform_buffer_bindings) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d UBO '%s' binding %u (array size %u) "
                                "exceeds GL_MAX_UNIFORM_BUFFER_BINDINGS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b, n,
                                STATE(var).max_uniform_buffer_bindings);
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
                    if (b >= STATE(var).max_shader_storage_buffer_bindings) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d SSBO '%s' binding %u "
                                "exceeds GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b,
                                STATE(var).max_shader_storage_buffer_bindings);
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
                    if (b >= STATE(var).max_image_units) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d image '%s' binding %u "
                                "exceeds GL_MAX_IMAGE_UNITS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b,
                                STATE(var).max_image_units);
                        binding_error = true;
                        break;
                    }
                }
            }

            /* Active image uniforms per stage (GLSL §4.4.6.2 / ARB_shader_image_load_store). */
            if (!binding_error) {
                MGLShaderResourceList *rl =
                    &pptr->shader_resources_list[stage][_STORAGE_IMAGE_RES];
                GLuint image_count = 0u;
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint elems = rl->list[i].gl_array_size > 0
                        ? (GLuint)rl->list[i].gl_array_size : 1u;
                    image_count += elems;
                }
                GLuint stage_max = 0u;
                switch (stage) {
                case _VERTEX_SHADER:
                    stage_max = STATE(var).max_vertex_image_uniforms;
                    break;
                case _TESS_CONTROL_SHADER:
                    stage_max = STATE(var).max_tess_control_image_uniforms;
                    break;
                case _TESS_EVALUATION_SHADER:
                    stage_max = STATE(var).max_tess_evaluation_image_uniforms;
                    break;
                case _GEOMETRY_SHADER:
                    stage_max = STATE(var).max_geometry_image_uniforms;
                    break;
                case _FRAGMENT_SHADER:
                    stage_max = STATE(var).max_fragment_image_uniforms;
                    break;
                case _COMPUTE_SHADER:
                    stage_max = STATE(var).max_compute_image_uniforms;
                    break;
                default:
                    break;
                }
                if (stage_max > 0u && image_count > stage_max) {
                    fprintf(stderr,
                            "MGL LINK ERROR: program %u stage %d has %u active image "
                            "uniforms; exceeds stage limit (%u)\n",
                            pptr->name, stage, image_count, stage_max);
                    binding_error = true;
                }
            }

            /* Atomic counters: GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS */
            if (!binding_error) {
                MGLShaderResourceList *rl =
                    &pptr->shader_resources_list[stage][_ATOMIC_COUNTER_RES];
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint b = rl->list[i].gl_binding;
                    if (b >= STATE(var).max_atomic_counter_buffer_bindings) {
                        fprintf(stderr,
                                "MGL LINK ERROR: program %u stage %d atomic counter '%s' binding %u "
                                "exceeds GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS (%u)\n",
                                pptr->name, stage,
                                rl->list[i].name ? rl->list[i].name : "(null)",
                                b,
                                STATE(var).max_atomic_counter_buffer_bindings);
                        binding_error = true;
                        break;
                    }
                }
            }
        }

        if (!binding_error) {
            GLuint combined_images = 0u;
            for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
                if ((pptr->attached_shader_mask & (1u << stage)) == 0u) {
                    continue;
                }
                MGLShaderResourceList *rl =
                    &pptr->shader_resources_list[stage][_STORAGE_IMAGE_RES];
                for (GLuint i = 0; i < rl->count; i++) {
                    GLuint elems = rl->list[i].gl_array_size > 0
                        ? (GLuint)rl->list[i].gl_array_size : 1u;
                    combined_images += elems;
                }
            }
            if (combined_images > STATE(var).max_combined_image_uniforms) {
                fprintf(stderr,
                        "MGL LINK ERROR: program %u has %u combined active image "
                        "uniforms; exceeds GL_MAX_COMBINED_IMAGE_UNIFORMS (%u)\n",
                        pptr->name, combined_images,
                        STATE(var).max_combined_image_uniforms);
                binding_error = true;
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
                MGLBufferSlotFacts slot_facts;
                mglBufferSlotFactsFromProgram(pptr, stage, &slot_facts);
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
                        GLuint element_count =
                            mglStageBufferResourceElementCount(resource_type,
                                                               resource);
                        uint32_t bad_slot = 0u;
                        const int slot_status = mglBufferSlotCheckResource(
                            &slot_facts, resource->binding, element_count,
                            &bad_slot);
                        if (slot_status == MGL_BUFFER_SLOT_OK) {
                            continue;
                        }
                        const char *resource_name =
                            resource->name ? resource->name : "(null)";
                        if (slot_status == MGL_BUFFER_SLOT_EXCEEDS_LIMIT) {
                            fprintf(stderr,
                                    "MGL LINK ERROR: program %u %s %s '%s' "
                                    "Metal buffer slot %u exceeds user slot "
                                    "limit [0, %u)\n",
                                    pptr->name, mglShaderStageName(stage),
                                    mglMGLShaderResourceTypeName(resource_type),
                                    resource_name, bad_slot,
                                    (unsigned)kMGLMaxMetalUserBufferCount);
                        } else {
                            const char *reserved =
                                mglBufferSlotReservedName(bad_slot);
                            fprintf(stderr,
                                    "MGL LINK ERROR: program %u %s %s '%s' "
                                    "Metal buffer slot %u conflicts with %s\n",
                                    pptr->name, mglShaderStageName(stage),
                                    mglMGLShaderResourceTypeName(resource_type),
                                    resource_name, bad_slot,
                                    reserved ? reserved : "an internal buffer");
                        }
                        binding_error = true;
                    }
                }
            }
        }

        if (binding_error) {
            goto link_fail;
        }
    }

    /* GL 4.6 §7.7.2: distinct atomic counters sharing a binding must not
     * overlap in the counter buffer. */
    if (!mglValidateAtomicCounterOffsetOverlap(pptr)) {
        goto link_fail;
    }
    if (!mglValidateCrossStageLayoutBindings(pptr)) {
        goto link_fail;
    }

    if ((pptr->attached_shader_mask & VERTEX_SHADER_MASK_BIT) &&
        (pptr->attached_shader_mask & FRAGMENT_SHADER_MASK_BIT)) {
        Shader *vs = pptr->shader_slots[_VERTEX_SHADER];
        Shader *fs = pptr->shader_slots[_FRAGMENT_SHADER];
        if (vs && fs && vs->src && fs->src) {
            char iface_err[512] = {0};
            int adjacent = !(pptr->attached_shader_mask &
                             (GEOMETRY_SHADER_MASK_BIT |
                              (1u << _TESS_CONTROL_SHADER) |
                              (1u << _TESS_EVALUATION_SHADER)));
            int iface_rc = (vs->frontend_tu && fs->frontend_tu)
                ? mglShaderInterfaceCheckTU(vs->frontend_tu, fs->frontend_tu,
                                            adjacent, iface_err,
                                            sizeof iface_err)
                : mglShaderInterfaceCheck(vs->src, fs->src, adjacent,
                                          iface_err, sizeof iface_err);
            if (iface_rc != 0) {
                fprintf(stderr,
                        "MGL WARNING: mglLinkProgram failed program %u: %s\n",
                        pptr->name,
                        iface_err[0] ? iface_err : "stage interface mismatch");
                goto link_fail;
            }
        }
    }

    if ((pptr->attached_shader_mask & (1u << _TESS_CONTROL_SHADER)) &&
        (pptr->attached_shader_mask & (1u << _TESS_EVALUATION_SHADER))) {
        Shader *tcs = pptr->shader_slots[_TESS_CONTROL_SHADER];
        Shader *tes = pptr->shader_slots[_TESS_EVALUATION_SHADER];
        if (tcs && tes && tcs->src && tes->src) {
            char iface_err[512] = {0};
            int iface_rc = (tcs->frontend_tu && tes->frontend_tu)
                ? mglShaderTessInterfaceCheckTU(tcs->frontend_tu,
                                                tes->frontend_tu, iface_err,
                                                sizeof iface_err)
                : mglShaderTessInterfaceCheck(tcs->src, tes->src, iface_err,
                                              sizeof iface_err);
            if (iface_rc != 0) {
                fprintf(stderr,
                        "MGL WARNING: mglLinkProgram failed program %u: %s\n",
                        pptr->name,
                        iface_err[0] ? iface_err
                                     : "tessellation stage interface mismatch");
                goto link_fail;
            }
        }
    }

    if (have_saved_executable) {
        mglDiscardSavedLinkExecutable(&saved_executable);
        have_saved_executable = GL_FALSE;
        for (GLuint i = 0; i < TEXTURE_UNITS; i++) {
            pptr->sampler_units[i] = -1;
            pptr->sampler_units_explicit[i] = GL_FALSE;
        }
        for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
            for (GLuint i = 0; i < TEXTURE_UNITS; i++) {
                pptr->sampler_units_by_stage[stage][i] = -1;
                pptr->sampler_units_explicit_by_stage[stage][i] = GL_FALSE;
            }
        }
    }
    /* A18: successful link commit resets ProgramLinkState sampler + plain
     * uniform storage so prior uploads cannot pollute the new executable. */
    mglProgramClearPlainUniformStorage(ctx, pptr);

    pptr->link_success = GL_TRUE;
    pptr->link_state = MGL_PROGRAM_LINK_OK;
    pptr->linked_program_separable = pptr->program_separable;
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
        if (mglRenderProgramModuleMetallibBytes(pptr, _VERTEX_SHADER)) {
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
        pptr->uses_sample_params = GL_FALSE;
        {
            /* Fragment builtin usage from the published mask: gl_FragCoord
             * drives the frag-coord fixup params, and the sample-related
             * builtins (plus the `sample` qualifier / interpolateAtSample)
             * drive the sample params.  gl_NumSamples counts for the params
             * but not for per-sample MS values, matching the old scans. */
            const GLuint fsBits = pptr->air_builtin_mask[_FRAGMENT_SHADER];
            if (fsBits & (MGL_AIR_BUILTIN_FRAG_COORD | MGL_AIR_BUILTIN_POINT_COORD |
                          MGL_AIR_BUILTIN_DERIV_Y))
                pptr->usesFragCoordParams = GL_TRUE;
            if (fsBits & (MGL_AIR_BUILTIN_NUM_SAMPLES |
                          MGL_AIR_BUILTIN_SAMPLE_ID |
                          MGL_AIR_BUILTIN_SAMPLE_POSITION |
                          MGL_AIR_BUILTIN_SAMPLE_MASK |
                          MGL_AIR_BUILTIN_INTERPOLATE_AT_SAMPLE |
                          MGL_AIR_BUILTIN_SAMPLE_INTERPOLATION))
                pptr->uses_sample_params = GL_TRUE;
        }
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

    /* Apply GLSL uniform default initializers into plain-uniform slots so
     * the first draw matches the language defaults without baking them into
     * shader SSA (which would ignore later glUniform* updates). */
    mglSeedUniformInitializers(ctx, pptr);
    mglSeedLegacyMatrixIdentity(ctx, pptr);

    mglRendererBindProgram(ctx, pptr);

    //ERROR_CHECK_RETURN(pptr->mtl_data, GL_INVALID_OPERATION);
link_fail:
    if (have_saved_executable) {
        mglRestoreSavedLinkExecutable(pptr, &saved_executable);
        have_saved_executable = GL_FALSE;
    }
    return;

}
