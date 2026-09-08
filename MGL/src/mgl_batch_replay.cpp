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
 * O2.3: BatchReplay stage/bind expansion — domain ownership (not ObjC).
 */

#include "mgl_batch_replay.h"

#include "mgl_render.h"
#include "mgl_types_texture.h"
#include "mgl_vertex_attrib_query.h"
#include "mgl_vertex_format.h"

#include <string.h>

extern "C" bool mgl_batch_replay_collect_resource_binding(
    MGLRenderResourceBindingSnapshot *snapshot, uint32_t stage, uint32_t kind,
    void *resource, uint32_t index)
{
    if (!snapshot || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        kind > MGL_RENDER_RESOURCE_BINDING_SAMPLER) {
        return false;
    }
    uint32_t *count = stage == MGL_RENDER_BINDING_STAGE_VERTEX
                          ? &snapshot->vertex_op_count
                          : &snapshot->fragment_op_count;
    MGLRenderResourceBindingOp *ops = stage == MGL_RENDER_BINDING_STAGE_VERTEX
                                          ? snapshot->vertex_ops
                                          : snapshot->fragment_ops;
    if (*count >= MGL_RENDER_RESOURCE_BINDING_SNAPSHOT_MAX_OPS) {
        return false;
    }
    ops[(*count)++] = MGLRenderResourceBindingOp{
        .kind = kind,
        .index = index,
        .resource = resource,
    };
    return true;
}

extern "C" bool mgl_batch_replay_build_dynamic_vertex_array(
    GLMContext ctx, const VertexArray *base, const MGLDrawCommand *cmd,
    VertexArray *out)
{
    if (!ctx || !base || !cmd || !out ||
        cmd->dynamic_vertex_binding_count > MGL_MAX_DYNAMIC_VERTEX_BINDINGS) {
        return false;
    }

    *out = *base;
    for (uint8_t i = 0; i < cmd->dynamic_vertex_binding_count; i++) {
        const MGLDynamicVertexBinding *override_binding =
            &cmd->dynamic_vertex_bindings[i];
        if (!override_binding->buffer_name ||
            override_binding->binding_index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
            return false;
        }
        Buffer *resolved = mglNamedBuffer(ctx, override_binding->buffer_name);
        if (!resolved) {
            return false;
        }
        BufferBinding *binding = &out->bindings[override_binding->binding_index];
        binding->buffer = resolved;
        binding->offset = (GLintptr)override_binding->offset;
        /* Keep classic VertexAttribPointer mirror fields in sync so resolve
         * stays correct if a path still reads attrib.binding_offset. */
        for (GLuint attrib = 0; attrib < MAX_ATTRIBS; attrib++) {
            if ((out->enabled_attribs & (1u << attrib)) == 0u ||
                out->attrib[attrib].buffer_bindingindex !=
                    override_binding->binding_index) {
                continue;
            }
            out->attrib[attrib].buffer = resolved;
            out->attrib[attrib].binding_offset =
                (GLintptr)override_binding->offset;
        }
    }
    return true;
}

extern "C" bool mgl_batch_replay_attrib_can_bind_directly(
    Program *active_program, GLuint attrib_index, const VertexAttrib *attrib)
{
    if (!attrib ||
        mglRenderAttribNeedsConversion(attrib->long_attribute ? 1 : 0,
                                       (uint32_t)attrib->type,
                                       attrib->integer ? 1 : 0)) {
        return false;
    }
    if (!attrib->integer) {
        return true;
    }

    MGLShaderResource *resource =
        mglRendererProgramVertexAttribResource(active_program, attrib_index);
    GLuint shader_type = resource ? resource->gl_type : 0u;
    return !mglIntegerAttribNeedsConversion(attrib->type, shader_type,
                                            attrib->size, NULL);
}

extern "C" bool mgl_batch_replay_apply_uniform_range_overrides(
    GLMContext ctx, const MGLDrawCommand *cmd)
{
    if (!ctx || !cmd || !ctx->active_state) {
        return false;
    }
    for (uint8_t i = 0; i < cmd->dynamic_uniform_binding_count; i++) {
        const MGLDynamicUniformBinding *override_binding =
            &cmd->dynamic_uniform_bindings[i];
        if (override_binding->binding_index >= MAX_BINDABLE_BUFFERS) {
            return false;
        }
        BufferBaseTarget *slot =
            &ctx->active_state->buffer_base[_UNIFORM_BUFFER]
                 .buffers[override_binding->binding_index];
        if (!slot->buf) {
            return false;
        }
        slot->offset = override_binding->offset;
        slot->size = override_binding->size;
    }
    return true;
}

extern "C" bool mgl_batch_replay_apply_texture_overrides(
    GLMContext ctx, const MGLDrawCommand *cmd, bool *touched_units,
    uint32_t touched_units_count)
{
    if (!ctx || !cmd || !ctx->active_state || !touched_units ||
        touched_units_count < (uint32_t)TEXTURE_UNITS) {
        return false;
    }

    for (uint8_t i = 0; i < cmd->dynamic_texture_binding_count; i++) {
        const MGLDynamicTextureBinding *override_binding =
            &cmd->dynamic_texture_bindings[i];
        if (override_binding->unit >= TEXTURE_UNITS ||
            override_binding->target_index >= _MAX_TEXTURE_TYPES ||
            !override_binding->texture_name) {
            return false;
        }
        Texture *texture = mglNamedTexture(ctx, override_binding->texture_name);
        if (!texture) {
            return false;
        }
        if (texture->index != override_binding->target_index) {
            return false;
        }
        if (!touched_units[override_binding->unit]) {
            ctx->active_state->active_textures[override_binding->unit] = NULL;
            touched_units[override_binding->unit] = true;
        }
        ctx->active_state->texture_units[override_binding->unit]
            .textures[override_binding->target_index] = texture;
        if (override_binding->is_active) {
            ctx->active_state->active_textures[override_binding->unit] =
                texture;
        }
    }
    return true;
}
