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
 * draw_command.c
 * MGL
 *
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "glm_context.h"
#include "draw_command.h"

void mglInitCommandBuffer(MGLCommandBuffer *cb)
{
    if (!cb) return;
    memset(cb, 0, sizeof(*cb));
}

void mglResetCommandBuffer(MGLCommandBuffer *cb)
{
    if (!cb) return;

    for (uint32_t i = 0; i < cb->batch_count; i++) {
        MGLDrawBatch *batch = &cb->batches[i];
        if (batch->commands) {
            free(batch->commands);
            batch->commands = NULL;
        }
    }

    memset(cb, 0, sizeof(*cb));
}

static inline uint64_t mglRotateLeft64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

static uint64_t mglComputeTextureHash(GLMContext ctx)
{
    uint64_t hash = 0;
    unsigned *mask = ctx->state.active_texture_mask;
    for (int w = 0; w < 4; w++) {
        unsigned bits = mask[w];
        while (bits) {
            int i = __builtin_ctz(bits);
            bits &= bits - 1;
            int unit = w * 32 + i;
            if (unit < TEXTURE_UNITS) {
                Texture *tex = ctx->state.active_textures[unit];
                uint64_t tex_ptr = tex ? (uint64_t)(uintptr_t)tex : 0;
                hash ^= mglRotateLeft64(tex_ptr, unit & 63);
            }
        }
    }
    return hash;
}

static uint64_t mglComputeRenderStateHash(GLMContext ctx)
{
    uint64_t hash = 0;
    GLMCaps *caps = &ctx->state.caps;
    GLMParams *var = &ctx->state.var;

    if (caps->blend) {
        hash ^= mglRotateLeft64((uint64_t)var->blend_src, 0);
        hash ^= mglRotateLeft64((uint64_t)var->blend_dst, 16);
        hash ^= mglRotateLeft64((uint64_t)var->blend_src_alpha[0], 32);
        hash ^= mglRotateLeft64((uint64_t)var->blend_dst_alpha[0], 48);
    }
    if (caps->depth_test) {
        hash ^= mglRotateLeft64((uint64_t)var->depth_func, 8);
        hash ^= var->depth_writemask ? 0xAAAA5555ULL : 0ULL;
    }
    if (caps->stencil_test) {
        hash ^= mglRotateLeft64((uint64_t)var->stencil_func, 24);
        hash ^= mglRotateLeft64((uint64_t)var->stencil_back_func, 40);
        hash ^= mglRotateLeft64((uint64_t)var->stencil_writemask, 56);
    }
    hash ^= mglRotateLeft64((uint64_t)var->polygon_mode, 5);
    hash ^= caps->polygon_offset_fill  ? 0x11111111ULL : 0ULL;
    hash ^= caps->polygon_offset_line  ? 0x22222222ULL : 0ULL;
    hash ^= caps->polygon_offset_point ? 0x44444444ULL : 0ULL;

    return hash;
}

static uint8_t mglModeToPrimitiveType(GLenum mode)
{
    switch (mode) {
        case GL_POINTS:         return 0;  /* MTLPrimitiveTypePoint */
        case GL_LINES:          return 1;  /* MTLPrimitiveTypeLine */
        case GL_LINE_STRIP:     return 2;  /* MTLPrimitiveTypeLineStrip */
        case GL_TRIANGLES:      return 3;  /* MTLPrimitiveTypeTriangle */
        case GL_TRIANGLE_STRIP: return 4;  /* MTLPrimitiveTypeTriangleStrip */
        default:                return 0xFF;
    }
}

static uint16_t mglComputeCapsFlags(GLMContext ctx)
{
    uint16_t flags = 0;
    GLMCaps *caps = &ctx->state.caps;

    if (caps->cull_face)        flags |= (1u << 0);
    if (caps->depth_test)       flags |= (1u << 1);
    if (caps->stencil_test)     flags |= (1u << 2);
    if (caps->blend)            flags |= (1u << 3);
    if (caps->scissor_test)     flags |= (1u << 4);
    if (caps->polygon_offset_fill)  flags |= (1u << 5);
    if (caps->polygon_offset_line)  flags |= (1u << 6);
    if (caps->polygon_offset_point) flags |= (1u << 7);
    if (ctx->state.var.cull_face_mode == GL_FRONT_AND_BACK) flags |= (1u << 8);

    return flags;
}

void mglComputeStateKey(GLMContext ctx, GLenum mode, bool uses_elements, MGLStateKey *out)
{
    if (!ctx || !out) return;
    memset(out, 0, sizeof(*out));

    out->program_name = ctx->state.program_name;
    out->vao_name = ctx->state.vao ? ctx->state.vao->name : 0;
    out->fbo_name = ctx->state.framebuffer ? ctx->state.framebuffer->name : 0;

    for (int i = 0; i < 4; i++) {
        out->viewport[i] = (int16_t)ctx->state.viewport[i];
    }

    out->scissor_enabled = ctx->state.caps.scissor_test ? 1 : 0;
    if (out->scissor_enabled) {
        for (int i = 0; i < 4; i++) {
            out->scissor[i] = (int16_t)ctx->state.var.scissor_box[i];
        }
    }

    out->primitive_type = mglModeToPrimitiveType(mode);
    out->caps_flags = mglComputeCapsFlags(ctx);
    out->texture_hash = mglComputeTextureHash(ctx);
    out->render_state_hash = mglComputeRenderStateHash(ctx);

    (void)uses_elements;
}

bool mglStateKeysEqual(const MGLStateKey *a, const MGLStateKey *b)
{
    if (!a || !b) return false;
    return memcmp(a, b, sizeof(MGLStateKey)) == 0;
}

static bool mglBatchIsMDICompatible(const MGLDrawBatch *batch, const MGLDrawCommand *cmd)
{
    uint8_t prim_type = batch->key.primitive_type;
    if (prim_type == 0xFF) return false;

    MGLDrawCommandType cmd_type = cmd->type;
    bool cmd_uses_elements = (cmd_type != MGL_CMD_DRAW_ARRAYS &&
                              cmd_type != MGL_CMD_DRAW_ARRAYS_INSTANCED &&
                              cmd_type != MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE);

    if (cmd_uses_elements != batch->uses_elements) return false;

    /* Emulated modes can't use MDI */
    GLenum mode = cmd->mode;
    if (mode == GL_TRIANGLE_FAN || mode == GL_LINE_LOOP) return false;

    return true;
}

void mglAppendDrawCommand(GLMContext ctx, const MGLDrawCommand *cmd)
{
    if (!ctx || !cmd) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    MGLStateKey key;
    mglComputeStateKey(ctx, cmd->mode,
        cmd->type != MGL_CMD_DRAW_ARRAYS &&
        cmd->type != MGL_CMD_DRAW_ARRAYS_INSTANCED &&
        cmd->type != MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE,
        &key);

    /* Find matching batch (check last first for spatial locality) */
    MGLDrawBatch *batch = NULL;
    if (cb->batch_count > 0) {
        MGLDrawBatch *last = &cb->batches[cb->batch_count - 1];
        if (mglStateKeysEqual(&last->key, &key) &&
            last->command_count < MGL_MAX_DRAWS_PER_BATCH) {
            batch = last;
        }
    }
    if (!batch) {
        if (cb->batch_count >= MGL_MAX_BATCHES) {
            mglFlushCommandBuffer(ctx);
        }
        batch = &cb->batches[cb->batch_count];
        batch->key = key;
        batch->command_count = 0;
        batch->commands = NULL;
        batch->mdi_compatible = true;
        batch->uses_elements =
            (cmd->type != MGL_CMD_DRAW_ARRAYS &&
             cmd->type != MGL_CMD_DRAW_ARRAYS_INSTANCED &&
             cmd->type != MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE);
        cb->batch_count++;
    }

    /* Resize command array */
    MGLDrawCommand *new_cmds = (MGLDrawCommand *)realloc(batch->commands,
        (batch->command_count + 1) * sizeof(MGLDrawCommand));
    if (!new_cmds) {
        fprintf(stderr, "MGL Error: mglAppendDrawCommand: realloc failed\n");
        return;
    }
    batch->commands = new_cmds;
    batch->commands[batch->command_count] = *cmd;
    batch->command_count++;
    cb->total_commands++;

    if (!mglBatchIsMDICompatible(batch, cmd)) {
        batch->mdi_compatible = false;
    }

    if (batch->uses_elements) {
        cb->element_cmd_count++;
    } else {
        cb->array_cmd_count++;
    }

    /*
     * The current batch key is only a compact comparison key. It does not
     * snapshot texture bindings, uniform buffers, vertex-buffer bindings, or
     * per-resource sampler state. Replaying after later GL state changes can
     * therefore draw old commands with new resources. Keep correctness first
     * and flush immediately until command buffering stores full state.
     */
    mglFlushCommandBuffer(ctx);
}

void mglFlushCommandBuffer(GLMContext ctx)
{
    if (!ctx) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0) return;

    if (ctx->mtl_funcs.mtlFlushDrawBuffer) {
        ctx->mtl_funcs.mtlFlushDrawBuffer(ctx);
    }
}
