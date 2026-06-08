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
#include "mgl_safety.h"

void mglInitCommandBuffer(MGLCommandBuffer *cb)
{
    if (!cb) return;
    memset(cb, 0, sizeof(*cb));
}

static void mglDestroyTransientBuffer(GLMContext ctx, Buffer *buffer)
{
    if (!buffer) return;

    if (buffer->data.mtl_data && ctx && ctx->mtl_funcs.mtlDeleteMTLObj) {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, buffer->data.mtl_data);
        buffer->data.mtl_data = NULL;
    }
    if (buffer->data.buffer_data) {
        free((void *)(uintptr_t)buffer->data.buffer_data);
        buffer->data.buffer_data = 0;
        buffer->data.buffer_size = 0;
    }

    free(buffer);
}

static void mglReleaseBatch(GLMContext ctx, MGLDrawBatch *batch)
{
    if (!batch) return;

    if (batch->commands) {
        free(batch->commands);
        batch->commands = NULL;
    }
    if (batch->state_snapshot) {
        free(batch->state_snapshot);
        batch->state_snapshot = NULL;
    }
    if (batch->vao_snapshot) {
        free(batch->vao_snapshot);
        batch->vao_snapshot = NULL;
    }
    batch->source_vao = NULL;
    if (batch->retained_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_program);
        batch->retained_program = NULL;
    }
    if (batch->retained_vertex_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_vertex_program);
        batch->retained_vertex_program = NULL;
    }
    if (batch->retained_fragment_program) {
        mglReleaseProgramReference(ctx, (Program *)batch->retained_fragment_program);
        batch->retained_fragment_program = NULL;
    }
    if (batch->stream_vertex_buffer) {
        mglDestroyTransientBuffer(ctx, (Buffer *)batch->stream_vertex_buffer);
        batch->stream_vertex_buffer = NULL;
    }
    if (batch->stream_index_buffer) {
        mglDestroyTransientBuffer(ctx, (Buffer *)batch->stream_index_buffer);
        batch->stream_index_buffer = NULL;
    }
}

static Program *mglRetainBatchProgram(GLMContext ctx, MGLDrawBatch *batch, Program *program, GLuint expectedName, void **slot)
{
    if (!ctx || !batch || !program || !slot) {
        return NULL;
    }

    if (!mglObjectPointerLooksPlausible(program) ||
        !mglPointerRangeIsReadable(program, sizeof(*program))) {
        return NULL;
    }

    if (expectedName == 0u) {
        expectedName = program->name;
    }
    if (!mglProgramPointerUsableForName(ctx, program, expectedName)) {
        return NULL;
    }

    mglRetainProgramReference(ctx, program);
    *slot = program;
    return program;
}

static void mglRetainBatchProgramReferences(GLMContext ctx, MGLDrawBatch *batch)
{
    if (!ctx || !batch) {
        return;
    }

    (void)mglRetainBatchProgram(ctx,
                                batch,
                                ctx->state.program,
                                ctx->state.program_name,
                                &batch->retained_program);

    if (ctx->state.program_name != 0u || !ctx->state.program_pipeline) {
        return;
    }

    ProgramPipeline *pipeline = ctx->state.program_pipeline;
    if (!mglObjectPointerLooksPlausible(pipeline) ||
        !mglPointerRangeIsReadable(pipeline, sizeof(*pipeline))) {
        return;
    }

    (void)mglRetainBatchProgram(ctx,
                                batch,
                                pipeline->stage_programs[_VERTEX_SHADER],
                                pipeline->stage_programs[_VERTEX_SHADER]
                                    ? pipeline->stage_programs[_VERTEX_SHADER]->name
                                    : 0u,
                                &batch->retained_vertex_program);
    (void)mglRetainBatchProgram(ctx,
                                batch,
                                pipeline->stage_programs[_FRAGMENT_SHADER],
                                pipeline->stage_programs[_FRAGMENT_SHADER]
                                    ? pipeline->stage_programs[_FRAGMENT_SHADER]->name
                                    : 0u,
                                &batch->retained_fragment_program);
}

static bool mglInitializeBatchStateSnapshot(GLMContext ctx, MGLDrawBatch *batch)
{
    if (!ctx || !batch) return false;

    batch->state_snapshot = malloc(sizeof(ctx->state));
    batch->vao_snapshot = malloc(sizeof(VertexArray));
    if (!batch->state_snapshot || !batch->vao_snapshot) {
        return false;
    }
    memcpy(batch->state_snapshot, &ctx->state, sizeof(ctx->state));
    batch->source_vao = ctx->state.vao;
    if (ctx->state.vao) {
        memcpy(batch->vao_snapshot, ctx->state.vao, sizeof(VertexArray));
        ((VertexArray *)batch->vao_snapshot)->transient_batch_vao = GL_TRUE;
        ((GLMState *)batch->state_snapshot)->vao = (VertexArray *)batch->vao_snapshot;
    } else {
        free(batch->vao_snapshot);
        batch->vao_snapshot = NULL;
    }

    mglRetainBatchProgramReferences(ctx, batch);
    return true;
}

void mglResetCommandBufferForContext(GLMContext ctx, MGLCommandBuffer *cb)
{
    if (!cb) return;

    for (uint32_t i = 0; i < cb->batch_count; i++) {
        mglReleaseBatch(ctx, &cb->batches[i]);
    }

    memset(cb, 0, sizeof(*cb));
}

void mglResetCommandBuffer(MGLCommandBuffer *cb)
{
    mglResetCommandBufferForContext(NULL, cb);
}

static inline uint64_t mglRotateLeft64(uint64_t x, int n)
{
    n &= 63;
    if (n == 0) return x;
    return (x << n) | (x >> (64 - n));
}

static uint64_t mglHashBytes64(const void *data, size_t size, uint64_t seed)
{
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t hash = seed ^ 0xcbf29ce484222325ULL;

    if (!bytes) return hash;
    for (size_t i = 0; i < size; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

static inline bool mglRangesOverlap(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1)
{
    return a0 < b1 && b0 < a1;
}

static void mglNormalizeMutationRange(int64_t offset, int64_t size, uint64_t *start, uint64_t *end)
{
    if (!start || !end) return;

    if (offset < 0 || size < 0) {
        *start = 0;
        *end = UINT64_MAX;
        return;
    }

    *start = (uint64_t)offset;
    if (size == 0) {
        *end = *start;
        return;
    }

    if ((uint64_t)size > UINT64_MAX - *start) {
        *end = UINT64_MAX;
    } else {
        *end = *start + (uint64_t)size;
    }
}

static void mglTrackPendingReadRange(GLMContext ctx, Buffer *buffer, uint64_t start, uint64_t end)
{
    if (!ctx || !buffer || end <= start) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    for (uint32_t i = 0; i < cb->buffer_read_range_count; i++) {
        MGLBufferReadRange *range = &cb->buffer_read_ranges[i];
        if (range->buffer == buffer && mglRangesOverlap(range->start, range->end, start, end)) {
            if (start < range->start) range->start = start;
            if (end > range->end) range->end = end;
            return;
        }
    }

    if (cb->buffer_read_range_count >= MGL_MAX_PENDING_BUFFER_RANGES) {
        cb->buffer_read_range_overflow = true;
        return;
    }

    MGLBufferReadRange *range = &cb->buffer_read_ranges[cb->buffer_read_range_count++];
    range->buffer = buffer;
    range->start = start;
    range->end = end;
}

static void mglTrackPendingReadBytes(GLMContext ctx, Buffer *buffer, uint64_t start, uint64_t size)
{
    uint64_t end = (size > UINT64_MAX - start) ? UINT64_MAX : start + size;
    mglTrackPendingReadRange(ctx, buffer, start, end);
}

static void mglTrackPendingReadWholeBuffer(GLMContext ctx, Buffer *buffer)
{
    if (!buffer) return;
    uint64_t end = buffer->size > 0 ? (uint64_t)buffer->size : UINT64_MAX;
    mglTrackPendingReadRange(ctx, buffer, 0, end);
}

static Texture *mglPendingDrawAttachmentTexture(FBOAttachment *attachment)
{
    if (!attachment) return NULL;

    if (attachment->textarget == GL_RENDERBUFFER) {
        return (attachment->buf.rbo && attachment->buf.rbo->tex) ? attachment->buf.rbo->tex : NULL;
    }

    return attachment->buf.tex;
}

static bool mglDrawBufferToColorAttachmentIndex(GLMContext ctx, GLenum buffer, GLuint *outIndex)
{
    if (!ctx || !outIndex) return false;
    if (buffer < GL_COLOR_ATTACHMENT0) return false;

    GLuint index = (GLuint)(buffer - GL_COLOR_ATTACHMENT0);
    if (index >= ctx->state.max_color_attachments || index >= MAX_COLOR_ATTACHMENTS) {
        return false;
    }

    *outIndex = index;
    return true;
}

static void mglTrackPendingTextureWrite(GLMContext ctx, Texture *texture);

static void mglTrackPendingFramebufferDepthStencilWrites(GLMContext ctx, Framebuffer *fbo)
{
    if (!ctx || !fbo) return;

    mglTrackPendingTextureWrite(ctx, mglPendingDrawAttachmentTexture(&fbo->depth));
    if (fbo->stencil.buf.tex != fbo->depth.buf.tex ||
        fbo->stencil.buf.rbo != fbo->depth.buf.rbo ||
        fbo->stencil.textarget != fbo->depth.textarget) {
        mglTrackPendingTextureWrite(ctx, mglPendingDrawAttachmentTexture(&fbo->stencil));
    }
}

static void mglTrackPendingTextureWrite(GLMContext ctx, Texture *texture)
{
    if (!ctx || !texture) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    for (uint32_t i = 0; i < cb->texture_write_count; i++) {
        if (cb->texture_write_objects[i] == texture) {
            return;
        }
    }

    if (cb->texture_write_count >= MGL_MAX_PENDING_TEXTURE_WRITES) {
        cb->texture_write_overflow = true;
        return;
    }

    cb->texture_write_objects[cb->texture_write_count++] = texture;
}

static void mglTrackPendingTextureRead(GLMContext ctx, Texture *texture)
{
    if (!ctx || !texture) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    for (uint32_t i = 0; i < cb->texture_read_count; i++) {
        if (cb->texture_read_objects[i] == texture) {
            return;
        }
    }

    if (cb->texture_read_count >= MGL_MAX_PENDING_TEXTURE_READS) {
        cb->texture_read_overflow = true;
        return;
    }

    cb->texture_read_objects[cb->texture_read_count++] = texture;
}

static void mglTrackPendingSampledTextureReads(GLMContext ctx)
{
    if (!ctx) return;

    unsigned *mask = ctx->state.active_texture_mask;
    for (int w = 0; w < 4; w++) {
        unsigned bits = mask[w];
        while (bits) {
            int bit = __builtin_ctz(bits);
            bits &= bits - 1;
            int unit = (w * 32) + bit;
            if (unit >= TEXTURE_UNITS) {
                continue;
            }

            Texture *active = ctx->state.active_textures[unit];
            if (active) {
                mglTrackPendingTextureRead(ctx, active);
            }

            TextureUnit *textureUnit = &ctx->state.texture_units[unit];
            for (int target = 0; target < _MAX_TEXTURE_TYPES; target++) {
                Texture *bound = textureUnit->textures[target];
                if (bound) {
                    mglTrackPendingTextureRead(ctx, bound);
                }
            }
        }
    }
}

static void mglTrackPendingFramebufferTextureWrites(GLMContext ctx)
{
    if (!ctx) return;

    Framebuffer *fbo = ctx->state.framebuffer;
    if (!fbo) return;

    GLsizei drawBufferCount = ctx->state.draw_buffer_count;
    if (drawBufferCount <= 0 || drawBufferCount > (GLsizei)MAX_COLOR_ATTACHMENTS) {
        drawBufferCount = 1;
    }

    for (GLsizei slot = 0; slot < drawBufferCount; slot++) {
        GLenum drawBuffer = ctx->state.draw_buffers[slot];
        if (drawBuffer == GL_NONE) {
            continue;
        }

        GLuint attachmentIndex = 0u;
        if (!mglDrawBufferToColorAttachmentIndex(ctx, drawBuffer, &attachmentIndex)) {
            continue;
        }
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }

        mglTrackPendingTextureWrite(ctx,
                                    mglPendingDrawAttachmentTexture(&fbo->color_attachments[attachmentIndex]));
    }

    mglTrackPendingFramebufferDepthStencilWrites(ctx, fbo);
}

static void mglFlushPendingDrawsBeforeFramebufferTextureWrites(GLMContext ctx)
{
    if (!ctx || !ctx->draw_defer_enabled) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return;
    if (cb->texture_read_overflow) {
        mglFlushCommandBuffer(ctx);
        return;
    }

    Framebuffer *fbo = ctx->state.framebuffer;
    if (!fbo) return;

    GLsizei drawBufferCount = ctx->state.draw_buffer_count;
    if (drawBufferCount <= 0 || drawBufferCount > (GLsizei)MAX_COLOR_ATTACHMENTS) {
        drawBufferCount = 1;
    }

    for (GLsizei slot = 0; slot < drawBufferCount; slot++) {
        GLenum drawBuffer = ctx->state.draw_buffers[slot];
        if (drawBuffer == GL_NONE) {
            continue;
        }

        GLuint attachmentIndex = 0u;
        if (!mglDrawBufferToColorAttachmentIndex(ctx, drawBuffer, &attachmentIndex)) {
            continue;
        }
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            continue;
        }

        Texture *texture =
            mglPendingDrawAttachmentTexture(&fbo->color_attachments[attachmentIndex]);
        if (texture && mglPendingDrawsReadTexture(ctx, texture)) {
            static uint64_t s_framebufferWriteHazardFlushCount = 0;
            uint64_t hit = ++s_framebufferWriteHazardFlushCount;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                fprintf(stderr,
                        "MGL TRACE pending framebuffer write hazard flush hit=%llu tex=%u attachment=%u batches=%u commands=%u\n",
                        (unsigned long long)hit,
                        texture ? texture->name : 0u,
                        (unsigned)attachmentIndex,
                        ctx->draw_command_buffer.batch_count,
                        ctx->draw_command_buffer.total_commands);
            }
            mglFlushCommandBuffer(ctx);
            return;
        }
    }

    const struct {
        const char *name;
        FBOAttachment *attachment;
    } depthStencilAttachments[] = {
        { "depth", &fbo->depth },
        { "stencil", &fbo->stencil }
    };
    for (size_t i = 0; i < sizeof(depthStencilAttachments) / sizeof(depthStencilAttachments[0]); i++) {
        Texture *texture = mglPendingDrawAttachmentTexture(depthStencilAttachments[i].attachment);
        if (texture && mglPendingDrawsReadTexture(ctx, texture)) {
            static uint64_t s_framebufferDepthStencilWriteHazardFlushCount = 0;
            uint64_t hit = ++s_framebufferDepthStencilWriteHazardFlushCount;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                fprintf(stderr,
                        "MGL TRACE pending framebuffer %s write hazard flush hit=%llu tex=%u batches=%u commands=%u\n",
                        depthStencilAttachments[i].name,
                        (unsigned long long)hit,
                        texture ? texture->name : 0u,
                        ctx->draw_command_buffer.batch_count,
                        ctx->draw_command_buffer.total_commands);
            }
            mglFlushCommandBuffer(ctx);
            return;
        }
    }
}

bool mglPendingDrawsReadBufferRange(GLMContext ctx, void *buffer, int64_t offset, int64_t size)
{
    if (!ctx || !buffer || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;
    if (cb->buffer_read_range_overflow) return true;

    uint64_t start = 0;
    uint64_t end = 0;
    mglNormalizeMutationRange(offset, size, &start, &end);

    for (uint32_t i = 0; i < cb->buffer_read_range_count; i++) {
        MGLBufferReadRange *range = &cb->buffer_read_ranges[i];
        if (range->buffer == buffer && mglRangesOverlap(range->start, range->end, start, end)) {
            return true;
        }
    }

    return false;
}

bool mglPendingDrawsWriteTexture(GLMContext ctx, void *texture)
{
    if (!ctx || !texture || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;
    if (cb->texture_write_overflow) return true;

    for (uint32_t i = 0; i < cb->texture_write_count; i++) {
        if (cb->texture_write_objects[i] == texture) {
            return true;
        }
    }

    return false;
}

bool mglPendingDrawsReadTexture(GLMContext ctx, void *texture)
{
    if (!ctx || !texture || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;
    if (cb->texture_read_overflow) return true;

    for (uint32_t i = 0; i < cb->texture_read_count; i++) {
        if (cb->texture_read_objects[i] == texture) {
            return true;
        }
    }

    return false;
}

void mglFlushPendingDrawsForBuffer(GLMContext ctx, void *buffer)
{
    if (mglPendingDrawsReadBufferRange(ctx, buffer, 0, -1)) {
        mglFlushCommandBuffer(ctx);
    }
}

void mglFlushPendingDrawsForBufferRange(GLMContext ctx, void *buffer, int64_t offset, int64_t size)
{
    if (mglPendingDrawsReadBufferRange(ctx, buffer, offset, size)) {
        mglFlushCommandBuffer(ctx);
    }
}

static bool mglPendingDrawsReferenceVertexArray(GLMContext ctx, VertexArray *vao)
{
    if (!ctx || !vao || !ctx->draw_defer_enabled) return false;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return false;

    for (uint32_t i = 0; i < cb->batch_count; i++) {
        MGLDrawBatch *batch = &cb->batches[i];
        if (batch->command_count == 0) continue;

        /*
         * Stream-merged batches own a private VAO snapshot and copied transient
         * vertex/index data, so later mutations of the application's VAO cannot
         * change the already recorded draw.
         */
        if (batch->stream_merged) continue;

        if (batch->state_snapshot) {
            GLMState *snapshot = (GLMState *)batch->state_snapshot;
            if (batch->source_vao == vao || snapshot->vao == vao) {
                return true;
            }
            continue;
        }

        if (batch->key.vao_name == vao->name) {
            return true;
        }
    }

    return false;
}

void mglFlushPendingDrawsForVertexArray(GLMContext ctx, void *vao)
{
    if (mglPendingDrawsReferenceVertexArray(ctx, (VertexArray *)vao)) {
        mglFlushCommandBuffer(ctx);
    }
}

void mglFlushPendingDrawsForTexture(GLMContext ctx, void *texture)
{
    if (mglPendingDrawsWriteTexture(ctx, texture) ||
        mglPendingDrawsReadTexture(ctx, texture)) {
        mglFlushCommandBuffer(ctx);
    }
}

void mglFlushPendingDrawsBeforeTextureWrite(GLMContext ctx, void *texture)
{
    if (mglPendingDrawsReadTexture(ctx, texture) ||
        mglPendingDrawsWriteTexture(ctx, texture)) {
        static uint64_t s_textureWriteHazardFlushCount = 0;
        uint64_t hit = ++s_textureWriteHazardFlushCount;
        if (hit <= 64ull || (hit % 512ull) == 0ull) {
            Texture *tex = (Texture *)texture;
            fprintf(stderr,
                    "MGL TRACE pending texture write hazard flush hit=%llu tex=%u batches=%u commands=%u\n",
                    (unsigned long long)hit,
                    tex ? tex->name : 0u,
                    ctx ? ctx->draw_command_buffer.batch_count : 0u,
                    ctx ? ctx->draw_command_buffer.total_commands : 0u);
        }
        mglFlushCommandBuffer(ctx);
    }
}

void mglFlushPendingDrawsForActiveTextures(GLMContext ctx)
{
    if (!ctx || !ctx->draw_defer_enabled) return;

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    if (cb->batch_count == 0 || cb->total_commands == 0) return;
    if (cb->texture_write_overflow) {
        mglFlushCommandBuffer(ctx);
        return;
    }

    unsigned *mask = ctx->state.active_texture_mask;
    for (int w = 0; w < 4; w++) {
        unsigned bits = mask[w];
        while (bits) {
            int bit = __builtin_ctz(bits);
            bits &= bits - 1;
            int unit = (w * 32) + bit;
            if (unit >= TEXTURE_UNITS) {
                continue;
            }

            Texture *active = ctx->state.active_textures[unit];
            if (active && mglPendingDrawsWriteTexture(ctx, active)) {
                mglFlushCommandBuffer(ctx);
                return;
            }

            TextureUnit *textureUnit = &ctx->state.texture_units[unit];
            for (int target = 0; target < _MAX_TEXTURE_TYPES; target++) {
                Texture *bound = textureUnit->textures[target];
                if (bound && mglPendingDrawsWriteTexture(ctx, bound)) {
                    mglFlushCommandBuffer(ctx);
                    return;
                }
            }
        }
    }
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

                TextureUnit *tex_unit = &ctx->state.texture_units[unit];
                for (int t = 0; t < _MAX_TEXTURE_TYPES; t++) {
                    Texture *typed_tex = tex_unit->textures[t];
                    uint64_t typed_ptr = typed_tex ? (uint64_t)(uintptr_t)typed_tex : 0;
                    hash ^= mglRotateLeft64(typed_ptr + (uint64_t)(t + 1), (unit + t) & 63);
                }

                Sampler *sampler = ctx->state.texture_samplers[unit];
                uint64_t sampler_ptr = sampler ? (uint64_t)(uintptr_t)sampler : 0;
                hash ^= mglRotateLeft64(sampler_ptr, (unit + 17) & 63);
            }
        }
    }
    return hash;
}

static void mglHashBufferBaseBinding(uint64_t *hash,
                                     const BufferBaseTarget *binding,
                                     uint64_t salt)
{
    if (!hash || !binding) return;

    uint64_t binding_ptr = binding->buf ? (uint64_t)(uintptr_t)binding->buf : 0;
    *hash ^= mglRotateLeft64(binding_ptr ^ ((uint64_t)binding->buffer << 1), salt & 63);
    *hash ^= mglRotateLeft64((uint64_t)binding->offset, (salt + 17u) & 63);
    *hash ^= mglRotateLeft64((uint64_t)binding->size, (salt + 37u) & 63);
}

static uint64_t mglComputeDrawBufferBindingHash(GLMContext ctx)
{
    uint64_t hash = 0;

    const int draw_buffer_targets[] = {
        _UNIFORM_BUFFER,
        _UNIFORM_CONSTANT,
        _SHADER_STORAGE_BUFFER,
        _ATOMIC_COUNTER_BUFFER,
        _TEXTURE_BUFFER
    };

    for (size_t t = 0; t < sizeof(draw_buffer_targets) / sizeof(draw_buffer_targets[0]); t++) {
        int target = draw_buffer_targets[t];
        for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            mglHashBufferBaseBinding(&hash,
                                     &ctx->state.buffer_base[target].buffers[i],
                                     ((uint64_t)(target + 1) * 131u) + (uint64_t)i);
        }
    }

    Program *program = ctx->state.program;
    if (program) {
        for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            mglHashBufferBaseBinding(&hash,
                                     &program->plain_uniform_buffers[i],
                                     0x700u + (uint64_t)i);
        }
    }

    return hash;
}

static void mglHashBufferPointerName(uint64_t *hash, const Buffer *buffer, uint64_t salt)
{
    if (!hash) return;

    uint64_t ptr = buffer ? (uint64_t)(uintptr_t)buffer : 0u;
    uint64_t name = buffer ? (uint64_t)buffer->name : 0u;
    *hash ^= mglRotateLeft64(ptr ^ (name << 1), salt & 63);
}

static uint64_t mglComputeVertexArrayStateHash(GLMContext ctx, bool uses_elements)
{
    VertexArray *vao = ctx ? ctx->state.vao : NULL;
    if (!vao) return 0u;

    uint64_t hash = 0x56414f5354415445ULL;
    hash ^= mglRotateLeft64((uint64_t)vao->name, 3);
    hash ^= mglRotateLeft64((uint64_t)vao->enabled_attribs, 7);

    GLuint maxAttribs = MAX_ATTRIBS;
    for (GLuint i = 0; i < maxAttribs; i++) {
        const VertexAttrib *attrib = &vao->attrib[i];
        uint64_t salt = 0x100u + (uint64_t)i * 17u;
        mglHashBufferPointerName(&hash, attrib->buffer, salt);
        hash ^= mglRotateLeft64((uint64_t)attrib->size, (salt + 1u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->type, (salt + 2u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->normalized, (salt + 3u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->integer, (salt + 4u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->long_attribute, (salt + 5u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->stride, (salt + 6u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->divisor, (salt + 7u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->relativeoffset, (salt + 8u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->binding_offset, (salt + 9u) & 63);
        hash ^= mglRotateLeft64((uint64_t)attrib->buffer_bindingindex, (salt + 10u) & 63);
    }

    for (GLuint i = 0; i < MGL_MAX_VERTEX_ATTRIB_BINDINGS; i++) {
        const BufferBinding *binding = &vao->bindings[i];
        uint64_t salt = 0x500u + (uint64_t)i * 19u;
        mglHashBufferPointerName(&hash, binding->buffer, salt);
        hash ^= mglRotateLeft64((uint64_t)binding->offset, (salt + 1u) & 63);
        hash ^= mglRotateLeft64((uint64_t)binding->stride, (salt + 2u) & 63);
        hash ^= mglRotateLeft64((uint64_t)binding->divisor, (salt + 3u) & 63);
    }

    if (uses_elements) {
        mglHashBufferPointerName(&hash, vao->element_array.buffer, 0x900u);
        hash ^= mglRotateLeft64((uint64_t)vao->element_array.type, 11);
        hash ^= mglRotateLeft64((uint64_t)vao->element_array.size, 13);
        hash ^= mglRotateLeft64((uint64_t)(uintptr_t)vao->element_array.ptr, 17);
    }

    return hash;
}

static uint64_t mglComputeRenderStateHash(GLMContext ctx)
{
    uint64_t hash = mglHashBytes64(&ctx->state.caps,
                                   sizeof(ctx->state.caps),
                                   0xfeedfacecafebeefULL);
    GLMParams *var = &ctx->state.var;

    hash ^= mglHashBytes64(ctx->state.viewport, sizeof(ctx->state.viewport), 0x101u);
    hash ^= mglRotateLeft64((uint64_t)ctx->state.draw_buffer, 7);
    hash ^= mglRotateLeft64((uint64_t)ctx->state.draw_buffer_count, 13);
    hash ^= mglHashBytes64(ctx->state.draw_buffers, sizeof(ctx->state.draw_buffers), 0x102u);

    hash ^= mglHashBytes64(&var->point_size, sizeof(var->point_size), 0x201u);
    hash ^= mglHashBytes64(&var->line_width, sizeof(var->line_width), 0x202u);
    hash ^= mglRotateLeft64((uint64_t)var->polygon_mode, 5);
    hash ^= mglRotateLeft64((uint64_t)var->cull_face_mode, 11);
    hash ^= mglRotateLeft64((uint64_t)var->front_face, 17);
    hash ^= mglHashBytes64(var->depth_range, sizeof(var->depth_range), 0x203u);
    hash ^= var->depth_writemask ? 0xAAAA5555ULL : 0ULL;
    hash ^= mglRotateLeft64((uint64_t)var->depth_func, 8);

    hash ^= mglRotateLeft64((uint64_t)var->logic_op, 9);
    hash ^= mglRotateLeft64((uint64_t)var->logic_op_mode, 10);
    hash ^= mglHashBytes64(var->color_writemask, sizeof(var->color_writemask), 0x204u);
    hash ^= mglHashBytes64(var->blend_color, sizeof(var->blend_color), 0x205u);
    hash ^= mglHashBytes64(var->blend_dst_rgb, sizeof(var->blend_dst_rgb), 0x206u);
    hash ^= mglHashBytes64(var->blend_src_rgb, sizeof(var->blend_src_rgb), 0x207u);
    hash ^= mglHashBytes64(var->blend_dst_alpha, sizeof(var->blend_dst_alpha), 0x208u);
    hash ^= mglHashBytes64(var->blend_src_alpha, sizeof(var->blend_src_alpha), 0x209u);
    hash ^= mglHashBytes64(var->blend_equation_rgb, sizeof(var->blend_equation_rgb), 0x20au);
    hash ^= mglHashBytes64(var->blend_equation_alpha, sizeof(var->blend_equation_alpha), 0x20bu);

    hash ^= mglRotateLeft64((uint64_t)var->stencil_func, 24);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_value_mask, 25);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_fail, 26);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_pass_depth_fail, 27);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_pass_depth_pass, 28);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_ref, 29);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_writemask, 30);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_func, 31);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_fail, 32);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_pass_depth_fail, 33);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_pass_depth_pass, 34);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_ref, 35);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_value_mask, 36);
    hash ^= mglRotateLeft64((uint64_t)var->stencil_back_writemask, 37);

    hash ^= mglHashBytes64(&var->polygon_offset_units, sizeof(var->polygon_offset_units), 0x20cu);
    hash ^= mglHashBytes64(&var->polygon_offset_factor, sizeof(var->polygon_offset_factor), 0x20du);
    hash ^= mglHashBytes64(&var->sample_coverage_value, sizeof(var->sample_coverage_value), 0x20eu);
    hash ^= mglRotateLeft64((uint64_t)var->sample_coverage_invert, 38);
    hash ^= mglRotateLeft64((uint64_t)var->primitive_restart_index, 39);
    hash ^= mglRotateLeft64((uint64_t)var->clip_origin, 40);
    hash ^= mglRotateLeft64((uint64_t)var->clip_depth_mode, 41);
    hash ^= mglRotateLeft64((uint64_t)var->provoking_vertex, 42);

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
    out->program_pipeline_name = ctx->state.var.program_pipeline_binding;
    if (out->program_pipeline_name == 0 && ctx->state.program_pipeline) {
        out->program_pipeline_name = ctx->state.program_pipeline->name;
    }
    if (ctx->state.program_name == 0 && out->program_pipeline_name != 0) {
        ProgramPipeline *pipeline = ctx->state.program_pipeline;
        if (!pipeline || pipeline->name != out->program_pipeline_name) {
            pipeline = (ProgramPipeline *)searchHashTable(&ctx->state.program_pipeline_table,
                                                          out->program_pipeline_name);
        }
        out->vertex_program_name = pipeline && pipeline->stage_programs[_VERTEX_SHADER]
            ? pipeline->stage_programs[_VERTEX_SHADER]->name
            : 0u;
        out->fragment_program_name = pipeline && pipeline->stage_programs[_FRAGMENT_SHADER]
            ? pipeline->stage_programs[_FRAGMENT_SHADER]->name
            : 0u;
    }
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
    out->render_state_hash = mglComputeRenderStateHash(ctx) ^
                             mglComputeDrawBufferBindingHash(ctx) ^
                             mglComputeVertexArrayStateHash(ctx, uses_elements) ^
                             mglRotateLeft64((uint64_t)mode, 21);
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
    if (batch->uses_elements &&
        batch->command_count > 0 &&
        batch->commands &&
        batch->commands[0].indexType != cmd->indexType) {
        return false;
    }

    /* Emulated modes can't use MDI */
    GLenum mode = cmd->mode;
    if (mode == GL_TRIANGLE_FAN || mode == GL_LINE_LOOP) return false;

    return true;
}

static size_t mglCommandIndexSize(GLenum type)
{
    switch (type) {
        case GL_UNSIGNED_BYTE:  return 1;
        case GL_UNSIGNED_SHORT: return 2;
        case GL_UNSIGNED_INT:   return 4;
        default:                return 0;
    }
}

static size_t mglCommandAttribComponentSize(GLenum type)
{
    switch (type) {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            return 1;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
            return 2;
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
        case GL_FIXED:
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            return 4;
        case GL_DOUBLE:
            return 8;
        default:
            return 0;
    }
}

static size_t mglCommandAttribElementBytes(GLenum type, GLuint size)
{
    switch (type) {
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            return 4;
        default: {
            size_t component = mglCommandAttribComponentSize(type);
            if (component == 0 || size == 0 || size > SIZE_MAX / component) {
                return 0;
            }
            return component * (size_t)size;
        }
    }
}

typedef struct {
    VertexAttrib *attrib;
    Buffer       *buffer;
    GLintptr      binding_offset;
    GLuint        stride;
    GLuint        divisor;
    GLintptr      relativeoffset;
    GLuint        binding_index;
    bool          uses_binding_table;
} MGLCommandResolvedAttrib;

static bool mglCommandResolveVertexAttrib(VertexArray *vao,
                                          GLuint attribute,
                                          MGLCommandResolvedAttrib *out)
{
    if (!vao || attribute >= MAX_ATTRIBS || !out) {
        return false;
    }

    VertexAttrib *attrib = &vao->attrib[attribute];
    Buffer *buffer = attrib->buffer;
    GLintptr bindingOffset = attrib->binding_offset;
    GLuint stride = attrib->stride;
    GLuint divisor = attrib->divisor;
    GLuint bindingIndex = attrib->buffer_bindingindex;
    bool usesBindingTable = false;

    if (bindingIndex < MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
        BufferBinding *binding = &vao->bindings[bindingIndex];
        if (binding->buffer) {
            buffer = binding->buffer;
            bindingOffset = binding->offset;
            stride = (binding->stride > 0) ? (GLuint)binding->stride : attrib->stride;
            divisor = binding->divisor;
            usesBindingTable = true;
        }
    }

    if (!buffer) {
        return false;
    }

    out->attrib = attrib;
    out->buffer = buffer;
    out->binding_offset = bindingOffset;
    out->stride = stride;
    out->divisor = divisor;
    out->relativeoffset = attrib->relativeoffset;
    out->binding_index = bindingIndex;
    out->uses_binding_table = usesBindingTable;
    return true;
}

static bool mglCommandPrimitiveRestartIndex(GLMContext ctx, GLenum type, uint64_t *restart)
{
    if (!ctx || !restart ||
        (!ctx->state.caps.primitive_restart && !ctx->state.caps.primitive_restart_fixed_index)) {
        return false;
    }

    if (ctx->state.caps.primitive_restart_fixed_index) {
        switch (type) {
            case GL_UNSIGNED_BYTE:  *restart = 0xFFu; return true;
            case GL_UNSIGNED_SHORT: *restart = 0xFFFFu; return true;
            case GL_UNSIGNED_INT:   *restart = 0xFFFFFFFFu; return true;
            default: return false;
        }
    }

    *restart = (uint64_t)ctx->state.var.primitive_restart_index;
    return true;
}

static bool mglCommandReadIndexValue(const uint8_t *data, GLenum type, GLsizei i, uint64_t *value)
{
    if (!data || !value) return false;

    switch (type) {
        case GL_UNSIGNED_BYTE:
            *value = data[i];
            return true;
        case GL_UNSIGNED_SHORT: {
            uint16_t v;
            memcpy(&v, data + ((size_t)i * sizeof(v)), sizeof(v));
            *value = v;
            return true;
        }
        case GL_UNSIGNED_INT: {
            uint32_t v;
            memcpy(&v, data + ((size_t)i * sizeof(v)), sizeof(v));
            *value = v;
            return true;
        }
        default:
            return false;
    }
}

static bool mglCommandComputeElementVertexRange(GLMContext ctx,
                                                const MGLDrawCommand *cmd,
                                                uint64_t *minVertex,
                                                uint64_t *maxVertex)
{
    if (!ctx || !cmd || !minVertex || !maxVertex || cmd->count <= 0) return false;

    Buffer *elementBuffer = (Buffer *)cmd->elementBuffer;
    size_t indexSize = mglCommandIndexSize(cmd->indexType);
    if (!elementBuffer || !elementBuffer->data.buffer_data || indexSize == 0) return false;
    if (cmd->indexBufferOffset > (GLuint)elementBuffer->size) return false;
    if ((uint64_t)cmd->count > (UINT64_MAX / (uint64_t)indexSize)) return false;

    uint64_t byteCount = (uint64_t)cmd->count * (uint64_t)indexSize;
    uint64_t offset = (uint64_t)cmd->indexBufferOffset;
    if (byteCount > UINT64_MAX - offset ||
        offset + byteCount > (uint64_t)elementBuffer->size) {
        return false;
    }

    const uint8_t *data = (const uint8_t *)(uintptr_t)elementBuffer->data.buffer_data + offset;
    uint64_t restart = 0;
    bool hasRestart = mglCommandPrimitiveRestartIndex(ctx, cmd->indexType, &restart);

    uint64_t minIndex = UINT64_MAX;
    uint64_t maxIndex = 0;
    for (GLsizei i = 0; i < cmd->count; i++) {
        uint64_t index = 0;
        if (!mglCommandReadIndexValue(data, cmd->indexType, i, &index)) {
            return false;
        }
        if (hasRestart && index == restart) {
            continue;
        }
        if (index < minIndex) minIndex = index;
        if (index > maxIndex) maxIndex = index;
    }

    if (minIndex == UINT64_MAX) {
        return false;
    }

    if (cmd->baseVertex < 0) {
        uint64_t negBase = (uint64_t)(-((int64_t)cmd->baseVertex));
        *minVertex = (minIndex > negBase) ? (minIndex - negBase) : 0;
        *maxVertex = (maxIndex > negBase) ? (maxIndex - negBase) : 0;
    } else {
        uint64_t base = (uint64_t)cmd->baseVertex;
        if (minIndex > UINT64_MAX - base || maxIndex > UINT64_MAX - base) {
            return false;
        }
        *minVertex = minIndex + base;
        *maxVertex = maxIndex + base;
    }

    return true;
}

static bool mglCommandComputeArrayVertexRange(const MGLDrawCommand *cmd,
                                              uint64_t *minVertex,
                                              uint64_t *maxVertex)
{
    if (!cmd || !minVertex || !maxVertex || cmd->first < 0 || cmd->count <= 0) {
        return false;
    }

    uint64_t first = (uint64_t)cmd->first;
    uint64_t count = (uint64_t)cmd->count;
    if (count == 0 || first > UINT64_MAX - (count - 1u)) {
        return false;
    }

    *minVertex = first;
    *maxVertex = first + count - 1u;
    return true;
}

static void mglTrackPendingAttribRead(GLMContext ctx,
                                      const MGLCommandResolvedAttrib *resolved,
                                      uint64_t minVertex,
                                      uint64_t maxVertex,
                                      GLsizei instanceCount,
                                      GLuint baseInstance)
{
    if (!ctx || !resolved || !resolved->attrib || !resolved->buffer) return;

    VertexAttrib *attrib = resolved->attrib;
    size_t elemBytes = mglCommandAttribElementBytes(attrib->type, attrib->size);
    uint64_t stride = resolved->stride > 0 ? (uint64_t)resolved->stride : (uint64_t)elemBytes;
    if (elemBytes == 0 || stride == 0 ||
        resolved->binding_offset < 0 || resolved->relativeoffset < 0) {
        mglTrackPendingReadWholeBuffer(ctx, resolved->buffer);
        return;
    }

    uint64_t firstIndex = minVertex;
    uint64_t lastIndex = maxVertex;
    if (resolved->divisor != 0) {
        uint64_t instances = instanceCount > 0 ? (uint64_t)instanceCount : 1u;
        uint64_t base = (uint64_t)baseInstance;
        firstIndex = base / (uint64_t)resolved->divisor;
        lastIndex = (base + instances - 1u) / (uint64_t)resolved->divisor;
    }

    uint64_t baseOffset = (uint64_t)resolved->binding_offset + (uint64_t)resolved->relativeoffset;
    if (baseOffset > UINT64_MAX - ((uint64_t)elemBytes) ||
        firstIndex > UINT64_MAX / stride ||
        lastIndex > UINT64_MAX / stride) {
        mglTrackPendingReadWholeBuffer(ctx, resolved->buffer);
        return;
    }

    uint64_t start = baseOffset + firstIndex * stride;
    uint64_t end = baseOffset + lastIndex * stride + (uint64_t)elemBytes;
    if (end <= start) {
        mglTrackPendingReadWholeBuffer(ctx, resolved->buffer);
        return;
    }

    mglTrackPendingReadRange(ctx, resolved->buffer, start, end);
}

static void mglTrackPendingBaseBufferReads(GLMContext ctx)
{
    if (!ctx) return;

    const int trackedTargets[] = {
        _UNIFORM_BUFFER,
        _UNIFORM_CONSTANT,
        _SHADER_STORAGE_BUFFER,
        _ATOMIC_COUNTER_BUFFER,
        _TEXTURE_BUFFER
    };

    for (size_t t = 0; t < sizeof(trackedTargets) / sizeof(trackedTargets[0]); t++) {
        int target = trackedTargets[t];
        for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            BufferBaseTarget *binding = &ctx->state.buffer_base[target].buffers[i];
            if (!binding->buf) continue;
            if (binding->size > 0 && binding->offset >= 0) {
                mglTrackPendingReadBytes(ctx,
                                         binding->buf,
                                         (uint64_t)binding->offset,
                                         (uint64_t)binding->size);
            } else {
                mglTrackPendingReadWholeBuffer(ctx, binding->buf);
            }
        }
    }

    Program *program = ctx->state.program;
    if (program) {
        for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++) {
            BufferBaseTarget *binding = &program->plain_uniform_buffers[i];
            if (!binding->buf) continue;
            if (binding->size > 0 && binding->offset >= 0) {
                mglTrackPendingReadBytes(ctx,
                                         binding->buf,
                                         (uint64_t)binding->offset,
                                         (uint64_t)binding->size);
            } else {
                mglTrackPendingReadWholeBuffer(ctx, binding->buf);
            }
        }
    }
}

#define MGL_STREAM_MERGE_MAX_SOURCE_BYTES (64u * 1024u)
#define MGL_STREAM_MERGE_MAX_BATCH_BYTES  (4u * 1024u * 1024u)

typedef struct {
    VertexArray *vao;
    Buffer      *vertex_buffer;
    Buffer      *index_buffer;
    const uint8_t *index_bytes;
    size_t       index_size;
    size_t       vertex_bytes;
    size_t       vertex_stride;
    size_t       max_attrib_end;
    GLintptr     binding_offset;
    uint32_t     attrib_mask;
    uint64_t     layout_hash;
} MGLStreamMergeCandidate;

static size_t mglAlignUpSize(size_t value, size_t alignment)
{
    if (alignment <= 1u) return value;
    size_t rem = value % alignment;
    if (rem == 0u) return value;
    if (value > SIZE_MAX - (alignment - rem)) return SIZE_MAX;
    return value + (alignment - rem);
}

static Buffer *mglNewTransientBatchBuffer(GLenum target)
{
    Buffer *buffer = (Buffer *)calloc(1, sizeof(Buffer));
    if (!buffer) return NULL;

    buffer->name = 0;
    buffer->target = target;
    buffer->size = 0;
    buffer->written_min = -1;
    buffer->written_max = -1;
    buffer->transient_batch_buffer = GL_TRUE;
    buffer->data.dirty_bits = DIRTY_BUFFER_ADDR | DIRTY_BUFFER_DATA;
    return buffer;
}

static bool mglEnsureTransientBufferCapacity(Buffer *buffer, size_t needed)
{
    if (!buffer) return false;
    if (needed <= buffer->data.buffer_size) return true;

    size_t newCapacity = buffer->data.buffer_size ? buffer->data.buffer_size : 4096u;
    while (newCapacity < needed) {
        if (newCapacity > SIZE_MAX / 2u) {
            newCapacity = needed;
            break;
        }
        newCapacity *= 2u;
    }

    void *oldData = (void *)(uintptr_t)buffer->data.buffer_data;
    void *newData = realloc(oldData, newCapacity);
    if (!newData) return false;

    buffer->data.buffer_data = (vm_address_t)(uintptr_t)newData;
    buffer->data.buffer_size = newCapacity;
    buffer->data.dirty_bits |= DIRTY_BUFFER_ADDR | DIRTY_BUFFER_DATA;
    return true;
}

static void mglMarkTransientBufferWritten(Buffer *buffer, size_t bytes)
{
    if (!buffer) return;

    buffer->size = (GLsizeiptr)bytes;
    buffer->data.dirty_bits |= DIRTY_BUFFER_ADDR | DIRTY_BUFFER_DATA;
    buffer->has_initialized_data = GL_TRUE;
    buffer->ever_written = GL_TRUE;
    buffer->written_min = 0;
    buffer->written_max = (GLintptr)bytes;
    buffer->last_init_source = kInitBufferDataCopy;
    buffer->last_write_offset = 0;
    buffer->last_write_size = (GLsizeiptr)bytes;
    buffer->last_write_src_ptr = (const void *)(uintptr_t)buffer->data.buffer_data;
    buffer->last_write_src_hash = 0;
}

static uint64_t mglStreamHashAttrib(uint64_t hash,
                                    const MGLCommandResolvedAttrib *resolved,
                                    GLuint index)
{
    if (!resolved || !resolved->attrib) return hash;

    const VertexAttrib *attrib = resolved->attrib;
    hash ^= mglRotateLeft64((uint64_t)index, 3);
    hash ^= mglRotateLeft64((uint64_t)attrib->size, 7);
    hash ^= mglRotateLeft64((uint64_t)attrib->type, 11);
    hash ^= mglRotateLeft64((uint64_t)attrib->normalized, 13);
    hash ^= mglRotateLeft64((uint64_t)attrib->integer, 17);
    hash ^= mglRotateLeft64((uint64_t)attrib->long_attribute, 19);
    hash ^= mglRotateLeft64((uint64_t)resolved->stride, 23);
    hash ^= mglRotateLeft64((uint64_t)resolved->relativeoffset, 29);
    hash ^= mglRotateLeft64((uint64_t)resolved->binding_index, 31);
    hash ^= mglRotateLeft64((uint64_t)resolved->binding_offset, 41);
    hash ^= mglRotateLeft64((uint64_t)resolved->divisor, 47);
    return hash;
}

static bool mglLooksLikeWeatherParticleStream(GLMContext ctx,
                                              const VertexArray *vao,
                                              const Buffer *vertexBuffer,
                                              uint32_t attribMask,
                                              size_t vertexStride)
{
    if (!ctx || !vao || !vertexBuffer) return false;
    if (vertexBuffer->usage != GL_DYNAMIC_DRAW) return false;
    if (vertexBuffer->size <= 0 ||
        vertexBuffer->size > (GLsizeiptr)MGL_STREAM_MERGE_MAX_SOURCE_BYTES) {
        return false;
    }
    if (vertexStride != 28u || (attribMask & 0xfu) != 0xfu) return false;

    bool blendEnabled = (ctx->state.caps.blend == GL_TRUE);
    if (!blendEnabled) {
        GLuint maxDrawBuffers = ctx->state.var.max_draw_buffers;
        if (maxDrawBuffers > MAX_COLOR_ATTACHMENTS) maxDrawBuffers = MAX_COLOR_ATTACHMENTS;
        for (GLuint i = 0; i < maxDrawBuffers; i++) {
            if (ctx->state.caps.blendi[i] == GL_TRUE) {
                blendEnabled = true;
                break;
            }
        }
    }
    if (!blendEnabled || ctx->state.var.depth_writemask == GL_TRUE) return false;

    const VertexAttrib *pos = &vao->attrib[0];
    const VertexAttrib *uv = &vao->attrib[1];
    const VertexAttrib *color = &vao->attrib[2];
    const VertexAttrib *light = &vao->attrib[3];

    return pos->type == GL_FLOAT && pos->size == 3 && pos->relativeoffset == 0 &&
           uv->type == GL_FLOAT && uv->size == 2 && uv->relativeoffset == 12 &&
           color->type == GL_UNSIGNED_BYTE && color->size == 4 &&
           color->normalized == GL_TRUE && color->relativeoffset == 20 &&
           light->type == GL_SHORT && light->size == 2 &&
           light->integer == GL_TRUE && light->relativeoffset == 24;
}

static bool mglLooksLikeMinecraftChunkTerrainStream(const VertexArray *vao,
                                                    uint32_t attribMask,
                                                    size_t vertexStride)
{
    if (!vao) return false;
    if (vertexStride != 32u || (attribMask & 0x1fu) != 0x1fu) return false;

    const VertexAttrib *pos = &vao->attrib[0];
    const VertexAttrib *color = &vao->attrib[1];
    const VertexAttrib *uv = &vao->attrib[2];
    const VertexAttrib *light = &vao->attrib[3];
    const VertexAttrib *normal = &vao->attrib[4];

    return pos->type == GL_FLOAT && pos->size == 3 &&
           pos->relativeoffset == 0 &&
           color->type == GL_UNSIGNED_BYTE && color->size == 4 &&
           color->normalized == GL_TRUE && color->relativeoffset == 12 &&
           uv->type == GL_FLOAT && uv->size == 2 &&
           uv->relativeoffset == 16 &&
           light->type == GL_SHORT && light->size == 2 &&
           light->integer == GL_TRUE && light->relativeoffset == 24 &&
           normal->type == GL_BYTE && normal->size == 3 &&
           normal->normalized == GL_TRUE && normal->relativeoffset == 28;
}

static bool mglPrepareStreamMergeCandidate(GLMContext ctx,
                                           const MGLDrawCommand *cmd,
                                           bool uses_elements,
                                           MGLStreamMergeCandidate *out)
{
    if (!ctx || !cmd || !out || !uses_elements) return false;
    if (cmd->mode != GL_TRIANGLES || cmd->count <= 0) return false;
    if (cmd->instanceCount != 1 || cmd->baseInstance != 0) return false;
    if (ctx->state.var.polygon_mode != GL_FILL) return false;

    uint64_t restart = 0;
    if (mglCommandPrimitiveRestartIndex(ctx, cmd->indexType, &restart)) {
        return false;
    }

    VertexArray *vao = ctx->state.vao;
    Buffer *indexBuffer = (Buffer *)cmd->elementBuffer;
    size_t indexSize = mglCommandIndexSize(cmd->indexType);
    if (!vao || !indexBuffer || indexBuffer->mapped ||
        !indexBuffer->data.buffer_data || indexSize == 0u) {
        return false;
    }
    if (cmd->indexBufferOffset > (GLuint)indexBuffer->size) {
        return false;
    }
    if ((uint64_t)cmd->count > UINT64_MAX / (uint64_t)indexSize) {
        return false;
    }

    uint64_t indexBytes = (uint64_t)cmd->count * (uint64_t)indexSize;
    uint64_t indexOffset = (uint64_t)cmd->indexBufferOffset;
    if (indexBytes > UINT64_MAX - indexOffset ||
        indexOffset + indexBytes > (uint64_t)indexBuffer->size) {
        return false;
    }

    GLuint maxAttribs = MAX_ATTRIBS;

    bool explicitAttribs = (vao->enabled_attribs != 0u);
    Buffer *vertexBuffer = NULL;
    size_t vertexStride = 0;
    GLintptr bindingOffset = -1;
    size_t maxAttribEnd = 0;
    uint32_t attribMask = 0;
    uint64_t layoutHash = 0x9e3779b97f4a7c15ULL;

    for (GLuint i = 0; i < maxAttribs; i++) {
        bool useAttrib = explicitAttribs
            ? ((vao->enabled_attribs & (1u << i)) != 0u)
            : (vao->attrib[i].buffer != NULL);
        if (!useAttrib) {
            if (explicitAttribs && (vao->enabled_attribs >> (i + 1u)) == 0u) break;
            continue;
        }

        MGLCommandResolvedAttrib resolved = {0};
        if (!mglCommandResolveVertexAttrib(vao, i, &resolved) ||
            resolved.divisor != 0u ||
            resolved.binding_offset < 0 || resolved.relativeoffset < 0) {
            return false;
        }

        VertexAttrib *attrib = resolved.attrib;
        size_t elemBytes = mglCommandAttribElementBytes(attrib->type, attrib->size);
        size_t stride = resolved.stride > 0 ? (size_t)resolved.stride : elemBytes;
        if (elemBytes == 0u || stride == 0u) return false;
        if ((size_t)resolved.relativeoffset > SIZE_MAX - elemBytes) return false;
        size_t attribEnd = (size_t)resolved.relativeoffset + elemBytes;
        if (attribEnd > stride) return false;

        if (!vertexBuffer) {
            vertexBuffer = resolved.buffer;
            vertexStride = stride;
            bindingOffset = resolved.binding_offset;
        } else if (vertexBuffer != resolved.buffer ||
                   vertexStride != stride ||
                   bindingOffset != resolved.binding_offset) {
            return false;
        }

        attribMask |= (1u << i);
        if (attribEnd > maxAttribEnd) maxAttribEnd = attribEnd;
        layoutHash = mglStreamHashAttrib(layoutHash, &resolved, i);
    }

    if (!vertexBuffer || vertexBuffer->mapped ||
        attribMask == 0u || !vertexBuffer->data.buffer_data ||
        vertexBuffer->size <= 0 || vertexBuffer->size > (GLsizeiptr)MGL_STREAM_MERGE_MAX_SOURCE_BYTES) {
        return false;
    }
    if (bindingOffset < 0 || vertexStride == 0u ||
        ((size_t)bindingOffset % vertexStride) != 0u ||
        ((size_t)vertexBuffer->size % vertexStride) != 0u) {
        return false;
    }
    if (mglLooksLikeWeatherParticleStream(ctx, vao, vertexBuffer, attribMask, vertexStride)) {
        return false;
    }
    if (mglLooksLikeMinecraftChunkTerrainStream(vao, attribMask, vertexStride)) {
        return false;
    }

    const uint8_t *indices = (const uint8_t *)(uintptr_t)indexBuffer->data.buffer_data + indexOffset;
    for (GLsizei i = 0; i < cmd->count; i++) {
        uint64_t rawIndex = 0;
        if (!mglCommandReadIndexValue(indices, cmd->indexType, i, &rawIndex)) {
            return false;
        }

        int64_t sourceIndex = (int64_t)rawIndex + (int64_t)cmd->baseVertex;
        if (sourceIndex < 0) return false;

        uint64_t vertexByte = (uint64_t)bindingOffset +
                              ((uint64_t)sourceIndex * (uint64_t)vertexStride);
        if (vertexByte > UINT64_MAX - (uint64_t)maxAttribEnd ||
            vertexByte + (uint64_t)maxAttribEnd > (uint64_t)vertexBuffer->size) {
            return false;
        }
    }

    memset(out, 0, sizeof(*out));
    out->vao = vao;
    out->vertex_buffer = vertexBuffer;
    out->index_buffer = indexBuffer;
    out->index_bytes = indices;
    out->index_size = indexSize;
    out->vertex_bytes = (size_t)vertexBuffer->size;
    out->vertex_stride = vertexStride;
    out->max_attrib_end = maxAttribEnd;
    out->binding_offset = bindingOffset;
    out->attrib_mask = attribMask;
    out->layout_hash = layoutHash ^
                       mglRotateLeft64((uint64_t)vertexStride, 37) ^
                       mglRotateLeft64((uint64_t)bindingOffset, 43);
    return true;
}

static bool mglInitializeStreamMergedBatch(GLMContext ctx,
                                           MGLDrawBatch *batch,
                                           const MGLStreamMergeCandidate *candidate)
{
    if (!ctx || !batch || !candidate || !candidate->vao) return false;

    batch->state_snapshot = malloc(sizeof(ctx->state));
    batch->vao_snapshot = malloc(sizeof(VertexArray));
    batch->stream_vertex_buffer = mglNewTransientBatchBuffer(GL_ARRAY_BUFFER);
    batch->stream_index_buffer = mglNewTransientBatchBuffer(GL_ELEMENT_ARRAY_BUFFER);
    if (!batch->state_snapshot || !batch->vao_snapshot ||
        !batch->stream_vertex_buffer || !batch->stream_index_buffer) {
        return false;
    }

    memcpy(batch->state_snapshot, &ctx->state, sizeof(ctx->state));
    memcpy(batch->vao_snapshot, candidate->vao, sizeof(VertexArray));

    VertexArray *vao = (VertexArray *)batch->vao_snapshot;
    Buffer *vertexBuffer = (Buffer *)batch->stream_vertex_buffer;
    Buffer *indexBuffer = (Buffer *)batch->stream_index_buffer;

    vao->transient_batch_vao = GL_TRUE;
    vao->mtl_data = NULL;
    vao->dirty_bits |= DIRTY_VAO_ATTRIB | DIRTY_VAO_BUFFER_BASE;
    vao->element_array.buffer = indexBuffer;

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        if ((candidate->attrib_mask & (1u << i)) == 0u) continue;
        GLuint binding = vao->attrib[i].buffer_bindingindex;
        vao->attrib[i].buffer = vertexBuffer;
        vao->attrib[i].binding_offset = 0;
        if (binding < MGL_MAX_VERTEX_ATTRIB_BINDINGS &&
            vao->bindings[binding].buffer == candidate->vertex_buffer) {
            vao->bindings[binding].buffer = vertexBuffer;
            vao->bindings[binding].offset = 0;
        }
    }

    GLMState *snapshot = (GLMState *)batch->state_snapshot;
    snapshot->vao = vao;
    snapshot->buffers[_ARRAY_BUFFER] = vertexBuffer;
    snapshot->buffers[_ELEMENT_ARRAY_BUFFER] = indexBuffer;
    snapshot->var.array_buffer_binding = 0;
    snapshot->var.element_array_buffer_binding = 0;
    snapshot->dirty_bits |= DIRTY_VAO | DIRTY_BUFFER;

    batch->stream_merged = true;
    batch->stream_layout_hash = candidate->layout_hash;
    batch->stream_vertex_stride = candidate->vertex_stride;
    batch->mdi_compatible = false;
    mglRetainBatchProgramReferences(ctx, batch);
    return true;
}

static bool mglAppendStreamMergedData(MGLDrawBatch *batch,
                                      const MGLStreamMergeCandidate *candidate,
                                      const MGLDrawCommand *srcCmd,
                                      MGLDrawCommand *storedCmd)
{
    if (!batch || !candidate || !srcCmd || !storedCmd ||
        !batch->stream_vertex_buffer || !batch->stream_index_buffer) {
        return false;
    }

    Buffer *vertexBuffer = (Buffer *)batch->stream_vertex_buffer;
    Buffer *indexBuffer = (Buffer *)batch->stream_index_buffer;

    size_t vertexOffset = mglAlignUpSize(batch->stream_vertex_bytes, candidate->vertex_stride);
    if (vertexOffset == SIZE_MAX ||
        vertexOffset > MGL_STREAM_MERGE_MAX_BATCH_BYTES ||
        candidate->vertex_bytes > MGL_STREAM_MERGE_MAX_BATCH_BYTES - vertexOffset) {
        return false;
    }
    size_t newVertexBytes = vertexOffset + candidate->vertex_bytes;

    uint64_t vertexBase = ((uint64_t)vertexOffset + (uint64_t)candidate->binding_offset) /
                          (uint64_t)candidate->vertex_stride;

    if ((uint64_t)srcCmd->count > (UINT64_MAX / sizeof(uint32_t)) ||
        (size_t)srcCmd->count > (MGL_STREAM_MERGE_MAX_BATCH_BYTES - batch->stream_index_bytes) / sizeof(uint32_t)) {
        return false;
    }
    size_t indexWriteOffset = batch->stream_index_bytes;
    size_t indexBytesToAppend = (size_t)srcCmd->count * sizeof(uint32_t);
    size_t newIndexBytes = indexWriteOffset + indexBytesToAppend;

    for (GLsizei i = 0; i < srcCmd->count; i++) {
        uint64_t rawIndex = 0;
        if (!mglCommandReadIndexValue(candidate->index_bytes, srcCmd->indexType, i, &rawIndex)) {
            return false;
        }
        int64_t sourceIndex = (int64_t)rawIndex + (int64_t)srcCmd->baseVertex;
        if (sourceIndex < 0) return false;
        uint64_t mergedIndex = vertexBase + (uint64_t)sourceIndex;
        if (mergedIndex > UINT32_MAX) return false;
    }

    if (!mglEnsureTransientBufferCapacity(vertexBuffer, newVertexBytes) ||
        !mglEnsureTransientBufferCapacity(indexBuffer, newIndexBytes)) {
        return false;
    }

    uint8_t *vertexDst = (uint8_t *)(uintptr_t)vertexBuffer->data.buffer_data;
    if (vertexOffset > batch->stream_vertex_bytes) {
        memset(vertexDst + batch->stream_vertex_bytes, 0, vertexOffset - batch->stream_vertex_bytes);
    }
    memcpy(vertexDst + vertexOffset,
           (const void *)(uintptr_t)candidate->vertex_buffer->data.buffer_data,
           candidate->vertex_bytes);

    uint32_t *indexDst = (uint32_t *)((uint8_t *)(uintptr_t)indexBuffer->data.buffer_data + indexWriteOffset);
    for (GLsizei i = 0; i < srcCmd->count; i++) {
        uint64_t rawIndex = 0;
        (void)mglCommandReadIndexValue(candidate->index_bytes, srcCmd->indexType, i, &rawIndex);
        int64_t sourceIndex = (int64_t)rawIndex + (int64_t)srcCmd->baseVertex;
        indexDst[i] = (uint32_t)(vertexBase + (uint64_t)sourceIndex);
    }

    batch->stream_vertex_bytes = newVertexBytes;
    batch->stream_index_bytes = newIndexBytes;
    batch->stream_index_count += (size_t)srcCmd->count;
    mglMarkTransientBufferWritten(vertexBuffer, batch->stream_vertex_bytes);
    mglMarkTransientBufferWritten(indexBuffer, batch->stream_index_bytes);

    *storedCmd = *srcCmd;
    storedCmd->indexType = GL_UNSIGNED_INT;
    storedCmd->indexBufferOffset = (GLuint)indexWriteOffset;
    storedCmd->elementBuffer = indexBuffer;
    storedCmd->baseVertex = 0;
    storedCmd->baseInstance = 0;
    return true;
}

static void mglTrackPendingDrawBufferReads(GLMContext ctx,
                                           const MGLDrawCommand *cmd,
                                           bool uses_elements)
{
    if (!ctx || !cmd) return;

    if (uses_elements && cmd->elementBuffer && cmd->count > 0) {
        size_t indexSize = mglCommandIndexSize(cmd->indexType);
        if (indexSize > 0) {
            uint64_t indexBytes = (uint64_t)cmd->count * (uint64_t)indexSize;
            mglTrackPendingReadBytes(ctx,
                                     (Buffer *)cmd->elementBuffer,
                                     (uint64_t)cmd->indexBufferOffset,
                                     indexBytes);
        } else {
            mglTrackPendingReadWholeBuffer(ctx, (Buffer *)cmd->elementBuffer);
        }
    }

    uint64_t minVertex = 0;
    uint64_t maxVertex = 0;
    bool rangeKnown = uses_elements
        ? mglCommandComputeElementVertexRange(ctx, cmd, &minVertex, &maxVertex)
        : mglCommandComputeArrayVertexRange(cmd, &minVertex, &maxVertex);

    VertexArray *vao = ctx->state.vao;
    if (vao) {
        GLuint maxAttribs = MAX_ATTRIBS;

        bool explicitAttribs = (vao->enabled_attribs != 0u);
        for (GLuint i = 0; i < maxAttribs; i++) {
            bool useAttrib = explicitAttribs
                ? ((vao->enabled_attribs & (1u << i)) != 0u)
                : (mglCommandResolveVertexAttrib(vao, i, &(MGLCommandResolvedAttrib){0}));
            if (!useAttrib) {
                if (explicitAttribs && (vao->enabled_attribs >> (i + 1u)) == 0u) break;
                continue;
            }

            MGLCommandResolvedAttrib resolved = {0};
            if (!mglCommandResolveVertexAttrib(vao, i, &resolved)) continue;

            if (rangeKnown) {
                mglTrackPendingAttribRead(ctx,
                                          &resolved,
                                          minVertex,
                                          maxVertex,
                                          cmd->instanceCount,
                                          cmd->baseInstance);
            } else {
                mglTrackPendingReadWholeBuffer(ctx, resolved.buffer);
            }
        }
    }

    mglTrackPendingBaseBufferReads(ctx);
}

void mglAppendDrawCommand(GLMContext ctx, const MGLDrawCommand *cmd)
{
    if (!ctx || !cmd) return;

    mglFlushPendingDrawsForActiveTextures(ctx);
    mglFlushPendingDrawsBeforeFramebufferTextureWrites(ctx);

    MGLCommandBuffer *cb = &ctx->draw_command_buffer;
    bool cmd_uses_elements =
        (cmd->type != MGL_CMD_DRAW_ARRAYS &&
         cmd->type != MGL_CMD_DRAW_ARRAYS_INSTANCED &&
         cmd->type != MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE);

    MGLStateKey key;
    mglComputeStateKey(ctx, cmd->mode, cmd_uses_elements, &key);

    MGLStreamMergeCandidate streamCandidate;
    bool can_stream_merge =
        mglPrepareStreamMergeCandidate(ctx, cmd, cmd_uses_elements, &streamCandidate);
    /*
     * A normal deferred batch replays one captured GL state with many Metal
     * draws. That is only correct after stream-merge has copied all varying
     * vertex/index data into transient buffers. Otherwise per-draw VAO, UBO
     * range, texture, or render-state changes must keep their own snapshot.
     */
    bool can_reuse_batch = can_stream_merge;

    /* Find matching batch (check last first for spatial locality) */
    MGLDrawBatch *batch = NULL;
    if (can_reuse_batch && cb->batch_count > 0) {
        MGLDrawBatch *last = &cb->batches[cb->batch_count - 1];
        if (mglStateKeysEqual(&last->key, &key) &&
            last->uses_elements == cmd_uses_elements &&
            last->stream_merged == can_stream_merge &&
            (!can_stream_merge || last->stream_layout_hash == streamCandidate.layout_hash) &&
            last->command_count < MGL_MAX_DRAWS_PER_BATCH) {
            batch = last;
        }
    }
    if (!batch) {
        if (cb->batch_count >= MGL_MAX_BATCHES) {
            mglFlushCommandBuffer(ctx);
            if (cb->batch_count >= MGL_MAX_BATCHES) {
                fprintf(stderr, "MGL Error: mglAppendDrawCommand: batch buffer full after flush\n");
                return;
            }
        }
        batch = &cb->batches[cb->batch_count];
        memset(batch, 0, sizeof(*batch));
        batch->key = key;
        batch->mdi_compatible = can_stream_merge;
        batch->uses_elements = cmd_uses_elements;

        if (can_stream_merge) {
            if (!mglInitializeStreamMergedBatch(ctx, batch, &streamCandidate)) {
                fprintf(stderr, "MGL Warning: stream merged batch init failed; falling back to normal deferred draw\n");
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                batch->key = key;
                batch->mdi_compatible = false;
                batch->uses_elements = cmd_uses_elements;
                can_stream_merge = false;
            }
        }

        if (!can_stream_merge) {
            if (!mglInitializeBatchStateSnapshot(ctx, batch)) {
                fprintf(stderr, "MGL Error: mglAppendDrawCommand: state snapshot alloc failed\n");
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                return;
            }
        }

        cb->batch_count++;
    }

    MGLDrawCommand stored_cmd = *cmd;
    if (can_stream_merge) {
        if (!mglAppendStreamMergedData(batch, &streamCandidate, cmd, &stored_cmd)) {
            fprintf(stderr, "MGL Warning: stream merged append failed; falling back to normal deferred draw\n");
            if (batch->command_count == 0 &&
                cb->batch_count > 0 &&
                batch == &cb->batches[cb->batch_count - 1]) {
                mglReleaseBatch(ctx, batch);
                memset(batch, 0, sizeof(*batch));
                cb->batch_count--;
            }
            can_stream_merge = false;
            batch = NULL;
            if (!batch) {
                if (cb->batch_count >= MGL_MAX_BATCHES) {
                    mglFlushCommandBuffer(ctx);
                    if (cb->batch_count >= MGL_MAX_BATCHES) {
                        fprintf(stderr, "MGL Error: mglAppendDrawCommand: fallback batch buffer full after flush\n");
                        return;
                    }
                }
                batch = &cb->batches[cb->batch_count];
                memset(batch, 0, sizeof(*batch));
                batch->key = key;
                batch->mdi_compatible = false;
                batch->uses_elements = cmd_uses_elements;
                if (!mglInitializeBatchStateSnapshot(ctx, batch)) {
                    fprintf(stderr, "MGL Error: mglAppendDrawCommand: fallback state snapshot alloc failed\n");
                    mglReleaseBatch(ctx, batch);
                    memset(batch, 0, sizeof(*batch));
                    return;
                }
                cb->batch_count++;
            }
            stored_cmd = *cmd;
        }
    }

    /* Resize command array */
    MGLDrawCommand *new_cmds = (MGLDrawCommand *)realloc(batch->commands,
        (batch->command_count + 1) * sizeof(MGLDrawCommand));
    if (!new_cmds) {
        fprintf(stderr, "MGL Error: mglAppendDrawCommand: realloc failed\n");
        if (batch->command_count == 0 &&
            cb->batch_count > 0 &&
            batch == &cb->batches[cb->batch_count - 1]) {
            mglReleaseBatch(ctx, batch);
            memset(batch, 0, sizeof(*batch));
            cb->batch_count--;
        }
        return;
    }
    batch->commands = new_cmds;
    batch->commands[batch->command_count] = stored_cmd;
    batch->command_count++;
    cb->total_commands++;

    if (!can_stream_merge || !mglBatchIsMDICompatible(batch, &stored_cmd)) {
        batch->mdi_compatible = false;
    }

    if (batch->uses_elements) {
        cb->element_cmd_count++;
    } else {
        cb->array_cmd_count++;
    }

    if (can_stream_merge) {
        mglTrackPendingBaseBufferReads(ctx);
    } else {
        mglTrackPendingDrawBufferReads(ctx, cmd, cmd_uses_elements);
    }
    mglTrackPendingSampledTextureReads(ctx);
    mglTrackPendingFramebufferTextureWrites(ctx);
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

void mglFlushPendingDraws(GLMContext ctx)
{
    if (!ctx || !ctx->draw_defer_enabled) return;
    mglFlushCommandBuffer(ctx);
}
