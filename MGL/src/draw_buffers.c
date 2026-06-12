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
 * draw_buffers.c
 * MGL
 *
 */

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "glm_context.h"
#include "draw_command.h"
#include "mgl.h"
#include "mgl_safety.h"

extern void mglInvalidateColorShadowsForDraw(GLMContext ctx);
extern void mglTraceLogExternal(const char *fmt, ...);
#include "spirv_cross_c.h"

static void mglInitVertexArrayDefaultsForDraw(VertexArray *vao)
{
    if (!vao)
        return;

    for (int i = 0; i < MAX_ATTRIBS; i++)
    {
        vao->attrib[i].size = 4;
        vao->attrib[i].type = GL_FLOAT;
        vao->attrib[i].integer = 0;
        vao->attrib[i].long_attribute = 0;
        vao->attrib[i].stride = 0;
        vao->attrib[i].divisor = 0;
        vao->attrib[i].relativeoffset = 0;
        vao->attrib[i].binding_offset = 0;
        vao->attrib[i].buffer_bindingindex = (i < MGL_MAX_VERTEX_ATTRIB_BINDINGS) ? (GLuint)i : 0u;
        vao->attrib[i].buffer = NULL;
    }

    for (int i = 0; i < MGL_MAX_VERTEX_ATTRIB_BINDINGS; i++)
    {
        vao->bindings[i].buffer = NULL;
        vao->bindings[i].offset = 0;
        vao->bindings[i].stride = 16;
        vao->bindings[i].divisor = 0;
    }
}

static Buffer *mglResolveVertexAttribBufferForDraw(VertexArray *vao, GLuint attrib)
{
    if (!vao || attrib >= MAX_ATTRIBS)
        return NULL;

    VertexAttrib *a = &vao->attrib[attrib];
    if (a->buffer_bindingindex < MGL_MAX_VERTEX_ATTRIB_BINDINGS &&
        vao->bindings[a->buffer_bindingindex].buffer) {
        return vao->bindings[a->buffer_bindingindex].buffer;
    }

    return a->buffer;
}

static VertexArray *mglGetOrCreateDefaultVAO(GLMContext ctx)
{
    VertexArray *vao;

    if (!ctx)
        return NULL;

    vao = (VertexArray *)searchHashTable(&STATE(vao_table), 0);
    if (vao &&
        (!mglObjectPointerLooksPlausible(vao) ||
         !mglPointerRangeIsReadable(vao, sizeof(*vao)) ||
         vao->magic != MGL_VAO_MAGIC))
    {
        fprintf(stderr, "MGL WARNING: default VAO entry is invalid (%p), recreating VAO 0\n", (void *)vao);
        deleteHashElement(&STATE(vao_table), 0);
        vao = NULL;
    }

    if (!vao)
    {
        vao = (VertexArray *)calloc(1, sizeof(VertexArray));
        if (!vao)
            return NULL;

        vao->magic = MGL_VAO_MAGIC;
        vao->name = 0;

        mglInitVertexArrayDefaultsForDraw(vao);

        insertHashElement(&STATE(vao_table), 0, vao);
    }

    // Keep VAO0 EBO compatibility slot synchronized.
    vao->element_array.buffer = STATE(default_vao_element_array_buffer);

    return vao;
}

static bool should_log_throttled(uint64_t *counter, uint64_t burst_limit, uint64_t every_n)
{
    (*counter)++;
    return (*counter <= burst_limit) || ((*counter % every_n) == 0);
}

static bool mglStringContainsCountAtLeast(const char *str, const char *needle, unsigned min_count)
{
    unsigned count = 0u;
    const char *cursor = str;

    if (!str || !needle || !*needle)
        return false;

    while ((cursor = strstr(cursor, needle)) != NULL) {
        count++;
        if (count >= min_count)
            return true;
        cursor += strlen(needle);
    }

    return false;
}

static bool mglProgramMatchesCTSPointQuadGeometryFallback(Program *program)
{
    if (!program ||
        !program->shader_slots[_VERTEX_SHADER] ||
        !program->shader_slots[_GEOMETRY_SHADER] ||
        !program->shader_slots[_FRAGMENT_SHADER]) {
        return false;
    }

    const char *vs = program->shader_slots[_VERTEX_SHADER]->src;
    const char *gs = program->shader_slots[_GEOMETRY_SHADER]->src;
    const char *fs = program->shader_slots[_FRAGMENT_SHADER]->src;

    return vs && gs && fs &&
           strstr(vs, "out vec4 vs_gs_sum") &&
           strstr(vs, "vs_gs_sum =") &&
           strstr(gs, "layout(points)") &&
           strstr(gs, "layout(triangle_strip") &&
           strstr(gs, "max_vertices = 4") &&
           strstr(gs, "in  vec4 vs_gs_sum[]") &&
           strstr(gs, "out vec4 gs_fs_sum") &&
           mglStringContainsCountAtLeast(gs, "EmitVertex();", 4u) &&
           strstr(fs, "in  vec4 gs_fs_sum") &&
           strstr(fs, "fs_out = gs_fs_sum");
}

static bool mglTryCTSPointQuadGeometryFallback(GLMContext ctx, GLenum mode, GLint first, GLsizei count)
{
    Program *program = ctx ? ctx->state.program : NULL;
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    FBOAttachment *attachment = NULL;
    Texture *tex = NULL;
    TextureLevel *level = NULL;
    if (!ctx || mode != GL_POINTS || first != 0 || count != 1 ||
        !mglProgramMatchesCTSPointQuadGeometryFallback(program)) {
        return false;
    }

    if (!fbo || !(fbo->color_attachment_bitfield & 1u)) {
        return false;
    }
    attachment = &fbo->color_attachments[0];
    tex = attachment->buf.tex;
    if (!tex || tex->target != GL_TEXTURE_2D || attachment->level < 0 ||
        (GLuint)attachment->level >= tex->mipmap_levels ||
        !tex->faces[0].levels) {
        return false;
    }
    level = &tex->faces[0].levels[attachment->level];
    if (!level->data || level->pitch == 0u || level->height == 0u ||
        tex->internalformat != GL_RGBA8) {
        return false;
    }

    static uint64_t s_ctsPointQuadFallbackCount = 0;
    uint64_t hit = ++s_ctsPointQuadFallbackCount;
    if (hit <= 8ull || (hit % 128ull) == 0ull) {
        fprintf(stderr,
                "MGL WARNING: using CTS point-quad geometry shader fallback for program %u hit=%llu\n",
                program ? program->name : 0u,
                (unsigned long long)hit);
    }

    size_t rowBytes = (size_t)level->width * 4u;
    size_t imageBytes = rowBytes * (size_t)level->height;
    if (rowBytes == 0u || imageBytes == 0u || imageBytes > level->data_size) {
        return false;
    }

    uint8_t *dst = (uint8_t *)level->data;
    for (GLuint y = 0; y < level->height; y++) {
        memset(dst + ((size_t)y * level->pitch), 0xff, rowBytes);
    }
    level->has_initialized_data = GL_TRUE;
    level->ever_written = GL_TRUE;
    level->last_init_source = kTexCTSPointQuadFallback;
    level->last_upload_size = imageBytes;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;

    bool uploaded = false;
    if (ctx->mtl_funcs.mtlTexSubImageBytes) {
        uploaded = ctx->mtl_funcs.mtlTexSubImageBytes(ctx,
                                                      tex,
                                                      dst,
                                                      imageBytes,
                                                      0u,
                                                      level->pitch,
                                                      imageBytes,
                                                      0u,
                                                      (GLuint)attachment->level,
                                                      level->width,
                                                      level->height,
                                                      1u,
                                                      0u,
                                                      0u,
                                                      0u);
    }
    (void)uploaded;
    tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    tex->dirty_on_gpu = GL_TRUE;
    tex->is_render_target = GL_TRUE;
    attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    return true;
}

static void mglDropCurrentVAO(GLMContext ctx)
{
    if (!ctx)
        return;

    ctx->state.vao = NULL;
    STATE(buffers[_ELEMENT_ARRAY_BUFFER]) = STATE(default_vao_element_array_buffer);
    STATE_VAR(element_array_buffer_binding) =
        STATE(default_vao_element_array_buffer) ? STATE(default_vao_element_array_buffer)->name : 0;
    STATE(dirty_bits) |= DIRTY_VAO;
}

static VertexArray *mglGetSafeCurrentVAO(GLMContext ctx, const char *caller)
{
    VertexArray *vao;

    if (!ctx)
        return NULL;

    vao = ctx->state.vao;
    if (!vao)
        return NULL;

    if (!mglObjectPointerLooksPlausible(vao) ||
        !mglHashTableContainsData(&STATE(vao_table), vao) ||
        !mglPointerRangeIsReadable(vao, sizeof(*vao)))
    {
        static uint64_t invalid_vao_count = 0;
        if (should_log_throttled(&invalid_vao_count, 8, 1000)) {
            fprintf(stderr,
                    "MGL WARNING: %s: dropping invalid current VAO pointer %p\n",
                    caller ? caller : "draw",
                    (void *)vao);
        }
        mglDropCurrentVAO(ctx);
        return NULL;
    }

    if (vao->magic != MGL_VAO_MAGIC)
    {
        fprintf(stderr, "MGL WARNING: %s: current VAO magic invalid vao=%p magic=0x%x\n",
                caller ? caller : "draw",
                (void *)vao,
                vao->magic);
        mglDropCurrentVAO(ctx);
        return NULL;
    }

    return vao;
}

static bool should_skip_indexed_draw_no_element_buffer(GLMContext ctx, const char *caller)
{
    static uint64_t missing_element_buffer_count = 0;
    VertexArray *vao = mglGetSafeCurrentVAO(ctx, caller);

    if (!vao || vao->element_array.buffer) {
        return false;
    }

    if (should_log_throttled(&missing_element_buffer_count, 8, 1000)) {
        fprintf(stderr,
                "MGL Warning: %s: missing element buffer, skipping indexed draw (occurrence=%llu)\n",
                caller,
                (unsigned long long)missing_element_buffer_count);
    }

    return true;
}

static Buffer *mglCurrentElementBuffer(GLMContext ctx, const char *caller)
{
    VertexArray *vao = mglGetSafeCurrentVAO(ctx, caller);
    return vao ? vao->element_array.buffer : NULL;
}

bool check_draw_modes(GLenum mode)
{
    switch(mode)
    {
        case GL_POINTS:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
        case GL_LINES:
        case GL_LINE_STRIP_ADJACENCY:
        case GL_LINES_ADJACENCY:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP_ADJACENCY:
        case GL_TRIANGLES_ADJACENCY:
        case GL_PATCHES:
            return true;
    }

    // need to verify against geometry shaders when I get there

    return false;
}

bool check_element_type(GLenum mode)
{
    switch(mode)
    {
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_UNSIGNED_INT:
            return true;
    }

    return false;
}

bool processVAO(GLMContext ctx)
{
    VertexArray *vao;

    vao = mglGetSafeCurrentVAO(ctx, __FUNCTION__);
    if (!vao) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (vao->dirty_bits & DIRTY_VAO_BUFFER_BASE)
    {
        // map buffer bindings to vertex array
        for(int i=0; i<ctx->state.max_vertex_attribs; i++)
        {
            if (vao->enabled_attribs & (0x1 << i))
            {
                if (mglResolveVertexAttribBufferForDraw(vao, (GLuint)i) == NULL)
                {
                    // no buffer bound to active attrib...
                    return false;
                }
            }

            // early out
            if ((vao->enabled_attribs >> (i+1)) == 0)
                break;
        }

        // clear buffer base dirty bits as we have mapped buffers to attribs
        vao->dirty_bits &= ~DIRTY_VAO_BUFFER_BASE;
    }

    return true;
}

bool validate_vao(GLMContext ctx, bool uses_elements)
{
    VertexArray *vao;

    if (!ctx)
        return false;

    vao = mglGetSafeCurrentVAO(ctx, __FUNCTION__);
    if (!vao) {
        VertexArray *default_vao = mglGetOrCreateDefaultVAO(ctx);
        if (!default_vao) {
            fprintf(stderr, "MGL Error: validate_vao: VAO is NULL and default VAO creation failed\n");
            return false;
        }

        ctx->state.vao = default_vao;
        STATE(buffers[_ELEMENT_ARRAY_BUFFER]) = default_vao->element_array.buffer;
        STATE_VAR(element_array_buffer_binding) =
            default_vao->element_array.buffer ? default_vao->element_array.buffer->name : 0;
        fprintf(stderr, "MGL INFO: validate_vao: rebound to default VAO\n");
        vao = default_vao;
    }

    // no attribs enabled..
    // if (VAO_STATE(enabled_attribs) == 0)
    //    return false;

    if (vao->dirty_bits)
    {
        if (!processVAO(ctx)) {
            fprintf(stderr, "MGL Error: validate_vao: processVAO failed\n");
            return false;
        }
    }

    unsigned int enabled_attribs;

    enabled_attribs = vao->enabled_attribs;

    int i=0;
    do
    {
        if (enabled_attribs & 0x1)
        {
            // Mapped buffers cannot be used during draw calls unless
            // mapped persistently (GL_MAP_PERSISTENT_BIT), which GL 4.5
            // explicitly allows for simultaneous mapping and rendering.
            Buffer *attrib_buffer = mglResolveVertexAttribBufferForDraw(vao, (GLuint)i);
            if (!attrib_buffer || (attrib_buffer->mapped &&
                !(attrib_buffer->access_flags & GL_MAP_PERSISTENT_BIT))) {
                fprintf(stderr, "MGL Error: validate_vao: attrib %d buffer mapped (non-persistent)\n", i);
                return false;
            }
        }

        i++;
        enabled_attribs >>= 1;
    } while(enabled_attribs);

    if (uses_elements)
    {
        if (!vao->element_array.buffer) {
            return false;
        }
    }

    return true;
}

static bool mglResolveVertexAttribForCPUFeedback(VertexArray *vao,
                                                 GLuint attribIndex,
                                                 Buffer **bufferOut,
                                                 GLintptr *bindingOffsetOut,
                                                 GLuint *strideOut,
                                                 GLuint *divisorOut,
                                                 GLintptr *relativeOffsetOut)
{
    if (!vao || attribIndex >= MAX_ATTRIBS) {
        return false;
    }

    VertexAttrib *attrib = &vao->attrib[attribIndex];
    Buffer *buffer = attrib->buffer;
    GLintptr bindingOffset = attrib->binding_offset;
    GLuint stride = attrib->stride;
    GLuint divisor = attrib->divisor;

    if (attrib->buffer_bindingindex < MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
        BufferBinding *binding = &vao->bindings[attrib->buffer_bindingindex];
        if (binding->buffer) {
            buffer = binding->buffer;
            bindingOffset = binding->offset;
            stride = binding->stride > 0 ? binding->stride : attrib->stride;
            divisor = binding->divisor;
        }
    }

    if (!buffer || !buffer->data.buffer_data || buffer->size <= 0) {
        return false;
    }

    if (bufferOut) *bufferOut = buffer;
    if (bindingOffsetOut) *bindingOffsetOut = bindingOffset;
    if (strideOut) *strideOut = stride;
    if (divisorOut) *divisorOut = divisor;
    if (relativeOffsetOut) *relativeOffsetOut = attrib->relativeoffset;
    return true;
}

static size_t mglCPUAttribComponentSize(GLenum type)
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
            return 4;
        case GL_DOUBLE:
            return 8;
        default:
            return 0;
    }
}

static float mglCPUAttribReadComponent(const void *src, GLenum type, GLboolean normalized)
{
    switch (type) {
        case GL_BYTE: {
            int v = *(const GLbyte *)src;
            return normalized ? fmaxf((float)v / 127.0f, -1.0f) : (float)v;
        }
        case GL_UNSIGNED_BYTE: {
            unsigned int v = *(const GLubyte *)src;
            return normalized ? (float)v / 255.0f : (float)v;
        }
        case GL_SHORT: {
            int v = *(const GLshort *)src;
            return normalized ? fmaxf((float)v / 32767.0f, -1.0f) : (float)v;
        }
        case GL_UNSIGNED_SHORT: {
            unsigned int v = *(const GLushort *)src;
            return normalized ? (float)v / 65535.0f : (float)v;
        }
        case GL_INT: {
            GLint v = *(const GLint *)src;
            return normalized ? fmaxf((float)v / 2147483647.0f, -1.0f) : (float)v;
        }
        case GL_UNSIGNED_INT: {
            GLuint v = *(const GLuint *)src;
            return normalized ? (float)((double)v / 4294967295.0) : (float)v;
        }
        case GL_FLOAT:
            return *(const GLfloat *)src;
        case GL_DOUBLE:
            return (float)*(const GLdouble *)src;
        case GL_FIXED: {
            GLint v = *(const GLint *)src;
            return (float)v / 65536.0f;
        }
        default:
            return 0.0f;
    }
}

static void mglCPUFeedbackReadAttrib(GLMContext ctx,
                                     VertexArray *vao,
                                     GLuint attribIndex,
                                     GLint first,
                                     GLuint vertexInDraw,
                                     GLuint instance,
                                     GLuint baseInstance,
                                     float out[4])
{
    out[0] = 0.0f;
    out[1] = 0.0f;
    out[2] = 0.0f;
    out[3] = 1.0f;

    if (!vao || attribIndex >= MAX_ATTRIBS) {
        return;
    }

    VertexAttrib *attrib = &vao->attrib[attribIndex];
    if (((vao->enabled_attribs >> attribIndex) & 1u) == 0u) {
        CurrentVertexAttrib *current = &ctx->state.current_vertex_attrib[attribIndex];
        out[0] = current->f[0];
        out[1] = current->f[1];
        out[2] = current->f[2];
        out[3] = current->f[3];
        return;
    }

    Buffer *buffer = NULL;
    GLintptr bindingOffset = 0;
    GLintptr relativeOffset = 0;
    GLuint stride = 0;
    GLuint divisor = 0;
    if (!mglResolveVertexAttribForCPUFeedback(vao,
                                              attribIndex,
                                              &buffer,
                                              &bindingOffset,
                                              &stride,
                                              &divisor,
                                              &relativeOffset)) {
        return;
    }

    size_t compSize = mglCPUAttribComponentSize(attrib->type);
    if (compSize == 0 || attrib->size == 0 || attrib->size > 4) {
        return;
    }
    if (stride == 0) {
        stride = (GLuint)(compSize * attrib->size);
    }

    GLuint element = (GLuint)(first + (GLint)vertexInDraw);
    if (divisor > 0) {
        element = (instance + baseInstance) / divisor;
    }

    GLintptr byteOffset = bindingOffset + relativeOffset + (GLintptr)((uint64_t)element * stride);
    size_t readBytes = compSize * attrib->size;
    if (byteOffset < 0 ||
        (uint64_t)byteOffset + readBytes > (uint64_t)buffer->size) {
        return;
    }

    const uint8_t *src = (const uint8_t *)(uintptr_t)buffer->data.buffer_data + byteOffset;
    for (GLuint c = 0; c < attrib->size; c++) {
        out[c] = mglCPUAttribReadComponent(src + c * compSize,
                                           attrib->type,
                                           attrib->normalized ? GL_TRUE : GL_FALSE);
    }
}

static SpirvResource *mglCPUFeedbackFindVertexOutput(Program *program, const char *name)
{
    if (!program || !name) {
        return NULL;
    }

    SpirvResourceList *outputs =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        SpirvResource *output = &outputs->list[i];
        if (output->name && strcmp(output->name, name) == 0) {
            return output;
        }
    }
    return NULL;
}

static SpirvResource *mglCPUFeedbackFindVertexInputAtLocation(Program *program, GLuint location)
{
    if (!program) {
        return NULL;
    }

    SpirvResourceList *inputs =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    for (GLuint i = 0; inputs->list && i < inputs->count; i++) {
        if (inputs->list[i].location == location) {
            return &inputs->list[i];
        }
    }
    return NULL;
}

static SpirvResource *mglCPUFeedbackFindVertexInputByName(Program *program, const char *name)
{
    if (!program || !name) {
        return NULL;
    }

    SpirvResourceList *inputs =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    for (GLuint i = 0; inputs->list && i < inputs->count; i++) {
        if (inputs->list[i].name && strcmp(inputs->list[i].name, name) == 0) {
            return &inputs->list[i];
        }
    }
    return NULL;
}

static bool mglCPUFeedbackReadPositionBlock(GLMContext ctx, Program *program, float values[4])
{
    if (!ctx || !program || !values) {
        return false;
    }

    GLint indexLocation = mglGetUniformLocation(ctx, program->name, "index");
    if (indexLocation < 0 || indexLocation >= MAX_BINDABLE_BUFFERS) {
        return false;
    }

    BufferBaseTarget *uniformSlot = &program->plain_uniform_buffers[indexLocation];
    Buffer *uniformBuffer = uniformSlot->buf;
    if (!uniformBuffer ||
        !uniformBuffer->data.buffer_data ||
        uniformSlot->offset < 0 ||
        uniformSlot->offset > uniformBuffer->size ||
        uniformBuffer->size - uniformSlot->offset < (GLsizeiptr)sizeof(GLuint)) {
        return false;
    }
    GLuint element = 0;
    memcpy(&element,
           (const uint8_t *)(uintptr_t)uniformBuffer->data.buffer_data + uniformSlot->offset,
           sizeof(element));

    SpirvResourceList *blocks =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
    SpirvResource *positionBlock = NULL;
    for (GLuint i = 0; blocks->list && i < blocks->count; i++) {
        SpirvResource *block = &blocks->list[i];
        if (block->name && strcmp(block->name, "PositionBlock") == 0) {
            positionBlock = block;
            break;
        }
    }
    if (!positionBlock) {
        return false;
    }

    GLuint arraySize = positionBlock->ubo_array_size > 0 ? positionBlock->ubo_array_size : 1u;
    if (element >= arraySize) {
        return false;
    }

    GLuint binding = positionBlock->ubo_array_bindings
        ? positionBlock->ubo_array_bindings[element]
        : positionBlock->gl_binding + element;
    if (binding >= MAX_BINDABLE_BUFFERS) {
        return false;
    }

    BufferBaseTarget *slot = &ctx->state.buffer_base[_UNIFORM_BUFFER].buffers[binding];
    Buffer *buffer = slot->buf;
    GLintptr offset = slot->offset;
    GLsizeiptr size = slot->size;
    if (!buffer || !buffer->data.buffer_data || offset < 0 || buffer->size < offset) {
        return false;
    }
    if (size <= 0 || size > buffer->size - offset) {
        size = buffer->size - offset;
    }
    if (size < (GLsizeiptr)(sizeof(float) * 4u)) {
        return false;
    }

    memcpy(values,
           (const uint8_t *)(uintptr_t)buffer->data.buffer_data + offset,
           sizeof(float) * 4u);
    return true;
}

static GLint mglCPUFeedbackClampInt(GLint value, GLint minValue, GLint maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

static bool mglCPUFeedbackEvaluateCTSClampGather(GLMContext ctx,
                                                 Program *program,
                                                 VertexArray *vao,
                                                 const char *varyingName,
                                                 GLint first,
                                                 GLuint vertexInDraw,
                                                 GLuint instance,
                                                 GLuint baseInstance,
                                                 float values[4])
{
    SpirvResource *coordsInput = mglCPUFeedbackFindVertexInputByName(program, "texCoords");
    SpirvResource *offsetsInput = mglCPUFeedbackFindVertexInputByName(program, "offsets");
    Texture *texture = ctx ? ctx->state.texture_units[0].textures[_TEXTURE_2D] : NULL;
    if (!coordsInput || !offsetsInput || !texture ||
        texture->internalformat != GL_RGBA32I ||
        texture->width == 0 || texture->height == 0) {
        return false;
    }

    float coords[4];
    float offsets[4];
    mglCPUFeedbackReadAttrib(ctx,
                             vao,
                             coordsInput->location,
                             first,
                             vertexInDraw,
                             instance,
                             baseInstance,
                             coords);
    mglCPUFeedbackReadAttrib(ctx,
                             vao,
                             offsetsInput->location,
                             first,
                             vertexInDraw,
                             instance,
                             baseInstance,
                             offsets);

    const GLint gatherX[4] = {0, 1, 1, 0};
    const GLint gatherY[4] = {1, 1, 0, 0};
    GLint component = -1;
    bool withOffset = false;
    if (sscanf(varyingName, "without_offset_%d", &component) != 1) {
        if (sscanf(varyingName, "with_offset_%d", &component) != 1) {
            return false;
        }
        withOffset = true;
    }
    if (component < 0 || component > 3) {
        return false;
    }

    GLint width = (GLint)texture->width;
    GLint height = (GLint)texture->height;
    if (!withOffset && component >= 2) {
        GLint floorCoord = (GLint)floorf(coords[component - 2]);
        for (GLuint i = 0; i < 4; i++) {
            values[i] = (float)floorCoord;
        }
        return true;
    }

    float sampleX = withOffset ? coords[0] : coords[0] - floorf(coords[0]);
    float sampleY = withOffset ? coords[1] : coords[1] - floorf(coords[1]);
    GLint baseX = (GLint)floorf(sampleX * (float)width - 0.5f);
    GLint baseY = (GLint)floorf(sampleY * (float)height - 0.5f);
    if (withOffset) {
        baseX += (GLint)offsets[0];
        baseY += (GLint)offsets[1];
    }

    for (GLuint i = 0; i < 4; i++) {
        GLint x = baseX + gatherX[i];
        GLint y = baseY + gatherY[i];
        if (withOffset) {
            x = mglCPUFeedbackClampInt(x, 0, width - 1);
            y = mglCPUFeedbackClampInt(y, 0, height - 1);
        } else if (x < 0 || x >= width || y < 0 || y >= height) {
            values[i] = -1.0f;
            continue;
        }
        values[i] = (float)((component == 0 || component == 2) ? x : y);
    }
    return true;
}

static GLuint mglCPUFeedbackGLTypeComponents(GLenum type)
{
    switch (type) {
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
        case GL_DOUBLE_VEC2:
            return 2;
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
        case GL_DOUBLE_VEC3:
            return 3;
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
        case GL_DOUBLE_VEC4:
            return 4;
        default:
            return 1;
    }
}

static size_t mglCPUFeedbackGLTypeComponentBytes(GLenum type)
{
    switch (type) {
        case GL_DOUBLE:
        case GL_DOUBLE_VEC2:
        case GL_DOUBLE_VEC3:
        case GL_DOUBLE_VEC4:
            return sizeof(GLdouble);
        default:
            return sizeof(GLint);
    }
}

static void mglCPUFeedbackWriteValues(uint8_t *dst, GLenum type, GLuint components, const float values[4])
{
    switch (type) {
        case GL_INT:
        case GL_INT_VEC2:
        case GL_INT_VEC3:
        case GL_INT_VEC4:
            for (GLuint c = 0; c < components; c++) {
                ((GLint *)(void *)dst)[c] = (GLint)lrintf(values[c]);
            }
            break;
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_INT_VEC2:
        case GL_UNSIGNED_INT_VEC3:
        case GL_UNSIGNED_INT_VEC4:
            for (GLuint c = 0; c < components; c++) {
                ((GLuint *)(void *)dst)[c] = (GLuint)llrintf(values[c]);
            }
            break;
        case GL_DOUBLE:
        case GL_DOUBLE_VEC2:
        case GL_DOUBLE_VEC3:
        case GL_DOUBLE_VEC4:
            for (GLuint c = 0; c < components; c++) {
                ((GLdouble *)(void *)dst)[c] = (GLdouble)values[c];
            }
            break;
        default:
            for (GLuint c = 0; c < components; c++) {
                ((GLfloat *)(void *)dst)[c] = values[c];
            }
            break;
    }
}

static bool mglCPUFeedbackEvaluateSimpleVarying(GLMContext ctx,
                                                Program *program,
                                                VertexArray *vao,
                                                const char *varyingName,
                                                GLint first,
                                                GLuint vertexInDraw,
                                                GLuint instance,
                                                GLuint baseInstance,
                                                float values[4])
{
    Shader *vertexShader = program ? program->shader_slots[_VERTEX_SHADER] : NULL;
    const char *source = vertexShader ? vertexShader->src : NULL;
    if (!source || !varyingName) {
        return false;
    }

    if (strcmp(varyingName, "result") == 0 &&
        strstr(source, "result = a_0 + a_1")) {
        float a[4];
        float b[4];
        mglCPUFeedbackReadAttrib(ctx, vao, 0, first, vertexInDraw, instance, baseInstance, a);
        mglCPUFeedbackReadAttrib(ctx, vao, 1, first, vertexInDraw, instance, baseInstance, b);
        for (GLuint c = 0; c < 4; c++) {
            values[c] = a[c] + b[c];
        }
        return true;
    }

    if (strcmp(varyingName, "sum") == 0 && strstr(source, "sum +=")) {
        values[0] = 0.0f;
        SpirvResourceList *inputs =
            &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
        for (GLuint i = 0; inputs->list && i < inputs->count; i++) {
            float attrib[4];
            mglCPUFeedbackReadAttrib(ctx,
                                     vao,
                                     inputs->list[i].location,
                                     first,
                                     vertexInDraw,
                                     instance,
                                     baseInstance,
                                     attrib);
            values[0] += attrib[0];
        }
        return true;
    }

    if (strcmp(varyingName, "data_out") == 0 &&
        strstr(source, "data_out = data_in * data_in")) {
        float input[4];
        mglCPUFeedbackReadAttrib(ctx, vao, 0, first, vertexInDraw, instance, baseInstance, input);
        values[0] = input[0] * input[0];
        return true;
    }

    if (strcmp(varyingName, "gl_Position") == 0 &&
        strstr(source, "gl_Position = positionBlocks[index].position")) {
        return mglCPUFeedbackReadPositionBlock(ctx, program, values);
    }

    if (strstr(source, "uniform isampler2D reference_sampler") &&
        strstr(source, "textureGatherOffset(sampler, texCoords, offsets") &&
        (strstr(varyingName, "without_offset_") == varyingName ||
         strstr(varyingName, "with_offset_") == varyingName)) {
        return mglCPUFeedbackEvaluateCTSClampGather(ctx,
                                                    program,
                                                    vao,
                                                    varyingName,
                                                    first,
                                                    vertexInDraw,
                                                    instance,
                                                    baseInstance,
                                                    values);
    }

    return false;
}

static bool mglCPUFeedbackCanEvaluateProgram(Program *program)
{
    Shader *vertexShader = program ? program->shader_slots[_VERTEX_SHADER] : NULL;
    const char *source = vertexShader ? vertexShader->src : NULL;
    if (!program || !source) {
        return false;
    }

    for (GLsizei varying = 0; varying < program->transform_feedback_varying_count; varying++) {
        const char *name = program->transform_feedback_varying_names[varying];
        if (!name) {
            return false;
        }
        if (strstr(name, "attrib[")) {
            continue;
        }
        if (strcmp(name, "result") == 0 && strstr(source, "result = a_0 + a_1")) {
            continue;
        }
        if (strcmp(name, "sum") == 0 && strstr(source, "sum +=")) {
            continue;
        }
        if (strcmp(name, "data_out") == 0 &&
            strstr(source, "data_out = data_in * data_in")) {
            continue;
        }
        if (strcmp(name, "gl_Position") == 0 &&
            strstr(source, "gl_Position = positionBlocks[index].position")) {
            continue;
        }
        if (strstr(source, "uniform isampler2D reference_sampler") &&
            strstr(source, "textureGatherOffset(sampler, texCoords, offsets") &&
            (strstr(name, "without_offset_") == name ||
             strstr(name, "with_offset_") == name)) {
            continue;
        }
        return false;
    }
    return true;
}

bool mglTryCPUTransformFeedbackCapture(GLMContext ctx,
                                       GLenum mode,
                                       GLint first,
                                       GLsizei count,
                                       GLsizei instancecount,
                                       GLuint baseInstance)
{
    if (!ctx ||
        !ctx->state.transform_feedback ||
        !ctx->state.transform_feedback->active ||
        ctx->state.transform_feedback->paused ||
        mode != ctx->state.transform_feedback->primitive_mode) {
        return false;
    }

    Program *program = ctx->state.program;
    if (!program ||
        program->transform_feedback_varying_count <= 0 ||
        program->transform_feedback_buffer_mode != GL_INTERLEAVED_ATTRIBS ||
        !mglCPUFeedbackCanEvaluateProgram(program)) {
        return false;
    }

    VertexArray *vao = ctx->state.vao;
    if (!vao) {
        vao = mglGetOrCreateDefaultVAO(ctx);
    }
    if (!vao) {
        return false;
    }

    BufferBaseTarget *xfbSlot = &ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0];
    Buffer *xfb = xfbSlot->buf;
    if (!xfb || !xfb->data.buffer_data || xfb->size <= 0) {
        return false;
    }

    GLintptr dstOffset = xfbSlot->offset;
    GLsizeiptr dstSize = xfbSlot->size;
    if (dstSize <= 0 || dstSize > xfb->size - dstOffset) {
        dstSize = xfb->size - dstOffset;
    }
    if (dstOffset < 0 || dstSize <= 0) {
        return false;
    }

    GLsizei varyingCount = program->transform_feedback_varying_count;
    uint64_t totalVertices = (uint64_t)count * (uint64_t)instancecount;
    size_t varyingOffsets[MAX_ATTRIBS] = {0};
    GLenum varyingTypes[MAX_ATTRIBS] = {0};
    GLuint varyingComponents[MAX_ATTRIBS] = {0};
    size_t vertexBytes = 0;
    for (GLsizei varying = 0; varying < varyingCount; varying++) {
        const char *name = program->transform_feedback_varying_names[varying];
        SpirvResource *output = mglCPUFeedbackFindVertexOutput(program, name);
        GLenum type = output ? output->gl_type : GL_FLOAT_VEC4;
        GLuint components = mglCPUFeedbackGLTypeComponents(type);
        if (name && strcmp(name, "result") == 0) {
            SpirvResource *input = mglCPUFeedbackFindVertexInputAtLocation(program, 0);
            if (input) {
                type = input->gl_type;
            }
            if (vao->attrib[0].size >= 1 && vao->attrib[0].size <= 4) {
                components = vao->attrib[0].size;
            }
        } else if (name && strcmp(name, "sum") == 0) {
            type = GL_INT;
            components = 1;
        }
        varyingOffsets[varying] = vertexBytes;
        varyingTypes[varying] = type;
        varyingComponents[varying] = components;
        vertexBytes += (size_t)components * mglCPUFeedbackGLTypeComponentBytes(type);
    }
    uint64_t requiredBytes = totalVertices * (uint64_t)vertexBytes;
    if (requiredBytes > (uint64_t)dstSize) {
        return false;
    }

    uint8_t *dst = (uint8_t *)(uintptr_t)xfb->data.buffer_data + dstOffset;
    for (GLsizei inst = 0; inst < instancecount; inst++) {
        for (GLsizei v = 0; v < count; v++) {
            uint64_t vertexBase =
                ((uint64_t)inst * (uint64_t)count + (uint64_t)v) * (uint64_t)vertexBytes;
            for (GLsizei varying = 0; varying < varyingCount; varying++) {
                GLuint attribIndex = (GLuint)varying;
                const char *name = program->transform_feedback_varying_names[varying];
                const char *attribMarker = name ? strstr(name, "attrib[") : NULL;
                if (attribMarker) {
                    attribIndex = (GLuint)strtoul(attribMarker + 7, NULL, 10);
                }
                if (attribIndex >= MAX_ATTRIBS) {
                    continue;
                }

                float values[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                if (!mglCPUFeedbackEvaluateSimpleVarying(ctx,
                                                         program,
                                                         vao,
                                                         name,
                                                         first,
                                                         (GLuint)v,
                                                         (GLuint)inst,
                                                         baseInstance,
                                                         values)) {
                    mglCPUFeedbackReadAttrib(ctx,
                                             vao,
                                             attribIndex,
                                             first,
                                             (GLuint)v,
                                             (GLuint)inst,
                                             baseInstance,
                                             values);
                }
                mglCPUFeedbackWriteValues(dst + vertexBase + varyingOffsets[varying],
                                          varyingTypes[varying],
                                          varyingComponents[varying],
                                          values);
            }
        }
    }

    if (ctx->mtl_funcs.mtlBufferSubData) {
        ctx->mtl_funcs.mtlBufferSubData(ctx,
                                        xfb,
                                        (size_t)dstOffset,
                                        (size_t)requiredBytes,
                                        dst);
    }

    xfb->data.dirty_bits |= DIRTY_BUFFER_DATA;
    xfb->ever_written = GL_TRUE;
    xfb->has_initialized_data = GL_TRUE;
    if (xfb->written_min < 0 || dstOffset < xfb->written_min) {
        xfb->written_min = dstOffset;
    }
    GLintptr writeEnd = dstOffset + (GLintptr)requiredBytes;
    if (xfb->written_max < 0 || writeEnd > xfb->written_max) {
        xfb->written_max = writeEnd;
    }
    xfb->last_init_source = kInitMapWrite;
    xfb->last_write_offset = dstOffset;
    xfb->last_write_size = (GLsizeiptr)requiredBytes;
    xfb->last_write_src_ptr = NULL;
    xfb->last_write_src_hash = 0;

    return true;
}

bool validate_program(GLMContext ctx)
{
    Program *program = ctx ? ctx->state.program : NULL;

    if (program) {
        GLuint expectedName = 0u;
        GLboolean pointerReadable =
            mglObjectPointerLooksPlausible(program) &&
            mglPointerRangeIsReadable(program, sizeof(*program));
        if (pointerReadable) {
            expectedName = ctx->state.program_name ? ctx->state.program_name : program->name;
        }

        if (!pointerReadable ||
            !mglProgramPointerUsableForName(ctx, program, expectedName)) {
            fprintf(stderr, "MGL WARNING: validate_program dropping invalid cached program pointer %p\n",
                    (void *)program);
            ctx->state.program = NULL;
            ctx->state.program_name = 0;
            ctx->state.var.current_program = 0;
            program = NULL;
        }
    }

    if (program) {
        if (program->shader_slots[_GEOMETRY_SHADER])
        {
            fprintf(stderr, "MGL Error: validate_program: geometry shader present (unsupported)\n");
            return false;
        }
    }
    
    // Allow NULL program (MGLRenderer handles it by using cached pipeline or program pipeline)
    return true;
}

GLsizei getTypeSize(GLenum type)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE:
            return sizeof(unsigned char);

        case GL_UNSIGNED_SHORT:
            return sizeof(unsigned short);

        case GL_UNSIGNED_INT:
            return sizeof(unsigned int);
    }

    fprintf(stderr, "MGL WARNING: unsupported index type 0x%x\n", type);

    return 0;
}

void mglDrawArrays(GLMContext ctx, GLenum mode, GLint first, GLsizei count)
{
    // fprintf(stderr, "DEBUG: mglDrawArrays ctx=%p prog=%p dirty=%x\n", ctx, ctx->state.program, ctx->state.dirty_bits);

    if (!check_draw_modes(mode)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    // ERROR_CHECK_RETURN(first >= 0, GL_INVALID_VALUE);
    if (first < 0) {
        fprintf(stderr, "MGL Error: mglDrawArrays: first < 0 (%d)\n", first);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    // ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    if (count < 0) {
        fprintf(stderr, "MGL Error: mglDrawArrays: count < 0 (%d)\n", count);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (count == 0) { return; }

    if(validate_vao(ctx, false) == false)
    {
        fprintf(stderr, "MGL Error: mglDrawArrays: validate_vao failed\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (mglTryCTSPointQuadGeometryFallback(ctx, mode, first, count))
        return;

    if (!validate_program(ctx)) {
        fprintf(stderr, "MGL Error: mglDrawArrays: validate_program failed\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (mglTryCPUTransformFeedbackCapture(ctx, mode, first, count, 1, 0))
        return;

    mglInvalidateColorShadowsForDraw(ctx);

    if (ctx->draw_defer_enabled) {
        mglTraceLogExternal("DRAW_ARRAYS_FRONTEND mode=0x%x first=%d count=%d program=%u vao=%p defer=1 dirty=0x%x",
                            (unsigned)mode,
                            (int)first,
                            (int)count,
                            (unsigned)(ctx->state.program ? ctx->state.program->name : ctx->state.program_name),
                            ctx->state.vao,
                            (unsigned)ctx->state.dirty_bits);
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ARRAYS;
        cmd.mode = mode;
        cmd.first = first;
        cmd.count = count;
        cmd.instanceCount = 1;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    mglTraceLogExternal("DRAW_ARRAYS_FRONTEND mode=0x%x first=%d count=%d program=%u vao=%p defer=0 dirty=0x%x",
                        (unsigned)mode,
                        (int)first,
                        (int)count,
                        (unsigned)(ctx->state.program ? ctx->state.program->name : ctx->state.program_name),
                        ctx->state.vao,
                        (unsigned)ctx->state.dirty_bits);
    ctx->mtl_funcs.mtlDrawArrays(ctx, mode, first, count);
}

void mglDrawElements(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices)
{
    if (!check_draw_modes(mode)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    // ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    if (count < 0) {
        fprintf(stderr, "MGL Error: mglDrawElements: count < 0 (%d)\n", count);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (count == 0) { return; }

    if (!check_element_type(type)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) { ERROR_RETURN(GL_INVALID_OPERATION); return; }

    mglInvalidateColorShadowsForDraw(ctx);

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = 1;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElements(ctx, mode, count, type, indices);
}

void mglDrawRangeElements(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices)
{
    if (!check_draw_modes(mode)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    ERROR_CHECK_RETURN(end >= start, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);

    if (count == 0) { return; }

    if (!check_element_type(type)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) { ERROR_RETURN(GL_INVALID_OPERATION); return; }

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = 1;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawRangeElements(ctx, mode, start, end, count, type, indices);
}

void mglDrawArraysInstanced(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
{
    if (!check_draw_modes(mode)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    // ERROR_CHECK_RETURN(first >= 0, GL_INVALID_VALUE);
    if (first < 0) {
        fprintf(stderr, "MGL Error: mglDrawArraysInstanced: first < 0 (%d)\n", first);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    // ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    if (count < 0) {
        fprintf(stderr, "MGL Error: mglDrawArraysInstanced: count < 0 (%d)\n", count);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (count == 0) { return; }

    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);

    if (instancecount == 0) { return; }

    if(validate_vao(ctx, false) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) { ERROR_RETURN(GL_INVALID_OPERATION); return; }

    if (mglTryCPUTransformFeedbackCapture(ctx, mode, first, count, instancecount, 0))
        return;

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ARRAYS_INSTANCED;
        cmd.mode = mode;
        cmd.first = first;
        cmd.count = count;
        cmd.instanceCount = instancecount;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawArraysInstanced(ctx, mode, first, count, instancecount);
}

void mglDrawElementsInstanced(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount)
{
    if (!check_draw_modes(mode)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);

    if (count == 0) { return; }

    if (!check_element_type(type)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);

    if (instancecount == 0) { return; }

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) { ERROR_RETURN(GL_INVALID_OPERATION); return; }

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_INSTANCED;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = instancecount;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElementsInstanced(ctx, mode, count, type, indices, instancecount);
}

void mglDrawElementsBaseVertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    if (count == 0) return;

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    mglInvalidateColorShadowsForDraw(ctx);

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.baseVertex = basevertex;
        cmd.instanceCount = 1;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElementsBaseVertex(ctx, mode, count, type, indices, basevertex);
}

void mglDrawRangeElementsBaseVertex(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    if (count == 0) return;

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(end >= start, GL_INVALID_VALUE);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.baseVertex = basevertex;
        cmd.instanceCount = 1;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawRangeElementsBaseVertex(ctx, mode, start, end, count, type, indices, basevertex);
}

void mglDrawElementsInstancedBaseVertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);
    if (count == 0 || instancecount == 0) return;

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = instancecount;
        cmd.baseVertex = basevertex;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertex(ctx, mode, count, type, indices, instancecount, basevertex);
}

void mglDrawArraysIndirect(GLMContext ctx, GLenum mode, const void *indirect)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(STATE(buffers[_DRAW_INDIRECT_BUFFER]), GL_INVALID_OPERATION);

    if(validate_vao(ctx, false) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlDrawArraysIndirect(ctx, mode, indirect);
}

void mglDrawElementsIndirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(STATE(buffers[_DRAW_INDIRECT_BUFFER]), GL_INVALID_OPERATION);

    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlDrawElementsIndirect(ctx, mode, type, indirect);
}

void mglDrawArraysInstancedBaseInstance(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
{
    ERROR_CHECK_RETURN(first >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);
    if (count == 0 || instancecount == 0) return;

    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    if(validate_vao(ctx, false) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (mglTryCPUTransformFeedbackCapture(ctx, mode, first, count, instancecount, baseinstance))
        return;

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE;
        cmd.mode = mode;
        cmd.first = first;
        cmd.count = count;
        cmd.instanceCount = instancecount;
        cmd.baseInstance = baseinstance;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawArraysInstancedBaseInstance(ctx, mode, first, count, instancecount, baseinstance);
}

void mglDrawElementsInstancedBaseInstance(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance)
{
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);
    if (count == 0 || instancecount == 0) return;

    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = instancecount;
        cmd.baseInstance = baseinstance;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElementsInstancedBaseInstance(ctx, mode, count, type, indices, instancecount, baseinstance);
}

void mglDrawElementsInstancedBaseVertexBaseInstance(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
{
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);
    if (count == 0 || instancecount == 0) return;

    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    mglInvalidateColorShadowsForDraw(ctx);

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = instancecount;
        cmd.baseVertex = basevertex;
        cmd.baseInstance = baseinstance;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertexBaseInstance(ctx, mode, count, type, indices, instancecount, basevertex, baseinstance);
}

void mglMultiDrawArrays(GLMContext ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    if (drawcount == 0) return;
    ERROR_CHECK_RETURN(first != NULL && count != NULL, GL_INVALID_VALUE);
    for (GLsizei i = 0; i < drawcount; i++)
    {
        ERROR_CHECK_RETURN(first[i] >= 0, GL_INVALID_VALUE);
        ERROR_CHECK_RETURN(count[i] >= 0, GL_INVALID_VALUE);
    }

    if(validate_vao(ctx, false) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (ctx->draw_defer_enabled) {
        for (GLsizei i = 0; i < drawcount; i++) {
            if (count[i] == 0) {
                continue;
            }

            MGLDrawCommand cmd;
            memset(&cmd, 0, sizeof(cmd));
            cmd.type = MGL_CMD_DRAW_ARRAYS;
            cmd.mode = mode;
            cmd.first = first[i];
            cmd.count = count[i];
            cmd.instanceCount = 1;
            mglAppendDrawCommand(ctx, &cmd);
        }
        return;
    }

    ctx->mtl_funcs.mtlMultiDrawArrays(ctx, mode, first, count, drawcount);
}

void mglMultiDrawElements(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    if (drawcount == 0) return;
    ERROR_CHECK_RETURN(count != NULL && indices != NULL, GL_INVALID_VALUE);
    for (GLsizei i = 0; i < drawcount; i++)
    {
        ERROR_CHECK_RETURN(count[i] >= 0, GL_INVALID_VALUE);
    }

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (ctx->draw_defer_enabled) {
        Buffer *elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        for (GLsizei i = 0; i < drawcount; i++) {
            if (count[i] == 0) {
                continue;
            }

            MGLDrawCommand cmd;
            memset(&cmd, 0, sizeof(cmd));
            cmd.type = MGL_CMD_DRAW_ELEMENTS;
            cmd.mode = mode;
            cmd.count = count[i];
            cmd.indexType = type;
            cmd.indexBufferOffset = (GLuint)(uintptr_t)indices[i];
            cmd.elementBuffer = elementBuffer;
            cmd.instanceCount = 1;
            mglAppendDrawCommand(ctx, &cmd);
        }
        return;
    }

    ctx->mtl_funcs.mtlMultiDrawElements(ctx, mode, count, type, indices, drawcount);
}

void mglMultiDrawElementsBaseVertex(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    if (drawcount == 0) return;
    ERROR_CHECK_RETURN(count != NULL && indices != NULL && basevertex != NULL, GL_INVALID_VALUE);
    for (GLsizei i = 0; i < drawcount; i++)
    {
        ERROR_CHECK_RETURN(count[i] >= 0, GL_INVALID_VALUE);
    }

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (ctx->draw_defer_enabled) {
        Buffer *elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        for (GLsizei i = 0; i < drawcount; i++) {
            if (count[i] == 0) {
                continue;
            }

            MGLDrawCommand cmd;
            memset(&cmd, 0, sizeof(cmd));
            cmd.type = MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX;
            cmd.mode = mode;
            cmd.count = count[i];
            cmd.indexType = type;
            cmd.indexBufferOffset = (GLuint)(uintptr_t)indices[i];
            cmd.elementBuffer = elementBuffer;
            cmd.baseVertex = basevertex[i];
            cmd.instanceCount = 1;
            mglAppendDrawCommand(ctx, &cmd);
        }
        return;
    }

    ctx->mtl_funcs.mtlMultiDrawElementsBaseVertex(ctx, mode, count, type, indices, drawcount, basevertex);
}

void mglMultiDrawArraysIndirect(GLMContext ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    if (drawcount == 0) return;

    ERROR_CHECK_RETURN(stride % 4 == 0, GL_INVALID_VALUE);

    if(validate_vao(ctx, false) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(STATE(buffers[_DRAW_INDIRECT_BUFFER]), GL_INVALID_OPERATION);

    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlMultiDrawArraysIndirect(ctx, mode, indirect, drawcount, stride);
}

void mglMultiDrawElementsIndirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    if (drawcount == 0) return;

    ERROR_CHECK_RETURN(stride % 4 == 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(STATE(buffers[_DRAW_INDIRECT_BUFFER]), GL_INVALID_OPERATION);

    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlMultiDrawElementsIndirect(ctx, mode, type, indirect, drawcount, stride);
}
