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
#include "mgl_program_reflection.h"

extern void mglInvalidateColorShadowsForDraw(GLMContext ctx);
#include "mgl_trace_log.h"

static bool mglSkipOrRecordConditionalDraw(GLMContext ctx)
{
    if (mglShouldSkipConditionalRender(ctx))
        return true;
    mglRecordActiveSampleQueryDraw(ctx);
    return false;
}

static GLuint mglTraceDrawProgram(GLMContext ctx)
{
    return ctx ? ctx->state.program_name : 0u;
}

static bool mglValidateDrawIndirectCommands(GLMContext ctx,
                                            const char *label,
                                            const void *indirect,
                                            GLsizei drawcount,
                                            GLsizei stride,
                                            GLsizeiptr commandSize)
{
    Buffer *buf = ctx ? STATE(buffers[_DRAW_INDIRECT_BUFFER]) : NULL;
    const char *caller = label ? label : "DRAW_INDIRECT";
    intptr_t baseOffset = (intptr_t)indirect;
    GLsizeiptr commandStride;
    GLsizeiptr lastOffset;

    if (!buf) {
        mglTraceLogExternal("%s_SKIP reason=no_draw_indirect_buffer program=%u",
                            caller,
                            (unsigned)mglTraceDrawProgram(ctx));
        mglDispatchError(ctx, caller, GL_INVALID_OPERATION);
        return false;
    }
    if (buf->mapped && !(buf->access_flags & GL_MAP_PERSISTENT_BIT)) {
        mglTraceLogExternal("%s_SKIP reason=draw_indirect_buffer_mapped buffer=%u program=%u",
                            caller,
                            (unsigned)buf->name,
                            (unsigned)mglTraceDrawProgram(ctx));
        mglDispatchError(ctx, caller, GL_INVALID_OPERATION);
        return false;
    }
    if (baseOffset < 0 || (baseOffset & 3) != 0) {
        mglTraceLogExternal("%s_SKIP reason=bad_indirect_offset offset=%lld program=%u",
                            caller,
                            (long long)baseOffset,
                            (unsigned)mglTraceDrawProgram(ctx));
        mglDispatchError(ctx, caller, GL_INVALID_VALUE);
        return false;
    }
    if (drawcount < 0 || stride < 0 || (stride & 3) != 0) {
        mglTraceLogExternal("%s_SKIP reason=bad_drawcount_or_stride drawcount=%d stride=%d program=%u",
                            caller,
                            (int)drawcount,
                            (int)stride,
                            (unsigned)mglTraceDrawProgram(ctx));
        mglDispatchError(ctx, caller, GL_INVALID_VALUE);
        return false;
    }
    if (drawcount == 0) {
        return true;
    }
    if (commandSize <= 0 || buf->size < 0 || baseOffset > buf->size) {
        mglTraceLogExternal("%s_SKIP reason=bad_indirect_range offset=%lld size=%lld commandSize=%lld program=%u",
                            caller,
                            (long long)baseOffset,
                            (long long)(buf ? buf->size : -1),
                            (long long)commandSize,
                            (unsigned)mglTraceDrawProgram(ctx));
        mglDispatchError(ctx, caller, GL_INVALID_OPERATION);
        return false;
    }

    commandStride = stride ? (GLsizeiptr)stride : commandSize;
    if (drawcount > 1 &&
        commandStride > 0 &&
        (GLsizeiptr)(drawcount - 1) > (GLsizeiptr)((INTPTR_MAX - baseOffset) / commandStride)) {
        mglTraceLogExternal("%s_SKIP reason=indirect_offset_overflow offset=%lld drawcount=%d stride=%lld program=%u",
                            caller,
                            (long long)baseOffset,
                            (int)drawcount,
                            (long long)commandStride,
                            (unsigned)mglTraceDrawProgram(ctx));
        mglDispatchError(ctx, caller, GL_INVALID_OPERATION);
        return false;
    }

    lastOffset = (GLsizeiptr)baseOffset + (GLsizeiptr)(drawcount - 1) * commandStride;
    if (lastOffset > buf->size || commandSize > buf->size - lastOffset) {
        mglTraceLogExternal("%s_SKIP reason=indirect_range_oob buffer=%u offset=%lld drawcount=%d stride=%lld commandSize=%lld size=%lld program=%u",
                            caller,
                            (unsigned)buf->name,
                            (long long)baseOffset,
                            (int)drawcount,
                            (long long)commandStride,
                            (long long)commandSize,
                            (long long)buf->size,
                            (unsigned)mglTraceDrawProgram(ctx));
        mglDispatchError(ctx, caller, GL_INVALID_OPERATION);
        return false;
    }

    return true;
}

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

static void mglDropCurrentVAO(GLMContext ctx)
{
    if (!ctx)
        return;

    ctx->state.vao = NULL;
    STATE(buffers[_ELEMENT_ARRAY_BUFFER]) = STATE(default_vao_element_array_buffer);
    STATE_VAR(element_array_buffer_binding) =
        STATE(default_vao_element_array_buffer) ? STATE(default_vao_element_array_buffer)->name : 0;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_VAO);
}

static VertexArray *mglGetSafeCurrentVAO(GLMContext ctx, const char *caller)
{
    VertexArray *vao;

    if (!ctx)
        return NULL;

    vao = ctx->state.vao;
    if (!vao)
        return NULL;

    /* Table membership implies live memory (VAOs leave the table before
     * free), so no readability probe is needed on the hit path. */
    if (!mglObjectPointerLooksPlausible(vao) ||
        !mglHashTableContainsData(&STATE(vao_table), vao))
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

bool validate_program(GLMContext ctx)
{
    Program *program = ctx ? ctx->state.program : NULL;

    if (program) {
        GLuint expectedName = ctx->state.program_name;
        if (expectedName == 0u &&
            mglObjectPointerLooksPlausible(program) &&
            mglPointerRangeIsReadable(program, sizeof(*program))) {
            /* Name unknown: probe before dereferencing to read it.  With a
             * known name, UsableForName's table fast path needs no probe. */
            expectedName = program->name;
        }

        if (expectedName == 0u ||
            !mglProgramPointerUsableForName(ctx, program, expectedName)) {
            fprintf(stderr, "MGL WARNING: validate_program dropping invalid cached program pointer %p\n",
                    (void *)program);
            ctx->state.program = NULL;
            ctx->state.program_name = 0;
            program = NULL;
        }
    }

    if (program) {
        if (program->shader_slots[_GEOMETRY_SHADER]) {
            static uint64_t s_geometryShaderProgramNoticeCount = 0;
            uint64_t hit = ++s_geometryShaderProgramNoticeCount;
            if (hit <= 16ull || (hit % 512ull) == 0ull) {
                fprintf(stderr,
                        "MGL WARNING: validate_program allowing geometry shader program=%u hit=%llu\n",
                        program->name,
                        (unsigned long long)hit);
            }
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
        &program->spirv_resources_list[_VERTEX_SHADER][_STAGE_OUTPUT_RES];
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
        &program->spirv_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
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
        &program->spirv_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES];
    for (GLuint i = 0; inputs->list && i < inputs->count; i++) {
        if (inputs->list[i].name && strcmp(inputs->list[i].name, name) == 0) {
            return &inputs->list[i];
        }
    }
    return NULL;
}

/* (deleted mglCPUFeedbackReadPositionBlock — test-specific: hardcoded UBO
 * names "index" / "PositionBlock". Removed as part of the CTS-cheat cleanup.) */

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

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx, GLuint64 generated, GLuint64 written);

static bool mglCPUReadIndexValue(const uint8_t *src, GLenum type, GLuint index, GLuint *value)
{
    if (!src || !value) {
        return false;
    }
    switch (type) {
        case GL_UNSIGNED_BYTE:
            *value = src[index];
            return true;
        case GL_UNSIGNED_SHORT:
            *value = ((const GLushort *)(const void *)src)[index];
            return true;
        case GL_UNSIGNED_INT:
            *value = ((const GLuint *)(const void *)src)[index];
            return true;
        default:
            return false;
    }
}

/* (deleted mglCPUFeedbackEvaluateCTSGeometryVarying,                *
 *          mglCPUFeedbackInputVerticesPerPrimitive,                   *
 *          mglCPUFeedbackOutputVerticesPerInputPrimitive — all were   *
 *          test-specific: substring-matched GS source and hardcoded   *
 *          varying names / magic values like 256.0f / 1024.0f.        *
 *          Removed as part of the CTS-cheat cleanup. A general VS-    *
 *          passthrough path replaces them; GS XFB is honestly not      *
 *          supported on Metal (no geometry-shader stage).)            */

/* Forward declarations — definitions live further down in this file. */
static bool mglCPUFeedbackResolveXFBSlot(GLMContext ctx,
                                         GLuint varying,
                                         Buffer **bufferOut,
                                         GLintptr *offsetOut,
                                         GLsizeiptr *sizeOut);
static GLuint64 mglCPUFeedbackPrimitiveCount(GLenum mode, GLuint64 vertices);
static bool mglCPUFeedbackIsPassthroughProgram(Program *program);

/* (deleted mglCPUFeedbackEvaluateSimpleVarying and                 *
 *  mglCPUFeedbackCanEvaluateProgram — test-specific substring/name *
 *  gates. Replaced by mglCPUFeedbackIsPassthroughProgram above.)   */

static GLuint64 mglCPUFeedbackPrimitiveCount(GLenum mode, GLuint64 vertices)
{
    switch (mode) {
        case GL_POINTS: return vertices;
        case GL_LINES: return vertices / 2u;
        case GL_LINE_STRIP: return vertices > 1u ? vertices - 1u : 0u;
        case GL_LINE_LOOP: return vertices > 1u ? vertices : 0u;
        case GL_TRIANGLES: return vertices / 3u;
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN: return vertices > 2u ? vertices - 2u : 0u;
        default: return vertices;
    }
}

static bool mglCPUFeedbackResolveXFBSlot(GLMContext ctx,
                                         GLuint varying,
                                         Buffer **bufferOut,
                                         GLintptr *offsetOut,
                                         GLsizeiptr *sizeOut)
{
    GLuint slotIndex = 0;
    if (ctx->state.program &&
        ctx->state.program->transform_feedback_buffer_mode == GL_SEPARATE_ATTRIBS) {
        slotIndex = varying;
    }
    if (slotIndex >= MAX_BINDABLE_BUFFERS) {
        return false;
    }
    BufferBaseTarget *slot = &ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[slotIndex];
    Buffer *buffer = slot->buf;
    if (!buffer || !buffer->data.buffer_data || buffer->size <= 0) {
        return false;
    }
    GLintptr offset = slot->offset;
    GLsizeiptr size = slot->size;
    if (offset < 0 || offset > buffer->size) {
        return false;
    }
    if (size <= 0 || size > buffer->size - offset) {
        size = buffer->size - offset;
    }
    if (size <= 0) {
        return false;
    }
    if (bufferOut) *bufferOut = buffer;
    if (offsetOut) *offsetOut = offset;
    if (sizeOut) *sizeOut = size;
    return true;
}

/* Determine whether a program's transform-feedback varyings can be captured
 * by the CPU passthrough path.
 *
 * A program is passthrough-capturable when EVERY captured varying is provably
 * a 1:1 copy of a vertex-shader INPUT attribute: the VS must declare an output
 * with the same name (or base name, for "foo[0]") at some location L and type
 * T, AND must declare a stage INPUT at the same location L with the same type
 * T. Under that assumption the captured value equals the input attribute value
 * for that vertex, which the CPU can read directly from the VAO without
 * interpreting the shader body.
 *
 * Limitations (honest, not cheated):
 *  - Only vertex-shader programs. Geometry/tessellation XFB is not capturable
 *    on the CPU (Metal has no geometry stage; tess-factor evaluation isn't
 *    reproducible on the CPU). The TES GPU path in MGLRenderer.m handles TES
 *    XFB independently. GS XFB is simply unsupported.
 *  - A VS that transforms an input but keeps the location would satisfy this
 *    check yet capture the wrong value; we accept that risk rather than parse
 *    GLSL. Non-passthrough XFB requires the GPU capture path (follow-up).
 */
static bool mglCPUFeedbackIsPassthroughProgram(Program *program)
{
    if (!program ||
        program->transform_feedback_varying_count <= 0 ||
        program->transform_feedback_varying_count > MAX_ATTRIBS) {
        return false;
    }

    /* Reject any program with a pre-fragment stage other than the vertex
     * shader (GS / TCS / TES). Those need a real geometry/tessellation
     * path, which the CPU cannot provide. */
    if (program->attached_shader_mask &
        (GEOMETRY_SHADER_MASK_BIT |
         TESS_CONTROL_SHADER_MASK_BIT |
         TESS_EVALUATION_SHADER_MASK_BIT)) {
        return false;
    }

    for (GLsizei i = 0; i < program->transform_feedback_varying_count; i++) {
        const char *name = program->transform_feedback_varying_names[i];
        if (!name || name[0] == '\0') {
            return false;
        }

        /* Strip "[N]" subscript -> base name (mirrors
         * mglValidateTransformFeedbackVaryings in program.c). */
        char base_name[96];
        strncpy(base_name, name, sizeof(base_name) - 1);
        base_name[sizeof(base_name) - 1] = '\0';
        char *bracket = strchr(base_name, '[');
        if (bracket) {
            *bracket = '\0';
        }

        SpirvResource *output = mglCPUFeedbackFindVertexOutput(program, base_name);
        if (!output) {
            return false;
        }

        /* The captured output must trace to a VS input at the same location
         * and with the same GL type. Try location first, then name. */
        SpirvResource *input =
            mglCPUFeedbackFindVertexInputAtLocation(program, output->location);
        if (!input) {
            input = mglCPUFeedbackFindVertexInputByName(program, base_name);
        }
        if (!input || input->gl_type != output->gl_type) {
            return false;
        }
    }

    return true;
}

/* Resolve the XFB slot for a given varying index and record it in the
 * touched-buffer list (used to flush + dirty-track after the capture loop). */
static bool mglCPUFeedbackNoteXFBSlot(GLMContext ctx,
                                      GLuint varying,
                                      Buffer *touchedBuffers[],
                                      GLintptr touchedOffsets[],
                                      GLsizeiptr touchedSizes[],
                                      GLuint *touchedCountInOut,
                                      Buffer **xfbOut,
                                      GLintptr *offsetOut,
                                      GLsizeiptr *sizeOut)
{
    Buffer *xfb = NULL;
    GLintptr offset = 0;
    GLsizeiptr size = 0;
    if (!mglCPUFeedbackResolveXFBSlot(ctx, varying, &xfb, &offset, &size)) {
        return false;
    }
    GLuint touchedCount = *touchedCountInOut;
    bool alreadyTouched = false;
    for (GLuint t = 0; t < touchedCount; t++) {
        if (touchedBuffers[t] == xfb) {
            alreadyTouched = true;
            break;
        }
    }
    if (!alreadyTouched && touchedCount < MAX_ATTRIBS) {
        touchedBuffers[touchedCount] = xfb;
        touchedOffsets[touchedCount] = offset;
        touchedSizes[touchedCount] = size;
        *touchedCountInOut = touchedCount + 1;
    }
    *xfbOut = xfb;
    *offsetOut = offset;
    *sizeOut = size;
    return true;
}

/* Flush captured XFB buffers back to the Metal buffer and update dirty
 * tracking + query counters. Shared by the arrays and elements paths. */
static void mglCPUFeedbackFlushAndCount(GLMContext ctx,
                                        Buffer *touchedBuffers[],
                                        GLintptr touchedOffsets[],
                                        GLsizeiptr touchedSizes[],
                                        GLuint touchedCount,
                                        GLuint64 totalVertices,
                                        GLuint64 capturedVertices)
{
    for (GLuint t = 0; t < touchedCount; t++) {
        Buffer *xfb = touchedBuffers[t];
        GLintptr dstOffset = touchedOffsets[t];
        GLsizeiptr writeSize = touchedSizes[t];
        if (!xfb) {
            continue;
        }
        if (ctx->mtl_funcs.mtlBufferSubData) {
            ctx->mtl_funcs.mtlBufferSubData(ctx,
                                            xfb,
                                            (size_t)dstOffset,
                                            (size_t)writeSize,
                                            (uint8_t *)(uintptr_t)xfb->data.buffer_data + dstOffset);
        }
        xfb->data.dirty_bits |= DIRTY_BUFFER_DATA;
        xfb->ever_written = GL_TRUE;
        xfb->has_initialized_data = GL_TRUE;
        if (xfb->written_min < 0 || dstOffset < xfb->written_min) {
            xfb->written_min = dstOffset;
        }
        GLintptr writeEnd = dstOffset + writeSize;
        if (xfb->written_max < 0 || writeEnd > xfb->written_max) {
            xfb->written_max = writeEnd;
        }
        xfb->last_init_source = kInitMapWrite;
        xfb->last_write_offset = dstOffset;
        xfb->last_write_size = writeSize;
        xfb->last_write_src_ptr = NULL;
        xfb->last_write_src_hash = 0;
    }

    GLuint64 generated = mglCPUFeedbackPrimitiveCount(
        ctx->state.transform_feedback->primitive_mode, totalVertices);
    GLuint64 written = mglCPUFeedbackPrimitiveCount(
        ctx->state.transform_feedback->primitive_mode, capturedVertices);
    ctx->state.transform_feedback->primitives_generated = generated;
    ctx->state.transform_feedback->primitives_written = written;
    mglRecordActivePrimitiveQueryDraw(ctx, generated, written);
}

/* Compute the per-varying layout (type, components, byte offset within an
 * interleaved vertex) for a passthrough program. Returns false if any varying
 * cannot be laid out (unknown type). */
static bool mglCPUFeedbackLayoutVaryings(Program *program,
                                         GLenum varyingTypes[MAX_ATTRIBS],
                                         GLuint varyingComponents[MAX_ATTRIBS],
                                         size_t varyingOffsets[MAX_ATTRIBS],
                                         size_t *vertexBytesOut)
{
    GLsizei varyingCount = program->transform_feedback_varying_count;
    size_t vertexBytes = 0;
    for (GLsizei varying = 0; varying < varyingCount; varying++) {
        const char *name = program->transform_feedback_varying_names[varying];
        char base_name[96];
        strncpy(base_name, name ? name : "", sizeof(base_name) - 1);
        base_name[sizeof(base_name) - 1] = '\0';
        char *bracket = strchr(base_name, '[');
        if (bracket) {
            *bracket = '\0';
        }

        SpirvResource *output = mglCPUFeedbackFindVertexOutput(program, base_name);
        GLenum type = output ? output->gl_type : GL_FLOAT_VEC4;
        GLuint components = mglCPUFeedbackGLTypeComponents(type);

        /* TODO(xfb-layout): this naive packed layout matches the TES GPU path
         * (program.c mglFixMSLTesAsComputeKernel Step 6). It does NOT apply
         * GL's std140 vec3->vec4 padding, matrix strides, or double alignment.
         * Programs using those will lay out incorrectly; the honest outcome
         * is a wrong-data failure rather than a silent cheat. */
        varyingOffsets[varying] = vertexBytes;
        varyingTypes[varying] = type;
        varyingComponents[varying] = components;
        vertexBytes += (size_t)components * mglCPUFeedbackGLTypeComponentBytes(type);
    }
    *vertexBytesOut = vertexBytes;
    return true;
}

/* Resolve + bounds-clamp the XFB slot capacity for every varying, returning
 * the minimum vertex capacity across all slots (the number of vertices that
 * actually fit). */
static bool mglCPUFeedbackClampCapacity(GLMContext ctx,
                                        Program *program,
                                        const GLenum varyingTypes[MAX_ATTRIBS],
                                        const GLuint varyingComponents[MAX_ATTRIBS],
                                        size_t vertexBytes,
                                        GLuint64 totalVertices,
                                        GLuint64 *capturedVerticesOut)
{
    GLsizei varyingCount = program->transform_feedback_varying_count;
    GLuint64 capturedVertices = totalVertices;
    for (GLsizei varying = 0; varying < varyingCount; varying++) {
        Buffer *xfb = NULL;
        GLintptr dstOffset = 0;
        GLsizeiptr dstSize = 0;
        if (!mglCPUFeedbackResolveXFBSlot(ctx, (GLuint)varying, &xfb, &dstOffset, &dstSize)) {
            return false;
        }
        size_t bytesPerVertex =
            program->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS
                ? vertexBytes
                : (size_t)varyingComponents[varying] *
                      mglCPUFeedbackGLTypeComponentBytes(varyingTypes[varying]);
        if (bytesPerVertex == 0) {
            return false;
        }
        uint64_t slotVertices = (uint64_t)dstSize / (uint64_t)bytesPerVertex;
        if (slotVertices < capturedVertices) {
            capturedVertices = slotVertices;
        }
        if (program->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS) {
            break;
        }
    }
    *capturedVerticesOut = capturedVertices;
    return true;
}

/* Capture one output vertex's varyings into the resolved XFB buffers. */
static void mglCPUFeedbackCaptureVertex(GLMContext ctx,
                                        Program *program,
                                        VertexArray *vao,
                                        GLuint linearVertex,
                                        GLuint attribVertex,
                                        GLuint instance,
                                        GLuint baseInstance,
                                        GLint first,
                                        const GLenum varyingTypes[MAX_ATTRIBS],
                                        const GLuint varyingComponents[MAX_ATTRIBS],
                                        const size_t varyingOffsets[MAX_ATTRIBS],
                                        size_t vertexBytes)
{
    GLsizei varyingCount = program->transform_feedback_varying_count;
    for (GLsizei varying = 0; varying < varyingCount; varying++) {
        const char *name = program->transform_feedback_varying_names[varying];
        char base_name[96];
        strncpy(base_name, name ? name : "", sizeof(base_name) - 1);
        base_name[sizeof(base_name) - 1] = '\0';
        char *bracket = strchr(base_name, '[');
        if (bracket) {
            *bracket = '\0';
        }

        SpirvResource *output = mglCPUFeedbackFindVertexOutput(program, base_name);
        SpirvResource *input =
            output ? mglCPUFeedbackFindVertexInputAtLocation(program, output->location) : NULL;
        if (!input && output) {
            input = mglCPUFeedbackFindVertexInputByName(program, base_name);
        }
        if (!input) {
            continue;
        }

        float values[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        mglCPUFeedbackReadAttrib(ctx,
                                 vao,
                                 input->location,
                                 first,
                                 attribVertex,
                                 instance,
                                 baseInstance,
                                 values);

        GLenum writeType = varyingTypes[varying];
        GLuint writeComponents = varyingComponents[varying];

        Buffer *xfb = NULL;
        GLintptr dstOffset = 0;
        GLsizeiptr dstSize = 0;
        if (!mglCPUFeedbackResolveXFBSlot(ctx, (GLuint)varying, &xfb, &dstOffset, &dstSize)) {
            continue;
        }

        size_t dstOffsetBytes;
        if (program->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS) {
            dstOffsetBytes = (size_t)dstOffset +
                             (size_t)linearVertex * vertexBytes +
                             varyingOffsets[varying];
        } else {
            size_t bytesPerVertex =
                (size_t)writeComponents * mglCPUFeedbackGLTypeComponentBytes(writeType);
            dstOffsetBytes = (size_t)dstOffset + (size_t)linearVertex * bytesPerVertex;
        }
        if (dstOffsetBytes +
                (size_t)writeComponents * mglCPUFeedbackGLTypeComponentBytes(writeType) >
            (size_t)dstOffset + (size_t)dstSize) {
            continue;
        }
        mglCPUFeedbackWriteValues(
            (uint8_t *)(uintptr_t)xfb->data.buffer_data + dstOffsetBytes,
            writeType,
            writeComponents,
            values);
    }
}

/* Common gate for both arrays and elements CPU capture paths. */
static bool mglCPUFeedbackCaptureGate(GLMContext ctx,
                                      GLenum mode,
                                      Program **programOut,
                                      VertexArray **vaoOut)
{
    if (!ctx ||
        !ctx->state.transform_feedback ||
        !ctx->state.transform_feedback->active ||
        ctx->state.transform_feedback->paused) {
        return false;
    }

    Program *program = ctx->state.program;
    if (!program ||
        program->transform_feedback_varying_count <= 0 ||
        (program->transform_feedback_buffer_mode != GL_INTERLEAVED_ATTRIBS &&
         program->transform_feedback_buffer_mode != GL_SEPARATE_ATTRIBS) ||
        !mglCPUFeedbackIsPassthroughProgram(program)) {
        return false;
    }
    /* GL spec: for a program without a geometry shader, the draw primitive
     * mode must match the transform-feedback primitive mode. */
    if (!program->shader_slots[_GEOMETRY_SHADER] &&
        mode != ctx->state.transform_feedback->primitive_mode) {
        return false;
    }

    VertexArray *vao = ctx->state.vao;
    if (!vao) {
        vao = mglGetOrCreateDefaultVAO(ctx);
    }
    if (!vao) {
        return false;
    }

    *programOut = program;
    *vaoOut = vao;
    return true;
}

bool mglTryCPUTransformFeedbackCapture(GLMContext ctx,
                                       GLenum mode,
                                       GLint first,
                                       GLsizei count,
                                       GLsizei instancecount,
                                       GLuint baseInstance)
{
    Program *program = NULL;
    VertexArray *vao = NULL;
    if (!mglCPUFeedbackCaptureGate(ctx, mode, &program, &vao)) {
        return false;
    }
    if (count <= 0) {
        /* Nothing to capture, but XFB is active: record zero primitives. */
        ctx->state.transform_feedback->primitives_generated = 0;
        ctx->state.transform_feedback->primitives_written = 0;
        mglRecordActivePrimitiveQueryDraw(ctx, 0, 0);
        return true;
    }
    if (instancecount <= 0) {
        instancecount = 1;
    }

    GLenum varyingTypes[MAX_ATTRIBS] = {0};
    GLuint varyingComponents[MAX_ATTRIBS] = {0};
    size_t varyingOffsets[MAX_ATTRIBS] = {0};
    size_t vertexBytes = 0;
    if (!mglCPUFeedbackLayoutVaryings(program, varyingTypes, varyingComponents,
                                      varyingOffsets, &vertexBytes)) {
        return false;
    }

    /* VS-only passthrough: one output vertex per input vertex. */
    uint64_t totalVertices = (uint64_t)count * (uint64_t)instancecount;
    uint64_t capturedVertices = 0;
    if (!mglCPUFeedbackClampCapacity(ctx, program, varyingTypes, varyingComponents,
                                     vertexBytes, totalVertices, &capturedVertices)) {
        return false;
    }

    Buffer *touchedBuffers[MAX_ATTRIBS] = {0};
    GLintptr touchedOffsets[MAX_ATTRIBS] = {0};
    GLsizeiptr touchedSizes[MAX_ATTRIBS] = {0};
    GLuint touchedCount = 0;
    /* Pre-resolve every slot once so the per-vertex loop only does lookups,
     * and so the flush tail knows the full dirty range. */
    for (GLsizei varying = 0; varying < program->transform_feedback_varying_count; varying++) {
        Buffer *xfb = NULL;
        GLintptr offset = 0;
        GLsizeiptr size = 0;
        if (!mglCPUFeedbackNoteXFBSlot(ctx, (GLuint)varying, touchedBuffers,
                                       touchedOffsets, touchedSizes, &touchedCount,
                                       &xfb, &offset, &size)) {
            return false;
        }
        if (program->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS) {
            break;
        }
    }

    GLuint64 linearVertex = 0;
    for (GLsizei inst = 0; inst < instancecount; inst++) {
        for (GLint v = 0; v < count; v++) {
            if (linearVertex >= capturedVertices) {
                break;
            }
            GLuint attribVertex = (GLuint)((GLint)v + first);
            mglCPUFeedbackCaptureVertex(ctx, program, vao, (GLuint)linearVertex,
                                        attribVertex, (GLuint)inst, baseInstance,
                                        first, varyingTypes, varyingComponents,
                                        varyingOffsets, vertexBytes);
            linearVertex++;
        }
        if (linearVertex >= capturedVertices) {
            break;
        }
    }

    mglCPUFeedbackFlushAndCount(ctx, touchedBuffers, touchedOffsets, touchedSizes,
                                touchedCount, totalVertices, capturedVertices);
    return true;
}

/* General indexed-draw CPU transform-feedback capture. Handles every element
 * type and draw mode; resolves the index buffer from the VAO element_array.
 * Honors basevertex (added to each index) and baseinstance (for instanced
 * attribute divisor math, applied inside mglCPUFeedbackReadAttrib). */
bool mglTryCPUTransformFeedbackCaptureElements(GLMContext ctx,
                                               GLenum mode,
                                               GLsizei count,
                                               GLenum type,
                                               const void *indices,
                                               GLsizei instancecount,
                                               GLint basevertex,
                                               GLuint baseinstance)
{
    Program *program = NULL;
    VertexArray *vao = NULL;
    if (!mglCPUFeedbackCaptureGate(ctx, mode, &program, &vao)) {
        return false;
    }
    if (count <= 0) {
        ctx->state.transform_feedback->primitives_generated = 0;
        ctx->state.transform_feedback->primitives_written = 0;
        mglRecordActivePrimitiveQueryDraw(ctx, 0, 0);
        return true;
    }
    if (instancecount <= 0) {
        instancecount = 1;
    }

    if (!vao->element_array.buffer ||
        !vao->element_array.buffer->data.buffer_data) {
        return false;
    }
    Buffer *indexBuffer = vao->element_array.buffer;
    uintptr_t indexOffset = (uintptr_t)indices;
    size_t indexTypeSize =
        (type == GL_UNSIGNED_BYTE) ? 1u :
        (type == GL_UNSIGNED_SHORT) ? 2u :
        (type == GL_UNSIGNED_INT) ? 4u : 0u;
    if (indexTypeSize == 0) {
        return false;
    }
    size_t indexBytes = (size_t)count * indexTypeSize;
    if ((uint64_t)indexOffset + indexBytes > (uint64_t)indexBuffer->size) {
        return false;
    }
    const uint8_t *indexData =
        (const uint8_t *)(uintptr_t)indexBuffer->data.buffer_data + indexOffset;

    GLenum varyingTypes[MAX_ATTRIBS] = {0};
    GLuint varyingComponents[MAX_ATTRIBS] = {0};
    size_t varyingOffsets[MAX_ATTRIBS] = {0};
    size_t vertexBytes = 0;
    if (!mglCPUFeedbackLayoutVaryings(program, varyingTypes, varyingComponents,
                                      varyingOffsets, &vertexBytes)) {
        return false;
    }

    uint64_t totalVertices = (uint64_t)count * (uint64_t)instancecount;
    uint64_t capturedVertices = 0;
    if (!mglCPUFeedbackClampCapacity(ctx, program, varyingTypes, varyingComponents,
                                     vertexBytes, totalVertices, &capturedVertices)) {
        return false;
    }

    Buffer *touchedBuffers[MAX_ATTRIBS] = {0};
    GLintptr touchedOffsets[MAX_ATTRIBS] = {0};
    GLsizeiptr touchedSizes[MAX_ATTRIBS] = {0};
    GLuint touchedCount = 0;
    for (GLsizei varying = 0; varying < program->transform_feedback_varying_count; varying++) {
        Buffer *xfb = NULL;
        GLintptr offset = 0;
        GLsizeiptr size = 0;
        if (!mglCPUFeedbackNoteXFBSlot(ctx, (GLuint)varying, touchedBuffers,
                                       touchedOffsets, touchedSizes, &touchedCount,
                                       &xfb, &offset, &size)) {
            return false;
        }
        if (program->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS) {
            break;
        }
    }

    GLuint64 linearVertex = 0;
    for (GLsizei inst = 0; inst < instancecount; inst++) {
        for (GLsizei i = 0; i < count; i++) {
            if (linearVertex >= capturedVertices) {
                break;
            }
            GLuint idx = 0;
            if (!mglCPUReadIndexValue(indexData, type, (GLuint)i, &idx)) {
                return false;
            }
            GLuint attribVertex = (GLuint)((GLint)idx + basevertex);
            mglCPUFeedbackCaptureVertex(ctx, program, vao, (GLuint)linearVertex,
                                        attribVertex, (GLuint)inst, baseinstance,
                                        /*first=*/0, varyingTypes, varyingComponents,
                                        varyingOffsets, vertexBytes);
            linearVertex++;
        }
        if (linearVertex >= capturedVertices) {
            break;
        }
    }

    mglCPUFeedbackFlushAndCount(ctx, touchedBuffers, touchedOffsets, touchedSizes,
                                touchedCount, totalVertices, capturedVertices);
    return true;
}


/* ================================================================== */
/* Unified Draw Frontend: mglDrawDispatch                  */
/* ================================================================== */
/* All "normal" draw entries (arrays / elements / instanced /         */
/* basevertex / baseinstance / range) funnel through this single     */
/* function.  Indirect and multidraw entries are NOT unified (see     */
/* docs/stage1_entry_step_matrix.md §B).                              */
/*                                                                    */
/* Design rules:                                                      */
/*  - Pure C, no Metal calls (core principle 1)                       */
/*  - Behaviour aligns with the original mglDrawArrays reference     */
/*    (conditional check BEFORE program validation, per GL spec)      */

/* True for element-indexed draw command types. */
static inline bool mglCmdIsIndexed(MGLDrawCommandType t)
{
    return t == MGL_CMD_DRAW_ELEMENTS                       ||
           t == MGL_CMD_DRAW_ELEMENTS_INSTANCED             ||
           t == MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX           ||
           t == MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX ||
           t == MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE         ||
           t == MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE;
}

/* True for instanced draw command types (instanceCount > 1 expected). */
static inline bool mglCmdIsInstanced(MGLDrawCommandType t)
{
    return t == MGL_CMD_DRAW_ARRAYS_INSTANCED                        ||
           t == MGL_CMD_DRAW_ELEMENTS_INSTANCED                      ||
           t == MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX          ||
           t == MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE          ||
           t == MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE        ||
           t == MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE;
}

static Program *mglCurrentGeometryDrawProgram(GLMContext ctx)
{
    if (!ctx) {
        return NULL;
    }
    if (ctx->state.program_name != 0u) {
        Program *program = ctx->state.program;
        return program && program->shader_slots[_GEOMETRY_SHADER]
            ? program : NULL;
    }
    ProgramPipeline *pipeline = ctx->state.program_pipeline;
    return pipeline ? pipeline->stage_programs[_GEOMETRY_SHADER] : NULL;
}

static Program *mglCurrentExpandedGeometryDrawProgram(GLMContext ctx)
{
    Program *program = mglCurrentGeometryDrawProgram(ctx);
    return program && !mglProgramHasPassthroughGeometryShader(program)
        ? program : NULL;
}

static bool mglBlockUnsupportedGeometryDraw(GLMContext ctx, const char *label)
{
    Program *program = mglCurrentExpandedGeometryDrawProgram(ctx);
    if (!program) {
        return false;
    }

    static uint64_t s_unsupportedGeometryDrawCount = 0;
    uint64_t hit = ++s_unsupportedGeometryDrawCount;
    if (hit <= 16ull || (hit % 512ull) == 0ull) {
        fprintf(stderr,
                "MGL GS WARNING: blocking unsupported %s program=%u hit=%llu\n",
                label ? label : "draw", program->name,
                (unsigned long long)hit);
    }
    mglTraceLogExternal("GS_DRAW_BLOCK label=%s program=%u hit=%llu",
                        label ? label : "draw", (unsigned)program->name,
                        (unsigned long long)hit);
    return true;
}

/* The 8-step unified draw frontend (S1-S11 + S14/S15). */
static void mglDrawDispatch(GLMContext ctx, const MGLDrawCommand *cmd)
{
    const bool indexed   = mglCmdIsIndexed(cmd->type);
    const bool instanced = mglCmdIsInstanced(cmd->type);

    /* S1: mode validation */
    if (!check_draw_modes(cmd->mode)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    /* S2: parameter validation */
    if (!indexed) {
        if (cmd->first < 0) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
    }
    if (cmd->count < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (cmd->count == 0) return;
    if (instanced) {
        if (cmd->instanceCount < 0) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
        if (cmd->instanceCount == 0) return;
    }

    /* S3+S4: element-specific validation (indexed only) */
    if (indexed) {
        if (!check_element_type(cmd->indexType)) {
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
        }
        if (should_skip_indexed_draw_no_element_buffer(ctx, "mglDrawDispatch")) {
            return;
        }
    }

    /* S5: VAO validation */
    if (!validate_vao(ctx, indexed)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    /* S6: conditional render check (BEFORE program — matches mglDrawArrays
     * reference; per GL spec, conditional skip should short-circuit). */
    if (mglShouldSkipConditionalRender(ctx))
        return;

    /* S7: program validation */
    if (!validate_program(ctx)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    /* S8: active query draw recording */
    mglRecordActiveSampleQueryDraw(ctx);

    /* S9: CPU transform feedback capture */
    if (!indexed) {
        if (mglTryCPUTransformFeedbackCapture(ctx, cmd->mode, cmd->first,
                                               cmd->count, cmd->instanceCount,
                                               cmd->baseInstance))
            return;
    } else {
        if (mglTryCPUTransformFeedbackCaptureElements(ctx, cmd->mode, cmd->count,
                                                       cmd->indexType,
                                                       (const void *)(uintptr_t)cmd->indexBufferOffset,
                                                       cmd->instanceCount,
                                                       cmd->baseVertex,
                                                       cmd->baseInstance))
            return;
    }

    /* S10: color shadow invalidation (always called — no-op when no
     * rgb10a2_shadow textures exist; fixes omission in 7/11 old entries). */
    mglInvalidateColorShadowsForDraw(ctx);

    /* S11: GL_PATCHES bypasses deferred (tessellation compute kernel path).
     * Only applies to non-instanced arrays (matches original mglDrawArrays). */
    if (cmd->type == MGL_CMD_DRAW_ARRAYS && cmd->mode == GL_PATCHES) {
        ctx->mtl_funcs.mtlDrawArrays(ctx, cmd->mode, cmd->first, cmd->count);
        return;
    }

    /* Geometry expansion rotates render/compute encoders and therefore cannot
     * be replayed as an ordinary deferred render draw.  Preserve ordering by
     * draining older commands, then let the renderer's GS helper run the AIR
     * compute route for both array and indexed shapes (P1). */
    Program *geometryProgram = mglCurrentExpandedGeometryDrawProgram(ctx);
    if (geometryProgram) {
        mglFlushCommandBuffer(ctx);
        switch (cmd->type) {
            case MGL_CMD_DRAW_ARRAYS:
                ctx->mtl_funcs.mtlDrawArrays(
                    ctx, cmd->mode, cmd->first, cmd->count);
                return;
            case MGL_CMD_DRAW_ARRAYS_INSTANCED:
                ctx->mtl_funcs.mtlDrawArraysInstanced(
                    ctx, cmd->mode, cmd->first, cmd->count,
                    cmd->instanceCount);
                return;
            case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE:
                ctx->mtl_funcs.mtlDrawArraysInstancedBaseInstance(
                    ctx, cmd->mode, cmd->first, cmd->count,
                    cmd->instanceCount, cmd->baseInstance);
                return;
            case MGL_CMD_DRAW_ELEMENTS:
                ctx->mtl_funcs.mtlDrawElements(
                    ctx, cmd->mode, cmd->count, cmd->indexType,
                    (const void *)(uintptr_t)cmd->indexBufferOffset);
                return;
            case MGL_CMD_DRAW_ELEMENTS_INSTANCED:
                ctx->mtl_funcs.mtlDrawElementsInstanced(
                    ctx, cmd->mode, cmd->count, cmd->indexType,
                    (const void *)(uintptr_t)cmd->indexBufferOffset,
                    cmd->instanceCount);
                return;
            case MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX:
                ctx->mtl_funcs.mtlDrawElementsBaseVertex(
                    ctx, cmd->mode, cmd->count, cmd->indexType,
                    (const void *)(uintptr_t)cmd->indexBufferOffset,
                    cmd->baseVertex);
                return;
            case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX:
                ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertex(
                    ctx, cmd->mode, cmd->count, cmd->indexType,
                    (const void *)(uintptr_t)cmd->indexBufferOffset,
                    cmd->instanceCount, cmd->baseVertex);
                return;
            case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE:
                ctx->mtl_funcs.mtlDrawElementsInstancedBaseInstance(
                    ctx, cmd->mode, cmd->count, cmd->indexType,
                    (const void *)(uintptr_t)cmd->indexBufferOffset,
                    cmd->instanceCount, cmd->baseInstance);
                return;
            case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE:
                ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertexBaseInstance(
                    ctx, cmd->mode, cmd->count, cmd->indexType,
                    (const void *)(uintptr_t)cmd->indexBufferOffset,
                    cmd->instanceCount, cmd->baseVertex, cmd->baseInstance);
                return;
            default:
                break;
        }
    }

    /* S14: deferred path — record command for batch replay */
    if (ctx->draw_defer_enabled) {
        mglTraceLogExternal("DRAW_DISPATCH_FRONTEND type=%u mode=0x%x first=%d count=%d "
                            "inst=%d bv=%d bi=%u program=%u defer=1",
                            (unsigned)cmd->type, (unsigned)cmd->mode, (int)cmd->first,
                            (int)cmd->count, (int)cmd->instanceCount, (int)cmd->baseVertex,
                            (unsigned)cmd->baseInstance,
                            (unsigned)mglTraceDrawProgram(ctx));
        mglRecordDrawCommand(ctx, cmd);
        return;
    }

    /* S15: immediate path — dispatch to Metal bridge */
    mglTraceLogExternal("DRAW_DISPATCH_FRONTEND type=%u mode=0x%x first=%d count=%d "
                        "inst=%d bv=%d bi=%u program=%u defer=0",
                        (unsigned)cmd->type, (unsigned)cmd->mode, (int)cmd->first,
                        (int)cmd->count, (int)cmd->instanceCount, (int)cmd->baseVertex,
                        (unsigned)cmd->baseInstance,
                        (unsigned)mglTraceDrawProgram(ctx));
    if (getenv("MGL_CULL_DBG")) fprintf(stderr, "MGL_CULL_DBG: mglDrawDispatch immediate\n");

    switch (cmd->type) {
        case MGL_CMD_DRAW_ARRAYS:
            ctx->mtl_funcs.mtlDrawArrays(ctx, cmd->mode, cmd->first, cmd->count);
            break;
        case MGL_CMD_DRAW_ELEMENTS:
            ctx->mtl_funcs.mtlDrawElements(ctx, cmd->mode, cmd->count, cmd->indexType,
                                           (const void *)(uintptr_t)cmd->indexBufferOffset);
            break;
        case MGL_CMD_DRAW_ARRAYS_INSTANCED:
            ctx->mtl_funcs.mtlDrawArraysInstanced(ctx, cmd->mode, cmd->first, cmd->count,
                                                  cmd->instanceCount);
            break;
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED:
            ctx->mtl_funcs.mtlDrawElementsInstanced(ctx, cmd->mode, cmd->count, cmd->indexType,
                                                    (const void *)(uintptr_t)cmd->indexBufferOffset,
                                                    cmd->instanceCount);
            break;
        case MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX:
            /* DrawRangeElementsBaseVertex also lands here (start/end ignored
             * by Metal backend — verified at MGLRenderer.m:32670). */
            ctx->mtl_funcs.mtlDrawElementsBaseVertex(ctx, cmd->mode, cmd->count, cmd->indexType,
                                                     (const void *)(uintptr_t)cmd->indexBufferOffset,
                                                     cmd->baseVertex);
            break;
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX:
            ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertex(ctx, cmd->mode, cmd->count,
                                                              cmd->indexType,
                                                              (const void *)(uintptr_t)cmd->indexBufferOffset,
                                                              cmd->instanceCount,
                                                              cmd->baseVertex);
            break;
        case MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE:
            ctx->mtl_funcs.mtlDrawArraysInstancedBaseInstance(ctx, cmd->mode, cmd->first,
                                                              cmd->count, cmd->instanceCount,
                                                              cmd->baseInstance);
            break;
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE:
            ctx->mtl_funcs.mtlDrawElementsInstancedBaseInstance(ctx, cmd->mode, cmd->count,
                                                                cmd->indexType,
                                                                (const void *)(uintptr_t)cmd->indexBufferOffset,
                                                                cmd->instanceCount,
                                                                cmd->baseInstance);
            break;
        case MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE:
            ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertexBaseInstance(ctx, cmd->mode, cmd->count,
                                                                          cmd->indexType,
                                                                          (const void *)(uintptr_t)cmd->indexBufferOffset,
                                                                          cmd->instanceCount,
                                                                          cmd->baseVertex,
                                                                          cmd->baseInstance);
            break;
        default:
            /* Unreachable — indirect/multidraw types never enter dispatch. */
            fprintf(stderr, "MGL Error: mglDrawDispatch unknown type %u\n",
                    (unsigned)cmd->type);
            ERROR_RETURN(GL_INVALID_OPERATION);
            break;
    }
}


void mglDrawArrays(GLMContext ctx, GLenum mode, GLint first, GLsizei count)
{
    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type          = MGL_CMD_DRAW_ARRAYS;
    cmd.mode          = mode;
    cmd.first         = first;
    cmd.count         = count;
    cmd.instanceCount = 1;
    mglDrawDispatch(ctx, &cmd);
}

void mglDrawElements(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices)
{
    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type              = MGL_CMD_DRAW_ELEMENTS;
    cmd.mode              = mode;
    cmd.count             = count;
    cmd.indexType         = type;
    cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
    cmd.elementBuffer     = mglCurrentElementBuffer(ctx, __func__);
    cmd.instanceCount     = 1;
    mglDrawDispatch(ctx, &cmd);
}

void mglDrawRangeElements(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices)
{
    /* Range-specific validation (start/end not stored in MGLDrawCommand;
     * Metal backend ignores them — verified at MGLRenderer.m:32670). */
    if (end < start) { ERROR_RETURN(GL_INVALID_VALUE); return; }

    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type              = MGL_CMD_DRAW_ELEMENTS;
    cmd.mode              = mode;
    cmd.count             = count;
    cmd.indexType         = type;
    cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
    cmd.elementBuffer     = mglCurrentElementBuffer(ctx, __func__);
    cmd.instanceCount     = 1;
    mglDrawDispatch(ctx, &cmd);
}

void mglDrawArraysInstanced(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
{
    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type          = MGL_CMD_DRAW_ARRAYS_INSTANCED;
    cmd.mode          = mode;
    cmd.first         = first;
    cmd.count         = count;
    cmd.instanceCount = instancecount;
    mglDrawDispatch(ctx, &cmd);
}

void mglDrawElementsInstanced(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount)
{
    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type              = MGL_CMD_DRAW_ELEMENTS_INSTANCED;
    cmd.mode              = mode;
    cmd.count             = count;
    cmd.indexType         = type;
    cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
    cmd.elementBuffer     = mglCurrentElementBuffer(ctx, __func__);
    cmd.instanceCount     = instancecount;
    mglDrawDispatch(ctx, &cmd);
}

void mglDrawElementsBaseVertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type              = MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX;
    cmd.mode              = mode;
    cmd.count             = count;
    cmd.indexType         = type;
    cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
    cmd.elementBuffer     = mglCurrentElementBuffer(ctx, __func__);
    cmd.baseVertex        = basevertex;
    cmd.instanceCount     = 1;
    mglDrawDispatch(ctx, &cmd);
}

void mglDrawRangeElementsBaseVertex(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    /* Range-specific validation (start/end ignored by Metal backend). */
    if (end < start) { ERROR_RETURN(GL_INVALID_VALUE); return; }

    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type              = MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX;
    cmd.mode              = mode;
    cmd.count             = count;
    cmd.indexType         = type;
    cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
    cmd.elementBuffer     = mglCurrentElementBuffer(ctx, __func__);
    cmd.baseVertex        = basevertex;
    cmd.instanceCount     = 1;
    mglDrawDispatch(ctx, &cmd);
}

void mglDrawElementsInstancedBaseVertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex)
{
    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type              = MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX;
    cmd.mode              = mode;
    cmd.count             = count;
    cmd.indexType         = type;
    cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
    cmd.elementBuffer     = mglCurrentElementBuffer(ctx, __func__);
    cmd.instanceCount     = instancecount;
    cmd.baseVertex        = basevertex;
    mglDrawDispatch(ctx, &cmd);
}

void mglDrawArraysIndirect(GLMContext ctx, GLenum mode, const void *indirect)
{
    mglTraceLogExternal("DRAW_ARRAYS_INDIRECT_FRONTEND_ENTRY mode=0x%x indirect=%p program=%u",
                        (unsigned)mode, indirect, (unsigned)mglTraceDrawProgram(ctx));

    if (!check_draw_modes(mode)) {
        mglTraceLogExternal("DRAW_ARRAYS_INDIRECT_FRONTEND_SKIP reason=bad_mode mode=0x%x program=%u",
                            (unsigned)mode, (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (!mglValidateDrawIndirectCommands(ctx,
                                         "DRAW_ARRAYS_INDIRECT_FRONTEND",
                                         indirect,
                                         1,
                                         0,
                                         (GLsizeiptr)sizeof(DrawArraysIndirectCommand))) {
        return;
    }

    if(validate_vao(ctx, false) == false)
    {
        mglTraceLogExternal("DRAW_ARRAYS_INDIRECT_FRONTEND_SKIP reason=validate_vao program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) {
        mglTraceLogExternal("DRAW_ARRAYS_INDIRECT_FRONTEND_SKIP reason=validate_program program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (mglSkipOrRecordConditionalDraw(ctx)) {
        mglTraceLogExternal("DRAW_ARRAYS_INDIRECT_FRONTEND_SKIP reason=conditional_render program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        return;
    }

    if (mglBlockUnsupportedGeometryDraw(ctx, "drawArraysIndirect")) {
        return;
    }

    mglFlushCommandBuffer(ctx);
    mglTraceLogExternal("DRAW_ARRAYS_INDIRECT_FRONTEND_DISPATCH mode=0x%x indirect=%p program=%u",
                        (unsigned)mode, indirect, (unsigned)mglTraceDrawProgram(ctx));
    ctx->mtl_funcs.mtlDrawArraysIndirect(ctx, mode, indirect);
}

void mglDrawElementsIndirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect)
{
    mglTraceLogExternal("DRAW_ELEMENTS_INDIRECT_FRONTEND_ENTRY mode=0x%x type=0x%x indirect=%p program=%u",
                        (unsigned)mode, (unsigned)type, indirect, (unsigned)mglTraceDrawProgram(ctx));

    if (!check_draw_modes(mode)) {
        mglTraceLogExternal("DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=bad_mode mode=0x%x program=%u",
                            (unsigned)mode, (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (!mglValidateDrawIndirectCommands(ctx,
                                         "DRAW_ELEMENTS_INDIRECT_FRONTEND",
                                         indirect,
                                         1,
                                         0,
                                         (GLsizeiptr)sizeof(DrawElementsIndirectCommand))) {
        return;
    }

    if (!check_element_type(type)) {
        mglTraceLogExternal("DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=bad_type type=0x%x program=%u",
                            (unsigned)type, (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        mglTraceLogExternal("DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=no_element_buffer program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        mglTraceLogExternal("DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=validate_vao program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) {
        mglTraceLogExternal("DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=validate_program program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (mglSkipOrRecordConditionalDraw(ctx)) {
        mglTraceLogExternal("DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=conditional_render program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        return;
    }

    if (mglBlockUnsupportedGeometryDraw(ctx, "drawElementsIndirect")) {
        return;
    }

    mglFlushCommandBuffer(ctx);
    mglTraceLogExternal("DRAW_ELEMENTS_INDIRECT_FRONTEND_DISPATCH mode=0x%x type=0x%x indirect=%p program=%u",
                        (unsigned)mode, (unsigned)type, indirect, (unsigned)mglTraceDrawProgram(ctx));
    ctx->mtl_funcs.mtlDrawElementsIndirect(ctx, mode, type, indirect);
}

void mglDrawArraysInstancedBaseInstance(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
{
    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type          = MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE;
    cmd.mode          = mode;
    cmd.first         = first;
    cmd.count         = count;
    cmd.instanceCount = instancecount;
    cmd.baseInstance  = baseinstance;
    mglDrawDispatch(ctx, &cmd);
}

void mglDrawElementsInstancedBaseInstance(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance)
{
    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type              = MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE;
    cmd.mode              = mode;
    cmd.count             = count;
    cmd.indexType         = type;
    cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
    cmd.elementBuffer     = mglCurrentElementBuffer(ctx, __func__);
    cmd.instanceCount     = instancecount;
    cmd.baseInstance      = baseinstance;
    mglDrawDispatch(ctx, &cmd);
}

void mglDrawElementsInstancedBaseVertexBaseInstance(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
{
    MGLDrawCommand cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.type              = MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE;
    cmd.mode              = mode;
    cmd.count             = count;
    cmd.indexType         = type;
    cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
    cmd.elementBuffer     = mglCurrentElementBuffer(ctx, __func__);
    cmd.instanceCount     = instancecount;
    cmd.baseVertex        = basevertex;
    cmd.baseInstance      = baseinstance;
    mglDrawDispatch(ctx, &cmd);
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

    if (mglBlockUnsupportedGeometryDraw(ctx, "multiDrawArrays")) {
        return;
    }

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
            mglRecordDrawCommand(ctx, &cmd);
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

    if (mglBlockUnsupportedGeometryDraw(ctx, "multiDrawElements")) {
        return;
    }

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
            mglRecordDrawCommand(ctx, &cmd);
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

    if (mglBlockUnsupportedGeometryDraw(ctx,
                                        "multiDrawElementsBaseVertex")) {
        return;
    }

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
            mglRecordDrawCommand(ctx, &cmd);
        }
        return;
    }

    ctx->mtl_funcs.mtlMultiDrawElementsBaseVertex(ctx, mode, count, type, indices, drawcount, basevertex);
}

void mglMultiDrawArraysIndirect(GLMContext ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_FRONTEND_ENTRY mode=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                        (unsigned)mode, indirect, (int)drawcount, (int)stride, (unsigned)mglTraceDrawProgram(ctx));

    if (!check_draw_modes(mode)) {
        mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_FRONTEND_SKIP reason=bad_mode mode=0x%x program=%u",
                            (unsigned)mode, (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (drawcount < 0) {
        mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_FRONTEND_SKIP reason=bad_drawcount drawcount=%d program=%u",
                            (int)drawcount, (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (drawcount == 0) {
        mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_FRONTEND_SKIP reason=zero_drawcount program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        return;
    }
    if (stride % 4 != 0) {
        mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_FRONTEND_SKIP reason=bad_stride stride=%d program=%u",
                            (int)stride, (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if(validate_vao(ctx, false) == false)
    {
        mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_FRONTEND_SKIP reason=validate_vao program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) {
        mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_FRONTEND_SKIP reason=validate_program program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!mglValidateDrawIndirectCommands(ctx,
                                         "MULTI_DRAW_ARRAYS_INDIRECT_FRONTEND",
                                         indirect,
                                         drawcount,
                                         stride,
                                         (GLsizeiptr)sizeof(DrawArraysIndirectCommand))) {
        return;
    }

    if (mglBlockUnsupportedGeometryDraw(ctx,
                                        "multiDrawArraysIndirect")) {
        return;
    }

    mglFlushCommandBuffer(ctx);
    mglTraceLogExternal("MULTI_DRAW_ARRAYS_INDIRECT_FRONTEND_DISPATCH mode=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                        (unsigned)mode, indirect, (int)drawcount, (int)stride, (unsigned)mglTraceDrawProgram(ctx));
    ctx->mtl_funcs.mtlMultiDrawArraysIndirect(ctx, mode, indirect, drawcount, stride);
}

void mglMultiDrawElementsIndirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND_ENTRY mode=0x%x type=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                        (unsigned)mode, (unsigned)type, indirect, (int)drawcount, (int)stride, (unsigned)mglTraceDrawProgram(ctx));

    if (!check_draw_modes(mode)) {
        mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=bad_mode mode=0x%x program=%u",
                            (unsigned)mode, (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (drawcount < 0) {
        mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=bad_drawcount drawcount=%d program=%u",
                            (int)drawcount, (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (drawcount == 0) {
        mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=zero_drawcount program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        return;
    }
    if (stride % 4 != 0) {
        mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=bad_stride stride=%d program=%u",
                            (int)stride, (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!check_element_type(type)) {
        mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=bad_type type=0x%x program=%u",
                            (unsigned)type, (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=no_element_buffer program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=validate_vao program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) {
        mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND_SKIP reason=validate_program program=%u",
                            (unsigned)mglTraceDrawProgram(ctx));
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!mglValidateDrawIndirectCommands(ctx,
                                         "MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND",
                                         indirect,
                                         drawcount,
                                         stride,
                                         (GLsizeiptr)sizeof(DrawElementsIndirectCommand))) {
        return;
    }

    if (mglBlockUnsupportedGeometryDraw(ctx,
                                        "multiDrawElementsIndirect")) {
        return;
    }

    mglFlushCommandBuffer(ctx);
    mglTraceLogExternal("MULTI_DRAW_ELEMENTS_INDIRECT_FRONTEND_DISPATCH mode=0x%x type=0x%x indirect=%p drawcount=%d stride=%d program=%u",
                        (unsigned)mode, (unsigned)type, indirect, (int)drawcount, (int)stride, (unsigned)mglTraceDrawProgram(ctx));
    ctx->mtl_funcs.mtlMultiDrawElementsIndirect(ctx, mode, type, indirect, drawcount, stride);
}
