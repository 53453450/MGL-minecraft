/*
 * Copyright (C) Michael Larson on on 1/6/25.
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
 * uniforms.c
 * MGL
 *
 */

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <mach/mach.h>
#include "spirv_cross_c.h"

#include "shaders.h"
#include "programs.h"
#include "buffers.h"
#include "glm_context.h"

extern GLMContext _ctx;


#pragma mark uniforms

#define MGL_INTERNAL_UNIFORM_BUFFER_NAME_BASE 0xf0000000u

static GLboolean mglPointerRangeReadable(const void *ptr, size_t size)
{
    if (size == 0) {
        return GL_TRUE;
    }
    if (!ptr) {
        return GL_FALSE;
    }

    uintptr_t start = (uintptr_t)ptr;
    if (start < 0x10000u || start > UINTPTR_MAX - size + 1u) {
        return GL_FALSE;
    }

    vm_address_t address = (vm_address_t)start;
    vm_size_t regionSize = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t objectName = MACH_PORT_NULL;
    kern_return_t kr = vm_region_64(mach_task_self(),
                                    &address,
                                    &regionSize,
                                    VM_REGION_BASIC_INFO_64,
                                    (vm_region_info_t)&info,
                                    &count,
                                    &objectName);
    if (objectName != MACH_PORT_NULL) {
        mach_port_deallocate(mach_task_self(), objectName);
    }
    if (kr != KERN_SUCCESS || start < (uintptr_t)address) {
        return GL_FALSE;
    }

    uintptr_t end = start + size;
    uintptr_t regionEnd = (uintptr_t)address + (uintptr_t)regionSize;
    if (regionEnd < (uintptr_t)address || end > regionEnd) {
        return GL_FALSE;
    }

    return (info.protection & VM_PROT_READ) ? GL_TRUE : GL_FALSE;
}

static GLMContext mglUniformResolveContext(GLMContext ctx, const char *func)
{
    GLMContext current = _ctx;
    if (!current || !mglPointerRangeReadable(current, sizeof(*current))) {
        fprintf(stderr,
                "MGL WARNING: dropping uniform update in %s with invalid current ctx=%p arg=%p\n",
                func ? func : "(null)",
                (void *)current,
                (void *)ctx);
        return NULL;
    }

    if (ctx != current) {
        static unsigned long long s_stale_uniform_ctx_count = 0;
        s_stale_uniform_ctx_count++;
        if (s_stale_uniform_ctx_count <= 32ull || (s_stale_uniform_ctx_count % 1024ull) == 0ull) {
            fprintf(stderr,
                    "MGL WARNING: uniform update in %s used stale ctx=%p current=%p hit=%llu\n",
                    func ? func : "(null)",
                    (void *)ctx,
                    (void *)current,
                    s_stale_uniform_ctx_count);
        }
    }

    return current;
}

static void mglUniformSetError(GLMContext ctx, GLenum error)
{
    if (!ctx || !mglPointerRangeReadable(ctx, sizeof(*ctx))) {
        return;
    }
    if (ctx->state.error == GL_NO_ERROR) {
        ctx->state.error = error;
    }
}

static GLint mglKnownPlainUniformLocation(const char *name)
{
    if (!name) {
        return -1;
    }

    if (!strcmp(name, "ModelViewMat")) {
        return 0;
    }
    if (!strcmp(name, "ProjMat")) {
        return 1;
    }
    if (!strcmp(name, "TextureMat")) {
        return 2;
    }
    if (!strcmp(name, "ColorModulator")) {
        return 3;
    }
    if (!strcmp(name, "FogStart")) {
        return 4;
    }
    if (!strcmp(name, "FogEnd")) {
        return 5;
    }
    if (!strcmp(name, "FogColor")) {
        return 6;
    }
    if (!strcmp(name, "FogShape")) {
        return 7;
    }
    if (!strcmp(name, "GameTime")) {
        return 8;
    }
    if (!strcmp(name, "ScreenSize")) {
        return 9;
    }
    if (!strcmp(name, "LineWidth")) {
        return 10;
    }
    if (!strcmp(name, "IViewRotMat")) {
        return 11;
    }
    if (!strcmp(name, "ChunkOffset")) {
        return 12;
    }

    return -1;
}

static GLint mglPlainUniformResourceLocation(const SpirvResource *res)
{
    if (!res) {
        return -1;
    }

    GLint known = mglKnownPlainUniformLocation(res->name);
    if (known >= 0) {
        return known;
    }
    if (res->uniform_location >= 0 && res->uniform_location < MAX_BINDABLE_BUFFERS) {
        return res->uniform_location;
    }
    if (res->location < MAX_BINDABLE_BUFFERS) {
        return (GLint)res->location;
    }
    if (res->gl_binding < MAX_BINDABLE_BUFFERS) {
        return (GLint)res->gl_binding;
    }
    if (res->binding < MAX_BINDABLE_BUFFERS) {
        return (GLint)res->binding;
    }

    return -1;
}

static GLboolean mglUniformLocationMatchesResource(const SpirvResource *res, int res_type, GLint location)
{
    if (!res || location < 0) {
        return GL_FALSE;
    }

    // Reflected sampler/image resources often report location 0 even when they
    // are distinct uniforms. Once we assign a synthetic location, only that
    // location should match; otherwise unrelated integer uniforms at location 0
    // can accidentally rewrite sampler units.
    if (res_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT) {
        GLint plain_location = mglPlainUniformResourceLocation(res);
        if (plain_location >= 0 && plain_location == location) {
            return GL_TRUE;
        }
    }

    if (res->uniform_location >= 0 && res->uniform_location == location) {
        return GL_TRUE;
    }
    if (res->uniform_location >= 0) {
        return GL_FALSE;
    }

    if (res->location != 0xffffffffu && (GLint)res->location == location) {
        return GL_TRUE;
    }

    return (GLint)res->binding == location ? GL_TRUE : GL_FALSE;
}

static GLboolean mglFindSamplerUniformBinding(Program *program, GLint location, int *stage_out, GLuint *binding_out)
{
    static const int sampler_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS
    };

    if (!program || location < 0 || !binding_out) {
        return GL_FALSE;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (size_t rt = 0; rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); rt++) {
            int res_type = sampler_resource_types[rt];
            SpirvResourceList *resources = &program->spirv_resources_list[stage][res_type];
            for (GLuint i = 0; i < resources->count; i++) {
                SpirvResource *res = &resources->list[i];
                if (!mglUniformLocationMatchesResource(res, res_type, location)) {
                    continue;
                }

                if (res->binding < TEXTURE_UNITS) {
                    if (stage_out) {
                        *stage_out = stage;
                    }
                    *binding_out = res->binding;
                    return GL_TRUE;
                }
            }
        }
    }

    return GL_FALSE;
}

static GLboolean mglSetSamplerUniformUnit(GLMContext ctx, GLint location, GLint unit)
{
    if (!ctx || location < 0) {
        return GL_FALSE;
    }

    Program *program = ctx->state.program;
    if (!program) {
        return GL_FALSE;
    }

    int stage = -1;
    GLuint binding = 0;
    if (!mglFindSamplerUniformBinding(program, location, &stage, &binding)) {
        return GL_FALSE;
    }

    if (unit < 0 || unit >= TEXTURE_UNITS) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, GL_TRUE);
    }

    program->sampler_units[binding] = unit;
    if (stage >= 0 && stage < _MAX_SHADER_TYPES) {
        program->sampler_units_by_stage[stage][binding] = unit;
    }
    return GL_TRUE;
}

static size_t mglRoundUpUniformBlockSize(size_t value)
{
    return value ? ((value + 15) & ~(size_t)15) : 0;
}

static SpirvResource *mglFindUniformBlockByIndex(Program *program, GLuint uniformBlockIndex, int *stage_out)
{
    SpirvResource *first_by_ordinal = NULL;
    GLuint ordinal = 0;

    if (!program || uniformBlockIndex == GL_INVALID_INDEX) {
        return NULL;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (res->gl_binding == uniformBlockIndex) {
                if (stage_out) {
                    *stage_out = stage;
                }
                return res;
            }
            if (ordinal == uniformBlockIndex && !first_by_ordinal) {
                first_by_ordinal = res;
                if (stage_out) {
                    *stage_out = stage;
                }
            }
            ordinal++;
        }
    }

    return first_by_ordinal;
}

static size_t mglUniformBlockRequiredSize(Program *program, const SpirvResource *block)
{
    size_t required_size = 0;

    if (!program || !block) {
        return 0;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            GLboolean same_block = GL_FALSE;

            if (block->name && res->name && !strcmp(block->name, res->name)) {
                same_block = GL_TRUE;
            } else if (res->gl_binding == block->gl_binding) {
                same_block = GL_TRUE;
            }

            if (same_block && res->required_size > required_size) {
                required_size = res->required_size;
            }
        }
    }

    return mglRoundUpUniformBlockSize(required_size);
}

static GLboolean mglUniformBlockReferencedByStage(Program *program, const SpirvResource *block, int query_stage)
{
    if (!program || !block || query_stage < 0 || query_stage >= _MAX_SHADER_TYPES) {
        return GL_FALSE;
    }

    SpirvResourceList *resources = &program->spirv_resources_list[query_stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
    for (GLuint i = 0; i < resources->count; i++) {
        SpirvResource *res = &resources->list[i];
        if ((block->name && res->name && !strcmp(block->name, res->name)) ||
            res->gl_binding == block->gl_binding) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

GLint  mglGetUniformLocation(GLMContext ctx, GLuint program, const GLchar *name)
{
    if (!ctx) {
        return -1;
    }

    if (!name) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, -1);
    }

    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, -1);
    }

    Program *ptr = getProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, -1);
    }

    if (ptr->linked_glsl_program == NULL) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, -1);
    }

    const int resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    for (size_t rt = 0; rt < sizeof(resource_types) / sizeof(resource_types[0]); rt++)
    {
        int res_type = resource_types[rt];
        for (int stage=_VERTEX_SHADER; stage<_MAX_SHADER_TYPES; stage++)
        {
            int count = ptr->spirv_resources_list[stage][res_type].count;
            SpirvResource *list = ptr->spirv_resources_list[stage][res_type].list;
            if (count <= 0 || !list) {
                continue;
            }

            for (int i=0; i<count; i++)
            {
                const char *str = list[i].name;
                if (!str) {
                    continue;
                }

                if (!strcmp(str, name))
                {
                    GLuint binding = list[i].binding;
                    GLint location = (res_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT)
                        ? mglPlainUniformResourceLocation(&list[i])
                        : ((list[i].uniform_location >= 0)
                            ? list[i].uniform_location
                            : ((list[i].location != 0xffffffffu) ? (GLint)list[i].location : (GLint)binding));

                    if (strstr(name, "CloudFaces")) {
                        fprintf(stderr,
                                "MGL REFLECT UniformLocation program=%u name=%s -> loc=%d binding=%u type=%d stage=%d dim=%u\n",
                                program,
                                name,
                                location,
                                binding,
                                res_type,
                                stage,
                                list[i].image_dim);
                    }

                    return location;
                }
            }
        }
    }

    if (strstr(name, "CloudFaces")) {
        fprintf(stderr,
                "MGL REFLECT UniformLocation MISS program=%u name=%s\n",
                program,
                name);
    }
    
    return -1;
}

void mglGetUniformfv(GLMContext ctx, GLuint program, GLint location, GLfloat *params)
{
    (void)location;
    if (!ctx) {
        return;
    }
    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (params) {
        *params = 0.0f;
    }
}

void mglGetUniformiv(GLMContext ctx, GLuint program, GLint location, GLint *params)
{
    if (!ctx) {
        return;
    }
    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (params) {
        Program *ptr = getProgram(ctx, program);
        GLuint binding = 0;
        int stage = -1;
        if (ptr && mglFindSamplerUniformBinding(ptr, location, &stage, &binding)) {
            *params = (stage >= 0 && stage < _MAX_SHADER_TYPES)
                ? ptr->sampler_units_by_stage[stage][binding]
                : ptr->sampler_units[binding];
        } else {
            *params = 0;
        }
    }
}


void mglGetUniformIndices(GLMContext ctx, GLuint program, GLsizei uniformCount, const GLchar *const*uniformNames, GLuint *uniformIndices)
{
    if (!ctx) {
        return;
    }
    if (uniformCount < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (uniformCount > 0 && (!uniformNames || !uniformIndices)) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < uniformCount; i++) {
        uniformIndices[i] = (GLuint)-1;
    }
}

void mglGetActiveUniformsiv(GLMContext ctx, GLuint program, GLsizei uniformCount, const GLuint *uniformIndices, GLenum pname, GLint *params)
{
    (void)uniformIndices;
    (void)pname;
    if (!ctx) {
        return;
    }
    if (uniformCount < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (uniformCount > 0 && !params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < uniformCount; i++) {
        params[i] = 0;
    }
}

void mglGetActiveUniformName(GLMContext ctx, GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformName)
{
    (void)uniformIndex;
    if (!ctx) {
        return;
    }
    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (length) {
        *length = 0;
    }
    if (uniformName && bufSize > 0) {
        uniformName[0] = '\0';
    }
}

GLuint  mglGetUniformBlockIndex(GLMContext ctx, GLuint program, const GLchar *uniformBlockName)
{
    if (!ctx) {
        return (GLuint)-1;
    }

    if (!uniformBlockName) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, (GLuint)-1);
    }

    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, (GLuint)-1);
    }

    Program *ptr = getProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, (GLuint)-1);
    }

    if (ptr->linked_glsl_program == NULL) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, (GLuint)-1);
    }

    for (int stage=_VERTEX_SHADER; stage<_MAX_SHADER_TYPES; stage++)
    {
        int count = ptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER].count;
        SpirvResource *list = ptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER].list;
        if (count <= 0 || !list) {
            continue;
        }

        for (int i=0; i<count; i++)
        {
            const char *str = list[i].name;
            if (!str) {
                continue;
            }

            if (!strcmp(str, uniformBlockName))
            {
                GLuint binding = list[i].gl_binding;

                return binding;
            }
        }
    }

    fprintf(stderr, "MGL WARNING: uniform block '%s' binding not found, returning GL_INVALID_INDEX\n", uniformBlockName ? uniformBlockName : "(null)");
    return (GLuint)-1;
}

void mglGetActiveUniformBlockiv(GLMContext ctx, GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params)
{
    if (!params) {
        return;
    }
    *params = 0;

    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = getProgram(ctx, program);
    if (!ptr || ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    SpirvResource *block = mglFindUniformBlockByIndex(ptr, uniformBlockIndex, NULL);
    if (!block) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch (pname) {
        case GL_UNIFORM_BLOCK_BINDING:
            *params = (GLint)block->gl_binding;
            break;
        case GL_UNIFORM_BLOCK_DATA_SIZE:
            *params = (GLint)mglUniformBlockRequiredSize(ptr, block);
            break;
        case GL_UNIFORM_BLOCK_NAME_LENGTH:
            *params = (GLint)(block->name ? strlen(block->name) + 1 : 1);
            break;
        case GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS:
            *params = 0;
            break;
        case GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER:
            *params = mglUniformBlockReferencedByStage(ptr, block, _VERTEX_SHADER);
            break;
        case GL_UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER:
            *params = mglUniformBlockReferencedByStage(ptr, block, _GEOMETRY_SHADER);
            break;
        case GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER:
            *params = mglUniformBlockReferencedByStage(ptr, block, _FRAGMENT_SHADER);
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}

void mglGetActiveUniformBlockName(GLMContext ctx, GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformBlockName)
{
    if (length) {
        *length = 0;
    }
    if (uniformBlockName && bufSize > 0) {
        uniformBlockName[0] = '\0';
    }
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = getProgram(ctx, program);
    if (!ptr || ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    SpirvResource *block = mglFindUniformBlockByIndex(ptr, uniformBlockIndex, NULL);
    if (!block) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    const char *src = block->name ? block->name : "";
    GLsizei src_len = (GLsizei)strlen(src);
    if (length) {
        *length = src_len;
    }
    if (uniformBlockName && bufSize > 0) {
        GLsizei copy_len = src_len < (bufSize - 1) ? src_len : (bufSize - 1);
        if (copy_len > 0) {
            memcpy(uniformBlockName, src, (size_t)copy_len);
        }
        uniformBlockName[copy_len] = '\0';
    }
}

void mglUniformBlockBinding(GLMContext ctx, GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding)
{
    // Accept as no-op for now; bindings are auto-mapped during shader compilation.
    (void)ctx;
    (void)program;
    (void)uniformBlockIndex;
    (void)uniformBlockBinding;
}

bool checkUniformParams(GLMContext ctx, GLint location)
{
    ctx = mglUniformResolveContext(ctx, __FUNCTION__);
    if (!ctx) {
        return false;
    }

    Program* ptr = ctx->state.program;
    
    if (!ptr) {
        mglUniformSetError(ctx, GL_INVALID_OPERATION);
        return false;
    }

    if (location < 0) {
        mglUniformSetError(ctx, GL_INVALID_OPERATION);
        return false;
    }
        
    if (location >= MAX_BINDABLE_BUFFERS) {
        mglUniformSetError(ctx, GL_INVALID_OPERATION);
        return false;
    }

    return true;
}

static bool checkUniformUploadParams(GLMContext ctx, GLint location, const void *ptr, GLsizei count, size_t element_size, GLsizeiptr *size_out)
{
    if (!checkUniformParams(ctx, location)) {
        return false;
    }

    if (count < 0) {
        ctx = mglUniformResolveContext(ctx, __FUNCTION__);
        mglUniformSetError(ctx, GL_INVALID_VALUE);
        return false;
    }

    size_t element_count = (size_t)count;
    if (element_size != 0 && element_count > (SIZE_MAX / element_size)) {
        ctx = mglUniformResolveContext(ctx, __FUNCTION__);
        mglUniformSetError(ctx, GL_OUT_OF_MEMORY);
        return false;
    }

    size_t total = element_count * element_size;
    if (total > (size_t)PTRDIFF_MAX) {
        ctx = mglUniformResolveContext(ctx, __FUNCTION__);
        mglUniformSetError(ctx, GL_OUT_OF_MEMORY);
        return false;
    }

    if (total > 0 && !mglPointerRangeReadable(ptr, total)) {
        fprintf(stderr,
                "MGL WARNING: dropping uniform update location=%d count=%d bytes=%zu unreadable value=%p\n",
                location,
                count,
                total,
                ptr);
        ctx = mglUniformResolveContext(ctx, __FUNCTION__);
        mglUniformSetError(ctx, GL_INVALID_VALUE);
        return false;
    }

    if (size_out) {
        *size_out = (GLsizeiptr)total;
    }
    return true;
}

void mglUniform(GLMContext ctx, GLint location, void *ptr, GLsizeiptr size)
{
    ctx = mglUniformResolveContext(ctx, __FUNCTION__);
    if (!ctx) {
        return;
    }
    if (!checkUniformParams(ctx, location)) {
        return;
    }
    if (size < 0) {
        mglUniformSetError(ctx, GL_INVALID_VALUE);
        return;
    }
    if (size > 0 && !mglPointerRangeReadable(ptr, (size_t)size)) {
        fprintf(stderr,
                "MGL WARNING: dropping uniform update location=%d bytes=%lld unreadable value=%p\n",
                location,
                (long long)size,
                ptr);
        mglUniformSetError(ctx, GL_INVALID_VALUE);
        return;
    }
    
    Program *program = ctx->state.program;
    if (!program) {
        mglUniformSetError(ctx, GL_INVALID_OPERATION);
        return;
    }

    BufferBaseTarget *uniformSlot = &program->plain_uniform_buffers[location];
    Buffer *buf = uniformSlot->buf;
    
    if(buf == NULL)
    {
        GLuint internalName = MGL_INTERNAL_UNIFORM_BUFFER_NAME_BASE |
                              (((GLuint)program->name & 0x0fffu) << 12) |
                              (GLuint)location;
        uniformSlot->buf = newBuffer(ctx, GL_UNIFORM_BUFFER, internalName);
        buf = uniformSlot->buf;
        if (buf) {
            insertHashElement(&ctx->state.buffer_table, internalName, buf);
        }
    }
    
    initBufferData(ctx, buf, size, ptr, true);
    uniformSlot->buffer = buf ? buf->name : 0u;
    uniformSlot->offset = 0;
    uniformSlot->size = size;

    /*
     * Minecraft's shader layer can reuse the same logical plain uniform values
     * across generated program variants. Keep the legacy global slot as a
     * fallback for programs that have not received an explicit upload yet, while
     * still preferring the per-program storage above when it exists.
     */
    BufferBaseTarget *globalSlot = &ctx->state.buffer_base[_UNIFORM_CONSTANT].buffers[location];
    if (!globalSlot->buf) {
        GLuint globalName = MGL_INTERNAL_UNIFORM_BUFFER_NAME_BASE |
                            0x00fff000u |
                            (GLuint)location;
        globalSlot->buf = newBuffer(ctx, GL_UNIFORM_BUFFER, globalName);
        if (globalSlot->buf) {
            insertHashElement(&ctx->state.buffer_table, globalName, globalSlot->buf);
        }
    }
    if (globalSlot->buf) {
        initBufferData(ctx, globalSlot->buf, size, ptr, true);
        globalSlot->buffer = globalSlot->buf->name;
        globalSlot->offset = 0;
        globalSlot->size = size;
    }

    ctx->state.dirty_bits |= DIRTY_BUFFER_BASE_STATE;
}

void mglUniform1d(GLMContext ctx, GLint location, GLdouble x)
{
    mglUniform(ctx, location, &x, sizeof(GLdouble));
}

void mglUniform1dv(GLMContext ctx, GLint location, GLsizei count, const GLdouble *value)
{
    mglUniform(ctx, location, (void *)value, count * sizeof(GLdouble));
}

void mglUniform1f(GLMContext ctx, GLint location, GLfloat v0)
{
    mglUniform(ctx, location, &v0, sizeof(GLfloat));
}

void mglUniform1fv(GLMContext ctx, GLint location, GLsizei count, const GLfloat *value)
{
    mglUniform(ctx, location, (void *)value, count * sizeof(GLfloat));
}

void mglUniform1i(GLMContext ctx, GLint location, GLint v0)
{
    if (mglSetSamplerUniformUnit(ctx, location, v0)) {
        return;
    }

    mglUniform(ctx, location, &v0, sizeof(GLint));
}

void mglUniform1iv(GLMContext ctx, GLint location, GLsizei count, const GLint *value)
{
    if (count > 0 && value && mglSetSamplerUniformUnit(ctx, location, value[0])) {
        return;
    }

    mglUniform(ctx, location, (void *)value, count * sizeof(GLint));
}

void mglUniform1ui(GLMContext ctx, GLint location, GLuint v0)
{
    if (v0 <= (GLuint)INT_MAX && mglSetSamplerUniformUnit(ctx, location, (GLint)v0)) {
        return;
    }

    mglUniform(ctx, location, &v0, sizeof(GLuint));
}

void mglUniform1uiv(GLMContext ctx, GLint location, GLsizei count, const GLuint *value)
{
    if (count > 0 && value && value[0] <= (GLuint)INT_MAX &&
        mglSetSamplerUniformUnit(ctx, location, (GLint)value[0])) {
        return;
    }

    mglUniform(ctx, location, (void *)value, count * sizeof(GLuint));
}

void mglUniform2d(GLMContext ctx, GLint location, volatile GLdouble x, volatile GLdouble y)
{
    GLdouble data[] = {x, y};
    
    mglUniform(ctx, location, data, 2 * sizeof(GLdouble));
}

void mglUniform2dv(GLMContext ctx, GLint location, GLsizei count, const GLdouble *value)
{
    mglUniform(ctx, location, (void *)value, 2 * count * sizeof(GLdouble));
}

void mglUniform2f(GLMContext ctx, GLint location, GLfloat v0, GLfloat v1)
{
    GLfloat data[] = {v0, v1};
    
    mglUniform(ctx, location, data, 2 * sizeof(GLfloat));
}

void mglUniform2fv(GLMContext ctx, GLint location, GLsizei count, const GLfloat *value)
{
    mglUniform(ctx, location, (void *)value, 2 * count * sizeof(GLfloat));
}

void mglUniform2i(GLMContext ctx, GLint location, GLint v0, GLint v1)
{
    GLint data[] = {v0, v1};
    
    mglUniform(ctx, location, data, 2 * sizeof(GLint));
}

void mglUniform2iv(GLMContext ctx, GLint location, GLsizei count, const GLint *value)
{
    mglUniform(ctx, location, (void *)value, 2 * count * sizeof(GLint));
}

void mglUniform2ui(GLMContext ctx, GLint location, GLuint v0, GLuint v1)
{
    GLuint data[] = {v0, v1};
    
    mglUniform(ctx, location, data, 2 * sizeof(GLuint));
}

void mglUniform2uiv(GLMContext ctx, GLint location, GLsizei count, const GLuint *value)
{
    mglUniform(ctx, location, (void *)value, 2 * count * sizeof(GLuint));
}

void mglUniform3d(GLMContext ctx, GLint location, GLdouble x, GLdouble y, GLdouble z)
{
    GLdouble data[] = {x, y, z};
    
    mglUniform(ctx, location, data, 3 * sizeof(GLdouble));
}

void mglUniform3dv(GLMContext ctx, GLint location, GLsizei count, const GLdouble *value)
{
    mglUniform(ctx, location, (void *)value, 3 * count * sizeof(GLdouble));
}

void mglUniform3f(GLMContext ctx, GLint location, GLfloat v0, GLfloat v1, GLfloat v2)
{
    GLfloat data[] = {v0, v1, v2};
    
    mglUniform(ctx, location, data, 3 * sizeof(GLfloat));
}

void mglUniform3fv(GLMContext ctx, GLint location, GLsizei count, const GLfloat *value)
{
    mglUniform(ctx, location, (void *)value, 3 * count * sizeof(GLfloat));
}

void mglUniform3i(GLMContext ctx, GLint location, GLint v0, GLint v1, GLint v2)
{
    GLint data[] = {v0, v1, v2};
    
    mglUniform(ctx, location, data, 3 * sizeof(GLfloat));
}

void mglUniform3iv(GLMContext ctx, GLint location, GLsizei count, const GLint *value)
{
    mglUniform(ctx, location, (void *)value, 3 * count * sizeof(GLint));
}

void mglUniform3ui(GLMContext ctx, GLint location, GLuint v0, GLuint v1, GLuint v2)
{
    GLuint data[] = {v0, v1, v2};
    
    mglUniform(ctx, location, (void *)data, 3 * sizeof(GLuint));
}

void mglUniform3uiv(GLMContext ctx, GLint location, GLsizei count, const GLuint *value)
{
    mglUniform(ctx, location, (void *)value, 3 * count * sizeof(GLuint));
}

void mglUniform4d(GLMContext ctx, GLint location, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
    GLdouble data[] = {x, y, z, w};
    
    mglUniform(ctx, location, data, 4 * sizeof(GLdouble));
}

void mglUniform4dv(GLMContext ctx, GLint location, GLsizei count, const GLdouble *value)
{
    mglUniform(ctx, location, (void *)value, 4 * count * sizeof(GLdouble));
}

void mglUniform4f(GLMContext ctx, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
{
    GLfloat data[] = {v0, v1, v2, v3};
    
    mglUniform(ctx, location, (void *)data, 4 * sizeof(GLfloat));
}

void mglUniform4fv(GLMContext ctx, GLint location, GLsizei count, const GLfloat *value)
{
    mglUniform(ctx, location, (void *)value, 4 * count * sizeof(GLfloat));
}

void mglUniform4i(GLMContext ctx, GLint location, GLint v0, GLint v1, GLint v2, GLint v3)
{
    GLint data[] = {v0, v1, v2, v3};
    
    mglUniform(ctx, location, data, 4 * sizeof(GLint));
}

void mglUniform4iv(GLMContext ctx, GLint location, GLsizei count, const GLint *value)
{
    mglUniform(ctx, location, (void *)value, 4 * count * sizeof(GLint));
}

void mglUniform4ui(GLMContext ctx, GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3)
{
    GLuint data[] = {v0, v1, v2, v3};
    
    mglUniform(ctx, location, data, 4 * sizeof(GLuint));
}

void mglUniform4uiv(GLMContext ctx, GLint location, GLsizei count, const GLuint *value)
{
    mglUniform(ctx, location, (void *)value, 4 * count * sizeof(GLuint));
}


// Macro to define matrix types
#define DEFINE_MATRIX_TYPE(_type_, _rows_, _cols_, _name_) \
typedef struct { \
    _type_ d[_rows_][_cols_]; \
} _name_;

// Macro to define transpose functions
#define DEFINE_TRANSPOSE_FUNC(_type_, _rows_, _cols_, _name_, _transposed_name_) \
void _name_##Transpose (const _name_ *matrix, _transposed_name_ *result) { \
    for (int i = 0; i < _rows_; i++) { \
        for (int j = 0; j < _cols_; j++) { \
            result->d[j][i] = matrix->d[i][j]; \
        } \
    } \
}

// Generalized function for uniform matrix upload
#define HANDLE_MATRIX_TRANSPOSE(_type_, _src_type_, _dst_type_, _transpose_func_) \
    ctx = mglUniformResolveContext(ctx, __FUNCTION__); \
    if (!ctx) { \
        return; \
    } \
    GLsizeiptr uniformBytes = 0; \
    if (!checkUniformUploadParams(ctx, location, value, count, sizeof(_src_type_), &uniformBytes)) { \
        return; \
    } \
    if (transpose) { \
        const _src_type_ *src = (const _src_type_ *)value; \
        /* CRITICAL SECURITY FIX: Prevent integer overflow in uniform matrix allocation */ \
        if (count > SIZE_MAX / sizeof(_dst_type_)) { \
            fprintf(stderr, "MGL SECURITY ERROR: Uniform matrix count %d would cause allocation overflow\n", count); \
            STATE(error) = GL_OUT_OF_MEMORY; \
            return; \
        } \
        size_t alloc_size = count * sizeof(_dst_type_); \
        _dst_type_ *dst = (_dst_type_ *)malloc(alloc_size); \
        if (!dst) { \
            fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate %zu bytes for uniform matrix\n", alloc_size); \
            STATE(error) = GL_OUT_OF_MEMORY; \
            return; \
        } \
        for (int i = 0; i < count; i++) { \
            _transpose_func_(&src[i], &dst[i]); \
        } \
        mglUniform(ctx, location, (void *)dst, count * sizeof(_dst_type_)); \
        free(dst); \
    } else { \
        mglUniform(ctx, location, (void *)value, uniformBytes); \
    }

DEFINE_MATRIX_TYPE(GLdouble, 2, 2, Mat2x2dv)       // 2x2 matrix type
DEFINE_MATRIX_TYPE(GLdouble, 2, 2, Mat2x2dvTrans) // Transposed matrix type (same dimensions for 2x2)
DEFINE_TRANSPOSE_FUNC(GLdouble, 2, 2, Mat2x2dv, Mat2x2dvTrans)

void mglUniformMatrix2dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat2x2dv,          // Source matrix type
                            Mat2x2dvTrans,     // Destination matrix type
                            Mat2x2dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 2, 2, Mat2x2fv)       // 2x2 matrix type
DEFINE_MATRIX_TYPE(GLfloat, 2, 2, Mat2x2fvTrans) // Transposed matrix type (same dimensions for 2x2)
DEFINE_TRANSPOSE_FUNC(GLfloat, 2, 2, Mat2x2fv, Mat2x2fvTrans)

void mglUniformMatrix2fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat2x2fv,          // Source matrix type
                            Mat2x2fvTrans,     // Destination matrix type
                            Mat2x2fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 2, 3, Mat2x3dv)       // 2x3 matrix type
DEFINE_MATRIX_TYPE(GLdouble, 2, 3, Mat2x3dvTrans) // Transposed matrix type (same dimensions for 2x3)
DEFINE_TRANSPOSE_FUNC(GLdouble, 2, 3, Mat2x3dv, Mat2x3dvTrans)

void mglUniformMatrix2x3dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,         // Element type
                            Mat2x3dv,          // Source matrix type
                            Mat2x3dvTrans,     // Destination matrix type
                            Mat2x3dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 2, 3, Mat2x3fv)       // 2x3 matrix type
DEFINE_MATRIX_TYPE(GLfloat, 2, 3, Mat2x3fvTrans) // Transposed matrix type (same dimensions for 2x3)
DEFINE_TRANSPOSE_FUNC(GLfloat, 2, 3, Mat2x3fv, Mat2x3fvTrans)

void mglUniformMatrix2x3fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat2x3fv,          // Source matrix type
                            Mat2x3fvTrans,     // Destination matrix type
                            Mat2x3fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 2, 4, Mat2x4dv)       // 2x4 matrix type
DEFINE_MATRIX_TYPE(GLdouble, 2, 4, Mat2x4dvTrans) // Transposed matrix type (same dimensions for 2x4)
DEFINE_TRANSPOSE_FUNC(GLdouble, 2, 4, Mat2x4dv, Mat2x4dvTrans)

void mglUniformMatrix2x4dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat2x4dv,          // Source matrix type
                            Mat2x4dvTrans,     // Destination matrix type
                            Mat2x4dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 2, 4, Mat2x4fv)       // 2x4 matrix type
DEFINE_MATRIX_TYPE(GLfloat, 2, 4, Mat2x4fvTrans) // Transposed matrix type (same dimensions for 2x4)
DEFINE_TRANSPOSE_FUNC(GLfloat, 2, 4, Mat2x4fv, Mat2x4fvTrans)

void mglUniformMatrix2x4fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat2x4fv,          // Source matrix type
                            Mat2x4fvTrans,     // Destination matrix type
                            Mat2x4fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 3, 3, Mat3x3dv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLdouble, 3, 3, Mat3x3dvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLdouble, 3, 3, Mat3x3dv, Mat3x3dvTrans)

void mglUniformMatrix3dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat3x3dv,          // Source matrix type
                            Mat3x3dvTrans,     // Destination matrix type
                            Mat3x3dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 3, 3, Mat3x3fv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLfloat, 3, 3, Mat3x3fvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLfloat, 3, 3, Mat3x3fv, Mat3x3fvTrans)

void mglUniformMatrix3fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat3x3fv,          // Source matrix type
                            Mat3x3fvTrans,     // Destination matrix type
                            Mat3x3fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 3, 2, Mat3x2dv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLdouble, 3, 2, Mat3x2dvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLdouble, 3, 2, Mat3x2dv, Mat3x2dvTrans)

void mglUniformMatrix3x2dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat3x2dv,          // Source matrix type
                            Mat3x2dvTrans,     // Destination matrix type
                            Mat3x2dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 3, 2, Mat3x2fv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLfloat, 3, 2, Mat3x2fvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLfloat, 3, 2, Mat3x2fv, Mat3x2fvTrans)

void mglUniformMatrix3x2fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat3x2fv,          // Source matrix type
                            Mat3x2fvTrans,     // Destination matrix type
                            Mat3x2fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 3, 4, Mat3x4dv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLdouble, 3, 4, Mat3x4dvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLdouble, 3, 4, Mat3x4dv, Mat3x4dvTrans)

void mglUniformMatrix3x4dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat3x4dv,          // Source matrix type
                            Mat3x4dvTrans,     // Destination matrix type
                            Mat3x4dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 3, 4, Mat3x4fv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLfloat, 3, 4, Mat3x4fvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLfloat, 3, 4, Mat3x4fv, Mat3x4fvTrans)

void mglUniformMatrix3x4fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat3x4fv,          // Source matrix type
                            Mat3x4fvTrans,     // Destination matrix type
                            Mat3x4fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 4, 4, Mat4x4dv)
DEFINE_MATRIX_TYPE(GLdouble, 4, 4, Mat4x4dvTrans)
DEFINE_TRANSPOSE_FUNC(GLdouble, 4, 4, Mat4x4dv, Mat4x4dvTrans)

void mglUniformMatrix4dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat4x4dv,          // Source matrix type
                            Mat4x4dvTrans,     // Destination matrix type
                            Mat4x4dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 4, 4, Mat4x4fv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLfloat, 4, 4, Mat4x4fvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLfloat, 4, 4, Mat4x4fv, Mat4x4fvTrans)

void mglUniformMatrix4fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat4x4fv,          // Source matrix type
                            Mat4x4fvTrans,     // Destination matrix type
                            Mat4x4fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 4, 2, Mat4x2dv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLdouble, 4, 2, Mat4x2dvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLdouble, 4, 2, Mat4x2dv, Mat4x2dvTrans)

void mglUniformMatrix4x2dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat4x4dv,          // Source matrix type
                            Mat4x4dvTrans,     // Destination matrix type
                            Mat4x4dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 4, 2, Mat4x2fv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLfloat, 4, 2, Mat4x2fvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLfloat, 4, 2, Mat4x2fv, Mat4x2fvTrans)

void mglUniformMatrix4x2fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat4x2fv,          // Source matrix type
                            Mat4x2fvTrans,     // Destination matrix type
                            Mat4x2fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 4, 3, Mat4x3dv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLdouble, 4, 3, Mat4x3dvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLdouble, 4, 3, Mat4x3dv, Mat4x3dvTrans)

void mglUniformMatrix4x3dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat4x3dv,          // Source matrix type
                            Mat4x3dvTrans,     // Destination matrix type
                            Mat4x3dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 4, 3, Mat4x3fv)       // 3x3 matrix type
DEFINE_MATRIX_TYPE(GLfloat, 4, 3, Mat4x3fvTrans) // Transposed matrix type (same dimensions for 3x3)
DEFINE_TRANSPOSE_FUNC(GLfloat, 4, 3, Mat4x3fv, Mat4x3fvTrans)

void mglUniformMatrix4x3fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat4x3fv,          // Source matrix type
                            Mat4x3fvTrans,     // Destination matrix type
                            Mat4x3fvTranspose  // Transpose function
        );
}
