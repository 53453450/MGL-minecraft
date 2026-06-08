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
#include "mgl_safety.h"
#include "draw_command.h"

#pragma mark uniforms

#define MGL_INTERNAL_UNIFORM_BUFFER_NAME_BASE 0xf0000000u
#define MGL_SAFE_SPIRV_RESOURCE_MAX 4096u
#define MGL_SYNTHETIC_SAMPLER_LOCATION_BASE 0x4000

extern void mglTraceLogExternal(const char *fmt, ...);

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
    GLMContext current = MGLgetCurrentContext();
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

#define MGL_SAFE_CSTRING_MAX 4096u

static GLboolean mglSafeCStringLength(const char *str, size_t *length_out)
{
    if (length_out) {
        *length_out = 0u;
    }
    if (!str) {
        return GL_FALSE;
    }

    for (size_t i = 0; i < MGL_SAFE_CSTRING_MAX; i++) {
        if (!mglPointerRangeReadable(str + i, 1u)) {
            return GL_FALSE;
        }
        if (str[i] == '\0') {
            if (length_out) {
                *length_out = i;
            }
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static GLboolean mglSafeCStringEquals(const char *lhs, const char *rhs)
{
    if (!lhs || !rhs) {
        return GL_FALSE;
    }

    for (size_t i = 0; i < MGL_SAFE_CSTRING_MAX; i++) {
        if (!mglPointerRangeReadable(lhs + i, 1u) ||
            !mglPointerRangeReadable(rhs + i, 1u)) {
            return GL_FALSE;
        }

        char a = lhs[i];
        char b = rhs[i];
        if (a != b) {
            return GL_FALSE;
        }
        if (a == '\0') {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static GLboolean mglSafeCStringContains(const char *haystack, const char *needle)
{
    size_t hay_len = 0u;
    size_t needle_len = 0u;

    if (!mglSafeCStringLength(haystack, &hay_len) ||
        !mglSafeCStringLength(needle, &needle_len)) {
        return GL_FALSE;
    }
    if (needle_len == 0u) {
        return GL_TRUE;
    }
    if (needle_len > hay_len) {
        return GL_FALSE;
    }

    for (size_t i = 0; i <= hay_len - needle_len; i++) {
        GLboolean match = GL_TRUE;
        for (size_t j = 0; j < needle_len; j++) {
            if (haystack[i + j] != needle[j]) {
                match = GL_FALSE;
                break;
            }
        }
        if (match) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static const char *mglSafeCStringForLog(const char *str)
{
    return mglSafeCStringLength(str, NULL) ? str : "(invalid)";
}

static SpirvResourceList *mglUniformSafeResourceList(Program *program, int stage, int res_type, const char *func)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        res_type < 0 || res_type >= _MAX_SPIRV_RES) {
        return NULL;
    }

    SpirvResourceList *resources = &program->spirv_resources_list[stage][res_type];
    if (!mglPointerRangeReadable(resources, sizeof(*resources))) {
        fprintf(stderr,
                "MGL WARNING: %s dropping unreadable resource list program=%p stage=%d type=%d\n",
                func ? func : "uniform",
                (void *)program,
                stage,
                res_type);
        return NULL;
    }

    if (resources->count == 0u) {
        return resources;
    }
    if (resources->count > MGL_SAFE_SPIRV_RESOURCE_MAX) {
        fprintf(stderr,
                "MGL WARNING: %s dropping suspicious resource list program=%p stage=%d type=%d count=%u\n",
                func ? func : "uniform",
                (void *)program,
                stage,
                res_type,
                resources->count);
        return NULL;
    }
    size_t resource_bytes = (size_t)resources->count * sizeof(SpirvResource);
    if (!resources->list ||
        !mglPointerRangeReadable(resources->list, resource_bytes)) {
        fprintf(stderr,
                "MGL WARNING: %s dropping unreadable resource storage program=%p stage=%d type=%d count=%u list=%p\n",
                func ? func : "uniform",
                (void *)program,
                stage,
                res_type,
                resources->count,
                (void *)resources->list);
        return NULL;
    }

    return resources;
}

static Program *mglUniformValidateProgramPointer(GLMContext ctx, Program *program, const char *func)
{
    if (!ctx || !program) {
        return NULL;
    }

    GLboolean pointerReadable =
        mglObjectPointerLooksPlausible(program) &&
        mglPointerRangeReadable(program, sizeof(*program));
    GLuint expectedName = pointerReadable ? program->name : 0u;

    if (!pointerReadable ||
        !mglProgramPointerUsableForName(ctx, program, expectedName)) {
        fprintf(stderr,
                "MGL WARNING: %s dropping invalid program pointer %p\n",
                func ? func : "uniform",
                (void *)program);
        if (ctx->state.program == program) {
            ctx->state.program = NULL;
            ctx->state.program_name = 0;
            ctx->state.var.current_program = 0;
        }
        return NULL;
    }

    return program;
}

static Program *mglUniformGetCurrentProgram(GLMContext ctx, const char *func)
{
    if (!ctx) {
        return NULL;
    }
    return mglUniformValidateProgramPointer(ctx, ctx->state.program, func);
}

static Program *mglUniformGetNamedProgram(GLMContext ctx, GLuint program, const char *func)
{
    if (!ctx || program == 0u) {
        return NULL;
    }
    return mglUniformValidateProgramPointer(ctx, getProgram(ctx, program), func);
}

static GLint mglKnownPlainUniformLocation(const char *name)
{
    if (!name) {
        return -1;
    }

    if (!mglSafeCStringLength(name, NULL)) {
        return -1;
    }

    if (mglSafeCStringEquals(name, "ModelViewMat")) {
        return 0;
    }
    if (mglSafeCStringEquals(name, "ProjMat")) {
        return 1;
    }
    if (mglSafeCStringEquals(name, "TextureMat")) {
        return 2;
    }
    if (mglSafeCStringEquals(name, "ColorModulator")) {
        return 3;
    }
    if (mglSafeCStringEquals(name, "FogStart")) {
        return 4;
    }
    if (mglSafeCStringEquals(name, "FogEnd")) {
        return 5;
    }
    if (mglSafeCStringEquals(name, "FogColor")) {
        return 6;
    }
    if (mglSafeCStringEquals(name, "FogShape")) {
        return 7;
    }
    if (mglSafeCStringEquals(name, "GameTime")) {
        return 8;
    }
    if (mglSafeCStringEquals(name, "ScreenSize")) {
        return 9;
    }
    if (mglSafeCStringEquals(name, "LineWidth")) {
        return 10;
    }
    if (mglSafeCStringEquals(name, "IViewRotMat")) {
        return 11;
    }
    if (mglSafeCStringEquals(name, "ChunkOffset")) {
        return 12;
    }
    if (mglSafeCStringEquals(name, "u_ProjectionMatrix")) {
        return 0;
    }
    if (mglSafeCStringEquals(name, "u_ModelViewMatrix")) {
        return 1;
    }
    if (mglSafeCStringEquals(name, "u_RegionOffset")) {
        return 2;
    }
    if (mglSafeCStringEquals(name, "u_TexCoordShrink")) {
        return 3;
    }
    if (mglSafeCStringEquals(name, "u_FogColor")) {
        return 4;
    }
    if (mglSafeCStringEquals(name, "u_EnvironmentFog")) {
        return 5;
    }
    if (mglSafeCStringEquals(name, "u_RenderFog")) {
        return 6;
    }

    return -1;
}

static GLboolean mglUniformNameLooksSamplerLike(const char *name)
{
    if (!name || !mglSafeCStringLength(name, NULL)) {
        return GL_FALSE;
    }

    return (mglSafeCStringContains(name, "Sampler") ||
            mglSafeCStringEquals(name, "CloudFaces")) ? GL_TRUE : GL_FALSE;
}

static GLboolean mglUniformResourceLooksSamplerLike(const SpirvResource *res, int res_type)
{
    if (!res) {
        return GL_FALSE;
    }

    switch (res_type) {
        case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS:
        case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
            return GL_TRUE;
        case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
            return (res->image_dim != 0u ||
                    res->uniform_location >= MGL_SYNTHETIC_SAMPLER_LOCATION_BASE ||
                    mglUniformNameLooksSamplerLike(res->name)) ? GL_TRUE : GL_FALSE;
        default:
            return GL_FALSE;
    }
}

static const int mglActiveUniformResourceTypes[] = {
    SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
    SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
    SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
    SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
    SPVC_RESOURCE_TYPE_STORAGE_IMAGE
};

static GLboolean mglActiveUniformResourceHasName(const SpirvResource *res)
{
    return res && mglSafeCStringLength(res->name, NULL);
}

static GLboolean mglActiveUniformNamesMatch(const char *resource_name, const char *query_name)
{
    size_t resource_len = 0u;
    size_t query_len = 0u;

    if (!mglSafeCStringLength(resource_name, &resource_len) ||
        !mglSafeCStringLength(query_name, &query_len)) {
        return GL_FALSE;
    }

    if (resource_len == query_len && memcmp(resource_name, query_name, resource_len) == 0) {
        return GL_TRUE;
    }

    if (query_len == resource_len + 3u &&
        query_name[resource_len] == '[' &&
        query_name[resource_len + 1u] == '0' &&
        query_name[resource_len + 2u] == ']' &&
        memcmp(resource_name, query_name, resource_len) == 0) {
        return GL_TRUE;
    }

    if (resource_len == query_len + 3u &&
        resource_name[query_len] == '[' &&
        resource_name[query_len + 1u] == '0' &&
        resource_name[query_len + 2u] == ']' &&
        memcmp(resource_name, query_name, query_len) == 0) {
        return GL_TRUE;
    }

    return GL_FALSE;
}

static GLboolean mglActiveUniformNameSeenBefore(Program *program,
                                                GLint target_type_ordinal,
                                                int target_stage,
                                                GLuint target_index,
                                                const char *name)
{
    if (!program || target_type_ordinal < 0 || !mglSafeCStringLength(name, NULL)) {
        return GL_FALSE;
    }

    for (GLint type_ordinal = 0;
         type_ordinal <= target_type_ordinal &&
         type_ordinal < (GLint)(sizeof(mglActiveUniformResourceTypes) / sizeof(mglActiveUniformResourceTypes[0]));
         type_ordinal++) {
        int res_type = mglActiveUniformResourceTypes[type_ordinal];
        for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
            SpirvResourceList *resources = mglUniformSafeResourceList(program, stage, res_type, __FUNCTION__);
            if (!resources) {
                continue;
            }
            for (GLuint i = 0; i < resources->count; i++) {
                if (type_ordinal == target_type_ordinal &&
                    stage == target_stage &&
                    i >= target_index) {
                    return GL_FALSE;
                }
                if (mglActiveUniformNamesMatch(resources->list[i].name, name)) {
                    return GL_TRUE;
                }
            }
        }
    }

    return GL_FALSE;
}

SpirvResource *mglProgramActiveUniformAt(Program *program, GLuint index, int *stage_out, int *res_type_out)
{
    if (stage_out) {
        *stage_out = -1;
    }
    if (res_type_out) {
        *res_type_out = -1;
    }
    if (!program) {
        return NULL;
    }

    GLuint ordinal = 0u;
    const size_t type_count = sizeof(mglActiveUniformResourceTypes) / sizeof(mglActiveUniformResourceTypes[0]);
    for (size_t type_ordinal = 0; type_ordinal < type_count; type_ordinal++) {
        int res_type = mglActiveUniformResourceTypes[type_ordinal];
        for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
            SpirvResourceList *resources = mglUniformSafeResourceList(program, stage, res_type, __FUNCTION__);
            if (!resources) {
                continue;
            }
            for (GLuint i = 0; i < resources->count; i++) {
                SpirvResource *res = &resources->list[i];
                if (!mglActiveUniformResourceHasName(res) ||
                    mglActiveUniformNameSeenBefore(program, (GLint)type_ordinal, stage, i, res->name)) {
                    continue;
                }
                if (ordinal == index) {
                    if (stage_out) {
                        *stage_out = stage;
                    }
                    if (res_type_out) {
                        *res_type_out = res_type;
                    }
                    return res;
                }
                ordinal++;
            }
        }
    }

    return NULL;
}

GLint mglProgramActiveUniformCount(Program *program)
{
    if (!program) {
        return 0;
    }

    GLint count = 0;
    while (mglProgramActiveUniformAt(program, (GLuint)count, NULL, NULL)) {
        count++;
    }
    return count;
}

GLint mglProgramActiveUniformIndexByName(Program *program, const GLchar *name)
{
    if (!program || !mglSafeCStringLength(name, NULL)) {
        return -1;
    }

    GLint count = mglProgramActiveUniformCount(program);
    for (GLint i = 0; i < count; i++) {
        SpirvResource *res = mglProgramActiveUniformAt(program, (GLuint)i, NULL, NULL);
        if (res && mglActiveUniformNamesMatch(res->name, name)) {
            return i;
        }
    }

    return -1;
}

static GLint mglKnownPlainUniformType(const char *name)
{
    if (!name || !mglSafeCStringLength(name, NULL)) {
        return GL_FLOAT;
    }

    if (mglSafeCStringEquals(name, "ModelViewMat") ||
        mglSafeCStringEquals(name, "ProjMat") ||
        mglSafeCStringEquals(name, "TextureMat") ||
        mglSafeCStringEquals(name, "u_ProjectionMatrix") ||
        mglSafeCStringEquals(name, "u_ModelViewMatrix")) {
        return GL_FLOAT_MAT4;
    }
    if (mglSafeCStringEquals(name, "IViewRotMat")) {
        return GL_FLOAT_MAT3;
    }
    if (mglSafeCStringEquals(name, "ColorModulator") ||
        mglSafeCStringEquals(name, "FogColor") ||
        mglSafeCStringEquals(name, "u_FogColor") ||
        mglSafeCStringEquals(name, "u_TexCoordShrink")) {
        return GL_FLOAT_VEC4;
    }
    if (mglSafeCStringEquals(name, "ScreenSize")) {
        return GL_FLOAT_VEC2;
    }
    if (mglSafeCStringEquals(name, "ChunkOffset") ||
        mglSafeCStringEquals(name, "u_RegionOffset")) {
        return GL_FLOAT_VEC3;
    }
    if (mglSafeCStringEquals(name, "FogShape") ||
        mglSafeCStringEquals(name, "u_EnvironmentFog") ||
        mglSafeCStringEquals(name, "u_RenderFog")) {
        return GL_INT;
    }

    return GL_FLOAT;
}

static GLint mglSamplerUniformGLType(const SpirvResource *res, int res_type)
{
    if (!res) {
        return 0;
    }

    if (res_type == SPVC_RESOURCE_TYPE_STORAGE_IMAGE) {
        return (res->image_dim == 5u) ? GL_INT_IMAGE_BUFFER : GL_INT_IMAGE_2D;
    }

    if (res_type == SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS) {
        return GL_SAMPLER_2D;
    }

    if (res_type == SPVC_RESOURCE_TYPE_SAMPLED_IMAGE ||
        res_type == SPVC_RESOURCE_TYPE_SEPARATE_IMAGE ||
        mglUniformResourceLooksSamplerLike(res, res_type)) {
        switch (res->image_dim) {
            case 0: return res->image_arrayed ? GL_SAMPLER_1D_ARRAY : GL_SAMPLER_1D;
            case 1: return res->image_arrayed ? GL_SAMPLER_2D_ARRAY : GL_SAMPLER_2D;
            case 2: return GL_SAMPLER_3D;
            case 3: return res->image_arrayed ? GL_SAMPLER_CUBE_MAP_ARRAY : GL_SAMPLER_CUBE;
            case 5: return GL_INT_SAMPLER_BUFFER;
            default: return GL_SAMPLER_2D;
        }
    }

    return 0;
}

GLint mglProgramActiveUniformGLType(const SpirvResource *res, int res_type)
{
    GLint sampler_type = mglSamplerUniformGLType(res, res_type);
    if (sampler_type != 0) {
        return sampler_type;
    }
    return mglKnownPlainUniformType(res ? res->name : NULL);
}

GLint mglProgramActiveUniformSize(const SpirvResource *res, int res_type)
{
    (void)res;
    (void)res_type;
    return 1;
}

GLsizei mglProgramActiveUniformNameLength(const SpirvResource *res)
{
    size_t len = 0u;
    if (!res || !mglSafeCStringLength(res->name, &len)) {
        return 0;
    }
    return (GLsizei)len;
}

GLint mglProgramActiveUniformMaxNameLength(Program *program)
{
    GLint max_len = 0;
    GLint count = mglProgramActiveUniformCount(program);
    for (GLint i = 0; i < count; i++) {
        SpirvResource *res = mglProgramActiveUniformAt(program, (GLuint)i, NULL, NULL);
        GLsizei len = mglProgramActiveUniformNameLength(res);
        if ((GLint)len + 1 > max_len) {
            max_len = (GLint)len + 1;
        }
    }
    return max_len;
}

void mglProgramCopyActiveUniformName(const SpirvResource *res, GLsizei bufSize, GLsizei *length, GLchar *name)
{
    GLsizei src_len = mglProgramActiveUniformNameLength(res);
    const char *src = (src_len > 0 && res) ? res->name : "";

    if (length) {
        *length = src_len;
    }
    if (name && bufSize > 0) {
        GLsizei copy_len = src_len < (bufSize - 1) ? src_len : (bufSize - 1);
        if (copy_len > 0) {
            memcpy(name, src, (size_t)copy_len);
        }
        name[copy_len] = '\0';
    }
}

static GLint mglPlainUniformResourceLocation(const SpirvResource *res)
{
    if (!res) {
        return -1;
    }

    if (mglUniformResourceLooksSamplerLike(res, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT)) {
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
    if (mglUniformResourceLooksSamplerLike(res, res_type)) {
        return (res->uniform_location >= 0 && res->uniform_location == location) ? GL_TRUE : GL_FALSE;
    }

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

    return (GLint)res->gl_binding == location ? GL_TRUE : GL_FALSE;
}

static SpirvResource *mglFindSamplerUniformResource(Program *program,
                                                    GLint location,
                                                    int *stage_out,
                                                    GLuint *metal_binding_out)
{
    static const int sampler_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    if (stage_out) {
        *stage_out = -1;
    }
    if (metal_binding_out) {
        *metal_binding_out = 0u;
    }

    if (!program || location < 0) {
        return NULL;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (size_t rt = 0; rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); rt++) {
            int res_type = sampler_resource_types[rt];
            SpirvResourceList *resources = mglUniformSafeResourceList(program, stage, res_type, __FUNCTION__);
            if (!resources) {
                continue;
            }
            for (GLuint i = 0; i < resources->count; i++) {
                SpirvResource *res = &resources->list[i];
                if (!mglUniformResourceLooksSamplerLike(res, res_type)) {
                    continue;
                }
                if (!mglUniformLocationMatchesResource(res, res_type, location)) {
                    continue;
                }

                if (stage_out) {
                    *stage_out = stage;
                }
                if (metal_binding_out) {
                    *metal_binding_out = res->binding;
                }
                return res;
            }
        }
    }

    return NULL;
}

static GLboolean mglSamplerResourcesShareUniform(const SpirvResource *a, int a_type,
                                                 const SpirvResource *b, int b_type)
{
    if (!a || !b ||
        !mglUniformResourceLooksSamplerLike(a, a_type) ||
        !mglUniformResourceLooksSamplerLike(b, b_type)) {
        return GL_FALSE;
    }

    if (mglSafeCStringLength(a->name, NULL) &&
        mglSafeCStringLength(b->name, NULL) &&
        mglSafeCStringEquals(a->name, b->name)) {
        return GL_TRUE;
    }

    if (a->uniform_location >= 0 &&
        b->uniform_location >= 0 &&
        a->uniform_location == b->uniform_location) {
        return GL_TRUE;
    }

    return GL_FALSE;
}

static GLboolean mglSamplerResourceMatchesUniformWrite(SpirvResource *res,
                                                       int res_type,
                                                       GLint location,
                                                       SpirvResource *primary,
                                                       int primary_type)
{
    if (!res || !mglUniformResourceLooksSamplerLike(res, res_type)) {
        return GL_FALSE;
    }

    if (mglUniformLocationMatchesResource(res, res_type, location)) {
        return GL_TRUE;
    }

    return mglSamplerResourcesShareUniform(res, res_type, primary, primary_type);
}

static GLboolean mglMetalSamplerSlotSharedAcrossResources(Program *program, GLuint metal_binding)
{
    static const int sampler_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    if (!program || metal_binding >= TEXTURE_UNITS) {
        return GL_FALSE;
    }

    unsigned hits = 0u;
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (size_t rt = 0; rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); rt++) {
            int res_type = sampler_resource_types[rt];
            SpirvResourceList *resources = mglUniformSafeResourceList(program, stage, res_type, __FUNCTION__);
            if (!resources) {
                continue;
            }

            for (GLuint i = 0; i < resources->count; i++) {
                SpirvResource *res = &resources->list[i];
                if (res->binding != metal_binding ||
                    !mglUniformResourceLooksSamplerLike(res, res_type)) {
                    continue;
                }

                if (++hits > 1u) {
                    return GL_TRUE;
                }
            }
        }
    }

    return GL_FALSE;
}

static GLboolean mglSetSamplerUniformUnit(GLMContext ctx, GLint location, GLint unit)
{
    static const int sampler_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    ctx = mglUniformResolveContext(ctx, __FUNCTION__);
    if (!ctx || location < 0) {
        return GL_FALSE;
    }

    Program *program = mglUniformGetCurrentProgram(ctx, __FUNCTION__);
    if (!program) {
        return GL_FALSE;
    }

    GLboolean matched = GL_FALSE;
    GLboolean needs_update = GL_FALSE;
    const char *firstMatchedName = NULL;
    GLuint firstMatchedBinding = 0u;
    int firstMatchedStage = -1;
    SpirvResource *primaryResource = NULL;
    int primaryResourceType = -1;

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (size_t rt = 0; rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); rt++) {
            int res_type = sampler_resource_types[rt];
            SpirvResourceList *resources = mglUniformSafeResourceList(program, stage, res_type, __FUNCTION__);
            if (!resources) {
                continue;
            }

            for (GLuint i = 0; i < resources->count; i++) {
                SpirvResource *res = &resources->list[i];
                if (!mglUniformResourceLooksSamplerLike(res, res_type) ||
                    !mglUniformLocationMatchesResource(res, res_type, location)) {
                    continue;
                }

                matched = GL_TRUE;
                if (!firstMatchedName) {
                    firstMatchedName = res->name;
                    firstMatchedBinding = res->binding;
                    firstMatchedStage = stage;
                }
                if (!primaryResource) {
                    primaryResource = res;
                    primaryResourceType = res_type;
                }
            }
        }
    }

    if (!matched) {
        return GL_FALSE;
    }

    if (unit < 0 || unit >= TEXTURE_UNITS) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, GL_TRUE);
    }

    GLuint matchedResourceCount = 0u;
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (size_t rt = 0; rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); rt++) {
            int res_type = sampler_resource_types[rt];
            SpirvResourceList *resources = mglUniformSafeResourceList(program, stage, res_type, __FUNCTION__);
            if (!resources) {
                continue;
            }

            for (GLuint i = 0; i < resources->count; i++) {
                SpirvResource *res = &resources->list[i];
                if (!mglSamplerResourceMatchesUniformWrite(res,
                                                           res_type,
                                                           location,
                                                           primaryResource,
                                                           primaryResourceType)) {
                    continue;
                }

                matchedResourceCount++;
                if (res->sampler_unit != unit || !res->sampler_unit_explicit) {
                    needs_update = GL_TRUE;
                } else if (res->binding < TEXTURE_UNITS) {
                    GLboolean shared_slot =
                        mglMetalSamplerSlotSharedAcrossResources(program, res->binding);
                    if (program->sampler_units_by_stage[stage][res->binding] != unit ||
                        !program->sampler_units_explicit_by_stage[stage][res->binding] ||
                        (!shared_slot &&
                         (program->sampler_units[res->binding] != unit ||
                          !program->sampler_units_explicit[res->binding]))) {
                        needs_update = GL_TRUE;
                    }
                }
            }
        }
    }

    if (needs_update) {
        mglFlushPendingDraws(ctx);
    }

    GLboolean changed = GL_FALSE;
    GLboolean explicit_changed = GL_FALSE;
    GLuint updatedResourceCount = 0u;

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (size_t rt = 0; rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); rt++) {
            int res_type = sampler_resource_types[rt];
            SpirvResourceList *resources = mglUniformSafeResourceList(program, stage, res_type, __FUNCTION__);
            if (!resources) {
                continue;
            }

            for (GLuint i = 0; i < resources->count; i++) {
                SpirvResource *res = &resources->list[i];
                if (!mglSamplerResourceMatchesUniformWrite(res,
                                                           res_type,
                                                           location,
                                                           primaryResource,
                                                           primaryResourceType)) {
                    continue;
                }

                updatedResourceCount++;
                if (res->sampler_unit != unit) {
                    changed = GL_TRUE;
                    res->sampler_unit = unit;
                }
                if (!res->sampler_unit_explicit) {
                    explicit_changed = GL_TRUE;
                    res->sampler_unit_explicit = GL_TRUE;
                }

                if (res->binding < TEXTURE_UNITS) {
                    GLboolean shared_slot =
                        mglMetalSamplerSlotSharedAcrossResources(program, res->binding);
                    if (!shared_slot && program->sampler_units[res->binding] != unit) {
                        changed = GL_TRUE;
                        program->sampler_units[res->binding] = unit;
                    }
                    if (!shared_slot && !program->sampler_units_explicit[res->binding]) {
                        explicit_changed = GL_TRUE;
                        program->sampler_units_explicit[res->binding] = GL_TRUE;
                    }
                    if (program->sampler_units_by_stage[stage][res->binding] != unit) {
                        changed = GL_TRUE;
                        program->sampler_units_by_stage[stage][res->binding] = unit;
                    }
                    if (!program->sampler_units_explicit_by_stage[stage][res->binding]) {
                        explicit_changed = GL_TRUE;
                        program->sampler_units_explicit_by_stage[stage][res->binding] = GL_TRUE;
                    }
                }
            }
        }
    }

    if (changed || explicit_changed) {
        static unsigned long long s_sampler_uniform_update_count = 0;
        unsigned long long hit = ++s_sampler_uniform_update_count;
        mglTraceLogExternal("SAMPLER_UNIFORM_SET program=%u location=%d unit=%d firstStage=%d firstBinding=%u firstName=%s resources=%u/%u changed=%d explicitChanged=%d hit=%llu",
                            (unsigned)program->name,
                            (int)location,
                            (int)unit,
                            firstMatchedStage,
                            (unsigned)firstMatchedBinding,
                            mglSafeCStringForLog(firstMatchedName),
                            (unsigned)updatedResourceCount,
                            (unsigned)matchedResourceCount,
                            changed ? 1 : 0,
                            explicit_changed ? 1 : 0,
                            hit);
        ctx->state.dirty_bits |= DIRTY_TEX_BINDING | DIRTY_SAMPLER;
    }
    return GL_TRUE;
}

static size_t mglRoundUpUniformBlockSize(size_t value)
{
    return value ? ((value + 15) & ~(size_t)15) : 0;
}

static GLboolean mglUniformBlockHasName(const SpirvResource *block)
{
    return block && mglSafeCStringLength(block->name, NULL);
}

static GLboolean mglUniformBlockSameIdentity(const SpirvResource *a, const SpirvResource *b)
{
    if (!a || !b) {
        return GL_FALSE;
    }

    if (mglUniformBlockHasName(a) && mglUniformBlockHasName(b)) {
        return mglSafeCStringEquals(a->name, b->name);
    }

    return a == b;
}

static GLboolean mglUniformBlockNameSeenBefore(Program *program, int block_stage, GLuint block_index, const char *name)
{
    if (!program || !mglSafeCStringLength(name, NULL)) {
        return GL_FALSE;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = mglUniformSafeResourceList(program, stage, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, __FUNCTION__);
        if (!resources) {
            continue;
        }

        for (GLuint i = 0; i < resources->count; i++) {
            if (stage > block_stage || (stage == block_stage && i >= block_index)) {
                return GL_FALSE;
            }
            if (mglSafeCStringEquals(name, resources->list[i].name)) {
                return GL_TRUE;
            }
        }
    }

    return GL_FALSE;
}

static SpirvResource *mglFindUniformBlockByIndex(Program *program, GLuint uniformBlockIndex, int *stage_out)
{
    GLuint ordinal = 0;

    if (!program || uniformBlockIndex == GL_INVALID_INDEX) {
        return NULL;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = mglUniformSafeResourceList(program, stage, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, __FUNCTION__);
        if (!resources) {
            continue;
        }
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (mglUniformBlockNameSeenBefore(program, stage, i, res->name)) {
                continue;
            }
            if (ordinal == uniformBlockIndex) {
                if (stage_out) {
                    *stage_out = stage;
                }
                return res;
            }
            ordinal++;
        }
    }

    return NULL;
}

static size_t mglUniformBlockRequiredSize(Program *program, const SpirvResource *block)
{
    size_t required_size = 0;

    if (!program || !block) {
        return 0;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = mglUniformSafeResourceList(program, stage, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, __FUNCTION__);
        if (!resources) {
            continue;
        }
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (mglUniformBlockSameIdentity(block, res) && res->required_size > required_size) {
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

    SpirvResourceList *resources = mglUniformSafeResourceList(program, query_stage, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, __FUNCTION__);
    if (!resources) {
        return GL_FALSE;
    }
    for (GLuint i = 0; i < resources->count; i++) {
        SpirvResource *res = &resources->list[i];
        if (mglUniformBlockSameIdentity(block, res)) {
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

    if (!name || !mglSafeCStringLength(name, NULL)) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, -1);
    }

    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, -1);
    }

    Program *ptr = mglUniformGetNamedProgram(ctx, program, __FUNCTION__);
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
            SpirvResourceList *resources = mglUniformSafeResourceList(ptr, stage, res_type, __FUNCTION__);
            if (!resources || resources->count == 0u) {
                continue;
            }

            for (GLuint i=0; i<resources->count; i++)
            {
                SpirvResource *list = resources->list;
                const char *str = list[i].name;
                if (!mglSafeCStringLength(str, NULL)) {
                    continue;
                }

                if (mglSafeCStringEquals(str, name))
                {
                    GLuint gl_binding = list[i].gl_binding;
                    GLboolean sampler_like = mglUniformResourceLooksSamplerLike(&list[i], res_type);
                    GLint location = sampler_like
                        ? list[i].uniform_location
                        : ((res_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT)
                            ? mglPlainUniformResourceLocation(&list[i])
                            : ((list[i].uniform_location >= 0)
                                ? list[i].uniform_location
                                : ((list[i].location != 0xffffffffu) ? (GLint)list[i].location : (GLint)gl_binding)));

                    return location;
                }
            }
        }
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
        Program *ptr = mglUniformGetNamedProgram(ctx, program, __FUNCTION__);
        GLuint metal_binding = 0;
        int stage = -1;
        SpirvResource *res =
            ptr ? mglFindSamplerUniformResource(ptr, location, &stage, &metal_binding) : NULL;
        if (res && res->sampler_unit >= 0 && res->sampler_unit < TEXTURE_UNITS) {
            *params = res->sampler_unit;
        } else if (ptr && metal_binding < TEXTURE_UNITS &&
                   stage >= 0 && stage < _MAX_SHADER_TYPES &&
                   ptr->sampler_units_by_stage[stage][metal_binding] >= 0) {
            *params = ptr->sampler_units_by_stage[stage][metal_binding];
        } else if (ptr && metal_binding < TEXTURE_UNITS &&
                   ptr->sampler_units[metal_binding] >= 0) {
            *params = ptr->sampler_units[metal_binding];
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

    Program *ptr = mglUniformGetNamedProgram(ctx, program, __FUNCTION__);
    if (!ptr || ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    for (GLsizei i = 0; i < uniformCount; i++) {
        GLint index = mglProgramActiveUniformIndexByName(ptr, uniformNames[i]);
        uniformIndices[i] = (index >= 0) ? (GLuint)index : GL_INVALID_INDEX;
    }
}

void mglGetActiveUniformsiv(GLMContext ctx, GLuint program, GLsizei uniformCount, const GLuint *uniformIndices, GLenum pname, GLint *params)
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
    if (uniformCount > 0 && !params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (uniformCount > 0 && !uniformIndices) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = mglUniformGetNamedProgram(ctx, program, __FUNCTION__);
    if (!ptr || ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    for (GLsizei i = 0; i < uniformCount; i++) {
        int stage = -1;
        int res_type = -1;
        SpirvResource *res = mglProgramActiveUniformAt(ptr, uniformIndices[i], &stage, &res_type);
        (void)stage;
        if (!res) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }

        switch (pname) {
            case GL_UNIFORM_TYPE:
                params[i] = mglProgramActiveUniformGLType(res, res_type);
                break;
            case GL_UNIFORM_SIZE:
                params[i] = mglProgramActiveUniformSize(res, res_type);
                break;
            case GL_UNIFORM_NAME_LENGTH:
                params[i] = (GLint)mglProgramActiveUniformNameLength(res) + 1;
                break;
            case GL_UNIFORM_BLOCK_INDEX:
            case GL_UNIFORM_OFFSET:
            case GL_UNIFORM_ARRAY_STRIDE:
            case GL_UNIFORM_MATRIX_STRIDE:
            case GL_UNIFORM_ATOMIC_COUNTER_BUFFER_INDEX:
                params[i] = -1;
                break;
            case GL_UNIFORM_IS_ROW_MAJOR:
                params[i] = GL_FALSE;
                break;
            default:
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
        }
    }
}

void mglGetActiveUniformName(GLMContext ctx, GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformName)
{
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

    Program *ptr = mglUniformGetNamedProgram(ctx, program, __FUNCTION__);
    if (!ptr || ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    SpirvResource *res = mglProgramActiveUniformAt(ptr, uniformIndex, NULL, NULL);
    if (!res) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglProgramCopyActiveUniformName(res, bufSize, length, uniformName);
}

GLuint  mglGetUniformBlockIndex(GLMContext ctx, GLuint program, const GLchar *uniformBlockName)
{
    if (!ctx) {
        return (GLuint)-1;
    }

    if (!uniformBlockName || !mglSafeCStringLength(uniformBlockName, NULL)) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, (GLuint)-1);
    }

    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, (GLuint)-1);
    }

    Program *ptr = mglUniformGetNamedProgram(ctx, program, __FUNCTION__);
    if (!ptr) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, (GLuint)-1);
    }

    if (ptr->linked_glsl_program == NULL) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, (GLuint)-1);
    }

    GLuint block_index = 0;
    for (int stage=_VERTEX_SHADER; stage<_MAX_SHADER_TYPES; stage++)
    {
        SpirvResourceList *resources = mglUniformSafeResourceList(ptr, stage, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, __FUNCTION__);
        if (!resources || resources->count == 0u) {
            continue;
        }

        for (GLuint i=0; i<resources->count; i++)
        {
            SpirvResource *list = resources->list;
            const char *str = list[i].name;
            if (!mglSafeCStringLength(str, NULL)) {
                continue;
            }
            if (mglUniformBlockNameSeenBefore(ptr, stage, i, str)) {
                continue;
            }

            if (mglSafeCStringEquals(str, uniformBlockName))
            {
                return block_index;
            }
            block_index++;
        }
    }

    fprintf(stderr, "MGL WARNING: uniform block '%s' binding not found, returning GL_INVALID_INDEX\n",
            mglSafeCStringForLog(uniformBlockName));
    return (GLuint)-1;
}

void mglGetActiveUniformBlockiv(GLMContext ctx, GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params)
{
    if (!params) {
        return;
    }
    *params = 0;

    if (!ctx) {
        return;
    }

    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = mglUniformGetNamedProgram(ctx, program, __FUNCTION__);
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
            {
                size_t block_name_len = 0u;
                *params = (GLint)(mglSafeCStringLength(block->name, &block_name_len) ? block_name_len + 1u : 1u);
            }
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
    if (!ctx) {
        return;
    }
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = mglUniformGetNamedProgram(ctx, program, __FUNCTION__);
    if (!ptr || ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    SpirvResource *block = mglFindUniformBlockByIndex(ptr, uniformBlockIndex, NULL);
    if (!block) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    size_t safe_src_len = 0u;
    const char *src = mglSafeCStringLength(block->name, &safe_src_len) ? block->name : "";
    GLsizei src_len = (GLsizei)safe_src_len;
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
    if (!ctx) {
        return;
    }

    if (uniformBlockBinding >= MAX_BINDABLE_BUFFERS) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (isProgram(ctx, program) == GL_FALSE) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = mglUniformGetNamedProgram(ctx, program, __FUNCTION__);
    if (!ptr || ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    int block_stage = -1;
    SpirvResource *block = mglFindUniformBlockByIndex(ptr, uniformBlockIndex, &block_stage);
    if (!block) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    const char *block_name = mglSafeCStringLength(block->name, NULL) ? block->name : NULL;
    GLuint old_binding = block->gl_binding;
    if (old_binding != uniformBlockBinding) {
        mglFlushPendingDraws(ctx);
    }

    mglTraceLogExternal("UNIFORM_BLOCK_BINDING program=%u blockIndex=%u blockName=%s oldBinding=%u newBinding=%u",
                        (unsigned)program,
                        (unsigned)uniformBlockIndex,
                        block_name ? block_name : "(invalid)",
                        (unsigned)old_binding,
                        (unsigned)uniformBlockBinding);

    /*
     * `binding` is the Metal argument slot after MSL reflection/repair.
     * `gl_binding` is the client-side UBO binding point used to find
     * glBindBufferRange state. glUniformBlockBinding changes only the latter.
     *
     * Match the block identity, not the old binding number. Minecraft's 1.21.x
     * shaders legitimately remap blocks through occupied binding points during
     * setup (for example Fog 3->2 followed by Lighting 2->3). Updating every
     * resource with the old binding makes those blocks alias the same UBO.
     */
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = mglUniformSafeResourceList(ptr, stage, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, __FUNCTION__);
        if (!resources) {
            continue;
        }
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            GLboolean same_named_block =
                block_name &&
                mglSafeCStringLength(res->name, NULL) &&
                mglSafeCStringEquals(block_name, res->name);
            GLboolean same_resource =
                !block_name && stage == block_stage && res == block;
            if (same_named_block || same_resource) {
                res->gl_binding = uniformBlockBinding;
            }
        }
    }

    ctx->state.dirty_bits |= DIRTY_BUFFER_BASE_STATE | DIRTY_PROGRAM;
}

bool checkUniformParams(GLMContext ctx, GLint location)
{
    ctx = mglUniformResolveContext(ctx, __FUNCTION__);
    if (!ctx) {
        return false;
    }

    Program* ptr = mglUniformGetCurrentProgram(ctx, __FUNCTION__);
    
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

static GLboolean mglUniformBufferDataWouldChange(Buffer *buf, GLsizeiptr size, const void *data)
{
    if (!buf) {
        return GL_FALSE;
    }
    if (buf->size != size) {
        return GL_TRUE;
    }
    if (size <= 0) {
        return GL_FALSE;
    }
    if (!buf->data.buffer_data || !data) {
        return GL_TRUE;
    }
    return memcmp((const void *)(uintptr_t)buf->data.buffer_data, data, (size_t)size) != 0;
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
    
    Program *program = mglUniformGetCurrentProgram(ctx, __FUNCTION__);
    if (!program) {
        mglUniformSetError(ctx, GL_INVALID_OPERATION);
        return;
    }

    /*
     * Deferred draws snapshot the GL state struct, but Program-owned uniform
     * storage is mutable shared state. Match Apple OpenGL call ordering by
     * submitting queued draws before this glUniform* changes that storage.
     */
    mglFlushPendingDraws(ctx);

    BufferBaseTarget *uniformSlot = &program->plain_uniform_buffers[location];
    Buffer *buf = uniformSlot->buf;
    if (mglUniformBufferDataWouldChange(buf, size, ptr)) {
        mglFlushPendingDrawsForBuffer(ctx, buf);
    }
    
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
        if (mglUniformBufferDataWouldChange(globalSlot->buf, size, ptr)) {
            mglFlushPendingDrawsForBuffer(ctx, globalSlot->buf);
        }
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
    if (count > 0 && value && mglPointerRangeReadable(value, sizeof(*value)) &&
        mglSetSamplerUniformUnit(ctx, location, value[0])) {
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
    if (count > 0 && value && mglPointerRangeReadable(value, sizeof(*value)) &&
        value[0] <= (GLuint)INT_MAX &&
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
    
    mglUniform(ctx, location, data, 3 * sizeof(GLint));
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

DEFINE_MATRIX_TYPE(GLdouble, 3, 2, Mat2x3dv)
DEFINE_MATRIX_TYPE(GLdouble, 2, 3, Mat2x3dvTrans)
DEFINE_TRANSPOSE_FUNC(GLdouble, 3, 2, Mat2x3dv, Mat2x3dvTrans)

void mglUniformMatrix2x3dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,         // Element type
                            Mat2x3dv,          // Source matrix type
                            Mat2x3dvTrans,     // Destination matrix type
                            Mat2x3dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 3, 2, Mat2x3fv)
DEFINE_MATRIX_TYPE(GLfloat, 2, 3, Mat2x3fvTrans)
DEFINE_TRANSPOSE_FUNC(GLfloat, 3, 2, Mat2x3fv, Mat2x3fvTrans)

void mglUniformMatrix2x3fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat2x3fv,          // Source matrix type
                            Mat2x3fvTrans,     // Destination matrix type
                            Mat2x3fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 4, 2, Mat2x4dv)
DEFINE_MATRIX_TYPE(GLdouble, 2, 4, Mat2x4dvTrans)
DEFINE_TRANSPOSE_FUNC(GLdouble, 4, 2, Mat2x4dv, Mat2x4dvTrans)

void mglUniformMatrix2x4dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat2x4dv,          // Source matrix type
                            Mat2x4dvTrans,     // Destination matrix type
                            Mat2x4dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 4, 2, Mat2x4fv)
DEFINE_MATRIX_TYPE(GLfloat, 2, 4, Mat2x4fvTrans)
DEFINE_TRANSPOSE_FUNC(GLfloat, 4, 2, Mat2x4fv, Mat2x4fvTrans)

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

DEFINE_MATRIX_TYPE(GLdouble, 2, 3, Mat3x2dv)
DEFINE_MATRIX_TYPE(GLdouble, 3, 2, Mat3x2dvTrans)
DEFINE_TRANSPOSE_FUNC(GLdouble, 2, 3, Mat3x2dv, Mat3x2dvTrans)

void mglUniformMatrix3x2dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat3x2dv,          // Source matrix type
                            Mat3x2dvTrans,     // Destination matrix type
                            Mat3x2dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 2, 3, Mat3x2fv)
DEFINE_MATRIX_TYPE(GLfloat, 3, 2, Mat3x2fvTrans)
DEFINE_TRANSPOSE_FUNC(GLfloat, 2, 3, Mat3x2fv, Mat3x2fvTrans)

void mglUniformMatrix3x2fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat3x2fv,          // Source matrix type
                            Mat3x2fvTrans,     // Destination matrix type
                            Mat3x2fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 4, 3, Mat3x4dv)
DEFINE_MATRIX_TYPE(GLdouble, 3, 4, Mat3x4dvTrans)
DEFINE_TRANSPOSE_FUNC(GLdouble, 4, 3, Mat3x4dv, Mat3x4dvTrans)

void mglUniformMatrix3x4dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat3x4dv,          // Source matrix type
                            Mat3x4dvTrans,     // Destination matrix type
                            Mat3x4dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 4, 3, Mat3x4fv)
DEFINE_MATRIX_TYPE(GLfloat, 3, 4, Mat3x4fvTrans)
DEFINE_TRANSPOSE_FUNC(GLfloat, 4, 3, Mat3x4fv, Mat3x4fvTrans)

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

DEFINE_MATRIX_TYPE(GLdouble, 2, 4, Mat4x2dv)
DEFINE_MATRIX_TYPE(GLdouble, 4, 2, Mat4x2dvTrans)
DEFINE_TRANSPOSE_FUNC(GLdouble, 2, 4, Mat4x2dv, Mat4x2dvTrans)

void mglUniformMatrix4x2dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat4x2dv,          // Source matrix type
                            Mat4x2dvTrans,     // Destination matrix type
                            Mat4x2dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 2, 4, Mat4x2fv)
DEFINE_MATRIX_TYPE(GLfloat, 4, 2, Mat4x2fvTrans)
DEFINE_TRANSPOSE_FUNC(GLfloat, 2, 4, Mat4x2fv, Mat4x2fvTrans)

void mglUniformMatrix4x2fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat4x2fv,          // Source matrix type
                            Mat4x2fvTrans,     // Destination matrix type
                            Mat4x2fvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLdouble, 3, 4, Mat4x3dv)
DEFINE_MATRIX_TYPE(GLdouble, 4, 3, Mat4x3dvTrans)
DEFINE_TRANSPOSE_FUNC(GLdouble, 3, 4, Mat4x3dv, Mat4x3dvTrans)

void mglUniformMatrix4x3dv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLdouble,        // Element type
                            Mat4x3dv,          // Source matrix type
                            Mat4x3dvTrans,     // Destination matrix type
                            Mat4x3dvTranspose  // Transpose function
        );
}

DEFINE_MATRIX_TYPE(GLfloat, 3, 4, Mat4x3fv)
DEFINE_MATRIX_TYPE(GLfloat, 4, 3, Mat4x3fvTrans)
DEFINE_TRANSPOSE_FUNC(GLfloat, 3, 4, Mat4x3fv, Mat4x3fvTrans)

void mglUniformMatrix4x3fv(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
    HANDLE_MATRIX_TRANSPOSE(
                            GLfloat,        // Element type
                            Mat4x3fv,          // Source matrix type
                            Mat4x3fvTrans,     // Destination matrix type
                            Mat4x3fvTranspose  // Transpose function
        );
}
