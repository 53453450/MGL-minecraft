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
#include <malloc/malloc.h>
#include <CoreFoundation/CoreFoundation.h>
#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>
#include "spirv-tools/libspirv.h"
#include "spirv_cross_c.h"
#include "spirv.h"

#include "glm_context.h"
#include "shaders.h"
#include "buffers.h"
#include "mgl_safety.h"

#ifndef MGL_VERBOSE_PROGRAM_LOGS
#define MGL_VERBOSE_PROGRAM_LOGS 0
#endif

#define MGL_TEXEL_BUFFER_TEXTURE_WIDTH 4096u
#define MGL_SYNTHETIC_SAMPLER_LOCATION_BASE 0x4000

static size_t mglRoundUpSize(size_t value, size_t alignment)
{
    return alignment ? ((value + alignment - 1) / alignment) * alignment : value;
}

static GLboolean mglPointerLooksMallocOwned(const void *ptr)
{
    uintptr_t value = (uintptr_t)ptr;
    if (!ptr || value < 0x10000u) {
        return GL_FALSE;
    }

    return malloc_size(ptr) > 0 ? GL_TRUE : GL_FALSE;
}

static void mglFreeProgramAttribName(Program *program, GLuint index, const char *reason)
{
    if (!program || index >= MAX_ATTRIBS) {
        return;
    }

    char *name = program->attrib_location_names[index];
    GLboolean owned = program->attrib_location_name_owned[index];
    program->attrib_location_names[index] = NULL;
    program->attrib_location_name_owned[index] = GL_FALSE;

    if (!name) {
        return;
    }

    if (owned && mglPointerLooksMallocOwned(name)) {
        free(name);
        return;
    }

    fprintf(stderr,
            "MGL WARNING: skipped invalid attrib name free program=%u index=%u ptr=%p owned=%d reason=%s\n",
            program->name,
            index,
            (void *)name,
            owned,
            reason ? reason : "(unknown)");
}

static GLboolean mglSetProgramAttribName(Program *program, GLuint index, const char *name)
{
    if (!program || index >= MAX_ATTRIBS || !name) {
        return GL_FALSE;
    }

    mglFreeProgramAttribName(program, index, "replace");
    program->attrib_location_names[index] = strdup(name);
    if (!program->attrib_location_names[index]) {
        return GL_FALSE;
    }
    program->attrib_location_name_owned[index] = GL_TRUE;
    return GL_TRUE;
}

static GLboolean mglUniformBlockNameSeen(Program *program, int max_stage, GLuint max_index, const char *name, GLuint gl_binding)
{
    for (int stage = _VERTEX_SHADER; stage <= max_stage && stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        GLuint limit = (stage == max_stage) ? max_index : resources->count;
        for (GLuint i = 0; i < limit; i++) {
            SpirvResource *res = &resources->list[i];
            if (name && name[0] != '\0') {
                if (res->name && !strcmp(name, res->name)) {
                    return GL_TRUE;
                }
                continue;
            }
            if ((!res->name || res->name[0] == '\0') && res->gl_binding == gl_binding) {
                return GL_TRUE;
            }
        }
    }
    return GL_FALSE;
}

static GLint mglActiveUniformBlockCount(Program *program)
{
    GLint total = 0;

    if (!program) {
        return 0;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (!mglUniformBlockNameSeen(program, stage, i, res->name, res->gl_binding)) {
                total++;
            }
        }
    }

    return total;
}

static GLint mglActiveUniformBlockMaxNameLength(Program *program)
{
    GLint max_len = 0;

    if (!program) {
        return 0;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (mglUniformBlockNameSeen(program, stage, i, res->name, res->gl_binding)) {
                continue;
            }
            GLint len = (GLint)(res->name ? strlen(res->name) + 1 : 1);
            if (len > max_len) {
                max_len = len;
            }
        }
    }

    return max_len;
}

static SpirvResourceList *mglProgramActiveAttribList(Program *program)
{
    if (!program) {
        return NULL;
    }

    return &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
}

static GLboolean mglProgramActiveAttribHasName(const SpirvResource *res)
{
    return (res && res->name && res->name[0] != '\0') ? GL_TRUE : GL_FALSE;
}

static GLint mglProgramActiveAttribCount(Program *program)
{
    SpirvResourceList *resources = mglProgramActiveAttribList(program);
    if (!resources || !resources->list) {
        return 0;
    }

    GLint count = 0;
    for (GLuint i = 0; i < resources->count; i++) {
        if (mglProgramActiveAttribHasName(&resources->list[i])) {
            count++;
        }
    }

    return count;
}

static SpirvResource *mglProgramActiveAttribAt(Program *program, GLuint index)
{
    SpirvResourceList *resources = mglProgramActiveAttribList(program);
    if (!resources || !resources->list) {
        return NULL;
    }

    GLuint ordinal = 0;
    for (GLuint i = 0; i < resources->count; i++) {
        SpirvResource *res = &resources->list[i];
        if (!mglProgramActiveAttribHasName(res)) {
            continue;
        }
        if (ordinal == index) {
            return res;
        }
        ordinal++;
    }

    return NULL;
}

static GLint mglProgramActiveAttribMaxNameLength(Program *program)
{
    GLint max_len = 0;
    GLint count = mglProgramActiveAttribCount(program);

    for (GLint i = 0; i < count; i++) {
        SpirvResource *res = mglProgramActiveAttribAt(program, (GLuint)i);
        GLint len = (GLint)(res && res->name ? strlen(res->name) + 1 : 1);
        if (len > max_len) {
            max_len = len;
        }
    }

    return max_len;
}

static GLenum mglProgramActiveAttribType(const SpirvResource *res)
{
    const char *name = res ? res->name : NULL;

    if (!name || !name[0]) {
        return GL_FLOAT;
    }

    if (!strcmp(name, "Position") ||
        !strcmp(name, "Normal")) {
        return GL_FLOAT_VEC3;
    }
    if (!strcmp(name, "Color")) {
        return GL_FLOAT_VEC4;
    }
    if (!strcmp(name, "UV") ||
        !strcmp(name, "UV0") ||
        !strcmp(name, "TexCoord") ||
        !strcmp(name, "texCoord")) {
        return GL_FLOAT_VEC2;
    }
    if (!strcmp(name, "UV1") ||
        !strcmp(name, "UV2")) {
        return GL_INT_VEC2;
    }

    if (strstr(name, "Color")) {
        return GL_FLOAT_VEC4;
    }
    if (strstr(name, "UV") ||
        strstr(name, "TexCoord") ||
        strstr(name, "texCoord")) {
        return GL_FLOAT_VEC2;
    }
    if (strstr(name, "Normal")) {
        return GL_FLOAT_VEC3;
    }

    return GL_FLOAT_VEC4;
}

static GLint mglSyntheticSamplerUniformLocation(int stage, int res_type, GLuint index)
{
    return 0x4000 + (stage * 0x1000) + (res_type * 0x100) + (GLint)index;
}

static bool mglIsSamplerResourceType(int res_type)
{
    return res_type == SPVC_RESOURCE_TYPE_SAMPLED_IMAGE ||
           res_type == SPVC_RESOURCE_TYPE_SEPARATE_IMAGE ||
           res_type == SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS ||
           res_type == SPVC_RESOURCE_TYPE_STORAGE_IMAGE;
}

static bool mglUniformNameLooksSamplerLike(const char *name)
{
    if (!name || !*name) {
        return false;
    }

    return strstr(name, "Sampler") != NULL ||
           strcmp(name, "CloudFaces") == 0;
}

static bool mglUniformConstantBaseTypeIsSamplerLike(spvc_basetype basetype)
{
    return basetype == SPVC_BASETYPE_IMAGE ||
           basetype == SPVC_BASETYPE_SAMPLED_IMAGE ||
           basetype == SPVC_BASETYPE_SAMPLER;
}

static GLint mglKnownPlainUniformLocationForName(const char *name)
{
    if (!name || !*name) {
        return -1;
    }

    if (!strcmp(name, "ModelViewMat")) return 0;
    if (!strcmp(name, "ProjMat")) return 1;
    if (!strcmp(name, "TextureMat")) return 2;
    if (!strcmp(name, "ColorModulator")) return 3;
    if (!strcmp(name, "FogStart")) return 4;
    if (!strcmp(name, "FogEnd")) return 5;
    if (!strcmp(name, "FogColor")) return 6;
    if (!strcmp(name, "FogShape")) return 7;
    if (!strcmp(name, "GameTime")) return 8;
    if (!strcmp(name, "ScreenSize")) return 9;
    if (!strcmp(name, "LineWidth")) return 10;
    if (!strcmp(name, "IViewRotMat")) return 11;
    if (!strcmp(name, "ChunkOffset")) return 12;
    if (!strcmp(name, "u_ProjectionMatrix")) return 0;
    if (!strcmp(name, "u_ModelViewMatrix")) return 1;
    if (!strcmp(name, "u_RegionOffset")) return 2;
    if (!strcmp(name, "u_TexCoordShrink")) return 3;
    if (!strcmp(name, "u_FogColor")) return 4;
    if (!strcmp(name, "u_EnvironmentFog")) return 5;
    if (!strcmp(name, "u_RenderFog")) return 6;

    return -1;
}

static bool mglProgramResourceLooksSamplerLike(const SpirvResource *res, int res_type)
{
    if (!res) {
        return false;
    }

    switch (res_type) {
        case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS:
        case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
            return true;
        case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
            return res->image_dim != 0u ||
                   res->uniform_location >= MGL_SYNTHETIC_SAMPLER_LOCATION_BASE ||
                   mglUniformNameLooksSamplerLike(res->name);
        default:
            return false;
    }
}

static void mglUnifySamplerUniformLocations(Program *program)
{
    if (!program) {
        return;
    }

    static const int sampler_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    for (int leader_stage = _VERTEX_SHADER; leader_stage < _MAX_SHADER_TYPES; leader_stage++) {
        for (size_t leader_rt = 0; leader_rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); leader_rt++) {
            int leader_type = sampler_resource_types[leader_rt];
            SpirvResourceList *leaders = &program->spirv_resources_list[leader_stage][leader_type];
            for (GLuint leader_i = 0; leaders->list && leader_i < leaders->count; leader_i++) {
                SpirvResource *leader = &leaders->list[leader_i];
                if (!mglProgramResourceLooksSamplerLike(leader, leader_type) ||
                    !leader->name ||
                    leader->uniform_location < 0) {
                    continue;
                }

                for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
                    for (size_t rt = 0; rt < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); rt++) {
                        int res_type = sampler_resource_types[rt];
                        SpirvResourceList *resources = &program->spirv_resources_list[stage][res_type];
                        for (GLuint i = 0; resources->list && i < resources->count; i++) {
                            SpirvResource *res = &resources->list[i];
                            if (res == leader ||
                                !mglProgramResourceLooksSamplerLike(res, res_type) ||
                                !res->name ||
                                strcmp(res->name, leader->name) != 0) {
                                continue;
                            }

                            res->uniform_location = leader->uniform_location;
                        }
                    }
                }
            }
        }
    }
}

static SpirvResource *mglFindAssignedPlainUniformResource(Program *program, const char *name)
{
    if (!program || !name || !*name) {
        return NULL;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (res->uniform_location < 0 ||
                mglProgramResourceLooksSamplerLike(res, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT) ||
                !res->name ||
                strcmp(res->name, name) != 0) {
                continue;
            }
            return res;
        }
    }

    return NULL;
}

static GLint mglFirstFreePlainUniformLocation(const bool used[MAX_BINDABLE_BUFFERS])
{
    for (GLint location = 0; location < MAX_BINDABLE_BUFFERS; location++) {
        if (!used[location]) {
            return location;
        }
    }

    return -1;
}

static void mglAssignPlainUniformLocations(Program *program)
{
    if (!program) {
        return;
    }

    bool used[MAX_BINDABLE_BUFFERS] = {false};

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (mglProgramResourceLooksSamplerLike(res, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT)) {
                continue;
            }

            GLint known = mglKnownPlainUniformLocationForName(res->name);
            if (known >= 0 && known < MAX_BINDABLE_BUFFERS) {
                res->uniform_location = known;
                used[known] = true;
            } else if (res->uniform_location >= 0 &&
                       res->uniform_location < MAX_BINDABLE_BUFFERS) {
                used[res->uniform_location] = true;
            }
        }
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (mglProgramResourceLooksSamplerLike(res, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT)) {
                continue;
            }
            if (res->uniform_location >= 0 &&
                res->uniform_location < MAX_BINDABLE_BUFFERS) {
                continue;
            }

            SpirvResource *assigned = mglFindAssignedPlainUniformResource(program, res->name);
            if (assigned && assigned->uniform_location >= 0 &&
                assigned->uniform_location < MAX_BINDABLE_BUFFERS) {
                res->uniform_location = assigned->uniform_location;
                continue;
            }

            GLint preferred = -1;
            if (res->location < MAX_BINDABLE_BUFFERS && !used[res->location]) {
                preferred = (GLint)res->location;
            } else if (res->gl_binding < MAX_BINDABLE_BUFFERS && !used[res->gl_binding]) {
                preferred = (GLint)res->gl_binding;
            } else {
                preferred = mglFirstFreePlainUniformLocation(used);
            }

            if (preferred < 0) {
                fprintf(stderr,
                        "MGL WARNING: no plain uniform location left program=%u name=%s stage=%d\n",
                        program->name,
                        res->name ? res->name : "(null)",
                        stage);
                continue;
            }

            res->uniform_location = preferred;
            used[preferred] = true;
            fprintf(stderr,
                    "MGL PLAIN UNIFORM FIX: program=%u stage=%d name=%s loc=%d metal=%u\n",
                    program->name,
                    stage,
                    res->name ? res->name : "(null)",
                    preferred,
                    (unsigned)res->binding);
        }
    }
}

static GLint mglDefaultSamplerUnitForProgramResource(Program *program, const SpirvResource *res)
{
    (void)program;
    (void)res;

    /*
     * OpenGL initializes sampler uniforms to texture unit 0. Minecraft 1.20+
     * then uploads the real units from ShaderInstance/RenderPipeline sampler
     * lists via glUniform1i. Guessing from names such as Sampler1/Sampler2
     * corrupts pipelines where a shader declares an optional sampler that the
     * pipeline intentionally does not bind.
     */
    return 0;
}

static void mglApplyDefaultSamplerUnit(Program *program, int stage, int res_type, SpirvResource *res)
{
    if (!program || !res || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }
    if (!mglProgramResourceLooksSamplerLike(res, res_type)) {
        return;
    }

    GLint unit = mglDefaultSamplerUnitForProgramResource(program, res);
    if (unit < 0 || unit >= TEXTURE_UNITS) {
        return;
    }

    /*
     * Store the OpenGL sampler uniform default on the resource itself. The
     * resource binding is now the Metal argument slot and can be shared by
     * unrelated resources such as vertex Sampler2 and fragment Sampler0.
     */
    res->sampler_unit = unit;
    res->sampler_unit_explicit = GL_FALSE;
}

static bool mglMSLIdentifierChar(char c)
{
    return (c == '_') ||
           (c >= '0' && c <= '9') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z');
}

typedef enum MGLMSLBindingKind {
    MGL_MSL_BINDING_TEXTURE,
    MGL_MSL_BINDING_BUFFER,
    MGL_MSL_BINDING_SAMPLER
} MGLMSLBindingKind;

typedef struct MGLMSLBindingEntry {
    MGLMSLBindingKind kind;
    GLuint index;
    const char *segment;
    size_t segment_len;
} MGLMSLBindingEntry;

#define MGL_MSL_BINDING_MAP_MAX 512u

typedef struct MGLMSLBindingMap {
    MGLMSLBindingEntry entries[MGL_MSL_BINDING_MAP_MAX];
    size_t count;
} MGLMSLBindingMap;

static GLboolean mglSegmentContainsIdentifier(const char *segment,
                                              size_t segment_len,
                                              const char *name)
{
    if (!segment || !name) {
        return GL_FALSE;
    }

    size_t name_len = strlen(name);
    if (name_len == 0 || name_len > segment_len) {
        return GL_FALSE;
    }

    const char *end = segment + segment_len;
    for (const char *cursor = segment; cursor + name_len <= end; cursor++) {
        if (memcmp(cursor, name, name_len) != 0) {
            continue;
        }

        char before = (cursor == segment) ? '\0' : cursor[-1];
        char after = (cursor + name_len == end) ? '\0' : cursor[name_len];
        if (!mglMSLIdentifierChar(before) && !mglMSLIdentifierChar(after)) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static const char *mglPreviousMSLArgumentBoundary(const char *msl, const char *attribute)
{
    const char *cursor = attribute;
    unsigned angle_depth = 0;

    while (cursor > msl) {
        char c = cursor[-1];
        if (c == '\n' || c == '\r') {
            break;
        }
        if (c == '>') {
            angle_depth++;
        } else if (c == '<') {
            if (angle_depth > 0) {
                angle_depth--;
            }
        } else if (c == ',' && angle_depth == 0) {
            break;
        }
        cursor--;
    }

    while (*cursor == ' ' || *cursor == '\t') {
        cursor++;
    }

    return cursor;
}

static const char *mglNextMSLArgumentBoundary(const char *attribute)
{
    const char *cursor = attribute;
    unsigned angle_depth = 0;

    while (*cursor) {
        char c = *cursor;
        if (c == '\n' || c == '\r') {
            break;
        }
        if (c == '<') {
            angle_depth++;
        } else if (c == '>') {
            if (angle_depth > 0) {
                angle_depth--;
            }
        } else if (c == ',' && angle_depth == 0) {
            break;
        }
        cursor++;
    }

    return cursor;
}

static GLboolean mglParseMSLBindingAttribute(const char *attribute,
                                             const char *prefix,
                                             GLuint *index_out)
{
    if (!attribute || !prefix || !index_out) {
        return GL_FALSE;
    }

    size_t prefix_len = strlen(prefix);
    if (strncmp(attribute, prefix, prefix_len) != 0) {
        return GL_FALSE;
    }

    const char *index_start = attribute + prefix_len;
    char *end = NULL;
    unsigned long value = strtoul(index_start, &end, 10);
    if (end == index_start || value >= TEXTURE_UNITS) {
        return GL_FALSE;
    }

    *index_out = (GLuint)value;
    return GL_TRUE;
}

static void mglMSLBindingMapAdd(MGLMSLBindingMap *map,
                                MGLMSLBindingKind kind,
                                GLuint index,
                                const char *segment_start,
                                const char *segment_end)
{
    if (!map || !segment_start || !segment_end || segment_end < segment_start ||
        map->count >= MGL_MSL_BINDING_MAP_MAX) {
        return;
    }

    while (segment_end > segment_start &&
           (segment_end[-1] == ' ' || segment_end[-1] == '\t')) {
        segment_end--;
    }

    MGLMSLBindingEntry *entry = &map->entries[map->count++];
    entry->kind = kind;
    entry->index = index;
    entry->segment = segment_start;
    entry->segment_len = (size_t)(segment_end - segment_start);
}

static void mglBuildMSLBindingMap(const char *msl, MGLMSLBindingMap *map)
{
    if (!map) {
        return;
    }

    memset(map, 0, sizeof(*map));
    if (!msl) {
        return;
    }

    const char *cursor = msl;
    while (*cursor) {
        const char *texture_attr = strstr(cursor, "[[texture(");
        const char *buffer_attr = strstr(cursor, "[[buffer(");
        const char *sampler_attr = strstr(cursor, "[[sampler(");
        const char *attribute = texture_attr;
        const char *prefix = "[[texture(";
        MGLMSLBindingKind kind = MGL_MSL_BINDING_TEXTURE;

        if (!attribute || (buffer_attr && buffer_attr < attribute)) {
            attribute = buffer_attr;
            prefix = "[[buffer(";
            kind = MGL_MSL_BINDING_BUFFER;
        }
        if (!attribute || (sampler_attr && sampler_attr < attribute)) {
            attribute = sampler_attr;
            prefix = "[[sampler(";
            kind = MGL_MSL_BINDING_SAMPLER;
        }
        if (!attribute) {
            break;
        }

        GLuint index = 0;
        if (mglParseMSLBindingAttribute(attribute, prefix, &index)) {
            const char *segment_start = mglPreviousMSLArgumentBoundary(msl, attribute);
            const char *segment_end = mglNextMSLArgumentBoundary(attribute);
            mglMSLBindingMapAdd(map, kind, index, segment_start, segment_end);
        }

        cursor = attribute + 2;
    }
}

static GLboolean mglFindMSLResourceIndexInMap(const MGLMSLBindingMap *map,
                                              MGLMSLBindingKind kind,
                                              const char *name,
                                              GLuint *index_out)
{
    if (!map || !name || !index_out) {
        return GL_FALSE;
    }

    for (size_t i = 0; i < map->count; i++) {
        const MGLMSLBindingEntry *entry = &map->entries[i];
        if (entry->kind == kind &&
            mglSegmentContainsIdentifier(entry->segment, entry->segment_len, name)) {
            *index_out = entry->index;
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static void applyMSLResourceBindings(Program *pptr, int stage, const char *msl)
{
    if (!pptr || !msl || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }

    MGLMSLBindingMap binding_map;
    mglBuildMSLBindingMap(msl, &binding_map);

    const int texture_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    for (size_t t = 0; t < sizeof(texture_resource_types) / sizeof(texture_resource_types[0]); t++) {
        int res_type = texture_resource_types[t];
        SpirvResourceList *resources = &pptr->spirv_resources_list[stage][res_type];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            GLuint metal_index = 0;
            if (!res->name ||
                !mglFindMSLResourceIndexInMap(&binding_map, MGL_MSL_BINDING_TEXTURE, res->name, &metal_index)) {
                continue;
            }

            if (res->binding != metal_index) {
                fprintf(stderr,
                        "MGL RESOURCE FIX: program=%u stage=%d type=%d %s texture binding %u -> %u\n",
                        pptr->name,
                        stage,
                        res_type,
                        res->name,
                        (unsigned)res->binding,
                        (unsigned)metal_index);
                res->binding = metal_index;
            }
        }
    }

    const int buffer_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT
    };

    for (size_t t = 0; t < sizeof(buffer_resource_types) / sizeof(buffer_resource_types[0]); t++) {
        int res_type = buffer_resource_types[t];
        SpirvResourceList *resources = &pptr->spirv_resources_list[stage][res_type];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            GLuint metal_index = 0;
            if (!res->name ||
                !mglFindMSLResourceIndexInMap(&binding_map, MGL_MSL_BINDING_BUFFER, res->name, &metal_index)) {
                continue;
            }

            if (res->binding != metal_index) {
                fprintf(stderr,
                        "MGL RESOURCE FIX: program=%u stage=%d type=%d %s buffer binding %u -> %u (gl=%u)\n",
                        pptr->name,
                        stage,
                        res_type,
                        res->name,
                        (unsigned)res->binding,
                        (unsigned)metal_index,
                        (unsigned)res->gl_binding);
                res->binding = metal_index;
            }
        }
    }

    SpirvResourceList *samplers =
        &pptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS];
    for (GLuint i = 0; i < samplers->count; i++) {
        SpirvResource *res = &samplers->list[i];
        GLuint metal_index = 0;
        if (!res->name ||
            !mglFindMSLResourceIndexInMap(&binding_map, MGL_MSL_BINDING_SAMPLER, res->name, &metal_index)) {
            continue;
        }
        if (res->binding != metal_index) {
            fprintf(stderr,
                    "MGL RESOURCE FIX: program=%u stage=%d type=%d %s sampler binding %u -> %u\n",
                    pptr->name,
                    stage,
                    SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
                    res->name,
                    (unsigned)res->binding,
                    (unsigned)metal_index);
            res->binding = metal_index;
        }
    }

    const int sampler_resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    for (size_t t = 0; t < sizeof(sampler_resource_types) / sizeof(sampler_resource_types[0]); t++) {
        int res_type = sampler_resource_types[t];
        SpirvResourceList *resources = &pptr->spirv_resources_list[stage][res_type];
        for (GLuint i = 0; i < resources->count; i++) {
            mglApplyDefaultSamplerUnit(pptr, stage, res_type, &resources->list[i]);
        }
    }
}

// Program Pipeline management
ProgramPipeline *newProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr;

    ptr = (ProgramPipeline *)malloc(sizeof(ProgramPipeline));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate program pipeline %u\n", pipeline);
        return NULL;
    }

    bzero(ptr, sizeof(ProgramPipeline));
    ptr->name = pipeline;

    return ptr;
}

ProgramPipeline *findProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    return (ProgramPipeline *)searchHashTable(&STATE(program_pipeline_table), pipeline);
}

ProgramPipeline *getProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr = findProgramPipeline(ctx, pipeline);

    if (!ptr)
    {
        ptr = newProgramPipeline(ctx, pipeline);
        if (!ptr)
            return NULL;
        insertHashElement(&STATE(program_pipeline_table), pipeline, ptr);
    }

    return ptr;
}

// Transform Feedback management
TransformFeedback *newTransformFeedback(GLMContext ctx, GLuint name)
{
    TransformFeedback *ptr;

    ptr = (TransformFeedback *)malloc(sizeof(TransformFeedback));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate transform feedback %u\n", name);
        return NULL;
    }

    bzero(ptr, sizeof(TransformFeedback));
    ptr->name = name;
    ptr->target = GL_TRANSFORM_FEEDBACK;
    ptr->active = GL_FALSE;
    ptr->paused = GL_FALSE;
    ptr->primitive_mode = GL_NONE;

    return ptr;
}

TransformFeedback *findTransformFeedback(GLMContext ctx, GLuint name)
{
    return (TransformFeedback *)searchHashTable(&STATE(transform_feedback_table), name);
}

TransformFeedback *getTransformFeedback(GLMContext ctx, GLuint name)
{
    TransformFeedback *ptr = findTransformFeedback(ctx, name);

    if (!ptr)
    {
        ptr = newTransformFeedback(ctx, name);
        if (!ptr)
            return NULL;
        insertHashElement(&STATE(transform_feedback_table), name, ptr);
    }

    return ptr;
}

Program *newProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    ptr = (Program *)malloc(sizeof(Program));
    if (!ptr) {
        if (ctx)
            STATE(error) = GL_OUT_OF_MEMORY;
        fprintf(stderr, "MGL ERROR: failed to allocate program %u\n", program);
        return NULL;
    }

    bzero(ptr, sizeof(Program));

    ptr->name = program;
    for (GLuint i = 0; i < TEXTURE_UNITS; i++) {
        ptr->sampler_units[i] = -1;
    }
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        for (GLuint i = 0; i < TEXTURE_UNITS; i++) {
            ptr->sampler_units_by_stage[stage][i] = -1;
        }
    }

    return ptr;
}

Program *getProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    if (!ctx || program == 0u)
    {
        return NULL;
    }

    ptr = (Program *)searchHashTable(&STATE(program_table), program);

    if (!ptr)
    {
        ptr = newProgram(ctx, program);
        if (!ptr)
            return NULL;

        insertHashElement(&STATE(program_table), program, ptr);
    }

    return ptr;
}

int isProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    if (!ctx || program == 0u)
    {
        return 0;
    }

    ptr = (Program *)searchHashTable(&STATE(program_table), program);

    if (ptr)
        return 1;

    return 0;
}

Program *findProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    if (!ctx || program == 0u)
    {
        return NULL;
    }

    ptr = (Program *)searchHashTable(&STATE(program_table), program);

    return ptr;
}

GLuint mglCreateProgram(GLMContext ctx)
{
    GLuint program;

    program = getNewName(&STATE(program_table));

    if (!getProgram(ctx, program))
        return 0;

    return program;
}

void mglFreeProgram(GLMContext ctx, Program *ptr)
{
    /* linked_glsl_program is used as a linked-state marker only. Do not delete
     * here: glslang_program_delete has been observed to crash on some runtime
     * paths (SIGSEGV in native code). */
    ptr->linked_glsl_program = NULL;

    if (ptr->mtl_data)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->mtl_data);
    }

    for(int i=0; i<_MAX_SHADER_TYPES; i++)
    {
        // CRITICAL FIX: Add NULL checks before all free/release operations to prevent double-frees
        if (ptr->spirv[i].ir) {
            free(ptr->spirv[i].ir);
            ptr->spirv[i].ir = NULL;
        }
        if (ptr->spirv[i].msl_str) {
            free(ptr->spirv[i].msl_str);
            ptr->spirv[i].msl_str = NULL;
        }
        if (ptr->spirv[i].entry_point) {
            free(ptr->spirv[i].entry_point);
            ptr->spirv[i].entry_point = NULL;
        }
        if (ptr->spirv[i].mtl_function) {
            CFRelease(ptr->spirv[i].mtl_function);
            ptr->spirv[i].mtl_function = NULL;
        }
        if (ptr->spirv[i].mtl_library) {
            CFRelease(ptr->spirv[i].mtl_library);
            ptr->spirv[i].mtl_library = NULL;
        }
        
        for(int j=0; j<_MAX_SPIRV_RES; j++)
        {
            // CRITICAL FIX: Add NULL checks and clear pointers to prevent double-frees
            if (ptr->spirv_resources_list[i][j].list) {
                free(ptr->spirv_resources_list[i][j].list);
                ptr->spirv_resources_list[i][j].list = NULL;
            }
        }
        
        if (ptr->shader_slots[i])
        {
            Shader *sptr = ptr->shader_slots[i];
            sptr->refcount--;
            if (sptr->refcount == 0 && sptr->delete_status)
            {
                mglFreeShader(ctx, sptr);
            }
        }
    }

    for (int i = 0; i < MAX_ATTRIBS; i++) {
        mglFreeProgramAttribName(ptr, (GLuint)i, "program delete");
    }

    free(ptr);
}

GLboolean mglProgramPointerUsableForName(GLMContext ctx, Program *program, GLuint expectedName)
{
    if (!ctx || !program || expectedName == 0u) {
        return GL_FALSE;
    }

    if (!mglObjectPointerLooksPlausible(program) ||
        !mglPointerRangeIsReadable(program, sizeof(*program)) ||
        program->name != expectedName) {
        return GL_FALSE;
    }

    if (mglHashTableContainsData(&STATE(program_table), program)) {
        return GL_TRUE;
    }

    /*
     * glDeleteProgram removes the name immediately, but the current program
     * and any deferred draws that captured it must keep using the object until
     * their references are released.
     */
    if (program->delete_status &&
        program->refcount > 0 &&
        program->linked_glsl_program != NULL) {
        return GL_TRUE;
    }

    return GL_FALSE;
}

void mglRetainProgramReference(GLMContext ctx, Program *program)
{
    if (!ctx || !program) {
        return;
    }

    GLuint programName = 0u;
    if (mglObjectPointerLooksPlausible(program) &&
        mglPointerRangeIsReadable(program, sizeof(*program))) {
        programName = program->name;
    }

    if (programName == 0u ||
        !mglProgramPointerUsableForName(ctx, program, programName)) {
        return;
    }

    program->refcount++;
}

void mglReleaseProgramReference(GLMContext ctx, Program *program)
{
    if (!ctx || !program ||
        !mglObjectPointerLooksPlausible(program) ||
        !mglPointerRangeIsReadable(program, sizeof(*program))) {
        return;
    }

    if (program->refcount > 0) {
        program->refcount--;
    }
    if (program->refcount == 0 && program->delete_status) {
        mglFreeProgram(ctx, program);
    }
}

void mglDeleteProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    ptr = findProgram(ctx, program);

    if (!ptr)
    {
        // // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION; // Silent ignore if not found? OpenGL says GL_INVALID_VALUE usually, but delete is often silent for 0.
        // But if program != 0 and not found, it's GL_INVALID_VALUE.
        return;
    }

    mglFlushPendingDraws(ctx);

    deleteHashElement(&STATE(program_table), program);
    
    ptr->delete_status = GL_TRUE;
    
    if (ptr->refcount == 0)
    {
        mglFreeProgram(ctx, ptr);
    }
}

GLboolean mglIsProgram(GLMContext ctx, GLuint program)
{
    if (isProgram(ctx, program))
        return GL_TRUE;

    return GL_FALSE;
}

void mglAttachShader(GLMContext ctx, GLuint program, GLuint shader)
{
    Program *pptr;
    Shader *sptr;
    GLuint index;

    sptr = findShader(ctx, shader);

    if (!sptr)
    {
        // CRITICAL FIX: Handle missing shader gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Shader %u not found in attach shader\n", shader);
        STATE(error) = GL_INVALID_VALUE;
        return;
    }

    pptr = findProgram(ctx, program);

    if (!pptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;

        return;
    }

    index = sptr->glm_type;

    mglFlushPendingDraws(ctx);

    pptr->shader_slots[index] = sptr;
    sptr->refcount++;
    pptr->dirty_bits |= DIRTY_PROGRAM;
}

void mglDetachShader(GLMContext ctx, GLuint program, GLuint shader)
{
    Program *pptr;
    Shader *sptr;
    GLuint index;

    pptr = findProgram(ctx, program);
    if (!pptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    sptr = findShader(ctx, shader);

    if (!sptr)
    {
        // If not found in hash table, check if it is attached to the program
        for (int i=0; i<_MAX_SHADER_TYPES; i++) {
            if (pptr->shader_slots[i] && pptr->shader_slots[i]->name == shader) {
                sptr = pptr->shader_slots[i];
                break;
            }
        }
    }

    if (!sptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    index = sptr->glm_type;

    if (pptr->shader_slots[index] != sptr)
    {
        return;
    }

    mglFlushPendingDraws(ctx);

    pptr->shader_slots[index] = NULL;
    sptr->refcount--;
    
    if (sptr->refcount == 0 && sptr->delete_status)
    {
        mglFreeShader(ctx, sptr);
    }
    
    pptr->dirty_bits |= DIRTY_PROGRAM;
}

void error_callback(void *userdata, const char *error)
{
    if (!error)
        return;
    DEBUG_PRINT("parseSPIRVShader error:%s\n", error);
}


static_assert(_VERTEX_SHADER == GLSLANG_STAGE_VERTEX, "_VERTEX_SHADER == GLSLANG_STAGE_VERTEX failed");
static_assert(_TESS_CONTROL_SHADER == GLSLANG_STAGE_TESSCONTROL, "_TESS_CONTROL_SHADER == GLSLANG_STAGE_TESSCONTROL failed");
static_assert(_TESS_EVALUATION_SHADER == GLSLANG_STAGE_TESSEVALUATION, "_TESS_EVALUATION_SHADER == GLSLANG_STAGE_TESSEVALUATION failed");
static_assert(_GEOMETRY_SHADER == GLSLANG_STAGE_GEOMETRY, "_GEOMETRY_SHADER == GLSLANG_STAGE_GEOMETRY failed");
static_assert(_FRAGMENT_SHADER == GLSLANG_STAGE_FRAGMENT, "_FRAGMENT_SHADER == GLSLANG_STAGE_FRAGMENT failed");
static_assert(_COMPUTE_SHADER == GLSLANG_STAGE_COMPUTE, "_COMPUTE_SHADER == GLSLANG_STAGE_COMPUTE failed");

void addShadersToProgram(GLMContext ctx, Program *pptr, glslang_program_t *glsl_program)
{
    // add shaders
    for(int i=0;i<_MAX_SHADER_TYPES; i++)
    {
        Shader *ptr;

        ptr = pptr->shader_slots[i];

        if(ptr)
        {
            // should have glsl shader here
            if (!ptr->compiled_glsl_shader) {
                fprintf(stderr,
                        "MGL ERROR: program %u shader stage %d has no compiled GLSL shader\n",
                        pptr ? pptr->name : 0u,
                        i);
                if (ctx)
                    STATE(error) = GL_INVALID_OPERATION;
                continue;
            }

            glslang_program_add_shader(glsl_program, ptr->compiled_glsl_shader);
        }
    }
}

static void replace_all_substr(char **pstr, const char *from, const char *to)
{
    char *src;
    char *pos;
    size_t from_len;
    size_t to_len;
    size_t count = 0;
    size_t src_len;
    size_t new_len;
    char *dst;
    char *out;

    if (!pstr || !*pstr || !from || !to) {
        return;
    }

    src = *pstr;
    from_len = strlen(from);
    to_len = strlen(to);
    if (from_len == 0) {
        return;
    }

    pos = src;
    while ((pos = strstr(pos, from)) != NULL) {
        count++;
        pos += from_len;
    }

    if (count == 0) {
        return;
    }

    src_len = strlen(src);
    if (to_len >= from_len) {
        new_len = src_len + count * (to_len - from_len);
    } else {
        new_len = src_len - count * (from_len - to_len);
    }
    out = (char *)malloc(new_len + 1);
    if (!out) {
        return;
    }

    pos = src;
    dst = out;
    while (1) {
        char *match = strstr(pos, from);
        size_t chunk_len;
        if (!match) {
            strcpy(dst, pos);
            break;
        }
        chunk_len = (size_t)(match - pos);
        memcpy(dst, pos, chunk_len);
        dst += chunk_len;
        memcpy(dst, to, to_len);
        dst += to_len;
        pos = match + from_len;
    }

    free(*pstr);
    *pstr = out;
}

static size_t count_substr(const char *str, const char *needle)
{
    size_t count = 0;
    size_t needle_len;
    const char *pos;

    if (!str || !needle) {
        return 0;
    }

    needle_len = strlen(needle);
    if (needle_len == 0) {
        return 0;
    }

    pos = str;
    while ((pos = strstr(pos, needle)) != NULL) {
        count++;
        pos += needle_len;
    }

    return count;
}

static GLboolean mglProgramStageHasResourceName(Program *program, int stage, int res_type, const char *name)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        res_type < 0 || res_type >= _MAX_SPIRV_RES || !name) {
        return GL_FALSE;
    }

    SpirvResourceList *resources = &program->spirv_resources_list[stage][res_type];
    for (GLuint i = 0; resources->list && i < resources->count; i++) {
        if (resources->list[i].name && strcmp(resources->list[i].name, name) == 0) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static void applyMSLCloudVertexIDFix(Program *pptr, int stage, char **msl_ptr)
{
    /* Metal's [[vertex_id]] for indexed draws already carries the index-buffer
     * value, matching OpenGL gl_VertexID for the CloudFaces shader. */
    (void)pptr; (void)stage; (void)msl_ptr;
}

static GLboolean mglVertexShaderLooksLikeMinecraftBlitScreen(Program *program)
{
    return mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "Position") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "UV") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "Color") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, "ModelViewMat") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, "ProjMat") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_OUTPUT, "texCoord") &&
           mglProgramStageHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_OUTPUT, "vertexColor");
}

static GLboolean mglFindMSLUserLocationForName(const char *msl, const char *name, GLuint *location_out)
{
    if (!msl || !name || !location_out) {
        return GL_FALSE;
    }

    size_t name_len = strlen(name);
    if (name_len == 0) {
        return GL_FALSE;
    }

    const char *cursor = msl;
    while ((cursor = strstr(cursor, name)) != NULL) {
        const char *line_start = cursor;
        const char *line_end = cursor;

        while (line_start > msl && line_start[-1] != '\n' && line_start[-1] != '\r') {
            line_start--;
        }
        while (*line_end && *line_end != '\n' && *line_end != '\r') {
            line_end++;
        }

        char before = (cursor == line_start) ? '\0' : cursor[-1];
        char after = cursor[name_len];
        GLboolean before_ident = (before == '_') ||
                                  (before >= '0' && before <= '9') ||
                                  (before >= 'A' && before <= 'Z') ||
                                  (before >= 'a' && before <= 'z');
        GLboolean after_ident = (after == '_') ||
                                 (after >= '0' && after <= '9') ||
                                 (after >= 'A' && after <= 'Z') ||
                                 (after >= 'a' && after <= 'z');
        if (!before_ident && !after_ident) {
            const char *loc = strstr(cursor, "[[user(locn");
            if (loc && loc < line_end) {
                loc += strlen("[[user(locn");
                char *end = NULL;
                unsigned long parsed = strtoul(loc, &end, 10);
                if (end && end > loc && end <= line_end) {
                    *location_out = (GLuint)parsed;
                    return GL_TRUE;
                }
            }
        }

        cursor += name_len;
    }

    return GL_FALSE;
}

static bool mglUniformStructName(const char *name)
{
    if (!name || !*name) {
        return false;
    }

    size_t len = strlen(name);
    if ((len >= 3 && strcmp(name + len - 3, "_in") == 0) ||
        (len >= 4 && strcmp(name + len - 4, "_out") == 0)) {
        return false;
    }

    return true;
}

static void applyMSLUniformBufferPacking(Program *pptr, int stage)
{
    if (!pptr || !pptr->spirv[stage].msl_str) {
        return;
    }

    /*
     * Uniform buffers use GL/std140 layout. A vec3 member still occupies a
     * 16-byte slot there, and Metal's float3 struct alignment matches that
     * requirement. Rewriting uniform float3 fields to packed_float3 shifts
     * subsequent members, e.g. CloudInfo.CellSize moves from offset 32 to 28.
     */
    return;

    const char *src = pptr->spirv[stage].msl_str;
    size_t len = strlen(src);
    size_t cap = len + 1;
    char *out = (char *)malloc(cap);
    if (!out) {
        return;
    }

    size_t out_len = 0;
    bool in_struct = false;
    bool patch_struct = false;
    unsigned patched = 0;

    const char *p = src;
    while (*p) {
        const char *line_start = p;
        const char *line_end = strchr(p, '\n');
        size_t line_len = line_end ? (size_t)(line_end - line_start + 1) : strlen(line_start);
        size_t content_len = line_end ? line_len - 1 : line_len;

        if (!in_struct) {
            char struct_name[128] = {0};
            if (sscanf(line_start, "struct %127s", struct_name) == 1) {
                char *brace = strchr(struct_name, '{');
                if (brace) {
                    *brace = '\0';
                }
                in_struct = true;
                patch_struct = mglUniformStructName(struct_name);
            }
        }

        const char *needle = "float3 ";
        const char *match = NULL;
        if (in_struct && patch_struct) {
            match = strstr(line_start, needle);
            if (match && match >= line_start + content_len) {
                match = NULL;
            }
        }

        size_t replacement_len = strlen("packed_float3 ");
        size_t needle_len = strlen(needle);
        size_t extra = match ? replacement_len - needle_len : 0;
        if (out_len + line_len + extra + 1 > cap) {
            while (out_len + line_len + extra + 1 > cap) {
                cap *= 2;
            }
            char *grown = (char *)realloc(out, cap);
            if (!grown) {
                free(out);
                return;
            }
            out = grown;
        }

        if (match) {
            size_t prefix_len = (size_t)(match - line_start);
            memcpy(out + out_len, line_start, prefix_len);
            out_len += prefix_len;
            memcpy(out + out_len, "packed_float3 ", replacement_len);
            out_len += replacement_len;
            size_t suffix_off = prefix_len + needle_len;
            memcpy(out + out_len, line_start + suffix_off, line_len - suffix_off);
            out_len += line_len - suffix_off;
            patched++;
        } else {
            memcpy(out + out_len, line_start, line_len);
            out_len += line_len;
        }

        if (in_struct && memchr(line_start, '}', content_len)) {
            in_struct = false;
            patch_struct = false;
        }

        p = line_end ? line_end + 1 : line_start + line_len;
    }

    out[out_len] = '\0';
    if (patched > 0) {
        fprintf(stderr,
                "MGL MSL PACKING FIX: program=%u stage=%d converted %u uniform float3 field(s) to packed_float3\n",
                pptr->name,
                stage,
                patched);
        free(pptr->spirv[stage].msl_str);
        pptr->spirv[stage].msl_str = out;
    } else {
        free(out);
    }
}

char *parseSPIRVShaderToMetal(GLMContext ctx, Program *ptr, int stage)
{
    const SpvId *spirv;
    size_t word_count;
    char *str_ret;
    int parse_res;

    spvc_context context = NULL;
    spvc_parsed_ir ir = NULL;
    spvc_compiler compiler_msl = NULL;
    spvc_compiler_options options = NULL;
    spvc_resources resources = NULL;
    const spvc_reflected_resource *list = NULL;
    const char *result = NULL;
    size_t count;
    size_t i;

    if (!ptr || stage < 0 || stage >= _MAX_SHADER_TYPES || !ptr->shader_slots[stage]) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    spirv = ptr->spirv[stage].ir;
    word_count = ptr->spirv[stage].size;
    if (!spirv || word_count == 0) {
        fprintf(stderr,
                "MGL ERROR: parseSPIRVShaderToMetal missing SPIR-V program=%u stage=%d words=%zu\n",
                ptr->name,
                stage,
                word_count);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // Create context.
    if (spvc_context_create(&context) != SPVC_SUCCESS || !context) {
        fprintf(stderr, "MGL ERROR: spvc_context_create failed\n");
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // Set debug callback.
    spvc_context_set_error_callback(context, error_callback, ctx);

    // Parse the SPIR-V.
    parse_res = spvc_context_parse_spirv(context, spirv, word_count, &ir);
    if (parse_res != SPVC_SUCCESS || !ir) {
        fprintf(stderr,
                "MGL ERROR: spvc_context_parse_spirv failed program=%u stage=%d err=%d\n",
                ptr->name,
                stage,
                parse_res);
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // Hand it off to a compiler instance and give it ownership of the IR.
    if (spvc_context_create_compiler(context, SPVC_BACKEND_MSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &compiler_msl) != SPVC_SUCCESS ||
        !compiler_msl) {
        fprintf(stderr, "MGL ERROR: spvc_context_create_compiler failed program=%u stage=%d\n", ptr->name, stage);
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }
    // ERROR_CHECK_RETURN(spvc_compiler_msl_add_discrete_descriptor_set(compiler_msl, 3) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    if (spvc_compiler_msl_add_discrete_descriptor_set(compiler_msl, 3) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_msl_add_discrete_descriptor_set failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // Modify options.
    // ERROR_CHECK_RETURN(spvc_compiler_create_compiler_options(compiler_msl, &options) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    if (spvc_compiler_create_compiler_options(compiler_msl, &options) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_create_compiler_options failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // ERROR_CHECK_RETURN(spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    if (spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_options_set_bool(SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS) failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // ERROR_CHECK_RETURN(spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(3,1,0)) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    if (spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(3,1,0)) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_options_set_uint(SPVC_COMPILER_OPTION_MSL_VERSION) failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    if (spvc_compiler_options_set_uint(options,
                                       SPVC_COMPILER_OPTION_MSL_TEXEL_BUFFER_TEXTURE_WIDTH,
                                       MGL_TEXEL_BUFFER_TEXTURE_WIDTH) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_options_set_uint(SPVC_COMPILER_OPTION_MSL_TEXEL_BUFFER_TEXTURE_WIDTH) failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    if (spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_FIXUP_DEPTH_CONVENTION, SPVC_TRUE) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_options_set_bool(SPVC_COMPILER_OPTION_FIXUP_DEPTH_CONVENTION) failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    //ERROR_CHECK_RETURN(spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_GLSL_VERSION, 4.5) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    // ERROR_CHECK_RETURN(spvc_compiler_install_compiler_options(compiler_msl, options) == SPVC_SUCCESS, GL_INVALID_OPERATION);
    if (spvc_compiler_install_compiler_options(compiler_msl, options) != SPVC_SUCCESS) {
        fprintf(stderr, "MGL Error: spvc_compiler_install_compiler_options failed\n");
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    
    // create an entry point for metal based on the shader type and name
    GLuint name;
    char entry_point[128];
    name = ptr->shader_slots[stage]->name;

    SpvExecutionModel model = SpvExecutionModelVertex; // CRITICAL FIX: Initialize with safe default
    switch(stage)
    {
        case _VERTEX_SHADER: model = SpvExecutionModelVertex; break;
        case _TESS_CONTROL_SHADER: model = SpvExecutionModelTessellationControl; break;
        case _TESS_EVALUATION_SHADER: model = SpvExecutionModelTessellationEvaluation; break;
        case _GEOMETRY_SHADER: model = SpvExecutionModelGeometry; break;
        case _FRAGMENT_SHADER: model = SpvExecutionModelFragment; break;
        case _COMPUTE_SHADER: model = SpvExecutionModelGLCompute; break;
        default: // CRITICAL FIX: Handle error gracefully instead of crashing
            fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
            STATE(error) = GL_INVALID_OPERATION;
            return NULL;
    }

    switch(stage)
    {
        case _VERTEX_SHADER: snprintf(entry_point, sizeof(entry_point), "vertex_%d_main",name); break;
        case _TESS_CONTROL_SHADER: snprintf(entry_point, sizeof(entry_point), "tess_control_%d_main",name); break;
        case _TESS_EVALUATION_SHADER: snprintf(entry_point, sizeof(entry_point), "tess_evaluation_%d_main",name); break;
        case _GEOMETRY_SHADER: snprintf(entry_point, sizeof(entry_point), "geometry_%d",name); break;
        case _FRAGMENT_SHADER: snprintf(entry_point, sizeof(entry_point), "fragment_%d",name); break;
        case _COMPUTE_SHADER: snprintf(entry_point, sizeof(entry_point), "compute_%d",name); break;
        default: // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
    }

    const char *cleansed_entry_point;
    cleansed_entry_point = spvc_compiler_get_cleansed_entry_point_name(compiler_msl, "main", model);

    spvc_result err;
    err = spvc_compiler_rename_entry_point(compiler_msl, cleansed_entry_point, entry_point, model);
    if (err != SPVC_SUCCESS) {
        fprintf(stderr,
                "MGL ERROR: spvc_compiler_rename_entry_point failed program=%u stage=%d err=%d\n",
                ptr->name,
                stage,
                err);
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // set the entry point for metal
    ptr->shader_slots[stage]->entry_point = strdup(entry_point);
    ptr->spirv[stage].entry_point = strdup(entry_point);

    // compute shader
    if (stage == _COMPUTE_SHADER)
    {
        spvc_result res;
        const spvc_entry_point *entry_points;
        size_t num_entry_points;

        res = spvc_compiler_get_entry_points(compiler_msl, &entry_points, &num_entry_points);
        if (res != SPVC_SUCCESS) {
            fprintf(stderr,
                    "MGL ERROR: spvc_compiler_get_entry_points failed program=%u err=%d\n",
                    ptr->name,
                    res);
            spvc_context_destroy(context);
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
        }
        
        for(int i=0; i<num_entry_points; i++)
        {
            DEBUG_PRINT("Entry point: %s Execution Model: %d\n", entry_points[i].name, entry_points[i].execution_model);
        }

        ptr->local_workgroup_size.x = spvc_compiler_get_execution_mode_argument_by_index(compiler_msl, SpvExecutionModeLocalSize, 0);
        ptr->local_workgroup_size.y = spvc_compiler_get_execution_mode_argument_by_index(compiler_msl, SpvExecutionModeLocalSize, 1);
        ptr->local_workgroup_size.z = spvc_compiler_get_execution_mode_argument_by_index(compiler_msl, SpvExecutionModeLocalSize, 2);
    }
    
    // Do some basic reflection.
    if (spvc_compiler_create_shader_resources(compiler_msl, &resources) != SPVC_SUCCESS || !resources) {
        fprintf(stderr,
                "MGL ERROR: spvc_compiler_create_shader_resources failed program=%u stage=%d\n",
                ptr->name,
                stage);
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }
    for (int res_type=SPVC_RESOURCE_TYPE_UNIFORM_BUFFER; res_type < SPVC_RESOURCE_TYPE_ACCELERATION_STRUCTURE; res_type++)
    {
#if DEBUG
        const char *res_name[] = {"NONE", "UNIFORM_BUFFER", "UNIFORM_CONSTANT", "STORAGE_BUFFER", "STAGE_INPUT", "STAGE_OUTPUT",
            "SUBPASS_INPUT", "STORAGE_INPUT", "SAMPLED_IMAGE", "ATOMIC_COUNTER", "PUSH_CONSTANT", "SEPARATE_IMAGE",
            "SEPARATE_SAMPLERS", "ACCELERATION_STRUCTURE", "RAY_QUERY"};
#endif
        
        spvc_resources_get_resource_list_for_type(resources, res_type, &list, &count);

        ptr->spirv_resources_list[stage][res_type].count = (GLuint)count;

        // CRITICAL SECURITY FIX: Prevent integer overflow in resource allocation
        // Check if count * sizeof(SpirvResource) would overflow size_t
        if (count > SIZE_MAX / sizeof(SpirvResource)) {
            fprintf(stderr, "MGL SECURITY ERROR: Resource count %zu would cause allocation overflow\n", count);
            spvc_context_destroy(context);
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, NULL);
        }

        size_t alloc_size = count * sizeof(SpirvResource);
        if (count == 0) {
            ptr->spirv_resources_list[stage][res_type].list = NULL;
        } else {
            ptr->spirv_resources_list[stage][res_type].list = (SpirvResource *)malloc(alloc_size);
        }
        if (count != 0 && !ptr->spirv_resources_list[stage][res_type].list) {
            fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate %zu bytes for resource list\n", alloc_size);
            spvc_context_destroy(context);
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, NULL);
        }

        for (i = 0; i < count; i++)
        {
            DEBUG_PRINT("res_type: %s ID: %u, BaseTypeID: %u, TypeID: %u, Name: %s ", res_name[res_type], list[i].id, list[i].base_type_id, list[i].type_id,
                   list[i].name);
            
            switch(res_type)
            {
                case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
                case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
                case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
                case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
                    DEBUG_PRINT("Set: %u, Binding: %u Uniform: %d offset: %d\n",
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationDescriptorSet),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationBinding),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationUniform),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationOffset));
                    break;

                case SPVC_RESOURCE_TYPE_STAGE_INPUT:
                case SPVC_RESOURCE_TYPE_STAGE_OUTPUT:
                case SPVC_RESOURCE_TYPE_SUBPASS_INPUT:
                    DEBUG_PRINT("Set: %u, Location: %d Index: %d, offset: %d\n",
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationDescriptorSet),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationLocation),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationIndex),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationOffset));
                    break;
                    
                case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
                case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
                    DEBUG_PRINT("Set: %u, Location: %d\n",
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationDescriptorSet),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationLocation));
                    break;

                default:
                    DEBUG_PRINT("Set: %u, Binding: %u Location: %d Index: %d, Uniform: %d offset: %d\n",
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationDescriptorSet),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationBinding),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationLocation),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationIndex),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationUniform),
                           spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationOffset));
                    break;
            }
            
            spvc_type reflected_type = NULL;
            spvc_basetype reflected_basetype = SPVC_BASETYPE_UNKNOWN;
            if (list[i].type_id) {
                reflected_type = spvc_compiler_get_type_handle(compiler_msl, list[i].type_id);
            }
            if (!reflected_type && list[i].base_type_id) {
                reflected_type = spvc_compiler_get_type_handle(compiler_msl, list[i].base_type_id);
            }
            if (reflected_type) {
                reflected_basetype = spvc_type_get_basetype(reflected_type);
            }

            bool uniform_constant_sampler_like =
                (res_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT) &&
                (mglUniformConstantBaseTypeIsSamplerLike(reflected_basetype) ||
                 mglUniformNameLooksSamplerLike(list[i].name));

            ptr->spirv_resources_list[stage][res_type].list[i]._id = list[i].id;
            ptr->spirv_resources_list[stage][res_type].list[i].base_type_id = list[i].base_type_id;
            ptr->spirv_resources_list[stage][res_type].list[i].type_id = list[i].type_id;
            ptr->spirv_resources_list[stage][res_type].list[i].name = strdup(list[i].name);
            ptr->spirv_resources_list[stage][res_type].list[i].set = spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationDescriptorSet);
            ptr->spirv_resources_list[stage][res_type].list[i].gl_binding = spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationBinding);
            ptr->spirv_resources_list[stage][res_type].list[i].binding = ptr->spirv_resources_list[stage][res_type].list[i].gl_binding;
            ptr->spirv_resources_list[stage][res_type].list[i].location = spvc_compiler_get_decoration(compiler_msl, list[i].id, SpvDecorationLocation);
            ptr->spirv_resources_list[stage][res_type].list[i].uniform_location =
                (mglIsSamplerResourceType(res_type) || uniform_constant_sampler_like)
                    ? mglSyntheticSamplerUniformLocation(stage, res_type, (GLuint)i)
                    : -1;
            ptr->spirv_resources_list[stage][res_type].list[i].sampler_unit = -1;
            ptr->spirv_resources_list[stage][res_type].list[i].sampler_unit_explicit = GL_FALSE;
            ptr->spirv_resources_list[stage][res_type].list[i].required_size = 0;
            ptr->spirv_resources_list[stage][res_type].list[i].image_dim = 0;
            ptr->spirv_resources_list[stage][res_type].list[i].image_arrayed = 0;
            ptr->spirv_resources_list[stage][res_type].list[i].image_multisampled = 0;

            bool resource_has_image_type =
                res_type == SPVC_RESOURCE_TYPE_SAMPLED_IMAGE ||
                res_type == SPVC_RESOURCE_TYPE_SEPARATE_IMAGE ||
                res_type == SPVC_RESOURCE_TYPE_STORAGE_IMAGE ||
                (res_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT &&
                 (reflected_basetype == SPVC_BASETYPE_IMAGE ||
                  reflected_basetype == SPVC_BASETYPE_SAMPLED_IMAGE));

            if (resource_has_image_type) {
                spvc_type image_type = reflected_type;

                if (image_type) {
                    ptr->spirv_resources_list[stage][res_type].list[i].image_dim =
                        (GLuint)spvc_type_get_image_dimension(image_type);
                    ptr->spirv_resources_list[stage][res_type].list[i].image_arrayed =
                        (GLuint)spvc_type_get_image_arrayed(image_type);
                    ptr->spirv_resources_list[stage][res_type].list[i].image_multisampled =
                        (GLuint)spvc_type_get_image_multisampled(image_type);

                    if (ptr->spirv_resources_list[stage][res_type].list[i].image_dim == (GLuint)SpvDimCube) {
                        fprintf(stderr,
                                "MGL SPIRV IMAGE resource program=%u stage=%d type=%d name=%s binding=%u dim=Cube arrayed=%u multisampled=%u\n",
                                ptr->name,
                                stage,
                                res_type,
                                list[i].name ? list[i].name : "(null)",
                                ptr->spirv_resources_list[stage][res_type].list[i].binding,
                                ptr->spirv_resources_list[stage][res_type].list[i].image_arrayed,
                                ptr->spirv_resources_list[stage][res_type].list[i].image_multisampled);
                    }
                }
            }

            if (res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER ||
                res_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT ||
                res_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER ||
                res_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER ||
                res_type == SPVC_RESOURCE_TYPE_PUSH_CONSTANT) {
                size_t declared_size = 0;

                if (reflected_type && spvc_type_get_basetype(reflected_type) == SPVC_BASETYPE_STRUCT) {
                    size_t struct_size = 0;
                    if (spvc_compiler_get_declared_struct_size(compiler_msl, reflected_type, &struct_size) == SPVC_SUCCESS) {
                        declared_size = struct_size;
                    }
                }

                /* Some Minecraft 1.21 shaders can make SPIRV-Cross crash while
                 * traversing active buffer ranges. The declared struct size is
                 * enough for our uniform buffer sizing, so avoid that fragile
                 * optional reflection pass. */

                if (res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER && declared_size > 0) {
                    declared_size = mglRoundUpSize(declared_size, 16);
                }

                ptr->spirv_resources_list[stage][res_type].list[i].required_size = declared_size;
            }
        }
    }

    if (spvc_compiler_compile(compiler_msl, &result) != SPVC_SUCCESS || !result) {
        fprintf(stderr,
                "MGL ERROR: spvc_compiler_compile failed program=%u stage=%d\n",
                ptr->name,
                stage);
        spvc_context_destroy(context);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }
    DEBUG_PRINT("\n%s\n", result);

    str_ret = strdup(result);
    if (str_ret) {
        /* Some generated MSL uses `sampler` as an identifier, which collides
         * with Metal's `sampler` type in function signatures. Normalize these
         * generated helper names to keep compilation valid. */
        replace_all_substr(&str_ret,
                           "texture2d<float> sampler, sampler samplerSmplr",
                           "texture2d<float> sourceTex, sampler sourceSmplr");
        replace_all_substr(&str_ret,
                           " sampler.sample(samplerSmplr,",
                           " sourceTex.sample(sourceSmplr,");
        replace_all_substr(&str_ret, "thread const bool&", "bool");
        replace_all_substr(&str_ret, "thread const int&", "int");
        replace_all_substr(&str_ret, "thread const uint&", "uint");
        replace_all_substr(&str_ret, "thread const float&", "float");
        replace_all_substr(&str_ret, "thread const float2&", "float2");
        replace_all_substr(&str_ret, "thread const float3&", "float3");
        replace_all_substr(&str_ret, "thread const float4&", "float4");
        replace_all_substr(&str_ret, "thread const int2&", "int2");
        replace_all_substr(&str_ret, "thread const int3&", "int3");
        replace_all_substr(&str_ret, "thread const int4&", "int4");
        replace_all_substr(&str_ret, "thread const uint2&", "uint2");
        replace_all_substr(&str_ret, "thread const uint3&", "uint3");
        replace_all_substr(&str_ret, "thread const uint4&", "uint4");
        replace_all_substr(&str_ret, "thread const float3x3&", "float3x3");
        replace_all_substr(&str_ret, "thread const float4x4&", "float4x4");
        replace_all_substr(&str_ret,
                           "float4x4 end_portal_layer(thread const float& layer, thread const float& GameTime)",
                           "float4x4 end_portal_layer(float layer, float GameTime)");
        if (stage == _VERTEX_SHADER &&
            strstr(str_ret, "float2 screenPos = (in.Position.xy * 2.0) - float2(1.0);") &&
            strstr(str_ret, "out.gl_Position = float4(screenPos.x, screenPos.y, 1.0, 1.0);") &&
            strstr(str_ret, "out.texCoord = in.Position.xy;")) {
            fprintf(stderr,
                    "MGL MSL FULLSCREEN FIX: program=%u flips sampled framebuffer texcoord Y\n",
                    ptr->name);
            replace_all_substr(&str_ret,
                               "out.texCoord = in.Position.xy;",
                               "out.texCoord = float2(in.Position.x, 1.0 - in.Position.y);");
        }
        if (stage == _VERTEX_SHADER && mglVertexShaderLooksLikeMinecraftBlitScreen(ptr)) {
            size_t fix_count = 0;
            static const struct {
                const char *from;
                const char *to;
            } blit_screen_uv_rewrites[] = {
                { "out.texCoord = in.UV;", "out.texCoord = float2(in.UV.x, 1.0 - in.UV.y);" },
                { "out.texCoord = in.UV0;", "out.texCoord = float2(in.UV0.x, 1.0 - in.UV0.y);" },
                { "out.texCoord = in._mgl_in_UV;", "out.texCoord = float2(in._mgl_in_UV.x, 1.0 - in._mgl_in_UV.y);" },
                { "out.texCoord = in.in_var_UV;", "out.texCoord = float2(in.in_var_UV.x, 1.0 - in.in_var_UV.y);" },
                { "out.texCoord = in.in_var_TEXCOORD0;", "out.texCoord = float2(in.in_var_TEXCOORD0.x, 1.0 - in.in_var_TEXCOORD0.y);" },
            };

            for (size_t i = 0; i < sizeof(blit_screen_uv_rewrites) / sizeof(blit_screen_uv_rewrites[0]); i++) {
                size_t hits = count_substr(str_ret, blit_screen_uv_rewrites[i].from);
                if (hits > 0) {
                    replace_all_substr(&str_ret,
                                       blit_screen_uv_rewrites[i].from,
                                       blit_screen_uv_rewrites[i].to);
                    fix_count += hits;
                }
            }

            if (fix_count > 0) {
                fprintf(stderr,
                        "MGL MSL BLIT_SCREEN FIX: program=%u flips sampled framebuffer texcoord Y hits=%zu\n",
                        ptr->name,
                        fix_count);
            } else {
                const char *texcoord = strstr(str_ret, "texCoord");
                char snippet[257] = {0};
                if (texcoord) {
                    size_t copy_len = strlen(texcoord);
                    if (copy_len > sizeof(snippet) - 1) {
                        copy_len = sizeof(snippet) - 1;
                    }
                    memcpy(snippet, texcoord, copy_len);
                    for (size_t i = 0; i < copy_len; i++) {
                        if (snippet[i] == '\n' || snippet[i] == '\r' || snippet[i] == '\t') {
                            snippet[i] = ' ';
                        }
                    }
                }
                fprintf(stderr,
                        "MGL MSL BLIT_SCREEN WARNING: program=%u matched resources but no UV assignment pattern; snippet=%s\n",
                        ptr->name,
                        snippet[0] ? snippet : "(no texCoord token)");
            }
        }
        applyMSLCloudVertexIDFix(ptr, stage, &str_ret);
        applyMSLResourceBindings(ptr, stage, str_ret);
    }

    // Frees all memory we allocated so far.
    spvc_context_destroy(context);

    return str_ret;
}

static void clearStageCompileState(Program *pptr, int stage)
{
    if (pptr->spirv[stage].ir) {
        free(pptr->spirv[stage].ir);
        pptr->spirv[stage].ir = NULL;
    }
    if (pptr->spirv[stage].msl_str) {
        free(pptr->spirv[stage].msl_str);
        pptr->spirv[stage].msl_str = NULL;
    }
    if (pptr->spirv[stage].entry_point) {
        free(pptr->spirv[stage].entry_point);
        pptr->spirv[stage].entry_point = NULL;
    }
    if (pptr->spirv[stage].mtl_function) {
        CFRelease(pptr->spirv[stage].mtl_function);
        pptr->spirv[stage].mtl_function = NULL;
    }
    if (pptr->spirv[stage].mtl_library) {
        CFRelease(pptr->spirv[stage].mtl_library);
        pptr->spirv[stage].mtl_library = NULL;
    }

    for (int res_type = 0; res_type < _MAX_SPIRV_RES; res_type++) {
        if (pptr->spirv_resources_list[stage][res_type].list) {
            free(pptr->spirv_resources_list[stage][res_type].list);
            pptr->spirv_resources_list[stage][res_type].list = NULL;
        }
        pptr->spirv_resources_list[stage][res_type].count = 0;
    }
}

static GLint mglDefaultAttribLocationForName(const char *name)
{
    if (!name) {
        return -1;
    }

    if (strcmp(name, "Position") == 0) return 0;
    if (strcmp(name, "Color") == 0) return 1;
    if (strcmp(name, "UV0") == 0) return 2;
    if (strcmp(name, "UV1") == 0) return 3;
    if (strcmp(name, "UV2") == 0) return 4;
    if (strcmp(name, "Normal") == 0) return 5;

    return -1;
}

static GLint mglProgramVertexInputOrdinal(Program *pptr, const char *name)
{
    if (!pptr || !name) {
        return -1;
    }

    SpirvResourceList *vertex_inputs =
        &pptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    if (!vertex_inputs->list) {
        return -1;
    }

    for (GLuint i = 0; i < vertex_inputs->count; i++) {
        const char *input_name = vertex_inputs->list[i].name;
        if (input_name && strcmp(input_name, name) == 0) {
            return (GLint)i;
        }
    }

    return -1;
}

static GLboolean mglProgramHasVertexInputNamed(Program *pptr, const char *name)
{
    return mglProgramVertexInputOrdinal(pptr, name) >= 0 ? GL_TRUE : GL_FALSE;
}

static GLint mglContextualDefaultAttribLocationForName(Program *pptr, const char *name)
{
    if (!pptr || !name) {
        return -1;
    }

    /*
     * Mojang's shader names are stable, but the set of inputs is not. Newer
     * GUI/item shaders often omit UV0 or UV1, so the vanilla fallback table
     * must collapse around the attributes that are actually present.
     */
    GLboolean has_color = mglProgramHasVertexInputNamed(pptr, "Color");
    GLboolean has_uv0 = mglProgramHasVertexInputNamed(pptr, "UV0");
    GLboolean has_uv1 = mglProgramHasVertexInputNamed(pptr, "UV1");
    GLboolean has_uv2 = mglProgramHasVertexInputNamed(pptr, "UV2");

    if (strcmp(name, "UV2") == 0) {
        if (!has_uv0 && !has_uv1) {
            return 2;
        }
        if (has_uv0 && !has_uv1) {
            return 3;
        }
        return 4;
    }

    if (strcmp(name, "Normal") == 0) {
        if (has_uv2 && !has_uv1) {
            return has_uv0 ? 4 : 3;
        }
        return 5;
    }

    if (has_color && has_uv0 && !has_uv1 && !has_uv2) {
        GLint color_ordinal = mglProgramVertexInputOrdinal(pptr, "Color");
        GLint uv0_ordinal = mglProgramVertexInputOrdinal(pptr, "UV0");
        if (uv0_ordinal >= 0 && color_ordinal >= 0 &&
            uv0_ordinal < color_ordinal) {
            if (strcmp(name, "UV0") == 0) {
                return 1;
            }
            if (strcmp(name, "Color") == 0) {
                return 2;
            }
        }
    }

    return mglDefaultAttribLocationForName(name);
}

static GLint mglDesiredAttribLocationForName(Program *pptr, const char *name)
{
    if (!pptr || !name) {
        return -1;
    }

    for (int i = 0; i < MAX_ATTRIBS; i++) {
        if (pptr->attrib_location_names[i] &&
            strcmp(pptr->attrib_location_names[i], name) == 0) {
            return i;
        }
    }

    return mglContextualDefaultAttribLocationForName(pptr, name);
}

static void applyVertexInputLocations(Program *pptr)
{
    if (!pptr || !pptr->spirv[_VERTEX_SHADER].msl_str) {
        return;
    }

    SpirvResourceList *vertex_inputs =
        &pptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    if (!vertex_inputs->list) {
        return;
    }

    for (GLuint i = 0; i < vertex_inputs->count; i++) {
        SpirvResource *vs_in = &vertex_inputs->list[i];
        GLint desiredLocation = mglDesiredAttribLocationForName(pptr, vs_in->name);
        if (desiredLocation < 0 || desiredLocation >= MAX_ATTRIBS ||
            vs_in->location == (GLuint)desiredLocation) {
            continue;
        }

        char from[256];
        char to[256];
        snprintf(from, sizeof(from), "%s [[attribute(%u)]]",
                 vs_in->name, (unsigned)vs_in->location);
        snprintf(to, sizeof(to), "%s [[attribute(%u)]]",
                 vs_in->name, (unsigned)desiredLocation);

        if (strstr(pptr->spirv[_VERTEX_SHADER].msl_str, from)) {
            fprintf(stderr,
                    "MGL ATTRIB FIX: program=%u vertex input %s loc %u -> %d\n",
                    pptr->name,
                    vs_in->name,
                    (unsigned)vs_in->location,
                    desiredLocation);
            replace_all_substr(&pptr->spirv[_VERTEX_SHADER].msl_str, from, to);
            vs_in->location = (GLuint)desiredLocation;
        } else {
            fprintf(stderr,
                    "MGL ATTRIB WARNING: program=%u wanted %s loc %u -> %d but MSL pattern was not found\n",
                    pptr->name,
                    vs_in->name,
                    (unsigned)vs_in->location,
                    desiredLocation);
        }
    }
}

static void alignFragmentInputLocationsToVertexOutputs(Program *pptr)
{
    if (!pptr ||
        !pptr->spirv[_FRAGMENT_SHADER].msl_str ||
        !pptr->spirv[_VERTEX_SHADER].msl_str) {
        return;
    }

    SpirvResourceList *vertex_outputs =
        &pptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
    SpirvResourceList *fragment_inputs =
        &pptr->spirv_resources_list[_FRAGMENT_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];

    if (!vertex_outputs->list || !fragment_inputs->list) {
        return;
    }

    for (GLuint f = 0; f < fragment_inputs->count; f++) {
        SpirvResource *fs_in = &fragment_inputs->list[f];
        if (!fs_in->name || fs_in->name[0] == '\0') {
            continue;
        }

        for (GLuint v = 0; v < vertex_outputs->count; v++) {
            SpirvResource *vs_out = &vertex_outputs->list[v];
            if (!vs_out->name || strcmp(fs_in->name, vs_out->name) != 0) {
                continue;
            }

            GLuint desired_location = vs_out->location;
            GLuint current_location = fs_in->location;
            if (mglFindMSLUserLocationForName(pptr->spirv[_VERTEX_SHADER].msl_str,
                                              vs_out->name,
                                              &desired_location)) {
                vs_out->location = desired_location;
            }
            if (mglFindMSLUserLocationForName(pptr->spirv[_FRAGMENT_SHADER].msl_str,
                                              fs_in->name,
                                              &current_location)) {
                fs_in->location = current_location;
            }

            if (current_location == desired_location) {
                break;
            }

            char from[256];
            char to[256];
            snprintf(from, sizeof(from), "%s [[user(locn%u)]]",
                     fs_in->name, (unsigned)current_location);
            snprintf(to, sizeof(to), "%s [[user(locn%u)]]",
                     fs_in->name, (unsigned)desired_location);

            if (strstr(pptr->spirv[_FRAGMENT_SHADER].msl_str, from)) {
                fprintf(stderr,
                        "MGL IFACE FIX: program=%u fragment input %s loc %u -> %u to match vertex output\n",
                        pptr->name,
                        fs_in->name,
                        (unsigned)current_location,
                        (unsigned)desired_location);
                replace_all_substr(&pptr->spirv[_FRAGMENT_SHADER].msl_str, from, to);
                fs_in->location = desired_location;
            } else {
                fprintf(stderr,
                        "MGL IFACE WARNING: program=%u wanted to align %s loc %u -> %u but MSL pattern was not found\n",
                        pptr->name,
                        fs_in->name,
                        (unsigned)current_location,
                        (unsigned)desired_location);
            }
            break;
        }
    }
}

static bool compileStageFromLinkedProgram(GLMContext ctx, Program *pptr, glslang_program_t *glsl_program, int stage)
{
    const char *spirv_messages;

    /* Safety check: ensure we have a shader for this stage */
    if (!pptr->shader_slots[stage]) {
        return true;
    }

    clearStageCompileState(pptr, stage);

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Generating SPIRV for stage %d\n", stage);
    }
    glslang_program_SPIRV_generate(glsl_program, stage);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: SPIRV generated\n");
    }

    spirv_messages = glslang_program_SPIRV_get_messages(glsl_program);
    if (spirv_messages && spirv_messages[0] != '\0')
    {
        fprintf(stderr, "MGL Error: glslang_program_SPIRV_get_messages:\n%s\n", spirv_messages);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return false;
    }

    // save SPIRV code
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Getting SPIRV size\n");
    }
    pptr->spirv[stage].size = glslang_program_SPIRV_get_size(glsl_program);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: SPIRV size: %zu\n", pptr->spirv[stage].size);
    }

    // CRITICAL SECURITY FIX: Prevent integer overflow in SPIRV allocation
    // Check if size * sizeof(unsigned) would overflow size_t
    if (pptr->spirv[stage].size > SIZE_MAX / sizeof(unsigned)) {
        fprintf(stderr, "MGL SECURITY ERROR: SPIRV size %zu would cause allocation overflow\n", pptr->spirv[stage].size);
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return false;
    }

    size_t alloc_size = pptr->spirv[stage].size * sizeof(unsigned);
    pptr->spirv[stage].ir = (unsigned int *)malloc(alloc_size);
    if (!pptr->spirv[stage].ir) {
        fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate %zu bytes for SPIRV\n", alloc_size);
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return false;
    }
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Getting SPIRV IR\n");
    }
    glslang_program_SPIRV_get(glsl_program, pptr->spirv[stage].ir);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: SPIRV IR obtained\n");
    }

    // compile SPIRV to Metal
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: About to parse SPIRV to Metal\n");
    }
    pptr->spirv[stage].msl_str = parseSPIRVShaderToMetal(ctx, pptr, stage);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: SPIRV parsed to Metal\n");
    }
    if (pptr->spirv[stage].msl_str == NULL) {
        fprintf(stderr, "MGL Error: parseSPIRVShaderToMetal failed for stage %d\n", stage);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return false;
    }
    applyMSLUniformBufferPacking(pptr, stage);

    return true;
}

void mglLinkProgram(GLMContext ctx, GLuint program)
{
    Program *pptr;
    glslang_program_t *glsl_program;
    int err;
    bool link_ok = true;
    bool has_any_shader = false;

    pptr = findProgram(ctx, program);

    if (!pptr)
    {
        // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;

        return;
    }

    mglFlushPendingDraws(ctx);

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        if (pptr->shader_slots[stage]) {
            has_any_shader = true;
            break;
        }
    }

    if (!has_any_shader) {
        fprintf(stderr, "MGL WARNING: mglLinkProgram called with no attached shaders\n");
        pptr->linked_glsl_program = NULL;
        return;
    }

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Creating glslang program for full-link\n");
    }
    glsl_program = glslang_program_create();
    if (!glsl_program) {
        fprintf(stderr, "MGL Error: glslang_program_create failed\n");
        pptr->linked_glsl_program = NULL;
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Adding shaders to program\n");
    }
    addShadersToProgram(ctx, pptr, glsl_program);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Shaders added\n");
    }

    err = glslang_program_link(glsl_program, GLSLANG_MSG_DEFAULT_BIT);
    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL DEBUG: Program link returned %d\n", err);
    }
    if (!err)
    {
        fprintf(stderr, "MGL Error: glslang_program_link failed err: %d\n", err);
        fprintf(stderr, "MGL Error: glslang_program_SPIRV_get_messages:\n%s\n", glslang_program_SPIRV_get_messages(glsl_program));
        fprintf(stderr, "MGL Error: glslang_program_get_info_log:\n%s\n", glslang_program_get_info_log(glsl_program));
        fprintf(stderr, "MGL Error: glslang_program_get_info_debug_log:\n%s\n", glslang_program_get_info_debug_log(glsl_program));
        pptr->linked_glsl_program = NULL;
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    err = glslang_program_map_io(glsl_program);
    if (!err)
    {
        fprintf(stderr, "MGL WARNING: glslang_program_map_io failed; continuing with linked program\n");
    }

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
    {
        if (!compileStageFromLinkedProgram(ctx, pptr, glsl_program, stage)) {
            link_ok = false;
            break;
        }
    }

    if (!link_ok) {
        pptr->linked_glsl_program = NULL;
        return;
    }

    applyVertexInputLocations(pptr);
    alignFragmentInputLocationsToVertexOutputs(pptr);
    mglAssignPlainUniformLocations(pptr);
    mglUnifySamplerUniformLocations(pptr);

    /* linked_glsl_program is used as a linked-state marker only. */
    pptr->linked_glsl_program = (glslang_program_t *)pptr;
    pptr->dirty_bits |= DIRTY_PROGRAM;

    /* Only call mtlBindProgram if Metal functions are initialized */
    if (ctx->mtl_funcs.mtlBindProgram) {
        ctx->mtl_funcs.mtlBindProgram(ctx, pptr);
    } else {
        fprintf(stderr, "WARNING: Metal functions not initialized, skipping mtlBindProgram\n");
    }

    //ERROR_CHECK_RETURN(pptr->mtl_data, GL_INVALID_OPERATION);
}

void mglUseProgram(GLMContext ctx, GLuint program)
{
    Program *pptr = NULL;
    static GLuint s_last_unlinked_program = 0;
    static unsigned int s_unlinked_program_hits = 0;

    if (program)
    {
        pptr = findProgram(ctx, program);

        if (!pptr ||
            !mglObjectPointerLooksPlausible(pptr) ||
            !mglHashTableContainsData(&STATE(program_table), pptr) ||
            !mglPointerRangeIsReadable(pptr, sizeof(*pptr)))
        {
            fprintf(stderr, "MGL Error: mglUseProgram program %u not found or invalid\n", program);
            // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;

            return;
        }

        if (!pptr->linked_glsl_program)
        {
            // Compatibility fallback: some pipelines can probe/use programs before
            // link is completed/available in this backend. Skip instead of poisoning
            // global GL error state every frame.
            s_unlinked_program_hits++;
            if (s_last_unlinked_program != program || (s_unlinked_program_hits % 128u) == 1u) {
                fprintf(stderr, "MGL WARNING: mglUseProgram skipping unlinked program %u (hit=%u)\n",
                        program, s_unlinked_program_hits);
                s_last_unlinked_program = program;
            }
            return;
        }
    }
    else
    {
        pptr = NULL;
    }

    bool bindingChanged =
        ctx->state.program != pptr ||
        ctx->state.program_name != program ||
        ctx->state.var.current_program != program;

    if (bindingChanged)
    {
        Program *oldProgram = ctx->state.program;
        if (oldProgram &&
            !mglProgramPointerUsableForName(ctx,
                                            oldProgram,
                                            oldProgram->name ? oldProgram->name : ctx->state.program_name))
        {
            fprintf(stderr, "MGL WARNING: mglUseProgram dropping invalid cached program pointer %p\n",
                    (void *)oldProgram);
            oldProgram = NULL;
            ctx->state.program = NULL;
        }

        if (oldProgram)
        {
            oldProgram->refcount--;
            if (oldProgram->refcount == 0 && oldProgram->delete_status)
            {
                mglFreeProgram(ctx, oldProgram);
            }
        }

        ctx->state.program = pptr;

        if (pptr)
        {
            pptr->refcount++;
        }
        ctx->state.dirty_bits |= DIRTY_PROGRAM;
    }

    /*
     * Keep program name and pointer state in sync so renderer-side recovery can
     * re-resolve by name if the cached pointer is lost.
     */
    ctx->state.program_name = program;
    ctx->state.var.current_program = program;

    if (MGL_VERBOSE_PROGRAM_LOGS) {
        fprintf(stderr, "MGL UseProgram program=%u resolved=%p\n",
                program, (void *)ctx->state.program);
    }
}

void mglBindAttribLocation(GLMContext ctx, GLuint program, GLuint index, const GLchar *name)
{
    if (index >= MAX_ATTRIBS || !name) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!mglSetProgramAttribName(ptr, index, name)) {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }

    ptr->dirty_bits |= DIRTY_PROGRAM;
}

void mglGetActiveAttrib(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    if (length) {
        *length = 0;
    }
    if (size) {
        *size = 0;
    }
    if (type) {
        *type = 0;
    }
    if (name && bufSize > 0) {
        name[0] = '\0';
    }

    if (!ctx) {
        return;
    }
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    SpirvResource *res = mglProgramActiveAttribAt(ptr, index);
    if (!res) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (size) {
        *size = 1;
    }
    if (type) {
        *type = mglProgramActiveAttribType(res);
    }

    const char *src = res->name ? res->name : "";
    GLsizei src_len = (GLsizei)strlen(src);
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

void mglGetActiveUniform(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    if (length) {
        *length = 0;
    }
    if (size) {
        *size = 0;
    }
    if (type) {
        *type = 0;
    }
    if (name && bufSize > 0) {
        name[0] = '\0';
    }

    if (!ctx) {
        return;
    }
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (ptr->linked_glsl_program == NULL) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    int stage = -1;
    int res_type = -1;
    SpirvResource *res = mglProgramActiveUniformAt(ptr, index, &stage, &res_type);
    (void)stage;
    if (!res) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (size) {
        *size = mglProgramActiveUniformSize(res, res_type);
    }
    if (type) {
        *type = (GLenum)mglProgramActiveUniformGLType(res, res_type);
    }
    mglProgramCopyActiveUniformName(res, bufSize, length, name);
}

void mglGetAttachedShaders(GLMContext ctx, GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders)
{
    if (count) {
        *count = 0;
    }
    if (!ctx) {
        return;
    }
    if (maxCount < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Program *ptr = findProgram(ctx, program);
    if (!ptr) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    GLsizei written = 0;
    for (int i = 0; i < _MAX_SHADER_TYPES; i++) {
        Shader *shader = ptr->shader_slots[i];
        if (!shader) {
            continue;
        }
        if (shaders && written < maxCount) {
            shaders[written++] = shader->name;
        }
    }

    if (count) {
        *count = written;
    }
}

GLint  mglGetAttribLocation(GLMContext ctx, GLuint program, const GLchar *name)
{
	if (isProgram(ctx, program) == GL_FALSE)
	{
		ERROR_RETURN(GL_INVALID_OPERATION); // also may be GL_INVALID_VALUE ????

		return -1;
	}

	Program *ptr;

	ptr = getProgram(ctx, program);
	if (!ptr)
	{
		ERROR_RETURN(GL_INVALID_OPERATION);
		return -1;
	}

	if (ptr->linked_glsl_program == NULL)
	{
		ERROR_RETURN(GL_INVALID_OPERATION);

		return -1;
	}

    SpirvResourceList *vertex_inputs =
        &ptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    for (GLuint i = 0; vertex_inputs->list && i < vertex_inputs->count; i++)
    {
        const char *str = vertex_inputs->list[i].name;

        if (str && !strcmp(str, name))
        {
            return (GLint)vertex_inputs->list[i].location;
        }
    }
	
	return -1;
}

void mglGetProgramiv(GLMContext ctx, GLuint program, GLenum pname, GLint *params)
{
    Program *pptr = findProgram(ctx, program);
    ERROR_CHECK_RETURN(pptr, GL_INVALID_VALUE);
    
    switch (pname) {
        case GL_LINK_STATUS:
            *params = pptr->linked_glsl_program ? GL_TRUE : GL_FALSE;
            break;
        case GL_DELETE_STATUS:
            *params = GL_FALSE;  /* Programs are not deleted by default */
            break;
        case GL_VALIDATE_STATUS:
            *params = GL_TRUE;  /* Assume valid */
            break;
        case GL_INFO_LOG_LENGTH:
            *params = 0;  /* No info log for now */
            break;
        case GL_ATTACHED_SHADERS:
            {
                int count = 0;
                for (int i = 0; i < _MAX_SHADER_TYPES; i++) {
                    if (pptr->shader_slots[i]) count++;
                }
                *params = count;
            }
            break;
        case GL_ACTIVE_ATTRIBUTES:
            *params = mglProgramActiveAttribCount(pptr);
            break;
        case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH:
            *params = mglProgramActiveAttribMaxNameLength(pptr);
            break;
        case GL_ACTIVE_UNIFORMS:
            *params = mglProgramActiveUniformCount(pptr);
            break;
        case GL_ACTIVE_UNIFORM_MAX_LENGTH:
            *params = mglProgramActiveUniformMaxNameLength(pptr);
            break;
        case GL_ACTIVE_UNIFORM_BLOCKS:
            *params = mglActiveUniformBlockCount(pptr);
            break;
        case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH:
            *params = mglActiveUniformBlockMaxNameLength(pptr);
            break;
        case GL_COMPUTE_WORK_GROUP_SIZE:
            if (pptr->shader_slots[_COMPUTE_SHADER]) {
                /* Return local workgroup size for compute shaders */
                params[0] = pptr->local_workgroup_size.x;
                params[1] = pptr->local_workgroup_size.y;
                params[2] = pptr->local_workgroup_size.z;
            } else {
                params[0] = params[1] = params[2] = 0;
            }
            break;
        default:
            fprintf(stderr, "mglGetProgramiv: unhandled pname 0x%x\n", pname);
            *params = 0;
            break;
    }
}

void mglGetProgramInfoLog(GLMContext ctx, GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    Program *pptr = findProgram(ctx, program);
    ERROR_CHECK_RETURN(pptr, GL_INVALID_VALUE);
    
    /* For now, always return an empty info log */
    if (bufSize > 0 && infoLog) {
        infoLog[0] = '\0';
        if (length) {
            *length = 0;
        }
    }
}



#pragma mark program pipelines
void mglGenProgramPipelines(GLMContext ctx, GLsizei n, GLuint *pipelines)
{
    for (GLsizei i = 0; i < n; i++)
    {
        pipelines[i] = getNewName(&STATE(program_pipeline_table));
        getProgramPipeline(ctx, pipelines[i]);
    }
}

GLboolean mglIsProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr = findProgramPipeline(ctx, pipeline);
    return ptr ? GL_TRUE : GL_FALSE;
}

void mglDeleteProgramPipelines(GLMContext ctx, GLsizei n, const GLuint *pipelines)
{
    mglFlushPendingDraws(ctx);

    for (GLsizei i = 0; i < n; i++)
    {
        if (pipelines[i] == 0)
            continue;
            
        ProgramPipeline *ptr = findProgramPipeline(ctx, pipelines[i]);
        if (!ptr)
            continue;
            
        // If deleting currently bound pipeline, unbind it
        if (STATE(program_pipeline) && STATE(program_pipeline)->name == pipelines[i])
        {
            STATE(program_pipeline) = NULL;
            STATE(var.program_pipeline_binding) = 0;
            STATE(dirty_bits) |= DIRTY_PROGRAM;
        }
        
        // Remove from hash table and free
        deleteHashElement(&STATE(program_pipeline_table), pipelines[i]);
        free(ptr);
    }
}

void mglBindProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    if (pipeline == 0)
    {
        STATE(program_pipeline) = NULL;
        STATE(var.program_pipeline_binding) = 0;
        STATE(dirty_bits) |= DIRTY_PROGRAM;
        return;
    }
    
    ProgramPipeline *ptr = getProgramPipeline(ctx, pipeline);
    STATE(program_pipeline) = ptr;
    STATE(var.program_pipeline_binding) = ptr ? pipeline : 0;
    STATE(dirty_bits) |= DIRTY_PROGRAM;
}

void mglUseProgramStages(GLMContext ctx, GLuint pipeline, GLbitfield stages, GLuint program)
{
    ProgramPipeline *pipe_ptr = findProgramPipeline(ctx, pipeline);
    if (!pipe_ptr)
    {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    mglFlushPendingDraws(ctx);

    Program *prog_ptr = NULL;
    if (program != 0)
    {
        prog_ptr = findProgram(ctx, program);
        if (!prog_ptr)
        {
            STATE(error) = GL_INVALID_VALUE;
            return;
        }
    }
    
    // Attach program to specified stages
    if (stages & GL_VERTEX_SHADER_BIT)
        pipe_ptr->stage_programs[_VERTEX_SHADER] = prog_ptr;
    if (stages & GL_FRAGMENT_SHADER_BIT)
        pipe_ptr->stage_programs[_FRAGMENT_SHADER] = prog_ptr;
    if (stages & GL_GEOMETRY_SHADER_BIT)
        pipe_ptr->stage_programs[_GEOMETRY_SHADER] = prog_ptr;
    if (stages & GL_TESS_CONTROL_SHADER_BIT)
        pipe_ptr->stage_programs[_TESS_CONTROL_SHADER] = prog_ptr;
    if (stages & GL_TESS_EVALUATION_SHADER_BIT)
        pipe_ptr->stage_programs[_TESS_EVALUATION_SHADER] = prog_ptr;
    if (stages & GL_COMPUTE_SHADER_BIT)
        pipe_ptr->stage_programs[_COMPUTE_SHADER] = prog_ptr;
        
    pipe_ptr->validated = GL_FALSE;
    STATE(dirty_bits) |= DIRTY_PROGRAM;
}
