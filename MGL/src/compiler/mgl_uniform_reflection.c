/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_uniform_reflection.h"
#include "mgl_binding_policy.h"  /* mglRenderResourceLooksSamplerLike */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MGL_SYNTHETIC_SAMPLER_LOCATION_BASE 0x4000

static GLboolean mglUniformBlockNameSeen(Program *program,
                                         int max_stage,
                                         GLuint max_index,
                                         const char *name,
                                         GLuint gl_binding)
{
    for (int stage = _VERTEX_SHADER;
         stage <= max_stage && stage < _MAX_SHADER_TYPES;
         stage++) {
        MGLShaderResourceList *resources =
            &program->shader_resources_list[stage][_UNIFORM_BUFFER_RES];
        GLuint limit = stage == max_stage ? max_index : resources->count;
        for (GLuint index = 0; index < limit; index++) {
            MGLShaderResource *resource = &resources->list[index];
            if (name && name[0] != '\0') {
                if (resource->name && strcmp(name, resource->name) == 0) {
                    return GL_TRUE;
                }
            } else if ((!resource->name || resource->name[0] == '\0') &&
                       resource->gl_binding == gl_binding) {
                return GL_TRUE;
            }
        }
    }
    return GL_FALSE;
}

static GLuint mglProgramUniformBlockArraySize(const MGLShaderResource *block)
{
    return block && block->ubo_array_size > 0 ? block->ubo_array_size : 1u;
}

GLint mglActiveUniformBlockCount(Program *program)
{
    GLint total = 0;
    if (!program) {
        return 0;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        MGLShaderResourceList *resources =
            &program->shader_resources_list[stage][_UNIFORM_BUFFER_RES];
        for (GLuint index = 0; index < resources->count; index++) {
            MGLShaderResource *resource = &resources->list[index];
            if (!mglUniformBlockNameSeen(program, stage, index,
                                         resource->name,
                                         resource->gl_binding)) {
                total += (GLint)mglProgramUniformBlockArraySize(resource);
            }
        }
    }
    return total;
}

GLint mglActiveAtomicCounterBufferCount(Program *program)
{
    GLint total = 0;
    GLboolean seen[MAX_BINDABLE_BUFFERS] = {GL_FALSE};
    if (!program) {
        return 0;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        MGLShaderResourceList *resources =
            &program->shader_resources_list[stage][_ATOMIC_COUNTER_RES];
        for (GLuint index = 0; index < resources->count; index++) {
            GLuint binding = resources->list[index].gl_binding;
            if (binding < MAX_BINDABLE_BUFFERS && !seen[binding]) {
                seen[binding] = GL_TRUE;
                total++;
            }
        }
    }
    return total;
}

GLint mglActiveUniformBlockMaxNameLength(Program *program)
{
    GLint max_length = 0;
    if (!program) {
        return 0;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        MGLShaderResourceList *resources =
            &program->shader_resources_list[stage][_UNIFORM_BUFFER_RES];
        for (GLuint index = 0; index < resources->count; index++) {
            MGLShaderResource *resource = &resources->list[index];
            if (mglUniformBlockNameSeen(program, stage, index,
                                        resource->name,
                                        resource->gl_binding)) {
                continue;
            }

            GLuint element_count = mglProgramUniformBlockArraySize(resource);
            for (GLuint element = 0; element < element_count; element++) {
                GLint length = 1;
                if (resource->name) {
                    length = (GLint)strlen(resource->name) + 1;
                    if (resource->ubo_is_array || element_count > 1u) {
                        char suffix[32];
                        snprintf(suffix, sizeof(suffix), "[%u]", element);
                        length += (GLint)strlen(suffix);
                    }
                }
                if (length > max_length) {
                    max_length = length;
                }
            }
        }
    }
    return max_length;
}

static MGLShaderResourceList *mglProgramActiveAttribList(Program *program)
{
    return program
        ? &program->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES]
        : NULL;
}

static GLboolean mglProgramActiveAttribHasName(const MGLShaderResource *resource)
{
    return resource && resource->name && resource->name[0] != '\0';
}

GLint mglProgramActiveAttribCount(Program *program)
{
    MGLShaderResourceList *resources = mglProgramActiveAttribList(program);
    GLint count = 0;
    if (!resources || !resources->list) {
        return 0;
    }
    for (GLuint index = 0; index < resources->count; index++) {
        count += mglProgramActiveAttribHasName(&resources->list[index]) ? 1 : 0;
    }
    return count;
}

MGLShaderResource *mglProgramActiveAttribAt(Program *program, GLuint index)
{
    MGLShaderResourceList *resources = mglProgramActiveAttribList(program);
    GLuint ordinal = 0;
    if (!resources || !resources->list) {
        return NULL;
    }

    for (GLuint resource_index = 0;
         resource_index < resources->count;
         resource_index++) {
        MGLShaderResource *resource = &resources->list[resource_index];
        if (!mglProgramActiveAttribHasName(resource)) {
            continue;
        }
        if (ordinal++ == index) {
            return resource;
        }
    }
    return NULL;
}

GLint mglProgramActiveAttribMaxNameLength(Program *program)
{
    GLint max_length = 0;
    GLint count = mglProgramActiveAttribCount(program);
    for (GLint index = 0; index < count; index++) {
        MGLShaderResource *resource =
            mglProgramActiveAttribAt(program, (GLuint)index);
        GLint length = (GLint)(resource && resource->name
            ? strlen(resource->name) + 1u : 1u);
        if (length > max_length) {
            max_length = length;
        }
    }
    return max_length;
}

GLenum mglProgramActiveAttribType(const MGLShaderResource *resource)
{
    /* AIR reflection (push_resource -> mglAirGLTypeFromIR) always sets a
     * non-zero gl_type for every resource.  The SPIRV-era name heuristic
     * that guessed the type from "Position"/"Color"/"UV"/"Normal" is gone:
     * the IR knows the exact type. */
    if (resource && resource->gl_type != 0u) {
        return resource->gl_type;
    }
    return GL_FLOAT;
}

GLint mglSyntheticSamplerUniformLocation(int stage,
                                         int resource_type,
                                         GLuint index)
{
    return MGL_SYNTHETIC_SAMPLER_LOCATION_BASE +
           stage * 0x1000 + resource_type * 0x100 + (GLint)index;
}

/* Sampler-like classification goes through the shared policy predicate; the
 * SPIRV-era synthetic-location and name heuristics are gone (see
 * mglRenderResourceLooksSamplerLike). */
static bool mglProgramResourceLooksSamplerLike(const MGLShaderResource *resource,
                                               int resource_type)
{
    if (!resource) {
        return false;
    }
    return mglRenderResourceLooksSamplerLike((uint32_t)resource_type,
                                             (uint32_t)resource->image_dim) != 0;
}

static bool mglSamplerResourceNamesMatch(const char *left,
                                         const char *right)
{
    if (!left || !right) {
        return false;
    }
    if (strcmp(left, right) == 0) {
        return true;
    }

    size_t left_length = strlen(left);
    size_t right_length = strlen(right);
    if (left_length >= 3u && strcmp(left + left_length - 3u, "[0]") == 0) {
        left_length -= 3u;
    }
    if (right_length >= 3u && strcmp(right + right_length - 3u, "[0]") == 0) {
        right_length -= 3u;
    }
    return left_length == right_length &&
           strncmp(left, right, left_length) == 0;
}

static GLint mglPlainUniformLocationCount(const MGLShaderResource *resource);
static int mglPlainUniformSpanAvailable(
    const bool used[MAX_PLAIN_UNIFORM_LOCATIONS],
    const char *used_by[MAX_PLAIN_UNIFORM_LOCATIONS],
    GLint base, GLint span, const char *name);
static void mglPlainUniformMarkSpan(
    bool used[MAX_PLAIN_UNIFORM_LOCATIONS],
    const char *used_by[MAX_PLAIN_UNIFORM_LOCATIONS],
    GLint base, GLint span, const char *name);
static GLint mglFirstFreePlainUniformSpan(
    const bool used[MAX_PLAIN_UNIFORM_LOCATIONS], GLint span);

void mglAssignSamplerUniformLocations(Program *program)
{
    static const int resource_types[] = {
        _UNIFORM_CONSTANT_RES,
        _SAMPLED_IMAGE_RES,
        _SEPARATE_IMAGE_RES,
        _SEPARATE_SAMPLERS_RES,
        _STORAGE_IMAGE_RES
    };
    bool used[MAX_PLAIN_UNIFORM_LOCATIONS] = {false};
    if (!program) {
        return;
    }

    /* Reserve locations already claimed by non-sampler plain uniforms so
     * sampler/image locations share the same GL location space (GL 4.6
     * §7.6.1).  Reflection may leave synthetic ids (>= 0x4000) on opaque
     * uniforms; those are rewritten below into free low slots. */
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        MGLShaderResourceList *resources =
            &program->shader_resources_list[stage][_UNIFORM_CONSTANT_RES];
        for (GLuint index = 0;
             resources->list && index < resources->count;
             index++) {
            MGLShaderResource *resource = &resources->list[index];
            if (mglProgramResourceLooksSamplerLike(resource,
                                                   _UNIFORM_CONSTANT_RES)) {
                continue;
            }
            if (resource->ubo_members && resource->ubo_member_count > 0u) {
                for (GLuint m = 0u; m < resource->ubo_member_count; m++) {
                    const SpirvUBOMember *member = &resource->ubo_members[m];
                    GLint base = member->location_offset;
                    GLint span = mglUniformTypeLocationSpan(member->gl_type,
                                                            member->size);
                    if (base >= 0 &&
                        base + span <= (GLint)MAX_PLAIN_UNIFORM_LOCATIONS) {
                        mglPlainUniformMarkSpan(used, NULL, base, span, NULL);
                    }
                }
                continue;
            }
            GLint base = resource->uniform_location;
            GLint span = mglPlainUniformLocationCount(resource);
            if (base >= 0 &&
                base + span <= (GLint)MAX_PLAIN_UNIFORM_LOCATIONS) {
                mglPlainUniformMarkSpan(used, NULL, base, span, NULL);
            }
        }
    }

    /* Pass 0: keep explicit / already-dense sampler locations and mark them. */
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (size_t ti = 0;
             ti < sizeof(resource_types) / sizeof(resource_types[0]);
             ti++) {
            int resource_type = resource_types[ti];
            MGLShaderResourceList *resources =
                &program->shader_resources_list[stage][resource_type];
            for (GLuint index = 0;
                 resources->list && index < resources->count;
                 index++) {
                MGLShaderResource *resource = &resources->list[index];
                if (!mglProgramResourceLooksSamplerLike(resource,
                                                       resource_type) ||
                    !resource->name || resource->uniform_location < 0) {
                    continue;
                }
                GLint base = resource->uniform_location;
                GLint span = resource->gl_array_size > 1
                                 ? resource->gl_array_size
                                 : 1;
                if (base >= MGL_SYNTHETIC_SAMPLER_LOCATION_BASE ||
                    base + span > (GLint)MAX_PLAIN_UNIFORM_LOCATIONS) {
                    continue;
                }
                if (!mglPlainUniformSpanAvailable(used, NULL, base, span,
                                                  NULL)) {
                    /* Collision with a plain uniform or earlier sampler:
                     * force reassignment in pass 1. */
                    resource->uniform_location =
                        MGL_SYNTHETIC_SAMPLER_LOCATION_BASE;
                    continue;
                }
                mglPlainUniformMarkSpan(used, NULL, base, span, NULL);
            }
        }
    }

    /* Pass 1: rewrite synthetic / colliding locations into free low slots,
     * sharing one location across stages for the same sampler name. */
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        for (size_t ti = 0;
             ti < sizeof(resource_types) / sizeof(resource_types[0]);
             ti++) {
            int resource_type = resource_types[ti];
            MGLShaderResourceList *resources =
                &program->shader_resources_list[stage][resource_type];
            for (GLuint index = 0;
                 resources->list && index < resources->count;
                 index++) {
                MGLShaderResource *resource = &resources->list[index];
                if (!mglProgramResourceLooksSamplerLike(resource,
                                                       resource_type) ||
                    !resource->name) {
                    continue;
                }
                GLint span = resource->gl_array_size > 1
                                 ? resource->gl_array_size
                                 : 1;
                if (resource->uniform_location >= 0 &&
                    resource->uniform_location <
                        MGL_SYNTHETIC_SAMPLER_LOCATION_BASE &&
                    resource->uniform_location + span <=
                        (GLint)MAX_PLAIN_UNIFORM_LOCATIONS) {
                    continue;
                }

                GLint shared = -1;
                for (int s2 = _VERTEX_SHADER; s2 < _MAX_SHADER_TYPES; s2++) {
                    for (size_t tj = 0;
                         tj < sizeof(resource_types) / sizeof(resource_types[0]);
                         tj++) {
                        int rt2 = resource_types[tj];
                        MGLShaderResourceList *others =
                            &program->shader_resources_list[s2][rt2];
                        for (GLuint j = 0;
                             others->list && j < others->count;
                             j++) {
                            MGLShaderResource *other = &others->list[j];
                            if (other == resource ||
                                !mglProgramResourceLooksSamplerLike(other,
                                                                    rt2) ||
                                !other->name ||
                                !mglSamplerResourceNamesMatch(other->name,
                                                              resource->name)) {
                                continue;
                            }
                            if (other->uniform_location >= 0 &&
                                other->uniform_location <
                                    MGL_SYNTHETIC_SAMPLER_LOCATION_BASE &&
                                other->uniform_location + span <=
                                    (GLint)MAX_PLAIN_UNIFORM_LOCATIONS) {
                                shared = other->uniform_location;
                                break;
                            }
                        }
                        if (shared >= 0) {
                            break;
                        }
                    }
                    if (shared >= 0) {
                        break;
                    }
                }

                GLint assigned = shared;
                if (assigned < 0) {
                    assigned = mglFirstFreePlainUniformSpan(used, span);
                }
                if (assigned < 0) {
                    fprintf(stderr,
                            "MGL WARNING: no sampler uniform location left "
                            "program=%u name=%s stage=%d\n",
                            program->name,
                            resource->name ? resource->name : "(null)",
                            stage);
                    continue;
                }
                resource->uniform_location = assigned;
                if (shared < 0) {
                    mglPlainUniformMarkSpan(used, NULL, assigned, span, NULL);
                }
            }
        }
    }
}

void mglUnifySamplerUniformLocations(Program *program)
{
    static const int resource_types[] = {
        _UNIFORM_CONSTANT_RES,
        _SAMPLED_IMAGE_RES,
        _SEPARATE_IMAGE_RES,
        _SEPARATE_SAMPLERS_RES,
        _STORAGE_IMAGE_RES
    };
    if (!program) {
        return;
    }

    for (int leader_stage = _VERTEX_SHADER;
         leader_stage < _MAX_SHADER_TYPES;
         leader_stage++) {
        for (size_t leader_type_index = 0;
             leader_type_index < sizeof(resource_types) / sizeof(resource_types[0]);
             leader_type_index++) {
            int leader_type = resource_types[leader_type_index];
            MGLShaderResourceList *leaders =
                &program->shader_resources_list[leader_stage][leader_type];
            for (GLuint leader_index = 0;
                 leaders->list && leader_index < leaders->count;
                 leader_index++) {
                MGLShaderResource *leader = &leaders->list[leader_index];
                if (!mglProgramResourceLooksSamplerLike(leader, leader_type) ||
                    !leader->name || leader->uniform_location < 0) {
                    continue;
                }

                GLint sampler_unit = leader->sampler_unit;
                for (int stage = _VERTEX_SHADER;
                     stage < _MAX_SHADER_TYPES;
                     stage++) {
                    for (size_t type_index = 0;
                         type_index < sizeof(resource_types) / sizeof(resource_types[0]);
                         type_index++) {
                        int resource_type = resource_types[type_index];
                        MGLShaderResourceList *resources =
                            &program->shader_resources_list[stage][resource_type];
                        for (GLuint index = 0;
                             resources->list && index < resources->count;
                             index++) {
                            MGLShaderResource *resource = &resources->list[index];
                            if (mglProgramResourceLooksSamplerLike(resource,
                                                                   resource_type) &&
                                resource->name &&
                                mglSamplerResourceNamesMatch(resource->name,
                                                             leader->name) &&
                                resource->sampler_unit > sampler_unit) {
                                sampler_unit = resource->sampler_unit;
                            }
                        }
                    }
                }

                leader->sampler_unit = sampler_unit;
                for (int stage = _VERTEX_SHADER;
                     stage < _MAX_SHADER_TYPES;
                     stage++) {
                    for (size_t type_index = 0;
                         type_index < sizeof(resource_types) / sizeof(resource_types[0]);
                         type_index++) {
                        int resource_type = resource_types[type_index];
                        MGLShaderResourceList *resources =
                            &program->shader_resources_list[stage][resource_type];
                        for (GLuint index = 0;
                             resources->list && index < resources->count;
                             index++) {
                            MGLShaderResource *resource = &resources->list[index];
                            if (resource == leader ||
                                !mglProgramResourceLooksSamplerLike(resource,
                                                                    resource_type) ||
                                !resource->name ||
                                !mglSamplerResourceNamesMatch(resource->name,
                                                              leader->name)) {
                                continue;
                            }
                            resource->uniform_location = leader->uniform_location;
                            resource->sampler_unit = sampler_unit;
                        }
                    }
                }
            }
        }
    }
}

static MGLShaderResource *mglFindAssignedPlainUniformResource(Program *program,
                                                          const char *name)
{
    if (!program || !name || !name[0]) {
        return NULL;
    }
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        MGLShaderResourceList *resources =
            &program->shader_resources_list[stage][_UNIFORM_CONSTANT_RES];
        for (GLuint index = 0;
             resources->list && index < resources->count;
             index++) {
            MGLShaderResource *resource = &resources->list[index];
            if (resource->uniform_location >= 0 && resource->name &&
                !mglProgramResourceLooksSamplerLike(resource,
                                                    _UNIFORM_CONSTANT_RES) &&
                strcmp(resource->name, name) == 0) {
                return resource;
            }
        }
    }
    return NULL;
}

/* GL 4.6 §7.6.1: an array of basic types occupies sequential locations,
 * one per element. Struct leaves are assigned separately. */
static GLint mglPlainUniformLocationCount(const MGLShaderResource *resource)
{
    if (!resource) {
        return 1;
    }
    if (resource->ubo_members && resource->ubo_member_count > 0u) {
        return 1;
    }
    if (resource->gl_array_size > 1) {
        return resource->gl_array_size;
    }
    return 1;
}

static int mglPlainUniformSpanAvailable(
    const bool used[MAX_PLAIN_UNIFORM_LOCATIONS],
    const char *used_by[MAX_PLAIN_UNIFORM_LOCATIONS],
    GLint base, GLint span, const char *name)
{
    if (base < 0 || span < 1 ||
        (GLint)MAX_PLAIN_UNIFORM_LOCATIONS - span < base) {
        return 0;
    }
    for (GLint i = 0; i < span; i++) {
        if (!used[base + i]) {
            continue;
        }
        if (!name || !used_by[base + i] ||
            strcmp(used_by[base + i], name) != 0) {
            return 0;
        }
    }
    return 1;
}

static void mglPlainUniformMarkSpan(
    bool used[MAX_PLAIN_UNIFORM_LOCATIONS],
    const char *used_by[MAX_PLAIN_UNIFORM_LOCATIONS],
    GLint base, GLint span, const char *name)
{
    for (GLint i = 0; i < span; i++) {
        used[base + i] = true;
        if (name) {
            used_by[base + i] = name;
        }
    }
}

static GLint mglFirstFreePlainUniformSpan(
    const bool used[MAX_PLAIN_UNIFORM_LOCATIONS], GLint span)
{
    if (span < 1) {
        span = 1;
    }
    for (GLint location = 0;
         location <= (GLint)MAX_PLAIN_UNIFORM_LOCATIONS - span;
         location++) {
        int ok = 1;
        for (GLint i = 0; i < span; i++) {
            if (used[location + i]) {
                ok = 0;
                break;
            }
        }
        if (ok) {
            return location;
        }
    }
    return -1;
}

void mglAssignPlainUniformLocations(Program *program)
{
    bool used[MAX_PLAIN_UNIFORM_LOCATIONS] = {false};
    const char *used_by[MAX_PLAIN_UNIFORM_LOCATIONS] = {NULL};
    if (!program) {
        return;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        MGLShaderResourceList *resources =
            &program->shader_resources_list[stage][_UNIFORM_CONSTANT_RES];
        for (GLuint index = 0;
             resources->list && index < resources->count;
             index++) {
            MGLShaderResource *resource = &resources->list[index];
            if (mglProgramResourceLooksSamplerLike(resource,
                                                   _UNIFORM_CONSTANT_RES)) {
                continue;
            }
            GLint span = mglPlainUniformLocationCount(resource);

            if (resource->location != 0xffffffffu &&
                resource->location < MAX_PLAIN_UNIFORM_LOCATIONS) {
                GLint candidate = (GLint)resource->location;
                if (mglPlainUniformSpanAvailable(used, used_by, candidate,
                                                 span, resource->name)) {
                    resource->uniform_location = candidate;
                    mglPlainUniformMarkSpan(used, used_by, candidate, span,
                                            resource->name);
                } else {
                    resource->uniform_location = -1;
                }
            } else if (resource->uniform_location >= 0 &&
                       resource->uniform_location < MAX_PLAIN_UNIFORM_LOCATIONS) {
                mglPlainUniformMarkSpan(used, used_by,
                                        resource->uniform_location, span,
                                        resource->name);
            }
        }
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        MGLShaderResourceList *resources =
            &program->shader_resources_list[stage][_UNIFORM_CONSTANT_RES];
        for (GLuint index = 0;
             resources->list && index < resources->count;
             index++) {
            MGLShaderResource *resource = &resources->list[index];
            if (mglProgramResourceLooksSamplerLike(resource,
                                                   _UNIFORM_CONSTANT_RES) ||
                resource->uniform_location >= 0) {
                continue;
            }

            MGLShaderResource *assigned =
                mglFindAssignedPlainUniformResource(program, resource->name);
            if (assigned && assigned->uniform_location >= 0 &&
                assigned->uniform_location < MAX_PLAIN_UNIFORM_LOCATIONS) {
                resource->uniform_location = assigned->uniform_location;
                continue;
            }

            GLint span = mglPlainUniformLocationCount(resource);
            GLint preferred = -1;
            if (resource->location < MAX_PLAIN_UNIFORM_LOCATIONS &&
                mglPlainUniformSpanAvailable(used, used_by,
                                             (GLint)resource->location, span,
                                             NULL)) {
                preferred = (GLint)resource->location;
            } else if (resource->gl_binding < MAX_PLAIN_UNIFORM_LOCATIONS &&
                       mglPlainUniformSpanAvailable(used, used_by,
                                                    (GLint)resource->gl_binding,
                                                    span, NULL)) {
                preferred = (GLint)resource->gl_binding;
            } else {
                preferred = mglFirstFreePlainUniformSpan(used, span);
            }

            if (preferred < 0) {
                fprintf(stderr,
                        "MGL WARNING: no plain uniform location left "
                        "program=%u name=%s stage=%d\n",
                        program->name,
                        resource->name ? resource->name : "(null)",
                        stage);
                continue;
            }
            resource->uniform_location = preferred;
            mglPlainUniformMarkSpan(used, used_by, preferred, span,
                                    resource->name);
        }
    }
}

typedef struct MGLUniformMemberLocation {
    char *name;
    GLint location;
} MGLUniformMemberLocation;

int mglAssignAggregateMemberLocations(Program *program)
{
    MGLUniformMemberLocation *assigned = NULL;
    size_t assigned_count = 0u;
    bool used[MAX_PLAIN_UNIFORM_LOCATIONS] = {false};
    const char *used_by[MAX_PLAIN_UNIFORM_LOCATIONS] = {NULL};
    int fail = 0;
    if (!program) {
        return 0;
    }

    /* GL 4.6 §4.4.3: explicit layout(location) first, then unused slots
     * for implicit uniforms. Same name across stages shares one location. */
    for (int pass = 0; pass < 2 && !fail; pass++) {
        for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
            MGLShaderResourceList *resources =
                &program->shader_resources_list[stage][_UNIFORM_CONSTANT_RES];
            for (GLuint index = 0;
                 resources->list && index < resources->count;
                 index++) {
                MGLShaderResource *resource = &resources->list[index];
                if (!resource->ubo_members || resource->ubo_member_count == 0u) {
                    continue;
                }
                resource->uniform_location = 0;
                for (GLuint member_index = 0;
                     member_index < resource->ubo_member_count;
                     member_index++) {
                    SpirvUBOMember *member = &resource->ubo_members[member_index];
                    const char *name = member->name ? member->name : "";
                    int has_explicit = member->explicit_location >= 0;
                    if (pass == 0 && !has_explicit) {
                        continue;
                    }
                    if (pass == 1 && has_explicit) {
                        continue;
                    }
                    GLint location = -1;
                    for (size_t found = 0; found < assigned_count; found++) {
                        if (strcmp(assigned[found].name, name) == 0) {
                            location = assigned[found].location;
                            break;
                        }
                    }
                    GLint location_span = mglUniformTypeLocationSpan(
                        member->gl_type, member->size);
                    if (location >= 0) {
                        if (has_explicit &&
                            location != member->explicit_location) {
                            fail = 1;
                            break;
                        }
                        member->location_offset = location;
                        continue;
                    }
                    if (has_explicit) {
                        location = member->explicit_location;
                        if (!mglPlainUniformSpanAvailable(
                                used, used_by, location, location_span,
                                name)) {
                            fail = 1;
                            break;
                        }
                    } else {
                        location = mglFirstFreePlainUniformSpan(
                            used, location_span);
                    }
                    if (location < 0) {
                        fail = 1;
                        break;
                    }
                    MGLUniformMemberLocation *grown = realloc(
                        assigned,
                        (assigned_count + 1u) * sizeof(*assigned));
                    if (!grown) {
                        fail = 1;
                        goto cleanup;
                    }
                    assigned = grown;
                    assigned[assigned_count].name = strdup(name);
                    if (!assigned[assigned_count].name) {
                        fail = 1;
                        goto cleanup;
                    }
                    assigned[assigned_count].location = location;
                    assigned_count++;
                    mglPlainUniformMarkSpan(used, used_by, location,
                                            location_span, name);
                    member->location_offset = location;
                }
                if (fail) {
                    break;
                }
            }
            if (fail) {
                break;
            }
        }
    }

cleanup:
    for (size_t index = 0; index < assigned_count; index++) {
        free(assigned[index].name);
    }
    free(assigned);
    return fail ? -1 : 0;
}

void mglFreeMGLShaderResourceOwnedFields(MGLShaderResource *resource)
{
    if (!resource) {
        return;
    }

    free((void *)resource->name);
    resource->name = NULL;
    if (resource->ubo_members) {
        for (GLuint index = 0;
             index < resource->ubo_member_count;
             index++) {
            free((void *)resource->ubo_members[index].name);
            free(resource->ubo_members[index].query_name);
        }
        free(resource->ubo_members);
        resource->ubo_members = NULL;
    }
    resource->ubo_member_count = 0u;
    resource->ubo_member = NULL;
    free(resource->ubo_array_bindings);
    resource->ubo_array_bindings = NULL;
    free(resource->ubo_instance_name);
    resource->ubo_instance_name = NULL;
}
