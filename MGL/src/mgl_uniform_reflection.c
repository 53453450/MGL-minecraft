#include "mgl_uniform_reflection.h"

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
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][_UNIFORM_BUFFER_RES];
        GLuint limit = stage == max_stage ? max_index : resources->count;
        for (GLuint index = 0; index < limit; index++) {
            SpirvResource *resource = &resources->list[index];
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

static GLuint mglProgramUniformBlockArraySize(const SpirvResource *block)
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
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][_UNIFORM_BUFFER_RES];
        for (GLuint index = 0; index < resources->count; index++) {
            SpirvResource *resource = &resources->list[index];
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
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][_ATOMIC_COUNTER_RES];
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
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][_UNIFORM_BUFFER_RES];
        for (GLuint index = 0; index < resources->count; index++) {
            SpirvResource *resource = &resources->list[index];
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

static SpirvResourceList *mglProgramActiveAttribList(Program *program)
{
    return program
        ? &program->spirv_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES]
        : NULL;
}

static GLboolean mglProgramActiveAttribHasName(const SpirvResource *resource)
{
    return resource && resource->name && resource->name[0] != '\0';
}

GLint mglProgramActiveAttribCount(Program *program)
{
    SpirvResourceList *resources = mglProgramActiveAttribList(program);
    GLint count = 0;
    if (!resources || !resources->list) {
        return 0;
    }
    for (GLuint index = 0; index < resources->count; index++) {
        count += mglProgramActiveAttribHasName(&resources->list[index]) ? 1 : 0;
    }
    return count;
}

SpirvResource *mglProgramActiveAttribAt(Program *program, GLuint index)
{
    SpirvResourceList *resources = mglProgramActiveAttribList(program);
    GLuint ordinal = 0;
    if (!resources || !resources->list) {
        return NULL;
    }

    for (GLuint resource_index = 0;
         resource_index < resources->count;
         resource_index++) {
        SpirvResource *resource = &resources->list[resource_index];
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
        SpirvResource *resource =
            mglProgramActiveAttribAt(program, (GLuint)index);
        GLint length = (GLint)(resource && resource->name
            ? strlen(resource->name) + 1u : 1u);
        if (length > max_length) {
            max_length = length;
        }
    }
    return max_length;
}

GLenum mglProgramActiveAttribType(const SpirvResource *resource)
{
    if (resource && resource->gl_type != 0u) {
        return resource->gl_type;
    }

    const char *name = resource ? resource->name : NULL;
    if (!name || !name[0]) {
        return GL_FLOAT;
    }
    if (strcmp(name, "Position") == 0 || strcmp(name, "Normal") == 0) {
        return GL_FLOAT_VEC3;
    }
    if (strcmp(name, "Color") == 0 || strstr(name, "Color")) {
        return GL_FLOAT_VEC4;
    }
    if (strcmp(name, "UV1") == 0 || strcmp(name, "UV2") == 0) {
        return GL_INT_VEC2;
    }
    if (strcmp(name, "UV") == 0 || strcmp(name, "UV0") == 0 ||
        strcmp(name, "TexCoord") == 0 || strcmp(name, "texCoord") == 0 ||
        strstr(name, "UV") || strstr(name, "TexCoord") ||
        strstr(name, "texCoord")) {
        return GL_FLOAT_VEC2;
    }
    if (strcmp(name, "LineWidth") == 0) {
        return GL_FLOAT;
    }
    if (strstr(name, "Normal")) {
        return GL_FLOAT_VEC3;
    }
    return GL_FLOAT_VEC4;
}

GLint mglSyntheticSamplerUniformLocation(int stage,
                                         int resource_type,
                                         GLuint index)
{
    return MGL_SYNTHETIC_SAMPLER_LOCATION_BASE +
           stage * 0x1000 + resource_type * 0x100 + (GLint)index;
}

static GLint mglFindExplicitUniformLocation(const char *source,
                                            const char *resource_name)
{
    char base_name[256];
    if (!source || !resource_name || !resource_name[0]) {
        return -1;
    }

    const char *bracket = strchr(resource_name, '[');
    size_t name_length = bracket
        ? (size_t)(bracket - resource_name) : strlen(resource_name);
    if (name_length == 0u || name_length >= sizeof(base_name)) {
        return -1;
    }
    memcpy(base_name, resource_name, name_length);
    base_name[name_length] = '\0';

    const char *position = source;
    while ((position = strstr(position, base_name)) != NULL) {
        const char *after_name = position + name_length;
        if ((position > source &&
             (isalnum((unsigned char)position[-1]) || position[-1] == '_')) ||
            isalnum((unsigned char)*after_name) || *after_name == '_') {
            position = after_name;
            continue;
        }

        const char *scan = position;
        const char *layout = NULL;
        while (scan > source) {
            scan--;
            if (*scan == ';' || *scan == '}') {
                break;
            }
            if (*scan == 'l' && scan + 6 <= position &&
                strncmp(scan, "layout", 6) == 0 &&
                (scan == source ||
                 (!isalnum((unsigned char)scan[-1]) && scan[-1] != '_'))) {
                layout = scan;
                break;
            }
        }
        if (!layout) {
            return -1;
        }

        const char *entry = layout + 6;
        while (isspace((unsigned char)*entry)) {
            entry++;
        }
        if (*entry++ != '(') {
            return -1;
        }
        const char *end = strchr(entry, ')');
        if (!end || end > position) {
            return -1;
        }

        while (entry < end) {
            while (entry < end &&
                   (isspace((unsigned char)*entry) || *entry == ',')) {
                entry++;
            }
            if ((size_t)(end - entry) >= 8u &&
                strncmp(entry, "location", 8) == 0) {
                const char *value = entry + 8;
                while (value < end && isspace((unsigned char)*value)) {
                    value++;
                }
                if (value < end && *value++ == '=') {
                    while (value < end && isspace((unsigned char)*value)) {
                        value++;
                    }
                    char *parsed_end = NULL;
                    unsigned long parsed = strtoul(value, &parsed_end, 10);
                    if (parsed_end != value && parsed_end <= end) {
                        return (GLint)parsed;
                    }
                }
            }
            entry++;
        }
        return -1;
    }
    return -1;
}

GLint mglSamplerUniformLocationFromReflection(GLuint reflected_location,
                                              int stage,
                                              int resource_type,
                                              GLuint index,
                                              const char *glsl_src,
                                              const char *resource_name)
{
    (void)reflected_location;
    GLint explicit_location =
        mglFindExplicitUniformLocation(glsl_src, resource_name);
    return explicit_location >= 0
        ? explicit_location
        : mglSyntheticSamplerUniformLocation(stage, resource_type, index);
}

bool mglUniformNameLooksSamplerLike(const char *name)
{
    return name && name[0] &&
           (strstr(name, "Sampler") != NULL ||
            strcmp(name, "CloudFaces") == 0);
}

static bool mglProgramResourceLooksSamplerLike(const SpirvResource *resource,
                                               int resource_type)
{
    if (!resource) {
        return false;
    }
    switch (resource_type) {
        case _SAMPLED_IMAGE_RES:
        case _SEPARATE_IMAGE_RES:
        case _SEPARATE_SAMPLERS_RES:
        case _STORAGE_IMAGE_RES:
            return true;
        case _UNIFORM_CONSTANT_RES:
            return resource->image_dim != MGL_IMAGE_DIM_NONE ||
                   resource->uniform_location >=
                       MGL_SYNTHETIC_SAMPLER_LOCATION_BASE ||
                   mglUniformNameLooksSamplerLike(resource->name);
        default:
            return false;
    }
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
            SpirvResourceList *leaders =
                &program->spirv_resources_list[leader_stage][leader_type];
            for (GLuint leader_index = 0;
                 leaders->list && leader_index < leaders->count;
                 leader_index++) {
                SpirvResource *leader = &leaders->list[leader_index];
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
                        SpirvResourceList *resources =
                            &program->spirv_resources_list[stage][resource_type];
                        for (GLuint index = 0;
                             resources->list && index < resources->count;
                             index++) {
                            SpirvResource *resource = &resources->list[index];
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
                        SpirvResourceList *resources =
                            &program->spirv_resources_list[stage][resource_type];
                        for (GLuint index = 0;
                             resources->list && index < resources->count;
                             index++) {
                            SpirvResource *resource = &resources->list[index];
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

static SpirvResource *mglFindAssignedPlainUniformResource(Program *program,
                                                          const char *name)
{
    if (!program || !name || !name[0]) {
        return NULL;
    }
    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][_UNIFORM_CONSTANT_RES];
        for (GLuint index = 0;
             resources->list && index < resources->count;
             index++) {
            SpirvResource *resource = &resources->list[index];
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

static GLint mglFirstFreePlainUniformLocation(
    const bool used[MAX_BINDABLE_BUFFERS])
{
    for (GLint location = 0; location < MAX_BINDABLE_BUFFERS; location++) {
        if (!used[location]) {
            return location;
        }
    }
    return -1;
}

void mglAssignPlainUniformLocations(Program *program)
{
    bool used[MAX_BINDABLE_BUFFERS] = {false};
    const char *used_by[MAX_BINDABLE_BUFFERS] = {NULL};
    if (!program) {
        return;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][_UNIFORM_CONSTANT_RES];
        for (GLuint index = 0;
             resources->list && index < resources->count;
             index++) {
            SpirvResource *resource = &resources->list[index];
            if (mglProgramResourceLooksSamplerLike(resource,
                                                   _UNIFORM_CONSTANT_RES)) {
                continue;
            }

            if (resource->location != 0xffffffffu &&
                resource->location < MAX_BINDABLE_BUFFERS) {
                GLint candidate = (GLint)resource->location;
                bool same_name = used_by[candidate] && resource->name &&
                    strcmp(used_by[candidate], resource->name) == 0;
                if (!used[candidate] || same_name) {
                    resource->uniform_location = candidate;
                    used[candidate] = true;
                    if (resource->name) {
                        used_by[candidate] = resource->name;
                    }
                } else {
                    resource->uniform_location = -1;
                }
            } else if (resource->location != 0xffffffffu &&
                       resource->location < 1024u) {
                resource->uniform_location = (GLint)resource->location;
            } else if (resource->uniform_location >= 0 &&
                       resource->uniform_location < MAX_BINDABLE_BUFFERS) {
                used[resource->uniform_location] = true;
                if (resource->name) {
                    used_by[resource->uniform_location] = resource->name;
                }
            }
        }
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][_UNIFORM_CONSTANT_RES];
        for (GLuint index = 0;
             resources->list && index < resources->count;
             index++) {
            SpirvResource *resource = &resources->list[index];
            if (mglProgramResourceLooksSamplerLike(resource,
                                                   _UNIFORM_CONSTANT_RES) ||
                resource->uniform_location >= 0) {
                continue;
            }

            SpirvResource *assigned =
                mglFindAssignedPlainUniformResource(program, resource->name);
            if (assigned && assigned->uniform_location >= 0 &&
                assigned->uniform_location < MAX_BINDABLE_BUFFERS) {
                resource->uniform_location = assigned->uniform_location;
                continue;
            }

            GLint preferred = -1;
            if (resource->location < MAX_BINDABLE_BUFFERS &&
                !used[resource->location]) {
                preferred = (GLint)resource->location;
            } else if (resource->gl_binding < MAX_BINDABLE_BUFFERS &&
                       !used[resource->gl_binding]) {
                preferred = (GLint)resource->gl_binding;
            } else {
                preferred = mglFirstFreePlainUniformLocation(used);
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
            used[preferred] = true;
        }
    }
}

typedef struct MGLUniformMemberLocation {
    char *name;
    GLint location;
} MGLUniformMemberLocation;

void mglAssignAggregateMemberLocations(Program *program)
{
    MGLUniformMemberLocation *assigned = NULL;
    size_t assigned_count = 0u;
    GLint next_location = 0;
    if (!program) {
        return;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][_UNIFORM_CONSTANT_RES];
        for (GLuint index = 0;
             resources->list && index < resources->count;
             index++) {
            SpirvResource *resource = &resources->list[index];
            if (!resource->ubo_members || resource->ubo_member_count == 0u) {
                continue;
            }
            resource->uniform_location = 0;
            for (GLuint member_index = 0;
                 member_index < resource->ubo_member_count;
                 member_index++) {
                SpirvUBOMember *member = &resource->ubo_members[member_index];
                const char *name = member->name ? member->name : "";
                GLint location = -1;
                for (size_t found = 0; found < assigned_count; found++) {
                    if (strcmp(assigned[found].name, name) == 0) {
                        location = assigned[found].location;
                        break;
                    }
                }
                if (location < 0) {
                    MGLUniformMemberLocation *grown = realloc(
                        assigned,
                        (assigned_count + 1u) * sizeof(*assigned));
                    if (!grown) {
                        goto cleanup;
                    }
                    assigned = grown;
                    assigned[assigned_count].name = strdup(name);
                    if (!assigned[assigned_count].name) {
                        goto cleanup;
                    }
                    GLint location_span = member->size > 1
                        ? member->size : 1;
                    location = next_location;
                    next_location += location_span;
                    assigned[assigned_count].location = location;
                    assigned_count++;
                }
                member->location_offset = location;
            }
        }
    }

cleanup:
    for (size_t index = 0; index < assigned_count; index++) {
        free(assigned[index].name);
    }
    free(assigned);
}

void mglFreeSpirvResourceOwnedFields(SpirvResource *resource)
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
