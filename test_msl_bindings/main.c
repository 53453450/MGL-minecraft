/* Focused regression tests for final MSL resource-slot reconciliation. */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glcorearb.h>

#include "MGLRenderer.h"
#include "glm_context.h"
#include "hash_table.h"
#include "mgl_spirv_compile.h"

static int test_binding_attribute_parser(void)
{
    GLuint index = 0;

    if (!mglParseMSLBindingAttribute("[[buffer(127)]]", "[[buffer(", &index) ||
        index != 127u) {
        fprintf(stderr, "msl-bindings: valid maximum index was rejected\n");
        return 1;
    }
    if (mglParseMSLBindingAttribute("[[buffer(128)]]", "[[buffer(", &index)) {
        fprintf(stderr, "msl-bindings: out-of-range index was accepted\n");
        return 1;
    }
    if (mglParseMSLBindingAttribute("[[buffer()]]", "[[buffer(", &index)) {
        fprintf(stderr, "msl-bindings: empty index was accepted\n");
        return 1;
    }
    if (mglParseMSLBindingAttribute("[[buffer(7oops)]]", "[[buffer(", &index)) {
        fprintf(stderr, "msl-bindings: malformed suffix was accepted\n");
        return 1;
    }
    if (mglParseMSLBindingAttribute("[[buffer( 7)]]", "[[buffer(", &index)) {
        fprintf(stderr, "msl-bindings: non-canonical whitespace was accepted\n");
        return 1;
    }
    return 0;
}

static int test_synthetic_validator(void)
{
    static const char *valid_msl =
        "fragment float4 main0(texture2d<float> mgl_sampler_tex "
        "[[texture(3)]], sampler mgl_sampler_texSmplr [[sampler(3)]], "
        "constant float& plain_value [[buffer(5)]], "
        "sampler separate_sampler [[sampler(6)]]) { return float4(1.0); }";
    static const char *missing_combined_sampler_msl =
        "fragment float4 main0(texture2d<float> mgl_sampler_tex "
        "[[texture(3)]], constant float& plain_value [[buffer(5)]], "
        "sampler separate_sampler [[sampler(6)]]) { return float4(1.0); }";
    static const char *mismatched_combined_sampler_msl =
        "fragment float4 main0(texture2d<float> mgl_sampler_tex "
        "[[texture(3)]], sampler mgl_sampler_texSmplr [[sampler(4)]], "
        "constant float& plain_value [[buffer(5)]], "
        "sampler separate_sampler [[sampler(6)]]) { return float4(1.0); }";
    static const char *empty_msl =
        "fragment float4 main0() { return float4(1.0); }";

    Program *program = (Program *)calloc(1, sizeof(*program));
    SpirvResource *sampled_resources =
        (SpirvResource *)calloc(2, sizeof(*sampled_resources));
    SpirvResource *buffer_resources =
        (SpirvResource *)calloc(1, sizeof(*buffer_resources));
    SpirvResource *sampler_resources =
        (SpirvResource *)calloc(1, sizeof(*sampler_resources));
    int failed = 1;
    if (!program || !sampled_resources || !buffer_resources ||
        !sampler_resources) {
        free(sampler_resources);
        free(buffer_resources);
        free(sampled_resources);
        free(program);
        fprintf(stderr, "msl-bindings: synthetic validator allocation failed\n");
        return 1;
    }

    program->name = 9001u;
    program->spirv_resources_list[_FRAGMENT_SHADER]
                                 [SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].count = 2u;
    program->spirv_resources_list[_FRAGMENT_SHADER]
                                 [SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].list = sampled_resources;
    program->spirv_resources_list[_FRAGMENT_SHADER]
                                 [SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT].count = 1u;
    program->spirv_resources_list[_FRAGMENT_SHADER]
                                 [SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT].list = buffer_resources;
    program->spirv_resources_list[_FRAGMENT_SHADER]
                                 [SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS].count = 1u;
    program->spirv_resources_list[_FRAGMENT_SHADER]
                                 [SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS].list = sampler_resources;

    sampled_resources[0]._id = 1u;
    sampled_resources[0].name = "sampler";
    sampled_resources[0].msl_name = "mgl_sampler_tex";
    sampled_resources[0].msl_combined_sampler_name = "mgl_sampler_texSmplr";
    sampled_resources[0].msl_combined_sampler_binding = 3u;
    sampled_resources[0].msl_active = GL_TRUE;
    sampled_resources[0].msl_has_combined_sampler = GL_TRUE;
    sampled_resources[0].msl_binding_kind = MGL_MSL_BINDING_TEXTURE;
    sampled_resources[0].binding = 3u;

    sampled_resources[1]._id = 2u;
    sampled_resources[1].name = "inactive_sampler";
    sampled_resources[1].msl_active = GL_FALSE;
    sampled_resources[1].msl_binding_kind = MGL_MSL_BINDING_NONE;
    sampled_resources[1].binding = 9u;

    buffer_resources[0]._id = 3u;
    buffer_resources[0].name = "plain_value";
    buffer_resources[0].msl_name = "plain_value";
    buffer_resources[0].msl_active = GL_TRUE;
    buffer_resources[0].msl_binding_kind = MGL_MSL_BINDING_BUFFER;
    buffer_resources[0].binding = 5u;

    sampler_resources[0]._id = 4u;
    sampler_resources[0].name = "separate_sampler";
    sampler_resources[0].msl_name = "separate_sampler";
    sampler_resources[0].msl_active = GL_TRUE;
    sampler_resources[0].msl_binding_kind = MGL_MSL_BINDING_SAMPLER;
    sampler_resources[0].binding = 6u;

    if (!mglValidateFinalMSLResourceBindings(program,
                                              _FRAGMENT_SHADER,
                                              valid_msl)) {
        fprintf(stderr, "msl-bindings: valid active/inactive matrix failed\n");
        goto done;
    }

    sampled_resources[0].binding = 4u;
    if (mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             valid_msl)) {
        fprintf(stderr, "msl-bindings: slot mismatch was accepted\n");
        goto done;
    }
    sampled_resources[0].binding = 3u;

    sampled_resources[0].msl_name = "missing_active_name";
    if (mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             valid_msl)) {
        fprintf(stderr, "msl-bindings: missing active name was accepted\n");
        goto done;
    }
    sampled_resources[0].msl_name = "mgl_sampler_tex";

    sampled_resources[0].msl_binding_kind = MGL_MSL_BINDING_BUFFER;
    if (mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             valid_msl)) {
        fprintf(stderr, "msl-bindings: wrong attribute kind was accepted\n");
        goto done;
    }
    sampled_resources[0].msl_binding_kind = MGL_MSL_BINDING_TEXTURE;

    buffer_resources[0].binding = 7u;
    if (mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             valid_msl)) {
        fprintf(stderr, "msl-bindings: buffer slot mismatch was accepted\n");
        goto done;
    }
    buffer_resources[0].binding = 5u;

    sampler_resources[0].binding = 8u;
    if (mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             valid_msl)) {
        fprintf(stderr, "msl-bindings: sampler slot mismatch was accepted\n");
        goto done;
    }
    sampler_resources[0].binding = 6u;

    if (mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             missing_combined_sampler_msl)) {
        fprintf(stderr, "msl-bindings: missing combined sampler was accepted\n");
        goto done;
    }
    if (mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             mismatched_combined_sampler_msl)) {
        fprintf(stderr, "msl-bindings: combined sampler slot mismatch was accepted\n");
        goto done;
    }

    sampled_resources[0].msl_active = GL_FALSE;
    buffer_resources[0].msl_active = GL_FALSE;
    sampler_resources[0].msl_active = GL_FALSE;
    if (!mglValidateFinalMSLResourceBindings(program,
                                              _FRAGMENT_SHADER,
                                              empty_msl)) {
        fprintf(stderr, "msl-bindings: inactive missing resources were rejected\n");
        goto done;
    }

    failed = 0;

done:
    free(sampler_resources);
    free(buffer_resources);
    free(sampled_resources);
    free(program);
    return failed;
}

static GLuint compile_shader(GLenum type, const char *source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[2048] = {0};
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "msl-bindings: shader compile failed: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint link_sampler_program(GLint *linked_out)
{
    static const char *vertex_source =
        "#version 330 core\n"
        "layout(location=0) in vec2 position;\n"
        "void main() { gl_Position = vec4(position, 0.0, 1.0); }\n";
    static const char *fragment_source =
        "#version 330 core\n"
        "uniform sampler2D sampler;\n"
        "uniform sampler2D inactive_sampler;\n"
        "layout(std140) uniform ActiveBlock { vec4 tint; };\n"
        "layout(std140) uniform InactiveBlock { vec4 inactive_tint; };\n"
        "layout(location=0) out vec4 color;\n"
        "void main() { color = texture(sampler, vec2(0.5)) * tint; }\n";

    GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (!vertex || !fragment) {
        if (vertex) glDeleteShader(vertex);
        if (fragment) glDeleteShader(fragment);
        *linked_out = GL_FALSE;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    glGetProgramiv(program, GL_LINK_STATUS, linked_out);
    return program;
}

static GLuint link_texture_buffer_program(GLint *linked_out)
{
    static const char *vertex_source =
        "#version 330 core\n"
        "void main() { gl_Position = vec4(0.0); }\n";
    static const char *fragment_source =
        "#version 330 core\n"
        "uniform samplerBuffer buffer_sampler;\n"
        "layout(location=0) out vec4 color;\n"
        "void main() { color = texelFetch(buffer_sampler, 0); }\n";

    GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (!vertex || !fragment) {
        if (vertex) glDeleteShader(vertex);
        if (fragment) glDeleteShader(fragment);
        *linked_out = GL_FALSE;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    glGetProgramiv(program, GL_LINK_STATUS, linked_out);
    return program;
}

static GLuint link_uniform_block_array_program(GLint *linked_out)
{
    static const char *vertex_source =
        "#version 430 core\n"
        "void main() { gl_Position = vec4(0.0); }\n";
    static const char *fragment_source =
        "#version 430 core\n"
        "layout(std140, binding=2) uniform ActiveBlocks { vec4 value; } blocks[2];\n"
        "layout(location=0) out vec4 color;\n"
        "void main() { color = blocks[0].value + blocks[1].value; }\n";

    GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (!vertex || !fragment) {
        if (vertex) glDeleteShader(vertex);
        if (fragment) glDeleteShader(fragment);
        *linked_out = GL_FALSE;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    glGetProgramiv(program, GL_LINK_STATUS, linked_out);
    return program;
}

static GLuint link_storage_block_array_program(GLint *linked_out)
{
    static const char *compute_source =
        "#version 430 core\n"
        "layout(local_size_x=1) in;\n"
        "layout(std430, binding=0) buffer Data { uint value; } datas[2];\n"
        "void main() { datas[0].value = datas[1].value; }\n";

    GLuint compute = compile_shader(GL_COMPUTE_SHADER, compute_source);
    if (!compute) {
        *linked_out = GL_FALSE;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, compute);
    glLinkProgram(program);
    glDeleteShader(compute);
    glGetProgramiv(program, GL_LINK_STATUS, linked_out);
    return program;
}

static GLuint link_mixed_texture_program(GLint *linked_out)
{
    static const char *vertex_source =
        "#version 430 core\n"
        "void main() { gl_Position = vec4(0.0); }\n";
    static const char *fragment_source =
        "#version 430 core\n"
        "uniform samplerBuffer buffer_sampler;\n"
        "uniform sampler2D regular_sampler;\n"
        "layout(location=0) out vec4 color;\n"
        "void main() { color = texelFetch(buffer_sampler, 0) + "
        "texture(regular_sampler, vec2(0.5)); }\n";

    GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (!vertex || !fragment) {
        if (vertex) glDeleteShader(vertex);
        if (fragment) glDeleteShader(fragment);
        *linked_out = GL_FALSE;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    glGetProgramiv(program, GL_LINK_STATUS, linked_out);
    return program;
}

static GLuint link_sampler_shadowing_program(GLint *linked_out)
{
    static const char *vertex_source =
        "#version 430 core\n"
        "void main() { gl_Position = vec4(0.0); }\n";
    static const char *fragment_source =
        "#version 430 core\n"
        "uniform sampler2D samplerSmplr;\n"
        "layout(location=0) out vec4 color;\n"
        "vec4 fetchIt(sampler2D sampler, vec2 uv) { "
        "return texture(sampler, uv); }\n"
        "void main() { color = fetchIt(samplerSmplr, vec2(0.5)); }\n";

    GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (!vertex || !fragment) {
        if (vertex) glDeleteShader(vertex);
        if (fragment) glDeleteShader(fragment);
        *linked_out = GL_FALSE;
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    glGetProgramiv(program, GL_LINK_STATUS, linked_out);
    return program;
}

static char *replace_once(const char *source,
                          const char *from,
                          const char *to)
{
    const char *match = source && from ? strstr(source, from) : NULL;
    if (!match || !to) {
        return NULL;
    }
    size_t source_len = strlen(source);
    size_t from_len = strlen(from);
    size_t to_len = strlen(to);
    if (source_len < from_len ||
        source_len - from_len > SIZE_MAX - to_len - 1u) {
        return NULL;
    }

    size_t prefix_len = (size_t)(match - source);
    char *result = malloc(source_len - from_len + to_len + 1u);
    if (!result) {
        return NULL;
    }
    memcpy(result, source, prefix_len);
    memcpy(result + prefix_len, to, to_len);
    memcpy(result + prefix_len + to_len,
           match + from_len,
           source_len - prefix_len - from_len + 1u);
    return result;
}

static SpirvResource *find_resource(Program *program,
                                    int stage,
                                    const char *name,
                                    int *res_type_out)
{
    static const int resource_types[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS
    };

    for (size_t t = 0;
         t < sizeof(resource_types) / sizeof(resource_types[0]);
         t++) {
        int res_type = resource_types[t];
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][res_type];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (res->name && strcmp(res->name, name) == 0) {
                if (res_type_out) {
                    *res_type_out = res_type;
                }
                return res;
            }
        }
    }
    return NULL;
}

static int test_real_shader_smoke(GLMContext ctx)
{
    GLint linked = GL_FALSE;
    GLuint program_name = link_sampler_program(&linked);
    if (!program_name || linked != GL_TRUE) {
        fprintf(stderr, "msl-bindings: renamed sampler program did not link\n");
        if (program_name) glDeleteProgram(program_name);
        return 1;
    }

    Program *program = (Program *)searchHashTable(
        &ctx->active_state->program_table, program_name);
    if (!program) {
        fprintf(stderr, "msl-bindings: linked program metadata is missing\n");
        glDeleteProgram(program_name);
        return 1;
    }
    SpirvResource *active = find_resource(program,
                                          _FRAGMENT_SHADER,
                                          "sampler",
                                          NULL);
    SpirvResource *inactive = find_resource(program,
                                            _FRAGMENT_SHADER,
                                            "inactive_sampler",
                                            NULL);
    SpirvResource *active_block = find_resource(program,
                                                _FRAGMENT_SHADER,
                                                "ActiveBlock",
                                                NULL);
    SpirvResource *inactive_block = find_resource(program,
                                                  _FRAGMENT_SHADER,
                                                  "InactiveBlock",
                                                  NULL);
    if (!active || !inactive || !active_block || !inactive_block) {
        fprintf(stderr, "msl-bindings: expected reflected resources are missing\n");
        glDeleteProgram(program_name);
        return 1;
    }
    if (!active->msl_active ||
        active->msl_binding_kind != MGL_MSL_BINDING_TEXTURE ||
        !active->msl_name || strcmp(active->msl_name, "mgl_sampler_tex") != 0 ||
        !active->msl_has_combined_sampler ||
        !active->msl_combined_sampler_name ||
        strcmp(active->msl_combined_sampler_name,
               "mgl_sampler_texSmplr") != 0) {
        fprintf(stderr,
                "msl-bindings: active renamed sampler metadata is incorrect\n");
        glDeleteProgram(program_name);
        return 1;
    }
    if (inactive->msl_active || inactive->msl_name) {
        fprintf(stderr,
                "msl-bindings: optimized-out sampler was marked active\n");
        glDeleteProgram(program_name);
        return 1;
    }
    if (!active_block->msl_active ||
        active_block->msl_binding_kind != MGL_MSL_BINDING_BUFFER ||
        !active_block->msl_name ||
        inactive_block->msl_active || inactive_block->msl_name) {
        fprintf(stderr, "msl-bindings: active/inactive UBO metadata is incorrect\n");
        glDeleteProgram(program_name);
        return 1;
    }
    const char *final_msl = program->spirv[_FRAGMENT_SHADER].msl_str;
    if (!final_msl ||
        !mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             final_msl)) {
        fprintf(stderr, "msl-bindings: final sampler MSL did not reconcile\n");
        glDeleteProgram(program_name);
        return 1;
    }

    MGLMSLBindingMap binding_map;
    GLuint sampler_index = 0;
    GLuint ignored_index = 0;
    mglBuildMSLBindingMap(final_msl, &binding_map);
    if (!mglFindMSLResourceIndexInMap(&binding_map,
                                      MGL_MSL_BINDING_SAMPLER,
                                      active->msl_combined_sampler_name,
                                      &sampler_index) ||
        sampler_index != active->msl_combined_sampler_binding ||
        mglFindMSLResourceIndexInMap(&binding_map,
                                     MGL_MSL_BINDING_TEXTURE,
                                     "inactive_sampler",
                                     &ignored_index) ||
        mglFindMSLResourceIndexInMap(&binding_map,
                                     MGL_MSL_BINDING_BUFFER,
                                     "InactiveBlock",
                                     &ignored_index)) {
        fprintf(stderr, "msl-bindings: inactive resource has an MSL argument\n");
        glDeleteProgram(program_name);
        return 1;
    }

    glDeleteProgram(program_name);
    return 0;
}

static int test_texture_buffer_has_no_companion_sampler(GLMContext ctx)
{
    GLint linked = GL_FALSE;
    GLuint program_name = link_texture_buffer_program(&linked);
    if (!program_name || linked != GL_TRUE) {
        fprintf(stderr, "msl-bindings: texture-buffer program did not link\n");
        if (program_name) glDeleteProgram(program_name);
        return 1;
    }

    Program *program = (Program *)searchHashTable(
        &ctx->active_state->program_table, program_name);
    SpirvResource *resource = program
        ? find_resource(program,
                        _FRAGMENT_SHADER,
                        "buffer_sampler",
                        NULL)
        : NULL;
    const char *final_msl = program
        ? program->spirv[_FRAGMENT_SHADER].msl_str
        : NULL;
    if (!resource || !resource->msl_active ||
        resource->msl_binding_kind != MGL_MSL_BINDING_TEXTURE ||
        !resource->msl_name ||
        resource->msl_has_combined_sampler ||
        resource->msl_combined_sampler_name ||
        !final_msl ||
        !mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             final_msl)) {
        fprintf(stderr,
                "msl-bindings: texture-buffer companion metadata is incorrect\n");
        glDeleteProgram(program_name);
        return 1;
    }

    MGLMSLBindingMap binding_map;
    GLuint texture_index = 0;
    mglBuildMSLBindingMap(final_msl, &binding_map);
    if (!mglFindMSLResourceIndexInMap(&binding_map,
                                      MGL_MSL_BINDING_TEXTURE,
                                      resource->msl_name,
                                      &texture_index) ||
        texture_index != resource->binding) {
        fprintf(stderr,
                "msl-bindings: texture-buffer texture slot did not reconcile\n");
        glDeleteProgram(program_name);
        return 1;
    }

    glDeleteProgram(program_name);
    return 0;
}

static int test_uniform_block_array_slots(GLMContext ctx)
{
    GLint linked = GL_FALSE;
    GLuint program_name = link_uniform_block_array_program(&linked);
    if (!program_name || linked != GL_TRUE) {
        char log[2048] = {0};
        if (program_name) {
            glGetProgramInfoLog(program_name, sizeof(log), NULL, log);
        }
        fprintf(stderr,
                "msl-bindings: uniform-block array program did not link: %s\n",
                log);
        if (program_name) glDeleteProgram(program_name);
        return 1;
    }

    Program *program = (Program *)searchHashTable(
        &ctx->active_state->program_table, program_name);
    SpirvResource *resource = NULL;
    if (program) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[_FRAGMENT_SHADER]
                                          [SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            if (resources->list[i].msl_active &&
                resources->list[i].ubo_array_size == 2u) {
                resource = &resources->list[i];
                break;
            }
        }
    }

    const char *final_msl = program
        ? program->spirv[_FRAGMENT_SHADER].msl_str
        : NULL;
    if (!resource || !resource->msl_name ||
        resource->msl_argument_count != 2u ||
        !resource->msl_argument_names ||
        !resource->msl_argument_names[0] ||
        !resource->msl_argument_names[1] || !final_msl ||
        !mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             final_msl)) {
        fprintf(stderr,
                "msl-bindings: uniform-block array metadata did not reconcile\n");
        glDeleteProgram(program_name);
        return 1;
    }

    char from[256];
    char to[256];
    snprintf(from,
             sizeof(from),
             "%s [[buffer(%u)]]",
             resource->msl_argument_names[1],
             (unsigned)(resource->binding + 1u));
    snprintf(to,
             sizeof(to),
             "%s [[buffer(%u)]]",
             resource->msl_argument_names[1],
             (unsigned)(resource->binding + 2u));
    char *mismatched_msl = replace_once(final_msl, from, to);
    if (!mismatched_msl ||
        mglValidateFinalMSLResourceBindings(program,
                                            _FRAGMENT_SHADER,
                                            mismatched_msl)) {
        fprintf(stderr,
                "msl-bindings: UBO array element slot mismatch was accepted\n");
        free(mismatched_msl);
        glDeleteProgram(program_name);
        return 1;
    }
    free(mismatched_msl);

    glDeleteProgram(program_name);
    return 0;
}

static int test_storage_block_array_slots(GLMContext ctx)
{
    GLint linked = GL_FALSE;
    GLuint program_name = link_storage_block_array_program(&linked);
    if (!program_name || linked != GL_TRUE) {
        fprintf(stderr, "msl-bindings: storage-block array program did not link\n");
        if (program_name) glDeleteProgram(program_name);
        return 1;
    }

    Program *program = (Program *)searchHashTable(
        &ctx->active_state->program_table, program_name);
    SpirvResource *resource = program
        ? find_resource(program, _COMPUTE_SHADER, "Data", NULL)
        : NULL;
    const char *final_msl = program
        ? program->spirv[_COMPUTE_SHADER].msl_str
        : NULL;
    if (!resource || resource->gl_array_size != 2 ||
        resource->msl_argument_count != 2u ||
        !resource->msl_argument_names ||
        !resource->msl_argument_names[0] ||
        !resource->msl_argument_names[1] || !final_msl ||
        !mglValidateFinalMSLResourceBindings(program,
                                             _COMPUTE_SHADER,
                                             final_msl)) {
        fprintf(stderr,
                "msl-bindings: storage-block array metadata did not reconcile\n");
        glDeleteProgram(program_name);
        return 1;
    }

    char from[256];
    char to[256];
    snprintf(from,
             sizeof(from),
             "%s [[buffer(%u)]]",
             resource->msl_argument_names[1],
             (unsigned)(resource->binding + 1u));
    snprintf(to,
             sizeof(to),
             "%s [[buffer(%u)]]",
             resource->msl_argument_names[1],
             (unsigned)(resource->binding + 2u));
    char *mismatched_msl = replace_once(final_msl, from, to);
    if (!mismatched_msl ||
        mglValidateFinalMSLResourceBindings(program,
                                            _COMPUTE_SHADER,
                                            mismatched_msl)) {
        fprintf(stderr,
                "msl-bindings: SSBO array element slot mismatch was accepted\n");
        free(mismatched_msl);
        glDeleteProgram(program_name);
        return 1;
    }
    free(mismatched_msl);
    glDeleteProgram(program_name);
    return 0;
}

static int test_independent_combined_sampler_slot(GLMContext ctx)
{
    GLint linked = GL_FALSE;
    GLuint program_name = link_mixed_texture_program(&linked);
    if (!program_name || linked != GL_TRUE) {
        fprintf(stderr, "msl-bindings: mixed texture program did not link\n");
        if (program_name) glDeleteProgram(program_name);
        return 1;
    }

    Program *program = (Program *)searchHashTable(
        &ctx->active_state->program_table, program_name);
    SpirvResource *buffer_sampler = program
        ? find_resource(program,
                        _FRAGMENT_SHADER,
                        "buffer_sampler",
                        NULL)
        : NULL;
    SpirvResource *regular_sampler = program
        ? find_resource(program,
                        _FRAGMENT_SHADER,
                        "regular_sampler",
                        NULL)
        : NULL;
    const char *final_msl = program
        ? program->spirv[_FRAGMENT_SHADER].msl_str
        : NULL;
    if (!buffer_sampler || buffer_sampler->msl_has_combined_sampler ||
        !regular_sampler || !regular_sampler->msl_has_combined_sampler ||
        regular_sampler->msl_combined_sampler_binding ==
            regular_sampler->binding ||
        !final_msl ||
        !mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             final_msl)) {
        fprintf(stderr,
                "msl-bindings: independent combined sampler slot was lost\n");
        glDeleteProgram(program_name);
        return 1;
    }

    glDeleteProgram(program_name);
    return 0;
}

static int test_sampler_shadowing_is_scoped(GLMContext ctx)
{
    GLint linked = GL_FALSE;
    GLuint program_name = link_sampler_shadowing_program(&linked);
    if (!program_name || linked != GL_TRUE) {
        fprintf(stderr, "msl-bindings: scoped sampler-shadow program did not link\n");
        if (program_name) glDeleteProgram(program_name);
        return 1;
    }

    Program *program = (Program *)searchHashTable(
        &ctx->active_state->program_table, program_name);
    SpirvResource *resource = program
        ? find_resource(program,
                        _FRAGMENT_SHADER,
                        "samplerSmplr",
                        NULL)
        : NULL;
    const char *final_msl = program
        ? program->spirv[_FRAGMENT_SHADER].msl_str
        : NULL;
    if (!resource || !resource->msl_name ||
        strcmp(resource->msl_name, "samplerSmplr") != 0 ||
        !resource->msl_combined_sampler_name ||
        strcmp(resource->msl_combined_sampler_name,
               "samplerSmplrSmplr") != 0 ||
        !final_msl || !strstr(final_msl, "_mglTex") ||
        !mglValidateFinalMSLResourceBindings(program,
                                             _FRAGMENT_SHADER,
                                             final_msl)) {
        fprintf(stderr,
                "msl-bindings: sampler-shadow patch changed resource metadata\n");
        glDeleteProgram(program_name);
        return 1;
    }

    glDeleteProgram(program_name);
    return 0;
}

static int test_validation_failure_propagates(void)
{
    setenv("MGL_TEST_FORCE_MSL_BINDING_VALIDATION_FAILURE", "1", 1);
    GLint linked = GL_TRUE;
    GLuint program = link_sampler_program(&linked);
    unsetenv("MGL_TEST_FORCE_MSL_BINDING_VALIDATION_FAILURE");

    if (!program || linked != GL_FALSE) {
        fprintf(stderr,
                "msl-bindings: forced final validation failure did not fail link\n");
        if (program) glDeleteProgram(program);
        return 1;
    }

    glDeleteProgram(program);
    return 0;
}

int main(void)
{
    unsetenv("MGL_TEST_FORCE_MSL_BINDING_VALIDATION_FAILURE");
    if (test_binding_attribute_parser() != 0 ||
        test_synthetic_validator() != 0) {
        return 1;
    }

    GLMContext ctx = createGLMContext(
        GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
        GL_DEPTH_COMPONENT, GL_FLOAT, 0, 0);
    if (!ctx || !CppCreateMGLRendererHeadless(ctx)) {
        fprintf(stderr, "msl-bindings: failed to create headless context\n");
        return 1;
    }
    MGLsetCurrentContext(ctx);

    if (test_real_shader_smoke(ctx) != 0 ||
        test_texture_buffer_has_no_companion_sampler(ctx) != 0 ||
        test_uniform_block_array_slots(ctx) != 0 ||
        test_storage_block_array_slots(ctx) != 0 ||
        test_independent_combined_sampler_slot(ctx) != 0 ||
        test_sampler_shadowing_is_scoped(ctx) != 0 ||
        test_validation_failure_propagates() != 0) {
        return 1;
    }

    printf("MSL binding reconciliation regression: PASS\n");
    return 0;
}
