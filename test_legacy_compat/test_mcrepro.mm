/*
 * test_mcrepro.mm
 * Repro: MC 1.21.11-style shaders (anonymous std140 UBO blocks + samplers)
 * through the AIR backend.  Prints the reflection's Metal slot assignment
 * next to the air.location_index values the emitted IR actually declares,
 * so a mismatch between the two is visible directly.
 */

#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_shader_abi.h"
#include "mgl_types_program.h"

/* This standalone reflection diagnostic intentionally avoids the product
 * uniform-reflection dependency tree. Sampler locations are not part of the
 * slot dump, so use the same self-contained stub as the AIR smoke tests. */
extern "C" GLint mglSyntheticSamplerUniformLocation(int stage, int res_type,
                                                     GLuint index) {
    (void)stage;
    (void)res_type;
    (void)index;
    return -1;
}

static const char *kVS =
    "#version 150 core\n"
    "in vec3 Position;\n"
    "in vec4 Color;\n"
    "in vec2 UV0;\n"
    "in ivec2 UV2;\n"
    "layout(std140) uniform DynamicTransforms {\n"
    "    mat4 ModelViewMat;\n"
    "    vec4 ColorModulator;\n"
    "    vec3 ModelOffset;\n"
    "    mat4 TextureMat;\n"
    "    float LineWidth;\n"
    "};\n"
    "layout(std140) uniform Projection {\n"
    "    mat4 ProjMat;\n"
    "};\n"
    "uniform sampler2D Sampler2;\n"
    "out vec4 vertexColor;\n"
    "out vec2 texCoord0;\n"
    "void main() {\n"
    "    gl_Position = ProjMat * ModelViewMat * vec4(Position + ModelOffset, 1.0);\n"
    "    vertexColor = Color * texelFetch(Sampler2, UV2 / 16, 0);\n"
    "    texCoord0 = UV0;\n"
    "}\n";

static const char *kFS =
    "#version 150 core\n"
    "uniform sampler2D Sampler0;\n"
    "layout(std140) uniform DynamicTransforms {\n"
    "    mat4 ModelViewMat;\n"
    "    vec4 ColorModulator;\n"
    "    vec3 ModelOffset;\n"
    "    mat4 TextureMat;\n"
    "    float LineWidth;\n"
    "};\n"
    "layout(std140) uniform Fog {\n"
    "    vec4 FogColor;\n"
    "    float FogEnvironmentalStart;\n"
    "    float FogEnvironmentalEnd;\n"
    "    float FogRenderDistanceStart;\n"
    "    float FogRenderDistanceEnd;\n"
    "    float FogSkyEnd;\n"
    "    float FogCloudsEnd;\n"
    "};\n"
    "in vec4 vertexColor;\n"
    "in vec2 texCoord0;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "    vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;\n"
    "    fragColor = vec4(color.rgb * FogColor.rgb, color.a);\n"
    "}\n";

static const char *kResName[_MAX_SPIRV_RES] = {
    "UNKNOWN", "UNIFORM_BUFFER", "UNIFORM_CONSTANT", "STORAGE_BUFFER",
    "STAGE_INPUT", "STAGE_OUTPUT", "SUBPASS_INPUT", "STORAGE_IMAGE",
    "SAMPLED_IMAGE", "ATOMIC_COUNTER", "PUSH_CONSTANT", "SEPARATE_IMAGE",
    "SEPARATE_SAMPLERS", "ACCEL_STRUCT", "RAY_QUERY",
};

static int dumpStage(const char *label, const char *src, int stage)
{
    unsigned char *lib = NULL;
    size_t libSize = 0;
    SpirvResourceList lists[_MAX_SPIRV_RES];
    memset(lists, 0, sizeof(lists));
    char err[1024] = {0};

    printf("\n================ %s ================\n", label);
    int rc = mglAirCompileGLSLWithReflect(src, stage, NULL, &lib, &libSize,
                                          lists, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "COMPILE FAILED: %s\n", err);
        return 1;
    }
    printf("metallib: %zu bytes\n", libSize);

    for (int t = 0; t < _MAX_SPIRV_RES; t++) {
        for (GLuint i = 0; i < lists[t].count; i++) {
            SpirvResource *r = &lists[t].list[i];
            printf("  [%-16s] name=%-20s binding=%-3u location=%-11u uniform_location=%d\n",
                   kResName[t], r->name ? r->name : "?", r->binding,
                   r->location, r->uniform_location);
        }
    }
    mglShaderFree(lib);
    return 0;
}

int main(void)
{
    /* MGL_DUMP_IR makes the backend print the module to stderr; run with
     * MGL_DUMP_IR=1 to diff air.location_index against the table above. */
    int failed = 0;
    failed |= dumpStage("VERTEX", kVS, MGL_STAGE_VERTEX);
    failed |= dumpStage("FRAGMENT", kFS, MGL_STAGE_FRAGMENT);
    return failed ? 1 : 0;
}
