/*
 * test_mglair_gtest.cpp
 * Unit tests for the M1/M2 AIR backend via the public compile API:
 * every shader category compiles to a valid metallib container, error
 * paths report failures, and the cross-stage interface check behaves.
 * No GPU work is performed here.
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "mgl_air_reflect.h"
#include "mgl_glsl_parser.h"
#include "mgl_shader_abi.h"
}

/* The gtest target intentionally links only the AIR/reflection frontend.
 * Sampler uniform synthesis belongs to the renderer's full reflection
 * dependency tree, so keep this compile-only target self-contained. */
extern "C" GLint mglSyntheticSamplerUniformLocation(int stage, int res_type,
                                                     GLuint index) {
    (void)stage;
    (void)res_type;
    (void)index;
    return -1;
}

namespace {

MGLIRModule *semacheck(const char *src, int stage, MGLTranslationUnit **tu_out) {
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    if (!tu || tu->error) return nullptr;
    MGLIRModule *mod = (MGLIRModule *)calloc(1, sizeof(MGLIRModule));
    MGLSemaError *errs = nullptr;
    uint32_t ec = 0;
    mglGLSLSemanticCheck(tu, stage, mod, &errs, &ec);
    mglGLSLSemanticCheckDestroy(errs, ec);
    *tu_out = tu;
    return mod;
}

struct CompileResult {
    int rc = -1;
    std::vector<unsigned char> bytes;
    std::string err;
};

CompileResult compile(const char *src, int stage) {
    unsigned char *out = nullptr;
    size_t size = 0;
    char err[512] = {0};
    int rc = mglShaderCompileGLSL(src, stage, &out, &size, err, sizeof(err));
    CompileResult r;
    r.rc = rc;
    if (out) {
        r.bytes.assign(out, out + size);
        mglShaderFree(out);
    }
    r.err = err;
    return r;
}

CompileResult compileCapture(const char *src) {
    unsigned char *out = nullptr;
    size_t size = 0;
    char err[512] = {0};
    int rc = mglShaderCompileGLSLCapture(src, &out, &size, err, sizeof(err));
    CompileResult r;
    r.rc = rc;
    if (out) {
        r.bytes.assign(out, out + size);
        mglShaderFree(out);
    }
    r.err = err;
    return r;
}

const char *kVS =
    "#version 460 core\n"
    "uniform mat4 mvp;\n"
    "in vec3 inPos;\n"
    "out vec2 vUV;\n"
    "void main() {\n"
    "    gl_Position = mvp * vec4(inPos, 1.0);\n"
    "    vUV = inPos.xy * 0.5 + vec2(0.5);\n"
    "}\n";

const char *kFS =
    "#version 460 core\n"
    "in vec2 vUV;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "    fragColor = vec4(vUV, 0.5, 1.0);\n"
    "}\n";

const char *kCS =
    "#version 460 core\n"
    "layout(local_size_x = 1) in;\n"
    "uniform int uCounter;\n"
    "void main() {\n"
    "    uCounter += 1 + int(gl_GlobalInvocationID.x);\n"
    "}\n";

const char *kTCS =
    "#version 450 core\n"
    "layout(vertices = 3) out;\n"
    "void main() {\n"
    "    gl_out[gl_InvocationID].gl_Position =\n"
    "        gl_in[gl_InvocationID].gl_Position;\n"
    "    gl_TessLevelOuter[0] = 1.0;\n"
    "    gl_TessLevelOuter[1] = 1.0;\n"
    "    gl_TessLevelOuter[2] = 1.0;\n"
    "    gl_TessLevelOuter[3] = float(gl_PatchVerticesIn + gl_PrimitiveID);\n"
    "    gl_TessLevelInner[0] = 1.0;\n"
    "}\n";

const char *kTES =
    "#version 450 core\n"
    "layout(triangles, equal_spacing, cw) in;\n"
    "out vec2 vUV;\n"
    "void main() {\n"
    "    gl_Position = gl_in[0].gl_Position +\n"
    "        vec4(gl_TessCoord, float(gl_PrimitiveID));\n"
    "    vUV = gl_TessCoord.xy;\n"
    "}\n";

const char *kSSBO =
    "#version 460 core\n"
    "layout(local_size_x = 1) in;\n"
    "layout(std430) buffer B { float data[4]; } b;\n"
    "layout(std430) buffer A { int counter; } a;\n"
    "void main() {\n"
    "    b.data[0] = 1.0;\n"
    "    b.data[1] = b.data[0] * 2.0;\n"
    "    a.counter += 5;\n"
    "    atomicAdd(a.counter, 7);\n"
    "}\n";

const char *kTex =
    "#version 460 core\n"
    "layout(local_size_x = 1) in;\n"
    "uniform sampler2D tex;\n"
    "uniform int uCounter;\n"
    "void main() {\n"
    "    vec4 tc = texture(tex, vec2(0.5));\n"
    "    vec4 tl = textureLod(tex, vec2(0.25), 0.0);\n"
    "    uCounter += int(tc.r * 100.0) + int(textureSize(tex, 0).x);\n"
    "}\n";

const char *kBuiltins =
    "#version 460 core\n"
    "layout(local_size_x = 1) in;\n"
    "uniform int uCounter;\n"
    "void main() {\n"
    "    vec3 vc = cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));\n"
    "    float t2 = atan(1.0, 1.0);\n"
    "    vec2 r1 = unpackUnorm2x16(packUnorm2x16(vec2(1.0, 1.0)));\n"
    "    vec2 r2 = unpackSnorm2x16(packSnorm2x16(vec2(1.0, 1.0)));\n"
    "    vec2 r3 = unpackHalf2x16(0x3800u);\n"
    "    uCounter += int(vc.z * 100.0) + int(t2 * 100.0)\n"
    "              + int(r1.x * 100.0) + int(r2.x * 100.0)\n"
    "              + int(r3.x * 100.0);\n"
    "}\n";

const char *kVSX =
    "#version 460 core\n"
    "uniform mat4 mvp;\n"
    "in vec3 inPos;\n"
    "out vec2 vUV;\n"
    "void main() {\n"
    "    gl_Position = mvp * vec4(inPos, 1.0);\n"
    "    gl_Position.y += float(gl_VertexID);\n"
    "    vUV = inPos.xy;\n"
    "}\n";

const char *kVSCullDistance =
    "#version 460 core\n"
    "in vec3 inPos;\n"
    "void main() {\n"
    "    gl_Position = vec4(inPos, 1.0);\n"
    "    gl_CullDistance[0] = inPos.x;\n"
    "    gl_CullDistance[1] = -inPos.y;\n"
    "}\n";

const char *kBadSyntax =
    "#version 460 core\n"
    "void main() {\n"
    "    float x = ;\n"
    "}\n";

const char *kBadUndeclared =
    "#version 460 core\n"
    "void main() {\n"
    "    float x = missing_var;\n"
    "}\n";

const char *kFSMismatch =
    "#version 460 core\n"
    "in vec3 vUV;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "    fragColor = vec4(vUV, 1.0);\n"
    "}\n";

const char *kFSMissing =
    "#version 460 core\n"
    "in vec2 vMissing;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "    fragColor = vec4(vMissing, 1.0);\n"
    "}\n";

struct CompileCase {
    const char *name;
    const char *src;
    int stage;
};

const CompileCase kCompileCases[] = {
    {"vertex", kVS, MGL_STAGE_VERTEX},
    {"fragment", kFS, MGL_STAGE_FRAGMENT},
    {"compute", kCS, MGL_STAGE_COMPUTE},
    {"ssbo", kSSBO, MGL_STAGE_COMPUTE},
    {"texture", kTex, MGL_STAGE_COMPUTE},
    {"builtins", kBuiltins, MGL_STAGE_COMPUTE},
    {"cull_distance", kVSCullDistance, MGL_STAGE_VERTEX},
};

class CompileTest : public ::testing::TestWithParam<CompileCase> {};

TEST_P(CompileTest, CompilesToValidMetallib) {
    const CompileCase &c = GetParam();
    CompileResult r = compile(c.src, c.stage);
    EXPECT_EQ(0, r.rc) << c.name << ": " << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4)) << c.name;
    uint64_t fileSize = 0;
    memcpy(&fileSize, r.bytes.data() + 16, 8);
    EXPECT_EQ(r.bytes.size(), fileSize) << c.name;
}

INSTANTIATE_TEST_SUITE_P(Shaders, CompileTest,
                         ::testing::ValuesIn(kCompileCases),
                         [](const ::testing::TestParamInfo<CompileCase> &i) {
                             return std::string(i.param.name);
                         });

TEST(Metallib, XfbCaptureVariant) {
    CompileResult r = compileCapture(kVSX);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
}

TEST(Metallib, TessVertexCaptureVariant) {
    unsigned char *bytes = nullptr;
    size_t size = 0;
    char err[512] = {0};
    ASSERT_EQ(0, mglShaderCompileGLSLTessCapture(
        kVSX, &bytes, &size, err, sizeof(err))) << err;
    ASSERT_NE(nullptr, bytes);
    ASSERT_GE(size, 4u);
    EXPECT_EQ(0, memcmp(bytes, "MTLB", 4));
    mglShaderFree(bytes);
}

TEST(Metallib, TessControlAirKernel) {
    CompileResult r = compile(kTCS, MGL_STAGE_TESS_CONTROL);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
}

TEST(Metallib, CullDistanceTransfersAcrossProgrammableStages) {
    static const char *tcs =
        "#version 450 core\n"
        "layout(vertices = 3) out;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = "
        "gl_in[gl_InvocationID].gl_Position;\n"
        "  gl_out[gl_InvocationID].gl_CullDistance[0] = "
        "gl_in[gl_InvocationID].gl_CullDistance[0];\n"
        "  gl_TessLevelOuter[0] = 1.0;\n"
        "  gl_TessLevelInner[0] = 1.0;\n"
        "}\n";
    CompileResult tcsResult = compile(tcs, MGL_STAGE_TESS_CONTROL);
    EXPECT_EQ(0, tcsResult.rc) << tcsResult.err;

    static const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position;\n"
        "  gl_CullDistance[0] = gl_in[0].gl_CullDistance[0];\n"
        "}\n";
    CompileResult tesResult = compile(tes, MGL_STAGE_TESS_EVALUATION);
    EXPECT_EQ(0, tesResult.rc) << tesResult.err;

    static const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position;\n"
        "  gl_CullDistance[0] = gl_in[0].gl_CullDistance[0];\n"
        "  EmitVertex(); EndPrimitive();\n"
        "}\n";
    CompileResult gsResult = compile(gs, MGL_STAGE_GEOMETRY);
    EXPECT_EQ(0, gsResult.rc) << gsResult.err;
}

TEST(Metallib, TessControlPacksUserVaryings) {
    static const char *src =
        "#version 450 core\n"
        "layout(vertices = 3) out;\n"
        "in vec4 customIn[];\n"
        "out vec4 customOut[];\n"
        "void main() { customOut[gl_InvocationID] = customIn[gl_InvocationID]; }\n";
    CompileResult r = compile(src, MGL_STAGE_TESS_CONTROL);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
}

TEST(Metallib, TessellationPacksPatchVaryings) {
    const char *tcs =
        "#version 410 core\n"
        "layout(vertices = 3) out;\n"
        "layout(location = 2) patch out vec4 patchColor;\n"
        "void main() {\n"
        "  gl_out[gl_InvocationID].gl_Position = "
        "gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    patchColor = vec4(0.25, 0.5, 0.75, 1.0);\n"
        "    gl_TessLevelOuter[0] = 1.0;\n"
        "    gl_TessLevelOuter[1] = 1.0;\n"
        "    gl_TessLevelOuter[2] = 1.0;\n"
        "    gl_TessLevelInner[0] = 1.0;\n"
        "  }\n"
        "}\n";
    CompileResult tcsResult = compile(tcs, MGL_STAGE_TESS_CONTROL);
    ASSERT_EQ(0, tcsResult.rc) << tcsResult.err;
    ASSERT_FALSE(tcsResult.bytes.empty());
    EXPECT_EQ(0, memcmp(tcsResult.bytes.data(), "MTLB", 4));

    const char *tes =
        "#version 410 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "layout(location = 2) patch in vec4 patchColor;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position;\n"
        "  color = patchColor;\n"
        "}\n";
    CompileResult tesResult = compile(tes, MGL_STAGE_TESS_EVALUATION);
    ASSERT_EQ(0, tesResult.rc) << tesResult.err;
    ASSERT_FALSE(tesResult.bytes.empty());
    EXPECT_EQ(0, memcmp(tesResult.bytes.data(), "MTLB", 4));
}

TEST(Metallib, TessControlRejectsExplicitReturn) {
    static const char *src =
        "#version 450 core\n"
        "layout(vertices = 3) out;\n"
        "void main() { return; }\n";
    CompileResult r = compile(src, MGL_STAGE_TESS_CONTROL);
    EXPECT_NE(0, r.rc);
    EXPECT_NE(std::string::npos, r.err.find("explicit return"));
}

TEST(Metallib, CullDistanceCompilesInTessEvaluationStage) {
    static const char *src =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "void main() { gl_Position = gl_in[0].gl_Position; "
        "gl_CullDistance[0] = 1.0; }\n";
    CompileResult r = compile(src, MGL_STAGE_TESS_EVALUATION);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));

    MGLAIRStageInfo info = {};
    char err[512] = {0};
    ASSERT_EQ(0, mglAirReflectGLSLStageInfo(
        src, MGL_STAGE_TESS_EVALUATION, &info, err, sizeof(err))) << err;
    EXPECT_EQ(1u, info.uses_cull_distance);
    EXPECT_EQ(1u, info.cull_distance_count);
}

TEST(Metallib, CullDistanceStageInfo) {
    MGLAIRStageInfo info = {};
    char err[512] = {0};
    ASSERT_EQ(0, mglAirReflectGLSLStageInfo(
        kVSCullDistance, MGL_STAGE_VERTEX, &info, err, sizeof(err))) << err;
    EXPECT_EQ(1u, info.uses_cull_distance);
    EXPECT_EQ(2u, info.cull_distance_count);
}

TEST(Metallib, CullDistanceCaptureVariant) {
    unsigned char *bytes = nullptr;
    size_t size = 0;
    char err[512] = {0};
    ASSERT_EQ(0, mglShaderCompileGLSLCullDistanceCapture(
        kVSCullDistance, &bytes, &size, err, sizeof(err))) << err;
    ASSERT_NE(nullptr, bytes);
    ASSERT_GE(size, 4u);
    EXPECT_EQ(0, memcmp(bytes, "MTLB", 4));
    mglShaderFree(bytes);
}

TEST(Metallib, CullDistancePerVertexBlock) {
    static const char *src =
        "#version 460 core\n"
        "out gl_PerVertex {\n"
        "  vec4 gl_Position;\n"
        "  float gl_PointSize;\n"
        "  float gl_CullDistance[8];\n"
        "};\n"
        "in vec3 inPos;\n"
        "void main() { gl_Position = vec4(inPos, 1.0); "
        "gl_CullDistance[0] = inPos.x; }\n";
    CompileResult r = compile(src, MGL_STAGE_VERTEX);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
    MGLAIRStageInfo info = {};
    char err[512] = {0};
    ASSERT_EQ(0, mglAirReflectGLSLStageInfo(
        src, MGL_STAGE_VERTEX, &info, err, sizeof(err))) << err;
    EXPECT_EQ(1u, info.uses_cull_distance);
    EXPECT_EQ(8u, info.cull_distance_count);
}

TEST(Metallib, TessEvaluationAirVertex) {
    CompileResult r = compile(kTES, MGL_STAGE_TESS_EVALUATION);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
}

TEST(Metallib, TessEvaluationResources) {
    static const char *src =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "uniform float scale;\n"
        "layout(std140, binding = 0) uniform Params { vec4 offset; };\n"
        "layout(std430, binding = 1) buffer Data { vec4 values[]; } dataBuffer;\n"
        "uniform sampler2D tex;\n"
        "void main() {\n"
        "  vec4 sampled = texture(tex, gl_TessCoord.xy);\n"
        "  gl_Position = gl_in[0].gl_Position * scale + offset + dataBuffer.values[0] + sampled;\n"
        "}\n";
    unsigned char *bytes = nullptr;
    size_t size = 0;
    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES] = {{0}};
    char err[512] = {0};
    ASSERT_EQ(0, mglAirCompileGLSLWithReflect(
        src, MGL_STAGE_TESS_EVALUATION, nullptr, &bytes, &size, lists,
        err, sizeof(err))) << err;
    ASSERT_NE(nullptr, bytes);
    ASSERT_GE(size, 4u);
    EXPECT_EQ(0, memcmp(bytes, "MTLB", 4));

    ASSERT_EQ(1u, lists[_UNIFORM_CONSTANT_RES].count);
    EXPECT_EQ(1u, lists[_UNIFORM_CONSTANT_RES].list[0].binding);
    ASSERT_EQ(1u, lists[_STORAGE_BUFFER_RES].count);
    EXPECT_EQ(2u, lists[_STORAGE_BUFFER_RES].list[0].binding);
    ASSERT_EQ(1u, lists[_UNIFORM_BUFFER_RES].count);
    EXPECT_EQ(3u, lists[_UNIFORM_BUFFER_RES].list[0].binding);

    mglAirReflectDestroy(lists);
    mglShaderFree(bytes);
}

TEST(Metallib, RuntimeSSBOArrayLengthAcrossStages) {
    static const char *sources[] = {
        "#version 460 core\n"
        "layout(std430, binding=0) buffer Data { uint prefix; float values[]; } dataBuffer;\n"
        "void main() { gl_Position = vec4(float(dataBuffer.values.length())); }\n",
        "#version 460 core\n"
        "layout(std430, binding=0) buffer Data { uint prefix; float values[]; } dataBuffer;\n"
        "out vec4 color;\n"
        "void main() { color = vec4(float(dataBuffer.values.length())); }\n",
        "#version 460 core\n"
        "layout(local_size_x=1) in;\n"
        "layout(std430, binding=0) buffer Data { uint prefix; float values[]; } dataBuffer;\n"
        "int runtimeLength() { return dataBuffer.values.length(); }\n"
        "void main() { dataBuffer.prefix = uint(runtimeLength()); }\n",
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=1) out;\n"
        "layout(std430, binding=0) buffer Data { uint prefix; float values[]; } dataBuffer;\n"
        "void main() { dataBuffer.prefix = uint(dataBuffer.values.length()); "
        "gl_Position = gl_in[0].gl_Position; EmitVertex(); EndPrimitive(); }\n",
        "#version 460 core\n"
        "layout(isolines, equal_spacing, cw) in;\n"
        "layout(std430, binding=0) buffer Data { uint prefix; float values[]; } dataBuffer;\n"
        "void main() { dataBuffer.prefix = uint(dataBuffer.values.length()); "
        "gl_Position = gl_in[0].gl_Position; }\n",
    };
    static const int stages[] = {
        MGL_STAGE_VERTEX, MGL_STAGE_FRAGMENT, MGL_STAGE_COMPUTE,
        MGL_STAGE_GEOMETRY, MGL_STAGE_TESS_EVALUATION,
    };
    for (size_t i = 0; i < sizeof(stages) / sizeof(stages[0]); i++) {
        unsigned char *bytes = nullptr;
        size_t size = 0;
        char err[512] = {0};
        MGLAIRStageInfo info = {};
        ASSERT_EQ(0, mglAirCompileGLSLWithReflectInfo(
            sources[i], stages[i], nullptr, &bytes, &size, nullptr,
            &info, err, sizeof(err))) << "stage " << stages[i] << ": " << err;
        EXPECT_EQ(1u, info.needs_runtime_array_size_buffer);
        ASSERT_NE(nullptr, bytes);
        ASSERT_GE(size, 4u);
        EXPECT_EQ(0, memcmp(bytes, "MTLB", 4));
        mglShaderFree(bytes);
    }
}

TEST(Metallib, ComputeWorkGroupID) {
    static const char *src =
        "#version 460 core\n"
        "layout(local_size_x=1) in;\n"
        "layout(std430, binding=0) buffer Data { uint values[4]; } data;\n"
        "void main() {\n"
        "  data.values[gl_WorkGroupID.x] = gl_WorkGroupID.x;\n"
        "}\n";
    CompileResult r = compile(src, MGL_STAGE_COMPUTE);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
}

TEST(Metallib, FixedArrayLengthDoesNotNeedSizeBuffer) {
    static const char *src =
        "#version 460 core\n"
        "layout(local_size_x=1) in;\n"
        "layout(std430, binding=0) buffer Data { uint values[4]; } dataBuffer;\n"
        "void main() { dataBuffer.values[0] = uint(dataBuffer.values.length()); }\n";
    unsigned char *bytes = nullptr;
    size_t size = 0;
    char err[512] = {0};
    MGLAIRStageInfo info = {};
    ASSERT_EQ(0, mglAirCompileGLSLWithReflectInfo(
        src, MGL_STAGE_COMPUTE, nullptr, &bytes, &size, nullptr,
        &info, err, sizeof(err))) << err;
    EXPECT_EQ(0u, info.needs_runtime_array_size_buffer);
    mglShaderFree(bytes);
}

TEST(Metallib, ArrayLengthRejectsNonArrayReceiver) {
    static const char *src =
        "#version 460 core\n"
        "layout(local_size_x=1) in;\n"
        "void main() { float value = 1.0; int count = value.length(); }\n";
    CompileResult r = compile(src, MGL_STAGE_COMPUTE);
    EXPECT_NE(0, r.rc);
    EXPECT_NE(std::string::npos, r.err.find("requires an array"));
}

TEST(Metallib, TessEvaluationRejectsUnsupportedPrimitive) {
    /* isolines became a legal emulated path with P2E (TES compute
     * expansion), so it must compile; only unknown topologies fail. */
    static const char *src =
        "#version 450 core\n"
        "layout(isolines, equal_spacing, cw) in;\n"
        "void main() { gl_Position = vec4(gl_TessCoord, 1.0); }\n";
    CompileResult r = compile(src, MGL_STAGE_TESS_EVALUATION);
    EXPECT_EQ(0, r.rc) << r.err;
}

TEST(Metallib, TessEvaluationQuadAirVertex) {
    static const char *src =
        "#version 450 core\n"
        "layout(quads, fractional_even_spacing, ccw) in;\n"
        "void main() { gl_Position = gl_in[3].gl_Position + "
        "vec4(gl_TessCoord, 1.0); }\n";
    CompileResult r = compile(src, MGL_STAGE_TESS_EVALUATION);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
}

TEST(Metallib, TessEvaluationControlPointInput) {
    static const char *src =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, cw) in;\n"
        "layout(location = 1) in vec2 customIn[];\n"
        "out vec2 customOut;\n"
        "void main() { customOut = customIn[1]; "
        "gl_Position = gl_in[0].gl_Position; }\n";
    CompileResult r = compile(src, MGL_STAGE_TESS_EVALUATION);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
}

TEST(Metallib, TessellationStageInfo) {
    unsigned char *bytes = nullptr;
    size_t size = 0;
    char err[512] = {0};
    MGLAIRStageInfo info = {};

    ASSERT_EQ(0, mglAirCompileGLSLWithReflectInfo(
        kTCS, MGL_STAGE_TESS_CONTROL, nullptr, &bytes, &size, nullptr,
        &info, err, sizeof(err))) << err;
    EXPECT_EQ(3u, info.tess_control_output_vertices);
    mglShaderFree(bytes);

    bytes = nullptr;
    size = 0;
    info = {};
    info.tess_patch_vertices = 1u;
    ASSERT_EQ(0, mglAirCompileGLSLWithReflectInfo(
        kTES, MGL_STAGE_TESS_EVALUATION, nullptr, &bytes, &size, nullptr,
        &info, err, sizeof(err))) << err;
    EXPECT_EQ(1u, info.tess_patch_vertices);
    EXPECT_EQ((uint32_t)GL_TRIANGLES, info.tess_gen_mode);
    EXPECT_EQ((uint32_t)GL_EQUAL, info.tess_gen_spacing);
    EXPECT_EQ((uint32_t)GL_CW, info.tess_gen_vertex_order);
    EXPECT_EQ(0u, info.tess_gen_point_mode);
    mglShaderFree(bytes);
}

TEST(Metallib, GeometryStageInfo) {
    static const char *kGS =
        "#version 450 core\n"
        "layout(lines_adjacency) in;\n"
        "layout(line_strip, max_vertices=7) out;\n"
        "layout(invocations=3) in;\n"
        "void main() { gl_Position = gl_in[0].gl_Position; "
        "EmitVertex(); EndPrimitive(); }\n";
    MGLAIRStageInfo info = {};
    char err[512] = {0};
    ASSERT_EQ(0, mglAirReflectGLSLStageInfo(
        kGS, MGL_STAGE_GEOMETRY, &info, err, sizeof(err))) << err;
    EXPECT_EQ((uint32_t)GL_LINES_ADJACENCY, info.geometry_input_type);
    EXPECT_EQ((uint32_t)GL_LINE_STRIP, info.geometry_output_type);
    EXPECT_EQ(7u, info.geometry_vertices_out);
    EXPECT_EQ(1u, info.geometry_max_vertices_specified);
    EXPECT_EQ(3u, info.geometry_invocations);
}

TEST(Metallib, GeometryInvocationsMustBePositive) {
    static const char *kGS =
        "#version 450 core\n"
        "layout(points, invocations=0) in;\n"
        "layout(points, max_vertices=1) out;\n"
        "void main() { gl_Position = gl_in[0].gl_Position; EmitVertex(); }\n";
    CompileResult r = compile(kGS, MGL_STAGE_GEOMETRY);
    EXPECT_NE(0, r.rc);
}

TEST(Metallib, GeometryMissingMaxVerticesIsUnspecified) {
    static const char *kGS =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points) out;\n"
        "void main() { gl_Position = gl_in[0].gl_Position; EmitVertex(); }\n";
    MGLAIRStageInfo info = {};
    char err[512] = {0};
    ASSERT_EQ(0, mglAirReflectGLSLStageInfo(
        kGS, MGL_STAGE_GEOMETRY, &info, err, sizeof(err))) << err;
    EXPECT_EQ(0u, info.geometry_max_vertices_specified);
    CompileResult r = compile(kGS, MGL_STAGE_GEOMETRY);
    EXPECT_EQ(0, r.rc) << r.err;
}

TEST(Metallib, GeometryAirKernelPositionExpansion) {
    static const char *kGS =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices=6) out;\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position; gl_PointSize = gl_in[0].gl_PointSize; EmitVertex();\n"
        "  gl_Position = gl_in[1].gl_Position; gl_PointSize = gl_in[1].gl_PointSize; EmitVertex();\n"
        "  gl_Position = gl_in[2].gl_Position; gl_PointSize = gl_in[2].gl_PointSize; EmitVertex();\n"
        "  EndPrimitive();\n"
        "  gl_Position = gl_in[2].gl_Position; EmitVertex();\n"
        "  gl_Position = gl_in[1].gl_Position; EmitVertex();\n"
        "  gl_Position = gl_in[0].gl_Position; EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    CompileResult r = compile(kGS, MGL_STAGE_GEOMETRY);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
}

TEST(Metallib, GeometryAirKernelSupportsPointTopology) {
    static const char *kGS =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=1) out;\n"
        "layout(invocations=2) in;\n"
        "void main() { gl_Position = gl_in[0].gl_Position + "
        "vec4(float(gl_InvocationID)); EmitVertex(); }\n";
    CompileResult r = compile(kGS, MGL_STAGE_GEOMETRY);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
}

TEST(Metallib, GeometryAirKernelPacksUserVaryings) {
    static const char *src =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location = 2) in vec2 inputUV[];\n"
        "layout(location = 2) out vec2 outputUV;\n"
        "void main() {\n"
        "  for (int i = 0; i < 3; ++i) {\n"
        "    gl_Position = gl_in[i].gl_Position;\n"
        "    outputUV = inputUV[i];\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    CompileResult r = compile(src, MGL_STAGE_GEOMETRY);
    EXPECT_EQ(0, r.rc) << r.err;
    ASSERT_FALSE(r.bytes.empty());
    EXPECT_EQ(0, memcmp(r.bytes.data(), "MTLB", 4));
}

TEST(Metallib, RejectsSyntaxError) {
    CompileResult r = compile(kBadSyntax, MGL_STAGE_VERTEX);
    EXPECT_NE(0, r.rc);
    EXPECT_FALSE(r.err.empty());
}

TEST(Metallib, RejectsUndeclaredIdentifier) {
    CompileResult r = compile(kBadUndeclared, MGL_STAGE_VERTEX);
    EXPECT_NE(0, r.rc);
    EXPECT_FALSE(r.err.empty());
}

TEST(Metallib, RejectsUnsupportedStage) {
    CompileResult r = compile(kVS, 99);
    EXPECT_NE(0, r.rc);
    EXPECT_FALSE(r.err.empty());
}

TEST(Metallib, NullArgumentsRejected) {
    EXPECT_NE(0, mglShaderCompileGLSL(nullptr, MGL_STAGE_VERTEX, nullptr,
                                      nullptr, nullptr, 0));
}

TEST(Interface, MatchingStagesAccepted) {
    char err[512] = {0};
    EXPECT_EQ(0, mglShaderInterfaceCheck(kVS, kFS, err, sizeof(err)));
}

TEST(Interface, TypeMismatchRejected) {
    char err[512] = {0};
    EXPECT_NE(0, mglShaderInterfaceCheck(kVS, kFSMismatch, err,
                                         sizeof(err)));
    EXPECT_FALSE(err[0] == 0);
}

TEST(Interface, MissingVaryingRejected) {
    char err[512] = {0};
    EXPECT_NE(0, mglShaderInterfaceCheck(kVS, kFSMissing, err,
                                         sizeof(err)));
    EXPECT_FALSE(err[0] == 0);
}

TEST(Interface, NullSourcesRejected) {
    char err[512] = {0};
    EXPECT_NE(0, mglShaderInterfaceCheck(nullptr, kFS, err, sizeof(err)));
}

/* ---- reflection exporter ---- */

TEST(Reflect, VertexResources) {
    static const char *src =
        "#version 460 core\n"
        "layout(location = 0) in vec3 inPos;\n"
        "layout(location = 1) in vec2 inUV;\n"
        "layout(location = 0) out vec2 vUV;\n"
        "uniform mat4 mvp;\n"
        "uniform float uTime;\n"
        "void main() {\n"
        "    gl_Position = mvp * vec4(inPos, 1.0);\n"
        "    vUV = inUV;\n"
        "}\n";
    MGLTranslationUnit *tu = nullptr;
    MGLIRModule *mod = semacheck(src, MGL_STAGE_VERTEX, &tu);
    ASSERT_NE(nullptr, mod);
    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES] = {{0}};
    ASSERT_EQ(0, mglAirReflectModule(mod, MGL_STAGE_VERTEX, nullptr,
                                     lists, nullptr, 0));

    /* two vertex inputs with explicit locations */
    ASSERT_EQ(2u, lists[_STAGE_INPUT_RES].count);
    EXPECT_STREQ("inPos", lists[_STAGE_INPUT_RES].list[0].name);
    EXPECT_EQ(0u, lists[_STAGE_INPUT_RES].list[0].location);
    EXPECT_EQ(GL_FLOAT_VEC3, lists[_STAGE_INPUT_RES].list[0].gl_type);
    EXPECT_STREQ("inUV", lists[_STAGE_INPUT_RES].list[1].name);
    EXPECT_EQ(1u, lists[_STAGE_INPUT_RES].list[1].location);

    /* one varying output; plain uniforms aggregate into one struct-packed
     * resource with mvp (mat4) and uTime (float) members. */
    ASSERT_EQ(1u, lists[_STAGE_OUTPUT_RES].count);
    EXPECT_EQ(GL_FLOAT_VEC2, lists[_STAGE_OUTPUT_RES].list[0].gl_type);
    ASSERT_EQ(1u, lists[_UNIFORM_CONSTANT_RES].count);
    EXPECT_EQ(2u, lists[_UNIFORM_CONSTANT_RES].list[0].ubo_member_count);
    EXPECT_STREQ("mvp", lists[_UNIFORM_CONSTANT_RES].list[0].ubo_members[0].name);
    EXPECT_EQ(GL_FLOAT_MAT4, lists[_UNIFORM_CONSTANT_RES].list[0].ubo_members[0].gl_type);
    EXPECT_STREQ("uTime", lists[_UNIFORM_CONSTANT_RES].list[0].ubo_members[1].name);
    EXPECT_EQ(GL_FLOAT, lists[_UNIFORM_CONSTANT_RES].list[0].ubo_members[1].gl_type);
    EXPECT_GT(lists[_UNIFORM_CONSTANT_RES].list[0].required_size, 0u);

    mglAirReflectDestroy(lists);
    mglIRModuleDestroy(mod);
    mglGLSLTranslationUnitDestroy(tu);
}

TEST(Reflect, UniformBlockInstanceArray) {
    static const char *src =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "layout(binding = 3) uniform UniformBlock { int entry; } blocks[4];\n"
        "void main() { gl_Position = vec4(float(blocks[3].entry)); EmitVertex(); }\n";
    MGLTranslationUnit *tu = nullptr;
    MGLIRModule *mod = semacheck(src, MGL_STAGE_GEOMETRY, &tu);
    ASSERT_NE(nullptr, mod);
    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES] = {{0}};
    ASSERT_EQ(0, mglAirReflectModule(mod, MGL_STAGE_GEOMETRY, nullptr,
                                     lists, nullptr, 0));

    ASSERT_EQ(1u, lists[_UNIFORM_BUFFER_RES].count);
    const MGLShaderResource &block = lists[_UNIFORM_BUFFER_RES].list[0];
    EXPECT_STREQ("blocks", block.name);
    EXPECT_EQ(4u, block.ubo_array_size);
    EXPECT_TRUE(block.ubo_is_array);
    EXPECT_EQ(0u, block.binding);
    EXPECT_EQ(3u, block.gl_binding);
    ASSERT_NE(nullptr, block.ubo_array_bindings);
    for (GLuint element = 0; element < 4; element++) {
        EXPECT_EQ(3u + element, block.ubo_array_bindings[element]);
    }
    ASSERT_EQ(1u, block.ubo_member_count);
    EXPECT_STREQ("entry", block.ubo_members[0].name);
    EXPECT_EQ(GL_INT, block.ubo_members[0].gl_type);
    EXPECT_EQ(16u, block.required_size);
    EXPECT_EQ(0u, lists[_UNIFORM_CONSTANT_RES].count);

    mglAirReflectDestroy(lists);
    mglIRModuleDestroy(mod);
    mglGLSLTranslationUnitDestroy(tu);
}

TEST(Reflect, ComputeResources) {
    static const char *src =
        "#version 460 core\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 3) buffer B { float data[4]; } b;\n"
        "uniform sampler2D tex;\n"
        "uniform int uCounter;\n"
        "void main() {\n"
        "    uCounter += int(texture(tex, vec2(0.5)).r);\n"
        "    b.data[0] = 1.0;\n"
        "}\n";
    MGLTranslationUnit *tu = nullptr;
    MGLIRModule *mod = semacheck(src, MGL_STAGE_COMPUTE, &tu);
    ASSERT_NE(nullptr, mod);
    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES] = {{0}};
    ASSERT_EQ(0, mglAirReflectModule(mod, MGL_STAGE_COMPUTE, nullptr,
                                     lists, nullptr, 0));

    ASSERT_EQ(1u, lists[_STORAGE_BUFFER_RES].count);
    EXPECT_STREQ("b", lists[_STORAGE_BUFFER_RES].list[0].name);
    /* gl_binding remains the client-visible GLSL binding (3); the AIR slot
     * is tracked separately in `binding` and is allocated after the packed
     * plain-uniform buffer. */
    EXPECT_EQ(3u, lists[_STORAGE_BUFFER_RES].list[0].gl_binding);
    ASSERT_EQ(1u, lists[_STORAGE_BUFFER_RES].list[0].ubo_member_count);
    EXPECT_STREQ("data", lists[_STORAGE_BUFFER_RES].list[0].ubo_members[0].name);
    EXPECT_EQ(GL_FLOAT, lists[_STORAGE_BUFFER_RES].list[0].ubo_members[0].gl_type);
    EXPECT_EQ(4, lists[_STORAGE_BUFFER_RES].list[0].ubo_members[0].size);
    EXPECT_EQ(0u, lists[_STORAGE_BUFFER_RES].list[0].ubo_members[0].offset);
    EXPECT_EQ(16u, lists[_STORAGE_BUFFER_RES].list[0].required_size);
    EXPECT_EQ(1u, lists[_STORAGE_BUFFER_RES].list[0].binding);

    ASSERT_EQ(1u, lists[_SAMPLED_IMAGE_RES].count);
    EXPECT_STREQ("tex", lists[_SAMPLED_IMAGE_RES].list[0].name);
    EXPECT_EQ(GL_SAMPLER_2D, lists[_SAMPLED_IMAGE_RES].list[0].gl_type);
    EXPECT_TRUE(lists[_SAMPLED_IMAGE_RES].list[0].has_combined_sampler);

    ASSERT_EQ(1u, lists[_UNIFORM_CONSTANT_RES].count);
    ASSERT_EQ(1u, lists[_UNIFORM_CONSTANT_RES].list[0].ubo_member_count);
    EXPECT_EQ(GL_INT, lists[_UNIFORM_CONSTANT_RES].list[0].ubo_members[0].gl_type);

    mglAirReflectDestroy(lists);
    mglIRModuleDestroy(mod);
    mglGLSLTranslationUnitDestroy(tu);
}

}  // namespace
