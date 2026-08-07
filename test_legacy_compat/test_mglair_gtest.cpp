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

namespace {

MGLIRModule *semacheck(const char *src, MGLTranslationUnit **tu_out) {
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    if (!tu || tu->error) return nullptr;
    MGLIRModule *mod = (MGLIRModule *)calloc(1, sizeof(MGLIRModule));
    MGLSemaError *errs = nullptr;
    uint32_t ec = 0;
    mglGLSLSemanticCheck(tu, mod, &errs, &ec);
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
    MGLIRModule *mod = semacheck(src, &tu);
    ASSERT_NE(nullptr, mod);
    SpirvResourceList lists[_MAX_SPIRV_RES] = {{0}};
    ASSERT_EQ(0, mglAirReflectModule(mod, _VERTEX_SHADER, lists, nullptr, 0));

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
    MGLIRModule *mod = semacheck(src, &tu);
    ASSERT_NE(nullptr, mod);
    SpirvResourceList lists[_MAX_SPIRV_RES] = {{0}};
    ASSERT_EQ(0, mglAirReflectModule(mod, _COMPUTE_SHADER, lists, nullptr, 0));

    ASSERT_EQ(1u, lists[_STORAGE_BUFFER_RES].count);
    EXPECT_STREQ("b", lists[_STORAGE_BUFFER_RES].list[0].name);
    EXPECT_EQ(3u, lists[_STORAGE_BUFFER_RES].list[0].gl_binding);
    ASSERT_EQ(1u, lists[_STORAGE_BUFFER_RES].list[0].ubo_member_count);
    EXPECT_STREQ("data", lists[_STORAGE_BUFFER_RES].list[0].ubo_members[0].name);
    EXPECT_EQ(GL_FLOAT, lists[_STORAGE_BUFFER_RES].list[0].ubo_members[0].gl_type);
    EXPECT_EQ(4, lists[_STORAGE_BUFFER_RES].list[0].ubo_members[0].size);
    EXPECT_EQ(0u, lists[_STORAGE_BUFFER_RES].list[0].ubo_members[0].offset);

    ASSERT_EQ(1u, lists[_SAMPLED_IMAGE_RES].count);
    EXPECT_STREQ("tex", lists[_SAMPLED_IMAGE_RES].list[0].name);
    EXPECT_EQ(GL_SAMPLER_2D, lists[_SAMPLED_IMAGE_RES].list[0].gl_type);
    EXPECT_TRUE(lists[_SAMPLED_IMAGE_RES].list[0].msl_has_combined_sampler);

    ASSERT_EQ(1u, lists[_UNIFORM_CONSTANT_RES].count);
    ASSERT_EQ(1u, lists[_UNIFORM_CONSTANT_RES].list[0].ubo_member_count);
    EXPECT_EQ(GL_INT, lists[_UNIFORM_CONSTANT_RES].list[0].ubo_members[0].gl_type);

    mglAirReflectDestroy(lists);
    mglIRModuleDestroy(mod);
    mglGLSLTranslationUnitDestroy(tu);
}

}  // namespace
