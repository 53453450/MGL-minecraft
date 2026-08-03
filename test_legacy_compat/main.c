/*
 * test_legacy_compat.c
 *
 * Standalone test for mgl_legacy_detect / mgl_translate_legacy_glsl.
 * Compiles directly with mgl_legacy_compat.c (no dylib dependency).
 *
 * Build:
 *   cc -Wall -Wextra -O0 -g \
 *     -IMGL/include -IMGL/include/GL \
 *     test_legacy_compat/main.c MGL/src/mgl_legacy_compat.c \
 *     -o build/test_legacy_compat
 */
#include "mgl_legacy_compat.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define BUF_SIZE 8192

static void check(int condition, const char *label, const char *detail)
{
    tests_run++;
    if (condition) {
        tests_passed++;
        printf("  [PASS] %s\n", label);
    } else {
        tests_failed++;
        printf("  [FAIL] %s%s%s\n", label,
               detail ? " — " : "", detail ? detail : "");
    }
}

static int contains(const char *haystack, const char *needle)
{
    return strstr(haystack, needle) != NULL;
}

static int not_contains(const char *haystack, const char *needle)
{
    return strstr(haystack, needle) == NULL;
}

static void copy_to_buf(const char *src, char *buf, size_t cap)
{
    memset(buf, 0, cap);
    strncpy(buf, src, cap - 1);
    buf[cap - 1] = '\0';
}

/* ===== Test cases ===== */

static void test_detect_vertex_110(void)
{
    printf("\n=== test_detect_vertex_110 ===\n");
    const char *src =
        "#version 110\n"
        "attribute vec3 a_pos;\n"
        "attribute vec2 a_uv;\n"
        "varying vec2 v_uv;\n"
        "uniform mat4 gl_ModelViewProjectionMatrix;\n"
        "void main() {\n"
        "    v_uv = a_uv;\n"
        "    gl_Position = gl_ModelViewProjectionMatrix * vec4(a_pos, 1.0);\n"
        "}\n";

    mgl_legacy_features_t feat;
    mgl_legacy_detect(src, &feat);

    check(feat.has_attribute, "detect attribute", NULL);
    check(feat.has_varying, "detect varying", NULL);
    check(!feat.has_gl_FragColor, "no gl_FragColor in VS", NULL);
    check(!feat.has_gl_TexCoord, "no gl_TexCoord", NULL);
    check(feat.needs_translation, "needs_translation", NULL);
}

static void test_detect_fragment_110(void)
{
    printf("\n=== test_detect_fragment_110 ===\n");
    const char *src =
        "#version 110\n"
        "varying vec2 v_uv;\n"
        "uniform sampler2D tex;\n"
        "void main() {\n"
        "    gl_FragColor = texture2D(tex, v_uv);\n"
        "}\n";

    mgl_legacy_features_t feat;
    mgl_legacy_detect(src, &feat);

    check(feat.has_varying, "detect varying in FS", NULL);
    check(feat.has_gl_FragColor, "detect gl_FragColor", NULL);
    check(feat.has_texture2D, "detect texture2D", NULL);
    check(feat.needs_translation, "needs_translation", NULL);
}

static void test_detect_comment_aware(void)
{
    printf("\n=== test_detect_comment_aware ===\n");
    /* Keywords only appear in comments — should NOT be detected. */
    const char *src =
        "#version 330\n"
        "// this uses attribute and varying in a comment\n"
        "/* gl_FragColor and texture2D in block comment */\n"
        "in vec3 a_pos;\n"
        "void main() { gl_Position = vec4(a_pos, 1.0); }\n";

    mgl_legacy_features_t feat;
    mgl_legacy_detect(src, &feat);

    check(!feat.has_attribute, "attribute in comment not detected", NULL);
    check(!feat.has_varying, "varying in comment not detected", NULL);
    check(!feat.has_gl_FragColor, "gl_FragColor in comment not detected", NULL);
    check(!feat.has_texture2D, "texture2D in comment not detected", NULL);
    check(!feat.needs_translation, "no false translation needed", NULL);
}

static void test_detect_string_aware(void)
{
    printf("\n=== test_detect_string_aware ===\n");
    /* GLSL doesn't have string literals, but the scanner should not crash
     * on a lone double-quote in a comment. */
    const char *src =
        "#version 110\n"
        "// comment with \"quoted attribute word\"\n"
        "attribute vec3 a_pos;\n"
        "void main() { gl_Position = vec4(a_pos, 1.0); }\n";

    mgl_legacy_features_t feat;
    mgl_legacy_detect(src, &feat);

    check(feat.has_attribute, "attribute outside comment detected", NULL);
    check(feat.needs_translation, "needs_translation", NULL);
}

static void test_translate_vertex_110(void)
{
    printf("\n=== test_translate_vertex_110 ===\n");
    const char *src =
        "#version 110\n"
        "attribute vec3 a_pos;\n"
        "attribute vec2 a_uv;\n"
        "varying vec2 v_uv;\n"
        "void main() {\n"
        "    v_uv = a_uv;\n"
        "    gl_Position = vec4(a_pos, 1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1 (modified)", NULL);

    check(contains(buf, "in vec3 a_pos;"), "attribute -> in", NULL);
    check(contains(buf, "in vec2 a_uv;"), "second attribute -> in", NULL);
    check(contains(buf, "out vec2 v_uv;"), "varying -> out (VS)", NULL);
    check(not_contains(buf, "attribute "), "no 'attribute' keyword left", NULL);
    check(not_contains(buf, "varying "), "no 'varying' keyword left", NULL);
    check(contains(buf, "#version 110"), "version line preserved", NULL);
}

static void test_translate_fragment_110(void)
{
    printf("\n=== test_translate_fragment_110 ===\n");
    const char *src =
        "#version 110\n"
        "varying vec2 v_uv;\n"
        "uniform sampler2D tex;\n"
        "void main() {\n"
        "    gl_FragColor = texture2D(tex, v_uv);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_FRAGMENT_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1 (modified)", NULL);

    check(contains(buf, "in vec2 v_uv;"), "varying -> in (FS)", NULL);
    check(contains(buf, "_mglFragColor = texture(tex,"), "gl_FragColor renamed + texture2D -> texture", NULL);
    check(contains(buf, "layout(location = 0) out vec4 _mglFragColor;"), "gl_FragColor declaration injected", NULL);
    check(not_contains(buf, "gl_FragColor"), "no gl_FragColor left", NULL);
    check(not_contains(buf, "texture2D"), "no texture2D left", NULL);
}

static void test_translate_fragdata_120(void)
{
    printf("\n=== test_translate_fragdata_120 ===\n");
    const char *src =
        "#version 120\n"
        "varying vec4 v_color;\n"
        "void main() {\n"
        "    gl_FragData[0] = v_color;\n"
        "    gl_FragData[1] = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_FRAGMENT_SHADER, 120, NULL);
    check(ret == 1, "translate returns 1", NULL);

    check(contains(buf, "_mglFragData[0]"), "gl_FragData[0] renamed", NULL);
    check(contains(buf, "_mglFragData[1]"), "gl_FragData[1] renamed", NULL);
    check(contains(buf, "layout(location = 0) out vec4 _mglFragData["), "gl_FragData declaration injected", NULL);
    check(not_contains(buf, "gl_FragData"), "no gl_FragData left", NULL);
}

static void test_translate_texcoord_110(void)
{
    printf("\n=== test_translate_texcoord_110 ===\n");
    const char *src =
        "#version 110\n"
        "void main() {\n"
        "    gl_TexCoord[0] = gl_MultiTexCoord0;\n"
        "    gl_Position = ftransform();\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);

    check(contains(buf, "_mglTexCoord[0]"), "gl_TexCoord[0] renamed", NULL);
    check(contains(buf, "out vec4 _mglTexCoord["), "gl_TexCoord declaration injected (VS out)", NULL);
    check(not_contains(buf, "gl_TexCoord"), "no gl_TexCoord left", NULL);
}

static void test_translate_texture_funcs(void)
{
    printf("\n=== test_translate_texture_funcs ===\n");
    const char *src =
        "#version 110\n"
        "uniform sampler2D tex2d;\n"
        "uniform sampler3D tex3d;\n"
        "uniform samplerCube texcube;\n"
        "varying vec2 v_uv;\n"
        "varying vec3 v_dir;\n"
        "void main() {\n"
        "    vec4 a = texture2D(tex2d, v_uv);\n"
        "    vec4 b = texture3D(tex3d, vec3(v_uv, 0.0));\n"
        "    vec4 c = textureCube(texcube, v_dir);\n"
        "    gl_FragColor = a + b + c;\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_FRAGMENT_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);

    check(contains(buf, "texture(tex2d,"), "texture2D -> texture", NULL);
    check(contains(buf, "texture(tex3d,"), "texture3D -> texture", NULL);
    check(contains(buf, "texture(texcube,"), "textureCube -> texture", NULL);
    check(not_contains(buf, "texture2D"), "no texture2D left", NULL);
    check(not_contains(buf, "texture3D"), "no texture3D left", NULL);
    check(not_contains(buf, "textureCube"), "no textureCube left", NULL);
}

static void test_translate_texture2DProj(void)
{
    printf("\n=== test_translate_texture2DProj ===\n");
    const char *src =
        "#version 110\n"
        "uniform sampler2D tex;\n"
        "varying vec4 v_uvq;\n"
        "void main() {\n"
        "    gl_FragColor = texture2DProj(tex, v_uvq);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_FRAGMENT_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);

    check(contains(buf, "textureProj(tex,"), "texture2DProj -> textureProj", NULL);
    /* texture2DProj must not be partially replaced to textureProjProj
     * or leave "Proj" dangling */
    check(not_contains(buf, "texture2DProj"), "no texture2DProj left", NULL);
    check(not_contains(buf, "textureProjProj"), "no double Proj", NULL);
}

static void test_no_regression_330(void)
{
    printf("\n=== test_no_regression_330 ===\n");
    const char *src =
        "#version 330 core\n"
        "in vec3 a_pos;\n"
        "out vec2 v_uv;\n"
        "void main() {\n"
        "    v_uv = a_pos.xy;\n"
        "    gl_Position = vec4(a_pos, 1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 330, NULL);
    check(ret == 0, "translate returns 0 (no changes for 330)", NULL);
    check(contains(buf, "in vec3 a_pos;"), "source unchanged", NULL);
}

static void test_identifier_boundary(void)
{
    printf("\n=== test_identifier_boundary ===\n");
    /* 'my_attribute' should NOT be rewritten to 'my_in' */
    const char *src =
        "#version 110\n"
        "attribute float my_attribute;\n"
        "varying float my_varying;\n"
        "void main() {\n"
        "    my_varying = my_attribute;\n"
        "    gl_Position = vec4(1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);

    check(contains(buf, "in float my_attribute;"), "attribute keyword -> in, my_attribute unchanged", NULL);
    check(contains(buf, "out float my_varying;"), "varying keyword -> out, my_varying unchanged", NULL);
    check(not_contains(buf, "my_in"), "my_attribute not mangled", NULL);
    check(not_contains(buf, "my_out"), "my_varying not mangled", NULL);
}

static void test_gl_VertexID_not_touched(void)
{
    printf("\n=== test_gl_VertexID_not_touched ===\n");
    /* gl_VertexID contains 'gl_Vertex' as prefix — but gl_Vertex is handled
     * by mglRewriteLegacyGLSL, not by this module.  This test just confirms
     * our module does not touch gl_VertexID when checking for unrelated
     * legacy features. */
    const char *src =
        "#version 110\n"
        "void main() {\n"
        "    int id = gl_VertexID;\n"
        "    gl_Position = vec4(float(id));\n"
        "}\n";

    mgl_legacy_features_t feat;
    mgl_legacy_detect(src, &feat);

    check(!feat.has_gl_TexCoord, "gl_VertexID not confused with gl_TexCoord", NULL);
    check(!feat.has_gl_FragColor, "gl_VertexID not confused with gl_FragColor", NULL);
    check(!feat.needs_translation || feat.has_ftransform == 0,
          "no false legacy detection from gl_VertexID", NULL);
}

static void test_combined_vertex_110(void)
{
    printf("\n=== test_combined_vertex_110 ===\n");
    /* Realistic GLSL 110 vertex shader combining multiple legacy features. */
    const char *src =
        "#version 110\n"
        "attribute vec3 inPosition;\n"
        "attribute vec2 inTexCoord;\n"
        "attribute vec3 inNormal;\n"
        "varying vec2 outTexCoord;\n"
        "varying vec3 outNormal;\n"
        "uniform mat4 mvp;\n"
        "void main() {\n"
        "    gl_TexCoord[0] = vec4(inTexCoord, 0.0, 0.0);\n"
        "    outTexCoord = inTexCoord;\n"
        "    outNormal = inNormal;\n"
        "    gl_Position = mvp * vec4(inPosition, 1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);

    check(contains(buf, "in vec3 inPosition;"), "attribute -> in (pos)", NULL);
    check(contains(buf, "in vec2 inTexCoord;"), "attribute -> in (uv)", NULL);
    check(contains(buf, "in vec3 inNormal;"), "attribute -> in (normal)", NULL);
    check(contains(buf, "out vec2 outTexCoord;"), "varying -> out", NULL);
    check(contains(buf, "out vec3 outNormal;"), "varying -> out (normal)", NULL);
    check(contains(buf, "_mglTexCoord[0]"), "gl_TexCoord renamed", NULL);
    check(contains(buf, "out vec4 _mglTexCoord["), "gl_TexCoord decl injected", NULL);
    check(not_contains(buf, "attribute "), "no attribute keyword", NULL);
    check(not_contains(buf, "varying "), "no varying keyword", NULL);
    check(not_contains(buf, "gl_TexCoord"), "no gl_TexCoord", NULL);
}

static void test_combined_fragment_110(void)
{
    printf("\n=== test_combined_fragment_110 ===\n");
    const char *src =
        "#version 110\n"
        "varying vec2 outTexCoord;\n"
        "varying vec3 outNormal;\n"
        "uniform sampler2D diffuseMap;\n"
        "uniform samplerCube envMap;\n"
        "void main() {\n"
        "    vec4 diff = texture2D(diffuseMap, outTexCoord);\n"
        "    vec4 env = textureCube(envMap, outNormal);\n"
        "    gl_FragColor = diff * 0.7 + env * 0.3;\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_FRAGMENT_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);

    check(contains(buf, "in vec2 outTexCoord;"), "varying -> in (FS)", NULL);
    check(contains(buf, "in vec3 outNormal;"), "varying -> in (FS normal)", NULL);
    check(contains(buf, "texture(diffuseMap,"), "texture2D -> texture", NULL);
    check(contains(buf, "texture(envMap,"), "textureCube -> texture", NULL);
    check(contains(buf, "_mglFragColor ="), "gl_FragColor renamed", NULL);
    check(contains(buf, "layout(location = 0) out vec4 _mglFragColor;"), "decl injected", NULL);
}

static void test_version_150_transitional(void)
{
    printf("\n=== test_version_150_transitional ===\n");
    /* GLSL 1.50 is transitional — both old and new syntax may appear.
     * This shader uses legacy gl_FragColor but version 150. */
    const char *src =
        "#version 150\n"
        "in vec2 v_uv;\n"
        "uniform sampler2D tex;\n"
        "void main() {\n"
        "    gl_FragColor = texture(tex, v_uv);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_FRAGMENT_SHADER, 150, NULL);
    check(ret == 1, "translate returns 1 (150 < 330)", NULL);
    check(contains(buf, "_mglFragColor"), "gl_FragColor translated for 150", NULL);
}

static void test_null_safety(void)
{
    printf("\n=== test_null_safety ===\n");
    mgl_legacy_features_t feat;
    mgl_legacy_detect(NULL, &feat);
    check(!feat.needs_translation, "detect(NULL) safe", NULL);

    int ret = mgl_translate_legacy_glsl(NULL, 100, GL_VERTEX_SHADER, 110, NULL);
    check(ret == -1, "translate(NULL) returns -1", NULL);

    char buf[64];
    copy_to_buf("#version 110\n", buf, sizeof(buf));
    ret = mgl_translate_legacy_glsl(buf, 0, GL_VERTEX_SHADER, 110, NULL);
    check(ret == -1, "translate(capacity=0) returns -1", NULL);
}

/* ===== New tests for spec-compliance additions ===== */

static void test_translate_texture1D(void)
{
    printf("\n=== test_translate_texture1D ===\n");
    const char *src =
        "#version 110\n"
        "uniform sampler1D tex;\n"
        "varying float v_coord;\n"
        "void main() {\n"
        "    gl_FragColor = texture1D(tex, v_coord);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_FRAGMENT_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);
    check(contains(buf, "texture(tex,"), "texture1D -> texture", NULL);
    check(not_contains(buf, "texture1D"), "no texture1D left", NULL);
}

static void test_translate_texture1DProj_3DProj(void)
{
    printf("\n=== test_translate_texture1DProj_3DProj ===\n");
    const char *src =
        "#version 110\n"
        "uniform sampler1D tex1d;\n"
        "uniform sampler3D tex3d;\n"
        "varying vec4 v_uvq;\n"
        "void main() {\n"
        "    vec4 a = texture1DProj(tex1d, v_uvq);\n"
        "    vec4 b = texture3DProj(tex3d, v_uvq);\n"
        "    gl_FragColor = a + b;\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_FRAGMENT_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);
    check(contains(buf, "textureProj(tex1d,"), "texture1DProj -> textureProj", NULL);
    check(contains(buf, "textureProj(tex3d,"), "texture3DProj -> textureProj", NULL);
    check(not_contains(buf, "texture1DProj"), "no texture1DProj left", NULL);
    check(not_contains(buf, "texture3DProj"), "no texture3DProj left", NULL);
}

static void test_translate_gl_Normal(void)
{
    printf("\n=== test_translate_gl_Normal ===\n");
    const char *src =
        "#version 110\n"
        "attribute vec3 a_pos;\n"
        "void main() {\n"
        "    vec3 n = gl_Normal;\n"
        "    gl_Position = vec4(a_pos + n * 0.01, 1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);
    check(contains(buf, "_mglNormal"), "gl_Normal -> _mglNormal", NULL);
    check(not_contains(buf, "gl_Normal"), "no gl_Normal left", NULL);
    check(contains(buf, "in vec3 _mglNormal;"), "gl_Normal declaration injected", NULL);
}

static void test_translate_gl_Color_VS(void)
{
    printf("\n=== test_translate_gl_Color_VS ===\n");
    const char *src =
        "#version 110\n"
        "void main() {\n"
        "    gl_FrontColor = gl_Color;\n"
        "    gl_Position = vec4(1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);
    check(contains(buf, "_mglFrontColor = _mglColor;"), "gl_FrontColor->out, gl_Color->in (VS)", NULL);
    check(contains(buf, "in vec4 _mglColor;"), "gl_Color VS decl injected (in)", NULL);
    check(contains(buf, "out vec4 _mglFrontColor;"), "gl_FrontColor VS decl injected (out)", NULL);
    check(not_contains(buf, "gl_Color"), "no gl_Color left", NULL);
    check(not_contains(buf, "gl_FrontColor"), "no gl_FrontColor left", NULL);
}

static void test_translate_gl_Color_FS(void)
{
    printf("\n=== test_translate_gl_Color_FS ===\n");
    /* In FS, gl_Color is a varying input corresponding to VS's gl_FrontColor.
     * It should be renamed to _mglFrontColor to link with VS output. */
    const char *src =
        "#version 110\n"
        "varying vec4 gl_Color; /* implicit */\n"
        "void main() {\n"
        "    gl_FragColor = gl_Color;\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_FRAGMENT_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);
    check(contains(buf, "_mglFrontColor"), "FS gl_Color -> _mglFrontColor", NULL);
    check(contains(buf, "in vec4 _mglFrontColor;"), "FS gl_Color decl injected (in)", NULL);
}

static void test_translate_gl_MultiTexCoord(void)
{
    printf("\n=== test_translate_gl_MultiTexCoord ===\n");
    const char *src =
        "#version 110\n"
        "void main() {\n"
        "    gl_TexCoord[0] = gl_MultiTexCoord0;\n"
        "    gl_TexCoord[1] = gl_MultiTexCoord1;\n"
        "    gl_Position = vec4(1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);
    check(contains(buf, "_mglMultiTexCoord0"), "gl_MultiTexCoord0 renamed", NULL);
    check(contains(buf, "_mglMultiTexCoord1"), "gl_MultiTexCoord1 renamed", NULL);
    check(contains(buf, "in vec4 _mglMultiTexCoord0;"), "gl_MultiTexCoord0 decl injected", NULL);
    check(contains(buf, "in vec4 _mglMultiTexCoord1;"), "gl_MultiTexCoord1 decl injected", NULL);
    check(not_contains(buf, "gl_MultiTexCoord0"), "no gl_MultiTexCoord0 left", NULL);
    check(not_contains(buf, "gl_MultiTexCoord1"), "no gl_MultiTexCoord1 left", NULL);
}

static void test_translate_gl_FogFragCoord(void)
{
    printf("\n=== test_translate_gl_FogFragCoord ===\n");
    /* gl_FogFragCoord is VS:out, FS:in — same name both stages */
    const char *vs_src =
        "#version 110\n"
        "varying float gl_FogFragCoord;\n"
        "void main() {\n"
        "    gl_FogFragCoord = 0.5;\n"
        "    gl_Position = vec4(1.0);\n"
        "}\n";
    char vs_buf[BUF_SIZE];
    copy_to_buf(vs_src, vs_buf, BUF_SIZE);
    int vs_ret = mgl_translate_legacy_glsl(vs_buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(vs_ret == 1, "VS translate returns 1", NULL);
    check(contains(vs_buf, "out float _mglFogFragCoord;"), "VS gl_FogFragCoord decl (out)", NULL);

    const char *fs_src =
        "#version 110\n"
        "varying float gl_FogFragCoord;\n"
        "void main() {\n"
        "    gl_FragColor = vec4(gl_FogFragCoord);\n"
        "}\n";
    char fs_buf[BUF_SIZE];
    copy_to_buf(fs_src, fs_buf, BUF_SIZE);
    int fs_ret = mgl_translate_legacy_glsl(fs_buf, BUF_SIZE, GL_FRAGMENT_SHADER, 110, NULL);
    check(fs_ret == 1, "FS translate returns 1", NULL);
    check(contains(fs_buf, "in float _mglFogFragCoord;"), "FS gl_FogFragCoord decl (in)", NULL);
}

static void test_translate_gl_ClipVertex(void)
{
    printf("\n=== test_translate_gl_ClipVertex ===\n");
    const char *src =
        "#version 110\n"
        "void main() {\n"
        "    gl_ClipVertex = gl_ModelViewMatrix * vec4(1.0);\n"
        "    gl_Position = vec4(1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);
    check(contains(buf, "_mglClipVertex"), "gl_ClipVertex -> _mglClipVertex", NULL);
    check(contains(buf, "out vec4 _mglClipVertex;"), "gl_ClipVertex decl injected (out)", NULL);
    check(not_contains(buf, "gl_ClipVertex"), "no gl_ClipVertex left", NULL);
}

static void test_translate_gl_BackColor(void)
{
    printf("\n=== test_translate_gl_BackColor ===\n");
    const char *src =
        "#version 110\n"
        "void main() {\n"
        "    gl_FrontColor = vec4(1.0);\n"
        "    gl_BackColor = vec4(0.5);\n"
        "    gl_Position = vec4(1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);
    check(contains(buf, "_mglFrontColor"), "gl_FrontColor renamed", NULL);
    check(contains(buf, "_mglBackColor"), "gl_BackColor renamed", NULL);
    check(contains(buf, "out vec4 _mglFrontColor;"), "gl_FrontColor decl injected", NULL);
    check(contains(buf, "out vec4 _mglBackColor;"), "gl_BackColor decl injected", NULL);
}

static void test_full_legacy_vertex_110(void)
{
    printf("\n=== test_full_legacy_vertex_110 ===\n");
    /* Comprehensive GLSL 1.10 vertex shader using many legacy features. */
    const char *src =
        "#version 110\n"
        "attribute vec3 inPosition;\n"
        "attribute vec3 inNormal;\n"
        "attribute vec2 inTexCoord;\n"
        "varying vec3 vNormal;\n"
        "varying vec2 vTexCoord;\n"
        "uniform mat4 mvp;\n"
        "void main() {\n"
        "    gl_TexCoord[0] = vec4(inTexCoord, 0.0, 0.0);\n"
        "    gl_FrontColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "    gl_FogFragCoord = 0.0;\n"
        "    vNormal = inNormal;\n"
        "    vTexCoord = inTexCoord;\n"
        "    gl_Position = mvp * vec4(inPosition, 1.0);\n"
        "}\n";

    char buf[BUF_SIZE];
    copy_to_buf(src, buf, BUF_SIZE);

    int ret = mgl_translate_legacy_glsl(buf, BUF_SIZE, GL_VERTEX_SHADER, 110, NULL);
    check(ret == 1, "translate returns 1", NULL);
    check(contains(buf, "in vec3 inPosition;"), "attribute -> in", NULL);
    check(contains(buf, "out vec3 vNormal;"), "varying -> out", NULL);
    check(contains(buf, "_mglTexCoord[0]"), "gl_TexCoord renamed", NULL);
    check(contains(buf, "_mglFrontColor"), "gl_FrontColor renamed", NULL);
    check(contains(buf, "_mglFogFragCoord"), "gl_FogFragCoord renamed", NULL);
    check(contains(buf, "out vec4 _mglTexCoord["), "gl_TexCoord decl", NULL);
    check(contains(buf, "out vec4 _mglFrontColor;"), "gl_FrontColor decl", NULL);
    check(contains(buf, "out float _mglFogFragCoord;"), "gl_FogFragCoord decl", NULL);
    check(not_contains(buf, "attribute "), "no attribute keyword", NULL);
    check(not_contains(buf, "varying "), "no varying keyword", NULL);
}

int main(void)
{
    printf("=== mgl_legacy_compat tests ===\n");

    test_detect_vertex_110();
    test_detect_fragment_110();
    test_detect_comment_aware();
    test_detect_string_aware();
    test_translate_vertex_110();
    test_translate_fragment_110();
    test_translate_fragdata_120();
    test_translate_texcoord_110();
    test_translate_texture_funcs();
    test_translate_texture2DProj();
    test_no_regression_330();
    test_identifier_boundary();
    test_gl_VertexID_not_touched();
    test_combined_vertex_110();
    test_combined_fragment_110();
    test_version_150_transitional();
    test_null_safety();

    /* Spec-compliance additions */
    test_translate_texture1D();
    test_translate_texture1DProj_3DProj();
    test_translate_gl_Normal();
    test_translate_gl_Color_VS();
    test_translate_gl_Color_FS();
    test_translate_gl_MultiTexCoord();
    test_translate_gl_FogFragCoord();
    test_translate_gl_ClipVertex();
    test_translate_gl_BackColor();
    test_full_legacy_vertex_110();

    printf("\n=== Summary ===\n");
    printf("Total: %d, Passed: %d, Failed: %d\n",
           tests_run, tests_passed, tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
