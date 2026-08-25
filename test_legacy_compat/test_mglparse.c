/*
 * test_mglparse.c
 *
 * M0.5 verification for the self-written GLSL parser skeleton: parses a
 * hello vertex shader + a compute shader exercising statements, verifies
 * the AST shape via a small dumper, checks that destruction is clean.
 *
 * Build (same pattern as test_mglir):
 *   cc -Wall -Wextra -O0 -g \
 *     -IMGL/include \
 *     test_legacy_compat/test_mglparse.c \
 *     MGL/src/mgl_glsl_parser.c MGL/src/mgl_glsl_lexer.c \
 *     -o build/test_mglparse
 */
#include "mgl_glsl_parser.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, label)                                    \
    do {                                                      \
        tests_run++;                                          \
        if (cond) {                                           \
            tests_passed++;                                   \
            printf("  [PASS] %s\n", (label));                 \
        } else {                                              \
            printf("  [FAIL] %s\n", (label));                 \
        }                                                     \
    } while (0)

/* Recursive AST dumper used to sanity-check structure after parse. */
static void test_hello(void)
{
    const char *src =
        "#version 450 core\n"
        "layout(location = 0) in vec3 pos;\n"
        "layout(location = 1) in vec3 col;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "    color = vec4(pos + col, 1.0);\n"
        "}\n";
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    CHECK(tu != NULL, "parse returns TU");
    CHECK(tu->version == 450, "version = 450");
    CHECK(tu->version_profile && strcmp(tu->version_profile, "core") == 0,
          "profile = core");
    CHECK(tu->error == NULL, "no parse error");
    CHECK(tu->decl_count == 4, "decl_count == 4 (3 vars + main)");

    if (tu && tu->decl_count >= 4) {
        MGLDecl *inpos = tu->decls[0];
        MGLDecl *mainf = tu->decls[3];
        CHECK(inpos->qualifiers & MGL_AST_Q_IN, "pos qualified in");
        CHECK(inpos->type->vec_size == 3, "pos is vec3");
        CHECK(mainf->body != NULL, "main has body");
        CHECK(mainf->param_count == 0, "main no params");
    }
    if (tu) {
        mglGLSLTranslationUnitDestroy(tu);
        printf("  (destroy ok)\n");
    }
}

/* full? no — fragment shader exercising more constructs */
static void test_fragment(void)
{
    const char *src =
        "#version 450 core\n"
        "layout(std140, binding = 0) uniform Block {\n"
        "    mat4 mvp;\n"
        "    vec4 tint;\n"
        "} blk;\n"
        "layout(binding = 1) uniform sampler2D tex;\n"
        "layout(location = 0) in flat vec2 uv;\n"
        "layout(location = 0) out vec4 frag;\n"
        "void main() {\n"
        "    vec4 c = texture(tex, uv);\n"
        "    if (c.a < 0.5) { discard; }\n"
        "    frag = c * blk.tint + blk.mvp[0][0];\n"
        "}\n";
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    CHECK(tu != NULL, "fragment TU created");
    CHECK(tu->error == NULL, "fragment no error");
    CHECK(tu->decl_count == 5, "fragment decl_count == 5");
    if (tu) {
        /* first decl is the UBO block (Block) */
        MGLDecl *blk = tu->decls[0];
        CHECK(blk->qualifiers & MGL_AST_Q_UNIFORM, "block uniform");
        CHECK(blk->struct_member_count == 2, "block has 2 members");
        CHECK(blk->layout == MGL_AST_LAYOUT_STD140, "block std140");
        CHECK(blk->name != NULL && strcmp(blk->name, "blk") == 0,
              "block instance name blk");
        MGLDecl *main = tu->decls[4];
        CHECK(main->body != NULL, "fragment main body");
    }
    if (tu) {
        mglGLSLTranslationUnitDestroy(tu);
    }
}

static void test_expr_path(void)
{
    /* expression-heavy body to exercise binary/assign/postfix/call */
    const char *src =
        "#version 450 core\n"
        "layout(location = 0) in vec4 p;\n"
        "layout(location = 0) out vec4 o;\n"
        "void main() {\n"
        "    float t = dot(p, p) * 0.5;\n"
        "    vec2 v2 = p.xy + vec2(1.0, 2.0);\n"
        "    o = p.x > 0.5 ? vec4(1.0) : vec4(0.0);\n"
        "    for (int i = 0; i < 4; i++) { t += i; }\n"
        "    if (t < 1.0) { t = 0.0; } else { t = 1.0; }\n"
        "}\n";
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    CHECK(tu != NULL, "expr TU");
    CHECK(tu->error == NULL, "expr no error");
    if (tu) {
        CHECK(tu->decl_count == 3, "expr decl_count == 3");
        MGLDecl *main = tu->decls[2];
        CHECK(main->body != NULL, "main body");
    }
    if (tu) {
        mglGLSLTranslationUnitDestroy(tu);
    }
}

static void test_precision_statements(void)
{
    const char *src =
        "#version 460 core\n"
        "precision lowp int;\n"
        "precision mediump float;\n"
        "precision highp sampler2D;\n"
        "lowp int low_value;\n"
        "mediump float medium_value;\n"
        "highp uint high_value;\n"
        "void main() { medium_value = 1.0; }\n";
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    CHECK(tu != NULL, "precision TU");
    CHECK(tu->error == NULL, "precision statements accepted");
    CHECK(tu->decl_count == 4,
          "precision statements do not create declarations");
    if (tu && tu->decl_count >= 3) {
        CHECK(tu->decls[0]->type->precision == MGL_AST_PRECISION_LOWP,
              "lowp declaration precision");
        CHECK(tu->decls[1]->type->precision == MGL_AST_PRECISION_MEDIUMP,
              "mediump declaration precision");
        CHECK(tu->decls[2]->type->precision == MGL_AST_PRECISION_HIGHP,
              "highp declaration precision");
    }
    if (tu) {
        mglGLSLTranslationUnitDestroy(tu);
    }
}

static void test_image_memory_qualifiers(void)
{
    const char *src =
        "#version 440 core\n"
        "layout(rgba32i, binding = 0) writeonly uniform highp "
        "iimage2DArray array_image;\n"
        "void main() {}\n";
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    CHECK(tu != NULL, "image qualifier TU created");
    CHECK(tu && tu->error == NULL,
          "image memory qualifier and integer array image accepted");
    if (tu && tu->decl_count >= 1) {
        MGLDecl *image = tu->decls[0];
        CHECK((image->qualifiers & MGL_AST_Q_UNIFORM) != 0,
              "image remains uniform");
        CHECK(image->type && image->type->base == MGL_AST_TYPE_IMAGE,
              "iimage2DArray classified as image");
        CHECK(image->type && image->type->name &&
              strcmp(image->type->name, "iimage2DArray") == 0,
              "integer array image typename preserved");
    }
    if (tu) {
        mglGLSLTranslationUnitDestroy(tu);
    }
}

static void test_error_report(void)
{
    const char *src = "void main( { }\n"; /* missing ')' */
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    CHECK(tu != NULL, "bad TU still returned");
    CHECK(tu->error != NULL, "parse error reported");
    if (tu) {
        mglGLSLTranslationUnitDestroy(tu);
    }
}

int main(void)
{
    printf("MGLGLSL parser skeleton tests\n");
    test_hello();
    test_fragment();
    test_expr_path();
    test_precision_statements();
    test_image_memory_qualifiers();
    test_error_report();
    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
