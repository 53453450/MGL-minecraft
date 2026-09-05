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
#include "mgl_glsl_cpp.h"

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

/* GLSL 4.60 §4.1 / §6.1: comma-separated declarators and precision-
 * qualified function parameters (KHR-GL46.shaders uniform_block/loops
 * front-door constructs). */
static void test_declarator_lists_and_param_precision(void)
{
    const char *src =
        "#version 460 core\n"
        "uniform mediump int ui_zero, ui_one, ui_two;\n"
        "float pair[2], triple[3];\n"
        "void main() {\n"
        "  int a = 1, b = 2, c;\n"
        "  c = a + b;\n"
        "  ui_zero = c + ui_one + ui_two;\n"
        "}\n";
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    CHECK(tu != NULL, "declarator list TU");
    CHECK(tu && tu->error == NULL, "comma declarators accepted");
    if (tu && tu->decl_count >= 2) {
        MGLDecl *u = tu->decls[0];
        MGLDecl *g = tu->decls[1];
        int uniform_count = 0;
        for (MGLDecl *cur = u; cur; cur = cur->next_declarator)
            uniform_count++;
        CHECK(uniform_count == 3, "uniform chain has 3 declarators");
        CHECK(u->type_shared == 0 && u->next_declarator &&
              u->next_declarator->type_shared == 1,
              "chain nodes share the first type spec");
        CHECK(u->next_declarator && u->next_declarator->name &&
              strcmp(u->next_declarator->name, "ui_one") == 0,
              "second uniform declarator named ui_one");
        CHECK(u->type->precision == MGL_AST_PRECISION_MEDIUMP,
              "precision qualifier recorded on shared type");
        CHECK(g->next_declarator != NULL &&
              g->next_declarator->array_count == 1 &&
              g->next_declarator->array_dims[0] == 3,
              "per-declarator array dims (triple[3])");
    }
    if (tu) {
        mglGLSLTranslationUnitDestroy(tu);
    }

    const char *src2 =
        "#version 460 core\n"
        "mediump float compare(highp float a, highp float b) "
        "{ return a < b ? 1.0 : 0.0; }\n"
        "float mix2(in highp float x, highp in float y) "
        "{ return compare(x, y); }\n"
        "void main() { }\n";
    MGLTranslationUnit *tu2 = mglGLSLParse(src2, strlen(src2));
    CHECK(tu2 != NULL, "param precision TU");
    CHECK(tu2 && tu2->error == NULL, "precision-qualified params accepted");
    if (tu2 && tu2->decl_count >= 2) {
        MGLDecl *fn = tu2->decls[0];
        CHECK(fn->param_count == 2 && fn->params &&
              fn->params[0]->type->precision == MGL_AST_PRECISION_HIGHP,
              "param precision recorded (highp float)");
        MGLDecl *fn2 = tu2->decls[1];
        CHECK(fn2->param_count == 2 && fn2->params &&
              (fn2->params[0]->qualifiers & MGL_AST_Q_IN) &&
              fn2->params[0]->type->precision == MGL_AST_PRECISION_HIGHP,
              "in highp float accepted");
        CHECK((fn2->params[1]->qualifiers & MGL_AST_Q_IN) &&
              fn2->params[1]->type->precision == MGL_AST_PRECISION_HIGHP,
              "highp in float accepted");
    }
    if (tu2) {
        mglGLSLTranslationUnitDestroy(tu2);
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

static void test_gs_layout_errors(void)
{
    const char *zero_inv =
        "#version 450 core\n"
        "layout(points, invocations=0) in;\n"
        "layout(points, max_vertices=1) out;\n"
        "void main() { }\n";
    MGLTranslationUnit *tu = mglGLSLParse(zero_inv, strlen(zero_inv));
    CHECK(tu != NULL, "invocations=0 still returns TU");
    CHECK(tu && tu->error != NULL, "invocations=0 is a parse error");
    if (tu) mglGLSLTranslationUnitDestroy(tu);

    const char *ok_inv =
        "#version 450 core\n"
        "layout(points, invocations=2) in;\n"
        "layout(points, max_vertices=1) out;\n"
        "void main() { }\n";
    tu = mglGLSLParse(ok_inv, strlen(ok_inv));
    CHECK(tu != NULL && tu->error == NULL, "invocations=2 parses");
    CHECK(tu && tu->layout_invocations == 2, "invocations=2 recorded");
    CHECK(tu && tu->layout_max_vertices == 1, "max_vertices=1 recorded");
    if (tu) mglGLSLTranslationUnitDestroy(tu);

    const char *missing_max =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points) out;\n"
        "void main() { }\n";
    tu = mglGLSLParse(missing_max, strlen(missing_max));
    CHECK(tu != NULL && tu->error == NULL, "missing max_vertices still parses");
    CHECK(tu && tu->layout_max_vertices < 0, "max_vertices unspecified is -1");
    if (tu) mglGLSLTranslationUnitDestroy(tu);
}

static int parse_ok(const char *src)
{
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    int ok = tu && tu->error == NULL;
    if (tu) {
        mglGLSLTranslationUnitDestroy(tu);
    }
    return ok;
}

static int parse_fails(const char *src)
{
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    int ok = tu && tu->error != NULL;
    if (tu) {
        mglGLSLTranslationUnitDestroy(tu);
    }
    return ok;
}

static int pp_contains(const char *src, const char *needle)
{
    char err[256];
    char *out = mglGLSLPreprocess(src, strlen(src), err, sizeof(err));
    int ok;
    if (!out) {
        printf("    pp error: %s\n", err);
        return 0;
    }
    ok = strstr(out, needle) != NULL;
    if (!ok) {
        printf("    pp out: [%s]\n", out);
    }
    free(out);
    return ok;
}

static void test_preprocessor(void)
{
    CHECK(pp_contains("#version 450\n#define A 3\nint x = A;\n", "3"),
          "object macro expands");
    CHECK(pp_contains("#version 450\n#define F(x) x\nint x = F(7);\n", "7"),
          "function macro expands");
    CHECK(pp_contains("#version 450\n#define P(a,b) a##b\nfloat varfoo;\n"
                      "float x = P(var, foo);\n",
                      "varfoo"),
          "token pasting ##");
    CHECK(pp_contains("#version 330\n#if 1 + 2 * 3 == 7\nint x = 1;\n"
                      "#else\nint x = 0;\n#endif\n",
                      "x = 1"),
          "#if * binds tighter than +");
    CHECK(pp_contains("#version 330\n#if 1 || UNDEF\nint x = 1;\n"
                      "#else\nint x = 0;\n#endif\n",
                      "x = 1"),
          "#if 1 || undefined is taken");
    CHECK(parse_ok("#version 330\n#if UNDEF\nint x = 1;\n#endif\n"
                      "void main() {}\n"),
          "undefined #if identifier is 0");
    CHECK(parse_fails("#version 330\n#defin AAA\nvoid main() {}\n"),
          "unknown directive fails");
    CHECK(parse_fails("#version 330\n#define GL_VALUE 1\nvoid main() {}\n"),
          "GL_ reserved name fails");
    CHECK(parse_fails("#version 330\n#define S(a) #a\nvoid main() {}\n"),
          "stringification is rejected");
    CHECK(parse_fails("precision mediump float;\n#version 330\nvoid main() {}\n"),
          "#version not first fails");
    CHECK(parse_fails("#version 329\nvoid main() {}\n"),
          "unsupported #version fails");
    CHECK(parse_fails("#version 330\n#define VALUE (AAA - 1.0)\n"
                      "#define VALUE (AAA- 1.0)\nvoid main() {}\n"),
          "redefinition whitespace differs");
    CHECK(parse_ok("#version 450 core\n#define VALUE\n"
                   "void main() { float x = VALUE - 1.0; }\n"),
          "empty object macro");
    CHECK(parse_ok("#version 450 core\n#define A(X) var##X\n"
                   "void main() { float varfoo = 1.0; float y = A(foo); }\n"),
          "token paste identifier compiles");
}

static void test_array_type_syntax(void)
{
    CHECK(parse_ok("#version 330\n"
                   "void main() {\n"
                   "  float[3] x;\n"
                   "  x = float[3](1.0, 2.0, 3.0);\n"
                   "  float[] y = float[](1.0, 2.0);\n"
                   "  int n = x.length();\n"
                   "}\n"),
          "T[N] decl, ctor, length");
    CHECK(parse_ok("#version 330\n"
                   "float[3] f(float[3] a) { return float[3](a[2], a[0], a[1]); }\n"
                   "void main() { float[3] x = f(float[3](1.0, 2.0, 3.0)); }\n"),
          "array return and param");
    CHECK(parse_ok("#version 330\n"
                   "float[3] f(float[3]);\n"
                   "void main() { float[3] a = float[3](1.0, 2.0, 3.0); a = f(a); }\n"
                   "float[3] f(float[3] a) { return a; }\n"),
          "unnamed array param prototype");
    CHECK(parse_fails("#version 330\nvoid main() { float a[5][3]; }\n"),
          "arrays of arrays rejected in 330");
    CHECK(parse_fails("#version 330\nvoid main() { float[5] a[3]; }\n"),
          "type-prefix + postfix arrays rejected in 330");
}

static void test_int_mod_truncated(void)
{
    /* GLSL `%` on int is truncated remainder (C `%`): -7 % 3 == -1.
     * Floor-style mod would yield 2, making a+2 == 4. */
    const char *src =
        "#version 450 core\n"
        "const int a = -7 % 3;\n"
        "float x[a + 2];\n"
        "void main() {}\n";
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    CHECK(tu != NULL && tu->error == NULL, "int % truncated parse ok");
    int ok = 0;
    if (tu && tu->decl_count >= 2) {
        MGLDecl *x = tu->decls[1];
        ok = x && x->name && strcmp(x->name, "x") == 0 &&
             x->array_count == 1 && x->array_dims &&
             x->array_dims[0] == 1u;
    }
    CHECK(ok, "const int a = -7 % 3 folds to -1 (x[a+2] size 1)");
    if (tu)
        mglGLSLTranslationUnitDestroy(tu);
}

int main(void)
{
    printf("MGLGLSL parser skeleton tests\n");
    test_hello();
    test_fragment();
    test_expr_path();
    test_precision_statements();
    test_image_memory_qualifiers();
    test_declarator_lists_and_param_precision();
    test_error_report();
    test_gs_layout_errors();
    test_preprocessor();
    test_array_type_syntax();
    test_int_mod_truncated();
    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
