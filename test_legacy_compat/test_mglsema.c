/*
 * test_mglsema.c
 *
 * M0.5/M1 verification for the semantic analysis skeleton: symbol table
 * resolution, type resolution (MGLTypeSpec -> MGLIRType), expression type
 * checking with implicit conversions, and uniform/buffer block layout.
 *
 * Build (same pattern as test_mglir):
 *   cc -Wall -Wextra -O0 -g \
 *     -IMGL/include \
 *     test_legacy_compat/test_mglsema.c \
 *     MGL/src/mgl_glsl_sema.c MGL/src/mgl_glsl_parser.c \
 *     MGL/src/mgl_glsl_lexer.c MGL/src/mgl_ir.c \
 *     -o build/test_mglsema
 */
#include "mgl_glsl_parser.h"
#include "mgl_glsl_sema.h"
#include "mgl_ir.h"
#include "mgl_shader_abi.h" /* MGL_STAGE_* */

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

static MGLIRModule module;
static MGLSemaError *errors;
static uint32_t error_count;

static void analyze_ex(const char *src, int stage, MGLIRModule *mod,
                       MGLSemaError **es, uint32_t *ec)
{
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    if (!tu) {
        fprintf(stderr, "parse failed (tu NULL)\n");
        exit(2);
    }
    if (tu->error) {
        fprintf(stderr, "parse error: %s (line %u)\n", tu->error,
                tu->error_line);
        exit(2);
    }
    memset(mod, 0, sizeof(*mod));
    *es = NULL;
    *ec = 0;
    mglGLSLSemanticCheck(tu, stage, mod, es, ec);
    mglGLSLTranslationUnitDestroy(tu);
}

static void analyze(const char *src)
{
    analyze_ex(src, MGL_STAGE_VERTEX, &module, &errors, &error_count);
}

static int has_error(const char *needle)
{
    for (uint32_t i = 0; i < error_count; i++) {
        if (errors[i].message && strstr(errors[i].message, needle)) {
            return 1;
        }
    }
    return 0;
}

static void teardown(void)
{
    mglGLSLSemanticCheckDestroy(errors, error_count);
    errors = NULL;
    error_count = 0;
    mglIRModuleDestroy(&module);
}

static MGLIRSymbol *find_sym(const char *name)
{
    for (uint32_t i = 0; i < module.symbol_count; i++) {
        if (module.symbols[i]->name &&
            strcmp(module.symbols[i]->name, name) == 0) {
            return module.symbols[i];
        }
    }
    return NULL;
}

static void test_hello(void)
{
    analyze("#version 450 core\n"
            "layout(location = 0) in vec3 pos;\n"
            "layout(location = 0) out vec4 color;\n"
            "void main() {\n"
            "    color = vec4(pos, 1.0);\n"
            "}\n");
    CHECK(error_count == 0, "hello clean");
    MGLIRSymbol *pos = find_sym("pos");
    CHECK(pos != NULL, "pos in module");
    CHECK(pos && pos->type && pos->type->kind == MGLIR_TYPE_VECTOR &&
          pos->type->cols == 3 && pos->type->scalar == MGLIR_SCALAR_FLOAT,
          "pos vec3 float");
    MGLIRSymbol *mainf = find_sym("main");
    CHECK(mainf && mainf->is_function == 1, "main is a function");
    CHECK(mainf && mainf->return_type &&
          mainf->return_type->scalar == MGLIR_SCALAR_VOID,
          "main returns void");
    teardown();
}

static void test_tess_geom_m3(void)
{
    /* TCS: layout(vertices) + gl_in/gl_out/gl_TessLevel builtins */
    analyze_ex("#version 450 core\n"
               "layout(vertices = 3) out;\n"
               "void main() {\n"
               "    gl_out[gl_InvocationID].gl_Position =\n"
               "        gl_in[gl_InvocationID].gl_Position;\n"
               "    gl_TessLevelOuter[0] = 1.0;\n"
               "    gl_TessLevelInner[0] = 1.0;\n"
               "}\n",
               MGL_STAGE_TESS_CONTROL, &module, &errors, &error_count);
    CHECK(error_count == 0, "TCS layout/builtins parse+sema");
    teardown();

    /* TES: primitive mode layout + gl_TessCoord */
    analyze_ex("#version 450 core\n"
               "layout(quads, equal_spacing, cw) in;\n"
               "void main() {\n"
               "    gl_Position = vec4(gl_TessCoord, 1.0);\n"
               "}\n",
               MGL_STAGE_TESS_EVALUATION, &module, &errors, &error_count);
    CHECK(error_count == 0, "TES layout/builtins parse+sema");
    teardown();

    /* GS: input/output topologies + EmitVertex/EndPrimitive + gl_in[] */
    analyze_ex("#version 450 core\n"
               "layout(triangles) in;\n"
               "layout(triangle_strip, max_vertices = 3) out;\n"
               "void main() {\n"
               "    gl_Position = gl_in[0].gl_Position;\n"
               "    EmitVertex();\n"
               "    gl_Position = gl_in[1].gl_Position;\n"
               "    EmitVertex();\n"
               "    EndPrimitive();\n"
               "}\n",
               MGL_STAGE_GEOMETRY, &module, &errors, &error_count);
    CHECK(error_count == 0, "GS layout/builtins parse+sema");
    teardown();
}

static void test_swizzle_checks(void)
{
    analyze("#version 450 core\n"
            "void main() {\n"
            "    mat3 M3 = mat3(1.0);\n"
            "    mat3 I3 = mat3(1.0);\n"
            "    vec3 v = (M3 * inverse(M3) - I3) * vec3(1.0);\n"
            "    vec2 w = v.xy;\n"
            "    vec2 bad = vec2(1.0, 2.0).zw;\n"
            "    vec4 c = vec4(1.0);\n"
            "    vec2 bad2 = c.rx;\n"
            "    vec3 bad3 = M3.xy;\n"
            "}\n");
    CHECK(error_count == 3, "swizzle range/namespace/matrix errors");
    CHECK(has_error("invalid swizzle 'zw'"), "out-of-range swizzle");
    CHECK(has_error("invalid swizzle 'rx'"), "mixed namespace swizzle");
    teardown();
}

static void test_undeclared(void)
{
    analyze("#version 450 core\n"
            "void main() {\n"
            "    float x = missing + 1.0;\n"
            "}\n");
    CHECK(error_count == 1, "one error for undeclared");
    CHECK(has_error("undeclared identifier 'missing'"),
          "undeclared message");
    teardown();
}

static void test_type_mismatch(void)
{
    analyze("#version 450 core\n"
            "void main() {\n"
            "    float x = 1;\n"
            "    bool b = x;\n"
            "}\n");
    CHECK(error_count >= 1, "bool = float rejected");
    CHECK(has_error("type mismatch"), "assign error message");
    teardown();
}

static void test_implicit_conv(void)
{
    analyze("#version 450 core\n"
            "void main() {\n"
            "    float f = 1;        /* int -> float ok */\n"
            "    uint u = 2;\n"
            "    float g = u + 1.5;  /* uint+float ok */\n"
            "}\n");
    CHECK(error_count == 0, "implicit conversions accepted");
    teardown();
}

static void test_redecl(void)
{
    analyze("#version 450 core\n"
            "float a;\n"
            "float a;\n"
            "void main() {}\n");
    CHECK(error_count >= 1, "redeclaration rejected");
    CHECK(has_error("redeclaration"), "redecl message");
    teardown();
}

static void test_block_layout(void)
{
    analyze("#version 450 core\n"
            "layout(std140, binding = 0) uniform Block {\n"
            "    vec3 a;\n"
            "    float b;\n"
            "    mat3 m;\n"
            "} blk;\n"
            "void main() {}\n");
    CHECK(error_count == 0, "block clean");
    MGLIRSymbol *blk = find_sym("blk");
    CHECK(blk != NULL, "blk in module");
    CHECK(blk && blk->type && blk->type->kind == MGLIR_TYPE_STRUCT,
          "blk is struct");
    CHECK(blk && blk->type && blk->type->layout_valid,
          "block layout computed");
    /* std140: vec3 @0 (size12, align16), float @12, mat3 @16 (16-aligned)
     * with mat3 occupying 48 bytes (3 cols x stride 16). */
    CHECK(blk && blk->type && blk->type->member_offsets &&
          blk->type->member_offsets[0] == 0, "member a @0");
    CHECK(blk && blk->type && blk->type->member_offsets &&
          blk->type->member_offsets[1] == 12, "member b @12");
    CHECK(blk && blk->type && blk->type->member_offsets &&
          blk->type->member_offsets[2] == 16, "member m @16");
    teardown();
}

static void test_call_arg_check(void)
{
    analyze("#version 450 core\n"
            "float twice(float x) { return x * 2.0; }\n"
            "void main() {\n"
            "    float y = twice(1.0);\n"
            "}\n");
    CHECK(error_count == 0, "call arg ok");
    teardown();
}

static void test_integer_vector_types(void)
{
    analyze("#version 450 core\n"
            "layout(location = 0) in ivec3 iv;\n"
            "layout(location = 1) in uvec4 uv;\n"
            "layout(location = 2) in bvec2 bv;\n"
            "layout(location = 3) in dvec3 dv;\n"
            "uniform dmat2 dm;\n"
            "uniform dmat4x3 dm2;\n"
            "layout(location = 0) out ivec4 outv;\n"
            "void main() {\n"
            "    outv = ivec4(iv, 1);\n"
            "    float f = float(iv[0]);\n"
            "    bool b = bv[0];\n"
            "}\n");
    CHECK(error_count == 0, "integer/boolean/double types clean");
    MGLIRSymbol *iv = find_sym("iv");
    CHECK(iv && iv->type && iv->type->kind == MGLIR_TYPE_VECTOR &&
          iv->type->scalar == MGLIR_SCALAR_INT && iv->type->cols == 3,
          "ivec3 is int vec3");
    MGLIRSymbol *uv = find_sym("uv");
    CHECK(uv && uv->type && uv->type->kind == MGLIR_TYPE_VECTOR &&
          uv->type->scalar == MGLIR_SCALAR_UINT && uv->type->cols == 4,
          "uvec4 is uint vec4");
    MGLIRSymbol *bv = find_sym("bv");
    CHECK(bv && bv->type && bv->type->kind == MGLIR_TYPE_VECTOR &&
          bv->type->scalar == MGLIR_SCALAR_BOOL && bv->type->cols == 2,
          "bvec2 is bool vec2");
    MGLIRSymbol *dv = find_sym("dv");
    CHECK(dv && dv->type && dv->type->kind == MGLIR_TYPE_VECTOR &&
          dv->type->scalar == MGLIR_SCALAR_DOUBLE && dv->type->cols == 3,
          "dvec3 is double vec3");
    MGLIRSymbol *dm = find_sym("dm");
    CHECK(dm && dm->type && dm->type->kind == MGLIR_TYPE_MATRIX &&
          dm->type->scalar == MGLIR_SCALAR_DOUBLE &&
          dm->type->cols == 2 && dm->type->rows == 2,
          "dmat2 is double mat2");
    MGLIRSymbol *dm2 = find_sym("dm2");
    CHECK(dm2 && dm2->type && dm2->type->kind == MGLIR_TYPE_MATRIX &&
          dm2->type->scalar == MGLIR_SCALAR_DOUBLE &&
          dm2->type->cols == 4 && dm2->type->rows == 3,
          "dmat4x3 is double mat4x3");
    teardown();
}

static void test_matrix_arith(void)
{
    analyze("#version 450 core\n"
            "uniform mat4 m4;\n"
            "uniform mat3 m3;\n"
            "uniform mat2x3 m23;\n"
            "uniform mat3x2 m32;\n"
            "layout(location = 0) in vec4 p;\n"
            "layout(location = 0) out vec4 o;\n"
            "void main() {\n"
            "    vec4 a = m4 * p;\n"
            "    vec4 b = p * m4;\n"
            "    vec3 c = m3 * vec3(1.0);\n"
            "    vec3 e = m23 * vec2(1.0);\n"
            "    vec2 f = vec3(1.0) * m23;\n"
            "    mat2 g = m32 * m23;\n"
            "    mat4 h = m4 + m4;\n"
            "    mat4 k = m4 * 2.0;\n"
            "    o = a + b + c + vec4(e, 0.0);\n"
            "}\n");
    CHECK(error_count == 0, "matrix arithmetic clean");
    teardown();

    analyze("#version 450 core\n"
            "uniform mat3 m3;\n"
            "void main() {\n"
            "    vec4 bad = m3 * vec4(1.0);\n"
            "}\n");
    CHECK(error_count == 1, "mat*vec dimension mismatch rejected");
    CHECK(has_error("must be multiplied by a vector of length 3"),
          "dimension mismatch message");
    teardown();
}

static void test_interface_ok(void)
{
    /* Stage-local declarations never trip the link check by themselves:
     * "one side only" is legal. */
    analyze("#version 450 core\n"
            "layout(location = 0) in vec3 pos;\n"
            "layout(location = 0) out vec4 color;\n"
            "void main() { color = vec4(pos, 1.0); }\n");
    MGLIRModule vs = module;
    MGLSemaError *vs_err = errors;
    uint32_t vs_ec = error_count;
    errors = NULL;
    error_count = 0;
    memset(&module, 0, sizeof(module));

    analyze("#version 450 core\n"
            "layout(location = 0) in vec4 color;\n"
            "layout(location = 0) out vec4 frag;\n"
            "void main() { frag = color; }\n");
    MGLSemaError *le = NULL;
    uint32_t lec = 0;
    int r = mglGLSLInterfaceCheck(&vs, &module, &le, &lec);
    CHECK(r == 0 && lec == 0, "matching in/out passes");
    mglGLSLSemanticCheckDestroy(le, lec);

    /* Teardown must survive after the linked modules are gone. */
    mglIRModuleDestroy(&vs);
    mglGLSLSemanticCheckDestroy(vs_err, vs_ec);
    teardown();
}

static void test_interface_mismatch(void)
{
    analyze("#version 450 core\n"
            "layout(location = 0) out vec4 color;\n"
            "void main() { color = vec4(1.0); }\n");
    MGLIRModule vs = module;
    MGLSemaError *vs_err = errors;
    uint32_t vs_ec = error_count;
    errors = NULL;
    error_count = 0;
    memset(&module, 0, sizeof(module));

    analyze("#version 450 core\n"
            "layout(location = 0) in vec3 color;\n"
            "void main() { vec3 c = color; }\n");
    MGLSemaError *le = NULL;
    uint32_t lec = 0;
    int r = mglGLSLInterfaceCheck(&vs, &module, &le, &lec);
    CHECK(r == 1 && lec == 1, "mismatched in/out rejected");
    CHECK(lec == 1 && le[0].message &&
          strstr(le[0].message, "interface variable 'color'"),
          "mismatch message");
    mglGLSLSemanticCheckDestroy(le, lec);
    mglIRModuleDestroy(&vs);
    mglGLSLSemanticCheckDestroy(vs_err, vs_ec);
    teardown();
}

static void test_interface_blocks(void)
{
    analyze("#version 450 core\n"
            "layout(std140) uniform B { vec4 v; } b1;\n"
            "void main() {}\n");
    MGLIRModule vs = module;
    MGLSemaError *vs_err = errors;
    uint32_t vs_ec = error_count;
    errors = NULL;
    error_count = 0;
    memset(&module, 0, sizeof(module));

    /* Same block name, same members, different instance name: legal. */
    analyze("#version 450 core\n"
            "layout(std140) uniform B { vec4 v; } b2;\n"
            "void main() {}\n");
    MGLSemaError *le = NULL;
    uint32_t lec = 0;
    CHECK(mglGLSLInterfaceCheck(&vs, &module, &le, &lec) == 0 && lec == 0,
          "block same members ok");
    mglGLSLSemanticCheckDestroy(le, lec);
    mglIRModuleDestroy(&vs);
    mglGLSLSemanticCheckDestroy(vs_err, vs_ec);
    teardown();

    analyze("#version 450 core\n"
            "layout(std140) uniform B { vec4 v; } b1;\n"
            "void main() {}\n");
    vs = module;
    vs_err = errors;
    vs_ec = error_count;
    errors = NULL;
    error_count = 0;
    memset(&module, 0, sizeof(module));

    /* Different members: rejected. */
    analyze("#version 450 core\n"
            "layout(std140) uniform B { vec4 v; float f; } b2;\n"
            "void main() {}\n");
    le = NULL;
    lec = 0;
    CHECK(mglGLSLInterfaceCheck(&vs, &module, &le, &lec) == 1 && lec == 1,
          "block member mismatch rejected");
    CHECK(lec == 1 && le[0].message &&
          strstr(le[0].message, "interface block 'B'"),
          "block mismatch message");
    mglGLSLSemanticCheckDestroy(le, lec);
    mglIRModuleDestroy(&vs);
    mglGLSLSemanticCheckDestroy(vs_err, vs_ec);
    teardown();
}

static void test_builtins(void)
{
    analyze("#version 450 core\n"
            "uniform sampler2D uTex;\n"
            "layout(location = 0) in vec2 uv;\n"
            "layout(location = 0) in vec3 pos;\n"
            "layout(location = 0) out vec4 o;\n"
            "void main() {\n"
            "    vec4 t = texture(uTex, uv);\n"
            "    vec4 tl = textureLod(uTex, uv, 0.0);\n"
            "    vec2 ts = textureSize(uTex, 0);\n"
            "    vec3 n = normalize(pos);\n"
            "    float d = dot(pos, normalize(pos));\n"
            "    float len = length(pos);\n"
            "    float dist = distance(pos, vec3(0.0));\n"
            "    vec3 c = clamp(pos, vec3(0.0), vec3(1.0));\n"
            "    vec3 c2 = clamp(pos, 0.0, 1.0);\n"
            "    vec3 m = mix(pos, vec3(1.0), c);\n"
            "    vec3 m2 = mix(pos, vec3(1.0), 0.5);\n"
            "    vec3 a = abs(pos);\n"
            "    float af = abs(-1.0);\n"
            "    o = t + tl + vec4(n * d + len + dist, 1.0) +\n"
            "        c + c2 + m + m2 + a + vec4(ts, af, 0.0);\n"
            "}\n");
    CHECK(error_count == 0, "builtin calls clean");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    float d = dot(vec3(1.0), vec4(1.0));\n"
            "}\n");
    CHECK(error_count == 1, "dot dimension mismatch rejected");
    CHECK(has_error("no matching overload of builtin 'dot'"),
          "dot mismatch message");
    teardown();

    analyze("#version 450 core\n"
            "uniform mat3 m3;\n"
            "void main() {\n"
            "    vec3 n = normalize(m3);\n"
            "}\n");
    CHECK(error_count == 1, "normalize(mat) rejected");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    vec4 t = texture(vec2(1.0), vec2(0.5));\n"
            "}\n");
    CHECK(error_count == 1, "texture with non-sampler rejected");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    vec3 c = clamp(vec3(1.0), vec2(0.0), vec2(1.0));\n"
            "}\n");
    CHECK(error_count == 1, "clamp gen dimension conflict rejected");
    teardown();
}

static void test_image2d_array_store(void)
{
    analyze_ex("#version 440 core\n"
               "layout(rgba32i, binding = 0) writeonly uniform highp "
               "iimage2DArray array_image;\n"
               "void main() {\n"
               "    imageStore(array_image, ivec3(1, 2, 3), "
               "ivec4(0, 255, 0, 0));\n"
               "}\n",
               MGL_STAGE_FRAGMENT, &module, &errors, &error_count);
    CHECK(error_count == 0,
          "integer image2DArray imageStore accepted");
    MGLIRSymbol *image = find_sym("array_image");
    CHECK(image && image->type && image->type->kind == MGLIR_TYPE_IMAGE,
          "array image lowered as storage image");
    CHECK(image && image->type &&
          image->type->tex_kind == MGLIR_TEX_2D_ARRAY &&
          image->type->tex_storage == MGLIR_SCALAR_INT,
          "array image preserves 2D-array signed storage type");
    teardown();
}

static void test_constructors(void)
{
    analyze("#version 450 core\n"
            "uniform vec3 v3;\n"
            "uniform ivec3 iv3;\n"
            "void main() {\n"
            "    vec2 a = vec2(1.0);\n"
            "    vec3 b = vec3(1.0, vec2(2.0));\n"
            "    vec4 c = vec4(v3, 1.0);\n"
            "    vec3 d = vec3(iv3);\n"
            "    float e = float(1);\n"
            "    int f = int(1.5);\n"
            "    mat2 h = mat2(1.0);\n"
            "    mat3 i = mat3(vec3(1.0), vec3(2.0), vec3(3.0));\n"
            "    mat3 j = mat3(v3, v3, v3);\n"
            "    bool k = bool(1);\n"
            "    vec4 l = vec4(1.0, vec2(2.0), 3.0);\n"
            "    vec3 n = vec3(a, 1.0);\n"
            "}\n");
    CHECK(error_count == 0, "valid constructors clean");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    vec3 a = vec3(vec2(1.0));\n"
            "}\n");
    CHECK(error_count == 1, "vec3(vec2) rejected");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    vec3 a = vec3(1.0, 2.0);\n"
            "}\n");
    CHECK(error_count == 1, "vec3 from 2 components rejected");
    CHECK(has_error("expected 3"), "component count message");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    vec2 a = vec2(mat2(1.0));\n"
            "}\n");
    CHECK(error_count == 1, "vec2(mat2) rejected");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    mat3 a = mat3(vec3(1.0), vec3(2.0));\n"
            "}\n");
    CHECK(error_count == 1, "mat3 from 2 columns rejected");
    CHECK(has_error("expects 3 column vector(s)"), "column count message");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    mat3 a = mat3(vec2(1.0), vec2(2.0), vec2(3.0));\n"
            "}\n");
    CHECK(error_count == 1, "mat3 from vec2 columns rejected");
    CHECK(has_error("must be a vec3"), "column dimension message");
    teardown();
}

int main(void)
{
    printf("MGLGLSL sema skeleton tests\n");
    test_hello();
    test_tess_geom_m3();
    test_swizzle_checks();
    test_undeclared();
    test_type_mismatch();
    test_implicit_conv();
    test_redecl();
    test_block_layout();
    test_call_arg_check();
    test_integer_vector_types();
    test_matrix_arith();
    test_builtins();
    test_image2d_array_store();
    test_constructors();
    test_interface_ok();
    test_interface_mismatch();
    test_interface_blocks();
    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}