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
#include "mgl_glsl_cpp.h"
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

    analyze_ex("#version 410\n"
               "layout (points) in;\n"
               "layout (points, max_vertices = 4) out;\n"
               "in gl_PerVertex { float gl_ClipDistance[]; } gl_in[];\n"
               "out gl_PerVertex { float gl_ClipDistance[]; };\n"
               "void main() {\n"
               "    gl_ClipDistance[0] = 0.5;\n"
               "    EmitVertex();\n"
               "}\n",
               MGL_STAGE_GEOMETRY, &module, &errors, &error_count);
    CHECK(error_count == 0, "GS unsized gl_ClipDistance PerVertex redecl");
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

static void test_flattened_ssbo_member_layout(void)
{
    analyze("#version 430 core\n"
            "layout(std430, binding = 0) buffer Out { int data[8]; };\n"
            "void main() {}\n");
    CHECK(error_count == 0, "flattened SSBO clean");
    MGLIRSymbol *data = find_sym("data");
    CHECK(data != NULL && data->block_name &&
              strcmp(data->block_name, "Out") == 0,
          "flattened SSBO member published");
    CHECK(data && data->type && data->type->kind == MGLIR_TYPE_ARRAY &&
              data->type->layout_valid &&
              data->type->layout_standard == MGLIR_LAYOUT_STD430 &&
              data->type->layout.array_stride == sizeof(int),
          "flattened SSBO member retains std430 array stride");
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
            "uniform bvec2 bv;\n"
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
    int r = mglGLSLInterfaceCheck(&vs, &module, 0, &le, &lec);
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
    int r = mglGLSLInterfaceCheck(&vs, &module, 0, &le, &lec);
    CHECK(r == 1 && lec == 1, "mismatched in/out rejected");
    CHECK(lec == 1 && le[0].message &&
          strstr(le[0].message, "interface variable 'color'"),
          "mismatch message");
    mglGLSLSemanticCheckDestroy(le, lec);
    mglIRModuleDestroy(&vs);
    mglGLSLSemanticCheckDestroy(vs_err, vs_ec);
    teardown();
}

static void test_interface_abi_qualifiers(void)
{
    analyze("#version 450 core\n"
            "layout(location = 1) out vec4 color;\n"
            "void main() { color = vec4(1.0); }\n");
    MGLIRModule vs = module;
    MGLSemaError *vs_err = errors;
    uint32_t vs_ec = error_count;
    errors = NULL;
    error_count = 0;
    memset(&module, 0, sizeof(module));

    analyze("#version 450 core\n"
            "layout(location = 2) in vec4 color;\n"
            "void main() { vec4 c = color; }\n");
    MGLSemaError *le = NULL;
    uint32_t lec = 0;
    CHECK(mglGLSLInterfaceCheck(&vs, &module, 0, &le, &lec) == 1 && lec == 1,
          "explicit interface location mismatch rejected");
    mglGLSLSemanticCheckDestroy(le, lec);
    mglIRModuleDestroy(&vs);
    mglGLSLSemanticCheckDestroy(vs_err, vs_ec);
    teardown();

    analyze("#version 450 core\n"
            "layout(location = 0) flat out vec4 color;\n"
            "void main() { color = vec4(1.0); }\n");
    vs = module;
    vs_err = errors;
    vs_ec = error_count;
    errors = NULL;
    error_count = 0;
    memset(&module, 0, sizeof(module));
    analyze("#version 450 core\n"
            "layout(location = 0) in vec4 color;\n"
            "void main() { vec4 c = color; }\n");
    le = NULL;
    lec = 0;
    CHECK(mglGLSLInterfaceCheck(&vs, &module, 0, &le, &lec) == 1 && lec == 1,
          "interpolation qualifier mismatch rejected");
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
    CHECK(mglGLSLInterfaceCheck(&vs, &module, 0, &le, &lec) == 0 && lec == 0,
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
    CHECK(mglGLSLInterfaceCheck(&vs, &module, 0, &le, &lec) == 1 && lec == 1,
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
            "uniform sampler2DShadow uShadow;\n"
            "layout(location = 0) in vec2 uv;\n"
            "layout(location = 0) in vec3 pos;\n"
            "layout(location = 0) out vec4 o;\n"
            "void main() {\n"
            "    vec4 t = texture(uTex, uv);\n"
            "    vec4 tl = textureLod(uTex, uv, 0.0);\n"
            "    vec4 tg = textureGather(uTex, uv);\n"
            "    vec4 tg1 = textureGather(uTex, uv, 1);\n"
            "    vec4 tgo = textureGatherOffset(uTex, uv, ivec2(1, 0));\n"
            "    ivec2 offs[4] = ivec2[4](ivec2(0), ivec2(1,0), ivec2(0,1), ivec2(1));\n"
            "    vec4 tgos = textureGatherOffsets(uTex, uv, offs);\n"
            "    vec4 tgs = textureGather(uShadow, uv, 0.5);\n"
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
            "    bool lte = all(lessThanEqual(pos, vec3(1.0)));\n"
            "    o = t + tl + tg + tg1 + tgo + tgos + tgs +\n"
            "        vec4(n * d + len + dist, 1.0) +\n"
            "        c + c2 + m + m2 + a + vec4(ts, af, 0.0);\n"
            "    if (lte) o.w = 1.0;\n"
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

    analyze("#version 450 core\n"
            "layout(rgba32f) uniform image2D i2;\n"
            "layout(rgba32f) uniform image3D i3;\n"
            "layout(rgba32f) uniform image2DArray ia;\n"
            "layout(rgba32f) uniform image1D i1;\n"
            "void main() {\n"
            "    ivec2 a = imageSize(i2);\n"
            "    ivec3 b = imageSize(i3);\n"
            "    ivec3 c = imageSize(ia);\n"
            "    int d = imageSize(i1);\n"
            "}\n");
    CHECK(error_count == 0, "imageSize return types by dimension");
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
            "    float e2 = float(v3);\n"
            "    int f2 = int(iv3);\n"
            "    float e3 = float(mat2(1.0));\n"
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
            "    vec3 b = vec3(vec4(1.0, 2.0, 3.0, 4.0));\n"
            "}\n");
    CHECK(error_count == 0, "matrix/extra-component vector ctors clean");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    vec2 a = vec2(mat2(1.0), 1.0);\n"
            "}\n");
    CHECK(error_count == 1, "vec2 extra unused arg rejected");
    CHECK(has_error("unused extra arguments"), "unused arg message");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    mat3 a = mat3(vec3(1.0), vec3(2.0));\n"
            "}\n");
    CHECK(error_count == 1, "mat3 from 2 columns rejected");
    CHECK(has_error("expected 9"), "component count message");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    mat3 a = mat3(vec2(1.0), vec2(2.0), vec2(3.0));\n"
            "}\n");
    CHECK(error_count == 1, "mat3 from 6 components rejected");
    CHECK(has_error("expected 9"), "short matrix ctor message");
    teardown();

    analyze("#version 450 core\n"
            "void main() {\n"
            "    mat2x3 a = mat2x3(vec2(1.0), vec2(2.0), vec2(3.0));\n"
            "    mat3x2 b = mat3x2(vec3(1.0), vec3(2.0));\n"
            "}\n");
    CHECK(error_count == 0, "non-square matrix component ctors clean");
    teardown();
}

static void test_invalid_vs_inputs(void)
{
    static const char *bad[] = {
        "in bool data;\nvoid main() {}\n",
        "in bvec2 data;\nvoid main() {}\n",
        "in bvec3 data;\nvoid main() {}\n",
        "in bvec4 data;\nvoid main() {}\n",
        "in sampler2D data;\nvoid main() {}\n",
        "in struct { vec4 test; } data;\nvoid main() {}\n",
        "centroid in vec4 data;\nvoid main() {}\n",
        "patch in vec4 data;\nvoid main() {}\n",
        "sample in vec4 data;\nvoid main() {}\n",
    };
    for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        char src[256];
        snprintf(src, sizeof(src), "#version 140\n%s", bad[i]);
        analyze(src);
        CHECK(error_count > 0, bad[i]);
        teardown();
    }
    analyze("#version 140\nin vec4 data;\nvoid main() {}\n");
    CHECK(error_count == 0, "legal VS in vec4");
    teardown();
}

static void test_reserved_idents(void)
{
    analyze("#version 460 core\n"
            "const vec4 common = vec4(1.0);\n"
            "void main() {}\n");
    CHECK(error_count > 0, "reserved ident common");
    teardown();

    analyze("#version 460 core\n"
            "void float(in vec4 t) {}\n"
            "void main() {}\n");
    CHECK(error_count > 0, "keyword as function name");
    teardown();

    analyze("#version 460 core\n"
            "in vec4 ok;\n"
            "void main() { gl_Position = ok; }\n");
    CHECK(error_count == 0, "legal ident still accepted");
    teardown();

    analyze_ex("#version 430 core\n"
               "layout(local_size_x = 1) in;\n"
               "buffer float {\n"
               "    vec4 test;\n"
               "};\n"
               "void main() {}\n",
               MGL_STAGE_COMPUTE, &module, &errors, &error_count);
    CHECK(error_count > 0, "keyword as SSBO block name");
    teardown();

    analyze("#version 460 core\n"
            "void main() {\n"
            "    struct attribute { vec4 test; };\n"
            "}\n");
    CHECK(error_count > 0, "keyword as struct name");
    teardown();

    analyze_ex("#version 430 core\n"
               "layout(local_size_x = 1) in;\n"
               "buffer atomic_uint {\n"
               "    vec4 test;\n"
               "};\n"
               "void main() {}\n",
               MGL_STAGE_COMPUTE, &module, &errors, &error_count);
    CHECK(error_count > 0, "atomic_uint as SSBO block name");
    teardown();
}

static void test_compute_compile_errors(void)
{
    printf("test_compute_compile_errors\n");

    analyze_ex("#version 420 core\n"
               "layout(local_size_x = 1) in;\n"
               "layout(std430) buffer Output { uint g_output[]; };\n"
               "void main() { g_output[gl_GlobalInvocationID.x] = 0; }\n",
               MGL_STAGE_COMPUTE, &module, &errors, &error_count);
    CHECK(error_count > 0, "compute builtin before version 430");
    teardown();

    {
        const char *src =
            "#version 430 core\n"
            "layout(local_size_x = 1) in;\n"
            "layout(local_size_x = 2) in;\n"
            "void main() {}\n";
        MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
        CHECK(tu && tu->error, "conflicting local_size_x");
        if (tu)
            mglGLSLTranslationUnitDestroy(tu);
    }

    analyze_ex("#version 430 core\n"
               "layout(local_size_x = 1) in;\n"
               "in uint x;\n"
               "layout(std430) buffer Output { uint g_output[]; };\n"
               "void main() { g_output[gl_GlobalInvocationID.x] = x; }\n",
               MGL_STAGE_COMPUTE, &module, &errors, &error_count);
    CHECK(error_count > 0, "user in on compute shader");
    teardown();

    analyze_ex("#version 430 core\n"
               "layout(local_size_x = 1) in;\n"
               "out uint x;\n"
               "layout(std430) buffer Output { uint g_output[]; };\n"
               "void main() { g_output[gl_GlobalInvocationID.x] = 0; x = 0; }\n",
               MGL_STAGE_COMPUTE, &module, &errors, &error_count);
    CHECK(error_count > 0, "user out on compute shader");
    teardown();

    analyze_ex("#version 430 core\n"
               "layout(local_size_x = 1025) in;\n"
               "void main() {}\n",
               MGL_STAGE_COMPUTE, &module, &errors, &error_count);
    CHECK(error_count > 0, "local_size_x exceeds max");
    teardown();

    analyze_ex("#version 430 core\n"
               "layout(local_size_x = 1) in;\n"
               "layout(std430) buffer Output { uint g_output[]; };\n"
               "void main() { g_output[gl_GlobalInvocationID.x] = 0; }\n",
               MGL_STAGE_COMPUTE, &module, &errors, &error_count);
    CHECK(error_count == 0, "legal compute still accepted");
    teardown();

    analyze_ex("#version 430 core\n"
               "layout(local_size_x = 2, local_size_y = 3, local_size_z = 4) in;\n"
               "layout(std430, binding = 0) buffer Output {\n"
               "  uint g_buffer[22 + gl_WorkGroupSize.x];\n"
               "};\n"
               "shared uint g_shared[gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z];\n"
               "void main() { g_buffer[0] = g_shared[0]; }\n",
               MGL_STAGE_COMPUTE, &module, &errors, &error_count);
    CHECK(error_count == 0, "gl_WorkGroupSize is a constant expression");
    teardown();
}

/* GLSL 4.60 §4.3.4: a member of an anonymous input block is a read-only
 * global input, like a plain `in`; an `in` parameter stays writable. */
static void test_input_block_member_readonly(void)
{
    analyze_ex("#version 400\n"
               "in inputBlock { vec4 data; };\n"
               "out vec4 result;\n"
               "void main() { data += vec4(1.0); result = data; }\n",
               MGL_STAGE_FRAGMENT, &module, &errors, &error_count);
    CHECK(has_error("cannot write to in variable 'data'"),
          "write to anonymous input block member rejected");
    teardown();
    analyze_ex("#version 400\n"
               "in inputBlock { vec4 data; };\n"
               "out vec4 result;\n"
               "void f(inout vec4 v) { v += vec4(1.0); }\n"
               "void main() { f(data); result = data; }\n",
               MGL_STAGE_FRAGMENT, &module, &errors, &error_count);
    CHECK(has_error("cannot pass 'data' to an out/inout parameter"),
          "anonymous input block member rejected as inout argument");
    teardown();
    analyze_ex("#version 400\n"
               "in inputBlock { vec4 data; };\n"
               "out vec4 result;\n"
               "vec4 f(in vec4 v) { v += vec4(1.0); return v; }\n"
               "void main() { result = f(data); }\n",
               MGL_STAGE_FRAGMENT, &module, &errors, &error_count);
    CHECK(error_count == 0, "in parameter stays writable");
    teardown();
}

/* `uniform struct S { ... } s;` declares a struct uniform in the default
 * block, exactly like `struct S { ... }; uniform S s;`; only
 * `uniform Block { ... }` is an interface block. */
static void test_inline_struct_uniform(void)
{
    analyze("#version 450 core\n"
            "uniform struct S { float a; vec2 b; } s;\n"
            "uniform struct N { struct I { float i; } o; } n[2];\n"
            "struct T { float a; };\n"
            "uniform T t;\n"
            "layout(std140) uniform Block { float c; } blk;\n"
            "void main() { gl_Position = vec4(s.a + n[1].o.i + t.a + blk.c); }\n");
    CHECK(error_count == 0, "inline struct uniforms clean");
    MGLIRSymbol *si = find_sym("s");
    MGLIRSymbol *ni = find_sym("n");
    MGLIRSymbol *ti = find_sym("t");
    MGLIRSymbol *bi = find_sym("blk");
    CHECK(si && !si->is_interface_block, "inline struct uniform is not a block");
    CHECK(ni && !ni->is_interface_block,
          "nested inline struct uniform array is not a block");
    CHECK(ti && !ti->is_interface_block, "named struct uniform is not a block");
    CHECK(bi && bi->is_interface_block, "uniform block is a block");
    teardown();
}

/* GLSL 4.40 §4.4: integer layout values are integral constant
 * expressions; earlier versions take a literal only. */
static void test_layout_constant_expression(void)
{
    const char *src440 = "#version 440 core\n"
                         "layout(local_size_x = (4), local_size_y = 2 * 3) in;\n"
                         "void main() {}\n";
    MGLTranslationUnit *tu = mglGLSLParse(src440, strlen(src440));
    CHECK(tu && !tu->error && tu->layout_local_size_x == 4 &&
              tu->layout_local_size_y == 6,
          "layout(local_size_x = (4)) parses at 440");
    if (tu) mglGLSLTranslationUnitDestroy(tu);
    const char *src430 = "#version 430 core\n"
                         "layout(local_size_x = (4)) in;\n"
                         "void main() {}\n";
    tu = mglGLSLParse(src430, strlen(src430));
    CHECK(tu && tu->error, "layout(local_size_x = (4)) rejected at 430");
    if (tu) mglGLSLTranslationUnitDestroy(tu);
    const char *neg = "#version 450 core\n"
                      "layout(location = (0 - 1)) in vec4 v;\n"
                      "void main() {}\n";
    tu = mglGLSLParse(neg, strlen(neg));
    CHECK(tu && tu->error, "negative layout(location) expression rejected");
    if (tu) mglGLSLTranslationUnitDestroy(tu);
}

/* GLSL 4.60 §3.3: a supported extension's macro is defined without an
 * #extension directive (SPIRV-Cross output probes it with #if defined). */
static void test_extension_macros_predefined(void)
{
    const char *src = "#version 450\n"
                      "#if !defined(GL_ARB_gpu_shader_int64) || "
                      "GL_ARB_gpu_shader_int64 != 1\n"
                      "#error missing\n"
                      "#endif\n"
                      "#ifdef GL_ARB_gpu_shader_fp64\n"
                      "#error unadvertised extension defined\n"
                      "#endif\n"
                      "void main() {}\n";
    char err[256] = {0};
    char *out = mglGLSLPreprocess(src, strlen(src), err, sizeof err);
    CHECK(out != NULL, "advertised extension macro predefined");
    free(out);
}

/* GLSL 4.60 §8.9.2: texelFetch / texelFetchOffset exist for every
 * non-shadow gsampler except cube and cube-array; rect and buffer take no
 * lod, and the rect offset form is (P, offset). */
static void test_texel_fetch_overloads(void)
{
    analyze_ex("#version 450 core\n"
               "uniform sampler1D s1; uniform isampler2D s2;\n"
               "uniform usampler3D s3; uniform sampler2DRect sr;\n"
               "uniform isampler1DArray s1a; uniform usampler2DArray s2a;\n"
               "uniform samplerBuffer sb; uniform isamplerBuffer isb;\n"
               "uniform sampler2DMS sms; uniform usampler2DMSArray smsa;\n"
               "uniform usampler2DRect usr;\n"
               "out vec4 f;\n"
               "void main() {\n"
               "    vec4 a = texelFetch(s1, 1, 0);\n"
               "    ivec4 b = texelFetch(s2, ivec2(1), 0);\n"
               "    uvec4 c = texelFetch(s3, ivec3(1), 0);\n"
               "    vec4 d = texelFetch(sr, ivec2(1));\n"
               "    ivec4 e = texelFetch(s1a, ivec2(1), 0);\n"
               "    uvec4 g = texelFetch(s2a, ivec3(1), 0);\n"
               "    vec4 h = texelFetch(sb, 3);\n"
               "    ivec4 i = texelFetch(isb, 3);\n"
               "    vec4 j = texelFetch(sms, ivec2(1), 2);\n"
               "    uvec4 k = texelFetch(smsa, ivec3(1), 2);\n"
               "    vec4 l = texelFetchOffset(s1, 1, 0, 1);\n"
               "    ivec4 m = texelFetchOffset(s2, ivec2(1), 0, ivec2(1));\n"
               "    uvec4 n = texelFetchOffset(s3, ivec3(1), 0, ivec3(1));\n"
               "    vec4 o = texelFetchOffset(sr, ivec2(1), ivec2(1));\n"
               "    ivec4 p = texelFetchOffset(s1a, ivec2(1), 0, 1);\n"
               "    uvec4 q = texelFetchOffset(s2a, ivec3(1), 0, ivec2(1));\n"
               "    uvec4 r = texelFetchOffset(usr, ivec2(1), ivec2(-1));\n"
               "    f = a + vec4(b) + vec4(c) + d + vec4(e) + vec4(g) + h +\n"
               "        vec4(i) + j + vec4(k) + l + vec4(m) + vec4(n) + o +\n"
               "        vec4(p) + vec4(q) + vec4(r);\n"
               "}\n",
               MGL_STAGE_FRAGMENT, &module, &errors, &error_count);
    CHECK(error_count == 0, "texelFetch/texelFetchOffset legal overloads clean");
    teardown();

    static const struct {
        const char *decl;
        const char *call;
        const char *label;
    } bad[] = {
        { "samplerCube", "texelFetch(t, ivec3(1, 1, 0), 0)",
          "texelFetch(samplerCube) rejected" },
        { "isamplerCube", "vec4(texelFetch(t, ivec3(0), 0))",
          "texelFetch(isamplerCube) rejected" },
        { "samplerCubeArray", "texelFetch(t, ivec4(0), 0)",
          "texelFetch(samplerCubeArray) rejected" },
        { "samplerCubeShadow", "vec4(texelFetch(t, ivec3(0), 0))",
          "texelFetch(samplerCubeShadow) rejected" },
        { "samplerCubeArrayShadow", "vec4(texelFetch(t, ivec4(0), 0))",
          "texelFetch(samplerCubeArrayShadow) rejected" },
        { "sampler1DShadow", "vec4(texelFetch(t, 0, 0))",
          "texelFetch(sampler1DShadow) rejected" },
        { "sampler2DShadow", "vec4(texelFetch(t, ivec2(0), 0))",
          "texelFetch(sampler2DShadow) rejected" },
        { "sampler2DRectShadow", "vec4(texelFetch(t, ivec2(0)))",
          "texelFetch(sampler2DRectShadow) rejected" },
        { "sampler1DArrayShadow", "vec4(texelFetch(t, ivec2(0), 0))",
          "texelFetch(sampler1DArrayShadow) rejected" },
        { "sampler2DArrayShadow", "vec4(texelFetch(t, ivec3(0), 0))",
          "texelFetch(sampler2DArrayShadow) rejected" },
        { "samplerBuffer", "texelFetch(t, 0, 0)",
          "texelFetch(samplerBuffer) with lod rejected" },
        { "sampler2DRect", "texelFetch(t, ivec2(0), 0)",
          "texelFetch(sampler2DRect) with lod rejected" },
        { "sampler2D", "texelFetch(t, ivec2(0))",
          "texelFetch(sampler2D) without lod rejected" },
        { "sampler2D", "texelFetch(t, ivec3(0), 0)",
          "texelFetch(sampler2D, ivec3) rejected" },
        { "sampler2D", "texelFetch(t, 0, 0)",
          "texelFetch(sampler2D, int) rejected" },
        { "sampler2D", "texelFetch(t, vec2(0.0), 0)",
          "texelFetch(sampler2D, vec2) rejected" },
        { "sampler2D", "texelFetch(t, uvec2(0u), 0)",
          "texelFetch(sampler2D, uvec2) rejected" },
        { "sampler1D", "texelFetch(t, ivec2(0), 0)",
          "texelFetch(sampler1D, ivec2) rejected" },
        { "sampler2DMS", "texelFetch(t, ivec3(0), 0)",
          "texelFetch(sampler2DMS, ivec3) rejected" },
        { "sampler2DRect", "texelFetch(t, 0)",
          "texelFetch(sampler2DRect, int) rejected" },
        { "samplerCube", "texelFetchOffset(t, ivec3(0), 0, ivec3(0))",
          "texelFetchOffset(samplerCube) rejected" },
        { "sampler2DMS", "texelFetchOffset(t, ivec2(0), 0, ivec2(0))",
          "texelFetchOffset(sampler2DMS) rejected" },
        { "samplerBuffer", "texelFetchOffset(t, 0, 0)",
          "texelFetchOffset(samplerBuffer) rejected" },
        { "sampler2DShadow",
          "vec4(texelFetchOffset(t, ivec2(0), 0, ivec2(0)))",
          "texelFetchOffset(sampler2DShadow) rejected" },
        { "sampler2DRect", "texelFetchOffset(t, ivec2(0), 0, ivec2(0))",
          "texelFetchOffset(sampler2DRect) with lod rejected" },
        { "sampler2D", "texelFetchOffset(t, ivec2(0), 0, ivec3(0))",
          "texelFetchOffset(sampler2D, ivec3 offset) rejected" },
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        char src[512];
        snprintf(src, sizeof src,
                 "#version 400 core\n"
                 "uniform %s t;\n"
                 "out vec4 f;\n"
                 "void main() { f = vec4(%s); }\n",
                 bad[i].decl, bad[i].call);
        analyze_ex(src, MGL_STAGE_FRAGMENT, &module, &errors, &error_count);
        CHECK(error_count >= 1 &&
                  (has_error("no matching overload of builtin 'texelFetch'") ||
                   has_error("no matching overload of builtin "
                             "'texelFetchOffset'")),
              bad[i].label);
        teardown();
    }
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
    test_flattened_ssbo_member_layout();
    test_call_arg_check();
    test_integer_vector_types();
    test_matrix_arith();
    test_builtins();
    test_texel_fetch_overloads();
    test_image2d_array_store();
    test_constructors();
    test_invalid_vs_inputs();
    test_reserved_idents();
    test_compute_compile_errors();
    test_interface_ok();
    test_interface_mismatch();
    test_interface_abi_qualifiers();
    test_interface_blocks();
    test_input_block_member_readonly();
    test_inline_struct_uniform();
    test_layout_constant_expression();
    test_extension_macros_predefined();
    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
