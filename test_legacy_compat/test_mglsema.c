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

static void analyze(const char *src)
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
    memset(&module, 0, sizeof(module));
    errors = NULL;
    error_count = 0;
    mglGLSLSemanticCheck(tu, &module, &errors, &error_count);
    mglGLSLTranslationUnitDestroy(tu);
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

int main(void)
{
    printf("MGLGLSL sema skeleton tests\n");
    test_hello();
    test_undeclared();
    test_type_mismatch();
    test_implicit_conv();
    test_redecl();
    test_block_layout();
    test_call_arg_check();
    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}