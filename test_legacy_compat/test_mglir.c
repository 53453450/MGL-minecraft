/*
 * test_mglir.c
 *
 * M0 verification for mgl_ir type system + std140/std430 layout.
 * Compiles directly with MGL/src/mgl_ir.c (no dylib dependency).
 *
 * Build:
 *   cc -Wall -Wextra -O0 -g \
 *     -IMGL/include \
 *     test_legacy_compat/test_mglir.c MGL/src/mgl_ir.c \
 *     -o build/test_mglir
 */
#include "mgl_ir.h"

#include <stdio.h>
#include <stdlib.h>

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

#define EXPECT_SIZE(t, layout, want, label)                            \
    do {                                                               \
        uint32_t sz = 0;                                               \
        int rc = mglIRComputeLayout((t), (layout), &sz);               \
        CHECK(rc == 0 && sz == (uint32_t)(want), label);               \
    } while (0)

static void test_std140_scalars(void)
{
    printf("std140 scalars\n");
    MGLIRType *f = mglIRTypeScalar(MGLIR_SCALAR_FLOAT);
    EXPECT_SIZE(f, MGLIR_LAYOUT_STD140, 4, "float size=4");
    mglIRTypeDestroy(f);

    MGLIRType *v2 = mglIRTypeVector(MGLIR_SCALAR_FLOAT, 2);
    EXPECT_SIZE(v2, MGLIR_LAYOUT_STD140, 8, "vec2 size=8");
    mglIRTypeDestroy(v2);

    MGLIRType *v3 = mglIRTypeVector(MGLIR_SCALAR_FLOAT, 3);
    EXPECT_SIZE(v3, MGLIR_LAYOUT_STD140, 12, "vec3 size=12");
    mglIRTypeDestroy(v3);

    MGLIRType *v4 = mglIRTypeVector(MGLIR_SCALAR_FLOAT, 4);
    EXPECT_SIZE(v4, MGLIR_LAYOUT_STD140, 16, "vec4 size=16");
    mglIRTypeDestroy(v4);
}

static void test_std140_matrices(void)
{
    printf("std140 matrices (GLSL 4.60 example values)\n");
    /* float m4; col-major array of 4 vec4 -> 64 bytes, align 16. */
    MGLIRType *m4 = mglIRTypeMatrix(MGLIR_SCALAR_FLOAT, 4, 4);
    EXPECT_SIZE(m4, MGLIR_LAYOUT_STD140, 64, "mat4 std140 size=64");
    mglIRTypeDestroy(m4);

    /* mat3 col-major -> 3 columns of vec3, stride 16 -> 48. */
    MGLIRType *m3 = mglIRTypeMatrix(MGLIR_SCALAR_FLOAT, 3, 3);
    {
        uint32_t sz = 0;
        int rc = mglIRComputeLayout(m3, MGLIR_LAYOUT_STD140, &sz);
        CHECK(rc == 0 && sz == 48, "mat3 std140 size=48");
    }
    mglIRTypeDestroy(m3);
}

static void test_std140_array(void)
{
    printf("std140 arrays\n");
    /* float[3] std140: glslang ArrayStride 16, size = 3*16 = 48. */
    MGLIRType *fa = mglIRTypeArray(mglIRTypeScalar(MGLIR_SCALAR_FLOAT), 3);
    EXPECT_SIZE(fa, MGLIR_LAYOUT_STD140, 48, "float[3] std140 size=48");
    mglIRTypeDestroy(fa);

    /* vec3[2] std140: glslang: m140@0 / v4[2]@32 / x@64 -> 2*16 = 32. */
    MGLIRType *va = mglIRTypeArray(mglIRTypeVector(MGLIR_SCALAR_FLOAT, 3), 2);
    EXPECT_SIZE(va, MGLIR_LAYOUT_STD140, 32, "vec3[2] std140 size=32");
    mglIRTypeDestroy(va);
}

static void test_std140_struct_literal(void)
{
    /* glslang authoritative layout (spirv-dis verified):
     * struct { vec3 a; float b; } std140: a@0, b@12, size 16. */
    MGLIRType *mem[] = {
        mglIRTypeVector(MGLIR_SCALAR_FLOAT, 3),
        mglIRTypeScalar(MGLIR_SCALAR_FLOAT),
    };
    const char *names[] = {"v", "f"};
    MGLIRType *s = mglIRTypeStruct(mem, names, 2, "S");
    uint32_t sz = 0;
    int rc = mglIRComputeLayout(s, MGLIR_LAYOUT_STD140, &sz);
    CHECK(rc == 0 && sz == 16, "struct{vec3,float} std140 size=16");
    CHECK(s->member_offsets[0] == 0, "vec3 member offset=0");
    CHECK(s->member_offsets[1] == 12, "float member offset=12");
    mglIRTypeDestroy(s);
}

static void test_std430(void)
{
    printf("std430\n");
    /* float[3] std430 -> stride 4, size 12. */
    MGLIRType *fa = mglIRTypeArray(mglIRTypeScalar(MGLIR_SCALAR_FLOAT), 3);
    EXPECT_SIZE(fa, MGLIR_LAYOUT_STD430, 12, "float[3] std430 size=12");
    mglIRTypeDestroy(fa);

    /* vec3[2] std430: glslang v430[2]@16 / y@48 -> 2*16 = 32. */
    MGLIRType *va = mglIRTypeArray(mglIRTypeVector(MGLIR_SCALAR_FLOAT, 3), 2);
    EXPECT_SIZE(va, MGLIR_LAYOUT_STD430, 32, "vec3[2] std430 size=32");
    mglIRTypeDestroy(va);

    /* vec3 std430 size=12. */
    MGLIRType *v3 = mglIRTypeVector(MGLIR_SCALAR_FLOAT, 3);
    EXPECT_SIZE(v3, MGLIR_LAYOUT_STD430, 12, "vec3 std430 size=12");
    mglIRTypeDestroy(v3);

    /* mat4 std430 -> 4 vec4 stride 16 -> 64. */
    MGLIRType *m4 = mglIRTypeMatrix(MGLIR_SCALAR_FLOAT, 4, 4);
    EXPECT_SIZE(m4, MGLIR_LAYOUT_STD430, 64, "mat4 std430 size=64");
    mglIRTypeDestroy(m4);

    /* mat2 std430: MatrixStride 8, size = 2*8 = 16. */
    MGLIRType *m2 = mglIRTypeMatrix(MGLIR_SCALAR_FLOAT, 2, 2);
    EXPECT_SIZE(m2, MGLIR_LAYOUT_STD430, 16, "mat2 std430 size=16");
    mglIRTypeDestroy(m2);

    /* mat3 std430: MatrixStride 16, size = 3*16 = 48. */
    MGLIRType *m3 = mglIRTypeMatrix(MGLIR_SCALAR_FLOAT, 3, 3);
    EXPECT_SIZE(m3, MGLIR_LAYOUT_STD430, 48, "mat3 std430 size=48");
    mglIRTypeDestroy(m3);

    /* struct{vec3,float} std430: a@0, b@12, size 16. */
    MGLIRType *mem[] = {
        mglIRTypeVector(MGLIR_SCALAR_FLOAT, 3),
        mglIRTypeScalar(MGLIR_SCALAR_FLOAT),
    };
    const char *names[] = {"v", "f"};
    MGLIRType *s = mglIRTypeStruct(mem, names, 2, "S");
    uint32_t sz = 0;
    int rc = mglIRComputeLayout(s, MGLIR_LAYOUT_STD430, &sz);
    CHECK(rc == 0 && sz == 16, "struct{vec3,float} std430 size=16");
    CHECK(s->member_offsets[1] == 12, "struct{vec3,float} std430 float@12");
    mglIRTypeDestroy(s);

    /* SSBO tail runtime array: contributes no space to the struct. */
    MGLIRType *ra = mglIRTypeRuntimeArray(mglIRTypeScalar(MGLIR_SCALAR_FLOAT));
    MGLIRType *sm[] = { mglIRTypeVector(MGLIR_SCALAR_FLOAT, 4), ra };
    const char *sn[] = { "head", "data" };
    MGLIRType *ssbo = mglIRTypeStruct(sm, sn, 2, "S");
    uint32_t ssz = 0;
    rc = mglIRComputeLayout(ssbo, MGLIR_LAYOUT_STD430, &ssz);
    CHECK(rc == 0 && ssz == 16, "struct with runtime array std430 size=16");
    CHECK(ssbo->member_offsets[1] == 16, "runtime array member offset=16");
    CHECK(ra->layout_valid && ra->layout.array_stride == 4,
          "runtime array stride computed");
    mglIRTypeDestroy(ssbo);
}

int main(void)
{
    printf("MGLIR std140/std430 layout tests\n");
    test_std140_scalars();
    test_std140_matrices();
    test_std140_array();
    test_std140_struct_literal();
    test_std430();
    printf("\n%d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}