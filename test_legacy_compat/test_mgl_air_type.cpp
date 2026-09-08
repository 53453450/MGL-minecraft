/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * test_mgl_air_type.cpp
 *
 * Non-Metal golden for C1b mgl_air_type helpers: typeFromIR, airTypeMangle /
 * mslTypeName, and float-carrier predicates.  Links mgl_air_type.cpp +
 * mgl_ir.c + LLVM only — no Metal, no mgl_air_backend, no Codegen IR emit.
 *
 * Linux smoke (llvm-19):
 *   g++ -std=c++20 -fno-exceptions \
 *     -I/usr/lib/llvm-19/include -IMGL/include \
 *     -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS \
 *     -D__STDC_LIMIT_MACROS \
 *     test_legacy_compat/test_mgl_air_type.cpp MGL/src/mgl_air_type.cpp \
 *     -x c MGL/src/mgl_ir.c -x none \
 *     -L/usr/lib/llvm-19/lib -lLLVM-19 -o build/test_mgl_air_type
 */

#include "mgl_air_type.h"

#include <cstdio>
#include <cstring>
#include <string>

extern "C" {
#include "mgl_ir.h"
}

using mgl::air::MType;
using mgl::air::airGenerated;
using mgl::air::airTypeMangle;
using mgl::air::floatCarrierType;
using mgl::air::mslTypeName;
using mgl::air::typeFromIR;
using mgl::air::uintUsesSplitFloatCarrier;
using mgl::air::varyingUsesFloatCarrier;

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

static void test_type_from_ir(void)
{
    printf("typeFromIR\n");
    MGLIRType *f = mglIRTypeScalar(MGLIR_SCALAR_FLOAT);
    MType tf = typeFromIR(f);
    CHECK(tf.scalar == MGLIR_SCALAR_FLOAT && !tf.vec && !tf.cols && !tf.arr,
          "float -> scalar float");
    mglIRTypeDestroy(f);

    MGLIRType *v4 = mglIRTypeVector(MGLIR_SCALAR_FLOAT, 4);
    MType tv4 = typeFromIR(v4);
    CHECK(tv4.scalar == MGLIR_SCALAR_FLOAT && tv4.vec == 4u && !tv4.cols,
          "vec4 -> vec=4");
    mglIRTypeDestroy(v4);

    MGLIRType *iv2 = mglIRTypeVector(MGLIR_SCALAR_INT, 2);
    MType tiv2 = typeFromIR(iv2);
    CHECK(tiv2.scalar == MGLIR_SCALAR_INT && tiv2.vec == 2u,
          "ivec2 -> int vec=2");
    mglIRTypeDestroy(iv2);

    MGLIRType *m3 = mglIRTypeMatrix(MGLIR_SCALAR_FLOAT, 3, 3);
    MType tm3 = typeFromIR(m3);
    CHECK(tm3.cols == 3u && tm3.rows == 3u && tm3.scalar == MGLIR_SCALAR_FLOAT,
          "mat3 -> cols=rows=3");
    mglIRTypeDestroy(m3);

    MGLIRType *el = mglIRTypeScalar(MGLIR_SCALAR_FLOAT);
    MGLIRType *arr = mglIRTypeArray(el, 4);
    MType ta = typeFromIR(arr);
    CHECK(ta.arr == 4u && ta.scalar == MGLIR_SCALAR_FLOAT && !ta.vec,
          "float[4] -> arr=4");
    mglIRTypeDestroy(arr);

    MGLIRType *vel = mglIRTypeVector(MGLIR_SCALAR_FLOAT, 4);
    MGLIRType *varr = mglIRTypeArray(vel, 2);
    MType tva = typeFromIR(varr);
    CHECK(tva.arr == 2u && tva.vec == 4u && tva.scalar == MGLIR_SCALAR_FLOAT,
          "vec4[2] -> arr=2 vec=4");
    mglIRTypeDestroy(varr);
}

static void test_mangle_msl(void)
{
    printf("airTypeMangle / mslTypeName\n");
    MType f{};
    f.scalar = MGLIR_SCALAR_FLOAT;
    CHECK(airTypeMangle(f) == "f", "mangle float=f");
    CHECK(mslTypeName(f) == "float", "msl float");

    MType i{};
    i.scalar = MGLIR_SCALAR_INT;
    CHECK(airTypeMangle(i) == "i", "mangle int=i");
    CHECK(mslTypeName(i) == "int", "msl int");

    MType u{};
    u.scalar = MGLIR_SCALAR_UINT;
    CHECK(airTypeMangle(u) == "j", "mangle uint=j");
    CHECK(mslTypeName(u) == "uint", "msl uint");

    MType b{};
    b.scalar = MGLIR_SCALAR_BOOL;
    CHECK(airTypeMangle(b) == "b", "mangle bool=b");
    CHECK(mslTypeName(b) == "bool", "msl bool");

    MType v4 = f;
    v4.vec = 4;
    CHECK(airTypeMangle(v4) == "Dv4_f", "mangle vec4=Dv4_f");
    CHECK(mslTypeName(v4) == "float4", "msl float4");
    CHECK(mslTypeName(MType{MGLIR_SCALAR_FLOAT, 2, 0, 0, 0}) == "float2",
          "msl float2");
    CHECK(mslTypeName(MType{MGLIR_SCALAR_FLOAT, 3, 0, 0, 0}) == "float3",
          "msl float3");

    MType iv3{};
    iv3.scalar = MGLIR_SCALAR_INT;
    iv3.vec = 3;
    CHECK(airTypeMangle(iv3) == "Dv3_i", "mangle ivec3=Dv3_i");
    CHECK(mslTypeName(iv3) == "int3", "msl int3");

    MType m4{};
    m4.scalar = MGLIR_SCALAR_FLOAT;
    m4.cols = 4;
    m4.rows = 4;
    CHECK(airTypeMangle(m4) == "float4x4", "mangle mat4 uses msl name");
    CHECK(mslTypeName(m4) == "float4x4", "msl float4x4");

    MType arr = f;
    arr.arr = 4;
    CHECK(airTypeMangle(arr) == "float", "mangle float[4] peels to msl float");
    CHECK(mslTypeName(arr) == "float", "msl float[4] peels");

    CHECK(airGenerated("vUV", v4) == "generated(3vUVDv4_f)",
          "airGenerated vUV vec4");
}

static void test_carriers(void)
{
    printf("float carriers\n");
    MType i{};
    i.scalar = MGLIR_SCALAR_INT;
    CHECK(varyingUsesFloatCarrier(i, false), "int needs float carrier (no GS)");
    CHECK(varyingUsesFloatCarrier(i, true), "int needs float carrier (GS)");

    MType f{};
    f.scalar = MGLIR_SCALAR_FLOAT;
    CHECK(!varyingUsesFloatCarrier(f, false), "float skips carrier");
    CHECK(!varyingUsesFloatCarrier(f, true), "float skips carrier (GS)");

    MType b{};
    b.scalar = MGLIR_SCALAR_BOOL;
    CHECK(!varyingUsesFloatCarrier(b, true), "bool never uses float carrier");

    MType u{};
    u.scalar = MGLIR_SCALAR_UINT;
    CHECK(uintUsesSplitFloatCarrier(u, false),
          "scalar uint split carrier (no GS)");
    CHECK(!uintUsesSplitFloatCarrier(u, true),
          "scalar uint no split under GS");

    MType uv2 = u;
    uv2.vec = 2;
    CHECK(!uintUsesSplitFloatCarrier(uv2, false),
          "uvec2 uses ordinary float carrier");

    MType fc = floatCarrierType(i);
    CHECK(fc.scalar == MGLIR_SCALAR_FLOAT && fc.vec == i.vec,
          "floatCarrierType(int) -> float");
    MType fcv = floatCarrierType(uv2);
    CHECK(fcv.scalar == MGLIR_SCALAR_FLOAT && fcv.vec == 2u,
          "floatCarrierType(uvec2) -> float2 shape");
}

int main(void)
{
    printf("mgl_air_type golden (non-Metal)\n");
    test_type_from_ir();
    test_mangle_msl();
    test_carriers();
    printf("%d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
