/* SPDX-License-Identifier: LGPL-3.0-only */

#include "mgl_air_expr_support.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace mgl::air;

static int tests_run;
static int tests_passed;

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

struct ExprFixture {
    llvm::LLVMContext context;
    llvm::Module module;
    llvm::Function *function;
    llvm::BasicBlock *entry;
    llvm::IRBuilder<> builder;
    Codegen cg{};

    ExprFixture()
        : module("expr_support_test", context),
          function(llvm::Function::Create(
              llvm::FunctionType::get(llvm::Type::getVoidTy(context), false),
              llvm::Function::ExternalLinkage, "test", module)),
          entry(llvm::BasicBlock::Create(context, "entry", function)),
          builder(entry)
    {
        cg.ctx = &context;
        cg.b = &builder;
        cg.fn = function;
        cg.mod = &module;
    }
};

static bool intResult(llvm::Value *value, uint64_t expected)
{
    auto *constant = llvm::dyn_cast_or_null<llvm::ConstantInt>(value);
    return constant && constant->getValue() == expected;
}

static bool floatResult(llvm::Value *value, double expected)
{
    auto *constant = llvm::dyn_cast_or_null<llvm::ConstantFP>(value);
    return constant && constant->getValueAPF().convertToDouble() == expected;
}

static void testSymbolsAndSwizzles()
{
    MGLIRSymbol functionSymbol{};
    functionSymbol.name = (char *)"value";
    functionSymbol.is_function = 1;
    MGLIRSymbol valueSymbol{};
    valueSymbol.name = (char *)"value";
    MGLIRSymbol *symbols[] = {&functionSymbol, &valueSymbol};
    MGLIRModule module{};
    module.symbols = symbols;
    module.symbol_count = 2;
    CHECK(findSymbol(&module, "value") == &valueSymbol,
          "findSymbol skips functions");
    CHECK(findSymbol(&module, "missing") == nullptr,
          "findSymbol reports a missing name");

    std::vector<uint32_t> indices;
    CHECK(swizzleIndices("rgba", &indices) &&
              indices == std::vector<uint32_t>({0, 1, 2, 3}),
          "rgba maps to xyzw indices");
    CHECK(swizzleIndices("stpq", &indices) &&
              indices == std::vector<uint32_t>({0, 1, 2, 3}),
          "stpq maps to xyzw indices");
    CHECK(!swizzleIndices("", &indices) && indices.empty(),
          "empty swizzle is rejected");
    CHECK(!swizzleIndices("xh", &indices),
          "invalid swizzle component is rejected");

    MType vec3{};
    vec3.scalar = MGLIR_SCALAR_FLOAT;
    vec3.vec = 3;
    CHECK(swizzleType(vec3, 1).vec == 0,
          "one-lane swizzle produces a scalar");
    CHECK(swizzleType(vec3, 2).vec == 2,
          "two-lane swizzle produces vec2");
    MType matrix{};
    matrix.scalar = MGLIR_SCALAR_FLOAT;
    matrix.cols = 3;
    matrix.rows = 3;
    CHECK(swizzleType(matrix, 2).isMatrix(),
          "matrix type remains unchanged");

    ExprFixture fixture;
    llvm::Constant *input = llvm::ConstantDataVector::get(
        fixture.context, std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f});
    llvm::Value *swizzled = emitSwizzleValue(fixture.cg, input, {2, 0});
    auto *constant = llvm::dyn_cast_or_null<llvm::Constant>(swizzled);
    CHECK(constant && constant->getType()->isVectorTy() &&
              floatResult(constant->getAggregateElement(0u), 30.0) &&
              floatResult(constant->getAggregateElement(1u), 10.0),
          "swizzle emission preserves lane order");
}

static void testNumericExpressions()
{
    ExprFixture fixture;
    llvm::Type *i32 = llvm::Type::getInt32Ty(fixture.context);
    llvm::Constant *seven = llvm::ConstantInt::get(i32, 7, true);
    llvm::Constant *two = llvm::ConstantInt::get(i32, 2, true);
    CHECK(intResult(tryFoldConst(fixture.cg, MGL_OP_DIV, seven, two, false), 3),
          "signed constant division");
    CHECK(intResult(tryFoldConst(fixture.cg, MGL_OP_MOD, seven, two, false), 1),
          "signed constant remainder");

    MType uintType{};
    uintType.scalar = MGLIR_SCALAR_UINT;
    MType floatType{};
    floatType.scalar = MGLIR_SCALAR_FLOAT;
    llvm::Value *largeUint = llvm::ConstantInt::get(i32, 0x80000000u);
    llvm::Value *floatTwo = llvm::ConstantFP::get(
        llvm::Type::getFloatTy(fixture.context), 2.0);
    CHECK(floatResult(emitNumericBinOp(fixture.cg, MGL_OP_MUL, largeUint,
                                       floatTwo, uintType, floatType),
                      4294967296.0),
          "mixed uint/float uses unsigned conversion");

    llvm::Constant *left = llvm::ConstantDataVector::get(
        fixture.context, std::vector<uint32_t>{1, 2});
    llvm::Constant *same = llvm::ConstantDataVector::get(
        fixture.context, std::vector<uint32_t>{1, 2});
    llvm::Constant *different = llvm::ConstantDataVector::get(
        fixture.context, std::vector<uint32_t>{1, 3});
    CHECK(intResult(tryFoldConst(fixture.cg, MGL_OP_EQ, left, same, false), 1),
          "vector equality folds to scalar true");
    CHECK(intResult(tryFoldConst(fixture.cg, MGL_OP_NE, left, different, false), 1),
          "vector inequality folds to scalar true");

    llvm::StructType *pairTy = llvm::StructType::get(fixture.context, {i32, i32});
    llvm::Constant *pairA = llvm::ConstantStruct::get(
        pairTy, std::vector<llvm::Constant *>{seven, two});
    llvm::Constant *pairB = llvm::ConstantStruct::get(
        pairTy, std::vector<llvm::Constant *>{seven, two});
    CHECK(intResult(emitAggregateCompare(fixture.cg, MGL_OP_EQ, pairA, pairB), 1),
          "aggregate equality compares members");
    CHECK(tryFoldConst(fixture.cg, MGL_OP_EQ, pairA, pairB, false) == nullptr,
          "generic constant folder rejects aggregates");

    llvm::Type *f32 = llvm::Type::getFloatTy(fixture.context);
    llvm::Value *splat = broadcastTo(
        fixture.cg, llvm::ConstantFP::get(f32, 2.0),
        llvm::FixedVectorType::get(f32, 3));
    CHECK(splat->getType()->isVectorTy() &&
              llvm::cast<llvm::FixedVectorType>(splat->getType())->getNumElements() == 3,
          "broadcast creates the requested vector width");
    llvm::Constant *a = llvm::ConstantDataVector::get(
        fixture.context, std::vector<float>{1.0f, 2.0f, 3.0f});
    llvm::Constant *b = llvm::ConstantDataVector::get(
        fixture.context, std::vector<float>{4.0f, 5.0f, 6.0f});
    CHECK(floatResult(dotProduct(fixture.cg, a, b), 32.0),
          "dot product preserves fixed lane order");
}

int main()
{
    printf("mgl_air_expr_support golden (non-Metal)\n");
    testSymbolsAndSwizzles();
    testNumericExpressions();
    printf("%d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
