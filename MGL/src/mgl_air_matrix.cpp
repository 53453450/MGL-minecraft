/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_matrix.cpp
 * C1f — AIR matrix builtins / binops extracted from mgl_air_backend.cpp
 * (emitMatrixBuiltin / emitMatrixBinOp + det helpers).  Backend keeps a
 * thin AirMatrixDeps facade; emitExpr stays in the monolith.
 */

#include "mgl_air_matrix.h"

#include <cstring>

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

namespace mgl {
namespace air {

/* det of the 2x2 block (c0,c1) x (r0,r1) of a matrix. */
static llvm::Value *det2Sel(Codegen &cg, llvm::Value *c0, llvm::Value *c1,
                            uint32_t r0, uint32_t r1) {
    auto cI = [&](uint32_t v) {
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), v);
    };
    llvm::Value *a = cg.b->CreateExtractElement(c0, cI(r0));
    llvm::Value *b = cg.b->CreateExtractElement(c1, cI(r0));
    llvm::Value *c = cg.b->CreateExtractElement(c0, cI(r1));
    llvm::Value *d = cg.b->CreateExtractElement(c1, cI(r1));
    return cg.b->CreateFSub(cg.b->CreateFMul(a, d), cg.b->CreateFMul(b, c));
}

/* det of the 3x3 block (cols[c0..c2]) x (rows r0..r2) of a matrix. */
static llvm::Value *det3Sel(Codegen &cg, llvm::Value *const *cols,
                            uint32_t c0, uint32_t c1, uint32_t c2,
                            uint32_t r0, uint32_t r1, uint32_t r2) {
    auto cI = [&](uint32_t v) {
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), v);
    };
    auto el = [&](uint32_t c, uint32_t r) {
        return cg.b->CreateExtractElement(cols[c], cI(r));
    };
    llvm::Value *a = el(c0, r0), *b = el(c1, r0), *cc = el(c2, r0);
    llvm::Value *d = el(c0, r1), *e = el(c1, r1), *f = el(c2, r1);
    llvm::Value *g = el(c0, r2), *h = el(c1, r2), *ii = el(c2, r2);
    /* a(ei - fh) - b(di - fg) + c(dh - eg) */
    llvm::Value *t1 = cg.b->CreateFSub(cg.b->CreateFMul(e, ii),
                                       cg.b->CreateFMul(f, h));
    llvm::Value *t2 = cg.b->CreateFSub(cg.b->CreateFMul(d, ii),
                                       cg.b->CreateFMul(f, g));
    llvm::Value *t3 = cg.b->CreateFSub(cg.b->CreateFMul(d, h),
                                       cg.b->CreateFMul(e, g));
    llvm::Value *r0v = cg.b->CreateFSub(cg.b->CreateFMul(a, t1),
                                        cg.b->CreateFMul(b, t2));
    return cg.b->CreateFAdd(r0v, cg.b->CreateFMul(cc, t3));
}

/* Determinant of a square float matrix ([N x <N x float>]). */
static llvm::Value *detMatrix(Codegen &cg, llvm::Value *m) {
    auto *arr = llvm::cast<llvm::ArrayType>(m->getType());
    uint32_t C = (uint32_t)arr->getNumElements();
    auto cI = [&](uint32_t v) {
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), v);
    };
    llvm::Value *cols[4];
    for (uint32_t c = 0; c < C; c++)
        cols[c] = cg.b->CreateExtractValue(m, c);
    if (C == 2) return det2Sel(cg, cols[0], cols[1], 0, 1);
    if (C == 3) return det3Sel(cg, cols, 0, 1, 2, 0, 1, 2);
    llvm::Value *acc = cg.b->CreateFMul(
        cg.b->CreateExtractElement(cols[0], cI(0)),
        det3Sel(cg, cols, 1, 2, 3, 1, 2, 3));
    llvm::Value *t = cg.b->CreateFMul(
        cg.b->CreateExtractElement(cols[0], cI(1)),
        det3Sel(cg, cols, 1, 2, 3, 0, 2, 3));
    acc = cg.b->CreateFSub(acc, t);
    t = cg.b->CreateFMul(
        cg.b->CreateExtractElement(cols[0], cI(2)),
        det3Sel(cg, cols, 1, 2, 3, 0, 1, 3));
    acc = cg.b->CreateFAdd(acc, t);
    t = cg.b->CreateFMul(
        cg.b->CreateExtractElement(cols[0], cI(3)),
        det3Sel(cg, cols, 1, 2, 3, 0, 1, 2));
    return cg.b->CreateFSub(acc, t);
}

/* Matrix builtins: transpose, matrixCompMult, outerProduct, determinant
 * and inverse (square float matrices, sema-typed subset).  Returns NULL
 * when `name` is not a matrix builtin handled here. */
llvm::Value *emitMatrixBuiltin(Codegen &cg, const MGLExpr *e,
                               const char *name, const MGLIRModule *mod,
                               const std::map<std::string, MType> &locals,
                               const AirMatrixDeps &deps) {
    bool isT = !strcmp(name, "transpose");
    bool isC = !strcmp(name, "matrixCompMult");
    bool isO = !strcmp(name, "outerProduct");
    bool isD = !strcmp(name, "determinant");
    bool isI = !strcmp(name, "inverse");
    if (!isT && !isC && !isO && !isD && !isI) return nullptr;

    llvm::Value *a = deps.emitExpr(cg, e->u.call.args[0], mod, locals);
    if (!a) return nullptr;
    llvm::Value *b = nullptr;
    if (e->u.call.arg_count == 2) {
        b = deps.emitExpr(cg, e->u.call.args[1], mod, locals);
        if (!b) return nullptr;
    }
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    auto cI = [&](uint32_t v) {
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), v);
    };

    if (isT) {
        auto *arr = llvm::cast<llvm::ArrayType>(a->getType());
        uint32_t C = (uint32_t)arr->getNumElements();
        uint32_t R = (uint32_t)llvm::cast<llvm::FixedVectorType>(
                         arr->getElementType())
                         ->getElementCount()
                         .getFixedValue();
        llvm::Type *outTy =
            llvm::ArrayType::get(llvm::FixedVectorType::get(f32, C), R);
        llvm::Value *out = llvm::UndefValue::get(outTy);
        for (uint32_t j = 0; j < R; j++) {
            llvm::Value *col = llvm::UndefValue::get(
                llvm::FixedVectorType::get(f32, C));
            for (uint32_t i = 0; i < C; i++) {
                llvm::Value *x = cg.b->CreateExtractElement(
                    cg.b->CreateExtractValue(a, i), cI(j));
                col = cg.b->CreateInsertElement(col, x, cI(i));
            }
            out = cg.b->CreateInsertValue(out, col, j);
        }
        return out;
    }
    if (isC) {
        auto *arr = llvm::cast<llvm::ArrayType>(a->getType());
        uint32_t C = (uint32_t)arr->getNumElements();
        llvm::Value *out = llvm::UndefValue::get(a->getType());
        for (uint32_t c = 0; c < C; c++) {
            llvm::Value *m = cg.b->CreateFMul(
                cg.b->CreateExtractValue(a, c), cg.b->CreateExtractValue(b, c));
            out = cg.b->CreateInsertValue(out, m, c);
        }
        return out;
    }
    if (isO) {
        auto *va = llvm::cast<llvm::FixedVectorType>(a->getType());
        auto *vb = llvm::cast<llvm::FixedVectorType>(b->getType());
        uint32_t C = (uint32_t)va->getElementCount().getFixedValue();
        uint32_t R = (uint32_t)vb->getElementCount().getFixedValue();
        llvm::Type *outTy =
            llvm::ArrayType::get(llvm::FixedVectorType::get(f32, R), C);
        llvm::Value *out = llvm::UndefValue::get(outTy);
        for (uint32_t c = 0; c < C; c++) {
            llvm::Value *coef = cg.b->CreateExtractElement(a, cI(c));
            llvm::Value *col = cg.b->CreateFMul(
                cg.b->CreateVectorSplat(R, coef), b);
            out = cg.b->CreateInsertValue(out, col, c);
        }
        return out;
    }
    if (isD) {
        return detMatrix(cg, a);
    }
    if (isI) {
        auto *arr = llvm::cast<llvm::ArrayType>(a->getType());
        uint32_t C = (uint32_t)arr->getNumElements();
        uint32_t R = (uint32_t)llvm::cast<llvm::FixedVectorType>(
                         arr->getElementType())
                         ->getElementCount()
                         .getFixedValue();
        llvm::Value *cols[4];
        for (uint32_t c = 0; c < C; c++)
            cols[c] = cg.b->CreateExtractValue(a, c);
        llvm::Value *inv = cg.b->CreateFDiv(
            llvm::ConstantFP::get(f32, 1.0), detMatrix(cg, a));
        llvm::Value *out = llvm::UndefValue::get(
            llvm::ArrayType::get(llvm::FixedVectorType::get(f32, R), C));
        if (C == 2) {
            /* inv = 1/det * [[a11, -a01], [-a10, a00]] in column-major
             * order: col0 = (a11, -a10), col1 = (-a01, a00). */
            llvm::Value *a00 = cg.b->CreateExtractElement(cols[0], cI(0));
            llvm::Value *a10 = cg.b->CreateExtractElement(cols[0], cI(1));
            llvm::Value *a01 = cg.b->CreateExtractElement(cols[1], cI(0));
            llvm::Value *a11 = cg.b->CreateExtractElement(cols[1], cI(1));
            llvm::Value *col0 = llvm::UndefValue::get(
                llvm::FixedVectorType::get(f32, 2));
            col0 = cg.b->CreateInsertElement(col0,
                cg.b->CreateFMul(a11, inv), cI(0));
            col0 = cg.b->CreateInsertElement(col0,
                cg.b->CreateFMul(cg.b->CreateFNeg(a10), inv), cI(1));
            llvm::Value *col1 = llvm::UndefValue::get(
                llvm::FixedVectorType::get(f32, 2));
            col1 = cg.b->CreateInsertElement(col1,
                cg.b->CreateFMul(cg.b->CreateFNeg(a01), inv), cI(0));
            col1 = cg.b->CreateInsertElement(col1,
                cg.b->CreateFMul(a00, inv), cI(1));
            out = cg.b->CreateInsertValue(out, col0, 0);
            return cg.b->CreateInsertValue(out, col1, 1);
        }
        /* Cofactor formula: inv[i][j] = (-1)^(i+j) * det(minor row j,
         * col i) / det(A). */
        auto otherIdx = [](uint32_t n, uint32_t skip, uint32_t out[3]) {
            uint32_t k = 0;
            for (uint32_t c = 0; c < n; c++)
                if (c != skip) out[k++] = c;
        };
        for (uint32_t j = 0; j < C; j++) {
            llvm::Value *col = llvm::UndefValue::get(
                llvm::FixedVectorType::get(f32, R));
            for (uint32_t i = 0; i < R; i++) {
                uint32_t cs[3], rs[3];
                otherIdx(C, i, cs);
                otherIdx(C, j, rs);
                llvm::Value *m = C == 3
                    ? det2Sel(cg, cols[cs[0]], cols[cs[1]], rs[0], rs[1])
                    : det3Sel(cg, cols, cs[0], cs[1], cs[2],
                              rs[0], rs[1], rs[2]);
                if (((i + j) & 1) != 0)
                    m = cg.b->CreateFNeg(m);
                m = cg.b->CreateFMul(m, inv);
                col = cg.b->CreateInsertElement(col, m, cI(i));
            }
            out = cg.b->CreateInsertValue(out, col, j);
        }
        return out;
    }
    return nullptr;
}


/* Resolve a sampler argument: a global sampler2D uniform (cg.texValues)
 * or a user-function parameter bound in cg.lvalues. */
static llvm::Value *samplerTexValue(Codegen &cg, const char *name) {
    auto t = cg.texValues.find(name);
    if (t != cg.texValues.end()) return t->second;
    auto l = cg.lvalues.find(name);
    if (l != cg.lvalues.end()) return l->second;
    return nullptr;
}

/* Scalar base for an LLVM type, used to coerce call arguments. */
static MGLIRScalar scalarFromType(llvm::Type *t) {
    if (auto *fv = llvm::dyn_cast<llvm::FixedVectorType>(t))
        t = fv->getElementType();
    if (t->isFloatingPointTy()) return MGLIR_SCALAR_FLOAT;
    if (t->isIntegerTy(1)) return MGLIR_SCALAR_BOOL;
    if (t->isIntegerTy(32)) return MGLIR_SCALAR_INT;
    return MGLIR_SCALAR_FLOAT;
}

/* Call a named AIR function (e.g. air.pack.unorm2x16.v2f32); the module
 * declaration is created on first use. */
llvm::Value *callAirFn(Codegen &cg, const char *fn, llvm::Type *retTy,
                       llvm::ArrayRef<llvm::Value *> args) {
    llvm::SmallVector<llvm::Type *, 4> argTys;
    for (llvm::Value *a : args) argTys.push_back(a->getType());
    llvm::FunctionType *ft =
        llvm::FunctionType::get(retTy, argTys, false);
    llvm::FunctionCallee callee = cg.mod->getOrInsertFunction(fn, ft);
    return cg.b->CreateCall(callee, args);
}

/* Matrix binary ops: M*vec, vec*M, M*M, M*scalar, scalar*M, M±M and
 * M±scalar (element-wise).  Column-major storage: the LLVM value is
 * [cols x <rows x float>].  Returns nullptr when neither operand is a
 * matrix, so the caller falls back to the scalar/vector path. */
llvm::Value *emitMatrixBinOp(Codegen &cg, uint32_t op, llvm::Value *l,
                             llvm::Value *r, const AirMatrixDeps &deps) {
    llvm::ArrayType *larr = llvm::dyn_cast<llvm::ArrayType>(l->getType());
    llvm::ArrayType *rarr = llvm::dyn_cast<llvm::ArrayType>(r->getType());
    if (!larr && !rarr) return nullptr;

    llvm::Type *elt = llvm::Type::getFloatTy(*cg.ctx);
    llvm::Constant *zero = llvm::ConstantFP::get(elt, 0.0);
    auto cI = [&](uint32_t v) {
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), v);
    };
    auto colCount = [](llvm::ArrayType *a) {
        return (uint32_t)a->getNumElements();
    };
    auto colType = [](llvm::ArrayType *a) {
        return llvm::cast<llvm::FixedVectorType>(a->getElementType());
    };
    auto rowCount = [&](llvm::ArrayType *a) {
        return (uint32_t)colType(a)->getElementCount().getFixedValue();
    };

    if (op == MGL_OP_MUL) {
        if (larr && r->getType()->isVectorTy()) {
            /* M * v: out = sum of col_c * v[c]. */
            uint32_t cols = colCount(larr), rows = rowCount(larr);
            llvm::Value *out = llvm::Constant::getNullValue(
                llvm::FixedVectorType::get(elt, rows));
            for (uint32_t c = 0; c < cols; c++) {
                llvm::Value *col = cg.b->CreateExtractValue(l, c);
                llvm::Value *splat = cg.b->CreateShuffleVector(r,
                    llvm::UndefValue::get(r->getType()),
                    llvm::ConstantVector::getSplat(
                        llvm::ElementCount::getFixed(rows), cI(c)));
                llvm::Value *term = cg.b->CreateFMul(col, splat);
                out = c == 0 ? term : cg.b->CreateFAdd(out, term);
            }
            return out;
        }
        if (l->getType()->isVectorTy() && rarr) {
            /* v * M: out[c] = dot(v, col_c). */
            uint32_t cols = colCount(rarr), rows = rowCount(rarr);
            llvm::Value *out = llvm::UndefValue::get(
                llvm::FixedVectorType::get(elt, cols));
            for (uint32_t c = 0; c < cols; c++) {
                llvm::Value *col = cg.b->CreateExtractValue(r, c);
                llvm::Value *d = deps.dotProduct(cg, l, col);
                out = cg.b->CreateInsertElement(out, d, cI(c));
            }
            return out;
        }
        if (larr && rarr) {
            /* M * M: out col c = sum_k splat(B[c][k]) * A[k]; sema
             * guarantees A->cols == B->rows. */
            uint32_t lc = colCount(larr), lr = rowCount(larr);
            uint32_t rc = colCount(rarr);
            llvm::Value *out = llvm::UndefValue::get(
                llvm::ArrayType::get(larr->getElementType(), rc));
            for (uint32_t c = 0; c < rc; c++) {
                llvm::Value *colB = cg.b->CreateExtractValue(r, c);
                llvm::Value *acc = llvm::Constant::getNullValue(
                    larr->getElementType());
                for (uint32_t k = 0; k < lc; k++) {
                    llvm::Value *coef = cg.b->CreateExtractElement(colB, cI(k));
                    llvm::Value *colA = cg.b->CreateExtractValue(l, k);
                    llvm::Value *term = cg.b->CreateFMul(colA,
                        cg.b->CreateVectorSplat(lr, coef));
                    acc = k == 0 ? term : cg.b->CreateFAdd(acc, term);
                }
                out = cg.b->CreateInsertValue(out, acc, c);
            }
            return out;
        }
        if (larr) {
            /* M * scalar: per-column scale. */
            llvm::Value *s = r;
            if (s->getType()->isVectorTy()) {
                llvm::Value *s0 = cg.b->CreateExtractElement(
                    s, llvm::ConstantInt::get(
                           llvm::Type::getInt32Ty(*cg.ctx), 0));
                s = s0;
            }
            llvm::Value *out = llvm::UndefValue::get(l->getType());
            for (uint32_t c = 0; c < colCount(larr); c++) {
                llvm::Value *col = cg.b->CreateExtractValue(l, c);
                llvm::Value *term = cg.b->CreateFMul(col,
                    cg.b->CreateVectorSplat(rowCount(larr), s));
                out = cg.b->CreateInsertValue(out, term, c);
            }
            return out;
        }
        if (rarr) {
            /* scalar * M. */
            llvm::Value *s = l;
            if (s->getType()->isVectorTy()) {
                llvm::Value *s0 = cg.b->CreateExtractElement(
                    s, llvm::ConstantInt::get(
                           llvm::Type::getInt32Ty(*cg.ctx), 0));
                s = s0;
            }
            llvm::Value *out = llvm::UndefValue::get(r->getType());
            for (uint32_t c = 0; c < colCount(rarr); c++) {
                llvm::Value *col = cg.b->CreateExtractValue(r, c);
                llvm::Value *term = cg.b->CreateFMul(col,
                    cg.b->CreateVectorSplat(rowCount(rarr), s));
                out = cg.b->CreateInsertValue(out, term, c);
            }
            return out;
        }
        return nullptr;
    }

    if (op == MGL_OP_ADD || op == MGL_OP_SUB) {
        bool sub = op == MGL_OP_SUB;
        if (larr && rarr) {
            llvm::Value *out = llvm::UndefValue::get(l->getType());
            for (uint32_t c = 0; c < colCount(larr); c++) {
                llvm::Value *lc = cg.b->CreateExtractValue(l, c);
                llvm::Value *rc = cg.b->CreateExtractValue(r, c);
                llvm::Value *m = sub ? cg.b->CreateFSub(lc, rc)
                                     : cg.b->CreateFAdd(lc, rc);
                out = cg.b->CreateInsertValue(out, m, c);
            }
            return out;
        }
        llvm::Value *arr = larr ? l : r;
        llvm::Value *s = larr ? r : l;
        if (s->getType()->isVectorTy()) {
            s = cg.b->CreateExtractElement(
                s, llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), 0));
        }
        llvm::Value *out = llvm::UndefValue::get(arr->getType());
        bool scalarOnRight = larr != nullptr;
        for (uint32_t c = 0; c < colCount(
                llvm::cast<llvm::ArrayType>(arr->getType())); c++) {
            llvm::Value *col = cg.b->CreateExtractValue(arr, c);
            llvm::Value *bs = cg.b->CreateVectorSplat(
                rowCount(llvm::cast<llvm::ArrayType>(arr->getType())), s);
            llvm::Value *m = sub
                ? (scalarOnRight ? cg.b->CreateFSub(col, bs)
                                 : cg.b->CreateFSub(bs, col))
                : cg.b->CreateFAdd(col, bs);
            out = cg.b->CreateInsertValue(out, m, c);
        }
        return out;
    }
    if ((op == MGL_OP_EQ || op == MGL_OP_NE) && larr && rarr) {
        /* GLSL 4.60 §5.9: mat == / != → scalar bool (all elements). */
        uint32_t cols = colCount(larr);
        if (cols != colCount(rarr) || rowCount(larr) != rowCount(rarr))
            return nullptr;
        llvm::Type *colTy = larr->getElementType();
        bool fp = colTy->isFPOrFPVectorTy() ||
                  (llvm::isa<llvm::FixedVectorType>(colTy) &&
                   llvm::cast<llvm::FixedVectorType>(colTy)
                       ->getElementType()
                       ->isFloatingPointTy());
        llvm::Value *acc = nullptr;
        for (uint32_t c = 0; c < cols; c++) {
            llvm::Value *lc = cg.b->CreateExtractValue(l, c);
            llvm::Value *rc = cg.b->CreateExtractValue(r, c);
            llvm::Value *cmp =
                fp ? (op == MGL_OP_EQ ? cg.b->CreateFCmpOEQ(lc, rc)
                                      : cg.b->CreateFCmpONE(lc, rc))
                   : (op == MGL_OP_EQ ? cg.b->CreateICmpEQ(lc, rc)
                                      : cg.b->CreateICmpNE(lc, rc));
            cmp = deps.scalarizeBoolCompare(cg, op, cmp);
            acc = !acc ? cmp
                       : (op == MGL_OP_EQ ? cg.b->CreateAnd(acc, cmp)
                                          : cg.b->CreateOr(acc, cmp));
        }
        return acc;
    }
    return nullptr;
}

} /* namespace air */
} /* namespace mgl */
