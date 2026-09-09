/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_math.cpp
 * C1d — AIR math builtins extracted from mgl_air_backend.cpp
 * (emitMathBuiltin + float-intrinsic helpers).  Backend keeps a thin
 * AirMathDeps facade; emitExpr stays in the monolith (matrix → C1f).
 */

#include "mgl_air_math.h"
#include "mgl_air_type.h"

#include <cmath>
#include <cstring>
#include <utility>

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

namespace mgl {
namespace air {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/* Element-wise float intrinsic (scalar or vector operand). */
static llvm::Value *callFloatIntrinsic(Codegen &cg, llvm::Intrinsic::ID id,
                                llvm::Value *v) {
    return cg.b->CreateIntrinsic(id, {v->getType()}, {v});
}

static bool typeIsIntLike(llvm::Type *t) {
    return t->isIntOrIntVectorTy() &&
           (!t->isVectorTy() || llvm::cast<llvm::FixedVectorType>(t)
                                    ->getElementType()
                                    ->isIntegerTy());
}

/* Constant 0.0 / 1.0 with the shape of `t` (scalar or vector). */
static llvm::Constant *fpConstOf(Codegen &cg, llvm::Type *t, double v) {
    llvm::Constant *c =
        llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), v);
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
        return llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(
                (uint32_t)vt->getElementCount().getFixedValue()),
            c);
    }
    return c;
}

/* ---- math builtins ----------------------------------------------------- */

llvm::Value *emitMathBuiltin(Codegen &cg, const MGLExpr *e,
                            const char *name, const MGLIRModule *mod,
                            const std::map<std::string, MType> &locals,
                            const AirMathDeps &deps)
{
    auto arg = [&](uint32_t i) -> llvm::Value * {
        return deps.emitExpr(cg, e->u.call.args[i], mod, locals);
    };
    auto farg = [&](uint32_t i) -> llvm::Value * {
        llvm::Value *v = arg(i);
        return v ? coerceScalar(cg, v, MGLIR_SCALAR_FLOAT) : nullptr;
    };
    auto need = [&](uint32_t want) -> bool {
        if (e->u.call.arg_count == want) return true;
        cg.err = 1;
        cg.errmsg = std::string("codegen: builtin '") + name + "' expects " +
                    std::to_string(want) + " argument(s)";
        return false;
    };

    /* --- float scalar/vector functions --- */
    llvm::Value *a0 = nullptr, *a1 = nullptr, *a2 = nullptr;
    (void)a1;
    (void)a2;

    if (strcmp(name, "lessThanEqual") == 0) {
        if (!need(2)) {
            return nullptr;
        }
        a0 = farg(0);
        a1 = farg(1);
        if (!a0 || !a1) {
            return nullptr;
        }
        return cg.b->CreateFCmp(llvm::CmpInst::FCMP_OLE, a0, a1);
    }
    if (strcmp(name, "lessThan") == 0 || strcmp(name, "greaterThan") == 0 ||
        strcmp(name, "greaterThanEqual") == 0 ||
        strcmp(name, "equal") == 0 || strcmp(name, "notEqual") == 0) {
        if (!need(2)) {
            return nullptr;
        }
        a0 = arg(0);
        a1 = arg(1);
        if (!a0 || !a1) {
            return nullptr;
        }
        llvm::CmpInst::Predicate fPred = llvm::CmpInst::FCMP_OLT;
        llvm::CmpInst::Predicate iPred = llvm::CmpInst::ICMP_SLT;
        if (strcmp(name, "greaterThan") == 0) {
            fPred = llvm::CmpInst::FCMP_OGT;
            iPred = llvm::CmpInst::ICMP_SGT;
        } else if (strcmp(name, "greaterThanEqual") == 0) {
            fPred = llvm::CmpInst::FCMP_OGE;
            iPred = llvm::CmpInst::ICMP_SGE;
        } else if (strcmp(name, "equal") == 0) {
            fPred = llvm::CmpInst::FCMP_OEQ;
            iPred = llvm::CmpInst::ICMP_EQ;
        } else if (strcmp(name, "notEqual") == 0) {
            fPred = llvm::CmpInst::FCMP_ONE;
            iPred = llvm::CmpInst::ICMP_NE;
        }
        if (a0->getType()->isFPOrFPVectorTy()) {
            a0 = coerceScalar(cg, a0, MGLIR_SCALAR_FLOAT);
            a1 = coerceScalar(cg, a1, MGLIR_SCALAR_FLOAT);
            return cg.b->CreateFCmp(fPred, a0, a1);
        }
        a0 = coerceScalar(cg, a0, MGLIR_SCALAR_INT);
        a1 = coerceScalar(cg, a1, MGLIR_SCALAR_INT);
        return cg.b->CreateICmp(iPred, a0, a1);
    }
    if (strcmp(name, "all") == 0) {
        if (!need(1)) {
            return nullptr;
        }
        a0 = arg(0);
        if (!a0) {
            return nullptr;
        }
        if (!a0->getType()->isVectorTy()) {
            return a0;
        }
        auto *vt = llvm::cast<llvm::FixedVectorType>(a0->getType());
        uint32_t n = (uint32_t)vt->getElementCount().getFixedValue();
        llvm::Value *acc = cg.b->CreateExtractElement(a0, (uint64_t)0);
        for (uint32_t i = 1; i < n; i++) {
            acc = cg.b->CreateAnd(
                acc, cg.b->CreateExtractElement(a0, (uint64_t)i));
        }
        return acc;
    }

    if (strcmp(name, "floatBitsToInt") == 0 ||
        strcmp(name, "floatBitsToUint") == 0) {
        if (!need(1)) return nullptr;
        a0 = arg(0);
        if (!a0) return nullptr;
        if (a0->getType()->isVectorTy()) {
            return cg.b->CreateBitCast(a0, llvm::VectorType::get(
                llvm::Type::getInt32Ty(*cg.ctx),
                llvm::cast<llvm::FixedVectorType>(a0->getType())->getNumElements(),
                false));
        }
        return cg.b->CreateBitCast(a0, cg.b->getInt32Ty());
    }

    if (strcmp(name, "sin") == 0 || strcmp(name, "cos") == 0 ||
        strcmp(name, "exp") == 0 || strcmp(name, "exp2") == 0 ||
        strcmp(name, "log") == 0 || strcmp(name, "log2") == 0 ||
        strcmp(name, "floor") == 0 || strcmp(name, "ceil") == 0 ||
        strcmp(name, "trunc") == 0 || strcmp(name, "round") == 0 ||
        strcmp(name, "roundEven") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        llvm::Intrinsic::ID id;
        if (strcmp(name, "sin") == 0) id = llvm::Intrinsic::sin;
        else if (strcmp(name, "cos") == 0) id = llvm::Intrinsic::cos;
        else if (strcmp(name, "exp") == 0) id = llvm::Intrinsic::exp;
        else if (strcmp(name, "exp2") == 0) id = llvm::Intrinsic::exp2;
        else if (strcmp(name, "log") == 0) id = llvm::Intrinsic::log;
        else if (strcmp(name, "log2") == 0) id = llvm::Intrinsic::log2;
        else if (strcmp(name, "floor") == 0) id = llvm::Intrinsic::floor;
        else if (strcmp(name, "ceil") == 0) id = llvm::Intrinsic::ceil;
        else if (strcmp(name, "trunc") == 0) id = llvm::Intrinsic::trunc;
        else if (strcmp(name, "round") == 0) id = llvm::Intrinsic::round;
        else id = llvm::Intrinsic::roundeven;
        return callFloatIntrinsic(cg, id, a0);
    }
    if (strcmp(name, "sqrt") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        return callFloatIntrinsic(cg, llvm::Intrinsic::sqrt, a0);
    }
    if (strcmp(name, "length") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        return callFloatIntrinsic(cg, llvm::Intrinsic::sqrt,
                                  deps.dotProduct(cg, a0, a0));
    }
    if (strcmp(name, "distance") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        if (!a0 || !a1) return nullptr;
        llvm::Value *d = cg.b->CreateFSub(a0, a1);
        return callFloatIntrinsic(cg, llvm::Intrinsic::sqrt,
                                  deps.dotProduct(cg, d, d));
    }
    if (strcmp(name, "normalize") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        llvm::Value *len = callFloatIntrinsic(cg, llvm::Intrinsic::sqrt,
                                              deps.dotProduct(cg, a0, a0));
        return cg.b->CreateFDiv(a0, deps.broadcastTo(cg, len, a0->getType()));
    }
    if (strcmp(name, "dot") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        if (!a0 || !a1) return nullptr;
        return deps.dotProduct(cg, a0, a1);
    }
    if (strcmp(name, "inversesqrt") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        return cg.b->CreateFDiv(fpConstOf(cg, a0->getType(), 1.0),
                                callFloatIntrinsic(cg, llvm::Intrinsic::sqrt,
                                                   a0));
    }
    if (strcmp(name, "tan") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        llvm::Value *s = callFloatIntrinsic(cg, llvm::Intrinsic::sin, a0);
        llvm::Value *c = callFloatIntrinsic(cg, llvm::Intrinsic::cos, a0);
        return cg.b->CreateFDiv(s, c);
    }
    if (strcmp(name, "fract") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        llvm::Value *fl = callFloatIntrinsic(cg, llvm::Intrinsic::floor, a0);
        return cg.b->CreateFSub(a0, fl);
    }
    if (strcmp(name, "sign") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        llvm::Type *t = a0->getType();
        llvm::Value *z = fpConstOf(cg, t, 0.0);
        llvm::Value *pos =
            cg.b->CreateFCmpOGT(a0, z);
        llvm::Value *one = fpConstOf(cg, t, 1.0);
        llvm::Value *neg = cg.b->CreateFCmpOLT(a0, z);
        llvm::Value *mone = fpConstOf(cg, t, -1.0);
        llvm::Value *sn = cg.b->CreateSelect(neg, mone, z);
        return cg.b->CreateSelect(pos, one, sn);
    }
    if (strcmp(name, "radians") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        return cg.b->CreateFMul(
            a0, fpConstOf(cg, a0->getType(),
                          M_PI / 180.0));
    }
    if (strcmp(name, "degrees") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        return cg.b->CreateFMul(
            a0, fpConstOf(cg, a0->getType(),
                          180.0 / M_PI));
    }
    if (strcmp(name, "pow") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        if (!a0 || !a1) return nullptr;
        return cg.b->CreateIntrinsic(llvm::Intrinsic::pow,
                                     {a0->getType()}, {a0, a1});
    }
    if (strcmp(name, "mod") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        if (!a0 || !a1) return nullptr;
        llvm::Type *t = a0->getType();
        if (t->isVectorTy()) a1 = deps.broadcastTo(cg, a1, t);
        llvm::Value *q = callFloatIntrinsic(cg, llvm::Intrinsic::floor,
                                            cg.b->CreateFDiv(a0, a1));
        return cg.b->CreateFSub(a0, cg.b->CreateFMul(q, a1));
    }
    if (strcmp(name, "step") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        if (!a0 || !a1) return nullptr;
        llvm::Type *t = a1->getType();
        if (t->isVectorTy()) a0 = deps.broadcastTo(cg, a0, t);
        llvm::Value *lt = cg.b->CreateFCmpOLT(a1, a0);
        return cg.b->CreateSelect(
            lt, fpConstOf(cg, t, 0.0), fpConstOf(cg, t, 1.0));
    }
    if (strcmp(name, "smoothstep") == 0) {
        if (!need(3)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        a2 = farg(2);
        if (!a0 || !a1 || !a2) return nullptr;
        llvm::Type *t = a2->getType();
        if (t->isVectorTy()) {
            a0 = deps.broadcastTo(cg, a0, t);
            a1 = deps.broadcastTo(cg, a1, t);
        }
        llvm::Value *tt = cg.b->CreateFDiv(cg.b->CreateFSub(a2, a0),
                                           cg.b->CreateFSub(a1, a0));
        llvm::Value *zero = fpConstOf(cg, t, 0.0);
        llvm::Value *one = fpConstOf(cg, t, 1.0);
        llvm::Value *t0 = cg.b->CreateFCmpOLT(tt, zero);
        llvm::Value *t1 = cg.b->CreateFCmpOGT(tt, one);
        tt = cg.b->CreateSelect(t0, zero, tt);
        tt = cg.b->CreateSelect(t1, one, tt);
        llvm::Value *tt2 = cg.b->CreateFMul(tt, tt);
        llvm::Value *three = fpConstOf(cg, t, 3.0);
        llvm::Value *two = fpConstOf(cg, t, 2.0);
        return cg.b->CreateFMul(tt2, cg.b->CreateFSub(three,
                                cg.b->CreateFMul(two, tt)));
    }
    if (strcmp(name, "min") == 0 || strcmp(name, "max") == 0) {
        if (!need(2)) return nullptr;
        a0 = arg(0);
        a1 = arg(1);
        if (!a0 || !a1) return nullptr;
        llvm::Type *t = a0->getType();
        if (typeIsIntLike(t)) {
            a1 = coerceScalar(cg, a1, MGLIR_SCALAR_INT);
            if (t->isVectorTy()) a1 = deps.broadcastTo(cg, a1, t);
            llvm::CmpInst::Predicate p = strcmp(name, "min") == 0
                ? llvm::CmpInst::ICMP_SLT : llvm::CmpInst::ICMP_SGT;
            return cg.b->CreateSelect(cg.b->CreateICmp(p, a0, a1), a0, a1);
        }
        a1 = coerceScalar(cg, a1, MGLIR_SCALAR_FLOAT);
        if (t->isVectorTy()) a1 = deps.broadcastTo(cg, a1, t);
        llvm::Intrinsic::ID id = strcmp(name, "min") == 0
            ? llvm::Intrinsic::minnum : llvm::Intrinsic::maxnum;
        return cg.b->CreateIntrinsic(id, {t}, {a0, a1});
    }
    if (strcmp(name, "clamp") == 0 || strcmp(name, "mix") == 0) {
        if (!need(3)) return nullptr;
        a0 = arg(0);
        a1 = arg(1);
        a2 = arg(2);
        if (!a0 || !a1 || !a2) return nullptr;
        llvm::Type *t = a0->getType();
        if (strcmp(name, "clamp") == 0) {
            if (typeIsIntLike(t)) {
                a1 = coerceScalar(cg, a1, MGLIR_SCALAR_INT);
                a2 = coerceScalar(cg, a2, MGLIR_SCALAR_INT);
                if (t->isVectorTy()) {
                    a1 = deps.broadcastTo(cg, a1, t);
                    a2 = deps.broadcastTo(cg, a2, t);
                }
                llvm::Value *mx = cg.b->CreateSelect(
                    cg.b->CreateICmp(llvm::CmpInst::ICMP_SGT, a0, a1), a0, a1);
                return cg.b->CreateSelect(
                    cg.b->CreateICmp(llvm::CmpInst::ICMP_SLT, mx, a2), mx,
                    a2);
            }
            a1 = coerceScalar(cg, a1, MGLIR_SCALAR_FLOAT);
            a2 = coerceScalar(cg, a2, MGLIR_SCALAR_FLOAT);
            if (t->isVectorTy()) {
                a1 = deps.broadcastTo(cg, a1, t);
                a2 = deps.broadcastTo(cg, a2, t);
            }
            llvm::Value *mx = cg.b->CreateIntrinsic(
                llvm::Intrinsic::maxnum, {t}, {a0, a1});
            return cg.b->CreateIntrinsic(llvm::Intrinsic::minnum, {t},
                                         {mx, a2});
        }
        /* mix(x, y, a) = fma(a, y, x * (1 - a)) */
        a1 = coerceScalar(cg, a1, MGLIR_SCALAR_FLOAT);
        a2 = coerceScalar(cg, a2, MGLIR_SCALAR_FLOAT);
        if (t->isVectorTy()) a2 = deps.broadcastTo(cg, a2, t);
        llvm::Value *one = fpConstOf(cg, t, 1.0);
        llvm::Value *s = cg.b->CreateFSub(one, a2);
        llvm::Value *term = cg.b->CreateFMul(a0, s);
        return cg.b->CreateIntrinsic(llvm::Intrinsic::fma, {t},
                                     {a1, a2, term});
    }
    if (strcmp(name, "abs") == 0) {
        if (!need(1)) return nullptr;
        a0 = arg(0);
        if (!a0) return nullptr;
        llvm::Type *t = a0->getType();
        if (typeIsIntLike(t)) {
            llvm::Value *neg = cg.b->CreateNeg(a0);
            return cg.b->CreateSelect(
                cg.b->CreateICmp(llvm::CmpInst::ICMP_SLT, a0,
                                 llvm::Constant::getNullValue(t)),
                neg, a0);
        }
        a0 = coerceScalar(cg, a0, MGLIR_SCALAR_FLOAT);
        return callFloatIntrinsic(cg, llvm::Intrinsic::fabs, a0);
    }
    if (strcmp(name, "reflect") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        if (!a0 || !a1) return nullptr;
        llvm::Value *d = deps.dotProduct(cg, a1, a0);
        d = cg.b->CreateFMul(d, fpConstOf(cg, d->getType(), 2.0));
        d = deps.broadcastTo(cg, d, a0->getType());
        llvm::Value *p = cg.b->CreateFMul(d, a1);
        return cg.b->CreateFSub(a0, p);
    }
    if (strcmp(name, "refract") == 0) {
        if (!need(3)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        a2 = farg(2);
        if (!a0 || !a1 || !a2) return nullptr;
        llvm::Type *t = a0->getType();
        llvm::Value *d = deps.dotProduct(cg, a1, a0);  /* scalar float */
        /* k = 1 - eta^2 * (1 - d^2);  r = eta*I - (eta*d + sqrt(k))*N */
        llvm::Constant *fone =
            llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 1.0);
        llvm::Value *k = cg.b->CreateFSub(
            fone,
            cg.b->CreateFMul(
                cg.b->CreateFMul(a2, a2),
                cg.b->CreateFSub(fone, cg.b->CreateFMul(d, d))));
        llvm::Value *kNeg = cg.b->CreateFCmpOLT(
            k, llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 0.0));
        llvm::Value *sk = callFloatIntrinsic(cg, llvm::Intrinsic::sqrt, k);
        llvm::Value *sc = cg.b->CreateFAdd(cg.b->CreateFMul(a2, d), sk);
        llvm::Value *r = cg.b->CreateFSub(
            cg.b->CreateFMul(a2, a0),
            cg.b->CreateFMul(deps.broadcastTo(cg, sc, t), a1));
        llvm::Value *zeroV = fpConstOf(cg, t, 0.0);
        if (t->isVectorTy()) {
            auto *vt = llvm::cast<llvm::FixedVectorType>(t);
            llvm::Value *mask = cg.b->CreateVectorSplat(
                (uint32_t)vt->getElementCount().getFixedValue(), kNeg);
            return cg.b->CreateSelect(mask, zeroV, r);
        }
        return cg.b->CreateSelect(kNeg, zeroV, r);
    }
    if (strcmp(name, "faceforward") == 0) {
        if (!need(3)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        a2 = farg(2);
        if (!a0 || !a1 || !a2) return nullptr;
        llvm::Value *d = deps.dotProduct(cg, a2, a1);
        llvm::Value *neg = cg.b->CreateFNeg(a0);
        return cg.b->CreateSelect(
            cg.b->CreateFCmpOLT(d, fpConstOf(cg, d->getType(), 0.0)),
            a0, neg);
    }
    if (strcmp(name, "cross") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        if (!a0 || !a1) return nullptr;
        auto cI = [&](uint32_t v) {
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), v);
        };
        auto sw = [&](llvm::Value *v, uint32_t i) {
            return cg.b->CreateExtractElement(v, cI(i));
        };
        llvm::Value *x = cg.b->CreateFSub(
            cg.b->CreateFMul(sw(a0, 1), sw(a1, 2)),
            cg.b->CreateFMul(sw(a0, 2), sw(a1, 1)));
        llvm::Value *y = cg.b->CreateFSub(
            cg.b->CreateFMul(sw(a0, 2), sw(a1, 0)),
            cg.b->CreateFMul(sw(a0, 0), sw(a1, 2)));
        llvm::Value *z = cg.b->CreateFSub(
            cg.b->CreateFMul(sw(a0, 0), sw(a1, 1)),
            cg.b->CreateFMul(sw(a0, 1), sw(a1, 0)));
        llvm::Value *r = llvm::UndefValue::get(a0->getType());
        r = cg.b->CreateInsertElement(r, x, cI(0));
        r = cg.b->CreateInsertElement(r, y, cI(1));
        r = cg.b->CreateInsertElement(r, z, cI(2));
        return r;
    }
    /* asin/acos/atan/atan(y,x): no LLVM intrinsics; AIR declares
     * air.fast_* entry points.  Vectors call the scalar variant per
     * component. */
    if (strcmp(name, "asin") == 0 || strcmp(name, "acos") == 0 ||
        strcmp(name, "atan") == 0) {
        uint32_t want = (strcmp(name, "atan") == 0 &&
                         e->u.call.arg_count == 2) ? 2 : 1;
        if (!need(want)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        llvm::Value *a1v = nullptr;
        if (want == 2) {
            a1v = farg(1);
            if (!a1v) return nullptr;
        }
        const char *airfn =
            strcmp(name, "asin") == 0   ? "air.fast_asin.f32"
            : strcmp(name, "acos") == 0 ? "air.fast_acos.f32"
            : want == 2                 ? "air.fast_atan2.f32"
                                        : "air.fast_atan.f32";
        llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
        llvm::Type *retT = a0->getType();
        auto cI = [&](uint32_t v) {
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), v);
        };
        if (retT->isVectorTy()) {
            auto *vt = llvm::cast<llvm::FixedVectorType>(retT);
            uint32_t n = vt->getElementCount().getFixedValue();
            llvm::Value *r = llvm::UndefValue::get(retT);
            for (uint32_t i = 0; i < n; i++) {
                llvm::Value *x = cg.b->CreateExtractElement(a0, cI(i));
                if (want == 2) {
                    llvm::Value *y = cg.b->CreateExtractElement(a1v, cI(i));
                    x = deps.callAirFn(cg, airfn, f32, {x, y});
                } else {
                    x = deps.callAirFn(cg, airfn, f32, {x});
                }
                r = cg.b->CreateInsertElement(r, x, cI(i));
            }
            return r;
        }
        if (want == 2) return deps.callAirFn(cg, airfn, f32, {a0, a1v});
        return deps.callAirFn(cg, airfn, f32, {a0});
    }
    /* pack/unpack (GLSL 4.60 8.4): AIR intrinsics. */
    if (strcmp(name, "packUnorm2x16") == 0 ||
        strcmp(name, "packSnorm2x16") == 0 ||
        strcmp(name, "unpackUnorm2x16") == 0 ||
        strcmp(name, "unpackSnorm2x16") == 0) {
        if (!need(1)) return nullptr;
        llvm::Value *av = arg(0);
        if (!av) return nullptr;
        llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        bool pack = name[0] == 'p';
        bool unorm = strcmp(name, "packUnorm2x16") == 0 ||
                     strcmp(name, "unpackUnorm2x16") == 0;
        const char *airfn =
            pack ? (unorm ? "air.pack.unorm2x16.v2f32"
                          : "air.pack.snorm2x16.v2f32")
                 : (unorm ? "air.unpack.unorm2x16.v2f32"
                          : "air.unpack.snorm2x16.v2f32");
        if (pack) return deps.callAirFn(cg, airfn, i32, {av});
        return deps.callAirFn(cg, airfn, llvm::FixedVectorType::get(f32, 2), {av});
    }
    if (strcmp(name, "packHalf2x16") == 0 ||
        strcmp(name, "unpackHalf2x16") == 0) {
        if (!need(1)) return nullptr;
        llvm::Value *av = arg(0);
        if (!av) return nullptr;
        llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
        llvm::Type *f16 = llvm::Type::getHalfTy(*cg.ctx);
        llvm::Type *v2f16 = llvm::FixedVectorType::get(f16, 2);
        llvm::Type *v2f32 = llvm::FixedVectorType::get(f32, 2);
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        if (strcmp(name, "packHalf2x16") == 0) {
            av = coerceScalar(cg, av, MGLIR_SCALAR_FLOAT);
            llvm::Value *h = deps.callAirFn(cg, "air.convert.f.v2f16.f.v2f32",
                                       v2f16, {av});
            return cg.b->CreateBitCast(h, i32);
        }
        llvm::Value *h = cg.b->CreateBitCast(av, v2f16);
        return deps.callAirFn(cg, "air.convert.f.v2f32.f.v2f16", v2f32, {h});
    }

    /* ---- integer / bitfield builtins (GLSL 4.60 §8.8 / §8.3) ---- */
    auto storeOut = [&](uint32_t argIndex, llvm::Value *val) -> bool {
        if (argIndex >= e->u.call.arg_count || !val) {
            cg.err = 1;
            cg.errmsg = "codegen: missing out-parameter for builtin";
            return false;
        }
        const MGLExpr *dst = e->u.call.args[argIndex];
        if (const MGLIRSymbol *sb = deps.ssboRootSym(dst, mod)) {
            deps.emitSSBOWrite(cg, dst, sb, mod, locals, val);
            return !cg.err;
        }
        const MGLExpr *rootE = dst;
        while (rootE && (rootE->kind == MGL_EXPR_INDEX ||
                         rootE->kind == MGL_EXPR_MEMBER)) {
            rootE = (rootE->kind == MGL_EXPR_INDEX)
                        ? rootE->u.index.object
                        : rootE->u.member.object;
        }
        if (!rootE || rootE->kind != MGL_EXPR_VAR_REF) {
            cg.err = 1;
            cg.errmsg = "codegen: out-parameter must be an lvalue";
            return false;
        }
        const char *name = rootE->u.var_ref.name;
        if (!cg.lvalues.count(name)) {
            llvm::Type *aggTy = nullptr;
            auto lit = locals.find(name);
            if (lit != locals.end())
                aggTy = llvmType(lit->second, *cg.ctx);
            else if (const MGLIRSymbol *sym = deps.findSymbol(mod, name))
                aggTy = llvmType(typeFromIR(sym->type), *cg.ctx);
            if (!aggTy) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: unknown out lvalue '") +
                            name + "'";
                return false;
            }
            cg.lvalues[name] = llvm::UndefValue::get(aggTy);
        }
        llvm::Value *nv =
            deps.updateIndexPath(cg, dst, cg.lvalues[name], val, mod, locals);
        if (!nv) return false;
        cg.lvalues[name] = nv;
        return true;
    };
    auto asSignedIntTy = [&](llvm::Value *v) -> llvm::Value * {
        llvm::Type *t = v->getType();
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
            llvm::Type *dst = llvm::FixedVectorType::get(
                llvm::Type::getInt32Ty(*cg.ctx), vt->getNumElements());
            return cg.b->CreateBitCast(v, dst);
        }
        return cg.b->CreateBitCast(v, llvm::Type::getInt32Ty(*cg.ctx));
    };

    if (strcmp(name, "bitCount") == 0) {
        if (!need(1)) return nullptr;
        a0 = arg(0);
        if (!a0) return nullptr;
        llvm::Value *c =
            cg.b->CreateIntrinsic(llvm::Intrinsic::ctpop, {a0->getType()},
                                  {a0});
        return asSignedIntTy(c);
    }
    if (strcmp(name, "bitfieldReverse") == 0) {
        if (!need(1)) return nullptr;
        a0 = arg(0);
        if (!a0) return nullptr;
        return cg.b->CreateIntrinsic(llvm::Intrinsic::bitreverse,
                                     {a0->getType()}, {a0});
    }
    if (strcmp(name, "findLSB") == 0) {
        if (!need(1)) return nullptr;
        a0 = arg(0);
        if (!a0) return nullptr;
        llvm::Type *t = a0->getType();
        llvm::Value *zero = llvm::Constant::getNullValue(t);
        llvm::Value *isZero = cg.b->CreateICmpEQ(a0, zero);
        llvm::Value *tz = cg.b->CreateIntrinsic(
            llvm::Intrinsic::cttz, {t}, {a0, cg.b->getInt1(false)});
        llvm::Value *neg1 = llvm::Constant::getAllOnesValue(
            llvm::Type::getInt32Ty(*cg.ctx));
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
            neg1 = cg.b->CreateVectorSplat(vt->getNumElements(), neg1);
            llvm::Type *i32v = llvm::FixedVectorType::get(
                llvm::Type::getInt32Ty(*cg.ctx), vt->getNumElements());
            tz = cg.b->CreateBitCast(tz, i32v);
        } else {
            tz = cg.b->CreateBitCast(tz, llvm::Type::getInt32Ty(*cg.ctx));
        }
        return cg.b->CreateSelect(isZero, neg1, tz);
    }
    if (strcmp(name, "findMSB") == 0) {
        if (!need(1)) return nullptr;
        a0 = arg(0);
        if (!a0) return nullptr;
        llvm::Type *t = a0->getType();
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        MType at = deps.exprType(cg, e->u.call.args[0], mod, locals);
        bool isSigned = (at.scalar == MGLIR_SCALAR_INT);
        llvm::Value *zero = llvm::Constant::getNullValue(t);
        llvm::Value *neg1i = llvm::Constant::getAllOnesValue(i32);
        llvm::Value *c31 = llvm::ConstantInt::get(i32, 31);
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
            neg1i = cg.b->CreateVectorSplat(vt->getNumElements(), neg1i);
            c31 = cg.b->CreateVectorSplat(vt->getNumElements(), c31);
        }
        llvm::Value *src = a0;
        llvm::Value *invalid = cg.b->CreateICmpEQ(a0, zero);
        if (isSigned) {
            llvm::Value *allOnes = llvm::Constant::getAllOnesValue(t);
            invalid = cg.b->CreateOr(invalid,
                                     cg.b->CreateICmpEQ(a0, allOnes));
            llvm::Value *neg = cg.b->CreateICmpSLT(
                a0, llvm::Constant::getNullValue(t));
            src = cg.b->CreateSelect(neg, cg.b->CreateNot(a0), a0);
        }
        llvm::Value *lz = cg.b->CreateIntrinsic(
            llvm::Intrinsic::ctlz, {t}, {src, cg.b->getInt1(false)});
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
            lz = cg.b->CreateBitCast(
                lz, llvm::FixedVectorType::get(i32, vt->getNumElements()));
        } else {
            lz = cg.b->CreateBitCast(lz, i32);
        }
        llvm::Value *msb = cg.b->CreateSub(c31, lz);
        return cg.b->CreateSelect(invalid, neg1i, msb);
    }
    if (strcmp(name, "bitfieldExtract") == 0) {
        if (!need(3)) return nullptr;
        a0 = arg(0);
        a1 = arg(1);
        a2 = arg(2);
        if (!a0 || !a1 || !a2) return nullptr;
        a1 = coerceScalar(cg, a1, MGLIR_SCALAR_INT);
        a2 = coerceScalar(cg, a2, MGLIR_SCALAR_INT);
        llvm::Type *t = a0->getType();
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        MType at = deps.exprType(cg, e->u.call.args[0], mod, locals);
        bool isSigned = (at.scalar == MGLIR_SCALAR_INT);
        llvm::Value *off = a1;
        llvm::Value *bits = a2;
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
            off = cg.b->CreateVectorSplat(vt->getNumElements(), off);
            bits = cg.b->CreateVectorSplat(vt->getNumElements(), bits);
        }
        llvm::Value *zero = llvm::Constant::getNullValue(t);
        llvm::Value *bitsZero = cg.b->CreateICmpEQ(bits, zero);
        llvm::Value *c32 = llvm::ConstantInt::get(i32, 32);
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t))
            c32 = cg.b->CreateVectorSplat(vt->getNumElements(), c32);
        if (isSigned) {
            llvm::Value *shlAmt = cg.b->CreateSub(c32, cg.b->CreateAdd(off, bits));
            llvm::Value *ashrAmt = cg.b->CreateSub(c32, bits);
            llvm::Value *tmp = cg.b->CreateShl(a0, shlAmt);
            llvm::Value *ext = cg.b->CreateAShr(tmp, ashrAmt);
            return cg.b->CreateSelect(bitsZero, zero, ext);
        }
        llvm::Value *one = llvm::ConstantInt::get(i32, 1);
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t))
            one = cg.b->CreateVectorSplat(vt->getNumElements(), one);
        llvm::Value *full = cg.b->CreateICmpEQ(bits, c32);
        llvm::Value *zeroBits = llvm::Constant::getNullValue(bits->getType());
        llvm::Value *safeBits = cg.b->CreateSelect(full, zeroBits, bits);
        llvm::Value *mask =
            cg.b->CreateSub(cg.b->CreateShl(one, safeBits), one);
        llvm::Value *allOnes = llvm::Constant::getAllOnesValue(t);
        mask = cg.b->CreateSelect(full, allOnes, mask);
        llvm::Value *shifted = cg.b->CreateLShr(a0, off);
        llvm::Value *ext = cg.b->CreateAnd(shifted, mask);
        ext = cg.b->CreateSelect(full, shifted, ext);
        return cg.b->CreateSelect(bitsZero, zero, ext);
    }
    if (strcmp(name, "bitfieldInsert") == 0) {
        if (!need(4)) return nullptr;
        a0 = arg(0);
        a1 = arg(1);
        a2 = arg(2);
        llvm::Value *a3 = arg(3);
        if (!a0 || !a1 || !a2 || !a3) return nullptr;
        a2 = coerceScalar(cg, a2, MGLIR_SCALAR_INT);
        a3 = coerceScalar(cg, a3, MGLIR_SCALAR_INT);
        llvm::Type *t = a0->getType();
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        llvm::Value *off = a2;
        llvm::Value *bits = a3;
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
            off = cg.b->CreateVectorSplat(vt->getNumElements(), off);
            bits = cg.b->CreateVectorSplat(vt->getNumElements(), bits);
        }
        llvm::Value *zero = llvm::Constant::getNullValue(t);
        llvm::Value *bitsZero = cg.b->CreateICmpEQ(bits, zero);
        llvm::Value *one = llvm::ConstantInt::get(i32, 1);
        llvm::Value *c32 = llvm::ConstantInt::get(i32, 32);
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
            one = cg.b->CreateVectorSplat(vt->getNumElements(), one);
            c32 = cg.b->CreateVectorSplat(vt->getNumElements(), c32);
        }
        llvm::Value *full = cg.b->CreateICmpEQ(bits, c32);
        llvm::Value *zeroBits = llvm::Constant::getNullValue(bits->getType());
        llvm::Value *safeBits = cg.b->CreateSelect(full, zeroBits, bits);
        llvm::Value *mask = cg.b->CreateShl(
            cg.b->CreateSub(cg.b->CreateShl(one, safeBits), one), off);
        mask = cg.b->CreateSelect(full,
                                  llvm::Constant::getAllOnesValue(t), mask);
        llvm::Value *insert = cg.b->CreateAnd(cg.b->CreateShl(a1, off), mask);
        llvm::Value *base = cg.b->CreateAnd(a0, cg.b->CreateNot(mask));
        llvm::Value *r = cg.b->CreateOr(base, insert);
        return cg.b->CreateSelect(bitsZero, a0, r);
    }
    if (strcmp(name, "uaddCarry") == 0 || strcmp(name, "usubBorrow") == 0) {
        if (!need(3)) return nullptr;
        a0 = arg(0);
        a1 = arg(1);
        if (!a0 || !a1) return nullptr;
        bool isAdd = strcmp(name, "uaddCarry") == 0;
        llvm::Value *sum = isAdd ? cg.b->CreateAdd(a0, a1)
                                 : cg.b->CreateSub(a0, a1);
        llvm::Value *flag =
            isAdd ? cg.b->CreateICmpULT(sum, a0)
                  : cg.b->CreateICmpUGT(a1, a0);
        llvm::Value *one = llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(*cg.ctx), 1);
        llvm::Value *zero = llvm::Constant::getNullValue(a0->getType());
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(a0->getType()))
            one = cg.b->CreateVectorSplat(vt->getNumElements(), one);
        llvm::Value *carry = cg.b->CreateSelect(flag, one, zero);
        if (!storeOut(2, carry)) return nullptr;
        return sum;
    }
    if (strcmp(name, "umulExtended") == 0 ||
        strcmp(name, "imulExtended") == 0) {
        if (!need(4)) return nullptr;
        a0 = arg(0);
        a1 = arg(1);
        if (!a0 || !a1) return nullptr;
        bool isSigned = strcmp(name, "imulExtended") == 0;
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        llvm::Type *i64 = llvm::Type::getInt64Ty(*cg.ctx);
        auto widenMul = [&](llvm::Value *x, llvm::Value *y) {
            llvm::Value *xx = isSigned ? cg.b->CreateSExt(x, i64)
                                       : cg.b->CreateZExt(x, i64);
            llvm::Value *yy = isSigned ? cg.b->CreateSExt(y, i64)
                                       : cg.b->CreateZExt(y, i64);
            llvm::Value *p = cg.b->CreateMul(xx, yy);
            llvm::Value *lsb = cg.b->CreateTrunc(p, i32);
            llvm::Value *msb = cg.b->CreateTrunc(
                cg.b->CreateLShr(p, llvm::ConstantInt::get(i64, 32)), i32);
            return std::pair<llvm::Value *, llvm::Value *>{msb, lsb};
        };
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(a0->getType())) {
            uint32_t n = vt->getNumElements();
            llvm::Value *msbV = llvm::UndefValue::get(a0->getType());
            llvm::Value *lsbV = llvm::UndefValue::get(a0->getType());
            for (uint32_t i = 0; i < n; i++) {
                llvm::Value *xi = cg.b->CreateExtractElement(a0, i);
                llvm::Value *yi = cg.b->CreateExtractElement(a1, i);
                auto p = widenMul(xi, yi);
                msbV = cg.b->CreateInsertElement(msbV, p.first, i);
                lsbV = cg.b->CreateInsertElement(lsbV, p.second, i);
            }
            if (!storeOut(2, msbV) || !storeOut(3, lsbV)) return nullptr;
        } else {
            auto p = widenMul(a0, a1);
            if (!storeOut(2, p.first) || !storeOut(3, p.second))
                return nullptr;
        }
        return cg.b->getInt32(0);
    }
    if (strcmp(name, "ldexp") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        a1 = arg(1);
        if (!a0 || !a1) return nullptr;
        a1 = coerceScalar(cg, a1, MGLIR_SCALAR_INT);
        /* Scalar llvm.exp2(sitofp(SSBO int)) has been observed to lower to
         * a dead Metal pipeline on AGX; vector exp2 is reliable.  Widen
         * scalar to <2 x float>, then extract. */
        llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
        if (!a0->getType()->isVectorTy()) {
            llvm::Value *ef = cg.b->CreateSIToFP(a1, f32);
            llvm::Type *v2 = llvm::FixedVectorType::get(f32, 2);
            llvm::Value *efv = cg.b->CreateVectorSplat(2, ef);
            llvm::Value *scalev =
                callFloatIntrinsic(cg, llvm::Intrinsic::exp2, efv);
            llvm::Value *scale = cg.b->CreateExtractElement(
                scalev, (uint64_t)0);
            return cg.b->CreateFMul(a0, scale);
        }
        llvm::Value *ef = cg.b->CreateSIToFP(a1, a0->getType());
        llvm::Value *scale =
            callFloatIntrinsic(cg, llvm::Intrinsic::exp2, ef);
        return cg.b->CreateFMul(a0, scale);
    }
    if (strcmp(name, "frexp") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        auto frexp1 = [&](llvm::Value *x) -> std::pair<llvm::Value *, llvm::Value *> {
            llvm::Value *ax =
                callFloatIntrinsic(cg, llvm::Intrinsic::fabs, x);
            llvm::Value *isZero = cg.b->CreateFCmpOEQ(
                ax, llvm::ConstantFP::get(f32, 0.0));
            llvm::Value *lg =
                callFloatIntrinsic(cg, llvm::Intrinsic::log2, ax);
            llvm::Value *fl =
                callFloatIntrinsic(cg, llvm::Intrinsic::floor, lg);
            llvm::Value *eF = cg.b->CreateFAdd(
                fl, llvm::ConstantFP::get(f32, 1.0));
            llvm::Value *eI = cg.b->CreateFPToSI(eF, i32);
            llvm::Value *scale = callFloatIntrinsic(
                cg, llvm::Intrinsic::exp2, cg.b->CreateFNeg(eF));
            llvm::Value *m = cg.b->CreateFMul(x, scale);
            m = cg.b->CreateSelect(isZero, llvm::ConstantFP::get(f32, 0.0),
                                   m);
            eI = cg.b->CreateSelect(isZero, cg.b->getInt32(0), eI);
            return {m, eI};
        };
        if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(a0->getType())) {
            uint32_t n = vt->getNumElements();
            llvm::Type *iv =
                llvm::FixedVectorType::get(i32, n);
            llvm::Value *mV = llvm::UndefValue::get(a0->getType());
            llvm::Value *eV = llvm::UndefValue::get(iv);
            for (uint32_t i = 0; i < n; i++) {
                llvm::Value *xi = cg.b->CreateExtractElement(a0, i);
                auto p = frexp1(xi);
                mV = cg.b->CreateInsertElement(mV, p.first, i);
                eV = cg.b->CreateInsertElement(eV, p.second, i);
            }
            if (!storeOut(1, eV)) return nullptr;
            return mV;
        }
        auto p = frexp1(a0);
        if (!storeOut(1, p.second)) return nullptr;
        return p.first;
    }
    if (strcmp(name, "packUnorm4x8") == 0 ||
        strcmp(name, "packSnorm4x8") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        bool unorm = strcmp(name, "packUnorm4x8") == 0;
        /* Prefer AIR pack when available; fall back to manual byte pack. */
        const char *airfn =
            unorm ? "air.pack.unorm4x8.v4f32" : "air.pack.snorm4x8.v4f32";
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
        /* Manual path (always correct; used when AIR name mismatches). */
        auto packLane = [&](llvm::Value *x, float scale) -> llvm::Value * {
            llvm::Value *lo = cg.b->CreateIntrinsic(
                llvm::Intrinsic::maxnum, {f32},
                {x, llvm::ConstantFP::get(f32, unorm ? 0.0 : -1.0)});
            llvm::Value *cl = cg.b->CreateIntrinsic(
                llvm::Intrinsic::minnum, {f32},
                {lo, llvm::ConstantFP::get(f32, 1.0)});
            /* CTS / GLSL pack*4x8: floor(c * range + 0.5). */
            llvm::Value *s = cg.b->CreateFMul(
                cl, llvm::ConstantFP::get(f32, scale));
            llvm::Value *biased = cg.b->CreateFAdd(
                s, llvm::ConstantFP::get(f32, 0.5));
            llvm::Value *r =
                callFloatIntrinsic(cg, llvm::Intrinsic::floor, biased);
            llvm::Value *iv = cg.b->CreateFPToSI(r, i32);
            return cg.b->CreateAnd(iv, cg.b->getInt32(0xff));
        };
        float scale = unorm ? 255.0f : 127.0f;
        llvm::Value *b0 =
            packLane(cg.b->CreateExtractElement(a0, (uint64_t)0), scale);
        llvm::Value *b1 =
            packLane(cg.b->CreateExtractElement(a0, (uint64_t)1), scale);
        llvm::Value *b2 =
            packLane(cg.b->CreateExtractElement(a0, (uint64_t)2), scale);
        llvm::Value *b3 =
            packLane(cg.b->CreateExtractElement(a0, (uint64_t)3), scale);
        llvm::Value *p = b0;
        p = cg.b->CreateOr(p, cg.b->CreateShl(b1, cg.b->getInt32(8)));
        p = cg.b->CreateOr(p, cg.b->CreateShl(b2, cg.b->getInt32(16)));
        p = cg.b->CreateOr(p, cg.b->CreateShl(b3, cg.b->getInt32(24)));
        (void)airfn;
        return p;
    }
    if (strcmp(name, "unpackUnorm4x8") == 0 ||
        strcmp(name, "unpackSnorm4x8") == 0) {
        if (!need(1)) return nullptr;
        a0 = arg(0);
        if (!a0) return nullptr;
        a0 = coerceScalar(cg, a0, MGLIR_SCALAR_UINT);
        bool unorm = strcmp(name, "unpackUnorm4x8") == 0;
        llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        llvm::Type *v4f32 = llvm::FixedVectorType::get(f32, 4);
        auto unpackLane = [&](uint32_t shift) -> llvm::Value * {
            llvm::Value *b = cg.b->CreateAnd(
                cg.b->CreateLShr(a0, cg.b->getInt32(shift)),
                cg.b->getInt32(0xff));
            if (unorm) {
                return cg.b->CreateFDiv(
                    cg.b->CreateUIToFP(b, f32),
                    llvm::ConstantFP::get(f32, 255.0));
            }
            /* snorm: byte as signed int8 */
            llvm::Value *sb = cg.b->CreateTrunc(
                b, llvm::Type::getInt8Ty(*cg.ctx));
            llvm::Value *si = cg.b->CreateSExt(sb, i32);
            llvm::Value *f = cg.b->CreateSIToFP(si, f32);
            llvm::Value *d = cg.b->CreateFDiv(
                f, llvm::ConstantFP::get(f32, 127.0));
            return cg.b->CreateIntrinsic(
                llvm::Intrinsic::maxnum, {f32},
                {d, llvm::ConstantFP::get(f32, -1.0)});
        };
        llvm::Value *r = llvm::UndefValue::get(v4f32);
        r = cg.b->CreateInsertElement(r, unpackLane(0), (uint64_t)0);
        r = cg.b->CreateInsertElement(r, unpackLane(8), (uint64_t)1);
        r = cg.b->CreateInsertElement(r, unpackLane(16), (uint64_t)2);
        r = cg.b->CreateInsertElement(r, unpackLane(24), (uint64_t)3);
        return r;
    }
    return nullptr;
}


} /* namespace air */
} /* namespace mgl */
