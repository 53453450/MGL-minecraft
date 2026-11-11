/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

#include "mgl_air_expr_support.h"

#include <cstring>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ConstantFold.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

namespace mgl {
namespace air {

const MGLIRSymbol *findSymbol(const MGLIRModule *mod, const char *name) {
    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        if (!mod->symbols[i]->is_function &&
            strcmp(mod->symbols[i]->name, name) == 0)
            return mod->symbols[i];
    }
    return nullptr;
}

bool swizzleIndices(const char *field, std::vector<uint32_t> *out) {
    static const char *valid = "xyzwrgbastpq";
    out->clear();
    for (const char *p = field; *p; p++) {
        const char *f = strchr(valid, *p);
        if (!f) return false;
        out->push_back((uint32_t)(f - valid) % 4);
    }
    return !out->empty();
}

MType swizzleType(const MType &base, size_t lanes) {
    MType t = base;
    if (base.isMatrix()) return base; /* unsupported, keep */
    /* GLSL 4.60 5.5: single-component swizzle yields a scalar. */
    t.vec = lanes == 1 ? 0 : (uint32_t)lanes;
    return t;
}

llvm::Value *emitSwizzleValue(Codegen &cg, llvm::Value *obj,
                              const std::vector<uint32_t> &idx)
{
    if (!obj || idx.empty()) return nullptr;
    if (!obj->getType()->isVectorTy()) {
        if (idx.size() == 1) return obj;
        llvm::Type *vt = llvm::FixedVectorType::get(
            obj->getType(), (unsigned)idx.size());
        llvm::Value *v = llvm::UndefValue::get(vt);
        for (unsigned i = 0; i < idx.size(); i++)
            v = cg.b->CreateInsertElement(v, obj, cg.b->getInt32(i));
        return v;
    }
    if (idx.size() == 1)
        return cg.b->CreateExtractElement(
            obj, llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx),
                                        idx[0]));
    llvm::SmallVector<llvm::Constant *, 4> mask;
    for (uint32_t i : idx)
        mask.push_back(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(*cg.ctx), i));
    return cg.b->CreateShuffleVector(
        obj, llvm::UndefValue::get(obj->getType()),
        llvm::ConstantVector::get(mask));
}

/* Constant-fold a numeric binary op when both operands are constants of
 * the same type; returns null when folding is not possible.  Mirrors the
 * runtime signedness/comparison semantics of emitNumericBinOp. */
/* GLSL 4.60 §5.9: == / != on vector operands yield a scalar bool
 * (all lanes equal / any lane differs).  Relational operators keep
 * their vector (bvec) results for any()/all()/equal(). */
llvm::Value *scalarizeBoolCompare(Codegen &cg, uint32_t op,
                                  llvm::Value *cmp) {
    if (op != MGL_OP_EQ && op != MGL_OP_NE) return cmp;
    if (!cmp->getType()->isVectorTy()) return cmp;
    auto *vt = llvm::cast<llvm::FixedVectorType>(cmp->getType());
    uint32_t n = (uint32_t)vt->getElementCount().getFixedValue();
    llvm::Value *acc = cg.b->CreateExtractElement(cmp, (uint64_t)0);
    for (uint32_t i = 1; i < n; i++) {
        llvm::Value *lane = cg.b->CreateExtractElement(cmp, (uint64_t)i);
        acc = op == MGL_OP_EQ ? cg.b->CreateAnd(acc, lane)
                              : cg.b->CreateOr(acc, lane);
    }
    return acc;
}

llvm::Value *tryFoldConst(Codegen &cg, uint32_t op, llvm::Value *l,
                          llvm::Value *r, bool uns) {
    auto *lc = llvm::dyn_cast<llvm::Constant>(l);
    auto *rc = llvm::dyn_cast<llvm::Constant>(r);
    if (!lc || !rc) return nullptr;
    if (l->getType() != r->getType()) return nullptr;
    /* Aggregate ==/!= must walk members; ConstantExpr::getCompare on
     * struct/array constants infinite-recurses in LLVM's folder. */
    if (l->getType()->isStructTy() || l->getType()->isArrayTy())
        return nullptr;
    bool fp = l->getType()->isFPOrFPVectorTy();
    if (op == MGL_OP_LAND || op == MGL_OP_LOR) {
        if (!l->getType()->isIntegerTy(1)) return nullptr;
        unsigned llo = op == MGL_OP_LAND ? llvm::Instruction::And
                                         : llvm::Instruction::Or;
        return llvm::ConstantFoldBinaryInstruction(llo, lc, rc);
    }
    unsigned llo;
    switch (op) {
    case MGL_OP_ADD: llo = fp ? llvm::Instruction::FAdd
                              : llvm::Instruction::Add; break;
    case MGL_OP_SUB: llo = fp ? llvm::Instruction::FSub
                              : llvm::Instruction::Sub; break;
    case MGL_OP_MUL: llo = fp ? llvm::Instruction::FMul
                              : llvm::Instruction::Mul; break;
    case MGL_OP_DIV: llo = fp ? llvm::Instruction::FDiv
                 : uns ? llvm::Instruction::UDiv
                       : llvm::Instruction::SDiv; break;
    case MGL_OP_MOD: llo = fp ? llvm::Instruction::FRem
                 : uns ? llvm::Instruction::URem
                       : llvm::Instruction::SRem; break;
    case MGL_OP_SHL: llo = llvm::Instruction::Shl; break;
    case MGL_OP_SHR: llo = uns ? llvm::Instruction::LShr
                               : llvm::Instruction::AShr; break;
    case MGL_OP_AND: llo = llvm::Instruction::And; break;
    case MGL_OP_OR:  llo = llvm::Instruction::Or; break;
    case MGL_OP_XOR: llo = llvm::Instruction::Xor; break;
    case MGL_OP_EQ:
        return scalarizeBoolCompare(cg, op,
            llvm::ConstantExpr::getCompare(fp ? llvm::CmpInst::FCMP_OEQ
                                              : llvm::CmpInst::ICMP_EQ,
                                           lc, rc));
    case MGL_OP_NE:
        return scalarizeBoolCompare(cg, op,
            llvm::ConstantExpr::getCompare(fp ? llvm::CmpInst::FCMP_UNE
                                              : llvm::CmpInst::ICMP_NE,
                                           lc, rc));
    case MGL_OP_LT:
        return llvm::ConstantExpr::getCompare(
            fp ? llvm::CmpInst::FCMP_OLT
               : uns ? llvm::CmpInst::ICMP_ULT : llvm::CmpInst::ICMP_SLT,
            lc, rc);
    case MGL_OP_GT:
        return llvm::ConstantExpr::getCompare(
            fp ? llvm::CmpInst::FCMP_OGT
               : uns ? llvm::CmpInst::ICMP_UGT : llvm::CmpInst::ICMP_SGT,
            lc, rc);
    case MGL_OP_LE:
        return llvm::ConstantExpr::getCompare(
            fp ? llvm::CmpInst::FCMP_OLE
               : uns ? llvm::CmpInst::ICMP_ULE : llvm::CmpInst::ICMP_SLE,
            lc, rc);
    case MGL_OP_GE:
        return llvm::ConstantExpr::getCompare(
            fp ? llvm::CmpInst::FCMP_OGE
               : uns ? llvm::CmpInst::ICMP_UGE : llvm::CmpInst::ICMP_SGE,
            lc, rc);
    default:
        return nullptr;
    }
    return llvm::ConstantFoldBinaryInstruction(llo, lc, rc);
}

/* Scalar/vector numeric binary op (no matrices).  Signedness follows the
 * operand types; comparisons yield bool per GLSL; && / || use LLVM's
 * branch-based short-circuit ops. */
llvm::Value *emitNumericBinOp(Codegen &cg, uint32_t op, llvm::Value *l,
                              llvm::Value *r, const MType &lt,
                              const MType &rt) {
    if (op == MGL_OP_LAND || op == MGL_OP_LOR) {
        auto truth = [&](llvm::Value *v, MGLIRScalar scalar) -> llvm::Value * {
            llvm::Type *ty = v->getType();
            if (ty->isIntOrIntVectorTy() && ty->getScalarSizeInBits() == 1)
                return v;
            if (scalar == MGLIR_SCALAR_DOUBLE)
                v = coerceScalar(cg, v, MGLIR_SCALAR_FLOAT, scalar);
            ty = v->getType();
            llvm::Value *zero = llvm::Constant::getNullValue(ty);
            return ty->isFPOrFPVectorTy()
                ? cg.b->CreateFCmpUNE(v, zero)
                : cg.b->CreateICmpNE(v, zero);
        };
        l = truth(l, lt.scalar);
        r = truth(r, rt.scalar);
        return op == MGL_OP_LAND ? cg.b->CreateLogicalAnd(l, r)
                                 : cg.b->CreateLogicalOr(l, r);
    }
    /* AGX double is an i64 payload with no f64 ALU.  Only the IR scalar kind
     * selects soft-float: true int64/uint64 values must keep integer ALU. */
    const bool l_double = lt.scalar == MGLIR_SCALAR_DOUBLE;
    const bool r_double = rt.scalar == MGLIR_SCALAR_DOUBLE;
    const bool soft_double = l_double || r_double;
    if (soft_double) {
        l = coerceScalar(cg, l, MGLIR_SCALAR_FLOAT, lt.scalar);
        r = coerceScalar(cg, r, MGLIR_SCALAR_FLOAT, rt.scalar);
    }
    bool lfp = l->getType()->isFPOrFPVectorTy();
    bool rfp = r->getType()->isFPOrFPVectorTy();
    if (lfp != rfp) {
        /* The integer side must be told its own signedness: `uintExpr * 2.0`
         * has to convert with UIToFP.  coerceScalar cannot recover it from the
         * LLVM value (i32 carries no signedness), and SIToFP would turn every
         * operand >= 2^31 into a NEGATIVE float.  Both operand MTypes are in
         * scope here — the only place in this function that needs them. */
        if (lfp) r = coerceScalar(cg, r, MGLIR_SCALAR_FLOAT, rt.scalar);
        else l = coerceScalar(cg, l, MGLIR_SCALAR_FLOAT, lt.scalar);
        rfp = r->getType()->isFPOrFPVectorTy();
    }
    bool uns = lt.scalar == MGLIR_SCALAR_UINT || rt.scalar == MGLIR_SCALAR_UINT ||
               lt.scalar == MGLIR_SCALAR_UINT64 || rt.scalar == MGLIR_SCALAR_UINT64;
    if (!lfp && !rfp && l->getType() != r->getType()) {
        if (op == MGL_OP_SHL || op == MGL_OP_SHR) {
            r = cg.b->CreateZExtOrTrunc(r, l->getType());
        } else if (l->getType()->isIntOrIntVectorTy() &&
                   r->getType()->isIntOrIntVectorTy()) {
            const bool to64 = l->getType()->getScalarSizeInBits() == 64 ||
                              r->getType()->getScalarSizeInBits() == 64;
            const MGLIRScalar target = to64
                ? (uns ? MGLIR_SCALAR_UINT64 : MGLIR_SCALAR_INT64)
                : (uns ? MGLIR_SCALAR_UINT : MGLIR_SCALAR_INT);
            l = coerceScalar(cg, l, target, lt.scalar);
            r = coerceScalar(cg, r, target, rt.scalar);
        }
    }
    bool lv = l->getType()->isVectorTy();
    bool rv = r->getType()->isVectorTy();
        if (lv != rv) {
        if (lv) r = broadcastTo(cg, r, l->getType());
        else l = broadcastTo(cg, l, r->getType());
    }
    bool fp = l->getType()->isFPOrFPVectorTy();
    auto widen_double = [&](llvm::Value *v) -> llvm::Value * {
        return soft_double ? coerceScalar(cg, v, MGLIR_SCALAR_DOUBLE) : v;
    };
    llvm::CmpInst::Predicate pred;
    switch (op) {
    case MGL_OP_ADD: return widen_double(fp ? cg.b->CreateFAdd(l, r)
                                            : cg.b->CreateAdd(l, r));
    case MGL_OP_SUB: return widen_double(fp ? cg.b->CreateFSub(l, r)
                                            : cg.b->CreateSub(l, r));
    case MGL_OP_MUL: return widen_double(fp ? cg.b->CreateFMul(l, r)
                                           : cg.b->CreateMul(l, r));
    case MGL_OP_DIV: return widen_double(fp ? cg.b->CreateFDiv(l, r)
                   : uns ? cg.b->CreateUDiv(l, r) : cg.b->CreateSDiv(l, r));
    case MGL_OP_MOD: return widen_double(fp ? cg.b->CreateFRem(l, r)
                   : uns ? cg.b->CreateURem(l, r) : cg.b->CreateSRem(l, r));
    case MGL_OP_SHL: return cg.b->CreateShl(l, r);
    case MGL_OP_SHR: return uns ? cg.b->CreateLShr(l, r) : cg.b->CreateAShr(l, r);
    case MGL_OP_AND: return cg.b->CreateAnd(l, r);
    case MGL_OP_OR:  return cg.b->CreateOr(l, r);
    case MGL_OP_XOR: return cg.b->CreateXor(l, r);
    case MGL_OP_LAND: return cg.b->CreateLogicalAnd(l, r);
    case MGL_OP_LOR: return cg.b->CreateLogicalOr(l, r);
    case MGL_OP_EQ: pred = fp ? llvm::CmpInst::FCMP_OEQ : llvm::CmpInst::ICMP_EQ; break;
    /* GLSL 4.60 §5.9: float != is true if either operand is NaN. */
    case MGL_OP_NE: pred = fp ? llvm::CmpInst::FCMP_UNE : llvm::CmpInst::ICMP_NE; break;
    case MGL_OP_LT: pred = fp ? llvm::CmpInst::FCMP_OLT
                     : uns ? llvm::CmpInst::ICMP_ULT : llvm::CmpInst::ICMP_SLT; break;
    case MGL_OP_LE: pred = fp ? llvm::CmpInst::FCMP_OLE
                     : uns ? llvm::CmpInst::ICMP_ULE : llvm::CmpInst::ICMP_SLE; break;
    case MGL_OP_GT: pred = fp ? llvm::CmpInst::FCMP_OGT
                     : uns ? llvm::CmpInst::ICMP_UGT : llvm::CmpInst::ICMP_SGT; break;
    case MGL_OP_GE: pred = fp ? llvm::CmpInst::FCMP_OGE
                     : uns ? llvm::CmpInst::ICMP_UGE : llvm::CmpInst::ICMP_SGE; break;
    default: return nullptr;
    }
    return scalarizeBoolCompare(
        cg, op, fp ? cg.b->CreateFCmp(pred, l, r) : cg.b->CreateICmp(pred, l, r));
}

/* GLSL 4.60 §5.9: struct/array == and != are element-wise, yielding a
 * scalar bool.  Vectors still go through emitNumericBinOp. */
llvm::Value *emitAggregateCompare(Codegen &cg, uint32_t op,
                                  llvm::Value *l, llvm::Value *r) {
    if (op != MGL_OP_EQ && op != MGL_OP_NE) return nullptr;
    if (!l || !r || l->getType() != r->getType()) return nullptr;
    llvm::Type *ty = l->getType();
    if (ty->isStructTy() || ty->isArrayTy()) {
        unsigned n = ty->isStructTy()
                         ? ty->getStructNumElements()
                         : (unsigned)ty->getArrayNumElements();
        llvm::Value *acc = nullptr;
        for (unsigned i = 0; i < n; i++) {
            llvm::Value *lv = cg.b->CreateExtractValue(l, i);
            llvm::Value *rv = cg.b->CreateExtractValue(r, i);
            llvm::Value *cmp = emitAggregateCompare(cg, op, lv, rv);
            if (!cmp) {
                MType dummy;
                dummy.scalar = MGLIR_SCALAR_INT;
                cmp = emitNumericBinOp(cg, op, lv, rv, dummy, dummy);
            }
            if (!cmp) return nullptr;
            acc = !acc ? cmp
                       : (op == MGL_OP_EQ ? cg.b->CreateAnd(acc, cmp)
                                          : cg.b->CreateOr(acc, cmp));
        }
        return acc ? acc
                   : llvm::ConstantInt::get(cg.b->getInt1Ty(),
                                            op == MGL_OP_EQ);
    }
    return nullptr;
}

/* Broadcast a scalar to a vector type; identity if already matching. */
llvm::Value *broadcastTo(Codegen &cg, llvm::Value *v, llvm::Type *vecTy) {
    if (v->getType() == vecTy) return v;
    if (!vecTy->isVectorTy()) return v;
    auto *vt = llvm::cast<llvm::FixedVectorType>(vecTy);
    return cg.b->CreateVectorSplat(
        (uint32_t)vt->getElementCount().getFixedValue(), v);
}

/* Scalar or vector dot product with a fixed lane order. */
llvm::Value *dotProduct(Codegen &cg, llvm::Value *a, llvm::Value *b) {
    llvm::Type *t = a->getType();
    if (!t->isVectorTy()) return cg.b->CreateFMul(a, b);
    auto *vt = llvm::cast<llvm::FixedVectorType>(t);
    uint32_t n = (uint32_t)vt->getElementCount().getFixedValue();
    llvm::Value *e0 = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), 0);
    llvm::Value *acc = cg.b->CreateFMul(
        cg.b->CreateExtractElement(a, e0), cg.b->CreateExtractElement(b, e0));
    for (uint32_t i = 1; i < n; i++) {
        llvm::Value *ix = llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(*cg.ctx), i);
        llvm::Value *p = cg.b->CreateFMul(
            cg.b->CreateExtractElement(a, ix), cg.b->CreateExtractElement(b, ix));
        acc = cg.b->CreateFAdd(acc, p);
    }
    return acc;
}

} /* namespace air */
} /* namespace mgl */
