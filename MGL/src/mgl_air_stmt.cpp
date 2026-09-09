/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_stmt.cpp
 * C1g — AIR statement emit extracted from mgl_air_backend.cpp
 * (emitStmt / emitCompound + stmtContainsBreakOrContinue).  Backend keeps
 * a thin AirStmtDeps facade; emitExpr stays in the monolith.
 */

#include "mgl_air_stmt.h"
#include "mgl_air_type.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"

namespace mgl {
namespace air {

static void emitCompound(Codegen &cg, const MGLStmt *st, const MGLIRModule *mod,
                         std::map<std::string, MType> *locals,
                         const AirStmtDeps &deps) {
    for (uint32_t i = 0; i < st->u.compound.count; i++)
        emitStmt(cg, st->u.compound.stmts[i], mod, locals, deps);
}

/* Tiny for-unroll must not erase loopStack/breakStack targets. */
static bool stmtContainsBreakOrContinue(const MGLStmt *st) {
    if (!st) return false;
    switch (st->kind) {
    case MGL_STMT_BREAK:
    case MGL_STMT_CONTINUE:
        return true;
    case MGL_STMT_COMPOUND:
        for (uint32_t i = 0; i < st->u.compound.count; i++)
            if (stmtContainsBreakOrContinue(st->u.compound.stmts[i]))
                return true;
        return false;
    case MGL_STMT_IF:
        return stmtContainsBreakOrContinue(st->u.ifs.then) ||
               stmtContainsBreakOrContinue(st->u.ifs.else_);
    case MGL_STMT_FOR:
        return stmtContainsBreakOrContinue(st->u.loop.init) ||
               stmtContainsBreakOrContinue(st->u.loop.body);
    case MGL_STMT_WHILE:
    case MGL_STMT_DO_WHILE:
        return stmtContainsBreakOrContinue(st->u.whilex.body);
    case MGL_STMT_SWITCH:
        return stmtContainsBreakOrContinue(st->u.switchx.body);
    default:
        return false;
    }
}

void emitStmt(Codegen &cg, const MGLStmt *st, const MGLIRModule *mod,
              std::map<std::string, MType> *locals, const AirStmtDeps &deps) {
    if (cg.err) return;
    switch (st->kind) {
    case MGL_STMT_COMPOUND:
        emitCompound(cg, st, mod, locals, deps);
        break;
    case MGL_STMT_EXPR:
        if (st->u.expr.expr)
            deps.emitExpr(cg, st->u.expr.expr, mod, *locals);
        break;
    case MGL_STMT_DECL: {
        /* Comma-separated declarators (`int a = 0, b = 1;`): every node
         * declares its own local. */
        for (MGLDecl *d = st->u.decl.decl; d; d = d->next_declarator) {
        /* Type-only `struct S { … };` — already in cg.structTypes. */
        if (!d->name)
            continue;
        MType t;
        if (d->type && d->type->base <= MGL_AST_TYPE_DOUBLE) {
            t.scalar = (MGLIRScalar)d->type->base;
            if (d->type->mat_cols > 0 && d->type->mat_rows > 0) {
                t.cols = d->type->mat_cols;
                t.rows = d->type->mat_rows;
            } else if (d->type->vec_size > 0) {
                t.vec = d->type->vec_size;
            }
            if (d->array_count > 0 && d->array_dims)
                t.arr = d->array_dims[0];
        } else if (d->init) {
            t = deps.exprType(cg, d->init, mod, *locals);
        } else {
            t.scalar = MGLIR_SCALAR_FLOAT;
            if (d->type && d->type->vec_size) t.vec = d->type->vec_size;
        }
        (*locals)[d->name] = t;
        /* Track IR type for local structs so member ExtractValue can
         * resolve field indices (MType cannot represent structs). */
        if (d->name && d->type && d->type->base == MGL_AST_TYPE_STRUCT &&
            d->type->name) {
            auto sit = cg.structTypes.find(d->type->name);
            if (sit != cg.structTypes.end()) {
                const MGLIRType *base = sit->second;
                uint32_t n = 0;
                if (d->array_count > 0 && d->array_dims)
                    n = d->array_dims[0];
                else if (d->init && d->init->kind == MGL_EXPR_CALL &&
                         d->init->u.call.is_array_ctor)
                    n = d->init->u.call.arg_count;
                else if (t.arr)
                    n = t.arr;
                if (n > 0) {
                    MGLIRType *arr =
                        mglIRTypeArray(deps.cloneIRType(base), n);
                    if (arr && cg.ownedIRTypes) {
                        cg.ownedIRTypes->push_back(arr);
                        cg.localIRTypes[d->name] = arr;
                    }
                } else {
                    cg.localIRTypes[d->name] = base;
                }
            }
        }
        if (d->init) {
            llvm::Value *v = deps.emitExpr(cg, d->init, mod, *locals);
            if (!v) return;
            v = coerceScalar(cg, v, t.scalar);
            if (needsArrayMem(cg, d->name ? d->name : "", t) && d->name) {
                llvm::Value *slot = ensureArrayMem(cg, d->name, t);
                cg.b->CreateAlignedStore(v, slot, llvm::Align(4));
            } else {
                cg.lvalues[d->name] = v;
            }
        } else if (d->name) {
            /* Uninitialized locals must still occupy an SSA slot before
             * any loop.  Lazy Undef on the first indexed store inside a
             * for-body is not in the loop phi set, so each iteration
             * rebuilds from Undef and leaves select(..., undef) values
             * that crash MTLCompilerService (XPC) when later read. */
            if (needsArrayMem(cg, d->name, t)) {
                ensureArrayMem(cg, d->name, t);
            } else {
                llvm::Type *ty = nullptr;
                auto irit = cg.localIRTypes.find(d->name);
                if (irit != cg.localIRTypes.end())
                    ty = llvmTypeFromIR(irit->second, *cg.ctx);
                else
                    ty = llvmType(t, *cg.ctx);
                if (ty)
                    cg.lvalues[d->name] = llvm::UndefValue::get(ty);
            }
        }
        }
        break;
    }
    case MGL_STMT_RETURN: {
        if (cg.inliningHelper) {
            /* Capture return for inlined GS/TCS/compute helpers; do not
             * terminate the enclosing stage function. */
            if (st->u.ret.value) {
                llvm::Value *v = deps.emitExpr(cg, st->u.ret.value, mod, *locals);
                if (!v) return;
                cg.inlineRetVal = v;
            }
            cg.err = 2;
            break;
        }
        if (st->u.ret.value) {
            llvm::Value *v = deps.emitExpr(cg, st->u.ret.value, mod, *locals);
            if (!v) return;
            /* Coerce to the LLVM function return type when shapes differ.
             * Doubles are i64 payloads on AGX — never emit f64 ALU; widen
             * int results with sext so RetInst types match. */
            llvm::Type *wantTy = cg.fn->getReturnType();
            if (!wantTy->isVoidTy() && v->getType() != wantTy) {
                if (wantTy->isFPOrFPVectorTy()) {
                    v = coerceScalar(cg, v, MGLIR_SCALAR_FLOAT);
                } else if (wantTy->isIntegerTy(64) &&
                           v->getType()->isIntegerTy(32)) {
                    v = cg.b->CreateSExt(v, wantTy);
                } else if (wantTy->isVectorTy() &&
                           v->getType()->isVectorTy()) {
                    auto *wvt = llvm::cast<llvm::FixedVectorType>(wantTy);
                    auto *vvt = llvm::cast<llvm::FixedVectorType>(v->getType());
                    if (wvt->getElementCount() == vvt->getElementCount() &&
                        wvt->getElementType()->isIntegerTy(64) &&
                        vvt->getElementType()->isIntegerTy(32))
                        v = cg.b->CreateSExt(v, wantTy);
                    else if (wvt->getElementType()->isFloatingPointTy())
                        v = coerceScalar(cg, v, MGLIR_SCALAR_FLOAT);
                } else if (wantTy->isIntOrIntVectorTy() &&
                           wantTy->getScalarSizeInBits() == 32) {
                    v = coerceScalar(cg, v, MGLIR_SCALAR_INT);
                }
                if (v->getType() != wantTy &&
                    v->getType()->getPrimitiveSizeInBits() ==
                        wantTy->getPrimitiveSizeInBits())
                    v = cg.b->CreateBitCast(v, wantTy);
            }
            if (v->getType() != wantTy && !wantTy->isVoidTy()) {
                cg.err = 1;
                cg.errmsg = "codegen: return type mismatch after coercion";
                return;
            }
            cg.b->CreateRet(v);
        } else if (cg.fn->getReturnType()->isVoidTy()) {
            cg.b->CreateRetVoid();
        } else {
            /* Bare return; in a non-void stage function: assemble the
             * outputs (position / varyings / frag color) as at end of
             * body. */
            cg.b->CreateRet(deps.assembleReturn(cg));
        }
        /* Stop emitting unreachable code. */
        cg.err = 2;
        break;
    }
    case MGL_STMT_DISCARD: {
        /* discard in fragment stage: lowers to air.discard_fragment(). */
        if (cg.isVS) {
            cg.err = 1;
            cg.errmsg = "codegen: discard is only allowed in fragment shaders";
            return;
        }
        llvm::Function *df = llvm::cast<llvm::Function>(
            cg.mod->getOrInsertFunction("air.discard_fragment",
                                        cg.b->getVoidTy())
                .getCallee());
        cg.b->CreateCall(df);
        cg.b->CreateRet(deps.assembleReturn(cg));
        cg.err = 2;
        break;
    }
    case MGL_STMT_IF: {
        /* if (cond) then [else else_]: SSA via phi at the merge block.
         * Nested ifs work recursively; a return inside a branch is
         * supported (no phi edge from that branch). */
        llvm::Value *cond = deps.emitExpr(cg, st->u.ifs.cond, mod, *locals);
        if (!cond) return;
        /* Constant condition: emit only the live branch. */
        if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(cond);
            ci && ci->getType()->isIntegerTy(1)) {
            if (ci->getValue().getBoolValue())
                emitStmt(cg, st->u.ifs.then, mod, locals, deps);
            else if (st->u.ifs.else_)
                emitStmt(cg, st->u.ifs.else_, mod, locals, deps);
            if (cg.err == 1) return;
            break;
        }
        if (!cond->getType()->isIntegerTy(1)) {
            cg.err = 1;
            cg.errmsg = "codegen: if condition must be a scalar bool";
            return;
        }
        int savedErr = cg.err;
        cg.err = 0;
        llvm::BasicBlock *condBB = cg.b->GetInsertBlock();
        llvm::BasicBlock *bbThen =
            llvm::BasicBlock::Create(*cg.ctx, "if.then", cg.fn);
        llvm::BasicBlock *bbElse = st->u.ifs.else_
            ? llvm::BasicBlock::Create(*cg.ctx, "if.else", cg.fn)
            : nullptr;
        llvm::BasicBlock *bbMerge =
            llvm::BasicBlock::Create(*cg.ctx, "if.end", cg.fn);
        cg.b->CreateCondBr(cond, bbThen, bbElse ? bbElse : bbMerge);

        std::map<std::string, llvm::Value *> snap = cg.lvalues;

        cg.b->SetInsertPoint(bbThen);
        emitStmt(cg, st->u.ifs.then, mod, locals, deps);
        if (cg.err == 1) return;
        llvm::BasicBlock *thenTail = cg.b->GetInsertBlock();
        bool thenRet = thenTail->getTerminator() &&
                       llvm::isa<llvm::ReturnInst>(thenTail->getTerminator());
        if (!thenRet) cg.b->CreateBr(bbMerge);
        std::map<std::string, llvm::Value *> thenL = cg.lvalues;

        std::map<std::string, llvm::Value *> elseL;
        if (bbElse) {
            /* Restart from the pre-if values: branch bodies are mutually
             * exclusive, so a value computed inside the then branch (a phi
             * in a then-side merge block) does not dominate the else side.
             * Letting the else body see it produced phi operands on
             * non-dominating edges -- invalid IR that crashed the AGX
             * compiler (MTLCompilerService SIGSEGV). */
            cg.lvalues = snap;
            cg.err = 0;
            cg.b->SetInsertPoint(bbElse);
            emitStmt(cg, st->u.ifs.else_, mod, locals, deps);
            if (cg.err == 1) return;
            llvm::BasicBlock *elseTail = cg.b->GetInsertBlock();
            bool elseRet = elseTail->getTerminator() &&
                           llvm::isa<llvm::ReturnInst>(elseTail->getTerminator());
            if (!elseRet) cg.b->CreateBr(bbMerge);
            elseL = cg.lvalues;
            if (thenRet && elseRet) {
                /* Both paths return; code after the if is unreachable. */
                cg.lvalues = snap;
            } else if (thenRet) {
                cg.lvalues = elseL;
            } else if (elseRet) {
                cg.lvalues = thenL;
            } else {
                cg.b->SetInsertPoint(bbMerge);
                for (auto &kv : thenL) {
                    auto it = elseL.find(kv.first);
                    if (it == elseL.end()) continue; /* then-only decl */
                    if (kv.second == it->second) continue;
                    if (kv.second->getType() != it->second->getType())
                        continue;
                    llvm::PHINode *phi =
                        cg.b->CreatePHI(kv.second->getType(), 2, kv.first);
                    phi->addIncoming(kv.second, thenTail);
                    phi->addIncoming(it->second, elseTail);
                    cg.lvalues[kv.first] = phi;
                }
                for (auto &kv : elseL)
                    if (!thenL.count(kv.first))
                        cg.lvalues[kv.first] = kv.second;
            }
        } else if (thenRet) {
            /* Return in the then branch; fall-through path skipped. */
            cg.lvalues = snap;
        } else {
            /* No else: merge changed values with the fall-through. */
            cg.b->SetInsertPoint(bbMerge);
            for (auto &kv : thenL) {
                auto it = snap.find(kv.first);
                if (it != snap.end() && it->second == kv.second) continue;
                llvm::Value *fall =
                    (it != snap.end() &&
                     it->second->getType() == kv.second->getType())
                        ? it->second
                        : llvm::UndefValue::get(kv.second->getType());
                llvm::PHINode *phi =
                    cg.b->CreatePHI(kv.second->getType(), 2, kv.first);
                phi->addIncoming(kv.second, thenTail);
                phi->addIncoming(fall, condBB);
                cg.lvalues[kv.first] = phi;
            }
        }
        cg.err = savedErr;
        cg.b->SetInsertPoint(bbMerge);
        break;
    }
    case MGL_STMT_WHILE:
    case MGL_STMT_FOR:
    case MGL_STMT_DO_WHILE: {
        /* Unroll tiny constant for-loops of the form
         *   for (T i = 0; i < N; ++i) ...
         * with N <= 16.  CTS 420pack binding_*_array uses this pattern to
         * index sampler/image arrays; keeping the loop makes Metal's
         * compiler crash on TCS (nested switch+phi of samples inside SSA
         * loop phis).  Unrolling yields constant indices → direct binds. */
        if (st->kind == MGL_STMT_FOR && st->u.loop.cond && st->u.loop.body &&
            st->u.loop.init && st->u.loop.init->kind == MGL_STMT_DECL &&
            st->u.loop.init->u.decl.decl &&
            st->u.loop.init->u.decl.decl->name &&
            st->u.loop.init->u.decl.decl->init &&
            st->u.loop.init->u.decl.decl->init->kind == MGL_EXPR_LITERAL &&
            st->u.loop.init->u.decl.decl->init->u.literal.value == 0.0 &&
            st->u.loop.cond->kind == MGL_EXPR_BINARY &&
            st->u.loop.cond->u.binary.op == MGL_OP_LT &&
            st->u.loop.cond->u.binary.lhs &&
            st->u.loop.cond->u.binary.lhs->kind == MGL_EXPR_VAR_REF &&
            st->u.loop.cond->u.binary.lhs->u.var_ref.name &&
            strcmp(st->u.loop.cond->u.binary.lhs->u.var_ref.name,
                   st->u.loop.init->u.decl.decl->name) == 0 &&
            st->u.loop.cond->u.binary.rhs &&
            st->u.loop.cond->u.binary.rhs->kind == MGL_EXPR_LITERAL) {
            uint32_t trip =
                (uint32_t)st->u.loop.cond->u.binary.rhs->u.literal.value;
            const char *indName = st->u.loop.init->u.decl.decl->name;
            bool incrOk = false;
            if (st->u.loop.incr && st->u.loop.incr->kind == MGL_EXPR_UNARY &&
                st->u.loop.incr->u.unary.op == MGL_OP_INC &&
                st->u.loop.incr->u.unary.operand &&
                st->u.loop.incr->u.unary.operand->kind == MGL_EXPR_VAR_REF &&
                st->u.loop.incr->u.unary.operand->u.var_ref.name &&
                strcmp(st->u.loop.incr->u.unary.operand->u.var_ref.name,
                       indName) == 0) {
                incrOk = true;
            }
            if (incrOk && trip > 0u && trip <= 16u &&
                !stmtContainsBreakOrContinue(st->u.loop.body)) {
                emitStmt(cg, st->u.loop.init, mod, locals, deps);
                if (cg.err) return;
                MType indTy = (*locals)[indName];
                for (uint32_t k = 0; k < trip; k++) {
                    cg.lvalues[indName] = llvm::ConstantInt::get(
                        llvmType(indTy, *cg.ctx), k, /*isSigned=*/true);
                    emitStmt(cg, st->u.loop.body, mod, locals, deps);
                    if (cg.err) return;
                }
                break;
            }
        }
        /* SSA loop lowering: a phi for every live value is placed at the
         * condition block (while/for) or the body head (do-while); the
         * back-edge operand is filled in after the body/incr is emitted.
         * break jumps to the merge block carrying a value snapshot;
         * continue jumps to the incr/merge block (values merge there with
         * the body tail before the condition phi sees them).  Nested
         * loops are handled through cg.loopStack. */
        LoopCtx lc;
        std::vector<std::string> names;
        for (auto &kv : cg.lvalues) names.push_back(kv.first);

        llvm::BasicBlock *bbCond =
            llvm::BasicBlock::Create(*cg.ctx, "loop.cond", cg.fn);
        llvm::BasicBlock *bbBody =
            llvm::BasicBlock::Create(*cg.ctx, "loop.body", cg.fn);
        llvm::BasicBlock *bbIncr =
            llvm::BasicBlock::Create(*cg.ctx, "loop.incr", cg.fn);
        llvm::BasicBlock *bbEnd =
            llvm::BasicBlock::Create(*cg.ctx, "loop.end", cg.fn);
        lc.condBB = bbCond;
        lc.endBB = bbEnd;
        lc.incrBB = bbIncr;

        if (st->kind == MGL_STMT_FOR && st->u.loop.init) {
            emitStmt(cg, st->u.loop.init, mod, locals, deps);
            if (cg.err) return;
            /* The init declaration is live across the loop; it must be
             * captured by the phi set too. */
            for (auto &kv : cg.lvalues)
                if (std::find(names.begin(), names.end(), kv.first) ==
                    names.end())
                    names.push_back(kv.first);
        }

        if (st->kind == MGL_STMT_DO_WHILE) {
            llvm::BasicBlock *pre = cg.b->GetInsertBlock();
            cg.b->CreateBr(bbBody);
            cg.b->SetInsertPoint(bbBody);
            for (auto &n : names) {
                auto *p = cg.b->CreatePHI(cg.lvalues[n]->getType(), 2, n);
                p->addIncoming(cg.lvalues[n], pre);
                lc.phis[n] = p;
                cg.lvalues[n] = p;
            }
        } else {
            llvm::BasicBlock *pre = cg.b->GetInsertBlock();
            cg.b->CreateBr(bbCond);
            cg.b->SetInsertPoint(bbCond);
            for (auto &n : names) {
                auto *p = cg.b->CreatePHI(cg.lvalues[n]->getType(), 2, n);
                p->addIncoming(cg.lvalues[n], pre);
                lc.phis[n] = p;
                cg.lvalues[n] = p;
            }
        }

        cg.loopStack.push_back(&lc);
        BreakCtx brk{bbEnd, {}};
        cg.breakStack.push_back(&brk);
        if (st->kind == MGL_STMT_DO_WHILE) {
            emitStmt(cg, st->u.whilex.body, mod, locals, deps);
            if (cg.err == 1) return;
            llvm::BasicBlock *tail = cg.b->GetInsertBlock();
            if (!tail->getTerminator()) cg.b->CreateBr(bbIncr);
            cg.b->SetInsertPoint(bbIncr);
            for (auto &n : names) {
                auto *p = cg.b->CreatePHI(cg.lvalues[n]->getType(),
                                          1 + lc.contSnaps.size(), n);
                bool isCont = false;
                for (auto &cs : lc.contSnaps)
                    if (cs.first == tail) { isCont = true; break; }
                if (!isCont) p->addIncoming(cg.lvalues[n], tail);
                for (auto &cs : lc.contSnaps) {
                    auto it = cs.second.find(n);
                    p->addIncoming(it != cs.second.end() ? it->second
                                                         : cg.lvalues[n],
                                   cs.first);
                }
                cg.lvalues[n] = p;
            }
            cg.b->CreateBr(bbCond);
            cg.b->SetInsertPoint(bbCond);
            llvm::Value *cond = deps.emitExpr(cg, st->u.whilex.cond, mod, *locals);
            if (cg.err) return;
            if (!cond->getType()->isIntegerTy(1)) {
                cg.err = 1;
                cg.errmsg = "codegen: do-while condition must be a scalar bool";
                return;
            }
            for (auto &n : names)
                lc.condExitSnap[n] = cg.lvalues[n];
            lc.condExitBB =
                llvm::BasicBlock::Create(*cg.ctx, "loop.cond.exit", cg.fn);
            for (auto &kv : lc.phis)
                kv.second->addIncoming(cg.lvalues[kv.first], bbCond);
            cg.b->CreateCondBr(cond, bbBody, lc.condExitBB);
            cg.b->SetInsertPoint(lc.condExitBB);
            cg.b->CreateBr(bbEnd);
        } else {
            llvm::Value *cond = st->kind == MGL_STMT_FOR
                ? (st->u.loop.cond ? deps.emitExpr(cg, st->u.loop.cond, mod,
                                              *locals)
                                   : nullptr)
                : deps.emitExpr(cg, st->u.whilex.cond, mod, *locals);
            if (cg.err) return;
            bool bodyDead = false;
            if (cond) {
                if (!cond->getType()->isIntegerTy(1)) {
                    cg.err = 1;
                    cg.errmsg = "codegen: loop condition must be a scalar bool";
                    return;
                }
                /* Constant-false condition: the body never runs.  The
                 * body/incr blocks were already created; terminate them
                 * so the IR stays valid, then jump straight to the merge. */
                if (auto *cint = llvm::dyn_cast<llvm::ConstantInt>(cond);
                    cint && !cint->getValue().getBoolValue()) {
                    llvm::BasicBlock *cur = cg.b->GetInsertBlock();
                    for (auto &n : names)
                        lc.condExitSnap[n] = cg.lvalues[n];
                    lc.condExitBB = cur;
                    cg.b->SetInsertPoint(bbBody);
                    cg.b->CreateUnreachable();
                    cg.b->SetInsertPoint(bbIncr);
                    cg.b->CreateUnreachable();
                    cg.b->SetInsertPoint(cur);
                    cg.b->CreateBr(bbEnd);
                    bodyDead = true;
                } else {
                    for (auto &n : names)
                        lc.condExitSnap[n] = cg.lvalues[n];
                    lc.condExitBB =
                        llvm::BasicBlock::Create(*cg.ctx, "loop.cond.exit",
                                                 cg.fn);
                    cg.b->CreateCondBr(cond, bbBody, lc.condExitBB);
                    cg.b->SetInsertPoint(lc.condExitBB);
                    cg.b->CreateBr(bbEnd);
                }
            } else {
                cg.b->CreateBr(bbBody);
            }
            if (!bodyDead) {
            cg.b->SetInsertPoint(bbBody);
            emitStmt(cg, st->kind == MGL_STMT_FOR ? st->u.loop.body
                                                  : st->u.whilex.body,
                     mod, locals, deps);
            if (cg.err == 1) return;
            llvm::BasicBlock *tail = cg.b->GetInsertBlock();
            if (!tail->getTerminator()) cg.b->CreateBr(bbIncr);
            cg.b->SetInsertPoint(bbIncr);
            /* Merge the values carried by the body tail and any continue
             * snapshots before running the for-loop increment, so the
             * condition phi keeps a single back-edge block. */
            for (auto &n : names) {
                auto *p = cg.b->CreatePHI(cg.lvalues[n]->getType(),
                                          1 + lc.contSnaps.size(), n);
                bool isCont = false;
                for (auto &cs : lc.contSnaps)
                    if (cs.first == tail) { isCont = true; break; }
                if (!isCont) p->addIncoming(cg.lvalues[n], tail);
                for (auto &cs : lc.contSnaps) {
                    auto it = cs.second.find(n);
                    p->addIncoming(it != cs.second.end() ? it->second
                                                         : cg.lvalues[n],
                                   cs.first);
                }
                cg.lvalues[n] = p;
            }
            if (st->kind == MGL_STMT_FOR && st->u.loop.incr) {
                deps.emitExpr(cg, st->u.loop.incr, mod, *locals);
                if (cg.err == 1) return;
            }
            for (auto &kv : lc.phis)
                kv.second->addIncoming(cg.lvalues[kv.first], bbIncr);
            cg.b->CreateBr(bbCond);
            }
        }
        cg.loopStack.pop_back();
        cg.breakStack.pop_back();

        cg.b->SetInsertPoint(bbEnd);
        for (auto &n : names) {
            llvm::Value *v = st->kind == MGL_STMT_DO_WHILE
                                 ? cg.lvalues[n]
                                 : lc.phis[n];
            unsigned nIn = (unsigned)brk.snaps.size() +
                           (lc.condExitBB ? 1u : 0u);
            llvm::PHINode *e =
                cg.b->CreatePHI(v->getType(), nIn, n);
            if (lc.condExitBB) {
                auto it = lc.condExitSnap.find(n);
                e->addIncoming(it != lc.condExitSnap.end() ? it->second : v,
                               lc.condExitBB);
            }
            for (auto &bs : brk.snaps) {
                auto it = bs.second.find(n);
                e->addIncoming(it != bs.second.end() ? it->second : v,
                               bs.first);
            }
            cg.lvalues[n] = e;
        }
        break;
    }
    case MGL_STMT_SWITCH: {
        /* switch (c) { case v: ... default: ... } lowered to a chain of
         * equality checks; each case/default starts a segment, segments
         * fall through to the next one (or the exit) unless a break (or
         * return) terminates them.  break carries a value snapshot and is
         * merged into phis at the exit block. */
        llvm::Value *cond = deps.emitExpr(cg, st->u.switchx.cond, mod, *locals);
        if (!cond) return;
        if (!cond->getType()->isIntegerTy()) {
            cg.err = 1;
            cg.errmsg = "codegen: switch condition must be an integer";
            return;
        }
        int savedErr = cg.err;
        cg.err = 0;
        std::map<std::string, llvm::Value *> snap = cg.lvalues;

        std::vector<MGLStmt *> bodyStmts;
        if (st->u.switchx.body->kind == MGL_STMT_COMPOUND) {
            const auto *cp = &st->u.switchx.body->u.compound;
            bodyStmts.assign(cp->stmts, cp->stmts + cp->count);
        } else {
            bodyStmts.push_back(st->u.switchx.body);
        }

        struct Seg {
            std::vector<llvm::ConstantInt *> vals;
            bool isDef = false;
            llvm::BasicBlock *entry = nullptr;
            std::vector<MGLStmt *> stmts;
        };
        std::vector<Seg> segs;
        for (auto *s : bodyStmts) {
            if (s->kind == MGL_STMT_CASE || s->kind == MGL_STMT_DEFAULT) {
                segs.push_back(Seg{});
                if (s->kind == MGL_STMT_DEFAULT) {
                    segs.back().isDef = true;
                    continue;
                }
                const MGLExpr *v = s->u.casex.value;
                if (v->kind != MGL_EXPR_LITERAL ||
                    (v->u.literal.base != MGL_AST_TYPE_INT &&
                     v->u.literal.base != MGL_AST_TYPE_UINT)) {
                    cg.err = 1;
                    cg.errmsg = "codegen: case value must be a constant integer";
                    return;
                }
                segs.back().vals.push_back(llvm::cast<llvm::ConstantInt>(
                    llvm::ConstantInt::get(cond->getType(),
                                           (uint64_t)v->u.literal.value,
                                           true)));
            } else if (!segs.empty()) {
                segs.back().stmts.push_back(s);
            }
        }

        llvm::BasicBlock *bbEnd =
            llvm::BasicBlock::Create(*cg.ctx, "switch.end", cg.fn);
        for (auto &seg : segs)
            seg.entry =
                llvm::BasicBlock::Create(*cg.ctx, "switch.case", cg.fn);

        BreakCtx brk{bbEnd, {}};
        cg.breakStack.push_back(&brk);

        /* Constant condition: emit only the matching segment (or the
         * default) and its fall-through chain; unselected segment entry
         * blocks are terminated so the IR stays valid. */
        if (auto *cint = llvm::dyn_cast<llvm::ConstantInt>(cond)) {
            int64_t cv = cint->getSExtValue();
            int sel = -1, defIdx = -1;
            for (size_t i = 0; i < segs.size(); i++) {
                if (segs[i].isDef) { defIdx = (int)i; continue; }
                for (auto *v : segs[i].vals)
                    if (v->getSExtValue() == cv) sel = (int)i;
            }
            if (sel < 0) sel = defIdx;
            if (sel >= 0) {
                llvm::BasicBlock *cur = cg.b->GetInsertBlock();
                if (!cur->getTerminator())
                    cg.b->CreateBr(segs[sel].entry);
            }
            llvm::BasicBlock *lastTail = nullptr;
            bool chainBroken = false;
            for (size_t i = 0; i < segs.size(); i++) {
                if (sel < 0 || (int)i < sel) continue;
                if (chainBroken) break;
                cg.b->SetInsertPoint(segs[i].entry);
                for (auto *s : segs[i].stmts) {
                    emitStmt(cg, s, mod, locals, deps);
                    if (cg.err == 1) return;
                }
                llvm::BasicBlock *tail = cg.b->GetInsertBlock();
                if (!tail->getTerminator()) {
                    if (tail->hasNPredecessors(0)) {
                        /* Dead block left by break/continue/return:
                         * the chain is broken; terminate it. */
                        chainBroken = true;
                        cg.b->CreateUnreachable();
                    } else if (i + 1 < segs.size()) {
                        cg.b->CreateBr(segs[i + 1].entry);
                    } else {
                        cg.b->CreateBr(bbEnd);
                        lastTail = tail;
                    }
                } else {
                    chainBroken = true;
                }
            }
            llvm::BasicBlock *noMatch = nullptr;
            if (sel < 0) {
                llvm::BasicBlock *cur = cg.b->GetInsertBlock();
                if (!cur->getTerminator()) {
                    cg.b->CreateBr(bbEnd);
                    noMatch = cg.b->GetInsertBlock();
                }
            }
            for (auto &seg : segs) {
                if (seg.entry->getTerminator()) continue;
                llvm::BasicBlock *ip = cg.b->GetInsertBlock();
                cg.b->SetInsertPoint(seg.entry);
                cg.b->CreateUnreachable();
                cg.b->SetInsertPoint(ip);
            }
            cg.breakStack.pop_back();
            cg.b->SetInsertPoint(bbEnd);
            for (auto &kv : snap) {
                llvm::Value *v = kv.second;
                llvm::PHINode *e = cg.b->CreatePHI(
                    v->getType(), 1 + brk.snaps.size() +
                                      (lastTail ? 1 : 0) +
                                      (noMatch ? 1 : 0),
                    kv.first);
                if (noMatch)
                    e->addIncoming(v, noMatch);
                if (lastTail)
                    e->addIncoming(cg.lvalues[kv.first], lastTail);
                for (auto &bs : brk.snaps) {
                    auto it = bs.second.find(kv.first);
                    e->addIncoming(it != bs.second.end() ? it->second : v,
                                   bs.first);
                }
                cg.lvalues[kv.first] = e;
            }
            cg.err = savedErr;
            break;
        }

        llvm::BasicBlock *check =
            llvm::BasicBlock::Create(*cg.ctx, "switch.check", cg.fn);
        cg.b->CreateBr(check);
        for (auto &seg : segs) {
            for (auto *v : seg.vals) {
                llvm::BasicBlock *next = llvm::BasicBlock::Create(
                    *cg.ctx, "switch.check", cg.fn);
                cg.b->SetInsertPoint(check);
                llvm::Value *eq = cg.b->CreateICmpEQ(cond, v);
                cg.b->CreateCondBr(eq, seg.entry, next);
                check = next;
            }
        }
        llvm::BasicBlock *defEntry = nullptr;
        for (auto &seg : segs)
            if (seg.isDef) { defEntry = seg.entry; break; }
        cg.b->SetInsertPoint(check);
        cg.b->CreateBr(defEntry ? defEntry : bbEnd);

        llvm::BasicBlock *lastTail = nullptr;
        for (size_t i = 0; i < segs.size(); i++) {
            cg.b->SetInsertPoint(segs[i].entry);
            for (auto *s : segs[i].stmts) {
                emitStmt(cg, s, mod, locals, deps);
                if (cg.err == 1) return;
            }
            llvm::BasicBlock *tail = cg.b->GetInsertBlock();
            if (!tail->getTerminator()) {
                if (i + 1 < segs.size())
                    cg.b->CreateBr(segs[i + 1].entry);
                else {
                    cg.b->CreateBr(bbEnd);
                    lastTail = tail;
                }
            }
        }
        cg.breakStack.pop_back();

        cg.b->SetInsertPoint(bbEnd);
        /* Merge over the union of pre-switch lvalues and anything first
         * assigned inside the switch (builtins like gl_Position are only
         * added to cg.lvalues when a case assigns them).  A name absent
         * from snap enters with an undefined value on edges where it was
         * never written. */
        std::set<std::string> mergeNames;
        for (auto &kv : snap) mergeNames.insert(kv.first);
        for (auto &kv : cg.lvalues) mergeNames.insert(kv.first);
        for (const auto &name : mergeNames) {
            llvm::Value *v = nullptr;
            auto sit = snap.find(name);
            if (sit != snap.end()) {
                v = sit->second;
            } else {
                auto lit = cg.lvalues.find(name);
                v = lit != cg.lvalues.end()
                        ? llvm::UndefValue::get(lit->second->getType())
                        : llvm::UndefValue::get(
                              llvm::Type::getVoidTy(*cg.ctx));
            }
            llvm::PHINode *e = cg.b->CreatePHI(
                v->getType(),
                1 + brk.snaps.size() + (lastTail ? 1 : 0) +
                    (defEntry ? 0 : 1),
                name);
            /* No default label: the last check block falls through to
             * the exit carrying the entry values. */
            if (!defEntry)
                e->addIncoming(v, check);
            if (lastTail)
                e->addIncoming(cg.lvalues[name], lastTail);
            for (auto &bs : brk.snaps) {
                auto it = bs.second.find(name);
                e->addIncoming(it != bs.second.end() ? it->second : v,
                               bs.first);
            }
            cg.lvalues[name] = e;
        }
        cg.err = savedErr;
        break;
    }
    case MGL_STMT_BREAK:
    case MGL_STMT_CONTINUE: {
        if (st->kind == MGL_STMT_BREAK) {
            if (cg.breakStack.empty()) {
                cg.err = 1;
                cg.errmsg = "codegen: break outside of a loop or switch";
                return;
            }
            BreakCtx *bc = cg.breakStack.back();
            std::map<std::string, llvm::Value *> snapB;
            for (auto &kv : cg.lvalues)
                snapB[kv.first] = kv.second;
            bc->snaps.push_back({cg.b->GetInsertBlock(), snapB});
            cg.b->CreateBr(bc->endBB);
        } else {
            if (cg.loopStack.empty()) {
                cg.err = 1;
                cg.errmsg = "codegen: continue outside of a loop";
                return;
            }
            LoopCtx *lc = cg.loopStack.back();
            if (lc->incrBB) {
                std::map<std::string, llvm::Value *> snap;
                for (auto &kv : lc->phis)
                    snap[kv.first] = cg.lvalues[kv.first];
                lc->contSnaps.push_back({cg.b->GetInsertBlock(), snap});
                cg.b->CreateBr(lc->incrBB);
            } else {
                for (auto &kv : lc->phis)
                    kv.second->addIncoming(cg.lvalues[kv.first],
                                           cg.b->GetInsertBlock());
                cg.b->CreateBr(lc->condBB);
            }
        }
        /* Code after break/continue is unreachable; emit it into a fresh
         * block so the following statements keep a valid insert point. */
        cg.b->SetInsertPoint(
            llvm::BasicBlock::Create(*cg.ctx, "dead", cg.fn));
        break;
    }
    default:
        cg.err = 1;
        break;
    }
}

} /* namespace air */
} /* namespace mgl */
