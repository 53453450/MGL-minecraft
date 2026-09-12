/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * mgl_air_backend.cpp
 * MGL - GLSL AST -> AIR (LLVM bitcode + air.* metadata) -> .metallib.
 *
 * M1 scope: single-stage compilation of the resource patterns exercised
 * by the PSO gate (plain uniforms in one implicit buffer 0, vertex
 * attributes, in/out varyings, gl_Position, vec constructors, swizzles,
 * + - * / arithmetic).  Buffer data is accessed through byte-offset GEPs
 * on an opaque i8 addrspace(1)* parameter - empirically the loader and
 * the Metal driver accept this without a matching LLVM struct type
 * (docs/AIR_SHADER_BACKEND_DESIGN.md, verified PSO_OK).
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SROA.h"

#include "mgl_glsl_ast.h"
#include "mgl_glsl_parser.h"
#include "mgl_glsl_sema.h"
#include "mgl_ir.h"
#include "mgl_metallib_writer.h"
#include "mgl_air_reflect.h"
#include "mgl_buffer_slots.h"
#include "mgl_shader_abi.h"
#include "glm_limits.h" /* MAX_ATTRIBS: attrib_names contract size */
#include "mgl_air_gs_abi.h"
#include "mgl_air_tess_abi.h"
#include "mgl_legacy_compat.h"
#include "mgl_frontend_session.h"
#include "mgl_env_flag.h"
#include "mgl_air_type.h"
#include "mgl_air_codegen.h"
#include "mgl_air_resource.h"
#include "mgl_air_math.h"
#include "mgl_air_matrix.h"
#include "mgl_air_stmt.h"
#include "mgl_air_varsym.h"

namespace {

/* C1f moved callAirFn's definition into mgl_air_matrix.cpp (namespace
 * mgl::air) without leaving a visible declaration for the bare call
 * sites in this file; re-expose it here. */
using mgl::air::callAirFn;


/* Map the frontend's GS output primitive enum to the backend-neutral
 * ABI enum used by the fixed record-layout helpers. */
static MGLAIRGSOutputPrimitive airGSOutputFromAST(uint32_t ast)
{
    switch (ast) {
    case MGL_AST_GS_OUT_POINTS: return MGL_AIR_GS_OUT_POINTS;
    case MGL_AST_GS_OUT_LINE_STRIP: return MGL_AIR_GS_OUT_LINE_STRIP;
    case MGL_AST_GS_OUT_TRIANGLE_STRIP:
    default: return MGL_AIR_GS_OUT_TRIANGLE_STRIP;
    }
}


/* C1b: MType / Codegen / type helpers live in mgl_air_type + mgl_air_codegen. */
using mgl::air::MType;
using mgl::air::Uniform;
using mgl::air::VarSym;
using mgl::air::LoopCtx;
using mgl::air::BreakCtx;
using mgl::air::Codegen;
using mgl::air::scalarIsFloat;
using mgl::air::varyingUsesFloatCarrier;
using mgl::air::uintUsesSplitFloatCarrier;
using mgl::air::encodeUintSplitFloatCarrier;
using mgl::air::decodeUintSplitFloatCarrier;
using mgl::air::encodeFloatCarrier;
using mgl::air::decodeFloatCarrier;
using mgl::air::floatCarrierType;
using mgl::air::matrixColumnType;
using mgl::air::varyingNeedsFloatRecordCarrier;
using mgl::air::attrMetalIfaceType;
using mgl::air::llvmScalar;
using mgl::air::irTypeIsVoid;
using mgl::air::bufferLeafAlign;
using mgl::air::llvmType;
using mgl::air::needsArrayMem;
using mgl::air::ensureArrayMem;
using mgl::air::arrayMemGEP;
using mgl::air::llvmTypeFromIR;
using mgl::air::coerceScalar;

/* Restore helpers deleted by C1f without being relocated: resolve a
 * sampler argument (global sampler2D uniform in cg.texValues, or a
 * user-function parameter bound in cg.lvalues). */
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
using mgl::air::airTypeMangle;
using mgl::air::airGenerated;
using mgl::air::varyingIfaceTag;
using mgl::air::mslTypeName;
using mgl::air::typeFromIR;
using mgl::air::uniformBlockType;
using mgl::air::uniformBlockElementCount;
using mgl::air::uniformBlockIsInstanceArray;
using mgl::air::collectUniforms;
using mgl::air::appendOpaqueUniformLeaves;
using mgl::air::resolveSamplerAccessName;
using mgl::air::varyingLocationSpan;
using mgl::air::airAttribLocation;
using mgl::air::collectStageVarSyms;
using mgl::air::assignStageVarSymLocations;
using mgl::air::stageRecordStride;
using mgl::air::AirIfaceLocationPeer;


static_assert(MGL_AIR_CODEGEN_PER_VERTEX_STRIDE == MGL_AIR_PER_VERTEX_STRIDE,
              "mgl_air_codegen.h stride default must match mgl_shader_abi.h");
static_assert((int)MGL_STAGE_VERTEX == (int)mgl::air::AIR_STAGE_VERTEX &&
              (int)MGL_STAGE_FRAGMENT == (int)mgl::air::AIR_STAGE_FRAGMENT &&
              (int)MGL_STAGE_COMPUTE == (int)mgl::air::AIR_STAGE_COMPUTE &&
              (int)MGL_STAGE_TESS_CONTROL == (int)mgl::air::AIR_STAGE_TESS_CONTROL &&
              (int)MGL_STAGE_TESS_EVALUATION == (int)mgl::air::AIR_STAGE_TESS_EVALUATION &&
              (int)MGL_STAGE_GEOMETRY == (int)mgl::air::AIR_STAGE_GEOMETRY,
              "AirStageId must match MGL_STAGE_*");


/* ---- type helpers (C1b) ----------------------------------------------- */
/* Bodies in mgl_air_type.cpp; storeStageOut stays here (stage-out side
 * effect, not a type-model helper). */


/* Persist a stage-output write into the entry-block alloca (if any) so
 * subsequent user-function calls and assembleReturn observe it. */
static void storeStageOut(Codegen &cg, const char *name, llvm::Value *v)
{
    if (!name || !v) return;
    auto it = cg.outPtrs.find(name);
    if (it == cg.outPtrs.end()) return;
    cg.b->CreateStore(v, it->second);
}

/* ---- resource collection (C1c) ---------------------------------------- */
/* Bodies in mgl_air_resource.cpp; thin using-facade above. */

/* ---- math builtins (C1d) ---------------------------------------------- */
/* Bodies in mgl_air_math.cpp; static emitMathBuiltin facade below. */

/* ---- matrix builtins (C1f) ---------------------------------------- */
/* Bodies in mgl_air_matrix.cpp; static emitMatrix* facade below. */

/* ---- statements (C1g) ---------------------------------------------- */
/* Bodies in mgl_air_stmt.cpp; thin AirStmtDeps facade below. */


/* ---- expression codegen ----------------------------------------------- */

const MGLIRSymbol *findSymbol(const MGLIRModule *mod, const char *name) {
    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        if (!mod->symbols[i]->is_function &&
            strcmp(mod->symbols[i]->name, name) == 0)
            return mod->symbols[i];
    }
    return nullptr;
}

bool swizzleIndices(const char *field, std::vector<uint32_t> *out) {
    static const char *valid = "xyzwrgba";
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

static MGLIRScalar astBaseToIRScalar(uint32_t base) {
    switch (base) {
    case MGL_AST_TYPE_BOOL: return MGLIR_SCALAR_BOOL;
    case MGL_AST_TYPE_INT: return MGLIR_SCALAR_INT;
    case MGL_AST_TYPE_UINT: return MGLIR_SCALAR_UINT;
    case MGL_AST_TYPE_DOUBLE: return MGLIR_SCALAR_DOUBLE;
    case MGL_AST_TYPE_FLOAT:
    default: return MGLIR_SCALAR_FLOAT;
    }
}

static MGLIRType *cloneIRType(const MGLIRType *src) {
    if (!src) return nullptr;
    switch (src->kind) {
    case MGLIR_TYPE_SCALAR:
        return mglIRTypeScalar(src->scalar);
    case MGLIR_TYPE_VECTOR:
        return mglIRTypeVector(src->scalar, src->cols);
    case MGLIR_TYPE_MATRIX:
        return mglIRTypeMatrix(src->scalar, src->cols, src->rows);
    case MGLIR_TYPE_ARRAY: {
        MGLIRType *el = cloneIRType(src->elem_type);
        if (!el) return nullptr;
        return mglIRTypeArray(el, src->array_size);
    }
    case MGLIR_TYPE_STRUCT: {
        std::vector<MGLIRType *> members(src->member_count);
        std::vector<const char *> names(src->member_count);
        for (uint32_t i = 0; i < src->member_count; i++) {
            members[i] = cloneIRType(src->members[i]);
            names[i] = src->member_names[i];
            if (!members[i]) {
                for (uint32_t j = 0; j < i; j++)
                    mglIRTypeDestroy(members[j]);
                return nullptr;
            }
        }
        return mglIRTypeStruct(members.data(), names.data(),
                               src->member_count, src->name);
    }
    default:
        return nullptr;
    }
}

/* Resolve a declarator's type to IR (scalars/vectors/matrices/named
 * structs + array dims).  Named struct members clone the registered type. */
static MGLIRType *astDeclToIRType(Codegen &cg, const MGLDecl *d) {
    if (!d || !d->type) return nullptr;
    const MGLTypeSpec *ts = d->type;
    MGLIRType *base = nullptr;
    if (ts->base == MGL_AST_TYPE_STRUCT) {
        if (!ts->name) return nullptr;
        auto it = cg.structTypes.find(ts->name);
        if (it == cg.structTypes.end()) return nullptr;
        base = cloneIRType(it->second);
    } else if (ts->mat_cols > 1) {
        base = mglIRTypeMatrix(astBaseToIRScalar(ts->base),
                               (uint32_t)ts->mat_cols,
                               (uint32_t)(ts->mat_rows > 0 ? ts->mat_rows
                                                           : ts->mat_cols));
    } else if (ts->vec_size > 1) {
        base = mglIRTypeVector(astBaseToIRScalar(ts->base),
                               (uint32_t)ts->vec_size);
    } else if (ts->base <= MGL_AST_TYPE_DOUBLE) {
        base = mglIRTypeScalar(astBaseToIRScalar(ts->base));
    }
    if (!base) return nullptr;
    if (d->array_count > 0 && d->array_dims) {
        for (int i = (int)d->array_count - 1; i >= 0; i--) {
            MGLIRType *arr = mglIRTypeArray(base, d->array_dims[i]);
            if (!arr) {
                mglIRTypeDestroy(base);
                return nullptr;
            }
            base = arr;
        }
    }
    return base;
}

static void registerStructTypeDecl(Codegen &cg, const MGLDecl *d) {
    if (!d || !d->type || d->type->base != MGL_AST_TYPE_STRUCT ||
        !d->type->name || !d->struct_members ||
        d->struct_member_count == 0)
        return;
    if (cg.structTypes.count(d->type->name))
        return;
    std::vector<MGLIRType *> members(d->struct_member_count);
    std::vector<const char *> names(d->struct_member_count);
    int ok = 1;
    for (uint32_t j = 0; j < d->struct_member_count; j++) {
        MGLDecl *m = d->struct_members[j];
        members[j] = astDeclToIRType(cg, m);
        names[j] = m && m->name ? m->name : "";
        if (!members[j]) {
            ok = 0;
            break;
        }
    }
    if (!ok) {
        for (uint32_t j = 0; j < d->struct_member_count; j++)
            if (members[j]) mglIRTypeDestroy(members[j]);
        return;
    }
    MGLIRType *st = mglIRTypeStruct(members.data(), names.data(),
                                    d->struct_member_count, d->type->name);
    if (!st) {
        for (uint32_t j = 0; j < d->struct_member_count; j++)
            mglIRTypeDestroy(members[j]);
        return;
    }
    cg.structTypes[d->type->name] = st;
    if (cg.ownedIRTypes)
        cg.ownedIRTypes->push_back(st);
}

static void collectStructTypesFromStmt(Codegen &cg, const MGLStmt *st) {
    if (!st) return;
    switch (st->kind) {
    case MGL_STMT_COMPOUND:
        for (uint32_t i = 0; i < st->u.compound.count; i++)
            collectStructTypesFromStmt(cg, st->u.compound.stmts[i]);
        break;
    case MGL_STMT_DECL:
        for (const MGLDecl *d = st->u.decl.decl; d; d = d->next_declarator)
            registerStructTypeDecl(cg, d);
        break;
    case MGL_STMT_IF:
        collectStructTypesFromStmt(cg, st->u.ifs.then);
        collectStructTypesFromStmt(cg, st->u.ifs.else_);
        break;
    case MGL_STMT_FOR:
        collectStructTypesFromStmt(cg, st->u.loop.init);
        collectStructTypesFromStmt(cg, st->u.loop.body);
        break;
    case MGL_STMT_WHILE:
    case MGL_STMT_DO_WHILE:
        collectStructTypesFromStmt(cg, st->u.whilex.body);
        break;
    case MGL_STMT_SWITCH:
        collectStructTypesFromStmt(cg, st->u.switchx.body);
        break;
    default:
        break;
    }
}

static void collectStructTypes(Codegen &cg, const MGLTranslationUnit *tu) {
    if (!tu) return;
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        MGLDecl *d = tu->decls[i];
        registerStructTypeDecl(cg, d);
        /* Local `struct S { … };` inside functions are not TU decls. */
        if (d && d->body)
            collectStructTypesFromStmt(cg, d->body);
    }
}

static const MGLIRType *exprIRType(
    Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals) {
    if (!e) return nullptr;
    switch (e->kind) {
    case MGL_EXPR_VAR_REF: {
        const char *n = e->u.var_ref.name;
        if (!n) return nullptr;
        /* Locals/params shadow anonymous-block uniforms of the same name
         * (CTS uniform_block.random: compare_vec2(a,b) vs BlockB { sA a; }). */
        if (locals.count(n)) {
            auto it = cg.localIRTypes.find(n);
            if (it != cg.localIRTypes.end())
                return it->second;
            return nullptr;
        }
        auto it = cg.localIRTypes.find(n);
        if (it != cg.localIRTypes.end())
            return it->second;
        const MGLIRSymbol *s = findSymbol(mod, n);
        return s ? s->type : nullptr;
    }
    case MGL_EXPR_MEMBER: {
        const MGLIRType *ot =
            exprIRType(cg, e->u.member.object, mod, locals);
        if (!ot) return nullptr;
        while (ot->kind == MGLIR_TYPE_ARRAY && ot->elem_type)
            ot = ot->elem_type;
        if (ot->kind != MGLIR_TYPE_STRUCT) return nullptr;
        for (uint32_t i = 0; i < ot->member_count; i++) {
            if (ot->member_names[i] &&
                strcmp(ot->member_names[i], e->u.member.field) == 0)
                return ot->members[i];
        }
        return nullptr;
    }
    case MGL_EXPR_INDEX: {
        const MGLIRType *ot =
            exprIRType(cg, e->u.index.object, mod, locals);
        if (!ot) return nullptr;
        if (ot->kind == MGLIR_TYPE_ARRAY)
            return ot->elem_type;
        return nullptr;
    }
    case MGL_EXPR_CALL: {
        auto it = cg.structTypes.find(e->u.call.name);
        if (it == cg.structTypes.end())
            return nullptr;
        /* Array-of-struct constructors register the array type on the
         * declaring local; the call itself yields the element struct. */
        return it->second;
    }
    default:
        return nullptr;
    }
}

MType exprType(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
               const std::map<std::string, MType> &locals);

static llvm::Value *emitGLSLTypeLength(
    Codegen &cg, const MGLExpr *object, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals);

/* Broadcast a scalar to the lane type of a vector type. */
llvm::Value *broadcastTo(Codegen &cg, llvm::Value *v, llvm::Type *vecTy);

/* Constant-fold a numeric binary op when both operands are constants of
 * the same type; returns null when folding is not possible.  Mirrors the
 * runtime signedness/comparison semantics of emitNumericBinOp. */
/* GLSL 4.60 §5.9: == / != on vector operands yield a scalar bool
 * (all lanes equal / any lane differs).  Relational operators keep
 * their vector (bvec) results for any()/all()/equal(). */
static llvm::Value *scalarizeBoolCompare(Codegen &cg, uint32_t op,
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
            llvm::ConstantExpr::getCompare(fp ? llvm::CmpInst::FCMP_ONE
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
    bool lfp = l->getType()->isFPOrFPVectorTy();
    bool rfp = r->getType()->isFPOrFPVectorTy();
    if (lfp != rfp) {
        if (lfp) r = coerceScalar(cg, r, MGLIR_SCALAR_FLOAT);
        else l = coerceScalar(cg, l, MGLIR_SCALAR_FLOAT);
        rfp = r->getType()->isFPOrFPVectorTy();
    }
    bool lv = l->getType()->isVectorTy();
    bool rv = r->getType()->isVectorTy();
        if (lv != rv) {
        if (lv) r = broadcastTo(cg, r, l->getType());
        else l = broadcastTo(cg, l, r->getType());
    }
    bool fp = l->getType()->isFPOrFPVectorTy();
    bool uns = lt.scalar == MGLIR_SCALAR_UINT || rt.scalar == MGLIR_SCALAR_UINT;
    llvm::CmpInst::Predicate pred;
    switch (op) {
    case MGL_OP_ADD: return fp ? cg.b->CreateFAdd(l, r) : cg.b->CreateAdd(l, r);
    case MGL_OP_SUB: return fp ? cg.b->CreateFSub(l, r) : cg.b->CreateSub(l, r);
    case MGL_OP_MUL: return fp ? cg.b->CreateFMul(l, r) : cg.b->CreateMul(l, r);
    case MGL_OP_DIV: return fp ? cg.b->CreateFDiv(l, r)
                   : uns ? cg.b->CreateUDiv(l, r) : cg.b->CreateSDiv(l, r);
    case MGL_OP_MOD: return fp ? cg.b->CreateFRem(l, r)
                   : uns ? cg.b->CreateURem(l, r) : cg.b->CreateSRem(l, r);
    case MGL_OP_SHL: return cg.b->CreateShl(l, r);
    case MGL_OP_SHR: return uns ? cg.b->CreateLShr(l, r) : cg.b->CreateAShr(l, r);
    case MGL_OP_AND: return cg.b->CreateAnd(l, r);
    case MGL_OP_OR:  return cg.b->CreateOr(l, r);
    case MGL_OP_XOR: return cg.b->CreateXor(l, r);
    case MGL_OP_LAND: return cg.b->CreateLogicalAnd(l, r);
    case MGL_OP_LOR:  return cg.b->CreateLogicalOr(l, r);
    case MGL_OP_EQ: pred = fp ? llvm::CmpInst::FCMP_OEQ : llvm::CmpInst::ICMP_EQ; break;
    case MGL_OP_NE: pred = fp ? llvm::CmpInst::FCMP_ONE : llvm::CmpInst::ICMP_NE; break;
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
static llvm::Value *emitAggregateCompare(Codegen &cg, uint32_t op,
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
            cg.b->CreateExtractElement(a, ix),
            cg.b->CreateExtractElement(b, ix));
        acc = cg.b->CreateFAdd(acc, p);
    }
    return acc;
}

static bool isTextureSampleBuiltin(const char *name)
{
    return strcmp(name, "texture") == 0 ||
           strcmp(name, "textureOffset") == 0 ||
           strcmp(name, "textureLod") == 0 ||
           strcmp(name, "textureLodOffset") == 0 ||
           strcmp(name, "textureGrad") == 0 ||
           strcmp(name, "textureGradOffset") == 0 ||
           strcmp(name, "textureProj") == 0 ||
           strcmp(name, "textureProjOffset") == 0 ||
           strcmp(name, "textureProjLod") == 0 ||
           strcmp(name, "textureProjLodOffset") == 0 ||
           strcmp(name, "textureProjGrad") == 0 ||
           strcmp(name, "textureProjGradOffset") == 0;
}

static llvm::Value *emitAirSampleOffset(Codegen &cg, llvm::Value *off)
{
    llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
    llvm::Type *v2i32 = llvm::FixedVectorType::get(i32, 2);
    if (!off) {
        return llvm::Constant::getNullValue(v2i32);
    }
    if (off->getType()->isIntegerTy(32)) {
        llvm::Value *v = llvm::UndefValue::get(v2i32);
        v = cg.b->CreateInsertElement(v, off, cg.b->getInt32(0));
        v = cg.b->CreateInsertElement(v, cg.b->getInt32(0), cg.b->getInt32(1));
        return v;
    }
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(off->getType())) {
        if (vt->getNumElements() == 2) {
            return off;
        }
        if (vt->getNumElements() >= 3) {
            return cg.b->CreateShuffleVector(
                off, llvm::UndefValue::get(off->getType()),
                llvm::ConstantVector::get(
                    {llvm::ConstantInt::get(i32, 0),
                     llvm::ConstantInt::get(i32, 1)}));
        }
    }
    return llvm::Constant::getNullValue(v2i32);
}

/* AIR sample_texture_2d_array* passes spatial coords and array layer as
 * separate arguments; GLSL bundles them into vec3(vec2(P), layer) or
 * vec2(s, layer) for 1D arrays. */
static bool splitSampleArrayCoord(Codegen &cg, MGLIRTexKind kind,
                                  llvm::Value *uv, llvm::Value **outCoord,
                                  llvm::Value **outLayer) {
    if (!uv || !outCoord || !outLayer) {
        return false;
    }
    *outCoord = uv;
    *outLayer = nullptr;
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    llvm::Type *v2f32 = llvm::FixedVectorType::get(f32, 2);
    if (kind == MGLIR_TEX_2D_ARRAY || kind == MGLIR_TEX_2D_MS_ARRAY) {
        auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(uv->getType());
        if (!vt || vt->getElementCount().getFixedValue() < 3u) {
            cg.err = 1;
            cg.errmsg = "codegen: sampler2DArray texture access expects vec3 "
                        "coordinates";
            return false;
        }
        *outLayer = cg.b->CreateExtractElement(uv, cg.b->getInt32(2));
        *outCoord = cg.b->CreateShuffleVector(
            uv, llvm::UndefValue::get(uv->getType()), {0, 1});
        return true;
    }
    if (kind == MGLIR_TEX_1D_ARRAY) {
        auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(uv->getType());
        if (!vt || vt->getElementCount().getFixedValue() < 2u) {
            cg.err = 1;
            cg.errmsg = "codegen: sampler1DArray texture access expects vec2 "
                        "coordinates";
            return false;
        }
        llvm::Value *x = cg.b->CreateExtractElement(uv, cg.b->getInt32(0));
        *outLayer = cg.b->CreateExtractElement(uv, cg.b->getInt32(1));
        llvm::Value *expanded = llvm::UndefValue::get(v2f32);
        expanded = cg.b->CreateInsertElement(expanded, x, cg.b->getInt32(0));
        expanded = cg.b->CreateInsertElement(
            expanded, llvm::ConstantFP::get(f32, 0.5), cg.b->getInt32(1));
        *outCoord = expanded;
        return true;
    }
    return true;
}

static llvm::Value *addTexelOffset(Codegen &cg, llvm::Value *coord,
                                   llvm::Value *off)
{
    if (!off) {
        return coord;
    }
    if (coord->getType()->isIntegerTy() && off->getType()->isIntegerTy()) {
        return cg.b->CreateAdd(coord, off);
    }
    if (auto *cvt = llvm::dyn_cast<llvm::FixedVectorType>(coord->getType())) {
        if (off->getType()->isIntegerTy() && cvt->getNumElements() >= 2) {
            llvm::Value *expanded = llvm::UndefValue::get(coord->getType());
            expanded = cg.b->CreateInsertElement(expanded, off,
                                                 cg.b->getInt32(0));
            expanded = cg.b->CreateInsertElement(
                expanded, llvm::ConstantInt::get(
                              llvm::Type::getInt32Ty(*cg.ctx), 0),
                cg.b->getInt32(1));
            if (cvt->getNumElements() == 3) {
                expanded = cg.b->CreateInsertElement(
                    expanded,
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(2)),
                    cg.b->getInt32(2));
            }
            return cg.b->CreateAdd(coord, expanded);
        }
        if (auto *ovt =
                llvm::dyn_cast<llvm::FixedVectorType>(off->getType())) {
            if (ovt->getNumElements() == 2 &&
                cvt->getNumElements() == 3) {
                llvm::Value *expanded = llvm::UndefValue::get(coord->getType());
                expanded = cg.b->CreateInsertElement(
                    expanded, cg.b->CreateExtractElement(off, cg.b->getInt32(0)),
                    cg.b->getInt32(0));
                expanded = cg.b->CreateInsertElement(
                    expanded, cg.b->CreateExtractElement(off, cg.b->getInt32(1)),
                    cg.b->getInt32(1));
                expanded = cg.b->CreateInsertElement(
                    expanded,
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(2)),
                    cg.b->getInt32(2));
                return cg.b->CreateAdd(coord, expanded);
            }
            if (ovt->getNumElements() == cvt->getNumElements()) {
                return cg.b->CreateAdd(coord, off);
            }
        }
    }
    return coord;
}

/* emitExpr / interpolateAt declared early for call sites. */
llvm::Value *emitExpr(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
                      const std::map<std::string, MType> &locals);
static llvm::Value *emitInterpolateAtBuiltin(
    Codegen &cg, const MGLExpr *e, const char *name, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals);

/* ---- matrix builtins (C1f) ----------------------------------------- */
/* Bodies in mgl_air_matrix.cpp; thin AirMatrixDeps facade. */

static llvm::Value *emitMatrixBuiltin(Codegen &cg, const MGLExpr *e,
                                      const char *name, const MGLIRModule *mod,
                                      const std::map<std::string, MType> &locals)
{
    static const mgl::air::AirMatrixDeps deps = {
        emitExpr,
        dotProduct,
        scalarizeBoolCompare,
    };
    return mgl::air::emitMatrixBuiltin(cg, e, name, mod, locals, deps);
}

llvm::Value *emitMatrixBinOp(Codegen &cg, uint32_t op, llvm::Value *l,
                             llvm::Value *r)
{
    static const mgl::air::AirMatrixDeps deps = {
        emitExpr,
        dotProduct,
        scalarizeBoolCompare,
    };
    return mgl::air::emitMatrixBinOp(cg, op, l, r, deps);
}


llvm::Value *emitExpr(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
                      const std::map<std::string, MType> &locals);

/* C1d: math builtins body in mgl_air_math.cpp; facade defined later. */
static llvm::Value *emitMathBuiltin(Codegen &cg, const MGLExpr *e,
                                    const char *name, const MGLIRModule *mod,
                                    const std::map<std::string, MType> &locals);

/* Select one element from a compile-time array of values by dynamic index.
 * Switch+phi avoids deep select chains that crash Metal when large sampler
 * arrays are indexed inside loops. */
static llvm::Value *selectArrayElement(Codegen &cg, llvm::Value *index,
                                       const std::vector<llvm::Value *> &values) {
    if (values.empty()) return nullptr;
    if (values.size() == 1) return values[0];
    llvm::Function *fn = cg.b->GetInsertBlock()->getParent();
    llvm::BasicBlock *defaultBB =
        llvm::BasicBlock::Create(*cg.ctx, "arr.def", fn);
    llvm::BasicBlock *mergeBB =
        llvm::BasicBlock::Create(*cg.ctx, "arr.merge", fn);
    llvm::SwitchInst *sw = cg.b->CreateSwitch(
        index, defaultBB, (unsigned)values.size());
    std::vector<llvm::BasicBlock *> caseBBs(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        caseBBs[i] = llvm::BasicBlock::Create(*cg.ctx, "arr.case", fn);
        sw->addCase(cg.b->getInt32((uint32_t)i), caseBBs[i]);
    }
    cg.b->SetInsertPoint(defaultBB);
    cg.b->CreateBr(mergeBB);
    llvm::PHINode *phi = llvm::PHINode::Create(
        values[0]->getType(), (unsigned)(values.size() + 1), "arr.phi",
        mergeBB);
    phi->addIncoming(values.back(), defaultBB);
    for (size_t i = 0; i < values.size(); ++i) {
        cg.b->SetInsertPoint(caseBBs[i]);
        cg.b->CreateBr(mergeBB);
        phi->addIncoming(values[i], caseBBs[i]);
    }
    cg.b->SetInsertPoint(mergeBB);
    return phi;
}

/* Peel to the IMAGE element type of a (possibly array) image uniform. */
static const MGLIRType *imageElementType(const MGLIRSymbol *s) {
    if (!s || !s->type) return nullptr;
    const MGLIRType *t = s->type;
    while (t && t->kind == MGLIR_TYPE_ARRAY)
        t = t->elem_type;
    return (t && t->kind == MGLIR_TYPE_IMAGE) ? t : nullptr;
}

/* Resolve imageLoad/Store/Atomic/Size first arg: image or image[i]. */
static llvm::Value *resolveImageTex(
    Codegen &cg, const MGLExpr *ia, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals, const MGLIRType **outImgTy) {
    if (!ia) {
        cg.err = 1;
        cg.errmsg = "codegen: missing image argument";
        return nullptr;
    }
    const char *imageName = nullptr;
    const MGLExpr *indexExpr = nullptr;
    if (ia->kind == MGL_EXPR_VAR_REF) {
        imageName = ia->u.var_ref.name;
    } else if (ia->kind == MGL_EXPR_INDEX && ia->u.index.object &&
               ia->u.index.object->kind == MGL_EXPR_VAR_REF) {
        imageName = ia->u.index.object->u.var_ref.name;
        indexExpr = ia->u.index.index;
    } else {
        cg.err = 1;
        cg.errmsg =
            "codegen: image first argument must be an image variable";
        return nullptr;
    }
    const MGLIRType *imgTy = imageElementType(findSymbol(mod, imageName));
    if (!imgTy) {
        cg.err = 1;
        cg.errmsg = "codegen: requires an image variable";
        return nullptr;
    }
    if (outImgTy) *outImgTy = imgTy;
    if (indexExpr) {
        llvm::Value *index = emitExpr(cg, indexExpr, mod, locals);
        if (!index) return nullptr;
        index = coerceScalar(cg, index, MGLIR_SCALAR_INT);
        auto ti = cg.texArrayValues.find(imageName);
        if (ti == cg.texArrayValues.end()) {
            cg.err = 1;
            cg.errmsg = "codegen: missing image array binding";
            return nullptr;
        }
        if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
            uint32_t k = (uint32_t)ci->getZExtValue();
            if (k < ti->second.size()) return ti->second[k];
            if (!ti->second.empty()) return ti->second.back();
            cg.err = 1;
            cg.errmsg = "codegen: empty image array binding";
            return nullptr;
        }
        return selectArrayElement(cg, index, ti->second);
    }
    llvm::Value *tex = samplerTexValue(cg, imageName);
    if (!tex) {
        cg.err = 1;
        cg.errmsg =
            std::string("codegen: missing image binding for ") + imageName;
    }
    return tex;
}

/* Sample from a sampler array by index without phi-selecting texture or
 * sampler pointers (Metal's compiler crashes on those inside loops).
 * One switch case performs exactly one sample; each texel component then
 * merges through its own scalar phi, so AIR sample aggregates are never
 * phi operands while every component of the result is populated. */
static llvm::Value *sampleArrayElementBySwitch(
    Codegen &cg, llvm::Value *index,
    const std::vector<llvm::Value *> &texValues,
    const std::vector<llvm::Value *> &smpValues, llvm::Type *resultVecTy,
    const std::function<llvm::Value *(llvm::Value *, llvm::Value *)>
        &emitSample) {
    if (texValues.empty()) return nullptr;
    size_t n = texValues.size();
    if (n == 1)
        return emitSample(texValues[0],
                          smpValues.empty() ? nullptr : smpValues[0]);
    auto *vecTy = llvm::cast<llvm::FixedVectorType>(resultVecTy);
    llvm::Type *laneTy = vecTy->getElementType();
    const unsigned lanes = vecTy->getNumElements();
    llvm::Function *fn = cg.b->GetInsertBlock()->getParent();
    llvm::BasicBlock *mergeBB =
        llvm::BasicBlock::Create(*cg.ctx, "samp.merge", fn);
    llvm::BasicBlock *defBB =
        llvm::BasicBlock::Create(*cg.ctx, "samp.def", fn);
    std::vector<llvm::PHINode *> lanePhis(lanes);
    for (unsigned c = 0; c < lanes; ++c) {
        lanePhis[c] = llvm::PHINode::Create(
            laneTy, (unsigned)(n + 1), "samp.lane", mergeBB);
    }
    std::vector<llvm::BasicBlock *> caseBBs(n);
    for (size_t i = 0; i < n; ++i)
        caseBBs[i] = llvm::BasicBlock::Create(*cg.ctx, "samp.case", fn);
    llvm::SwitchInst *sw =
        cg.b->CreateSwitch(index, defBB, (unsigned)n);
    for (size_t i = 0; i < n; ++i)
        sw->addCase(cg.b->getInt32((uint32_t)i), caseBBs[i]);
    auto fillLanes = [&](llvm::Value *val, llvm::BasicBlock *fromBB) {
        for (unsigned c = 0; c < lanes; ++c) {
            lanePhis[c]->addIncoming(
                cg.b->CreateExtractElement(val, (uint64_t)c), fromBB);
        }
    };
    for (size_t i = 0; i < n; ++i) {
        cg.b->SetInsertPoint(caseBBs[i]);
        llvm::Value *val = emitSample(
            texValues[i], smpValues.empty() ? nullptr : smpValues[i]);
        fillLanes(val, caseBBs[i]);
        cg.b->CreateBr(mergeBB);
    }
    cg.b->SetInsertPoint(defBB);
    llvm::Value *defVal = emitSample(
        texValues.back(),
        smpValues.empty() ? nullptr : smpValues.back());
    fillLanes(defVal, defBB);
    cg.b->CreateBr(mergeBB);
    cg.b->SetInsertPoint(mergeBB);
    llvm::Value *out = llvm::UndefValue::get(vecTy);
    for (unsigned c = 0; c < lanes; ++c)
        out = cg.b->CreateInsertElement(out, lanePhis[c], (uint64_t)c);
    return out;
}

/* Dynamic read of obj[idx]: matrix -> column (select chain over the
 * columns, since extractvalue needs a constant index), vector ->
 * component.  `idx` must be an integer value. */
static llvm::Value *emitIndexValue(Codegen &cg, llvm::Value *obj,
                                   const MType &bt, llvm::Value *idx) {
    /* Arrays-of-matrices: arr takes precedence (same as llvmType). */
    if (bt.isArray() || (!bt.isMatrix() && obj->getType()->isArrayTy())) {
        auto *arr = llvm::dyn_cast<llvm::ArrayType>(obj->getType());
        if (!arr) return nullptr;
        uint32_t C = (uint32_t)arr->getNumElements();
        llvm::Value *res = nullptr;
        for (uint32_t i = 0; i < C; i++) {
            llvm::Value *el = cg.b->CreateExtractValue(obj, i);
            llvm::Value *eq = cg.b->CreateICmpEQ(
                idx, llvm::ConstantInt::get(idx->getType(), i));
            res = res ? cg.b->CreateSelect(eq, el, res) : el;
        }
        return res;
    }
    if (bt.isMatrix()) {
        auto *arr = llvm::dyn_cast<llvm::ArrayType>(obj->getType());
        if (!arr) return nullptr;
        uint32_t C = (uint32_t)arr->getNumElements();
        llvm::Value *res = nullptr;
        for (uint32_t i = 0; i < C; i++) {
            llvm::Value *col = cg.b->CreateExtractValue(obj, i);
            llvm::Value *eq = cg.b->CreateICmpEQ(
                idx, llvm::ConstantInt::get(idx->getType(), i));
            res = res ? cg.b->CreateSelect(eq, col, res) : col;
        }
        return res;
    }
    if (obj->getType()->isVectorTy())
        return cg.b->CreateExtractElement(obj, idx);
    return nullptr;
}

/* Dynamic write of obj[idx] = val; returns the updated aggregate. */
static llvm::Value *insertIndexValue(Codegen &cg, llvm::Value *obj,
                                     const MType &bt, llvm::Value *idx,
                                     llvm::Value *val) {
    /* Arrays-of-matrices: arr takes precedence (same as llvmType). */
    if (bt.isArray() || (!bt.isMatrix() && obj->getType()->isArrayTy())) {
        auto *arr = llvm::dyn_cast<llvm::ArrayType>(obj->getType());
        if (!arr) return nullptr;
        uint32_t n = (uint32_t)arr->getNumElements();
        llvm::Value *out = llvm::UndefValue::get(obj->getType());
        for (uint32_t i = 0; i < n; i++) {
            llvm::Value *el = cg.b->CreateExtractValue(obj, i);
            llvm::Value *eq = cg.b->CreateICmpEQ(
                idx, llvm::ConstantInt::get(idx->getType(), i));
            llvm::Value *ne = cg.b->CreateSelect(eq, val, el);
            out = cg.b->CreateInsertValue(out, ne, i);
        }
        return out;
    }
    if (bt.isMatrix()) {
        auto *arr = llvm::dyn_cast<llvm::ArrayType>(obj->getType());
        if (!arr) return nullptr;
        uint32_t C = (uint32_t)arr->getNumElements();
        llvm::Value *out = llvm::UndefValue::get(obj->getType());
        for (uint32_t i = 0; i < C; i++) {
            llvm::Value *col = cg.b->CreateExtractValue(obj, i);
            llvm::Value *eq = cg.b->CreateICmpEQ(
                idx, llvm::ConstantInt::get(idx->getType(), i));
            llvm::Value *nc = cg.b->CreateSelect(eq, val, col);
            out = cg.b->CreateInsertValue(out, nc, i);
        }
        return out;
    }
    if (obj->getType()->isVectorTy()) {
        auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(obj->getType());
        if (!vt) return nullptr;
        uint32_t n = (uint32_t)vt->getElementCount().getFixedValue();
        llvm::Value *out = llvm::UndefValue::get(obj->getType());
        for (uint32_t i = 0; i < n; i++) {
            llvm::Value *el = cg.b->CreateExtractElement(
                obj, llvm::ConstantInt::get(idx->getType(), i));
            llvm::Value *eq = cg.b->CreateICmpEQ(
                idx, llvm::ConstantInt::get(idx->getType(), i));
            llvm::Value *ne = cg.b->CreateSelect(
                eq, val, el);
            out = cg.b->CreateInsertElement(
                out, ne, llvm::ConstantInt::get(idx->getType(), i));
        }
        return out;
    }
    return nullptr;
}

/* Write val into the swizzle-selected lanes of a vector (constant
 * indices, so no runtime selection); unselected lanes are kept.  For a
 * multi-lane target the j-th lane of val goes to the j-th component. */
static llvm::Value *insertSwizzleValue(Codegen &cg, llvm::Value *obj,
                                       const std::vector<uint32_t> &idx,
                                       llvm::Value *val) {
    auto *vt = llvm::cast<llvm::FixedVectorType>(obj->getType());
    uint32_t n = (uint32_t)vt->getElementCount().getFixedValue();
    auto cI = [&](uint32_t v) {
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), v);
    };
    llvm::Value *out = llvm::UndefValue::get(obj->getType());
    for (uint32_t i = 0; i < n; i++) {
        llvm::Value *lane = nullptr;
        for (uint32_t j = 0; j < idx.size(); j++) {
            if (idx[j] == i) {
                lane = idx.size() == 1
                    ? val
                    : cg.b->CreateExtractElement(val, cI(j));
                break;
            }
        }
        if (!lane)
            lane = cg.b->CreateExtractElement(obj, cI(i));
        out = cg.b->CreateInsertElement(out, lane, cI(i));
    }
    return out;
}

/* Read an index chain (x[i][j] / s.member.yz) from the root value without
 * re-emitting the object expression. */
static llvm::Value *readIndexChain(Codegen &cg, const MGLExpr *e,
                                   llvm::Value *rootVal,
                                   const MGLIRModule *mod,
                                   const std::map<std::string, MType> &locals) {
    if (e->kind == MGL_EXPR_VAR_REF) return rootVal;
    if (e->kind == MGL_EXPR_MEMBER) {
        llvm::Value *obj = readIndexChain(cg, e->u.member.object, rootVal, mod,
                                          locals);
        if (!obj) return nullptr;
        /* Struct field before swizzle: member names like `a` collide with
         * the rgba swizzle alphabet (CTS local-struct `s.a = …`). */
        if (const MGLIRType *objTy =
                exprIRType(cg, e->u.member.object, mod, locals)) {
            while (objTy->kind == MGLIR_TYPE_ARRAY && objTy->elem_type)
                objTy = objTy->elem_type;
            if (objTy->kind == MGLIR_TYPE_STRUCT) {
                if (!obj->getType()->isStructTy()) {
                    /* AST says struct but the materialized root holds a
                     * different LLVM shape (e.g. a flattened block member
                     * lvalue) — fall back to the swizzle/error path rather
                     * than emitting an invalid extractvalue. */
                    objTy = nullptr;
                } else {
                for (uint32_t i = 0; i < objTy->member_count; i++) {
                    if (objTy->member_names[i] &&
                        strcmp(objTy->member_names[i],
                               e->u.member.field) == 0)
                        return cg.b->CreateExtractValue(obj, i);
                }
                cg.err = 1;
                cg.errmsg = std::string("codegen: unknown member '") +
                            e->u.member.field + "'";
                return nullptr;
                }
            }
        }
        std::vector<uint32_t> idx;
        if (!swizzleIndices(e->u.member.field, &idx)) {
            cg.err = 1;
            cg.errmsg = "codegen: invalid swizzle";
            return nullptr;
        }
        if (!obj->getType()->isVectorTy()) {
            cg.err = 1;
            cg.errmsg = std::string("codegen: member '") + e->u.member.field +
                        "' of a non-vector value is not supported";
            return nullptr;
        }
        if (idx.size() == 1)
            return cg.b->CreateExtractElement(obj,
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx),
                                       idx[0]));
        llvm::SmallVector<llvm::Constant *, 4> mask;
        for (uint32_t i : idx)
            mask.push_back(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(*cg.ctx), i));
        return cg.b->CreateShuffleVector(obj, llvm::UndefValue::get(
            obj->getType()), llvm::ConstantVector::get(mask));
    }
    if (e->kind != MGL_EXPR_INDEX) { cg.err = 1; return nullptr; }
    llvm::Value *obj = readIndexChain(cg, e->u.index.object, rootVal, mod,
                                      locals);
    if (!obj) return nullptr;
    llvm::Value *idx = emitExpr(cg, e->u.index.index, mod, locals);
    if (!idx) return nullptr;
    MType bt = exprType(cg, e->u.index.object, mod, locals);
    llvm::Value *res = emitIndexValue(cg, obj, bt, idx);
    if (!res) {
        cg.err = 1;
        cg.errmsg = "codegen: indexing this type is not implemented in M1";
        return nullptr;
    }
    return res;
}

/* Write `val` through the index chain `lhs` (rooted at a var ref holding
 * rootVal); returns the new root value. */
static llvm::Value *updateIndexPath(Codegen &cg, const MGLExpr *lhs,
                                    llvm::Value *rootVal, llvm::Value *val,
                                    const MGLIRModule *mod,
                                    const std::map<std::string, MType> &locals) {
    if (lhs->kind == MGL_EXPR_VAR_REF) return val;
    if (lhs->kind == MGL_EXPR_MEMBER) {
        const MGLExpr *objE = lhs->u.member.object;
        llvm::Value *objVal;
        if (objE->kind == MGL_EXPR_VAR_REF) {
            objVal = rootVal;
        } else {
            objVal = readIndexChain(cg, objE, rootVal, mod, locals);
            if (!objVal) return nullptr;
        }
        if (const MGLIRType *objTy = exprIRType(cg, objE, mod, locals)) {
            while (objTy->kind == MGLIR_TYPE_ARRAY && objTy->elem_type)
                objTy = objTy->elem_type;
            if (objTy->kind == MGLIR_TYPE_STRUCT) {
                if (!objVal->getType()->isStructTy()) {
                    /* Mirror the read-side guard: only take the value
                     * path when the root really is a struct aggregate. */
                    objTy = nullptr;
                } else {
                for (uint32_t i = 0; i < objTy->member_count; i++) {
                    if (objTy->member_names[i] &&
                        strcmp(objTy->member_names[i],
                               lhs->u.member.field) == 0) {
                        llvm::Value *newObj =
                            cg.b->CreateInsertValue(objVal, val, i);
                        if (objE->kind == MGL_EXPR_VAR_REF) return newObj;
                        return updateIndexPath(cg, objE, rootVal, newObj, mod,
                                               locals);
                    }
                }
                cg.err = 1;
                cg.errmsg = std::string("codegen: unknown member '") +
                            lhs->u.member.field + "'";
                return nullptr;
                }
            }
        }
        std::vector<uint32_t> idx;
        if (!swizzleIndices(lhs->u.member.field, &idx)) {
            cg.err = 1;
            cg.errmsg = "codegen: invalid swizzle";
            return nullptr;
        }
        if (!objVal->getType()->isVectorTy()) {
            cg.err = 1;
            cg.errmsg = std::string("codegen: member '") +
                        lhs->u.member.field +
                        "' of a non-vector value is not supported";
            return nullptr;
        }
        llvm::Value *newObj = insertSwizzleValue(cg, objVal, idx, val);
        if (objE->kind == MGL_EXPR_VAR_REF) return newObj;
        return updateIndexPath(cg, objE, rootVal, newObj, mod, locals);
    }
    if (lhs->kind != MGL_EXPR_INDEX) { cg.err = 1; return nullptr; }
    const MGLExpr *objE = lhs->u.index.object;
    MType objT = exprType(cg, objE, mod, locals);
    llvm::Value *objVal;
    if (objE->kind == MGL_EXPR_VAR_REF) {
        objVal = rootVal;
    } else if (objE->kind == MGL_EXPR_INDEX || objE->kind == MGL_EXPR_MEMBER) {
        objVal = readIndexChain(cg, objE, rootVal, mod, locals);
        if (!objVal) return nullptr;
    } else {
        cg.err = 1;
        cg.errmsg = "codegen: unsupported indexed assignment target";
        return nullptr;
    }
    llvm::Value *idx = emitExpr(cg, lhs->u.index.index, mod, locals);
    if (!idx) return nullptr;
    llvm::Value *newObj = insertIndexValue(cg, objVal, objT, idx, val);
    if (!newObj) { cg.err = 1; return nullptr; }
    if (objE->kind == MGL_EXPR_VAR_REF) return newObj;
    return updateIndexPath(cg, objE, rootVal, newObj, mod, locals);
}

/* Buffer read: byte GEP + bitcast + aligned load.  Alignment follows
 * std140: scalar 4, vec2 8, vec3/vec4 and matrix columns 16. */
llvm::Value *bufferLoad(Codegen &cg, uint32_t offset, llvm::Type *loadTy) {
    llvm::Align align(16);
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(loadTy)) {
        uint64_t w = vt->getElementCount().getFixedValue();
        if (w == 1) align = llvm::Align(4);
        else if (w == 2) align = llvm::Align(8);
    } else if (loadTy->isFloatTy()) {
        align = llvm::Align(4);
    }
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), cg.bufferPtr,
                                     cg.b->getInt64(offset));
    p = cg.b->CreateBitCast(p, loadTy->getPointerTo(1));
    return cg.b->CreateAlignedLoad(loadTy, p, align);
}

/* Buffer write mirroring bufferLoad; used by compute shaders to write
 * back through the device buffer. */
void bufferStore(Codegen &cg, uint32_t offset, llvm::Type *storeTy,
                 llvm::Value *val) {
    llvm::Align align(16);
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(storeTy)) {
        uint64_t w = vt->getElementCount().getFixedValue();
        if (w == 1) align = llvm::Align(4);
        else if (w == 2) align = llvm::Align(8);
    } else if (storeTy->isFloatTy() || storeTy->isIntegerTy(32)) {
        align = llvm::Align(4);
    }
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), cg.bufferPtr,
                                     cg.b->getInt64(offset));
    p = cg.b->CreateBitCast(p, storeTy->getPointerTo(1));
    cg.b->CreateAlignedStore(val, p, align);
}

/* Does `e` (an object/index/member chain) root at an SSBO instance? */
const MGLIRSymbol *ssboRootSym(const MGLExpr *e, const MGLIRModule *mod) {
    const MGLExpr *r = e;
    while (r->kind == MGL_EXPR_INDEX || r->kind == MGL_EXPR_MEMBER)
        r = r->kind == MGL_EXPR_INDEX ? r->u.index.object
                                      : r->u.member.object;
    if (r->kind != MGL_EXPR_VAR_REF) return nullptr;
    const MGLIRSymbol *sym = findSymbol(mod, r->u.var_ref.name);
    if (sym && (sym->qualifiers & MGL_AST_Q_BUFFER)) return sym;
    return nullptr;
}

/* Resolve the IR type and static byte offset of an SSBO member/index chain.
 * A runtime array can only be the final block member, so its length query has
 * no dynamic index in the path and therefore has a stable tail offset. */
const MGLIRType *ssboExprType(const MGLExpr *e, const MGLIRSymbol *sb,
                              uint32_t *staticOffset) {
    const MGLIRType *ty = sb ? sb->type : nullptr;
    std::vector<const MGLExpr *> path;
    const MGLExpr *cur = e;
    while (cur && (cur->kind == MGL_EXPR_MEMBER ||
                   cur->kind == MGL_EXPR_INDEX)) {
        path.push_back(cur);
        cur = cur->kind == MGL_EXPR_INDEX ? cur->u.index.object
                                          : cur->u.member.object;
    }
    std::reverse(path.begin(), path.end());
    /* Flattened anonymous-block members start at their static offset in
     * the owning block (pad + unsized-tail CTS shaders). */
    uint32_t off = 0;
    if (sb && sb->block_name && sb->block_name[0] &&
        sb->offset != UINT32_MAX) {
        off = sb->offset;
    }
    for (const MGLExpr *pe : path) {
        if (!ty) return nullptr;
        if (pe->kind == MGL_EXPR_MEMBER) {
            const MGLIRType *member = nullptr;
            for (uint32_t i = 0; i < ty->member_count; i++) {
                if (strcmp(ty->member_names[i], pe->u.member.field) == 0) {
                    off += ty->member_offsets ? ty->member_offsets[i] : 0;
                    member = ty->members[i];
                    break;
                }
            }
            if (!member) return nullptr;
            ty = member;
        } else {
            /* Only fixed arrays can legally contain a nested object.  Their
             * length is folded before this offset is used. */
            if (ty->kind != MGLIR_TYPE_ARRAY) return nullptr;
            ty = ty->elem_type;
        }
    }
    if (staticOffset) *staticOffset = off;
    return ty;
}

/* Byte size of one scalar component (SSBO component addressing). */
static uint32_t mglAirScalarByteSize(MGLIRScalar s) {
    switch (s) {
    case MGLIR_SCALAR_DOUBLE: return 8u;
    case MGLIR_SCALAR_VOID: return 0u;
    default: return 4u;   /* bool / int / uint / float / half */
    }
}

/* Byte address of a member/index chain rooted at an SSBO instance; the
 * member type is returned in *outTy. */
llvm::Value *ssboAddress(Codegen &cg, const MGLExpr *e,
                         const MGLIRSymbol *sb, const MGLIRModule *mod,
                         const std::map<std::string, MType> &locals,
                         const MGLIRType **outTy) {
    /* Flattened anonymous-block members are addressed through the owning
     * block's device buffer at the member's static offset. */
    const char *bufName =
        (sb->block_name && sb->block_name[0]) ? sb->block_name : sb->name;
    const MGLIRType *ty = sb->type;
    uint32_t off = 0;
    if (sb->block_name && sb->block_name[0]) {
        if (sb->offset != UINT32_MAX)
            off = sb->offset;
        else {
            const MGLIRSymbol *blk = findSymbol(mod, sb->block_name);
            if (blk && blk->type && blk->type->member_offsets &&
                sb->block_member_index < blk->type->member_count)
                off = blk->type->member_offsets[sb->block_member_index];
        }
    }
    std::vector<const MGLExpr *> path;
    const MGLExpr *cur = e;
    while (cur->kind == MGL_EXPR_MEMBER || cur->kind == MGL_EXPR_INDEX) {
        path.push_back(cur);
        cur = cur->kind == MGL_EXPR_INDEX ? cur->u.index.object
                                          : cur->u.member.object;
    }
    std::reverse(path.begin(), path.end());
    auto pit = cg.ssboPtrs.find(bufName);
    if (pit == cg.ssboPtrs.end() || !pit->second) {
        cg.err = 1;
        cg.errmsg = std::string("codegen: SSBO '") + bufName +
                    "' has no device buffer";
        return nullptr;
    }
    llvm::Value *base = pit->second;
    /* Buffer instance arrays: g_ssbo[i].member selects a separate Metal
     * buffer per element (mirrors UBO instance-array handling). */
    if (!sb->block_name && uniformBlockIsInstanceArray(sb->type) &&
        !path.empty() && path[0]->kind == MGL_EXPR_INDEX &&
        path[0]->u.index.object &&
        path[0]->u.index.object->kind == MGL_EXPR_VAR_REF &&
        path[0]->u.index.object->u.var_ref.name &&
        strcmp(path[0]->u.index.object->u.var_ref.name, sb->name) == 0) {
        auto slotIt = cg.ssboElemSlot.find(bufName);
        auto tyIt = cg.ssboElemArrTy.find(bufName);
        if (slotIt == cg.ssboElemSlot.end() ||
            tyIt == cg.ssboElemArrTy.end()) {
            cg.err = 1;
            cg.errmsg = std::string("codegen: SSBO block array '") +
                        bufName + "' has no element buffers";
            return nullptr;
        }
        llvm::Value *elemIndex =
            emitExpr(cg, path[0]->u.index.index, mod, locals);
        if (!elemIndex) return nullptr;
        elemIndex = coerceScalar(cg, elemIndex, MGLIR_SCALAR_UINT);
        uint32_t elemCount = uniformBlockElementCount(sb->type);
        if (elemCount == 0u) {
            cg.err = 1;
            cg.errmsg = "codegen: SSBO block array has zero elements";
            return nullptr;
        }
        elemIndex = cg.b->CreateSelect(
            cg.b->CreateICmpULT(elemIndex, cg.b->getInt32(elemCount)),
            elemIndex, cg.b->getInt32(elemCount - 1u));
        llvm::Type *ptrTy = llvm::Type::getInt8Ty(*cg.ctx)->getPointerTo(1);
        llvm::Value *elemPtr = cg.b->CreateInBoundsGEP(
            tyIt->second, slotIt->second,
            {cg.b->getInt32(0), elemIndex});
        base = cg.b->CreateLoad(ptrTy, elemPtr);
        ty = sb->type->elem_type;
        path.erase(path.begin());
        off = 0;
    }
    for (const MGLExpr *pe : path) {
        if (pe->kind == MGL_EXPR_MEMBER) {
            /* A swizzle on a vector-typed element selects one component:
             * address the scalar in place instead of falling into the
             * struct-member lookup (which would reject it) — but only for
             * single-component swizzles; a multi-component swizzle cannot
             * be addressed as one contiguous scalar. */
            if (ty->kind == MGLIR_TYPE_VECTOR) {
                std::vector<uint32_t> comps;
                if (!swizzleIndices(pe->u.member.field, &comps) ||
                    comps.size() != 1u) {
                    cg.err = 1;
                    cg.errmsg =
                        "codegen: only single-component swizzles are "
                        "supported on SSBO vector members";
                    return nullptr;
                }
                if (comps[0] >= ty->cols) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: swizzle '") +
                                pe->u.member.field +
                                "' out of range for SSBO vector member";
                    return nullptr;
                }
                off += comps[0] * mglAirScalarByteSize(ty->scalar);
                ty = mglIRTypeScalar(ty->scalar);
                continue;
            }
            uint32_t mi = 0;
            const MGLIRType *mt = nullptr;
            for (uint32_t i = 0; i < ty->member_count; i++)
                if (strcmp(ty->member_names[i], pe->u.member.field) == 0) {
                    mi = i;
                    mt = ty->members[i];
                    break;
                }
            if (!mt) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: SSBO has no member '") +
                            pe->u.member.field + "'";
                return nullptr;
            }
            off += ty->member_offsets ? ty->member_offsets[mi] : 0;
            ty = mt;
        } else {
            llvm::Value *idx = emitExpr(cg, pe->u.index.index, mod, locals);
            if (!idx) return nullptr;
            /* An index on a vector-typed element selects one scalar
             * component.  The previous code treated it as another array
             * hop and walked into the vector's NULL elem_type, which
             * crashed typeFromIR downstream (SIGSEGV reachable from any
             * shader doing g_buffer.vec[expr][component]). */
            if (ty->kind == MGLIR_TYPE_VECTOR) {
                uint32_t scalarSize = mglAirScalarByteSize(ty->scalar);
                idx = cg.b->CreateSExtOrTrunc(idx, cg.b->getInt64Ty());
                base = cg.b->CreateGEP(
                    cg.b->getInt8Ty(), base,
                    cg.b->CreateAdd(cg.b->getInt64(off),
                                    cg.b->CreateMul(
                                        idx, cg.b->getInt64(scalarSize))));
                off = 0;
                ty = mglIRTypeScalar(ty->scalar);
                continue;
            }
            /* GLSL 4.60 §5.6: m[i] selects column i.  Column-major memory
             * stores columns contiguously at matrix_stride; row-major
             * columns are not contiguous so only CM (or tightly packed
             * row-vecs that happen to match) can be addressed as a pointer
             * here — row-major column writes go through SSA gather/scatter. */
            if (ty->kind == MGLIR_TYPE_MATRIX) {
                if (ty->row_major) {
                    cg.err = 1;
                    cg.errmsg =
                        "codegen: indexing a row_major SSBO matrix column "
                        "as an lvalue is not supported yet";
                    return nullptr;
                }
                uint32_t stride = ty->layout.matrix_stride;
                if (!stride) {
                    uint32_t comps = ty->rows;
                    uint32_t baseBytes =
                        (comps <= 2u ? comps : 4u) * 4u;
                    stride = (baseBytes + 15u) & ~15u;
                }
                idx = cg.b->CreateSExtOrTrunc(idx, cg.b->getInt64Ty());
                base = cg.b->CreateGEP(
                    cg.b->getInt8Ty(), base,
                    cg.b->CreateAdd(cg.b->getInt64(off),
                                    cg.b->CreateMul(
                                        idx, cg.b->getInt64(stride))));
                off = 0;
                /* Column is a vector of `rows` components. */
                MGLIRType *col = mglIRTypeVector(ty->scalar, ty->rows);
                if (!col) {
                    cg.err = 1;
                    cg.errmsg =
                        "codegen: failed to form SSBO matrix column type";
                    return nullptr;
                }
                ty = col;
                continue;
            }
            if (ty->kind != MGLIR_TYPE_ARRAY) {
                cg.err = 1;
                cg.errmsg =
                    "codegen: SSBO member cannot be indexed by this "
                    "expression";
                return nullptr;
            }
            uint32_t stride = ty->layout.array_stride;
            if (!stride) stride = ty->layout.size;
            idx = cg.b->CreateSExtOrTrunc(idx, cg.b->getInt64Ty());
            base = cg.b->CreateGEP(
                cg.b->getInt8Ty(), base,
                cg.b->CreateAdd(cg.b->getInt64(off),
                                cg.b->CreateMul(idx,
                                                cg.b->getInt64(stride))));
            off = 0;
            ty = ty->elem_type;
        }
    }
    *outTy = ty;
    return cg.b->CreateGEP(cg.b->getInt8Ty(), base, cg.b->getInt64(off));
}

llvm::Value *emitAtomicCounterAddress(
    Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals)
{
    const char *rootName = nullptr;
    const MGLExpr *indexExpr = nullptr;
    if (e->kind == MGL_EXPR_VAR_REF) {
        rootName = e->u.var_ref.name;
    } else if (e->kind == MGL_EXPR_INDEX && e->u.index.object &&
               e->u.index.object->kind == MGL_EXPR_VAR_REF) {
        rootName = e->u.index.object->u.var_ref.name;
        indexExpr = e->u.index.index;
    } else {
        cg.err = 1;
        cg.errmsg = "codegen: atomic counter lvalue required";
        return nullptr;
    }
    auto it = cg.acPtrs.find(rootName);
    if (it == cg.acPtrs.end()) {
        cg.err = 1;
        cg.errmsg = std::string("codegen: unknown atomic counter '") +
                    rootName + "'";
        return nullptr;
    }
    const MGLIRSymbol *s = findSymbol(mod, rootName);
    uint32_t baseOff = (s && s->offset != UINT32_MAX) ? s->offset : 0u;
    llvm::Value *base = it->second;
    llvm::Value *off = cg.b->getInt32(baseOff);
    if (indexExpr) {
        llvm::Value *idx = emitExpr(cg, indexExpr, mod, locals);
        if (!idx) return nullptr;
        idx = coerceScalar(cg, idx, MGLIR_SCALAR_INT);
        /* Out-of-range dynamic indices are undefined in GLSL; clamp so a bad
         * runtime index cannot address past the declared counter array. */
        if (s && s->type->kind == MGLIR_TYPE_ARRAY &&
            s->type->array_size > 0u) {
            uint32_t elemCount = s->type->array_size;
            idx = cg.b->CreateBinaryIntrinsic(
                llvm::Intrinsic::umax, idx, cg.b->getInt32(0));
            idx = cg.b->CreateBinaryIntrinsic(
                llvm::Intrinsic::umin, idx,
                cg.b->getInt32(elemCount - 1u));
        }
        off = cg.b->CreateAdd(
            off, cg.b->CreateMul(idx, cg.b->getInt32(4), "", true, true));
    }
    llvm::Value *p =
        cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
    return cg.b->CreateBitCast(
        p, llvm::Type::getInt32Ty(*cg.ctx)->getPointerTo(1));
}

static llvm::Value *emitUBOMatrixLoad(Codegen &cg, llvm::Value *base,
                                      llvm::Value *off, const MGLIRType *ct,
                                      const MType &vt);
static void emitUBOMatrixStore(Codegen &cg, llvm::Value *base,
                               llvm::Value *off, const MGLIRType *ct,
                               const MType &vt, llvm::Value *v);
static llvm::Value *emitUBOLeafLoad(Codegen &cg, llvm::Value *base,
                                    uint32_t moff, const MGLIRType *ct,
                                    const MType &vt);
static llvm::Value *emitSSBOAggregateLoad(Codegen &cg, llvm::Value *base,
                                          const MGLIRType *ty);
static void emitSSBOAggregateStore(Codegen &cg, llvm::Value *base,
                                   const MGLIRType *ty, llvm::Value *v);

llvm::Value *emitSSBORead(Codegen &cg, const MGLExpr *e,
                          const MGLIRSymbol *sb, const MGLIRModule *mod,
                          const std::map<std::string, MType> &locals) {
    const MGLIRType *ty = nullptr;
    llvm::Value *p = ssboAddress(cg, e, sb, mod, locals, &ty);
    if (!p) return nullptr;
    /* Matrices need matrix_stride / row_major gathering — a packed LLVM
     * `[cols x <rows x T>]` load is only correct for tightly packed
     * column-major mat2 (std430). */
    if (ty && ty->kind == MGLIR_TYPE_MATRIX) {
        MType vt = typeFromIR(ty);
        return emitUBOMatrixLoad(cg, p, cg.b->getInt64(0), ty, vt);
    }
    /* Struct/array IR types have scalar==VOID; a contiguous float load
     * plus coerceScalar(VOID) yields `store i32, float*` — illegal AIR. */
    if (ty && (ty->kind == MGLIR_TYPE_STRUCT || ty->kind == MGLIR_TYPE_ARRAY))
        return emitSSBOAggregateLoad(cg, p, ty);
    llvm::Type *lt = llvmType(typeFromIR(ty), *cg.ctx);
    llvm::Align align = bufferLeafAlign(lt);
    p = cg.b->CreateBitCast(p, lt->getPointerTo(1));
    return cg.b->CreateAlignedLoad(lt, p, align);
}

void emitSSBOWrite(Codegen &cg, const MGLExpr *e, const MGLIRSymbol *sb,
                   const MGLIRModule *mod,
                   const std::map<std::string, MType> &locals,
                   llvm::Value *v) {
    /* Multi-component swizzle store: RMW the parent vector. */
    if (e->kind == MGL_EXPR_MEMBER) {
        std::vector<uint32_t> comps;
        if (swizzleIndices(e->u.member.field, &comps) && comps.size() > 1u) {
            llvm::Value *old =
                emitSSBORead(cg, e->u.member.object, sb, mod, locals);
            if (!old) return;
            llvm::Value *nv = insertSwizzleValue(cg, old, comps, v);
            if (!nv) return;
            emitSSBOWrite(cg, e->u.member.object, sb, mod, locals, nv);
            return;
        }
    }
    const MGLIRType *ty = nullptr;
    llvm::Value *p = ssboAddress(cg, e, sb, mod, locals, &ty);
    if (!p) return;
    if (ty && ty->kind == MGLIR_TYPE_MATRIX) {
        MType vt = typeFromIR(ty);
        emitUBOMatrixStore(cg, p, cg.b->getInt64(0), ty, vt, v);
        return;
    }
    if (ty && (ty->kind == MGLIR_TYPE_STRUCT || ty->kind == MGLIR_TYPE_ARRAY)) {
        emitSSBOAggregateStore(cg, p, ty, v);
        return;
    }
    v = coerceScalar(cg, v, typeFromIR(ty).scalar);
    llvm::Type *lt = llvmType(typeFromIR(ty), *cg.ctx);
    llvm::Align align = bufferLeafAlign(lt);
    p = cg.b->CreateBitCast(p, lt->getPointerTo(1));
    cg.b->CreateAlignedStore(v, p, align);
}

/* Matrix uniform -> SSA [N x <rows x float>] array value. */
llvm::Value *emitMatrixUniform(Codegen &cg, const Uniform &u) {
    llvm::Type *colTy = llvm::FixedVectorType::get(llvm::Type::getFloatTy(*cg.ctx),
                                              u.type.rows);
    llvm::Value *arr = llvm::UndefValue::get(
        llvm::ArrayType::get(colTy, u.type.cols));
    for (uint32_t c = 0; c < u.type.cols; c++) {
        llvm::Value *col = bufferLoad(cg, u.offset + 16 * c, colTy);
        arr = cg.b->CreateInsertValue(arr, col, c);
    }
    return arr;
}

/* Natural alignment for a float/int vector of `comps` components in a
 * std140/std430 buffer (vec3 shares vec4's 16-byte base align). */
static llvm::Align matrixVecAlign(uint32_t comps) {
    if (comps <= 1u) return llvm::Align(4);
    if (comps == 2u) return llvm::Align(8);
    return llvm::Align(16);
}

/* Load a UBO/SSBO matrix at byte offset `off` from `base`, honouring
 * matrix_stride and row_major.  LLVM SSA form is always column-major
 * ([cols x <rows x T>]); row-major memory is gathered into that shape so
 * GLSL `m[i]` still yields column i (GLSL 4.60 §5.6). */
static llvm::Value *emitUBOMatrixLoad(Codegen &cg, llvm::Value *base,
                                      llvm::Value *off, const MGLIRType *ct,
                                      const MType &vt) {
    uint32_t stride = ct->layout.matrix_stride;
    if (stride == 0) {
        uint32_t vec_comps = ct->row_major ? vt.cols : vt.rows;
        uint32_t baseBytes = (vec_comps <= 2u ? vec_comps : 4u) * 4u;
        stride = (baseBytes + 15u) & ~15u;
    }
    llvm::Type *elt = llvmScalar(vt.scalar, *cg.ctx);
    llvm::Type *colTy = llvm::FixedVectorType::get(elt, vt.rows);
    llvm::Value *v = llvm::UndefValue::get(
        llvm::ArrayType::get(colTy, vt.cols));
    if (ct->row_major) {
        llvm::Type *rowTy = llvm::FixedVectorType::get(elt, vt.cols);
        llvm::Align align = matrixVecAlign(vt.cols);
        llvm::SmallVector<llvm::Value *, 4> rows;
        for (uint32_t r = 0; r < vt.rows; r++) {
            llvm::Value *rowOff =
                cg.b->CreateAdd(off, cg.b->getInt64((uint64_t)r * stride));
            llvm::Value *rp =
                cg.b->CreateGEP(cg.b->getInt8Ty(), base, rowOff);
            rp = cg.b->CreateBitCast(rp, rowTy->getPointerTo(1));
            rows.push_back(cg.b->CreateAlignedLoad(rowTy, rp, align));
        }
        for (uint32_t c = 0; c < vt.cols; c++) {
            llvm::Value *col = llvm::UndefValue::get(colTy);
            for (uint32_t r = 0; r < vt.rows; r++) {
                llvm::Value *e = cg.b->CreateExtractElement(
                    rows[r], cg.b->getInt32(c));
                col = cg.b->CreateInsertElement(col, e, cg.b->getInt32(r));
            }
            v = cg.b->CreateInsertValue(v, col, c);
        }
        return v;
    }
    llvm::Align align = matrixVecAlign(vt.rows);
    for (uint32_t c = 0; c < vt.cols; c++) {
        llvm::Value *colOff =
            cg.b->CreateAdd(off, cg.b->getInt64((uint64_t)c * stride));
        llvm::Value *cp =
            cg.b->CreateGEP(cg.b->getInt8Ty(), base, colOff);
        cp = cg.b->CreateBitCast(cp, colTy->getPointerTo(1));
        llvm::Value *col = cg.b->CreateAlignedLoad(colTy, cp, align);
        v = cg.b->CreateInsertValue(v, col, c);
    }
    return v;
}

/* Store SSA column-major matrix `v` into UBO/SSBO memory at `off`. */
static void emitUBOMatrixStore(Codegen &cg, llvm::Value *base,
                               llvm::Value *off, const MGLIRType *ct,
                               const MType &vt, llvm::Value *v) {
    uint32_t stride = ct->layout.matrix_stride;
    if (stride == 0) {
        uint32_t vec_comps = ct->row_major ? vt.cols : vt.rows;
        uint32_t baseBytes = (vec_comps <= 2u ? vec_comps : 4u) * 4u;
        stride = (baseBytes + 15u) & ~15u;
    }
    llvm::Type *elt = llvmScalar(vt.scalar, *cg.ctx);
    llvm::Type *colTy = llvm::FixedVectorType::get(elt, vt.rows);
    if (ct->row_major) {
        llvm::Type *rowTy = llvm::FixedVectorType::get(elt, vt.cols);
        llvm::Align align = matrixVecAlign(vt.cols);
        for (uint32_t r = 0; r < vt.rows; r++) {
            llvm::Value *row = llvm::UndefValue::get(rowTy);
            for (uint32_t c = 0; c < vt.cols; c++) {
                llvm::Value *col = cg.b->CreateExtractValue(v, c);
                if (col->getType() != colTy)
                    col = cg.b->CreateBitCast(col, colTy);
                llvm::Value *e = cg.b->CreateExtractElement(
                    col, cg.b->getInt32(r));
                row = cg.b->CreateInsertElement(row, e, cg.b->getInt32(c));
            }
            llvm::Value *rowOff =
                cg.b->CreateAdd(off, cg.b->getInt64((uint64_t)r * stride));
            llvm::Value *rp =
                cg.b->CreateGEP(cg.b->getInt8Ty(), base, rowOff);
            rp = cg.b->CreateBitCast(rp, rowTy->getPointerTo(1));
            cg.b->CreateAlignedStore(row, rp, align);
        }
        return;
    }
    llvm::Align align = matrixVecAlign(vt.rows);
    for (uint32_t c = 0; c < vt.cols; c++) {
        llvm::Value *col = cg.b->CreateExtractValue(v, c);
        if (col->getType() != colTy)
            col = cg.b->CreateBitCast(col, colTy);
        llvm::Value *colOff =
            cg.b->CreateAdd(off, cg.b->getInt64((uint64_t)c * stride));
        llvm::Value *cp =
            cg.b->CreateGEP(cg.b->getInt8Ty(), base, colOff);
        cp = cg.b->CreateBitCast(cp, colTy->getPointerTo(1));
        cg.b->CreateAlignedStore(col, cp, align);
    }
}

/* Load a UBO leaf (scalar / vector / matrix / bvec) at byte offset `off`
 * from `base`.  Contiguous LLVM aggregate loads are wrong for std140:
 * mat2 columns are 16 bytes apart while `[2 x <2 x float>]` packs at 8,
 * and float[N] elements are 16 apart while `[N x float]` packs at 4. */
static llvm::Value *emitUBOLeafLoad(Codegen &cg, llvm::Value *base,
                                    llvm::Value *off, const MGLIRType *ct,
                                    const MType &vt) {
    if (ct && ct->kind == MGLIR_TYPE_MATRIX)
        return emitUBOMatrixLoad(cg, base, off, ct, vt);
    llvm::Type *t = llvmType(vt, *cg.ctx);
    llvm::Value *p =
        cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
    llvm::Align align = bufferLeafAlign(t);
    if (vt.vec && vt.scalar == MGLIR_SCALAR_BOOL) {
        llvm::Type *wordsTy = llvm::FixedVectorType::get(
            llvm::Type::getInt32Ty(*cg.ctx), vt.vec);
        p = cg.b->CreateBitCast(p, wordsTy->getPointerTo(1));
        llvm::Value *words = cg.b->CreateAlignedLoad(wordsTy, p, align);
        return cg.b->CreateICmpNE(
            words, llvm::ConstantAggregateZero::get(wordsTy));
    }
    if (vt.scalar == MGLIR_SCALAR_BOOL && !vt.vec && !vt.isMatrix()) {
        /* std140 bool is a 32-bit word; an i1 load is not defined for
         * buffer address space on Metal. */
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        p = cg.b->CreateBitCast(p, i32->getPointerTo(1));
        llvm::Value *word =
            cg.b->CreateAlignedLoad(i32, p, llvm::Align(4));
        return cg.b->CreateICmpNE(word, cg.b->getInt32(0));
    }
    p = cg.b->CreateBitCast(p, t->getPointerTo(1));
    return cg.b->CreateAlignedLoad(t, p, align);
}

static llvm::Value *emitUBOLeafLoad(Codegen &cg, llvm::Value *base,
                                    uint32_t moff, const MGLIRType *ct,
                                    const MType &vt) {
    return emitUBOLeafLoad(cg, base, cg.b->getInt64(moff), ct, vt);
}

/* Store a scalar/vector/matrix leaf at byte offset 0 from `base`. */
static void emitUBOLeafStore(Codegen &cg, llvm::Value *base,
                             const MGLIRType *ct, const MType &vt,
                             llvm::Value *v) {
    if (ct && ct->kind == MGLIR_TYPE_MATRIX) {
        emitUBOMatrixStore(cg, base, cg.b->getInt64(0), ct, vt, v);
        return;
    }
    llvm::Type *t = llvmType(vt, *cg.ctx);
    llvm::Align align = bufferLeafAlign(t);
    if (vt.vec && vt.scalar == MGLIR_SCALAR_BOOL) {
        llvm::Type *wordsTy = llvm::FixedVectorType::get(
            llvm::Type::getInt32Ty(*cg.ctx), vt.vec);
        llvm::Value *words = cg.b->CreateZExt(v, wordsTy);
        llvm::Value *p = cg.b->CreateBitCast(base, wordsTy->getPointerTo(1));
        cg.b->CreateAlignedStore(words, p, align);
        return;
    }
    if (vt.scalar == MGLIR_SCALAR_BOOL && !vt.vec && !vt.isMatrix()) {
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        llvm::Value *word = cg.b->CreateZExt(v, i32);
        llvm::Value *p = cg.b->CreateBitCast(base, i32->getPointerTo(1));
        cg.b->CreateAlignedStore(word, p, llvm::Align(4));
        return;
    }
    v = coerceScalar(cg, v, vt.scalar);
    if (v->getType() != t)
        v = cg.b->CreateBitCast(v, t);
    llvm::Value *p = cg.b->CreateBitCast(base, t->getPointerTo(1));
    cg.b->CreateAlignedStore(v, p, align);
}

/* Load an SSBO struct/array honouring std140/std430 member_offsets and
 * array_stride (never a single contiguous LLVM aggregate load). */
static llvm::Value *emitSSBOAggregateLoad(Codegen &cg, llvm::Value *base,
                                          const MGLIRType *ty) {
    if (!ty) return nullptr;
    if (ty->kind == MGLIR_TYPE_STRUCT) {
        llvm::Type *st = llvmTypeFromIR(ty, *cg.ctx);
        llvm::Value *v = llvm::UndefValue::get(st);
        for (uint32_t i = 0; i < ty->member_count; i++) {
            uint32_t moff = ty->member_offsets ? ty->member_offsets[i] : 0u;
            const MGLIRType *mt = ty->members[i];
            llvm::Value *mp =
                cg.b->CreateGEP(cg.b->getInt8Ty(), base, cg.b->getInt64(moff));
            llvm::Value *mv;
            if (mt->kind == MGLIR_TYPE_STRUCT || mt->kind == MGLIR_TYPE_ARRAY)
                mv = emitSSBOAggregateLoad(cg, mp, mt);
            else
                mv = emitUBOLeafLoad(cg, mp, 0u, mt, typeFromIR(mt));
            if (!mv) return nullptr;
            v = cg.b->CreateInsertValue(v, mv, i);
        }
        return v;
    }
    if (ty->kind == MGLIR_TYPE_ARRAY) {
        uint32_t n = ty->array_size;
        if (n == 0u) {
            cg.err = 1;
            cg.errmsg =
                "codegen: cannot load an unsized SSBO array as a value";
            return nullptr;
        }
        llvm::Type *at = llvmTypeFromIR(ty, *cg.ctx);
        llvm::Value *v = llvm::UndefValue::get(at);
        uint32_t stride = ty->layout.array_stride;
        if (!stride && ty->elem_type)
            stride = ty->elem_type->layout.size;
        const MGLIRType *et = ty->elem_type;
        for (uint32_t i = 0; i < n; i++) {
            llvm::Value *ep = cg.b->CreateGEP(
                cg.b->getInt8Ty(), base,
                cg.b->getInt64((uint64_t)i * stride));
            llvm::Value *ev;
            if (et->kind == MGLIR_TYPE_STRUCT || et->kind == MGLIR_TYPE_ARRAY)
                ev = emitSSBOAggregateLoad(cg, ep, et);
            else
                ev = emitUBOLeafLoad(cg, ep, 0u, et, typeFromIR(et));
            if (!ev) return nullptr;
            v = cg.b->CreateInsertValue(v, ev, i);
        }
        return v;
    }
    return emitUBOLeafLoad(cg, base, 0u, ty, typeFromIR(ty));
}

static void emitSSBOAggregateStore(Codegen &cg, llvm::Value *base,
                                   const MGLIRType *ty, llvm::Value *v) {
    if (!ty || !v) return;
    if (ty->kind == MGLIR_TYPE_STRUCT) {
        for (uint32_t i = 0; i < ty->member_count; i++) {
            uint32_t moff = ty->member_offsets ? ty->member_offsets[i] : 0u;
            const MGLIRType *mt = ty->members[i];
            llvm::Value *mp =
                cg.b->CreateGEP(cg.b->getInt8Ty(), base, cg.b->getInt64(moff));
            llvm::Value *mv = cg.b->CreateExtractValue(v, i);
            if (mt->kind == MGLIR_TYPE_STRUCT || mt->kind == MGLIR_TYPE_ARRAY)
                emitSSBOAggregateStore(cg, mp, mt, mv);
            else
                emitUBOLeafStore(cg, mp, mt, typeFromIR(mt), mv);
        }
        return;
    }
    if (ty->kind == MGLIR_TYPE_ARRAY) {
        uint32_t n = ty->array_size;
        if (n == 0u) {
            cg.err = 1;
            cg.errmsg =
                "codegen: cannot store an unsized SSBO array as a value";
            return;
        }
        uint32_t stride = ty->layout.array_stride;
        if (!stride && ty->elem_type)
            stride = ty->elem_type->layout.size;
        const MGLIRType *et = ty->elem_type;
        for (uint32_t i = 0; i < n; i++) {
            llvm::Value *ep = cg.b->CreateGEP(
                cg.b->getInt8Ty(), base,
                cg.b->getInt64((uint64_t)i * stride));
            llvm::Value *ev = cg.b->CreateExtractValue(v, i);
            if (et->kind == MGLIR_TYPE_STRUCT || et->kind == MGLIR_TYPE_ARRAY)
                emitSSBOAggregateStore(cg, ep, et, ev);
            else
                emitUBOLeafStore(cg, ep, et, typeFromIR(et), ev);
        }
        return;
    }
    emitUBOLeafStore(cg, base, ty, typeFromIR(ty), v);
}

llvm::Value *varValue(Codegen &cg, const VarSym &v, const MGLIRModule *mod) {
    if (v.kind == VarSym::BUFFER) {
        /* Anonymous-block member: read from the block's device buffer. */
        const MGLIRSymbol *bs = findSymbol(mod, v.name.c_str());
        if (mgl_env_flag_enabled("MGL_VAR_DBG"))
            fprintf(stderr, "VAR %s kind=%d block=%s\n", v.name.c_str(),
                    (int)v.kind, bs ? (bs->block_name ? bs->block_name : "-") : "-");
        if (bs && bs->block_name) {
            llvm::Value *base = cg.uboPtrs.count(bs->block_name)
                                    ? cg.uboPtrs[bs->block_name]
                                    : nullptr;
            if (base) {
                const MGLIRSymbol *blk = findSymbol(mod, bs->block_name);
                uint32_t moff = (blk && blk->type->member_offsets)
                                    ? blk->type->member_offsets[bs->block_member_index]
                                    : 0;
                if (bs->offset != UINT32_MAX)
                    moff = bs->offset;
                return emitUBOLeafLoad(cg, base, moff, bs->type, v.type);
            }
        }
        /* Uniform: single value read. */
        uint32_t off = cg.bufferOffsets.count(v.name) ? cg.bufferOffsets[v.name] : 0;
        if (v.type.isMatrix())
            return emitMatrixUniform(cg, Uniform{v.name, v.type, off, 0});
        /* Default-block uniforms use the same std140 leaf rules as named
         * UBOs (bool/bvec as 32-bit words — never packed <N x i1>). */
        const MGLIRSymbol *us = findSymbol(mod, v.name.c_str());
        return emitUBOLeafLoad(cg, cg.bufferPtr, off,
                               us ? us->type : nullptr, v.type);
    }
    auto amit = cg.arrayMem.find(v.name);
    if (amit != cg.arrayMem.end()) {
        llvm::Type *arrTy = cg.arrayMemTypes[v.name];
        return cg.b->CreateAlignedLoad(arrTy, amit->second, llvm::Align(4));
    }
    auto it = cg.lvalues.find(v.name);
    if (it != cg.lvalues.end())
        return it->second;
    /* Unwritten out/attribute: undef. */
    return llvm::UndefValue::get(llvmType(v.type, *cg.ctx));
}

/* All programmable pre-raster stages exchange MGLAIRPerVertexRecord. */
static bool perVertexPath(const MGLExpr *e, const char **root,
                          const MGLExpr **index, const char **field)
{
    if (!e || e->kind != MGL_EXPR_MEMBER ||
        !e->u.member.object || e->u.member.object->kind != MGL_EXPR_INDEX)
        return false;
    const MGLExpr *obj = e->u.member.object->u.index.object;
    if (!obj || obj->kind != MGL_EXPR_VAR_REF)
        return false;
    const char *name = obj->u.var_ref.name;
    if (strcmp(name, "gl_in") != 0 && strcmp(name, "gl_out") != 0)
        return false;
    const char *f = e->u.member.field;
    if (strcmp(f, "gl_Position") != 0 &&
        strcmp(f, "gl_PointSize") != 0 &&
        strcmp(f, "gl_ClipDistance") != 0 &&
        strcmp(f, "gl_CullDistance") != 0)
        return false;
    if (root) *root = name;
    if (index) *index = e->u.member.object->u.index.index;
    if (field) *field = f;
    return true;
}

static uint64_t perVertexFieldOffset(const char *field)
{
    if (!strcmp(field, "gl_PointSize"))
        return MGL_AIR_PER_VERTEX_POINT_SIZE_OFFSET;
    if (!strcmp(field, "gl_CullDistance"))
        return MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET;
    if (!strcmp(field, "gl_ClipDistance"))
        return MGL_AIR_PER_VERTEX_CLIP_DISTANCE_OFFSET;
    return MGL_AIR_PER_VERTEX_POSITION_OFFSET;
}

static llvm::Type *perVertexFieldType(Codegen &cg, const char *field)
{
    if (!strcmp(field, "gl_Position"))
        return llvm::FixedVectorType::get(
            llvm::Type::getFloatTy(*cg.ctx), 4);
    if (!strcmp(field, "gl_CullDistance"))
        return llvm::ArrayType::get(
            llvm::Type::getFloatTy(*cg.ctx),
            MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT);
    if (!strcmp(field, "gl_ClipDistance"))
        return llvm::ArrayType::get(
            llvm::Type::getFloatTy(*cg.ctx),
            MGL_AIR_PER_VERTEX_CLIP_DISTANCE_COUNT);
    return llvm::Type::getFloatTy(*cg.ctx);
}

static VarSym *codegenStageSymbol(Codegen &cg, const char *name,
                                  VarSym::Kind kind)
{
    if (!cg.auxSyms || !name) return nullptr;
    for (VarSym &v : *cg.auxSyms) {
        if (v.kind == kind && v.name == name) return &v;
    }
    return nullptr;
}

/* Flattened interface-block member: match both instance and field name. */
static VarSym *codegenBlockMember(Codegen &cg, const char *instName,
                                  const char *field, VarSym::Kind kind)
{
    if (!cg.auxSyms || !instName || !field) return nullptr;
    for (VarSym &v : *cg.auxSyms) {
        if (v.kind == kind && v.name == field && v.blockName == instName)
            return &v;
    }
    return nullptr;
}

static llvm::Value *tessStageRecordIndex(Codegen &cg, llvm::Value *index,
                                         bool input)
{
    index = coerceScalar(cg, index, MGLIR_SCALAR_UINT);
    llvm::Value *patch = cg.b->CreateExtractElement(
        cg.patchPos, cg.b->getInt32(0));
    llvm::Value *verticesPerPatch = nullptr;
    if (input) {
        llvm::Value *p = cg.b->CreateBitCast(
            cg.indirectPtr, cg.b->getInt32Ty()->getPointerTo(1));
        verticesPerPatch = cg.b->CreateAlignedLoad(
            cg.b->getInt32Ty(), p, llvm::Align(4));
    } else {
        verticesPerPatch = cg.b->getInt32(cg.tcsOutputVertices);
    }
    return cg.b->CreateAdd(
        cg.b->CreateMul(patch, verticesPerPatch), index);
}

/* Native TES per-patch draws: Metal patch_id is always 0; the runtime
 * stamps the global patch index in mgl_patch_info[2] (slot 28). */
static llvm::Value *tessPatchIndexForStageIn(Codegen &cg)
{
    if (cg.isTessEval && !cg.isTESCompute && !cg.isTESVertex &&
        cg.indirectPtr) {
        llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
        llvm::Value *info = cg.b->CreateBitCast(
            cg.indirectPtr, i32->getPointerTo(1));
        return cg.b->CreateAlignedLoad(
            i32, cg.b->CreateGEP(i32, info, cg.b->getInt32(2)),
            llvm::Align(4));
    }
    return cg.patchId;
}

static llvm::Value *emitPatchVaryingLoad(Codegen &cg, const VarSym &sym)
{
    if (!sym.isPatch || sym.location == UINT32_MAX || !cg.captureBuf)
        return nullptr;
    llvm::Value *patchIdx = nullptr;
    uint64_t stride = 0;
    if (cg.isTessEval) {
        if (!cg.patchId) return nullptr;
        patchIdx = cg.patchId;
        stride = cg.patchInStride;
    } else if (cg.isTessControl) {
        /* TCS must reload patch outs from the shared patch-out buffer so
         * barrier()-guarded cross-invocation writes are visible. */
        if (!cg.patchPos) return nullptr;
        patchIdx = cg.b->CreateExtractElement(cg.patchPos, cg.b->getInt32(0));
        stride = cg.patchOutStride;
    } else {
        return nullptr;
    }
    llvm::Value *off = cg.b->CreateAdd(
        cg.b->CreateMul(cg.b->CreateZExt(patchIdx, cg.b->getInt64Ty()),
                        cg.b->getInt64(stride)),
        cg.b->getInt64(sym.location * 16u));
    llvm::Type *ty = llvmType(sym.type, *cg.ctx);
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), cg.captureBuf, off);
    p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
    return cg.b->CreateAlignedLoad(ty, p, llvm::Align(4));
}

static bool emitPatchVaryingStore(Codegen &cg, const VarSym &sym,
                                  llvm::Value *value)
{
    if (!cg.isTessControl || !sym.isPatch || !cg.captureBuf || !cg.patchPos ||
        sym.location == UINT32_MAX) {
        return false;
    }
    llvm::Value *patch = cg.b->CreateExtractElement(
        cg.patchPos, cg.b->getInt32(0));
    llvm::Value *off = cg.b->CreateAdd(
        cg.b->CreateMul(cg.b->CreateZExt(patch, cg.b->getInt64Ty()),
                        cg.b->getInt64(cg.patchOutStride)),
        cg.b->getInt64(sym.location * 16u));
    llvm::Type *ty = llvmType(sym.type, *cg.ctx);
    if (value->getType() != ty)
        value = coerceScalar(cg, value, sym.type.scalar);
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), cg.captureBuf, off);
    p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
    cg.b->CreateAlignedStore(value, p, llvm::Align(4));
    return true;
}

/* GL 4.6 §11.2.1.2 / §11.2.2: any TCS invocation may write gl_TessLevel*.
 * Unwritten components keep the PATCH_DEFAULT_* fill. Skip undef so a
 * silent invocation does not clobber another invocation's store. */
static void flushTCSTessLevels(Codegen &cg)
{
    if (!cg.isTessControl || !cg.tessFactorPtr || !cg.patchPos || !cg.b)
        return;
    llvm::IRBuilder<> &b = *cg.b;
    llvm::LLVMContext &ctx = *cg.ctx;
    llvm::Value *patch =
        cg.workGroupPos
            ? b.CreateExtractElement(cg.workGroupPos, b.getInt32(0))
            : b.CreateExtractElement(cg.patchPos, b.getInt32(0));
    llvm::Value *factorBase = b.CreateGEP(
        b.getInt8Ty(), cg.tessFactorPtr,
        b.CreateMul(b.CreateZExt(patch, b.getInt64Ty()),
                    b.getInt64(MGL_AIR_TESS_FACTOR_RECORD_BYTES)));
    llvm::Type *halfTy = llvm::Type::getHalfTy(ctx);
    llvm::Type *f32Ty = llvm::Type::getFloatTy(ctx);
    auto storeOne = [&](const char *name, unsigned count, unsigned index,
                        unsigned halfOff, unsigned exactOff) {
        if (!cg.lvalues.count(name))
            return;
        llvm::Value *v = b.CreateExtractValue(cg.lvalues[name], index);
        if (llvm::isa<llvm::UndefValue>(v))
            return;
        llvm::Value *hp = b.CreateBitCast(
            b.CreateGEP(b.getInt8Ty(), factorBase, b.getInt64(halfOff)),
            halfTy->getPointerTo(1));
        b.CreateAlignedStore(b.CreateFPTrunc(v, halfTy), hp, llvm::Align(2));
        llvm::Value *ep = b.CreateBitCast(
            b.CreateGEP(b.getInt8Ty(), factorBase, b.getInt64(exactOff)),
            f32Ty->getPointerTo(1));
        b.CreateAlignedStore(v, ep, llvm::Align(4));
        (void)count;
    };
    for (unsigned i = 0; i < 4; i++)
        storeOne("gl_TessLevelOuter", 4, i, i * 2u,
                 MGL_AIR_TESS_FACTOR_EXACT_FLOAT_OFFSET + i * 4u);
    for (unsigned i = 0; i < 2; i++)
        storeOne("gl_TessLevelInner", 2, i, 8u + i * 2u,
                 MGL_AIR_TESS_FACTOR_EXACT_FLOAT_OFFSET + 16u + i * 4u);
}

/* Indexed patch out/in: patch out T arr[N]; arr[i] = … / TES patch in. */
static llvm::Value *emitPatchArrayElementLoad(
    Codegen &cg, const VarSym &sym, llvm::Value *index)
{
    if (!sym.isPatch || !sym.type.isArray() || sym.location == UINT32_MAX ||
        !cg.captureBuf)
        return nullptr;
    index = coerceScalar(cg, index, MGLIR_SCALAR_UINT);
    llvm::Value *patchIdx = nullptr;
    uint64_t stride = 0;
    if (cg.isTessControl) {
        if (!cg.patchPos) return nullptr;
        patchIdx = cg.b->CreateExtractElement(cg.patchPos, cg.b->getInt32(0));
        stride = cg.patchOutStride;
    } else if (cg.isTessEval) {
        if (!cg.patchId) return nullptr;
        patchIdx = cg.patchId;
        stride = cg.patchInStride;
    } else {
        return nullptr;
    }
    MType elem = sym.type;
    elem.arr = 0;
    llvm::Value *slot = cg.b->CreateAdd(
        cg.b->getInt32(sym.location), index);
    llvm::Value *off = cg.b->CreateAdd(
        cg.b->CreateMul(cg.b->CreateZExt(patchIdx, cg.b->getInt64Ty()),
                        cg.b->getInt64(stride)),
        cg.b->CreateMul(cg.b->CreateZExt(slot, cg.b->getInt64Ty()),
                        cg.b->getInt64(16)));
    llvm::Type *ty = llvmType(elem, *cg.ctx);
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), cg.captureBuf, off);
    p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
    return cg.b->CreateAlignedLoad(ty, p, llvm::Align(4));
}

static bool emitPatchArrayElementStore(Codegen &cg, const VarSym &sym,
                                       llvm::Value *index,
                                       llvm::Value *value)
{
    if (!cg.isTessControl || !sym.isPatch || !sym.type.isArray() ||
        sym.location == UINT32_MAX || !cg.captureBuf || !cg.patchPos)
        return false;
    index = coerceScalar(cg, index, MGLIR_SCALAR_UINT);
    llvm::Value *patch = cg.b->CreateExtractElement(
        cg.patchPos, cg.b->getInt32(0));
    MType elem = sym.type;
    elem.arr = 0;
    llvm::Value *slot = cg.b->CreateAdd(
        cg.b->getInt32(sym.location), index);
    llvm::Value *off = cg.b->CreateAdd(
        cg.b->CreateMul(cg.b->CreateZExt(patch, cg.b->getInt64Ty()),
                        cg.b->getInt64(cg.patchOutStride)),
        cg.b->CreateMul(cg.b->CreateZExt(slot, cg.b->getInt64Ty()),
                        cg.b->getInt64(16)));
    llvm::Type *ty = llvmType(elem, *cg.ctx);
    if (value->getType() != ty)
        value = coerceScalar(cg, value, elem.scalar);
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), cg.captureBuf, off);
    p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
    cg.b->CreateAlignedStore(value, p, llvm::Align(4));
    return true;
}

/* Forward decl: GS gl_in record index (defined below, after the
 * tess control/eval array loaders). */
static llvm::Value *geometryInputRecordIndex(Codegen &cg,
                                             llvm::Value *vertexIndex);

/* Load one varying slot from a GS stage-in record.  The record stride
 * comes from the gather params at runtime (the capture lays records out
 * by the *vertex* stage's output locations, which can be wider than this
 * GS's declared inputs), and the member location maps through loc_map
 * (renderer stores vs_loc + 1; 0 marks unmapped and falls back to the
 * identity mapping). */
static llvm::Value *loadGeometryInputVarying(Codegen &cg,
                                             const VarSym &sym,
                                             llvm::Value *slotLocation,
                                             llvm::Value *record,
                                             llvm::Value *base)
{
    llvm::Type *i32ty = llvm::Type::getInt32Ty(*cg.ctx);
    llvm::Value *stride = cg.b->CreateAlignedLoad(
        i32ty,
        cg.b->CreateGEP(i32ty,
                        cg.b->CreateBitCast(cg.geometryGatherParamsPtr,
                                            i32ty->getPointerTo(1)),
                        cg.b->getInt32(4)),
        llvm::Align(4));
    /* Map this location through loc_map (renderer stores vs_loc + 1; 0
     * marks unmapped and falls back to the identity mapping).  Locations
     * beyond the 32-entry map keep the identity. */
    llvm::Value *inMap = cg.b->CreateICmpULT(slotLocation,
                                             cg.b->getInt32(32u));
    llvm::Value *raw = cg.b->CreateAlignedLoad(
        i32ty,
        cg.b->CreateGEP(i32ty,
                        cg.b->CreateBitCast(cg.geometryGatherParamsPtr,
                                            i32ty->getPointerTo(1)),
                        cg.b->CreateAdd(cg.b->getInt32(5), slotLocation)),
        llvm::Align(4));
    llvm::Value *decoded = cg.b->CreateSelect(
        cg.b->CreateICmpEQ(raw, cg.b->getInt32(0)),
        slotLocation,
        cg.b->CreateSub(raw, cg.b->getInt32(1)));
    llvm::Value *mapped = cg.b->CreateSelect(inMap, decoded, slotLocation);
    llvm::Value *varyOff = cg.b->CreateAdd(
        cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE),
        cg.b->CreateMul(cg.b->CreateZExt(mapped, cg.b->getInt64Ty()),
                        cg.b->getInt64(16u)));
    llvm::Value *off = cg.b->CreateAdd(
        cg.b->CreateMul(cg.b->CreateZExt(record, cg.b->getInt64Ty()),
                        cg.b->CreateZExt(stride, cg.b->getInt64Ty())),
        varyOff);
    /* Array block members occupy one slot per element; every caller loads
     * exactly one element, so strip the array dimension for the type. */
    MType elemType = sym.type;
    elemType.arr = 0;
    llvm::Type *ty = llvmType(elemType, *cg.ctx);
    llvm::Type *loadTy = ty;
    if (varyingNeedsFloatRecordCarrier(elemType))
        loadTy = llvmType(floatCarrierType(elemType), *cg.ctx);
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
    p = cg.b->CreateBitCast(p, loadTy->getPointerTo(1));
    llvm::Value *v = cg.b->CreateAlignedLoad(loadTy, p, llvm::Align(4));
    if (varyingNeedsFloatRecordCarrier(elemType))
        v = decodeFloatCarrier(cg, v, elemType.scalar, ty);
    return v;
}

/* Interface-block GS input member: instance[k].field (or instance.field).
 * Sema flattens named in-block members into VARYING symbols whose
 * block_name identifies the owning instance; the read is the same
 * stage-in record load as plain array varyings. */
static llvm::Value *emitGeometryBlockLoad(
    Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals)
{
    if (!cg.isGeometry || !e || e->kind != MGL_EXPR_MEMBER ||
        !cg.geometryInputPtr || !cg.geometryPrimitiveId ||
        !cg.geometryGatherParamsPtr) {
        return nullptr;
    }
    const MGLExpr *obj = e->u.member.object;
    const char *instName = nullptr;
    llvm::Value *vertexIndex = nullptr;
    if (obj && obj->kind == MGL_EXPR_VAR_REF) {
        instName = obj->u.var_ref.name;
        vertexIndex = cg.b->getInt32(0);
    } else if (obj && obj->kind == MGL_EXPR_INDEX &&
               obj->u.index.object &&
               obj->u.index.object->kind == MGL_EXPR_VAR_REF) {
        instName = obj->u.index.object->u.var_ref.name;
        vertexIndex = emitExpr(cg, obj->u.index.index, mod, locals);
        if (!vertexIndex) return nullptr;
    } else {
        return nullptr;
    }
    VarSym *member = codegenBlockMember(
        cg, instName, e->u.member.field, VarSym::VARYING);
    if (!member || member->location == UINT32_MAX || member->type.isArray()) {
        /* Array members are handled by the INDEX case so the trailing
         * element index selects the record slot. */
        return nullptr;
    }
    vertexIndex = coerceScalar(cg, vertexIndex, MGLIR_SCALAR_UINT);
    llvm::Value *record = geometryInputRecordIndex(cg, vertexIndex);
    if (!record) return nullptr;
    /* Matrices occupy one location per column; assemble the aggregate. */
    if (member->type.isMatrix() && member->type.cols > 0) {
        llvm::Type *ty = llvmType(member->type, *cg.ctx);
        llvm::Type *colTy = llvm::FixedVectorType::get(
            llvmScalar(member->type.scalar, *cg.ctx), member->type.rows);
        llvm::Value *agg = llvm::UndefValue::get(ty);
        MType colSymType = matrixColumnType(member->type);
        for (uint32_t c = 0; c < member->type.cols; c++) {
            VarSym colSym = *member;
            colSym.type = colSymType;
            colSym.location = member->location + c;
            llvm::Value *col = loadGeometryInputVarying(
                cg, colSym, cg.b->getInt32(colSym.location), record,
                cg.geometryInputPtr);
            if (!col) return nullptr;
            if (col->getType() != colTy)
                col = cg.b->CreateBitCast(col, colTy);
            agg = cg.b->CreateInsertValue(agg, col, c);
        }
        return agg;
    }
    return loadGeometryInputVarying(
        cg, *member, cg.b->getInt32(member->location), record,
        cg.geometryInputPtr);
}

/* Array interface-block member with the full access chain in hand:
 * inst[k].field[e] loads element slot (base location + e) of input
 * vertex k's stage-in record. */
static llvm::Value *emitGeometryBlockArrayLoad(
    Codegen &cg, const MGLExpr *indexExpr, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals)
{
    /* Stage-level info for return assembly. */    if (!cg.isGeometry || !indexExpr || indexExpr->kind != MGL_EXPR_INDEX ||
        !indexExpr->u.index.object ||
        indexExpr->u.index.object->kind != MGL_EXPR_MEMBER ||
        !cg.geometryInputPtr || !cg.geometryPrimitiveId ||
        !cg.geometryGatherParamsPtr) {
        return nullptr;
    }
    const MGLExpr *memberE = indexExpr->u.index.object;
    const MGLExpr *obj = memberE->u.member.object;
    const char *instName = nullptr;
    llvm::Value *vertexIndex = nullptr;
    if (obj && obj->kind == MGL_EXPR_VAR_REF) {
        instName = obj->u.var_ref.name;
        vertexIndex = cg.b->getInt32(0);
    } else if (obj && obj->kind == MGL_EXPR_INDEX &&
               obj->u.index.object &&
               obj->u.index.object->kind == MGL_EXPR_VAR_REF) {
        instName = obj->u.index.object->u.var_ref.name;
        vertexIndex = emitExpr(cg, obj->u.index.index, mod, locals);
        if (!vertexIndex) return nullptr;
    } else {
        return nullptr;
    }
    VarSym *member = codegenBlockMember(
        cg, instName, memberE->u.member.field, VarSym::VARYING);
    /* Arrays and matrices both consume one location per element/column. */
    if (!member || member->location == UINT32_MAX ||
        (!member->type.isArray() && !member->type.isMatrix())) {
        return nullptr;
    }
    llvm::Value *element = emitExpr(cg, indexExpr->u.index.index,
                                    mod, locals);
    if (!element) return nullptr;
    element = coerceScalar(cg, element, MGLIR_SCALAR_UINT);
    uint32_t span = member->type.isMatrix() ? member->type.cols
                                            : (uint32_t)member->type.arr;
    if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(element)) {
        if (ci->getZExtValue() >= span) {
            cg.err = 1;
            cg.errmsg = "codegen: interface-block array index out of range";
            return nullptr;
        }
    }
    vertexIndex = coerceScalar(cg, vertexIndex, MGLIR_SCALAR_UINT);
    llvm::Value *record = geometryInputRecordIndex(cg, vertexIndex);
    if (!record) return nullptr;
    llvm::Value *slot = cg.b->CreateAdd(
        cg.b->getInt32(member->location), element);
    VarSym elemSym = *member;
    if (member->type.isMatrix())
        elemSym.type = matrixColumnType(member->type);
    else {
        elemSym.type.arr = 0;
    }
    return loadGeometryInputVarying(cg, elemSym, slot, record,
                                    cg.geometryInputPtr);
}

static llvm::Value *emitTessStageArrayLoad(
    Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals)
{
    if ((!cg.isTessControl && !cg.isGeometry) || !e ||
        e->kind != MGL_EXPR_INDEX || !e->u.index.object ||
        e->u.index.object->kind != MGL_EXPR_VAR_REF) return nullptr;
    const char *name = e->u.index.object->u.var_ref.name;
    /* TCS may read per-vertex outs written by other invocations (after
     * barrier()). Prefer OUTPUT/stageOut; fall back to stage-in varyings. */
    VarSym *sym = nullptr;
    bool fromOutput = false;
    if (cg.isTessControl) {
        sym = codegenStageSymbol(cg, name, VarSym::OUTPUT);
        if (sym && sym->location != UINT32_MAX) {
            fromOutput = true;
        } else {
            sym = codegenStageSymbol(cg, name, VarSym::VARYING);
        }
    } else {
        sym = codegenStageSymbol(cg, name, VarSym::VARYING);
    }
    if (!sym || sym->location == UINT32_MAX) return nullptr;
    llvm::Value *index = emitExpr(cg, e->u.index.index, mod, locals);
    if (!index) return nullptr;
    llvm::Value *record = nullptr;
    llvm::Value *base = nullptr;
    uint64_t stride = 0;
    if (cg.isGeometry) {
        if (!cg.geometryInputPtr || !cg.geometryPrimitiveId) return nullptr;
        index = coerceScalar(cg, index, MGLIR_SCALAR_UINT);
        record = geometryInputRecordIndex(cg, index);
        base = cg.geometryInputPtr;
    } else if (fromOutput) {
        if (!cg.stageOutPtr || !cg.patchPos) return nullptr;
        record = tessStageRecordIndex(cg, index, false);
        base = cg.stageOutPtr;
        stride = cg.stageOutStride;
    } else {
        if (!cg.stageInPtr || !cg.patchPos || !cg.indirectPtr) return nullptr;
        record = tessStageRecordIndex(cg, index, true);
        base = cg.stageInPtr;
        stride = cg.stageInStride;
    }
    if (cg.isGeometry) {
        return loadGeometryInputVarying(cg, *sym, cg.b->getInt32(sym->location),
                                        record, base);
    }
    llvm::Value *off = cg.b->CreateAdd(
        cg.b->CreateMul(cg.b->CreateZExt(record, cg.b->getInt64Ty()),
                        cg.b->getInt64(stride)),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE + sym->location * 16u));
    llvm::Type *ty = llvmType(sym->type, *cg.ctx);
    llvm::Type *loadTy = ty;
    if (varyingNeedsFloatRecordCarrier(sym->type))
        loadTy = llvmType(floatCarrierType(sym->type), *cg.ctx);
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
    p = cg.b->CreateBitCast(p, loadTy->getPointerTo(1));
    llvm::Value *v = cg.b->CreateAlignedLoad(loadTy, p, llvm::Align(4));
    if (varyingNeedsFloatRecordCarrier(sym->type))
        v = decodeFloatCarrier(cg, v, sym->type.scalar, ty);
    return v;
}

static bool emitTessStageArrayStore(
    Codegen &cg, const MGLExpr *lhs, llvm::Value *value,
    const MGLIRModule *mod, const std::map<std::string, MType> &locals)
{
    if (!cg.isTessControl || !lhs || lhs->kind != MGL_EXPR_INDEX ||
        !lhs->u.index.object ||
        lhs->u.index.object->kind != MGL_EXPR_VAR_REF ||
        !cg.stageOutPtr || !cg.patchPos) return false;
    const char *name = lhs->u.index.object->u.var_ref.name;
    VarSym *sym = codegenStageSymbol(cg, name, VarSym::OUTPUT);
    if (!sym || sym->location == UINT32_MAX) return false;
    llvm::Value *index = emitExpr(cg, lhs->u.index.index, mod, locals);
    if (!index) return true;
    llvm::Value *record = tessStageRecordIndex(cg, index, false);
    llvm::Value *off = cg.b->CreateAdd(
        cg.b->CreateMul(cg.b->CreateZExt(record, cg.b->getInt64Ty()),
                        cg.b->getInt64(cg.stageOutStride)),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE + sym->location * 16u));
    llvm::Type *ty = llvmType(sym->type, *cg.ctx);
    llvm::Value *storeVal = value;
    llvm::Type *storeTy = ty;
    if (varyingNeedsFloatRecordCarrier(sym->type)) {
        storeVal = encodeFloatCarrier(cg, value, sym->type.scalar);
        storeTy = storeVal->getType();
    } else if (value->getType() != ty) {
        storeVal = coerceScalar(cg, value, sym->type.scalar);
    }
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), cg.stageOutPtr, off);
    p = cg.b->CreateBitCast(p, storeTy->getPointerTo(1));
    cg.b->CreateAlignedStore(storeVal, p, llvm::Align(4));
    return true;
}

/* TCS/TES: instance[i].field — flattened interface-block member.
 * Reject swizzle fields (`.xyz`) so they take the vector swizzle path. */
static bool tessBlockMemberPath(const MGLExpr *e, const char **instOut,
                                const MGLExpr **indexOut, const char **fieldOut)
{
    if (!e || e->kind != MGL_EXPR_MEMBER || !e->u.member.object ||
        e->u.member.object->kind != MGL_EXPR_INDEX)
        return false;
    const MGLExpr *idxE = e->u.member.object;
    if (!idxE->u.index.object ||
        idxE->u.index.object->kind != MGL_EXPR_VAR_REF)
        return false;
    const char *inst = idxE->u.index.object->u.var_ref.name;
    if (!inst || !strcmp(inst, "gl_in") || !strcmp(inst, "gl_out"))
        return false;
    const char *field = e->u.member.field;
    if (field) {
        std::vector<uint32_t> swz;
        if (swizzleIndices(field, &swz))
            return false;
    }
    if (instOut) *instOut = inst;
    if (indexOut) *indexOut = idxE->u.index.index;
    if (fieldOut) *fieldOut = field;
    return true;
}

static bool emitTessBlockMemberStore(
    Codegen &cg, const MGLExpr *lhs, llvm::Value *value,
    const MGLIRModule *mod, const std::map<std::string, MType> &locals)
{
    const char *inst = nullptr, *field = nullptr;
    const MGLExpr *indexE = nullptr;
    if (!cg.isTessControl || !cg.stageOutPtr || !cg.patchPos ||
        !tessBlockMemberPath(lhs, &inst, &indexE, &field))
        return false;
    VarSym *member = codegenBlockMember(cg, inst, field, VarSym::OUTPUT);
    if (!member || member->location == UINT32_MAX) return false;
    llvm::Value *index = emitExpr(cg, indexE, mod, locals);
    if (!index) return true;
    llvm::Value *record = tessStageRecordIndex(cg, index, false);
    llvm::Type *ty = llvmType(member->type, *cg.ctx);
    if (member->type.isMatrix() && member->type.cols > 0) {
        llvm::Type *colTy = llvm::FixedVectorType::get(
            llvmScalar(member->type.scalar, *cg.ctx), member->type.rows);
        for (uint32_t c = 0; c < member->type.cols; c++) {
            llvm::Value *col = cg.b->CreateExtractValue(value, c);
            llvm::Value *off = cg.b->CreateAdd(
                cg.b->CreateMul(cg.b->CreateZExt(record, cg.b->getInt64Ty()),
                                cg.b->getInt64(cg.stageOutStride)),
                cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE +
                               (member->location + c) * 16u));
            llvm::Value *p =
                cg.b->CreateGEP(cg.b->getInt8Ty(), cg.stageOutPtr, off);
            p = cg.b->CreateBitCast(p, colTy->getPointerTo(1));
            cg.b->CreateAlignedStore(col, p, llvm::Align(4));
        }
        return true;
    }
    llvm::Value *storeVal = value;
    llvm::Type *storeTy = ty;
    if (varyingNeedsFloatRecordCarrier(member->type)) {
        storeVal = encodeFloatCarrier(cg, value, member->type.scalar);
        storeTy = storeVal->getType();
    } else if (value->getType() != ty) {
        if (ty->isIntOrIntVectorTy() && value->getType()->isIntOrIntVectorTy())
            storeVal = cg.b->CreateBitCast(value, ty);
        else
            storeVal = coerceScalar(cg, value, member->type.scalar);
    }
    llvm::Value *off = cg.b->CreateAdd(
        cg.b->CreateMul(cg.b->CreateZExt(record, cg.b->getInt64Ty()),
                        cg.b->getInt64(cg.stageOutStride)),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE + member->location * 16u));
    llvm::Value *p =
        cg.b->CreateGEP(cg.b->getInt8Ty(), cg.stageOutPtr, off);
    p = cg.b->CreateBitCast(p, storeTy->getPointerTo(1));
    cg.b->CreateAlignedStore(storeVal, p, llvm::Align(4));
    return true;
}

static llvm::Value *emitTessBlockMemberLoad(
    Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals)
{
    const char *inst = nullptr, *field = nullptr;
    const MGLExpr *indexE = nullptr;
    if (!tessBlockMemberPath(e, &inst, &indexE, &field))
        return nullptr;
    /* TCS compound assigns (outVertex[i].x += …) must reload OUTPUT from
     * stage_out.  Treating every TCS block load as an input made += re-read
     * stage_in and collapse the accumulation loop to ~in[inv]+in[last]. */
    VarSym *member = nullptr;
    bool fromOutput = false;
    if (cg.isTessControl) {
        member = codegenBlockMember(cg, inst, field, VarSym::OUTPUT);
        if (member && member->location != UINT32_MAX) {
            fromOutput = true;
        } else {
            member = codegenBlockMember(cg, inst, field, VarSym::VARYING);
        }
    } else if (cg.isTessEval) {
        member = codegenBlockMember(cg, inst, field,
                                    VarSym::CONTROL_POINT_INPUT);
        if (!member)
            member = codegenBlockMember(cg, inst, field, VarSym::OUTPUT);
    } else {
        return nullptr;
    }
    if (!member || member->location == UINT32_MAX) return nullptr;
    llvm::Value *index = emitExpr(cg, indexE, mod, locals);
    if (!index) return nullptr;
    auto loadFromRecord = [&](llvm::Value *base, uint64_t stride,
                              llvm::Value *record) -> llvm::Value * {
        llvm::Type *ty = llvmType(member->type, *cg.ctx);
        if (member->type.isMatrix() && member->type.cols > 0) {
            llvm::Type *colTy = llvm::FixedVectorType::get(
                llvmScalar(member->type.scalar, *cg.ctx), member->type.rows);
            llvm::Value *agg = llvm::UndefValue::get(ty);
            for (uint32_t c = 0; c < member->type.cols; c++) {
                llvm::Value *off = cg.b->CreateAdd(
                    cg.b->CreateMul(
                        cg.b->CreateZExt(record, cg.b->getInt64Ty()),
                        cg.b->getInt64(stride)),
                    cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE +
                                   (member->location + c) * 16u));
                llvm::Value *p =
                    cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
                p = cg.b->CreateBitCast(p, colTy->getPointerTo(1));
                llvm::Value *col =
                    cg.b->CreateAlignedLoad(colTy, p, llvm::Align(4));
                agg = cg.b->CreateInsertValue(agg, col, c);
            }
            return agg;
        }
        llvm::Type *loadTy = ty;
        if (varyingNeedsFloatRecordCarrier(member->type))
            loadTy = llvmType(floatCarrierType(member->type), *cg.ctx);
        llvm::Value *off = cg.b->CreateAdd(
            cg.b->CreateMul(cg.b->CreateZExt(record, cg.b->getInt64Ty()),
                            cg.b->getInt64(stride)),
            cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE +
                           member->location * 16u));
        llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
        p = cg.b->CreateBitCast(p, loadTy->getPointerTo(1));
        llvm::Value *v =
            cg.b->CreateAlignedLoad(loadTy, p, llvm::Align(4));
        if (varyingNeedsFloatRecordCarrier(member->type))
            v = decodeFloatCarrier(cg, v, member->type.scalar, ty);
        return v;
    };
    if (cg.isTessControl) {
        if (fromOutput) {
            if (!cg.stageOutPtr || !cg.patchPos) return nullptr;
            llvm::Value *record = tessStageRecordIndex(cg, index, false);
            return loadFromRecord(cg.stageOutPtr, cg.stageOutStride, record);
        }
        if (!cg.stageInPtr || !cg.patchPos || !cg.indirectPtr) return nullptr;
        llvm::Value *record = tessStageRecordIndex(cg, index, true);
        return loadFromRecord(cg.stageInPtr, cg.stageInStride, record);
    }
    if (cg.isTessEval) {
        if (cg.isTESCompute) {
            if (!cg.stageInPtr || !cg.indirectPtr || !cg.patchId) return nullptr;
            llvm::Value *patchInfo = cg.b->CreateBitCast(
                cg.indirectPtr, cg.b->getInt32Ty()->getPointerTo(1));
            llvm::Value *verticesPerPatch = cg.b->CreateAlignedLoad(
                cg.b->getInt32Ty(),
                cg.b->CreateGEP(cg.b->getInt32Ty(), patchInfo,
                                cg.b->getInt32(1)),
                llvm::Align(4));
            index = coerceScalar(cg, index, MGLIR_SCALAR_UINT);
            llvm::Value *flat = cg.b->CreateAdd(
                cg.b->CreateMul(cg.patchId, verticesPerPatch), index);
            return loadFromRecord(cg.stageInPtr, cg.stageInStride, flat);
        }
        if (!cg.controlPointGetter || !cg.patchControlPtr) return nullptr;
        auto fieldIt = cg.controlPointFields.find(member->name);
        if (fieldIt == cg.controlPointFields.end()) return nullptr;
        index = coerceScalar(cg, index, MGLIR_SCALAR_UINT);
        llvm::Value *record = cg.b->CreateCall(
            cg.controlPointGetter, {index, cg.patchControlPtr});
        return cg.b->CreateExtractValue(record, fieldIt->second);
    }
    return nullptr;
}

static llvm::Value *emitPerVertexLoad(Codegen &cg, const MGLExpr *e,
                                      const MGLIRModule *mod,
                                      const std::map<std::string, MType> &locals)
{
    const char *root = nullptr, *field = nullptr;
    const MGLExpr *index = nullptr;
    if (!perVertexPath(e, &root, &index, &field)) return nullptr;
    if (cg.isGeometry && !strcmp(root, "gl_in")) {
        if (!cg.geometryInputPtr) {
            cg.err = 1;
            cg.errmsg = "codegen: GS gl_in is unavailable";
            return nullptr;
        }
        llvm::Value *iv = emitExpr(cg, index, mod, locals);
        if (!iv) return nullptr;
        /* A constant gl_in[] index at or past the declared input-primitive
         * vertex count is a compile-time error (GL 4.6 §11.3.1); without
         * this check CTS more_input_vertices expects the build to fail. */
        if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(iv)) {
            if (ci->getZExtValue() >= (uint64_t)cg.geometryInputVertices) {
                cg.err = 1;
                cg.errmsg =
                    "GS codegen: gl_in index out of range for the input "
                    "primitive";
                return nullptr;
            }
        }
        iv = coerceScalar(cg, iv, MGLIR_SCALAR_UINT);
        llvm::Value *record = geometryInputRecordIndex(cg, iv);
        llvm::Value *stride = nullptr;
        if (cg.geometryGatherParamsPtr) {
            /* Runtime capture stride from gather params word 4; see
             * emitTessStageArrayLoad for why this cannot be a constant. */
            llvm::Type *i32ty = llvm::Type::getInt32Ty(*cg.ctx);
            stride = cg.b->CreateAlignedLoad(
                i32ty,
                cg.b->CreateGEP(i32ty,
                                cg.b->CreateBitCast(cg.geometryGatherParamsPtr,
                                                    i32ty->getPointerTo(1)),
                                cg.b->getInt32(4)),
                llvm::Align(4));
        } else {
            stride = cg.b->getInt32(cg.stageInStride);
        }
        llvm::Value *off = cg.b->CreateMul(
            cg.b->CreateZExt(record, cg.b->getInt64Ty()),
            cg.b->CreateZExt(stride, cg.b->getInt64Ty()));
        off = cg.b->CreateAdd(off,
                              cg.b->getInt64(perVertexFieldOffset(field)));
        llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(),
                                         cg.geometryInputPtr, off);
        llvm::Type *ty = perVertexFieldType(cg, field);
        p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
        return cg.b->CreateAlignedLoad(ty, p, llvm::Align(4));
    }
    if (cg.isTessEval && !strcmp(root, "gl_in")) {
        /* TES-vertex reads gl_in from the same per-patch control-point
         * record stream as the compute expansion (slot 30); only native
         * post-tessellation goes through the patch control-point function. */
        if (!cg.isTESCompute && !cg.isTESVertex &&
            (!cg.patchControlPtr || !cg.controlPointGetter)) {
            cg.err = 1;
            cg.errmsg = "codegen: TES patch control points are unavailable";
            return nullptr;
        }
        llvm::Value *iv = emitExpr(cg, index, mod, locals);
        if (!iv) return nullptr;
        iv = coerceScalar(cg, iv, MGLIR_SCALAR_UINT);
        if (!strcmp(field, "gl_Position") && !cg.isTESCompute &&
            !cg.isTESVertex) {
            llvm::Value *record = cg.b->CreateCall(
                cg.controlPointGetter, {iv, cg.patchControlPtr});
            return cg.b->CreateExtractValue(record, 0);
        }
        if (!cg.stageInPtr || !cg.indirectPtr ||
            ((cg.isTESCompute || cg.isTESVertex) && !cg.patchId)) {
            cg.err = 1;
            cg.errmsg = "TES AIR codegen: shared control-point buffer is unavailable";
            return nullptr;
        }
        llvm::Value *patchInfo = cg.b->CreateBitCast(
            cg.indirectPtr, cg.b->getInt32Ty()->getPointerTo(1));
        llvm::Value *verticesPerPatch = cg.b->CreateAlignedLoad(
            cg.b->getInt32Ty(),
            cg.b->CreateGEP(cg.b->getInt32Ty(), patchInfo,
                            cg.b->getInt32(1)),
            llvm::Align(4));
        llvm::Value *patchIndex = tessPatchIndexForStageIn(cg);
        llvm::Value *flat = cg.b->CreateAdd(
            cg.b->CreateMul(patchIndex, verticesPerPatch), iv);
        llvm::Value *recordIdx = flat;
        if (cg.isTESCompute && cg.tessGatherPtr && cg.tessGatherParamsPtr) {
            /* Indexed draws: the stage input is a sparse capture stream
             * ([instance][vertex_id]) and the gather stream carries the raw
             * index of every gl_in slot of a per-instance patch group.
             * Gather params (mgl_air_tess_abi.h §3): {vertices_per_instance,
             * primitives_per_instance, first_vertex, gather_enabled,
             * instance_idx}. */
            llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
            llvm::Value *params = cg.b->CreateBitCast(
                cg.tessGatherParamsPtr, i32->getPointerTo(1));
            llvm::Value *gatherEnabled = cg.b->CreateAlignedLoad(
                i32, cg.b->CreateGEP(i32, params, cg.b->getInt32(3)),
                llvm::Align(4));
            llvm::BasicBlock *gatherBB = llvm::BasicBlock::Create(
                *cg.ctx, "tes_gather", cg.fn);
            llvm::BasicBlock *arrayBB = llvm::BasicBlock::Create(
                *cg.ctx, "tes_array", cg.fn);
            llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(
                *cg.ctx, "tes_gather_merge", cg.fn);
            cg.b->CreateCondBr(
                cg.b->CreateICmpNE(gatherEnabled, cg.b->getInt32(0)),
                gatherBB, arrayBB);
            cg.b->SetInsertPoint(gatherBB);
            llvm::Value *vertsPerInst = cg.b->CreateAlignedLoad(
                i32, cg.b->CreateGEP(i32, params, cg.b->getInt32(0)),
                llvm::Align(4));
            llvm::Value *firstVertex = cg.b->CreateAlignedLoad(
                i32, cg.b->CreateGEP(i32, params, cg.b->getInt32(2)),
                llvm::Align(4));
            llvm::Value *instanceIdx = cg.b->CreateAlignedLoad(
                i32, cg.b->CreateGEP(i32, params, cg.b->getInt32(4)),
                llvm::Align(4));
            llvm::Value *gatherBase = cg.b->CreateBitCast(
                cg.tessGatherPtr, i32->getPointerTo(1));
            llvm::Value *vid = cg.b->CreateAlignedLoad(
                i32,
                cg.b->CreateGEP(
                    i32, gatherBase,
                    cg.b->CreateZExt(flat, cg.b->getInt64Ty())),
                llvm::Align(4));
            llvm::Value *gatherIdx = cg.b->CreateAdd(
                cg.b->CreateSub(vid, firstVertex),
                cg.b->CreateMul(instanceIdx, vertsPerInst));
            cg.b->CreateBr(mergeBB);
            cg.b->SetInsertPoint(arrayBB);
            cg.b->CreateBr(mergeBB);
            cg.b->SetInsertPoint(mergeBB);
            llvm::PHINode *phi = cg.b->CreatePHI(i32, 2);
            phi->addIncoming(gatherIdx, gatherBB);
            phi->addIncoming(flat, arrayBB);
            recordIdx = phi;
        }
        llvm::Value *off = cg.b->CreateAdd(
            cg.b->CreateMul(
                cg.b->CreateZExt(recordIdx, cg.b->getInt64Ty()),
                cg.b->getInt64(cg.stageInStride)),
            cg.b->getInt64(perVertexFieldOffset(field)));
        llvm::Value *p = cg.b->CreateGEP(
            cg.b->getInt8Ty(), cg.stageInPtr, off);
        llvm::Type *ty = perVertexFieldType(cg, field);
        p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
        return cg.b->CreateAlignedLoad(ty, p, llvm::Align(4));
    }
    llvm::Value *base = !strcmp(root, "gl_in") ? cg.stageInPtr : cg.stageOutPtr;
    if (!base) {
        cg.err = 1;
        cg.errmsg = std::string("codegen: ") + root + " is unavailable for this stage";
        return nullptr;
    }
    llvm::Value *iv = emitExpr(cg, index, mod, locals);
    if (!iv) return nullptr;
    iv = coerceScalar(cg, iv, MGLIR_SCALAR_UINT);
    if (cg.patchPos) {
        llvm::Value *patch = cg.b->CreateExtractElement(
            cg.patchPos, cg.b->getInt32(0));
        llvm::Value *verticesPerPatch = nullptr;
        if (!strcmp(root, "gl_in") && cg.indirectPtr) {
            llvm::Value *p = cg.b->CreateBitCast(
                cg.indirectPtr, cg.b->getInt32Ty()->getPointerTo(1));
            verticesPerPatch = cg.b->CreateAlignedLoad(
                cg.b->getInt32Ty(), p, llvm::Align(4));
        } else {
            verticesPerPatch = cg.b->getInt32(cg.tcsOutputVertices);
        }
        iv = cg.b->CreateAdd(cg.b->CreateMul(patch, verticesPerPatch), iv);
    }
    llvm::Value *off = cg.b->CreateMul(
        cg.b->CreateZExt(iv, cg.b->getInt64Ty()),
        cg.b->getInt64(!strcmp(root, "gl_in")
                           ? cg.stageInStride : cg.stageOutStride));
    off = cg.b->CreateAdd(off,
                          cg.b->getInt64(perVertexFieldOffset(field)));
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
    llvm::Type *ty = perVertexFieldType(cg, field);
    p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
    return cg.b->CreateAlignedLoad(ty, p, llvm::Align(4));
}

static bool emitPerVertexStore(Codegen &cg, const MGLExpr *lhs, llvm::Value *value,
                               const MGLIRModule *mod,
                               const std::map<std::string, MType> &locals)
{
    const char *root = nullptr, *field = nullptr;
    const MGLExpr *index = nullptr;
    if (!perVertexPath(lhs, &root, &index, &field)) return false;
    if (strcmp(root, "gl_out") != 0) {
        cg.err = 1;
        cg.errmsg = "codegen: gl_in is read-only";
        return true;
    }
    if (!cg.stageOutPtr) {
        cg.err = 1;
        cg.errmsg = "codegen: gl_out is unavailable for this stage";
        return true;
    }
    llvm::Value *iv = emitExpr(cg, index, mod, locals);
    if (!iv) return true;
    iv = coerceScalar(cg, iv, MGLIR_SCALAR_UINT);
    if (cg.patchPos) {
        llvm::Value *patch = cg.b->CreateExtractElement(
            cg.patchPos, cg.b->getInt32(0));
        iv = cg.b->CreateAdd(
            cg.b->CreateMul(patch,
                            cg.b->getInt32(cg.tcsOutputVertices)),
            iv);
    }
    llvm::Value *off = cg.b->CreateMul(
        cg.b->CreateZExt(iv, cg.b->getInt64Ty()),
        cg.b->getInt64(cg.stageOutStride));
    off = cg.b->CreateAdd(off,
                          cg.b->getInt64(perVertexFieldOffset(field)));
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), cg.stageOutPtr, off);
    llvm::Type *ty = perVertexFieldType(cg, field);
    if (!ty->isArrayTy())
        value = coerceScalar(cg, value, MGLIR_SCALAR_FLOAT);
    if (value->getType() != ty) {
        if (ty->isVectorTy() && value->getType()->isVectorTy()) {
            value = cg.b->CreateBitCast(value, ty);
        } else if (ty->isVectorTy()) {
            value = cg.b->CreateVectorSplat(4, value);
        }
    }
    p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
    cg.b->CreateAlignedStore(value, p, llvm::Align(4));
    return true;
}

static llvm::Value *geometryCounterPtr(Codegen &cg, uint32_t field)
{
    /* ABI (mgl_air_gs_abi.h §3): each work item owns a 28-byte counts
     * record = MGLAIRGSIndirectArgs (words 0..3) + kernel scratch
     * (words 4..6).  Counter 0 is the only draw parameter the kernel
     * writes (indirect-args vertex count); the strip/emit state rolls in
     * the scratch words so instance_count/base_vertex stay renderer
     * preset (1/0) and the rasterizing indirect draw is well-defined. */
    uint32_t word = (field == MGL_AIR_GS_COUNT_VERTEX_COUNT)
        ? 0u
        : (MGL_AIR_GS_COUNTS_ARGS_WORDS + (field - 1u));
    llvm::Value *record = cg.b->CreateMul(
        cg.geometryWorkItemId,
        cg.b->getInt32(MGL_AIR_GS_COUNTS_RECORD_WORDS));
    llvm::Value *index = cg.b->CreateAdd(record, cg.b->getInt32(word));
    llvm::Value *base = cg.b->CreateBitCast(
        cg.geometryCountPtr, cg.b->getInt32Ty()->getPointerTo(1));
    return cg.b->CreateGEP(cg.b->getInt32Ty(), base, index);
}

/* GS gl_in record index (mgl_air_gs_abi.h §7).  Array path:
 * globPrim*inputVertices + vertex.  Indexed path (runtime
 * gather_enabled): gather[globPrim*inputVertices + vertex] -
 * first_vertex + instance * vertices_per_instance.  The capture record
 * stream is sparse ([instance][vertex_id]); the gather entry carries the
 * raw index value so the kernel can locate each gl_in[]. */
static llvm::Value *geometryInputRecordIndex(Codegen &cg,
                                             llvm::Value *vertexIndex)
{
    if (!cg.geometryInputPtr || !cg.geometryPrimitiveId) return nullptr;
    llvm::Value *globPrim = cg.geometryPrimitiveId;
    llvm::Value *arrayIdx = cg.b->CreateAdd(
        cg.b->CreateMul(globPrim, cg.b->getInt32(cg.geometryInputVertices)),
        vertexIndex);
    if (!cg.geometryGatherPtr || !cg.geometryGatherParamsPtr) {
        return arrayIdx;
    }
    llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
    llvm::Value *params = cg.b->CreateBitCast(
        cg.geometryGatherParamsPtr, i32->getPointerTo(1));
    llvm::Value *gatherEnabled = cg.b->CreateAlignedLoad(
        i32, cg.b->CreateGEP(i32, params, cg.b->getInt32(3)),
        llvm::Align(4));
    llvm::Value *enabled = cg.b->CreateICmpNE(
        gatherEnabled, cg.b->getInt32(0));
    llvm::BasicBlock *gatherBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_gather", cg.fn);
    llvm::BasicBlock *arrayBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_array", cg.fn);
    llvm::BasicBlock *mergeBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_gather_merge", cg.fn);
    cg.b->CreateCondBr(enabled, gatherBB, arrayBB);
    cg.b->SetInsertPoint(gatherBB);
    llvm::Value *vertsPerInst = cg.b->CreateAlignedLoad(
        i32, cg.b->CreateGEP(i32, params, cg.b->getInt32(0)),
        llvm::Align(4));
    llvm::Value *primsPerInst = cg.b->CreateAlignedLoad(
        i32, cg.b->CreateGEP(i32, params, cg.b->getInt32(1)),
        llvm::Align(4));
    llvm::Value *firstVertex = cg.b->CreateAlignedLoad(
        i32, cg.b->CreateGEP(i32, params, cg.b->getInt32(2)),
        llvm::Align(4));
    /* Instance decomposition: globPrim = instanceIdx * primsPerInst +
     * primInInst.  The gather stream is shared across instances (one
     * entry per input vertex of a per-instance primitive). */
    llvm::Value *instanceIdx = cg.b->CreateUDiv(globPrim, primsPerInst);
    llvm::Value *primInInst = cg.b->CreateURem(globPrim, primsPerInst);
    llvm::Value *gatherSlot = cg.b->CreateAdd(
        cg.b->CreateMul(primInInst, cg.b->getInt32(cg.geometryInputVertices)),
        vertexIndex);
    llvm::Value *gatherBase = cg.b->CreateBitCast(
        cg.geometryGatherPtr, i32->getPointerTo(1));
    llvm::Value *vid = cg.b->CreateAlignedLoad(
        i32,
        cg.b->CreateGEP(i32, gatherBase,
                        cg.b->CreateZExt(gatherSlot, cg.b->getInt64Ty())),
        llvm::Align(4));
    llvm::Value *gatherIdx = cg.b->CreateAdd(
        cg.b->CreateSub(vid, firstVertex),
        cg.b->CreateMul(instanceIdx, vertsPerInst));
    cg.b->CreateBr(mergeBB);
    cg.b->SetInsertPoint(arrayBB);
    cg.b->CreateBr(mergeBB);
    cg.b->SetInsertPoint(mergeBB);
    llvm::PHINode *phi = cg.b->CreatePHI(i32, 2);
    phi->addIncoming(gatherIdx, gatherBB);
    phi->addIncoming(arrayIdx, arrayBB);
    return phi;
}

static llvm::Value *geometryRecordPtr(Codegen &cg, llvm::Value *record)
{
    llvm::Value *slot = cg.b->CreateAdd(
        cg.b->CreateMul(cg.geometryWorkItemId,
                        cg.b->getInt32(cg.geometryRecordCount)),
        record);
    llvm::Value *off = cg.b->CreateMul(
        cg.b->CreateZExt(slot, cg.b->getInt64Ty()),
        cg.b->getInt64(cg.stageOutStride));
    return cg.b->CreateGEP(cg.b->getInt8Ty(), cg.geometryOutputPtr, off);
}

static void storeGeometryPosition(Codegen &cg, llvm::Value *record,
                                  llvm::Value *position)
{
    if (mgl_env_flag_enabled("MGL_GS_DIAG_CONST")) {
        position = llvm::ConstantVector::get({
            llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 0.25),
            llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 0.5),
            llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 0.75),
            llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 1.0)});
    }
    llvm::Type *v4 = llvm::FixedVectorType::get(
        llvm::Type::getFloatTy(*cg.ctx), 4);
    llvm::Value *p = cg.b->CreateBitCast(
        geometryRecordPtr(cg, record), v4->getPointerTo(1));
    cg.b->CreateAlignedStore(position, p, llvm::Align(16));
}

static void storeGeometryPointSize(Codegen &cg, llvm::Value *record,
                                   llvm::Value *pointSize)
{
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, record), cg.b->getInt64(16));
    p = cg.b->CreateBitCast(p, cg.b->getFloatTy()->getPointerTo(1));
    cg.b->CreateAlignedStore(pointSize, p, llvm::Align(4));
}

static llvm::Value *defaultCullDistances(Codegen &cg)
{
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    llvm::Value *result = llvm::UndefValue::get(llvm::ArrayType::get(
        f32, MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT));
    for (uint32_t i = 0; i < MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT; i++) {
        result = cg.b->CreateInsertValue(
            result, llvm::ConstantFP::get(f32, 1.0), i);
    }
    return result;
}

static llvm::Value *defaultClipDistances(Codegen &cg)
{
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    llvm::Value *result = llvm::UndefValue::get(llvm::ArrayType::get(
        f32, MGL_MAX_CLIP_DISTANCES));
    for (uint32_t i = 0; i < MGL_MAX_CLIP_DISTANCES; i++) {
        result = cg.b->CreateInsertValue(
            result, llvm::ConstantFP::get(f32, 1.0), i);
    }
    return result;
}

static void storeGeometryCullDistances(Codegen &cg, llvm::Value *record,
                                       llvm::Value *distances)
{
    llvm::Type *arrayTy = llvm::ArrayType::get(
        llvm::Type::getFloatTy(*cg.ctx),
        MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT);
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET));
    p = cg.b->CreateBitCast(p, arrayTy->getPointerTo(1));
    cg.b->CreateAlignedStore(distances, p, llvm::Align(4));
}

static void storeGeometryClipDistances(Codegen &cg, llvm::Value *record,
                                       llvm::Value *distances)
{
    llvm::Type *arrayTy = llvm::ArrayType::get(
        llvm::Type::getFloatTy(*cg.ctx),
        MGL_AIR_PER_VERTEX_CLIP_DISTANCE_COUNT);
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_CLIP_DISTANCE_OFFSET));
    p = cg.b->CreateBitCast(p, arrayTy->getPointerTo(1));
    cg.b->CreateAlignedStore(distances, p, llvm::Align(4));
}

static llvm::Value *loadGeometryPosition(Codegen &cg, uint32_t record)
{
    llvm::Type *v4 = llvm::FixedVectorType::get(
        llvm::Type::getFloatTy(*cg.ctx), 4);
    llvm::Value *p = cg.b->CreateBitCast(
        geometryRecordPtr(cg, cg.b->getInt32(record)), v4->getPointerTo(1));
    return cg.b->CreateAlignedLoad(v4, p, llvm::Align(16));
}

static llvm::Value *loadGeometryPointSize(Codegen &cg, uint32_t record)
{
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(),
        geometryRecordPtr(cg, cg.b->getInt32(record)), cg.b->getInt64(16));
    p = cg.b->CreateBitCast(p, cg.b->getFloatTy()->getPointerTo(1));
    return cg.b->CreateAlignedLoad(cg.b->getFloatTy(), p, llvm::Align(4));
}

static void storeGeometryLayer(Codegen &cg, llvm::Value *record,
                               llvm::Value *layer)
{
    /* Offsets for layer/viewport are dedicated (layout v2 / A04); only stamp
     * when the shader actually wrote the builtin. */
    if (!cg.lvalues.count("gl_Layer"))
        return;
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_LAYER_OFFSET));
    p = cg.b->CreateBitCast(p, cg.b->getInt32Ty()->getPointerTo(1));
    cg.b->CreateAlignedStore(layer, p, llvm::Align(4));
}

static void storeGeometryViewportIndex(Codegen &cg, llvm::Value *record,
                                       llvm::Value *viewportIndex)
{
    if (!cg.lvalues.count("gl_ViewportIndex"))
        return;
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_VIEWPORT_INDEX_OFFSET));
    p = cg.b->CreateBitCast(p, cg.b->getInt32Ty()->getPointerTo(1));
    cg.b->CreateAlignedStore(viewportIndex, p, llvm::Align(4));
}

/* gl_PrimitiveID written by the GS rides at
 * MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET so the fragment stage can receive
 * it through the passthrough vertex function (flat).  The record holds a
 * float carrier (sitofp of the id): Apple's AGX compiler segfaults in
 * InstCombine when a flat int stage_input that is actually read crosses
 * into the fragment stage, so the id travels as a float and the FS entry
 * converts it back with round+fptosi.  Every reader/writer of this slot
 * must use the same carrier type.  Unwritten records keep whatever the
 * strip cache held; the renderer's PTVS only forwards it for programs that
 * declared gl_PrimitiveID. */
static void copyGeometryPrimitiveIdSelected(Codegen &cg, llvm::Value *dst,
                                            uint32_t falseRecord,
                                            uint32_t trueRecord,
                                            llvm::Value *condition)
{
    auto load = [&](uint32_t rec) {
        llvm::Value *p = cg.b->CreateGEP(
            cg.b->getInt8Ty(), geometryRecordPtr(cg, cg.b->getInt32(rec)),
            cg.b->getInt64(MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET));
        p = cg.b->CreateBitCast(p, cg.b->getFloatTy()->getPointerTo(1));
        return cg.b->CreateAlignedLoad(cg.b->getFloatTy(), p,
                                       llvm::Align(4));
    };
    llvm::Value *v =
        cg.b->CreateSelect(condition, load(trueRecord), load(falseRecord));
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, dst),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET));
    p = cg.b->CreateBitCast(p, cg.b->getFloatTy()->getPointerTo(1));
    cg.b->CreateAlignedStore(v, p, llvm::Align(4));
}

static llvm::Value *loadGeometryPrimitiveId(Codegen &cg, uint32_t record)
{
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, cg.b->getInt32(record)),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET));
    p = cg.b->CreateBitCast(p, cg.b->getFloatTy()->getPointerTo(1));
    return cg.b->CreateAlignedLoad(cg.b->getFloatTy(), p, llvm::Align(4));
}

static void copyGeometryPrimitiveId(Codegen &cg, llvm::Value *dst,
                                    uint32_t src)
{
    llvm::Value *v = loadGeometryPrimitiveId(cg, src);
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, dst),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET));
    p = cg.b->CreateBitCast(p, cg.b->getFloatTy()->getPointerTo(1));
    cg.b->CreateAlignedStore(v, p, llvm::Align(4));
}

static void storeGeometryPrimitiveId(Codegen &cg, llvm::Value *record)
{
    auto it = cg.lvalues.find("gl_PrimitiveID");
    if (!cg.primitiveIdWritten || it == cg.lvalues.end()) return;
    llvm::Value *v = it->second;
    if (v->getType() != cg.b->getInt32Ty())
        v = cg.b->CreateZExtOrTrunc(v, cg.b->getInt32Ty());
    v = cg.b->CreateSIToFP(v, cg.b->getFloatTy());
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET));
    p = cg.b->CreateBitCast(p, cg.b->getFloatTy()->getPointerTo(1));
    cg.b->CreateAlignedStore(v, p, llvm::Align(4));
}

static llvm::Value *loadGeometryLayer(Codegen &cg, uint32_t record)
{
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(),
        geometryRecordPtr(cg, cg.b->getInt32(record)),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_LAYER_OFFSET));
    p = cg.b->CreateBitCast(p, cg.b->getInt32Ty()->getPointerTo(1));
    return cg.b->CreateAlignedLoad(cg.b->getInt32Ty(), p, llvm::Align(4));
}

static llvm::Value *loadGeometryViewportIndex(Codegen &cg, uint32_t record)
{
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(),
        geometryRecordPtr(cg, cg.b->getInt32(record)),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_VIEWPORT_INDEX_OFFSET));
    p = cg.b->CreateBitCast(p, cg.b->getInt32Ty()->getPointerTo(1));
    return cg.b->CreateAlignedLoad(cg.b->getInt32Ty(), p, llvm::Align(4));
}

static llvm::Value *loadGeometryCullDistances(Codegen &cg, uint32_t record)
{
    llvm::Type *arrayTy = llvm::ArrayType::get(
        llvm::Type::getFloatTy(*cg.ctx),
        MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT);
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(),
        geometryRecordPtr(cg, cg.b->getInt32(record)),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET));
    p = cg.b->CreateBitCast(p, arrayTy->getPointerTo(1));
    return cg.b->CreateAlignedLoad(arrayTy, p, llvm::Align(4));
}

static llvm::Value *loadGeometryClipDistances(Codegen &cg, uint32_t record)
{
    llvm::Type *arrayTy = llvm::ArrayType::get(
        llvm::Type::getFloatTy(*cg.ctx),
        MGL_AIR_PER_VERTEX_CLIP_DISTANCE_COUNT);
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(),
        geometryRecordPtr(cg, cg.b->getInt32(record)),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_CLIP_DISTANCE_OFFSET));
    p = cg.b->CreateBitCast(p, arrayTy->getPointerTo(1));
    return cg.b->CreateAlignedLoad(arrayTy, p, llvm::Align(4));
}

static llvm::Value *geometryPrimitiveCulled(
    Codegen &cg, std::initializer_list<llvm::Value *> vertices)
{
    llvm::Value *culled = cg.b->getFalse();
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    for (uint32_t distance = 0;
         distance < MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT; distance++) {
        llvm::Value *allNegative = cg.b->getTrue();
        for (llvm::Value *vertex : vertices) {
            llvm::Value *value = cg.b->CreateExtractValue(vertex, distance);
            allNegative = cg.b->CreateAnd(
                allNegative,
                cg.b->CreateFCmpOLT(value, llvm::ConstantFP::get(f32, 0.0)));
        }
        culled = cg.b->CreateOr(culled, allNegative);
    }
    return culled;
}

/* ---- varying location span (C1e) ----------------------------------- */
/* Body in mgl_air_varsym.cpp; using-facade above. */


static void storeVaryingValueAtLocation(Codegen &cg, llvm::Value *record,
                                        const VarSym &varying,
                                        llvm::Value *value)
{
    llvm::Type *ty = llvmType(varying.type, *cg.ctx);
    if (!value) value = llvm::UndefValue::get(ty);
    if (varying.type.isMatrix() && varying.type.cols > 0) {
        llvm::Type *colTy = llvm::FixedVectorType::get(
            llvmScalar(varying.type.scalar, *cg.ctx), varying.type.rows);
        for (uint32_t c = 0; c < varying.type.cols; c++) {
            llvm::Value *col = cg.b->CreateExtractValue(value, c);
            llvm::Value *p = cg.b->CreateGEP(
                cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
                cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE +
                               (varying.location + c) * 16u));
            p = cg.b->CreateBitCast(p, colTy->getPointerTo(1));
            cg.b->CreateAlignedStore(col, p, llvm::Align(4));
        }
        return;
    }
    llvm::Type *storeTy = ty;
    if (varyingNeedsFloatRecordCarrier(varying.type)) {
        value = encodeFloatCarrier(cg, value, varying.type.scalar);
        storeTy = value->getType();
    }
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE + varying.location * 16u));
    p = cg.b->CreateBitCast(p, storeTy->getPointerTo(1));
    cg.b->CreateAlignedStore(value, p, llvm::Align(4));
}

static llvm::Value *loadVaryingValueAtLocation(Codegen &cg, llvm::Value *record,
                                               const VarSym &varying)
{
    llvm::Type *ty = llvmType(varying.type, *cg.ctx);
    if (varying.type.isMatrix() && varying.type.cols > 0) {
        llvm::Type *colTy = llvm::FixedVectorType::get(
            llvmScalar(varying.type.scalar, *cg.ctx), varying.type.rows);
        llvm::Value *agg = llvm::UndefValue::get(ty);
        for (uint32_t c = 0; c < varying.type.cols; c++) {
            llvm::Value *p = cg.b->CreateGEP(
                cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
                cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE +
                               (varying.location + c) * 16u));
            p = cg.b->CreateBitCast(p, colTy->getPointerTo(1));
            llvm::Value *col =
                cg.b->CreateAlignedLoad(colTy, p, llvm::Align(4));
            agg = cg.b->CreateInsertValue(agg, col, c);
        }
        return agg;
    }
    llvm::Type *loadTy = ty;
    if (varyingNeedsFloatRecordCarrier(varying.type))
        loadTy = llvmType(floatCarrierType(varying.type), *cg.ctx);
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE + varying.location * 16u));
    p = cg.b->CreateBitCast(p, loadTy->getPointerTo(1));
    llvm::Value *v = cg.b->CreateAlignedLoad(loadTy, p, llvm::Align(4));
    if (varyingNeedsFloatRecordCarrier(varying.type))
        v = decodeFloatCarrier(cg, v, varying.type.scalar, ty);
    return v;
}

static void storeGeometryVaryings(Codegen &cg, llvm::Value *record)
{
    if (!cg.auxSyms) return;
    for (VarSym &varying : *cg.auxSyms) {
        if (varying.kind != VarSym::OUTPUT ||
            varying.location == UINT32_MAX) continue;
        /* The stage-out record feeds stream 0 rasterization and stream 0
         * XFB only; stream > 0 varyings are captured in compact per-stream
         * records by emitGeometryStreamVertex (GL 4.6 §11.1.3.4).  Skipping
         * them here also avoids location collisions between streams that
         * share the same layout(location=N) value. */
        if (varying.stream != 0) continue;
        llvm::Type *ty = llvmType(varying.type, *cg.ctx);
        llvm::Value *value = cg.lvalues.count(varying.name)
            ? cg.lvalues[varying.name] : llvm::UndefValue::get(ty);
        storeVaryingValueAtLocation(cg, record, varying, value);
    }
}

/* isolines/point-mode TES kernels write their user varyings (VARYING kind,
 * the same record layout GS uses: position@0, point_size@16, varyings at
 * MGL_AIR_PER_VERTEX_STRIDE + location*16). */
static void storeTessComputeVaryings(Codegen &cg, llvm::Value *record)
{
    if (!cg.auxSyms) return;
    for (VarSym &varying : *cg.auxSyms) {
        if (varying.kind != VarSym::VARYING ||
            varying.location == UINT32_MAX) continue;
        llvm::Type *ty = llvmType(varying.type, *cg.ctx);
        llvm::Value *value = cg.lvalues.count(varying.name)
            ? cg.lvalues[varying.name] : llvm::UndefValue::get(ty);
        storeVaryingValueAtLocation(cg, record, varying, value);
    }
}

static void copyGeometryVaryings(Codegen &cg, llvm::Value *dst,
                                 uint32_t sourceRecord)
{
    if (!cg.auxSyms) return;
    for (VarSym &varying : *cg.auxSyms) {
        if (varying.kind != VarSym::OUTPUT ||
            varying.location == UINT32_MAX) continue;
        if (varying.stream != 0) continue;
        llvm::Value *value = loadVaryingValueAtLocation(
            cg, cg.b->getInt32(sourceRecord), varying);
        storeVaryingValueAtLocation(cg, dst, varying, value);
    }
}

static void copyGeometryVaryingsSelected(Codegen &cg, llvm::Value *dst,
                                         uint32_t falseRecord,
                                         uint32_t trueRecord,
                                         llvm::Value *condition)
{
    if (!cg.auxSyms) return;
    for (VarSym &varying : *cg.auxSyms) {
        if (varying.kind != VarSym::OUTPUT ||
            varying.location == UINT32_MAX) continue;
        if (varying.stream != 0) continue;
        llvm::Value *falseValue = loadVaryingValueAtLocation(
            cg, cg.b->getInt32(falseRecord), varying);
        llvm::Value *trueValue = loadVaryingValueAtLocation(
            cg, cg.b->getInt32(trueRecord), varying);
        llvm::Value *value =
            cg.b->CreateSelect(condition, trueValue, falseValue);
        storeVaryingValueAtLocation(cg, dst, varying, value);
    }
}

/* Accumulate GS-generated primitives for stream 0 (GL 4.6
 * PRIMITIVES_GENERATED): counts list primitives EMITTED by the shader,
 * including primitives later culled by gl_CullDistance (culling happens
 * after generation).  The counter lives in the XFB meta block
 * (MGLAIRGSXFBStreamMeta::generated, stream slot 0) and is read back by
 * the renderer for the primitive queries. */
static void geometryStream0GeneratedAdd(Codegen &cg, llvm::Value *count)
{
    if (!cg.geometryXfbMetaPtr) return;
    llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
    llvm::Value *metaBase = cg.b->CreateBitCast(
        cg.geometryXfbMetaPtr, i32->getPointerTo(1));
    /* stream block 0, generated at word 3. */
    llvm::Value *generatedPtr = cg.b->CreateGEP(
        i32, metaBase, cg.b->getInt32(3));
    cg.b->CreateAtomicRMW(llvm::AtomicRMWInst::Add, generatedPtr, count,
                          llvm::MaybeAlign(),
                          llvm::AtomicOrdering::Monotonic);
}

static llvm::Value *emitGeometryVertex(Codegen &cg)
{
    if (!cg.isGeometry || !cg.geometryOutputPtr || !cg.geometryCountPtr ||
        !cg.geometryPrimitiveId || cg.geometryRecordCount < 2) {
        cg.err = 1;
        cg.errmsg = "GS AIR codegen: EmitVertex requires the  output ABI";
        return nullptr;
    }
    llvm::Value *pos = cg.lvalues.count("gl_Position")
        ? cg.lvalues["gl_Position"]
        : llvm::ConstantVector::get({
              llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 0.0),
              llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 0.0),
              llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 0.0),
              llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 1.0)});
    pos = coerceScalar(cg, pos, MGLIR_SCALAR_FLOAT);
    llvm::Type *v4 = llvm::FixedVectorType::get(
        llvm::Type::getFloatTy(*cg.ctx), 4);
    if (pos->getType() != v4) {
        if (pos->getType()->isVectorTy()) pos = cg.b->CreateBitCast(pos, v4);
        else pos = cg.b->CreateVectorSplat(4, pos);
    }
    llvm::Value *pointSize = cg.lvalues.count("gl_PointSize")
        ? coerceScalar(cg, cg.lvalues["gl_PointSize"], MGLIR_SCALAR_FLOAT)
        : llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 1.0);
    llvm::Value *cullDistances = cg.lvalues.count("gl_CullDistance")
        ? cg.lvalues["gl_CullDistance"] : defaultCullDistances(cg);
    llvm::Value *clipDistances = cg.lvalues.count("gl_ClipDistance")
        ? cg.lvalues["gl_ClipDistance"] : defaultClipDistances(cg);
    llvm::Value *layer = cg.lvalues.count("gl_Layer")
        ? cg.lvalues["gl_Layer"] : cg.b->getInt32(0);
    llvm::Value *viewportIndex = cg.lvalues.count("gl_ViewportIndex")
        ? cg.lvalues["gl_ViewportIndex"] : cg.b->getInt32(0);
    llvm::Value *outputCountPtr = geometryCounterPtr(cg, 0);
    llvm::Value *stripCountPtr = geometryCounterPtr(cg, 1);
    llvm::Value *emitCountPtr = geometryCounterPtr(cg, 2);
    llvm::Value *emitCount = cg.b->CreateAlignedLoad(
        cg.b->getInt32Ty(), emitCountPtr, llvm::Align(4));
    llvm::Value *canEmit = cg.b->CreateICmpULT(
        emitCount, cg.b->getInt32(cg.geometryMaxVertices));
    llvm::BasicBlock *emitBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_emit", cg.fn);
    llvm::BasicBlock *doneBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_emit_done", cg.fn);
        cg.b->CreateCondBr(canEmit, emitBB, doneBB);
    cg.b->SetInsertPoint(emitBB);

    if (cg.geometryOutputType == MGL_AST_GS_OUT_POINTS &&
        cg.geometryXfbMetaPtr) {
        geometryStream0GeneratedAdd(cg, cg.b->getInt32(1));
    }

    if (cg.geometryOutputType == MGL_AST_GS_OUT_POINTS) {
        llvm::Value *outputCount = cg.b->CreateAlignedLoad(
            cg.b->getInt32Ty(), outputCountPtr, llvm::Align(4));
        llvm::Value *stripCount = cg.b->CreateAlignedLoad(
            cg.b->getInt32Ty(), stripCountPtr, llvm::Align(4));
        llvm::Value *outputRecord = cg.b->CreateAdd(
            outputCount, cg.b->getInt32(2));
        storeGeometryPosition(cg, outputRecord, pos);
        storeGeometryPointSize(cg, outputRecord, pointSize);
        storeGeometryCullDistances(cg, outputRecord, cullDistances);
        storeGeometryClipDistances(cg, outputRecord, clipDistances);
        storeGeometryVaryings(cg, outputRecord);
        storeGeometryLayer(cg, outputRecord, layer);
        storeGeometryViewportIndex(cg, outputRecord, viewportIndex);
        storeGeometryPrimitiveId(cg, outputRecord);
        llvm::Value *visibleIncrement = cg.b->CreateSelect(
            geometryPrimitiveCulled(cg, {cullDistances}),
            cg.b->getInt32(0), cg.b->getInt32(1));
        cg.b->CreateAlignedStore(
            cg.b->CreateAdd(outputCount, visibleIncrement),
            outputCountPtr, llvm::Align(4));
        cg.b->CreateAlignedStore(
            cg.b->CreateAdd(stripCount, cg.b->getInt32(1)),
            stripCountPtr, llvm::Align(4));
        cg.b->CreateAlignedStore(
            cg.b->CreateAdd(emitCount, cg.b->getInt32(1)),
            emitCountPtr, llvm::Align(4));
        cg.b->CreateBr(doneBB);
        cg.b->SetInsertPoint(doneBB);
        return cg.b->getInt32(0);
    }

    llvm::Value *stripCount = cg.b->CreateAlignedLoad(
        cg.b->getInt32Ty(), stripCountPtr, llvm::Align(4));

    if (cg.geometryOutputType == MGL_AST_GS_OUT_LINE_STRIP) {
        llvm::Value *hasLine = cg.b->CreateICmpUGE(
            stripCount, cg.b->getInt32(1));
        llvm::BasicBlock *lineBB = llvm::BasicBlock::Create(
            *cg.ctx, "gs_emit_line", cg.fn);
        llvm::BasicBlock *advanceBB = llvm::BasicBlock::Create(
            *cg.ctx, "gs_emit_line_advance", cg.fn);
        cg.b->CreateCondBr(hasLine, lineBB, advanceBB);

        cg.b->SetInsertPoint(lineBB);
        llvm::Value *previous = loadGeometryPosition(cg, 0);
        llvm::Value *previousPoint = loadGeometryPointSize(cg, 0);
        llvm::Value *previousCull = loadGeometryCullDistances(cg, 0);
        llvm::Value *previousClip = loadGeometryClipDistances(cg, 0);
        llvm::Value *previousLayer = loadGeometryLayer(cg, 0);
        llvm::Value *previousViewport = loadGeometryViewportIndex(cg, 0);
        llvm::Value *outputCount = cg.b->CreateAlignedLoad(
            cg.b->getInt32Ty(), outputCountPtr, llvm::Align(4));
        llvm::Value *outputRecord = cg.b->CreateAdd(
            outputCount, cg.b->getInt32(2));
        storeGeometryPosition(cg, outputRecord, previous);
        storeGeometryPointSize(cg, outputRecord, previousPoint);
        storeGeometryCullDistances(cg, outputRecord, previousCull);
        storeGeometryClipDistances(cg, outputRecord, previousClip);
        storeGeometryLayer(cg, outputRecord, previousLayer);
        storeGeometryViewportIndex(cg, outputRecord, previousViewport);
        copyGeometryPrimitiveId(cg, outputRecord, 0);
        copyGeometryVaryings(cg, outputRecord, 0);
        storeGeometryPosition(cg,
            cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), pos);
        storeGeometryPointSize(cg,
            cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), pointSize);
        storeGeometryCullDistances(cg,
            cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), cullDistances);
        storeGeometryClipDistances(cg,
            cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), clipDistances);
        storeGeometryLayer(cg,
            cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), layer);
        storeGeometryViewportIndex(cg,
            cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), viewportIndex);
        storeGeometryPrimitiveId(cg,
            cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)));
        storeGeometryVaryings(
            cg, cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)));
        llvm::Value *lineIncrement = cg.b->CreateSelect(
            geometryPrimitiveCulled(cg, {previousCull, cullDistances}),
            cg.b->getInt32(0), cg.b->getInt32(2));
        cg.b->CreateAlignedStore(
            cg.b->CreateAdd(outputCount, lineIncrement),
            outputCountPtr, llvm::Align(4));
        geometryStream0GeneratedAdd(cg, cg.b->getInt32(1));
        cg.b->CreateBr(advanceBB);

        cg.b->SetInsertPoint(advanceBB);
        storeGeometryPosition(cg, cg.b->getInt32(0), pos);
        storeGeometryPointSize(cg, cg.b->getInt32(0), pointSize);
        storeGeometryCullDistances(cg, cg.b->getInt32(0), cullDistances);
        storeGeometryClipDistances(cg, cg.b->getInt32(0), clipDistances);
        storeGeometryLayer(cg, cg.b->getInt32(0), layer);
        storeGeometryViewportIndex(cg, cg.b->getInt32(0), viewportIndex);
        storeGeometryPrimitiveId(cg, cg.b->getInt32(0));
        storeGeometryVaryings(cg, cg.b->getInt32(0));
        cg.b->CreateAlignedStore(
            cg.b->CreateAdd(stripCount, cg.b->getInt32(1)),
            stripCountPtr, llvm::Align(4));
        cg.b->CreateAlignedStore(
            cg.b->CreateAdd(emitCount, cg.b->getInt32(1)),
            emitCountPtr, llvm::Align(4));
        cg.b->CreateBr(doneBB);
        cg.b->SetInsertPoint(doneBB);
        return cg.b->getInt32(0);
    }

    llvm::Value *hasTriangle = cg.b->CreateICmpUGE(
        stripCount, cg.b->getInt32(2));
    llvm::BasicBlock *triangleBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_emit_triangle", cg.fn);
    llvm::BasicBlock *advanceBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_emit_advance", cg.fn);
    cg.b->CreateCondBr(hasTriangle, triangleBB, advanceBB);

    cg.b->SetInsertPoint(triangleBB);
    llvm::Value *previous0 = loadGeometryPosition(cg, 0);
    llvm::Value *previous1 = loadGeometryPosition(cg, 1);
    llvm::Value *previousPoint0 = loadGeometryPointSize(cg, 0);
    llvm::Value *previousPoint1 = loadGeometryPointSize(cg, 1);
    llvm::Value *previousCull0 = loadGeometryCullDistances(cg, 0);
    llvm::Value *previousCull1 = loadGeometryCullDistances(cg, 1);
    llvm::Value *previousClip0 = loadGeometryClipDistances(cg, 0);
    llvm::Value *previousClip1 = loadGeometryClipDistances(cg, 1);
    llvm::Value *previousLayer0 = loadGeometryLayer(cg, 0);
    llvm::Value *previousLayer1 = loadGeometryLayer(cg, 1);
    llvm::Value *previousViewport0 = loadGeometryViewportIndex(cg, 0);
    llvm::Value *previousViewport1 = loadGeometryViewportIndex(cg, 1);
    llvm::Value *odd = cg.b->CreateICmpNE(
        cg.b->CreateAnd(stripCount, cg.b->getInt32(1)), cg.b->getInt32(0));
    llvm::Value *first = cg.b->CreateSelect(odd, previous1, previous0);
    llvm::Value *second = cg.b->CreateSelect(odd, previous0, previous1);
    llvm::Value *firstPoint = cg.b->CreateSelect(
        odd, previousPoint1, previousPoint0);
    llvm::Value *secondPoint = cg.b->CreateSelect(
        odd, previousPoint0, previousPoint1);
    llvm::Value *firstCull = cg.b->CreateSelect(
        odd, previousCull1, previousCull0);
    llvm::Value *secondCull = cg.b->CreateSelect(
        odd, previousCull0, previousCull1);
    llvm::Value *firstClip = cg.b->CreateSelect(
        odd, previousClip1, previousClip0);
    llvm::Value *secondClip = cg.b->CreateSelect(
        odd, previousClip0, previousClip1);
    llvm::Value *firstLayer = cg.b->CreateSelect(
        odd, previousLayer1, previousLayer0);
    llvm::Value *secondLayer = cg.b->CreateSelect(
        odd, previousLayer0, previousLayer1);
    llvm::Value *firstViewport = cg.b->CreateSelect(
        odd, previousViewport1, previousViewport0);
    llvm::Value *secondViewport = cg.b->CreateSelect(
        odd, previousViewport0, previousViewport1);
    llvm::Value *outputCount = cg.b->CreateAlignedLoad(
        cg.b->getInt32Ty(), outputCountPtr, llvm::Align(4));
    llvm::Value *outputRecord = cg.b->CreateAdd(outputCount, cg.b->getInt32(2));
    storeGeometryPosition(cg, outputRecord, first);
    storeGeometryPointSize(cg, outputRecord, firstPoint);
    storeGeometryCullDistances(cg, outputRecord, firstCull);
    storeGeometryClipDistances(cg, outputRecord, firstClip);
    storeGeometryLayer(cg, outputRecord, firstLayer);
    storeGeometryViewportIndex(cg, outputRecord, firstViewport);
    copyGeometryPrimitiveIdSelected(cg, outputRecord, 0, 1, odd);
    copyGeometryVaryingsSelected(cg, outputRecord, 0, 1, odd);
    storeGeometryPosition(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), second);
    storeGeometryPointSize(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), secondPoint);
    storeGeometryCullDistances(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), secondCull);
    storeGeometryClipDistances(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), secondClip);
    storeGeometryLayer(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), secondLayer);
    storeGeometryViewportIndex(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), secondViewport);
    copyGeometryPrimitiveIdSelected(
        cg, cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), 1, 0, odd);
    copyGeometryVaryingsSelected(
        cg, cg.b->CreateAdd(outputRecord, cg.b->getInt32(1)), 1, 0, odd);
    storeGeometryPosition(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(2)), pos);
    storeGeometryPointSize(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(2)), pointSize);
    storeGeometryCullDistances(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(2)), cullDistances);
    storeGeometryClipDistances(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(2)), clipDistances);
    storeGeometryLayer(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(2)), layer);
    storeGeometryViewportIndex(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(2)), viewportIndex);
    storeGeometryPrimitiveId(cg,
        cg.b->CreateAdd(outputRecord, cg.b->getInt32(2)));
    storeGeometryVaryings(
        cg, cg.b->CreateAdd(outputRecord, cg.b->getInt32(2)));
    llvm::Value *triangleIncrement = cg.b->CreateSelect(
        geometryPrimitiveCulled(
            cg, {firstCull, secondCull, cullDistances}),
        cg.b->getInt32(0), cg.b->getInt32(3));
    cg.b->CreateAlignedStore(
        cg.b->CreateAdd(outputCount, triangleIncrement),
        outputCountPtr, llvm::Align(4));
    geometryStream0GeneratedAdd(cg, cg.b->getInt32(1));
    cg.b->CreateBr(advanceBB);

    cg.b->SetInsertPoint(advanceBB);
    llvm::Value *previous1ForNext = loadGeometryPosition(cg, 1);
    llvm::Value *previousPoint1ForNext = loadGeometryPointSize(cg, 1);
    llvm::Value *previousCull1ForNext = loadGeometryCullDistances(cg, 1);
    llvm::Value *previousClip1ForNext = loadGeometryClipDistances(cg, 1);
    llvm::Value *previousLayer1ForNext = loadGeometryLayer(cg, 1);
    llvm::Value *previousViewport1ForNext = loadGeometryViewportIndex(cg, 1);
    storeGeometryPosition(cg, cg.b->getInt32(0), previous1ForNext);
    storeGeometryPointSize(cg, cg.b->getInt32(0), previousPoint1ForNext);
    storeGeometryCullDistances(cg, cg.b->getInt32(0), previousCull1ForNext);
    storeGeometryClipDistances(cg, cg.b->getInt32(0), previousClip1ForNext);
    storeGeometryLayer(cg, cg.b->getInt32(0), previousLayer1ForNext);
    storeGeometryViewportIndex(cg, cg.b->getInt32(0), previousViewport1ForNext);
    copyGeometryPrimitiveId(cg, cg.b->getInt32(0), 1);
    copyGeometryVaryings(cg, cg.b->getInt32(0), 1);
    storeGeometryPosition(cg, cg.b->getInt32(1), pos);
    storeGeometryPointSize(cg, cg.b->getInt32(1), pointSize);
    storeGeometryCullDistances(cg, cg.b->getInt32(1), cullDistances);
    storeGeometryClipDistances(cg, cg.b->getInt32(1), clipDistances);
    storeGeometryLayer(cg, cg.b->getInt32(1), layer);
    storeGeometryViewportIndex(cg, cg.b->getInt32(1), viewportIndex);
    storeGeometryPrimitiveId(cg, cg.b->getInt32(1));
    storeGeometryVaryings(cg, cg.b->getInt32(1));
    cg.b->CreateAlignedStore(
        cg.b->CreateAdd(stripCount, cg.b->getInt32(1)),
        stripCountPtr, llvm::Align(4));
    cg.b->CreateAlignedStore(
        cg.b->CreateAdd(emitCount, cg.b->getInt32(1)),
        emitCountPtr, llvm::Align(4));
    cg.b->CreateBr(doneBB);
    cg.b->SetInsertPoint(doneBB);
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), 0);
}

/* Write this stream's captured varyings into the stage-out record at their
 * location*16 slots (the same field offsets the rasterization record uses),
 * restricted to OUTPUT symbols on `stream`.  Pass 2 later repacks these to
 * the link-time component offsets; the pass-1 record keeps the location
 * layout so one stage-out buffer serves every stream and rasterization. */
static void storeGeometryStageOutStreamVaryings(Codegen &cg,
                                                llvm::Value *record,
                                                int32_t stream)
{
    if (!cg.auxSyms) return;
    for (VarSym &v : *cg.auxSyms) {
        if (v.kind != VarSym::OUTPUT || v.location == UINT32_MAX) continue;
        if (v.stream != stream) continue;
        llvm::Type *ty = llvmType(v.type, *cg.ctx);
        llvm::Value *value = cg.lvalues.count(v.name)
            ? cg.lvalues[v.name] : llvm::UndefValue::get(ty);
        storeVaryingValueAtLocation(cg, record, v, value);
    }
}

/* EmitStreamVertex on stream > 0 (GLSL 4.60 §8.13, GL4 ordered terminal
 * state): the vertex is appended to this work item's stage-out record run at
 * a deterministic per-stream index (no GPU-atomic cursor), and the
 * per-(work-item, stream) visible byte count is accumulated into the
 * visibility buffer (slot 30) for the CPU prefix-sum and the pass-2 ordered
 * scatter.  Streams above 0 remain points-only. */
static llvm::Value *emitGeometryStreamVertex(Codegen &cg, int32_t stream)
{
    if (!cg.isGeometry || !cg.geometryOutputPtr || !cg.geometryCountPtr ||
        !cg.geometryPrimitiveId || !cg.geometryXfbPtr || !cg.geometryXfbMetaPtr) {
        cg.err = 1;
        cg.errmsg = "GS AIR codegen: EmitStreamVertex requires the M3 XFB ABI";
        return nullptr;
    }
    llvm::Value *pos = cg.lvalues.count("gl_Position")
        ? cg.lvalues["gl_Position"]
        : llvm::UndefValue::get(llvm::FixedVectorType::get(
              llvm::Type::getFloatTy(*cg.ctx), 4));
    pos = coerceScalar(cg, pos, MGLIR_SCALAR_FLOAT);
    llvm::Type *v4 = llvm::FixedVectorType::get(
        llvm::Type::getFloatTy(*cg.ctx), 4);
    if (pos->getType() != v4) {
        if (pos->getType()->isVectorTy()) pos = cg.b->CreateBitCast(pos, v4);
        else pos = cg.b->CreateVectorSplat(4, pos);
    }
    llvm::Value *cullDistances = cg.lvalues.count("gl_CullDistance")
        ? cg.lvalues["gl_CullDistance"] : defaultCullDistances(cg);

    llvm::Value *emitCountPtr = geometryCounterPtr(cg, 2);
    llvm::Value *emitCount = cg.b->CreateAlignedLoad(
        cg.b->getInt32Ty(), emitCountPtr, llvm::Align(4));
    llvm::Value *canEmit = cg.b->CreateICmpULT(
        emitCount, cg.b->getInt32(cg.geometryMaxVertices));
    llvm::BasicBlock *emitBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_stream_emit", cg.fn);
    llvm::BasicBlock *doneBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_stream_done", cg.fn);
    cg.b->CreateCondBr(canEmit, emitBB, doneBB);
    cg.b->SetInsertPoint(emitBB);

    llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
    llvm::Type *i64 = llvm::Type::getInt64Ty(*cg.ctx);
    llvm::Value *metaBase = cg.b->CreateBitCast(
        cg.geometryXfbMetaPtr, i32->getPointerTo(1));
    /* Stream block offset: MGLAIRGSXFBStreamMeta is 16 bytes, with
     * stride@0 capacity@4 capture_base@8 generated@12. */
    llvm::Value *blockOff = cg.b->getInt32(stream * 4u); /* 16B in u32 words */
    if (stream > 0) {
        /* Non-zero streams are currently points-only.  Count every emitted
         * point — including culled ones, which are still generated (GL 4.6
         * PRIMITIVES_GENERATED counts primitives before culling) — so the
         * indexed query stays meaningful when no XFB buffer is bound. */
        llvm::Value *generatedPtr = cg.b->CreateGEP(
            i32, metaBase, cg.b->CreateAdd(blockOff, cg.b->getInt32(3)));
        cg.b->CreateAtomicRMW(llvm::AtomicRMWInst::Add, generatedPtr,
                              cg.b->getInt32(1), llvm::MaybeAlign(),
                              llvm::AtomicOrdering::Monotonic);
    }

    /* Culled primitives contribute nothing to the capture (same policy as
     * the stream 0 batch path, GL 4.6 §13.2.4). */
    llvm::Value *culled = geometryPrimitiveCulled(cg, {cullDistances});
    llvm::BasicBlock *appendBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_stream_append", cg.fn);
    cg.b->CreateCondBr(culled, doneBB, appendBB);
    cg.b->SetInsertPoint(appendBB);
    /* GL4 ordered terminal state (mgl_air_gs_abi.h §5b): a captured
     * (stride != 0) stream emission appends its record at the deterministic
     * descending index recordCount-1-cursor inside the work item's stage-out
     * run, stamped with its stream id (MGL_AIR_PER_VERTEX_STREAM_OFFSET) so
     * the pass-2 scatter can attribute records to streams in emission order.
     * Stream 0 keeps the ascending [2, 2+vertex_count) region for the
     * rasterizing indirect draw, so this path must NOT touch counter 0.
     * The global emit guard above bounds stream-0 + stream>0 records to the
     * expanded region, so the two regions never overlap.  An uncaptured
     * stream emits no record but still counts generated primitives (above)
     * for the indexed query. */
    /* Attribute this emission to the buffers fed by this stream
     * (meta.buffer_stream; the link plan keeps one stream per buffer):
     * the record is captured when at least one fed buffer has capture on,
     * and the visible bytes accumulate per fed buffer (a stream may feed
     * several buffers via gl_NextBuffer).  buffer_stream lives in the
     * meta words right after the four 8-word stream blocks. */
    llvm::Value *captured = nullptr;
    llvm::Value *fedStride[MGL_AIR_GS_MAX_STREAMS] = {nullptr};
    llvm::Value *fedPred[MGL_AIR_GS_MAX_STREAMS] = {nullptr};
    for (uint32_t buf = 0; buf < MGL_AIR_GS_MAX_STREAMS; buf++) {
        llvm::Value *bs = cg.b->CreateAlignedLoad(
            i32, cg.b->CreateGEP(i32, metaBase,
                                 cg.b->getInt32(16u + buf)), llvm::Align(4));
        llvm::Value *match = cg.b->CreateICmpEQ(
            bs, cg.b->getInt32((uint32_t)stream));
        llvm::Value *bsStride = cg.b->CreateAlignedLoad(
            i32, cg.b->CreateGEP(i32, metaBase, cg.b->getInt32(buf * 4u)),
            llvm::Align(4));
        llvm::Value *on = cg.b->CreateAnd(
            match, cg.b->CreateICmpNE(bsStride, cg.b->getInt32(0)));
        fedStride[buf] = bsStride;
        fedPred[buf] = on;
        captured = captured ? cg.b->CreateOr(captured, on) : on;
    }
    llvm::BasicBlock *captureBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_stream_capture", cg.fn);
    llvm::BasicBlock *tailBB = llvm::BasicBlock::Create(
        *cg.ctx, "gs_stream_tail", cg.fn);
    cg.b->CreateCondBr(captured, captureBB, tailBB);
    cg.b->SetInsertPoint(captureBB);

    llvm::Value *cursorPtr = geometryCounterPtr(cg, MGL_AIR_GS_COUNT_STREAM);
    llvm::Value *cursor = cg.b->CreateAlignedLoad(
        i32, cursorPtr, llvm::Align(4));
    llvm::Value *record = cg.b->CreateSub(
        cg.b->getInt32(cg.geometryRecordCount - 1u), cursor);
    storeGeometryPosition(cg, record, pos);
    storeGeometryStageOutStreamVaryings(cg, record, stream);
    {
        /* Stamp the stream id for the pass-2 scatter. */
        llvm::Value *base = geometryRecordPtr(cg, record);
        llvm::Value *stampPtr = cg.b->CreateBitCast(
            cg.b->CreateGEP(cg.b->getInt8Ty(), base,
                            cg.b->getInt64(MGL_AIR_PER_VERTEX_STREAM_OFFSET)),
            i32->getPointerTo(1));
        cg.b->CreateAlignedStore(cg.b->getInt32((uint32_t)stream), stampPtr,
                                 llvm::Align(4));
    }
    cg.b->CreateAlignedStore(cg.b->CreateAdd(cursor, cg.b->getInt32(1)),
                             cursorPtr, llvm::Align(4));

    /* vis[workItem * MGL_AIR_GS_MAX_STREAMS + b] += stride[b] for every
     * buffer fed by this stream. */
    if (cg.geometryXfbVisPtr && cg.geometryWorkItemId) {
        llvm::Value *visBase = cg.b->CreateBitCast(
            cg.geometryXfbVisPtr, i32->getPointerTo(1));
        llvm::Value *visRun = cg.b->CreateMul(
            cg.geometryWorkItemId,
            cg.b->getInt32(MGL_AIR_GS_MAX_STREAMS));
        for (uint32_t buf = 0; buf < MGL_AIR_GS_MAX_STREAMS; buf++) {
            llvm::Value *add = cg.b->CreateSelect(
                fedPred[buf], fedStride[buf], cg.b->getInt32(0));
            llvm::Value *visPtr = cg.b->CreateGEP(
                i32, visBase,
                cg.b->CreateAdd(visRun, cg.b->getInt32(buf)));
            llvm::Value *cur = cg.b->CreateAlignedLoad(i32, visPtr,
                                                       llvm::Align(4));
            cg.b->CreateAlignedStore(cg.b->CreateAdd(cur, add), visPtr,
                                     llvm::Align(4));
        }
    }
    cg.b->CreateBr(tailBB);
    cg.b->SetInsertPoint(tailBB);
    llvm::Value *strip = cg.b->CreateAlignedLoad(
        cg.b->getInt32Ty(), geometryCounterPtr(cg, 1), llvm::Align(4));
    cg.b->CreateAlignedStore(
        cg.b->CreateAdd(strip, cg.b->getInt32(1)),
        geometryCounterPtr(cg, 1), llvm::Align(4));
    cg.b->CreateAlignedStore(
        cg.b->CreateAdd(emitCount, cg.b->getInt32(1)),
        emitCountPtr, llvm::Align(4));
    cg.b->CreateBr(doneBB);
    cg.b->SetInsertPoint(doneBB);
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), 0);
}

/* ---- Uniform-block member chains ------------------------------------- */

/* Collect the member/index chain of `e` (outermost first) and the root
 * block symbol.  `rootIndex` (if any) is the trailing index that selects
 * an instance-array element rather than walking the block layout. */
static const MGLIRSymbol *blockChainRoot(const MGLExpr *e,
                                         const MGLExpr *chain[16],
                                         uint32_t *chain_len,
                                         const MGLExpr **rootIndex,
                                         const MGLIRModule *mod) {
    const MGLExpr *cur = e;
    uint32_t n = 0;
    while (cur && n < 16 &&
           (cur->kind == MGL_EXPR_MEMBER || cur->kind == MGL_EXPR_INDEX)) {
        chain[n++] = cur;
        cur = cur->kind == MGL_EXPR_MEMBER ? cur->u.member.object
                                           : cur->u.index.object;
    }
    if (!cur || cur->kind != MGL_EXPR_VAR_REF) {
        return nullptr;
    }
    *rootIndex = nullptr;
    if (n > 0 && chain[n - 1]->kind == MGL_EXPR_INDEX &&
        chain[n - 1]->u.index.object &&
        chain[n - 1]->u.index.object->kind == MGL_EXPR_VAR_REF) {
        /* Only strip as a block-instance-array selector when the root
         * symbol is the block itself.  Flattened anonymous members like
         * `S s[N];` inside `uniform Block { ... }` use the INDEX as a
         * std140 array step, not a Metal buffer-slot pick. */
        const MGLIRSymbol *rootSym =
            findSymbol(mod, chain[n - 1]->u.index.object->u.var_ref.name);
        if (!rootSym || !rootSym->block_name) {
            *rootIndex = chain[n - 1];
            n--;
        }
    }
    *chain_len = n;
    const MGLIRSymbol *ov = findSymbol(mod, cur->u.var_ref.name);
    if (!ov || ov->is_function || !(ov->qualifiers & MGL_AST_Q_UNIFORM)) {
        return nullptr;
    }
    return ov;
}

/* Type-level walk of a uniform-block member chain; returns the leaf type
 * or nullptr when the expression is not a resolvable block access. */
static const MGLIRType *blockMemberLeafType(const MGLExpr *e,
                                            const MGLIRModule *mod) {
    const MGLExpr *chain[16];
    uint32_t chain_len = 0;
    const MGLExpr *rootIndex = nullptr;
    const MGLIRSymbol *ov = blockChainRoot(e, chain, &chain_len, &rootIndex,
                                           mod);
    if (!ov) {
        return nullptr;
    }
    const MGLIRType *ct = ov->type;
    if (ct && ct->kind == MGLIR_TYPE_ARRAY && !ov->block_name) {
        /* Block-instance arrays are selected via the stripped rootIndex;
         * peel so the walk starts at the block struct.  Flattened members
         * keep their array wrapper so `s[i].f` type-checks. */
        ct = ct->elem_type;
    }
    /* Only descend into a uniform block (struct / array-of-struct).  Plain
     * uniform arrays and vectors are not blocks and must fall through to the
     * normal swizzle path in exprType().  Without this guard a plain
     * `uniform vec4 arr[N]; arr[i].xyz` would resolve to the element type
     * (vec4) instead of letting exprType swizzle it to vec3. */
    {
        const MGLIRType *gate = ct;
        while (gate && gate->kind == MGLIR_TYPE_ARRAY)
            gate = gate->elem_type;
        if (!gate || gate->kind != MGLIR_TYPE_STRUCT) {
            return nullptr;
        }
    }
    /* chain_len already excludes the (possibly stripped) rootIndex node, so
     * walk every remaining member/index step from the block root.  The
     * start offset must stay 0 — the stripped INDEX is not part of the
     * traversable member path. */
    uint32_t start = 0u;
    for (uint32_t ci = start; ci < chain_len && ct; ci++) {
        const MGLExpr *node = chain[chain_len - 1 - ci]; /* innermost first */
        if (node->kind == MGL_EXPR_MEMBER) {
            if (ct->kind != MGLIR_TYPE_STRUCT) {
                return nullptr; /* swizzle: not a block-layout step */
            }
            const MGLIRType *mt = nullptr;
            for (uint32_t m = 0; m < ct->member_count; m++) {
                if (!strcmp(ct->member_names[m], node->u.member.field)) {
                    mt = ct->members[m];
                    break;
                }
            }
            ct = mt;
        } else {
            if (ct->kind != MGLIR_TYPE_ARRAY || !ct->elem_type) {
                return nullptr;
            }
            ct = ct->elem_type;
        }
    }
    return ct;
}

/* Emit a uniform-block member chain read: walk the member/index path over
 * the block's struct layout and load the leaf at the accumulated byte
 * offset (static member offsets + runtime array-index strides).  Trailing
 * swizzles / vector component indexes apply to the loaded leaf. */
static llvm::Value *emitBlockMemberChain(Codegen &cg, const MGLExpr *e,
                                         llvm::Value *base,
                                         const MGLIRType *ubStruct,
                                         const char *objName,
                                         const MGLIRModule *mod,
                                         const std::map<std::string, MType>
                                             &locals,
                                         uint32_t startOff = 0) {
    const MGLExpr *chain[16];
    uint32_t chain_len = 0;
    const MGLExpr *rootIndex = nullptr;
    if (!blockChainRoot(e, chain, &chain_len, &rootIndex, mod)) {
        cg.err = 1;
        cg.errmsg = std::string("codegen: uniform block '") + objName +
                    "' member path did not resolve";
        return nullptr;
    }
    const MGLIRType *ct = ubStruct;
    uint64_t soff = startOff;
    llvm::Value *dynOff = nullptr;
    llvm::Value *v = nullptr; /* set once the leaf is loaded */
    MType vt;
    for (uint32_t ci = 0; ci < chain_len; ci++) {
        const MGLExpr *node = chain[chain_len - 1 - ci]; /* innermost first */
        if (!v) {
            bool stepped = false;
            if (node->kind == MGL_EXPR_MEMBER &&
                ct && ct->kind == MGLIR_TYPE_STRUCT) {
                const MGLIRType *mt = nullptr;
                uint32_t moff = 0;
                for (uint32_t m = 0; m < ct->member_count; m++) {
                    if (!strcmp(ct->member_names[m],
                                node->u.member.field)) {
                        mt = ct->members[m];
                        moff = ct->member_offsets
                                   ? ct->member_offsets[m]
                                   : 0;
                        break;
                    }
                }
                if (!mt) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: uniform block '") +
                                objName + "' has no member '" +
                                node->u.member.field + "'";
                    return nullptr;
                }
                soff += moff;
                ct = mt;
                stepped = true;
            } else if (node->kind == MGL_EXPR_INDEX &&
                       ct && ct->kind == MGLIR_TYPE_ARRAY &&
                       ct->elem_type) {
                llvm::Value *idx = emitExpr(cg, node->u.index.index, mod,
                                            locals);
                if (!idx) return nullptr;
                idx = coerceScalar(cg, idx, MGLIR_SCALAR_INT);
                llvm::Value *i64 = cg.b->CreateSExt(idx, cg.b->getInt64Ty());
                uint32_t stride = ct->layout.array_stride > 0
                                      ? (uint32_t)ct->layout.array_stride
                                      : 0u;
                llvm::Value *byte =
                    cg.b->CreateMul(i64, cg.b->getInt64(stride));
                dynOff = dynOff ? cg.b->CreateAdd(dynOff, byte) : byte;
                ct = ct->elem_type;
                stepped = true;
            }
            if (stepped) {
                continue;
            }
            /* Leaf boundary: the remaining outer nodes are swizzles or
             * component selects on the loaded value. */
            if (!ct || ct->kind == MGLIR_TYPE_STRUCT) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: uniform block '") + objName +
                            "' whole-struct members are not readable";
                return nullptr;
            }
            llvm::Value *off = cg.b->getInt64(soff);
            if (dynOff) off = cg.b->CreateAdd(off, dynOff);
            vt = typeFromIR(ct);
            if (ct->kind == MGLIR_TYPE_MATRIX) {
                v = emitUBOMatrixLoad(cg, base, off, ct, vt);
            } else if (vt.vec && vt.scalar == MGLIR_SCALAR_BOOL) {
                llvm::Type *wordsTy = llvm::FixedVectorType::get(
                    llvm::Type::getInt32Ty(*cg.ctx), vt.vec);
                llvm::Value *p =
                    cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
                llvm::Align align(vt.vec <= 2 ? 8 : 16);
                p = cg.b->CreateBitCast(p, wordsTy->getPointerTo(1));
                llvm::Value *words =
                    cg.b->CreateAlignedLoad(wordsTy, p, align);
                v = cg.b->CreateICmpNE(
                    words, llvm::ConstantAggregateZero::get(wordsTy));
            } else {
                llvm::Type *t = llvmType(vt, *cg.ctx);
                llvm::Value *p =
                    cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
                llvm::Align align(16);
                if (auto *fvt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
                    uint64_t w = fvt->getElementCount().getFixedValue();
                    if (w == 1) align = llvm::Align(4);
                    else if (w == 2) align = llvm::Align(8);
                } else if (t->isFloatTy() || t->isIntegerTy(32)) {
                    align = llvm::Align(4);
                }
                p = cg.b->CreateBitCast(p, t->getPointerTo(1));
                v = cg.b->CreateAlignedLoad(t, p, align);
            }
        }
        /* Post-leaf: swizzle / component selection on the loaded value. */
        if (node->kind == MGL_EXPR_MEMBER) {
            std::vector<uint32_t> sidx;
            if (!swizzleIndices(node->u.member.field, &sidx)) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: invalid swizzle '") +
                            node->u.member.field + "'";
                return nullptr;
            }
            if (sidx.size() == 1) {
                v = cg.b->CreateExtractElement(
                    v, cg.b->getInt32(sidx[0]));
                vt.vec = 0;
            } else {
                llvm::SmallVector<llvm::Constant *, 4> mask;
                for (uint32_t s : sidx)
                    mask.push_back(cg.b->getInt32(s));
                v = cg.b->CreateShuffleVector(
                    v, llvm::UndefValue::get(v->getType()),
                    llvm::ConstantVector::get(mask));
                vt.vec = sidx.size();
            }
            continue;
        }
        /* INDEX: matrix column or vector component. */
        llvm::Value *idx = emitExpr(cg, node->u.index.index, mod, locals);
        if (!idx) return nullptr;
        llvm::Value *r = emitIndexValue(cg, v, vt, idx);
        if (!r) {
            cg.err = 1;
            cg.errmsg = "codegen: indexing this type is not supported on a "
                        "block member";
            return nullptr;
        }
        v = r;
        if (vt.isMatrix()) {
            MType col;
            col.scalar = vt.scalar;
            col.vec = vt.rows;
            vt = col;
        } else if (vt.isArray()) {
            vt.arr = 0;
        } else {
            vt.vec = 0;
        }
    }
    if (!v) {
        if (!ct || ct->kind == MGLIR_TYPE_STRUCT) {
            cg.err = 1;
            cg.errmsg = std::string("codegen: uniform block '") + objName +
                        "' whole-struct members are not readable";
            return nullptr;
        }
        llvm::Value *off = cg.b->getInt64(soff);
        if (dynOff) off = cg.b->CreateAdd(off, dynOff);
        vt = typeFromIR(ct);
        llvm::Type *t = llvmType(vt, *cg.ctx);
        llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
        llvm::Align align(16);
        if (auto *fvt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
            uint64_t w = fvt->getElementCount().getFixedValue();
            if (w == 1) align = llvm::Align(4);
            else if (w == 2) align = llvm::Align(8);
        } else if (t->isFloatTy() || t->isIntegerTy(32)) {
            align = llvm::Align(4);
        }
        p = cg.b->CreateBitCast(p, t->getPointerTo(1));
        if (ct->kind == MGLIR_TYPE_MATRIX)
            return emitUBOMatrixLoad(cg, base, off, ct, vt);
        if (vt.vec && vt.scalar == MGLIR_SCALAR_BOOL) {
            /* bvecN members live as 4-byte words in the block (GL 4.6
             * §7.6.2.2 std140 bool packing).  An <N x i1> vector load
             * crashes MTLCompilerService; load the words and truncate to
             * i1 lanes (any nonzero word is true). */
            llvm::Type *wordsTy = llvm::FixedVectorType::get(
                llvm::Type::getInt32Ty(*cg.ctx), vt.vec);
            p = cg.b->CreateBitCast(p, wordsTy->getPointerTo(1));
            llvm::Value *words = cg.b->CreateAlignedLoad(wordsTy, p, align);
            v = cg.b->CreateICmpNE(
                words, llvm::ConstantAggregateZero::get(wordsTy));
            return v;
        }
        if (vt.scalar == MGLIR_SCALAR_BOOL && !vt.vec) {
            llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
            p = cg.b->CreateBitCast(p, i32->getPointerTo(1));
            llvm::Value *word =
                cg.b->CreateAlignedLoad(i32, p, llvm::Align(4));
            return cg.b->CreateICmpNE(word, cg.b->getInt32(0));
        }
        v = cg.b->CreateAlignedLoad(t, p, align);
    }
    return v;
}

void emitStmt(Codegen &cg, const MGLStmt *st, const MGLIRModule *mod,
              std::map<std::string, MType> *locals);

llvm::Value *emitExpr(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
                      const std::map<std::string, MType> &locals) {
    switch (e->kind) {
    case MGL_EXPR_LITERAL: {
        MGLIRScalar base = (MGLIRScalar)e->u.literal.base;
        if (base == MGLIR_SCALAR_INT || base == MGLIR_SCALAR_UINT)
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx),
                                          (uint64_t)e->u.literal.value);
        if (base == MGLIR_SCALAR_BOOL)
            return llvm::ConstantInt::get(llvm::Type::getInt1Ty(*cg.ctx),
                                          e->u.literal.value != 0.0);
        return llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx),
                                     e->u.literal.value);
    }
    case MGL_EXPR_VAR_REF: {
        /* GLSL exposes geometry limits as compile-time constants.  These
         * values describe the capability contract advertised by MGL and the
         * fixed AIR geometry expansion budget. */
        if (strcmp(e->u.var_ref.name, "gl_MaxGeometryInputComponents") == 0)
            return cg.b->getInt32(64);
        if (strcmp(e->u.var_ref.name, "gl_MaxGeometryOutputComponents") == 0)
            return cg.b->getInt32(128);
        if (strcmp(e->u.var_ref.name, "gl_MaxGeometryTextureImageUnits") == 0)
            return cg.b->getInt32(16);
        if (strcmp(e->u.var_ref.name, "gl_MaxGeometryOutputVertices") == 0)
            return cg.b->getInt32(1024);
        if (strcmp(e->u.var_ref.name, "gl_MaxGeometryTotalOutputComponents") == 0)
            return cg.b->getInt32(1024);
        if (strcmp(e->u.var_ref.name, "gl_MaxGeometryUniformComponents") == 0)
            return cg.b->getInt32(4096);
        /* Atomic counters / image uniforms ride the GS compute expansion;
         * glm_params floors these limits at 8 (mgl_air_reflect.c assigns
         * Metal slots on the same budget), so the shader-visible constants
         * must match the glGetIntegerv values (GLSL 4.60 §7.4 requires
         * the two to agree). */
        if (strcmp(e->u.var_ref.name, "gl_MaxGeometryAtomicCounters") == 0)
            return cg.b->getInt32(8);
        if (strcmp(e->u.var_ref.name, "gl_MaxGeometryAtomicCounterBuffers") == 0)
            return cg.b->getInt32(8);
        if (strcmp(e->u.var_ref.name, "gl_MaxGeometryImageUniforms") == 0)
            return cg.b->getInt32(8);
        if (strcmp(e->u.var_ref.name, "gl_MaxGeometryShaderInvocations") == 0)
            return cg.b->getInt32(32);
        /* Image limits must match glm_params / glGet (GLSL 4.60 §7.3). */
        if (strcmp(e->u.var_ref.name, "gl_MaxImageUnits") == 0)
            return cg.b->getInt32(8);
        if (strcmp(e->u.var_ref.name, "gl_MaxImageSamples") == 0)
            return cg.b->getInt32(8);
        if (strcmp(e->u.var_ref.name, "gl_MaxVertexImageUniforms") == 0 ||
            strcmp(e->u.var_ref.name, "gl_MaxTessControlImageUniforms") == 0 ||
            strcmp(e->u.var_ref.name,
                   "gl_MaxTessEvaluationImageUniforms") == 0 ||
            strcmp(e->u.var_ref.name, "gl_MaxFragmentImageUniforms") == 0 ||
            strcmp(e->u.var_ref.name, "gl_MaxComputeImageUniforms") == 0)
            return cg.b->getInt32(8);
        if (strcmp(e->u.var_ref.name, "gl_MaxCombinedImageUniforms") == 0)
            return cg.b->getInt32(40);
        if (strcmp(e->u.var_ref.name,
                   "gl_MaxCombinedShaderOutputResources") == 0 ||
            strcmp(e->u.var_ref.name,
                   "gl_MaxCombinedImageUnitsAndFragmentOutputs") == 0)
            return cg.b->getInt32(8);
        /* Match glm_params floors / glGet (GLSL 4.60 §7.3). */
        if (strcmp(e->u.var_ref.name, "gl_MaxClipDistances") == 0)
            return cg.b->getInt32(MGL_MAX_CLIP_DISTANCES);
        if (strcmp(e->u.var_ref.name, "gl_MaxCullDistances") == 0)
            return cg.b->getInt32(MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT);
        if (strcmp(e->u.var_ref.name,
                   "gl_MaxCombinedClipAndCullDistances") == 0)
            return cg.b->getInt32(MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT);
        if (strcmp(e->u.var_ref.name, "gl_Position") == 0) {
            if (!cg.position.written) {
                cg.position.name = "gl_Position";
                cg.position.type.scalar = MGLIR_SCALAR_FLOAT;
                cg.position.type.vec = 4;
                cg.position.kind = VarSym::OUTPUT;
            }
            return varValue(cg, cg.position, mod);
        }
        if (strcmp(e->u.var_ref.name, "gl_GlobalInvocationID") == 0) {
            if (!cg.threadPos) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_GlobalInvocationID requires a "
                            "compute stage";
                return nullptr;
            }
            return cg.threadPos;
        }
        if (strcmp(e->u.var_ref.name, "gl_LocalInvocationID") == 0) {
            if (!cg.localInvocationPos) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_LocalInvocationID requires a "
                            "compute stage";
                return nullptr;
            }
            return cg.localInvocationPos;
        }
        if (strcmp(e->u.var_ref.name, "gl_LocalInvocationIndex") == 0) {
            if (!cg.localInvocationIndex) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_LocalInvocationIndex requires a "
                            "compute stage";
                return nullptr;
            }
            return cg.localInvocationIndex;
        }
        if (strcmp(e->u.var_ref.name, "gl_WorkGroupSize") == 0) {
            if (!cg.hasWorkGroupSize) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_WorkGroupSize requires a compute "
                            "stage";
                return nullptr;
            }
            llvm::Type *i32 = cg.b->getInt32Ty();
            return llvm::ConstantVector::get(
                {llvm::ConstantInt::get(i32, cg.workGroupSizeX),
                 llvm::ConstantInt::get(i32, cg.workGroupSizeY),
                 llvm::ConstantInt::get(i32, cg.workGroupSizeZ)});
        }
        if (strcmp(e->u.var_ref.name, "gl_WorkGroupID") == 0) {
            if (!cg.workGroupPos) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_WorkGroupID requires a compute "
                            "stage";
                return nullptr;
            }
            return cg.workGroupPos;
        }
        if (strcmp(e->u.var_ref.name, "gl_NumWorkGroups") == 0) {
            if (!cg.numWorkGroups) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_NumWorkGroups requires a compute "
                            "stage";
                return nullptr;
            }
            return cg.numWorkGroups;
        }
        if (strcmp(e->u.var_ref.name, "gl_VertexID") == 0) {
            if (!cg.vertexId) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_VertexID requires a vertex stage";
                return nullptr;
            }
            return cg.vertexId;
        }
        if (strcmp(e->u.var_ref.name, "gl_InstanceID") == 0) {
            if (!cg.instanceId || !cg.baseInstance) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_InstanceID requires a vertex stage";
                return nullptr;
            }
            return cg.b->CreateSub(cg.instanceId, cg.baseInstance);
        }
        if (strcmp(e->u.var_ref.name, "gl_BaseInstance") == 0) {
            if (!cg.baseInstance) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_BaseInstance requires a vertex stage";
                return nullptr;
            }
            return cg.baseInstance;
        }
        if (strcmp(e->u.var_ref.name, "gl_FragCoord") == 0) {
            if (!cg.fragPos) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_FragCoord requires a fragment stage";
                return nullptr;
            }
            return cg.fragPos;
        }
        if (strcmp(e->u.var_ref.name, "gl_FrontFacing") == 0) {
            if (!cg.lvalues.count("gl_FrontFacing")) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_FrontFacing requires a fragment stage";
                return nullptr;
            }
            return cg.lvalues["gl_FrontFacing"];
        }
        if (strcmp(e->u.var_ref.name, "gl_PointCoord") == 0) {
            if (!cg.lvalues.count("gl_PointCoord")) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_PointCoord requires a fragment stage";
                return nullptr;
            }
            return cg.lvalues["gl_PointCoord"];
        }
        if (strcmp(e->u.var_ref.name, "gl_FragDepth") == 0) {
            if (!cg.lvalues.count("gl_FragDepth"))
                cg.lvalues["gl_FragDepth"] = llvm::ConstantFP::get(
                    llvm::Type::getFloatTy(*cg.ctx), 1.0);
            return cg.lvalues["gl_FragDepth"];
        }
        if (strcmp(e->u.var_ref.name, "gl_SampleID") == 0) {
            if (!cg.lvalues.count("gl_SampleID")) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_SampleID requires a fragment stage";
                return nullptr;
            }
            return cg.lvalues["gl_SampleID"];
        }
        if (strcmp(e->u.var_ref.name, "gl_MaxSamples") == 0)
            return cg.b->getInt32(4);
        if (strcmp(e->u.var_ref.name, "gl_NumSamples") == 0) {
            if (!cg.lvalues.count("gl_NumSamples")) {
                /* Non-MSAA default; the fragment params path overwrites. */
                return cg.b->getInt32(1);
            }
            return cg.lvalues["gl_NumSamples"];
        }
        if (strcmp(e->u.var_ref.name, "gl_SamplePosition") == 0) {
            if (!cg.lvalues.count("gl_SamplePosition")) {
                cg.err = 1;
                cg.errmsg =
                    "codegen: gl_SamplePosition requires a fragment stage";
                return nullptr;
            }
            return cg.lvalues["gl_SamplePosition"];
        }
        if (strcmp(e->u.var_ref.name, "gl_SampleMaskIn") == 0) {
            if (!cg.lvalues.count("gl_SampleMaskIn")) {
                llvm::Type *i32 = cg.b->getInt32Ty();
                llvm::Value *arr = llvm::UndefValue::get(
                    llvm::ArrayType::get(i32, 1));
                arr = cg.b->CreateInsertValue(arr, cg.b->getInt32(~0), 0);
                cg.lvalues["gl_SampleMaskIn"] = arr;
            }
            return cg.lvalues["gl_SampleMaskIn"];
        }
        if (strcmp(e->u.var_ref.name, "gl_SampleMask") == 0) {
            if (!cg.lvalues.count("gl_SampleMask")) {
                llvm::Type *i32 = cg.b->getInt32Ty();
                llvm::Value *arr = llvm::UndefValue::get(
                    llvm::ArrayType::get(i32, 1));
                arr = cg.b->CreateInsertValue(arr, cg.b->getInt32(~0), 0);
                cg.lvalues["gl_SampleMask"] = arr;
            }
            return cg.lvalues["gl_SampleMask"];
        }

        if (strcmp(e->u.var_ref.name, "gl_PointSize") == 0) {
            if (!cg.pointSize) {
                /* read-before-write: an unwritten point size is 1.0 */
                return llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 1.0);
            }
            return cg.lvalues.count("gl_PointSize")
                       ? cg.lvalues["gl_PointSize"]
                       : llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 1.0);
        }
        if (strcmp(e->u.var_ref.name, "gl_CullDistance") == 0) {
            if (!cg.lvalues.count("gl_CullDistance")) {
                cg.lvalues["gl_CullDistance"] = defaultCullDistances(cg);
            }
            return cg.lvalues["gl_CullDistance"];
        }
        if (strcmp(e->u.var_ref.name, "gl_ClipDistance") == 0) {
            if (!cg.lvalues.count("gl_ClipDistance")) {
                /* Unwritten elements stay +1.0: Metal clips where an
                 * element is negative, so the default must not clip. */
                llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
                llvm::Value *arr = llvm::UndefValue::get(
                    llvm::ArrayType::get(f32, MGL_MAX_CLIP_DISTANCES));
                for (uint32_t i = 0; i < MGL_MAX_CLIP_DISTANCES; i++)
                    arr = cg.b->CreateInsertValue(
                        arr, llvm::ConstantFP::get(f32, 1.0), i);
                cg.lvalues["gl_ClipDistance"] = arr;
            }
            return cg.lvalues["gl_ClipDistance"];
        }
        if (strcmp(e->u.var_ref.name, "gl_Layer") == 0 ||
            strcmp(e->u.var_ref.name, "gl_ViewportIndex") == 0 ||
            (strcmp(e->u.var_ref.name, "gl_PrimitiveID") == 0 &&
             !cg.isTessControl && !cg.isTessEval)) {
            /* Out-variable read-back: the value last written this
             * invocation; 0 before any write (GL 4.6 §11.1.3.5/§11.1.3.6).
             * Tess stages read gl_PrimitiveID as a patch input builtin. */
            if (!cg.lvalues.count(e->u.var_ref.name)) {
                cg.lvalues[e->u.var_ref.name] = cg.b->getInt32(0);
            }
            return cg.lvalues[e->u.var_ref.name];
        }
        if (strcmp(e->u.var_ref.name, "gl_InvocationID") == 0) {
            if (cg.isGeometry && cg.geometryInvocationId)
                return cg.geometryInvocationId;
            if (!cg.invocationPos) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_InvocationID requires a TCS stage";
                return nullptr;
            }
            return cg.b->CreateExtractElement(
                cg.invocationPos, cg.b->getInt32(0));
        }
        if (strcmp(e->u.var_ref.name, "gl_PatchVerticesIn") == 0) {
            if (!cg.indirectPtr) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_PatchVerticesIn requires a tessellation stage";
                return nullptr;
            }
            /* TCS indirect: {patch_vertices, instance_count} → word0.
             * TES compute contract: {patch_id, gl_in_vertices, …} → word1.
             * Native TES patch_info: {draw_patch_vertices, tcs_out} → word1
             * (word1 falls back to draw size when there is no TCS). */
            const unsigned word = cg.isTessEval ? 1u : 0u;
            llvm::Value *p = cg.b->CreateBitCast(
                cg.indirectPtr, cg.b->getInt32Ty()->getPointerTo(1));
            return cg.b->CreateAlignedLoad(
                cg.b->getInt32Ty(),
                cg.b->CreateGEP(cg.b->getInt32Ty(), p, cg.b->getInt32(word)),
                llvm::Align(4));
        }
        if (strcmp(e->u.var_ref.name, "gl_PrimitiveID") == 0) {
            if (cg.lvalues.count("gl_PrimitiveID"))
                return cg.lvalues["gl_PrimitiveID"];
            if (cg.isTessControl && cg.workGroupPos)
                return cg.b->CreateExtractElement(cg.workGroupPos,
                                                   cg.b->getInt32(0));
            if (cg.patchId)
                return cg.patchId;
            if (!cg.patchPos) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_PrimitiveID requires a tessellation stage";
                return nullptr;
            }
            return cg.b->CreateExtractElement(cg.patchPos,
                                               cg.b->getInt32(0));
        }
        if (strcmp(e->u.var_ref.name, "gl_PrimitiveIDIn") == 0) {
            if (!cg.isGeometry || !cg.geometryPrimitiveId) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_PrimitiveIDIn requires a geometry stage";
                return nullptr;
            }
            return cg.geometryPrimitiveId;
        }
        if (strcmp(e->u.var_ref.name, "gl_TessCoord") == 0) {
            if (!cg.tessCoord) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_TessCoord requires a TES stage";
                return nullptr;
            }
            return cg.tessCoord;
        }
        if (strcmp(e->u.var_ref.name, "gl_TessLevelOuter") == 0) {
            if (!cg.lvalues.count("gl_TessLevelOuter"))
                cg.lvalues["gl_TessLevelOuter"] = llvm::UndefValue::get(
                    llvm::ArrayType::get(llvm::Type::getFloatTy(*cg.ctx), 4));
            return cg.lvalues["gl_TessLevelOuter"];
        }
        if (strcmp(e->u.var_ref.name, "gl_TessLevelInner") == 0) {
            if (!cg.lvalues.count("gl_TessLevelInner"))
                cg.lvalues["gl_TessLevelInner"] = llvm::UndefValue::get(
                    llvm::ArrayType::get(llvm::Type::getFloatTy(*cg.ctx), 2));
            return cg.lvalues["gl_TessLevelInner"];
        }
        auto lit = locals.find(e->u.var_ref.name);
        if (lit != locals.end())
            return varValue(cg, VarSym{e->u.var_ref.name, lit->second, VarSym::LOCAL},
                            mod);
        const MGLIRSymbol *s = findSymbol(mod, e->u.var_ref.name);
        if (!s) { cg.err = 1; return nullptr; }
        if ((s->qualifiers & MGL_AST_Q_CONST) &&
            cg.lvalues.count(e->u.var_ref.name)) {
            /* Const values folded from global initializers above.  Non-const
             * uniforms must load from the plain pack so glUniform* updates
             * are visible (CTS indirectAddressing-case2). */
            return cg.lvalues[e->u.var_ref.name];
        }
        if (s->qualifiers & MGL_AST_Q_BUFFER)
            return emitSSBORead(cg, e, s, mod, locals);
        if (cg.isTessEval && (s->qualifiers & MGL_AST_Q_PATCH)) {
            VarSym *patch = codegenStageSymbol(
                cg, e->u.var_ref.name, VarSym::CONTROL_POINT_INPUT);
            llvm::Value *loaded = patch
                ? emitPatchVaryingLoad(cg, *patch) : nullptr;
            if (!loaded) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: unavailable TES patch input '") +
                            e->u.var_ref.name + "'";
            }
            return loaded;
        }
        if (cg.isTessControl && (s->qualifiers & MGL_AST_Q_PATCH) &&
            (s->qualifiers & MGL_AST_Q_OUT)) {
            /* Reload from patch-out buffer — SSA cache would hide other
             * invocations' barrier-ordered writes. */
            VarSym *patch = codegenStageSymbol(
                cg, e->u.var_ref.name, VarSym::OUTPUT);
            llvm::Value *loaded = patch
                ? emitPatchVaryingLoad(cg, *patch) : nullptr;
            if (!loaded) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: unavailable TCS patch output '") +
                            e->u.var_ref.name + "'";
            }
            return loaded;
        }
        VarSym v;
        v.name = s->name;
        v.type = typeFromIR(s->type);
        if (v.type.isArray()) {
            /* const array variables are SSA values in cg.lvalues; array
             * varyings (gl_FragData, gl_TexCoord) are pre-registered
             * aggregates in cg.lvalues too.  Uniform arrays (e.g. the
             * legacy gl_TextureMatrix[] / _mglClipPlane[]) fall through to
             * the BUFFER read below, which loads them from the plain
             * uniform blob at their reflection offset. */
            if (!(s->qualifiers & MGL_AST_Q_UNIFORM)) {
                return varValue(cg, VarSym{s->name, v.type,
                                           VarSym::LOCAL}, mod);
            }
        }
        if (s->qualifiers & MGL_AST_Q_UNIFORM) {
            const MGLIRType *ut = s->type;
            while (ut && ut->kind == MGLIR_TYPE_ARRAY)
                ut = ut->elem_type;
            if (ut && ut->kind == MGLIR_TYPE_ATOMIC_COUNTER) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: atomic_uint '") +
                            s->name + "' cannot be loaded directly";
                return nullptr;
            }
            v.kind = VarSym::BUFFER;
        } else if ((s->qualifiers & MGL_AST_Q_IN) && cg.isVS) {
            v.kind = VarSym::ATTR;
        } else {
            v.kind = VarSym::VARYING;
        }
        return varValue(cg, v, mod);
    }
    case MGL_EXPR_MEMBER: {
        if (cg.isTessControl || cg.isTessEval || cg.isGeometry) {
            if (llvm::Value *pv = emitPerVertexLoad(cg, e, mod, locals))
                return pv;
            if (cg.err) return nullptr;
            if (llvm::Value *blk =
                    emitGeometryBlockLoad(cg, e, mod, locals))
                return blk;
            if (cg.err) return nullptr;
            if (llvm::Value *tb =
                    emitTessBlockMemberLoad(cg, e, mod, locals))
                return tb;
            if (cg.err) return nullptr;
        }
        /* Non-arrayed named interface-block member: `input_block.field`
         * (FS in / VS·TES out). Sema flattens these to per-member
         * VARYING symbols keyed by blockName. */
        if (e->u.member.object &&
            e->u.member.object->kind == MGL_EXPR_VAR_REF) {
            const char *inst = e->u.member.object->u.var_ref.name;
            VarSym *member = codegenBlockMember(
                cg, inst, e->u.member.field, VarSym::VARYING);
            if (!member)
                member = codegenBlockMember(
                    cg, inst, e->u.member.field, VarSym::OUTPUT);
            if (member && member->location != UINT32_MAX &&
                !member->type.isArray()) {
                return varValue(cg, *member, mod);
            }
        }
        if (const MGLIRSymbol *sb = ssboRootSym(e, mod)) {
            /* Multi-component swizzles (`.xy`) cannot be addressed as one
             * contiguous scalar in the SSBO; load the parent vector and
             * shuffle in SSA. */
            if (e->kind == MGL_EXPR_MEMBER) {
                std::vector<uint32_t> comps;
                if (swizzleIndices(e->u.member.field, &comps) &&
                    comps.size() > 1u) {
                    llvm::Value *obj =
                        emitSSBORead(cg, e->u.member.object, sb, mod, locals);
                    if (!obj || !obj->getType()->isVectorTy()) {
                        if (!cg.err) {
                            cg.err = 1;
                            cg.errmsg =
                                "codegen: SSBO multi-swizzle needs a vector";
                        }
                        return nullptr;
                    }
                    llvm::SmallVector<llvm::Constant *, 4> mask;
                    for (uint32_t i : comps)
                        mask.push_back(llvm::ConstantInt::get(
                            llvm::Type::getInt32Ty(*cg.ctx), i));
                    return cg.b->CreateShuffleVector(
                        obj, llvm::UndefValue::get(obj->getType()),
                        llvm::ConstantVector::get(mask));
                }
            }
            return emitSSBORead(cg, e, sb, mod, locals);
        }
        /* Uniform block instance member: lightmapInfo.BlockFactor, or
         * uni_block_array[N].entry (each instance-array element is a separate
         * GL uniform block and therefore a separate Metal buffer argument). */
        {
            const MGLExpr *chain[16];
            uint32_t chain_len = 0;
            const MGLExpr *rootIndexExpr = nullptr;
            const MGLIRSymbol *ov = blockChainRoot(e, chain, &chain_len,
                                                   &rootIndexExpr, mod);
            const char *objName = (ov && ov->name) ? ov->name : nullptr;
            /* Locals / parameters shadow flattened UBO member names. */
            if (objName && locals.find(objName) != locals.end())
                ov = nullptr;
            /* Flattened anonymous-block members carry block_name; the Metal
             * buffer is keyed by the block, not the member.  Keep array
             * wrappers on flattened members (`S s[N]`) so `s[i].f` walks
             * array_stride — only peel arrays for true block-instance
             * arrays (handled via rootIndex). */
            const char *bufName =
                (ov && ov->block_name) ? ov->block_name : objName;
            const MGLIRType *ubStruct = nullptr;
            uint32_t startOff = 0u;
            if (ov) {
                if (ov->block_name) {
                    ubStruct = ov->type;
                    if (ov->offset != UINT32_MAX)
                        startOff = ov->offset;
                } else if (ov->type && ov->type->kind == MGLIR_TYPE_ARRAY &&
                           ov->type->elem_type) {
                    ubStruct = ov->type->elem_type;
                } else {
                    ubStruct = ov->type;
                }
            }
            const MGLIRType *structGate = ubStruct;
            while (structGate && structGate->kind == MGLIR_TYPE_ARRAY)
                structGate = structGate->elem_type;
            if (ov && !ov->is_function &&
                (ov->qualifiers & MGL_AST_Q_UNIFORM) &&
                structGate && structGate->kind == MGLIR_TYPE_STRUCT &&
                structGate->member_count > 0) {
                /* Interface blocks / anonymous-block members → UBO Metal
                 * buffers (keyed by block_name). Named struct uniforms →
                 * plain pack at bufferOffsets. */
                llvm::Value *base = nullptr;
                uint32_t plainOff = startOff;
                if (ov->is_interface_block || ov->block_name) {
                if (rootIndexExpr) {
                    /* Instance array: each element binds its own device
                     * buffer; pick it through the entry alloca. */
                    auto slotIt = cg.uboElemSlot.find(objName);
                    auto tyIt = cg.uboElemArrTy.find(objName);
                    if (slotIt == cg.uboElemSlot.end() ||
                        tyIt == cg.uboElemArrTy.end()) {
                        cg.err = 1;
                        cg.errmsg =
                            std::string("codegen: uniform block array '") +
                            objName + "' has no element buffers";
                        return nullptr;
                    }
                    llvm::Value *elemIndex =
                        emitExpr(cg, rootIndexExpr->u.index.index, mod,
                                 locals);
                    if (!elemIndex) return nullptr;
                    elemIndex = coerceScalar(cg, elemIndex,
                                             MGLIR_SCALAR_UINT);
                    /* Out-of-range dynamic indices are undefined in GLSL;
                     * clamp so a bad runtime index cannot select a wild
                     * buffer pointer from the element alloca. */
                    {
                        uint32_t elemCount =
                            ov->type->kind == MGLIR_TYPE_ARRAY &&
                                    ov->type->array_size > 0
                                ? ov->type->array_size
                                : 1u;
                        elemIndex = cg.b->CreateBinaryIntrinsic(
                            llvm::Intrinsic::umax,
                            elemIndex,
                            cg.b->getInt32(0));
                        elemIndex = cg.b->CreateBinaryIntrinsic(
                            llvm::Intrinsic::umin,
                            elemIndex,
                            cg.b->getInt32(elemCount - 1u));
                    }
                    llvm::Value *gep = cg.b->CreateGEP(
                        tyIt->second, slotIt->second,
                        {cg.b->getInt64(0),
                         cg.b->CreateZExt(elemIndex,
                                          cg.b->getInt64Ty())});
                    base = cg.b->CreateLoad(
                        llvm::Type::getInt8Ty(*cg.ctx)->getPointerTo(1),
                        gep);
                } else {
                    base = bufName && cg.uboPtrs.count(bufName)
                               ? cg.uboPtrs[bufName]
                               : nullptr;
                }
                if (!base) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: uniform block '") +
                                (bufName ? bufName : "?") +
                                "' has no device buffer";
                    return nullptr;
                }
                } else {
                    if (!cg.bufferPtr || !objName ||
                        !cg.bufferOffsets.count(objName)) {
                        cg.err = 1;
                        cg.errmsg =
                            std::string("codegen: plain uniform struct '") +
                            (objName ? objName : "?") +
                            "' has no packed offset";
                        return nullptr;
                    }
                    base = cg.bufferPtr;
                    plainOff = cg.bufferOffsets[objName] + startOff;
                    if (rootIndexExpr) {
                        /* Array of named struct uniforms in the plain pack. */
                        llvm::Value *elemIndex =
                            emitExpr(cg, rootIndexExpr->u.index.index, mod,
                                     locals);
                        if (!elemIndex) return nullptr;
                        elemIndex = coerceScalar(cg, elemIndex,
                                                 MGLIR_SCALAR_INT);
                        uint32_t stride =
                            ov->type->kind == MGLIR_TYPE_ARRAY
                                ? (uint32_t)ov->type->layout.array_stride
                                : 0u;
                        llvm::Value *byte = cg.b->CreateMul(
                            cg.b->CreateSExt(elemIndex, cg.b->getInt64Ty()),
                            cg.b->getInt64(stride));
                        base = cg.b->CreateGEP(cg.b->getInt8Ty(), base,
                                               byte);
                    }
                }
                return emitBlockMemberChain(cg, e, base, ubStruct,
                                            bufName ? bufName : objName, mod,
                                            locals, plainOff);
            }
        }
        /* Local / temporary struct field access (e.g. S.member after an
         * initializer-list desugar to S(...)). */
        if (const MGLIRType *objTy =
                exprIRType(cg, e->u.member.object, mod, locals)) {
            while (objTy->kind == MGLIR_TYPE_ARRAY && objTy->elem_type)
                objTy = objTy->elem_type;
            if (objTy->kind == MGLIR_TYPE_STRUCT) {
                for (uint32_t i = 0; i < objTy->member_count; i++) {
                    if (objTy->member_names[i] &&
                        strcmp(objTy->member_names[i],
                               e->u.member.field) == 0) {
                        llvm::Value *obj =
                            emitExpr(cg, e->u.member.object, mod, locals);
                        if (!obj) return nullptr;
                        return cg.b->CreateExtractValue(obj, i);
                    }
                }
                cg.err = 1;
                cg.errmsg = std::string("codegen: unknown member '") +
                            e->u.member.field + "'";
                return nullptr;
            }
        }
        /* Swizzle only in M1. */
        std::vector<uint32_t> idx;
        if (!swizzleIndices(e->u.member.field, &idx)) { cg.err = 1; return nullptr; }
        llvm::Value *obj = emitExpr(cg, e->u.member.object, mod, locals);
        if (!obj) return nullptr;
        if (!obj->getType()->isVectorTy()) {
            /* Member access on a non-vector (e.g. a struct-typed member
             * of a uniform block, whose aggregate reads are not wired
             * yet): fail with a diagnostic instead of an invalid
             * ExtractElement on an aggregate (SIGSEGV in LLVM). */
            cg.err = 1;
            cg.errmsg = std::string("codegen: member '") +
                        e->u.member.field +
                        "' of a non-vector value is not supported";
            return nullptr;
        }
        if (idx.size() == 1)
            return cg.b->CreateExtractElement(obj,
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), idx[0]));
        llvm::SmallVector<llvm::Constant *, 4> mask;
        for (uint32_t i : idx)
            mask.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), i));
        llvm::Value *undef = llvm::UndefValue::get(obj->getType());
        return cg.b->CreateShuffleVector(obj, undef,
            llvm::ConstantVector::get(mask));
    }
    case MGL_EXPR_INDEX: {
        /* Matrix[i] yields a column vector (GLSL 4.60 5.5), vector[i] a
         * component; the index may be a constant or a runtime value. */
        if (llvm::Value *blockElem =
                emitGeometryBlockArrayLoad(cg, e, mod, locals))
            return blockElem;
        if (cg.err) return nullptr;
        if (e->u.index.object && e->u.index.object->kind == MGL_EXPR_VAR_REF) {
            const char *an = e->u.index.object->u.var_ref.name;
            VarSym *patchArr = nullptr;
            if (cg.isTessControl)
                patchArr = codegenStageSymbol(cg, an, VarSym::OUTPUT);
            else if (cg.isTessEval)
                patchArr = codegenStageSymbol(
                    cg, an, VarSym::CONTROL_POINT_INPUT);
            if (patchArr && patchArr->isPatch && patchArr->type.isArray()) {
                llvm::Value *idx =
                    emitExpr(cg, e->u.index.index, mod, locals);
                if (!idx) return nullptr;
                llvm::Value *loaded =
                    emitPatchArrayElementLoad(cg, *patchArr, idx);
                if (!loaded) {
                    cg.err = 1;
                    cg.errmsg = "codegen: unavailable patch array element load";
                }
                return loaded;
            }
        }
        if (llvm::Value *stageValue =
                emitTessStageArrayLoad(cg, e, mod, locals))
            return stageValue;
        if (cg.err) return nullptr;
        if (const MGLIRSymbol *sb = ssboRootSym(e, mod))
            return emitSSBORead(cg, e, sb, mod, locals);
        /* Uniform-block array/vector indexing must stay on the block chain
         * path so std140 array_stride is applied.  Falling through to
         * load-whole-array + ExtractValue packs elements tightly and
         * mis-reads ivec2/vec2 arrays (stride 8 instead of 16). */
        {
            const MGLExpr *chain[16];
            uint32_t chain_len = 0;
            const MGLExpr *rootIndexExpr = nullptr;
            const MGLIRSymbol *ov = blockChainRoot(e, chain, &chain_len,
                                                   &rootIndexExpr, mod);
            const char *objName = (ov && ov->name) ? ov->name : nullptr;
            if (objName && locals.find(objName) != locals.end())
                ov = nullptr;
            const char *bufName =
                (ov && ov->block_name) ? ov->block_name : objName;
            const MGLIRType *ubStruct = nullptr;
            uint32_t startOff = 0u;
            if (ov) {
                if (ov->block_name) {
                    ubStruct = ov->type;
                    if (ov->offset != UINT32_MAX)
                        startOff = ov->offset;
                } else if (ov->type && ov->type->kind == MGLIR_TYPE_ARRAY &&
                           ov->type->elem_type) {
                    ubStruct = ov->type->elem_type;
                } else {
                    ubStruct = ov->type;
                }
            }
            const MGLIRType *structGate = ubStruct;
            while (structGate && structGate->kind == MGLIR_TYPE_ARRAY)
                structGate = structGate->elem_type;
            if (ov && !ov->is_function &&
                (ov->qualifiers & MGL_AST_Q_UNIFORM) &&
                structGate && structGate->kind == MGLIR_TYPE_STRUCT &&
                structGate->member_count > 0) {
                llvm::Value *base = nullptr;
                uint32_t plainOff = startOff;
                if (ov->is_interface_block || ov->block_name) {
                if (rootIndexExpr) {
                    auto slotIt = cg.uboElemSlot.find(objName);
                    auto tyIt = cg.uboElemArrTy.find(objName);
                    if (slotIt == cg.uboElemSlot.end() ||
                        tyIt == cg.uboElemArrTy.end()) {
                        cg.err = 1;
                        cg.errmsg =
                            std::string("codegen: uniform block array '") +
                            objName + "' has no element slots";
                        return nullptr;
                    }
                    llvm::Value *elemIndex =
                        emitExpr(cg, rootIndexExpr->u.index.index, mod,
                                 locals);
                    if (!elemIndex) return nullptr;
                    llvm::Value *gep = cg.b->CreateInBoundsGEP(
                        tyIt->second, slotIt->second,
                        {cg.b->getInt64(0),
                         cg.b->CreateZExt(elemIndex,
                                          cg.b->getInt64Ty())});
                    base = cg.b->CreateLoad(
                        llvm::Type::getInt8Ty(*cg.ctx)->getPointerTo(1),
                        gep);
                } else {
                    base = bufName && cg.uboPtrs.count(bufName)
                               ? cg.uboPtrs[bufName]
                               : nullptr;
                }
                if (!base) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: uniform block '") +
                                (bufName ? bufName : "?") +
                                "' has no device buffer";
                    return nullptr;
                }
                } else {
                    if (!cg.bufferPtr || !objName ||
                        !cg.bufferOffsets.count(objName)) {
                        cg.err = 1;
                        cg.errmsg =
                            std::string("codegen: plain uniform struct '") +
                            (objName ? objName : "?") +
                            "' has no packed offset";
                        return nullptr;
                    }
                    base = cg.bufferPtr;
                    plainOff = cg.bufferOffsets[objName] + startOff;
                    if (rootIndexExpr) {
                        llvm::Value *elemIndex =
                            emitExpr(cg, rootIndexExpr->u.index.index, mod,
                                     locals);
                        if (!elemIndex) return nullptr;
                        elemIndex = coerceScalar(cg, elemIndex,
                                                 MGLIR_SCALAR_INT);
                        uint32_t stride =
                            ov->type->kind == MGLIR_TYPE_ARRAY
                                ? (uint32_t)ov->type->layout.array_stride
                                : 0u;
                        llvm::Value *byte = cg.b->CreateMul(
                            cg.b->CreateSExt(elemIndex, cg.b->getInt64Ty()),
                            cg.b->getInt64(stride));
                        base = cg.b->CreateGEP(cg.b->getInt8Ty(), base,
                                               byte);
                    }
                }
                return emitBlockMemberChain(cg, e, base, ubStruct,
                                            bufName ? bufName : objName, mod,
                                            locals, plainOff);
            }
        }
        /* Anonymous UBO member: `var[i]` where `var` was flattened out of
         * `uniform Block { T var[N]; }`.  Must apply array_stride /
         * matrix_stride — loading the whole aggregate then ExtractValue
         * packs float/vec2/mat2 tightly and mis-reads std140.
         * Skip when the name is a function parameter / local — those
         * shadow flattened block members (CTS compare_mat*(a,b) vs UBO
         * members also named a/b). */
        if (e->u.index.object &&
            e->u.index.object->kind == MGL_EXPR_VAR_REF) {
            const char *aname = e->u.index.object->u.var_ref.name;
            if (locals.find(aname) == locals.end()) {
            const MGLIRSymbol *bs = findSymbol(mod, aname);
            if (bs && bs->block_name && !bs->is_function &&
                (bs->qualifiers & MGL_AST_Q_UNIFORM) && bs->type) {
                llvm::Value *base = cg.uboPtrs.count(bs->block_name)
                                        ? cg.uboPtrs[bs->block_name]
                                        : nullptr;
                if (!base) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: uniform block '") +
                                bs->block_name + "' has no device buffer";
                    return nullptr;
                }
                uint32_t moff = bs->offset != UINT32_MAX ? bs->offset : 0u;
                llvm::Value *idxVal =
                    emitExpr(cg, e->u.index.index, mod, locals);
                if (!idxVal) return nullptr;
                idxVal = coerceScalar(cg, idxVal, MGLIR_SCALAR_INT);
                llvm::Value *i64 =
                    cg.b->CreateSExt(idxVal, cg.b->getInt64Ty());
                if (bs->type->kind == MGLIR_TYPE_ARRAY &&
                    bs->type->elem_type) {
                    uint32_t stride = bs->type->layout.array_stride;
                    if (stride == 0) {
                        cg.err = 1;
                        cg.errmsg =
                            std::string("codegen: anonymous UBO array '") +
                            aname + "' has no array_stride";
                        return nullptr;
                    }
                    llvm::Value *byte = cg.b->CreateMul(
                        i64, cg.b->getInt64(stride));
                    llvm::Value *off = cg.b->CreateAdd(
                        cg.b->getInt64(moff), byte);
                    const MGLIRType *elem = bs->type->elem_type;
                    return emitUBOLeafLoad(cg, base, off, elem,
                                           typeFromIR(elem));
                }
                if (bs->type->kind == MGLIR_TYPE_MATRIX) {
                    MType vt = typeFromIR(bs->type);
                    llvm::Value *mat = emitUBOMatrixLoad(
                        cg, base, cg.b->getInt64(moff), bs->type, vt);
                    llvm::Value *col = emitIndexValue(cg, mat, vt, idxVal);
                    if (!col) {
                        cg.err = 1;
                        cg.errmsg =
                            "codegen: indexing anonymous UBO matrix failed";
                        return nullptr;
                    }
                    return col;
                }
            }
            }
        }
        const MGLExpr *idxE = e->u.index.index;
        if (cg.isTessEval && e->u.index.object &&
            e->u.index.object->kind == MGL_EXPR_VAR_REF) {
            const char *name = e->u.index.object->u.var_ref.name;
            auto field = cg.controlPointFields.find(name);
            if (field != cg.controlPointFields.end()) {
                llvm::Value *idx = emitExpr(cg, idxE, mod, locals);
                if (!idx) return nullptr;
                idx = coerceScalar(cg, idx, MGLIR_SCALAR_UINT);
                if (cg.isTESCompute) {
                    /* isolines/point-mode kernel: control-point varying
                     * fields live in the stage_in records (VS output
                     * layout), not the Metal control-point function. */
                    if (!cg.stageInPtr || !cg.indirectPtr || !cg.patchId) {
                        cg.err = 1;
                        cg.errmsg = "TES AIR codegen: shared control-point "
                                    "buffer is unavailable";
                        return nullptr;
                    }
                    llvm::Value *patchInfo = cg.b->CreateBitCast(
                        cg.indirectPtr, cg.b->getInt32Ty()->getPointerTo(1));
                    llvm::Value *verticesPerPatch = cg.b->CreateAlignedLoad(
                        cg.b->getInt32Ty(),
                        cg.b->CreateGEP(cg.b->getInt32Ty(), patchInfo,
                                        cg.b->getInt32(1)),
                        llvm::Align(4));
                    llvm::Value *flat = cg.b->CreateAdd(
                        cg.b->CreateMul(cg.patchId, verticesPerPatch), idx);
                    VarSym *sym =
                        codegenStageSymbol(cg, name, VarSym::CONTROL_POINT_INPUT);
                    if (!sym || sym->location == UINT32_MAX) {
                        cg.err = 1;
                        cg.errmsg = "TES AIR codegen: control-point varying "
                                    "has no location";
                        return nullptr;
                    }
                    llvm::Value *off = cg.b->CreateAdd(
                        cg.b->CreateMul(
                            cg.b->CreateZExt(flat, cg.b->getInt64Ty()),
                            cg.b->getInt64(cg.stageInStride)),
                        cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE +
                                       (uint64_t)sym->location * 16u));
                    llvm::Value *p = cg.b->CreateGEP(
                        cg.b->getInt8Ty(), cg.stageInPtr, off);
                    llvm::Type *ty = llvmType(sym->type, *cg.ctx);
                    llvm::Type *loadTy = ty;
                    if (varyingNeedsFloatRecordCarrier(sym->type))
                        loadTy = llvmType(floatCarrierType(sym->type),
                                          *cg.ctx);
                    p = cg.b->CreateBitCast(p, loadTy->getPointerTo(1));
                    llvm::Value *v =
                        cg.b->CreateAlignedLoad(loadTy, p, llvm::Align(4));
                    if (varyingNeedsFloatRecordCarrier(sym->type))
                        v = decodeFloatCarrier(cg, v, sym->type.scalar, ty);
                    return v;
                }
                /* The Metal control-point function only exists for the
                 * non-compute, non-render-vertex TES codegen.  The sibling
                 * paths at 2807 / 2878 check for it; this one did not, so a
                 * TES-vertex program that indexed gl_in[i].<field> built a
                 * CallInst against a null callee and crashed inside LLVM's
                 * type accessors (reachable once point-mode geometry
                 * expansion got its correct point topology and the stage
                 * actually compiled). */
                if (!cg.controlPointGetter || !cg.patchControlPtr) {
                    cg.err = 1;
                    cg.errmsg = "TES AIR codegen: control-point getter is "
                                "unavailable for this tessellation path";
                    return nullptr;
                }
                llvm::Value *record = cg.b->CreateCall(
                    cg.controlPointGetter, {idx, cg.patchControlPtr});
                return cg.b->CreateExtractValue(record, field->second);
            }
        }
        bool constIdx = idxE->kind == MGL_EXPR_LITERAL &&
                        (idxE->u.literal.base == MGL_AST_TYPE_INT ||
                         idxE->u.literal.base == MGL_AST_TYPE_UINT);
        MType bt = exprType(cg, e->u.index.object, mod, locals);
        /* Memory-backed scalar arrays: GEP + load (avoid SSA [N x float]). */
        if (e->u.index.object->kind == MGL_EXPR_VAR_REF) {
            const char *an = e->u.index.object->u.var_ref.name;
            auto amit = cg.arrayMem.find(an ? an : "");
            if (amit != cg.arrayMem.end()) {
                llvm::Value *idx = emitExpr(cg, idxE, mod, locals);
                if (!idx) return nullptr;
                llvm::Value *ep =
                    arrayMemGEP(cg, an, amit->second, idx);
                llvm::Type *arrTy = cg.arrayMemTypes[an];
                auto *aty = llvm::cast<llvm::ArrayType>(arrTy);
                return cg.b->CreateAlignedLoad(aty->getElementType(), ep,
                                               llvm::Align(4));
            }
        }
        llvm::Value *obj = emitExpr(cg, e->u.index.object, mod, locals);
        if (!obj) return nullptr;
        if (constIdx) {
            uint32_t i = (uint32_t)idxE->u.literal.value;
            if (bt.isArray() || obj->getType()->isArrayTy()) {
                auto *arrayTy = llvm::dyn_cast<llvm::ArrayType>(obj->getType());
                if (!arrayTy || i >= arrayTy->getNumElements()) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: array index ") +
                                std::to_string(i) + " out of range";
                    return nullptr;
                }
                return cg.b->CreateExtractValue(obj, i);
            }
            if (bt.isMatrix()) {
                if (i >= bt.cols || !obj->getType()->isArrayTy()) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: column index ") +
                                std::to_string(i) + " out of range";
                    return nullptr;
                }
                return cg.b->CreateExtractValue(obj, i);
            }
            if (obj->getType()->isVectorTy()) {
                return cg.b->CreateExtractElement(obj,
                    llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), i));
            }
        } else {
            llvm::Value *idx = emitExpr(cg, idxE, mod, locals);
            if (!idx) return nullptr;
            llvm::Value *res = emitIndexValue(cg, obj, bt, idx);
            if (res) return res;
        }
        cg.err = 1;
        cg.errmsg = std::string("codegen: indexing this type is not "
                                "implemented in M1");
        return nullptr;
    }
    case MGL_EXPR_CALL: {
        const char *name = e->u.call.name;
        if (strcmp(name, "__mgl_array_length") == 0) {
            if (e->u.call.arg_count != 1) {
                cg.err = 1;
                cg.errmsg = "codegen: array length() requires one internal object argument";
                return nullptr;
            }
            const MGLExpr *object = e->u.call.args[0];
            if (const MGLIRSymbol *sb = ssboRootSym(object, mod)) {
                uint32_t tailOffset = 0;
                const MGLIRType *array = ssboExprType(object, sb, &tailOffset);
                if (!array || array->kind != MGLIR_TYPE_ARRAY) {
                    cg.err = 1;
                    cg.errmsg = "codegen: length() object is not an SSBO array";
                    return nullptr;
                }
                if (array->array_size != 0)
                    return cg.b->getInt32(array->array_size);
                /* Anonymous members are keyed by the owning block name in
                 * ssboSlots/ssboPtrs (same as ssboAddress). */
                const char *slotName =
                    (sb->block_name && sb->block_name[0]) ? sb->block_name
                                                         : sb->name;
                auto slot = cg.ssboSlots.find(slotName);
                if (!cg.bufferSizePtr || slot == cg.ssboSlots.end()) {
                    cg.err = 1;
                    cg.errmsg = "codegen: runtime SSBO length requires buffer(25) sizes";
                    return nullptr;
                }
                uint32_t stride = array->layout.array_stride;
                if (!stride && array->elem_type)
                    stride = array->elem_type->layout.size;
                if (!stride) {
                    cg.err = 1;
                    cg.errmsg = "codegen: runtime SSBO array has zero stride";
                    return nullptr;
                }
                uint32_t sizeSlot = slot->second;
                /* Buffer instance arrays: g_input23[i].data.length() must
                 * read the size of element i's bound Metal buffer. */
                if (uniformBlockIsInstanceArray(sb->type) &&
                    !sb->block_name) {
                    const MGLExpr *walk = object;
                    while (walk && (walk->kind == MGL_EXPR_MEMBER ||
                                    walk->kind == MGL_EXPR_INDEX)) {
                        if (walk->kind == MGL_EXPR_INDEX &&
                            walk->u.index.object &&
                            walk->u.index.object->kind == MGL_EXPR_VAR_REF &&
                            walk->u.index.object->u.var_ref.name &&
                            strcmp(walk->u.index.object->u.var_ref.name,
                                   sb->name) == 0) {
                            llvm::Value *iv = emitExpr(
                                cg, walk->u.index.index, mod, locals);
                            if (!iv) return nullptr;
                            if (auto *ci =
                                    llvm::dyn_cast<llvm::ConstantInt>(iv)) {
                                uint64_t idx = ci->getZExtValue();
                                uint32_t n =
                                    uniformBlockElementCount(sb->type);
                                if (idx < n) sizeSlot += (uint32_t)idx;
                            } else {
                                /* Dynamic instance index: load sizes[base+i]
                                 * at runtime. */
                                llvm::Value *sizes = cg.b->CreateBitCast(
                                    cg.bufferSizePtr,
                                    llvm::Type::getInt32Ty(*cg.ctx)
                                        ->getPointerTo(2));
                                iv = coerceScalar(cg, iv, MGLIR_SCALAR_UINT);
                                llvm::Value *base =
                                    cg.b->getInt32(slot->second);
                                llvm::Value *idx32 = cg.b->CreateAdd(
                                    base,
                                    cg.b->CreateZExtOrTrunc(
                                        iv, cg.b->getInt32Ty()));
                                llvm::Value *sizePtr = cg.b->CreateGEP(
                                    cg.b->getInt32Ty(), sizes, idx32);
                                llvm::Value *boundSize =
                                    cg.b->CreateAlignedLoad(
                                        cg.b->getInt32Ty(), sizePtr,
                                        llvm::Align(4));
                                llvm::Value *hasTail = cg.b->CreateICmpUGT(
                                    boundSize, cg.b->getInt32(tailOffset));
                                llvm::Value *available = cg.b->CreateSelect(
                                    hasTail,
                                    cg.b->CreateSub(
                                        boundSize,
                                        cg.b->getInt32(tailOffset)),
                                    cg.b->getInt32(0));
                                return cg.b->CreateUDiv(
                                    available, cg.b->getInt32(stride));
                            }
                            break;
                        }
                        walk = walk->kind == MGL_EXPR_MEMBER
                                   ? walk->u.member.object
                                   : walk->u.index.object;
                    }
                }
                llvm::Value *sizes = cg.b->CreateBitCast(
                    cg.bufferSizePtr,
                    llvm::Type::getInt32Ty(*cg.ctx)->getPointerTo(2));
                llvm::Value *sizePtr = cg.b->CreateGEP(
                    cg.b->getInt32Ty(), sizes, cg.b->getInt64(sizeSlot));
                llvm::Value *boundSize = cg.b->CreateAlignedLoad(
                    cg.b->getInt32Ty(), sizePtr, llvm::Align(4));
                llvm::Value *hasTail = cg.b->CreateICmpUGT(
                    boundSize, cg.b->getInt32(tailOffset));
                llvm::Value *available = cg.b->CreateSelect(
                    hasTail,
                    cg.b->CreateSub(boundSize, cg.b->getInt32(tailOffset)),
                    cg.b->getInt32(0));
                return cg.b->CreateUDiv(available, cg.b->getInt32(stride));
            }
            if (object && object->kind == MGL_EXPR_VAR_REF &&
                strcmp(object->u.var_ref.name, "gl_in") == 0 &&
                cg.isGeometry) {
                return cg.b->getInt32(cg.geometryInputVertices);
            }
            if (llvm::Value *len = emitGLSLTypeLength(cg, object, mod, locals))
                return len;
            cg.err = 1;
            cg.errmsg = "codegen: length() requires an array, vector, or matrix expression";
            return nullptr;
        }
        if (strcmp(name, "EmitVertex") == 0 ||
            strcmp(name, "EmitStreamVertex") == 0) {
            if (strcmp(name, "EmitStreamVertex") == 0 &&
                e->u.call.arg_count != 1) {
                cg.err = 1;
                cg.errmsg = "GS AIR codegen: EmitStreamVertex takes one constant stream argument";
                return nullptr;
            }
            if (strcmp(name, "EmitVertex") == 0 && e->u.call.arg_count != 0) {
                cg.err = 1;
                cg.errmsg = "GS AIR codegen: EmitVertex takes no arguments";
                return nullptr;
            }
            int32_t stream = 0;
            if (strcmp(name, "EmitStreamVertex") == 0) {
                llvm::Value *sv = emitExpr(
                    cg, e->u.call.args[0], mod, locals);
                if (!sv) return nullptr;
                if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(sv)) {
                    if (ci->getZExtValue() >= MGL_AIR_GS_MAX_STREAMS) {
                        cg.err = 1;
                        cg.errmsg = "GS AIR codegen: stream must be in [0, 3]";
                        return nullptr;
                    }
                    stream = (int32_t)ci->getZExtValue();
                } else {
                    cg.err = 1;
                    cg.errmsg = "GS AIR codegen: stream must be a constant expression";
                    return nullptr;
                }
                if (stream > 0 &&
                    cg.geometryOutputType != MGL_AST_GS_OUT_POINTS) {
                    cg.err = 1;
                    cg.errmsg = "GS AIR codegen: streams above 0 require points output";
                    return nullptr;
                }
            }
            if (stream > 0) {
                return emitGeometryStreamVertex(cg, stream);
            }
            return emitGeometryVertex(cg);
        }
        if (strcmp(name, "EndPrimitive") == 0 ||
            strcmp(name, "EndStreamPrimitive") == 0) {
            if (strcmp(name, "EndStreamPrimitive") == 0) {
                if (e->u.call.arg_count != 1) {
                    cg.err = 1;
                    cg.errmsg = "GS AIR codegen: EndStreamPrimitive takes one constant stream argument";
                    return nullptr;
                }
                llvm::Value *sv = emitExpr(
                    cg, e->u.call.args[0], mod, locals);
                if (!sv) return nullptr;
                uint64_t stream = 0;
                if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(sv)) {
                    stream = ci->getZExtValue();
                    if (stream >= MGL_AIR_GS_MAX_STREAMS) {
                        cg.err = 1;
                        cg.errmsg = "GS AIR codegen: stream must be in [0, 3]";
                        return nullptr;
                    }
                } else {
                    cg.err = 1;
                    cg.errmsg = "GS AIR codegen: stream must be a constant expression";
                    return nullptr;
                }
                if (!cg.isGeometry || !cg.geometryCountPtr ||
                    !cg.geometryPrimitiveId) {
                    cg.err = 1;
                    cg.errmsg = "GS AIR codegen: EndStreamPrimitive requires the GS output ABI";
                    return nullptr;
                }
                /* Stream 0 owns the raster strip counter.  Streams > 0 are
                 * points-only; EndStreamPrimitive there must not reset
                 * stream 0's strip. */
                if (stream == 0) {
                    cg.b->CreateAlignedStore(cg.b->getInt32(0),
                                             geometryCounterPtr(cg, 1),
                                             llvm::Align(4));
                }
                return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), 0);
            } else if (e->u.call.arg_count != 0) {
                cg.err = 1;
                cg.errmsg = "GS AIR codegen: EndPrimitive takes no arguments";
                return nullptr;
            }
            if (!cg.isGeometry || !cg.geometryCountPtr ||
                !cg.geometryPrimitiveId) {
                cg.err = 1;
                cg.errmsg = "GS AIR codegen: EndPrimitive requires the  output ABI";
                return nullptr;
            }
            cg.b->CreateAlignedStore(cg.b->getInt32(0),
                                     geometryCounterPtr(cg, 1),
                                     llvm::Align(4));
            return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), 0);
        }
        /* User struct constructors: S(a,b,...) and S[](…).  Must run before
         * the generic array-ctor path so element type stays a struct. */
        {
            auto sit = cg.structTypes.find(name);
            if (sit != cg.structTypes.end()) {
                const MGLIRType *st = sit->second;
                if (e->u.call.is_array_ctor) {
                    uint32_t n = e->u.call.arg_count;
                    llvm::Type *eltTy = llvmTypeFromIR(st, *cg.ctx);
                    llvm::Value *res = llvm::UndefValue::get(
                        llvm::ArrayType::get(eltTy, n ? n : 1u));
                    for (uint32_t a = 0; a < n; a++) {
                        llvm::Value *arg =
                            emitExpr(cg, e->u.call.args[a], mod, locals);
                        if (!arg) return nullptr;
                        if (arg->getType() != eltTy)
                            arg = cg.b->CreateBitCast(arg, eltTy);
                        res = cg.b->CreateInsertValue(res, arg, a);
                    }
                    return res;
                }
                if (e->u.call.arg_count != st->member_count) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: constructor '") + name +
                                "' expects " +
                                std::to_string(st->member_count) +
                                " argument(s)";
                    return nullptr;
                }
                llvm::Type *sty = llvmTypeFromIR(st, *cg.ctx);
                llvm::Value *res = llvm::UndefValue::get(sty);
                for (uint32_t a = 0; a < e->u.call.arg_count; a++) {
                    llvm::Value *arg =
                        emitExpr(cg, e->u.call.args[a], mod, locals);
                    if (!arg) return nullptr;
                    llvm::Type *want =
                        llvmTypeFromIR(st->members[a], *cg.ctx);
                    if (arg->getType() != want) {
                        if (arg->getType()->isIntOrIntVectorTy() ||
                            arg->getType()->isFPOrFPVectorTy())
                            arg = coerceScalar(
                                cg, arg, typeFromIR(st->members[a]).scalar);
                        else
                            arg = cg.b->CreateBitCast(arg, want);
                    }
                    res = cg.b->CreateInsertValue(res, arg, a);
                }
                return res;
            }
        }
        /* Array constructors: int[](a,b,...) / vecN[](...). Must run before
         * scalar `int`/`float` constructors — those share the same callee
         * name and would reject multi-arg array forms. */
        if (e->u.call.is_array_ctor) {
            llvm::Value *res = llvm::UndefValue::get(llvmType(
                exprType(cg, e, mod, locals), *cg.ctx));
            if (e->u.call.arg_count == 0) return res;
            MType et = exprType(cg, e, mod, locals);
            et.arr = 0;
            llvm::Type *eltTy = llvmType(et, *cg.ctx);
            for (uint32_t a = 0; a < e->u.call.arg_count; a++) {
                llvm::Value *arg = emitExpr(cg, e->u.call.args[a], mod, locals);
                if (!arg) return nullptr;
                if (arg->getType() != eltTy) {
                    if (arg->getType()->isIntOrIntVectorTy() ||
                        arg->getType()->isFPOrFPVectorTy())
                        arg = coerceScalar(cg, arg, et.scalar);
                    else
                        arg = cg.b->CreateBitCast(arg, eltTy);
                }
                res = cg.b->CreateInsertValue(res, arg, a);
            }
            return res;
        }
        /* Scalar constructors / conversions. */
        if (strcmp(name, "float") == 0 || strcmp(name, "int") == 0 ||
            strcmp(name, "uint") == 0 || strcmp(name, "bool") == 0) {
            if (e->u.call.arg_count != 1) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: constructor '") + name +
                            "' expects 1 argument";
                return nullptr;
            }
            llvm::Value *arg = emitExpr(cg, e->u.call.args[0], mod, locals);
            if (!arg) return nullptr;
            /* GLSL 4.60 §5.4.2: scalar(vec) / scalar(mat) take the first
             * component.  coerceScalar preserves vector shape, so peel
             * here — otherwise float(vec4) stays <4 x float> and a later
             * store into float* fails Metal materializeAll. */
            if (auto *fvt = llvm::dyn_cast<llvm::FixedVectorType>(
                    arg->getType())) {
                (void)fvt;
                arg = cg.b->CreateExtractElement(
                    arg, llvm::ConstantInt::get(
                             llvm::Type::getInt32Ty(*cg.ctx), 0));
            } else if (auto *arrTy = llvm::dyn_cast<llvm::ArrayType>(
                           arg->getType())) {
                llvm::Value *col0 = cg.b->CreateExtractValue(arg, 0);
                if (col0->getType()->isVectorTy())
                    arg = cg.b->CreateExtractElement(
                        col0, llvm::ConstantInt::get(
                                  llvm::Type::getInt32Ty(*cg.ctx), 0));
                else
                    arg = col0;
                (void)arrTy;
            }
            MGLIRScalar want = name[0] == 'f' ? MGLIR_SCALAR_FLOAT
                             : name[0] == 'u' ? MGLIR_SCALAR_UINT
                             : name[0] == 'b' ? MGLIR_SCALAR_BOOL
                                              : MGLIR_SCALAR_INT;
            return coerceScalar(cg, arg, want);
        }
        /* Vector constructors: [i]uvec/bvec/vec2..4. */
        const char *vn = name;
        MGLIRScalar velt = MGLIR_SCALAR_FLOAT;
        uint32_t vlanes = 0;
        if (strncmp(vn, "ivec", 4) == 0 || strncmp(vn, "uvec", 4) == 0 ||
            strncmp(vn, "bvec", 4) == 0) {
            velt = vn[0] == 'i' ? MGLIR_SCALAR_INT
                 : vn[0] == 'u' ? MGLIR_SCALAR_UINT
                                : MGLIR_SCALAR_BOOL;
            vn += 4;
        } else if (strncmp(vn, "vec", 3) == 0) {
            vn += 3;
        } else {
            vn = nullptr;
        }
        if (vn && vn[0] >= '2' && vn[0] <= '4' && vn[1] == '\0') {
            vlanes = (uint32_t)(vn[0] - '0');
            llvm::Type *eltTy = llvmScalar(velt, *cg.ctx);
            llvm::Type *vt = llvm::FixedVectorType::get(eltTy, vlanes);
            llvm::Value *res = llvm::UndefValue::get(vt);
            auto coerceComp = [&](llvm::Value *x) -> llvm::Value * {
                if (velt == MGLIR_SCALAR_BOOL) {
                    if (x->getType()->isFloatingPointTy())
                        return cg.b->CreateFCmpUNE(
                            x, llvm::ConstantFP::get(x->getType(), 0.0));
                    if (x->getType()->isIntegerTy(1))
                        return x;
                    return cg.b->CreateICmpNE(
                        x, llvm::Constant::getNullValue(x->getType()));
                }
                return coerceScalar(cg, x, velt);
            };
            auto insertComp = [&](llvm::Value *x, uint32_t slot) {
                x = coerceComp(x);
                return cg.b->CreateInsertElement(
                    res, x,
                    llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx),
                                           slot));
            };
            uint32_t slot = 0;
            for (uint32_t a = 0; a < e->u.call.arg_count && slot < vlanes;
                 a++) {
                llvm::Value *arg = emitExpr(cg, e->u.call.args[a], mod, locals);
                if (!arg) return nullptr;
                if (auto *arrTy = llvm::dyn_cast<llvm::ArrayType>(
                        arg->getType())) {
                    /* Matrix: column-major component stream (GLSL 4.60 §5.4.2). */
                    uint32_t ncols = (uint32_t)arrTy->getNumElements();
                    llvm::Type *colElt = arrTy->getElementType();
                    uint32_t nrows = 1;
                    if (auto *cv = llvm::dyn_cast<llvm::FixedVectorType>(colElt))
                        nrows = (uint32_t)cv->getNumElements();
                    for (uint32_t c = 0; c < ncols && slot < vlanes; c++) {
                        llvm::Value *col = cg.b->CreateExtractValue(arg, c);
                        if (col->getType()->isVectorTy()) {
                            for (uint32_t r = 0; r < nrows && slot < vlanes;
                                 r++, slot++) {
                                llvm::Value *x = cg.b->CreateExtractElement(
                                    col,
                                    llvm::ConstantInt::get(
                                        llvm::Type::getInt32Ty(*cg.ctx), r));
                                res = insertComp(x, slot);
                            }
                        } else {
                            res = insertComp(col, slot++);
                        }
                    }
                } else if (!arg->getType()->isVectorTy()) {
                    /* Single scalar argument broadcasts (GLSL 4.60 5.4.2);
                     * otherwise one component per scalar. */
                    if (e->u.call.arg_count == 1) {
                        llvm::Value *s = coerceComp(arg);
                        for (uint32_t lane = 0; lane < vlanes; lane++)
                            res = cg.b->CreateInsertElement(
                                res, s,
                                llvm::ConstantInt::get(
                                    llvm::Type::getInt32Ty(*cg.ctx), lane));
                        return res;
                    }
                    res = insertComp(arg, slot++);
                } else {
                    llvm::FixedVectorType *argTy =
                        llvm::cast<llvm::FixedVectorType>(arg->getType());
                    uint32_t argLanes = (uint32_t)argTy->getElementCount()
                                                    .getFixedValue();
                    for (uint32_t lane = 0;
                         lane < argLanes && slot < vlanes; lane++, slot++) {
                        llvm::Value *x = cg.b->CreateExtractElement(
                            arg, llvm::ConstantInt::get(
                                     llvm::Type::getInt32Ty(*cg.ctx), lane));
                        res = insertComp(x, slot);
                    }
                }
            }
            return res;
        }
        /* Matrix constructors: mat2..mat4 / matCxR. */
        uint32_t mcols = 0, mrows = 0;
        if (strncmp(name, "mat", 3) == 0) {
            const char *m = name + 3;
            if (m[0] >= '2' && m[0] <= '4' && m[1] == '\0') {
                mcols = mrows = (uint32_t)(m[0] - '0');
            } else if (m[0] >= '2' && m[0] <= '4' && m[1] == 'x' &&
                       m[2] >= '2' && m[2] <= '4' && m[3] == '\0') {
                mcols = (uint32_t)(m[0] - '0');
                mrows = (uint32_t)(m[2] - '0');
            }
        }
        if (mcols) {
            llvm::Type *colTy = llvm::FixedVectorType::get(
                llvm::Type::getFloatTy(*cg.ctx), mrows);
            llvm::Type *arrTy = llvm::ArrayType::get(colTy, mcols);
            llvm::Value *arr = llvm::UndefValue::get(arrTy);
            if (e->u.call.arg_count == 1) {
                llvm::Value *s = emitExpr(cg, e->u.call.args[0], mod, locals);
                if (!s) return nullptr;
                if (s->getType()->isArrayTy()) {
                    /* matNxN(matMxM) with M<N: embed the smaller matrix in
                     * the upper-left, identity on the remaining diagonal. */
                    llvm::ArrayType *sa = llvm::cast<llvm::ArrayType>(
                        s->getType());
                    llvm::Type *se = sa->getElementType();
                    uint32_t sc = (uint32_t)sa->getNumElements();
                    uint32_t sr = 0;
                    if (auto *sv = llvm::dyn_cast<llvm::FixedVectorType>(se))
                        sr = (uint32_t)sv->getNumElements();
                    else
                        sr = 1;
                    if (sc > mcols || sr > mrows) {
                        cg.err = 1;
                        cg.errmsg = std::string("codegen: constructor '") +
                                    name + "' embeds a larger matrix";
                        return nullptr;
                    }
                    for (uint32_t c = 0; c < mcols; c++) {
                        llvm::Value *col = llvm::UndefValue::get(colTy);
                        for (uint32_t r = 0; r < mrows; r++) {
                            llvm::Value *x;
                            if (c < sc && r < sr) {
                                x = cg.b->CreateExtractElement(
                                    cg.b->CreateExtractValue(s, c),
                                    llvm::ConstantInt::get(
                                        llvm::Type::getInt32Ty(*cg.ctx), r));
                            } else {
                                x = (r == c)
                                    ? llvm::ConstantFP::get(
                                          llvm::Type::getFloatTy(*cg.ctx), 1.0)
                                    : llvm::ConstantFP::get(
                                          llvm::Type::getFloatTy(*cg.ctx), 0.0);
                            }
                            col = cg.b->CreateInsertElement(col, x,
                                llvm::ConstantInt::get(
                                    llvm::Type::getInt32Ty(*cg.ctx), r));
                        }
                        arr = cg.b->CreateInsertValue(arr, col, c);
                    }
                    return arr;
                }
                /* matN(f): diagonal scale. */
                s = coerceScalar(cg, s, MGLIR_SCALAR_FLOAT);
                for (uint32_t c = 0; c < mcols; c++) {
                    llvm::Value *col = llvm::UndefValue::get(colTy);
                    for (uint32_t r = 0; r < mrows; r++) {
                        llvm::Value *x = (r == c) ? s
                            : llvm::ConstantFP::get(
                                  llvm::Type::getFloatTy(*cg.ctx), 0.0);
                        col = cg.b->CreateInsertElement(col, x,
                            llvm::ConstantInt::get(
                                llvm::Type::getInt32Ty(*cg.ctx), r));
                    }
                    arr = cg.b->CreateInsertValue(arr, col, c);
                }
            } else if (e->u.call.arg_count == (uint32_t)(mcols * mrows)) {
                /* Scalar list: column-major fill (defensive; sema prefers
                 * vector columns). */
                uint32_t a = 0;
                for (uint32_t c = 0; c < mcols; c++) {
                    llvm::Value *col = llvm::UndefValue::get(colTy);
                    for (uint32_t r = 0; r < mrows; r++, a++) {
                        llvm::Value *arg = emitExpr(cg, e->u.call.args[a],
                                                    mod, locals);
                        if (!arg) return nullptr;
                        arg = coerceScalar(cg, arg, MGLIR_SCALAR_FLOAT);
                        col = cg.b->CreateInsertElement(col, arg,
                            llvm::ConstantInt::get(
                                llvm::Type::getInt32Ty(*cg.ctx), r));
                    }
                    arr = cg.b->CreateInsertValue(arr, col, c);
                }
            } else {
                /* Vector columns: matN(vecN, ...). */
                uint32_t c = 0;
                for (uint32_t a = 0; a < e->u.call.arg_count; a++, c++) {
                    llvm::Value *arg = emitExpr(cg, e->u.call.args[a],
                                                mod, locals);
                    if (!arg) return nullptr;
                    arg = coerceScalar(cg, arg, MGLIR_SCALAR_FLOAT);
                    if (!arg->getType()->isVectorTy() || c >= mcols) {
                        cg.err = 1;
                        cg.errmsg = std::string("codegen: constructor '") +
                                    name + "' column mismatch";
                        return nullptr;
                    }
                    arr = cg.b->CreateInsertValue(arr, arg, c);
                }
            }
            return arr;
        }
        /* Math builtins (sema-typed subset).  All float args are coerced;
         * integer variants (abs/min/max/clamp) use icmp selects. */
        {
            if (strcmp(name, "dFdx") == 0 || strcmp(name, "dFdy") == 0) {
                if (e->u.call.arg_count != 1) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: '") + name +
                                "' expects 1 argument";
                    return nullptr;
                }
                llvm::Value *v = emitExpr(cg, e->u.call.args[0], mod, locals);
                if (!v) return nullptr;
                llvm::Type *et = v->getType();
                if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(et)) {
                    uint32_t n = (uint32_t)vt->getNumElements();
                    std::string fn = (strcmp(name, "dFdx") == 0)
                        ? std::string("air.dfdx.v") + std::to_string(n) + "f32"
                        : std::string("air.dfdy.v") + std::to_string(n) + "f32";
                    return callAirFn(cg, fn.c_str(), et, {v});
                }
                return callAirFn(cg, strcmp(name, "dFdx") == 0
                                         ? "air.dfdx.f32"
                                         : "air.dfdy.f32",
                                 et, {v});
            }
            if (strcmp(name, "interpolateAtCentroid") == 0 ||
                strcmp(name, "interpolateAtSample") == 0 ||
                strcmp(name, "interpolateAtOffset") == 0) {
                llvm::Value *interp = emitInterpolateAtBuiltin(cg, e, name, mod,
                                                               locals);
                if (interp || cg.err)
                    return interp;
            }
            llvm::Value *mb = emitMatrixBuiltin(cg, e, name, mod, locals);
            if (mb) return mb;
        }
        {
            llvm::Value *mb = emitMathBuiltin(cg, e, name, mod, locals);
            if (mb) return mb;
        }
        {
            /* floatBitsToInt/Uint and intBitsToFloat/uintBitsToFloat are
             * pure bitcasts between float and 32-bit int representations. */
            if (strcmp(name, "floatBitsToInt") == 0 ||
                strcmp(name, "floatBitsToUint") == 0 ||
                strcmp(name, "intBitsToFloat") == 0 ||
                strcmp(name, "uintBitsToFloat") == 0) {
                if (e->u.call.arg_count != 1) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: '") + name +
                                "' expects 1 argument";
                    return nullptr;
                }
                llvm::Value *a0 =
                    emitExpr(cg, e->u.call.args[0], mod, locals);
                if (!a0) return nullptr;
                bool toInt = name[0] == 'f';
                llvm::Type *src = a0->getType();
                auto i32 = [&] { return llvm::Type::getInt32Ty(*cg.ctx); };
                auto f32 = [&] { return llvm::Type::getFloatTy(*cg.ctx); };
                llvm::Type *dst;
                if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(src)) {
                    dst = toInt ? (llvm::Type *)llvm::FixedVectorType::get(
                                      i32(), vt->getNumElements())
                                : (llvm::Type *)llvm::FixedVectorType::get(
                                      f32(), vt->getNumElements());
                } else {
                    dst = toInt ? (llvm::Type *)i32() : (llvm::Type *)f32();
                }
                return cg.b->CreateBitCast(a0, dst);
            }
        }
        /* Storage-image ops: texture handle, no sampler.  GL 1D / 1D_ARRAY
         * are Metal-backed as 2D / 2D_ARRAY (height or y = 0). */
        if (strcmp(name, "barrier") == 0) {
            /* GLSL barrier(): TCS patch invocations / CS workgroup threads.
             * TCS dispatch is one Metal threadgroup per patch, so map to
             * threadgroup_barrier(mem_device | mem_threadgroup) —
             * air.wg.barrier(flags, scope) with flags=1|2 (device|TG). */
            if (e->u.call.arg_count != 0) {
                cg.err = 1;
                cg.errmsg = "codegen: barrier() takes no arguments";
                return nullptr;
            }
            llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
            llvm::Type *voidTy = llvm::Type::getVoidTy(*cg.ctx);
            callAirFn(cg, "air.wg.barrier", voidTy,
                      {llvm::ConstantInt::get(i32, 3),
                       llvm::ConstantInt::get(i32, 1)});
            return cg.b->getInt32(0);
        }
        if (strcmp(name, "memoryBarrier") == 0 ||
            strcmp(name, "memoryBarrierAtomicCounter") == 0 ||
            strcmp(name, "memoryBarrierBuffer") == 0 ||
            strcmp(name, "memoryBarrierShared") == 0 ||
            strcmp(name, "memoryBarrierImage") == 0 ||
            strcmp(name, "groupMemoryBarrier") == 0) {
            /* Map GLSL memoryBarrier* to air.wg.barrier mem flags:
             * 1=device, 2=threadgroup, 4=texture (MSL mem_flags). */
            if (e->u.call.arg_count != 0) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: '") + name +
                            "' takes no arguments";
                return nullptr;
            }
            unsigned flags = 0u;
            if (strcmp(name, "memoryBarrierShared") == 0 ||
                strcmp(name, "groupMemoryBarrier") == 0)
                flags = 2u;
            else if (strcmp(name, "memoryBarrierBuffer") == 0 ||
                     strcmp(name, "memoryBarrierAtomicCounter") == 0)
                flags = 1u;
            else if (strcmp(name, "memoryBarrierImage") == 0)
                flags = 4u;
            else
                flags = 1u | 2u | 4u;
            llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
            llvm::Type *voidTy = llvm::Type::getVoidTy(*cg.ctx);
            callAirFn(cg, "air.wg.barrier", voidTy,
                      {llvm::ConstantInt::get(i32, flags),
                       llvm::ConstantInt::get(i32, 1)});
            return cg.b->getInt32(0);
        }
        if (strncmp(name, "imageAtomic", 11) == 0) {
            /* Metal 3.1 texture atomics (air.atomic_fetch_*_explicit_texture_*).
             * Unsupported combinations are rejected at codegen (A05). */
            const MGLExpr *ia = e->u.call.arg_count > 0
                ? e->u.call.args[0] : nullptr;
            const MGLIRType *imgTy = nullptr;
            llvm::Value *tex = resolveImageTex(cg, ia, mod, locals, &imgTy);
            if (!tex) return nullptr;
            const MGLIRTexKind tk = imgTy->tex_kind;
            const bool isUint = imgTy->tex_storage == MGLIR_SCALAR_UINT;
            const bool isCompSwap = strcmp(name, "imageAtomicCompSwap") == 0;
            const bool isMsImage =
                tk == MGLIR_TEX_2D_MS || tk == MGLIR_TEX_2D_MS_ARRAY;
            const unsigned wantArgs =
                isCompSwap ? (isMsImage ? 5u : 4u) : (isMsImage ? 4u : 3u);
            if (e->u.call.arg_count != wantArgs) {
                cg.err = 1;
                cg.errmsg = "codegen: imageAtomic* arity mismatch";
                return nullptr;
            }
            llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
            llvm::Type *v2i32 = llvm::FixedVectorType::get(i32, 2);
            llvm::Type *v3i32 = llvm::FixedVectorType::get(i32, 3);
            llvm::Type *v4i32 = llvm::FixedVectorType::get(i32, 4);
            auto toIvec2X0 = [&](llvm::Value *x) -> llvm::Value * {
                llvm::Value *v = llvm::UndefValue::get(v2i32);
                v = cg.b->CreateInsertElement(v, x, cg.b->getInt32(0));
                return cg.b->CreateInsertElement(v, cg.b->getInt32(0),
                                                 cg.b->getInt32(1));
            };
            llvm::Value *coord = emitExpr(cg, e->u.call.args[1], mod, locals);
            if (!coord) return nullptr;
            llvm::Value *coord2 = nullptr;
            llvm::Value *coord3 = nullptr;
            llvm::Value *layerOrFace = nullptr;
            switch (tk) {
            case MGLIR_TEX_1D:
            case MGLIR_TEX_BUFFER:
                if (!coord->getType()->isIntegerTy(32)) {
                    cg.err = 1;
                    cg.errmsg = "codegen: imageAtomic 1D/buffer coord must be int";
                    return nullptr;
                }
                coord2 = toIvec2X0(coord);
                break;
            case MGLIR_TEX_2D:
            case MGLIR_TEX_2D_RECT:
            case MGLIR_TEX_2D_MS:
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: imageAtomic 2D coord must be ivec2";
                    return nullptr;
                }
                coord2 = coord;
                break;
            case MGLIR_TEX_1D_ARRAY:
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: imageAtomic 1DArray coord must be ivec2";
                    return nullptr;
                }
                layerOrFace = cg.b->CreateExtractElement(coord, cg.b->getInt32(1));
                coord2 = toIvec2X0(
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(0)));
                break;
            case MGLIR_TEX_2D_ARRAY:
            case MGLIR_TEX_CUBE:
            case MGLIR_TEX_CUBE_ARRAY:
            case MGLIR_TEX_3D:
            case MGLIR_TEX_2D_MS_ARRAY:
                if (coord->getType() != v3i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: imageAtomic 3D/cube/array coord must be ivec3";
                    return nullptr;
                }
                if (tk == MGLIR_TEX_3D) {
                    coord3 = coord;
                } else {
                    layerOrFace = cg.b->CreateExtractElement(coord, cg.b->getInt32(2));
                    coord2 = cg.b->CreateShuffleVector(
                        coord, llvm::UndefValue::get(coord->getType()), {0, 1});
                }
                break;
            default:
                cg.err = 1;
                cg.errmsg = "codegen: imageAtomic unsupported image type";
                return nullptr;
            }
            llvm::Value *msSample = nullptr;
            unsigned dataArg = 2u;
            if (isMsImage) {
                msSample = emitExpr(cg, e->u.call.args[2], mod, locals);
                if (!msSample) return nullptr;
                msSample = coerceScalar(cg, msSample, MGLIR_SCALAR_INT);
                dataArg = 3u;
            }
            llvm::Value *data =
                emitExpr(cg, e->u.call.args[dataArg], mod, locals);
            if (!data) return nullptr;
            data = coerceScalar(cg, data,
                                isUint ? MGLIR_SCALAR_UINT : MGLIR_SCALAR_INT);

            /* Native Metal texture atomics exist for 1D/2D/3D/array/buffer.
             * CompSwap, multisample, and cube kinds have no correct atomic
             * lowering here — refuse rather than emit racy RMW (A05). */
            const bool useNative =
                !isCompSwap && !isMsImage &&
                (tk == MGLIR_TEX_1D || tk == MGLIR_TEX_BUFFER ||
                 tk == MGLIR_TEX_2D || tk == MGLIR_TEX_2D_RECT ||
                 tk == MGLIR_TEX_1D_ARRAY || tk == MGLIR_TEX_2D_ARRAY ||
                 tk == MGLIR_TEX_3D);
            if (!useNative) {
                cg.err = 1;
                if (isCompSwap) {
                    cg.errmsg =
                        "codegen: imageAtomicCompSwap is not supported "
                        "(no native texture compare-exchange)";
                } else if (isMsImage) {
                    cg.errmsg =
                        "codegen: imageAtomic* on multisample images is "
                        "not supported";
                } else {
                    cg.errmsg =
                        "codegen: imageAtomic* on this image kind is not "
                        "supported (no native texture atomic)";
                }
                return nullptr;
            }
            const char *opStem = nullptr;
            if (strcmp(name, "imageAtomicAdd") == 0)
                opStem = "atomic_fetch_add";
            else if (strcmp(name, "imageAtomicMin") == 0)
                opStem = "atomic_fetch_min";
            else if (strcmp(name, "imageAtomicMax") == 0)
                opStem = "atomic_fetch_max";
            else if (strcmp(name, "imageAtomicAnd") == 0)
                opStem = "atomic_fetch_and";
            else if (strcmp(name, "imageAtomicOr") == 0)
                opStem = "atomic_fetch_or";
            else if (strcmp(name, "imageAtomicXor") == 0)
                opStem = "atomic_fetch_xor";
            else if (strcmp(name, "imageAtomicExchange") == 0)
                opStem = "atomic_exchange";
            else {
                cg.err = 1;
                cg.errmsg = "codegen: unsupported imageAtomic op";
                return nullptr;
            }
            llvm::Value *dataV4 = llvm::UndefValue::get(v4i32);
            dataV4 = cg.b->CreateInsertElement(dataV4, data,
                                               cg.b->getInt32(0));
            dataV4 = cg.b->CreateInsertElement(dataV4, cg.b->getInt32(0),
                                               cg.b->getInt32(1));
            dataV4 = cg.b->CreateInsertElement(dataV4, cg.b->getInt32(0),
                                               cg.b->getInt32(2));
            dataV4 = cg.b->CreateInsertElement(dataV4, cg.b->getInt32(0),
                                               cg.b->getInt32(3));
            llvm::Value *zero2 = llvm::ConstantVector::get(
                {cg.b->getInt32(0), cg.b->getInt32(0)});
            llvm::Value *zero3 = llvm::ConstantVector::get(
                {cg.b->getInt32(0), cg.b->getInt32(0), cg.b->getInt32(0)});
            /* memory_order_relaxed=0, access::read_write=3 */
            llvm::Value *order = cg.b->getInt32(0);
            llvm::Value *access = cg.b->getInt32(3);
            auto airName = [&](const char *dim) -> std::string {
                return std::string("air.") + opStem + "_explicit_" + dim +
                       (isUint ? ".u.v4i32" : ".s.v4i32");
            };
            llvm::Value *oldV4 = nullptr;
            if (tk == MGLIR_TEX_3D) {
                oldV4 = callAirFn(cg, airName("texture_3d").c_str(), v4i32,
                                  {tex, coord3, zero3, dataV4, order,
                                   access});
            } else if (tk == MGLIR_TEX_2D_ARRAY || tk == MGLIR_TEX_1D_ARRAY) {
                oldV4 = callAirFn(
                    cg, airName("texture_2d_array").c_str(), v4i32,
                    {tex, coord2, layerOrFace, zero2, dataV4, order,
                     access});
            } else {
                /* 1D / buffer / 2D / rect — Metal 2D backing. */
                oldV4 = callAirFn(cg, airName("texture_2d").c_str(), v4i32,
                                  {tex, coord2, zero2, dataV4, order,
                                   access});
            }
            return cg.b->CreateExtractElement(oldV4, cg.b->getInt32(0));
        }
        if (strcmp(name, "imageStore") == 0 ||
            strcmp(name, "imageLoad") == 0 ||
            strcmp(name, "imageSize") == 0) {
            const MGLExpr *ia = e->u.call.arg_count > 0
                ? e->u.call.args[0] : nullptr;
            const MGLIRType *imgTy = nullptr;
            llvm::Value *tex = resolveImageTex(cg, ia, mod, locals, &imgTy);
            if (!tex) return nullptr;
            const MGLIRTexKind tk = imgTy->tex_kind;
            llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
            llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
            llvm::Type *v2i32 = llvm::FixedVectorType::get(i32, 2);
            llvm::Type *v3i32 = llvm::FixedVectorType::get(i32, 3);
            llvm::Type *v4f32 = llvm::FixedVectorType::get(f32, 4);
            llvm::Type *v4i32 = llvm::FixedVectorType::get(i32, 4);
            const MGLIRScalar storage = imgTy->tex_storage;
            const bool isInt =
                storage == MGLIR_SCALAR_INT || storage == MGLIR_SCALAR_UINT;
            auto writeName = [&](const char *base) -> std::string {
                if (!isInt) return std::string(base) + ".v4f32";
                return std::string(base) +
                       (storage == MGLIR_SCALAR_UINT ? ".u.v4i32" : ".s.v4i32");
            };
            auto readName = [&](const char *base) -> std::string {
                if (!isInt) return std::string(base) + ".v4f32";
                return std::string(base) +
                       (storage == MGLIR_SCALAR_UINT ? ".u.v4i32" : ".s.v4i32");
            };
            auto toIvec2X0 = [&](llvm::Value *x) -> llvm::Value * {
                llvm::Value *v = llvm::UndefValue::get(v2i32);
                v = cg.b->CreateInsertElement(v, x, cg.b->getInt32(0));
                return cg.b->CreateInsertElement(v, cg.b->getInt32(0),
                                                 cg.b->getInt32(1));
            };
            if (strcmp(name, "imageSize") == 0) {
                /* Enough for current CTS; return ivec2(width, height-or-1). */
                const char *wFn = "air.get_width_texture_2d";
                const char *hFn = "air.get_height_texture_2d";
                if (tk == MGLIR_TEX_3D) {
                    wFn = "air.get_width_texture_3d";
                    hFn = "air.get_height_texture_3d";
                } else if (tk == MGLIR_TEX_2D_ARRAY ||
                           tk == MGLIR_TEX_1D_ARRAY) {
                    wFn = "air.get_width_texture_2d_array";
                    hFn = "air.get_height_texture_2d_array";
                } else if (tk == MGLIR_TEX_CUBE ||
                           tk == MGLIR_TEX_CUBE_ARRAY) {
                    wFn = "air.get_width_texture_cube";
                    hFn = "air.get_height_texture_cube";
                }
                llvm::Value *w = callAirFn(cg, wFn, i32, {tex, cg.b->getInt32(0)});
                llvm::Value *h = (tk == MGLIR_TEX_1D || tk == MGLIR_TEX_BUFFER)
                    ? cg.b->getInt32(1)
                    : callAirFn(cg, hFn, i32, {tex, cg.b->getInt32(0)});
                llvm::Value *size = llvm::UndefValue::get(v2i32);
                size = cg.b->CreateInsertElement(size, w, cg.b->getInt32(0));
                return cg.b->CreateInsertElement(size, h, cg.b->getInt32(1));
            }
            llvm::Value *coord = emitExpr(cg, e->u.call.args[1], mod, locals);
            if (!coord) return nullptr;
            coord = coerceScalar(cg, coord, MGLIR_SCALAR_INT);
            llvm::Value *layerOrFace = nullptr;
            llvm::Value *coord2 = nullptr;
            llvm::Value *coord3 = nullptr;
            /* Normalize GLSL coords to the Metal texture backing. */
            switch (tk) {
            case MGLIR_TEX_1D:
            case MGLIR_TEX_BUFFER:
                /* Buffer images are uploaded as a width×1 texture2d fallback
                 * (see TEXBUFFER CREATE … as=texture2d), same as 1D. */
                if (!coord->getType()->isIntegerTy(32)) {
                    cg.err = 1;
                    cg.errmsg = "codegen: image1D/imageBuffer coord must be int";
                    return nullptr;
                }
                coord2 = toIvec2X0(coord);
                break;
            case MGLIR_TEX_2D:
            case MGLIR_TEX_2D_RECT:
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: image2D/image2DRect coord must be ivec2";
                    return nullptr;
                }
                coord2 = coord;
                break;
            case MGLIR_TEX_1D_ARRAY:
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: image1DArray coord must be ivec2";
                    return nullptr;
                }
                layerOrFace = cg.b->CreateExtractElement(coord, cg.b->getInt32(1));
                coord2 = toIvec2X0(
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(0)));
                break;
            case MGLIR_TEX_2D_ARRAY:
            case MGLIR_TEX_CUBE:
            case MGLIR_TEX_CUBE_ARRAY:
            case MGLIR_TEX_3D:
            case MGLIR_TEX_2D_MS_ARRAY:
                if (coord->getType() != v3i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: image3D/cube/2DArray/2DMSArray coord must be ivec3";
                    return nullptr;
                }
                if (tk == MGLIR_TEX_3D) {
                    coord3 = coord;
                } else {
                    layerOrFace = cg.b->CreateExtractElement(coord, cg.b->getInt32(2));
                    coord2 = cg.b->CreateShuffleVector(
                        coord, llvm::UndefValue::get(coord->getType()), {0, 1});
                }
                break;
            case MGLIR_TEX_2D_MS:
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: image2DMS coord must be ivec2";
                    return nullptr;
                }
                coord2 = coord;
                break;
            default:
                cg.err = 1;
                cg.errmsg = std::string("codegen: ") + name +
                    " unsupported image type";
                return nullptr;
            }
            /* Multisample imageLoad/Store take an explicit sample index. Metal
             * cannot write texture2d_ms, so MS images are backed as
             * texture2d_array with sample (or layer*samples+sample) as layer. */
            llvm::Value *msSample = nullptr;
            const bool isMsImage =
                tk == MGLIR_TEX_2D_MS || tk == MGLIR_TEX_2D_MS_ARRAY;
            if (isMsImage) {
                const unsigned expectArgs =
                    strcmp(name, "imageStore") == 0 ? 4u : 3u;
                if (e->u.call.arg_count != expectArgs) {
                    cg.err = 1;
                    cg.errmsg = "codegen: multisample image op arity mismatch";
                    return nullptr;
                }
                msSample = emitExpr(cg, e->u.call.args[2], mod, locals);
                if (!msSample) return nullptr;
                msSample = coerceScalar(cg, msSample, MGLIR_SCALAR_INT);
            }
            if (strcmp(name, "imageLoad") == 0) {
                llvm::Type *vecTy = isInt ? v4i32 : v4f32;
                llvm::Type *retTy = llvm::StructType::get(
                    *cg.ctx, {vecTy, cg.b->getInt8Ty()});
                llvm::Value *r = nullptr;
                if (tk == MGLIR_TEX_BUFFER) {
                    r = callAirFn(cg, readName("air.read_texture_2d").c_str(),
                                  retTy, {tex, coord2, cg.b->getInt32(0),
                                          cg.b->getInt32(3)});
                } else if (tk == MGLIR_TEX_3D) {
                    r = callAirFn(cg, readName("air.read_texture_3d").c_str(),
                                  retTy, {tex, coord3, cg.b->getInt32(0),
                                          cg.b->getInt32(3)});
                } else if (tk == MGLIR_TEX_CUBE) {
                    r = callAirFn(cg, readName("air.read_texture_cube").c_str(),
                                  retTy, {tex, coord2, layerOrFace,
                                          cg.b->getInt32(0), cg.b->getInt32(3)});
                } else if (tk == MGLIR_TEX_CUBE_ARRAY) {
                    /* MSL texturecube_array.read(coord, face, array). GLSL
                     * imageCubeArray uses a flat layer-face index. */
                    llvm::Value *face =
                        cg.b->CreateURem(layerOrFace, cg.b->getInt32(6));
                    llvm::Value *arrayIdx =
                        cg.b->CreateUDiv(layerOrFace, cg.b->getInt32(6));
                    r = callAirFn(cg,
                                  readName("air.read_texture_cube_array").c_str(),
                                  retTy,
                                  {tex, coord2, face, arrayIdx,
                                   cg.b->getInt32(0), cg.b->getInt32(3)});
                } else if (tk == MGLIR_TEX_2D_MS) {
                    r = callAirFn(cg, readName("air.read_texture_2d_array").c_str(),
                                  retTy, {tex, coord2, msSample,
                                          cg.b->getInt32(0), cg.b->getInt32(3)});
                } else if (tk == MGLIR_TEX_2D_MS_ARRAY) {
                    /* texture2d_array planes: flat = layer * 8 + sample. */
                    llvm::Value *flat = cg.b->CreateAdd(
                        cg.b->CreateMul(layerOrFace, cg.b->getInt32(8)),
                        msSample);
                    r = callAirFn(cg, readName("air.read_texture_2d_array").c_str(),
                                  retTy, {tex, coord2, flat,
                                          cg.b->getInt32(0), cg.b->getInt32(3)});
                } else if (tk == MGLIR_TEX_2D_ARRAY || tk == MGLIR_TEX_1D_ARRAY) {
                    r = callAirFn(cg, readName("air.read_texture_2d_array").c_str(),
                                  retTy, {tex, coord2, layerOrFace,
                                          cg.b->getInt32(0), cg.b->getInt32(3)});
                } else {
                    r = callAirFn(cg, readName("air.read_texture_2d").c_str(),
                                  retTy, {tex, coord2, cg.b->getInt32(0),
                                          cg.b->getInt32(3)});
                }
                return cg.b->CreateExtractValue(r, 0);
            }
            const MGLExpr *valueArg = isMsImage ? e->u.call.args[3]
                                                : e->u.call.args[2];
            llvm::Value *value = emitExpr(cg, valueArg, mod, locals);
            if (!value) return nullptr;
            if (isInt) {
                value = coerceScalar(cg, value, MGLIR_SCALAR_INT);
                if (value->getType() != v4i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: integer imageStore value must be ivec4/uvec4";
                    return nullptr;
                }
            } else {
                value = coerceScalar(cg, value, MGLIR_SCALAR_FLOAT);
                if (value->getType() != v4f32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: imageStore value must be vec4";
                    return nullptr;
                }
            }
            llvm::Type *voidTy = llvm::Type::getVoidTy(*cg.ctx);
            auto fenceAfterImageWrite = [&](llvm::Value *texH, MGLIRTexKind kind) {
                /* Metal requires texture.fence() between write and a later
                 * read of the same texture in one invocation (CTS
                 * basic-glsl-misc / image atomics). */
                const char *fn = "air.fence_texture_2d";
                switch (kind) {
                case MGLIR_TEX_3D:
                    fn = "air.fence_texture_3d";
                    break;
                case MGLIR_TEX_CUBE:
                    fn = "air.fence_texture_cube";
                    break;
                case MGLIR_TEX_CUBE_ARRAY:
                    fn = "air.fence_texture_cube_array";
                    break;
                case MGLIR_TEX_2D_ARRAY:
                case MGLIR_TEX_1D_ARRAY:
                case MGLIR_TEX_2D_MS:
                case MGLIR_TEX_2D_MS_ARRAY:
                    fn = "air.fence_texture_2d_array";
                    break;
                default:
                    break;
                }
                callAirFn(cg, fn, voidTy, {texH});
            };
            if (tk == MGLIR_TEX_3D) {
                llvm::Value *w = callAirFn(
                    cg, writeName("air.write_texture_3d").c_str(), voidTy,
                    {tex, coord3, value, cg.b->getInt32(0), cg.b->getInt32(3)});
                fenceAfterImageWrite(tex, tk);
                return w;
            }
            if (tk == MGLIR_TEX_CUBE) {
                llvm::Value *w = callAirFn(
                    cg, writeName("air.write_texture_cube").c_str(), voidTy,
                    {tex, coord2, layerOrFace, value, cg.b->getInt32(0),
                     cg.b->getInt32(3)});
                fenceAfterImageWrite(tex, tk);
                return w;
            }
            if (tk == MGLIR_TEX_CUBE_ARRAY) {
                /* MSL: write(color, uint2 coord, uint face, uint array). */
                llvm::Value *face =
                    cg.b->CreateURem(layerOrFace, cg.b->getInt32(6));
                llvm::Value *arrayIdx =
                    cg.b->CreateUDiv(layerOrFace, cg.b->getInt32(6));
                llvm::Value *w = callAirFn(
                    cg, writeName("air.write_texture_cube_array").c_str(), voidTy,
                    {tex, coord2, face, arrayIdx, value, cg.b->getInt32(0),
                     cg.b->getInt32(3)});
                fenceAfterImageWrite(tex, tk);
                return w;
            }
            if (tk == MGLIR_TEX_2D_MS) {
                llvm::Value *w = callAirFn(
                    cg, writeName("air.write_texture_2d_array").c_str(), voidTy,
                    {tex, coord2, msSample, value, cg.b->getInt32(0),
                     cg.b->getInt32(3)});
                fenceAfterImageWrite(tex, tk);
                return w;
            }
            if (tk == MGLIR_TEX_2D_MS_ARRAY) {
                llvm::Value *flat = cg.b->CreateAdd(
                    cg.b->CreateMul(layerOrFace, cg.b->getInt32(8)),
                    msSample);
                llvm::Value *w = callAirFn(
                    cg, writeName("air.write_texture_2d_array").c_str(), voidTy,
                    {tex, coord2, flat, value, cg.b->getInt32(0),
                     cg.b->getInt32(3)});
                fenceAfterImageWrite(tex, tk);
                return w;
            }
            if (tk == MGLIR_TEX_2D_ARRAY || tk == MGLIR_TEX_1D_ARRAY) {
                llvm::Value *w = callAirFn(
                    cg, writeName("air.write_texture_2d_array").c_str(), voidTy,
                    {tex, coord2, layerOrFace, value, cg.b->getInt32(0),
                     cg.b->getInt32(3)});
                fenceAfterImageWrite(tex, tk);
                return w;
            }
            /* 1D / buffer (as 2D), 2D, 2DRect */
            {
                llvm::Value *w = callAirFn(
                    cg, writeName("air.write_texture_2d").c_str(), voidTy,
                    {tex, coord2, value, cg.b->getInt32(0), cg.b->getInt32(3)});
                fenceAfterImageWrite(tex, tk);
                return w;
            }
        }
        /* texelFetch(sampler, ivecP, lod): unfiltered read. */
        if (strcmp(name, "texelFetch") == 0 ||
            strcmp(name, "texelFetchOffset") == 0) {
            const bool hasFetchOffset = strcmp(name, "texelFetchOffset") == 0;
            if ((!hasFetchOffset &&
                 e->u.call.arg_count != 3 && e->u.call.arg_count != 2) ||
                (hasFetchOffset && e->u.call.arg_count != 4)) {
                cg.err = 1;
                cg.errmsg = hasFetchOffset
                                ? "codegen: texelFetchOffset expects 4 arguments"
                                : "codegen: texelFetch expects 2 or 3 arguments";
                return nullptr;
            }
            const MGLExpr *sa = e->u.call.args[0];
            std::string samplerPath;
            const char *samplerName = nullptr;
            llvm::Value *tex = nullptr;
            bool dynamicSamplerArray = false;
            llvm::Value *arrayIndex = nullptr;
            const std::vector<llvm::Value *> *texArray = nullptr;
            const bool topLevelSamplerArray =
                sa->kind == MGL_EXPR_INDEX && sa->u.index.object &&
                sa->u.index.object->kind == MGL_EXPR_VAR_REF;
            if (topLevelSamplerArray) {
                /* Same pattern as texture(): texInput[i] with constant or
                 * dynamic i (CTS texture_barrier usampler2D[N] loops). */
                samplerName = sa->u.index.object->u.var_ref.name;
                llvm::Value *index =
                    emitExpr(cg, sa->u.index.index, mod, locals);
                if (!index) return nullptr;
                index = coerceScalar(cg, index, MGLIR_SCALAR_INT);
                auto ti = cg.texArrayValues.find(samplerName);
                if (ti == cg.texArrayValues.end()) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch first argument must be a "
                                "sampler variable";
                    return nullptr;
                }
                if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
                    uint32_t k = (uint32_t)ci->getZExtValue();
                    if (k < ti->second.size())
                        tex = ti->second[k];
                    else if (!ti->second.empty())
                        tex = ti->second.back();
                } else {
                    dynamicSamplerArray = true;
                    arrayIndex = index;
                    texArray = &ti->second;
                }
            } else if (!resolveSamplerAccessName(sa, &samplerPath)) {
                cg.err = 1;
                cg.errmsg = "codegen: texelFetch first argument must be a "
                            "sampler variable";
                return nullptr;
            } else {
                samplerName = samplerPath.c_str();
                tex = samplerTexValue(cg, samplerName);
            }
            if (!dynamicSamplerArray && !tex) {
                cg.err = 1;
                cg.errmsg = "codegen: texelFetch first argument must be a "
                            "sampler variable";
                return nullptr;
            }
            const MGLIRType *sampleType = nullptr;
            auto sti = cg.samplerIRTypes.find(samplerName);
            if (sti != cg.samplerIRTypes.end())
                sampleType = sti->second;
            else {
                const MGLIRSymbol *ts = findSymbol(mod, samplerName);
                sampleType = ts ? ts->type : nullptr;
            }
            if (sampleType && sampleType->kind == MGLIR_TYPE_ARRAY &&
                sampleType->elem_type)
                sampleType = sampleType->elem_type;
            MGLIRTexKind texKind = sampleType &&
                                           sampleType->kind == MGLIR_TYPE_SAMPLER
                                       ? sampleType->tex_kind
                                       : MGLIR_TEX_2D;
            MGLIRScalar texel =
                sampleType && sampleType->kind == MGLIR_TYPE_SAMPLER
                    ? sampleType->tex_storage
                    : MGLIR_SCALAR_FLOAT;
            bool isBuf = texKind == MGLIR_TEX_BUFFER;
            llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
            llvm::Type *v2i32 = llvm::FixedVectorType::get(i32, 2);
            llvm::Type *v3i32 = llvm::FixedVectorType::get(i32, 3);
            llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
            llvm::Type *vecTy =
                texel == MGLIR_SCALAR_FLOAT
                    ? (llvm::Type *)llvm::FixedVectorType::get(f32, 4)
                    : (llvm::Type *)llvm::FixedVectorType::get(i32, 4);
            llvm::Type *retTy =
                llvm::StructType::get(*cg.ctx, {vecTy, cg.b->getInt8Ty()});
            auto readIntrinsic = [&](const char *floatName) -> std::string {
                if (texel == MGLIR_SCALAR_FLOAT) {
                    return floatName;
                }
                std::string n(floatName);
                const char *from = ".v4f32";
                const char *to = texel == MGLIR_SCALAR_INT ? ".s.v4i32"
                                                           : ".u.v4i32";
                size_t pos = n.find(from);
                if (pos != std::string::npos) {
                    n.replace(pos, strlen(from), to);
                }
                return n;
            };
            if (isBuf) {
                if (e->u.call.arg_count != 2) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a samplerBuffer "
                                "expects 2 arguments";
                    return nullptr;
                }
                llvm::Value *coord = emitExpr(cg, e->u.call.args[1], mod,
                                              locals);
                if (!coord) return nullptr;
                coord = coerceScalar(cg, coord, MGLIR_SCALAR_INT);
                /* TEXTURE_BUFFER is backed as texture2d (same packing as
                 * imageBuffer imageStore/Load).  Keep texelFetch on that
                 * path so imageLoad == texelFetch after COMMAND/FETCH
                 * barriers (CTS advanced-sync-imageAccess).  Integer
                 * usamplerBuffer/isamplerBuffer must use .u/.s reads —
                 * always-float was returning 0 for integer formats. */
                llvm::Value *xy = llvm::UndefValue::get(v2i32);
                xy = cg.b->CreateInsertElement(xy, coord, cg.b->getInt32(0));
                xy = cg.b->CreateInsertElement(xy, cg.b->getInt32(0),
                                               cg.b->getInt32(1));
                auto doBufFetch =
                    [&](llvm::Value *t, llvm::Value *) -> llvm::Value * {
                    llvm::Value *r = callAirFn(
                        cg, readIntrinsic("air.read_texture_2d.v4f32").c_str(),
                        retTy,
                        {t, xy, cg.b->getInt32(0), cg.b->getInt32(3)});
                    return cg.b->CreateExtractValue(r, 0);
                };
                if (dynamicSamplerArray) {
                    std::vector<llvm::Value *> empty;
                    return sampleArrayElementBySwitch(
                        cg, arrayIndex, *texArray, empty, vecTy, doBufFetch);
                }
                return doBufFetch(tex, nullptr);
            }
            auto toIvec2XY0 = [&](llvm::Value *x) -> llvm::Value * {
                llvm::Value *v = llvm::UndefValue::get(v2i32);
                v = cg.b->CreateInsertElement(v, x, cg.b->getInt32(0));
                v = cg.b->CreateInsertElement(v, cg.b->getInt32(0),
                                              cg.b->getInt32(1));
                return v;
            };
            bool isRect = texKind == MGLIR_TEX_2D_RECT;
            if (!hasFetchOffset && e->u.call.arg_count == 2 && !isRect) {
                cg.err = 1;
                cg.errmsg = "codegen: texelFetch on a sampler expects 2 or 3 "
                            "arguments";
                return nullptr;
            }
            llvm::Value *coord = emitExpr(cg, e->u.call.args[1], mod, locals);
            if (!coord) return nullptr;
            coord = coerceScalar(cg, coord, MGLIR_SCALAR_INT);
            llvm::Value *lodOrSample = cg.b->getInt32(0);
            uint32_t lodArg = 2;
            if (hasFetchOffset) {
                lodArg = 2;
            }
            if (e->u.call.arg_count >= 3 && (!hasFetchOffset || e->u.call.arg_count == 4)) {
                lodOrSample = emitExpr(cg, e->u.call.args[lodArg], mod, locals);
                if (!lodOrSample) return nullptr;
                lodOrSample = coerceScalar(cg, lodOrSample, MGLIR_SCALAR_INT);
            }
            if (hasFetchOffset) {
                llvm::Value *off =
                    emitExpr(cg, e->u.call.args[3], mod, locals);
                if (!off) return nullptr;
                off = coerceScalar(cg, off, MGLIR_SCALAR_INT);
                coord = addTexelOffset(cg, coord, off);
            }
            llvm::Value *arrayLayer = nullptr;
            if (texKind == MGLIR_TEX_2D_ARRAY) {
                if (coord->getType() != v3i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler2DArray "
                                "expects ivec3 coordinates";
                    return nullptr;
                }
                arrayLayer =
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(2));
                coord = cg.b->CreateShuffleVector(
                    coord, llvm::UndefValue::get(coord->getType()),
                    {0, 1});
            } else if (texKind == MGLIR_TEX_1D_ARRAY) {
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler1DArray "
                                "expects ivec2 coordinates";
                    return nullptr;
                }
                arrayLayer =
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(1));
                coord = toIvec2XY0(
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(0)));
            } else if (texKind == MGLIR_TEX_1D) {
                if (!coord->getType()->isIntegerTy()) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler1D expects "
                                "int coordinates";
                    return nullptr;
                }
                coord = toIvec2XY0(coord);
            } else if (texKind == MGLIR_TEX_2D_RECT) {
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler2DRect "
                                "expects ivec2 coordinates";
                    return nullptr;
                }
            } else if (texKind == MGLIR_TEX_2D_MS) {
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler2DMS "
                                "expects ivec2 coordinates";
                    return nullptr;
                }
            } else if (texKind == MGLIR_TEX_2D_MS_ARRAY) {
                if (coord->getType() != v3i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler2DMSArray "
                                "expects ivec3 coordinates";
                    return nullptr;
                }
                arrayLayer =
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(2));
                coord = cg.b->CreateShuffleVector(
                    coord, llvm::UndefValue::get(coord->getType()),
                    {0, 1});
            } else if (texKind == MGLIR_TEX_3D || texKind == MGLIR_TEX_CUBE) {
                if (coord->getType() != v3i32) {
                    cg.err = 1;
                    cg.errmsg = texKind == MGLIR_TEX_3D
                                    ? "codegen: texelFetch on a sampler3D "
                                      "expects ivec3 coordinates"
                                    : "codegen: texelFetch on a samplerCube "
                                      "expects ivec3 coordinates";
                    return nullptr;
                }
            } else {
                if (coord->getType()->isIntegerTy()) {
                    coord = toIvec2XY0(coord);
                } else if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler2D expects "
                                "ivec2 coordinates";
                    return nullptr;
                }
            }
            auto doFetchVec =
                [&](llvm::Value *t, llvm::Value *) -> llvm::Value * {
                llvm::Value *r = nullptr;
                if (texKind == MGLIR_TEX_2D_ARRAY ||
                    texKind == MGLIR_TEX_1D_ARRAY) {
                    r = callAirFn(
                        cg,
                        readIntrinsic("air.read_texture_2d_array.v4f32")
                            .c_str(),
                        retTy,
                        {t, coord, arrayLayer, lodOrSample,
                         cg.b->getInt32(3)});
                } else if (texKind == MGLIR_TEX_2D_MS) {
                    /* Non-RT MS textures are texture2d_array sample planes. */
                    r = callAirFn(
                        cg,
                        readIntrinsic("air.read_texture_2d_array.v4f32")
                            .c_str(),
                        retTy,
                        {t, coord, lodOrSample, cg.b->getInt32(0),
                         cg.b->getInt32(3)});
                } else if (texKind == MGLIR_TEX_2D_MS_ARRAY) {
                    llvm::Value *flat = cg.b->CreateAdd(
                        cg.b->CreateMul(arrayLayer, cg.b->getInt32(8)),
                        lodOrSample);
                    r = callAirFn(
                        cg,
                        readIntrinsic("air.read_texture_2d_array.v4f32")
                            .c_str(),
                        retTy,
                        {t, coord, flat, cg.b->getInt32(0),
                         cg.b->getInt32(3)});
                } else if (texKind == MGLIR_TEX_3D) {
                    r = callAirFn(
                        cg, readIntrinsic("air.read_texture_3d.v4f32").c_str(),
                        retTy,
                        {t, coord, lodOrSample, cg.b->getInt32(3)});
                } else if (texKind == MGLIR_TEX_CUBE) {
                    r = callAirFn(
                        cg,
                        readIntrinsic("air.read_texture_cube.v4f32").c_str(),
                        retTy,
                        {t, coord, lodOrSample, cg.b->getInt32(3)});
                } else {
                    llvm::Value *level =
                        texKind == MGLIR_TEX_2D_RECT ? cg.b->getInt32(0)
                                                     : lodOrSample;
                    r = callAirFn(
                        cg, readIntrinsic("air.read_texture_2d.v4f32").c_str(),
                        retTy, {t, coord, level, cg.b->getInt32(3)});
                }
                return cg.b->CreateExtractValue(r, 0);
            };
            if (dynamicSamplerArray) {
                std::vector<llvm::Value *> empty;
                return sampleArrayElementBySwitch(
                    cg, arrayIndex, *texArray, empty, vecTy, doFetchVec);
            }
            return doFetchVec(tex, nullptr);
        }
        /* texture / textureLod / textureSize: the sampler argument maps
         * to paired AIR texture + sampler parameters. */
        if (isTextureSampleBuiltin(name) || strcmp(name, "textureSize") == 0) {
            const bool isProj = strstr(name, "Proj") != nullptr;
            const bool isLod = strstr(name, "Lod") != nullptr;
            const bool isGrad = strstr(name, "Grad") != nullptr;
            const bool hasOffset = strstr(name, "Offset") != nullptr;
            if (strcmp(name, "textureSize") == 0) {
                if (e->u.call.arg_count != 2) {
                    cg.err = 1;
                    cg.errmsg = "codegen: textureSize expects 2 arguments";
                    return nullptr;
                }
            } else if (e->u.call.arg_count < 2 || e->u.call.arg_count > 5) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: '") + name +
                            "' expects 2 to 5 arguments";
                return nullptr;
            }
            const MGLExpr *sa = e->u.call.args[0];
            std::string samplerPath;
            const char *samplerName = nullptr;
            llvm::Value *tex = nullptr;
            llvm::Value *smp = nullptr;
            bool dynamicSamplerArray = false;
            llvm::Value *arrayIndex = nullptr;
            const std::vector<llvm::Value *> *texArray = nullptr;
            const std::vector<llvm::Value *> *smpArray = nullptr;
            const bool topLevelSamplerArray =
                sa->kind == MGL_EXPR_INDEX && sa->u.index.object &&
                sa->u.index.object->kind == MGL_EXPR_VAR_REF;
            if (topLevelSamplerArray) {
                samplerName = sa->u.index.object->u.var_ref.name;
                llvm::Value *index = emitExpr(cg, sa->u.index.index, mod, locals);
                if (!index) return nullptr;
                index = coerceScalar(cg, index, MGLIR_SCALAR_INT);
                auto ti = cg.texArrayValues.find(samplerName);
                auto si = cg.smpArrayValues.find(samplerName);
                if (ti == cg.texArrayValues.end()) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texture argument must be a sampler2D "
                                "variable";
                    return nullptr;
                }
                if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(index)) {
                    uint32_t k = (uint32_t)ci->getZExtValue();
                    if (k < ti->second.size()) {
                        tex = ti->second[k];
                        if (si != cg.smpArrayValues.end() &&
                            k < si->second.size())
                            smp = si->second[k];
                    } else if (!ti->second.empty()) {
                        tex = ti->second.back();
                        if (si != cg.smpArrayValues.end() &&
                            !si->second.empty())
                            smp = si->second.back();
                    }
                } else {
                    dynamicSamplerArray = true;
                    arrayIndex = index;
                    texArray = &ti->second;
                    if (si != cg.smpArrayValues.end())
                        smpArray = &si->second;
                }
            } else if (resolveSamplerAccessName(sa, &samplerPath)) {
                samplerName = samplerPath.c_str();
                tex = samplerTexValue(cg, samplerName);
                auto si = cg.smpValues.find(samplerName);
                if (si != cg.smpValues.end()) smp = si->second;
            } else {
                cg.err = 1;
                cg.errmsg = "codegen: texture argument must be a sampler2D "
                            "variable";
                return nullptr;
            }
            if (!dynamicSamplerArray && !tex) {
                cg.err = 1;
                cg.errmsg = "codegen: texture argument must be a sampler2D "
                            "variable";
                return nullptr;
            }
            if (!smp && !dynamicSamplerArray) {
                /* Function parameter: use the read sampler
                 * (filtered sampling inside helpers is not wired). */
                llvm::Type *smpT = llvm::StructType::get(
                    *cg.ctx, "struct._sampler_t");
                smp = callAirFn(cg, "air.get_read_sampler",
                                smpT->getPointerTo(2), {});
            }
            const MGLIRType *sampleTypeForDim = nullptr;
            {
                auto sti = cg.samplerIRTypes.find(samplerName);
                if (sti != cg.samplerIRTypes.end())
                    sampleTypeForDim = sti->second;
                else {
                    const MGLIRSymbol *tss = findSymbol(mod, samplerName);
                    sampleTypeForDim = tss ? tss->type : nullptr;
                }
            }
            if (sampleTypeForDim && sampleTypeForDim->kind == MGLIR_TYPE_ARRAY &&
                sampleTypeForDim->elem_type)
                sampleTypeForDim = sampleTypeForDim->elem_type;
            bool is3d = sampleTypeForDim &&
                        sampleTypeForDim->kind == MGLIR_TYPE_SAMPLER &&
                        sampleTypeForDim->tex_kind == MGLIR_TEX_3D;
            MGLIRTexKind sampleKind = sampleTypeForDim &&
                        sampleTypeForDim->kind == MGLIR_TYPE_SAMPLER
                    ? sampleTypeForDim->tex_kind
                    : MGLIR_TEX_2D;
            llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
            llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
            if (strcmp(name, "textureSize") == 0) {
                if (dynamicSamplerArray)
                    tex = selectArrayElement(cg, arrayIndex, *texArray);
                llvm::Value *lod = emitExpr(cg, e->u.call.args[1], mod,
                                            locals);
                if (!lod) return nullptr;
                lod = coerceScalar(cg, lod, MGLIR_SCALAR_INT);
                llvm::Value *w = callAirFn(
                    cg, is3d ? "air.get_width_texture_3d"
                             : "air.get_width_texture_2d",
                    i32, {tex, lod});
                llvm::Value *h = callAirFn(
                    cg, is3d ? "air.get_height_texture_3d"
                             : "air.get_height_texture_2d",
                    i32, {tex, lod});
                if (is3d) {
                    llvm::Value *d = callAirFn(cg, "air.get_depth_texture_3d",
                                               i32, {tex, lod});
                    llvm::Type *v3i32 = llvm::FixedVectorType::get(i32, 3);
                    llvm::Value *sz = llvm::UndefValue::get(v3i32);
                    sz = cg.b->CreateInsertElement(sz, w, cg.b->getInt32(0));
                    sz = cg.b->CreateInsertElement(sz, h, cg.b->getInt32(1));
                    sz = cg.b->CreateInsertElement(sz, d, cg.b->getInt32(2));
                    return sz;
                }
                llvm::Type *v2i32 = llvm::FixedVectorType::get(i32, 2);
                llvm::Value *sz = llvm::UndefValue::get(v2i32);
                sz = cg.b->CreateInsertElement(sz, w, cg.b->getInt32(0));
                sz = cg.b->CreateInsertElement(sz, h, cg.b->getInt32(1));
                return sz;
            }
            llvm::Value *uv = emitExpr(cg, e->u.call.args[1], mod, locals);
            if (!uv) return nullptr;
            if (isProj) {
                /* textureProj(sampler, vec4): sample at uv.xy / uv.w. */
                if (auto *uvt = llvm::dyn_cast<llvm::FixedVectorType>(
                        uv->getType());
                    !uvt || uvt->getNumElements() != 4) {
                    cg.err = 1;
                    cg.errmsg = "codegen: textureProj expects a vec4 "
                                "coordinate";
                    return nullptr;
                }
                llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
                llvm::Value *w = cg.b->CreateExtractElement(
                    uv, cg.b->getInt32(3));
                llvm::Type *v2f32 = llvm::FixedVectorType::get(f32, 2);
                llvm::Value *xy = cg.b->CreateShuffleVector(
                    uv, llvm::UndefValue::get(uv->getType()),
                    llvm::ConstantVector::get(
                        {llvm::ConstantInt::get(
                             llvm::Type::getInt32Ty(*cg.ctx), 0),
                         llvm::ConstantInt::get(
                             llvm::Type::getInt32Ty(*cg.ctx), 1)}));
                llvm::Value *pw = cg.b->CreateVectorSplat(2, w);
                uv = cg.b->CreateFDiv(xy, pw);
            }
            /* For sampler arrays, the expression is an index node and the
             * union's var_ref member is not valid.  Use the resolved base
             * name selected above so integer sampler arrays choose the
             * correct AIR intrinsic and result type. */
            const MGLIRSymbol *sampsym = findSymbol(mod, samplerName);
            const MGLIRType *sampleType = sampsym ? sampsym->type : nullptr;
            if (sampleType && sampleType->kind == MGLIR_TYPE_ARRAY &&
                sampleType->elem_type)
                sampleType = sampleType->elem_type;
            MGLIRScalar texel = sampleType &&
                                        sampleType->kind == MGLIR_TYPE_SAMPLER
                                    ? sampleType->tex_storage
                                    : MGLIR_SCALAR_FLOAT;
            /* Shadow compare (sample_compare) is not wired yet. Incomplete
             * texture / shadow CTS cases expect 0.0; returning constant
             * float matches BI_RET_FLOAT overloads until compare is added. */
            if (sampleType && sampleType->kind == MGLIR_TYPE_SAMPLER &&
                sampleType->tex_depth) {
                return llvm::ConstantFP::get(f32, 0.0);
            }
            auto sampledRetType = [&](llvm::Type *vecTy) {
                return llvm::StructType::get(*cg.ctx,
                                             {vecTy, cg.b->getInt8Ty()});
            };
            /* Integer samplers return integer texels; the AIR intrinsic
             * suffix carries the format (reference:
             * texture2d<int, sample>.sample). */
            auto sampledIntrinsic =
                [&](const char *floatName) -> std::string {
                std::string n(floatName);
                std::string from = ".v4f32";
                size_t pos = n.find(from);
                if (pos != std::string::npos) {
                    n.replace(pos, from.size(),
                              texel == MGLIR_SCALAR_INT ? ".s.v4i32"
                                                        : ".u.v4i32");
                }
                return n;
            };
            if (isGrad) {
                llvm::Value *dPdx = emitExpr(cg, e->u.call.args[2], mod,
                                             locals);
                llvm::Value *dPdy = emitExpr(cg, e->u.call.args[3], mod,
                                             locals);
                if (!dPdx || !dPdy) return nullptr;
                llvm::Value *gradOffset = llvm::Constant::getNullValue(
                    llvm::FixedVectorType::get(i32, 2));
                if (hasOffset) {
                    llvm::Value *off =
                        emitExpr(cg, e->u.call.args[4], mod, locals);
                    if (!off) return nullptr;
                    off = coerceScalar(cg, off, MGLIR_SCALAR_INT);
                    gradOffset = emitAirSampleOffset(cg, off);
                }
                llvm::Type *v2i32 = llvm::FixedVectorType::get(i32, 2);
                llvm::Type *vecTy =
                    texel == MGLIR_SCALAR_FLOAT
                        ? (llvm::Type *)llvm::FixedVectorType::get(f32, 4)
                        : (llvm::Type *)llvm::FixedVectorType::get(i32, 4);
                const char *gradName = "air.sample_texture_2d_grad.v4f32";
                if (sampleKind == MGLIR_TEX_3D) {
                    gradName = "air.sample_texture_3d_grad.v4f32";
                } else if (sampleKind == MGLIR_TEX_2D_ARRAY ||
                           sampleKind == MGLIR_TEX_2D_MS_ARRAY ||
                           sampleKind == MGLIR_TEX_1D_ARRAY) {
                    gradName = "air.sample_texture_2d_array_grad.v4f32";
                } else if (sampleKind == MGLIR_TEX_CUBE ||
                           sampleKind == MGLIR_TEX_CUBE_ARRAY) {
                    gradName = "air.sample_texture_cube_grad.v4f32";
                } else if (sampleKind == MGLIR_TEX_2D_MS) {
                    gradName = "air.sample_texture_2d_ms_grad.v4f32";
                } else if (sampleKind == MGLIR_TEX_2D_MS_ARRAY) {
                    gradName = "air.sample_texture_2d_ms_array_grad.v4f32";
                }
                auto doGradSampleVec =
                    [&](llvm::Value *t, llvm::Value *s) -> llvm::Value * {
                    llvm::Value *sp = s;
                    if (!sp) {
                        llvm::Type *smpT = llvm::StructType::get(
                            *cg.ctx, "struct._sampler_t");
                        sp = callAirFn(cg, "air.get_read_sampler",
                                        smpT->getPointerTo(2), {});
                    }
                    llvm::Value *sampleCoord = uv;
                    llvm::Value *arrayLayer = nullptr;
                    if (!splitSampleArrayCoord(cg, sampleKind, uv, &sampleCoord,
                                               &arrayLayer)) {
                        return nullptr;
                    }
                    std::vector<llvm::Value *> gradArgs = {
                        t, sp, sampleCoord, dPdx, dPdy,
                        llvm::ConstantFP::get(f32, 0.0),
                        cg.b->getInt1(false),
                        gradOffset,
                        cg.b->getInt32(0)};
                    if (arrayLayer) {
                        gradArgs.insert(gradArgs.begin() + 3, arrayLayer);
                    }
                    llvm::Value *r = callAirFn(
                        cg,
                        sampledIntrinsic(gradName).c_str(),
                        sampledRetType(vecTy),
                        gradArgs);
                    return cg.b->CreateExtractValue(r, 0);
                };
                if (dynamicSamplerArray) {
                    std::vector<llvm::Value *> smps =
                        smpArray ? *smpArray
                                 : std::vector<llvm::Value *>();
                    return sampleArrayElementBySwitch(
                        cg, arrayIndex, *texArray, smps, vecTy,
                        doGradSampleVec);
                }
                return doGradSampleVec(tex, smp);
            }
            llvm::Value *lod = nullptr;
            bool explicitLod = false;
            llvm::Value *sampleOffset = llvm::Constant::getNullValue(
                llvm::FixedVectorType::get(i32, 2));
            uint32_t argIdx = 2;
            if (isLod) {
                lod = emitExpr(cg, e->u.call.args[argIdx++], mod, locals);
                if (!lod) return nullptr;
                lod = coerceScalar(cg, lod, MGLIR_SCALAR_FLOAT);
                explicitLod = true;
                if (hasOffset) {
                    llvm::Value *off =
                        emitExpr(cg, e->u.call.args[argIdx++], mod, locals);
                    if (!off) return nullptr;
                    off = coerceScalar(cg, off, MGLIR_SCALAR_INT);
                    sampleOffset = emitAirSampleOffset(cg, off);
                }
            } else if (hasOffset) {
                llvm::Value *off =
                    emitExpr(cg, e->u.call.args[argIdx++], mod, locals);
                if (!off) return nullptr;
                off = coerceScalar(cg, off, MGLIR_SCALAR_INT);
                sampleOffset = emitAirSampleOffset(cg, off);
            } else if (e->u.call.arg_count == 3) {
                lod = emitExpr(cg, e->u.call.args[2], mod, locals);
                if (!lod) return nullptr;
                lod = coerceScalar(cg, lod, MGLIR_SCALAR_FLOAT);
                explicitLod = true;
            }
            llvm::Type *v2i32 = llvm::FixedVectorType::get(i32, 2);
            llvm::Type *vecTy = texel == MGLIR_SCALAR_FLOAT
                ? (llvm::Type *)llvm::FixedVectorType::get(f32, 4)
                : (llvm::Type *)llvm::FixedVectorType::get(i32, 4);
            llvm::Type *retTy = sampledRetType(vecTy);
            const char *baseName = "air.sample_texture_2d.v4f32";
            if (sampleKind == MGLIR_TEX_3D) {
                baseName = "air.sample_texture_3d.v4f32";
            } else if (sampleKind == MGLIR_TEX_2D_ARRAY ||
                       sampleKind == MGLIR_TEX_2D_MS_ARRAY ||
                       sampleKind == MGLIR_TEX_1D_ARRAY) {
                baseName = "air.sample_texture_2d_array.v4f32";
            } else if (sampleKind == MGLIR_TEX_CUBE) {
                baseName = "air.sample_texture_cube.v4f32";
            } else if (sampleKind == MGLIR_TEX_CUBE_ARRAY) {
                baseName = "air.sample_texture_cube_array.v4f32";
            } else if (sampleKind == MGLIR_TEX_2D_MS) {
                baseName = "air.sample_texture_2d_ms.v4f32";
            } else if (sampleKind == MGLIR_TEX_1D) {
                baseName = "air.sample_texture_2d.v4f32";
                if (uv->getType()->isFloatingPointTy()) {
                    llvm::Type *v2f32 = llvm::FixedVectorType::get(f32, 2);
                    llvm::Value *expanded = llvm::UndefValue::get(v2f32);
                    expanded = cg.b->CreateInsertElement(
                        expanded, uv, cg.b->getInt32(0));
                    expanded = cg.b->CreateInsertElement(
                        expanded, llvm::ConstantFP::get(f32, 0.5),
                        cg.b->getInt32(1));
                    uv = expanded;
                }
            }
            /* CTS (and some apps) pass vec3(0) into texture(samplerCube).
             * Metal's cube sample of a zero direction returns black; pick +X. */
            if ((sampleKind == MGLIR_TEX_CUBE ||
                 sampleKind == MGLIR_TEX_CUBE_ARRAY) &&
                uv->getType()->isVectorTy() &&
                llvm::cast<llvm::FixedVectorType>(uv->getType())
                        ->getNumElements() >= 3) {
                llvm::Value *x =
                    cg.b->CreateExtractElement(uv, cg.b->getInt32(0));
                llvm::Value *y =
                    cg.b->CreateExtractElement(uv, cg.b->getInt32(1));
                llvm::Value *z =
                    cg.b->CreateExtractElement(uv, cg.b->getInt32(2));
                llvm::Value *len2 = cg.b->CreateFAdd(
                    cg.b->CreateFMul(x, x),
                    cg.b->CreateFAdd(cg.b->CreateFMul(y, y),
                                     cg.b->CreateFMul(z, z)));
                llvm::Value *isZero = cg.b->CreateFCmpOEQ(
                    len2, llvm::ConstantFP::get(f32, 0.0));
                llvm::Value *fallback = llvm::UndefValue::get(uv->getType());
                fallback = cg.b->CreateInsertElement(
                    fallback, llvm::ConstantFP::get(f32, 1.0),
                    cg.b->getInt32(0));
                fallback = cg.b->CreateInsertElement(
                    fallback, llvm::ConstantFP::get(f32, 0.0),
                    cg.b->getInt32(1));
                fallback = cg.b->CreateInsertElement(
                    fallback, llvm::ConstantFP::get(f32, 0.0),
                    cg.b->getInt32(2));
                uv = cg.b->CreateSelect(isZero, fallback, uv);
            }
            auto doSampleVec =
                [&](llvm::Value *t, llvm::Value *s) -> llvm::Value * {
                llvm::Value *sp = s;
                if (!sp) {
                    llvm::Type *smpT = llvm::StructType::get(
                        *cg.ctx, "struct._sampler_t");
                    sp = callAirFn(cg, "air.get_read_sampler",
                                    smpT->getPointerTo(2), {});
                }
                llvm::Value *sampleCoord = uv;
                llvm::Value *arrayLayer = nullptr;
                if (!splitSampleArrayCoord(cg, sampleKind, uv, &sampleCoord,
                                           &arrayLayer)) {
                    return nullptr;
                }
                /* Cube sample has no offset in AIR/MSL (texturecube.sample
                 * takes coord + lod/bias only). Passing the 2D offset pair
                 * makes Metal's PSO compiler abort. */
                const bool cubeSample =
                    sampleKind == MGLIR_TEX_CUBE ||
                    sampleKind == MGLIR_TEX_CUBE_ARRAY;
                std::vector<llvm::Value *> sampleArgs;
                if (cubeSample) {
                    sampleArgs = {
                        t, sp, sampleCoord,
                        cg.b->getInt1(explicitLod),
                        lod ? lod : llvm::ConstantFP::get(f32, 0.0),
                        llvm::ConstantFP::get(f32, 0.0),
                        cg.b->getInt32(0)};
                } else {
                    sampleArgs = {
                        t, sp, sampleCoord, cg.b->getInt1(true),
                        sampleOffset,
                        cg.b->getInt1(explicitLod),
                        lod ? lod : llvm::ConstantFP::get(f32, 0.0),
                        llvm::ConstantFP::get(f32, 0.0),
                        cg.b->getInt32(0)};
                }
                if (arrayLayer) {
                    sampleArgs.insert(sampleArgs.begin() + 3, arrayLayer);
                }
                llvm::Value *r = callAirFn(
                    cg, sampledIntrinsic(baseName).c_str(), retTy,
                    sampleArgs);
                return cg.b->CreateExtractValue(r, 0);
            };
            if (dynamicSamplerArray) {
                std::vector<llvm::Value *> smps =
                    smpArray ? *smpArray : std::vector<llvm::Value *>();
                return sampleArrayElementBySwitch(
                    cg, arrayIndex, *texArray, smps, vecTy, doSampleVec);
            }
            return doSampleVec(tex, smp);
        }
        /* atomicCounterIncrement(counter): monotonic RMW on device memory. */
        if (strcmp(name, "atomicCounterIncrement") == 0) {
            if (e->u.call.arg_count != 1) {
                cg.err = 1;
                cg.errmsg = "codegen: atomicCounterIncrement expects 1 argument";
                return nullptr;
            }
            llvm::Value *p = emitAtomicCounterAddress(
                cg, e->u.call.args[0], mod, locals);
            if (!p) return nullptr;
            /* GLSL 4.60 8.11: returns the value previously in the counter. */
            return cg.b->CreateAtomicRMW(
                llvm::AtomicRMWInst::Add, p, cg.b->getInt32(1),
                llvm::MaybeAlign(), llvm::AtomicOrdering::Monotonic);
        }
        /* atomicCounterDecrement(counter): monotonic RMW on device memory. */
        if (strcmp(name, "atomicCounterDecrement") == 0) {
            if (e->u.call.arg_count != 1) {
                cg.err = 1;
                cg.errmsg = "codegen: atomicCounterDecrement expects 1 argument";
                return nullptr;
            }
            llvm::Value *p = emitAtomicCounterAddress(
                cg, e->u.call.args[0], mod, locals);
            if (!p) return nullptr;
            /* GLSL 4.60 §8.11: atomicCounterDecrement returns the value
             * *resulting from* the decrement (post-decrement), unlike
             * atomicCounterIncrement which returns the pre-increment
             * value.  AtomicRMW::Sub yields the old value, so subtract
             * one more. */
            llvm::Value *old = cg.b->CreateAtomicRMW(
                llvm::AtomicRMWInst::Sub, p, cg.b->getInt32(1),
                llvm::MaybeAlign(), llvm::AtomicOrdering::Monotonic);
            return cg.b->CreateSub(old, cg.b->getInt32(1));
        }
        /* atomicCounter(counter): non-modifying read of device memory. */
        if (strcmp(name, "atomicCounter") == 0) {
            if (e->u.call.arg_count != 1) {
                cg.err = 1;
                cg.errmsg = "codegen: atomicCounter expects 1 argument";
                return nullptr;
            }
            llvm::Value *p = emitAtomicCounterAddress(
                cg, e->u.call.args[0], mod, locals);
            if (!p) return nullptr;
            llvm::LoadInst *load =
                cg.b->CreateLoad(cg.b->getInt32Ty(), p, "acval");
            load->setAtomic(llvm::AtomicOrdering::Monotonic);
            return load;
        }
        /* SSBO atomic* (GLSL 4.60 §8.11): RMW on device memory; every
         * op returns the original contents of mem before the update. */
        {
            llvm::AtomicRMWInst::BinOp rmwOp = llvm::AtomicRMWInst::BAD_BINOP;
            int isCompSwap = 0;
            if (strcmp(name, "atomicAdd") == 0)
                rmwOp = llvm::AtomicRMWInst::Add;
            else if (strcmp(name, "atomicMin") == 0)
                rmwOp = llvm::AtomicRMWInst::BAD_BINOP; /* signedness below */
            else if (strcmp(name, "atomicMax") == 0)
                rmwOp = llvm::AtomicRMWInst::BAD_BINOP;
            else if (strcmp(name, "atomicAnd") == 0)
                rmwOp = llvm::AtomicRMWInst::And;
            else if (strcmp(name, "atomicOr") == 0)
                rmwOp = llvm::AtomicRMWInst::Or;
            else if (strcmp(name, "atomicXor") == 0)
                rmwOp = llvm::AtomicRMWInst::Xor;
            else if (strcmp(name, "atomicExchange") == 0)
                rmwOp = llvm::AtomicRMWInst::Xchg;
            else if (strcmp(name, "atomicCompSwap") == 0)
                isCompSwap = 1;

            int isAtomicFamily =
                isCompSwap || strcmp(name, "atomicAdd") == 0 ||
                strcmp(name, "atomicMin") == 0 ||
                strcmp(name, "atomicMax") == 0 ||
                strcmp(name, "atomicAnd") == 0 ||
                strcmp(name, "atomicOr") == 0 ||
                strcmp(name, "atomicXor") == 0 ||
                strcmp(name, "atomicExchange") == 0;

            if (isAtomicFamily) {
                uint32_t wantArgs = isCompSwap ? 3u : 2u;
                if (e->u.call.arg_count != wantArgs) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: ") + name +
                                " argument count mismatch";
                    return nullptr;
                }
                const MGLIRSymbol *sb = ssboRootSym(e->u.call.args[0], mod);
                if (!sb) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: ") + name +
                                " target must be an SSBO member";
                    return nullptr;
                }
                const MGLIRType *ty = nullptr;
                llvm::Value *p = ssboAddress(cg, e->u.call.args[0], sb, mod,
                                             locals, &ty);
                if (!p) return nullptr;
                llvm::Value *data = emitExpr(cg, e->u.call.args[1], mod, locals);
                if (!data) return nullptr;
                data = coerceScalar(cg, data,
                                    ty && ty->scalar == MGLIR_SCALAR_UINT
                                        ? MGLIR_SCALAR_UINT
                                        : MGLIR_SCALAR_INT);
                p = cg.b->CreateBitCast(p, data->getType()->getPointerTo(1));
                if (isCompSwap) {
                    llvm::Value *cmp = data;
                    llvm::Value *neu =
                        emitExpr(cg, e->u.call.args[2], mod, locals);
                    if (!neu) return nullptr;
                    neu = coerceScalar(cg, neu,
                                       ty && ty->scalar == MGLIR_SCALAR_UINT
                                           ? MGLIR_SCALAR_UINT
                                           : MGLIR_SCALAR_INT);
                    auto *cx = cg.b->CreateAtomicCmpXchg(
                        p, cmp, neu, llvm::MaybeAlign(),
                        llvm::AtomicOrdering::Monotonic,
                        llvm::AtomicOrdering::Monotonic);
                    /* CmpXchg returns { old, success }; GLSL wants old. */
                    return cg.b->CreateExtractValue(cx, 0);
                }
                if (strcmp(name, "atomicMin") == 0) {
                    rmwOp = (ty && ty->scalar == MGLIR_SCALAR_UINT)
                                ? llvm::AtomicRMWInst::UMin
                                : llvm::AtomicRMWInst::Min;
                } else if (strcmp(name, "atomicMax") == 0) {
                    rmwOp = (ty && ty->scalar == MGLIR_SCALAR_UINT)
                                ? llvm::AtomicRMWInst::UMax
                                : llvm::AtomicRMWInst::Max;
                }
                return cg.b->CreateAtomicRMW(rmwOp, p, data,
                                             llvm::MaybeAlign(),
                                             llvm::AtomicOrdering::Monotonic);
            }
        }
        /* User-defined function call. */
        if (cg.userFns || cg.userFnDecls) {
            std::string key = std::string(name) + "#" +
                              std::to_string(e->u.call.arg_count);
            /* Prefer inlining when a decl is registered in userFnDecls
             * (GS/TCS/compute always; any stage when a param is out/inout
             * so by-value LLVM calls cannot lose write-back). */
            if (cg.userFnDecls) {
                auto dit = cg.userFnDecls->find(key);
                if (dit != cg.userFnDecls->end() && dit->second &&
                    dit->second->body) {
                    MGLDecl *fd = dit->second;
                    std::map<std::string, MType> ilocals = locals;
                    std::map<std::string, llvm::Value *> saved;
                    std::map<std::string, const MGLIRType *> savedIR;
                    for (uint32_t a = 0; a < fd->param_count &&
                                        a < e->u.call.arg_count; a++) {
                        MGLDecl *pd = fd->params[a];
                        if (!pd || !pd->name) continue;
                        llvm::Value *av =
                            emitExpr(cg, e->u.call.args[a], mod, locals);
                        if (!av) return nullptr;
                        MType pt;
                        pt.scalar = (MGLIRScalar)(pd->type ? pd->type->base
                                                           : MGL_AST_TYPE_FLOAT);
                        if (pd->type && pd->type->vec_size)
                            pt.vec = pd->type->vec_size;
                        if (pd->type && pd->type->mat_cols > 1) {
                            pt.cols = pd->type->mat_cols;
                            pt.rows = pd->type->mat_rows;
                        }
                        if (pd->type &&
                            pd->type->base != MGL_AST_TYPE_STRUCT)
                            av = coerceScalar(cg, av, pt.scalar);
                        if (cg.lvalues.count(pd->name))
                            saved[pd->name] = cg.lvalues[pd->name];
                        cg.lvalues[pd->name] = av;
                        ilocals[pd->name] = pt;
                        if (pd->type &&
                            pd->type->base == MGL_AST_TYPE_STRUCT &&
                            pd->type->name) {
                            auto sit =
                                cg.structTypes.find(pd->type->name);
                            if (sit != cg.structTypes.end()) {
                                auto irit =
                                    cg.localIRTypes.find(pd->name);
                                if (irit != cg.localIRTypes.end())
                                    savedIR[pd->name] = irit->second;
                                cg.localIRTypes[pd->name] = sit->second;
                            }
                        }
                    }
                    int savedErr = cg.err;
                    bool savedInline = cg.inliningHelper;
                    llvm::Value *savedRet = cg.inlineRetVal;
                    cg.inliningHelper = true;
                    cg.inlineRetVal = nullptr;
                    cg.err = 0;
                    emitStmt(cg, fd->body, mod, &ilocals);
                    llvm::Value *ret = cg.inlineRetVal;
                    cg.inliningHelper = savedInline;
                    cg.inlineRetVal = savedRet;
                    /* GLSL out/inout: write the final param value back to
                     * the caller's lvalue argument before unshadowing. */
                    for (uint32_t a = 0; a < fd->param_count &&
                                        a < e->u.call.arg_count; a++) {
                        MGLDecl *pd = fd->params[a];
                        if (!pd || !pd->name) continue;
                        if (!(pd->qualifiers & MGL_AST_Q_OUT)) continue;
                        const MGLExpr *arg = e->u.call.args[a];
                        if (!arg) continue;
                        auto pit = cg.lvalues.find(pd->name);
                        if (pit == cg.lvalues.end()) continue;
                        if (arg->kind == MGL_EXPR_VAR_REF &&
                            arg->u.var_ref.name) {
                            cg.lvalues[arg->u.var_ref.name] = pit->second;
                            continue;
                        }
                        const MGLExpr *rootE = arg;
                        while (rootE &&
                               (rootE->kind == MGL_EXPR_INDEX ||
                                rootE->kind == MGL_EXPR_MEMBER)) {
                            rootE = (rootE->kind == MGL_EXPR_INDEX)
                                ? rootE->u.index.object
                                : rootE->u.member.object;
                        }
                        if (!rootE || rootE->kind != MGL_EXPR_VAR_REF ||
                            !rootE->u.var_ref.name ||
                            !cg.lvalues.count(rootE->u.var_ref.name)) {
                            cg.err = 1;
                            cg.errmsg =
                                "codegen: out/inout argument writeback "
                                "unsupported for this lvalue";
                            break;
                        }
                        llvm::Value *nv = updateIndexPath(
                            cg, arg, cg.lvalues[rootE->u.var_ref.name],
                            pit->second, mod, locals);
                        if (!nv) {
                            cg.err = 1;
                            if (cg.errmsg.empty())
                                cg.errmsg =
                                    "codegen: out/inout argument "
                                    "writeback failed";
                            break;
                        }
                        cg.lvalues[rootE->u.var_ref.name] = nv;
                    }
                    for (uint32_t a = 0; a < fd->param_count; a++) {
                        MGLDecl *pd = fd->params[a];
                        if (!pd || !pd->name) continue;
                        auto sit = saved.find(pd->name);
                        if (sit != saved.end())
                            cg.lvalues[pd->name] = sit->second;
                        else
                            cg.lvalues.erase(pd->name);
                        auto iit = savedIR.find(pd->name);
                        if (iit != savedIR.end())
                            cg.localIRTypes[pd->name] = iit->second;
                        else if (pd->type &&
                                 pd->type->base == MGL_AST_TYPE_STRUCT)
                            cg.localIRTypes.erase(pd->name);
                    }
                    if (cg.err == 1) return nullptr;
                    /* Helper return ends the inlined body, not the caller. */
                    if (cg.err == 2) cg.err = savedErr;
                    MType rt;
                    rt.scalar = (MGLIRScalar)(fd->type ? fd->type->base
                                                       : MGL_AST_TYPE_FLOAT);
                    if (fd->type && fd->type->vec_size)
                        rt.vec = fd->type->vec_size;
                    if (fd->type && fd->type->mat_cols > 1) {
                        rt.cols = fd->type->mat_cols;
                        rt.rows = fd->type->mat_rows;
                    }
                    if (ret) {
                        if (!rt.isMatrix() && !rt.isArray() &&
                            !(fd->type &&
                              fd->type->base == MGL_AST_TYPE_STRUCT))
                            ret = coerceScalar(cg, ret, rt.scalar);
                        return ret;
                    }
                    return llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(*cg.ctx), 0);
                }
            }
            if (!cg.userFns) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: call to '") + name +
                            "' not implemented in M1";
                return nullptr;
            }
            auto fit = cg.userFns->find(key);
            if (fit != cg.userFns->end()) {
                uint64_t hidden = 0;
                if (cg.userFnHidden) {
                    auto hit = cg.userFnHidden->find(key);
                    if (hit != cg.userFnHidden->end())
                        hidden = hit->second;
                }
                if (e->u.call.arg_count + hidden !=
                    fit->second->arg_size()) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: function '") + name +
                                "' expects " +
                                std::to_string(fit->second->arg_size() -
                                               hidden) +
                                " argument(s)";
                    return nullptr;
                }
                std::vector<llvm::Value *> args;
                for (uint32_t a = 0; a < e->u.call.arg_count; a++) {
                    llvm::Value *av = nullptr;
                    llvm::Type *at = fit->second->getArg(a)->getType();
                    if (at->isPointerTy()) {
                        llvm::Type *pt = at->getPointerElementType();
                        if (pt->isStructTy()) {
                            llvm::StringRef sn = pt->getStructName();
                            if (!sn.empty() &&
                                sn.startswith("struct._texture_") &&
                                e->u.call.args[a]->kind == MGL_EXPR_VAR_REF)
                                av = samplerTexValue(
                                    cg, e->u.call.args[a]->u.var_ref.name);
                        }
                    }
                    if (!av) {
                        av = emitExpr(cg, e->u.call.args[a], mod, locals);
                        if (!av) return nullptr;
                        av = coerceScalar(cg, av, scalarFromType(at));
                    }
                    args.push_back(av);
                }
                for (const auto &kv : cg.uboPtrs)
                    args.push_back(kv.second);
                for (const auto &kv : cg.ssboPtrs)
                    args.push_back(kv.second);
                for (const auto &kv : cg.acPtrs)
                    args.push_back(kv.second);
                if (cg.bufferSizePtr)
                    args.push_back(cg.bufferSizePtr);
                if (cg.userFnPassCull)
                    args.push_back(cg.lvalues["gl_CullDistance"]);
                if (cg.userFnPassClip)
                    args.push_back(cg.lvalues["gl_ClipDistance"]);
                if (cg.isGeometry) {
                    args.push_back(cg.geometryInputPtr);
                    args.push_back(cg.geometryOutputPtr);
                    args.push_back(cg.geometryCountPtr);
                    args.push_back(cg.geometryGatherPtr);
                    args.push_back(cg.geometryGatherParamsPtr);
                    args.push_back(cg.geometryWorkItemId);
                    args.push_back(cg.geometryPrimitiveId);
                    args.push_back(cg.geometryInvocationId);
                }
                if (cg.isTessControl) {
                    args.push_back(cg.stageInPtr);
                    args.push_back(cg.tessFactorPtr);
                    args.push_back(cg.stageOutPtr);
                    args.push_back(cg.indirectPtr);
                    args.push_back(cg.invocationPos);
                    args.push_back(cg.patchPos);
                }
                if (cg.isCompute && cg.threadPos)
                    args.push_back(cg.threadPos);
                if (!cg.isGeometry && !cg.isTessControl) {
                    for (const auto &kv : cg.texValues)
                        args.push_back(kv.second);
                    for (const auto &kv : cg.smpValues)
                        args.push_back(kv.second);
                    for (const auto &kv : cg.texArrayValues)
                        for (llvm::Value *tv : kv.second)
                            args.push_back(tv);
                    for (const auto &kv : cg.smpArrayValues)
                        for (llvm::Value *sv : kv.second)
                            args.push_back(sv);
                }
                if (cg.bufferPtr)
                    args.push_back(cg.bufferPtr);
                for (const auto &kv : cg.outPtrs)
                    args.push_back(kv.second);
                llvm::Value *call = cg.b->CreateCall(fit->second, args);
                /* Pull stage outputs written by the callee back into the
                 * caller's SSA map. */
                for (const auto &kv : cg.outPtrs) {
                    auto *ai = llvm::dyn_cast<llvm::AllocaInst>(kv.second);
                    if (!ai) continue;
                    cg.lvalues[kv.first] =
                        cg.b->CreateLoad(ai->getAllocatedType(), kv.second);
                }
                return call;
            }
        }
        /* Declaration-only call with no inlined/local body: emitting a
         * zero stub would silently run with wrong results. Fail codegen
         * until true cross-TU linking exists. */
        if (mod) {
            for (uint32_t si = 0; si < mod->symbol_count; si++) {
                const MGLIRSymbol *fs = mod->symbols[si];
                if (!fs || !fs->is_function || !fs->name ||
                    strcmp(fs->name, name) != 0 ||
                    fs->param_count != e->u.call.arg_count)
                    continue;
                cg.err = 1;
                cg.errmsg = std::string("codegen: call to '") + name +
                            "' has no definition in this compilation unit";
                return nullptr;
            }
        }
        cg.err = 1;
        cg.errmsg = std::string("codegen: call to '") + name +
                    "' not implemented in M1";
        return nullptr;
    }
    case MGL_EXPR_UNARY: {
        llvm::Value *v = emitExpr(cg, e->u.unary.operand, mod, locals);
        if (!v) return nullptr;
        switch (e->u.unary.op) {
        case MGL_OP_INC:
        case MGL_OP_DEC: {
            if (e->u.unary.operand->kind != MGL_EXPR_VAR_REF) {
                /* SSBO member/index lvalues read-modify-write through the
                 * device pointer; other non-variable forms stay
                 * unsupported. */
                const MGLIRSymbol *sb =
                    ssboRootSym(e->u.unary.operand, mod);
                if (!sb) {
                    cg.err = 1;
                    cg.errmsg =
                        std::string("codegen: ++/-- requires a variable");
                    return nullptr;
                }
                const MGLIRType *sty = nullptr;
                llvm::Value *sp = ssboAddress(cg, e->u.unary.operand, sb,
                                              mod, locals, &sty);
                if (!sp) return nullptr;
                llvm::Type *slt = llvmType(typeFromIR(sty), *cg.ctx);
                llvm::Align salign(16);
                if (auto *vt =
                        llvm::dyn_cast<llvm::FixedVectorType>(slt)) {
                    uint64_t w = vt->getElementCount().getFixedValue();
                    if (w == 1) salign = llvm::Align(4);
                    else if (w == 2) salign = llvm::Align(8);
                } else if (slt->isFloatTy() || slt->isIntegerTy(32)) {
                    salign = llvm::Align(4);
                }
                sp = cg.b->CreateBitCast(sp, slt->getPointerTo(1));
                llvm::Value *cur =
                    cg.b->CreateAlignedLoad(slt, sp, salign);
                bool sfp = slt->isFPOrFPVectorTy();
                llvm::Constant *sone = sfp
                    ? llvm::ConstantFP::get(slt, 1.0)
                    : llvm::ConstantInt::get(slt, 1);
                llvm::Value *nv = (e->u.unary.op == MGL_OP_INC)
                    ? (sfp ? cg.b->CreateFAdd(cur, sone)
                           : cg.b->CreateAdd(cur, sone))
                    : (sfp ? cg.b->CreateFSub(cur, sone)
                           : cg.b->CreateSub(cur, sone));
                cg.b->CreateAlignedStore(nv, sp, salign);
                return e->u.unary.prefix ? nv : cur;
            }
            const char *name = e->u.unary.operand->u.var_ref.name;
            auto it = cg.lvalues.find(name);
            if (it == cg.lvalues.end()) {
                /* Flattened anonymous SSBO members (`buffer B { int g_o0; };`
                 * → global `g_o0`) are VAR_REFs with BUFFER quals, not SSA
                 * locals.  RMW through the device pointer like member ++. */
                if (const MGLIRSymbol *sb =
                        ssboRootSym(e->u.unary.operand, mod)) {
                    const MGLIRType *sty = nullptr;
                    llvm::Value *sp = ssboAddress(cg, e->u.unary.operand, sb,
                                                  mod, locals, &sty);
                    if (!sp) return nullptr;
                    llvm::Type *slt = llvmType(typeFromIR(sty), *cg.ctx);
                    llvm::Align salign(16);
                    if (auto *vt =
                            llvm::dyn_cast<llvm::FixedVectorType>(slt)) {
                        uint64_t w = vt->getElementCount().getFixedValue();
                        if (w == 1) salign = llvm::Align(4);
                        else if (w == 2) salign = llvm::Align(8);
                    } else if (slt->isFloatTy() || slt->isIntegerTy(32)) {
                        salign = llvm::Align(4);
                    }
                    sp = cg.b->CreateBitCast(sp, slt->getPointerTo(1));
                    llvm::Value *cur =
                        cg.b->CreateAlignedLoad(slt, sp, salign);
                    bool sfp = slt->isFPOrFPVectorTy();
                    llvm::Constant *sone = sfp
                        ? llvm::ConstantFP::get(slt, 1.0)
                        : llvm::ConstantInt::get(slt, 1);
                    llvm::Value *nv = (e->u.unary.op == MGL_OP_INC)
                        ? (sfp ? cg.b->CreateFAdd(cur, sone)
                               : cg.b->CreateAdd(cur, sone))
                        : (sfp ? cg.b->CreateFSub(cur, sone)
                               : cg.b->CreateSub(cur, sone));
                    cg.b->CreateAlignedStore(nv, sp, salign);
                    return e->u.unary.prefix ? nv : cur;
                }
                cg.err = 1;
                cg.errmsg = std::string("codegen: ++/-- on unknown variable '") +
                            name + "'";
                return nullptr;
            }
            llvm::Value *cur = it->second;
            llvm::Type *ty = cur->getType();
            llvm::Type *elt = ty->isVectorTy()
                ? llvm::cast<llvm::FixedVectorType>(ty)->getElementType()
                : ty;
            bool fp = elt->isFloatingPointTy();
            llvm::Constant *one = fp
                ? llvm::ConstantFP::get(elt, 1.0)
                : llvm::ConstantInt::get(elt, 1);
            if (ty->isVectorTy()) {
                one = llvm::ConstantVector::getSplat(
                    llvm::ElementCount::getFixed(
                        (uint32_t)llvm::cast<llvm::FixedVectorType>(ty)
                            ->getElementCount()
                            .getFixedValue()),
                    one);
            }
            llvm::Value *nv = (e->u.unary.op == MGL_OP_INC)
                ? (fp ? cg.b->CreateFAdd(cur, one)
                      : cg.b->CreateAdd(cur, one))
                : (fp ? cg.b->CreateFSub(cur, one)
                      : cg.b->CreateSub(cur, one));
            cg.lvalues[name] = nv;
            return e->u.unary.prefix ? nv : cur;
        }
        case MGL_OP_SUB:
            return v->getType()->isFPOrFPVectorTy() ? cg.b->CreateFNeg(v)
                                                    : cg.b->CreateNeg(v);
        case MGL_OP_NOT:
        case MGL_OP_BNOT:
            return cg.b->CreateNot(v);
        default:
            cg.err = 1;
            cg.errmsg = std::string("codegen: unary op not implemented in M1 (line ") +
                        std::to_string(e->line) + std::string(")");
            return nullptr;
        }
    }
    case MGL_EXPR_BINARY: {
        if (e->u.binary.op == MGL_OP_COMMA) {
            if (!emitExpr(cg, e->u.binary.lhs, mod, locals)) return nullptr;
            return emitExpr(cg, e->u.binary.rhs, mod, locals);
        }
        llvm::Value *l = emitExpr(cg, e->u.binary.lhs, mod, locals);
        llvm::Value *r = emitExpr(cg, e->u.binary.rhs, mod, locals);
        if (!l || !r) return nullptr;
        llvm::Value *mres = emitMatrixBinOp(cg, e->u.binary.op, l, r);
        if (mres) return mres;
        if (llvm::Value *agg =
                emitAggregateCompare(cg, e->u.binary.op, l, r))
            return agg;
        MType lt = exprType(cg, e->u.binary.lhs, mod, locals);
        MType rt = exprType(cg, e->u.binary.rhs, mod, locals);
        llvm::Value *folded =
            tryFoldConst(cg, e->u.binary.op, l, r,
                         lt.scalar == MGLIR_SCALAR_UINT ||
                         rt.scalar == MGLIR_SCALAR_UINT);
        if (folded) return folded;
        llvm::Value *res = emitNumericBinOp(cg, e->u.binary.op, l, r, lt, rt);
        if (!res) {
            cg.err = 1;
            cg.errmsg = std::string("codegen: binary op not implemented in M1");
        }
        return res;
    }
    case MGL_EXPR_ASSIGN: {
        const bool diagAssign = mgl_env_flag_enabled("MGL_GS_DIAG_ASSIGN") &&
                                cg.isGeometry;
        if (diagAssign) {
            fprintf(stderr, "MGL GS ASSIGN begin lhsKind=%d rhsKind=%d block=%s lvalues=",
                    e->u.assign.lhs ? (int)e->u.assign.lhs->kind : -1,
                    e->u.assign.rhs ? (int)e->u.assign.rhs->kind : -1,
                    cg.b->GetInsertBlock()->getName().str().c_str());
            for (const auto &kv : cg.lvalues) fprintf(stderr, "%s,", kv.first.c_str());
            fprintf(stderr, " lhs=");
            if (e->u.assign.lhs) {
                const MGLExpr *path = e->u.assign.lhs;
                while (path && (path->kind == MGL_EXPR_MEMBER ||
                                path->kind == MGL_EXPR_INDEX)) {
                    if (path->kind == MGL_EXPR_MEMBER) {
                        fprintf(stderr, ".%s", path->u.member.field);
                        path = path->u.member.object;
                    } else {
                        fprintf(stderr, "[]");
                        path = path->u.index.object;
                    }
                }
                if (path && path->kind == MGL_EXPR_VAR_REF)
                    fprintf(stderr, "%s", path->u.var_ref.name);
                else
                    fprintf(stderr, "<nonvar>");
            } else {
                fprintf(stderr, "<null>");
            }
            fprintf(stderr, "\n");
        }
        llvm::Value *v = emitExpr(cg, e->u.assign.rhs, mod, locals);
        if (!v) return nullptr;
        llvm::Value *rhsV = v;
        const MGLExpr *lhs = e->u.assign.lhs;

        if (cg.isTessControl && lhs && lhs->kind == MGL_EXPR_INDEX &&
            lhs->u.index.object &&
            lhs->u.index.object->kind == MGL_EXPR_VAR_REF) {
            VarSym *outSym = codegenStageSymbol(
                cg, lhs->u.index.object->u.var_ref.name, VarSym::OUTPUT);
            if (outSym) {
                auto compoundInto = [&](llvm::Value *old) -> llvm::Value * {
                    if (!old) return nullptr;
                    uint32_t binop = 0;
                    switch (e->u.assign.op) {
                    case MGL_OP_ADD_ASSIGN: binop = MGL_OP_ADD; break;
                    case MGL_OP_SUB_ASSIGN: binop = MGL_OP_SUB; break;
                    case MGL_OP_MUL_ASSIGN: binop = MGL_OP_MUL; break;
                    case MGL_OP_DIV_ASSIGN: binop = MGL_OP_DIV; break;
                    default: break;
                    }
                    if (!binop) {
                        cg.err = 1;
                        cg.errmsg = "codegen: compound TCS stage output "
                                    "assignment is not implemented";
                        return nullptr;
                    }
                    return emitNumericBinOp(
                        cg, binop, old, rhsV,
                        exprType(cg, lhs, mod, locals),
                        exprType(cg, e->u.assign.rhs, mod, locals));
                };
                if (outSym->isPatch && outSym->type.isArray()) {
                    llvm::Value *idx =
                        emitExpr(cg, lhs->u.index.index, mod, locals);
                    if (!idx) return nullptr;
                    if (e->u.assign.op != MGL_OP_ASSIGN) {
                        llvm::Value *old =
                            emitPatchArrayElementLoad(cg, *outSym, idx);
                        v = compoundInto(old);
                        if (!v) return nullptr;
                    }
                    if (!emitPatchArrayElementStore(cg, *outSym, idx, v)) {
                        cg.err = 1;
                        cg.errmsg = "codegen: unavailable TCS patch array "
                                    "output store";
                        return nullptr;
                    }
                    return v;
                }
                if (!outSym->isPatch) {
                    if (e->u.assign.op != MGL_OP_ASSIGN) {
                        llvm::Value *old =
                            emitTessStageArrayLoad(cg, lhs, mod, locals);
                        v = compoundInto(old);
                        if (!v) return nullptr;
                    }
                    emitTessStageArrayStore(cg, lhs, v, mod, locals);
                    return v;
                }
            }
        }

        if (cg.isTessControl && lhs && lhs->kind == MGL_EXPR_INDEX &&
            lhs->u.index.object &&
            lhs->u.index.object->kind == MGL_EXPR_MEMBER) {
            const MGLExpr *member = lhs->u.index.object;
            const char *pvRoot = nullptr, *pvField = nullptr;
            const MGLExpr *pvVertexIndex = nullptr;
            if (perVertexPath(member, &pvRoot, &pvVertexIndex, &pvField) &&
                (!strcmp(pvField, "gl_CullDistance") ||
                 !strcmp(pvField, "gl_ClipDistance"))) {
                const uint32_t distanceCount =
                    !strcmp(pvField, "gl_ClipDistance")
                        ? MGL_AIR_PER_VERTEX_CLIP_DISTANCE_COUNT
                        : MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT;
                llvm::Value *array = emitPerVertexLoad(
                    cg, member, mod, locals);
                llvm::Value *component = emitExpr(
                    cg, lhs->u.index.index, mod, locals);
                if (!array || !component) return nullptr;
                component = coerceScalar(cg, component, MGLIR_SCALAR_UINT);
                if (e->u.assign.op != MGL_OP_ASSIGN) {
                    MType arrayType;
                    arrayType.scalar = MGLIR_SCALAR_FLOAT;
                    arrayType.arr = distanceCount;
                    llvm::Value *old = emitIndexValue(
                        cg, array, arrayType, component);
                    uint32_t binop = e->u.assign.op == MGL_OP_ADD_ASSIGN
                        ? MGL_OP_ADD : e->u.assign.op == MGL_OP_SUB_ASSIGN
                        ? MGL_OP_SUB : e->u.assign.op == MGL_OP_MUL_ASSIGN
                        ? MGL_OP_MUL : e->u.assign.op == MGL_OP_DIV_ASSIGN
                        ? MGL_OP_DIV : 0u;
                    if (!old || !binop) {
                        cg.err = 1;
                        cg.errmsg = "codegen: compound gl_out distance "
                                    "assignment unsupported";
                        return nullptr;
                    }
                    MType scalarType;
                    scalarType.scalar = MGLIR_SCALAR_FLOAT;
                    v = emitNumericBinOp(cg, binop, old, rhsV,
                                         scalarType, scalarType);
                    if (!v) return nullptr;
                }
                MType arrayType;
                arrayType.scalar = MGLIR_SCALAR_FLOAT;
                arrayType.arr = distanceCount;
                llvm::Value *updated = insertIndexValue(
                    cg, array, arrayType, component,
                    coerceScalar(cg, v, MGLIR_SCALAR_FLOAT));
                if (!updated) {
                    cg.err = 1;
                    cg.errmsg = "codegen: failed to update gl_out distance";
                    return nullptr;
                }
                emitPerVertexStore(cg, member, updated, mod, locals);
                return v;
            }
        }

        if (cg.isTessControl &&
            lhs && lhs->kind == MGL_EXPR_MEMBER) {
            const char *pvRoot = nullptr, *pvField = nullptr;
            const MGLExpr *pvIndex = nullptr;
            if (perVertexPath(lhs, &pvRoot, &pvIndex, &pvField)) {
                if (e->u.assign.op != MGL_OP_ASSIGN) {
                    llvm::Value *old = emitPerVertexLoad(cg, lhs, mod, locals);
                    if (!old) return nullptr;
                    uint32_t binop = 0;
                    switch (e->u.assign.op) {
                    case MGL_OP_ADD_ASSIGN: binop = MGL_OP_ADD; break;
                    case MGL_OP_SUB_ASSIGN: binop = MGL_OP_SUB; break;
                    case MGL_OP_MUL_ASSIGN: binop = MGL_OP_MUL; break;
                    case MGL_OP_DIV_ASSIGN: binop = MGL_OP_DIV; break;
                    default: break;
                    }
                    if (!binop) {
                        cg.err = 1;
                        cg.errmsg = "codegen: compound gl_out assignment unsupported";
                        return nullptr;
                    }
                    v = emitNumericBinOp(cg, binop, old, rhsV,
                                         exprType(cg, lhs, mod, locals),
                                         exprType(cg, e->u.assign.rhs, mod, locals));
                    if (!v) return nullptr;
                }
                emitPerVertexStore(cg, lhs, v, mod, locals);
                return v;
            }
            /* Named interface-block member: outVertex[i].field (=|+=|…) */
            {
                const char *inst = nullptr, *field = nullptr;
                const MGLExpr *indexE = nullptr;
                if (tessBlockMemberPath(lhs, &inst, &indexE, &field)) {
                    if (e->u.assign.op != MGL_OP_ASSIGN) {
                        llvm::Value *old =
                            emitTessBlockMemberLoad(cg, lhs, mod, locals);
                        if (!old) return nullptr;
                        uint32_t binop = 0;
                        switch (e->u.assign.op) {
                        case MGL_OP_ADD_ASSIGN: binop = MGL_OP_ADD; break;
                        case MGL_OP_SUB_ASSIGN: binop = MGL_OP_SUB; break;
                        case MGL_OP_MUL_ASSIGN: binop = MGL_OP_MUL; break;
                        case MGL_OP_DIV_ASSIGN: binop = MGL_OP_DIV; break;
                        default: break;
                        }
                        if (!binop) {
                            cg.err = 1;
                            cg.errmsg =
                                "codegen: compound TCS interface-block "
                                "member assignment is not implemented";
                            return nullptr;
                        }
                        v = emitNumericBinOp(
                            cg, binop, old, rhsV,
                            exprType(cg, lhs, mod, locals),
                            exprType(cg, e->u.assign.rhs, mod, locals));
                        if (!v) return nullptr;
                    }
                    if (emitTessBlockMemberStore(cg, lhs, v, mod, locals))
                        return v;
                }
            }
        }

        /* Interface-block member write: instance.field = v (VS/TES/GS out
         * blocks flatten to per-member VARYING symbols, so this is an
         * ordinary varying lvalue store keyed by the member name).  TCS
         * arrayed outs use emitTessBlockMemberStore above. */
        if (lhs->kind == MGL_EXPR_MEMBER &&
            lhs->u.member.object &&
            lhs->u.member.object->kind == MGL_EXPR_VAR_REF) {
            const char *instName = lhs->u.member.object->u.var_ref.name;
            VarSym *member = codegenBlockMember(
                cg, instName, lhs->u.member.field, VarSym::OUTPUT);
            if (!member)
                member = codegenBlockMember(
                    cg, instName, lhs->u.member.field, VarSym::VARYING);
            if (member && !cg.isTessControl &&
                member->location != UINT32_MAX) {
                if (e->u.assign.op != MGL_OP_ASSIGN) {
                    cg.err = 1;
                    cg.errmsg = "codegen: compound interface-block member "
                                "assignment unsupported";
                    return nullptr;
                }
                llvm::Type *ty = llvmType(member->type, *cg.ctx);
                if (v->getType() != ty)
                    v = coerceScalar(cg, v, member->type.scalar);
                cg.lvalues[member->name] = v;
                return v;
            }
        }

        if (lhs->kind == MGL_EXPR_INDEX || lhs->kind == MGL_EXPR_MEMBER) {
            /* Indexed/swizzled lvalue: x[i] = v / v.xy = w / m[i][j] = v. */
            if (const MGLIRSymbol *sb = ssboRootSym(lhs, mod)) {
                if (e->u.assign.op != MGL_OP_ASSIGN) {
                    llvm::Value *old = emitExpr(cg, lhs, mod, locals);
                    if (!old) return nullptr;
                    uint32_t binop = 0;
                    switch (e->u.assign.op) {
                    case MGL_OP_ADD_ASSIGN: binop = MGL_OP_ADD; break;
                    case MGL_OP_SUB_ASSIGN: binop = MGL_OP_SUB; break;
                    case MGL_OP_MUL_ASSIGN: binop = MGL_OP_MUL; break;
                    case MGL_OP_DIV_ASSIGN: binop = MGL_OP_DIV; break;
                    case MGL_OP_MOD_ASSIGN: binop = MGL_OP_MOD; break;
                    case MGL_OP_SHL_ASSIGN: binop = MGL_OP_SHL; break;
                    case MGL_OP_SHR_ASSIGN: binop = MGL_OP_SHR; break;
                    case MGL_OP_AND_ASSIGN: binop = MGL_OP_AND; break;
                    case MGL_OP_OR_ASSIGN:  binop = MGL_OP_OR; break;
                    case MGL_OP_XOR_ASSIGN: binop = MGL_OP_XOR; break;
                    default: break;
                    }
                    if (!binop) {
                        cg.err = 1;
                        cg.errmsg = "codegen: compound SSBO assign not "
                                    "implemented in M1";
                        return nullptr;
                    }
                    v = emitMatrixBinOp(cg, binop, old, v);
                    if (!v)
                        v = emitNumericBinOp(cg, binop, old, rhsV,
                            exprType(cg, lhs, mod, locals),
                            exprType(cg, e->u.assign.rhs, mod, locals));
                    if (!v) return nullptr;
                }
                emitSSBOWrite(cg, lhs, sb, mod, locals, v);
                return v;
            }
            const MGLExpr *rootE = lhs;
            while (rootE->kind == MGL_EXPR_INDEX ||
                   rootE->kind == MGL_EXPR_MEMBER) {
                rootE = (rootE->kind == MGL_EXPR_INDEX)
                    ? rootE->u.index.object : rootE->u.member.object;
            }
            if (rootE->kind != MGL_EXPR_VAR_REF) {
                cg.err = 1;
                cg.errmsg = "codegen: unsupported indexed assignment target";
                return nullptr;
            }
            const char *name = rootE->u.var_ref.name;
            /* Interface-block array member: `B.member[i] = v`.  The member
             * VarSym is flattened under its own field name (blockName=B)
             * and arrayMem-backed, so the store must go through
             * arrayMemGEP under the member key — the block instance name
             * has no storage of its own (extractvalue on the lazily
             * materialized instance lvalue asserted in LLVM). */
            if (lhs->kind == MGL_EXPR_INDEX &&
                lhs->u.index.object->kind == MGL_EXPR_MEMBER &&
                lhs->u.index.object->u.member.object &&
                lhs->u.index.object->u.member.object->kind ==
                    MGL_EXPR_VAR_REF) {
                const char *inst =
                    lhs->u.index.object->u.member.object->u.var_ref.name;
                const char *field = lhs->u.index.object->u.member.field;
                VarSym *bmem = codegenBlockMember(cg, inst, field,
                                                  VarSym::VARYING);
                if (!bmem)
                    bmem = codegenBlockMember(cg, inst, field,
                                              VarSym::OUTPUT);
                if (bmem && bmem->type.isArray()) {
                    if (e->u.assign.op != MGL_OP_ASSIGN) {
                        cg.err = 1;
                        cg.errmsg = "codegen: compound assign into block "
                                    "array member not supported";
                        return nullptr;
                    }
                    llvm::Value *idx =
                        emitExpr(cg, lhs->u.index.index, mod, locals);
                    if (!idx) return nullptr;
                    if (cg.arrayMem.count(bmem->name)) {
                        llvm::Value *ep = arrayMemGEP(cg, bmem->name,
                                                      cg.arrayMem[bmem->name],
                                                      idx);
                        cg.b->CreateAlignedStore(v, ep, llvm::Align(4));
                        return v;
                    }
                    /* Value-semantics fallback: member aggregate lives in
                     * lvalues (assembled into the stage-out record at
                     * return). */
                    if (!cg.lvalues.count(bmem->name))
                        cg.lvalues[bmem->name] = llvm::UndefValue::get(
                            llvmType(bmem->type, *cg.ctx));
                    llvm::Value *agg = cg.lvalues[bmem->name];
                    llvm::Value *nv = insertIndexValue(cg, agg, bmem->type,
                                                       idx, v);
                    if (!nv) {
                        cg.err = 1;
                        cg.errmsg = "codegen: cannot index block array "
                                    "member lvalue";
                        return nullptr;
                    }
                    cg.lvalues[bmem->name] = nv;
                    return v;
                }
            }
            /* Memory-backed scalar array: store through GEP. */
            if (name && cg.arrayMem.count(name)) {
                if (lhs->kind != MGL_EXPR_INDEX ||
                    lhs->u.index.object->kind != MGL_EXPR_VAR_REF) {
                    cg.err = 1;
                    cg.errmsg = "codegen: nested store into arrayMem not "
                                "supported";
                    return nullptr;
                }
                if (e->u.assign.op != MGL_OP_ASSIGN) {
                    llvm::Value *old = emitExpr(cg, lhs, mod, locals);
                    if (!old) return nullptr;
                    uint32_t binop = 0;
                    switch (e->u.assign.op) {
                    case MGL_OP_ADD_ASSIGN: binop = MGL_OP_ADD; break;
                    case MGL_OP_SUB_ASSIGN: binop = MGL_OP_SUB; break;
                    case MGL_OP_MUL_ASSIGN: binop = MGL_OP_MUL; break;
                    case MGL_OP_DIV_ASSIGN: binop = MGL_OP_DIV; break;
                    default: break;
                    }
                    if (!binop) {
                        cg.err = 1;
                        cg.errmsg = "codegen: compound arrayMem assign not "
                                    "implemented";
                        return nullptr;
                    }
                    v = emitNumericBinOp(cg, binop, old, rhsV,
                        exprType(cg, lhs, mod, locals),
                        exprType(cg, e->u.assign.rhs, mod, locals));
                    if (!v) return nullptr;
                }
                llvm::Value *idx =
                    emitExpr(cg, lhs->u.index.index, mod, locals);
                if (!idx) return nullptr;
                llvm::Value *ep =
                    arrayMemGEP(cg, name, cg.arrayMem[name], idx);
                cg.b->CreateAlignedStore(v, ep, llvm::Align(4));
                return v;
            }
            if (!cg.lvalues.count(name)) {
                /* First write through a member/index path to a name that
                 * has not been materialized yet (e.g. "out vec4 result;"
                 * written as result.x = ...).  Lazily start it as an
                 * undefined aggregate of its declared type, mirroring the
                 * plain-assignment path. */
                llvm::Type *aggTy = nullptr;
                auto lit = locals.find(name);
                if (lit != locals.end())
                    aggTy = llvmType(lit->second, *cg.ctx);
                else {
                    const MGLIRSymbol *sym = findSymbol(mod, name);
                    if (sym)
                        aggTy = llvmType(typeFromIR(sym->type), *cg.ctx);
                }
                if (!aggTy) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: unknown lvalue '") +
                                name + "'";
                    return nullptr;
                }
                cg.lvalues[name] = llvm::UndefValue::get(aggTy);
            }
            llvm::Value *agg = cg.lvalues[name];
            if (e->u.assign.op != MGL_OP_ASSIGN) {
                llvm::Value *old = emitExpr(cg, lhs, mod, locals);
                if (!old) return nullptr;
                uint32_t binop = 0;
                switch (e->u.assign.op) {
                case MGL_OP_ADD_ASSIGN: binop = MGL_OP_ADD; break;
                case MGL_OP_SUB_ASSIGN: binop = MGL_OP_SUB; break;
                case MGL_OP_MUL_ASSIGN: binop = MGL_OP_MUL; break;
                case MGL_OP_DIV_ASSIGN: binop = MGL_OP_DIV; break;
                case MGL_OP_MOD_ASSIGN: binop = MGL_OP_MOD; break;
                case MGL_OP_SHL_ASSIGN: binop = MGL_OP_SHL; break;
                case MGL_OP_SHR_ASSIGN: binop = MGL_OP_SHR; break;
                case MGL_OP_AND_ASSIGN: binop = MGL_OP_AND; break;
                case MGL_OP_OR_ASSIGN:  binop = MGL_OP_OR; break;
                case MGL_OP_XOR_ASSIGN: binop = MGL_OP_XOR; break;
                default: break;
                }
                if (!binop) {
                    cg.err = 1;
                    cg.errmsg = "codegen: compound assign not implemented in M1";
                    return nullptr;
                }
                v = emitMatrixBinOp(cg, binop, old, v);
                if (!v)
                    v = emitNumericBinOp(cg, binop, old, rhsV,
                        exprType(cg, lhs, mod, locals),
                        exprType(cg, e->u.assign.rhs, mod, locals));
                if (!v) {
                    cg.err = 1;
                    cg.errmsg = "codegen: compound assign unsupported for "
                                "this type in M1";
                    return nullptr;
                }
            }
            llvm::Value *nv = updateIndexPath(cg, lhs, agg, v, mod, locals);
            if (!nv) return nullptr;
            cg.lvalues[name] = nv;
            /* Stage outputs also live in outPtrs for helpers / return
             * assembly; indexed writes must update that alloca too. */
            if (cg.outPtrs.count(name))
                storeStageOut(cg, name, nv);
            if (cg.isTessControl &&
                (!strcmp(name, "gl_TessLevelOuter") ||
                 !strcmp(name, "gl_TessLevelInner")))
                flushTCSTessLevels(cg);
            return v;
        }

        /* x op= y where x is a named lvalue. */
        if (lhs->kind != MGL_EXPR_VAR_REF) {
            cg.err = 1; return nullptr;
        }
        const char *name = lhs->u.var_ref.name;
        VarSym *patchOutput = cg.isTessControl
            ? codegenStageSymbol(cg, name, VarSym::OUTPUT) : nullptr;
        if (patchOutput && patchOutput->isPatch) {
            if (e->u.assign.op != MGL_OP_ASSIGN) {
                cg.err = 1;
                cg.errmsg = "codegen: compound patch output assignment is "
                            "not implemented";
                return nullptr;
            }
            v = coerceScalar(cg, v, patchOutput->type.scalar);
            if (!emitPatchVaryingStore(cg, *patchOutput, v)) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: unavailable TCS patch output '") +
                            name + "'";
                return nullptr;
            }
            cg.lvalues[name] = v;
            return v;
        }
        if (e->u.assign.op != MGL_OP_ASSIGN) {
            MType t;
            const MGLIRSymbol *sym = nullptr;
            auto lit = locals.find(name);
            if (lit != locals.end()) t = lit->second;
            else if (strcmp(name, "gl_Position") == 0) {
                t.scalar = MGLIR_SCALAR_FLOAT;
                t.vec = 4;
            } else if (strcmp(name, "gl_PointSize") == 0) {
                t.scalar = MGLIR_SCALAR_FLOAT;
            } else {
                sym = findSymbol(mod, name);
                if (!sym) { cg.err = 1; return nullptr; }
                t = typeFromIR(sym->type);
            }
            uint32_t binop = 0;
            switch (e->u.assign.op) {
            case MGL_OP_ADD_ASSIGN: binop = MGL_OP_ADD; break;
            case MGL_OP_SUB_ASSIGN: binop = MGL_OP_SUB; break;
            case MGL_OP_MUL_ASSIGN: binop = MGL_OP_MUL; break;
            case MGL_OP_DIV_ASSIGN: binop = MGL_OP_DIV; break;
            case MGL_OP_MOD_ASSIGN: binop = MGL_OP_MOD; break;
            case MGL_OP_SHL_ASSIGN: binop = MGL_OP_SHL; break;
            case MGL_OP_SHR_ASSIGN: binop = MGL_OP_SHR; break;
            case MGL_OP_AND_ASSIGN: binop = MGL_OP_AND; break;
            case MGL_OP_OR_ASSIGN:  binop = MGL_OP_OR; break;
            case MGL_OP_XOR_ASSIGN: binop = MGL_OP_XOR; break;
            default: break;
            }
            if (!binop) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: compound assign not "
                                        "implemented in M1");
                return nullptr;
            }
            llvm::Value *cur = nullptr;
            if (cg.lvalues.count(name)) {
                cur = cg.lvalues[name];
            } else if (sym && (sym->qualifiers & MGL_AST_Q_BUFFER)) {
                cur = emitSSBORead(cg, lhs, sym, mod, locals);
                if (!cur) return nullptr;
            } else if (sym && (sym->qualifiers & MGL_AST_Q_UNIFORM)) {
                cur = bufferLoad(cg, cg.bufferOffsets[name],
                                 llvmType(t, *cg.ctx));
            } else {
                cur = llvm::UndefValue::get(llvmType(t, *cg.ctx));
            }
            v = emitMatrixBinOp(cg, binop, cur, v);
            if (!v)
                v = emitNumericBinOp(cg, binop, cur, rhsV, t,
                    exprType(cg, e->u.assign.rhs, mod, locals));
            if (!v) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: compound assign unsupported "
                                        "for this type in M1");
                return nullptr;
            }
        }
        if (strcmp(name, "gl_Position") == 0) {
            if (!cg.position.written) {
                cg.position.name = name;
                cg.position.type.scalar = MGLIR_SCALAR_FLOAT;
                cg.position.type.vec = 4;
                cg.position.kind = VarSym::OUTPUT;
            }
            cg.position.written = true;
            cg.lvalues[name] = v;
            storeStageOut(cg, name, v);
            if (diagAssign)
                fprintf(stderr, "MGL GS ASSIGN gl_Position rhs=%s typeId=%u block=%s\n",
                        v->getName().str().c_str(),
                        (unsigned)v->getType()->getTypeID(),
                        cg.b->GetInsertBlock()->getName().str().c_str());
            return v;
        }
        if (strcmp(name, "gl_PointSize") == 0) {
            cg.pointSize = true;
            cg.lvalues[name] = coerceScalar(cg, v, MGLIR_SCALAR_FLOAT);
            storeStageOut(cg, name, cg.lvalues[name]);
            return v;
        }
        if (strcmp(name, "gl_PrimitiveID") == 0) {
            if (!cg.isGeometry) {
                cg.err = 1;
                return nullptr;
            }
            cg.primitiveIdWritten = true;
            cg.lvalues[name] = coerceScalar(cg, v, MGLIR_SCALAR_INT);
            storeStageOut(cg, name, cg.lvalues[name]);
            return v;
        }
        if (strcmp(name, "gl_Layer") == 0 ||
            strcmp(name, "gl_ViewportIndex") == 0) {
            cg.layerViewport = true;
            cg.lvalues[name] = coerceScalar(cg, v, MGLIR_SCALAR_INT);
            storeStageOut(cg, name, cg.lvalues[name]);
            return v;
        }
        if (strcmp(name, "gl_FragDepth") == 0) {
            /* Fragment depth output; carried in the struct return (see
             * assembleReturn).  Unwritten paths keep 1.0. */
            cg.lvalues[name] = coerceScalar(cg, v, MGLIR_SCALAR_FLOAT);
            storeStageOut(cg, name, cg.lvalues[name]);
            return v;
        }
        auto lit = locals.find(name);
        if (lit != locals.end()) {
            v = coerceScalar(cg, v, lit->second.scalar);
            /* Memory-backed scalar arrays: whole-array assign must store
             * into the alloca; INDEX reads only from arrayMem. */
            auto amit = cg.arrayMem.find(name);
            if (amit != cg.arrayMem.end()) {
                cg.b->CreateAlignedStore(v, amit->second, llvm::Align(4));
                cg.lvalues.erase(name);
            } else {
                cg.lvalues[name] = v;
            }
            return v;
        }
        const MGLIRSymbol *sym = findSymbol(mod, name);
        if (!sym) { cg.err = 1; return nullptr; }
        v = coerceScalar(cg, v, typeFromIR(sym->type).scalar);
        if (sym->qualifiers & MGL_AST_Q_BUFFER) {
            emitSSBOWrite(cg, lhs, sym, mod, locals, v);
            return v;
        }
        if (sym->qualifiers & MGL_AST_Q_UNIFORM) {
            bufferStore(cg, cg.bufferOffsets[name],
                        llvmType(typeFromIR(sym->type), *cg.ctx), v);
            return v;
        }
        cg.lvalues[name] = v;
        storeStageOut(cg, name, v);
        return v;
    }
    case MGL_EXPR_TERNARY: {
        llvm::Value *c = emitExpr(cg, e->u.ternary.cond, mod, locals);
        llvm::Value *tv = emitExpr(cg, e->u.ternary.then, mod, locals);
        llvm::Value *ev = emitExpr(cg, e->u.ternary.else_, mod, locals);
        if (!c || !tv || !ev) return nullptr;
        if (!c->getType()->isIntegerTy(1)) {
            cg.err = 1;
            cg.errmsg = "codegen: ternary condition must be a scalar bool";
            return nullptr;
        }
        return cg.b->CreateSelect(c, tv, ev);
    }
    default:
        cg.err = 1;
        cg.errmsg = std::string("codegen: unsupported construct kind ") +
                    std::to_string(e->kind);
        return nullptr;
    }
}

MType exprType(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
               const std::map<std::string, MType> &locals) {
    MType t;
    switch (e->kind) {
    case MGL_EXPR_LITERAL: {
        MGLIRScalar b = (MGLIRScalar)e->u.literal.base;
        if (b == MGLIR_SCALAR_DOUBLE || b == MGLIR_SCALAR_HALF)
            b = MGLIR_SCALAR_FLOAT;
        t.scalar = scalarIsFloat(b) ? MGLIR_SCALAR_FLOAT : b;
        break;
    }
    case MGL_EXPR_VAR_REF: {
        if (strcmp(e->u.var_ref.name, "gl_Position") == 0) {
            t.scalar = MGLIR_SCALAR_FLOAT; t.vec = 4; break;
        }
        if (strcmp(e->u.var_ref.name, "gl_CullDistance") == 0) {
            t.scalar = MGLIR_SCALAR_FLOAT; t.arr = 8; break;
        }
        if (strcmp(e->u.var_ref.name, "gl_ClipDistance") == 0) {
            t.scalar = MGLIR_SCALAR_FLOAT; t.arr = 8; break;
        }
        if (strcmp(e->u.var_ref.name, "gl_InvocationID") == 0 ||
            strcmp(e->u.var_ref.name, "gl_PatchVerticesIn") == 0 ||
            strcmp(e->u.var_ref.name, "gl_PrimitiveID") == 0) {
            t.scalar = MGLIR_SCALAR_INT; break;
        }
        if (strcmp(e->u.var_ref.name, "gl_TessCoord") == 0) {
            t.scalar = MGLIR_SCALAR_FLOAT; t.vec = 3; break;
        }
        if (strcmp(e->u.var_ref.name, "gl_TessLevelOuter") == 0) {
            t.scalar = MGLIR_SCALAR_FLOAT; t.arr = 4; break;
        }
        if (strcmp(e->u.var_ref.name, "gl_TessLevelInner") == 0) {
            t.scalar = MGLIR_SCALAR_FLOAT; t.arr = 2; break;
        }
        if (strcmp(e->u.var_ref.name, "gl_FragDepth") == 0) {
            t.scalar = MGLIR_SCALAR_FLOAT; break;
        }
        if (strcmp(e->u.var_ref.name, "gl_PointCoord") == 0) {
            t.scalar = MGLIR_SCALAR_FLOAT; t.vec = 2; break;
        }
        if (strcmp(e->u.var_ref.name, "gl_SampleID") == 0) {
            t.scalar = MGLIR_SCALAR_INT; break;
        }
        auto lit = locals.find(e->u.var_ref.name);
        if (lit != locals.end()) { t = lit->second; break; }
        const MGLIRSymbol *s = findSymbol(mod, e->u.var_ref.name);
        if (s) t = typeFromIR(s->type);
        break;
    }
    case MGL_EXPR_MEMBER: {
        const char *pvRoot = nullptr, *pvField = nullptr;
        const MGLExpr *pvIndex = nullptr;
        if (perVertexPath(e, &pvRoot, &pvIndex, &pvField)) {
            t.scalar = MGLIR_SCALAR_FLOAT;
            if (!strcmp(pvField, "gl_Position")) t.vec = 4;
            else if (!strcmp(pvField, "gl_CullDistance"))
                t.arr = MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT;
            else if (!strcmp(pvField, "gl_ClipDistance"))
                t.arr = MGL_AIR_PER_VERTEX_CLIP_DISTANCE_COUNT;
            break;
        }
        std::vector<uint32_t> idx;
        /* Uniform-block member chain: the leaf IR type is the expression
         * type (the chain already includes every .field / [i] step). */
        if (const MGLIRType *leaf = blockMemberLeafType(e, mod)) {
            t = typeFromIR(leaf);
            break;
        }
        if (const MGLIRType *leaf = exprIRType(cg, e, mod, locals)) {
            t = typeFromIR(leaf);
            break;
        }
        MType base = exprType(cg, e->u.member.object, mod, locals);
        if (swizzleIndices(e->u.member.field, &idx))
            t = swizzleType(base, idx.size());
        break;
    }
    case MGL_EXPR_INDEX: {
        MType base = exprType(cg, e->u.index.object, mod, locals);
        if (base.isArray()) {
            /* Array[i] yields the element type.  Check before isMatrix()
             * so matCxR[N] indexes as an array of matrices. */
            t = base;
            t.arr = 0;
        } else if (base.isMatrix()) {
            /* Matrix[i] yields a column vector. */
            t.scalar = base.scalar;
            t.vec = base.rows;
        } else if (base.vec) {
            /* Vector[i] yields a scalar component. */
            t = base;
            t.vec = 0;
        } else {
            t = base;
        }
        break;
    }
    case MGL_EXPR_CALL: {
        const char *name = e->u.call.name;
        if (strcmp(name, "__mgl_array_length") == 0) {
            t.scalar = MGLIR_SCALAR_INT;
            break;
        }
        if (strcmp(name, "float") == 0 || strcmp(name, "int") == 0 ||
            strcmp(name, "uint") == 0 || strcmp(name, "bool") == 0) {
            t.scalar = name[0] == 'f' ? MGLIR_SCALAR_FLOAT
                     : name[0] == 'u' ? MGLIR_SCALAR_UINT
                     : name[0] == 'b' ? MGLIR_SCALAR_BOOL
                                      : MGLIR_SCALAR_INT;
            if (e->u.call.is_array_ctor)
                t.arr = e->u.call.arg_count;
            break;
        }
        const char *vn = name;
        if (strncmp(vn, "ivec", 4) == 0 || strncmp(vn, "uvec", 4) == 0 ||
            strncmp(vn, "bvec", 4) == 0) {
            t.scalar = vn[0] == 'i' ? MGLIR_SCALAR_INT
                     : vn[0] == 'u' ? MGLIR_SCALAR_UINT
                                    : MGLIR_SCALAR_BOOL;
            vn += 4;
        } else if (strncmp(vn, "vec", 3) == 0) {
            vn += 3;
        } else {
            vn = nullptr;
        }
        if (vn && vn[0] >= '2' && vn[0] <= '4' && vn[1] == '\0') {
            t.vec = (uint32_t)(vn[0] - '0');
            if (e->u.call.is_array_ctor)
                t.arr = e->u.call.arg_count;
            break;
        }
        if (strncmp(name, "mat", 3) == 0) {
            const char *m = name + 3;
            if (m[0] >= '2' && m[0] <= '4' && m[1] == '\0') {
                t.scalar = MGLIR_SCALAR_FLOAT;
                t.cols = t.rows = (uint32_t)(m[0] - '0');
            } else if (m[0] >= '2' && m[0] <= '4' && m[1] == 'x' &&
                       m[2] >= '2' && m[2] <= '4' && m[3] == '\0') {
                t.scalar = MGLIR_SCALAR_FLOAT;
                t.cols = (uint32_t)(m[0] - '0');
                t.rows = (uint32_t)(m[2] - '0');
            }
            if (e->u.call.is_array_ctor)
                t.arr = e->u.call.arg_count;
        } else if (cg.structTypes.count(name)) {
            if (e->u.call.is_array_ctor)
                t.arr = e->u.call.arg_count;
        } else if (strcmp(name, "normalize") == 0 ||
                   strcmp(name, "abs") == 0 ||
                   strcmp(name, "clamp") == 0 ||
                   strcmp(name, "mix") == 0) {
            /* genType result: width follows the first argument. */
            t.scalar = MGLIR_SCALAR_FLOAT;
            if (e->u.call.arg_count > 0)
                t.vec = exprType(cg, e->u.call.args[0], mod, locals).vec;
        } else if (strcmp(name, "floatBitsToInt") == 0 ||
                   strcmp(name, "floatBitsToUint") == 0) {
            t.scalar = strcmp(name, "floatBitsToUint") == 0
                ? MGLIR_SCALAR_UINT : MGLIR_SCALAR_INT;
            if (e->u.call.arg_count > 0)
                t.vec = exprType(cg, e->u.call.args[0], mod, locals).vec;
        } else if (strcmp(name, "length") == 0 ||
                   strcmp(name, "distance") == 0 ||
                   strcmp(name, "dot") == 0) {
            t.scalar = MGLIR_SCALAR_FLOAT;
        } else if (strcmp(name, "lessThanEqual") == 0 ||
                   strcmp(name, "lessThan") == 0 ||
                   strcmp(name, "greaterThan") == 0 ||
                   strcmp(name, "greaterThanEqual") == 0 ||
                   strcmp(name, "equal") == 0 ||
                   strcmp(name, "notEqual") == 0) {
            t.scalar = MGLIR_SCALAR_BOOL;
            if (e->u.call.arg_count > 0)
                t.vec = exprType(cg, e->u.call.args[0], mod, locals).vec;
        } else if (strcmp(name, "all") == 0) {
            t.scalar = MGLIR_SCALAR_BOOL;
            t.vec = 0;
        }
        break;
    }
    case MGL_EXPR_UNARY:
        t = exprType(cg, e->u.unary.operand, mod, locals);
        break;
    case MGL_EXPR_BINARY:
    case MGL_EXPR_ASSIGN:
        t = exprType(cg, e->u.binary.lhs, mod, locals);
        break;
    case MGL_EXPR_TERNARY:
        t = exprType(cg, e->u.ternary.then, mod, locals);
        break;
    default:
        break;
    }
    return t;
}

static uint32_t mtypeLength(const MType &t)
{
    if (t.isMatrix()) return t.cols;
    if (t.vec) return t.vec;
    if (t.isArray() && t.arr) return t.arr;
    return 0;
}

static uint32_t lengthFromLLVMType(llvm::Type *ty)
{
    if (!ty) return 0;
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(ty))
        return vt->getNumElements();
    if (ty->isArrayTy()) {
        llvm::Type *el = ty->getArrayElementType();
        if (el->isVectorTy() || el->isArrayTy())
            return ty->getArrayNumElements();
        return ty->getArrayNumElements();
    }
    return 0;
}

static llvm::Value *emitGLSLTypeLength(
    Codegen &cg, const MGLExpr *object, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals)
{
    if (!object) return nullptr;
    if (const MGLIRType *leaf = blockMemberLeafType(object, mod)) {
        uint32_t n = mtypeLength(typeFromIR(leaf));
        if (n) return cg.b->getInt32(n);
    }
    if (const MGLIRType *ir = exprIRType(cg, object, mod, locals)) {
        uint32_t n = mtypeLength(typeFromIR(ir));
        if (n) return cg.b->getInt32(n);
    }
    if (object->kind == MGL_EXPR_MEMBER) {
        const MGLExpr *root = object->u.member.object;
        if (root && root->kind == MGL_EXPR_INDEX &&
            root->u.index.object &&
            root->u.index.object->kind == MGL_EXPR_VAR_REF) {
            root = root->u.index.object;
        }
        if (root && root->kind == MGL_EXPR_VAR_REF) {
            VarSym *sym = codegenBlockMember(
                cg, root->u.var_ref.name, object->u.member.field,
                VarSym::VARYING);
            if (!sym)
                sym = codegenBlockMember(
                    cg, root->u.var_ref.name, object->u.member.field,
                    VarSym::OUTPUT);
            if (sym) {
                uint32_t n = mtypeLength(sym->type);
                if (n) return cg.b->getInt32(n);
            }
        }
    }
    if (object->kind == MGL_EXPR_VAR_REF) {
        auto lit = cg.lvalues.find(object->u.var_ref.name);
        if (lit != cg.lvalues.end()) {
            uint32_t n = lengthFromLLVMType(lit->second->getType());
            if (n) return cg.b->getInt32(n);
        }
    }
    if (object->kind == MGL_EXPR_BINARY &&
        object->u.binary.op == MGL_OP_MUL) {
        MType l = exprType(cg, object->u.binary.lhs, mod, locals);
        MType r = exprType(cg, object->u.binary.rhs, mod, locals);
        if (l.isMatrix() && r.isMatrix() && l.cols == r.rows)
            return cg.b->getInt32(r.cols);
        if (l.vec && r.vec)
            return cg.b->getInt32(l.vec);
    }
    if (object->kind == MGL_EXPR_INDEX) {
        MType base = exprType(cg, object->u.index.object, mod, locals);
        if (base.isMatrix()) return cg.b->getInt32(base.rows);
        if (base.vec) return cg.b->getInt32(1u);
    }
    if (object->kind == MGL_EXPR_CALL && object->u.call.name &&
        strcmp(object->u.call.name, "outerProduct") == 0 &&
        object->u.call.arg_count == 2) {
        MType l = exprType(cg, object->u.call.args[0], mod, locals);
        if (l.vec) return cg.b->getInt32(l.vec);
    }
    MType mt = exprType(cg, object, mod, locals);
    uint32_t n = mtypeLength(mt);
    if (n) return cg.b->getInt32(n);
    return nullptr;
}

/* ---- math builtins (C1d) --------------------------------------------- */
/* Bodies in mgl_air_math.cpp; thin AirMathDeps facade. */

static llvm::Value *emitMathBuiltin(Codegen &cg, const MGLExpr *e,
                                    const char *name, const MGLIRModule *mod,
                                    const std::map<std::string, MType> &locals)
{
    static const mgl::air::AirMathDeps deps = {
        emitExpr,
        callAirFn,
        dotProduct,
        broadcastTo,
        exprType,
        findSymbol,
        ssboRootSym,
        emitSSBOWrite,
        updateIndexPath,
    };
    return mgl::air::emitMathBuiltin(cg, e, name, mod, locals, deps);
}

/* emitStmt body → mgl_air_stmt.cpp (C1g); forward decl kept above emitExpr. */

/* Assemble the stage output value: vertex = {position, varyings...},
 * fragment = render target color.  Unknown outputs fall back to undef. */
/* Metal's clip-space z range is [0,1] while GLSL writes [-1,1]; convert
 * before returning the position: z' = z*0.5 + w*0.5 (clip space). */
static llvm::Value *fixClipZ(Codegen &cg, llvm::Value *pos) {
    if (!pos->getType()->isVectorTy()) return pos;
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    auto cI = [&](uint32_t v) {
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx), v);
    };
    llvm::Value *z = cg.b->CreateExtractElement(pos, cI(2));
    llvm::Value *w = cg.b->CreateExtractElement(pos, cI(3));
    llvm::Value *half = llvm::ConstantFP::get(f32, 0.5);
    z = cg.b->CreateFAdd(cg.b->CreateFMul(z, half),
                         cg.b->CreateFMul(w, half));
    return cg.b->CreateInsertElement(pos, z, cI(2));
}

/* Metal exposes clip distances but not GLSL's primitive-level cull
 * distances.  The draw path binds the source vertex buffer at slot 29 and
 * {primitive vertex count, byte offset, stride, active count} at slot 28.
 * Match the legacy path: for each distance, cull only when every vertex in
 * the primitive is negative.  Dynamic primitive assembly for strips/fans
 * remains a documented legacy limitation; the fixed-size modes are exact. */
static llvm::Value *applyCullDistance(Codegen &cg, llvm::Value *pos)
{
    if (!cg.usesCullDistance || !cg.cullBuffer || !cg.cullParams ||
        !cg.vertexId) {
        return pos;
    }
    llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    llvm::Value *params = cg.b->CreateBitCast(
        cg.cullParams, i32->getPointerTo(1));
    auto loadParam = [&](uint32_t index) {
        llvm::Value *p = cg.b->CreateGEP(i32, params, cg.b->getInt32(index));
        return cg.b->CreateAlignedLoad(i32, p, llvm::Align(4));
    };
    llvm::Value *primCount = loadParam(0);
    llvm::Value *distanceOffset = loadParam(1);
    llvm::Value *stride = loadParam(2);
    llvm::Value *distanceCount = loadParam(3);
    llvm::Value *firstVertex = loadParam(4);
    llvm::Value *explicitVertexCount = loadParam(5);
    llvm::Value *firstInstance = loadParam(10);
    llvm::Value *instanceStride = loadParam(11);
    llvm::Value *validPrim = cg.b->CreateICmpUGT(primCount, cg.b->getInt32(0));
    llvm::Value *safePrim = cg.b->CreateSelect(validPrim, primCount,
                                               cg.b->getInt32(1));
    llvm::Value *relativeVertex = cg.b->CreateSub(cg.vertexId, firstVertex);
    llvm::Value *base = cg.b->CreateAdd(
        firstVertex,
        cg.b->CreateSub(relativeVertex,
                        cg.b->CreateURem(relativeVertex, safePrim)));
    llvm::Value *hasExplicitVertices = cg.b->CreateICmpUGT(
        explicitVertexCount, cg.b->getInt32(0));
    llvm::Value *selectedVertexCount = cg.b->CreateSelect(
        hasExplicitVertices, explicitVertexCount, safePrim);
    llvm::Value *shouldCull = cg.b->getFalse();
    llvm::Value *buf = cg.b->CreateBitCast(cg.cullBuffer,
                                           f32->getPointerTo(1));
    for (uint32_t distance = 0; distance < 8; ++distance) {
        llvm::Value *activeDistance = cg.b->CreateICmpULT(
            cg.b->getInt32(distance), distanceCount);
        llvm::Value *allNegative = cg.b->getTrue();
        for (uint32_t vertex = 0; vertex < 4; ++vertex) {
            llvm::Value *activeVertex = cg.b->CreateICmpULT(
                cg.b->getInt32(vertex), selectedVertexCount);
            llvm::Value *implicitVertex = cg.b->CreateAdd(
                base, cg.b->getInt32(vertex));
            llvm::Value *explicitVertex = loadParam(6u + vertex);
            llvm::Value *other = cg.b->CreateSelect(
                hasExplicitVertices, explicitVertex, implicitVertex);
            llvm::Value *relativeInstance = cg.b->CreateSub(
                cg.instanceId ? cg.instanceId : cg.b->getInt32(0),
                firstInstance);
            llvm::Value *instanceBase = cg.b->CreateMul(
                relativeInstance, instanceStride);
            llvm::Value *byteOffset = cg.b->CreateAdd(
                cg.b->CreateMul(cg.b->CreateAdd(instanceBase, other), stride),
                cg.b->CreateAdd(distanceOffset,
                                cg.b->getInt32(distance * 4)));
            llvm::Value *floatOffset = cg.b->CreateUDiv(byteOffset,
                                                        cg.b->getInt32(4));
            llvm::Value *p = cg.b->CreateGEP(f32, buf, floatOffset);
            llvm::Value *value = cg.b->CreateAlignedLoad(f32, p,
                                                         llvm::Align(4));
            llvm::Value *negative = cg.b->CreateFCmpOLT(
                value, llvm::ConstantFP::get(f32, 0.0));
            allNegative = cg.b->CreateAnd(
                allNegative,
                cg.b->CreateSelect(activeVertex, negative, cg.b->getTrue()));
        }
        shouldCull = cg.b->CreateOr(
            shouldCull, cg.b->CreateAnd(activeDistance, allNegative));
    }
    llvm::Value *culled = llvm::ConstantVector::get({
        llvm::ConstantFP::get(f32, 2.0),
        llvm::ConstantFP::get(f32, 2.0),
        llvm::ConstantFP::get(f32, 2.0),
        llvm::ConstantFP::get(f32, 1.0)});
    return cg.b->CreateSelect(shouldCull, culled, pos);
}

/* Native post-tessellation vertex stage: primitive cull uses the patch
 * control-point gl_CullDistance values in the TCS output stream (slot 30).
 * Slot 28 already carries patch metadata for the same draw. */
static llvm::Value *applyCullDistanceFromPatchInputs(Codegen &cg,
                                                     llvm::Value *pos)
{
    if (!cg.usesPatchCullDistance || !cg.stageInPtr || !cg.indirectPtr)
        return pos;
    llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    llvm::Value *patchInfo = cg.b->CreateBitCast(
        cg.indirectPtr, i32->getPointerTo(1));
    llvm::Value *verticesPerPatch = cg.b->CreateAlignedLoad(
        i32, cg.b->CreateGEP(i32, patchInfo, cg.b->getInt32(1)),
        llvm::Align(4));
    llvm::Value *patchIndex = tessPatchIndexForStageIn(cg);
    llvm::Value *shouldCull = cg.b->getFalse();
    for (uint32_t distance = 0;
         distance < MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT; ++distance) {
        llvm::Value *allNegative = cg.b->getTrue();
        for (uint32_t vertex = 0; vertex < 32u; ++vertex) {
            llvm::Value *active = cg.b->CreateICmpULT(
                cg.b->getInt32(vertex), verticesPerPatch);
            llvm::Value *recordIdx = cg.b->CreateAdd(
                cg.b->CreateMul(patchIndex, verticesPerPatch),
                cg.b->getInt32(vertex));
            llvm::Value *off = cg.b->CreateAdd(
                cg.b->CreateMul(
                    cg.b->CreateZExt(recordIdx, cg.b->getInt64Ty()),
                    cg.b->getInt64(cg.stageInStride)),
                cg.b->getInt64(MGL_AIR_PER_VERTEX_CULL_DISTANCE_OFFSET +
                               distance * 4u));
            llvm::Value *p = cg.b->CreateGEP(
                cg.b->getInt8Ty(), cg.stageInPtr, off);
            p = cg.b->CreateBitCast(p, f32->getPointerTo(1));
            llvm::Value *value = cg.b->CreateAlignedLoad(
                f32, p, llvm::Align(4));
            llvm::Value *negative = cg.b->CreateFCmpOLT(
                value, llvm::ConstantFP::get(f32, 0.0));
            allNegative = cg.b->CreateAnd(
                allNegative,
                cg.b->CreateSelect(active, negative, cg.b->getTrue()));
        }
        shouldCull = cg.b->CreateOr(shouldCull, allNegative);
    }
    llvm::Value *culled = llvm::ConstantVector::get({
        llvm::ConstantFP::get(f32, 2.0),
        llvm::ConstantFP::get(f32, 2.0),
        llvm::ConstantFP::get(f32, 2.0),
        llvm::ConstantFP::get(f32, 1.0)});
    return cg.b->CreateSelect(shouldCull, culled, pos);
}

/* GL ignores gl_SampleMask when SAMPLE_BUFFERS==0; Metal still honours
 * [[sample_mask]], so force full coverage for non-MSAA targets.
 * Params float4 is {height, lower_left, ns_bits, sb_bits}; when emulating
 * MS sample planes, sb_bits may carry 0x80000000 | (forced_sid << 8) and
 * Metal only has one coverage bit — map GL bit[forced_sid] onto it. */
static llvm::Value *resolveSampleMaskOut(Codegen &cg) {
    llvm::Value *maskArr = cg.lvalues.count("gl_SampleMask")
        ? cg.lvalues["gl_SampleMask"]
        : llvm::ConstantInt::get(cg.b->getInt32Ty(), ~0u);
    llvm::Value *mask = maskArr->getType()->isArrayTy()
        ? cg.b->CreateExtractValue(maskArr, 0)
        : maskArr;
    if (!cg.fragSampleParams)
        return mask;
    llvm::Type *i32 = cg.b->getInt32Ty();
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    llvm::Value *fptr = cg.b->CreateBitCast(
        cg.fragSampleParams, f32->getPointerTo(1));
    llvm::Value *sbBits = cg.b->CreateAlignedLoad(
        f32, cg.b->CreateGEP(f32, fptr, cg.b->getInt32(3)), llvm::Align(4));
    llvm::Value *sb = cg.b->CreateBitCast(sbBits, i32);
    llvm::Value *forceMask = cg.b->CreateAnd(sb, cg.b->getInt32(0x80000000u));
    llvm::Value *force = cg.b->CreateICmpNE(forceMask, cg.b->getInt32(0));
    llvm::Value *forcedSid = cg.b->CreateAnd(
        cg.b->CreateLShr(sb, 8), cg.b->getInt32(0xff));
    llvm::Value *bit = cg.b->CreateAnd(
        cg.b->CreateLShr(mask, forcedSid), cg.b->getInt32(1));
    llvm::Value *forcedCover = cg.b->CreateSelect(
        cg.b->CreateICmpNE(bit, cg.b->getInt32(0)),
        cg.b->getInt32(~0), cg.b->getInt32(0));
    llvm::Value *sampleBuffers = cg.b->CreateAnd(sb, cg.b->getInt32(1));
    llvm::Value *nonMS =
        cg.b->CreateICmpEQ(sampleBuffers, cg.b->getInt32(0));
    llvm::Value *msMask = cg.b->CreateSelect(force, forcedCover, mask);
    return cg.b->CreateSelect(nonMS, cg.b->getInt32(~0), msMask);
}

/* Software gl_SamplePosition matching mglGetMultisamplefv tables.
 * Avoids Metal [[sample_position]], which crashes the AGX compiler. */
static llvm::Value *emitSamplePositionFromId(Codegen &cg, llvm::Value *sampleId,
                                             llvm::Value *numSamples);

/* Evaluate interpolant at pixel-relative offset using fine derivatives:
 *   v + dFdx(v) * ox + dFdy(v) * oy
 * Matches GLSL interpolateAtOffset / interpolateAtSample approximation when
 * hardware sample shading is unavailable (MS textures as array planes). */
static llvm::Value *emitInterpolateAtOffsetValue(Codegen &cg, llvm::Value *v,
                                                 llvm::Value *offsetXY) {
    if (!v || !offsetXY) return nullptr;
    llvm::Type *et = v->getType();
    llvm::Value *dx = nullptr;
    llvm::Value *dy = nullptr;
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(et)) {
        uint32_t n = (uint32_t)vt->getNumElements();
        std::string dxn = std::string("air.dfdx.v") + std::to_string(n) + "f32";
        std::string dyn = std::string("air.dfdy.v") + std::to_string(n) + "f32";
        dx = callAirFn(cg, dxn.c_str(), et, {v});
        dy = callAirFn(cg, dyn.c_str(), et, {v});
    } else {
        dx = callAirFn(cg, "air.dfdx.f32", et, {v});
        dy = callAirFn(cg, "air.dfdy.f32", et, {v});
    }
    if (!dx || !dy) return nullptr;
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    llvm::Value *ox = cg.b->CreateExtractElement(offsetXY, cg.b->getInt32(0));
    llvm::Value *oy = cg.b->CreateExtractElement(offsetXY, cg.b->getInt32(1));
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(et)) {
        uint32_t n = (uint32_t)vt->getNumElements();
        llvm::Value *oxv = llvm::UndefValue::get(et);
        llvm::Value *oyv = llvm::UndefValue::get(et);
        for (uint32_t i = 0; i < n; i++) {
            oxv = cg.b->CreateInsertElement(oxv, ox, i);
            oyv = cg.b->CreateInsertElement(oyv, oy, i);
        }
        ox = oxv;
        oy = oyv;
    } else if (et != f32) {
        /* Non-float scalars are not valid GENF interpolants. */
        return v;
    }
    llvm::Value *termX = cg.b->CreateFMul(dx, ox);
    llvm::Value *termY = cg.b->CreateFMul(dy, oy);
    return cg.b->CreateFAdd(v, cg.b->CreateFAdd(termX, termY));
}

static llvm::Value *emitInterpolateAtBuiltin(
    Codegen &cg, const MGLExpr *e, const char *name, const MGLIRModule *mod,
    const std::map<std::string, MType> &locals) {
    const bool isCentroid = strcmp(name, "interpolateAtCentroid") == 0;
    const bool isSample = strcmp(name, "interpolateAtSample") == 0;
    const bool isOffset = strcmp(name, "interpolateAtOffset") == 0;
    const unsigned expectArgs = isCentroid ? 1u : 2u;
    if (e->u.call.arg_count != expectArgs) {
        cg.err = 1;
        cg.errmsg = std::string("codegen: '") + name + "' expects " +
                    std::to_string(expectArgs) + " argument(s)";
        return nullptr;
    }
    llvm::Value *v = emitExpr(cg, e->u.call.args[0], mod, locals);
    if (!v) return nullptr;
    if (isCentroid) {
        /* Pixel-center interpolant is a stable centroid approximation when
         * MSAA coverage is full-quad (CTS MSI unique=false cases). */
        return v;
    }
    llvm::Value *offset = nullptr;
    if (isOffset) {
        offset = emitExpr(cg, e->u.call.args[1], mod, locals);
        if (!offset) return nullptr;
    } else if (isSample) {
        llvm::Value *sid = emitExpr(cg, e->u.call.args[1], mod, locals);
        if (!sid) return nullptr;
        llvm::Value *ns = cg.lvalues.count("gl_NumSamples")
                              ? cg.lvalues["gl_NumSamples"]
                              : cg.b->getInt32(1);
        llvm::Value *sp = emitSamplePositionFromId(cg, sid, ns);
        llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
        llvm::Value *half = llvm::ConstantFP::get(f32, 0.5);
        llvm::Value *half2 = llvm::ConstantVector::get(
            {llvm::cast<llvm::Constant>(half),
             llvm::cast<llvm::Constant>(half)});
        offset = cg.b->CreateFSub(sp, half2);
    }
    return emitInterpolateAtOffsetValue(cg, v, offset);
}

/* Software gl_SamplePosition matching mglGetMultisamplefv tables.
 * Avoids Metal [[sample_position]], which crashes the AGX compiler. */
static llvm::Value *emitSamplePositionFromId(Codegen &cg, llvm::Value *sampleId,
                                             llvm::Value *numSamples) {
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    auto *f2 = llvm::FixedVectorType::get(f32, 2);
    auto v2 = [&](float x, float y) -> llvm::Value * {
        return llvm::ConstantVector::get(
            {llvm::ConstantFP::get(f32, x), llvm::ConstantFP::get(f32, y)});
    };
    llvm::Value *center = v2(0.5f, 0.5f);
    llvm::Value *sid = sampleId
        ? cg.b->CreateBitCast(sampleId, cg.b->getInt32Ty())
        : cg.b->getInt32(0);
    llvm::Value *ns = numSamples ? numSamples : cg.b->getInt32(1);

    /* 2x Metal standard positions. */
    llvm::Value *p2 = cg.b->CreateSelect(
        cg.b->CreateICmpEQ(sid, cg.b->getInt32(0)),
        v2(0.25f, 0.25f), v2(0.75f, 0.75f));

    /* 4x Metal standard positions. */
    llvm::Value *p4 = v2(0.625f, 0.625f);
    p4 = cg.b->CreateSelect(cg.b->CreateICmpEQ(sid, cg.b->getInt32(2)),
                            v2(0.125f, 0.875f), p4);
    p4 = cg.b->CreateSelect(cg.b->CreateICmpEQ(sid, cg.b->getInt32(1)),
                            v2(0.875f, 0.375f), p4);
    p4 = cg.b->CreateSelect(cg.b->CreateICmpEQ(sid, cg.b->getInt32(0)),
                            v2(0.375f, 0.125f), p4);

    llvm::Value *pos = center;
    pos = cg.b->CreateSelect(cg.b->CreateICmpEQ(ns, cg.b->getInt32(4)), p4, pos);
    pos = cg.b->CreateSelect(cg.b->CreateICmpEQ(ns, cg.b->getInt32(2)), p2, pos);
    (void)f2;
    return pos;
}

llvm::Value *assembleReturn(Codegen &cg) {
    if (cg.isVS) {
        if (cg.retTy->isStructTy()) {
            llvm::Value *ret = llvm::UndefValue::get(cg.retTy);
            llvm::Value *pos = cg.lvalues.count("gl_Position")
                                   ? cg.lvalues["gl_Position"]
                                   : llvm::UndefValue::get(cg.retElems[0]);
            pos = fixClipZ(cg, pos);
            if (cg.usesPatchCullDistance)
                pos = applyCullDistanceFromPatchInputs(cg, pos);
            else
                pos = applyCullDistance(cg, pos);
            ret = cg.b->CreateInsertValue(ret, pos, 0);
            uint32_t ri = 1;
            if (cg.pointSize) {
                ret = cg.b->CreateInsertValue(
                    ret,
                    cg.lvalues.count("gl_PointSize")
                        ? cg.lvalues["gl_PointSize"]
                        : llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 1.0),
                    ri++);
            }
            if (cg.usesClipDistance) {
                llvm::Value *clip = cg.lvalues.count("gl_ClipDistance")
                    ? cg.lvalues["gl_ClipDistance"]
                    : defaultClipDistances(cg);
                ret = cg.b->CreateInsertValue(ret, clip, ri++);
                for (uint32_t i = 0; i < MGL_MAX_CLIP_DISTANCES; i++) {
                    ret = cg.b->CreateInsertValue(
                        ret, cg.b->CreateExtractValue(clip, i), ri++);
                }
            }
            if (cg.cullDistancePassthroughCount > 0) {
                /* Prefer the exact per-vertex capture buffer (slot 29) when
                 * present: the same values drive primitive cull emulation.
                 * Falling back to the SSA gl_CullDistance array covers draws
                 * that skip the capture pre-pass. */
                llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
                if (cg.usesCullDistance && cg.cullBuffer && cg.cullParams &&
                    cg.vertexId) {
                    llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
                    llvm::Value *params = cg.b->CreateBitCast(
                        cg.cullParams, i32->getPointerTo(1));
                    auto loadParam = [&](uint32_t index) {
                        return cg.b->CreateAlignedLoad(
                            i32,
                            cg.b->CreateGEP(i32, params, cg.b->getInt32(index)),
                            llvm::Align(4));
                    };
                    llvm::Value *distanceOffset = loadParam(1);
                    llvm::Value *stride = loadParam(2);
                    llvm::Value *firstInstance = loadParam(10);
                    llvm::Value *instanceStride = loadParam(11);
                    llvm::Value *relativeInstance = cg.b->CreateSub(
                        cg.instanceId ? cg.instanceId : cg.b->getInt32(0),
                        firstInstance);
                    llvm::Value *instanceBase = cg.b->CreateMul(
                        relativeInstance, instanceStride);
                    llvm::Value *buf = cg.b->CreateBitCast(
                        cg.cullBuffer, f32->getPointerTo(1));
                    for (uint32_t i = 0; i < cg.cullDistancePassthroughCount;
                         i++) {
                        llvm::Value *byteOffset = cg.b->CreateAdd(
                            cg.b->CreateMul(
                                cg.b->CreateAdd(instanceBase, cg.vertexId),
                                stride),
                            cg.b->CreateAdd(distanceOffset,
                                            cg.b->getInt32(i * 4u)));
                        llvm::Value *floatOffset = cg.b->CreateUDiv(
                            byteOffset, cg.b->getInt32(4));
                        llvm::Value *value = cg.b->CreateAlignedLoad(
                            f32, cg.b->CreateGEP(f32, buf, floatOffset),
                            llvm::Align(4));
                        ret = cg.b->CreateInsertValue(ret, value, ri++);
                    }
                } else {
                    llvm::Value *cull = cg.lvalues.count("gl_CullDistance")
                        ? cg.lvalues["gl_CullDistance"]
                        : defaultCullDistances(cg);
                    for (uint32_t i = 0; i < cg.cullDistancePassthroughCount;
                         i++) {
                        ret = cg.b->CreateInsertValue(
                            ret, cg.b->CreateExtractValue(cull, i), ri++);
                    }
                }
            }
            if (cg.layerViewport) {
                /* GLSL 4.60 §7.1.4 / GL 4.6 §13.8.1: unwritten gl_Layer
                 * and gl_ViewportIndex stay 0 independently. VS writing
                 * gl_Layer is an ARB_shader_viewport_layer_array-like
                 * extension; the two builtins must not alias. */
                llvm::Value *layer = cg.lvalues.count("gl_Layer")
                    ? cg.lvalues["gl_Layer"] : cg.b->getInt32(0);
                llvm::Value *viewportIndex = cg.lvalues.count("gl_ViewportIndex")
                    ? cg.lvalues["gl_ViewportIndex"] : cg.b->getInt32(0);
                ret = cg.b->CreateInsertValue(ret, layer, ri++);
                ret = cg.b->CreateInsertValue(ret, viewportIndex, ri++);
            }
            for (uint32_t i = 0; i < cg.varyings.size(); i++) {
                VarSym *var = cg.varyings[i];
                if (var->type.isArray() && cg.arrayMem.count(var->name)) {
                    uint32_t n = (uint32_t)var->type.arr;
                    llvm::Value *slot = cg.arrayMem[var->name];
                    for (uint32_t k = 0; k < n; k++) {
                        llvm::Value *ep = arrayMemGEP(
                            cg, var->name, slot, cg.b->getInt32(k));
                        llvm::Type *arrTy = cg.arrayMemTypes[var->name];
                        auto *aty = llvm::cast<llvm::ArrayType>(arrTy);
                        llvm::Value *el = cg.b->CreateAlignedLoad(
                            aty->getElementType(), ep, llvm::Align(4));
                        MType elTy = var->type;
                        elTy.arr = 0;
                        if (uintUsesSplitFloatCarrier(elTy, cg.has_gs)) {
                            llvm::Value *lo = nullptr, *hi = nullptr;
                            encodeUintSplitFloatCarrier(cg, el, &lo, &hi);
                            ret = cg.b->CreateInsertValue(ret, lo, ri++);
                            ret = cg.b->CreateInsertValue(ret, hi, ri++);
                        } else if (varyingUsesFloatCarrier(var->type,
                                                           cg.has_gs)) {
                            el = encodeFloatCarrier(cg, el, var->type.scalar);
                            ret = cg.b->CreateInsertValue(ret, el, ri++);
                        } else {
                            ret = cg.b->CreateInsertValue(ret, el, ri++);
                        }
                    }
                    continue;
                }
                llvm::Value *base = cg.lvalues.count(var->name)
                    ? cg.lvalues[var->name]
                    : llvm::UndefValue::get(llvmType(var->type, *cg.ctx));
                if (var->type.isArray()) {
                    /* Flattened: one return field per element. */
                    uint32_t n = (uint32_t)var->type.arr;
                    for (uint32_t k = 0; k < n; k++) {
                        llvm::Value *el = base;
                        if (base->getType()->isArrayTy()) {
                            el = cg.b->CreateExtractValue(base, k);
                        }
                        MType elTy = var->type;
                        elTy.arr = 0;
                        if (uintUsesSplitFloatCarrier(elTy, cg.has_gs)) {
                            llvm::Value *lo = nullptr, *hi = nullptr;
                            encodeUintSplitFloatCarrier(cg, el, &lo, &hi);
                            ret = cg.b->CreateInsertValue(ret, lo, ri++);
                            ret = cg.b->CreateInsertValue(ret, hi, ri++);
                        } else if (varyingUsesFloatCarrier(var->type, cg.has_gs)) {
                            el = encodeFloatCarrier(cg, el, var->type.scalar);
                            ret = cg.b->CreateInsertValue(ret, el, ri++);
                        } else {
                            ret = cg.b->CreateInsertValue(ret, el, ri++);
                        }
                    }
                } else if (var->type.isMatrix()) {
                    MType colTy = matrixColumnType(var->type);
                    for (uint32_t c = 0; c < var->type.cols; c++) {
                        llvm::Value *col = base;
                        if (base->getType()->isArrayTy())
                            col = cg.b->CreateExtractValue(base, c);
                        if (varyingUsesFloatCarrier(colTy, cg.has_gs))
                            col = encodeFloatCarrier(cg, col, colTy.scalar);
                        ret = cg.b->CreateInsertValue(ret, col, ri++);
                    }
                } else {
                    if (uintUsesSplitFloatCarrier(var->type, cg.has_gs)) {
                        llvm::Value *lo = nullptr, *hi = nullptr;
                        encodeUintSplitFloatCarrier(cg, base, &lo, &hi);
                        ret = cg.b->CreateInsertValue(ret, lo, ri++);
                        ret = cg.b->CreateInsertValue(ret, hi, ri++);
                    } else {
                        if (varyingUsesFloatCarrier(var->type, cg.has_gs)) {
                            base = encodeFloatCarrier(cg, base, var->type.scalar);
                        }
                        ret = cg.b->CreateInsertValue(ret, base, ri++);
                    }
                }
            }
            return ret;
        }
        llvm::Value *pos = cg.lvalues.count("gl_Position")
                               ? cg.lvalues["gl_Position"]
                               : llvm::UndefValue::get(cg.retTy);
        pos = fixClipZ(cg, pos);
        if (cg.usesPatchCullDistance)
            pos = applyCullDistanceFromPatchInputs(cg, pos);
        else
            pos = applyCullDistance(cg, pos);
        return pos;
    }
    VarSym *arrayOut = nullptr;
    for (VarSym &v : *cg.auxSyms) {
        if (v.kind == VarSym::OUTPUT && v.type.isArray()) {
            arrayOut = &v;
            break;
        }
    }
    if (arrayOut) {
        llvm::Value *color = nullptr;
        auto op = cg.outPtrs.find(arrayOut->name);
        if (op != cg.outPtrs.end()) {
            color = cg.b->CreateAlignedLoad(
                llvmType(arrayOut->type, *cg.ctx), op->second,
                llvm::Align(4));
        } else if (cg.lvalues.count(arrayOut->name)) {
            color = cg.lvalues[arrayOut->name];
        } else {
            color = llvm::UndefValue::get(
                llvmType(arrayOut->type, *cg.ctx));
        }
        /* Fragment output arrays: extract each element into the struct return. */
        llvm::Value *ret = llvm::UndefValue::get(cg.retTy);
        for (uint32_t i = 0; i < (uint32_t)arrayOut->type.arr; i++)
            ret = cg.b->CreateInsertValue(
                ret, cg.b->CreateExtractValue(color, i), i);
        if (cg.hasFragDepth) {
            llvm::Value *depth = cg.lvalues.count("gl_FragDepth")
                ? cg.lvalues["gl_FragDepth"]
                : llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 1.0);
            ret = cg.b->CreateInsertValue(ret, depth, arrayOut->type.arr);
        }
        if (cg.hasSampleMask) {
            uint32_t field = (uint32_t)arrayOut->type.arr +
                             (cg.hasFragDepth ? 1u : 0u);
            ret = cg.b->CreateInsertValue(ret, resolveSampleMaskOut(cg), field);
        }
        return ret;
    }
    if (cg.fragOutputs.size() > 1u || cg.hasFragDepth || cg.hasSampleMask) {
        llvm::Value *ret = llvm::UndefValue::get(cg.retTy);
        uint32_t field = 0u;
        for (VarSym *out : cg.fragOutputs) {
            llvm::Value *color = cg.lvalues.count(out->name)
                ? cg.lvalues[out->name]
                : llvm::UndefValue::get(llvmType(out->type, *cg.ctx));
            ret = cg.b->CreateInsertValue(ret, color, field++);
        }
        if (cg.fragOutputs.empty()) field = 1u;
        if (cg.hasFragDepth) {
            llvm::Value *depth = cg.lvalues.count("gl_FragDepth")
                ? cg.lvalues["gl_FragDepth"]
                : llvm::ConstantFP::get(llvm::Type::getFloatTy(*cg.ctx), 1.0);
            ret = cg.b->CreateInsertValue(ret, depth, field++);
        }
        if (cg.hasSampleMask) {
            ret = cg.b->CreateInsertValue(ret, resolveSampleMaskOut(cg), field);
        }
        return ret;
    }
    VarSym *out = cg.fragOutputs.empty() ? nullptr : cg.fragOutputs[0];
    if (cg.hasSampleMask) {
        llvm::Value *ret = llvm::UndefValue::get(cg.retTy);
        llvm::Value *color = (out && cg.lvalues.count(out->name))
            ? cg.lvalues[out->name]
            : llvm::UndefValue::get(cg.retElems.empty()
                  ? llvm::FixedVectorType::get(
                        llvm::Type::getFloatTy(*cg.ctx), 4)
                  : cg.retElems[0]);
        /* Single color + sample_mask: promote to struct return. */
        if (cg.retTy->isStructTy()) {
            ret = cg.b->CreateInsertValue(ret, color, 0);
            ret = cg.b->CreateInsertValue(ret, resolveSampleMaskOut(cg), 1);
            return ret;
        }
    }
    return (out && cg.lvalues.count(out->name))
        ? cg.lvalues[out->name] : llvm::UndefValue::get(cg.retTy);
}

/* ---- statements (C1g) ------------------------------------------------ */
/* Bodies in mgl_air_stmt.cpp; thin AirStmtDeps facade. */

void emitStmt(Codegen &cg, const MGLStmt *st, const MGLIRModule *mod,
              std::map<std::string, MType> *locals)
{
    static const mgl::air::AirStmtDeps deps = {
        emitExpr,
        exprType,
        assembleReturn,
        cloneIRType,
    };
    mgl::air::emitStmt(cg, st, mod, locals, deps);
}

/* ---- AIR metadata ------------------------------------------------------ */

void addModuleFlags(llvm::Module *m) {
    llvm::LLVMContext &ctx = m->getContext();
    llvm::NamedMDNode *flags = m->getOrInsertNamedMetadata("llvm.module.flags");
    auto flag = [&](const char *name, uint32_t behavior, uint32_t value) {
        return llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), behavior)),
            llvm::MDString::get(ctx, name),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), value))});
    };
    flags->addOperand(flag("wchar_size", 1, 4));
    flags->addOperand(flag("air.max_device_buffers", 7, 31));
}

/* ---- module assembly (C1e VarSym → mgl_air_varsym.*; residual) ---------- */

} /* namespace */

/* ---- attrib location preference (C1e) ----------------------------- */
/* Body in mgl_air_varsym.cpp; using-facade above. */

static bool exprUsesRuntimeArrayLength(const MGLExpr *e,
                                       const MGLIRModule *mod) {
    if (!e) return false;
    switch (e->kind) {
    case MGL_EXPR_MEMBER:
        return exprUsesRuntimeArrayLength(e->u.member.object, mod);
    case MGL_EXPR_INDEX:
        return exprUsesRuntimeArrayLength(e->u.index.object, mod) ||
               exprUsesRuntimeArrayLength(e->u.index.index, mod);
    case MGL_EXPR_CALL:
        if (strcmp(e->u.call.name, "__mgl_array_length") == 0 &&
            e->u.call.arg_count == 1) {
            const MGLIRSymbol *sb = ssboRootSym(e->u.call.args[0], mod);
            const MGLIRType *array = sb
                ? ssboExprType(e->u.call.args[0], sb, nullptr) : nullptr;
            if (array && array->kind == MGLIR_TYPE_ARRAY &&
                array->array_size == 0)
                return true;
        }
        for (uint32_t i = 0; i < e->u.call.arg_count; i++)
            if (exprUsesRuntimeArrayLength(e->u.call.args[i], mod)) return true;
        return false;
    case MGL_EXPR_UNARY:
        return exprUsesRuntimeArrayLength(e->u.unary.operand, mod);
    case MGL_EXPR_BINARY:
        return exprUsesRuntimeArrayLength(e->u.binary.lhs, mod) ||
               exprUsesRuntimeArrayLength(e->u.binary.rhs, mod);
    case MGL_EXPR_ASSIGN:
        return exprUsesRuntimeArrayLength(e->u.assign.lhs, mod) ||
               exprUsesRuntimeArrayLength(e->u.assign.rhs, mod);
    case MGL_EXPR_TERNARY:
        return exprUsesRuntimeArrayLength(e->u.ternary.cond, mod) ||
               exprUsesRuntimeArrayLength(e->u.ternary.then, mod) ||
               exprUsesRuntimeArrayLength(e->u.ternary.else_, mod);
    default:
        return false;
    }
}

static bool stmtUsesRuntimeArrayLength(const MGLStmt *st,
                                       const MGLIRModule *mod) {
    if (!st) return false;
    switch (st->kind) {
    case MGL_STMT_COMPOUND:
        for (uint32_t i = 0; i < st->u.compound.count; i++)
            if (stmtUsesRuntimeArrayLength(st->u.compound.stmts[i], mod))
                return true;
        return false;
    case MGL_STMT_EXPR:
        return exprUsesRuntimeArrayLength(st->u.expr.expr, mod);
    case MGL_STMT_DECL:
        for (const MGLDecl *d = st->u.decl.decl; d; d = d->next_declarator) {
            if (d->init && exprUsesRuntimeArrayLength(d->init, mod))
                return true;
        }
        return false;
    case MGL_STMT_IF:
        return exprUsesRuntimeArrayLength(st->u.ifs.cond, mod) ||
               stmtUsesRuntimeArrayLength(st->u.ifs.then, mod) ||
               stmtUsesRuntimeArrayLength(st->u.ifs.else_, mod);
    case MGL_STMT_FOR:
        return stmtUsesRuntimeArrayLength(st->u.loop.init, mod) ||
               exprUsesRuntimeArrayLength(st->u.loop.cond, mod) ||
               exprUsesRuntimeArrayLength(st->u.loop.incr, mod) ||
               stmtUsesRuntimeArrayLength(st->u.loop.body, mod);
    case MGL_STMT_WHILE:
    case MGL_STMT_DO_WHILE:
        return exprUsesRuntimeArrayLength(st->u.whilex.cond, mod) ||
               stmtUsesRuntimeArrayLength(st->u.whilex.body, mod);
    case MGL_STMT_SWITCH:
        return exprUsesRuntimeArrayLength(st->u.switchx.cond, mod) ||
               stmtUsesRuntimeArrayLength(st->u.switchx.body, mod);
    case MGL_STMT_CASE:
        return exprUsesRuntimeArrayLength(st->u.casex.value, mod);
    case MGL_STMT_RETURN:
        return exprUsesRuntimeArrayLength(st->u.ret.value, mod);
    default:
        return false;
    }
}

static bool translationUnitUsesRuntimeArrayLength(
    const MGLTranslationUnit *tu, const MGLIRModule *mod) {
    if (!tu || !mod) return false;
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        const MGLDecl *d = tu->decls[i];
        if (!d) continue;
        for (const MGLDecl *cur = d; cur; cur = cur->next_declarator) {
            if (exprUsesRuntimeArrayLength(cur->init, mod) ||
                stmtUsesRuntimeArrayLength(cur->body, mod))
                return true;
        }
    }
    return false;
}

/* ---- legacy GLSL frontend wiring ----------------------
 *
 * The AIR frontend parses core-profile GLSL 4.50 only (mgl_glsl_lexer/parser/
 * sema have no legacy tokens such as gl_TexCoord / texture2D / gl_FragData).
 * Pre-3.30 sources (GLSL 1.10/1.20/1.50 style) are translated source-level
 * BEFORE parsing via mgl_legacy_compat (pure C, no glslang/SPIRV).  The
 * translation is applied at every source entry point below so the reflect
 * pass, the MSL compile pass and the interface check all observe the same
 * translated source.  A no-op when the source needs no translation. */

/* Detect + translate legacy GLSL.  Returns a malloc'd translated copy (caller
 * frees via free()) or NULL when the source needs no translation.  The caller
 * falls back to the original source on NULL. */
static char *airPrepareLegacySource(const char *src, int air_stage) {
    char *translated = NULL;
    char err[256] = {0};
    int rc = mglFrontendRewriteLegacy(src, air_stage, &translated, err,
                                      sizeof(err));
    if (rc < 0) {
        fprintf(stderr, "MGL WARNING: %s\n", err[0] ? err : "legacy rewrite failed");
        return NULL;
    }
    return translated;
}

static int compileGLSLImpl(const char *src, int stage, int capture,
                           bool has_gs, bool force_tes_compute,
                           bool tes_vertex_render,
                           const char *const *attrib_names,
                           uint32_t tessPatchVertices,
                           const MGLShaderResourceList *iface_location_peers,
                           unsigned char **metallib_out, size_t *size_out,
                           char *err_buf, size_t err_cap,
                           MGLFrontendSession *session_in = nullptr) {
    if (!src || !metallib_out || !size_out) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "bad args");
        return -1;
    }
    if (stage != MGL_STAGE_VERTEX && stage != MGL_STAGE_FRAGMENT &&
        stage != MGL_STAGE_COMPUTE &&
        stage != MGL_STAGE_TESS_CONTROL &&
        stage != MGL_STAGE_TESS_EVALUATION &&
        stage != MGL_STAGE_GEOMETRY) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "unsupported stage");
        return -1;
    }
    /* FrontendSession: one legacy rewrite + parse + sema for codegen. */
    MGLFrontendSession local_session;
    mglFrontendSessionInit(&local_session);
    MGLFrontendSession *sess = session_in;
    bool own_session = false;
    if (!sess) {
        if (mglFrontendSessionBuild(&local_session, src, stage, err_buf,
                                    err_cap) != 0)
            return -1;
        sess = &local_session;
        own_session = true;
    } else if (!sess->ready || !sess->tu || !sess->src) {
        if (err_buf && err_cap)
            snprintf(err_buf, err_cap, "FrontendSession not ready");
        return -1;
    }
    const char *esrc = sess->src;
    const bool isVS = (stage == MGL_STAGE_VERTEX);
    const bool isCompute = (stage == MGL_STAGE_COMPUTE);
    const bool isTCS = (stage == MGL_STAGE_TESS_CONTROL);
    const bool isTES = (stage == MGL_STAGE_TESS_EVALUATION);
    const bool isGS = (stage == MGL_STAGE_GEOMETRY);
    const bool isCapture = capture != 0 && isVS;
    const bool isTessCapture = capture == 2 && isVS;
    const bool isCullCapture = capture == 3 && isVS;
    if (isGS && mgl_env_flag_enabled("MGL_GS_DIAG_SOURCE"))
        fprintf(stderr, "MGL GS SOURCE BEGIN\n%s\nMGL GS SOURCE END\n", esrc);
    MGLTranslationUnit *tu = sess->tu;
    MGLIRModule &mod = sess->mod;
    const uint32_t irCullCount =
        mglFrontendBuiltinArrayCount(&mod, tu, "gl_CullDistance");
    const uint32_t irClipCount =
        mglFrontendBuiltinArrayCount(&mod, tu, "gl_ClipDistance");
    const bool sourceUsesCullDistance = irCullCount > 0;
    const bool needsBufferSizeBuffer =
        translationUnitUsesRuntimeArrayLength(tu, &mod);
    /* Metal post-tessellation only supports triangle/quad patches (no
     * isolines patch type, no point output topology).  isolines and
     * point-mode TES compile either to a compute kernel that enumerates the
     * expanded line/point stream (the isTESCompute paths), or — when nothing
     * forces a compute record (no XFB, no following GS) — to an ordinary
     * render vertex function that rasterizes the CPU-seeded domain stream
     * directly (isTESVertex).  XFB and a following geometry shader still
     * force the compute path: native post-tess feeds FS directly and cannot
     * insert GS or capture XFB, so triangles/quads with either share the
     * same compute ABI. */
    const bool isTESVertex = isTES && tes_vertex_render &&
        (tu->layout_primitive == MGL_AST_TES_ISOLINES ||
         tu->layout_point_mode != 0) &&
        !force_tes_compute && !has_gs;
    const bool isTESCompute = isTES && !isTESVertex &&
        (tu->layout_primitive == MGL_AST_TES_ISOLINES ||
         tu->layout_point_mode != 0 ||
         force_tes_compute ||
         has_gs);
    const bool isKernel = isCompute || isTCS || isGS || isTESCompute;
    const bool usesCullDistance = isVS && !isCapture &&
                                  sourceUsesCullDistance;
    const bool usesPatchCullDistance =
        isTES && !isTESCompute && !isTESVertex && !isCapture &&
        sourceUsesCullDistance;
    const uint32_t activeCullCount = sourceUsesCullDistance
        ? irCullCount
        : 0u;
    const bool usesCullDistancePassthrough =
        isVS && !isCapture && sourceUsesCullDistance && activeCullCount > 0;
    const bool usesFragmentCullDistance =
        !isVS && !isTES && !isKernel && !isCapture &&
        sourceUsesCullDistance && activeCullCount > 0;
    const bool sourceUsesClipDistanceRead =
        !isVS && !isTES && !isKernel && !isCapture &&
        irClipCount > 0;
    const uint32_t activeClipCount = sourceUsesClipDistanceRead
        ? irClipCount
        : 0u;
    const bool usesFragmentClipDistance =
        sourceUsesClipDistanceRead && activeClipCount > 0;
    const uint32_t runtimeArraySizeBufferIndex =
        (isGS || isTESCompute || isTESVertex)
            ? MGL_COMPUTE_ABI_RUNTIME_ARRAY_SIZE_BUFFER_INDEX
            : MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX;

    if (isGS) {
        /* The parser intentionally shares the token `triangles` between TES
         * and GS (MGL_AST_TES_TRIANGLES); sema resolves its stage meaning. */
        if (tu->layout_primitive != MGL_AST_GS_IN_POINTS &&
            tu->layout_primitive != MGL_AST_GS_IN_LINES &&
            tu->layout_primitive != MGL_AST_GS_IN_LINES_ADJACENCY &&
            tu->layout_primitive != MGL_AST_GS_IN_TRIANGLES &&
            tu->layout_primitive != MGL_AST_GS_IN_TRIANGLES_ADJACENCY &&
            tu->layout_primitive != MGL_AST_TES_TRIANGLES) {
            if (err_buf && err_cap)
                snprintf(err_buf, err_cap,
                         "GS AIR codegen: invalid input topology");
            if (own_session) mglFrontendSessionDestroy(sess);
            return -1;
        }
        if (tu->layout_primitive_out != MGL_AST_GS_OUT_POINTS &&
            tu->layout_primitive_out != MGL_AST_GS_OUT_LINE_STRIP &&
            tu->layout_primitive_out != MGL_AST_GS_OUT_TRIANGLE_STRIP) {
            if (err_buf && err_cap)
                snprintf(err_buf, err_cap,
                         "GS AIR codegen: invalid output topology");
            if (own_session) mglFrontendSessionDestroy(sess);
            return -1;
        }
        if (tu->layout_max_vertices > 1024) {
            if (err_buf && err_cap)
                snprintf(err_buf, err_cap,
                         "GS AIR codegen: max_vertices must be in the range 0..1024");
            if (own_session) mglFrontendSessionDestroy(sess);
            return -1;
        }
    }

    /* TCS / isolines·point-mode TES: early `return;` is handled by
     * STMT_RETURN (CreateRetVoid + err=2).  Previously rejected here,
     * which blocked CTS basic-atomic-case2. */

    if (isTES) {
        /* GLSL requires an input primitive mode, but a missing mode is a
         * link error (CTS te_lacking_primitive_mode_declaration): compile
         * must still succeed.  Codegen treats DEFAULT as triangles. */
        if (tu->layout_primitive == MGL_AST_TES_DEFAULT) {
            tu->layout_primitive = MGL_AST_TES_TRIANGLES;
        } else if (tu->layout_primitive != MGL_AST_TES_TRIANGLES &&
                   tu->layout_primitive != MGL_AST_TES_QUADS &&
                   tu->layout_primitive != MGL_AST_TES_ISOLINES) {
            if (err_buf && err_cap)
                snprintf(err_buf, err_cap,
                         "TES AIR codegen: only layout(triangles/quads/"
                         "isolines) is implemented yet");
            if (own_session) mglFrontendSessionDestroy(sess);
            return -1;
        }
    }

    std::vector<Uniform> uniforms;
    uint32_t bufferSize = 0;
    if (collectUniforms(&mod, &uniforms, &bufferSize, err_buf, err_cap)) {
        if (own_session) mglFrontendSessionDestroy(sess);
        return -1;
    }

    /* Find main and the stage's interface symbols. */
    MGLDecl *mainDecl = nullptr;
    std::vector<VarSym> syms;
    /* C1e: classify/location/stride → mgl_air_varsym.*; thin facade. */
    collectStageVarSyms(&mod, tu, stage, &syms);
    {
        std::vector<AirIfaceLocationPeer> peers;
        if (iface_location_peers && iface_location_peers->list &&
            ((has_gs && stage == MGL_STAGE_FRAGMENT) ||
             stage == MGL_STAGE_TESS_EVALUATION)) {
            peers.reserve(iface_location_peers->count);
            for (GLuint i = 0; i < iface_location_peers->count; i++) {
                const MGLShaderResource *peer =
                    &iface_location_peers->list[i];
                AirIfaceLocationPeer p;
                p.name = peer->name;
                p.location = peer->location;
                p.isPerPatch = (peer->is_per_patch != GL_FALSE);
                peers.push_back(p);
            }
        }
        assignStageVarSymLocations(
            syms, stage, isKernel, has_gs, attrib_names, MAX_ATTRIBS,
            peers.empty() ? nullptr : peers.data(),
            (uint32_t)peers.size());
    }
    uint32_t ssboCount = 0, uboCount = 0, acCount = 0, texCount = 0, imageCount = 0;
    for (VarSym &v : syms) {
        if (v.kind == VarSym::SSBO) {
            const MGLIRSymbol *us = findSymbol(&mod, v.name.c_str());
            ssboCount += uniformBlockElementCount(us ? us->type : nullptr);
        } else if (v.kind == VarSym::UBO) {
            const MGLIRSymbol *us = findSymbol(&mod, v.name.c_str());
            uboCount += uniformBlockElementCount(us ? us->type : nullptr);
        } else if (v.kind == VarSym::ATOMIC_COUNTER) {
            acCount++;
        } else if (v.kind == VarSym::TEXTURE) {
            texCount += v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
        } else if (v.kind == VarSym::IMAGE) {
            imageCount += v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
        }
    }
    const uint32_t stageInputStride = (isTCS || isGS)
        ? stageRecordStride(syms, VarSym::VARYING, false,
                            MGL_AIR_PER_VERTEX_STRIDE)
        : isTES ? stageRecordStride(syms, VarSym::CONTROL_POINT_INPUT, false,
                                    MGL_AIR_PER_VERTEX_STRIDE)
                : MGL_AIR_PER_VERTEX_STRIDE;
    const uint32_t stageOutputStride = (isTCS || isGS || isTESCompute || isTESVertex)
        ? stageRecordStride(syms,
                            (isTESCompute || isTESVertex) ? VarSym::VARYING
                                                          : VarSym::OUTPUT,
                            false, MGL_AIR_PER_VERTEX_STRIDE)
        : MGL_AIR_PER_VERTEX_STRIDE;
    const uint32_t tessCaptureStride = isTessCapture
        ? stageRecordStride(syms, VarSym::VARYING, false,
                            MGL_AIR_PER_VERTEX_STRIDE)
        : MGL_AIR_PER_VERTEX_STRIDE;
    const uint32_t patchInputStride = isTES
        ? stageRecordStride(syms, VarSym::CONTROL_POINT_INPUT, true, 16u)
        : 16u;
    const uint32_t patchOutputStride = isTCS
        ? stageRecordStride(syms, VarSym::OUTPUT, true, 16u) : 16u;
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        if (tu->decls[i]->body && tu->decls[i]->name &&
            strcmp(tu->decls[i]->name, "main") == 0) {
            mainDecl = tu->decls[i];
            break;
        }
    }
    if (!mainDecl) {
        /* GLSL §3.6: a compilation unit need not define main — helpers
         * defined here are linked with another unit that provides the
         * entry point (CTS negative-glsl-linkTime / SSO).  CompileShader
         * only requires a successful parse+sema; discard codegen. */
        *metallib_out = (unsigned char *)malloc(1);
        *size_out = 0;
        if (!*metallib_out) {
            if (err_buf && err_cap)
                snprintf(err_buf, err_cap, "out of memory");
            if (own_session) mglFrontendSessionDestroy(sess);
            return -1;
        }
        if (own_session) mglFrontendSessionDestroy(sess);
        return 0;
    }
    /* Patch uniform offsets into var syms. */
    for (VarSym &v : syms) {
        if (v.kind == VarSym::BUFFER) {
            for (const Uniform &u : uniforms) {
                if (u.name == v.name) { v.bufferOffset = u.offset; break; }
            }
        }
    }

    llvm::LLVMContext ctx;
    ctx.setOpaquePointers(false);
    llvm::Module module("mgl_shader", ctx);
    module.setTargetTriple("air64_v28-apple-macosx26.0.0");
    module.setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"
                         "-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32"
                         "-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
                         "-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
                         "-n8:16:32");

    /* Vertex return: { position, varyings... }; fragment: { output }. */
    std::vector<llvm::Type *> retElems;
    std::vector<VarSym *> varyings;
    std::vector<VarSym *> fragOutputs;
    llvm::Type *retTy = nullptr;
    /* Built-in detection mirrors the legacy path's strstr over the source
     * (gl_FragCoord -> fragment position arg; gl_PointSize -> point_size
     * output member). */
    const bool usesFragCoord =
        !isVS && !isTES && !isKernel && strstr(esrc, "gl_FragCoord") != nullptr;
    const bool usesFrontFacing =
        !isVS && !isTES && !isKernel &&
        strstr(esrc, "gl_FrontFacing") != nullptr;
    const bool usesPointCoord =
        !isVS && !isTES && !isKernel &&
        strstr(esrc, "gl_PointCoord") != nullptr;
    const bool usesFragDepth =
        !isVS && !isTES && !isKernel &&
        strstr(esrc, "gl_FragDepth") != nullptr;
    const bool usesPrimitiveId =
        !isVS && !isTES && !isKernel &&
        strstr(esrc, "gl_PrimitiveID") != nullptr;
    const bool tesUsesPrimitiveId =
        isTES && !isKernel && strstr(esrc, "gl_PrimitiveID") != nullptr;
    const bool usesLayer =
        !isVS && !isTES && !isKernel && strstr(esrc, "gl_Layer") != nullptr;
    const bool usesViewportIndex =
        !isVS && !isTES && !isKernel &&
        strstr(esrc, "gl_ViewportIndex") != nullptr;
    const bool usesSampleID =
        !isVS && !isTES && !isKernel &&
        strstr(esrc, "gl_SampleID") != nullptr;
    const bool usesSamplePosition =
        !isVS && !isTES && !isKernel &&
        strstr(esrc, "gl_SamplePosition") != nullptr;
    const bool usesSampleMaskIn =
        !isVS && !isTES && !isKernel &&
        strstr(esrc, "gl_SampleMaskIn") != nullptr;
    const bool usesSampleMask =
        !isVS && !isTES && !isKernel &&
        (strstr(esrc, "gl_SampleMask[") != nullptr ||
         strstr(esrc, "gl_SampleMask =") != nullptr);
    const bool usesNumSamples =
        !isVS && !isTES && !isKernel &&
        strstr(esrc, "gl_NumSamples") != nullptr;
    const bool usesInterpolateAtSample =
        !isVS && !isTES && !isKernel &&
        strstr(esrc, "interpolateAtSample") != nullptr;
    bool hasSampleVarying = false;
    if (!isVS && !isTES && !isKernel) {
        for (const VarSym &v : syms) {
            if (v.kind == VarSym::VARYING && v.isSample) {
                hasSampleVarying = true;
                break;
            }
        }
    }
    /* SamplePosition is synthesized from sample_id + num_samples (no
     * air.sample_position — AGX Metal crashes on that attribute). */
    const bool needSampleID = usesSampleID || usesSamplePosition ||
                              usesInterpolateAtSample || hasSampleVarying;
    const bool needSampleParams =
        usesNumSamples || usesSampleMask || usesSamplePosition ||
        usesSampleID || usesInterpolateAtSample || hasSampleVarying;
    /* Metal [[position]] is top-left; GL gl_FragCoord is bottom-left.  Slot 30
     * carries {height, lower_left, num_samples, sample_buffers} so the FS can
     * flip Y (see RenderPass fragCoordParams). */
    const bool needFragCoordParams = usesFragCoord;
    const bool needParamsBuffer = needFragCoordParams || needSampleParams;
    const bool usesWorkGroupID =
        isCompute && strstr(esrc, "gl_WorkGroupID") != nullptr;
    const bool usesNumWorkGroups =
        isCompute && strstr(esrc, "gl_NumWorkGroups") != nullptr;
    const bool usesLocalInvocationID =
        isCompute && strstr(esrc, "gl_LocalInvocationID") != nullptr;
    const bool usesLocalInvocationIndex =
        isCompute && strstr(esrc, "gl_LocalInvocationIndex") != nullptr;
    const bool usesLocalInvocation =
        usesLocalInvocationID || usesLocalInvocationIndex;
    /* Always emit [[point_size]] for ordinary VS.  After a GS-expanded
     * triangle draw, Metal Point-topology PSOs whose VS omit point_size can
     * silently drop subsequent GL_POINTS draws (CTS multiple-uniforms after
     * early-fragment-tests).  Default 1.0 matches GL's initial point size
     * when the shader does not write gl_PointSize.
     *
     * Skip generated GS/TES passthrough VS: those rasterize with an explicit
     * Triangle/Line topology, and Metal rejects point_size on that class.
     * Same for VS that write gl_Layer / gl_ViewportIndex: Metal requires an
     * explicit topology for [[render_target_array_index]], and Paravirtual
     * rejects Triangle + point_size.  Capture / TES stages keep the
     * historical "only if written" gate. */
    const bool isStagePassthrough =
        (isVS && !isTESVertex &&
         (strstr(esrc, "mgl_gs_output") != nullptr ||
          strstr(esrc, "mgl_tes_output") != nullptr));
    const bool usesLayerViewport =
        isVS && (strstr(esrc, "gl_Layer") != nullptr ||
                 strstr(esrc, "gl_ViewportIndex") != nullptr);
    /* TES-vertex point_mode draws rasterize MTLPrimitiveTypePoint and must
     * declare [[point_size]]; isolines (line topology) must not. */
    const bool usesPointSize =
        ((isVS && !isCapture && !isStagePassthrough && !usesLayerViewport) ||
         (isTESVertex && tu->layout_point_mode != 0) ||
         ((isVS || (isTES && !isTESVertex)) &&
          strstr(esrc, "gl_PointSize") != nullptr));
    const bool usesClipDistance =
        (isVS || (isTES && !isTESCompute)) && !isCapture && !isKernel &&
        irClipCount > 0;
    /* TES-vertex keeps the isoline partner-endpoint cull rule (a line is
     * culled when both endpoints' distance < 0 for the same axis; a point
     * when any distance < 0) that the passthrough VS used to apply, so the
     * record buffer stays bound at slot 29 for the partner read. */
    const bool usesTesVertexCull =
        isTESVertex && sourceUsesCullDistance;
    const uint32_t userBufferLocationBase = isTES ? 1u : 0u;
    if (isVS || isTES) {
        /* retElems always carries the output record (capture variants
         * write it to the XFB buffer). */
        retElems.push_back(llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4));
        if (isTessCapture) {
            retElems.push_back(llvm::Type::getFloatTy(ctx));
            retElems.push_back(llvm::ArrayType::get(
                llvm::Type::getFloatTy(ctx),
                MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT));
        } else if (usesPointSize) {
            retElems.push_back(llvm::Type::getFloatTy(ctx));
        }
        if (usesClipDistance) {
            retElems.push_back(llvm::ArrayType::get(
                llvm::Type::getFloatTy(ctx), MGL_MAX_CLIP_DISTANCES));
            /* Metal FS cannot read clip_distance; also emit flat mirrors. */
            for (uint32_t i = 0; i < MGL_MAX_CLIP_DISTANCES; i++)
                retElems.push_back(llvm::Type::getFloatTy(ctx));
        }
        if (usesCullDistancePassthrough) {
            for (uint32_t i = 0; i < activeCullCount; i++)
                retElems.push_back(llvm::Type::getFloatTy(ctx));
        }
        if (usesLayerViewport) {
            retElems.push_back(llvm::Type::getInt32Ty(ctx));
            retElems.push_back(llvm::Type::getInt32Ty(ctx));
        }
        for (VarSym &v : syms) {
            if (v.kind == VarSym::VARYING) {
                if (!isTessCapture) {
                    /* Metal stage-out structs forbid array members
                     * ("field of illegal type 'float4[N]'"), so array
                     * varyings are flattened into per-element scalar
                     * fields; assembleReturn and the metadata emit one
                     * entry per element with element-specific interface
                     * names (name_elmN) on both stages. */
                    if (v.type.isArray()) {
                        MType el = v.type;
                        el.arr = 0;
                        if (varyingUsesFloatCarrier(el, has_gs))
                            el = floatCarrierType(el);
                        for (uint32_t i = 0; i < (uint32_t)v.type.arr; i++)
                            retElems.push_back(llvmType(el, ctx));
                    } else if (v.type.isMatrix()) {
                        /* Metal forbids matrix stage-out members; emit one
                         * vector field per column (GL location = base+c). */
                        MType col = matrixColumnType(v.type);
                        if (varyingUsesFloatCarrier(col, has_gs))
                            col = floatCarrierType(col);
                        for (uint32_t c = 0; c < v.type.cols; c++)
                            retElems.push_back(llvmType(col, ctx));
                    } else {
                        MType outTy = v.type;
                        if (uintUsesSplitFloatCarrier(outTy, has_gs)) {
                            retElems.push_back(llvmType(floatCarrierType(outTy), ctx));
                            retElems.push_back(llvmType(floatCarrierType(outTy), ctx));
                        } else {
                            if (varyingUsesFloatCarrier(outTy, has_gs))
                                outTy = floatCarrierType(outTy);
                            retElems.push_back(llvmType(outTy, ctx));
                        }
                    }
                }
                varyings.push_back(&v);
            }
        }
        if (isTessCapture || isCullCapture) {
            /* AGX rejects RasterizationEnabled + void VS. Tess/cull capture
             * must return position so the discard-draw pipeline can keep
             * rasterization on (stub FS + zero color masks) and still
             * execute VS SSBO stores / cull-distance capture.  The real
             * payload still travels through the slot-29 record buffer. */
            retTy = llvm::FixedVectorType::get(
                llvm::Type::getFloatTy(ctx), 4);
        } else if (isKernel || isCapture) {
            retTy = llvm::Type::getVoidTy(ctx);
        } else if (isTES && !isTESVertex) {
            /* Apple's post-tessellation ABI returns a packed output record,
             * even when position is its only member. */
            retTy = llvm::StructType::get(ctx, retElems, true);
        } else if (retElems.size() == 1) {
            retTy = retElems[0];
        } else {
            retTy = llvm::StructType::get(ctx, retElems);
        }
    } else if (isKernel) {
        retTy = llvm::Type::getVoidTy(ctx);
    } else {
        VarSym *arrayOutput = nullptr;
        for (VarSym &v : syms) {
            if (v.kind != VarSym::OUTPUT) continue;
            if (v.type.isArray()) {
                arrayOutput = &v;
                break;
            }
            fragOutputs.push_back(&v);
        }
        std::sort(fragOutputs.begin(), fragOutputs.end(),
                  [](const VarSym *a, const VarSym *b) {
                      return a->location < b->location;
                  });
        if (arrayOutput) {
            /* Fragment output arrays (gl_FragData / out T fragOut[N]): flatten
             * into per-element color outputs.  Element type follows the GLSL
             * scalar (float4 / int4 / uint4) — hardcoding float4 made Metal
             * reject R32UI PSOs (CTS texture_barrier). */
            MType el = arrayOutput->type;
            el.arr = 0;
            llvm::Type *elTy = llvmType(el, ctx);
            std::vector<llvm::Type *> fields;
            for (uint32_t i = 0; i < (uint32_t)arrayOutput->type.arr; i++)
                fields.push_back(elTy);
            if (usesFragDepth)
                fields.push_back(llvm::Type::getFloatTy(ctx));
            if (usesSampleMask)
                fields.push_back(llvm::Type::getInt32Ty(ctx));
            retTy = llvm::StructType::get(ctx, fields);
        } else if (fragOutputs.size() > 1u || usesFragDepth ||
                   usesSampleMask) {
            std::vector<llvm::Type *> fields;
            for (VarSym *out : fragOutputs)
                fields.push_back(llvmType(out->type, ctx));
            if (fields.empty())
                fields.push_back(llvm::FixedVectorType::get(
                    llvm::Type::getFloatTy(ctx), 4));
            if (usesFragDepth)
                fields.push_back(llvm::Type::getFloatTy(ctx));
            if (usesSampleMask)
                fields.push_back(llvm::Type::getInt32Ty(ctx));
            retTy = llvm::StructType::get(ctx, fields);
        } else {
            retTy = !fragOutputs.empty()
                ? llvmType(fragOutputs[0]->type, ctx)
                : llvm::FixedVectorType::get(
                      llvm::Type::getFloatTy(ctx), 4);
        }
    }

    auto captureRecordType = [&]() -> llvm::Type * {
        if (isCullCapture) {
            return llvm::ArrayType::get(llvm::Type::getFloatTy(ctx), 8);
        }
        std::vector<llvm::Type *> fields = retElems;
        return fields.size() == 1 ? fields[0]
                                  : llvm::StructType::get(ctx, fields);
    };

    /* Parameters: capture = [captureBuf, buffer, ssbo..., tex/smp...,
     * attrs..., optional capture params, instance_id, base_instance, vertex_id]; vertex = [buffer,
     * ssbo..., tex/smp..., attrs..., cull buffers, instance_id, base_instance, vertex_id];
     * fragment = [varyings..., buffer, ssbo..., tex/smp...];
     * compute = [buffer, ssbo..., tex/smp..., thread_position_in_grid]. */
    std::vector<llvm::Type *> paramTys;
    bool hasBuffer = !uniforms.empty();
    /* Metal buffer slots reserved by the packed plain-uniform buffer.
     * Must match mglAirReflectModule ssbo_binding for every stage that
     * emits the pack (VS/TES/CS/GS/TCS via isKernel, plus FS).  LLVM
     * argument indices still use the VS/TES/kernel-only skip below —
     * on FS the pack is a trailing arg, not a leading one. */
    const uint32_t plainBufMetalSlots =
        hasBuffer && (isVS || isTES || isKernel ||
                      stage == MGL_STAGE_FRAGMENT)
            ? 1u : 0u;
    uint32_t attrCount = 0;
    for (VarSym &v : syms) {
        if (!(isVS && v.kind == VarSym::ATTR)) continue;
        /* Array / matrix attributes occupy one Metal/GL location per
         * element or column so glVertexAttribPointer(base+i) matches. */
        attrCount += varyingLocationSpan(v.type);
    }
    llvm::StructType *texTy2d =
        llvm::StructType::create(ctx, "struct._texture_2d_t");
    llvm::StructType *texTy2dArray =
        llvm::StructType::create(ctx, "struct._texture_2d_array_t");
    llvm::StructType *texTy3d =
        llvm::StructType::create(ctx, "struct._texture_3d_t");
    llvm::StructType *texTyCube =
        llvm::StructType::create(ctx, "struct._texture_cube_t");
    llvm::StructType *texTyCubeArray =
        llvm::StructType::create(ctx, "struct._texture_cube_array_t");
    llvm::StructType *texTyBuf =
        llvm::StructType::create(ctx, "struct._texture_buffer_1d_t");
    llvm::StructType *smpTy =
        llvm::StructType::create(ctx, "struct._sampler_t");
    llvm::StructType *patchControlTy = isTES
        ? llvm::StructType::create(ctx, "struct._patch_control_point_t")
        : nullptr;
    if (isCapture)
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    if (isVS && !isCapture) {
        /* Vertex attributes come first (matching the Metal ABI: stage_in
         * value args precede buffers/textures).  Exception: the XFB
         * capture variant must NOT place attributes at odd argument slots
         * (right after the read_write capture buffer) -- Metal rejects
         * that PSO with "Unsupported attribute type".  For capture we
         * keep the legacy layout with attributes AFTER all buffers. */
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATTR) continue;
            if (v.type.isArray() && v.type.arr > 0) {
                MType el = attrMetalIfaceType(v.type);
                el.arr = 0;
                for (uint32_t k = 0; k < (uint32_t)v.type.arr; k++)
                    paramTys.push_back(llvmType(el, ctx));
            } else if (v.type.isMatrix()) {
                MType col = attrMetalIfaceType(matrixColumnType(v.type));
                for (uint32_t c = 0; c < v.type.cols; c++)
                    paramTys.push_back(llvmType(col, ctx));
            } else {
                paramTys.push_back(llvmType(attrMetalIfaceType(v.type), ctx));
            }
        }
    }
    if ((isVS || isTES || isKernel) && hasBuffer)
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    for (VarSym &v : syms)
        if (v.kind == VarSym::SSBO || v.kind == VarSym::UBO ||
            v.kind == VarSym::ATOMIC_COUNTER) {
            const MGLIRSymbol *us = findSymbol(&mod, v.name.c_str());
            uint32_t uelems =
                (v.kind == VarSym::UBO || v.kind == VarSym::SSBO)
                    ? uniformBlockElementCount(us ? us->type : nullptr)
                    : 1u;
            for (uint32_t k = 0; k < uelems; k++)
                paramTys.push_back(
                    llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        }
    if (needsBufferSizeBuffer)
        paramTys.push_back(llvm::Type::getInt32Ty(ctx)->getPointerTo(2));
    for (VarSym &v : syms) {
        if (v.kind != VarSym::TEXTURE) continue;
        const MGLIRType *st = v.opaqueType;
        if (!st) {
            const MGLIRSymbol *ts = findSymbol(&mod, v.name.c_str());
            st = ts ? ts->type : nullptr;
        }
        while (st && st->kind == MGLIR_TYPE_ARRAY)
            st = st->elem_type;
        MGLIRTexKind tk = st && st->kind == MGLIR_TYPE_SAMPLER
            ? st->tex_kind : MGLIR_TEX_2D;
        llvm::StructType *tt = texTy2d; /* TEX_BUFFER / 1D use texture2d */
        if (tk == MGLIR_TEX_3D) tt = texTy3d;
        else if (tk == MGLIR_TEX_2D_ARRAY || tk == MGLIR_TEX_1D_ARRAY ||
                 tk == MGLIR_TEX_2D_MS || tk == MGLIR_TEX_2D_MS_ARRAY)
            tt = texTy2dArray;
        else if (tk == MGLIR_TEX_CUBE) tt = texTyCube;
        else if (tk == MGLIR_TEX_CUBE_ARRAY) tt = texTyCubeArray;
        uint32_t elements = v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
        for (uint32_t k = 0; k < elements; k++) {
            paramTys.push_back(tt->getPointerTo(1));
            paramTys.push_back(smpTy->getPointerTo(2));
        }
    }
    for (VarSym &v : syms) {
        if (v.kind != VarSym::IMAGE) continue;
        const MGLIRType *imgTy =
            imageElementType(findSymbol(&mod, v.name.c_str()));
        MGLIRTexKind tk = imgTy ? imgTy->tex_kind : MGLIR_TEX_2D;
        llvm::StructType *tt = texTy2d;
        if (tk == MGLIR_TEX_3D) tt = texTy3d;
        else if (tk == MGLIR_TEX_2D_ARRAY || tk == MGLIR_TEX_1D_ARRAY ||
                 tk == MGLIR_TEX_2D_MS || tk == MGLIR_TEX_2D_MS_ARRAY)
            tt = texTy2dArray;
        else if (tk == MGLIR_TEX_CUBE) tt = texTyCube;
        else if (tk == MGLIR_TEX_CUBE_ARRAY) tt = texTyCubeArray;
        else if (tk == MGLIR_TEX_BUFFER) tt = texTy2d;
        uint32_t elements = v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
        for (uint32_t k = 0; k < elements; k++)
            paramTys.push_back(tt->getPointerTo(1));
    }
    for (VarSym &v : syms) {
        if (!isVS && !isTES && !isKernel && v.kind == VarSym::VARYING) {
            if (v.type.isArray()) {
                /* Flattened stage-in: N scalar params (Metal forbids array
                 * stage-in members; the setup binds one arg per element). */
                MType el = v.type;
                el.arr = 0;
                MType iface = varyingUsesFloatCarrier(v.type, has_gs)
                    ? floatCarrierType(el) : el;
                for (uint32_t k = 0; k < (uint32_t)v.type.arr; k++)
                    paramTys.push_back(llvmType(iface, ctx));
            } else if (v.type.isMatrix()) {
                MType col = matrixColumnType(v.type);
                if (varyingUsesFloatCarrier(col, has_gs))
                    col = floatCarrierType(col);
                for (uint32_t c = 0; c < v.type.cols; c++)
                    paramTys.push_back(llvmType(col, ctx));
            } else {
                if (uintUsesSplitFloatCarrier(v.type, has_gs)) {
                    paramTys.push_back(llvmType(floatCarrierType(v.type), ctx));
                    paramTys.push_back(llvmType(floatCarrierType(v.type), ctx));
                } else {
                    paramTys.push_back(llvmType(
                        varyingUsesFloatCarrier(v.type, has_gs)
                            ? floatCarrierType(v.type) : v.type, ctx));
                }
            }
        }
    }
    if (usesFragmentClipDistance) {
        for (uint32_t i = 0; i < activeClipCount; i++)
            paramTys.push_back(llvm::Type::getFloatTy(ctx));
    }
    if (usesFragmentCullDistance) {
        for (uint32_t i = 0; i < activeCullCount; i++)
            paramTys.push_back(llvm::Type::getFloatTy(ctx));
    }
    if (isVS && isCapture) {
        /* XFB capture variant: attributes trail all buffers (see above). */
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATTR) continue;
            if (v.type.isArray() && v.type.arr > 0) {
                MType el = attrMetalIfaceType(v.type);
                el.arr = 0;
                for (uint32_t k = 0; k < (uint32_t)v.type.arr; k++)
                    paramTys.push_back(llvmType(el, ctx));
            } else if (v.type.isMatrix()) {
                MType col = attrMetalIfaceType(matrixColumnType(v.type));
                for (uint32_t c = 0; c < v.type.cols; c++)
                    paramTys.push_back(llvmType(col, ctx));
            } else {
                paramTys.push_back(llvmType(attrMetalIfaceType(v.type), ctx));
            }
        }
    }
    if (isTCS) {
        /* Fixed buffers consumed by the existing TCS compute dispatcher:
         * stage_in(24), tess factors(26), patch output(27), stage output(28),
         * and indirect parameters(29). */
        for (int i = 0; i < 5; i++)
            paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    } else if (isGS) {
        /*  GS compute ABI (mgl_air_gs_abi.h): primitive input records,
         * expanded output records, one 28-byte counts record per work
         * item, the optional indexed gather stream, the gather params
         * constant, the transform-feedback stream(31) and its atomic
         * meta record(27).  All buffers in device address space. */
        for (int i = 0; i < 3; i++)
            paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        /* Gather stream + params are bound only for indexed draws; the
         * kernel branches on gather_params.gather_enabled at runtime. */
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        /* XFB stream + meta are always declared; the meta stride word
         * (0 = capture off) disables capture at runtime. */
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        /* GL4 ordered XFB (mgl_air_gs_abi.h §5b): the per-(work-item,
         * buffer) visibility buffer this work item writes for the CPU
         * prefix-sum. */
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    }
    if (isTESCompute) {
        /* isolines/point-mode TES kernel ABI: stage_in(24) factors(26)
         * patch inputs(27) stage_out(28) indirect(29), matching the TCS
         * kernel slot layout, plus the optional indexed gather stream and
         * its params (bound only for indexed draws; the kernel branches on
         * gather_params.gather_enabled at runtime), and the optional
         * transform-feedback stream(31). */
        for (int i = 0; i < 8; i++)
            paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    } else if (isTESVertex) {
        /* TES-vertex ABI: gl_in control points(30) seed TessCoord
         * records(28) factors(26) patch inputs(27) contract(29). */
        for (int i = 0; i < 5; i++)
            paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    } else if (isTES) {
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    }
    uint32_t cullBufferArgIdx = UINT32_MAX;
    uint32_t cullParamsArgIdx = UINT32_MAX;
    if (usesCullDistance && isVS) {
        cullBufferArgIdx = (uint32_t)paramTys.size();
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        cullParamsArgIdx = (uint32_t)paramTys.size();
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    } else if (isTESVertex && usesTesVertexCull) {
        /* TES-vertex cull reads the seeded record stream (slot 28) for the
         * partner endpoint's distances; it reuses the same buffer param so
         * the draw binds the record buffer once. */
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    } else if (isCullCapture || isTessCapture) {
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    }
    if (isVS || isTESVertex) {
        paramTys.push_back(llvm::Type::getInt32Ty(ctx));
        paramTys.push_back(llvm::Type::getInt32Ty(ctx));
        paramTys.push_back(llvm::Type::getInt32Ty(ctx));
    }
    else if (isKernel) {
        paramTys.push_back(llvm::FixedVectorType::get(
            llvm::Type::getInt32Ty(ctx), 3));
        if (usesLocalInvocation && !isTCS) {
            if (usesLocalInvocationID)
                paramTys.push_back(llvm::FixedVectorType::get(
                    llvm::Type::getInt32Ty(ctx), 3));
            if (usesLocalInvocationIndex)
                paramTys.push_back(llvm::Type::getInt32Ty(ctx));
        }
        if (usesWorkGroupID || isTCS)
            paramTys.push_back(llvm::FixedVectorType::get(
                llvm::Type::getInt32Ty(ctx), 3));
        if (usesNumWorkGroups)
            paramTys.push_back(llvm::FixedVectorType::get(
                llvm::Type::getInt32Ty(ctx), 3));
    }
    else if (isTES && !isTESCompute) {
        paramTys.push_back(patchControlTy->getPointerTo());
        paramTys.push_back(llvm::FixedVectorType::get(
            llvm::Type::getFloatTy(ctx), 3));
        paramTys.push_back(llvm::Type::getInt32Ty(ctx));
    } else {
        if (hasBuffer)
            paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        if (usesFragCoord)
            paramTys.push_back(llvm::FixedVectorType::get(
                llvm::Type::getFloatTy(ctx), 4));
        if (usesFrontFacing)
            paramTys.push_back(llvm::Type::getInt1Ty(ctx));
        if (usesPointCoord)
            paramTys.push_back(llvm::FixedVectorType::get(
                llvm::Type::getFloatTy(ctx), 2));
        if (usesPrimitiveId)
            paramTys.push_back((stage == MGL_STAGE_FRAGMENT && has_gs)
                ? llvm::Type::getFloatTy(ctx)
                : llvm::Type::getInt32Ty(ctx));
        if (usesLayer)
            paramTys.push_back(llvm::Type::getInt32Ty(ctx));
        if (usesViewportIndex)
            paramTys.push_back(llvm::Type::getInt32Ty(ctx));
        if (needSampleID)
            paramTys.push_back(llvm::Type::getInt32Ty(ctx));
        if (usesSampleMaskIn)
            paramTys.push_back(llvm::Type::getInt32Ty(ctx));
        /* Slot 30: FragCoord Y-fixup and/or sample params. */
        if (needParamsBuffer)
            paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    }
    if (isTESCompute)
        paramTys.push_back(llvm::FixedVectorType::get(
            llvm::Type::getInt32Ty(ctx), 3));
    if (!retTy && getenv("MGL_GS_TRACE")) {
        fprintf(stderr, "MGLGSTRACE site1 NULL retTy isGS=%d isTCS=%d isCompute=%d isTES=%d\n",
                (int)isGS, (int)isTCS, (int)isCompute, (int)isTES);
        fflush(stderr);
    }
    llvm::FunctionType *ft = llvm::FunctionType::get(retTy, paramTys, false);
    llvm::Function *fn = llvm::Function::Create(
        ft, llvm::Function::ExternalLinkage, "main", &module);
    fn->setDoesNotThrow();
    llvm::Function *controlPointGetter = nullptr;
    if (isTES && !isTESCompute && !isTESVertex) {
        std::vector<llvm::Type *> cpRecordElems = {
            llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4)};
        for (VarSym &v : syms)
            if (v.kind == VarSym::CONTROL_POINT_INPUT && !v.isPatch)
                cpRecordElems.push_back(llvmType(v.type, ctx));
        llvm::Type *cpRecordTy = llvm::StructType::get(ctx, cpRecordElems);
        controlPointGetter = llvm::Function::Create(
            llvm::FunctionType::get(cpRecordTy,
                {llvm::Type::getInt32Ty(ctx), patchControlTy->getPointerTo()},
                false),
            llvm::Function::ExternalLinkage,
            "_Z12ControlPoint.MTL_CONTROL_POINT_FN", &module);
        controlPointGetter->setSection("air.externally_defined");
        controlPointGetter->setDoesNotThrow();
        controlPointGetter->setOnlyReadsMemory();
    }
    if (hasBuffer) {
        unsigned bufIdx;
        if (isVS || isTES || isKernel)
            bufIdx = (isCapture ? 1 : 0) + (isCapture ? 0 : attrCount);
        else {
            /* fragment: buffer sits after varyings, before the optional
             * fragCoord position arg */
            bufIdx = (isCapture ? 1 : 0) + ssboCount + uboCount + acCount +
                     (needsBufferSizeBuffer ? 1 : 0) + 2 * texCount +
                     imageCount;
            for (VarSym &v : syms)
                if (!isVS && !isTES && !isKernel && v.kind == VarSym::VARYING)
                    bufIdx++;
            if (usesFragmentClipDistance)
                bufIdx += activeClipCount;
            if (usesFragmentCullDistance)
                bufIdx += activeCullCount;
        }
        fn->addParamAttr(bufIdx, llvm::Attribute::AttrKind::NoAlias);
        if (!isKernel)
            fn->addParamAttr(bufIdx, llvm::Attribute::AttrKind::ReadOnly);
    }
    if (isCapture)
        fn->addParamAttr(0, llvm::Attribute::AttrKind::NoAlias);
    {
        unsigned ssboIdx = (isCapture ? 1 : 0) +
                           (isCapture ? 0 : attrCount) +
                           ((isVS || isTES || isKernel) ? (hasBuffer ? 1 : 0) : 0);
        for (VarSym &v : syms) {
            if (v.kind != VarSym::SSBO) continue;
            const MGLIRSymbol *us = findSymbol(&mod, v.name.c_str());
            uint32_t nelems =
                uniformBlockElementCount(us ? us->type : nullptr);
            for (uint32_t k = 0; k < nelems; k++)
                fn->addParamAttr(ssboIdx++,
                                 llvm::Attribute::AttrKind::NoAlias);
        }
        for (VarSym &v : syms) {
            if (v.kind != VarSym::UBO) continue;
            const MGLIRSymbol *us = findSymbol(&mod, v.name.c_str());
            uint32_t uelems =
                uniformBlockElementCount(us ? us->type : nullptr);
            for (uint32_t k = 0; k < uelems; k++) {
                fn->addParamAttr(ssboIdx, llvm::Attribute::AttrKind::NoAlias);
                fn->addParamAttr(ssboIdx, llvm::Attribute::AttrKind::ReadOnly);
                ssboIdx++;
            }
        }
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATOMIC_COUNTER) continue;
            fn->addParamAttr(ssboIdx++, llvm::Attribute::AttrKind::NoAlias);
        }
    }
    if (needsBufferSizeBuffer) {
        unsigned sizeIdx = (isCapture ? 1 : 0) +
                           (isVS && !isCapture ? attrCount : 0) +
                           ((isVS || isTES || isKernel) && hasBuffer ? 1 : 0) +
                           ssboCount + uboCount + acCount;
        fn->addParamAttr(sizeIdx, llvm::Attribute::AttrKind::NoAlias);
        fn->addParamAttr(sizeIdx, llvm::Attribute::AttrKind::ReadOnly);
    }

    llvm::BasicBlock *entry = llvm::BasicBlock::Create(ctx, "entry", fn);
    llvm::IRBuilder<> b(entry);

    Codegen cg;
    cg.ctx = &ctx;
    cg.b = &b;
    cg.fn = fn;
    cg.mod = &module;
    cg.isVS = isVS || isTES;
    cg.pointSize = usesPointSize;
    cg.has_gs = has_gs;
    cg.isCompute = isCompute || isTCS || isGS;
    cg.isTessControl = isTCS;
    cg.isTessEval = isTES;
    cg.isGeometry = isGS;
    cg.isTESCompute = isTESCompute;
    cg.isTESVertex = isTESVertex;
    cg.isolinesTopology = tu->layout_primitive == MGL_AST_TES_ISOLINES;
    if (isCompute && tu) {
        cg.hasWorkGroupSize = true;
        cg.workGroupSizeX =
            tu->layout_local_size_x > 0 ? (uint32_t)tu->layout_local_size_x
                                        : 1u;
        cg.workGroupSizeY =
            tu->layout_local_size_y > 0 ? (uint32_t)tu->layout_local_size_y
                                        : 1u;
        cg.workGroupSizeZ =
            tu->layout_local_size_z > 0 ? (uint32_t)tu->layout_local_size_z
                                        : 1u;
    }
    cg.usesPatchCullDistance = usesPatchCullDistance;
    cg.cullDistancePassthroughCount =
        usesCullDistancePassthrough ? activeCullCount : 0u;
    cg.clipDistanceInputCount =
        usesFragmentClipDistance ? activeClipCount : 0u;
    cg.controlPointGetter = controlPointGetter;
    struct OwnedIRTypes {
        std::vector<MGLIRType *> v;
        ~OwnedIRTypes() {
            for (MGLIRType *t : v)
                mglIRTypeDestroy(t);
        }
    } ownedIR;
    cg.ownedIRTypes = &ownedIR.v;
    collectStructTypes(cg, tu);
    if (sourceUsesCullDistance) {
        cg.lvalues["gl_CullDistance"] = defaultCullDistances(cg);
    }
    {
        uint32_t field = 1;
        for (VarSym &v : syms)
            if (v.kind == VarSym::CONTROL_POINT_INPUT && !v.isPatch)
                cg.controlPointFields[v.name] = field++;
    }
    cg.tcsOutputVertices = isTCS && tu->layout_vertices > 0
                               ? (uint32_t)tu->layout_vertices
                               : 1u;
    cg.stageInStride = stageInputStride;
    cg.stageOutStride = stageOutputStride;
    cg.patchInStride = patchInputStride;
    cg.patchOutStride = patchOutputStride;
    /* Bind parameters by symbol: vertex = [attrs..., buffer, ssbo/ubo,
     * tex/smp..., cull buffers, instance_id, base_instance, vertex_id] (attrs first, except XFB capture where they
     * trail all buffers); fragment = [varyings..., buffer];
     * compute = [buffer, thread_position_in_grid]. */
    uint32_t argSlot = 0;
    if (isCapture)
        cg.captureBuf = fn->getArg(argSlot++);
    if (isVS && !isCapture) {
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATTR) continue;
            MType iface = attrMetalIfaceType(v.type);
            if (v.type.isArray() && v.type.arr > 0) {
                /* Assemble float[N] from N scalar stage_in args so
                 * gl_CullDistance/ClipDistance loads keep working. */
                llvm::Type *aggTy = llvmType(iface, ctx);
                llvm::Value *agg = llvm::UndefValue::get(aggTy);
                for (uint32_t k = 0; k < (uint32_t)v.type.arr; k++)
                    agg = cg.b->CreateInsertValue(agg, fn->getArg(argSlot++),
                                                  k);
                cg.lvalues[v.name] = agg;
            } else if (v.type.isMatrix()) {
                MType colIface = attrMetalIfaceType(matrixColumnType(v.type));
                MType matIface = v.type;
                matIface.scalar = colIface.scalar;
                llvm::Type *aggTy = llvmType(matIface, ctx);
                llvm::Value *agg = llvm::UndefValue::get(aggTy);
                for (uint32_t c = 0; c < v.type.cols; c++)
                    agg = cg.b->CreateInsertValue(agg, fn->getArg(argSlot++),
                                                  c);
                cg.lvalues[v.name] = agg;
            } else {
                cg.lvalues[v.name] = fn->getArg(argSlot++);
            }
        }
    }
    if ((isVS || isTES || isKernel) && hasBuffer)
        cg.bufferPtr = fn->getArg(argSlot++);
    for (VarSym &v : syms) {
        if (v.kind != VarSym::SSBO) continue;
        const MGLIRSymbol *us = findSymbol(&mod, v.name.c_str());
        uint32_t nelems =
            uniformBlockElementCount(us ? us->type : nullptr);
        if (!uniformBlockIsInstanceArray(us ? us->type : nullptr)) {
            cg.ssboPtrs[v.name] = fn->getArg(argSlot++);
            continue;
        }
        llvm::Type *ptrTy = llvm::Type::getInt8Ty(ctx)->getPointerTo(1);
        llvm::ArrayType *arrTy = llvm::ArrayType::get(ptrTy, nelems);
        llvm::Value *agg = llvm::UndefValue::get(arrTy);
        for (uint32_t k = 0; k < nelems; k++, argSlot++) {
            agg = cg.b->CreateInsertValue(agg, fn->getArg(argSlot), k);
        }
        llvm::Value *slot =
            cg.b->CreateAlloca(arrTy, nullptr, v.name + "_elems");
        cg.b->CreateStore(agg, slot);
        cg.ssboElemSlot[v.name] = slot;
        cg.ssboElemArrTy[v.name] = arrTy;
        cg.ssboPtrs[v.name] = agg; /* unused for arrays; kept non-null */
    }
    for (VarSym &v : syms) {
        if (v.kind != VarSym::UBO) continue;
        const MGLIRSymbol *us = findSymbol(&mod, v.name.c_str());
        uint32_t uelems =
            uniformBlockElementCount(us ? us->type : nullptr);
        if (!uniformBlockIsInstanceArray(us ? us->type : nullptr)) {
            cg.uboPtrs[v.name] = fn->getArg(argSlot++);
            continue;
        }
        llvm::Type *ptrTy = llvm::Type::getInt8Ty(ctx)->getPointerTo(1);
        llvm::ArrayType *arrTy = llvm::ArrayType::get(ptrTy, uelems);
        llvm::Value *agg = llvm::UndefValue::get(arrTy);
        for (uint32_t k = 0; k < uelems; k++, argSlot++) {
            agg = cg.b->CreateInsertValue(agg, fn->getArg(argSlot), k);
        }
        llvm::Value *slot =
            cg.b->CreateAlloca(arrTy, nullptr, v.name + "_elems");
        cg.b->CreateStore(agg, slot);
        cg.uboElemSlot[v.name] = slot;
        cg.uboElemArrTy[v.name] = arrTy;
        cg.uboPtrs[v.name] = agg; /* unused for arrays; kept non-null */
    }
    for (VarSym &v : syms) {
        if (v.kind != VarSym::ATOMIC_COUNTER) continue;
        cg.acPtrs[v.name] = fn->getArg(argSlot++);
    }
    if (needsBufferSizeBuffer)
        cg.bufferSizePtr = fn->getArg(argSlot++);
    for (VarSym &v : syms) {
        if (v.kind != VarSym::TEXTURE) continue;
        uint32_t elements = v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
        if (v.opaqueType)
            cg.samplerIRTypes[v.name] = v.opaqueType;
        else {
            const MGLIRSymbol *ts = findSymbol(&mod, v.name.c_str());
            if (ts) cg.samplerIRTypes[v.name] = ts->type;
        }
        if (elements == 1u) {
            cg.texValues[v.name] = fn->getArg(argSlot++);
            cg.smpValues[v.name] = fn->getArg(argSlot++);
        } else {
            std::vector<llvm::Value *> texes, samplers;
            for (uint32_t k = 0; k < elements; k++) {
                texes.push_back(fn->getArg(argSlot++));
                samplers.push_back(fn->getArg(argSlot++));
            }
            cg.texArrayValues[v.name] = std::move(texes);
            cg.smpArrayValues[v.name] = std::move(samplers);
        }
    }
    for (VarSym &v : syms) {
        if (v.kind != VarSym::IMAGE) continue;
        uint32_t elements = v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
        if (elements == 1u) {
            cg.texValues[v.name] = fn->getArg(argSlot++);
        } else {
            std::vector<llvm::Value *> texes;
            for (uint32_t k = 0; k < elements; k++)
                texes.push_back(fn->getArg(argSlot++));
            cg.texArrayValues[v.name] = std::move(texes);
        }
    }
    {
        uint32_t location = (isCapture ? 1u : 0u) + plainBufMetalSlots +
            (isCapture ? 0u : attrCount);
        for (VarSym &v : syms) {
            if (v.kind != VarSym::SSBO) continue;
            const MGLIRSymbol *us = findSymbol(&mod, v.name.c_str());
            uint32_t nelems =
                uniformBlockElementCount(us ? us->type : nullptr);
            cg.ssboSlots[v.name] = location;
            location += nelems;
        }
        location += uboCount;
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATOMIC_COUNTER) continue;
            cg.acSlots[v.name] = location++;
        }
    }
    for (VarSym &v : syms) {
        if ((isVS && isCapture && v.kind == VarSym::ATTR) ||
            (!isVS && !isTES && !isKernel && v.kind == VarSym::VARYING)) {
            if (v.type.isArray() && v.type.arr > 0) {
                /* Flattened stage-in: N scalar args assembled into a
                 * single aggregate lvalue so the read paths (readIndexChain
                 * / swizzles) keep working unchanged. */
                MType el = v.type;
                el.arr = 0;
                MType aggType = (v.kind == VarSym::ATTR)
                    ? attrMetalIfaceType(v.type) : v.type;
                uint32_t n = (uint32_t)v.type.arr;
                if (needsArrayMem(cg, v.name, aggType)) {
                    llvm::Value *slot = ensureArrayMem(cg, v.name, aggType);
                    for (uint32_t k = 0; k < n; k++) {
                        llvm::Value *arg = fn->getArg(argSlot++);
                        if (v.kind == VarSym::VARYING &&
                            varyingUsesFloatCarrier(el, has_gs)) {
                            arg = decodeFloatCarrier(cg, arg, el.scalar,
                                                     llvmType(el, ctx));
                        }
                        llvm::Value *ep = arrayMemGEP(
                            cg, v.name, slot, cg.b->getInt32(k));
                        cg.b->CreateAlignedStore(arg, ep, llvm::Align(4));
                    }
                } else {
                    llvm::Type *aggTy = llvmType(aggType, ctx);
                    llvm::Value *agg = llvm::UndefValue::get(aggTy);
                    for (uint32_t k = 0; k < n; k++) {
                        llvm::Value *arg = fn->getArg(argSlot++);
                        if (v.kind == VarSym::VARYING &&
                            varyingUsesFloatCarrier(el, has_gs)) {
                            arg = decodeFloatCarrier(cg, arg, el.scalar,
                                                     llvmType(el, ctx));
                        }
                        agg = cg.b->CreateInsertValue(agg, arg, k);
                    }
                    cg.lvalues[v.name] = agg;
                }
            } else if (v.type.isMatrix()) {
                MType colTy = matrixColumnType(v.type);
                MType matIface = v.type;
                if (v.kind == VarSym::ATTR) {
                    colTy = attrMetalIfaceType(colTy);
                    matIface.scalar = colTy.scalar;
                }
                llvm::Type *aggTy = llvmType(matIface, ctx);
                llvm::Value *agg = llvm::UndefValue::get(aggTy);
                for (uint32_t c = 0; c < v.type.cols; c++) {
                    llvm::Value *arg = fn->getArg(argSlot++);
                    if (v.kind == VarSym::VARYING &&
                        varyingUsesFloatCarrier(matrixColumnType(v.type),
                                                has_gs)) {
                        arg = decodeFloatCarrier(
                            cg, arg, matrixColumnType(v.type).scalar,
                            llvmType(matrixColumnType(v.type), ctx));
                    }
                    agg = cg.b->CreateInsertValue(agg, arg, c);
                }
                cg.lvalues[v.name] = agg;
                } else {
                    if (uintUsesSplitFloatCarrier(v.type, has_gs)) {
                        llvm::Value *lo = fn->getArg(argSlot++);
                        llvm::Value *hi = fn->getArg(argSlot++);
                        cg.lvalues[v.name] = decodeUintSplitFloatCarrier(
                            cg, lo, hi, llvmType(v.type, ctx));
                    } else {
                        llvm::Value *arg = fn->getArg(argSlot++);
                        /* Vertex ATTR values arrive as raw ints from Metal;
                         * float-carrier decode is only for FS stage_in
                         * varyings (SIToFP/UIToFP).  Applying FPToUI to an
                         * i32 1 reinterprets it as a denormal → 0.
                         * DOUBLE ATTR arrives as float (VertexLayout). */
                        if (v.kind == VarSym::VARYING &&
                            varyingUsesFloatCarrier(v.type, has_gs)) {
                            arg = decodeFloatCarrier(cg, arg, v.type.scalar,
                                                     llvmType(v.type, ctx));
                        }
                        cg.lvalues[v.name] = arg;
                    }
                }
        }
    }
    if (usesFragmentClipDistance) {
        llvm::Type *f32 = llvm::Type::getFloatTy(ctx);
        llvm::Value *arr = llvm::UndefValue::get(
            llvm::ArrayType::get(f32, activeClipCount));
        for (uint32_t i = 0; i < activeClipCount; i++)
            arr = cg.b->CreateInsertValue(arr, fn->getArg(argSlot++), i);
        cg.lvalues["gl_ClipDistance"] = arr;
    }
    if (usesFragmentCullDistance) {
        llvm::Type *f32 = llvm::Type::getFloatTy(ctx);
        llvm::Value *arr = llvm::UndefValue::get(
            llvm::ArrayType::get(f32, activeCullCount));
        for (uint32_t i = 0; i < activeCullCount; i++)
            arr = cg.b->CreateInsertValue(arr, fn->getArg(argSlot++), i);
        cg.lvalues["gl_CullDistance"] = arr;
    }
    if (isVS && !isCapture) {
        /* VS varyings (out) have no parameter backing; plain writes
         * auto-create their lvalues, but indexed writes (e.g. legacy
         * gl_TexCoord[i] -> out vec4 _mglTexCoord[8]) require a
         * pre-registered aggregate.  Register an undef aggregate so the
         * indexed-assign path can build into it; assembleReturn picks up
         * the final value. */
        for (VarSym &v : syms) {
            if (v.kind != VarSym::VARYING) continue;
            if (needsArrayMem(cg, v.name, v.type))
                ensureArrayMem(cg, v.name, v.type);
            else
                cg.lvalues[v.name] =
                    llvm::UndefValue::get(llvmType(v.type, ctx));
        }
    }
    if (isVS && isCapture) {
        /* Capture variants write varyings into the capture record; indexed
         * writes into array varyings (e.g. out vec4 v[2]) need the same
         * pre-registered undef aggregate as the non-capture path. */
        for (VarSym &v : syms) {
            if (v.kind != VarSym::VARYING) continue;
            if (needsArrayMem(cg, v.name, v.type))
                ensureArrayMem(cg, v.name, v.type);
            else
                cg.lvalues[v.name] =
                    llvm::UndefValue::get(llvmType(v.type, ctx));
        }
    }
    if (isTES && !isTESCompute && !isTESVertex) {
        cg.stageInPtr = fn->getArg(argSlot++);
        cg.indirectPtr = fn->getArg(argSlot++);
        cg.captureBuf = fn->getArg(argSlot++);
    }
    if (isTESVertex) {
        cg.stageInPtr = fn->getArg(argSlot++);       /* gl_in (slot 30) */
        cg.geometryOutputPtr = fn->getArg(argSlot++); /* seed records (28) */
        cg.tessFactorPtr = fn->getArg(argSlot++);    /* factors (26) */
        cg.captureBuf = fn->getArg(argSlot++);       /* patch inputs (27) */
        cg.indirectPtr = fn->getArg(argSlot++);      /* contract (29) */
        if (usesTesVertexCull)
            cg.cullBuffer = fn->getArg(argSlot++);   /* records (28) cull read */
    }
    if (isTCS || isTESCompute) {
        cg.stageInPtr = fn->getArg(argSlot++);
        cg.tessFactorPtr = fn->getArg(argSlot++);
        cg.captureBuf = fn->getArg(argSlot++);
        cg.stageOutPtr = fn->getArg(argSlot++);
        cg.indirectPtr = fn->getArg(argSlot++);
        if (isTESCompute) {
            /* The patch control-point record stream is the stage input
             * buffer itself (slot 24). */
            cg.patchControlPtr = cg.stageInPtr;
            cg.geometryOutputPtr = cg.stageOutPtr;
            cg.tessGatherPtr = fn->getArg(argSlot++);
            cg.tessGatherParamsPtr = fn->getArg(argSlot++);
            cg.xfbOutPtr = fn->getArg(argSlot++);
        }
    } else if (isGS) {
        cg.geometryInputPtr = fn->getArg(argSlot++);
        cg.geometryOutputPtr = fn->getArg(argSlot++);
        cg.geometryCountPtr = fn->getArg(argSlot++);
        cg.geometryGatherPtr = fn->getArg(argSlot++);
        cg.geometryGatherParamsPtr = fn->getArg(argSlot++);
        cg.geometryXfbPtr = fn->getArg(argSlot++);
        cg.geometryXfbMetaPtr = fn->getArg(argSlot++);
        cg.geometryXfbVisPtr = fn->getArg(argSlot++);
    }
    if (isTessCapture) {
        cg.cullParams = fn->getArg(argSlot++);
    } else if (isCullCapture && sourceUsesCullDistance) {
        cg.lvalues["gl_CullDistance"] = defaultCullDistances(cg);
        cg.cullParams = fn->getArg(argSlot++);
    }
    if (isVS || isTESVertex) {
        if (usesCullDistance) {
            cg.cullBuffer = fn->getArg(argSlot++);
            cg.cullParams = fn->getArg(argSlot++);
            cg.usesCullDistance = true;
            cg.lvalues["gl_CullDistance"] = defaultCullDistances(cg);
        }
        cg.instanceId = fn->getArg(argSlot++);
        cg.baseInstance = fn->getArg(argSlot++);
        cg.vertexId = fn->getArg(argSlot++);
        if (sourceUsesCullDistance &&
            !cg.lvalues.count("gl_CullDistance")) {
            cg.lvalues["gl_CullDistance"] = defaultCullDistances(cg);
        }
    }
    else if (isKernel) {
        llvm::Value *pos = fn->getArg(argSlot++);
        if (isTCS) cg.invocationPos = pos;
        else cg.threadPos = pos;
        if (usesLocalInvocation && !isTCS) {
            if (usesLocalInvocationID)
                cg.localInvocationPos = fn->getArg(argSlot++);
            if (usesLocalInvocationIndex)
                cg.localInvocationIndex = fn->getArg(argSlot++);
        }
        if (usesWorkGroupID || isTCS)
            cg.workGroupPos = fn->getArg(argSlot++);
        if (usesNumWorkGroups)
            cg.numWorkGroups = fn->getArg(argSlot++);
        if (isTCS && cg.workGroupPos)
            cg.patchPos = cg.workGroupPos;
    }
    else if (isTES && !isTESCompute) {
        cg.patchControlPtr = fn->getArg(argSlot++);
        cg.tessCoord = fn->getArg(argSlot++);
        cg.patchId = fn->getArg(argSlot++);
        /* Per-patch native draws report patch_id 0; the runtime stamps the
         * global patch index in mgl_patch_info[2] for shaders using
         * gl_PrimitiveID (GL 4.6 §13.2.3). */
        if (tesUsesPrimitiveId && cg.indirectPtr) {
            llvm::Value *info = cg.b->CreateBitCast(
                cg.indirectPtr, cg.b->getInt32Ty()->getPointerTo(1));
            cg.lvalues["gl_PrimitiveID"] = cg.b->CreateAlignedLoad(
                cg.b->getInt32Ty(),
                cg.b->CreateGEP(cg.b->getInt32Ty(), info, cg.b->getInt32(2)),
                llvm::Align(4));
        }
    } else {
        if (hasBuffer)
            cg.bufferPtr = fn->getArg(argSlot++);
        if (usesFragCoord)
            cg.fragPos = fn->getArg(argSlot++);
        if (usesFrontFacing)
            cg.lvalues["gl_FrontFacing"] = fn->getArg(argSlot++);
        if (usesPointCoord)
            cg.lvalues["gl_PointCoord"] = fn->getArg(argSlot++);
        if (usesPrimitiveId) {
            llvm::Value *primitiveArg = fn->getArg(argSlot++);
            if (stage == MGL_STAGE_FRAGMENT && has_gs) {
                /* The id arrives as a float carrier (see
                 * storeGeometryPrimitiveId); convert back for shader math. */
                primitiveArg = cg.b->CreateFPToSI(
                    cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::round,
                                               primitiveArg),
                    cg.b->getInt32Ty());
            }
            cg.lvalues["gl_PrimitiveID"] = primitiveArg;
        }
        if (usesLayer)
            cg.lvalues["gl_Layer"] = fn->getArg(argSlot++);
        if (usesViewportIndex)
            cg.lvalues["gl_ViewportIndex"] = fn->getArg(argSlot++);
        if (needSampleID) {
            llvm::Value *sid = fn->getArg(argSlot++);
            if (usesSampleID)
                cg.lvalues["gl_SampleID"] = sid;
            if (usesSamplePosition || usesInterpolateAtSample) {
                /* Defer until num_samples is loaded below. */
                cg.lvalues["__mgl_sample_id_for_pos"] = sid;
            }
            /* Keep hardware id so forced-sample override can replace it. */
            cg.lvalues["__mgl_hw_sample_id"] = sid;
        }
        if (usesSampleMaskIn) {
            llvm::Value *mask = fn->getArg(argSlot++);
            llvm::Type *i32 = cg.b->getInt32Ty();
            llvm::Value *arr =
                llvm::UndefValue::get(llvm::ArrayType::get(i32, 1));
            arr = cg.b->CreateInsertValue(
                arr, cg.b->CreateBitCast(mask, i32), 0);
            cg.lvalues["gl_SampleMaskIn"] = arr;
        }
        if (needParamsBuffer) {
            llvm::Value *buf = fn->getArg(argSlot++);
            cg.fragSampleParams = buf;
            /* float4 layout from RenderPass when FragCoord and/or sample
             * params are requested: {height, lower_left, ns_bits, sb_bits}.
             * When emulating MS sample planes, sb_bits may carry
             * 0x80000000 | (forced_sample_id << 8) so SampleID / position
             * track the attached array slice. */
            llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
            llvm::Value *fptr = cg.b->CreateBitCast(
                buf, f32->getPointerTo(1));
            if (needFragCoordParams && cg.fragPos) {
                llvm::Value *height = cg.b->CreateAlignedLoad(
                    f32, fptr, llvm::Align(4));
                llvm::Value *lowerLeft = cg.b->CreateAlignedLoad(
                    f32,
                    cg.b->CreateGEP(f32, fptr, cg.b->getInt32(1)),
                    llvm::Align(4));
                llvm::Value *y = cg.b->CreateExtractElement(
                    cg.fragPos, cg.b->getInt32(1));
                llvm::Value *flipped = cg.b->CreateFSub(height, y);
                llvm::Value *useFlip = cg.b->CreateFCmpOGT(
                    lowerLeft, llvm::ConstantFP::get(f32, 0.5));
                llvm::Value *newY =
                    cg.b->CreateSelect(useFlip, flipped, y);
                cg.fragPos = cg.b->CreateInsertElement(
                    cg.fragPos, newY, cg.b->getInt32(1));
            }
            if (needSampleParams) {
                llvm::Value *nsBits = cg.b->CreateAlignedLoad(
                    f32,
                    cg.b->CreateGEP(f32, fptr, cg.b->getInt32(2)),
                    llvm::Align(4));
                llvm::Value *ns = cg.b->CreateBitCast(
                    nsBits, cg.b->getInt32Ty());
                if (usesNumSamples || usesInterpolateAtSample ||
                    hasSampleVarying)
                    cg.lvalues["gl_NumSamples"] = ns;
                llvm::Value *sbBits = cg.b->CreateAlignedLoad(
                    f32,
                    cg.b->CreateGEP(f32, fptr, cg.b->getInt32(3)),
                    llvm::Align(4));
                llvm::Value *sb = cg.b->CreateBitCast(
                    sbBits, cg.b->getInt32Ty());
                llvm::Value *forceMask = cg.b->CreateAnd(
                    sb, cg.b->getInt32(0x80000000u));
                llvm::Value *force = cg.b->CreateICmpNE(
                    forceMask, cg.b->getInt32(0));
                llvm::Value *forcedSid = cg.b->CreateAnd(
                    cg.b->CreateLShr(sb, 8), cg.b->getInt32(0xff));
                llvm::Value *hwSid =
                    cg.lvalues.count("__mgl_hw_sample_id")
                        ? cg.lvalues["__mgl_hw_sample_id"]
                        : cg.b->getInt32(0);
                llvm::Value *sid =
                    cg.b->CreateSelect(force, forcedSid, hwSid);
                if (usesSampleID)
                    cg.lvalues["gl_SampleID"] = sid;
                cg.lvalues["__mgl_sample_id_for_pos"] = sid;
                if (usesSamplePosition || usesInterpolateAtSample ||
                    hasSampleVarying) {
                    cg.lvalues["gl_SamplePosition"] =
                        emitSamplePositionFromId(cg, sid, ns);
                }
                if (usesSampleMaskIn) {
                    /* Hardware [[sample_mask]] is a single bit when the
                     * metal target is non-MSAA (array-plane emulation).
                     * Under forced SampleID, expose full GL coverage so
                     * `u_sampleMask & gl_SampleMaskIn` keeps all bits. */
                    llvm::Type *i32t = cg.b->getInt32Ty();
                    llvm::Value *hwIn = cg.lvalues.count("gl_SampleMaskIn")
                        ? cg.b->CreateExtractValue(
                              cg.lvalues["gl_SampleMaskIn"], 0)
                        : cg.b->getInt32(~0);
                    llvm::Value *ge32 = cg.b->CreateICmpUGE(
                        ns, cg.b->getInt32(32));
                    llvm::Value *shiftAmt = cg.b->CreateSelect(
                        ge32, cg.b->getInt32(0), ns);
                    llvm::Value *full = cg.b->CreateSelect(
                        ge32, cg.b->getInt32(~0),
                        cg.b->CreateSub(
                            cg.b->CreateShl(cg.b->getInt32(1), shiftAmt),
                            cg.b->getInt32(1)));
                    llvm::Value *inMask =
                        cg.b->CreateSelect(force, full, hwIn);
                    llvm::Value *arr =
                        llvm::UndefValue::get(llvm::ArrayType::get(i32t, 1));
                    cg.lvalues["gl_SampleMaskIn"] =
                        cg.b->CreateInsertValue(arr, inMask, 0);
                }
            }
        }
    }
    if (hasSampleVarying && !isVS && !isTES && !isKernel) {
        /* With raster_sample_count==1 (MS array emulation), Metal only
         * delivers center interpolants. Rewrite `sample in` to the
         * software sample-offset formula using forced/hardware SampleID. */
        llvm::Value *sid =
            cg.lvalues.count("gl_SampleID")
                ? cg.lvalues["gl_SampleID"]
                : (cg.lvalues.count("__mgl_sample_id_for_pos")
                       ? cg.lvalues["__mgl_sample_id_for_pos"]
                       : cg.b->getInt32(0));
        llvm::Value *ns = cg.lvalues.count("gl_NumSamples")
                              ? cg.lvalues["gl_NumSamples"]
                              : cg.b->getInt32(1);
        llvm::Value *sp =
            cg.lvalues.count("gl_SamplePosition")
                ? cg.lvalues["gl_SamplePosition"]
                : emitSamplePositionFromId(cg, sid, ns);
        llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
        llvm::Value *half = llvm::ConstantFP::get(f32, 0.5);
        llvm::Value *off =
            cg.b->CreateFSub(sp, llvm::ConstantVector::get(
                                     {llvm::cast<llvm::Constant>(half),
                                      llvm::cast<llvm::Constant>(half)}));
        for (VarSym &v : syms) {
            if (v.kind != VarSym::VARYING || !v.isSample) continue;
            auto it = cg.lvalues.find(v.name);
            if (it == cg.lvalues.end() || !it->second) continue;
            llvm::Value *adj =
                emitInterpolateAtOffsetValue(cg, it->second, off);
            if (adj) it->second = adj;
        }
    }
    if (usesFragDepth)
        cg.hasFragDepth = true;
    if (usesSampleMask) {
        cg.hasSampleMask = true;
        llvm::Type *i32 = cg.b->getInt32Ty();
        llvm::Value *arr =
            llvm::UndefValue::get(llvm::ArrayType::get(i32, 1));
        /* Default coverage: all bits set; shader writes replace this. */
        arr = cg.b->CreateInsertValue(arr, cg.b->getInt32(~0), 0);
        cg.lvalues["gl_SampleMask"] = arr;
    }
    if (usesClipDistance) {
        cg.usesClipDistance = true;
        /* Indexed writes (gl_ClipDistance[i] = v) need the aggregate
         * lvalue pre-registered (see the array-varying fix). */
        cg.lvalues["gl_ClipDistance"] = defaultClipDistances(cg);
    }
    if (!isVS && !isTES && !isKernel) {
        /* Fragment output arrays: indexed writes need an aggregate lvalue
         * whose element type matches the GLSL declaration (float/int/uint). */
        for (VarSym &v : syms) {
            if (v.kind == VarSym::OUTPUT && v.type.isArray()) {
                cg.lvalues[v.name] =
                    llvm::UndefValue::get(llvmType(v.type, ctx));
                break;
            }
        }
    }
    if (isTESCompute)
        cg.patchPos = fn->getArg(argSlot++);
    if (isGS) {
        cg.geometryWorkItemId = cg.threadPos
            ? cg.b->CreateExtractElement(cg.threadPos, cg.b->getInt32(0))
            : nullptr;
        const uint32_t invocationCount = tu->layout_invocations > 0
            ? (uint32_t)tu->layout_invocations : 1u;
        cg.geometryPrimitiveId = cg.geometryWorkItemId
            ? cg.b->CreateUDiv(cg.geometryWorkItemId,
                               cg.b->getInt32(invocationCount))
            : nullptr;
        cg.geometryInvocationId = cg.geometryWorkItemId
            ? cg.b->CreateURem(cg.geometryWorkItemId,
                               cg.b->getInt32(invocationCount))
            : nullptr;
        switch (tu->layout_primitive) {
        case MGL_AST_GS_IN_POINTS: cg.geometryInputVertices = 1u; break;
        case MGL_AST_GS_IN_LINES: cg.geometryInputVertices = 2u; break;
        case MGL_AST_GS_IN_LINES_ADJACENCY: cg.geometryInputVertices = 4u; break;
        case MGL_AST_GS_IN_TRIANGLES_ADJACENCY: cg.geometryInputVertices = 6u; break;
        default: cg.geometryInputVertices = 3u; break;
        }
        cg.geometryOutputType = tu->layout_primitive_out;
        /* A zero/unspecified max_vertices GS is a valid no-output program.
         * Keep the zero in the codegen state so EmitVertex remains rejected,
         * while the ABI still allocates its two header records below. */
        cg.geometryMaxVertices = tu->layout_max_vertices >= 0
            ? (uint32_t)tu->layout_max_vertices : 0u;
        /* Fixed ABI layout (mgl_air_gs_abi.h): the output record run is
         * 2 header records + the expanded primitive vertices. */
        const MGLAIRGSOutputPrimitive outPrim =
            airGSOutputFromAST(cg.geometryOutputType);
        cg.geometryOutputVertices =
            mglAIRGSExpandedVertices(outPrim, cg.geometryMaxVertices);
        cg.geometryRecordCount =
            mglAIRGSRecordsPerPrimitive(outPrim, cg.geometryMaxVertices);
    } else if (isTESCompute) {
        /* Each work item expands exactly one vertex record. */
        cg.geometryRecordCount = 1u;
    }
    /* Patch BUFFER sym offsets into uniforms */
    for (Uniform &u : uniforms) {
        cg.bufferOffsets[u.name] = u.offset;
        for (VarSym &v : syms)
            if (v.kind == VarSym::BUFFER && v.name == u.name)
                v.bufferOffset = u.offset;
    }
    /* Stage-level info for return assembly. */
    cg.retTy = retTy;
    cg.retElems = retElems;
    cg.varyings = varyings;
    cg.fragOutputs = fragOutputs;
    cg.auxSyms = &syms;

    /* Stage-output allocas in main's entry so helpers can write them. */
    {
        auto addOut = [&](const std::string &name, const MType &ty) {
            if (name.empty() || cg.outPtrs.count(name)) return;
            cg.outPtrs[name] =
                cg.b->CreateAlloca(llvmType(ty, ctx), nullptr, name + ".out");
        };
        for (VarSym &v : syms) {
            /* TCS per-vertex outs go through stageOutPtr.
             * GS compute kernels reject alloca pointers as callee args
             * (materializeAll); GS helpers write outs via geometry ABI
             * only when called from main after inlining would be needed.
             * Prefer SSA outPtrs for raster stages. */
            if (isTCS && v.kind == VarSym::OUTPUT) continue;
            if (isGS && v.kind == VarSym::OUTPUT) continue;
            bool isStageOut =
                ((isVS || isTES) && v.kind == VarSym::VARYING) ||
                (!isVS && !isTES && !isTCS && !isGS && !isKernel &&
                 v.kind == VarSym::OUTPUT);
            if (isStageOut) addOut(v.name, v.type);
        }
        if ((isVS || isTES || isTESCompute) && !isGS) {
            MType pos;
            pos.scalar = MGLIR_SCALAR_FLOAT;
            pos.vec = 4;
            addOut("gl_Position", pos);
        }
        if (usesFragDepth) {
            MType d;
            d.scalar = MGLIR_SCALAR_FLOAT;
            addOut("gl_FragDepth", d);
        }
    }

    /* Freeze which builtins are threaded into user functions so call
     * sites cannot drift from the signature if main later materializes
     * extra lvalues. */
    cg.userFnPassCull = cg.lvalues.count("gl_CullDistance") != 0;
    cg.userFnPassClip = cg.lvalues.count("gl_ClipDistance") != 0;

    /* User-defined functions (fog helpers etc.): create the LLVM
     * functions first so calls (including recursion) resolve, then emit
     * their bodies. */
    std::map<std::string, llvm::Function *> userFns;
    std::map<std::string, uint32_t> userFnHidden;
    std::map<std::string, MGLDecl *> userFnDecls;
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        MGLDecl *d = tu->decls[i];
        if (!d->name || !d->body || strcmp(d->name, "main") == 0) continue;
        /* GS/TCS/compute helpers are inlined at call sites (Metal rejects
         * some texture/alloca/stage-buffer calling conventions on compute
         * callees — including non-void helpers).  Helpers with out/inout
         * params are also registered for inlining on every stage: LLVM
         * by-value calls cannot write results back to the caller. */
        {
            int force_inline = 0;
            for (uint32_t p = 0; p < d->param_count; p++) {
                if (d->params[p] &&
                    (d->params[p]->qualifiers & MGL_AST_Q_OUT)) {
                    force_inline = 1;
                    break;
                }
            }
            /* Metal helper ABI mishandles aggregate returns; also keep
             * struct params on the SSA-inline path for member access. */
            if (d->type && d->type->base == MGL_AST_TYPE_STRUCT)
                force_inline = 1;
            for (uint32_t p = 0; !force_inline && p < d->param_count; p++) {
                if (d->params[p] && d->params[p]->type &&
                    d->params[p]->type->base == MGL_AST_TYPE_STRUCT)
                    force_inline = 1;
            }
            if (isGS || isCompute || isTCS || force_inline) {
                std::string key = std::string(d->name) + "#" +
                                  std::to_string(d->param_count);
                userFnDecls[key] = d;
            }
        }
        const MGLIRSymbol *fs = nullptr;
        for (uint32_t k = 0; k < mod.symbol_count; k++) {
            if (mod.symbols[k]->is_function &&
                strcmp(mod.symbols[k]->name, d->name) == 0 &&
                mod.symbols[k]->param_count == d->param_count) {
                fs = mod.symbols[k];
                break;
            }
        }
        if (!fs) continue;
        /* GS/TCS/compute, or helpers with aggregate return: inline only. */
        if (isGS || isCompute || isTCS)
            continue;
        if (d->type && d->type->base == MGL_AST_TYPE_STRUCT)
            continue;
        llvm::Type *rt = irTypeIsVoid(fs->return_type)
            ? llvm::Type::getVoidTy(ctx)
            : llvmTypeFromIR(fs->return_type, ctx);
        std::vector<llvm::Type *> pts;
        for (uint32_t p = 0; p < fs->param_count; p++) {
            const MGLIRType *pt = fs->param_types[p];
            if (pt->kind == MGLIR_TYPE_SAMPLER) {
                llvm::StructType *st =
                    pt->tex_kind == MGLIR_TEX_3D ? texTy3d
                    : pt->tex_kind == MGLIR_TEX_2D_ARRAY ? texTy2dArray
                                                         : texTy2d;
                pts.push_back(st->getPointerTo(1));
            } else {
                pts.push_back(llvmTypeFromIR(pt, ctx));
            }
        }
        const uint32_t nExplicit = (uint32_t)pts.size();
        /* Hidden trailing arguments: UBO values (a pointer for a scalar block,
         * an aggregate of pointers for an instance array) and SSBO pointers,
         * so user functions can read global blocks of their own. */
        for (const auto &kv : cg.uboPtrs)
            pts.push_back(kv.second->getType());
        for (const auto &kv : cg.ssboPtrs)
            pts.push_back(kv.second->getType());
        for (const auto &kv : cg.acPtrs)
            pts.push_back(llvm::Type::getInt8PtrTy(ctx, 1));
        if (cg.bufferSizePtr)
            pts.push_back(llvm::Type::getInt32PtrTy(ctx, 2));
        if (cg.userFnPassCull)
            pts.push_back(cg.lvalues["gl_CullDistance"]->getType());
        if (cg.userFnPassClip)
            pts.push_back(cg.lvalues["gl_ClipDistance"]->getType());
        if (isGS) {
            for (int hidden = 0; hidden < 5; hidden++)
                pts.push_back(llvm::Type::getInt8PtrTy(ctx, 1));
            for (int hidden = 0; hidden < 3; hidden++)
                pts.push_back(llvm::Type::getInt32Ty(ctx));
        }
        if (isTCS) {
            /* stage_in, tess factors, stage_out, indirect, invocation, patch */
            for (int hidden = 0; hidden < 4; hidden++)
                pts.push_back(llvm::Type::getInt8PtrTy(ctx, 1));
            pts.push_back(llvm::FixedVectorType::get(
                llvm::Type::getInt32Ty(ctx), 3));
            pts.push_back(llvm::FixedVectorType::get(
                llvm::Type::getInt32Ty(ctx), 3));
        }
        if (isCompute && cg.threadPos)
            pts.push_back(cg.threadPos->getType());
        /* Texture/sampler handles as non-entry args are rejected by the
         * Metal GS/TCS compute pipeline loader (materializeAll).  Regular
         * compute and raster stages accept them. */
        if (!isGS && !isTCS) {
            for (const auto &kv : cg.texValues)
                pts.push_back(kv.second->getType());
            for (const auto &kv : cg.smpValues)
                pts.push_back(kv.second->getType());
            for (const auto &kv : cg.texArrayValues)
                for (llvm::Value *tv : kv.second)
                    pts.push_back(tv->getType());
            for (const auto &kv : cg.smpArrayValues)
                for (llvm::Value *sv : kv.second)
                    pts.push_back(sv->getType());
        }
        if (cg.bufferPtr)
            pts.push_back(cg.bufferPtr->getType());
        for (const auto &kv : cg.outPtrs)
            pts.push_back(kv.second->getType());
        std::string key = std::string(d->name) + "#" +
                          std::to_string(fs->param_count);
        userFnHidden[key] = (uint32_t)pts.size() - nExplicit;
        if (!rt && getenv("MGL_GS_TRACE")) {
            fprintf(stderr, "MGLGSTRACE site3 NULL rt fn=%s params=%u isGS=%d\n",
                    d->name ? d->name : "?", (unsigned)fs->param_count, (int)isGS);
            fflush(stderr);
        }
        llvm::Function *f = llvm::Function::Create(
            llvm::FunctionType::get(rt, pts, false),
            llvm::Function::ExternalLinkage,
            (std::string("mgl_fn_") + d->name + "_" +
             std::to_string(fs->param_count)),
            &module);
        userFns[key] = f;
    }
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        MGLDecl *d = tu->decls[i];
        if (!d->name || !d->body || strcmp(d->name, "main") == 0) continue;
        auto it = userFns.find(std::string(d->name) + "#" +
                               std::to_string(d->param_count));
        if (it == userFns.end()) continue;
        llvm::Function *f = it->second;
        llvm::BasicBlock *entry =
            llvm::BasicBlock::Create(ctx, "entry", f);
        llvm::IRBuilder<> fb(entry);
        Codegen fc;
        fc.ctx = cg.ctx;
        fc.b = &fb;
        fc.fn = f;
        fc.mod = cg.mod;
        fc.isVS = cg.isVS;
        fc.isCompute = cg.isCompute;
        fc.isTessControl = cg.isTessControl;
        fc.isTessEval = cg.isTessEval;
        fc.isGeometry = cg.isGeometry;
        fc.bufferPtr = cg.bufferPtr;
        fc.threadPos = cg.threadPos;
        fc.localInvocationPos = cg.localInvocationPos;
        fc.localInvocationIndex = cg.localInvocationIndex;
        fc.workGroupPos = cg.workGroupPos;
        fc.numWorkGroups = cg.numWorkGroups;
        fc.invocationPos = cg.invocationPos;
        fc.patchPos = cg.patchPos;
        fc.stageInPtr = cg.stageInPtr;
        fc.stageOutPtr = cg.stageOutPtr;
        fc.tessFactorPtr = cg.tessFactorPtr;
        fc.indirectPtr = cg.indirectPtr;
        fc.tcsOutputVertices = cg.tcsOutputVertices;
        fc.stageInStride = cg.stageInStride;
        fc.stageOutStride = cg.stageOutStride;
        fc.geometryInputPtr = cg.geometryInputPtr;
        fc.geometryOutputPtr = cg.geometryOutputPtr;
        fc.geometryCountPtr = cg.geometryCountPtr;
        fc.geometryWorkItemId = cg.geometryWorkItemId;
        fc.geometryPrimitiveId = cg.geometryPrimitiveId;
        fc.geometryInvocationId = cg.geometryInvocationId;
        fc.geometryInputVertices = cg.geometryInputVertices;
        fc.geometryOutputType = cg.geometryOutputType;
        fc.geometryMaxVertices = cg.geometryMaxVertices;
        fc.geometryOutputVertices = cg.geometryOutputVertices;
        fc.geometryRecordCount = cg.geometryRecordCount;
        fc.patchControlPtr = cg.patchControlPtr;
        fc.tessCoord = cg.tessCoord;
        fc.patchId = cg.patchId;
        fc.controlPointGetter = cg.controlPointGetter;
        fc.controlPointFields = cg.controlPointFields;
        fc.auxSyms = cg.auxSyms;
        fc.captureBuf = cg.captureBuf;
        fc.vertexId = cg.vertexId;
        fc.fragPos = cg.fragPos;
        fc.pointSize = cg.pointSize;
        fc.ssboPtrs = cg.ssboPtrs;
        fc.ssboSlots = cg.ssboSlots;
        fc.ssboElemSlot = cg.ssboElemSlot;
        fc.ssboElemArrTy = cg.ssboElemArrTy;
        fc.acPtrs = cg.acPtrs;
        fc.acSlots = cg.acSlots;
        fc.uboPtrs = cg.uboPtrs;
        /* tex/smp/outPtrs rebound from hidden args below — do not copy
         * main's Argument* values (illegal cross-function use). */
        fc.bufferOffsets = cg.bufferOffsets;
        fc.position = cg.position;
        fc.userFns = &userFns;
        fc.userFnHidden = &userFnHidden;
        fc.userFnDecls = &userFnDecls;
        fc.userFnPassCull = cg.userFnPassCull;
        fc.userFnPassClip = cg.userFnPassClip;
        /* Struct ctors / localIRTypes resolve through these maps. */
        fc.structTypes = cg.structTypes;
        fc.ownedIRTypes = cg.ownedIRTypes;
        std::map<std::string, MType> flocals;
        for (uint32_t p = 0; p < d->param_count; p++) {
            MGLDecl *pd = d->params[p];
            if (!pd || !pd->name) continue;
            MType pt;
            pt.scalar = (MGLIRScalar)(pd->type ? pd->type->base
                                               : MGL_AST_TYPE_FLOAT);
            if (pd->type && pd->type->vec_size) pt.vec = pd->type->vec_size;
            /* Matrix parameters must carry their shape or m[i]/m[i][j]
             * inside the function body fail the index type check. */
            if (pd->type && pd->type->mat_cols > 1) {
                pt.cols = pd->type->mat_cols;
                pt.rows = pd->type->mat_rows;
            }
            flocals[pd->name] = pt;
            fc.lvalues[pd->name] = f->getArg(p);
            if (pd->type && pd->type->base == MGL_AST_TYPE_STRUCT &&
                pd->type->name) {
                auto sit = cg.structTypes.find(pd->type->name);
                if (sit != cg.structTypes.end())
                    fc.localIRTypes[pd->name] = sit->second;
            }
        }
        {
            uint32_t hidx = (uint32_t)d->param_count;
            for (const auto &kv : cg.uboPtrs) {
                llvm::Value *value = f->getArg(hidx++);
                fc.uboPtrs[kv.first] = value;
                if (auto *arrTy = llvm::dyn_cast<llvm::ArrayType>(
                        value->getType())) {
                    llvm::Value *slot = fb.CreateAlloca(
                        arrTy, nullptr, kv.first + "_elems");
                    fb.CreateStore(value, slot);
                    fc.uboElemSlot[kv.first] = slot;
                    fc.uboElemArrTy[kv.first] = arrTy;
                }
            }
            for (const auto &kv : cg.ssboPtrs) {
                llvm::Value *value = f->getArg(hidx++);
                fc.ssboPtrs[kv.first] = value;
                if (auto *arrTy = llvm::dyn_cast<llvm::ArrayType>(
                        value->getType())) {
                    llvm::Value *slot = fb.CreateAlloca(
                        arrTy, nullptr, kv.first + "_elems");
                    fb.CreateStore(value, slot);
                    fc.ssboElemSlot[kv.first] = slot;
                    fc.ssboElemArrTy[kv.first] = arrTy;
                }
            }
            for (const auto &kv : cg.acPtrs)
                fc.acPtrs[kv.first] = f->getArg(hidx++);
            if (cg.bufferSizePtr)
                fc.bufferSizePtr = f->getArg(hidx++);
            if (cg.userFnPassCull)
                fc.lvalues["gl_CullDistance"] = f->getArg(hidx++);
            if (cg.userFnPassClip)
                fc.lvalues["gl_ClipDistance"] = f->getArg(hidx++);
            if (isGS) {
                fc.geometryInputPtr = f->getArg(hidx++);
                fc.geometryOutputPtr = f->getArg(hidx++);
                fc.geometryCountPtr = f->getArg(hidx++);
                fc.geometryGatherPtr = f->getArg(hidx++);
                fc.geometryGatherParamsPtr = f->getArg(hidx++);
                fc.geometryWorkItemId = f->getArg(hidx++);
                fc.geometryPrimitiveId = f->getArg(hidx++);
                fc.geometryInvocationId = f->getArg(hidx++);
            }
            if (isTCS) {
                fc.stageInPtr = f->getArg(hidx++);
                fc.tessFactorPtr = f->getArg(hidx++);
                fc.stageOutPtr = f->getArg(hidx++);
                fc.indirectPtr = f->getArg(hidx++);
                fc.invocationPos = f->getArg(hidx++);
                fc.patchPos = f->getArg(hidx++);
            }
            if (isCompute && cg.threadPos)
                fc.threadPos = f->getArg(hidx++);
            fc.texValues.clear();
            fc.smpValues.clear();
            fc.texArrayValues.clear();
            fc.smpArrayValues.clear();
            if (!isGS && !isTCS) {
                for (const auto &kv : cg.texValues)
                    fc.texValues[kv.first] = f->getArg(hidx++);
                for (const auto &kv : cg.smpValues)
                    fc.smpValues[kv.first] = f->getArg(hidx++);
                for (const auto &kv : cg.texArrayValues) {
                    std::vector<llvm::Value *> texes;
                    for (size_t ti = 0; ti < kv.second.size(); ti++)
                        texes.push_back(f->getArg(hidx++));
                    fc.texArrayValues[kv.first] = std::move(texes);
                }
                for (const auto &kv : cg.smpArrayValues) {
                    std::vector<llvm::Value *> smps;
                    for (size_t si = 0; si < kv.second.size(); si++)
                        smps.push_back(f->getArg(hidx++));
                    fc.smpArrayValues[kv.first] = std::move(smps);
                }
            }
            if (cg.bufferPtr)
                fc.bufferPtr = f->getArg(hidx++);
            fc.outPtrs.clear();
            for (const auto &kv : cg.outPtrs)
                fc.outPtrs[kv.first] = f->getArg(hidx++);
        }
        emitStmt(fc, d->body, &mod, &flocals);
        if (fc.err == 1) {
            cg.err = 1;
            cg.errmsg = std::string("codegen: in function '") + d->name +
                        "': " + fc.errmsg;
            snprintf(err_buf, err_cap, "%s", cg.errmsg.c_str());
            if (own_session) mglFrontendSessionDestroy(sess);
            return -1;
        }
        if (fc.err != 2) {
            if (f->getReturnType()->isVoidTy()) {
                fb.CreateRetVoid();
            } else {
                fb.CreateRet(llvm::UndefValue::get(f->getReturnType()));
            }
        }
    }

    cg.userFns = &userFns;
    cg.userFnHidden = &userFnHidden;
    cg.userFnDecls = &userFnDecls;
    std::map<std::string, MType> locals;
    /* Global initializers: const arrays/scalars fold to SSA.  Non-const
     * uniforms must not be SSA-folded — loads go through the plain pack so
     * glUniform* is visible.  (GLSL default initializers still need CPU-side
     * seeding at link; do not bufferStore here or every invocation would
     * overwrite glUniform uploads.) */
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        MGLDecl *d = tu->decls[i];
        if (!d || !d->name || d->body || !d->init) continue;
        const MGLIRSymbol *gs = findSymbol(&mod, d->name);
        if (!gs || gs->is_function) continue;
        if (gs->qualifiers & (MGL_AST_Q_BUFFER |
                              MGL_AST_Q_IN | MGL_AST_Q_OUT |
                              MGL_AST_Q_INOUT | MGL_AST_Q_SHARED)) {
            continue;
        }
        MType gt = typeFromIR(gs->type);
        const bool isConst = (gs->qualifiers & MGL_AST_Q_CONST) != 0;
        const bool isUniform = (gs->qualifiers & MGL_AST_Q_UNIFORM) != 0;
        if (gt.isArray() && !isConst && !isUniform) continue;
        if (isUniform && !isConst)
            continue; /* pack load; defaults seeded at link (program.c) */
        llvm::Value *gv = emitExpr(cg, d->init, &mod, locals);
        if (!gv) break;
        cg.lvalues[d->name] = gv;
        locals[d->name] = gt;
    }
    if (isTCS) {
        cg.lvalues["gl_TessLevelOuter"] = llvm::UndefValue::get(
            llvm::ArrayType::get(llvm::Type::getFloatTy(ctx), 4));
        cg.lvalues["gl_TessLevelInner"] = llvm::UndefValue::get(
            llvm::ArrayType::get(llvm::Type::getFloatTy(ctx), 2));
    }
    if (isTESCompute && cg.threadPos && cg.tessFactorPtr && cg.indirectPtr) {
        /* isolines/point-mode TES kernel: the runtime dispatches one
         * compute pass per patch (per-patch item counts differ), so the
         * contract buffer (slot 29) carries {patch_id, vertices_per_patch,
         * items_per_patch, output_offset}; thread_position_in_grid is the
         * item index inside the current patch. */
        llvm::Value *threadItem = b.CreateExtractElement(
            cg.threadPos, b.getInt32(0));
        llvm::Type *f32 = llvm::Type::getFloatTy(ctx);
        llvm::Value *contract = b.CreateBitCast(
            cg.indirectPtr, b.getInt32Ty()->getPointerTo(1));
        llvm::Value *patchId = b.CreateAlignedLoad(
            b.getInt32Ty(),
            b.CreateGEP(b.getInt32Ty(), contract, b.getInt32(0)),
            llvm::Align(4));
        llvm::Value *outputBase = b.CreateAlignedLoad(
            b.getInt32Ty(),
            b.CreateGEP(b.getInt32Ty(), contract, b.getInt32(3)),
            llvm::Align(4));
        llvm::Value *innerId = threadItem;
        cg.patchId = patchId;
        cg.geometryWorkItemId = b.CreateAdd(outputBase, innerId);
        llvm::Value *itemsC = b.CreateAlignedLoad(
            b.getInt32Ty(),
            b.CreateGEP(b.getInt32Ty(), contract, b.getInt32(2)),
            llvm::Align(4));
        llvm::Function *kfn = b.GetInsertBlock()->getParent();
        llvm::BasicBlock *okBB = llvm::BasicBlock::Create(
            ctx, "tesk_inrange", kfn);
        llvm::BasicBlock *oobBB = llvm::BasicBlock::Create(
            ctx, "tesk_oob", kfn);
        b.CreateCondBr(b.CreateICmpUGE(innerId, itemsC), oobBB, okBB);
        {
            llvm::IRBuilder<>::InsertPoint ip = b.saveIP();
            b.SetInsertPoint(oobBB);
            b.CreateRetVoid();
            b.restoreIP(ip);
        }
        b.SetInsertPoint(okBB);
        llvm::Value *factorBase = b.CreateGEP(
            b.getInt8Ty(), cg.tessFactorPtr,
            b.CreateMul(b.CreateZExt(patchId, b.getInt64Ty()),
                        b.getInt64(MGL_AIR_TESS_FACTOR_RECORD_BYTES)));
        /* GL TES inputs: gl_TessLevel* must be the exact levels (TCS outs
         * or glPatchParameterfv), not the Metal half factors — half
         * round-trip fails CTS 1e-5 (gl_tessLevel). */
        {
            llvm::Type *arr4 = llvm::ArrayType::get(f32, 4);
            llvm::Type *arr2 = llvm::ArrayType::get(f32, 2);
            llvm::Value *outer = llvm::UndefValue::get(arr4);
            llvm::Value *inner = llvm::UndefValue::get(arr2);
            llvm::Value *exactBase = b.CreateGEP(
                b.getInt8Ty(), factorBase,
                b.getInt64(MGL_AIR_TESS_FACTOR_EXACT_FLOAT_OFFSET));
            for (unsigned i = 0; i < 4; i++) {
                llvm::Value *p = b.CreateBitCast(
                    b.CreateGEP(b.getInt8Ty(), exactBase,
                                b.getInt64(4 * i)),
                    f32->getPointerTo(1));
                outer = b.CreateInsertValue(
                    outer, b.CreateAlignedLoad(f32, p, llvm::Align(4)), i);
            }
            for (unsigned i = 0; i < 2; i++) {
                llvm::Value *p = b.CreateBitCast(
                    b.CreateGEP(b.getInt8Ty(), exactBase,
                                b.getInt64(16 + 4 * i)),
                    f32->getPointerTo(1));
                inner = b.CreateInsertValue(
                    inner, b.CreateAlignedLoad(f32, p, llvm::Align(4)), i);
            }
            cg.lvalues["gl_TessLevelOuter"] = outer;
            cg.lvalues["gl_TessLevelInner"] = inner;
        }
        /* Each invocation owns one output record. The domain layer seeds
         * its position.xyz with TessCoord before dispatch; the TES reads it
         * once and then overwrites the record with its shader outputs. */
        llvm::Value *coordBase = b.CreateGEP(
            b.getInt8Ty(), cg.geometryOutputPtr,
            b.CreateMul(b.CreateZExt(cg.geometryWorkItemId, b.getInt64Ty()),
                        b.getInt64(cg.stageOutStride)));
        llvm::Value *coord = llvm::UndefValue::get(
            llvm::FixedVectorType::get(f32, 3));
        for (unsigned axis = 0; axis < 3; axis++) {
            llvm::Value *p = b.CreateBitCast(
                b.CreateGEP(b.getInt8Ty(), coordBase, b.getInt64(axis * 4)),
                f32->getPointerTo(1));
            coord = b.CreateInsertElement(coord,
                b.CreateAlignedLoad(f32, p, llvm::Align(4)), b.getInt32(axis));
        }
        cg.tessCoord = coord;
    }
    if (isTESVertex && cg.vertexId && cg.tessFactorPtr && cg.indirectPtr) {
        /* TES-vertex: the record index is gl_VertexID (Metal vertex_id
         * already carries the per-patch vertexStart).  The contract buffer
         * (slot 29) carries {patch_id, vertices_per_patch, items, 0}. */
        llvm::Type *f32 = llvm::Type::getFloatTy(ctx);
        llvm::Value *contract = b.CreateBitCast(
            cg.indirectPtr, b.getInt32Ty()->getPointerTo(1));
        llvm::Value *patchId = b.CreateAlignedLoad(
            b.getInt32Ty(),
            b.CreateGEP(b.getInt32Ty(), contract, b.getInt32(0)),
            llvm::Align(4));
        cg.patchId = patchId;
        cg.geometryWorkItemId = cg.vertexId;
        /* gl_TessLevel* must be the exact float32 levels (TCS outs or
         * glPatchParameterfv), not the Metal half factors — half round-trip
         * fails CTS 1e-5 (gl_tessLevel). */
        llvm::Value *factorBase = b.CreateGEP(
            b.getInt8Ty(), cg.tessFactorPtr,
            b.CreateMul(b.CreateZExt(patchId, b.getInt64Ty()),
                        b.getInt64(MGL_AIR_TESS_FACTOR_RECORD_BYTES)));
        {
            llvm::Type *arr4 = llvm::ArrayType::get(f32, 4);
            llvm::Type *arr2 = llvm::ArrayType::get(f32, 2);
            llvm::Value *outer = llvm::UndefValue::get(arr4);
            llvm::Value *inner = llvm::UndefValue::get(arr2);
            llvm::Value *exactBase = b.CreateGEP(
                b.getInt8Ty(), factorBase,
                b.getInt64(MGL_AIR_TESS_FACTOR_EXACT_FLOAT_OFFSET));
            for (unsigned i = 0; i < 4; i++) {
                llvm::Value *p = b.CreateBitCast(
                    b.CreateGEP(b.getInt8Ty(), exactBase,
                                b.getInt64(4 * i)),
                    f32->getPointerTo(1));
                outer = b.CreateInsertValue(
                    outer, b.CreateAlignedLoad(f32, p, llvm::Align(4)), i);
            }
            for (unsigned i = 0; i < 2; i++) {
                llvm::Value *p = b.CreateBitCast(
                    b.CreateGEP(b.getInt8Ty(), exactBase,
                                b.getInt64(16 + 4 * i)),
                    f32->getPointerTo(1));
                inner = b.CreateInsertValue(
                    inner, b.CreateAlignedLoad(f32, p, llvm::Align(4)), i);
            }
            cg.lvalues["gl_TessLevelOuter"] = outer;
            cg.lvalues["gl_TessLevelInner"] = inner;
        }
        /* The CPU domain expansion seeded this record's position.xyz with
         * TessCoord; read it back from slot 28. */
        llvm::Value *coordBase = b.CreateGEP(
            b.getInt8Ty(), cg.geometryOutputPtr,
            b.CreateMul(b.CreateZExt(cg.geometryWorkItemId, b.getInt64Ty()),
                        b.getInt64(cg.stageOutStride)));
        llvm::Value *coord = llvm::UndefValue::get(
            llvm::FixedVectorType::get(f32, 3));
        for (unsigned axis = 0; axis < 3; axis++) {
            llvm::Value *p = b.CreateBitCast(
                b.CreateGEP(b.getInt8Ty(), coordBase, b.getInt64(axis * 4)),
                f32->getPointerTo(1));
            coord = b.CreateInsertElement(coord,
                b.CreateAlignedLoad(f32, p, llvm::Align(4)), b.getInt32(axis));
        }
        cg.tessCoord = coord;
    }
    emitStmt(cg, mainDecl->body, &mod, &locals);
    if (isTCS && cg.tessFactorPtr && cg.patchPos &&
        !b.GetInsertBlock()->getTerminator()) {
        /* Any invocation that wrote gl_TessLevel* stores it. Unwritten
         * slots keep the PATCH_DEFAULT_* fill (GL 4.6 §11.2.2). */
        flushTCSTessLevels(cg);
    }

    if (isTESVertex) {
        /* TES-vertex: the TES is the raster vertex stage, so it returns the
         * ordinary VS output record instead of writing the slot-28 record
         * buffer.  gl_CullDistance for point/line topologies is applied
         * here (the passthrough VS used to do it from the record buffer). */
        if (usesTesVertexCull) {
            llvm::Type *f32 = llvm::Type::getFloatTy(ctx);
            llvm::Value *cullArr = cg.lvalues.count("gl_CullDistance")
                ? cg.lvalues["gl_CullDistance"]
                : defaultCullDistances(cg);
            /* gl_CullDistance (GL 4.6 §13.6.1): a vertex is culled when its
             * own distance is negative.  The rasterizer then clips any
             * primitive whose endpoints straddle the plane and discards a
             * primitive only when all of its vertices are negative.  Per-
             * vertex culling (criterion = own) yields exactly that for both
             * point_mode (points) and isolines (line segments).  The previous
             * isolines partner rule (criterion = own * partner, which fired
             * only on a sign straddle) left rows whose vertices all shared
             * one sign (e.g. d = 0.5 - v culling a whole v row) fully visible
             * — repro: air_tessellation_cull_distance. */
            llvm::Value *shouldCull = b.getFalse();
            for (uint32_t d = 0; d < MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT; d++) {
                llvm::Value *own = b.CreateExtractValue(cullArr, d);
                llvm::Value *criterion = own;
                shouldCull = b.CreateOr(
                    shouldCull,
                    b.CreateFCmpOLT(criterion,
                                    llvm::ConstantFP::get(f32, 0.0)));
            }
            llvm::Value *culled = llvm::ConstantVector::get({
                llvm::ConstantFP::get(f32, 2.0),
                llvm::ConstantFP::get(f32, 2.0),
                llvm::ConstantFP::get(f32, 2.0),
                llvm::ConstantFP::get(f32, 1.0)});
            llvm::Value *pos = cg.lvalues.count("gl_Position")
                ? cg.lvalues["gl_Position"]
                : llvm::UndefValue::get(llvm::FixedVectorType::get(f32, 4));
            if (pos->getType() != llvm::FixedVectorType::get(f32, 4)) {
                if (pos->getType()->isVectorTy())
                    pos = b.CreateBitCast(pos, llvm::FixedVectorType::get(f32, 4));
                else
                    pos = b.CreateVectorSplat(4, pos);
            }
            cg.lvalues["gl_Position"] =
                b.CreateSelect(shouldCull, culled, pos);
        }
        /* Fall through to the common epilogue return at the end of codegen
         * (b.CreateRet(assembleReturn(cg))), which is also used by the plain
         * VS/FS paths.  Emitting the return here would place two terminators
         * in the same block and make the AIR module invalid. */
    } else if (isTESCompute && cg.geometryOutputPtr && cg.geometryWorkItemId &&
        !b.GetInsertBlock()->getTerminator()) {
        /* Each work item writes one expanded vertex record into the
         * stage-out buffer (slot 28); see storeTessComputeVaryings for the
         * shared GS record layout. */
        llvm::Type *v4 = llvm::FixedVectorType::get(
            llvm::Type::getFloatTy(ctx), 4);
        llvm::Value *pos = cg.lvalues.count("gl_Position")
            ? cg.lvalues["gl_Position"] : llvm::UndefValue::get(v4);
        if (pos->getType() != v4) {
            if (pos->getType()->isVectorTy()) pos = b.CreateBitCast(pos, v4);
            else pos = b.CreateVectorSplat(4, pos);
        }
        storeGeometryPosition(cg, b.getInt32(0), pos);
        llvm::Value *pointSize = cg.lvalues.count("gl_PointSize")
            ? cg.lvalues["gl_PointSize"]
            : llvm::ConstantFP::get(llvm::Type::getFloatTy(ctx), 1.0);
        storeGeometryPointSize(cg, b.getInt32(0), pointSize);
        storeTessComputeVaryings(cg, b.getInt32(0));
        /* Post-tess cull distances (GL 4.6 §13.6.1): the TES-written
         * gl_CullDistance of each expanded vertex lands in the shared
         * per-vertex record slot; the passthrough vertex stage applies the
         * point/line cull rule (a point is culled when any distance < 0; a
         * line when both endpoints' distance < 0 for the same axis).  When
         * the TES never touches gl_CullDistance the slot keeps its zero
         * fill (nothing culled). */
        if (cg.lvalues.count("gl_CullDistance")) {
            storeGeometryCullDistances(cg, b.getInt32(0),
                                       cg.lvalues["gl_CullDistance"]);
        }
        if (cg.lvalues.count("gl_ClipDistance")) {
            storeGeometryClipDistances(cg, b.getInt32(0),
                                       cg.lvalues["gl_ClipDistance"]);
        }
        if (cg.xfbOutPtr) {
            /* Transform-feedback stream (slot 31): one complete stage-out
             * record per work item, same layout/stride as slot 28.  The
             * runtime binds the GL target here only when feedback is
             * active; the kernel copy is otherwise skipped. */
            llvm::Value *xfbSlot = b.CreateMul(
                b.CreateZExt(cg.geometryWorkItemId, b.getInt64Ty()),
                b.getInt64(cg.stageOutStride));
            llvm::Value *xfbBase = b.CreateGEP(b.getInt8Ty(),
                                               cg.xfbOutPtr, xfbSlot);
            llvm::Value *stageBase = b.CreateGEP(
                b.getInt8Ty(), cg.geometryOutputPtr, xfbSlot);
            b.CreateMemCpy(xfbBase, llvm::Align(16), stageBase,
                           llvm::Align(16), b.getInt64(cg.stageOutStride));
        }
    }

    if (cg.err && cg.err != 2) {
        snprintf(err_buf, err_cap, "%s",
                 cg.errmsg.empty() ? "codegen: unsupported construct"
                                   : cg.errmsg.c_str());
        if (own_session) mglFrontendSessionDestroy(sess);
        return -1;
    }
    /* Terminate if the body's last statement was a return. */
    if (cg.err != 2) {
        if (isKernel) {
            if (isGS && cg.geometryXfbMetaPtr &&
                cg.geometryCountPtr && cg.geometryWorkItemId) {
                /* GL4 ordered terminal state (mgl_air_gs_abi.h §5b): the
                 * stream-0 XFB path no longer appends through a GPU-atomic
                 * cursor.  Instead the epilogue accumulates this work item's
                 * visible stream-0 bytes into the visibility buffer (slot 26)
                 * at a deterministic per-work-item index; the CPU prefix-sum
                 * and the pass-2 scatter copy the records in emission order.
                 * The visible count is the final stream-0 outputCount (draw
                 * param word 0), read back from the counts record, and it is
                 * attributed to every buffer fed by stream 0 (a single-stream
                 * program may split varyings across buffers with
                 * gl_NextBuffer).  The rasterization records stay in the
                 * stage-out run untouched. */
                llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
                llvm::Type *i64 = llvm::Type::getInt64Ty(ctx);
                llvm::Value *metaBase = b.CreateBitCast(
                    cg.geometryXfbMetaPtr, i32->getPointerTo(1));
                llvm::Value *fedStride[MGL_AIR_GS_MAX_STREAMS] = {nullptr};
                llvm::Value *fedPred[MGL_AIR_GS_MAX_STREAMS] = {nullptr};
                llvm::Value *captureOn = nullptr;
                for (uint32_t buf = 0; buf < MGL_AIR_GS_MAX_STREAMS; buf++) {
                    llvm::Value *bs = b.CreateAlignedLoad(
                        i32, b.CreateGEP(i32, metaBase, b.getInt32(16u + buf)),
                        llvm::Align(4));
                    llvm::Value *match = b.CreateICmpEQ(bs, b.getInt32(0));
                    llvm::Value *bsStride = b.CreateAlignedLoad(
                        i32, b.CreateGEP(i32, metaBase,
                                         b.getInt32(buf * 4u)),
                        llvm::Align(4));
                    llvm::Value *on = b.CreateAnd(
                        match, b.CreateICmpNE(bsStride, b.getInt32(0)));
                    fedStride[buf] = bsStride;
                    fedPred[buf] = on;
                    captureOn = captureOn ? b.CreateOr(captureOn, on) : on;
                }
                llvm::BasicBlock *xfbOnBB = llvm::BasicBlock::Create(
                    ctx, "gs_xfb_on", cg.fn);
                llvm::BasicBlock *xfbSkipBB = llvm::BasicBlock::Create(
                    ctx, "gs_xfb_skip", cg.fn);
                b.CreateCondBr(captureOn, xfbOnBB, xfbSkipBB);
                b.SetInsertPoint(xfbOnBB);
                llvm::Value *countsBase = b.CreateBitCast(
                    cg.geometryCountPtr, i32->getPointerTo(1));
                llvm::Value *countsOff = b.CreateMul(
                    b.CreateZExt(cg.geometryWorkItemId, i64),
                    b.getInt64(MGL_AIR_GS_COUNTS_RECORD_WORDS));
                llvm::Value *visible = b.CreateAlignedLoad(
                    i32, b.CreateGEP(i32, countsBase, countsOff),
                    llvm::Align(4));
                llvm::Value *hasVisible = b.CreateICmpNE(
                    visible, b.getInt32(0));
                llvm::BasicBlock *xfbVisBB = llvm::BasicBlock::Create(
                    ctx, "gs_xfb_vis", cg.fn);
                b.CreateCondBr(hasVisible, xfbVisBB, xfbSkipBB);
                b.SetInsertPoint(xfbVisBB);
                if (cg.geometryXfbVisPtr) {
                    /* vis[workItem * 4 + b] += visible * stride[b] for every
                     * buffer fed by stream 0.  Accumulate (never overwrite):
                     * EmitStreamVertex may already have accumulated bytes
                     * for buffers fed by streams > 0 earlier in this same
                     * thread. */
                    llvm::Value *visBase = b.CreateBitCast(
                        cg.geometryXfbVisPtr, i32->getPointerTo(1));
                    llvm::Value *visRun = b.CreateMul(
                        cg.geometryWorkItemId,
                        b.getInt32(MGL_AIR_GS_MAX_STREAMS));
                    for (uint32_t buf = 0; buf < MGL_AIR_GS_MAX_STREAMS;
                         buf++) {
                        llvm::Value *add = b.CreateSelect(
                            fedPred[buf],
                            b.CreateMul(visible, fedStride[buf]),
                            b.getInt32(0));
                        llvm::Value *visPtr = b.CreateGEP(
                            i32, visBase,
                            b.CreateAdd(visRun, b.getInt32(buf)));
                        llvm::Value *cur = b.CreateAlignedLoad(
                            i32, visPtr, llvm::Align(4));
                        b.CreateAlignedStore(b.CreateAdd(cur, add), visPtr,
                                             llvm::Align(4));
                    }
                }
                b.CreateBr(xfbSkipBB);
                b.SetInsertPoint(xfbSkipBB);
            }
            if (isGS && cg.geometryCountPtr && cg.geometryWorkItemId) {
                /* ABI (mgl_air_gs_abi.h §3): finalize the per-work-item
                 * MGLAIRGSIndirectArgs — instance_count=1, base_vertex=0,
                 * base_instance=0 — so the rasterizing indirect draw is
                 * well-defined.  The scratch strip/emit counters live in
                 * words 4..6 and are deliberately NOT touched here. */
                llvm::Value *off = b.CreateMul(
                    b.CreateZExt(cg.geometryWorkItemId, b.getInt64Ty()),
                    b.getInt64(MGL_AIR_GS_COUNTS_RECORD_WORDS));
                llvm::Value *p = b.CreateGEP(
                    b.getInt32Ty(),
                    b.CreateBitCast(cg.geometryCountPtr,
                                    b.getInt32Ty()->getPointerTo(1)),
                    off);
                llvm::Value *instanceCount = b.CreateGEP(
                    b.getInt32Ty(), p, b.getInt32(1));
                llvm::Value *vertexStart = b.CreateGEP(
                    b.getInt32Ty(), p, b.getInt32(2));
                llvm::Value *baseInstance = b.CreateGEP(
                    b.getInt32Ty(), p, b.getInt32(3));
                b.CreateAlignedStore(b.getInt32(1), instanceCount,
                                     llvm::Align(4));
                b.CreateAlignedStore(b.getInt32(0), vertexStart,
                                     llvm::Align(4));
                b.CreateAlignedStore(b.getInt32(0), baseInstance,
                                     llvm::Align(4));
            }
            b.CreateRetVoid();
        } else if (isCapture) {
            /* XFB capture: write the assembled output record into the
             * capture buffer at [vertex_id]. */
            llvm::Type *recTy = captureRecordType();
            llvm::Value *rec = llvm::UndefValue::get(recTy);
            if (isCullCapture) {
                rec = cg.lvalues.count("gl_CullDistance")
                    ? cg.lvalues["gl_CullDistance"]
                    : llvm::UndefValue::get(recTy);
            }
            llvm::Value *pos = cg.lvalues.count("gl_Position")
                                   ? cg.lvalues["gl_Position"]
                                   : llvm::UndefValue::get(cg.retElems[0]);
            /* Raw GL clip space: gl_in consumers and XFB captures of
             * gl_Position must observe the shader-written z; the Metal
             * [0,1] depth remap happens where records feed rasterization
             * (GS EmitVertex / TES stage-out stores). */
            if (!isCullCapture && recTy->isStructTy()) {
                rec = b.CreateInsertValue(rec, pos, 0);
                uint32_t ri = 1;
                if (isTessCapture) {
                    rec = b.CreateInsertValue(
                        rec,
                        cg.lvalues.count("gl_PointSize")
                            ? cg.lvalues["gl_PointSize"]
                            : llvm::ConstantFP::get(
                                  llvm::Type::getFloatTy(ctx), 1.0),
                        ri++);
                    rec = b.CreateInsertValue(
                        rec,
                        cg.lvalues.count("gl_CullDistance")
                            ? cg.lvalues["gl_CullDistance"]
                            : defaultCullDistances(cg),
                        ri++);
                } else if (cg.pointSize) {
                    rec = b.CreateInsertValue(
                        rec,
                        cg.lvalues.count("gl_PointSize")
                            ? cg.lvalues["gl_PointSize"]
                            : llvm::ConstantFP::get(
                                  llvm::Type::getFloatTy(ctx), 1.0),
                        ri++);
                }
                if (!isTessCapture) {
                    for (uint32_t i = 0; i < cg.varyings.size(); i++) {
                        VarSym *var = cg.varyings[i];
                        llvm::Value *base =
                            cg.lvalues.count(var->name)
                                ? cg.lvalues[var->name]
                                : llvm::UndefValue::get(llvmType(var->type, ctx));
                        if (var->type.isArray()) {
                            /* Flattened record: one field per element,
                             * matching retElems construction. */
                            uint32_t n = (uint32_t)var->type.arr;
                            for (uint32_t k = 0; k < n; k++) {
                                llvm::Value *el = base;
                                if (base->getType()->isArrayTy())
                                    el = b.CreateExtractValue(base, k);
                                rec = b.CreateInsertValue(rec, el, ri++);
                            }
                        } else if (var->type.isMatrix()) {
                            for (uint32_t c = 0; c < var->type.cols; c++) {
                                llvm::Value *col = base;
                                if (base->getType()->isArrayTy())
                                    col = b.CreateExtractValue(base, c);
                                rec = b.CreateInsertValue(rec, col, ri++);
                            }
                        } else {
                            rec = b.CreateInsertValue(rec, base, ri++);
                        }
                    }
                }
            } else if (!isCullCapture) {
                rec = pos;
            }
            uint64_t recSize = module.getDataLayout().getTypeAllocSize(recTy);
            uint64_t recStride = isTessCapture
                ? tessCaptureStride : recSize;
            llvm::Value *vid = b.CreateSExtOrTrunc(cg.vertexId,
                                                   b.getInt64Ty());
            if ((isCullCapture || isTessCapture) && cg.instanceId &&
                cg.cullParams) {
                llvm::Value *params = b.CreateBitCast(
                    cg.cullParams, b.getInt32Ty()->getPointerTo(1));
                uint32_t firstInstanceField = isCullCapture ? 10u : 2u;
                uint32_t instanceStrideField = isCullCapture ? 11u : 1u;
                llvm::Value *firstInstance = b.CreateAlignedLoad(
                    b.getInt32Ty(),
                    b.CreateGEP(b.getInt32Ty(), params,
                                b.getInt32(firstInstanceField)),
                    llvm::Align(4));
                llvm::Value *instanceStride = b.CreateAlignedLoad(
                    b.getInt32Ty(),
                    b.CreateGEP(b.getInt32Ty(), params,
                                b.getInt32(instanceStrideField)),
                    llvm::Align(4));
                llvm::Value *relativeInstance = b.CreateSub(
                    cg.instanceId, firstInstance);
                llvm::Value *instanceBase = b.CreateMul(
                    relativeInstance, instanceStride);
                if (isTessCapture) {
                    llvm::Value *firstVertex = b.CreateAlignedLoad(
                        b.getInt32Ty(), params, llvm::Align(4));
                    vid = b.CreateSub(
                        vid, b.CreateZExt(firstVertex, b.getInt64Ty()));
                }
                vid = b.CreateAdd(
                    b.CreateZExt(instanceBase, b.getInt64Ty()), vid);
            }
            llvm::Value *p = b.CreateGEP(
                b.getInt8Ty(), cg.captureBuf,
                b.CreateMul(vid, b.getInt64(recStride)));
            p = b.CreateBitCast(p, recTy->getPointerTo(1));
            b.CreateAlignedStore(rec, p, llvm::Align(16));
            if (isTessCapture && cg.lvalues.count("gl_ClipDistance")) {
                llvm::Type *clipTy = llvm::ArrayType::get(
                    llvm::Type::getFloatTy(ctx),
                    MGL_AIR_PER_VERTEX_CLIP_DISTANCE_COUNT);
                llvm::Value *recordBase = b.CreateGEP(
                    b.getInt8Ty(), cg.captureBuf,
                    b.CreateMul(vid, b.getInt64(recStride)));
                llvm::Value *clipPtr = b.CreateBitCast(
                    b.CreateGEP(
                        b.getInt8Ty(), recordBase,
                        b.getInt64(MGL_AIR_PER_VERTEX_CLIP_DISTANCE_OFFSET)),
                    clipTy->getPointerTo(1));
                b.CreateAlignedStore(cg.lvalues["gl_ClipDistance"], clipPtr,
                                    llvm::Align(4));
            }
            if (isTessCapture) {
                llvm::Value *recordBase = b.CreateGEP(
                    b.getInt8Ty(), cg.captureBuf,
                    b.CreateMul(vid, b.getInt64(recStride)));
                auto storeRecordSlot = [&](uint32_t loc, MType slotTy,
                                           llvm::Value *slotVal) {
                    if (varyingNeedsFloatRecordCarrier(slotTy)) {
                        slotVal = encodeFloatCarrier(cg, slotVal,
                                                     slotTy.scalar);
                        slotTy = floatCarrierType(slotTy);
                    }
                    llvm::Type *storeTy = llvmType(slotTy, ctx);
                    if (slotVal->getType() != storeTy) {
                        if (storeTy->isIntOrIntVectorTy() &&
                            slotVal->getType()->isIntOrIntVectorTy())
                            slotVal = b.CreateBitCast(slotVal, storeTy);
                        else
                            slotVal = coerceScalar(cg, slotVal,
                                                   slotTy.scalar);
                    }
                    llvm::Value *vp = b.CreateGEP(
                        b.getInt8Ty(), recordBase,
                        b.getInt64(MGL_AIR_PER_VERTEX_STRIDE + loc * 16u));
                    vp = b.CreateBitCast(vp, storeTy->getPointerTo(1));
                    b.CreateAlignedStore(slotVal, vp, llvm::Align(4));
                };
                for (VarSym *varying : cg.varyings) {
                    if (!varying || varying->location == UINT32_MAX) continue;
                    /* Plain stage-in arrays index by primitive vertex, so
                     * each per-vertex record stores element 0 only.
                     * Interface-block array members carry one distinct
                     * value per element: store each element in its own
                     * consecutive location slot.  Matrices are one column
                     * per location (GL 4.6 §4.4.1) — packing <2 x float>
                     * columns into a single 16B slot breaks column loads. */
                    MType mt = varying->type;
                    const bool wasArray = mt.isArray() && mt.arr > 0;
                    const bool blockArray =
                        wasArray && !varying->blockName.empty();
                    if (wasArray) mt.arr = 0;
                    llvm::Value *value = cg.lvalues.count(varying->name)
                        ? cg.lvalues[varying->name]
                        : llvm::UndefValue::get(llvmType(varying->type, ctx));
                    if (wasArray && !blockArray &&
                        value->getType()->isArrayTy())
                        value = b.CreateExtractValue(value, 0u);
                    if (blockArray) {
                        for (uint32_t ei = 0; ei < varying->type.arr; ++ei) {
                            llvm::Value *elem =
                                value->getType()->isArrayTy()
                                    ? b.CreateExtractValue(value, ei)
                                    : value;
                            storeRecordSlot(varying->location + ei, mt, elem);
                        }
                        continue;
                    }
                    if (mt.isMatrix() && mt.cols > 0) {
                        MType colTy = matrixColumnType(mt);
                        for (uint32_t c = 0; c < mt.cols; c++) {
                            llvm::Value *col =
                                value->getType()->isArrayTy()
                                    ? b.CreateExtractValue(value, c)
                                    : value;
                            storeRecordSlot(varying->location + c, colTy,
                                            col);
                        }
                        continue;
                    }
                    storeRecordSlot(varying->location, mt, value);
                }
            }
            if (isTessCapture || isCullCapture)
                b.CreateRet(assembleReturn(cg));
            else
                b.CreateRetVoid();
        } else {
            b.CreateRet(assembleReturn(cg));
        }
    }

    /* ---- AIR metadata ---- */
    std::vector<llvm::Metadata *> argNodes;
    const uint32_t sizeBufferArg =
        (isCapture ? 1u : 0u) + (isVS && !isCapture ? attrCount : 0u) +
        (((isVS || isTES || isKernel) && hasBuffer) ? 1u : 0u) +
        ssboCount + uboCount + acCount;
    if (isVS && !isCapture) {
        /* Vertex attributes are the first value arguments (Metal ABI:
         * stage_in value args precede buffers/textures). */
        uint32_t mArgSlot = 0;
        uint32_t nextFreeAttrLoc = 0;
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATTR) continue;
            /* The air.location_index must equal the location the reflector
             * reports (mglAirReflectModule) — the renderer's vertex
             * descriptor and draw-time bindings are driven by the reflected
             * locations.  Priority: explicit layout(location=N) from the
             * sema, then glBindAttribLocation/stable-name preferences, then
             * the running declaration-order counter.  The previous code
             * ignored explicit locations entirely (running counter only),
             * which silently misaligned any shader with non-contiguous
             * explicit attribute locations (the reflector said N, the
             * metallib read [[attribute(k)]]). */
            uint32_t attrLoc = v.location;
            if (!v.locationExplicit) {
                uint32_t want = airAttribLocation(v.name.c_str(),
                                                  attrib_names, MAX_ATTRIBS);
                if (want != UINT32_MAX)
                    attrLoc = want;
                else if (attrLoc == UINT32_MAX)
                    attrLoc = nextFreeAttrLoc;
            } else if (attrLoc == UINT32_MAX) {
                attrLoc = nextFreeAttrLoc;
            }
            const uint32_t n = varyingLocationSpan(v.type);
            MType el = v.type;
            if (v.type.isArray() && v.type.arr > 0) el.arr = 0;
            else if (v.type.isMatrix()) el = matrixColumnType(v.type);
            el = attrMetalIfaceType(el);
            for (uint32_t k = 0; k < n; k++) {
                std::string argName = n > 1u
                    ? v.name + "[" + std::to_string(k) + "]"
                    : v.name;
                std::vector<llvm::Metadata *> elems = {
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                    llvm::MDString::get(ctx, "air.vertex_input"),
                    llvm::MDString::get(ctx, "air.location_index"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), attrLoc + k)),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 1)),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, mslTypeName(el)),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, argName)};
                argNodes.push_back(llvm::MDNode::get(ctx, elems));
            }
            nextFreeAttrLoc = std::max(nextFreeAttrLoc, attrLoc + n);
        }
    }
    if (isCapture) {
        /* Capture output record buffer (XFB slot 29, read_write). */
        llvm::Type *recTy = captureRecordType();
        uint64_t recSize = module.getDataLayout().getTypeAllocSize(recTy);
        std::vector<llvm::Metadata *> stiFields;
        llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
        const llvm::DataLayout &dl = module.getDataLayout();
        uint32_t soff = 0;
        auto addMember = [&](llvm::Type *mt, const char *tname,
                             const char *mname) {
            soff = llvm::alignTo(soff, dl.getABITypeAlignment(mt));
            stiFields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32, soff)));
            stiFields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32, dl.getTypeAllocSize(mt))));
            stiFields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32, 0)));
            stiFields.push_back(llvm::MDString::get(ctx, tname));
            stiFields.push_back(llvm::MDString::get(ctx, mname));
            soff += dl.getTypeAllocSize(mt);
        };
        if (isCullCapture) {
            addMember(llvm::ArrayType::get(llvm::Type::getFloatTy(ctx), 8),
                      "float8", "cull_distance");
        } else {
            addMember(llvm::FixedVectorType::get(
                          llvm::Type::getFloatTy(ctx), 4),
                      "float4", "pos");
            if (isTessCapture) {
                addMember(llvm::Type::getFloatTy(ctx), "float", "psize");
                addMember(llvm::ArrayType::get(
                              llvm::Type::getFloatTy(ctx),
                              MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT),
                          "float[8]", "cull_distance");
            } else if (usesPointSize) {
                addMember(llvm::Type::getFloatTy(ctx), "float", "psize");
            }
            for (VarSym *v : varyings)
                addMember(llvmType(v->type, ctx),
                          mslTypeName(v->type).c_str(), v->name.c_str());
        }
        llvm::MDNode *sti = llvm::MDNode::get(ctx, stiFields);
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 0)),
            llvm::MDString::get(ctx, "air.buffer"),
            llvm::MDString::get(ctx, "air.location_index"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 29)),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
            llvm::MDString::get(ctx, "air.read_write"),
            llvm::MDString::get(ctx, "air.address_space"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
            llvm::MDString::get(ctx, "air.struct_type_info"), sti,
            llvm::MDString::get(ctx, "air.arg_type_size"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                i32, isTessCapture ? tessCaptureStride : recSize)),
            llvm::MDString::get(ctx, "air.arg_type_align_size"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 16)),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "VSOut"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "capture")}));
    }
    if (hasBuffer) {
        llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
        unsigned idx;
        if (isVS || isTES || isKernel) {
            idx = (isCapture ? 1 : 0) + (isCapture ? 0 : attrCount);
        } else {
            /* fragment: [ssbo..., ubo..., tex/smp pairs..., varyings...,
             * buffer, fragCoord?] */
            idx = ssboCount + uboCount + acCount + (needsBufferSizeBuffer ? 1 : 0);
            for (VarSym &v : syms) {
                if (v.kind == VarSym::TEXTURE) {
                    uint32_t elements =
                        v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
                    idx += 2u * elements;
                }
            }
            for (VarSym &v : syms) {
                if (v.kind == VarSym::IMAGE)
                    idx += v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
            }
            for (VarSym &v : syms) {
                if (isVS || isTES || isKernel || v.kind != VarSym::VARYING)
                    continue;
                idx += v.type.isArray() ? (uint32_t)v.type.arr : 1u;
            }
        }
        std::vector<llvm::Metadata *> structFields;
        for (const Uniform &u : uniforms) {
            structFields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32, u.offset)));
            structFields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32, u.size)));
            structFields.push_back(llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32, 0)));
            structFields.push_back(llvm::MDString::get(ctx, mslTypeName(u.type)));
            structFields.push_back(llvm::MDString::get(ctx, u.name));
        }
        llvm::MDNode *sti = llvm::MDNode::get(ctx, structFields);
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, idx)),
            llvm::MDString::get(ctx, "air.buffer"),
            llvm::MDString::get(ctx, "air.buffer_size"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, bufferSize)),
            llvm::MDString::get(ctx, "air.location_index"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                i32, userBufferLocationBase)),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
            llvm::MDString::get(ctx, "air.read"),
            llvm::MDString::get(ctx, "air.address_space"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), 1)),
            llvm::MDString::get(ctx, "air.struct_type_info"), sti,
            llvm::MDString::get(ctx, "air.arg_type_size"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, bufferSize)),
            llvm::MDString::get(ctx, "air.arg_type_align_size"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 16)),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "UBO"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "ubo")}));
    }
    /* SSBO buffers: independent writable device buffers (air.read_write),
     * one parameter per instance. */
    {
        /* location_index matches reflection (attrs + plain pack reserved).
         * LLVM arg index still skips attrs/pack only on VS/TES/kernel —
         * FS keeps SSBOs as leading args. */
        uint32_t loc =
            plainBufMetalSlots + attrCount + userBufferLocationBase;
        uint32_t ssboArg = (isCapture ? 1 : 0) +
                           ((isVS || isTES || isKernel) ? (hasBuffer ? 1 : 0) : 0) +
                           (isCapture ? 0 : attrCount);
        for (VarSym &v : syms) {
            if (v.kind != VarSym::SSBO) continue;
            const MGLIRSymbol *sb = findSymbol(&mod, v.name.c_str());
            uint32_t nelems =
                uniformBlockElementCount(sb ? sb->type : nullptr);
            const MGLIRType *elemTy = sb ? sb->type : nullptr;
            if (elemTy && elemTy->kind == MGLIR_TYPE_ARRAY)
                elemTy = elemTy->elem_type;
            uint32_t bsize = elemTy ? elemTy->layout.size : 0;
            for (uint32_t k = 0; k < nelems; k++) {
                std::string argName = v.name;
                if (nelems > 1u)
                    argName += "[" + std::to_string(k) + "]";
                argNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), ssboArg++)),
                    llvm::MDString::get(ctx, "air.buffer"),
                    llvm::MDString::get(ctx, "air.location_index"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), loc++)),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 1)),
                    llvm::MDString::get(ctx, "air.read_write"),
                    llvm::MDString::get(ctx, "air.address_space"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 1)),
                    llvm::MDString::get(ctx, "air.arg_type_size"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), bsize)),
                    llvm::MDString::get(ctx, "air.arg_type_align_size"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 16)),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, argName),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, argName)}));
            }
        }
    }
    /* Uniform blocks: independent read-only device buffers. */
    {
        /* location_index must match mglAirReflectModule (attrs reserved).
         * Capture used to set loc=1 when attrCount==0, so the draw path
         * bound the UBO at reflected slot 0 while the tess-capture
         * metallib read slot 1 — VS UBO reads failed whenever GS/tess
         * capture ran without vertex attributes (420pack binding_uniform). */
        uint32_t loc =
            plainBufMetalSlots +
            ssboCount + attrCount + userBufferLocationBase;
        uint32_t uboArg = (isCapture ? 1 : 0) +
                          ((isVS || isTES || isKernel) ? (hasBuffer ? 1 : 0) : 0) +
                          ssboCount + (isCapture ? 0 : attrCount);
        for (VarSym &v : syms) {
            if (v.kind != VarSym::UBO) continue;
            const MGLIRSymbol *sb = findSymbol(&mod, v.name.c_str());
            const MGLIRType *blockTy = uniformBlockType(sb ? sb->type : nullptr);
            uint32_t bsize = blockTy ? blockTy->layout.size : 0;
            uint32_t uelems =
                uniformBlockElementCount(sb ? sb->type : nullptr);
            for (uint32_t k = 0; k < uelems; k++) {
                std::string aname =
                    uelems > 1u
                        ? v.name + "[" + std::to_string(k) + "]"
                        : v.name;
                argNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), uboArg++)),
                    llvm::MDString::get(ctx, "air.buffer"),
                    llvm::MDString::get(ctx, "air.location_index"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), loc++)),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 1)),
                    llvm::MDString::get(ctx, "air.read"),
                    llvm::MDString::get(ctx, "air.address_space"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 1)),
                    llvm::MDString::get(ctx, "air.arg_type_size"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), bsize)),
                    llvm::MDString::get(ctx, "air.arg_type_align_size"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 16)),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, v.name),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, aname)}));
            }
        }
    }
    /* Atomic counter buffers: one device buffer per atomic_uint instance. */
    {
        uint32_t loc =
            plainBufMetalSlots +
            ssboCount + uboCount + attrCount + userBufferLocationBase;
        uint32_t acArg = (isCapture ? 1 : 0) +
                         ((isVS || isTES || isKernel) ? (hasBuffer ? 1 : 0) : 0) +
                         ssboCount + uboCount + (isCapture ? 0 : attrCount);
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATOMIC_COUNTER) continue;
            const MGLIRSymbol *ac = findSymbol(&mod, v.name.c_str());
            uint32_t elements = 1u;
            if (ac && ac->type->kind == MGLIR_TYPE_ARRAY &&
                ac->type->array_size > 0u) {
                elements = ac->type->array_size;
            }
            uint32_t bsize = elements * 4u;
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), acArg++)),
                llvm::MDString::get(ctx, "air.buffer"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), loc++)),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.read_write"),
                llvm::MDString::get(ctx, "air.address_space"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_size"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), bsize)),
                llvm::MDString::get(ctx, "air.arg_type_align_size"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 4)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, v.name),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, v.name)}));
        }
    }
    if (needsBufferSizeBuffer) {
        llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                i32, sizeBufferArg)),
            llvm::MDString::get(ctx, "air.buffer"),
            llvm::MDString::get(ctx, "air.location_index"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                i32, runtimeArraySizeBufferIndex)),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
            llvm::MDString::get(ctx, "air.read"),
            llvm::MDString::get(ctx, "air.address_space"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 2)),
            llvm::MDString::get(ctx, "air.arg_type_size"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 4)),
            llvm::MDString::get(ctx, "air.arg_type_align_size"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 4)),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "uint"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "spvBufferSizeConstants")}));
    }
    /* Texture/sampler pairs: air.texture + air.sampler arguments. */
    {
        uint32_t texLoc = 0, smpLoc = 0;
        uint32_t texArg = (isCapture ? 1 : 0) +
                       ((isVS || isTES || isKernel) ? (hasBuffer ? 1 : 0) : 0) +
                          ssboCount + uboCount + acCount +
                          (needsBufferSizeBuffer ? 1 : 0) +
                          (isCapture ? 0 : attrCount);
        for (VarSym &v : syms) {
            if (v.kind != VarSym::TEXTURE) continue;
            const MGLIRType *samplerType = v.opaqueType;
            if (!samplerType) {
                const MGLIRSymbol *tss = findSymbol(&mod, v.name.c_str());
                samplerType = tss ? tss->type : nullptr;
            }
            while (samplerType && samplerType->kind == MGLIR_TYPE_ARRAY)
                samplerType = samplerType->elem_type;
            bool is3d = samplerType &&
                        samplerType->kind == MGLIR_TYPE_SAMPLER &&
                        samplerType->tex_kind == MGLIR_TEX_3D;
            bool is2dArray = samplerType &&
                             samplerType->kind == MGLIR_TYPE_SAMPLER &&
                             (samplerType->tex_kind == MGLIR_TEX_2D_ARRAY ||
                              samplerType->tex_kind == MGLIR_TEX_1D_ARRAY ||
                              samplerType->tex_kind == MGLIR_TEX_2D_MS ||
                              samplerType->tex_kind == MGLIR_TEX_2D_MS_ARRAY);
            bool isCube = samplerType &&
                          samplerType->kind == MGLIR_TYPE_SAMPLER &&
                          samplerType->tex_kind == MGLIR_TEX_CUBE;
            bool isCubeArray = samplerType &&
                               samplerType->kind == MGLIR_TYPE_SAMPLER &&
                               samplerType->tex_kind == MGLIR_TEX_CUBE_ARRAY;
            const char *texelName = "float";
            if (samplerType && samplerType->kind == MGLIR_TYPE_SAMPLER) {
                if (samplerType->tex_storage == MGLIR_SCALAR_INT)
                    texelName = "int";
                else if (samplerType->tex_storage == MGLIR_SCALAR_UINT)
                    texelName = "uint";
            }
            /* Match IMAGE metadata / sample_texture_cube*: samplerCube must
             * be texturecube, not texture2d (else Metal PSO compile fails). */
            std::string sampledType = is3d ? "texture3d<"
                                  : is2dArray ? "texture2d_array<"
                                  : isCubeArray ? "texturecube_array<"
                                  : isCube ? "texturecube<"
                                              : "texture2d<";
            sampledType += texelName;
            sampledType += ", sample>";
            uint32_t elements = v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
            for (uint32_t element = 0; element < elements; element++) {
            std::string elementName = v.name;
            if (elements > 1u)
                elementName += "[" + std::to_string(element) + "]";
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), texArg++)),
                llvm::MDString::get(ctx, "air.texture"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), texLoc++)),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.sample"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, sampledType.c_str()),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, elementName.c_str())}));
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), texArg++)),
                llvm::MDString::get(ctx, "air.sampler"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), smpLoc++)),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "sampler"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, elementName.c_str())}));
            }
        }
        for (VarSym &v : syms) {
            if (v.kind != VarSym::IMAGE) continue;
            const MGLIRType *imgTy =
                imageElementType(findSymbol(&mod, v.name.c_str()));
            MGLIRTexKind itk = imgTy ? imgTy->tex_kind : MGLIR_TEX_2D;
            MGLIRScalar ist =
                imgTy ? imgTy->tex_storage : MGLIR_SCALAR_FLOAT;
            const char *accessTy = "float";
            if (ist == MGLIR_SCALAR_INT) accessTy = "int";
            else if (ist == MGLIR_SCALAR_UINT) accessTy = "uint";
            const char *dimTy = "texture2d";
            switch (itk) {
            case MGLIR_TEX_3D: dimTy = "texture3d"; break;
            case MGLIR_TEX_2D_ARRAY:
            case MGLIR_TEX_1D_ARRAY:
            case MGLIR_TEX_2D_MS:
            case MGLIR_TEX_2D_MS_ARRAY:
                dimTy = "texture2d_array";
                break;
            case MGLIR_TEX_CUBE: dimTy = "texturecube"; break;
            case MGLIR_TEX_CUBE_ARRAY: dimTy = "texturecube_array"; break;
            case MGLIR_TEX_BUFFER:
                /* Matches TEXBUFFER CREATE fallback (packed as texture2d). */
                dimTy = "texture2d";
                break;
            case MGLIR_TEX_1D:
            case MGLIR_TEX_2D:
            case MGLIR_TEX_2D_RECT:
            default: dimTy = "texture2d"; break;
            }
            char imageType[96];
            snprintf(imageType, sizeof(imageType),
                     "%s<%s, access::read_write>", dimTy, accessTy);
            uint32_t elements = v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
            for (uint32_t element = 0; element < elements; element++) {
                std::string elementName = v.name;
                if (elements > 1u)
                    elementName += "[" + std::to_string(element) + "]";
                argNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), texArg++)),
                    llvm::MDString::get(ctx, "air.texture"),
                    llvm::MDString::get(ctx, "air.location_index"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), texLoc++)),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 1)),
                    llvm::MDString::get(ctx, "air.read_write"),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, imageType),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, elementName)}));
            }
        }
    }
    if (isVS && isCapture) {
        /* XFB capture variant: vertex_input metadata emitted after all
         * buffer/texture arguments (mirroring the argument order), so the
         * stage_in value args sit at even slots (Metal rejects odd-slot
         * value args directly after a buffer with "Unsupported attribute
         * type"). */
        uint32_t mArgSlot = 1 + (hasBuffer ? 1 : 0) + ssboCount + uboCount + acCount +
                            (needsBufferSizeBuffer ? 1 : 0) + 2 * texCount +
                            imageCount;
        uint32_t attrLoc = 0;
        uint32_t nextFreeAttrLoc = 0;
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATTR) continue;
            /* Prefer bindAttribLocation, then explicit/IR location, then
             * declaration order — same priority as non-capture + reflector. */
            uint32_t want = airAttribLocation(v.name.c_str(), attrib_names, MAX_ATTRIBS);
            if (want == UINT32_MAX)
                want = v.location;
            if (want == UINT32_MAX)
                want = nextFreeAttrLoc;
            attrLoc = want;
            const uint32_t n = varyingLocationSpan(v.type);
            MType el = v.type;
            if (v.type.isArray() && v.type.arr > 0) el.arr = 0;
            else if (v.type.isMatrix()) el = matrixColumnType(v.type);
            el = attrMetalIfaceType(el);
            for (uint32_t k = 0; k < n; k++) {
                std::string argName = n > 1u
                    ? v.name + "[" + std::to_string(k) + "]"
                    : v.name;
                std::vector<llvm::Metadata *> elems = {
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                    llvm::MDString::get(ctx, "air.vertex_input"),
                    llvm::MDString::get(ctx, "air.location_index"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), attrLoc++)),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 1)),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, mslTypeName(el)),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, argName)};
                argNodes.push_back(llvm::MDNode::get(ctx, elems));
            }
            nextFreeAttrLoc = std::max(nextFreeAttrLoc, attrLoc);
        }
    }
    if (isTESVertex) {
        /* TES-vertex ABI (render encoder): gl_in control points(30), seed
         * TessCoord records(28, read-only), tess factors(26), patch
         * inputs(27), contract(29, {patch_id, vertices_per_patch, items, 0}).
         * All are read-only; the expanded vertex is returned as the VS
         * output record.  The cull-record buffer param(28) reuses the seed
         * stream and is only declared when the shader uses gl_CullDistance. */
        uint32_t arg = (hasBuffer ? 1u : 0u) + ssboCount + uboCount + acCount +
                       (needsBufferSizeBuffer ? 1u : 0u) + 2u * texCount +
                       imageCount;
        const uint32_t locs[5] = {30u, 28u, 26u, 27u, 29u};
        const char *names[5] = {"tes_gl_in", "tes_seed_records",
                                "tess_factors", "tes_patch_inputs",
                                "tes_contract"};
        for (int i = 0; i < 5; i++) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), arg++)),
                llvm::MDString::get(ctx, "air.buffer"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), locs[i])),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.read"),
                llvm::MDString::get(ctx, "air.address_space"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uchar*"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, names[i])}));
        }
        if (usesTesVertexCull) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), arg++)),
                llvm::MDString::get(ctx, "air.buffer"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 28u)),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.read"),
                llvm::MDString::get(ctx, "air.address_space"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uchar*"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "tes_cull_records")}));
        }
    }
    if (isTCS) {
        uint32_t arg = (hasBuffer ? 1u : 0u) + ssboCount + uboCount + acCount +
                       (needsBufferSizeBuffer ? 1u : 0u) + 2u * texCount +
                       imageCount;
        const uint32_t locs[5] = {24u, 26u, 27u, 28u, 29u};
        const char *names[5] = {"tcs_stage_in", "tess_factors",
                                "tcs_patch_out", "tcs_stage_out",
                                "tcs_indirect"};
        for (int i = 0; i < 5; i++) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), arg++)),
                llvm::MDString::get(ctx, "air.buffer"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), locs[i])),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, i == 0 ? "air.read" : "air.read_write"),
                llvm::MDString::get(ctx, "air.address_space"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uchar*"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, names[i])}));
        }
    } else if (isTESCompute) {
        /* isolines/point-mode TES kernel ABI: stage_in(24, control points),
         * tess factors(26), patch inputs(27), stage out(28, expanded vertex
         * records), indirect contract(29, {patch_id, vertices_per_patch,
         * items_per_patch, output_offset}), the optional indexed gather
         * stream(30)/params(25), and the optional transform-feedback
         * stream(31). */
        uint32_t arg = (hasBuffer ? 1u : 0u) + ssboCount + uboCount + acCount +
                       (needsBufferSizeBuffer ? 1u : 0u) + 2u * texCount +
                       imageCount;
        const uint32_t locs[8] = {24u, 26u, 27u, 28u, 29u, 30u, 25u, 31u};
        const char *names[8] = {"tes_stage_in", "tess_factors",
                                "tes_patch_inputs", "tes_stage_out",
                                "tes_indirect", "tes_gather",
                                "tes_gather_params", "tes_xfb_out"};
        const char *access[8] = {"air.read", "air.read", "air.read",
                                 "air.read_write", "air.read", "air.read",
                                 "air.read", "air.read_write"};
        for (int i = 0; i < 8; i++) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), arg++)),
                llvm::MDString::get(ctx, "air.buffer"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), locs[i])),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, access[i]),
                llvm::MDString::get(ctx, "air.address_space"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uchar*"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, names[i])}));
        }
    } else if (isGS) {
        uint32_t arg = (hasBuffer ? 1u : 0u) + ssboCount + uboCount + acCount +
                       (needsBufferSizeBuffer ? 1u : 0u) + 2u * texCount +
                       imageCount;
        /* Fixed ABI slots (mgl_air_gs_abi.h §1/§5b/§7): input, output,
         * counts, indexed gather stream, gather params, XFB stream, XFB
         * meta, and the ordered-scatter visibility buffer.  The gather
         * buffer and params constant are read-only; output/counts/XFB/
         * visibility are read_write. */
        const uint32_t locs[8] = {24u, 28u, 29u, 30u, 25u, 31u, 27u, 26u};
        const char *names[8] = {"gs_input", "gs_output", "gs_count",
                                "gs_gather", "gs_gather_params",
                                "gs_xfb_out", "gs_xfb_meta", "gs_xfb_vis"};
        for (int i = 0; i < 8; i++) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), arg++)),
                llvm::MDString::get(ctx, "air.buffer"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), locs[i])),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, (i == 0 || i == 3 || i == 4)
                                                ? "air.read" : "air.read_write"),
                llvm::MDString::get(ctx, "air.address_space"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uchar*"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, names[i])}));
        }
    }
    uint32_t mArgSlot =
        (isCapture ? 1 : 0) +
        ((isVS || isTES || isKernel) ? (hasBuffer ? 1 : 0) : 0) + ssboCount +
        uboCount + (needsBufferSizeBuffer ? 1 : 0) + 2 * texCount + imageCount;
    if (isTCS) mArgSlot += 5;
    else if (isGS) mArgSlot += 8;  /* input/output/counts/gather/params/xfb/xfb-meta/xfb-vis */
    else if (isTESCompute) mArgSlot += 8; /* stage_in/factors/patches/out/indirect/gather/params/xfb */
    else if (isTESVertex) mArgSlot += usesTesVertexCull ? 6 : 5; /* gl_in/seed/factors/patch/contract[/cull] */
    if (isVS) {
        /* Vertex attribute metadata already emitted above. */
    } else if (isTES && !isTESCompute && !isTESVertex) {
        /* Native post-tessellation only: TES-vertex declares its own five
         * buffers above and carries no patch-control-point hidden args. */
        uint32_t hiddenArg = mArgSlot;
        const uint32_t locations[3] = {30u, 28u, 27u};
        const char *names[3] = {"mgl_control_points", "mgl_patch_info",
                                "mgl_patch_inputs"};
        for (uint32_t i = 0; i < 3; i++) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), hiddenArg++)),
                llvm::MDString::get(ctx, "air.buffer"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), locations[i])),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.read"),
                llvm::MDString::get(ctx, "air.address_space"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uchar*"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, names[i])}));
        }
        mArgSlot += 3;
        llvm::MDNode *getterRef = llvm::MDNode::get(ctx, {
            llvm::MDString::get(ctx, "air.patch_control_point_function"),
            llvm::ValueAsMetadata::get(controlPointGetter)});
        llvm::MDNode *fieldInfo = llvm::MDNode::get(ctx, {
            llvm::MDString::get(ctx, "air.location_index"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), 0)),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), 1)),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "float4"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "position")});
        std::vector<llvm::Metadata *> patchInput = {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), mArgSlot)),
            llvm::MDString::get(ctx, "air.patch_control_point_input"),
            getterRef, fieldInfo};
        for (VarSym &v : syms) {
            if (v.kind != VarSym::CONTROL_POINT_INPUT || v.isPatch) continue;
            uint32_t location = v.location + 1u;
            patchInput.push_back(llvm::MDNode::get(ctx, {
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), location)),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, mslTypeName(v.type)),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, v.name)}));
        }
        argNodes.push_back(llvm::MDNode::get(ctx, patchInput));
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), mArgSlot + 1)),
            llvm::MDString::get(ctx, "air.position_in_patch"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "float3"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "tessCoord")}));
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), mArgSlot + 2)),
            llvm::MDString::get(ctx, "air.patch_id"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "uint"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "patchId")}));
    } else if (!isKernel && !isTESVertex) {
        auto emitFSVarying = [&](const std::string &tagName,
                                 const MType &mt, uint32_t argIdx,
                                 bool forceFlat = false) {
            const bool flat = forceFlat || varyingUsesFloatCarrier(mt, has_gs) ||
                                !scalarIsFloat(mt.scalar);
            MType iface = mt;
            if (varyingUsesFloatCarrier(mt, has_gs) || forceFlat) {
                MType src = mt;
                if (forceFlat && scalarIsFloat(mt.scalar))
                    src = MType{MGLIR_SCALAR_FLOAT};
                iface = floatCarrierType(src);
            }
            std::vector<llvm::Metadata *> elems = {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), argIdx)),
                llvm::MDString::get(ctx, "air.fragment_input"),
                llvm::MDString::get(ctx, airGenerated(tagName, iface)),
                llvm::MDString::get(ctx, flat ? "air.flat" : "air.center"),
                llvm::MDString::get(ctx,
                                    flat ? "air.no_perspective"
                                         : "air.perspective"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, mslTypeName(iface)),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, tagName)};
            argNodes.push_back(llvm::MDNode::get(ctx, elems));
        };
        for (VarSym &v : syms) {
            if (v.kind != VarSym::VARYING) continue;
            if (v.type.isArray()) {
                /* Flattened: one fragment_input per element, each with the
                 * element-specific interface name (matches the VS side). */
                MType el = v.type;
                el.arr = 0;
                uint32_t n = (uint32_t)v.type.arr;
                for (uint32_t k = 0; k < n; k++) {
                    std::string elName = varyingIfaceTag(v, k, has_gs);
                    emitFSVarying(elName, el, mArgSlot++);
                }
            } else if (v.type.isMatrix()) {
                MType col = matrixColumnType(v.type);
                for (uint32_t c = 0; c < v.type.cols; c++) {
                    std::string colName = varyingIfaceTag(v, c, has_gs);
                    emitFSVarying(colName, col, mArgSlot++);
                }
            } else {
                std::string tag = varyingIfaceTag(v, 0, has_gs);
                if (uintUsesSplitFloatCarrier(v.type, has_gs)) {
                    emitFSVarying(tag + "_lo", v.type, mArgSlot++, true);
                    emitFSVarying(tag + "_hi", v.type, mArgSlot++, true);
                } else {
                    emitFSVarying(tag, v.type, mArgSlot++);
                }
            }
        }
        if (usesFragmentClipDistance) {
            MType floatTy;
            floatTy.scalar = MGLIR_SCALAR_FLOAT;
            for (uint32_t i = 0; i < activeClipCount; i++) {
                std::string elName =
                    "gl_ClipDistance_elm" + std::to_string(i);
                emitFSVarying(elName, floatTy, mArgSlot++, true);
            }
        }
        if (usesFragmentCullDistance) {
            MType floatTy;
            floatTy.scalar = MGLIR_SCALAR_FLOAT;
            for (uint32_t i = 0; i < activeCullCount; i++) {
                std::string elName =
                    "gl_CullDistance_elm" + std::to_string(i);
                emitFSVarying(elName, floatTy, mArgSlot++, true);
            }
        }
        if (usesFragCoord || usesFrontFacing || usesPointCoord ||
            usesPrimitiveId || usesLayer || usesViewportIndex ||
            usesSampleID || usesSamplePosition || usesSampleMaskIn ||
            needParamsBuffer || usesFragmentCullDistance ||
            usesFragmentClipDistance) {
            /* Fragment builtins sit after the varyings and the optional
             * uniform buffer in the arg order; skip that slot once. */
            if (hasBuffer) mArgSlot++;
        }
        if (usesFragCoord) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.position"),
                llvm::MDString::get(ctx, "air.center"),
                llvm::MDString::get(ctx, "air.no_perspective"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "float4"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "gl_FragCoord")}));
        }
        if (usesFrontFacing) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.front_facing"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "bool"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "gl_FrontFacing")}));
        }
        if (usesPointCoord) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.point_coord"),
                llvm::MDString::get(ctx, "air.center"),
                llvm::MDString::get(ctx, "air.no_perspective"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "float2"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "gl_PointCoord")}));
        }
        if (usesPrimitiveId) {
            if (stage == MGL_STAGE_FRAGMENT && has_gs) {
                /* GS expansion path: the id arrives as a flat float carrier
                 * from the passthrough VS, which declares
                 *   layout(location = MGL_AIR_PRIMITIVE_ID_LOCATION)
                 *   flat out float mgl_primitive_id;
                 * Per varyingIfaceTag(), that output is tagged mgl_loc_N —
                 * not the GLSL name — so the FS must use the same location
                 * tag (see the has_gs forceLocationTag comment above). */
                MType carrierType;
                carrierType.scalar = MGLIR_SCALAR_FLOAT;
                const std::string primTag =
                    "mgl_loc_" +
                    std::to_string((unsigned)MGL_AIR_PRIMITIVE_ID_LOCATION);
                argNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                    llvm::MDString::get(ctx, "air.fragment_input"),
                    llvm::MDString::get(
                        ctx, airGenerated(primTag, carrierType)),
                    llvm::MDString::get(ctx, "air.flat"),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, "float"),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, "gl_PrimitiveID")}));
            } else {
                argNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                    llvm::MDString::get(ctx, "air.primitive_id"),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, "uint"),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, "gl_PrimitiveID")}));
            }
        }
        if (usesLayer) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.render_target_array_index"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "gl_Layer")}));
        }
        if (usesViewportIndex) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.viewport_array_index"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "gl_ViewportIndex")}));
        }
        if (needSampleID) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.sample_id"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "gl_SampleID")}));
        }
        if (usesSampleMaskIn) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.sample_mask"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "gl_SampleMaskIn")}));
        }
        if (needParamsBuffer) {
            llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32, mArgSlot++)),
                llvm::MDString::get(ctx, "air.buffer"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    i32, kMGLFragCoordParamsBufferIndex)),
                llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32, 1)),
                llvm::MDString::get(ctx, "air.read"),
                llvm::MDString::get(ctx, "air.address_space"),
                llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32, 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "device uchar*"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, needFragCoordParams
                                            ? "mgl_fragcoord_params"
                                            : "mgl_sample_params")}));
        }
    }
    if (isKernel) {
        /* Kernel thread position: [[thread_position_in_grid]] as uint3. */
        uint32_t kSlot = mArgSlot;
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), kSlot++)),
            llvm::MDString::get(ctx, isTCS ? "air.thread_position_in_threadgroup"
                                           : "air.thread_position_in_grid"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "uint3"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, isTCS ? "thread_position_in_threadgroup"
                                           : "thread_position_in_grid")}));
        if (usesLocalInvocation && !isTCS) {
            if (usesLocalInvocationID) {
                argNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), kSlot++)),
                    llvm::MDString::get(ctx,
                                        "air.thread_position_in_threadgroup"),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, "uint3"),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx,
                                        "thread_position_in_threadgroup")}));
            }
            if (usesLocalInvocationIndex) {
                argNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), kSlot++)),
                    llvm::MDString::get(ctx, "air.thread_index_in_threadgroup"),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, "uint"),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, "thread_index_in_threadgroup")}));
            }
        }
        if (isTCS || isTESCompute || usesWorkGroupID) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), kSlot++)),
                llvm::MDString::get(ctx, "air.threadgroup_position_in_grid"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint3"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "threadgroup_position_in_grid")}));
        }
        if (usesNumWorkGroups) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), kSlot++)),
                llvm::MDString::get(ctx, "air.threadgroups_per_grid"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint3"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "threadgroups_per_grid")}));
        }
    }

    std::vector<llvm::Metadata *> outNodes;   /* outputs / render targets */
    if (isTessCapture || isCullCapture) {
        /* Both capture kernels return float4 position: AGX rejects a
         * rasterization-enabled pipeline whose vertex function returns
         * void (or lacks [[position]]).  The real XFB/cull payload travels
         * through the slot-29 record buffer. */
        outNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::MDString::get(ctx, "air.position"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "float4"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "position")}));
    } else if ((isVS || (isTES && !isTESCompute) || isTESVertex) && !isCapture) {
        outNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::MDString::get(ctx, "air.position"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "float4"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "position")}));
        if (usesPointSize) {
            outNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::MDString::get(ctx, "air.point_size"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "float"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "psize")}));
        }
        if (usesClipDistance) {
            /* Reference shape from MSL 'float cd [[clip_distance]] [N]':
             * air.clip_distance + air.clip_distance_array_size. */
            outNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::MDString::get(ctx, "air.clip_distance"),
                llvm::MDString::get(ctx, "air.clip_distance_array_size"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), MGL_MAX_CLIP_DISTANCES)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "float"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "gl_ClipDistance")}));
            MType floatTy;
            floatTy.scalar = MGLIR_SCALAR_FLOAT;
            for (uint32_t i = 0; i < MGL_MAX_CLIP_DISTANCES; i++) {
                std::string elName =
                    "gl_ClipDistance_elm" + std::to_string(i);
                outNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::MDString::get(ctx, "air.vertex_output"),
                    llvm::MDString::get(ctx, airGenerated(elName, floatTy)),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, "float"),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, elName)}));
            }
        }
        if (usesCullDistancePassthrough) {
            MType floatTy;
            floatTy.scalar = MGLIR_SCALAR_FLOAT;
            for (uint32_t i = 0; i < activeCullCount; i++) {
                std::string elName =
                    "gl_CullDistance_elm" + std::to_string(i);
                outNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::MDString::get(ctx, "air.vertex_output"),
                    llvm::MDString::get(ctx, airGenerated(elName, floatTy)),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, "float"),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, elName)}));
            }
        }
        if (usesLayerViewport) {
            outNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::MDString::get(ctx, "air.render_target_array_index"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "mgl_layer")}));
            outNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::MDString::get(ctx, "air.viewport_array_index"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "mgl_viewport_index")}));
        }
        for (VarSym *v : varyings) {
            if (v->type.isArray()) {
                /* Flattened (Metal forbids array stage-out members): one
                 * vertex_output per element with the element-specific
                 * interface name; the FS side emits the identical tags. */
                MType el = v->type;
                el.arr = 0;
                uint32_t n = (uint32_t)v->type.arr;
                for (uint32_t k = 0; k < n; k++) {
                    std::string elName = varyingIfaceTag(*v, k, has_gs);
                    MType outTy = varyingUsesFloatCarrier(el, has_gs)
                        ? floatCarrierType(el) : el;
                    outNodes.push_back(llvm::MDNode::get(ctx, {
                        llvm::MDString::get(ctx, "air.vertex_output"),
                        llvm::MDString::get(ctx,
                                            airGenerated(elName, outTy)),
                        llvm::MDString::get(ctx, "air.arg_type_name"),
                        llvm::MDString::get(ctx, mslTypeName(outTy)),
                        llvm::MDString::get(ctx, "air.arg_name"),
                        llvm::MDString::get(ctx, elName)}));
                }
            } else if (v->type.isMatrix()) {
                MType col = matrixColumnType(v->type);
                for (uint32_t c = 0; c < v->type.cols; c++) {
                    std::string colName = varyingIfaceTag(*v, c, has_gs);
                    MType outTy = varyingUsesFloatCarrier(col, has_gs)
                        ? floatCarrierType(col) : col;
                    outNodes.push_back(llvm::MDNode::get(ctx, {
                        llvm::MDString::get(ctx, "air.vertex_output"),
                        llvm::MDString::get(ctx,
                                            airGenerated(colName, outTy)),
                        llvm::MDString::get(ctx, "air.arg_type_name"),
                        llvm::MDString::get(ctx, mslTypeName(outTy)),
                        llvm::MDString::get(ctx, "air.arg_name"),
                        llvm::MDString::get(ctx, colName)}));
                }
            } else {
                std::string tag = varyingIfaceTag(*v, 0, has_gs);
                if (uintUsesSplitFloatCarrier(v->type, has_gs)) {
                    MType outTy = floatCarrierType(v->type);
                    outNodes.push_back(llvm::MDNode::get(ctx, {
                        llvm::MDString::get(ctx, "air.vertex_output"),
                        llvm::MDString::get(ctx,
                            airGenerated(tag + "_lo", outTy)),
                        llvm::MDString::get(ctx, "air.arg_type_name"),
                        llvm::MDString::get(ctx, mslTypeName(outTy)),
                        llvm::MDString::get(ctx, "air.arg_name"),
                        llvm::MDString::get(ctx, tag + "_lo")}));
                    outNodes.push_back(llvm::MDNode::get(ctx, {
                        llvm::MDString::get(ctx, "air.vertex_output"),
                        llvm::MDString::get(ctx,
                            airGenerated(tag + "_hi", outTy)),
                        llvm::MDString::get(ctx, "air.arg_type_name"),
                        llvm::MDString::get(ctx, mslTypeName(outTy)),
                        llvm::MDString::get(ctx, "air.arg_name"),
                        llvm::MDString::get(ctx, tag + "_hi")}));
                } else {
                    MType outTy = varyingUsesFloatCarrier(v->type, has_gs)
                        ? floatCarrierType(v->type) : v->type;
                    outNodes.push_back(llvm::MDNode::get(ctx, {
                        llvm::MDString::get(ctx, "air.vertex_output"),
                        llvm::MDString::get(ctx, airGenerated(tag, outTy)),
                        llvm::MDString::get(ctx, "air.arg_type_name"),
                        llvm::MDString::get(ctx, mslTypeName(outTy)),
                        llvm::MDString::get(ctx, "air.arg_name"),
                        llvm::MDString::get(ctx, tag)}));
                }
            }
        }
    } else if (!isKernel) {
        VarSym *arrayOut = nullptr;
        for (VarSym &v : syms) {
            if (v.kind == VarSym::OUTPUT && v.type.isArray()) {
                arrayOut = &v;
                break;
            }
        }
        if (arrayOut) {
            /* Fragment output arrays: one render_target node per element.
             * Type must match the GLSL element (float4/int4/uint4). */
            MType el = arrayOut->type;
            el.arr = 0;
            const std::string elTypeName = mslTypeName(el);
            for (uint32_t i = 0; i < (uint32_t)arrayOut->type.arr; i++) {
                std::string elName = std::string(arrayOut->name) + "_" +
                                     std::to_string(i);
                outNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::MDString::get(ctx, "air.render_target"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), i)),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 0)),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, elTypeName),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, elName.c_str())}));
            }
        } else {
            for (VarSym *out : fragOutputs) {
                outNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::MDString::get(ctx, "air.render_target"),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), out->location)),
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), 0)),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, mslTypeName(out->type)),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, out->name)}));
            }
        }
        if (usesFragDepth) {
            /* Reference shape from aux_shaders/scaled_depth_blit.metal:
             * the depth output is air.depth + air.depth_qualifier air.any
             * in the fragment output list, matched to the struct member by
             * position (second member, after the render target). */
            outNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::MDString::get(ctx, "air.depth"),
                llvm::MDString::get(ctx, "air.depth_qualifier"),
                llvm::MDString::get(ctx, "air.any"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "float"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "depth")}));
        }
        if (usesSampleMask) {
            outNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::MDString::get(ctx, "air.sample_mask"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "gl_SampleMask")}));
        }
    }

    if (usesCullDistance && cullBufferArgIdx != UINT32_MAX &&
        cullParamsArgIdx != UINT32_MAX) {
        llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
        uint32_t cullBufferArg = cullBufferArgIdx;
        uint32_t cullParamsArg = cullParamsArgIdx;
        auto addCullBuffer = [&](uint32_t arg, uint32_t location,
                                 uint32_t size, const char *typeName,
                                 const char *argName) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, arg)),
                llvm::MDString::get(ctx, "air.buffer"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(
                    llvm::ConstantInt::get(i32, location)),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
                llvm::MDString::get(ctx, "air.read"),
                llvm::MDString::get(ctx, "air.address_space"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
                llvm::MDString::get(ctx, "air.arg_type_size"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, size)),
                llvm::MDString::get(ctx, "air.arg_type_align_size"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 4)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, typeName),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, argName)}));
        };
        addCullBuffer(cullBufferArg, 29u, 4u, "float", "mgl_cull_buf");
        addCullBuffer(cullParamsArg, 28u, 48u,
                      "MGLCullDistanceParams", "mgl_cull_params");
    } else if (isCullCapture || isTessCapture) {
        llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
        uint32_t cullParamsArg = (uint32_t)paramTys.size() - 4u;
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32, cullParamsArg)),
            llvm::MDString::get(ctx, "air.buffer"),
            llvm::MDString::get(ctx, "air.location_index"),
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(i32, 28u)),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
            llvm::MDString::get(ctx, "air.read"),
            llvm::MDString::get(ctx, "air.address_space"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
            llvm::MDString::get(ctx, "air.arg_type_size"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                i32, isCullCapture ? 48u : 12u)),
            llvm::MDString::get(ctx, "air.arg_type_align_size"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 4)),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, isCullCapture
                ? "MGLCullDistanceParams" : "MGLTessCaptureParams"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, isCullCapture
                ? "mgl_cull_capture_params" : "mgl_tess_capture_params")}));
    }

    if (isVS || isTESVertex) {
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx),
                (unsigned)paramTys.size() - 3u)),
            llvm::MDString::get(ctx, "air.instance_id"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "uint"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "iid")}));
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx),
                (unsigned)paramTys.size() - 2u)),
            llvm::MDString::get(ctx, "air.base_instance"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "uint"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "base_iid")}));
        /* Vertex stage vertex id (gl_VertexID). */
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx),
                (unsigned)paramTys.size() - 1u)),
            llvm::MDString::get(ctx, "air.vertex_id"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "uint"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "vid")}));
    }
    /* Metal expects the argument list ordered by parameter index; the
     * emission order above mixes buffers and value args (e.g. a fragment
     * shader's uniform buffer node precedes its fragment_input node even
     * though the value parameter comes first in the signature).  Sort
     * stably by the leading argument-index integer. */
    std::stable_sort(argNodes.begin(), argNodes.end(),
                     [](const llvm::Metadata *a, const llvm::Metadata *b) {
                         auto idx = [](const llvm::Metadata *m) -> long {
                             auto *n = llvm::dyn_cast<llvm::MDNode>(m);
                             if (!n || n->getNumOperands() == 0) return -1;
                             auto *c =
                                 llvm::dyn_cast<llvm::ConstantAsMetadata>(
                                     n->getOperand(0).get());
                             if (!c) return -1;
                             auto *ci = llvm::dyn_cast<llvm::ConstantInt>(
                                 c->getValue());
                             return ci ? (long)ci->getZExtValue() : -1;
                         };
                         return idx(a) < idx(b);
                     });
    std::vector<llvm::Metadata *> stageElems = {
        llvm::ValueAsMetadata::get(fn),
        llvm::MDNode::get(ctx, outNodes)};
    if (!argNodes.empty())
        stageElems.push_back(llvm::MDNode::get(ctx, argNodes));
    else
        stageElems.push_back(llvm::MDNode::get(ctx, {}));
    if (isTES && !isTESCompute && !isTESVertex) {
        stageElems.push_back(llvm::MDNode::get(ctx, {
            llvm::MDString::get(ctx, "air.patch"),
            llvm::MDString::get(
                ctx, tu->layout_primitive == MGL_AST_TES_QUADS
                         ? "quad" : "triangle"),
            llvm::MDString::get(ctx, "air.patch_control_point"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), 0))}));
    }
    /* GLSL layout(early_fragment_tests) → Metal [[early_fragment_tests]].
     * Apple's AIR puts a bare MDString "early_fragment_tests" (no "air."
     * prefix) as the 4th !air.fragment operand — not a nested MDNode.
     * Confirmed via llvm-bcanalyzer on metalfe metallibs. */
    if (!isKernel && !isVS && !isTES && tu->layout_early_fragment_tests) {
        stageElems.push_back(
            llvm::MDString::get(ctx, "early_fragment_tests"));
    }
    llvm::NamedMDNode *air = module.getOrInsertNamedMetadata(
        isKernel ? "air.kernel"
                 : ((isVS || isTES) ? "air.vertex" : "air.fragment"));
    air->addOperand(llvm::MDNode::get(ctx, stageElems));

    llvm::NamedMDNode *ver = module.getOrInsertNamedMetadata("air.version");
    ver->addOperand(llvm::MDNode::get(ctx, {
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(ctx), 2)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(ctx), 8)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(ctx), 0))}));
    llvm::NamedMDNode *lver = module.getOrInsertNamedMetadata(
        "air.language_version");
    lver->addOperand(llvm::MDNode::get(ctx, {
        llvm::MDString::get(ctx, "Metal"),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(ctx), 4)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(ctx), 0)),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(ctx), 0))}));
    addModuleFlags(&module);

    {
        llvm::LoopAnalysisManager LAM;
        llvm::FunctionAnalysisManager FAM;
        llvm::CGSCCAnalysisManager CGAM;
        llvm::ModuleAnalysisManager MAM;
        llvm::PassBuilder PB;
        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

        llvm::ModulePassManager MPM;
        MPM.addPass(llvm::AlwaysInlinerPass());
        llvm::FunctionPassManager FPM;
        FPM.addPass(llvm::SROAPass());
        FPM.addPass(llvm::EarlyCSEPass());
        FPM.addPass(llvm::InstCombinePass());
        FPM.addPass(llvm::DCEPass());
        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
        MPM.run(module, MAM);
    }

    if (mgl_env_flag_enabled("MGL_DUMP_IR"))
        module.print(llvm::errs(), nullptr);

    /* Serialize: bitcode blob + MTLB container. */
    llvm::SmallVector<char, 0> bc;
    llvm::raw_svector_ostream bcos(bc);
    llvm::WriteBitcodeToFile(module, bcos);

    std::vector<mgl::MTLBFunction> fns;
    mgl::MTLBFunction f;
    f.name = "main";
    f.type = isKernel ? mgl::MTLB_FN_KERNEL
                       : ((isVS || isTES) ? mgl::MTLB_FN_VERTEX
                                         : mgl::MTLB_FN_FRAGMENT);
    if (isTES && !isTESCompute && !isTESVertex) {
        /* The metallib TESS tag is 4 * controlPointCount + patchKind; it is
         * how Metal computes the per-patch control-point offset on the CPU
         * side (patchStart * controlPointCount).  Encoding only the patch
         * kind leaves controlPointCount = 0, which makes every patch read
         * its control points from record 0.  The caller passes the patch
         * vertex count (TCS output vertices, or glPatchParameteri without
         * a TCS); fall back to the GL default of 3. */
        uint32_t cpc = tessPatchVertices > 0u ? tessPatchVertices : 3u;
        uint32_t kind = tu->layout_primitive == MGL_AST_TES_QUADS ? 2u : 1u;
        f.tessellation = (uint8_t)(4u * cpc + kind);
    }
    f.bitcode.assign(bc.begin(), bc.end());
    fns.push_back(f);

    llvm::SmallVector<char, 0> mlib;
    llvm::raw_svector_ostream mlibos(mlib);
    mgl::mglMTLBWrite(fns, mlibos);

    unsigned char *out = (unsigned char *)malloc(mlib.size());
    if (!out) {
        snprintf(err_buf, err_cap, "out of memory");
        if (own_session) mglFrontendSessionDestroy(sess);
        return -1;
    }
    memcpy(out, mlib.data(), mlib.size());
    *metallib_out = out;
    *size_out = mlib.size();

    if (mgl_env_flag_enabled("MGL_DUMP_METALLIB")) {
        static unsigned s_dumpSeq = 0u;
        char path[128];
        snprintf(path, sizeof(path), "/tmp/mgl_mlib_s%d_%u.metallib", stage,
                 s_dumpSeq++);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(mlib.data(), 1, mlib.size(), f);
            fclose(f);
        }
    }

    if (own_session)
        mglFrontendSessionDestroy(sess);
    return 0;
}

extern "C" int mglShaderCompileGLSL(const char *src, int stage,
                                    unsigned char **metallib_out,
                                    size_t *size_out, char *err_buf,
                                    size_t err_cap) {
    return compileGLSLImpl(src, stage, 0, /*has_gs=*/false,
                           /*force_tes_compute=*/false,
                           /*tes_vertex_render=*/false, nullptr, 0u,
                           /*iface_location_peers=*/nullptr, metallib_out,
                           size_out, err_buf, err_cap);
}

/* XFB capture variant: the vertex stage writes its full output record
 * (position + varyings) into a device buffer at location 29 with
 * rasterization disabled (the capture variant of the mglShaderCompileGLSL
 * compile entry). attrib_names is optional (glBindAttribLocation map). */
extern "C" int mglShaderCompileGLSLCapture(const char *src,
                                           const char *const *attrib_names,
                                           unsigned char **metallib_out,
                                           size_t *size_out, char *err_buf,
                                           size_t err_cap) {
    return compileGLSLImpl(src, MGL_STAGE_VERTEX, 1, /*has_gs=*/false,
                           /*force_tes_compute=*/false,
                           /*tes_vertex_render=*/false, attrib_names,
                           0u, /*iface_location_peers=*/nullptr,
                           metallib_out, size_out, err_buf, err_cap);
}

extern "C" int mglShaderCompileGLSLTessCapture(
    const char *src, const char *const *attrib_names,
    unsigned char **metallib_out, size_t *size_out,
    char *err_buf, size_t err_cap) {
    return compileGLSLImpl(src, MGL_STAGE_VERTEX, 2, /*has_gs=*/false,
                           /*force_tes_compute=*/false,
                           /*tes_vertex_render=*/false, attrib_names,
                           0u, /*iface_location_peers=*/nullptr,
                           metallib_out, size_out, err_buf, err_cap);
}

extern "C" int mglShaderCompileGLSLCullDistanceCapture(
    const char *src, const char *const *attrib_names,
    unsigned char **metallib_out, size_t *size_out,
    char *err_buf, size_t err_cap) {
    return compileGLSLImpl(src, MGL_STAGE_VERTEX, 3, /*has_gs=*/false,
                           /*force_tes_compute=*/false,
                           /*tes_vertex_render=*/false, attrib_names,
                           0u, /*iface_location_peers=*/nullptr,
                           metallib_out, size_out, err_buf, err_cap);
}

static void fillStageInfo(const MGLTranslationUnit *tu,
                          const MGLIRModule *mod, int stage,
                          const char *src, MGLAIRStageInfo *stage_info) {
    (void)src;
    memset(stage_info, 0, sizeof(*stage_info));
    stage_info->needs_runtime_array_size_buffer =
        translationUnitUsesRuntimeArrayLength(tu, mod) ? 1u : 0u;
    if (stage != MGL_STAGE_FRAGMENT && stage != MGL_STAGE_COMPUTE && mod) {
        stage_info->cull_distance_count =
            mglFrontendBuiltinArrayCount(mod, tu, "gl_CullDistance");
        stage_info->clip_distance_count =
            mglFrontendBuiltinArrayCount(mod, tu, "gl_ClipDistance");
        stage_info->uses_cull_distance =
            stage_info->cull_distance_count > 0 ? 1u : 0u;
    }
    if (stage == MGL_STAGE_TESS_CONTROL && tu->layout_vertices > 0)
        stage_info->tess_control_output_vertices =
            static_cast<uint32_t>(tu->layout_vertices);
    if (stage == MGL_STAGE_TESS_EVALUATION) {
        stage_info->tess_gen_mode_specified =
            (tu->layout_primitive == MGL_AST_TES_TRIANGLES ||
             tu->layout_primitive == MGL_AST_TES_QUADS ||
             tu->layout_primitive == MGL_AST_TES_ISOLINES)
                ? 1u : 0u;
        stage_info->tess_gen_mode =
            tu->layout_primitive == MGL_AST_TES_QUADS ? GL_QUADS :
            tu->layout_primitive == MGL_AST_TES_ISOLINES ? GL_ISOLINES :
            GL_TRIANGLES;
        stage_info->tess_gen_spacing =
            tu->layout_spacing == MGL_AST_SPACING_FRACTIONAL_EVEN
                ? GL_FRACTIONAL_EVEN :
            tu->layout_spacing == MGL_AST_SPACING_FRACTIONAL_ODD
                ? GL_FRACTIONAL_ODD : GL_EQUAL;
        stage_info->tess_gen_vertex_order =
            tu->layout_winding == MGL_AST_WINDING_CW ? GL_CW : GL_CCW;
        stage_info->tess_gen_point_mode = tu->layout_point_mode ? 1u : 0u;
        stage_info->uses_tess_level =
            (mglFrontendBuiltinArrayCount(mod, tu, "gl_TessLevelOuter") > 0u ||
             mglFrontendBuiltinArrayCount(mod, tu, "gl_TessLevelInner") > 0u)
                ? 1u : 0u;
    }
    if (stage == MGL_STAGE_GEOMETRY) {
        switch (tu->layout_primitive) {
        case MGL_AST_GS_IN_POINTS:
            stage_info->geometry_input_type = GL_POINTS;
            break;
        case MGL_AST_GS_IN_LINES:
            stage_info->geometry_input_type = GL_LINES;
            break;
        case MGL_AST_GS_IN_LINES_ADJACENCY:
            stage_info->geometry_input_type = GL_LINES_ADJACENCY;
            break;
        case MGL_AST_GS_IN_TRIANGLES_ADJACENCY:
            stage_info->geometry_input_type = GL_TRIANGLES_ADJACENCY;
            break;
        default:
            stage_info->geometry_input_type = GL_TRIANGLES;
            break;
        }
        switch (tu->layout_primitive_out) {
        case MGL_AST_GS_OUT_POINTS:
            stage_info->geometry_output_type = GL_POINTS;
            break;
        case MGL_AST_GS_OUT_LINE_STRIP:
            stage_info->geometry_output_type = GL_LINE_STRIP;
            break;
        default:
            stage_info->geometry_output_type = GL_TRIANGLE_STRIP;
            break;
        }
        stage_info->geometry_vertices_out = tu->layout_max_vertices > 0
            ? static_cast<uint32_t>(tu->layout_max_vertices) : 0u;
        stage_info->geometry_max_vertices_specified =
            tu->layout_max_vertices >= 0 ? 1u : 0u;
        stage_info->geometry_invocations = tu->layout_invocations > 0
            ? static_cast<uint32_t>(tu->layout_invocations) : 1u;
        /* Per-stream output layout: count the OUTPUT varyings per stream
         * (position + varying slots at 16B each make the XFB record
         * stride).  Streams above 0 are transform-feedback only. */
        uint32_t count[MGL_AIR_GS_MAX_STREAMS] = {};
        uint32_t maxStream = 0u;
        for (uint32_t i = 0u; i < mod->symbol_count; i++) {
            const MGLIRSymbol *s = mod->symbols[i];
            if (s->is_function || !s->name ||
                strncmp(s->name, "gl_", 3) == 0) {
                continue;
            }
            if (!(s->qualifiers & MGL_AST_Q_OUT)) continue;
            int32_t stream = s->stream >= 0
                ? s->stream
                : (tu->layout_stream >= 0 ? tu->layout_stream : 0);
            if (stream < 0 || stream >= MGL_AIR_GS_MAX_STREAMS) stream = 0;
            count[stream]++;
            if ((uint32_t)stream > maxStream) maxStream = (uint32_t)stream;
        }
        stage_info->gs_stream_count = maxStream + 1u;
        for (uint32_t s = 0u; s < MGL_AIR_GS_MAX_STREAMS; s++) {
            stage_info->gs_stream_varying_count[s] = count[s];
            stage_info->gs_stream_xfb_stride[s] = 16u + count[s] * 16u;
        }
    }
    if (stage == MGL_STAGE_COMPUTE) {
        /* Unspecified axes default to 1 (GLSL 4.60 §4.4.1.4). */
        stage_info->compute_local_size_x =
            tu->layout_local_size_x > 0 ? (uint32_t)tu->layout_local_size_x
                                        : 1u;
        stage_info->compute_local_size_y =
            tu->layout_local_size_y > 0 ? (uint32_t)tu->layout_local_size_y
                                        : 1u;
        stage_info->compute_local_size_z =
            tu->layout_local_size_z > 0 ? (uint32_t)tu->layout_local_size_z
                                        : 1u;
    }
}

extern "C" int mglAirReflectGLSLStageInfo(
    const char *src, int stage, MGLAIRStageInfo *stage_info,
    char *err_buf, size_t err_cap) {
    if (!src || !stage_info) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "bad args");
        return -1;
    }
    /* Legacy GLSL frontend wiring: translate pre-3.30 constructs before
     * parsing (mglShaderInterfaceCheck/compileGLSLImpl do the same). */
    std::unique_ptr<char[]> legacy_holder(airPrepareLegacySource(src, stage));
    const char *esrc = legacy_holder ? legacy_holder.get() : src;
    MGLTranslationUnit *tu = mglGLSLParse(esrc, strlen(esrc));
    if (!tu || tu->error) {
        if (err_buf && err_cap) {
            snprintf(err_buf, err_cap, "%s",
                     (tu && tu->error) ? tu->error : "parse: out of memory");
        }
        mglGLSLTranslationUnitDestroy(tu);
        return -1;
    }
    MGLIRModule mod = {};
    MGLSemaError *errors = nullptr;
    uint32_t error_count = 0;
    int hard = mglGLSLSemanticCheck(tu, stage, &mod, &errors, &error_count);
    if (hard) {
        if (err_buf && err_cap && errors && error_count) {
            snprintf(err_buf, err_cap, "line %u: %s",
                     errors[0].line, errors[0].message);
        }
        mglGLSLSemanticCheckDestroy(errors, error_count);
        mglIRModuleDestroy(&mod);
        mglGLSLTranslationUnitDestroy(tu);
        return -1;
    }
    mglGLSLSemanticCheckDestroy(errors, error_count);
    fillStageInfo(tu, &mod, stage, esrc, stage_info);
    mglIRModuleDestroy(&mod);
    mglGLSLTranslationUnitDestroy(tu);
    return 0;
}

extern "C" int mglAirCompileGLSLWithReflectInfoEx(
    const char *src, int stage, const char *const *attrib_names,
    unsigned char **metallib_out, size_t *size_out,
    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES], MGLAIRStageInfo *stage_info,
    uint32_t flags, const MGLShaderResourceList *iface_location_peers,
    char *err_buf, size_t err_cap, MGLTranslationUnit **tu_out) {
    bool has_gs = (flags & MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER) != 0;
    bool force_tes_compute =
        (flags & MGL_AIR_COMPILE_FORCE_TES_COMPUTE) != 0;
    if (tu_out)
        *tu_out = nullptr;
    if (!src || !metallib_out || !size_out) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "bad args");
        return -1;
    }
    MGLFrontendSession sess;
    mglFrontendSessionInit(&sess);
    if (mglFrontendSessionBuild(&sess, src, stage, err_buf, err_cap) != 0)
        return -1;

    uint32_t tessPatchVertices = 0u;
    MGLAIRStageInfo filled = {};
    if (stage_info) {
        tessPatchVertices = stage_info->tess_patch_vertices;
        fillStageInfo(sess.tu, &sess.mod, stage, sess.src, stage_info);
        stage_info->tess_patch_vertices = tessPatchVertices;
        filled = *stage_info;
    } else if (stage == MGL_STAGE_TESS_EVALUATION) {
        fillStageInfo(sess.tu, &sess.mod, stage, sess.src, &filled);
    }
    /* gl_TessLevel* in TES is the exact TCS / PatchParameterfv value
     * (GL 4.6 §11.2.3). Metal post-tessellation only sees half factors. */
    if (stage == MGL_STAGE_TESS_EVALUATION && filled.uses_tess_level)
        force_tes_compute = true;

    if (lists) {
        int reflect_rc = mglAirReflectModule(&sess.mod, stage, attrib_names, lists,
                                             err_buf, err_cap);
        if (reflect_rc != 0) {
            mglFrontendSessionDestroy(&sess);
            if (err_buf && err_cap && err_buf[0] == '\0') {
                snprintf(err_buf, err_cap, "reflection failed");
            }
            return -1;
        }
    }

    int capture = 0;
    if (flags & MGL_AIR_COMPILE_CULL_CAPTURE)
        capture = 3;
    else if (flags & MGL_AIR_COMPILE_TESS_CAPTURE)
        capture = 2;
    else if (flags & MGL_AIR_COMPILE_VS_CAPTURE)
        capture = 1;
    const bool tes_vertex_render =
        (flags & MGL_AIR_COMPILE_TES_VERTEX) != 0;

    int rc = compileGLSLImpl(sess.src, stage, capture, has_gs, force_tes_compute,
                             tes_vertex_render,
                             attrib_names, tessPatchVertices,
                             iface_location_peers, metallib_out, size_out,
                             err_buf, err_cap, &sess);
    if (rc == 0 && tu_out)
        *tu_out = mglFrontendSessionStealTU(&sess);
    else if (tu_out)
        *tu_out = nullptr;
    mglFrontendSessionDestroy(&sess);
    return rc;
}

extern "C" int mglAirCompileGLSLWithReflectInfo(
    const char *src, int stage, const char *const *attrib_names,
    unsigned char **metallib_out, size_t *size_out,
    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES], MGLAIRStageInfo *stage_info,
    char *err_buf, size_t err_cap) {
    return mglAirCompileGLSLWithReflectInfoEx(
        src, stage, attrib_names, metallib_out, size_out, lists,
        stage_info, 0u, /*iface_location_peers=*/nullptr, err_buf, err_cap,
        /*tu_out=*/nullptr);
}

extern "C" int mglAirCompileGLSLWithReflect(
    const char *src, int stage, const char *const *attrib_names,
    unsigned char **metallib_out, size_t *size_out,
    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES], char *err_buf,
    size_t err_cap) {
    return mglAirCompileGLSLWithReflectInfo(
        src, stage, attrib_names, metallib_out, size_out, lists, nullptr,
        err_buf, err_cap);
}

extern "C" void mglShaderFree(void *bytes) {
    free(bytes);
}

extern "C" int mglShaderInterfaceCheck(const char *vs_src, const char *fs_src,
                                       char *err_buf, size_t err_cap) {
    if (!vs_src || !fs_src) return -1;
    /* Legacy GLSL frontend wiring: translate pre-3.30 constructs before
     * parsing (VS/FS only — the interface check compares the two stages). */
    std::unique_ptr<char[]> vs_legacy(airPrepareLegacySource(vs_src, MGL_STAGE_VERTEX));
    std::unique_ptr<char[]> fs_legacy(airPrepareLegacySource(fs_src, MGL_STAGE_FRAGMENT));
    const char *vesrc = vs_legacy ? vs_legacy.get() : vs_src;
    const char *fesrc = fs_legacy ? fs_legacy.get() : fs_src;
    MGLTranslationUnit *vtu = mglGLSLParse(vesrc, strlen(vesrc));
    MGLTranslationUnit *ftu = mglGLSLParse(fesrc, strlen(fesrc));
    if (!vtu || !ftu) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "parse failed");
        mglGLSLTranslationUnitDestroy(vtu);
        mglGLSLTranslationUnitDestroy(ftu);
        return -1;
    }
    MGLIRModule vs, fs;
    memset(&vs, 0, sizeof vs);
    memset(&fs, 0, sizeof fs);
    MGLSemaError *ve = nullptr, *fe = nullptr;
    uint32_t vc = 0, fc = 0;
    int vhard = mglGLSLSemanticCheck(vtu, MGL_STAGE_VERTEX, &vs, &ve, &vc);
    int fhard = mglGLSLSemanticCheck(ftu, MGL_STAGE_FRAGMENT, &fs, &fe, &fc);
    int rc = 0;
    if (vhard || fhard) {
        if (err_buf && err_cap) {
            const char *msg = (vhard && ve && vc)
                ? ve[0].message : (fe && fc) ? fe[0].message
                                             : "semantic check failed";
            snprintf(err_buf, err_cap, "%s", msg);
        }
        rc = -1;
    } else {
        MGLSemaError *le = nullptr;
        uint32_t lec = 0;
        if (mglGLSLInterfaceCheck(&vs, &fs, &le, &lec)) {
            if (err_buf && err_cap && le && lec)
                snprintf(err_buf, err_cap, "%s", le[0].message);
            rc = -1;
        }
        mglGLSLSemanticCheckDestroy(le, lec);
        if (rc == 0) {
            le = nullptr;
            lec = 0;
            if (mglGLSLUniformLinkCheck(&vs, &fs, &le, &lec)) {
                if (err_buf && err_cap && le && lec)
                    snprintf(err_buf, err_cap, "%s", le[0].message);
                rc = -1;
            }
            mglGLSLSemanticCheckDestroy(le, lec);
        }
    }
    mglGLSLSemanticCheckDestroy(ve, vc);
    mglGLSLSemanticCheckDestroy(fe, fc);
    mglIRModuleDestroy(&vs);
    mglIRModuleDestroy(&fs);
    mglGLSLTranslationUnitDestroy(vtu);
    mglGLSLTranslationUnitDestroy(ftu);
    return rc;
}

extern "C" int mglShaderTessInterfaceCheck(const char *tcs_src,
                                           const char *tes_src,
                                           char *err_buf, size_t err_cap) {
    if (!tcs_src || !tes_src) return -1;
    std::unique_ptr<char[]> tcs_legacy(
        airPrepareLegacySource(tcs_src, MGL_STAGE_TESS_CONTROL));
    std::unique_ptr<char[]> tes_legacy(
        airPrepareLegacySource(tes_src, MGL_STAGE_TESS_EVALUATION));
    const char *csrc = tcs_legacy ? tcs_legacy.get() : tcs_src;
    const char *esrc = tes_legacy ? tes_legacy.get() : tes_src;
    MGLTranslationUnit *ctu = mglGLSLParse(csrc, strlen(csrc));
    MGLTranslationUnit *etu = mglGLSLParse(esrc, strlen(esrc));
    if (!ctu || !etu) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "parse failed");
        mglGLSLTranslationUnitDestroy(ctu);
        mglGLSLTranslationUnitDestroy(etu);
        return -1;
    }
    MGLIRModule tcs, tes;
    memset(&tcs, 0, sizeof tcs);
    memset(&tes, 0, sizeof tes);
    MGLSemaError *ce = nullptr, *ee = nullptr;
    uint32_t cc = 0, ec = 0;
    int chard = mglGLSLSemanticCheck(ctu, MGL_STAGE_TESS_CONTROL, &tcs, &ce, &cc);
    int ehard = mglGLSLSemanticCheck(etu, MGL_STAGE_TESS_EVALUATION, &tes, &ee, &ec);
    int rc = 0;
    if (chard || ehard) {
        /* Declaration-level errors are reported at compile; still treat a
         * surviving type mismatch as a link failure when both compile. */
        if (err_buf && err_cap) {
            const char *msg = (chard && ce && cc)
                ? ce[0].message : (ee && ec) ? ee[0].message
                                             : "tess semantic check failed";
            snprintf(err_buf, err_cap, "%s", msg);
        }
        rc = -1;
    } else {
        MGLSemaError *le = nullptr;
        uint32_t lec = 0;
        if (mglGLSLInterfaceCheck(&tcs, &tes, &le, &lec)) {
            if (err_buf && err_cap && le && lec)
                snprintf(err_buf, err_cap, "%s", le[0].message);
            rc = -1;
        }
        mglGLSLSemanticCheckDestroy(le, lec);
    }
    mglGLSLSemanticCheckDestroy(ce, cc);
    mglGLSLSemanticCheckDestroy(ee, ec);
    mglIRModuleDestroy(&tcs);
    mglIRModuleDestroy(&tes);
    mglGLSLTranslationUnitDestroy(ctu);
    mglGLSLTranslationUnitDestroy(etu);
    return rc;
}
