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
#include "llvm/Support/raw_ostream.h"

#include "mgl_glsl_ast.h"
#include "mgl_glsl_parser.h"
#include "mgl_glsl_sema.h"
#include "mgl_ir.h"
#include "mgl_metallib_writer.h"
#include "mgl_air_reflect.h"
#include "mgl_shader_abi.h"

namespace {

/* Lightweight type model for codegen.  Mirrors the MGLIR scalar/vector/
 * matrix shapes; the LLVM types are derived on demand. */
struct MType {
    MGLIRScalar scalar = MGLIR_SCALAR_FLOAT;
    uint32_t vec = 0;        /* vector width, 0 = scalar */
    uint32_t cols = 0;       /* matrix columns, 0 = not a matrix */
    uint32_t rows = 0;       /* matrix rows */

    bool isMatrix() const { return cols != 0; }
    uint32_t matrixCols() const { return isMatrix() ? cols : 1; }
    uint32_t lanes() const { return isMatrix() ? rows : (vec ? vec : 1); }
};

struct Uniform {
    std::string name;
    MType type;
    uint32_t offset;         /* byte offset in the implicit buffer */
    uint32_t size;           /* std140 byte size */
};

struct VarSym {
    std::string name;
    MType type;
    enum Kind { ATTR, VARYING, OUTPUT, BUFFER, SSBO, UBO, TEXTURE, LOCAL } kind = LOCAL;
    uint32_t bufferOffset = 0;
    bool written = false;
};

struct LoopCtx {
    llvm::BasicBlock *condBB = nullptr;  /* do-while continue target */
    llvm::BasicBlock *endBB = nullptr;   /* break target */
    llvm::BasicBlock *incrBB = nullptr;  /* merge block; while/for continue target */
    std::map<std::string, llvm::PHINode *> phis;
    std::vector<std::pair<llvm::BasicBlock *,
                          std::map<std::string, llvm::Value *>>> contSnaps;
};

/* Shared by loops and switch: break jumps to endBB carrying a snapshot
 * of the live values; the owner merges them into phis at endBB. */
struct BreakCtx {
    llvm::BasicBlock *endBB;
    std::vector<std::pair<llvm::BasicBlock *,
                          std::map<std::string, llvm::Value *>>> snaps;
};

struct Codegen {
    llvm::LLVMContext *ctx;
    llvm::IRBuilder<> *b;
    llvm::Function *fn;
    llvm::Module *mod = nullptr;       /* current LLVM module */
    bool isVS = false;
    bool isCompute = false;
    llvm::Value *bufferPtr = nullptr;    /* i8 addrspace(1)* */
    llvm::Value *threadPos = nullptr;    /* compute: <3 x i32> grid position */
    llvm::Value *captureBuf = nullptr;   /* capture variant: output buffer */
    llvm::Value *vertexId = nullptr;     /* capture variant: vertex_id */
    std::map<std::string, llvm::Value *> ssboPtrs;  /* SSBO instance -> buffer */
    std::map<std::string, llvm::Value *> uboPtrs;   /* uniform block -> buffer */
    std::map<std::string, llvm::Value *> texValues;  /* sampler name -> texture */
    std::map<std::string, llvm::Value *> smpValues;  /* sampler name -> sampler */
    std::map<std::string, uint32_t> bufferOffsets;  /* uniform name -> byte offset */
    std::map<std::string, llvm::Value *> lvalues;   /* register values */
    std::vector<VarSym *> varyings;      /* vertex out / fragment in, decl order */
    VarSym *fragOutput = nullptr;        /* fragment out vec4 */
    VarSym position;                     /* gl_Position */
    llvm::Type *retTy = nullptr;         /* stage return type */
    std::vector<llvm::Type *> retElems;  /* VS struct fields (incl. position) */
    std::vector<VarSym> *auxSyms = nullptr;  /* all stage symbols (frag output) */
    int err = 0;
    std::string errmsg;                  /* specific diagnostic when set */
    std::vector<LoopCtx *> loopStack;    /* innermost loop is last */
    std::vector<BreakCtx *> breakStack;  /* innermost loop/switch is last */
};

/* ---- type helpers ---------------------------------------------------- */

bool scalarIsFloat(MGLIRScalar s) {
    return s == MGLIR_SCALAR_FLOAT || s == MGLIR_SCALAR_DOUBLE ||
           s == MGLIR_SCALAR_HALF;
}

llvm::Type *llvmScalar(MGLIRScalar s, llvm::LLVMContext &ctx) {
    switch (s) {
    case MGLIR_SCALAR_BOOL: return llvm::Type::getInt1Ty(ctx);
    case MGLIR_SCALAR_INT:  return llvm::Type::getInt32Ty(ctx);
    case MGLIR_SCALAR_UINT: return llvm::Type::getInt32Ty(ctx);
    default:                return llvm::Type::getFloatTy(ctx);
    }
}

llvm::Type *llvmType(const MType &t, llvm::LLVMContext &ctx) {
    llvm::Type *s = llvmScalar(t.scalar, ctx);
    if (t.isMatrix())
        return llvm::ArrayType::get(llvm::FixedVectorType::get(s, t.rows), t.cols);
    if (t.vec)
        return llvm::FixedVectorType::get(s, t.vec);
    return s;
}

/* Implicit GLSL numeric conversion (sema allows any non-void scalar base
 * to convert to any other, GLSL 4.60 4.1.10).  Idempotent; works on
 * scalars and vectors of matching width. */
llvm::Value *coerceScalar(Codegen &cg, llvm::Value *v, MGLIRScalar want) {
    llvm::Type *cur = v->getType();
    if (!cur->isIntOrIntVectorTy() && !cur->isFPOrFPVectorTy())
        return v;  /* arrays / matrices / aggregates: no scalar cast */
    bool wantFP = scalarIsFloat(want);
    bool curFP = cur->isFPOrFPVectorTy();
    if (curFP == wantFP && want != MGLIR_SCALAR_BOOL &&
        cur->getScalarSizeInBits() == (want == MGLIR_SCALAR_BOOL ? 1 : 32))
        return v;
    llvm::LLVMContext &ctx = *cg.ctx;
    auto vt = [&](llvm::Type *elt) -> llvm::Type * {
        if (auto *fv = llvm::dyn_cast<llvm::FixedVectorType>(cur))
            return llvm::FixedVectorType::get(elt,
                fv->getElementCount().getFixedValue());
        return elt;
    };
    if (wantFP) {
        if (cur->getScalarSizeInBits() == 1)
            return cg.b->CreateUIToFP(v, vt(llvm::Type::getFloatTy(ctx)));
        return cg.b->CreateSIToFP(v, vt(llvm::Type::getFloatTy(ctx)));
    }
    if (curFP)
        return cg.b->CreateFPToSI(v, vt(llvm::Type::getInt32Ty(ctx)));
    if (want == MGLIR_SCALAR_BOOL)
        return cg.b->CreateICmpNE(v, llvm::Constant::getNullValue(cur));
    /* int: widen bool to i32, otherwise identity. */
    if (cur->getScalarSizeInBits() == 1)
        return cg.b->CreateZExt(v, vt(llvm::Type::getInt32Ty(ctx)));
    return v;
}

/* GLSL type name used in air.* metadata (MSL naming). */
std::string mslTypeName(const MType &t) {
    if (t.isMatrix()) {
        char buf[32];
        snprintf(buf, sizeof buf, "float%ux%u", t.cols, t.rows);
        return buf;
    }
    switch (t.scalar) {
    case MGLIR_SCALAR_INT:   return t.vec ? "int" + std::to_string(t.vec) : "int";
    case MGLIR_SCALAR_UINT:  return t.vec ? "uint" + std::to_string(t.vec) : "uint";
    default: break;
    }
    if (!t.vec) return "float";
    switch (t.vec) {
    case 2: return "float2";
    case 3: return "float3";
    default: return "float4";
    }
}

/* ---- resource collection ---------------------------------------------- */

MType typeFromIR(const MGLIRType *t) {
    MType r;
    r.scalar = t->scalar;
    switch (t->kind) {
    case MGLIR_TYPE_VECTOR: r.vec = t->cols; break;
    case MGLIR_TYPE_MATRIX: r.cols = t->cols; r.rows = t->rows; break;
    default: break;
    }
    return r;
}

/* One implicit buffer holds every plain uniform, packed with std140
 * alignment in declaration order.  Returns 0 on success. */
int collectUniforms(const MGLIRModule *mod, std::vector<Uniform> *out,
                    uint32_t *bufferSize, char *err, size_t errCap) {
    uint32_t off = 0;
    for (uint32_t i = 0; i < mod->symbol_count; i++) {
        MGLIRSymbol *s = mod->symbols[i];
        if (s->is_function || !(s->qualifiers & MGL_AST_Q_UNIFORM))
            continue;
        if (s->type->kind == MGLIR_TYPE_SAMPLER ||
            s->type->kind == MGLIR_TYPE_IMAGE) {
            continue;   /* texture/sampler params are separate AIR args */
        }
        /* Uniform blocks (struct types) and their anonymous-block members
         * are independent device buffers, not part of the plain uniform
         * pack. */
        if (s->block_name ||
            (s->type->kind == MGLIR_TYPE_STRUCT && s->type->member_count > 0))
            continue;
        uint32_t size = 0;
        if (mglIRComputeLayout(s->type, MGLIR_LAYOUT_STD140, &size) != 0) {
            snprintf(err, errCap, "layout failed for uniform %s", s->name);
            return -1;
        }
        off = (off + s->type->layout.alignment - 1) &
              ~(s->type->layout.alignment - 1);
        Uniform u;
        u.name = s->name;
        u.type = typeFromIR(s->type);
        u.offset = off;
        u.size = size;
        out->push_back(u);
        off += size;
    }
    *bufferSize = off;
    return 0;
}

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

MType exprType(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
               const std::map<std::string, MType> &locals);

/* Broadcast a scalar to the lane type of a vector type. */
llvm::Value *broadcastTo(Codegen &cg, llvm::Value *v, llvm::Type *vecTy);

/* Constant-fold a numeric binary op when both operands are constants of
 * the same type; returns null when folding is not possible.  Mirrors the
 * runtime signedness/comparison semantics of emitNumericBinOp. */
llvm::Value *tryFoldConst(Codegen &cg, uint32_t op, llvm::Value *l,
                          llvm::Value *r, bool uns) {
    auto *lc = llvm::dyn_cast<llvm::Constant>(l);
    auto *rc = llvm::dyn_cast<llvm::Constant>(r);
    if (!lc || !rc) return nullptr;
    if (l->getType() != r->getType()) return nullptr;
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
        return llvm::ConstantExpr::getCompare(fp ? llvm::CmpInst::FCMP_OEQ
                                                 : llvm::CmpInst::ICMP_EQ,
                                              lc, rc);
    case MGL_OP_NE:
        return llvm::ConstantExpr::getCompare(fp ? llvm::CmpInst::FCMP_ONE
                                                 : llvm::CmpInst::ICMP_NE,
                                              lc, rc);
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
    return fp ? cg.b->CreateFCmp(pred, l, r) : cg.b->CreateICmp(pred, l, r);
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

/* Matrix builtins (declared early; defined after emitExpr). */
llvm::Value *emitExpr(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
                      const std::map<std::string, MType> &locals);
static llvm::Value *emitMatrixBuiltin(Codegen &cg, const MGLExpr *e,
                                      const char *name, const MGLIRModule *mod,
                                      const std::map<std::string, MType> &locals);

/* ---- Matrix builtins -------------------------------------------------- */

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
static llvm::Value *emitMatrixBuiltin(Codegen &cg, const MGLExpr *e,
                                      const char *name, const MGLIRModule *mod,
                                      const std::map<std::string, MType> &locals) {
    bool isT = !strcmp(name, "transpose");
    bool isC = !strcmp(name, "matrixCompMult");
    bool isO = !strcmp(name, "outerProduct");
    bool isD = !strcmp(name, "determinant");
    bool isI = !strcmp(name, "inverse");
    if (!isT && !isC && !isO && !isD && !isI) return nullptr;

    llvm::Value *a = emitExpr(cg, e->u.call.args[0], mod, locals);
    if (!a) return nullptr;
    llvm::Value *b = nullptr;
    if (e->u.call.arg_count == 2) {
        b = emitExpr(cg, e->u.call.args[1], mod, locals);
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

/* Element-wise float intrinsic (scalar or vector operand). */
llvm::Value *callFloatIntrinsic(Codegen &cg, llvm::Intrinsic::ID id,
                                llvm::Value *v) {
    return cg.b->CreateIntrinsic(id, {v->getType()}, {v});
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
                             llvm::Value *r) {
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
                llvm::Value *d = dotProduct(cg, l, col);
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
    return nullptr;
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

/* Math builtins beyond the first wave: trigonometry, exponentials,
 * rounding, fract/sign/mod/step/smoothstep, min/max (float and integer),
 * geometric reflect/refract/faceforward, radians/degrees.  Returns NULL
 * when `name` is not a math builtin handled here. */
static llvm::Value *emitMathBuiltin(Codegen &cg, const MGLExpr *e,
                                    const char *name, const MGLIRModule *mod,
                                    const std::map<std::string, MType> &locals);

llvm::Value *emitExpr(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
                      const std::map<std::string, MType> &locals);

static llvm::Value *emitMatrixBuiltin(Codegen &cg, const MGLExpr *e,
                                      const char *name, const MGLIRModule *mod,
                                      const std::map<std::string, MType> &locals);

/* Dynamic read of obj[idx]: matrix -> column (select chain over the
 * columns, since extractvalue needs a constant index), vector ->
 * component.  `idx` must be an integer value. */
static llvm::Value *emitIndexValue(Codegen &cg, llvm::Value *obj,
                                   const MType &bt, llvm::Value *idx) {
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

/* Read an index chain (x[i][j]) from the root value without re-emitting
 * the object expression. */
static llvm::Value *readIndexChain(Codegen &cg, const MGLExpr *e,
                                   llvm::Value *rootVal,
                                   const MGLIRModule *mod,
                                   const std::map<std::string, MType> &locals) {
    if (e->kind == MGL_EXPR_VAR_REF) return rootVal;
    if (e->kind == MGL_EXPR_MEMBER) {
        llvm::Value *obj = readIndexChain(cg, e->u.member.object, rootVal, mod,
                                          locals);
        if (!obj) return nullptr;
        std::vector<uint32_t> idx;
        if (!swizzleIndices(e->u.member.field, &idx)) {
            cg.err = 1;
            cg.errmsg = "codegen: invalid swizzle";
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
        std::vector<uint32_t> idx;
        if (!swizzleIndices(lhs->u.member.field, &idx)) {
            cg.err = 1;
            cg.errmsg = "codegen: invalid swizzle";
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
    } else if (objE->kind == MGL_EXPR_INDEX) {
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

/* Byte address of a member/index chain rooted at an SSBO instance; the
 * member type is returned in *outTy. */
llvm::Value *ssboAddress(Codegen &cg, const MGLExpr *e,
                         const MGLIRSymbol *sb, const MGLIRModule *mod,
                         const std::map<std::string, MType> &locals,
                         const MGLIRType **outTy) {
    const MGLIRType *ty = sb->type;
    std::vector<const MGLExpr *> path;
    const MGLExpr *cur = e;
    while (cur->kind == MGL_EXPR_MEMBER || cur->kind == MGL_EXPR_INDEX) {
        path.push_back(cur);
        cur = cur->kind == MGL_EXPR_INDEX ? cur->u.index.object
                                          : cur->u.member.object;
    }
    std::reverse(path.begin(), path.end());
    llvm::Value *base = cg.ssboPtrs[sb->name];
    uint32_t off = 0;
    for (const MGLExpr *pe : path) {
        if (pe->kind == MGL_EXPR_MEMBER) {
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

llvm::Value *emitSSBORead(Codegen &cg, const MGLExpr *e,
                          const MGLIRSymbol *sb, const MGLIRModule *mod,
                          const std::map<std::string, MType> &locals) {
    const MGLIRType *ty = nullptr;
    llvm::Value *p = ssboAddress(cg, e, sb, mod, locals, &ty);
    if (!p) return nullptr;
    llvm::Type *lt = llvmType(typeFromIR(ty), *cg.ctx);
    llvm::Align align(16);
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(lt)) {
        uint64_t w = vt->getElementCount().getFixedValue();
        if (w == 1) align = llvm::Align(4);
        else if (w == 2) align = llvm::Align(8);
    } else if (lt->isFloatTy() || lt->isIntegerTy(32)) {
        align = llvm::Align(4);
    }
    p = cg.b->CreateBitCast(p, lt->getPointerTo(1));
    return cg.b->CreateAlignedLoad(lt, p, align);
}

void emitSSBOWrite(Codegen &cg, const MGLExpr *e, const MGLIRSymbol *sb,
                   const MGLIRModule *mod,
                   const std::map<std::string, MType> &locals,
                   llvm::Value *v) {
    const MGLIRType *ty = nullptr;
    llvm::Value *p = ssboAddress(cg, e, sb, mod, locals, &ty);
    if (!p) return;
    v = coerceScalar(cg, v, typeFromIR(ty).scalar);
    llvm::Type *lt = llvmType(typeFromIR(ty), *cg.ctx);
    llvm::Align align(16);
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(lt)) {
        uint64_t w = vt->getElementCount().getFixedValue();
        if (w == 1) align = llvm::Align(4);
        else if (w == 2) align = llvm::Align(8);
    } else if (lt->isFloatTy() || lt->isIntegerTy(32)) {
        align = llvm::Align(4);
    }
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

llvm::Value *varValue(Codegen &cg, const VarSym &v, const MGLIRModule *mod) {
    if (v.kind == VarSym::BUFFER) {
        /* Anonymous-block member: read from the block's device buffer. */
        const MGLIRSymbol *bs = findSymbol(mod, v.name.c_str());
        if (bs && bs->block_name) {
            llvm::Value *base = cg.uboPtrs.count(bs->block_name)
                                    ? cg.uboPtrs[bs->block_name]
                                    : nullptr;
            if (base) {
                const MGLIRSymbol *blk = findSymbol(mod, bs->block_name);
                uint32_t moff = (blk && blk->type->member_offsets)
                                    ? blk->type->member_offsets[bs->block_member_index]
                                    : 0;
                llvm::Type *t = llvmType(v.type, *cg.ctx);
                llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), base,
                                                 cg.b->getInt64(moff));
                llvm::Align align(16);
                if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
                    uint64_t w = vt->getElementCount().getFixedValue();
                    if (w == 1) align = llvm::Align(4);
                    else if (w == 2) align = llvm::Align(8);
                } else if (t->isFloatTy() || t->isIntegerTy(32)) {
                    align = llvm::Align(4);
                }
                p = cg.b->CreateBitCast(p, t->getPointerTo(1));
                return cg.b->CreateAlignedLoad(t, p, align);
            }
        }
        /* Uniform: single value read. */
        llvm::Type *t = llvmType(v.type, *cg.ctx);
        uint32_t off = cg.bufferOffsets.count(v.name) ? cg.bufferOffsets[v.name] : 0;
        if (v.type.isMatrix())
            return emitMatrixUniform(cg, Uniform{v.name, v.type, off, 0});
        return bufferLoad(cg, off, t);
    }
    auto it = cg.lvalues.find(v.name);
    if (it != cg.lvalues.end())
        return it->second;
    /* Unwritten out/attribute: undef. */
    return llvm::UndefValue::get(llvmType(v.type, *cg.ctx));
}

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
        if (strcmp(e->u.var_ref.name, "gl_VertexID") == 0) {
            if (!cg.vertexId) {
                cg.err = 1;
                cg.errmsg = "codegen: gl_VertexID requires the capture "
                            "variant";
                return nullptr;
            }
            return cg.vertexId;
        }
        auto lit = locals.find(e->u.var_ref.name);
        if (lit != locals.end())
            return varValue(cg, VarSym{e->u.var_ref.name, lit->second, VarSym::LOCAL},
                            mod);
        const MGLIRSymbol *s = findSymbol(mod, e->u.var_ref.name);
        if (!s) { cg.err = 1; return nullptr; }
        VarSym v;
        v.name = s->name;
        v.type = typeFromIR(s->type);
        if (s->qualifiers & MGL_AST_Q_UNIFORM) {
            v.kind = VarSym::BUFFER;
        } else if ((s->qualifiers & MGL_AST_Q_IN) && cg.isVS) {
            v.kind = VarSym::ATTR;
        } else {
            v.kind = VarSym::VARYING;
        }
        return varValue(cg, v, mod);
    }
    case MGL_EXPR_MEMBER: {
        if (const MGLIRSymbol *sb = ssboRootSym(e, mod))
            return emitSSBORead(cg, e, sb, mod, locals);
        /* Swizzle only in M1. */
        std::vector<uint32_t> idx;
        if (!swizzleIndices(e->u.member.field, &idx)) { cg.err = 1; return nullptr; }
        llvm::Value *obj = emitExpr(cg, e->u.member.object, mod, locals);
        if (!obj) return nullptr;
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
        if (const MGLIRSymbol *sb = ssboRootSym(e, mod))
            return emitSSBORead(cg, e, sb, mod, locals);
        const MGLExpr *idxE = e->u.index.index;
        bool constIdx = idxE->kind == MGL_EXPR_LITERAL &&
                        (idxE->u.literal.base == MGL_AST_TYPE_INT ||
                         idxE->u.literal.base == MGL_AST_TYPE_UINT);
        MType bt = exprType(cg, e->u.index.object, mod, locals);
        llvm::Value *obj = emitExpr(cg, e->u.index.object, mod, locals);
        if (!obj) return nullptr;
        if (constIdx) {
            uint32_t i = (uint32_t)idxE->u.literal.value;
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
            uint32_t slot = 0;
            for (uint32_t a = 0; a < e->u.call.arg_count; a++) {
                llvm::Value *arg = emitExpr(cg, e->u.call.args[a], mod, locals);
                if (!arg) return nullptr;
                if (!arg->getType()->isVectorTy()) {
                    /* Single scalar argument broadcasts (GLSL 4.60 5.4.2);
                     * otherwise one component per scalar. */
                    arg = coerceScalar(cg, arg, velt);
                    if (e->u.call.arg_count == 1) {
                        for (uint32_t lane = 0; lane < vlanes; lane++)
                            res = cg.b->CreateInsertElement(res, arg,
                                llvm::ConstantInt::get(
                                    llvm::Type::getInt32Ty(*cg.ctx), lane));
                        return res;
                    }
                    res = cg.b->CreateInsertElement(res, arg,
                        llvm::ConstantInt::get(llvm::Type::getInt32Ty(*cg.ctx),
                                               slot++));
                } else {
                    arg = coerceScalar(cg, arg, velt);
                    llvm::FixedVectorType *argTy =
                        llvm::cast<llvm::FixedVectorType>(arg->getType());
                    uint32_t argLanes = (uint32_t)argTy->getElementCount()
                                                    .getFixedValue();
                    for (uint32_t lane = 0;
                         lane < argLanes && slot < vlanes; lane++, slot++) {
                        llvm::Value *x = cg.b->CreateExtractElement(arg,
                            llvm::ConstantInt::get(
                                llvm::Type::getInt32Ty(*cg.ctx), lane));
                        res = cg.b->CreateInsertElement(res, x,
                            llvm::ConstantInt::get(
                                llvm::Type::getInt32Ty(*cg.ctx), slot));
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
                /* matN(f): diagonal scale. */
                llvm::Value *s = emitExpr(cg, e->u.call.args[0], mod, locals);
                if (!s) return nullptr;
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
            llvm::Value *mb = emitMatrixBuiltin(cg, e, name, mod, locals);
            if (mb) return mb;
        }
        {
            llvm::Value *mb = emitMathBuiltin(cg, e, name, mod, locals);
            if (mb) return mb;
        }
        /* texture / textureLod / textureSize: the sampler argument maps
         * to paired AIR texture + sampler parameters. */
        if (strcmp(name, "texture") == 0 || strcmp(name, "textureLod") == 0 ||
            strcmp(name, "textureSize") == 0) {
            if (e->u.call.arg_count < 2 || e->u.call.arg_count > 3) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: '") + name +
                            "' expects 2 or 3 arguments";
                return nullptr;
            }
            const MGLExpr *sa = e->u.call.args[0];
            if (sa->kind != MGL_EXPR_VAR_REF ||
                !cg.texValues.count(sa->u.var_ref.name)) {
                cg.err = 1;
                cg.errmsg = "codegen: texture argument must be a sampler2D "
                            "variable";
                return nullptr;
            }
            llvm::Value *tex = cg.texValues[sa->u.var_ref.name];
            llvm::Value *smp = cg.smpValues[sa->u.var_ref.name];
            llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
            llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
            if (strcmp(name, "textureSize") == 0) {
                llvm::Value *lod = emitExpr(cg, e->u.call.args[1], mod,
                                            locals);
                if (!lod) return nullptr;
                lod = coerceScalar(cg, lod, MGLIR_SCALAR_INT);
                llvm::Value *w = callAirFn(cg, "air.get_width_texture_2d",
                                           i32, {tex, lod});
                llvm::Value *h = callAirFn(cg, "air.get_height_texture_2d",
                                           i32, {tex, lod});
                llvm::Type *v2i32 = llvm::FixedVectorType::get(i32, 2);
                llvm::Value *sz = llvm::UndefValue::get(v2i32);
                sz = cg.b->CreateInsertElement(sz, w, cg.b->getInt32(0));
                sz = cg.b->CreateInsertElement(sz, h, cg.b->getInt32(1));
                return sz;
            }
            llvm::Value *uv = emitExpr(cg, e->u.call.args[1], mod, locals);
            if (!uv) return nullptr;
            llvm::Value *lod = nullptr;
            bool explicitLod = false;
            if (e->u.call.arg_count == 3) {
                lod = emitExpr(cg, e->u.call.args[2], mod, locals);
                if (!lod) return nullptr;
                lod = coerceScalar(cg, lod, MGLIR_SCALAR_FLOAT);
                explicitLod = true;
            }
            llvm::Type *v2i32 = llvm::FixedVectorType::get(i32, 2);
            llvm::Type *v4f32 = llvm::FixedVectorType::get(f32, 4);
            llvm::Type *retTy =
                llvm::StructType::get(*cg.ctx, {v4f32, cg.b->getInt8Ty()});
            llvm::Value *r = callAirFn(
                cg, "air.sample_texture_2d.v4f32", retTy,
                {tex, smp, uv, cg.b->getInt1(true),
                 llvm::Constant::getNullValue(v2i32),
                 cg.b->getInt1(explicitLod),
                 lod ? lod
                     : llvm::ConstantFP::get(f32, 0.0),
                 llvm::ConstantFP::get(f32, 0.0),
                 cg.b->getInt32(0)});
            return cg.b->CreateExtractValue(r, 0);
        }
        /* atomicAdd(ssbo_lvalue, value): monotonic RMW on device memory. */
        if (strcmp(name, "atomicAdd") == 0) {
            if (e->u.call.arg_count != 2) {
                cg.err = 1;
                cg.errmsg = "codegen: atomicAdd expects 2 arguments";
                return nullptr;
            }
            const MGLIRSymbol *sb = ssboRootSym(e->u.call.args[0], mod);
            if (!sb) {
                cg.err = 1;
                cg.errmsg = "codegen: atomicAdd target must be an SSBO member";
                return nullptr;
            }
            llvm::Value *val = emitExpr(cg, e->u.call.args[1], mod, locals);
            if (!val) return nullptr;
            const MGLIRType *ty = nullptr;
            llvm::Value *p = ssboAddress(cg, e->u.call.args[0], sb, mod,
                                         locals, &ty);
            if (!p) return nullptr;
            p = cg.b->CreateBitCast(p, val->getType()->getPointerTo(1));
            llvm::Value *old =
                cg.b->CreateAtomicRMW(llvm::AtomicRMWInst::Add, p, val,
                                      llvm::MaybeAlign(),
                                      llvm::AtomicOrdering::Monotonic);
            /* GLSL 4.60 8.11: atomicAdd returns the new value. */
            return cg.b->CreateAdd(old, val);
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
                cg.err = 1;
                cg.errmsg = std::string("codegen: ++/-- requires a variable");
                return nullptr;
            }
            const char *name = e->u.unary.operand->u.var_ref.name;
            auto it = cg.lvalues.find(name);
            if (it == cg.lvalues.end()) {
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
        llvm::Value *l = emitExpr(cg, e->u.binary.lhs, mod, locals);
        llvm::Value *r = emitExpr(cg, e->u.binary.rhs, mod, locals);
        if (!l || !r) return nullptr;
        llvm::Value *mres = emitMatrixBinOp(cg, e->u.binary.op, l, r);
        if (mres) return mres;
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
        llvm::Value *v = emitExpr(cg, e->u.assign.rhs, mod, locals);
        if (!v) return nullptr;
        llvm::Value *rhsV = v;
        const MGLExpr *lhs = e->u.assign.lhs;

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
            if (!cg.lvalues.count(name)) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: unknown lvalue '") + name +
                            "'";
                return nullptr;
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
            return nv;
        }

        /* x op= y where x is a named lvalue. */
        if (lhs->kind != MGL_EXPR_VAR_REF) {
            cg.err = 1; return nullptr;
        }
        const char *name = lhs->u.var_ref.name;
        if (e->u.assign.op != MGL_OP_ASSIGN) {
            MType t;
            const MGLIRSymbol *sym = nullptr;
            auto lit = locals.find(name);
            if (lit != locals.end()) t = lit->second;
            else if (strcmp(name, "gl_Position") == 0) {
                t.scalar = MGLIR_SCALAR_FLOAT;
                t.vec = 4;
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
            llvm::Value *cur = cg.lvalues.count(name)
                ? cg.lvalues[name]
                : ((sym->qualifiers & MGL_AST_Q_UNIFORM)
                       ? bufferLoad(cg, cg.bufferOffsets[name],
                                    llvmType(t, *cg.ctx))
                       : llvm::UndefValue::get(llvmType(t, *cg.ctx)));
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
            return v;
        }
        auto lit = locals.find(name);
        if (lit != locals.end()) {
            v = coerceScalar(cg, v, lit->second.scalar);
            cg.lvalues[name] = v;
            return v;
        }
        const MGLIRSymbol *sym = findSymbol(mod, name);
        if (!sym) { cg.err = 1; return nullptr; }
        v = coerceScalar(cg, v, typeFromIR(sym->type).scalar);
        if (sym->qualifiers & MGL_AST_Q_UNIFORM) {
            bufferStore(cg, cg.bufferOffsets[name],
                        llvmType(typeFromIR(sym->type), *cg.ctx), v);
            return v;
        }
        cg.lvalues[name] = v;
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
        auto lit = locals.find(e->u.var_ref.name);
        if (lit != locals.end()) { t = lit->second; break; }
        const MGLIRSymbol *s = findSymbol(mod, e->u.var_ref.name);
        if (s) t = typeFromIR(s->type);
        break;
    }
    case MGL_EXPR_MEMBER: {
        std::vector<uint32_t> idx;
        MType base = exprType(cg, e->u.member.object, mod, locals);
        if (swizzleIndices(e->u.member.field, &idx))
            t = swizzleType(base, idx.size());
        break;
    }
    case MGL_EXPR_INDEX: {
        MType base = exprType(cg, e->u.index.object, mod, locals);
        if (base.isMatrix()) {
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
        if (strcmp(name, "float") == 0 || strcmp(name, "int") == 0 ||
            strcmp(name, "uint") == 0 || strcmp(name, "bool") == 0) {
            t.scalar = name[0] == 'f' ? MGLIR_SCALAR_FLOAT
                     : name[0] == 'u' ? MGLIR_SCALAR_UINT
                     : name[0] == 'b' ? MGLIR_SCALAR_BOOL
                                      : MGLIR_SCALAR_INT;
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
        } else if (strcmp(name, "normalize") == 0 ||
                   strcmp(name, "abs") == 0 ||
                   strcmp(name, "clamp") == 0 ||
                   strcmp(name, "mix") == 0) {
            /* genType result: width follows the first argument. */
            t.scalar = MGLIR_SCALAR_FLOAT;
            if (e->u.call.arg_count > 0)
                t.vec = exprType(cg, e->u.call.args[0], mod, locals).vec;
        } else if (strcmp(name, "length") == 0 ||
                   strcmp(name, "distance") == 0 ||
                   strcmp(name, "dot") == 0) {
            t.scalar = MGLIR_SCALAR_FLOAT;
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

/* ---- math builtins ----------------------------------------------------- */

static llvm::Value *emitMathBuiltin(Codegen &cg, const MGLExpr *e,
                                    const char *name, const MGLIRModule *mod,
                                    const std::map<std::string, MType> &locals)
{
    auto arg = [&](uint32_t i) -> llvm::Value * {
        return emitExpr(cg, e->u.call.args[i], mod, locals);
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
                                  dotProduct(cg, a0, a0));
    }
    if (strcmp(name, "distance") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        if (!a0 || !a1) return nullptr;
        llvm::Value *d = cg.b->CreateFSub(a0, a1);
        return callFloatIntrinsic(cg, llvm::Intrinsic::sqrt,
                                  dotProduct(cg, d, d));
    }
    if (strcmp(name, "normalize") == 0) {
        if (!need(1)) return nullptr;
        a0 = farg(0);
        if (!a0) return nullptr;
        llvm::Value *len = callFloatIntrinsic(cg, llvm::Intrinsic::sqrt,
                                              dotProduct(cg, a0, a0));
        return cg.b->CreateFDiv(a0, broadcastTo(cg, len, a0->getType()));
    }
    if (strcmp(name, "dot") == 0) {
        if (!need(2)) return nullptr;
        a0 = farg(0);
        a1 = farg(1);
        if (!a0 || !a1) return nullptr;
        return dotProduct(cg, a0, a1);
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
        if (t->isVectorTy()) a1 = broadcastTo(cg, a1, t);
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
        if (t->isVectorTy()) a0 = broadcastTo(cg, a0, t);
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
            a0 = broadcastTo(cg, a0, t);
            a1 = broadcastTo(cg, a1, t);
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
            if (t->isVectorTy()) a1 = broadcastTo(cg, a1, t);
            llvm::CmpInst::Predicate p = strcmp(name, "min") == 0
                ? llvm::CmpInst::ICMP_SLT : llvm::CmpInst::ICMP_SGT;
            return cg.b->CreateSelect(cg.b->CreateICmp(p, a0, a1), a0, a1);
        }
        a1 = coerceScalar(cg, a1, MGLIR_SCALAR_FLOAT);
        if (t->isVectorTy()) a1 = broadcastTo(cg, a1, t);
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
                    a1 = broadcastTo(cg, a1, t);
                    a2 = broadcastTo(cg, a2, t);
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
                a1 = broadcastTo(cg, a1, t);
                a2 = broadcastTo(cg, a2, t);
            }
            llvm::Value *mx = cg.b->CreateIntrinsic(
                llvm::Intrinsic::maxnum, {t}, {a0, a1});
            return cg.b->CreateIntrinsic(llvm::Intrinsic::minnum, {t},
                                         {mx, a2});
        }
        /* mix(x, y, a) = fma(a, y, x * (1 - a)) */
        a1 = coerceScalar(cg, a1, MGLIR_SCALAR_FLOAT);
        a2 = coerceScalar(cg, a2, MGLIR_SCALAR_FLOAT);
        if (t->isVectorTy()) a2 = broadcastTo(cg, a2, t);
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
        llvm::Value *d = dotProduct(cg, a1, a0);
        d = cg.b->CreateFMul(d, fpConstOf(cg, d->getType(), 2.0));
        d = broadcastTo(cg, d, a0->getType());
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
        llvm::Value *d = dotProduct(cg, a1, a0);  /* scalar float */
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
            cg.b->CreateFMul(broadcastTo(cg, sc, t), a1));
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
        llvm::Value *d = dotProduct(cg, a2, a1);
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
                    x = callAirFn(cg, airfn, f32, {x, y});
                } else {
                    x = callAirFn(cg, airfn, f32, {x});
                }
                r = cg.b->CreateInsertElement(r, x, cI(i));
            }
            return r;
        }
        if (want == 2) return callAirFn(cg, airfn, f32, {a0, a1v});
        return callAirFn(cg, airfn, f32, {a0});
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
        if (pack) return callAirFn(cg, airfn, i32, {av});
        return callAirFn(cg, airfn, llvm::FixedVectorType::get(f32, 2), {av});
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
            llvm::Value *h = callAirFn(cg, "air.convert.f.v2f16.f.v2f32",
                                       v2f16, {av});
            return cg.b->CreateBitCast(h, i32);
        }
        llvm::Value *h = cg.b->CreateBitCast(av, v2f16);
        return callAirFn(cg, "air.convert.f.v2f32.f.v2f16", v2f32, {h});
    }
    return nullptr;
}

/* ---- statements -------------------------------------------------------- */

void emitStmt(Codegen &cg, const MGLStmt *st, const MGLIRModule *mod,
              std::map<std::string, MType> *locals);

/* Assemble the stage output value: vertex = {position, varyings...},
 * fragment = render target color.  Unknown outputs fall back to undef. */
llvm::Value *assembleReturn(Codegen &cg) {
    if (cg.isVS) {
        if (cg.retTy->isStructTy()) {
            llvm::Value *ret = llvm::UndefValue::get(cg.retTy);
            llvm::Value *pos = cg.lvalues.count("gl_Position")
                                   ? cg.lvalues["gl_Position"]
                                   : llvm::UndefValue::get(cg.retElems[0]);
            ret = cg.b->CreateInsertValue(ret, pos, 0);
            for (uint32_t i = 0; i < cg.varyings.size(); i++) {
                llvm::Value *vv = cg.lvalues.count(cg.varyings[i]->name)
                                      ? cg.lvalues[cg.varyings[i]->name]
                                      : llvm::UndefValue::get(cg.retElems[i + 1]);
                ret = cg.b->CreateInsertValue(ret, vv, i + 1);
            }
            return ret;
        }
        return cg.lvalues.count("gl_Position")
                   ? cg.lvalues["gl_Position"]
                   : llvm::UndefValue::get(cg.retTy);
    }
    VarSym *out = nullptr;
    for (VarSym &v : *cg.auxSyms)
        if (v.kind == VarSym::OUTPUT) { out = &v; break; }
    return (out && cg.lvalues.count(out->name))
               ? cg.lvalues[out->name]
               : llvm::UndefValue::get(cg.retTy);
}

void emitCompound(Codegen &cg, const MGLStmt *st, const MGLIRModule *mod,
                  std::map<std::string, MType> *locals) {
    for (uint32_t i = 0; i < st->u.compound.count; i++)
        emitStmt(cg, st->u.compound.stmts[i], mod, locals);
}

void emitStmt(Codegen &cg, const MGLStmt *st, const MGLIRModule *mod,
              std::map<std::string, MType> *locals) {
    if (cg.err) return;
    switch (st->kind) {
    case MGL_STMT_COMPOUND:
        emitCompound(cg, st, mod, locals);
        break;
    case MGL_STMT_EXPR:
        emitExpr(cg, st->u.expr.expr, mod, *locals);
        break;
    case MGL_STMT_DECL: {
        MGLDecl *d = st->u.decl.decl;
        MType t;
        if (d->type && d->type->base <= MGL_AST_TYPE_DOUBLE) {
            t.scalar = (MGLIRScalar)d->type->base;
            if (d->type->mat_cols > 1) {
                t.cols = d->type->mat_cols;
                t.rows = d->type->mat_rows;
            } else {
                t.vec = d->type->vec_size;
            }
        } else if (d->init) {
            t = exprType(cg, d->init, mod, *locals);
        } else {
            t.scalar = MGLIR_SCALAR_FLOAT;
            if (d->type && d->type->vec_size) t.vec = d->type->vec_size;
        }
        if (d->init) {
            llvm::Value *v = emitExpr(cg, d->init, mod, *locals);
            if (!v) return;
            v = coerceScalar(cg, v, t.scalar);
            cg.lvalues[d->name] = v;
        }
        (*locals)[d->name] = t;
        break;
    }
    case MGL_STMT_RETURN: {
        if (st->u.ret.value) {
            llvm::Value *v = emitExpr(cg, st->u.ret.value, mod, *locals);
            if (!v) return;
            cg.b->CreateRet(v);
        } else if (cg.fn->getReturnType()->isVoidTy()) {
            cg.b->CreateRetVoid();
        } else {
            /* Bare return; in a non-void stage function: assemble the
             * outputs (position / varyings / frag color) as at end of
             * body. */
            cg.b->CreateRet(assembleReturn(cg));
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
        cg.b->CreateRet(assembleReturn(cg));
        cg.err = 2;
        break;
    }
    case MGL_STMT_IF: {
        /* if (cond) then [else else_]: SSA via phi at the merge block.
         * Nested ifs work recursively; a return inside a branch is
         * supported (no phi edge from that branch). */
        llvm::Value *cond = emitExpr(cg, st->u.ifs.cond, mod, *locals);
        if (!cond) return;
        /* Constant condition: emit only the live branch. */
        if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(cond);
            ci && ci->getType()->isIntegerTy(1)) {
            if (ci->getValue().getBoolValue())
                emitStmt(cg, st->u.ifs.then, mod, locals);
            else if (st->u.ifs.else_)
                emitStmt(cg, st->u.ifs.else_, mod, locals);
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
        emitStmt(cg, st->u.ifs.then, mod, locals);
        if (cg.err == 1) return;
        llvm::BasicBlock *thenTail = cg.b->GetInsertBlock();
        bool thenRet = thenTail->getTerminator() &&
                       llvm::isa<llvm::ReturnInst>(thenTail->getTerminator());
        if (!thenRet) cg.b->CreateBr(bbMerge);
        std::map<std::string, llvm::Value *> thenL = cg.lvalues;

        std::map<std::string, llvm::Value *> elseL;
        if (bbElse) {
            cg.err = 0;
            cg.b->SetInsertPoint(bbElse);
            emitStmt(cg, st->u.ifs.else_, mod, locals);
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
                llvm::PHINode *phi =
                    cg.b->CreatePHI(kv.second->getType(), 2, kv.first);
                phi->addIncoming(kv.second, thenTail);
                phi->addIncoming(it != snap.end()
                                     ? it->second
                                     : llvm::UndefValue::get(
                                           kv.second->getType()),
                                 condBB);
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
            emitStmt(cg, st->u.loop.init, mod, locals);
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
            emitStmt(cg, st->u.whilex.body, mod, locals);
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
            llvm::Value *cond = emitExpr(cg, st->u.whilex.cond, mod, *locals);
            if (cg.err) return;
            if (!cond->getType()->isIntegerTy(1)) {
                cg.err = 1;
                cg.errmsg = "codegen: do-while condition must be a scalar bool";
                return;
            }
            for (auto &kv : lc.phis)
                kv.second->addIncoming(cg.lvalues[kv.first], bbCond);
            cg.b->CreateCondBr(cond, bbBody, bbEnd);
        } else {
            llvm::Value *cond = st->kind == MGL_STMT_FOR
                ? (st->u.loop.cond ? emitExpr(cg, st->u.loop.cond, mod,
                                              *locals)
                                   : nullptr)
                : emitExpr(cg, st->u.whilex.cond, mod, *locals);
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
                    cg.b->SetInsertPoint(bbBody);
                    cg.b->CreateUnreachable();
                    cg.b->SetInsertPoint(bbIncr);
                    cg.b->CreateUnreachable();
                    cg.b->SetInsertPoint(cur);
                    cg.b->CreateBr(bbEnd);
                    bodyDead = true;
                } else {
                    cg.b->CreateCondBr(cond, bbBody, bbEnd);
                }
            } else {
                cg.b->CreateBr(bbBody);
            }
            if (!bodyDead) {
            cg.b->SetInsertPoint(bbBody);
            emitStmt(cg, st->kind == MGL_STMT_FOR ? st->u.loop.body
                                                  : st->u.whilex.body,
                     mod, locals);
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
                emitExpr(cg, st->u.loop.incr, mod, *locals);
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
            llvm::PHINode *e =
                cg.b->CreatePHI(v->getType(), 1 + brk.snaps.size(), n);
            e->addIncoming(v, bbCond);
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
        llvm::Value *cond = emitExpr(cg, st->u.switchx.cond, mod, *locals);
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
                    emitStmt(cg, s, mod, locals);
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
                emitStmt(cg, s, mod, locals);
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
        for (auto &kv : snap) {
            llvm::Value *v = kv.second;
            llvm::PHINode *e = cg.b->CreatePHI(
                v->getType(),
                1 + brk.snaps.size() + (lastTail ? 1 : 0) +
                    (defEntry ? 0 : 1),
                kv.first);
            /* No default label: the last check block falls through to
             * the exit carrying the entry values. */
            if (!defEntry)
                e->addIncoming(v, check);
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

/* ---- module assembly ---------------------------------------------------- */

} /* namespace */

static int compileGLSLImpl(const char *src, int stage, int capture,
                           unsigned char **metallib_out, size_t *size_out,
                           char *err_buf, size_t err_cap) {
    if (!src || !metallib_out || !size_out) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "bad args");
        return -1;
    }
    if (stage != MGL_STAGE_VERTEX && stage != MGL_STAGE_FRAGMENT &&
        stage != MGL_STAGE_COMPUTE) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "unsupported stage");
        return -1;
    }
    const bool isVS = (stage == MGL_STAGE_VERTEX);
    const bool isCompute = (stage == MGL_STAGE_COMPUTE);
    const bool isCapture = capture && isVS;

    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    if (!tu) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "parse: out of memory");
        return -1;
    }
    if (tu->error) {
        if (err_buf && err_cap)
            snprintf(err_buf, err_cap, "parse line %u: %s",
                     tu->error_line, tu->error);
        mglGLSLTranslationUnitDestroy(tu);
        return -1;
    }

    MGLIRModule mod;
    memset(&mod, 0, sizeof mod);
    MGLSemaError *errors = nullptr;
    uint32_t error_count = 0;
    int hard = mglGLSLSemanticCheck(tu, &mod, &errors, &error_count);
    if (hard) {
        if (err_buf && err_cap && errors && error_count)
            snprintf(err_buf, err_cap, "line %u: %s",
                     errors[0].line, errors[0].message);
        mglGLSLSemanticCheckDestroy(errors, error_count);
        mglIRModuleDestroy(&mod);
        mglGLSLTranslationUnitDestroy(tu);
        return -1;
    }
    mglGLSLSemanticCheckDestroy(errors, error_count);

    std::vector<Uniform> uniforms;
    uint32_t bufferSize = 0;
    if (collectUniforms(&mod, &uniforms, &bufferSize, err_buf, err_cap)) {
        mglIRModuleDestroy(&mod);
        mglGLSLTranslationUnitDestroy(tu);
        return -1;
    }

    /* Find main and the stage's interface symbols. */
    MGLDecl *mainDecl = nullptr;
    std::vector<VarSym> syms;
    for (uint32_t i = 0; i < mod.symbol_count; i++) {
        MGLIRSymbol *s = mod.symbols[i];
        if (s->is_function) {
            continue;
        }
        VarSym v;
        v.name = s->name;
        v.type = typeFromIR(s->type);
        uint32_t q = s->qualifiers;
        if (q & MGL_AST_Q_UNIFORM) {
            if (s->type->kind == MGLIR_TYPE_SAMPLER) {
                v.kind = VarSym::TEXTURE;
            } else if (s->type->kind == MGLIR_TYPE_STRUCT &&
                       s->type->member_count > 0) {
                v.kind = VarSym::UBO;
            } else {
                v.kind = VarSym::BUFFER;
            }
        } else if (q & MGL_AST_Q_BUFFER) {
            v.kind = VarSym::SSBO;
        } else if (isCompute) {
            continue;   /* compute has no stage varyings */
        } else if (isVS && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::ATTR;
        } else if (isVS && (q & MGL_AST_Q_OUT)) {
            v.kind = VarSym::VARYING;
        } else if (!isVS && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::VARYING;
        } else if (!isVS && (q & MGL_AST_Q_OUT)) {
            v.kind = VarSym::OUTPUT;
        }
        syms.push_back(v);
    }
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        if (tu->decls[i]->body && tu->decls[i]->name &&
            strcmp(tu->decls[i]->name, "main") == 0) {
            mainDecl = tu->decls[i];
            break;
        }
    }
    if (!mainDecl) {
        snprintf(err_buf, err_cap, "no main function");
        mglIRModuleDestroy(&mod);
        mglGLSLTranslationUnitDestroy(tu);
        return -1;
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
    llvm::Type *retTy = nullptr;
    if (isVS) {
        /* retElems always carries the output record (capture variants
         * write it to the XFB buffer). */
        retElems.push_back(llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4));
        for (VarSym &v : syms) {
            if (v.kind == VarSym::VARYING) {
                retElems.push_back(llvmType(v.type, ctx));
                varyings.push_back(&v);
            }
        }
        if (isCompute || isCapture) {
            retTy = llvm::Type::getVoidTy(ctx);
        } else if (retElems.size() == 1) {
            retTy = retElems[0];
        } else {
            retTy = llvm::StructType::get(ctx, retElems);
        }
    } else {
        VarSym *out = nullptr;
        for (VarSym &v : syms) {
            if (v.kind == VarSym::OUTPUT) { out = &v; break; }
        }
        retTy = out ? llvmType(out->type, ctx)
                    : llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4);
    }

    /* Parameters: capture = [captureBuf, buffer, ssbo..., tex/smp...,
     * attrs..., vertex_id]; vertex = [buffer, ssbo..., tex/smp...,
     * attrs...]; fragment = [varyings..., buffer, ssbo..., tex/smp...];
     * compute = [buffer, ssbo..., tex/smp..., thread_position_in_grid]. */
    std::vector<llvm::Type *> paramTys;
    bool hasBuffer = !uniforms.empty();
    llvm::StructType *texTy =
        llvm::StructType::create(ctx, "struct._texture_2d_t");
    llvm::StructType *smpTy =
        llvm::StructType::create(ctx, "struct._sampler_t");
    if (isCapture)
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    if ((isVS || isCompute) && hasBuffer)
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    for (VarSym &v : syms)
        if (v.kind == VarSym::SSBO || v.kind == VarSym::UBO)
            paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    for (VarSym &v : syms) {
        if (v.kind != VarSym::TEXTURE) continue;
        paramTys.push_back(texTy->getPointerTo(1));
        paramTys.push_back(smpTy->getPointerTo(2));
    }
    for (VarSym &v : syms) {
        if ((isVS && v.kind == VarSym::ATTR))
            paramTys.push_back(llvmType(v.type, ctx));
        else if (!isVS && !isCompute && v.kind == VarSym::VARYING)
            paramTys.push_back(llvmType(v.type, ctx));
    }
    if (isCapture)
        paramTys.push_back(llvm::Type::getInt32Ty(ctx));
    else if (isCompute)
        paramTys.push_back(llvm::FixedVectorType::get(
            llvm::Type::getInt32Ty(ctx), 3));
    else if (!isVS && hasBuffer)
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));

    llvm::FunctionType *ft = llvm::FunctionType::get(retTy, paramTys, false);
    llvm::Function *fn = llvm::Function::Create(
        ft, llvm::Function::ExternalLinkage, "main", &module);
    fn->setDoesNotThrow();
    if (hasBuffer) {
        unsigned bufIdx = (isVS || isCompute)
                              ? (isCapture ? 1 : 0)
                              : (unsigned)paramTys.size() - 1;
        fn->addParamAttr(bufIdx, llvm::Attribute::AttrKind::NoAlias);
        if (!isCompute)
            fn->addParamAttr(bufIdx, llvm::Attribute::AttrKind::ReadOnly);
    }
    if (isCapture)
        fn->addParamAttr(0, llvm::Attribute::AttrKind::NoAlias);
    {
        unsigned ssboIdx = (isCapture ? 1 : 0) +
                           ((isVS || isCompute) ? (hasBuffer ? 1 : 0) : 0);
        for (VarSym &v : syms) {
            if (v.kind != VarSym::SSBO) continue;
            fn->addParamAttr(ssboIdx++, llvm::Attribute::AttrKind::NoAlias);
        }
        for (VarSym &v : syms) {
            if (v.kind != VarSym::UBO) continue;
            fn->addParamAttr(ssboIdx, llvm::Attribute::AttrKind::NoAlias);
            fn->addParamAttr(ssboIdx, llvm::Attribute::AttrKind::ReadOnly);
            ssboIdx++;
        }
    }

    llvm::BasicBlock *entry = llvm::BasicBlock::Create(ctx, "entry", fn);
    llvm::IRBuilder<> b(entry);

    Codegen cg;
    cg.ctx = &ctx;
    cg.b = &b;
    cg.fn = fn;
    cg.mod = &module;
    cg.isVS = isVS;
    cg.isCompute = isCompute;
    /* Bind parameters by symbol: vertex = [buffer, attrs...];
     * fragment = [varyings..., buffer];
     * compute = [buffer, thread_position_in_grid]. */
    uint32_t argSlot = 0;
    if (isCapture)
        cg.captureBuf = fn->getArg(argSlot++);
    if ((isVS || isCompute) && hasBuffer)
        cg.bufferPtr = fn->getArg(argSlot++);
    for (VarSym &v : syms) {
        if (v.kind != VarSym::SSBO) continue;
        cg.ssboPtrs[v.name] = fn->getArg(argSlot++);
    }
    for (VarSym &v : syms) {
        if (v.kind != VarSym::UBO) continue;
        cg.uboPtrs[v.name] = fn->getArg(argSlot++);
    }
    for (VarSym &v : syms) {
        if (v.kind != VarSym::TEXTURE) continue;
        cg.texValues[v.name] = fn->getArg(argSlot++);
        cg.smpValues[v.name] = fn->getArg(argSlot++);
    }
    for (VarSym &v : syms) {
        if ((isVS && v.kind == VarSym::ATTR) ||
            (!isVS && !isCompute && v.kind == VarSym::VARYING))
            cg.lvalues[v.name] = fn->getArg(argSlot++);
    }
    if (isCapture)
        cg.vertexId = fn->getArg(argSlot);
    else if (isCompute)
        cg.threadPos = fn->getArg(argSlot);
    else if (!isVS && hasBuffer)
        cg.bufferPtr = fn->getArg(argSlot);
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
    cg.auxSyms = &syms;
    std::map<std::string, MType> locals;
    emitStmt(cg, mainDecl->body, &mod, &locals);

    if (cg.err && cg.err != 2) {
        snprintf(err_buf, err_cap, "%s",
                 cg.errmsg.empty() ? "codegen: unsupported construct"
                                   : cg.errmsg.c_str());
        mglIRModuleDestroy(&mod);
        mglGLSLTranslationUnitDestroy(tu);
        return -1;
    }
    /* Terminate if the body's last statement was a return. */
    if (cg.err != 2) {
        if (isCompute) {
            b.CreateRetVoid();
        } else if (isCapture) {
            /* XFB capture: write the assembled output record into the
             * capture buffer at [vertex_id]. */
            llvm::Type *recTy = cg.retElems.size() == 1
                                    ? cg.retElems[0]
                                    : llvm::StructType::get(ctx, cg.retElems);
            llvm::Value *rec = llvm::UndefValue::get(recTy);
            llvm::Value *pos = cg.lvalues.count("gl_Position")
                                   ? cg.lvalues["gl_Position"]
                                   : llvm::UndefValue::get(cg.retElems[0]);
            if (recTy->isStructTy()) {
                rec = b.CreateInsertValue(rec, pos, 0);
                for (uint32_t i = 0; i < cg.varyings.size(); i++) {
                    llvm::Value *vv =
                        cg.lvalues.count(cg.varyings[i]->name)
                            ? cg.lvalues[cg.varyings[i]->name]
                            : llvm::UndefValue::get(cg.retElems[i + 1]);
                    rec = b.CreateInsertValue(rec, vv, i + 1);
                }
            } else {
                rec = pos;
            }
            uint64_t recSize = module.getDataLayout().getTypeAllocSize(recTy);
            llvm::Value *vid = b.CreateSExtOrTrunc(cg.vertexId,
                                                   b.getInt64Ty());
            llvm::Value *p = b.CreateGEP(
                b.getInt8Ty(), cg.captureBuf,
                b.CreateMul(vid, b.getInt64(recSize)));
            p = b.CreateBitCast(p, recTy->getPointerTo(1));
            b.CreateAlignedStore(rec, p, llvm::Align(16));
            b.CreateRetVoid();
        } else {
            b.CreateRet(assembleReturn(cg));
        }
    }

    /* ---- AIR metadata ---- */
    std::vector<llvm::Metadata *> argNodes;
    if (isCapture) {
        /* Capture output record buffer (XFB slot 29, read_write). */
        llvm::Type *recTy = retElems.size() == 1
                                ? retElems[0]
                                : llvm::StructType::get(ctx, retElems);
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
        addMember(llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4),
                  "float4", "pos");
        for (VarSym *v : varyings)
            addMember(llvmType(v->type, ctx),
                      mslTypeName(v->type).c_str(), v->name.c_str());
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
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, recSize)),
            llvm::MDString::get(ctx, "air.arg_type_align_size"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 16)),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "VSOut"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "capture")}));
    }
    if (hasBuffer) {
        llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
        unsigned idx = (isVS || isCompute)
                           ? (isCapture ? 1 : 0)
                           : (unsigned)paramTys.size() - 1;
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
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 0)),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
            llvm::MDString::get(ctx, "air.read"),
            llvm::MDString::get(ctx, "air.address_space"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i32, 1)),
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
        uint32_t loc = (isCapture ? 1 : 0) +
                       ((isVS || isCompute) ? (hasBuffer ? 1 : 0) : 0);
        uint32_t ssboArg = (isCapture ? 1 : 0) +
                           ((isVS || isCompute) ? (hasBuffer ? 1 : 0) : 0);
        for (VarSym &v : syms) {
            if (v.kind != VarSym::SSBO) continue;
            const MGLIRSymbol *sb = findSymbol(&mod, v.name.c_str());
            uint32_t bsize = sb ? sb->type->layout.size : 0;
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
                llvm::MDString::get(ctx, v.name),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, v.name)}));
        }
    }
    uint32_t ssboCount = 0;
    for (VarSym &v : syms)
        if (v.kind == VarSym::SSBO) ssboCount++;
    /* Uniform blocks: independent read-only device buffers. */
    {
        uint32_t loc = (isCapture ? 1 : 0) +
                       ((isVS || isCompute) ? (hasBuffer ? 1 : 0) : 0) +
                       ssboCount;
        uint32_t uboArg = loc;
        for (VarSym &v : syms) {
            if (v.kind != VarSym::UBO) continue;
            const MGLIRSymbol *sb = findSymbol(&mod, v.name.c_str());
            uint32_t bsize = sb ? sb->type->layout.size : 0;
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
                llvm::MDString::get(ctx, v.name)}));
        }
    }
    uint32_t uboCount = 0;
    for (VarSym &v : syms)
        if (v.kind == VarSym::UBO) uboCount++;
    /* Texture/sampler pairs: air.texture + air.sampler arguments. */
    {
        uint32_t texLoc = 0, smpLoc = 0;
        uint32_t texArg = (isCapture ? 1 : 0) +
                          ((isVS || isCompute) ? (hasBuffer ? 1 : 0) : 0) +
                          ssboCount + uboCount;
        for (VarSym &v : syms) {
            if (v.kind != VarSym::TEXTURE) continue;
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
                llvm::MDString::get(ctx, "texture2d<float, sample>"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, v.name)}));
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
                llvm::MDString::get(ctx, v.name)}));
        }
    }
    uint32_t texCount = 0;
    for (VarSym &v : syms)
        if (v.kind == VarSym::TEXTURE) texCount++;
    uint32_t mArgSlot =
        (isCapture ? 1 : 0) +
        ((isVS || isCompute) ? (hasBuffer ? 1 : 0) : 0) + ssboCount +
        uboCount + 2 * texCount;
    if (isVS) {
        /* Vertex attributes arrive as vertex_input value arguments; the
         * renderer feeds them through the vertex descriptor (buffer 16+). */
        uint32_t attrLoc = 0;
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATTR) continue;
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
                llvm::MDString::get(ctx, mslTypeName(v.type)),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, v.name)};
            argNodes.push_back(llvm::MDNode::get(ctx, elems));
        }
    } else if (!isCompute) {
        for (VarSym &v : syms) {
            if (v.kind != VarSym::VARYING) continue;
            std::vector<llvm::Metadata *> elems = {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.fragment_input"),
                llvm::MDString::get(ctx, "generated(" + v.name + ")"),
                llvm::MDString::get(ctx, "air.center"),
                llvm::MDString::get(ctx, "air.perspective"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, mslTypeName(v.type)),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, v.name)};
            argNodes.push_back(llvm::MDNode::get(ctx, elems));
        }
    }
    if (isCompute) {
        /* Kernel thread position: [[thread_position_in_grid]] as uint3. */
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), mArgSlot)),
            llvm::MDString::get(ctx, "air.thread_position_in_grid"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "uint3"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "thread_position_in_grid")}));
    }

    std::vector<llvm::Metadata *> outNodes;   /* outputs / render targets */
    if (isVS && !isCapture) {
        outNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::MDString::get(ctx, "air.position"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "float4"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "position")}));
        for (VarSym *v : varyings) {
            outNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::MDString::get(ctx, "air.vertex_output"),
                llvm::MDString::get(ctx, "generated(" + v->name + ")"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, mslTypeName(v->type)),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, v->name)}));
        }
    } else if (!isCompute) {
        VarSym *out = nullptr;
        for (VarSym &v : syms)
            if (v.kind == VarSym::OUTPUT) { out = &v; break; }
        if (out) {
            outNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::MDString::get(ctx, "air.render_target"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 0)),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 0)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, mslTypeName(out->type))}));
        }
    }

    if (isCapture) {
        /* Capture variants index their output record by vertex id. */
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx),
                (unsigned)paramTys.size() - 1)),
            llvm::MDString::get(ctx, "air.vertex_id"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "uint"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, "vid")}));
    }
    std::vector<llvm::Metadata *> stageElems = {
        llvm::ValueAsMetadata::get(fn),
        llvm::MDNode::get(ctx, outNodes)};
    if (!argNodes.empty())
        stageElems.push_back(llvm::MDNode::get(ctx, argNodes));
    else
        stageElems.push_back(llvm::MDNode::get(ctx, {}));
    llvm::NamedMDNode *air = module.getOrInsertNamedMetadata(
        isCompute ? "air.kernel" : (isVS ? "air.vertex" : "air.fragment"));
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

    if (getenv("MGL_DUMP_IR"))
        module.print(llvm::errs(), nullptr);

    /* Serialize: bitcode blob + MTLB container. */
    llvm::SmallVector<char, 0> bc;
    llvm::raw_svector_ostream bcos(bc);
    llvm::WriteBitcodeToFile(module, bcos);

    std::vector<mgl::MTLBFunction> fns;
    mgl::MTLBFunction f;
    f.name = "main";
    f.type = isCompute ? mgl::MTLB_FN_KERNEL
                       : (isVS ? mgl::MTLB_FN_VERTEX : mgl::MTLB_FN_FRAGMENT);
    f.bitcode.assign(bc.begin(), bc.end());
    fns.push_back(f);

    llvm::SmallVector<char, 0> mlib;
    llvm::raw_svector_ostream mlibos(mlib);
    mgl::mglMTLBWrite(fns, mlibos);

    unsigned char *out = (unsigned char *)malloc(mlib.size());
    if (!out) {
        snprintf(err_buf, err_cap, "out of memory");
        mglIRModuleDestroy(&mod);
        mglGLSLTranslationUnitDestroy(tu);
        return -1;
    }
    memcpy(out, mlib.data(), mlib.size());
    *metallib_out = out;
    *size_out = mlib.size();

    mglIRModuleDestroy(&mod);
    mglGLSLTranslationUnitDestroy(tu);
    return 0;
}

extern "C" int mglShaderCompileGLSL(const char *src, int stage,
                                    unsigned char **metallib_out,
                                    size_t *size_out, char *err_buf,
                                    size_t err_cap) {
    return compileGLSLImpl(src, stage, 0, metallib_out, size_out, err_buf,
                           err_cap);
}

/* XFB capture variant: the vertex stage writes its full output record
 * (position + varyings) into a device buffer at location 29 with
 * rasterization disabled, mirroring the legacy mglCompileMSLCaptureVariant
 * path. */
extern "C" int mglShaderCompileGLSLCapture(const char *src,
                                           unsigned char **metallib_out,
                                           size_t *size_out, char *err_buf,
                                           size_t err_cap) {
    return compileGLSLImpl(src, MGL_STAGE_VERTEX, 1, metallib_out, size_out,
                           err_buf, err_cap);
}

extern "C" int mglAirCompileGLSLWithReflect(
    const char *src, int stage, unsigned char **metallib_out,
    size_t *size_out, SpirvResourceList lists[_MAX_SPIRV_RES], char *err_buf,
    size_t err_cap) {
    if (!src || !metallib_out || !size_out) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "bad args");
        return -1;
    }
    MGLTranslationUnit *tu = mglGLSLParse(src, strlen(src));
    if (!tu || tu->error) {
        if (err_buf && err_cap) {
            snprintf(err_buf, err_cap, "%s",
                     (tu && tu->error) ? tu->error : "parse: out of memory");
        }
        mglGLSLTranslationUnitDestroy(tu);
        return -1;
    }
    MGLIRModule mod;
    memset(&mod, 0, sizeof mod);
    MGLSemaError *errors = nullptr;
    uint32_t error_count = 0;
    int hard = mglGLSLSemanticCheck(tu, &mod, &errors, &error_count);
    if (hard) {
        if (err_buf && err_cap && errors && error_count) {
            snprintf(err_buf, err_cap, "line %u: %s",
                     errors[0].line, errors[0].message);
        }
        mglGLSLSemanticCheckDestroy(errors, error_count);
        mglGLSLTranslationUnitDestroy(tu);
        return -1;
    }
    mglGLSLSemanticCheckDestroy(errors, error_count);

    if (lists)
        mglAirReflectModule(&mod, stage, lists, err_buf, err_cap);
    mglIRModuleDestroy(&mod);
    mglGLSLTranslationUnitDestroy(tu);

    return compileGLSLImpl(src, stage, 0, metallib_out, size_out, err_buf,
                           err_cap);
}

extern "C" void mglShaderFree(void *bytes) {
    free(bytes);
}

extern "C" int mglShaderInterfaceCheck(const char *vs_src, const char *fs_src,
                                       char *err_buf, size_t err_cap) {
    if (!vs_src || !fs_src) return -1;
    MGLTranslationUnit *vtu = mglGLSLParse(vs_src, strlen(vs_src));
    MGLTranslationUnit *ftu = mglGLSLParse(fs_src, strlen(fs_src));
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
    int vhard = mglGLSLSemanticCheck(vtu, &vs, &ve, &vc);
    int fhard = mglGLSLSemanticCheck(ftu, &fs, &fe, &fc);
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
    }
    mglGLSLSemanticCheckDestroy(ve, vc);
    mglGLSLSemanticCheckDestroy(fe, fc);
    mglIRModuleDestroy(&vs);
    mglIRModuleDestroy(&fs);
    mglGLSLTranslationUnitDestroy(vtu);
    mglGLSLTranslationUnitDestroy(ftu);
    return rc;
}
