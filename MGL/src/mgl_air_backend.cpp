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
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

#include "mgl_glsl_ast.h"
#include "mgl_glsl_parser.h"
#include "mgl_glsl_sema.h"
#include "mgl_ir.h"
#include "mgl_metallib_writer.h"
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
    enum Kind { ATTR, VARYING, OUTPUT, BUFFER, LOCAL } kind = LOCAL;
    uint32_t bufferOffset = 0;
    bool written = false;
};

struct Codegen {
    llvm::LLVMContext *ctx;
    llvm::IRBuilder<> *b;
    llvm::Function *fn;
    bool isVS = false;
    llvm::Value *bufferPtr = nullptr;    /* i8 addrspace(1)* */
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
            snprintf(err, errCap, "samplers/images not supported in M1");
            return -1;
        }
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

llvm::Value *emitExpr(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
                      const std::map<std::string, MType> &locals);

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
        /* M1: constant index only.  Matrix[i] yields a column vector
         * (GLSL 4.60 5.5), vector[i] a component. */
        if (e->u.index.index->kind != MGL_EXPR_LITERAL ||
            (e->u.index.index->u.literal.base != MGL_AST_TYPE_INT &&
             e->u.index.index->u.literal.base != MGL_AST_TYPE_UINT)) {
            cg.err = 1;
            cg.errmsg = std::string("codegen: matrix/vector index must be "
                                    "a constant integer");
            return nullptr;
        }
        uint32_t i = (uint32_t)e->u.index.index->u.literal.value;
        MType bt = exprType(cg, e->u.index.object, mod, locals);
        llvm::Value *obj = emitExpr(cg, e->u.index.object, mod, locals);
        if (!obj) return nullptr;
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
        cg.err = 1;
        cg.errmsg = std::string("codegen: call to '") + name +
                    "' not implemented in M1";
        return nullptr;
    }
    case MGL_EXPR_UNARY: {
        if (e->u.unary.op != MGL_OP_SUB || !e->u.unary.prefix) {
            cg.err = 1; return nullptr;
        }
        llvm::Value *v = emitExpr(cg, e->u.unary.operand, mod, locals);
        return v ? cg.b->CreateFNeg(v) : nullptr;
    }
    case MGL_EXPR_BINARY: {
        llvm::Value *l = emitExpr(cg, e->u.binary.lhs, mod, locals);
        llvm::Value *r = emitExpr(cg, e->u.binary.rhs, mod, locals);
        if (!l || !r) return nullptr;
        /* Matrix * vector: column-major dot, width = matrix rows. */
        if (llvm::ArrayType *arr =
                llvm::dyn_cast<llvm::ArrayType>(l->getType())) {
            llvm::FixedVectorType *colTy =
                llvm::dyn_cast<llvm::FixedVectorType>(arr->getElementType());
            if (e->u.binary.op != MGL_OP_MUL || !colTy ||
                !r->getType()->isVectorTy()) {
                cg.err = 1;
                cg.errmsg = "codegen: matrix operation unsupported in M1";
                return nullptr;
            }
            uint32_t cols = (uint32_t)arr->getNumElements();
            uint32_t rows = (uint32_t)colTy->getElementCount().getFixedValue();
            if (cols > rows) {
                cg.err = 1;
                cg.errmsg = "codegen: matrix with more columns than rows * vector unsupported in M1";
                return nullptr;
            }
            llvm::FixedVectorType *outTy =
                llvm::FixedVectorType::get(llvm::Type::getFloatTy(*cg.ctx),
                                           rows);
            llvm::Value *acc = llvm::Constant::getNullValue(outTy);
            for (uint32_t c = 0; c < cols; c++) {
                llvm::Value *col = cg.b->CreateExtractValue(l, c);
                llvm::Value *splat = cg.b->CreateShuffleVector(r,
                    llvm::UndefValue::get(r->getType()),
                    llvm::ConstantVector::getSplat(
                        llvm::ElementCount::getFixed(rows),
                        llvm::ConstantInt::get(
                            llvm::Type::getInt32Ty(*cg.ctx), c)));
                llvm::Value *term = cg.b->CreateFMul(col, splat);
                acc = c == 0 ? term : cg.b->CreateFAdd(acc, term);
            }
            return acc;
        }
        /* Implicit numeric conversion: if exactly one side is FP, promote
         * the other (GLSL 4.1.10). */
        bool lfp = l->getType()->isFPOrFPVectorTy();
        bool rfp = r->getType()->isFPOrFPVectorTy();
        if (lfp != rfp) {
            if (lfp) r = coerceScalar(cg, r, MGLIR_SCALAR_FLOAT);
            else l = coerceScalar(cg, l, MGLIR_SCALAR_FLOAT);
        }
        bool fp = l->getType()->isFPOrFPVectorTy();
        switch (e->u.binary.op) {
        case MGL_OP_ADD: return fp ? cg.b->CreateFAdd(l, r) : cg.b->CreateAdd(l, r);
        case MGL_OP_SUB: return fp ? cg.b->CreateFSub(l, r) : cg.b->CreateSub(l, r);
        case MGL_OP_MUL:
            if (fp) {
                return cg.b->CreateFMul(l, r);
            }
            return cg.b->CreateMul(l, r);
        case MGL_OP_DIV: return fp ? cg.b->CreateFDiv(l, r) : cg.b->CreateSDiv(l, r);
        default: cg.err = 1; return nullptr;
        }
    }
    case MGL_EXPR_ASSIGN: {
        /* x = y where x is a named lvalue. */
        if (e->u.assign.op != MGL_OP_ASSIGN ||
            e->u.assign.lhs->kind != MGL_EXPR_VAR_REF) {
            cg.err = 1; return nullptr;
        }
        const char *name = e->u.assign.lhs->u.var_ref.name;
        llvm::Value *v = emitExpr(cg, e->u.assign.rhs, mod, locals);
        if (!v) return nullptr;
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
        cg.lvalues[name] = v;
        return v;
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
    default:
        break;
    }
    return t;
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
    case MGL_STMT_IF:
        /* M1: single-level if without else is enough for the gate. */
        cg.err = 1;
        break;
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

extern "C" int mglShaderCompileGLSL(const char *src, int stage,
                                    unsigned char **metallib_out,
                                    size_t *size_out, char *err_buf,
                                    size_t err_cap) {
    if (!src || !metallib_out || !size_out) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "bad args");
        return -1;
    }
    if (stage != MGL_STAGE_VERTEX && stage != MGL_STAGE_FRAGMENT) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "compute unsupported");
        return -1;
    }
    const bool isVS = (stage == MGL_STAGE_VERTEX);

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
            v.kind = VarSym::BUFFER;
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
        retElems.push_back(llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4));
        for (VarSym &v : syms) {
            if (v.kind == VarSym::VARYING) {
                retElems.push_back(llvmType(v.type, ctx));
                varyings.push_back(&v);
            }
        }
        if (retElems.size() == 1)
            retTy = retElems[0];
        else
            retTy = llvm::StructType::get(ctx, retElems);
    } else {
        VarSym *out = nullptr;
        for (VarSym &v : syms) {
            if (v.kind == VarSym::OUTPUT) { out = &v; break; }
        }
        retTy = out ? llvmType(out->type, ctx)
                    : llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4);
    }

    /* Parameters: vertex = [buffer, attrs...]; fragment = [varyings..., buffer]. */
    std::vector<llvm::Type *> paramTys;
    bool hasBuffer = !uniforms.empty();
    if (isVS && hasBuffer)
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    for (VarSym &v : syms) {
        if ((isVS && v.kind == VarSym::ATTR) ||
            (!isVS && v.kind == VarSym::VARYING))
            paramTys.push_back(llvmType(v.type, ctx));
    }
    if (!isVS && hasBuffer)
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));

    llvm::FunctionType *ft = llvm::FunctionType::get(retTy, paramTys, false);
    llvm::Function *fn = llvm::Function::Create(
        ft, llvm::Function::ExternalLinkage, "main", &module);
    fn->setDoesNotThrow();
    if (hasBuffer) {
        unsigned bufIdx = isVS ? 0 : (unsigned)paramTys.size() - 1;
        fn->addParamAttr(bufIdx, llvm::Attribute::AttrKind::NoAlias);
        fn->addParamAttr(bufIdx, llvm::Attribute::AttrKind::ReadOnly);
    }

    llvm::BasicBlock *entry = llvm::BasicBlock::Create(ctx, "entry", fn);
    llvm::IRBuilder<> b(entry);

    Codegen cg;
    cg.ctx = &ctx;
    cg.b = &b;
    cg.fn = fn;
    cg.isVS = isVS;
    /* Bind parameters by symbol: vertex = [buffer, attrs...];
     * fragment = [varyings..., buffer]. */
    uint32_t argSlot = 0;
    if (isVS && hasBuffer)
        cg.bufferPtr = fn->getArg(argSlot++);
    for (VarSym &v : syms) {
        if ((isVS && v.kind == VarSym::ATTR) ||
            (!isVS && v.kind == VarSym::VARYING))
            cg.lvalues[v.name] = fn->getArg(argSlot++);
    }
    if (!isVS && hasBuffer)
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
    if (cg.err != 2)
        b.CreateRet(assembleReturn(cg));

    /* ---- AIR metadata ---- */
    std::vector<llvm::Metadata *> argNodes;
    if (hasBuffer) {
        llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
        unsigned idx = isVS ? 0 : (unsigned)paramTys.size() - 1;
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
    uint32_t mArgSlot = hasBuffer ? 1 : 0;
    if (isVS) {
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATTR) continue;
            std::vector<llvm::Metadata *> elems = {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.vertex_input"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 0)),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, mslTypeName(v.type)),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, v.name)};
            argNodes.push_back(llvm::MDNode::get(ctx, elems));
        }
    } else {
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

    std::vector<llvm::Metadata *> outNodes;   /* outputs / render targets */
    if (isVS) {
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
    } else {
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

    std::vector<llvm::Metadata *> stageElems = {
        llvm::ValueAsMetadata::get(fn),
        llvm::MDNode::get(ctx, outNodes)};
    if (!argNodes.empty())
        stageElems.push_back(llvm::MDNode::get(ctx, argNodes));
    else
        stageElems.push_back(llvm::MDNode::get(ctx, {}));
    llvm::NamedMDNode *air = module.getOrInsertNamedMetadata(
        isVS ? "air.vertex" : "air.fragment");
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
    f.type = isVS ? mgl::MTLB_FN_VERTEX : mgl::MTLB_FN_FRAGMENT;
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

extern "C" void mglShaderFree(void *bytes) {
    free(bytes);
}
