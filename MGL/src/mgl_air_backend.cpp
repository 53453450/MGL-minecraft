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

namespace {

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


/* Lightweight type model for codegen.  Mirrors the MGLIR scalar/vector/
 * matrix shapes; the LLVM types are derived on demand. */
struct MType {
    MGLIRScalar scalar = MGLIR_SCALAR_FLOAT;
    uint32_t vec = 0;        /* vector width, 0 = scalar */
    uint32_t cols = 0;       /* matrix columns, 0 = not a matrix */
    uint32_t rows = 0;       /* matrix rows */
    uint32_t arr = 0;        /* array element count, 0 = not an array */

    bool isMatrix() const { return cols != 0; }
    bool isArray() const { return arr != 0; }
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
    enum Kind { ATTR, CONTROL_POINT_INPUT, VARYING, OUTPUT, BUFFER, SSBO,
                UBO, TEXTURE, IMAGE, ATOMIC_COUNTER, LOCAL } kind = LOCAL;
    uint32_t bufferOffset = 0;
    uint32_t location = UINT32_MAX;
    int32_t stream = 0;          /* GS output stream for OUTPUT vars */
    std::string blockName;       /* owning interface block, or empty */
    bool isPatch = false;
    bool written = false;
};

struct LoopCtx {
    llvm::BasicBlock *condBB = nullptr;  /* do-while continue target */
    llvm::BasicBlock *endBB = nullptr;   /* break target */
    llvm::BasicBlock *incrBB = nullptr;  /* merge block; while/for continue target */
    llvm::BasicBlock *condExitBB = nullptr; /* false-condition exit (after side effects) */
    std::map<std::string, llvm::Value *> condExitSnap;
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
    bool isTessControl = false;
    bool isTessEval = false;
    bool isGeometry = false;
    llvm::Value *bufferPtr = nullptr;    /* i8 addrspace(1)* */
    llvm::Value *bufferSizePtr = nullptr; /* constant uint*, buffer(25) */
    llvm::Value *threadPos = nullptr;    /* compute: <3 x i32> grid position */
    llvm::Value *workGroupPos = nullptr; /* compute: <3 x i32> group position */
    llvm::Value *numWorkGroups = nullptr; /* compute: <3 x i32> dispatch grid */
    llvm::Value *invocationPos = nullptr; /* TCS: <3 x i32> threadgroup position */
    llvm::Value *patchPos = nullptr;      /* TCS: <3 x i32> threadgroup grid position */
    llvm::Value *stageInPtr = nullptr;   /* TCS gl_in replacement, buffer(24) */
    llvm::Value *stageOutPtr = nullptr;  /* TCS gl_out replacement, buffer(28) */
    llvm::Value *tessFactorPtr = nullptr; /* TCS factors, buffer(26) */
    llvm::Value *indirectPtr = nullptr;  /* TCS patch info, buffer(29) */
    uint32_t tcsOutputVertices = 0;
    uint32_t stageInStride = MGL_AIR_PER_VERTEX_STRIDE;
    uint32_t stageOutStride = MGL_AIR_PER_VERTEX_STRIDE;
    uint32_t patchInStride = 16;
    uint32_t patchOutStride = 16;
    llvm::Value *geometryInputPtr = nullptr;  /* GS primitive records */
    llvm::Value *geometryOutputPtr = nullptr; /* GS expanded records */
    llvm::Value *geometryCountPtr = nullptr;  /* GS indirect draw args */
    llvm::Value *geometryGatherPtr = nullptr; /* GS indexed gather stream */
    llvm::Value *geometryGatherParamsPtr = nullptr; /* GS gather params    */
    llvm::Value *geometryXfbPtr = nullptr;  /* GS XFB stream, buffer(31)   */
    llvm::Value *geometryXfbMetaPtr = nullptr; /* GS XFB meta, buffer(27)  */
    llvm::Value *geometryXfbVisPtr = nullptr;  /* GS XFB visibility, buffer(30) */
    llvm::Value *tessGatherPtr = nullptr;     /* TES compute gather stream */
    llvm::Value *tessGatherParamsPtr = nullptr; /* TES compute gather params*/
    llvm::Value *xfbOutPtr = nullptr;   /* TES compute XFB stream, buffer(31) */
    llvm::Value *geometryWorkItemId = nullptr;
    llvm::Value *geometryPrimitiveId = nullptr;
    llvm::Value *geometryInvocationId = nullptr;
    uint32_t geometryInputVertices = 3;
    uint32_t geometryOutputType = MGL_AST_GS_OUT_TRIANGLE_STRIP;
    uint32_t geometryMaxVertices = 0;
    uint32_t geometryOutputVertices = 0;
    uint32_t geometryRecordCount = 0;
    llvm::Value *patchControlPtr = nullptr; /* TES patch_control_point stage-in */
    llvm::Value *tessCoord = nullptr;    /* TES position_in_patch */
    llvm::Value *patchId = nullptr;      /* TES patch_id */
    llvm::Function *controlPointGetter = nullptr; /* TES stage-in accessor */
    std::map<std::string, uint32_t> controlPointFields;
    bool isTESCompute = false;   /* isolines/point-mode TES kernel: gl_in
                                  * reads come from the stage_in buffer
                                  * instead of the Metal control-point fn */
    llvm::Value *captureBuf = nullptr;   /* capture variant: output buffer */
    llvm::Value *vertexId = nullptr;     /* capture variant: vertex_id */
    llvm::Value *instanceId = nullptr;   /* vertex: instance_id */
    llvm::Value *baseInstance = nullptr; /* vertex: base_instance */
    llvm::Value *cullBuffer = nullptr;   /* VS cull-distance source buffer */
    llvm::Value *cullParams = nullptr;   /* VS cull-distance emu parameters */
    bool usesCullDistance = false;
    llvm::Value *fragPos = nullptr;      /* fragment: [[position]] (gl_FragCoord) */
    bool hasFragDepth = false;           /* fragment writes gl_FragDepth */
    bool fragDepthInit = false;          /* gl_FragDepth lvalue initialized */
    bool usesClipDistance = false;       /* vertex writes gl_ClipDistance */
    bool pointSize = false;              /* vertex: writes gl_PointSize */
    bool layerViewport = false;          /* writes gl_Layer / gl_ViewportIndex */
    bool primitiveIdWritten = false;     /* writes gl_PrimitiveID (GS out) */
    std::map<std::string, llvm::Value *> ssboPtrs;  /* SSBO instance -> buffer */
    std::map<std::string, llvm::Value *> acPtrs;  /* atomic_uint -> ACBO */
    /* UBO instance arrays: element pointers stashed in an entry-block
     * alloca so member reads can index by runtime value. */
    std::map<std::string, llvm::Value *> uboElemSlot;
    std::map<std::string, llvm::Type *> uboElemArrTy;
    std::map<std::string, uint32_t> ssboSlots;      /* SSBO instance -> Metal slot */
    std::map<std::string, uint32_t> acSlots;        /* atomic_uint -> Metal slot */
    std::map<std::string, llvm::Value *> uboPtrs;   /* uniform block -> buffer */
    std::map<std::string, llvm::Value *> texValues;  /* sampler name -> texture */
    std::map<std::string, llvm::Value *> smpValues;  /* sampler name -> sampler */
    std::map<std::string, std::vector<llvm::Value *>> texArrayValues;
    std::map<std::string, std::vector<llvm::Value *>> smpArrayValues;
    std::map<std::string, uint32_t> bufferOffsets;  /* uniform name -> byte offset */
    std::map<std::string, llvm::Value *> lvalues;   /* register values */
    std::vector<VarSym *> varyings;      /* vertex out / fragment in, decl order */
    std::vector<VarSym *> fragOutputs;   /* fragment outputs, return-field order */
    bool has_gs = false;                 /* fragment fed by GS passthrough VS */
    VarSym position;                     /* gl_Position */
    llvm::Type *retTy = nullptr;         /* stage return type */
    std::vector<llvm::Type *> retElems;  /* VS struct fields (incl. position) */
    std::vector<VarSym> *auxSyms = nullptr;  /* all stage symbols (frag output) */
    std::map<std::string, llvm::Function *> *userFns = nullptr;
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

/* Integer varyings on AGX must ride float carriers: GS expansion already
 * needed this for all integer types; flat integer varyings in plain VS/FS
 * pipelines also misread when carried as raw int/uint stage inputs. */
static bool varyingUsesFloatCarrier(const MType &t, bool has_gs) {
    if (t.scalar == MGLIR_SCALAR_BOOL || scalarIsFloat(t.scalar))
        return false;
    if (has_gs)
        return true;
    return t.scalar == MGLIR_SCALAR_INT || t.scalar == MGLIR_SCALAR_UINT;
}

static llvm::Value *encodeFloatCarrier(Codegen &cg, llvm::Value *value,
                                       MGLIRScalar scalar) {
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    if (scalar == MGLIR_SCALAR_UINT)
        return cg.b->CreateUIToFP(value, f32);
    return cg.b->CreateSIToFP(value, f32);
}

static llvm::Value *decodeFloatCarrier(Codegen &cg, llvm::Value *arg,
                                       MGLIRScalar scalar,
                                       llvm::Type *destTy) {
    arg = cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::round, arg);
    if (scalar == MGLIR_SCALAR_UINT)
        return cg.b->CreateFPToUI(arg, destTy);
    return cg.b->CreateFPToSI(arg, destTy);
}

static MType floatCarrierType(const MType &t) {
    MType f = t;
    f.scalar = MGLIR_SCALAR_FLOAT;
    return f;
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
    /* Arrays-of-matrices: arr takes precedence over the matrix shape. */
    if (t.isArray()) {
        MType el = t;
        el.arr = 0;
        return llvm::ArrayType::get(llvmType(el, ctx), t.arr);
    }
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
    /* bool before the generic float→int path so float→bool is a compare,
     * not FPToSI to i32 (which would then fail to match i1 uses). */
    if (want == MGLIR_SCALAR_BOOL) {
        if (curFP)
            return cg.b->CreateFCmpUNE(v, llvm::Constant::getNullValue(cur));
        if (cur->getScalarSizeInBits() == 1)
            return v;
        return cg.b->CreateICmpNE(v, llvm::Constant::getNullValue(cur));
    }
    if (curFP)
        return cg.b->CreateFPToSI(v, vt(llvm::Type::getInt32Ty(ctx)));
    /* int: widen bool to i32, otherwise identity. */
    if (cur->getScalarSizeInBits() == 1)
        return cg.b->CreateZExt(v, vt(llvm::Type::getInt32Ty(ctx)));
    return v;
}

/* Itanium-style type mangling for air.vertex_output / air.fragment_input
 * "generated(...)" tags (e.g. "1aDv4_f": len 1 + "a" + vec4<float>). */
std::string mslTypeName(const MType &t);

std::string airTypeMangle(const MType &t) {
    if (t.isMatrix() || t.isArray()) {
        return mslTypeName(t);
    }
    const char *elem;
    switch (t.scalar) {
    case MGLIR_SCALAR_INT:  elem = "i"; break;
    case MGLIR_SCALAR_UINT: elem = "j"; break;
    case MGLIR_SCALAR_BOOL: elem = "b"; break;
    default:                elem = "f"; break;
    }
    if (!t.vec) return elem;
    return "Dv" + std::to_string(t.vec) + "_" + elem;
}

std::string airGenerated(const std::string &name, const MType &t) {
    return "generated(" + std::to_string(name.size()) + name +
           airTypeMangle(t) + ")";
}

/* GLSL type name used in air.* metadata (MSL naming). */
std::string mslTypeName(const MType &t) {
    if (t.isArray()) {
        MType el = t;
        el.arr = 0;
        return mslTypeName(el);
    }
    if (t.isMatrix()) {
        char buf[32];
        snprintf(buf, sizeof buf, "float%ux%u", t.cols, t.rows);
        return buf;
    }
    switch (t.scalar) {
    case MGLIR_SCALAR_INT:   return t.vec ? "int" + std::to_string(t.vec) : "int";
    case MGLIR_SCALAR_UINT:  return t.vec ? "uint" + std::to_string(t.vec) : "uint";
    case MGLIR_SCALAR_BOOL:  return t.vec ? "bool" + std::to_string(t.vec) : "bool";
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
    case MGLIR_TYPE_ARRAY: {
        r.arr = t->array_size;
        const MGLIRType *el = t->elem_type;
        while (el && el->kind == MGLIR_TYPE_ARRAY) {
            el = el->elem_type;
            r.arr *= t->array_size;
        }
        if (el) {
            r.scalar = el->scalar;
            if (el->kind == MGLIR_TYPE_VECTOR) {
                r.vec = el->cols;
            } else if (el->kind == MGLIR_TYPE_MATRIX) {
                /* matNxM[K] must keep cols/rows so llvmType builds
                 * [K x [N x <M x float>]], not float[K]. */
                r.cols = el->cols;
                r.rows = el->rows;
            }
        }
        break;
    }
    default: break;
    }
    return r;
}

const MGLIRType *uniformBlockType(const MGLIRType *type) {
    if (type && type->kind == MGLIR_TYPE_ARRAY)
        type = type->elem_type;
    return type && type->kind == MGLIR_TYPE_STRUCT && type->member_count > 0
        ? type : nullptr;
}

uint32_t uniformBlockElementCount(const MGLIRType *type) {
    /* `uniform Block { } name[1]` is still an instance array — keep the
     * declared length so blockC[0].x can index element slots. */
    if (type && type->kind == MGLIR_TYPE_ARRAY && type->array_size > 0u)
        return type->array_size;
    return 1u;
}

static bool uniformBlockIsInstanceArray(const MGLIRType *type) {
    return type && type->kind == MGLIR_TYPE_ARRAY && type->array_size > 0u;
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
        const MGLIRType *ut = s->type;
        while (ut->kind == MGLIR_TYPE_ARRAY && ut->elem_type)
            ut = ut->elem_type;
        if (ut->kind == MGLIR_TYPE_SAMPLER ||
            ut->kind == MGLIR_TYPE_IMAGE ||
            ut->kind == MGLIR_TYPE_ATOMIC_COUNTER) {
            continue;   /* texture/sampler/atomic-counter params are separate
                         * AIR args; packing them into the plain uniform blob
                         * would desync reflection offsets. */
        }
        /* Uniform blocks (struct types, including instance arrays) and their
         * anonymous-block members are independent device buffers, not part
         * of the plain uniform pack. */
        if (s->block_name || uniformBlockType(s->type))
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
    if (bt.isArray() || obj->getType()->isArrayTy()) {
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
    if (bt.isArray() || obj->getType()->isArrayTy()) {
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
    uint32_t off = 0;
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

/* Load a UBO matrix at byte offset `off` from `base`, honouring
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
        llvm::SmallVector<llvm::Value *, 4> rows;
        for (uint32_t r = 0; r < vt.rows; r++) {
            llvm::Value *rowOff =
                cg.b->CreateAdd(off, cg.b->getInt64((uint64_t)r * stride));
            llvm::Value *rp =
                cg.b->CreateGEP(cg.b->getInt8Ty(), base, rowOff);
            rp = cg.b->CreateBitCast(rp, rowTy->getPointerTo(1));
            rows.push_back(
                cg.b->CreateAlignedLoad(rowTy, rp, llvm::Align(16)));
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
    for (uint32_t c = 0; c < vt.cols; c++) {
        llvm::Value *colOff =
            cg.b->CreateAdd(off, cg.b->getInt64((uint64_t)c * stride));
        llvm::Value *cp =
            cg.b->CreateGEP(cg.b->getInt8Ty(), base, colOff);
        cp = cg.b->CreateBitCast(cp, colTy->getPointerTo(1));
        llvm::Value *col =
            cg.b->CreateAlignedLoad(colTy, cp, llvm::Align(16));
        v = cg.b->CreateInsertValue(v, col, c);
    }
    return v;
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
    llvm::Align align(16);
    if (auto *fvt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
        uint64_t w = fvt->getElementCount().getFixedValue();
        if (w == 1) align = llvm::Align(4);
        else if (w == 2) align = llvm::Align(8);
    } else if (t->isFloatTy() || t->isIntegerTy(32)) {
        align = llvm::Align(4);
    }
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

llvm::Value *varValue(Codegen &cg, const VarSym &v, const MGLIRModule *mod) {
    if (v.kind == VarSym::BUFFER) {
        /* Anonymous-block member: read from the block's device buffer. */
        const MGLIRSymbol *bs = findSymbol(mod, v.name.c_str());
        if (getenv("MGL_VAR_DBG"))
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

static llvm::Value *emitPatchVaryingLoad(Codegen &cg, const VarSym &sym)
{
    if (!cg.isTessEval || !sym.isPatch || !cg.captureBuf || !cg.patchId ||
        sym.location == UINT32_MAX) {
        return nullptr;
    }
    llvm::Value *off = cg.b->CreateAdd(
        cg.b->CreateMul(cg.b->CreateZExt(cg.patchId, cg.b->getInt64Ty()),
                        cg.b->getInt64(cg.patchInStride)),
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
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
    p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
    return cg.b->CreateAlignedLoad(ty, p, llvm::Align(4));
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
    VarSym *member =
        codegenStageSymbol(cg, e->u.member.field, VarSym::VARYING);
    if (!member || member->location == UINT32_MAX ||
        member->blockName != instName || member->type.isArray()) {
        /* Array members are handled by the INDEX case so the trailing
         * element index selects the record slot. */
        return nullptr;
    }
    vertexIndex = coerceScalar(cg, vertexIndex, MGLIR_SCALAR_UINT);
    llvm::Value *record = geometryInputRecordIndex(cg, vertexIndex);
    if (!record) return nullptr;
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
    VarSym *member =
        codegenStageSymbol(cg, memberE->u.member.field, VarSym::VARYING);
    /* Stage-level info for return assembly. */    if (!member || member->location == UINT32_MAX ||
        member->blockName != instName || !member->type.isArray()) {
        return nullptr;
    }
    llvm::Value *element = emitExpr(cg, indexExpr->u.index.index,
                                    mod, locals);
    if (!element) return nullptr;
    element = coerceScalar(cg, element, MGLIR_SCALAR_UINT);
    if (auto *ci = llvm::dyn_cast<llvm::ConstantInt>(element)) {
        if (ci->getZExtValue() >= member->type.arr) {
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
    return loadGeometryInputVarying(cg, *member, slot, record,
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
    VarSym *sym = codegenStageSymbol(cg, name, VarSym::VARYING);
    if (!sym || sym->location == UINT32_MAX) return nullptr;
    llvm::Value *index = emitExpr(cg, e->u.index.index, mod, locals);
    if (!index) return nullptr;
    llvm::Value *record = nullptr;
    llvm::Value *base = nullptr;
    if (cg.isGeometry) {
        if (!cg.geometryInputPtr || !cg.geometryPrimitiveId) return nullptr;
        index = coerceScalar(cg, index, MGLIR_SCALAR_UINT);
        record = geometryInputRecordIndex(cg, index);
        base = cg.geometryInputPtr;
    } else {
        if (!cg.stageInPtr || !cg.patchPos || !cg.indirectPtr) return nullptr;
        record = tessStageRecordIndex(cg, index, true);
        base = cg.stageInPtr;
    }
    if (cg.isGeometry) {
        return loadGeometryInputVarying(cg, *sym, cg.b->getInt32(sym->location),
                                        record, base);
    }
    llvm::Value *off = cg.b->CreateAdd(
        cg.b->CreateMul(cg.b->CreateZExt(record, cg.b->getInt64Ty()),
                        cg.b->getInt64(cg.stageInStride)),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE + sym->location * 16u));
    llvm::Type *ty = llvmType(sym->type, *cg.ctx);
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), base, off);
    p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
    return cg.b->CreateAlignedLoad(ty, p, llvm::Align(4));
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
    if (value->getType() != ty)
        value = coerceScalar(cg, value, sym->type.scalar);
    llvm::Value *p = cg.b->CreateGEP(cg.b->getInt8Ty(), cg.stageOutPtr, off);
    p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
    cg.b->CreateAlignedStore(value, p, llvm::Align(4));
    return true;
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
        if (!cg.isTESCompute &&
            (!cg.patchControlPtr || !cg.controlPointGetter)) {
            cg.err = 1;
            cg.errmsg = "codegen: TES patch control points are unavailable";
            return nullptr;
        }
        llvm::Value *iv = emitExpr(cg, index, mod, locals);
        if (!iv) return nullptr;
        iv = coerceScalar(cg, iv, MGLIR_SCALAR_UINT);
        if (!strcmp(field, "gl_Position") && !cg.isTESCompute) {
            llvm::Value *record = cg.b->CreateCall(
                cg.controlPointGetter, {iv, cg.patchControlPtr});
            return cg.b->CreateExtractValue(record, 0);
        }
        if (!cg.stageInPtr || !cg.indirectPtr || !cg.patchId) {
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
        llvm::Value *flat = cg.b->CreateAdd(
            cg.b->CreateMul(cg.patchId, verticesPerPatch), iv);
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
    if (getenv("MGL_GS_DIAG_CONST")) {
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
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_LAYER_OFFSET));
    p = cg.b->CreateBitCast(p, cg.b->getInt32Ty()->getPointerTo(1));
    cg.b->CreateAlignedStore(layer, p, llvm::Align(4));
}

static void storeGeometryViewportIndex(Codegen &cg, llvm::Value *record,
                                       llvm::Value *viewportIndex)
{
    llvm::Value *p = cg.b->CreateGEP(
        cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
        cg.b->getInt64(MGL_AIR_PER_VERTEX_VIEWPORT_INDEX_OFFSET));
    p = cg.b->CreateBitCast(p, cg.b->getInt32Ty()->getPointerTo(1));
    cg.b->CreateAlignedStore(viewportIndex, p, llvm::Align(4));
}

/* gl_PrimitiveID written by the GS rides at offset 52 so the fragment
 * stage can receive it through the passthrough vertex function (flat).
 * The record holds a float carrier (sitofp of the id): Apple's AGX
 * compiler segfaults in InstCombine when a flat int stage_input that is
 * actually read crosses into the fragment stage, so the id travels as a
 * float and the FS entry converts it back with round+fptosi.  Every
 * reader/writer of this slot must use the same carrier type.
 * Unwritten records keep whatever the strip cache held; the renderer's
 * PTVS only forwards it for programs that declared gl_PrimitiveID. */
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
        llvm::Value *p = cg.b->CreateGEP(
            cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
            cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE +
                           varying.location * 16u));
        p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
        cg.b->CreateAlignedStore(value, p, llvm::Align(4));
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
        llvm::Value *p = cg.b->CreateGEP(
            cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
            cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE +
                           varying.location * 16u));
        p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
        cg.b->CreateAlignedStore(value, p, llvm::Align(4));
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
        llvm::Type *ty = llvmType(varying.type, *cg.ctx);
        uint64_t fieldOffset = MGL_AIR_PER_VERTEX_STRIDE +
                               varying.location * 16u;
        llvm::Value *src = cg.b->CreateGEP(
            cg.b->getInt8Ty(),
            geometryRecordPtr(cg, cg.b->getInt32(sourceRecord)),
            cg.b->getInt64(fieldOffset));
        src = cg.b->CreateBitCast(src, ty->getPointerTo(1));
        llvm::Value *value = cg.b->CreateAlignedLoad(ty, src, llvm::Align(4));
        llvm::Value *out = cg.b->CreateGEP(
            cg.b->getInt8Ty(), geometryRecordPtr(cg, dst),
            cg.b->getInt64(fieldOffset));
        out = cg.b->CreateBitCast(out, ty->getPointerTo(1));
        cg.b->CreateAlignedStore(value, out, llvm::Align(4));
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
        llvm::Type *ty = llvmType(varying.type, *cg.ctx);
        uint64_t fieldOffset = MGL_AIR_PER_VERTEX_STRIDE +
                               varying.location * 16u;
        auto load = [&](uint32_t source) {
            llvm::Value *p = cg.b->CreateGEP(
                cg.b->getInt8Ty(),
                geometryRecordPtr(cg, cg.b->getInt32(source)),
                cg.b->getInt64(fieldOffset));
            p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
            return cg.b->CreateAlignedLoad(ty, p, llvm::Align(4));
        };
        llvm::Value *value = cg.b->CreateSelect(
            condition, load(trueRecord), load(falseRecord));
        llvm::Value *out = cg.b->CreateGEP(
            cg.b->getInt8Ty(), geometryRecordPtr(cg, dst),
            cg.b->getInt64(fieldOffset));
        out = cg.b->CreateBitCast(out, ty->getPointerTo(1));
        cg.b->CreateAlignedStore(value, out, llvm::Align(4));
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
        llvm::Value *previousLayer = loadGeometryLayer(cg, 0);
        llvm::Value *previousViewport = loadGeometryViewportIndex(cg, 0);
        llvm::Value *outputCount = cg.b->CreateAlignedLoad(
            cg.b->getInt32Ty(), outputCountPtr, llvm::Align(4));
        llvm::Value *outputRecord = cg.b->CreateAdd(
            outputCount, cg.b->getInt32(2));
        storeGeometryPosition(cg, outputRecord, previous);
        storeGeometryPointSize(cg, outputRecord, previousPoint);
        storeGeometryCullDistances(cg, outputRecord, previousCull);
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
    llvm::Value *previousLayer1ForNext = loadGeometryLayer(cg, 1);
    llvm::Value *previousViewport1ForNext = loadGeometryViewportIndex(cg, 1);
    storeGeometryPosition(cg, cg.b->getInt32(0), previous1ForNext);
    storeGeometryPointSize(cg, cg.b->getInt32(0), previousPoint1ForNext);
    storeGeometryCullDistances(cg, cg.b->getInt32(0), previousCull1ForNext);
    storeGeometryLayer(cg, cg.b->getInt32(0), previousLayer1ForNext);
    storeGeometryViewportIndex(cg, cg.b->getInt32(0), previousViewport1ForNext);
    copyGeometryPrimitiveId(cg, cg.b->getInt32(0), 1);
    copyGeometryVaryings(cg, cg.b->getInt32(0), 1);
    storeGeometryPosition(cg, cg.b->getInt32(1), pos);
    storeGeometryPointSize(cg, cg.b->getInt32(1), pointSize);
    storeGeometryCullDistances(cg, cg.b->getInt32(1), cullDistances);
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
        llvm::Value *p = cg.b->CreateGEP(
            cg.b->getInt8Ty(), geometryRecordPtr(cg, record),
            cg.b->getInt64(MGL_AIR_PER_VERTEX_STRIDE + v.location * 16u));
        p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
        cg.b->CreateAlignedStore(value, p, llvm::Align(4));
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
                cg.errmsg = "codegen: gl_PatchVerticesIn requires a TCS stage";
                return nullptr;
            }
            llvm::Value *p = cg.b->CreateBitCast(
                cg.indirectPtr, cg.b->getInt32Ty()->getPointerTo(1));
            return cg.b->CreateAlignedLoad(cg.b->getInt32Ty(), p,
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
            return cg.lvalues[e->u.var_ref.name];
        }
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
        }
        if (const MGLIRSymbol *sb = ssboRootSym(e, mod))
            return emitSSBORead(cg, e, sb, mod, locals);
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
                llvm::Value *base = nullptr;
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
                return emitBlockMemberChain(cg, e, base, ubStruct,
                                            bufName ? bufName : objName, mod,
                                            locals, startOff);
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
                return emitBlockMemberChain(cg, e, base, ubStruct,
                                            bufName ? bufName : objName, mod,
                                            locals, startOff);
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
                    p = cg.b->CreateBitCast(p, ty->getPointerTo(1));
                    return cg.b->CreateAlignedLoad(ty, p, llvm::Align(4));
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
            if (bt.isArray()) {
                auto *arrayTy = llvm::dyn_cast<llvm::ArrayType>(obj->getType());
                if (!arrayTy || i >= arrayTy->getNumElements()) {
                    cg.err = 1;
                    cg.errmsg = std::string("codegen: array index ") +
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
                auto slot = cg.ssboSlots.find(sb->name);
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
                llvm::Value *sizes = cg.b->CreateBitCast(
                    cg.bufferSizePtr,
                    llvm::Type::getInt32Ty(*cg.ctx)->getPointerTo(2));
                llvm::Value *sizePtr = cg.b->CreateGEP(
                    cg.b->getInt32Ty(), sizes, cg.b->getInt64(slot->second));
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
            MType array = exprType(cg, object, mod, locals);
            if (array.arr != 0)
                return cg.b->getInt32(array.arr);
            cg.err = 1;
            cg.errmsg = "codegen: length() requires an array expression";
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
        /* Array constructors: vecN[](a, b, ...). */
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
        /* Storage-image operations use the texture handle table but have no
         * sampler parameter.  image2D keeps the original float path; the
         * array write path below mirrors Metal's texture2d_array integer ABI. */
        if (strcmp(name, "imageStore") == 0 ||
            strcmp(name, "imageLoad") == 0 ||
            strcmp(name, "imageSize") == 0) {
            const MGLExpr *ia = e->u.call.arg_count > 0
                ? e->u.call.args[0] : nullptr;
            if (!ia || ia->kind != MGL_EXPR_VAR_REF) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: ") + name +
                    " first argument must be an image variable";
                return nullptr;
            }
            const MGLIRSymbol *is = findSymbol(mod, ia->u.var_ref.name);
            bool is2DArray = is && is->type->kind == MGLIR_TYPE_IMAGE &&
                             is->type->tex_kind == MGLIR_TEX_2D_ARRAY;
            if (!is || is->type->kind != MGLIR_TYPE_IMAGE ||
                (is->type->tex_kind != MGLIR_TEX_2D && !is2DArray)) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: ") + name +
                    " currently requires image2D or image2DArray";
                return nullptr;
            }
            llvm::Value *tex = samplerTexValue(cg, ia->u.var_ref.name);
            if (!tex) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: missing image binding for ") +
                    ia->u.var_ref.name;
                return nullptr;
            }
            llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
            llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
            llvm::Type *v2i32 = llvm::FixedVectorType::get(i32, 2);
            llvm::Type *v3i32 = llvm::FixedVectorType::get(i32, 3);
            llvm::Type *v4f32 = llvm::FixedVectorType::get(f32, 4);
            llvm::Type *v4i32 = llvm::FixedVectorType::get(i32, 4);
            if (is2DArray && strcmp(name, "imageStore") != 0) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: ") + name +
                    " image2DArray is not implemented yet";
                return nullptr;
            }
            if (strcmp(name, "imageSize") == 0) {
                llvm::Value *w = callAirFn(cg, "air.get_width_texture_2d",
                                           i32, {tex, cg.b->getInt32(0)});
                llvm::Value *h = callAirFn(cg, "air.get_height_texture_2d",
                                           i32, {tex, cg.b->getInt32(0)});
                llvm::Value *size = llvm::UndefValue::get(v2i32);
                size = cg.b->CreateInsertElement(size, w, cg.b->getInt32(0));
                return cg.b->CreateInsertElement(size, h, cg.b->getInt32(1));
            }
            llvm::Value *coord = emitExpr(cg, e->u.call.args[1], mod, locals);
            if (!coord) return nullptr;
            coord = coerceScalar(cg, coord, MGLIR_SCALAR_INT);
            llvm::Value *layer = nullptr;
            if (is2DArray) {
                if (coord->getType() != v3i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: image2DArray coordinate must be ivec3";
                    return nullptr;
                }
                layer = cg.b->CreateExtractElement(coord, cg.b->getInt32(2));
                coord = cg.b->CreateShuffleVector(
                    coord, llvm::UndefValue::get(coord->getType()), {0, 1});
            } else if (coord->getType() != v2i32) {
                cg.err = 1;
                cg.errmsg = std::string("codegen: ") + name +
                    " image2D coordinate must be ivec2";
                return nullptr;
            }
            if (strcmp(name, "imageLoad") == 0) {
                MGLIRScalar storage = is->type->tex_storage;
                if (storage == MGLIR_SCALAR_INT ||
                    storage == MGLIR_SCALAR_UINT) {
                    llvm::Type *v4i32 = llvm::FixedVectorType::get(i32, 4);
                    llvm::Type *retTy = llvm::StructType::get(
                        *cg.ctx, {v4i32, cg.b->getInt8Ty()});
                    const char *intrinsic =
                        storage == MGLIR_SCALAR_UINT
                            ? "air.read_texture_2d.u.v4i32"
                            : "air.read_texture_2d.s.v4i32";
                    llvm::Value *r = callAirFn(cg, intrinsic, retTy,
                        {tex, coord, cg.b->getInt32(0),
                         cg.b->getInt32(3)});
                    return cg.b->CreateExtractValue(r, 0);
                }
                llvm::Type *retTy = llvm::StructType::get(
                    *cg.ctx, {v4f32, cg.b->getInt8Ty()});
                llvm::Value *r = callAirFn(cg, "air.read_texture_2d.v4f32",
                    retTy, {tex, coord, cg.b->getInt32(0),
                            cg.b->getInt32(3)});
                return cg.b->CreateExtractValue(r, 0);
            }
            llvm::Value *value = emitExpr(cg, e->u.call.args[2], mod, locals);
            if (!value) return nullptr;
            if (is2DArray) {
                if (value->getType() != v4i32 ||
                    (is->type->tex_storage != MGLIR_SCALAR_INT &&
                     is->type->tex_storage != MGLIR_SCALAR_UINT)) {
                    cg.err = 1;
                    cg.errmsg = "codegen: integer image2DArray store requires ivec4/uvec4";
                    return nullptr;
                }
                const char *intrinsic = is->type->tex_storage == MGLIR_SCALAR_UINT
                    ? "air.write_texture_2d_array.u.v4i32"
                    : "air.write_texture_2d_array.s.v4i32";
                return callAirFn(cg, intrinsic,
                                 llvm::Type::getVoidTy(*cg.ctx),
                                 {tex, coord, layer, value,
                                  cg.b->getInt32(0), cg.b->getInt32(3)});
            }
            value = coerceScalar(cg, value, MGLIR_SCALAR_FLOAT);
            if (value->getType() != v4f32) {
                cg.err = 1;
                cg.errmsg = "codegen: imageStore image2D value must be vec4";
                return nullptr;
            }
            return callAirFn(cg, "air.write_texture_2d.v4f32",
                             llvm::Type::getVoidTy(*cg.ctx),
                             {tex, coord, value, cg.b->getInt32(0),
                              cg.b->getInt32(3)});
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
            if (sa->kind != MGL_EXPR_VAR_REF) {
                cg.err = 1;
                cg.errmsg = "codegen: texelFetch first argument must be a "
                            "sampler variable";
                return nullptr;
            }
            llvm::Value *tex = samplerTexValue(cg, sa->u.var_ref.name);
            if (!tex) {
                cg.err = 1;
                cg.errmsg = "codegen: texelFetch first argument must be a "
                            "sampler variable";
                return nullptr;
            }
            const MGLIRSymbol *ts = findSymbol(mod, sa->u.var_ref.name);
            const MGLIRType *sampleType = ts ? ts->type : nullptr;
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
                llvm::Type *smp = llvm::StructType::get(
                    *cg.ctx, "struct._sampler_t");
                llvm::Value *rs = callAirFn(cg, "air.get_read_sampler",
                                            smp->getPointerTo(2), {});
                llvm::Type *v4f32 = llvm::FixedVectorType::get(
                    llvm::Type::getFloatTy(*cg.ctx), 4);
                llvm::Type *retTy = llvm::StructType::get(
                    *cg.ctx, {v4f32, cg.b->getInt8Ty()});
                llvm::Value *r = callAirFn(
                    cg, "air.read_texture_buffer_1d.v4f32", retTy,
                    {tex, rs, coord, cg.b->getInt32(1)});
                return cg.b->CreateExtractValue(r, 0);
            }
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
            auto toIvec2XY0 = [&](llvm::Value *x) -> llvm::Value * {
                llvm::Value *v = llvm::UndefValue::get(v2i32);
                v = cg.b->CreateInsertElement(v, x, cg.b->getInt32(0));
                v = cg.b->CreateInsertElement(v, cg.b->getInt32(0),
                                              cg.b->getInt32(1));
                return v;
            };
            auto unsampledRead2d = [&](llvm::Value *xy,
                                       llvm::Value *level) -> llvm::Value * {
                return callAirFn(
                    cg, readIntrinsic("air.read_texture_2d.v4f32").c_str(),
                    retTy, {tex, xy, level, cg.b->getInt32(3)});
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
            llvm::Value *r = nullptr;
            if (texKind == MGLIR_TEX_2D_ARRAY) {
                if (coord->getType() != v3i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler2DArray "
                                "expects ivec3 coordinates";
                    return nullptr;
                }
                llvm::Value *layer =
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(2));
                coord = cg.b->CreateShuffleVector(
                    coord, llvm::UndefValue::get(coord->getType()),
                    {0, 1});
                r = callAirFn(
                    cg,
                    readIntrinsic("air.read_texture_2d_array.v4f32").c_str(),
                    retTy,
                    {tex, coord, layer, lodOrSample, cg.b->getInt32(3)});
            } else if (texKind == MGLIR_TEX_1D_ARRAY) {
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler1DArray "
                                "expects ivec2 coordinates";
                    return nullptr;
                }
                llvm::Value *layer =
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(1));
                coord = cg.b->CreateExtractElement(coord, cg.b->getInt32(0));
                coord = toIvec2XY0(coord);
                r = callAirFn(
                    cg,
                    readIntrinsic("air.read_texture_2d_array.v4f32").c_str(),
                    retTy,
                    {tex, coord, layer, lodOrSample, cg.b->getInt32(3)});
            } else if (texKind == MGLIR_TEX_1D) {
                if (!coord->getType()->isIntegerTy()) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler1D expects "
                                "int coordinates";
                    return nullptr;
                }
                r = unsampledRead2d(toIvec2XY0(coord), lodOrSample);
            } else if (texKind == MGLIR_TEX_2D_RECT) {
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler2DRect "
                                "expects ivec2 coordinates";
                    return nullptr;
                }
                r = unsampledRead2d(coord, cg.b->getInt32(0));
            } else if (texKind == MGLIR_TEX_2D_MS) {
                if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler2DMS "
                                "expects ivec2 coordinates";
                    return nullptr;
                }
                r = callAirFn(
                    cg,
                    readIntrinsic("air.read_texture_2d_ms.v4f32").c_str(),
                    retTy,
                    {tex, coord, lodOrSample, cg.b->getInt32(3)});
            } else if (texKind == MGLIR_TEX_2D_MS_ARRAY) {
                if (coord->getType() != v3i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler2DMSArray "
                                "expects ivec3 coordinates";
                    return nullptr;
                }
                llvm::Value *layer =
                    cg.b->CreateExtractElement(coord, cg.b->getInt32(2));
                coord = cg.b->CreateShuffleVector(
                    coord, llvm::UndefValue::get(coord->getType()),
                    {0, 1});
                r = callAirFn(
                    cg,
                    readIntrinsic("air.read_texture_2d_ms_array.v4f32")
                        .c_str(),
                    retTy,
                    {tex, coord, layer, lodOrSample, cg.b->getInt32(3)});
            } else if (texKind == MGLIR_TEX_3D) {
                if (coord->getType() != v3i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler3D expects "
                                "ivec3 coordinates";
                    return nullptr;
                }
                r = callAirFn(
                    cg, readIntrinsic("air.read_texture_3d.v4f32").c_str(),
                    retTy,
                    {tex, coord, lodOrSample, cg.b->getInt32(3)});
            } else if (texKind == MGLIR_TEX_CUBE) {
                if (coord->getType() != v3i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a samplerCube expects "
                                "ivec3 coordinates";
                    return nullptr;
                }
                r = callAirFn(
                    cg, readIntrinsic("air.read_texture_cube.v4f32").c_str(),
                    retTy,
                    {tex, coord, lodOrSample, cg.b->getInt32(3)});
            } else {
                if (coord->getType()->isIntegerTy()) {
                    coord = toIvec2XY0(coord);
                } else if (coord->getType() != v2i32) {
                    cg.err = 1;
                    cg.errmsg = "codegen: texelFetch on a sampler2D expects "
                                "ivec2 coordinates";
                    return nullptr;
                }
                r = unsampledRead2d(coord, lodOrSample);
            }
            return cg.b->CreateExtractValue(r, 0);
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
            const char *samplerName = nullptr;
            if (sa->kind == MGL_EXPR_VAR_REF) {
                samplerName = sa->u.var_ref.name;
            } else if (sa->kind == MGL_EXPR_INDEX &&
                       sa->u.index.object &&
                       sa->u.index.object->kind == MGL_EXPR_VAR_REF) {
                samplerName = sa->u.index.object->u.var_ref.name;
            }
            if (!samplerName) {
                cg.err = 1;
                cg.errmsg = "codegen: texture argument must be a sampler2D "
                            "variable";
                return nullptr;
            }
            llvm::Value *tex = nullptr;
            llvm::Value *smp = nullptr;
            bool dynamicSamplerArray = false;
            llvm::Value *arrayIndex = nullptr;
            const std::vector<llvm::Value *> *texArray = nullptr;
            const std::vector<llvm::Value *> *smpArray = nullptr;
            if (sa->kind == MGL_EXPR_INDEX) {
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
            } else {
                tex = samplerTexValue(cg, samplerName);
                auto si = cg.smpValues.find(samplerName);
                if (si != cg.smpValues.end()) smp = si->second;
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
            const MGLIRSymbol *tss = findSymbol(mod, samplerName);
            const MGLIRType *sampleTypeForDim = tss ? tss->type : nullptr;
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
                    llvm::Value *r = callAirFn(
                        cg,
                        sampledIntrinsic(gradName).c_str(),
                        sampledRetType(vecTy),
                        {t, sp, uv, dPdx, dPdy,
                         llvm::ConstantFP::get(f32, 0.0),
                         cg.b->getInt1(false),
                         gradOffset,
                         cg.b->getInt32(0)});
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
            } else if (sampleKind == MGLIR_TEX_CUBE ||
                       sampleKind == MGLIR_TEX_CUBE_ARRAY) {
                baseName = "air.sample_texture_cube.v4f32";
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
            auto doSampleVec =
                [&](llvm::Value *t, llvm::Value *s) -> llvm::Value * {
                llvm::Value *sp = s;
                if (!sp) {
                    llvm::Type *smpT = llvm::StructType::get(
                        *cg.ctx, "struct._sampler_t");
                    sp = callAirFn(cg, "air.get_read_sampler",
                                    smpT->getPointerTo(2), {});
                }
                llvm::Value *r = callAirFn(
                    cg, sampledIntrinsic(baseName).c_str(), retTy,
                    {t, sp, uv, cg.b->getInt1(true),
                     sampleOffset,
                     cg.b->getInt1(explicitLod),
                     lod ? lod : llvm::ConstantFP::get(f32, 0.0),
                     llvm::ConstantFP::get(f32, 0.0),
                     cg.b->getInt32(0)});
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
        /* User-defined function call. */
        if (cg.userFns) {
            std::string key = std::string(name) + "#" +
                              std::to_string(e->u.call.arg_count);
            auto fit = cg.userFns->find(key);
            if (fit != cg.userFns->end()) {
                uint64_t hidden = cg.uboPtrs.size() + cg.ssboPtrs.size() +
                                  cg.acPtrs.size() +
                                  (cg.bufferSizePtr ? 1u : 0u) +
                                  (cg.isGeometry ? 8u : 0u);
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
                return cg.b->CreateCall(fit->second, args);
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
        const bool diagAssign = getenv("MGL_GS_DIAG_ASSIGN") && cg.isGeometry;
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
            lhs->u.index.object->kind == MGL_EXPR_VAR_REF &&
            codegenStageSymbol(
                cg, lhs->u.index.object->u.var_ref.name, VarSym::OUTPUT)) {
            if (e->u.assign.op != MGL_OP_ASSIGN) {
                cg.err = 1;
                cg.errmsg = "codegen: compound TCS stage output assignment "
                            "is not implemented";
                return nullptr;
            }
            emitTessStageArrayStore(cg, lhs, v, mod, locals);
            return v;
        }

        if (cg.isTessControl && lhs && lhs->kind == MGL_EXPR_INDEX &&
            lhs->u.index.object &&
            lhs->u.index.object->kind == MGL_EXPR_MEMBER) {
            const MGLExpr *member = lhs->u.index.object;
            const char *pvRoot = nullptr, *pvField = nullptr;
            const MGLExpr *pvVertexIndex = nullptr;
            if (perVertexPath(member, &pvRoot, &pvVertexIndex, &pvField) &&
                !strcmp(pvField, "gl_CullDistance")) {
                llvm::Value *array = emitPerVertexLoad(
                    cg, member, mod, locals);
                llvm::Value *component = emitExpr(
                    cg, lhs->u.index.index, mod, locals);
                if (!array || !component) return nullptr;
                component = coerceScalar(cg, component, MGLIR_SCALAR_UINT);
                if (e->u.assign.op != MGL_OP_ASSIGN) {
                    MType arrayType;
                    arrayType.scalar = MGLIR_SCALAR_FLOAT;
                    arrayType.arr = MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT;
                    llvm::Value *old = emitIndexValue(
                        cg, array, arrayType, component);
                    uint32_t binop = e->u.assign.op == MGL_OP_ADD_ASSIGN
                        ? MGL_OP_ADD : e->u.assign.op == MGL_OP_SUB_ASSIGN
                        ? MGL_OP_SUB : e->u.assign.op == MGL_OP_MUL_ASSIGN
                        ? MGL_OP_MUL : e->u.assign.op == MGL_OP_DIV_ASSIGN
                        ? MGL_OP_DIV : 0u;
                    if (!old || !binop) {
                        cg.err = 1;
                        cg.errmsg = "codegen: compound gl_out CullDistance "
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
                arrayType.arr = MGL_AIR_PER_VERTEX_CULL_DISTANCE_COUNT;
                llvm::Value *updated = insertIndexValue(
                    cg, array, arrayType, component,
                    coerceScalar(cg, v, MGLIR_SCALAR_FLOAT));
                if (!updated) {
                    cg.err = 1;
                    cg.errmsg = "codegen: failed to update gl_out CullDistance";
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
        }

        /* Interface-block member write: instance.field = v (VS/TES out
         * blocks flatten to per-member VARYING symbols, so this is an
         * ordinary varying lvalue store keyed by the member name). */
        if (lhs->kind == MGL_EXPR_MEMBER &&
            lhs->u.member.object &&
            lhs->u.member.object->kind == MGL_EXPR_VAR_REF) {
            const char *instName = lhs->u.member.object->u.var_ref.name;
            VarSym *member = codegenStageSymbol(
                cg, lhs->u.member.field, VarSym::VARYING);
            if (member && !cg.isGeometry && !cg.isTessControl &&
                member->location != UINT32_MAX &&
                member->blockName == instName) {
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
            return v;
        }
        if (strcmp(name, "gl_PrimitiveID") == 0) {
            if (!cg.isGeometry) {
                cg.err = 1;
                return nullptr;
            }
            cg.primitiveIdWritten = true;
            cg.lvalues[name] = coerceScalar(cg, v, MGLIR_SCALAR_INT);
            return v;
        }
        if (strcmp(name, "gl_Layer") == 0 ||
            strcmp(name, "gl_ViewportIndex") == 0) {
            cg.layerViewport = true;
            cg.lvalues[name] = coerceScalar(cg, v, MGLIR_SCALAR_INT);
            return v;
        }
        if (strcmp(name, "gl_FragDepth") == 0) {
            /* Fragment depth output; carried in the struct return (see
             * assembleReturn).  Unwritten paths keep 1.0. */
            cg.lvalues[name] = coerceScalar(cg, v, MGLIR_SCALAR_FLOAT);
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
            break;
        }
        std::vector<uint32_t> idx;
        /* Uniform-block member chain: the leaf IR type is the expression
         * type (the chain already includes every .field / [i] step). */
        if (const MGLIRType *leaf = blockMemberLeafType(e, mod)) {
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
        if (base.isMatrix()) {
            /* Matrix[i] yields a column vector. */
            t.scalar = base.scalar;
            t.vec = base.rows;
        } else if (base.isArray()) {
            /* Array[i] yields the element type. */
            t = base;
            t.arr = 0;
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

llvm::Value *assembleReturn(Codegen &cg) {
    if (cg.isVS) {
        if (cg.retTy->isStructTy()) {
            llvm::Value *ret = llvm::UndefValue::get(cg.retTy);
            llvm::Value *pos = cg.lvalues.count("gl_Position")
                                   ? cg.lvalues["gl_Position"]
                                   : llvm::UndefValue::get(cg.retElems[0]);
            pos = fixClipZ(cg, pos);
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
                ret = cg.b->CreateInsertValue(
                    ret,
                    cg.lvalues.count("gl_ClipDistance")
                        ? cg.lvalues["gl_ClipDistance"]
                        : defaultClipDistances(cg),
                    ri++);
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
                        if (varyingUsesFloatCarrier(var->type, cg.has_gs)) {
                            el = encodeFloatCarrier(cg, el, var->type.scalar);
                        }
                        ret = cg.b->CreateInsertValue(ret, el, ri++);
                    }
                } else {
                    if (varyingUsesFloatCarrier(var->type, cg.has_gs)) {
                        base = encodeFloatCarrier(cg, base, var->type.scalar);
                    }
                    ret = cg.b->CreateInsertValue(ret, base, ri++);
                }
            }
            return ret;
        }
        llvm::Value *pos = cg.lvalues.count("gl_Position")
                               ? cg.lvalues["gl_Position"]
                               : llvm::UndefValue::get(cg.retTy);
        return applyCullDistance(cg, fixClipZ(cg, pos));
    }
    VarSym *arrayOut = nullptr;
    for (VarSym &v : *cg.auxSyms) {
        if (v.kind == VarSym::OUTPUT && v.type.isArray()) {
            arrayOut = &v;
            break;
        }
    }
    if (arrayOut) {
        llvm::Value *color = cg.lvalues.count(arrayOut->name)
            ? cg.lvalues[arrayOut->name]
            : llvm::UndefValue::get(llvmType(arrayOut->type, *cg.ctx));
        /* gl_FragData[i]: extract each element into the struct return. */
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
        return ret;
    }
    if (cg.fragOutputs.size() > 1u || cg.hasFragDepth) {
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
            ret = cg.b->CreateInsertValue(ret, depth, field);
        }
        return ret;
    }
    VarSym *out = cg.fragOutputs.empty() ? nullptr : cg.fragOutputs[0];
    return (out && cg.lvalues.count(out->name))
        ? cg.lvalues[out->name] : llvm::UndefValue::get(cg.retTy);
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
        /* Comma-separated declarators (`int a = 0, b = 1;`): every node
         * declares its own local. */
        for (MGLDecl *d = st->u.decl.decl; d; d = d->next_declarator) {
        MType t;
        if (d->type && d->type->base <= MGL_AST_TYPE_DOUBLE) {
            t.scalar = (MGLIRScalar)d->type->base;
            if (d->type->mat_cols > 1) {
                t.cols = d->type->mat_cols;
                t.rows = d->type->mat_rows;
            } else {
                t.vec = d->type->vec_size;
            }
            if (d->array_count > 0 && d->array_dims)
                t.arr = d->array_dims[0];
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
        }
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
            /* Restart from the pre-if values: branch bodies are mutually
             * exclusive, so a value computed inside the then branch (a phi
             * in a then-side merge block) does not dominate the else side.
             * Letting the else body see it produced phi operands on
             * non-dominating edges -- invalid IR that crashed the AGX
             * compiler (MTLCompilerService SIGSEGV). */
            cg.lvalues = snap;
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

/* Desired vertex attribute location: explicit glBindAttribLocation
 * bindings first, then the Mojang stable names (mirroring the legacy
 * mglDesiredAttribLocationForName default table).  UINT32_MAX means no
 * preference (declaration order applies). */
static uint32_t airAttribLocation(const char *name,
                                  const char *const *attrib_names) {
    if (name && attrib_names) {
        for (int i = 0; i < MAX_ATTRIBS; i++) {
            if (attrib_names[i] && strcmp(attrib_names[i], name) == 0) {
                return (uint32_t)i;
            }
        }
    }
    if (name) {
        static const struct { const char *n; uint32_t l; } def[] = {
            {"Position", 0}, {"Color", 1}, {"UV0", 2},
            {"UV1", 3}, {"UV2", 4}, {"Normal", 5},
        };
        for (const auto &d : def) {
            if (strcmp(d.n, name) == 0) {
                return d.l;
            }
        }
    }
    return UINT32_MAX;
}

static bool stmtContainsReturn(const MGLStmt *st) {
    if (!st) return false;
    switch (st->kind) {
    case MGL_STMT_RETURN:
        return true;
    case MGL_STMT_COMPOUND:
        for (uint32_t i = 0; i < st->u.compound.count; i++)
            if (stmtContainsReturn(st->u.compound.stmts[i])) return true;
        return false;
    case MGL_STMT_IF:
        return stmtContainsReturn(st->u.ifs.then) ||
               stmtContainsReturn(st->u.ifs.else_);
    case MGL_STMT_FOR:
        return stmtContainsReturn(st->u.loop.init) ||
               stmtContainsReturn(st->u.loop.body);
    case MGL_STMT_WHILE:
    case MGL_STMT_DO_WHILE:
        return stmtContainsReturn(st->u.whilex.body);
    case MGL_STMT_SWITCH:
        return stmtContainsReturn(st->u.switchx.body);
    default:
        return false;
    }
}

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

static GLuint airStageToGLShaderType(int air_stage) {
    switch (air_stage) {
        case MGL_STAGE_VERTEX: return GL_VERTEX_SHADER;
        case MGL_STAGE_FRAGMENT: return GL_FRAGMENT_SHADER;
        case MGL_STAGE_TESS_CONTROL: return GL_TESS_CONTROL_SHADER;
        case MGL_STAGE_TESS_EVALUATION: return GL_TESS_EVALUATION_SHADER;
        case MGL_STAGE_GEOMETRY: return GL_GEOMETRY_SHADER;
        case MGL_STAGE_COMPUTE: return GL_COMPUTE_SHADER;
        default: return 0;
    }
}

/* GLSL version number from the #version directive; legacy default 110. */
static int airGLSLVersionOf(const char *src) {
    if (!src) return 110;
    const char *v = strstr(src, "#version");
    if (!v) return 110;
    int ver = 0;
    char prof[32] = {0};
    if (sscanf(v + 8, "%d %31s", &ver, prof) >= 1 && ver > 0) {
        return ver;
    }
    return 110;
}

/* Detect + translate legacy GLSL.  Returns a malloc'd translated copy (caller
 * frees via free()) or NULL when the source needs no translation.  The caller
 * falls back to the original source on NULL. */
static char *airPrepareLegacySource(const char *src, int air_stage) {
    if (!src) return NULL;
    /* The compile entry re-parses the translated source produced by the
     * reflect entry (and vice versa).  Matrix uniforms and gl_Vertex keep
     * their ORIGINAL names after translation (the AIR frontend accepts gl_
     * prefixed user declarations), so detecting them again would double-
     * inject the declarations.  The preamble marker identifies an
     * already-translated source. */
    if (strstr(src, "/* MGL legacy GLSL translation: renamed builtins declared as")) {
        return NULL;
    }
    mgl_legacy_features_t features;
    memset(&features, 0, sizeof(features));
    mgl_legacy_detect(src, &features);
    if (!features.needs_translation) return NULL;
    const GLuint shader_type = airStageToGLShaderType(air_stage);
    const int version = airGLSLVersionOf(src);
    const size_t len = strlen(src);
    /* The translator needs +2048 growth headroom (same convention the
     * standalone test harness uses). */
    char *translated = (char *)malloc(len + 2048);
    if (!translated) return NULL;
    memcpy(translated, src, len + 1);
    const int ret = mgl_translate_legacy_glsl(
        translated, len + 2048, shader_type, version, &features);
    if (ret != 1) {
        /* Not modified (or error): keep the original source. */
        free(translated);
        return NULL;
    }
    return translated;
}

static int compileGLSLImpl(const char *src, int stage, int capture,
                           bool has_gs, const char *const *attrib_names,
                           uint32_t tessPatchVertices,
                           unsigned char **metallib_out, size_t *size_out,
                           char *err_buf, size_t err_cap) {
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
    /* Legacy GLSL frontend wiring: translate pre-3.30 constructs before
     * parsing so the reflect + MSL passes see core-profile source. */
    std::unique_ptr<char[]> legacy_holder(airPrepareLegacySource(src, stage));
    const char *esrc = legacy_holder ? legacy_holder.get() : src;
    const bool isVS = (stage == MGL_STAGE_VERTEX);
    const bool isCompute = (stage == MGL_STAGE_COMPUTE);
    const bool isTCS = (stage == MGL_STAGE_TESS_CONTROL);
    const bool isTES = (stage == MGL_STAGE_TESS_EVALUATION);
    const bool isGS = (stage == MGL_STAGE_GEOMETRY);
    const bool isCapture = capture != 0 && isVS;
    const bool isTessCapture = capture == 2 && isVS;
    const bool isCullCapture = capture == 3 && isVS;
    /* AIR has no primitive-level cull-distance output.  Keep the GLSL
     * builtin as an SSA array and append two hidden vertex arguments so the
     * return path can reproduce the legacy primitive-cull emulation. */
    const bool sourceUsesCullDistance =
        strstr(esrc, "gl_CullDistance") != nullptr;
    const bool usesCullDistance = isVS && !isCapture &&
                                  sourceUsesCullDistance;
    if (isGS && getenv("MGL_GS_DIAG_SOURCE"))
        fprintf(stderr, "MGL GS SOURCE BEGIN\n%s\nMGL GS SOURCE END\n", esrc);
    MGLTranslationUnit *tu = mglGLSLParse(esrc, strlen(esrc));
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
    int hard = mglGLSLSemanticCheck(tu, stage, &mod, &errors, &error_count);
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
    const bool needsBufferSizeBuffer =
        translationUnitUsesRuntimeArrayLength(tu, &mod);
    /* Metal post-tessellation only supports triangle/quad patches (no
     * isolines patch type, no point output topology).  isolines and
     * point-mode TES compile to a compute kernel that enumerates the
     * expanded line/point stream instead (see the isTESCompute paths). */
    const bool isTESCompute = isTES &&
        (tu->layout_primitive == MGL_AST_TES_ISOLINES ||
         tu->layout_point_mode != 0);
    const bool isKernel = isCompute || isTCS || isGS || isTESCompute;
    const uint32_t runtimeArraySizeBufferIndex =
        (isGS || isTESCompute)
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
            mglIRModuleDestroy(&mod);
            mglGLSLTranslationUnitDestroy(tu);
            return -1;
        }
        if (tu->layout_primitive_out != MGL_AST_GS_OUT_POINTS &&
            tu->layout_primitive_out != MGL_AST_GS_OUT_LINE_STRIP &&
            tu->layout_primitive_out != MGL_AST_GS_OUT_TRIANGLE_STRIP) {
            if (err_buf && err_cap)
                snprintf(err_buf, err_cap,
                         "GS AIR codegen: invalid output topology");
            mglIRModuleDestroy(&mod);
            mglGLSLTranslationUnitDestroy(tu);
            return -1;
        }
        if (tu->layout_max_vertices > 1024) {
            if (err_buf && err_cap)
                snprintf(err_buf, err_cap,
                         "GS AIR codegen: max_vertices must be in the range 0..1024");
            mglIRModuleDestroy(&mod);
            mglGLSLTranslationUnitDestroy(tu);
            return -1;
        }
    }

    if (isTCS || isTESCompute) {
        for (uint32_t i = 0; i < tu->decl_count; i++) {
            MGLDecl *d = tu->decls[i];
            if (!d || !d->body || !d->name || strcmp(d->name, "main") != 0)
                continue;
            if (stmtContainsReturn(d->body)) {
                if (err_buf && err_cap)
                    snprintf(err_buf, err_cap,
                             isTCS ? "TCS AIR codegen: explicit return is not "
                                    "implemented yet"
                                   : "TES AIR codegen: explicit return in "
                                     "isolines/point-mode TES is not "
                                     "implemented yet");
                mglIRModuleDestroy(&mod);
                mglGLSLTranslationUnitDestroy(tu);
                return -1;
            }
        }
    }
    if (isTES) {
        if (tu->layout_primitive != MGL_AST_TES_TRIANGLES &&
            tu->layout_primitive != MGL_AST_TES_QUADS &&
            tu->layout_primitive != MGL_AST_TES_ISOLINES) {
            if (err_buf && err_cap)
                snprintf(err_buf, err_cap,
                         "TES AIR codegen: only layout(triangles/quads/"
                         "isolines) is implemented yet");
            mglIRModuleDestroy(&mod);
            mglGLSLTranslationUnitDestroy(tu);
            return -1;
        }
    }

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
        /* Builtin interface-block shells/members are lowered through the
         * dedicated gl_Position/gl_PointSize/gl_CullDistance paths below,
         * never as user varyings.  EXCEPTIONS (must become real kernel
         * parameters, mirroring mgl_air_reflect.c's refined skip):
         * uniform-qualified gl_ symbols (legacy fixed-function matrix
         * uniforms injected verbatim) and explicitly-located gl_ symbols
         * (legacy gl_Vertex injected with layout(location = 0)). */
        if (s->name && strncmp(s->name, "gl_", 3) == 0 &&
            !(s->qualifiers & MGL_AST_Q_UNIFORM) &&
            s->location == UINT32_MAX) {
            continue;
        }
        /* GS interface-block instances flatten into per-member VARYING
         * symbols (block_name set); the struct-typed instance symbol
         * itself carries no interface storage. */
        {
            const MGLIRType *it = s->type;
            bool structShaped =
                it->kind == MGLIR_TYPE_STRUCT ||
                (it->kind == MGLIR_TYPE_ARRAY && it->elem_type &&
                 it->elem_type->kind == MGLIR_TYPE_STRUCT);
            if (!s->block_name && structShaped &&
                (s->qualifiers & (MGL_AST_Q_IN | MGL_AST_Q_OUT)) &&
                !(s->qualifiers & (MGL_AST_Q_UNIFORM | MGL_AST_Q_BUFFER))) {
                continue;
            }
        }
        /* Anonymous UBO members (block_name set) are not Metal buffer
         * arguments — nested structs must not become extra UBO slots
         * (that shifts bindings for the real blocks). */
        if (s->block_name && (s->qualifiers & MGL_AST_Q_UNIFORM) &&
            !(s->qualifiers & MGL_AST_Q_BUFFER)) {
            continue;
        }
        VarSym v;
        v.name = s->name;
        v.type = typeFromIR(s->type);
        v.location = s->location;
        v.stream = s->stream;
        v.blockName = s->block_name ? s->block_name : "";
        uint32_t q = s->qualifiers;
        v.isPatch = (q & MGL_AST_Q_PATCH) != 0;
        if (q & MGL_AST_Q_UNIFORM) {
            const MGLIRType *ut = s->type;
            if (ut->kind == MGLIR_TYPE_ARRAY && ut->elem_type)
                ut = ut->elem_type; /* UBO instance array */
            if (ut->kind == MGLIR_TYPE_SAMPLER) {
                v.kind = VarSym::TEXTURE;
            } else if (ut->kind == MGLIR_TYPE_IMAGE) {
                v.kind = VarSym::IMAGE;
            } else if (ut->kind == MGLIR_TYPE_ATOMIC_COUNTER ||
                       (ut->kind == MGLIR_TYPE_ARRAY && ut->elem_type &&
                        ut->elem_type->kind == MGLIR_TYPE_ATOMIC_COUNTER)) {
                v.kind = VarSym::ATOMIC_COUNTER;
            } else if (ut->kind == MGLIR_TYPE_STRUCT &&
                       ut->member_count > 0) {
                v.kind = VarSym::UBO;
            } else {
                v.kind = VarSym::BUFFER;
            }
        } else if (q & MGL_AST_Q_BUFFER) {
            v.kind = VarSym::SSBO;
        } else if (isTCS && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::VARYING;
            if (!v.isPatch && s->type->kind == MGLIR_TYPE_ARRAY &&
                s->type->elem_type) {
                v.type = typeFromIR(s->type->elem_type);
            }
        } else if (isTCS && (q & MGL_AST_Q_OUT)) {
            v.kind = VarSym::OUTPUT;
            if (!v.isPatch && s->type->kind == MGLIR_TYPE_ARRAY &&
                s->type->elem_type) {
                v.type = typeFromIR(s->type->elem_type);
            }
        } else if (isGS && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::VARYING;
            /* Plain gl_in-style input arrays index by input vertex; keep
             * the element type.  Interface-block members (block_name set)
             * keep their array shape: indexing selects the element slot
             * at base location + index. */
            if (!s->block_name &&
                s->type->kind == MGLIR_TYPE_ARRAY && s->type->elem_type) {
                v.type = typeFromIR(s->type->elem_type);
            }
        } else if (isGS && (q & MGL_AST_Q_OUT)) {
            v.kind = VarSym::OUTPUT;
            if (v.stream < 0) {
                v.stream = tu->layout_stream >= 0
                    ? tu->layout_stream : 0;
            }
            if (v.stream < 0 || v.stream >= MGL_AIR_GS_MAX_STREAMS) {
                v.stream = 0;
            }
        } else if (isTES && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::CONTROL_POINT_INPUT;
            if (!v.isPatch && s->type->kind == MGLIR_TYPE_ARRAY &&
                s->type->elem_type) {
                v.type = typeFromIR(s->type->elem_type);
            }
        } else if (isVS && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::ATTR;
        } else if ((isVS || isTES) && (q & MGL_AST_Q_OUT)) {
            v.kind = VarSym::VARYING;
        } else if (!isVS && (q & MGL_AST_Q_IN)) {
            v.kind = VarSym::VARYING;
        } else if (!isVS && (q & MGL_AST_Q_OUT)) {
            v.kind = VarSym::OUTPUT;
        }
        syms.push_back(v);
    }
    {
        uint32_t nextInputLocation = 0;
        uint32_t nextOutputLocation = 0;
        uint32_t nextPatchInputLocation = 0;
        uint32_t nextPatchOutputLocation = 0;
        for (VarSym &v : syms) {
            bool input = ((isTCS || isGS) && v.kind == VarSym::VARYING) ||
                         (isTES && v.kind == VarSym::CONTROL_POINT_INPUT);
            bool output = ((isVS || isTES) && v.kind == VarSym::VARYING) ||
                          ((isTCS || isGS) && v.kind == VarSym::OUTPUT) ||
                          (!isVS && !isTES && !isKernel &&
                           v.kind == VarSym::OUTPUT);
            if (input) {
                uint32_t &next = v.isPatch
                    ? nextPatchInputLocation : nextInputLocation;
                if (v.location == UINT32_MAX) v.location = next;
                /* Interface-block array members span one location per
                 * element (each element is its own record slot). */
                uint32_t span = (!v.blockName.empty() && v.type.isArray())
                    ? v.type.arr : 1u;
                next = std::max(next, v.location + span);
            }
            if (output) {
                uint32_t &next = v.isPatch
                    ? nextPatchOutputLocation : nextOutputLocation;
                if (v.location == UINT32_MAX) v.location = next;
                next = std::max(next, v.location + 1u);
            }
        }
    }
    uint32_t ssboCount = 0, uboCount = 0, acCount = 0, texCount = 0, imageCount = 0;
    for (VarSym &v : syms) {
        if (v.kind == VarSym::SSBO) {
            ssboCount++;
        } else if (v.kind == VarSym::UBO) {
            const MGLIRSymbol *us = findSymbol(&mod, v.name.c_str());
            uboCount += uniformBlockElementCount(us ? us->type : nullptr);
        } else if (v.kind == VarSym::ATOMIC_COUNTER) {
            acCount++;
        } else if (v.kind == VarSym::TEXTURE) {
            texCount += v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
        } else if (v.kind == VarSym::IMAGE) {
            imageCount++;
        }
    }
    auto recordStrideFor = [&](VarSym::Kind kind) -> uint32_t {
        uint32_t stride = MGL_AIR_PER_VERTEX_STRIDE;
        for (const VarSym &v : syms) {
            if (v.kind != kind || v.isPatch ||
                v.location == UINT32_MAX) continue;
            uint64_t end = (uint64_t)MGL_AIR_PER_VERTEX_STRIDE +
                           ((uint64_t)v.location + 1u) * 16u;
            if (end > UINT32_MAX) return 0u;
            stride = std::max(stride, (uint32_t)end);
        }
        return stride;
    };
    auto patchStrideFor = [&](VarSym::Kind kind) -> uint32_t {
        uint32_t stride = 16u;
        for (const VarSym &v : syms) {
            if (v.kind != kind || !v.isPatch ||
                v.location == UINT32_MAX) continue;
            uint64_t end = ((uint64_t)v.location + 1u) * 16u;
            if (end > UINT32_MAX) return 0u;
            stride = std::max(stride, (uint32_t)end);
        }
        return stride;
    };
    const uint32_t stageInputStride = (isTCS || isGS)
        ? recordStrideFor(VarSym::VARYING)
        : isTES ? recordStrideFor(VarSym::CONTROL_POINT_INPUT)
                : MGL_AIR_PER_VERTEX_STRIDE;
    const uint32_t stageOutputStride = (isTCS || isGS || isTESCompute)
        ? recordStrideFor(isTESCompute ? VarSym::VARYING : VarSym::OUTPUT)
        : MGL_AIR_PER_VERTEX_STRIDE;
    const uint32_t tessCaptureStride = isTessCapture
        ? recordStrideFor(VarSym::VARYING)
        : MGL_AIR_PER_VERTEX_STRIDE;
    const uint32_t patchInputStride = isTES
        ? patchStrideFor(VarSym::CONTROL_POINT_INPUT) : 16u;
    const uint32_t patchOutputStride = isTCS
        ? patchStrideFor(VarSym::OUTPUT) : 16u;
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
    const bool usesWorkGroupID =
        isCompute && strstr(esrc, "gl_WorkGroupID") != nullptr;
    const bool usesNumWorkGroups =
        isCompute && strstr(esrc, "gl_NumWorkGroups") != nullptr;
    const bool usesPointSize =
        (isVS || isTES) && strstr(esrc, "gl_PointSize") != nullptr;
    const bool usesClipDistance =
        isVS && !isCapture && !isKernel &&
        strstr(esrc, "gl_ClipDistance") != nullptr;
    const bool usesLayerViewport =
        isVS && (strstr(esrc, "gl_Layer") != nullptr ||
                 strstr(esrc, "gl_ViewportIndex") != nullptr);
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
                    } else {
                        MType outTy = v.type;
                        if (varyingUsesFloatCarrier(outTy, has_gs))
                            outTy = floatCarrierType(outTy);
                        retElems.push_back(llvmType(outTy, ctx));
                    }
                }
                varyings.push_back(&v);
            }
        }
        if (isKernel || isCapture) {
            retTy = llvm::Type::getVoidTy(ctx);
        } else if (isTES) {
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
            /* gl_FragData[i]: array fragment outputs are flattened into
             * per-element color outputs (MSL forbids array members in
             * render-target structs — same constraint as array varyings).
             * Each element becomes a float4 [[color(i)]] member. */
            std::vector<llvm::Type *> fields;
            for (uint32_t i = 0; i < (uint32_t)arrayOutput->type.arr; i++)
                fields.push_back(llvm::FixedVectorType::get(
                    llvm::Type::getFloatTy(ctx), 4));
            if (usesFragDepth)
                fields.push_back(llvm::Type::getFloatTy(ctx));
            retTy = llvm::StructType::get(ctx, fields);
        } else if (fragOutputs.size() > 1u || usesFragDepth) {
            std::vector<llvm::Type *> fields;
            for (VarSym *out : fragOutputs)
                fields.push_back(llvmType(out->type, ctx));
            if (fields.empty())
                fields.push_back(llvm::FixedVectorType::get(
                    llvm::Type::getFloatTy(ctx), 4));
            if (usesFragDepth)
                fields.push_back(llvm::Type::getFloatTy(ctx));
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
    uint32_t attrCount = 0;
    for (VarSym &v : syms)
        if (isVS && v.kind == VarSym::ATTR) attrCount++;
    llvm::StructType *texTy2d =
        llvm::StructType::create(ctx, "struct._texture_2d_t");
    llvm::StructType *texTy2dArray =
        llvm::StructType::create(ctx, "struct._texture_2d_array_t");
    llvm::StructType *texTy3d =
        llvm::StructType::create(ctx, "struct._texture_3d_t");
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
        for (VarSym &v : syms)
            if (v.kind == VarSym::ATTR)
                paramTys.push_back(llvmType(v.type, ctx));
    }
    if ((isVS || isTES || isKernel) && hasBuffer)
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    for (VarSym &v : syms)
        if (v.kind == VarSym::SSBO || v.kind == VarSym::UBO ||
            v.kind == VarSym::ATOMIC_COUNTER) {
            const MGLIRSymbol *us = findSymbol(&mod, v.name.c_str());
            uint32_t uelems = v.kind == VarSym::UBO
                ? uniformBlockElementCount(us ? us->type : nullptr) : 1u;
            for (uint32_t k = 0; k < uelems; k++)
                paramTys.push_back(
                    llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        }
    if (needsBufferSizeBuffer)
        paramTys.push_back(llvm::Type::getInt32Ty(ctx)->getPointerTo(2));
    for (VarSym &v : syms) {
        if (v.kind != VarSym::TEXTURE) continue;
        const MGLIRSymbol *ts = findSymbol(&mod, v.name.c_str());
        MGLIRTexKind tk = ts && ts->type->kind == MGLIR_TYPE_SAMPLER
            ? ts->type->tex_kind : MGLIR_TEX_2D;
        llvm::StructType *tt = (tk == MGLIR_TEX_3D) ? texTy3d
                             : (tk == MGLIR_TEX_2D_ARRAY) ? texTy2dArray
                             : (tk == MGLIR_TEX_BUFFER) ? texTyBuf : texTy2d;
        uint32_t elements = v.type.arr > 0 ? (uint32_t)v.type.arr : 1u;
        for (uint32_t k = 0; k < elements; k++) {
            paramTys.push_back(tt->getPointerTo(1));
            paramTys.push_back(smpTy->getPointerTo(2));
        }
    }
    for (VarSym &v : syms) {
        if (v.kind != VarSym::IMAGE) continue;
        const MGLIRSymbol *ts = findSymbol(&mod, v.name.c_str());
        MGLIRTexKind tk = ts && ts->type->kind == MGLIR_TYPE_IMAGE
            ? ts->type->tex_kind : MGLIR_TEX_2D;
        llvm::StructType *tt = (tk == MGLIR_TEX_3D) ? texTy3d
                             : (tk == MGLIR_TEX_2D_ARRAY) ? texTy2dArray
                             : texTy2d;
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
            } else {
                paramTys.push_back(llvmType(
                    varyingUsesFloatCarrier(v.type, has_gs)
                        ? floatCarrierType(v.type) : v.type, ctx));
            }
        }
    }
    if (isVS && isCapture) {
        /* XFB capture variant: attributes trail all buffers (see above). */
        for (VarSym &v : syms)
            if (v.kind == VarSym::ATTR)
                paramTys.push_back(llvmType(v.type, ctx));
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
    } else if (isTES) {
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    }
    if (usesCullDistance) {
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    } else if (isCullCapture || isTessCapture) {
        paramTys.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo(1));
    }
    if (isVS) {
        paramTys.push_back(llvm::Type::getInt32Ty(ctx));
        paramTys.push_back(llvm::Type::getInt32Ty(ctx));
        paramTys.push_back(llvm::Type::getInt32Ty(ctx));
    }
    else if (isKernel) {
        paramTys.push_back(llvm::FixedVectorType::get(
            llvm::Type::getInt32Ty(ctx), 3));
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
        if (usesSampleID)
            paramTys.push_back(llvm::Type::getInt32Ty(ctx));
    }
    if (isTESCompute)
        paramTys.push_back(llvm::FixedVectorType::get(
            llvm::Type::getInt32Ty(ctx), 3));
    llvm::FunctionType *ft = llvm::FunctionType::get(retTy, paramTys, false);
    llvm::Function *fn = llvm::Function::Create(
        ft, llvm::Function::ExternalLinkage, "main", &module);
    fn->setDoesNotThrow();
    llvm::Function *controlPointGetter = nullptr;
    if (isTES && !isTESCompute) {
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
            fn->addParamAttr(ssboIdx++, llvm::Attribute::AttrKind::NoAlias);
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
    cg.has_gs = has_gs;
    cg.isCompute = isCompute || isTCS || isGS;
    cg.isTessControl = isTCS;
    cg.isTessEval = isTES;
    cg.isGeometry = isGS;
    cg.isTESCompute = isTESCompute;
    cg.controlPointGetter = controlPointGetter;
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
            if (v.kind == VarSym::ATTR)
                cg.lvalues[v.name] = fn->getArg(argSlot++);
        }
    }
    if ((isVS || isTES || isKernel) && hasBuffer)
        cg.bufferPtr = fn->getArg(argSlot++);
    for (VarSym &v : syms) {
        if (v.kind != VarSym::SSBO) continue;
        cg.ssboPtrs[v.name] = fn->getArg(argSlot++);
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
        cg.texValues[v.name] = fn->getArg(argSlot++);
    }
    {
        uint32_t location = (isCapture ? 1u : 0u) +
            (((isVS || isTES || isKernel) && hasBuffer) ? 1u : 0u) +
            (isCapture ? 0u : attrCount);
        for (VarSym &v : syms) {
            if (v.kind != VarSym::SSBO) continue;
            cg.ssboSlots[v.name] = location++;
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
            if (v.kind == VarSym::VARYING && v.type.isArray()) {
                /* Flattened (FS stage-in): N scalar args assembled into a
                 * single aggregate lvalue so the read paths (readIndexChain
                 * / swizzles) keep working unchanged. */
                MType el = v.type;
                el.arr = 0;
                llvm::Type *aggTy = llvmType(v.type, ctx);
                llvm::Value *agg = llvm::UndefValue::get(aggTy);
                uint32_t n = (uint32_t)v.type.arr;
                for (uint32_t k = 0; k < n; k++) {
                    llvm::Value *arg = fn->getArg(argSlot++);
                    if (varyingUsesFloatCarrier(el, has_gs)) {
                        arg = decodeFloatCarrier(cg, arg, el.scalar,
                                                 llvmType(el, ctx));
                    }
                    agg = cg.b->CreateInsertValue(agg, arg, k);
                }
                cg.lvalues[v.name] = agg;
                (void)el;
            } else {
                llvm::Value *arg = fn->getArg(argSlot++);
                if (varyingUsesFloatCarrier(v.type, has_gs)) {
                    arg = decodeFloatCarrier(cg, arg, v.type.scalar,
                                             llvmType(v.type, ctx));
                }
                cg.lvalues[v.name] = arg;
            }
        }
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
            cg.lvalues[v.name] =
                llvm::UndefValue::get(llvmType(v.type, ctx));
        }
    }
    if (isTES && !isTESCompute) {
        cg.stageInPtr = fn->getArg(argSlot++);
        cg.indirectPtr = fn->getArg(argSlot++);
        cg.captureBuf = fn->getArg(argSlot++);
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
    if (isVS) {
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
        if (usesSampleID)
            cg.lvalues["gl_SampleID"] = fn->getArg(argSlot++);
    }
    if (usesFragDepth)
        cg.hasFragDepth = true;
    if (usesClipDistance) {
        cg.usesClipDistance = true;
        /* Indexed writes (gl_ClipDistance[i] = v) need the aggregate
         * lvalue pre-registered (see the array-varying fix). */
        cg.lvalues["gl_ClipDistance"] = defaultClipDistances(cg);
    }
    if (!isVS && !isTES && !isKernel) {
        /* gl_FragData[i]: indexed writes need the aggregate lvalue
         * pre-registered (same fix as the array-varying aggregates). */
        for (VarSym &v : syms) {
            if (v.kind == VarSym::OUTPUT && v.type.isArray()) {
                cg.lvalues[v.name] = llvm::UndefValue::get(
                    llvm::ArrayType::get(
                        llvm::FixedVectorType::get(
                            llvm::Type::getFloatTy(ctx), 4),
                        v.type.arr));
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

    /* User-defined functions (fog helpers etc.): create the LLVM
     * functions first so calls (including recursion) resolve, then emit
     * their bodies. */
    std::map<std::string, llvm::Function *> userFns;
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        MGLDecl *d = tu->decls[i];
        if (!d->name || !d->body || strcmp(d->name, "main") == 0) continue;
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
        llvm::Type *rt = fs->return_type
            ? llvmType(typeFromIR(fs->return_type), ctx)
            : llvm::Type::getVoidTy(ctx);
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
                pts.push_back(llvmType(typeFromIR(pt), ctx));
            }
        }
        /* Hidden trailing arguments: UBO values (a pointer for a scalar block,
         * an aggregate of pointers for an instance array) and SSBO pointers,
         * so user functions can read global blocks of their own. */
        for (const auto &kv : cg.uboPtrs)
            pts.push_back(kv.second->getType());
        for (const auto &kv : cg.ssboPtrs)
            pts.push_back(llvm::Type::getInt8PtrTy(ctx, 1));
        for (const auto &kv : cg.acPtrs)
            pts.push_back(llvm::Type::getInt8PtrTy(ctx, 1));
        if (cg.bufferSizePtr)
            pts.push_back(llvm::Type::getInt32PtrTy(ctx, 2));
        if (isGS) {
            for (int hidden = 0; hidden < 5; hidden++)
                pts.push_back(llvm::Type::getInt8PtrTy(ctx, 1));
            for (int hidden = 0; hidden < 3; hidden++)
                pts.push_back(llvm::Type::getInt32Ty(ctx));
        }
        llvm::Function *f = llvm::Function::Create(
            llvm::FunctionType::get(rt, pts, false),
            llvm::Function::ExternalLinkage,
            (std::string("mgl_fn_") + d->name + "_" +
             std::to_string(fs->param_count)),
            &module);
        userFns[std::string(d->name) + "#" + std::to_string(fs->param_count)] =
            f;
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
        fc.acPtrs = cg.acPtrs;
        fc.acSlots = cg.acSlots;
        fc.uboPtrs = cg.uboPtrs;
        fc.texValues = cg.texValues;
        fc.smpValues = cg.smpValues;
        fc.bufferOffsets = cg.bufferOffsets;
        fc.position = cg.position;
        fc.userFns = &userFns;
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
            for (const auto &kv : cg.ssboPtrs)
                fc.ssboPtrs[kv.first] = f->getArg(hidx++);
            for (const auto &kv : cg.acPtrs)
                fc.acPtrs[kv.first] = f->getArg(hidx++);
            if (cg.bufferSizePtr)
                fc.bufferSizePtr = f->getArg(hidx++);
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
        }
        emitStmt(fc, d->body, &mod, &flocals);
        if (fc.err == 1) {
            cg.err = 1;
            cg.errmsg = std::string("codegen: in function '") + d->name +
                        "': " + fc.errmsg;
            snprintf(err_buf, err_cap, "%s", cg.errmsg.c_str());
            mglIRModuleDestroy(&mod);
            mglGLSLTranslationUnitDestroy(tu);
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
    std::map<std::string, MType> locals;
    /* Global initializers (const arrays/scalars and other file-scope
     * variables with constant initializers such as `int counter = 0;`):
     * evaluate into cg.lvalues before main so references fold to SSA
     * values instead of reading undef/unbound slots. */
    for (uint32_t i = 0; i < tu->decl_count; i++) {
        MGLDecl *d = tu->decls[i];
        if (!d || !d->name || d->body || !d->init) continue;
        const MGLIRSymbol *gs = findSymbol(&mod, d->name);
        if (!gs || gs->is_function) continue;
        if (gs->qualifiers & (MGL_AST_Q_UNIFORM | MGL_AST_Q_BUFFER |
                              MGL_AST_Q_IN | MGL_AST_Q_OUT |
                              MGL_AST_Q_INOUT | MGL_AST_Q_SHARED)) {
            continue;
        }
        MType gt = typeFromIR(gs->type);
        const bool isConst = (gs->qualifiers & MGL_AST_Q_CONST) != 0;
        if (gt.isArray() && !isConst) continue;
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
                        b.getInt64(12)));
        llvm::Type *halfTy = llvm::Type::getHalfTy(ctx);
        auto loadHalf = [&](unsigned i) -> llvm::Value * {
            llvm::Value *p = b.CreateBitCast(
                b.CreateGEP(b.getInt8Ty(), factorBase, b.getInt64(2 * i)),
                halfTy->getPointerTo(1));
            return b.CreateFPExt(
                b.CreateAlignedLoad(halfTy, p, llvm::Align(2)), f32);
        };
        auto ceilClamp = [&](llvm::Value *v, float fallback) -> llvm::Value * {
            llvm::Value *fb = llvm::ConstantFP::get(f32, fallback);
            llvm::Value *use = b.CreateSelect(
                b.CreateFCmpOGT(v, fb), v, fb);
            return b.CreateFPToUI(
                b.CreateIntrinsic(llvm::Intrinsic::ceil, {f32}, {use}),
                b.getInt32Ty());
        };
        auto toF = [&](llvm::Value *i) -> llvm::Value * {
            return b.CreateSIToFP(i, f32);
        };
        /* GL 4.6 §11.2.2.2: the subdivision count honours the TES layout
         * spacing declaration — integer keeps ceil(level), fractional_even
         * rounds up to the next even (min 2), fractional_odd to the next
         * odd.  isolines are exempt (spacing applies only to triangles and
         * quads). */
        auto roundLevel = [&](llvm::Value *ceilVal) -> llvm::Value * {
            if (tu->layout_spacing == MGL_AST_SPACING_FRACTIONAL_EVEN) {
                llvm::Value *odd = b.CreateAnd(ceilVal, b.getInt32(1));
                llvm::Value *even = b.CreateAdd(
                    ceilVal,
                    b.CreateSelect(b.CreateICmpNE(odd, b.getInt32(0)),
                                   b.getInt32(1), b.getInt32(0)));
                return b.CreateSelect(
                    b.CreateICmpULT(even, b.getInt32(2)), b.getInt32(2),
                    even);
            }
            if (tu->layout_spacing == MGL_AST_SPACING_FRACTIONAL_ODD) {
                llvm::Value *odd = b.CreateAnd(ceilVal, b.getInt32(1));
                return b.CreateAdd(
                    ceilVal,
                    b.CreateSelect(b.CreateICmpEQ(odd, b.getInt32(0)),
                                   b.getInt32(1), b.getInt32(0)));
            }
            return ceilVal; /* integer / default */
        };
        llvm::Value *u = nullptr, *v = nullptr;
        if (tu->layout_primitive == MGL_AST_TES_ISOLINES) {
            /* GL 4.6 §11.2.2.3: outer[0] selects the number of isolines n
             * at v = {0, 1/n, ..., (n-1)/n}; outer[1] selects the m
             * segments per row.  Each segment is emitted as one line
             * primitive with two vertices at u = {seg/m, (seg+1)/m}, so
             * each row contributes 2*m vertices. */
            llvm::Value *n = ceilClamp(loadHalf(0), 1.0f);
            llvm::Value *m = ceilClamp(loadHalf(1), 1.0f);
            llvm::Value *perLine = b.CreateMul(m, b.getInt32(2));
            llvm::Value *lineIdx = b.CreateUDiv(innerId, perLine);
            llvm::Value *t = b.CreateURem(innerId, perLine);
            llvm::Value *seg = b.CreateUDiv(t, b.getInt32(2));
            llvm::Value *vtx = b.CreateURem(t, b.getInt32(2));
            u = b.CreateFDiv(toF(b.CreateAdd(seg, vtx)), toF(m));
            v = b.CreateFDiv(toF(lineIdx), toF(n));
        } else if (tu->layout_primitive == MGL_AST_TES_QUADS) {
            /* point_mode quads: one point at each inner grid cell centre. */
            llvm::Value *nx = roundLevel(ceilClamp(loadHalf(4), 1.0f));
            llvm::Value *ny = roundLevel(ceilClamp(loadHalf(5), 1.0f));
            llvm::Value *i = b.CreateURem(innerId, nx);
            llvm::Value *j = b.CreateUDiv(innerId, nx);
            u = b.CreateFDiv(
                b.CreateFAdd(toF(i), llvm::ConstantFP::get(f32, 0.5)),
                toF(nx));
            v = b.CreateFDiv(
                b.CreateFAdd(toF(j), llvm::ConstantFP::get(f32, 0.5)),
                toF(ny));
        } else {
            /* point_mode triangles: one point per inner grid cell (n*n
             * cells), at the up-triangle centroid. */
            llvm::Value *n = roundLevel(ceilClamp(loadHalf(4), 1.0f));
            llvm::Value *i = b.CreateURem(innerId, n);
            llvm::Value *j = b.CreateUDiv(innerId, n);
            llvm::Value *three = b.getInt32(3);
            u = b.CreateFDiv(
                toF(b.CreateAdd(b.CreateMul(three, i), b.getInt32(1))),
                toF(b.CreateMul(three, n)));
            v = b.CreateFDiv(
                toF(b.CreateAdd(b.CreateMul(three, j), b.getInt32(1))),
                toF(b.CreateMul(three, n)));
        }
        llvm::Value *uv = llvm::UndefValue::get(llvm::FixedVectorType::get(
            f32, 3));
        llvm::Value *w = (tu->layout_primitive == MGL_AST_TES_TRIANGLES)
            ? b.CreateFSub(
                  llvm::ConstantFP::get(f32, 1.0),
                  b.CreateFAdd(u, v))
            : llvm::ConstantFP::get(f32, 0.0);
        cg.tessCoord = b.CreateInsertElement(
            b.CreateInsertElement(
                b.CreateInsertElement(uv, u, b.getInt32(0)), v, b.getInt32(1)),
            w, b.getInt32(2));
    }
    emitStmt(cg, mainDecl->body, &mod, &locals);

    if (isTCS && cg.tessFactorPtr && cg.invocationPos && cg.patchPos &&
        !b.GetInsertBlock()->getTerminator()) {
        /* Metal's tessellation-factor record is six half values: four edge
         * factors followed by two inner factors.  Only invocation zero owns
         * the patch-wide factors; all other TCS invocations skip the write. */
        llvm::BasicBlock *writeBB = llvm::BasicBlock::Create(
            ctx, "tcs_write_factors", fn);
        llvm::BasicBlock *doneBB = llvm::BasicBlock::Create(
            ctx, "tcs_factors_done", fn);
        llvm::Value *inv = b.CreateExtractElement(cg.invocationPos,
                                                   b.getInt32(0));
        llvm::Value *isZero = b.CreateICmpEQ(
            inv, llvm::ConstantInt::get(inv->getType(), 0));
        b.CreateCondBr(isZero, writeBB, doneBB);
        b.SetInsertPoint(writeBB);
        llvm::Value *patch =
            cg.isTessControl && cg.workGroupPos
                ? cg.b->CreateExtractElement(cg.workGroupPos, cg.b->getInt32(0))
                : cg.b->CreateExtractElement(cg.patchPos, cg.b->getInt32(0));
        llvm::Value *factorOff = b.CreateMul(
            b.CreateZExt(patch, b.getInt64Ty()), b.getInt64(12));
        llvm::Value *factorBase = b.CreateGEP(
            b.getInt8Ty(), cg.tessFactorPtr, factorOff);
        llvm::Type *halfTy = llvm::Type::getHalfTy(ctx);
        auto factor = [&](const char *name, unsigned count, unsigned index,
                          float fallback) {
            llvm::Value *arr = cg.lvalues.count(name)
                ? cg.lvalues[name]
                : llvm::UndefValue::get(llvm::ArrayType::get(
                      llvm::Type::getFloatTy(ctx), count));
            llvm::Value *v = b.CreateExtractValue(arr, index);
            if (llvm::isa<llvm::UndefValue>(v))
                v = llvm::ConstantFP::get(llvm::Type::getFloatTy(ctx), fallback);
            return v;
        };
        for (unsigned i = 0; i < 4; i++) {
            llvm::Value *p = b.CreateGEP(b.getInt8Ty(), factorBase,
                                         b.getInt64(i * 2));
            p = b.CreateBitCast(p, halfTy->getPointerTo(1));
            b.CreateAlignedStore(b.CreateFPTrunc(
                factor("gl_TessLevelOuter", 4, i, 1.0f), halfTy), p,
                llvm::Align(2));
        }
        for (unsigned i = 0; i < 2; i++) {
            llvm::Value *p = b.CreateGEP(b.getInt8Ty(), factorBase,
                                         b.getInt64(8 + i * 2));
            p = b.CreateBitCast(p, halfTy->getPointerTo(1));
            b.CreateAlignedStore(b.CreateFPTrunc(
                factor("gl_TessLevelInner", 2, i, 1.0f), halfTy), p,
                llvm::Align(2));
        }
        b.CreateBr(doneBB);
        b.SetInsertPoint(doneBB);
    }

    if (isTESCompute && cg.geometryOutputPtr && cg.geometryWorkItemId &&
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
            ? cg.lvalues["gl_PointSize"] : llvm::UndefValue::get(
                llvm::Type::getFloatTy(ctx));
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
        mglIRModuleDestroy(&mod);
        mglGLSLTranslationUnitDestroy(tu);
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
            if (isTessCapture) {
                llvm::Value *recordBase = b.CreateGEP(
                    b.getInt8Ty(), cg.captureBuf,
                    b.CreateMul(vid, b.getInt64(recStride)));
                for (VarSym *varying : cg.varyings) {
                    if (!varying || varying->location == UINT32_MAX) continue;
                    /* Plain stage-in arrays index by primitive vertex, so
                     * each per-vertex record stores element 0 only.
                     * Interface-block array members carry one distinct
                     * value per element: store each element in its own
                     * consecutive location slot. */
                    MType mt = varying->type;
                    const bool wasArray = mt.isArray() && mt.arr > 0;
                    const bool blockArray =
                        wasArray && !varying->blockName.empty();
                    if (wasArray) mt.arr = 0;
                    llvm::Type *varyingTy = llvmType(mt, ctx);
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
                            llvm::Value *vp = b.CreateGEP(
                                b.getInt8Ty(), recordBase,
                                b.getInt64(MGL_AIR_PER_VERTEX_STRIDE +
                                           (varying->location + ei) * 16u));
                            vp = b.CreateBitCast(
                                vp, varyingTy->getPointerTo(1));
                            b.CreateAlignedStore(elem, vp, llvm::Align(4));
                        }
                        continue;
                    }
                    llvm::Value *vp = b.CreateGEP(
                        b.getInt8Ty(), recordBase,
                        b.getInt64(MGL_AIR_PER_VERTEX_STRIDE +
                                   varying->location * 16u));
                    vp = b.CreateBitCast(vp, varyingTy->getPointerTo(1));
                    b.CreateAlignedStore(value, vp, llvm::Align(4));
                }
            }
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
            if (attrLoc == UINT32_MAX) {
                uint32_t want = airAttribLocation(v.name.c_str(),
                                                  attrib_names);
                attrLoc = (want != UINT32_MAX) ? want : nextFreeAttrLoc;
            }
            std::vector<llvm::Metadata *> elems = {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.vertex_input"),
                llvm::MDString::get(ctx, "air.location_index"),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), attrLoc)),
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), 1)),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, mslTypeName(v.type)),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, v.name)};
            argNodes.push_back(llvm::MDNode::get(ctx, elems));
            nextFreeAttrLoc = std::max(nextFreeAttrLoc, attrLoc + 1u);
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
            for (VarSym &v : syms)
                if (v.kind == VarSym::IMAGE)
                    idx++;
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
        uint32_t loc = (isCapture ? 1 : 0) +
                       ((isVS || isTES || isKernel) ? (hasBuffer ? 1 : 0) : 0) +
                       (isCapture ? 0 : attrCount) + userBufferLocationBase;
        uint32_t ssboArg = (isCapture ? 1 : 0) +
                           ((isVS || isTES || isKernel) ? (hasBuffer ? 1 : 0) : 0) +
                           (isCapture ? 0 : attrCount);
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
                llvm::MDString::get(ctx, isVS ? "air.read" : "air.read_write"),
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
    /* Uniform blocks: independent read-only device buffers. */
    {
        uint32_t loc = (isCapture ? 1 : 0) +
                       ((isVS || isTES || isKernel) ? (hasBuffer ? 1 : 0) : 0) +
                       ssboCount + (isCapture ? 0 : attrCount) +
                       userBufferLocationBase;
        uint32_t uboArg = loc;
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
        uint32_t loc = (isCapture ? 1 : 0) +
                       ((isVS || isTES || isKernel) ? (hasBuffer ? 1 : 0) : 0) +
                       ssboCount + uboCount + (isCapture ? 0 : attrCount) +
                       userBufferLocationBase;
        uint32_t acArg = loc;
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
            const MGLIRSymbol *tss = findSymbol(&mod, v.name.c_str());
            const MGLIRType *samplerType = tss ? tss->type : nullptr;
            while (samplerType && samplerType->kind == MGLIR_TYPE_ARRAY)
                samplerType = samplerType->elem_type;
            bool is3d = samplerType &&
                        samplerType->kind == MGLIR_TYPE_SAMPLER &&
                        samplerType->tex_kind == MGLIR_TEX_3D;
            bool is2dArray = samplerType &&
                             samplerType->kind == MGLIR_TYPE_SAMPLER &&
                             samplerType->tex_kind == MGLIR_TEX_2D_ARRAY;
            const char *texelName = "float";
            if (samplerType && samplerType->kind == MGLIR_TYPE_SAMPLER) {
                if (samplerType->tex_storage == MGLIR_SCALAR_INT)
                    texelName = "int";
                else if (samplerType->tex_storage == MGLIR_SCALAR_UINT)
                    texelName = "uint";
            }
            std::string sampledType = is3d ? "texture3d<"
                                  : is2dArray ? "texture2d_array<"
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
            const MGLIRSymbol *iss = findSymbol(&mod, v.name.c_str());
            bool is3d = iss && iss->type->kind == MGLIR_TYPE_IMAGE &&
                        iss->type->tex_kind == MGLIR_TEX_3D;
            bool is2dArray = iss && iss->type->kind == MGLIR_TYPE_IMAGE &&
                             iss->type->tex_kind == MGLIR_TEX_2D_ARRAY;
            const char *imageType = is3d
                ? "texture3d<float, access::read_write>"
                : "texture2d<float, access::read_write>";
            if (is2dArray) {
                imageType = iss->type->tex_storage == MGLIR_SCALAR_INT
                    ? "texture2d_array<int, read_write>"
                    : iss->type->tex_storage == MGLIR_SCALAR_UINT
                        ? "texture2d_array<uint, read_write>"
                        : "texture2d_array<float, read_write>";
            } else if (!is3d && iss) {
                if (iss->type->tex_storage == MGLIR_SCALAR_INT)
                    imageType = "texture2d<int, access::read_write>";
                else if (iss->type->tex_storage == MGLIR_SCALAR_UINT)
                    imageType = "texture2d<uint, access::read_write>";
            }
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
                llvm::MDString::get(ctx, v.name)}));
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
        for (VarSym &v : syms) {
            if (v.kind != VarSym::ATTR) continue;
            uint32_t want = airAttribLocation(v.name.c_str(), attrib_names);
            if (want != UINT32_MAX) attrLoc = want;
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
    if (isVS) {
        /* Vertex attribute metadata already emitted above. */
    } else if (isTES && !isTESCompute) {
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
    } else if (!isKernel) {
        auto emitFSVarying = [&](const std::string &tagName,
                                 const MType &mt, uint32_t argIdx) {
            bool carrier = varyingUsesFloatCarrier(mt, has_gs);
            const MType &iface = carrier ? floatCarrierType(mt) : mt;
            std::vector<llvm::Metadata *> elems = {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), argIdx)),
                llvm::MDString::get(ctx, "air.fragment_input"),
                llvm::MDString::get(ctx,
                                    airGenerated(tagName, iface)),
                llvm::MDString::get(ctx,
                                    carrier || !scalarIsFloat(mt.scalar)
                                        ? "air.flat"
                                        : "air.center"),
                llvm::MDString::get(ctx,
                                    carrier || !scalarIsFloat(mt.scalar)
                                        ? "air.no_perspective"
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
                std::string base = v.name;
                for (uint32_t k = 0; k < n; k++) {
                    std::string elName =
                        base + "_elm" + std::to_string(k);
                    emitFSVarying(elName, el, mArgSlot++);
                }
            } else {
                emitFSVarying(v.name, v.type, mArgSlot++);
            }
        }
        if (usesFragCoord || usesFrontFacing || usesPointCoord ||
            usesPrimitiveId || usesLayer || usesViewportIndex ||
            usesSampleID) {
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
                 * written by the passthrough vertex function, which declares
                 * its output as vertex_output + generated(<name>f).  The FS
                 * input must pair with that exact generated() name — a
                 * location-indexed stage_input here crashes Apple's compiler
                 * once the input is actually read. */
                /* Base name must match the passthrough vertex function's
                 * output variable ("mgl_primitive_id", see
                 * MGLRenderer+RenderPass.m) so the generated() names pair. */
                MType carrierType;
                argNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                        llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                    llvm::MDString::get(ctx, "air.fragment_input"),
                    llvm::MDString::get(
                        ctx, airGenerated("mgl_primitive_id", carrierType)),
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
        if (usesSampleID) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot++)),
                llvm::MDString::get(ctx, "air.sample_id"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "gl_SampleID")}));
        }
    }
    if (isKernel) {
        /* Kernel thread position: [[thread_position_in_grid]] as uint3. */
        argNodes.push_back(llvm::MDNode::get(ctx, {
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), mArgSlot)),
            llvm::MDString::get(ctx, isTCS ? "air.thread_position_in_threadgroup"
                                           : "air.thread_position_in_grid"),
            llvm::MDString::get(ctx, "air.arg_type_name"),
            llvm::MDString::get(ctx, "uint3"),
            llvm::MDString::get(ctx, "air.arg_name"),
            llvm::MDString::get(ctx, isTCS ? "thread_position_in_threadgroup"
                                           : "thread_position_in_grid")}));
        if (isTCS || isTESCompute || usesWorkGroupID) {
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), mArgSlot + 1)),
                llvm::MDString::get(ctx, "air.threadgroup_position_in_grid"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint3"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "threadgroup_position_in_grid")}));
        }
        if (usesNumWorkGroups) {
            uint32_t numWorkGroupsSlot = mArgSlot + 1u;
            if (isTCS || isTESCompute || usesWorkGroupID)
                numWorkGroupsSlot++;
            argNodes.push_back(llvm::MDNode::get(ctx, {
                llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                    llvm::Type::getInt32Ty(ctx), numWorkGroupsSlot)),
                llvm::MDString::get(ctx, "air.threadgroups_per_grid"),
                llvm::MDString::get(ctx, "air.arg_type_name"),
                llvm::MDString::get(ctx, "uint3"),
                llvm::MDString::get(ctx, "air.arg_name"),
                llvm::MDString::get(ctx, "threadgroups_per_grid")}));
        }
    }

    std::vector<llvm::Metadata *> outNodes;   /* outputs / render targets */
    if ((isVS || (isTES && !isTESCompute)) && !isCapture) {
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
                std::string base = v->name;
                uint32_t n = (uint32_t)v->type.arr;
                for (uint32_t k = 0; k < n; k++) {
                    std::string elName =
                        base + "_elm" + std::to_string(k);
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
            } else {
                MType outTy = varyingUsesFloatCarrier(v->type, has_gs)
                    ? floatCarrierType(v->type) : v->type;
                outNodes.push_back(llvm::MDNode::get(ctx, {
                    llvm::MDString::get(ctx, "air.vertex_output"),
                    llvm::MDString::get(ctx, airGenerated(v->name, outTy)),
                    llvm::MDString::get(ctx, "air.arg_type_name"),
                    llvm::MDString::get(ctx, mslTypeName(outTy)),
                    llvm::MDString::get(ctx, "air.arg_name"),
                    llvm::MDString::get(ctx, v->name)}));
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
            /* gl_FragData[i]: one render_target node per element with
             * (member index, color index) constants. */
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
                    llvm::MDString::get(ctx, "float4"),
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
    }

    if (usesCullDistance) {
        llvm::Type *i32 = llvm::Type::getInt32Ty(ctx);
        uint32_t cullBufferArg = (uint32_t)paramTys.size() - 5u;
        uint32_t cullParamsArg = cullBufferArg + 1u;
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

    if (isVS) {
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
    if (isTES && !isTESCompute) {
        stageElems.push_back(llvm::MDNode::get(ctx, {
            llvm::MDString::get(ctx, "air.patch"),
            llvm::MDString::get(
                ctx, tu->layout_primitive == MGL_AST_TES_QUADS
                         ? "quad" : "triangle"),
            llvm::MDString::get(ctx, "air.patch_control_point"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx), 0))}));
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

    if (getenv("MGL_DUMP_IR"))
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
    if (isTES && !isTESCompute) {
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
    return compileGLSLImpl(src, stage, 0, /*has_gs=*/false, nullptr, 0u,
                           metallib_out,
                           size_out, err_buf, err_cap);
}

/* XFB capture variant: the vertex stage writes its full output record
 * (position + varyings) into a device buffer at location 29 with
 * rasterization disabled (the capture variant of the mglShaderCompileGLSL
 * compile entry). */
extern "C" int mglShaderCompileGLSLCapture(const char *src,
                                           unsigned char **metallib_out,
                                           size_t *size_out, char *err_buf,
                                           size_t err_cap) {
    return compileGLSLImpl(src, MGL_STAGE_VERTEX, 1, /*has_gs=*/false, nullptr,
                           0u,
                           metallib_out, size_out, err_buf, err_cap);
}

extern "C" int mglShaderCompileGLSLTessCapture(
    const char *src, unsigned char **metallib_out, size_t *size_out,
    char *err_buf, size_t err_cap) {
    return compileGLSLImpl(src, MGL_STAGE_VERTEX, 2, /*has_gs=*/false, nullptr,
                           0u,
                           metallib_out, size_out, err_buf, err_cap);
}

extern "C" int mglShaderCompileGLSLCullDistanceCapture(
    const char *src, unsigned char **metallib_out, size_t *size_out,
    char *err_buf, size_t err_cap) {
    return compileGLSLImpl(src, MGL_STAGE_VERTEX, 3, /*has_gs=*/false, nullptr,
                           0u,
                           metallib_out, size_out, err_buf, err_cap);
}

static uint32_t reflectCullDistanceCount(const char *src)
{
    if (!src) return 0;
    const char *p = src;
    uint32_t count = 0;
    while ((p = strstr(p, "gl_CullDistance")) != nullptr) {
        p += strlen("gl_CullDistance");
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (*p != '[') {
            count = 8;
            continue;
        }
        ++p;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p < '0' || *p > '9') {
            count = 8;
            continue;
        }
        char *end = nullptr;
        unsigned long index = strtoul(p, &end, 10);
        uint32_t reflected = index < 8
            ? static_cast<uint32_t>(index + 1)
            : (index == 8 ? 8u : 0u);
        if (count < reflected) count = reflected;
        p = end ? end : p;
    }
    return count;
}

static void fillStageInfo(const MGLTranslationUnit *tu,
                          const MGLIRModule *mod, int stage,
                          const char *src, MGLAIRStageInfo *stage_info) {
    memset(stage_info, 0, sizeof(*stage_info));
    stage_info->needs_runtime_array_size_buffer =
        translationUnitUsesRuntimeArrayLength(tu, mod) ? 1u : 0u;
    if (stage != MGL_STAGE_FRAGMENT && stage != MGL_STAGE_COMPUTE && src &&
        strstr(src, "gl_CullDistance") != nullptr) {
        stage_info->uses_cull_distance = 1u;
        stage_info->cull_distance_count = reflectCullDistanceCount(src);
        if (stage_info->cull_distance_count == 0)
            stage_info->cull_distance_count = 8u;
    }
    if (stage == MGL_STAGE_TESS_CONTROL && tu->layout_vertices > 0)
        stage_info->tess_control_output_vertices =
            static_cast<uint32_t>(tu->layout_vertices);
    if (stage == MGL_STAGE_TESS_EVALUATION) {
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
    uint32_t flags, char *err_buf, size_t err_cap) {
    bool has_gs = (flags & MGL_AIR_COMPILE_HAS_GEOMETRY_SHADER) != 0;
    if (!src || !metallib_out || !size_out) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "bad args");
        return -1;
    }
    /* Legacy GLSL frontend wiring: translate pre-3.30 constructs before
     * parsing (compileGLSLImpl re-parses the same translated source). */
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
    MGLIRModule mod;
    memset(&mod, 0, sizeof mod);
    MGLSemaError *errors = nullptr;
    uint32_t error_count = 0;
    int hard = mglGLSLSemanticCheck(tu, stage, &mod, &errors, &error_count);
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

    uint32_t tessPatchVertices = 0u;
    if (stage_info) {
        tessPatchVertices = stage_info->tess_patch_vertices;
        fillStageInfo(tu, &mod, stage, esrc, stage_info);
        stage_info->tess_patch_vertices = tessPatchVertices;
    }

    if (lists)
        mglAirReflectModule(&mod, stage, attrib_names, lists, err_buf,
                            err_cap);
    mglIRModuleDestroy(&mod);
    mglGLSLTranslationUnitDestroy(tu);

    return compileGLSLImpl(esrc, stage, 0, has_gs, attrib_names,
                           tessPatchVertices,
                           metallib_out, size_out, err_buf, err_cap);
}

extern "C" int mglAirCompileGLSLWithReflectInfo(
    const char *src, int stage, const char *const *attrib_names,
    unsigned char **metallib_out, size_t *size_out,
    MGLShaderResourceList lists[MGL_MAX_SHADER_RESOURCES], MGLAIRStageInfo *stage_info,
    char *err_buf, size_t err_cap) {
    return mglAirCompileGLSLWithReflectInfoEx(
        src, stage, attrib_names, metallib_out, size_out, lists,
        stage_info, 0u, err_buf, err_cap);
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
