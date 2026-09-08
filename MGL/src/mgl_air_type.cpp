/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_type.cpp
 * C1b — AIR type helpers extracted from mgl_air_backend.cpp (~290–723 +
 * typeFromIR).  Backend keeps thin using-declarations into mgl::air.
 */

#include "mgl_air_type.h"
#include "mgl_air_codegen.h"

#include <cstdio>

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/ADT/SmallVector.h"

namespace mgl {
namespace air {

std::string mslTypeName(const MType &t);
MType typeFromIR(const MGLIRType *t);

bool scalarIsFloat(MGLIRScalar s) {
    return s == MGLIR_SCALAR_FLOAT || s == MGLIR_SCALAR_DOUBLE ||
           s == MGLIR_SCALAR_HALF;
}

/* Integer varyings on AGX must ride float carriers: GS expansion already
 * needed this for all integer types; flat integer varyings in plain VS/FS
 * pipelines also misread when carried as raw int/uint stage inputs. */
bool varyingUsesFloatCarrier(const MType &t, bool has_gs) {
    if (t.scalar == MGLIR_SCALAR_BOOL || scalarIsFloat(t.scalar))
        return false;
    if (has_gs)
        return true;
    return t.scalar == MGLIR_SCALAR_INT || t.scalar == MGLIR_SCALAR_UINT;
}

/* Full-range uint flat varyings cannot use a single float carrier: UIToFP
 * loses bits above float24 and intBitsToFloat produces NaN payloads that
 * AGX does not preserve.  Split into two exact float16-bit lanes instead.
 * Vectors keep the ordinary float carrier (per-component bitcast). */
bool uintUsesSplitFloatCarrier(const MType &t, bool has_gs) {
    return !has_gs && t.scalar == MGLIR_SCALAR_UINT && !t.vec &&
           !t.isArray() && !t.isMatrix();
}

void encodeUintSplitFloatCarrier(Codegen &cg, llvm::Value *value,
                                        llvm::Value **loOut,
                                        llvm::Value **hiOut) {
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    llvm::Value *lo16 =
        cg.b->CreateAnd(value, cg.b->getInt32(0xFFFF));
    llvm::Value *hi16 = cg.b->CreateLShr(value, 16);
    *loOut = cg.b->CreateUIToFP(lo16, f32);
    *hiOut = cg.b->CreateUIToFP(hi16, f32);
}

llvm::Value *decodeUintSplitFloatCarrier(Codegen &cg,
                                                llvm::Value *loF,
                                                llvm::Value *hiF,
                                                llvm::Type *destTy) {
    loF = cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::round, loF);
    hiF = cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::round, hiF);
    llvm::Type *i32 = llvm::Type::getInt32Ty(*cg.ctx);
    llvm::Value *lo = cg.b->CreateFPToUI(loF, i32);
    llvm::Value *hi = cg.b->CreateFPToUI(hiF, i32);
    llvm::Value *u = cg.b->CreateOr(lo, cg.b->CreateShl(hi, 16));
    if (destTy != i32)
        u = cg.b->CreateTruncOrBitCast(u, destTy);
    return u;
}

llvm::Value *encodeFloatCarrier(Codegen &cg, llvm::Value *value,
                                       MGLIRScalar scalar) {
    llvm::Type *f32 = llvm::Type::getFloatTy(*cg.ctx);
    llvm::Type *dst = f32;
    if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(value->getType()))
        dst = llvm::FixedVectorType::get(
            f32, vt->getElementCount().getFixedValue());
    if (scalar == MGLIR_SCALAR_DOUBLE) {
        /* ATTR/VS already produced f32; keep bits. i64 double payloads from
         * SSBO paths soft-truncate IEEE754 binary64→binary32 without f64 ALU. */
        if (value->getType()->isFPOrFPVectorTy() &&
            value->getType()->getScalarSizeInBits() == 32)
            return value;
        if (value->getType()->isIntOrIntVectorTy() &&
            value->getType()->getScalarSizeInBits() == 64) {
            auto conv1 = [&](llvm::Value *bits) -> llvm::Value * {
                llvm::Value *hi = cg.b->CreateLShr(bits, 32);
                hi = cg.b->CreateTrunc(hi, cg.b->getInt32Ty());
                llvm::Value *lo = cg.b->CreateTrunc(bits, cg.b->getInt32Ty());
                /* Approximate: take high 32 bits of the double payload when
                 * it was a float promoted into the high half; otherwise
                 * rebuild a float from sign/exp/mant of the i64. */
                llvm::Value *sign = cg.b->CreateAnd(
                    cg.b->CreateLShr(hi, 31), cg.b->getInt32(1));
                llvm::Value *exp = cg.b->CreateAnd(
                    cg.b->CreateLShr(hi, 20), cg.b->getInt32(0x7FF));
                llvm::Value *mantHi = cg.b->CreateAnd(hi, cg.b->getInt32(0xFFFFF));
                llvm::Value *mant = cg.b->CreateOr(
                    cg.b->CreateShl(mantHi, 3),
                    cg.b->CreateLShr(lo, 29));
                llvm::Value *exp32 = cg.b->CreateSub(exp, cg.b->getInt32(1023 - 127));
                llvm::Value *under = cg.b->CreateICmpSLT(exp32, cg.b->getInt32(1));
                llvm::Value *over = cg.b->CreateICmpSGT(exp32, cg.b->getInt32(254));
                llvm::Value *fbits = cg.b->CreateOr(
                    cg.b->CreateShl(sign, 31),
                    cg.b->CreateOr(cg.b->CreateShl(
                        cg.b->CreateAnd(exp32, cg.b->getInt32(0xFF)), 23),
                        cg.b->CreateAnd(mant, cg.b->getInt32(0x7FFFFF))));
                fbits = cg.b->CreateSelect(under, cg.b->CreateShl(sign, 31), fbits);
                fbits = cg.b->CreateSelect(
                    over,
                    cg.b->CreateOr(cg.b->CreateShl(sign, 31),
                                   cg.b->getInt32(0x7F800000)),
                    fbits);
                return cg.b->CreateBitCast(fbits, f32);
            };
            if (auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(value->getType())) {
                unsigned n = vt->getElementCount().getFixedValue();
                llvm::Value *out = llvm::UndefValue::get(dst);
                for (unsigned i = 0; i < n; i++) {
                    llvm::Value *el = cg.b->CreateExtractElement(value, i);
                    out = cg.b->CreateInsertElement(out, conv1(el), i);
                }
                return out;
            }
            return conv1(value);
        }
        return llvm::Constant::getNullValue(dst);
    }
    if (scalar == MGLIR_SCALAR_UINT)
        return cg.b->CreateUIToFP(value, dst);
    return cg.b->CreateSIToFP(value, dst);
}

llvm::Value *decodeFloatCarrier(Codegen &cg, llvm::Value *arg,
                                       MGLIRScalar scalar,
                                       llvm::Type *destTy) {
    arg = cg.b->CreateUnaryIntrinsic(llvm::Intrinsic::round, arg);
    if (scalar == MGLIR_SCALAR_UINT)
        return cg.b->CreateFPToUI(arg, destTy);
    return cg.b->CreateFPToSI(arg, destTy);
}

MType floatCarrierType(const MType &t) {
    MType f = t;
    f.scalar = MGLIR_SCALAR_FLOAT;
    return f;
}

/* Metal rejects matrix stage_in / stage_out attribute types
 * ("Unsupported attribute type").  Flatten to one vector per column. */
MType matrixColumnType(const MType &t)
{
    MType c;
    c.scalar = t.scalar;
    c.vec = t.rows > 0 ? t.rows : 1u;
    return c;
}

/* Stage-out records are float-oriented (passthrough VS reads vec4 slots).
 * Integer varyings must use SIToFP/UIToFP carriers — bitcasting small uint
 * values yields AGX-flushable denormals, and floatBitsToUint then returns 0.
 * DOUBLE is the same float carrier: VertexLayout converts GL_DOUBLE ATTR to
 * float, VS math stays f32 on AGX, and XFB pack expands float→GLdouble. */
bool varyingNeedsFloatRecordCarrier(const MType &t)
{
    if (t.isMatrix() || t.isArray()) return false;
    return t.scalar == MGLIR_SCALAR_INT || t.scalar == MGLIR_SCALAR_UINT ||
           t.scalar == MGLIR_SCALAR_BOOL || t.scalar == MGLIR_SCALAR_DOUBLE;
}

/* Vertex ATTR ABI must match VertexLayout: GL_DOUBLE is CPU-converted to
 * MTL Float*, so stage_in is floatN — never i64 (Metal reports i64 ATTR as
 * int1 and rejects Float→int1). */
MType attrMetalIfaceType(const MType &t)
{
    if (t.scalar != MGLIR_SCALAR_DOUBLE)
        return t;
    MType f = t;
    f.scalar = MGLIR_SCALAR_FLOAT;
    return f;
}

llvm::Type *llvmScalar(MGLIRScalar s, llvm::LLVMContext &ctx) {
    switch (s) {
    case MGLIR_SCALAR_BOOL: return llvm::Type::getInt1Ty(ctx);
    case MGLIR_SCALAR_INT:  return llvm::Type::getInt32Ty(ctx);
    case MGLIR_SCALAR_UINT: return llvm::Type::getInt32Ty(ctx);
    /* Metal has no f64 ALU on AGX; GLSL double is an i64 bit payload so
     * SSBO/UBO load/store stay bit-preserving without emitting f64 ops. */
    case MGLIR_SCALAR_DOUBLE: return llvm::Type::getInt64Ty(ctx);
    default:                return llvm::Type::getFloatTy(ctx);
    }
}

/* Sema stores void returns as a non-NULL SCALAR_VOID type, not a null
 * pointer — callers must not treat "return_type != NULL" as non-void. */
bool irTypeIsVoid(const MGLIRType *t)
{
    return !t || (t->kind == MGLIR_TYPE_SCALAR &&
                  t->scalar == MGLIR_SCALAR_VOID);
}

/* Natural align for a buffer leaf load/store of LLVM type `t`. */
llvm::Align bufferLeafAlign(llvm::Type *t) {
    if (auto *fvt = llvm::dyn_cast<llvm::FixedVectorType>(t)) {
        uint64_t w = fvt->getElementCount().getFixedValue();
        unsigned es =
            fvt->getElementType()->getPrimitiveSizeInBits() / 8u;
        if (es >= 8u) {
            if (w <= 1u) return llvm::Align(8);
            if (w == 2u) return llvm::Align(16);
            return llvm::Align(32); /* dvec3/dvec4 */
        }
        if (w == 1) return llvm::Align(4);
        if (w == 2) return llvm::Align(8);
        return llvm::Align(16);
    }
    if (t->isIntegerTy(64) || t->isDoubleTy())
        return llvm::Align(8);
    if (t->isFloatTy() || t->isIntegerTy(32))
        return llvm::Align(4);
    return llvm::Align(16);
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

/* Scalar arrays that Metal cannot keep in SSA aggregates.  Struct/matrix
 * arrays stay as SSA InsertValue aggregates (MType cannot tell them apart
 * from float[], so consult localIRTypes when present). */
bool needsArrayMem(Codegen &cg, const std::string &name,
                          const MType &t) {
    if (!t.isArray() || t.isMatrix() || t.vec != 0 || t.arr == 0)
        return false;
    if (!name.empty()) {
        auto it = cg.localIRTypes.find(name);
        if (it != cg.localIRTypes.end()) {
            const MGLIRType *e = it->second;
            while (e && e->kind == MGLIR_TYPE_ARRAY && e->elem_type)
                e = e->elem_type;
            if (e && (e->kind == MGLIR_TYPE_STRUCT ||
                      e->kind == MGLIR_TYPE_MATRIX))
                return false;
        }
    }
    return true;
}

llvm::Value *ensureArrayMem(Codegen &cg, const std::string &name,
                                   const MType &t) {
    auto it = cg.arrayMem.find(name);
    if (it != cg.arrayMem.end())
        return it->second;
    llvm::Type *arrTy = llvmType(t, *cg.ctx);
    llvm::BasicBlock *savedBB = cg.b->GetInsertBlock();
    auto savedPt = cg.b->GetInsertPoint();
    llvm::BasicBlock &entry = cg.fn->getEntryBlock();
    cg.b->SetInsertPoint(&entry, entry.begin());
    llvm::Value *slot = cg.b->CreateAlloca(arrTy, nullptr, name + ".amem");
    cg.b->SetInsertPoint(savedBB, savedPt);
    cg.arrayMem[name] = slot;
    cg.arrayMemTypes[name] = arrTy;
    return slot;
}

llvm::Value *arrayMemGEP(Codegen &cg, const std::string &name,
                                llvm::Value *slot, llvm::Value *idx) {
    llvm::Type *arrTy = cg.arrayMemTypes[name];
    idx = cg.b->CreateZExtOrTrunc(idx, cg.b->getInt64Ty());
    return cg.b->CreateInBoundsGEP(arrTy, slot,
                                   {cg.b->getInt64(0), idx});
}

/* SSA shape for an IR type, including nested structs/arrays.  Buffer
 * padding lives only in member_offsets / array_stride — the LLVM type
 * is tightly packed so InsertValue/ExtractValue can address fields. */
llvm::Type *llvmTypeFromIR(const MGLIRType *t, llvm::LLVMContext &ctx) {
    if (!t)
        return llvm::Type::getFloatTy(ctx);
    switch (t->kind) {
    case MGLIR_TYPE_STRUCT: {
        llvm::SmallVector<llvm::Type *, 8> elts;
        for (uint32_t i = 0; i < t->member_count; i++)
            elts.push_back(llvmTypeFromIR(t->members[i], ctx));
        return llvm::StructType::get(ctx, elts);
    }
    case MGLIR_TYPE_ARRAY: {
        uint32_t n = t->array_size > 0u ? t->array_size : 1u;
        return llvm::ArrayType::get(llvmTypeFromIR(t->elem_type, ctx), n);
    }
    default:
        return llvmType(typeFromIR(t), ctx);
    }
}

/* Implicit GLSL numeric conversion (sema allows any non-void scalar base
 * to convert to any other, GLSL 4.60 4.1.10).  Idempotent; works on
 * scalars and vectors of matching width. */
llvm::Value *coerceScalar(Codegen &cg, llvm::Value *v, MGLIRScalar want) {
    llvm::Type *cur = v->getType();
    if (!cur->isIntOrIntVectorTy() && !cur->isFPOrFPVectorTy())
        return v;  /* arrays / matrices / aggregates: no scalar cast */
    llvm::LLVMContext &ctx = *cg.ctx;
    auto vt = [&](llvm::Type *elt) -> llvm::Type * {
        if (auto *fv = llvm::dyn_cast<llvm::FixedVectorType>(cur))
            return llvm::FixedVectorType::get(elt,
                fv->getElementCount().getFixedValue());
        return elt;
    };
    /* Doubles are i64 payloads — never SIToFP/FPToSI them as float.
     * Numeric int/float→double conversion would emit f64 ALU that AGX
     * metallibs reject; bit-identical payloads from matching paths still
     * compare equal for CTS equality checks. */
    if (want == MGLIR_SCALAR_DOUBLE) {
        if (cur->isIntOrIntVectorTy() && cur->getScalarSizeInBits() == 64)
            return v;
        if (cur->isFPOrFPVectorTy() && cur->getScalarSizeInBits() == 64)
            return cg.b->CreateBitCast(v, vt(llvm::Type::getInt64Ty(ctx)));
        return v;
    }
    if (cur->isIntOrIntVectorTy() && cur->getScalarSizeInBits() == 64 &&
        want != MGLIR_SCALAR_DOUBLE)
        return v; /* keep double payload out of float/int coercions */
    bool wantFP = scalarIsFloat(want);
    bool curFP = cur->isFPOrFPVectorTy();
    if (curFP == wantFP && want != MGLIR_SCALAR_BOOL &&
        cur->getScalarSizeInBits() == (want == MGLIR_SCALAR_BOOL ? 1 : 32))
        return v;
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

/* Metal pairs VS vertex_output / FS fragment_input by air.generated()
 * identity (name mangling), not by GLSL identifier.  SSO programs often
 * use different names at the same layout(location=N) (CTS
 * advanced-sso-atomicCounters: o_color vs i_color).  Prefer a stable
 * location tag only when the location was explicit in source; auto-
 * assigned locations are per-stage and must not drive the tag (or
 * named matchings like vs_color break).
 *
 * Exception: GS / TES-compute passthrough VS always emit
 * layout(location=N) (ensureAIRGeometryPassthrough /
 * ensureAIRTessEvalPassthrough), so outs are tagged mgl_loc_N.  When
 * has_gs is set the FS must use the same tags even if inputs were only
 * auto-assigned — otherwise Metal rejects the pipeline with
 * "Fragment input(s) `name` ... not written by vertex shader". */
std::string varyingIfaceTag(const VarSym &v, uint32_t elem,
                                   bool forceLocationTag) {
    if (v.location != UINT32_MAX &&
        (v.locationExplicit || forceLocationTag)) {
        return "mgl_loc_" + std::to_string(v.location + elem);
    }
    if (v.type.isArray() || v.type.isMatrix() || elem != 0u) {
        return v.name + "_elm" + std::to_string(elem);
    }
    return v.name;
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

} /* namespace air */
} /* namespace mgl */
