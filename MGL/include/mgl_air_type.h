/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_type.h
 *
 * C1b domain strip from mgl_air_backend.cpp — lightweight MType model and
 * type helpers (float carriers, LLVM type mapping, AIR/MSL mangling,
 * typeFromIR).  Codegen-tied helpers take mgl::air::Codegen& (see
 * mgl_air_codegen.h).  Do not sink these back into mgl_air_backend.cpp;
 * do not put non-type logic into mgl_readback_policy.
 */

#ifndef MGL_AIR_TYPE_H
#define MGL_AIR_TYPE_H

#include <cstdint>
#include <string>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"

#include "mgl_ir.h"

namespace mgl {
namespace air {

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

struct Codegen;
struct VarSym;

bool scalarIsFloat(MGLIRScalar s);
bool varyingUsesFloatCarrier(const MType &t, bool has_gs);
bool uintUsesSplitFloatCarrier(const MType &t, bool has_gs);
void encodeUintSplitFloatCarrier(Codegen &cg, llvm::Value *value,
                                 llvm::Value **loOut, llvm::Value **hiOut);
llvm::Value *decodeUintSplitFloatCarrier(Codegen &cg, llvm::Value *loF,
                                         llvm::Value *hiF, llvm::Type *destTy);
llvm::Value *encodeFloatCarrier(Codegen &cg, llvm::Value *value,
                                MGLIRScalar scalar);
llvm::Value *decodeFloatCarrier(Codegen &cg, llvm::Value *arg,
                                MGLIRScalar scalar, llvm::Type *destTy);
MType floatCarrierType(const MType &t);
MType matrixColumnType(const MType &t);
bool varyingNeedsFloatRecordCarrier(const MType &t);
MType attrMetalIfaceType(const MType &t);
llvm::Type *llvmScalar(MGLIRScalar s, llvm::LLVMContext &ctx);
bool irTypeIsVoid(const MGLIRType *t);
llvm::Align bufferLeafAlign(llvm::Type *t);
llvm::Type *llvmType(const MType &t, llvm::LLVMContext &ctx);
bool needsArrayMem(Codegen &cg, const std::string &name, const MType &t);
llvm::Value *ensureArrayMem(Codegen &cg, const std::string &name,
                            const MType &t);
llvm::Value *arrayMemGEP(Codegen &cg, const std::string &name,
                         llvm::Value *slot, llvm::Value *idx);
llvm::Type *llvmTypeFromIR(const MGLIRType *t, llvm::LLVMContext &ctx);
llvm::Value *coerceScalar(Codegen &cg, llvm::Value *v, MGLIRScalar want);
std::string airTypeMangle(const MType &t);
std::string airGenerated(const std::string &name, const MType &t);
std::string varyingIfaceTag(const VarSym &v, uint32_t elem = 0,
                            bool forceLocationTag = false);
std::string mslTypeName(const MType &t);
MType typeFromIR(const MGLIRType *t);

} /* namespace air */
} /* namespace mgl */

#endif /* MGL_AIR_TYPE_H */
