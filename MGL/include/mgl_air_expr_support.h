/* SPDX-License-Identifier: LGPL-3.0-only */

#ifndef MGL_AIR_EXPR_SUPPORT_H
#define MGL_AIR_EXPR_SUPPORT_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "mgl_air_codegen.h"
#include "mgl_glsl_sema.h"

namespace mgl {
namespace air {

const MGLIRSymbol *findSymbol(const MGLIRModule *mod, const char *name);

bool swizzleIndices(const char *field, std::vector<uint32_t> *out);
MType swizzleType(const MType &base, size_t lanes);
llvm::Value *emitSwizzleValue(Codegen &cg, llvm::Value *obj,
                              const std::vector<uint32_t> &idx);

llvm::Value *scalarizeBoolCompare(Codegen &cg, uint32_t op, llvm::Value *cmp);
llvm::Value *tryFoldConst(Codegen &cg, uint32_t op, llvm::Value *l,
                          llvm::Value *r, bool uns);
llvm::Value *emitNumericBinOp(Codegen &cg, uint32_t op, llvm::Value *l,
                              llvm::Value *r, const MType &lt,
                              const MType &rt);
llvm::Value *emitAggregateCompare(Codegen &cg, uint32_t op,
                                  llvm::Value *l, llvm::Value *r);
llvm::Value *broadcastTo(Codegen &cg, llvm::Value *v, llvm::Type *vecTy);
llvm::Value *dotProduct(Codegen &cg, llvm::Value *a, llvm::Value *b);

} /* namespace air */
} /* namespace mgl */

#endif
