/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_math.h
 *
 * C1d domain strip from mgl_air_backend.cpp — GLSL math / pack / bitfield
 * builtins (emitMathBuiltin).  Backend keeps a thin AirMathDeps facade so
 * emitExpr stays in the monolith; matrix builtins live in mgl_air_matrix
 * (C1f).  Do not sink these back into mgl_air_backend.cpp; do not move
 * emitExpr or matrix builtins here.
 */

#ifndef MGL_AIR_MATH_H
#define MGL_AIR_MATH_H

#include <cstdint>
#include <map>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "mgl_air_codegen.h"
#include "mgl_glsl_ast.h"
#include "mgl_glsl_sema.h"
#include "mgl_ir.h"

namespace mgl {
namespace air {

/* Backend-provided hooks (anonymous-ns symbols from mgl_air_backend.cpp).
 * The monolith fills this at the thin facade call site. */
struct AirMathDeps {
    llvm::Value *(*emitExpr)(Codegen &cg, const MGLExpr *e,
                             const MGLIRModule *mod,
                             const std::map<std::string, MType> &locals);
    llvm::Value *(*callAirFn)(Codegen &cg, const char *fn, llvm::Type *retTy,
                              llvm::ArrayRef<llvm::Value *> args);
    llvm::Value *(*dotProduct)(Codegen &cg, llvm::Value *a, llvm::Value *b);
    llvm::Value *(*broadcastTo)(Codegen &cg, llvm::Value *v, llvm::Type *vecTy);
    MType (*exprType)(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
                      const std::map<std::string, MType> &locals);
    const MGLIRSymbol *(*findSymbol)(const MGLIRModule *mod, const char *name);
    const MGLIRSymbol *(*ssboRootSym)(const MGLExpr *e, const MGLIRModule *mod);
    void (*emitSSBOWrite)(Codegen &cg, const MGLExpr *e, const MGLIRSymbol *sb,
                          const MGLIRModule *mod,
                          const std::map<std::string, MType> &locals,
                          llvm::Value *v);
    llvm::Value *(*updateIndexPath)(
        Codegen &cg, const MGLExpr *lhs, llvm::Value *rootVal, llvm::Value *val,
        const MGLIRModule *mod, const std::map<std::string, MType> &locals);
};

/* Math builtins: trigonometry, exponentials, rounding, geometric,
 * pack/unpack, integer/bitfield.  Returns nullptr when `name` is not a
 * math builtin handled here. */
llvm::Value *emitMathBuiltin(Codegen &cg, const MGLExpr *e, const char *name,
                             const MGLIRModule *mod,
                             const std::map<std::string, MType> &locals,
                             const AirMathDeps &deps);

} /* namespace air */
} /* namespace mgl */

#endif /* MGL_AIR_MATH_H */
