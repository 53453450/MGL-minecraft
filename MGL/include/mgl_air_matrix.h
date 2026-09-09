/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_matrix.h
 *
 * C1f domain strip from mgl_air_backend.cpp — GLSL matrix builtins and
 * matrix binary ops (emitMatrixBuiltin / emitMatrixBinOp + det helpers).
 * Backend keeps a thin AirMatrixDeps facade so emitExpr stays in the
 * monolith.  Do not sink these back into mgl_air_backend.cpp; do not move
 * whole emitExpr here.
 */

#ifndef MGL_AIR_MATRIX_H
#define MGL_AIR_MATRIX_H

#include <cstdint>
#include <map>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Value.h"

#include "mgl_air_codegen.h"
#include "mgl_glsl_ast.h"
#include "mgl_glsl_sema.h"
#include "mgl_ir.h"

namespace mgl {
namespace air {

/* Backend-provided hooks (anonymous-ns symbols from mgl_air_backend.cpp).
 * The monolith fills this at the thin facade call site. */
struct AirMatrixDeps {
    llvm::Value *(*emitExpr)(Codegen &cg, const MGLExpr *e,
                             const MGLIRModule *mod,
                             const std::map<std::string, MType> &locals);
    llvm::Value *(*dotProduct)(Codegen &cg, llvm::Value *a, llvm::Value *b);
    llvm::Value *(*scalarizeBoolCompare)(Codegen &cg, uint32_t op,
                                         llvm::Value *cmp);
};

/* Matrix builtins: transpose, matrixCompMult, outerProduct, determinant
 * and inverse (square float matrices, sema-typed subset).  Returns nullptr
 * when `name` is not a matrix builtin handled here. */
llvm::Value *emitMatrixBuiltin(Codegen &cg, const MGLExpr *e, const char *name,
                               const MGLIRModule *mod,
                               const std::map<std::string, MType> &locals,
                               const AirMatrixDeps &deps);

/* Matrix binary ops: M*vec, vec*M, M*M, M*scalar, scalar*M, M±M and
 * M±scalar (element-wise).  Column-major storage: the LLVM value is
 * [cols x <rows x float>].  Returns nullptr when neither operand is a
 * matrix, so the caller falls back to the scalar/vector path. */
llvm::Value *emitMatrixBinOp(Codegen &cg, uint32_t op, llvm::Value *l,
                             llvm::Value *r, const AirMatrixDeps &deps);

/* Call a named AIR function (e.g. air.pack.unorm2x16.v2f32); the module
 * declaration is created on first use.  Definition in mgl_air_matrix.cpp
 * (relocated here from mgl_air_backend.cpp by C1f). */
llvm::Value *callAirFn(Codegen &cg, const char *fn, llvm::Type *retTy,
                       llvm::ArrayRef<llvm::Value *> args);

} /* namespace air */
} /* namespace mgl */

#endif /* MGL_AIR_MATRIX_H */
