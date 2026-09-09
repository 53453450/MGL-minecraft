/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_stmt.h
 *
 * C1g domain strip from mgl_air_backend.cpp — statement emit (emitStmt /
 * emitCompound + break/continue scan).  Backend keeps a thin AirStmtDeps
 * facade so emitExpr stays in the monolith.  Do not sink these back into
 * mgl_air_backend.cpp; do not move whole emitExpr here.
 */

#ifndef MGL_AIR_STMT_H
#define MGL_AIR_STMT_H

#include <cstdint>
#include <map>
#include <string>

#include "llvm/IR/Value.h"

#include "mgl_air_codegen.h"
#include "mgl_glsl_ast.h"
#include "mgl_glsl_sema.h"
#include "mgl_ir.h"

namespace mgl {
namespace air {

/* Backend-provided hooks (anonymous-ns symbols from mgl_air_backend.cpp).
 * The monolith fills this at the thin facade call site. */
struct AirStmtDeps {
    llvm::Value *(*emitExpr)(Codegen &cg, const MGLExpr *e,
                             const MGLIRModule *mod,
                             const std::map<std::string, MType> &locals);
    MType (*exprType)(Codegen &cg, const MGLExpr *e, const MGLIRModule *mod,
                      const std::map<std::string, MType> &locals);
    llvm::Value *(*assembleReturn)(Codegen &cg);
    MGLIRType *(*cloneIRType)(const MGLIRType *src);
};

/* Emit one AST statement (compound/expr/decl/return/discard/if/loops/
 * switch/break/continue).  Recurses via deps.emitExpr for expressions. */
void emitStmt(Codegen &cg, const MGLStmt *st, const MGLIRModule *mod,
              std::map<std::string, MType> *locals, const AirStmtDeps &deps);

} /* namespace air */
} /* namespace mgl */

#endif /* MGL_AIR_STMT_H */
