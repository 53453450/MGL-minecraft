#ifndef MGL_AIR_SPARSE_RESIDENCY_H
#define MGL_AIR_SPARSE_RESIDENCY_H

#include "mgl_air_codegen.h"

namespace mgl {
namespace air {

/* Return an i32 GL-style boolean. Nullptr means that this compile variant has
 * no commit table and the caller must not treat that as a resident result. */
llvm::Value *sparseCommitResidency(Codegen &cg, llvm::Value *metal_slot,
                                   llvm::Value *relative_level,
                                   llvm::Value *x, llvm::Value *y,
                                   llvm::Value *z);
llvm::Value *sparseCommitResidency(Codegen &cg, uint32_t metal_slot,
                                   llvm::Value *relative_level,
                                   llvm::Value *x, llvm::Value *y,
                                   llvm::Value *z);

llvm::Value *sparseSampleCount(Codegen &cg, uint32_t metal_slot);

} // namespace air
} // namespace mgl

#endif
