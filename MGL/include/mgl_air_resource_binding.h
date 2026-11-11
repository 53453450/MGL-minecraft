/* SPDX-License-Identifier: LGPL-3.0-only */

#ifndef MGL_AIR_RESOURCE_BINDING_H
#define MGL_AIR_RESOURCE_BINDING_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llvm {
class Type;
class Value;
}

struct MGLIRType;

namespace mgl {
namespace air {

/* Function-entry resource values and their Metal binding slots. */
struct ResourceBindingState {
    std::map<std::string, llvm::Value *> ssboPtrs;  /* SSBO instance -> buffer */
    std::map<std::string, llvm::Value *> acPtrs;  /* atomic_uint -> ACBO */
    /* UBO/SSBO instance arrays: element pointers stashed in an entry-block
     * alloca so member reads can index by runtime value. */
    std::map<std::string, llvm::Value *> uboElemSlot;
    std::map<std::string, llvm::Type *> uboElemArrTy;
    std::map<std::string, llvm::Value *> ssboElemSlot;
    std::map<std::string, llvm::Type *> ssboElemArrTy;
    std::map<std::string, uint32_t> ssboSlots;      /* SSBO instance -> Metal slot */
    std::map<std::string, uint32_t> acSlots;        /* atomic_uint -> Metal slot */
    std::map<std::string, llvm::Value *> uboPtrs;   /* uniform block -> buffer */
    std::map<std::string, llvm::Value *> texValues;  /* sampler name -> texture */
    std::map<std::string, llvm::Value *> smpValues;  /* sampler name -> sampler */
    std::map<std::string, uint32_t> texMetalIndex; /* sampler -> Metal tex slot */
    llvm::Value *samplerWrapPtr = nullptr; /* integer gather wrap/border table */
    /* GL 4.6 §8.14.1 packed table at kMGLLodBiasMaxBufferIndex:
     *   float biasmax;
     *   float deriv_scale_x;  // fb_w / viewport_w (Metal→GL dFdx)
     *   float deriv_scale_y;
     *   float _pad;
     *   float bias[TEXTURE_UNITS];  // TEXTURE_LOD_BIAS per Metal tex slot */
    llvm::Value *lodBiasPtr = nullptr;
    llvm::Value *sparseResidencyPtr = nullptr; /* dynamic commit rect table */
    std::map<std::string, std::vector<llvm::Value *>> texArrayValues;
    std::map<std::string, std::vector<llvm::Value *>> smpArrayValues;
    std::map<std::string, const MGLIRType *> samplerIRTypes;
};

} // namespace air
} // namespace mgl

#endif
