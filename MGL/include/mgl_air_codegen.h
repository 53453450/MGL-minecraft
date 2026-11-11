/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_air_codegen.h
 *
 * Backend-internal Codegen / symbol state shared by mgl_air_backend.cpp and
 * mgl_air_type.cpp (C1b).  Not a public GL ABI.  Do not include from render
 * or ObjC ports.
 */

#ifndef MGL_AIR_CODEGEN_H
#define MGL_AIR_CODEGEN_H

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

#include "mgl_air_control_flow.h"
#include "mgl_air_resource_binding.h"
#include "mgl_air_type.h"
#include "mgl_glsl_ast.h"
#include "mgl_ir.h"

/* Avoid pulling mgl_shader_abi.h (Mach) via GS/tess ABI headers — type.cpp
 * must Linux-smoke.  Keep defaults aligned with MGL_AIR_PER_VERTEX_STRIDE. */
#ifndef MGL_AIR_CODEGEN_PER_VERTEX_STRIDE
#define MGL_AIR_CODEGEN_PER_VERTEX_STRIDE 112u
#endif

namespace mgl {
namespace air {

struct Uniform {
    std::string name;
    MType type;
    uint32_t offset;         /* byte offset in the implicit buffer */
    uint32_t size;           /* std140 byte size */
    uint32_t elementStride;  /* bytes between array elements, 0 if scalar */
};

struct VarSym {
    std::string name;
    MType type;
    enum Kind { ATTR, CONTROL_POINT_INPUT, VARYING, OUTPUT, BUFFER, SSBO,
                UBO, TEXTURE, IMAGE, ATOMIC_COUNTER, LOCAL } kind = LOCAL;
    uint32_t bufferOffset = 0;
    uint32_t location = UINT32_MAX;
    bool locationExplicit = false; /* layout(location=N) in source */
    uint32_t component = 0;      /* layout(component=N) */
    int32_t stream = 0;          /* GS output stream for OUTPUT vars */
    std::string blockName;       /* owning interface block, or empty */
    bool isPatch = false;
    bool isSample = false;       /* `sample in` / `sample out` qualifier */
    bool isFlat = false;
    bool isNoPerspective = false;
    bool written = false;
    const MGLIRType *opaqueType = nullptr; /* nested sampler/image leaf */
};

struct InlineState {
    /* STMT_RETURN captures into retVal instead of emitting CreateRet. */
    bool inliningHelper = false;
    /* Bound recursive inlining even when a subroutine call reaches a cycle
     * that the sema call graph did not reject. */
    uint32_t inlineDepth = 0;
    llvm::Value *inlineRetVal = nullptr;
    llvm::BasicBlock *inlineExitBB = nullptr;
    llvm::PHINode *inlineRetPhi = nullptr;
    /* Return edges can originate in nested control flow; their live values are
     * merged at inlineExitBB rather than assumed to dominate that block. */
    std::vector<std::pair<llvm::BasicBlock *,
                          std::map<std::string, llvm::Value *>>> inlineExitSnaps;
};

struct Codegen {
    llvm::LLVMContext *ctx;
    llvm::IRBuilder<> *b;
    llvm::Function *fn;
    llvm::Module *mod = nullptr;       /* current LLVM module */
    /* AIR ABI stage of this compile; subroutine-uniform pack names are
     * stage-qualified with the same tag map (mgl_air_backend.cpp). */
    int stage = 0;
    bool isVS = false;
    bool isCompute = false;
    bool isTessControl = false;
    bool isTessEval = false;
    bool isGeometry = false;
    llvm::Value *bufferPtr = nullptr;    /* i8 addrspace(1)* */
    llvm::Value *bufferSizePtr = nullptr; /* constant uint*, buffer(25) */
    llvm::Value *threadPos = nullptr;    /* compute: <3 x i32> grid position */
    llvm::Value *localInvocationPos = nullptr; /* compute: <3 x i32> local */
    llvm::Value *localInvocationIndex = nullptr; /* compute: i32 flat local */
    llvm::Value *workGroupPos = nullptr; /* compute: <3 x i32> group position */
    llvm::Value *numWorkGroups = nullptr; /* compute: <3 x i32> dispatch grid */
    /* gl_WorkGroupSize: constant from layout(local_size_*); axes default to 1. */
    uint32_t workGroupSizeX = 1;
    uint32_t workGroupSizeY = 1;
    uint32_t workGroupSizeZ = 1;
    bool hasWorkGroupSize = false;
    llvm::Value *invocationPos = nullptr; /* TCS: <3 x i32> threadgroup position */
    llvm::Value *patchPos = nullptr;      /* TCS: <3 x i32> threadgroup grid position */
    llvm::Value *stageInPtr = nullptr;   /* TCS gl_in replacement, buffer(24) */
    llvm::Value *stageOutPtr = nullptr;  /* TCS gl_out replacement, buffer(28) */
    llvm::Value *tessFactorPtr = nullptr; /* TCS factors, buffer(26) */
    llvm::Value *indirectPtr = nullptr;  /* TCS patch info, buffer(29) */
    uint32_t tcsOutputVertices = 0;
    uint32_t stageInStride = MGL_AIR_CODEGEN_PER_VERTEX_STRIDE;
    uint32_t stageOutStride = MGL_AIR_CODEGEN_PER_VERTEX_STRIDE;
    uint32_t patchInStride = 16;
    uint32_t patchOutStride = 16;
    llvm::Value *geometryInputPtr = nullptr;  /* GS primitive records */
    llvm::Value *geometryOutputPtr = nullptr; /* GS expanded records */
    llvm::Value *geometryCountPtr = nullptr;  /* GS indirect draw args */
    llvm::Value *geometryGatherPtr = nullptr; /* GS indexed gather stream */
    llvm::Value *geometryGatherParamsPtr = nullptr; /* GS gather params    */
    llvm::Value *geometryXfbPtr = nullptr;  /* GS XFB stream, buffer(22)   */
    llvm::Value *geometryXfbMetaPtr = nullptr; /* GS XFB meta, buffer(27)  */
    llvm::Value *geometryXfbVisPtr = nullptr;  /* GS XFB visibility, buffer(30) */
    llvm::Value *tessGatherPtr = nullptr;     /* TES compute gather stream */
    llvm::Value *tessGatherParamsPtr = nullptr; /* TES compute gather params*/
    llvm::Value *xfbOutPtr = nullptr;   /* TES compute XFB stream, buffer(22) */
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
    bool isTESVertex = false;    /* isolines/point-mode TES compiled as a
                                  * render vertex function that rasterizes
                                  * the seeded domain stream directly. */
    bool isolinesTopology = false; /* TES-vertex: isolines (line) vs point. */
    llvm::Value *captureBuf = nullptr;   /* capture variant: output buffer */
    llvm::Value *vertexId = nullptr;     /* capture variant: vertex_id */
    llvm::Value *instanceId = nullptr;   /* vertex: instance_id */
    llvm::Value *baseInstance = nullptr; /* vertex: base_instance */
    llvm::Value *baseVertex = nullptr;   /* vertex: base_vertex (gl_BaseVertex) */
    llvm::Value *drawId = nullptr;       /* vertex: gl_DrawID (host/ICB) */
    llvm::Value *cullBuffer = nullptr;   /* VS cull-distance source buffer */
    llvm::Value *cullParams = nullptr;   /* VS cull-distance emu parameters */
    bool usesCullDistance = false;
    bool usesPatchCullDistance = false;  /* native TES: cull from gl_in records */
    bool clipDepthZeroToOne = false;     /* skip [-1,1]→[0,1] position.z remap */
    llvm::Value *fragPos = nullptr;      /* fragment: [[position]] (gl_FragCoord) */
    bool hasFragDepth = false;           /* fragment writes gl_FragDepth */
    bool fragDepthInit = false;          /* gl_FragDepth lvalue initialized */
    bool hasSampleMask = false;          /* fragment writes gl_SampleMask */
    /* Slot 30: {num_samples, sample_buffers}. sample_buffers==0 means
     * non-MSAA FB; GL ignores gl_SampleMask writes in that case. */
    llvm::Value *fragSampleParams = nullptr;
    /* +1 / -1 applied to dFdy: -1 on a window-system target whose rows count
     * downward (see MGL_AIR_BUILTIN_DERIV_Y). Null outside fragment stages. */
    llvm::Value *derivYSign = nullptr;
    const MGLExpr *derivXExpr = nullptr; /* GradClamp: explicit dPdx */
    const MGLExpr *derivYExpr = nullptr; /* GradClamp: explicit dPdy */
    llvm::Value *explicitLodOverride = nullptr; /* Clamp: max(auto, clamp) */
    bool usesClipDistance = false;       /* vertex writes gl_ClipDistance */
    bool emulateClipDiscard = false;     /* FS discards interpolated clip < 0 */
    uint32_t cullDistancePassthroughCount = 0; /* VS flat outs / FS ins */
    uint32_t clipDistanceInputCount = 0; /* FS clip varyings (read or emulate) */
    bool pointSize = false;              /* vertex: writes gl_PointSize */
    bool layerViewport = false;          /* writes gl_Layer / gl_ViewportIndex */
    bool primitiveIdWritten = false;     /* writes gl_PrimitiveID (GS out) */
    ResourceBindingState resources;
    /* Same-invocation imageStore→imageLoad forwarding (GLSL 4.60 §8.12 /
     * coherent).  Metal texture.fence()+read is not reliable for VS
     * imageBuffer on AGX (CTS advanced-memory-order).  Entries are only
     * reused when the load is still in the store's basic block; cleared on
     * barrier / memoryBarrier* / imageAtomic*.  Load path applies image
     * layout missing-component fill (r32f → (R,0,0,1)). */
    struct ImageStoreFwd {
        llvm::BasicBlock *bb = nullptr;
        llvm::Value *tex = nullptr;
        llvm::Value *coord = nullptr;
        llvm::Value *value = nullptr;
    };
    std::vector<ImageStoreFwd> imageStoreFwd;
    /* Stage outputs living in entry-block allocas so user functions can
     * write them through hidden pointer args (SET_RESULT→helper pattern). */
    std::map<std::string, llvm::Value *> outPtrs;
    /* Local/varying scalar arrays in entry-block memory.  Combined with
     * scalar(vec) peeling in constructors, this avoids illegal
     * store <N x float>, float* and keeps dynamic float[] indexing off
     * SSA select-of-float paths that Metal materializeAll rejects. */
    std::map<std::string, llvm::Value *> arrayMem;
    std::map<std::string, llvm::Type *> arrayMemTypes;
    std::map<std::string, uint32_t> bufferOffsets;  /* uniform name -> byte offset */
    /* Uniform name -> byte stride between array elements in the pack.  The
     * renderer lays the implicit uniform buffer out with the std140 element
     * stride (16 for an array of uints, not 4), and the subroutine dispatch
     * indexes the array with it. */
    std::map<std::string, uint32_t> bufferStrides;
    std::map<std::string, llvm::Value *> lvalues;   /* register values */
    /* AST copies handed to emitters that may retain the pointer. */
    /* Owned ephemeral MGLExpr copies handed to the inliner.  The inliner is
     * synchronous and only borrows them, but a caller that must mutate a node
     * (subroutine dispatch retargets u.call.name) parks the copy here so it
     * outlives the call.  Cleared by the Codegen destructor's owner. */
    std::vector<std::unique_ptr<MGLExpr>> exprPool;
    /* Block-qualified keys of per-vertex varying arrays whose elements a
     * shader stored straight into the stage-out record (TES compute path).
     * storeTessComputeVaryings must not overwrite those slots with undef. */
    std::set<std::string> directRecordVaryings;
    std::vector<VarSym *> varyings;      /* vertex out / fragment in, decl order */
    std::vector<VarSym *> fragOutputs;   /* fragment outputs, return-field order */
    bool has_gs = false;                 /* fragment fed by GS passthrough VS */
    VarSym position;                     /* gl_Position */
    llvm::Type *retTy = nullptr;         /* stage return type */
    std::vector<llvm::Type *> retElems;  /* VS struct fields (incl. position) */
    std::vector<VarSym> *auxSyms = nullptr;  /* all stage symbols (frag output) */
    std::map<std::string, llvm::Function *> *userFns = nullptr;
    std::map<std::string, uint32_t> *userFnHidden = nullptr;
    std::map<std::string, MGLDecl *> *userFnDecls = nullptr;
    bool userFnPassCull = false;
    bool userFnPassClip = false;
    InlineState inlining;
    /* Named user struct types from the TU (`struct S { … };`), used for
     * S(...) / S[](...) constructors and local member ExtractValue. */
    std::map<std::string, MGLIRType *> structTypes;
    std::vector<MGLIRType *> *ownedIRTypes = nullptr;
    std::map<std::string, const MGLIRType *> localIRTypes;
    int err = 0;
    std::string errmsg;                  /* specific diagnostic when set */
    ControlFlowState controlFlow;
};

} /* namespace air */
} /* namespace mgl */

#endif /* MGL_AIR_CODEGEN_H */
