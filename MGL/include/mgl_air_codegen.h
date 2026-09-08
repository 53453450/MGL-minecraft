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
#include <string>
#include <vector>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

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
};

struct VarSym {
    std::string name;
    MType type;
    enum Kind { ATTR, CONTROL_POINT_INPUT, VARYING, OUTPUT, BUFFER, SSBO,
                UBO, TEXTURE, IMAGE, ATOMIC_COUNTER, LOCAL } kind = LOCAL;
    uint32_t bufferOffset = 0;
    uint32_t location = UINT32_MAX;
    bool locationExplicit = false; /* layout(location=N) in source */
    int32_t stream = 0;          /* GS output stream for OUTPUT vars */
    std::string blockName;       /* owning interface block, or empty */
    bool isPatch = false;
    bool isSample = false;       /* `sample in` / `sample out` qualifier */
    bool written = false;
    const MGLIRType *opaqueType = nullptr; /* nested sampler/image leaf */
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
    bool usesPatchCullDistance = false;  /* native TES: cull from gl_in records */
    llvm::Value *fragPos = nullptr;      /* fragment: [[position]] (gl_FragCoord) */
    bool hasFragDepth = false;           /* fragment writes gl_FragDepth */
    bool fragDepthInit = false;          /* gl_FragDepth lvalue initialized */
    bool hasSampleMask = false;          /* fragment writes gl_SampleMask */
    /* Slot 30: {num_samples, sample_buffers}. sample_buffers==0 means
     * non-MSAA FB; GL ignores gl_SampleMask writes in that case. */
    llvm::Value *fragSampleParams = nullptr;
    bool usesClipDistance = false;       /* vertex writes gl_ClipDistance */
    uint32_t cullDistancePassthroughCount = 0; /* VS flat outs / FS ins */
    uint32_t clipDistanceInputCount = 0; /* FS reads gl_ClipDistance */
    bool pointSize = false;              /* vertex: writes gl_PointSize */
    bool layerViewport = false;          /* writes gl_Layer / gl_ViewportIndex */
    bool primitiveIdWritten = false;     /* writes gl_PrimitiveID (GS out) */
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
    std::map<std::string, std::vector<llvm::Value *>> texArrayValues;
    std::map<std::string, std::vector<llvm::Value *>> smpArrayValues;
    std::map<std::string, const MGLIRType *> samplerIRTypes;
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
    std::map<std::string, llvm::Value *> lvalues;   /* register values */
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
    /* When true, STMT_RETURN captures into inlineRetVal instead of
     * emitting CreateRet (GS/TCS/compute helper inlining). */
    bool inliningHelper = false;
    llvm::Value *inlineRetVal = nullptr;
    /* Named user struct types from the TU (`struct S { … };`), used for
     * S(...) / S[](...) constructors and local member ExtractValue. */
    std::map<std::string, MGLIRType *> structTypes;
    std::vector<MGLIRType *> *ownedIRTypes = nullptr;
    std::map<std::string, const MGLIRType *> localIRTypes;
    int err = 0;
    std::string errmsg;                  /* specific diagnostic when set */
    std::vector<LoopCtx *> loopStack;    /* innermost loop is last */
    std::vector<BreakCtx *> breakStack;  /* innermost loop/switch is last */
};

} /* namespace air */
} /* namespace mgl */

#endif /* MGL_AIR_CODEGEN_H */
