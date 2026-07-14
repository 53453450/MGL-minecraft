/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * MGLRenderer+Draw_Private.h
 * MGL
 *
 * Private method declarations, types, constants, and C helpers for the Draw
 * category (MGLRenderer+Draw.m).  Imports MGLRenderer_Private.h for ivar
 * access and shared types.
 */

#ifndef MGLRenderer_Draw_Private_h
#define MGLRenderer_Draw_Private_h

#import "MGLRenderer_Private.h"
#import "msl_patch_pipeline.h"  /* P1-6: mglResolvePassthroughPatchModeForContext moved here */

/* === Per-worker context for parallel command recording ===
 * Each worker thread encodes draws onto its own MTLRenderCommandEncoder
 * obtained from a shared MTLParallelRenderCommandEncoder.  The dedup state
 * must be per-worker to prevent one worker's bindings from causing another
 * to skip a needed bind. */
typedef struct {
    id<MTLRenderCommandEncoder> encoder;

    /* Per-worker GLMState snapshot.  During parallel encoding,
     * ctx->active_state is redirected to point here so that all
     * ctx->active_state->* accesses in the encoding hot path are
     * thread-safe.  Heap-allocated to avoid ~83KB stack frame in the
     * parallel-encode path (3 stack copies = ~250KB+ can overflow
     * small-stack threads).  Owned by encodeBatchForParallelWorker:
     * allocated on entry, freed on exit.  NULL when unused. */
    GLMState *workerState;

    MGLLastBoundBuffer lastBoundVertexBuffers[kMGLMaxBufferSlots];
    MGLLastBoundBuffer lastBoundFragmentBuffers[kMGLMaxBufferSlots];
    id<MTLTexture> lastBoundVertexTextures[TEXTURE_UNITS];
    id<MTLTexture> lastBoundFragmentTextures[TEXTURE_UNITS];
    id<MTLSamplerState> lastBoundVertexSamplers[TEXTURE_UNITS];
    id<MTLSamplerState> lastBoundFragmentSamplers[TEXTURE_UNITS];
    id<MTLRenderPipelineState> lastPipelineState;
    id<MTLDepthStencilState> lastDepthStencilState;
    MTLViewport lastViewport;
    MTLScissorRect lastScissorRect;
    MTLCullMode lastCullMode;
    MTLWinding lastFrontFacingWinding;
    MTLTriangleFillMode lastTriangleFillMode;
    float lastDepthBias;
    float lastDepthBiasClamp;
    float lastDepthSlopeScale;
    BOOL lastBoundValid;

    id<MTLRenderPipelineState> pipelineState;
    MTLPixelFormat pipelineColor0Format;
    MTLPixelFormat pipelineDepthFormat;
    MTLPixelFormat pipelineStencilFormat;
    GLuint pipelineProgramName;

    NSUInteger mdiArgsScratchOffset;

    /* Metal-slot masks for the dedup entries copied into this worker. */
    uint32_t saved_vbuf_mask;
    uint32_t saved_fbuf_mask;
    uint64_t saved_texture_mask[2];

    uint64_t traceReplayFlushId;
    uint32_t traceReplayBatchIndex;
} MGLWorkerContext;

/* === Resolved vertex-attrib binding === */
typedef struct MGLResolvedVertexAttribBinding_t {
    const VertexAttrib *attrib;
    Buffer *buffer;
    GLintptr binding_offset;
    GLuint stride;
    GLuint divisor;
    GLintptr relativeoffset;
    GLuint binding_index;
    bool uses_binding_table;
} MGLResolvedVertexAttribBinding;

bool mglRendererResolveVertexAttribBinding(GLMContext ctx,
                                           VertexArray *vao,
                                           GLuint attribute,
                                           const char *where,
                                           MGLResolvedVertexAttribBinding *out);
int mglRendererResolveVertexAttributeBufferIndex(GLMContext ctx,
                                                 VertexArray *vao,
                                                 GLuint attribute,
                                                 const char *where);

/* Cull distance emulation params. */
typedef struct {
    uint32_t prim_vertex_count;
    uint32_t culldist_offset;
    uint32_t vertex_stride;
    uint32_t culldist_size;
} MGLCullDistanceEmuParams;

/* === Diagnostic constants === */
static const BOOL kMGLDiagnosticStateLogs = NO;
static const BOOL kMGLDrawSubmitDiagnostics = NO;

/* === Draw binding/validation constants === */
static const BOOL kMGLVerboseBindLogs = NO;
static const NSUInteger kMGLMinimumStageBindingSize = 256;
static const NSUInteger kMGLDefaultStageFallbackBufferSize = 4096;
static const NSUInteger kMGLStageBindingStackScratchSize = 1024;
static const BOOL kMGLEnableVertexAllSlotFallback = YES;
static const BOOL kMGLEnableSampledTextureFallback = YES;
static const BOOL kMGLValidateDrawArraysVboRange = YES;
static const BOOL kMGLValidateDrawElementsVboRange = YES;

#define kMGLPointSizeParamBufferIndex     kMGLPointSizeBufferIndex
#define kMGLTCSStageInReplBufferIndex     kMGLBufferSlot_TCSStageInRepl

/* === Inline helpers === */
static inline BOOL mglRendererObjectPointerLikelyValid(const void *ptr)
{
    return mglObjectPointerLooksPlausible(ptr);
}

static inline bool mglShouldTraceCall(uint64_t count)
{
    if (!kMGLDiagnosticStateLogs) {
        return false;
    }
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}

static inline BOOL mglTraceRTYFlipDiagnosticsEnabled(void)
{
    return mglTraceLogIsEnabled() && mglEnvFlagEnabled("MGL_TRACE_RT_YFLIP");
}

static inline const char *mglYFlipDecisionName(MGLYFlipDecision decision)
{
    switch (decision) {
        case MGL_YFLIP_USE_ORIGINAL: return "original";
        case MGL_YFLIP_USE_SAMPLED_COPY: return "sampled-copy";
        case MGL_YFLIP_USE_ORIGINAL_AND_INJECT: return "original-inject";
        default: return "unknown";
    }
}

/* === C functions defined in MGLRenderer.m, used by MGLRenderer+Draw.m === */
Program *mglTraceResolveDrawProgram(GLMContext traceCtx);
bool mglTraceShouldLogReplay(GLMContext traceCtx, Program *program);
BOOL mglRendererTextureLooksRecoverableSampled2D(GLMContext glctx,
                                                  Texture *tex,
                                                  MTLTextureType expectedType,
                                                  MGLTextureDataKind expectedKind);
BOOL mglRendererTextureLooksLikeSampledColor2D(GLMContext glctx, Texture *tex);
MTLIndexType getMTLIndexType(GLenum type);
Buffer *getElementBuffer(GLMContext ctx);
Buffer *getIndirectBuffer(GLMContext ctx);
MTLPrimitiveType getMTLPrimitiveType(GLenum mode);

void mglRestoreProgramPipelinePair(GLMContext ctx, GLuint programName, GLuint pipelineName);
void mglRendererSyncFramebufferBindingNames(GLMContext ctx);
Texture *mglTraceFramebufferAttachmentTexture(GLMContext glctx, FBOAttachment *attachment);
BOOL mglRendererGLSampledCopyLooksUsable(Texture *tex,
                                                MTLTextureType expectedType,
                                                MGLTextureDataKind expectedKind,
                                                BOOL allowPreviousWriteVersion,
                                                id<MTLTexture> *copyOut,
                                                BOOL *usedPreviousWriteVersionOut);
void mglLogDrawWithoutSwapWatchdog(const char *kind,
                                          uint64_t drawCall,
                                          GLMContext ctx,
                                          id<MTLCommandBuffer> commandBuffer,
                                          id<MTLRenderCommandEncoder> renderEncoder,
                                          MTLRenderPassDescriptor *renderPassDescriptor);
Texture *mglFindFramebufferColorTexturePairedWithDepth(GLMContext glctx,
                                                              Texture *depthTexture,
                                                              GLuint *fboNameOut);
BOOL mglCurrentDrawFramebufferUsesColorTexture(GLMContext glctx,
                                                      Texture *texture,
                                                      GLuint expectedFboName,
                                                      NSUInteger *attachmentIndexOut);
Buffer *mglRendererGetValidatedBuffer(GLMContext ctx, Buffer *candidate, const char *where, NSUInteger slot);
NSUInteger mglRendererBuildCurrentVertexAttribBytes(GLMContext ctx,
                                                           GLuint attribute,
                                                           const VertexAttrib *attrib,
                                                           uint8_t bytes[16]);
void mglLogSkippedGLSampledRenderTargetCopy(GLMContext glctx,
                                                   Program *program,
                                                   Texture *tex,
                                                   const char *stage,
                                                   const char *sampledName,
                                                   GLuint binding,
                                                   GLuint textureUnit,
                                                   const char *reason);
bool mglShouldInspectDrawCall(uint64_t drawCall, GLuint programName);
void mglTraceDrawElementsAttrib(GLMContext ctx,
                                       VertexArray *vao,
                                       uint64_t drawCall,
                                       GLuint programName,
                                       const uint8_t *indexBytes,
                                       GLenum indexType,
                                       NSUInteger indexElement,
                                       GLint baseVertex,
                                       GLuint attrib,
                                       bool traceFile);
void mglTraceReplayCommandVertexAttribSamples(GLMContext traceCtx,
                                                     Program *program,
                                                     const MGLDrawCommand *cmd,
                                                     Buffer *ebo,
                                                     uint64_t flushId,
                                                     uint32_t batchIndex,
                                                     uint32_t commandIndex,
                                                     bool forceTrace);

/* P1-6: mglResolvePassthroughPatchModeForContext moved to msl_patch_pipeline.h. */

/* P2-1: mglRendererProgramHasSampledResourceNamed is defined in
 * MGLRenderer+Draw.m, also called from MGLRenderer+Batch.m. */
bool mglRendererProgramHasSampledResourceNamed(Program *program, const char *name);

@interface MGLRenderer ()

// === Draw batch scheduling and execution ===
- (MGLBatchPath)scheduleDrawBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (BOOL)checkBatchShouldExecute:(MGLDrawBatch *)batch
                        context:(GLMContext)glm_ctx
                        flushId:(uint64_t)flushId
                     batchIndex:(uint32_t)batchIndex
                    replayError:(GLenum *)replayError
                skippedCommands:(uint32_t *)skippedCommands;
- (void)recordBatchCommandStats:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (BOOL)issueStreamMergedMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (BOOL)issueIndirectCommandBufferBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (id<MTLBuffer>)mdiArgumentScratchBufferWithLength:(NSUInteger)length
                                             offset:(NSUInteger *)offsetOut;

// === Resource binding sync ===
- (bool)syncResourceBindingsForContext:(GLMContext)glm_ctx;
- (bool)bindVertexBuffersToCurrentRenderEncoder;
- (bool)bindFragmentBuffersToCurrentRenderEncoder;
- (bool)bindActiveTexturesToMTL;

// === Parallel command recording infrastructure ===
- (void)saveDedupStateToWorker:(MGLWorkerContext *)worker;
- (void)loadDedupStateFromWorker:(const MGLWorkerContext *)worker;
- (BOOL)parallelEncodeEnabled;
- (MGLBatchPath)encodeBatchForParallelWorker:(MGLWorkerContext *)worker
                                       batch:(MGLDrawBatch *)batch
                                     context:(GLMContext)glm_ctx
                                     flushId:(uint64_t)flushId
                                  batchIndex:(uint32_t)batchIndex
                                  savedState:(const GLMState *)savedState
                                    executed:(BOOL *)executedOut;

// === Dedup state management ===
- (void)invalidateLastBoundState;
- (void)recordLastBoundVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)recordLastBoundFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)invalidateLastBoundVertexBufferAtIndex:(NSUInteger)index;
- (void)invalidateLastBoundFragmentBufferAtIndex:(NSUInteger)index;
- (void)setVertexTextureIfNeeded:(id<MTLTexture>)texture atIndex:(NSUInteger)index;
- (void)setFragmentTextureIfNeeded:(id<MTLTexture>)texture atIndex:(NSUInteger)index;
- (void)setVertexSamplerStateIfNeeded:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index;
- (void)setFragmentSamplerStateIfNeeded:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index;
- (void)setViewportIfNeeded:(MTLViewport)viewport;
- (void)setScissorRectIfNeeded:(MTLScissorRect)rect;
- (void)setTriangleFillModeIfNeeded:(MTLTriangleFillMode)mode;

// === Locked draw variants ===
- (void)mtlDrawArraysLocked:(GLMContext)ctx mode:(GLenum)mode first:(GLint)first count:(GLsizei)count;
- (void)mtlDrawElementsLocked:(GLMContext)glm_ctx mode:(GLenum)mode count:(GLsizei)count type:(GLenum)type indices:(const void *)indices;

// === Methods defined in MGLRenderer.m, called from MGLRenderer+Draw.m ===
- (int)getVertexBufferIndexWithAttributeSet:(int)attribute;
- (NSUInteger)getProgramBindingRequiredSize:(int)stage type:(int)type index:(int)index;
- (NSUInteger)getProgramBindingRequiredSizeForStage:(int)stage clientBinding:(GLuint)clientBinding;
- (NSInteger)getProgramMetalBufferIndexForStage:(int)stage clientBinding:(GLuint)clientBinding;
- (id<MTLBuffer>)floatVertexBufferForDoubleAttrib:(Buffer *)sourceBuffer
                                         resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                             size:(GLuint)componentCount
                                         outStride:(NSUInteger *)outStride;
- (id<MTLBuffer>)floatVertexBufferForIntAttrib:(Buffer *)sourceBuffer
                                      resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                          size:(GLuint)componentCount
                                    normalized:(GLboolean)normalized
                                          type:(GLenum)type
                                     outStride:(NSUInteger *)outStride;
- (id<MTLBuffer>)integerVertexBufferForAttrib:(Buffer *)sourceBuffer
                                     resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                         size:(GLuint)componentCount
                                       srcType:(GLenum)srcType
                                     dstIsInt:(BOOL)dstIsInt
                                    outStride:(NSUInteger *)outStride;
- (id<MTLSamplerState>)fallbackSamplerState;
- (int)getProgramBinding:(int)stage type:(int)type index:(int)index;
- (int)getProgramGLBinding:(int)stage type:(int)type index:(int)index;
- (GLuint)textureUnitForSampledResource:(SpirvResource *)sampledResource
                            metalBinding:(GLuint)metalBinding
                                  stage:(int)stage;
- (MTLTextureType)getProgramExpectedTextureType:(int)stage type:(int)type index:(int)index;
- (MTLTextureType)getProgramDeclaredTextureType:(int)stage type:(int)type index:(int)index;
- (MGLTextureDataKind)getProgramExpectedTextureDataKind:(int)stage type:(int)type index:(int)index;
- (Texture *)textureForSampledResource:(SpirvResource *)sampledResource
                          metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
                           expectedType:(MTLTextureType)expectedType;
- (id<MTLTexture>)fallbackSampledTextureForExpectedType:(MTLTextureType)expectedType
                                               dataKind:(MGLTextureDataKind)dataKind;
- (int)textureIndexForExpectedMetalType:(MTLTextureType)expectedType;
- (void)traceSampledTextureReadback:(id<MTLTexture>)texture
                              glTex:(Texture *)glTex
                              level:(TextureLevel *)level0
                            program:(GLuint)program
                            binding:(GLuint)binding
                              stage:(NSString *)stage
                             reason:(NSString *)reason
                                hit:(uint64_t)hit;
- (bool)processBuffer:(Buffer *)ptr;
- (bool)dispatchTessControlShader:(GLMContext)glm_ctx
                          program:(Program *)tcsProgram
                            first:(GLint)first
                            count:(GLsizei)count
                        indexType:(GLenum)indexType
                          indices:(const void *)indices
                       baseVertex:(GLint)baseVertex
                     instanceCount:(GLsizei)drawInstanceCount
                     baseInstance:(GLuint)baseInstance;
- (bool)dispatchTessEvaluationShader:(GLMContext)glm_ctx
                            program:(Program *)tesProgram
                              first:(GLint)first
                            count:(GLsizei)count;

// === P2-1: Methods defined in MGLRenderer+Draw.m, called from MGLRenderer+Batch.m ===
- (void)issueMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (void)issueDirectBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (bool)bindTexturesToCurrentRenderEncoder;
- (BOOL)currentDrawRasterizationIsEmpty;
- (BOOL)currentDrawModeIsFullyCulled:(GLenum)mode;
- (void)applyPolygonOffsetForDrawMode:(GLenum)mode;
- (BOOL)resolveElementBufferForCommand:(const MGLDrawCommand *)cmd
                                  label:(const char *)label
                                context:(GLMContext)drawCtx
                               glBuffer:(Buffer **)glBufferOut
                              mtlBuffer:(id<MTLBuffer> *)mtlBufferOut;

// === P2-1: Methods defined in MGLRenderer+Batch.m, called from MGLRenderer+Draw.m ===
- (void)traceReplayCommand:(MGLDrawBatch *)batch
                   command:(MGLDrawCommand *)cmd
                   context:(GLMContext)glm_ctx
                   flushId:(uint64_t)flushId
                batchIndex:(uint32_t)batchIndex
              commandIndex:(uint32_t)commandIndex
                     phase:(const char *)phase
                    reason:(const char *)reason;
- (void)recordArrayDrawSubmittedMode:(GLenum)mode vertexCount:(uint64_t)vertexCount;
- (void)recordElementDrawSubmittedMode:(GLenum)mode indexCount:(uint64_t)indexCount;

@end

#endif /* MGLRenderer_Draw_Private_h */
