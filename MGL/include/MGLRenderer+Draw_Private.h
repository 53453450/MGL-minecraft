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

#import "MGLRenderer.h"
#include "mgl_air_tess_abi.h"
#include "mgl_air_gs_abi.h"

/* Encode target passed explicitly to the issue and bind methods instead of
 * read from _renderPassManager.state->currentRenderEncoder. */
typedef struct {
    id<MTLRenderCommandEncoder> encoder;
    void *render_encoder_owner;
} MGLEncodeContext;

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
    uint32_t first_vertex;
    uint32_t explicit_vertex_count;
    uint32_t explicit_vertices[4];
    uint32_t first_instance;
    uint32_t instance_stride;
} MGLCullDistanceEmuParams;

/* === Diagnostic constants === */
static const BOOL kMGLDiagnosticStateLogs = NO;
static const BOOL kMGLDrawSubmitDiagnostics = NO;

/* === Draw binding/validation constants === */
static const BOOL kMGLVerboseBindLogs = NO;
static const NSUInteger kMGLMinimumStageBindingSize = 256;
static const NSUInteger kMGLDefaultStageFallbackBufferSize = 4096;
/* Stack scratch used to zero-pad small inline stage bindings up to the size the
 * shader argument requires.  Matches Metal's 4 KB set*Bytes limit.  A macro so
 * it can size a local array. */
#define kMGLStageBindingStackScratchSize 4096u
static const BOOL kMGLEnableVertexAllSlotFallback = YES;
static const BOOL kMGLEnableSampledTextureFallback = YES;
/* VBO-range guards: block draws whose vertex inputs would read out of bounds.
 * Set MGL_VALIDATE_VBO_RANGE=0 to disable (per-draw attrib re-resolution can
 * cost ~0.5-1µs per draw call).  Defaults: on in debug builds, off in release. */
static inline BOOL mglVboRangeValidationEnabled(void)
{
#if defined(DEBUG) || defined(MGL_DEBUG)
    return YES;
#else
    return mglEnvFlagEnabledDefaultOn("MGL_VALIDATE_VBO_RANGE");
#endif
}

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
                                          void *commandBufferOwner,
                                          void *renderEncoderOwner,
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

/* mglRendererProgramHasSampledResourceNamed is defined in
 * MGLRenderer+Draw.m, also called from MGLRenderer+Batch.m. */
bool mglRendererProgramHasSampledResourceNamed(Program *program, const char *name);

@interface MGLRenderer ()

/* Temporary backend operation targets. These remain private to the renderer
 * while P5.4 moves draw execution into the C++ backend. */
- (void)mtlDrawArrays:(GLMContext)ctx mode:(GLenum)mode
                 first:(GLint)first count:(GLsizei)count;
- (void)mtlDrawElements:(GLMContext)ctx mode:(GLenum)mode
                   count:(GLsizei)count type:(GLenum)type
                 indices:(const void *)indices;
- (void)mtlDrawRangeElements:(GLMContext)ctx mode:(GLenum)mode
                         start:(GLuint)start end:(GLuint)end
                         count:(GLsizei)count type:(GLenum)type
                       indices:(const void *)indices;
- (void)mtlDrawArraysInstanced:(GLMContext)ctx mode:(GLenum)mode
                          first:(GLint)first count:(GLsizei)count
                  instancecount:(GLsizei)instancecount;
- (void)mtlDrawElementsInstanced:(GLMContext)ctx mode:(GLenum)mode
                            count:(GLsizei)count type:(GLenum)type
                          indices:(const void *)indices
                    instancecount:(GLsizei)instancecount;
- (void)mtlDrawElementsBaseVertex:(GLMContext)ctx mode:(GLenum)mode
                             count:(GLsizei)count type:(GLenum)type
                           indices:(const void *)indices
                        basevertex:(GLint)basevertex;
- (void)mtlDrawRangeElementsBaseVertex:(GLMContext)ctx mode:(GLenum)mode
                                   start:(GLuint)start end:(GLuint)end
                                   count:(GLsizei)count type:(GLenum)type
                                 indices:(const void *)indices
                              basevertex:(GLint)basevertex;
- (void)mtlDrawElementsInstancedBaseVertex:(GLMContext)ctx mode:(GLenum)mode
                                      count:(GLsizei)count type:(GLenum)type
                                    indices:(const void *)indices
                              instancecount:(GLsizei)instancecount
                                 basevertex:(GLint)basevertex;
- (void)mtlDrawArraysIndirect:(GLMContext)ctx mode:(GLenum)mode
                       indirect:(const void *)indirect;
- (void)mtlDrawElementsIndirect:(GLMContext)ctx mode:(GLenum)mode
                          type:(GLenum)type indirect:(const void *)indirect;
- (void)mtlDrawArraysInstancedBaseInstance:(GLMContext)ctx mode:(GLenum)mode
                                      first:(GLint)first count:(GLsizei)count
                              instancecount:(GLsizei)instancecount
                                baseinstance:(GLuint)baseinstance;
- (void)mtlDrawElementsInstancedBaseInstance:(GLMContext)ctx mode:(GLenum)mode
                                        count:(GLsizei)count type:(GLenum)type
                                      indices:(const void *)indices
                                instancecount:(GLsizei)instancecount
                                  baseinstance:(GLuint)baseinstance;
- (void)mtlDrawElementsInstancedBaseVertexBaseInstance:
            (GLMContext)ctx mode:(GLenum)mode count:(GLsizei)count
            type:(GLenum)type indices:(const void *)indices
            instancecount:(GLsizei)instancecount
            basevertex:(GLint)basevertex baseinstance:(GLuint)baseinstance;
- (void)mtlMultiDrawArrays:(GLMContext)ctx mode:(GLenum)mode
                      first:(const GLint *)first count:(const GLsizei *)count
                  drawcount:(GLsizei)drawcount;
- (void)mtlMultiDrawElements:(GLMContext)ctx mode:(GLenum)mode
                       count:(const GLsizei *)count type:(GLenum)type
                     indices:(const void *const *)indices
                   drawcount:(GLsizei)drawcount;
- (void)mtlMultiDrawElementsBaseVertex:(GLMContext)ctx mode:(GLenum)mode
                                 count:(const GLsizei *)count type:(GLenum)type
                               indices:(const void *const *)indices
                             drawcount:(GLsizei)drawcount
                            basevertex:(const GLint *)basevertex;
- (void)mtlMultiDrawArraysIndirect:(GLMContext)ctx mode:(GLenum)mode
                           indirect:(const void *)indirect
                          drawcount:(GLsizei)drawcount stride:(GLsizei)stride;
- (void)mtlMultiDrawElementsIndirect:(GLMContext)ctx mode:(GLenum)mode
                              type:(GLenum)type indirect:(const void *)indirect
                         drawcount:(GLsizei)drawcount stride:(GLsizei)stride;

// === Draw batch scheduling and execution ===
- (MGLBatchPath)scheduleDrawBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (BOOL)checkBatchShouldExecute:(MGLDrawBatch *)batch
                        context:(GLMContext)glm_ctx
                        flushId:(uint64_t)flushId
                     batchIndex:(uint32_t)batchIndex
                    replayError:(GLenum *)replayError
                skippedCommands:(uint32_t *)skippedCommands;
- (void)recordBatchCommandStats:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx;
- (BOOL)issueStreamMergedMDIBatch:(MGLDrawBatch *)batch
                          context:(GLMContext)glm_ctx
                    encodeContext:(const MGLEncodeContext *)encCtx;
- (BOOL)issueIndirectCommandBufferBatch:(MGLDrawBatch *)batch
                                context:(GLMContext)glm_ctx
                          encodeContext:(const MGLEncodeContext *)encCtx;
- (id<MTLBuffer>)mdiArgumentScratchBufferWithLength:(NSUInteger)length
                                             offset:(NSUInteger *)offsetOut;

// === Resource binding sync ===
/* Work already performed by processDirtyStateDomainsLocked within the same
 * processGLState invocation; syncResourceBindingsForContext skips these
 * steps instead of repeating the full rebind (which used to run twice per
 * draw). */
typedef struct {
    bool mappedBuffers;
    bool updatedBaseLists;
    bool boundActiveTextures;
} MGLResourceSyncWork;
- (bool)syncResourceBindingsForContext:(GLMContext)glm_ctx
                           alreadyDone:(const MGLResourceSyncWork *)done;
- (bool)bindVertexBuffersToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx;
- (bool)bindFragmentBuffersToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx;
- (bool)bindActiveTexturesToMTL;

// === Dedup state management ===
- (void)invalidateLastBoundState;
- (void)recordLastBoundVertexBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)recordLastBoundFragmentBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)invalidateLastBoundVertexBufferAtIndex:(NSUInteger)index;
- (void)invalidateLastBoundFragmentBufferAtIndex:(NSUInteger)index;
- (void)setViewportIfNeeded:(MTLViewport)viewport;
- (void)setScissorRectIfNeeded:(MTLScissorRect)rect;
- (void)setTriangleFillModeIfNeeded:(MTLTriangleFillMode)mode;

// === Locked draw variants ===
- (void)mtlDrawArraysLocked:(GLMContext)ctx mode:(GLenum)mode first:(GLint)first count:(GLsizei)count;
- (void)mtlDrawElementsLocked:(GLMContext)glm_ctx mode:(GLenum)mode count:(GLsizei)count type:(GLenum)type indices:(const void *)indices;

// === Methods defined in MGLRenderer.m, called from MGLRenderer+Draw.m ===
// getVertexBufferIndexWithAttributeSet: and floatVertexBufferFor*Attrib: are
// now declared in MGLRenderer+Buffer_Private.h (implemented in +Buffer.m).
// getProgramBinding* / getProgramMetalBufferIndexForStage: /
// getProgramBindingRequiredSize* / getProgramExpectedTexture* /
// getProgramDeclaredTextureType: are now declared in
// MGLRenderer+ProgramBinding_Private.h (implemented in +ProgramBinding.m).
- (id<MTLSamplerState>)fallbackSamplerState;
- (GLuint)textureUnitForSampledResource:(MGLShaderResource *)sampledResource
                            metalBinding:(GLuint)metalBinding
                                  stage:(int)stage;
/* program-resolved variant — skips mglResolveProgramForStageFromState. */
- (GLuint)textureUnitForSampledResource:(MGLShaderResource *)sampledResource
                                program:(Program *)program
                           metalBinding:(GLuint)metalBinding
                                  stage:(int)stage;
- (Texture *)textureForSampledResource:(MGLShaderResource *)sampledResource
                          metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
                           expectedType:(MTLTextureType)expectedType;
/* textureUnit-resolved variant — caller passes the already-computed
 * texture unit, skipping the internal textureUnitForSampledResource: call. */
- (Texture *)textureForSampledResource:(MGLShaderResource *)sampledResource
                          metalBinding:(GLuint)metalBinding
                                  stage:(int)stage
                           expectedType:(MTLTextureType)expectedType
                          textureUnit:(GLuint)textureUnit;
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
                         contract:(const MGLAIRTessDrawContract *)contract;
- (bool)dispatchTessEvaluationShader:(GLMContext)glm_ctx
                            program:(Program *)tesProgram
                           contract:(const MGLAIRTessDrawContract *)contract;

- (void)issueMDIBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
        encodeContext:(const MGLEncodeContext *)encCtx;
- (void)issueDirectBatch:(MGLDrawBatch *)batch context:(GLMContext)glm_ctx
           encodeContext:(const MGLEncodeContext *)encCtx;
- (bool)applySamplerSnapshotForCommand:(const MGLDrawCommand *)cmd
                                context:(GLMContext)glm_ctx
                          encodeContext:(const MGLEncodeContext *)encCtx;
- (bool)bindTexturesToCurrentRenderEncoder:(const MGLEncodeContext *)encCtx;
- (BOOL)currentDrawRasterizationIsEmpty;
- (BOOL)currentDrawModeIsFullyCulled:(GLenum)mode;
- (void)applyPolygonOffsetForDrawMode:(GLenum)mode;
- (BOOL)resolveElementBufferForCommand:(const MGLDrawCommand *)cmd
                                  label:(const char *)label
                                context:(GLMContext)drawCtx
                               glBuffer:(Buffer **)glBufferOut
                              mtlBuffer:(id<MTLBuffer> *)mtlBufferOut;

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
- (void)bindCullDistanceEmulationBuffers:(GLenum)mode
                             firstVertex:(GLuint)firstVertex
                        explicitVertices:(const GLuint *)explicitVertices
                      explicitVertexCount:(GLuint)explicitVertexCount
                           encodeContext:(const MGLEncodeContext *)encCtx;
- (BOOL)captureAIRCullDistancesForArrayDraw:(GLMContext)drawCtx
                                      first:(GLint)first
                                      count:(GLsizei)count
                              instanceCount:(GLsizei)instanceCount
                               baseInstance:(GLuint)baseInstance;
- (BOOL)captureAIRCullDistancesForElementDraw:(GLMContext)drawCtx
                                    indexBytes:(const uint8_t *)indexBytes
                                     indexType:(GLenum)indexType
                                         count:(GLsizei)count
                                    baseVertex:(GLint)baseVertex
                                 instanceCount:(GLsizei)instanceCount
                                  baseInstance:(GLuint)baseInstance;
- (BOOL)encodeCullDistanceElementDraw:(GLenum)mode
                            indexBytes:(const uint8_t *)indexBytes
                             indexType:(GLenum)indexType
                                 count:(GLsizei)count
                            baseVertex:(GLint)baseVertex
                         instanceCount:(GLsizei)instanceCount
                          baseInstance:(GLuint)baseInstance
                       polygonLineMode:(BOOL)polygonLineMode
                         encodeContext:(const MGLEncodeContext *)encCtx;
- (BOOL)prepareAndEncodeDirectCullDistanceElementDraw:(GLenum)mode
                                           indexBytes:(const uint8_t *)indexBytes
                                            indexType:(GLenum)indexType
                                                count:(GLsizei)count
                                           baseVertex:(GLint)baseVertex
                                        instanceCount:(GLsizei)instanceCount
                                         baseInstance:(GLuint)baseInstance
                                      polygonLineMode:(BOOL)polygonLineMode;
- (BOOL)encodeCullDistanceArrayDraw:(GLenum)mode
                               first:(GLint)first
                               count:(GLsizei)count
                       instanceCount:(GLsizei)instanceCount
                        baseInstance:(GLuint)baseInstance
                       encodeContext:(const MGLEncodeContext *)encCtx;
- (bool)validateDrawArraysVertexInputs:(GLMContext)drawCtx
                                  mode:(GLenum)mode
                                 first:(GLint)first
                                 count:(GLsizei)count
                              drawCall:(uint64_t)drawCall;
- (BOOL)resolveElementBufferForDraw:(const char *)label
                            context:(GLMContext)drawCtx
                           glBuffer:(Buffer **)glBufferOut
                          mtlBuffer:(id<MTLBuffer> *)mtlBufferOut;
- (BOOL)resolveIndirectBufferForDraw:(const char *)label
                             context:(GLMContext)drawCtx
                            glBuffer:(Buffer **)glBufferOut
                           mtlBuffer:(id<MTLBuffer> *)mtlBufferOut;
- (BOOL)prepareEmulatedIndirectCPURead:(GLMContext)drawCtx label:(const char *)label;
- (BOOL)handleTessellationPatchDrawIfNeeded:(GLMContext)drawCtx
                                        mode:(GLenum *)mode
                                       first:(GLint)first
                                       count:(GLsizei)count
                                   indexType:(GLenum)indexType
                                     indices:(const void *)indices
                                  baseVertex:(GLint)baseVertex
                               instanceCount:(GLsizei)instanceCount
                                baseInstance:(GLuint)baseInstance
                                       label:(const char *)label;
- (BOOL)handleGeometryDrawIfNeeded:(GLMContext)drawCtx
                              mode:(GLenum)mode
                             first:(GLint)first
                             count:(GLsizei)count
                         indexType:(GLenum)indexType
                           indices:(const void *)indices
                        baseVertex:(GLint)baseVertex
                     instanceCount:(GLsizei)instanceCount
                      baseInstance:(GLuint)baseInstance
                             label:(const char *)label;
- (id<MTLBuffer>)captureAIRVertexPositionsForGeometryIndexed:(GLMContext)drawCtx
                                                  indexBuffer:(id<MTLBuffer>)indexBuffer
                                                    indexType:(MTLIndexType)indexType
                                                  indexOffset:(NSUInteger)indexOffset
                                                        count:(GLsizei)count
                                                    baseVertex:(GLint)baseVertex
                                                 instanceCount:(GLsizei)instanceCount
                                                  baseInstance:(GLuint)baseInstance
                                                     maxIndex:(uint32_t)maxIndex
                                                     outOffset:(NSUInteger *)outOffset;
- (BOOL)ensureAIRGeometryPassthroughFunctionForProgram:(Program *)program
                                      outputPrimitive:(MTLPrimitiveType)outputPrimitive;
- (BOOL)ensureAIRTessEvalPassthroughFunctionForProgram:(Program *)program;
- (bool)bindBuffersToComputeEncoder:(id<MTLComputeCommandEncoder>)encoder
                               stage:(int)stage
                           copyBacks:(MGLStageBindingCopyBackList *)copyBacks;
- (bool)bindBuffersToComputeEncoder:(id<MTLComputeCommandEncoder>)encoder
                               stage:(int)stage
                           copyBacks:(MGLStageBindingCopyBackList *)copyBacks
                       executionPlan:(MGLRenderCppComputeExecutionPlan *)executionPlan
                        temporaries:(NSMutableArray *)temporaries;
- (bool)bindTexturesToComputeEncoder:(id<MTLComputeCommandEncoder>)encoder
                                stage:(int)stage;
- (bool)bindTexturesToComputeEncoder:(id<MTLComputeCommandEncoder>)encoder
                                stage:(int)stage
                        executionPlan:(MGLRenderCppComputeExecutionPlan *)executionPlan
                         temporaries:(NSMutableArray *)temporaries;

@end

#endif /* MGLRenderer_Draw_Private_h */
