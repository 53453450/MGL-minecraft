#ifndef MGLPipelineCache_h
#define MGLPipelineCache_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "glm_context.h"

NS_ASSUME_NONNULL_BEGIN

#define MGL_PIPELINE_CACHE_KEY_WORDS 7u

/* P4.2: final/simple/safe pipeline descriptor 的 value-state（完整定义在
 * mgl_air_loader.h）。ObjC 只构造 value-state，不再组装
 * MTLRenderPipelineDescriptor。 */
typedef struct MGLRenderCppPipelineDescriptorState
    MGLRenderCppPipelineDescriptorState;
typedef struct MGLRenderCppPipelineBlendState_t
    MGLRenderCppPipelineBlendState;

NS_ASSUME_NONNULL_END

typedef struct MGLPipelineCacheState_t {
    MTLBlendFactor src_blend_rgb_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor dst_blend_rgb_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor src_blend_alpha_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor dst_blend_alpha_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendOperation rgb_blend_operation[MAX_COLOR_ATTACHMENTS];
    MTLBlendOperation alpha_blend_operation[MAX_COLOR_ATTACHMENTS];
    MTLColorWriteMask color_mask[MAX_COLOR_ATTACHMENTS];
    id<MTLRenderPipelineState> __strong _Nullable pipelineState;
    MTLPixelFormat pipelineColor0Format;
    MTLPixelFormat pipelineDepthFormat;
    MTLPixelFormat pipelineStencilFormat;
    GLuint pipelineProgramName;
    id<MTLFunction> __strong _Nullable pipelineVertexFunction;
    id<MTLFunction> __strong _Nullable pipelineFragmentFunction;
    BOOL dsCacheEnabled;
    BOOL psoDedupEnabled;
} MGLPipelineCacheState;

NS_ASSUME_NONNULL_BEGIN

@interface MGLPipelineCache : NSObject {
@private
    MGLPipelineCacheState _state;
    id<MTLDevice> _device;
    void *_cppOwner;
    BOOL _binaryArchiveRequested;
}

@property(nonatomic, readonly) const MGLPipelineCacheState *state;
@property(nonatomic, strong, nullable) id<MTLDevice> device;
@property(nonatomic, readonly, getter=isBinaryArchiveEnabled)
    BOOL binaryArchiveEnabled;

- (instancetype)initWithPSODedupEnabled:(BOOL)psoDedupEnabled
                depthStencilCacheEnabled:(BOOL)depthStencilCacheEnabled
                     binaryArchiveEnabled:(BOOL)binaryArchiveEnabled;

- (nullable id<MTLDepthStencilState>)depthStencilStateForDescriptor:
    (MTLDepthStencilDescriptor *)descriptor;
/* Typed cache path backed exclusively by the C++ pipeline-cache owner. */
- (BOOL)lookupPipelineForWords:(const uint64_t * _Nonnull)words
                      pipeline:(id<MTLRenderPipelineState> _Nullable * _Nonnull)pipelineOut
                vertexFunction:(id<MTLFunction> _Nullable * _Nonnull)vertexFunctionOut
              fragmentFunction:(id<MTLFunction> _Nullable * _Nonnull)fragmentFunctionOut;
- (nullable MTLRenderPipelineDescriptor *)pipelineDescriptorForWords:
    (const uint64_t * _Nonnull)words;
/* Descriptor cache value-state lookup. A hit copies state and returns YES. */
- (BOOL)pipelineDescriptorStateForWords:
    (const uint64_t * _Nonnull)words
      state:(MGLRenderCppPipelineDescriptorState * _Nonnull)stateOut;
- (NSUInteger)storePipeline:(id<MTLRenderPipelineState>)pipeline
              vertexFunction:(nullable id<MTLFunction>)vertexFunction
            fragmentFunction:(nullable id<MTLFunction>)fragmentFunction
                    forWords:(const uint64_t * _Nonnull)words;
- (void)storePipelineDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                       forWords:(const uint64_t * _Nonnull)words;
- (void)storePipelineDescriptorState:
    (const MGLRenderCppPipelineDescriptorState * _Nonnull)state
                            forWords:(const uint64_t * _Nonnull)words;
/* Blend state lookup from the C++ owner. */
- (BOOL)blendStateForAttachment:(NSUInteger)index
                            out:(MGLRenderCppPipelineBlendState * _Nonnull)outState;

- (void)loadBinaryArchive;
- (void)saveBinaryArchive;
/* Complete VS+FS descriptors query the binary archive first. A miss falls
 * back to ordinary compilation and is then added exactly once; incomplete
 * descriptors bypass the archive. */
- (nullable id<MTLRenderPipelineState>)createRenderPipelineStateWithDescriptor:
    (MTLRenderPipelineDescriptor *)descriptor
    error:(NSError * _Nullable * _Nullable)error;
- (int)createRenderPipelineFromState:
    (const MGLRenderCppPipelineDescriptorState * _Nonnull)state
    vertexFunction:(void * _Nonnull)vertexFunction
    fragmentFunction:(void * _Nullable)fragmentFunction
    pipelineOut:(void * _Nullable * _Nonnull)pipelineOut
    errorMessage:(char * _Nullable)errorMessage
    errorCapacity:(size_t)errorCapacity;

- (void)resetCaches;
- (void)shutdown;

/* Low-level pipeline-state mutators used by the PSO build path. These write
 * the tracked state so the manager owns all writes to _state (no raw writable
 * state pointer escapes). */
- (void)invalidatePipelineState;
- (void)setPipelineState:(nullable id<MTLRenderPipelineState>)pipelineState;
- (void)activatePipelineState:(nullable id<MTLRenderPipelineState>)pipelineState
                 color0Format:(MTLPixelFormat)color0Format
                  depthFormat:(MTLPixelFormat)depthFormat
                stencilFormat:(MTLPixelFormat)stencilFormat
                  programName:(GLuint)programName
               vertexFunction:(nullable id<MTLFunction>)vertexFunction
             fragmentFunction:(nullable id<MTLFunction>)fragmentFunction;
- (void)setBlendFactorsForAttachment:(NSUInteger)index
                        srcRgbFactor:(MTLBlendFactor)srcRgbFactor
                      srcAlphaFactor:(MTLBlendFactor)srcAlphaFactor
                        dstRgbFactor:(MTLBlendFactor)dstRgbFactor
                      dstAlphaFactor:(MTLBlendFactor)dstAlphaFactor
                        rgbOperation:(MTLBlendOperation)rgbOperation
                      alphaOperation:(MTLBlendOperation)alphaOperation
                           colorMask:(MTLColorWriteMask)colorMask;
- (void)disableBinaryArchive;

@end

NS_ASSUME_NONNULL_END

#endif /* MGLPipelineCache_h */
