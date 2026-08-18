#ifndef MGLPipelineCache_h
#define MGLPipelineCache_h

#import <Foundation/Foundation.h>

#include "glm_context.h"

NS_ASSUME_NONNULL_BEGIN

#define MGL_PIPELINE_CACHE_KEY_WORDS 7u

/* P4.2: final/simple/safe pipeline descriptor 的 value-state（完整定义在
 * mgl_air_loader.h）。ObjC 只构造 value-state，不再组装平台对象。 */
typedef struct MGLRenderCppPipelineDescriptorState
    MGLRenderCppPipelineDescriptorState;
typedef struct MGLRenderCppPipelineBlendState_t
    MGLRenderCppPipelineBlendState;
typedef struct MGLRenderCppDepthStencilDescriptorState_t
    MGLRenderCppDepthStencilDescriptorState;

NS_ASSUME_NONNULL_END

typedef struct MGLPipelineCacheState_t {
    /* Metal objects are owned by the C++ cache owner.  These borrowed opaque
     * identities are retained only so the GL-semantic ObjC layer can test and
     * pass the active handles without importing Metal object types. */
    void * _Nullable pipelineState;
    /* Keep the native NSUInteger-sized value width while remaining a plain
     * C value; Metal enum names stay out of this interface. */
    uint64_t pipelineColor0Format;
    uint64_t pipelineDepthFormat;
    uint64_t pipelineStencilFormat;
    GLuint pipelineProgramName;
    void * _Nullable pipelineVertexFunction;
    void * _Nullable pipelineFragmentFunction;
    BOOL dsCacheEnabled;
    BOOL psoDedupEnabled;
} MGLPipelineCacheState;

NS_ASSUME_NONNULL_BEGIN

@interface MGLPipelineCache : NSObject {
@private
    MGLPipelineCacheState _state;
    void *_device;
    void *_cppOwner;
    BOOL _binaryArchiveRequested;
}

@property(nonatomic, readonly) const MGLPipelineCacheState *state;
/* Borrowed platform handle; the C++ owner performs the retain. */
@property(nonatomic, assign, nullable) id device;
@property(nonatomic, readonly, getter=isBinaryArchiveEnabled)
    BOOL binaryArchiveEnabled;

- (instancetype)initWithPSODedupEnabled:(BOOL)psoDedupEnabled
                depthStencilCacheEnabled:(BOOL)depthStencilCacheEnabled
                     binaryArchiveEnabled:(BOOL)binaryArchiveEnabled;

- (nullable id)depthStencilStateForValueState:
    (const MGLRenderCppDepthStencilDescriptorState *)state;
/* Typed cache path backed exclusively by the C++ pipeline-cache owner. */
- (BOOL)lookupPipelineForWords:(const uint64_t * _Nonnull)words
                      pipeline:(id _Nullable * _Nonnull)pipelineOut
                vertexFunction:(id _Nullable * _Nonnull)vertexFunctionOut
              fragmentFunction:(id _Nullable * _Nonnull)fragmentFunctionOut;
/* Descriptor cache value-state lookup. A hit copies state and returns YES. */
- (BOOL)pipelineDescriptorStateForWords:
    (const uint64_t * _Nonnull)words
      state:(MGLRenderCppPipelineDescriptorState * _Nonnull)stateOut;
- (NSUInteger)storePipeline:(id)pipeline
              vertexFunction:(nullable id)vertexFunction
            fragmentFunction:(nullable id)fragmentFunction
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
- (void)setPipelineState:(nullable id)pipelineState;
- (void)activatePipelineState:(nullable id)pipelineState
                 color0Format:(uint32_t)color0Format
                  depthFormat:(uint32_t)depthFormat
                stencilFormat:(uint32_t)stencilFormat
                  programName:(GLuint)programName
               vertexFunction:(nullable id)vertexFunction
             fragmentFunction:(nullable id)fragmentFunction;
- (void)setBlendFactorsForAttachment:(NSUInteger)index
                        srcRgbFactor:(uint32_t)srcRgbFactor
                      srcAlphaFactor:(uint32_t)srcAlphaFactor
                        dstRgbFactor:(uint32_t)dstRgbFactor
                      dstAlphaFactor:(uint32_t)dstAlphaFactor
                        rgbOperation:(uint32_t)rgbOperation
                      alphaOperation:(uint32_t)alphaOperation
                           colorMask:(uint32_t)colorMask;
- (void)disableBinaryArchive;

@end

NS_ASSUME_NONNULL_END

#endif /* MGLPipelineCache_h */
