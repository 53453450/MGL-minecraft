#ifndef MGLPipelineCache_h
#define MGLPipelineCache_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "glm_context.h"

NS_ASSUME_NONNULL_BEGIN

#define MGL_PIPELINE_CACHE_KEY_WORDS 7u

/* Immutable value key for the PSO/descriptor caches: {primaryKey, vertex MSL
 * instance/generation, fragment MSL instance/generation, pipeline descriptor
 * signature, vertex descriptor signature}.  Replaces the hex NSString key so
 * lookups hash/compare 7 words instead of formatting and comparing a
 * 118-char string on every pipeline resolve. */
@interface MGLPipelineCacheKey : NSObject <NSCopying>
- (instancetype)initWithWords:(const uint64_t[MGL_PIPELINE_CACHE_KEY_WORDS])words;
/* Rewrites the cached words of a reusable query key.  The returned object
 * must only be used for cache lookups; store paths must allocate a fresh
 * MGLPipelineCacheKey instead. */
- (void)overwriteWords:(const uint64_t[MGL_PIPELINE_CACHE_KEY_WORDS])words;
@end

@class MGLDepthStencilCacheKey;

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
    NSMutableDictionary<MGLPipelineCacheKey *, id> *__strong _Nullable pipelineStateCache;
    NSMutableOrderedSet<MGLPipelineCacheKey *> *__strong _Nullable pipelineStateCacheLRU;
    NSMutableDictionary<MGLPipelineCacheKey *, MTLRenderPipelineDescriptor *> *__strong _Nullable pipelineDescriptorCache;
    NSMutableOrderedSet<MGLPipelineCacheKey *> *__strong _Nullable pipelineDescriptorCacheLRU;
    NSMutableDictionary *__strong _Nullable depthStencilStateCache;
    NSMutableOrderedSet *__strong _Nullable depthStencilStateCacheLRU;
    MGLDepthStencilCacheKey *__strong _Nullable depthStencilCacheQueryKey;
    /* Reusable zero-alloc PSO lookup key (see MGLPipelineCacheKey
     * overwriteWords: contract). */
    MGLPipelineCacheKey *__strong _Nullable pipelineCacheQueryKey;
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
- (nullable id)pipelineEntryForKey:(MGLPipelineCacheKey *)key;
- (void)markPipelineEntryUsedForKey:(MGLPipelineCacheKey *)key;
- (nullable MTLRenderPipelineDescriptor *)pipelineDescriptorForKey:(MGLPipelineCacheKey *)key;
/* Returns a reusable query key populated with words.  Zero per-lookup
 * allocation; the object is only valid for cache lookups. */
- (nonnull MGLPipelineCacheKey *)pipelineQueryKeyForWords:
    (const uint64_t[MGL_PIPELINE_CACHE_KEY_WORDS])words;
- (NSUInteger)storePipelineEntry:(id)entry forKey:(MGLPipelineCacheKey *)key;
- (void)storePipelineDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                         forKey:(MGLPipelineCacheKey *)key;
/* Typed cache path used by the renderer. Metal-cpp mode stays allocation-free
 * on hits; the ObjC key/dictionary path remains only as the temporary A/B
 * baseline until the facade itself is deleted. */
- (BOOL)lookupPipelineForWords:(const uint64_t * _Nonnull)words
                      pipeline:(id<MTLRenderPipelineState> _Nullable * _Nonnull)pipelineOut
                vertexFunction:(id<MTLFunction> _Nullable * _Nonnull)vertexFunctionOut
              fragmentFunction:(id<MTLFunction> _Nullable * _Nonnull)fragmentFunctionOut;
- (nullable MTLRenderPipelineDescriptor *)pipelineDescriptorForWords:
    (const uint64_t * _Nonnull)words;
/* P4.2: descriptor cache 的 value-state 版（gate-on）。命中返回 YES 并拷贝
 * state；未命中返回 NO。gate-off 走 ObjC descriptor 字典（见上）。 */
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
/* P4.2: blend state owner-first 读取（gate-on 用）。命中 C++ owner 返回 YES；
 * 否则回退 ObjC 镜像并返回 YES；越界返回 NO。 */
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
