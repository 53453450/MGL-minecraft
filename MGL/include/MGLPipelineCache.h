#ifndef MGLPipelineCache_h
#define MGLPipelineCache_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifndef MGL_HAS_MTL4_COMPILER
#if __has_include(<Metal/MTL4Compiler.h>) && __has_include(<Metal/MTL4LibraryDescriptor.h>)
#import <Metal/MTL4Compiler.h>
#import <Metal/MTL4LibraryDescriptor.h>
#define MGL_HAS_MTL4_COMPILER 1
#else
#define MGL_HAS_MTL4_COMPILER 0
#endif
#endif

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
@end

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
    BOOL dsCacheEnabled;
    id<MTLBinaryArchive> __strong _Nullable binaryArchive;
    BOOL binaryArchiveEnabled;
#if MGL_HAS_MTL4_COMPILER
    id<MTL4Compiler> __strong _Nullable mtl4Compiler;
#endif
    BOOL psoDedupEnabled;
} MGLPipelineCacheState;

NS_ASSUME_NONNULL_BEGIN

@interface MGLPipelineCache : NSObject {
@private
    MGLPipelineCacheState _state;
    id<MTLDevice> _device;
}

@property(nonatomic, readonly) const MGLPipelineCacheState *state;
@property(nonatomic, strong, nullable) id<MTLDevice> device;

- (instancetype)initWithPSODedupEnabled:(BOOL)psoDedupEnabled
                depthStencilCacheEnabled:(BOOL)depthStencilCacheEnabled
                     binaryArchiveEnabled:(BOOL)binaryArchiveEnabled;

- (nullable id<MTLDepthStencilState>)depthStencilStateForDescriptor:
    (MTLDepthStencilDescriptor *)descriptor;
- (nullable id)pipelineEntryForKey:(MGLPipelineCacheKey *)key;
- (void)markPipelineEntryUsedForKey:(MGLPipelineCacheKey *)key;
- (nullable MTLRenderPipelineDescriptor *)pipelineDescriptorForKey:(MGLPipelineCacheKey *)key;
- (NSUInteger)storePipelineEntry:(id)entry forKey:(MGLPipelineCacheKey *)key;
- (void)storePipelineDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                         forKey:(MGLPipelineCacheKey *)key;

- (void)initializeCompilerIfAvailableUnlessDisabled:(BOOL)disabled;
- (nullable id<MTLLibrary>)newMetalLibraryWithSource:(NSString *)source
                                              options:(nullable MTLCompileOptions *)options
                                                label:(nullable NSString *)label
                                                error:(NSError **)error;

- (void)loadBinaryArchive;
- (void)saveBinaryArchive;
- (void)applyBinaryArchiveToDescriptor:(MTLRenderPipelineDescriptor *)descriptor;
- (void)addPipelineToBinaryArchive:(MTLRenderPipelineDescriptor *)descriptor;

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
