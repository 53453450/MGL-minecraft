#import "MGLPipelineCache.h"

#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"
#include "mgl_render_cpp_objc.h"
#include "mgl_air_loader.h"   /* MGLRenderCppPipelineDescriptorState */

@interface MGLPipelineCacheKey ()
- (void)copyWords:(uint64_t[MGL_PIPELINE_CACHE_KEY_WORDS])words;
@end

@interface MGLPipelineCache ()
- (BOOL)ensureCppOwnerCreated;
- (BOOL)ensureCppOwner;
@end

static BOOL mglPipelineCacheUsesMetalCpp(void)
{
    return mglRenderCppGetDevice() != NULL;
}

static NSError *mglPipelineCacheMetalCppError(const char *message,
                                              NSInteger code)
{
    NSString *description = message && message[0]
        ? [NSString stringWithUTF8String:message]
        : @"Metal-cpp operation failed";
    return [NSError errorWithDomain:@"MGLPipelineCache"
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey: description}];
}

static MGLMetalDepthStencilStateRef mglPipelineCacheCreateDepthStencilState(
    MGLMetalDeviceRef device,
    MTLDepthStencilDescriptor *descriptor)
{
    if (mglPipelineCacheUsesMetalCpp()) {
        void *state = NULL;
        if (mglRenderCppCreateDepthStencilState(
                (__bridge void *)descriptor, &state) == 0 && state) {
            return (__bridge_transfer MGLMetalDepthStencilStateRef)state;
        }
    }
    return [device newDepthStencilStateWithDescriptor:descriptor];
}

static MGLRenderCppStencilDescriptorState
mglPipelineCacheStencilState(MTLStencilDescriptor *descriptor)
{
    if (!descriptor) return (MGLRenderCppStencilDescriptorState){0};
    return (MGLRenderCppStencilDescriptorState){
        .present = 1,
        .compare_function = (uint32_t)descriptor.stencilCompareFunction,
        .read_mask = descriptor.readMask,
        .write_mask = descriptor.writeMask,
        .stencil_failure_operation =
            (uint32_t)descriptor.stencilFailureOperation,
        .depth_failure_operation =
            (uint32_t)descriptor.depthFailureOperation,
        .depth_stencil_pass_operation =
            (uint32_t)descriptor.depthStencilPassOperation,
    };
}

static MGLRenderCppDepthStencilDescriptorState
mglPipelineCacheDepthStencilState(MTLDepthStencilDescriptor *descriptor)
{
    return (MGLRenderCppDepthStencilDescriptorState){
        .depth_compare_function =
            (uint32_t)descriptor.depthCompareFunction,
        .depth_write_enabled = descriptor.depthWriteEnabled ? 1u : 0u,
        .front = mglPipelineCacheStencilState(descriptor.frontFaceStencil),
        .back = mglPipelineCacheStencilState(descriptor.backFaceStencil),
    };
}

@implementation MGLPipelineCacheKey {
    uint64_t _words[MGL_PIPELINE_CACHE_KEY_WORDS];
}

- (instancetype)initWithWords:(const uint64_t[MGL_PIPELINE_CACHE_KEY_WORDS])words
{
    self = [super init];
    if (self) {
        memcpy(_words, words, sizeof(_words));
    }
    return self;
}

/* Rewrites the cached words of a reusable query key.  The caller must never
 * store this object into the pipeline cache dictionaries/LRU — only the
 * hit-path lookup may use it, because later overwrites would corrupt any
 * dictionary that retained the object.  Miss paths must allocate a fresh
 * MGLPipelineCacheKey instead. */
- (void)overwriteWords:(const uint64_t[MGL_PIPELINE_CACHE_KEY_WORDS])words
{
    memcpy(_words, words, sizeof(_words));
}

- (void)copyWords:(uint64_t[MGL_PIPELINE_CACHE_KEY_WORDS])words
{
    memcpy(words, _words, sizeof(_words));
}

- (BOOL)isEqual:(id)object
{
    if (self == object) return YES;
    if (![object isKindOfClass:[MGLPipelineCacheKey class]]) return NO;
    MGLPipelineCacheKey *other = object;
    return memcmp(_words, other->_words, sizeof(_words)) == 0;
}

- (NSUInteger)hash
{
    uint64_t hash = 0x9e3779b97f4a7c15ull;
    for (unsigned i = 0; i < MGL_PIPELINE_CACHE_KEY_WORDS; i++) {
        hash ^= _words[i];
        hash *= 0x100000001b3ull;
    }
    return (NSUInteger)hash;
}

- (id)copyWithZone:(NSZone *)zone
{
    return self;  /* immutable */
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"%016llx-%016llx-%016llx-%016llx-%016llx-%016llx-%016llx",
            (unsigned long long)_words[0], (unsigned long long)_words[1],
            (unsigned long long)_words[2], (unsigned long long)_words[3],
            (unsigned long long)_words[4], (unsigned long long)_words[5],
            (unsigned long long)_words[6]];
}

@end

@interface MGLDepthStencilCacheKey : NSObject <NSCopying>
@property(nonatomic) MTLCompareFunction depthCompareFunction;
@property(nonatomic) BOOL depthWriteEnabled;
@property(nonatomic) BOOL frontStencilPresent;
@property(nonatomic) MTLCompareFunction frontStencilCompareFunction;
@property(nonatomic) uint32_t frontReadMask;
@property(nonatomic) uint32_t frontWriteMask;
@property(nonatomic) MTLStencilOperation frontStencilFailureOperation;
@property(nonatomic) MTLStencilOperation frontDepthFailureOperation;
@property(nonatomic) MTLStencilOperation frontDepthStencilPassOperation;
@property(nonatomic) BOOL backStencilPresent;
@property(nonatomic) MTLCompareFunction backStencilCompareFunction;
@property(nonatomic) uint32_t backReadMask;
@property(nonatomic) uint32_t backWriteMask;
@property(nonatomic) MTLStencilOperation backStencilFailureOperation;
@property(nonatomic) MTLStencilOperation backDepthFailureOperation;
@property(nonatomic) MTLStencilOperation backDepthStencilPassOperation;
@end

/* Populates the key's fields from a MTLDepthStencilDescriptor. */
static inline void MGLPopulateDepthStencilKey(MGLDepthStencilCacheKey *key,
                                              MTLDepthStencilDescriptor *descriptor)
{
    key.depthCompareFunction = descriptor.depthCompareFunction;
    key.depthWriteEnabled = descriptor.depthWriteEnabled;

    MTLStencilDescriptor *front = descriptor.frontFaceStencil;
    if (front) {
        key.frontStencilPresent = YES;
        key.frontStencilCompareFunction = front.stencilCompareFunction;
        key.frontReadMask = front.readMask;
        key.frontWriteMask = front.writeMask;
        key.frontStencilFailureOperation = front.stencilFailureOperation;
        key.frontDepthFailureOperation = front.depthFailureOperation;
        key.frontDepthStencilPassOperation = front.depthStencilPassOperation;
    }

    MTLStencilDescriptor *back = descriptor.backFaceStencil;
    if (back) {
        key.backStencilPresent = YES;
        key.backStencilCompareFunction = back.stencilCompareFunction;
        key.backReadMask = back.readMask;
        key.backWriteMask = back.writeMask;
        key.backStencilFailureOperation = back.stencilFailureOperation;
        key.backDepthFailureOperation = back.depthFailureOperation;
        key.backDepthStencilPassOperation = back.depthStencilPassOperation;
    }
}

@implementation MGLDepthStencilCacheKey

- (BOOL)isEqual:(id)object
{
    if (self == object) return YES;
    if (![object isKindOfClass:[MGLDepthStencilCacheKey class]]) return NO;

    MGLDepthStencilCacheKey *other = object;
    return _depthCompareFunction == other.depthCompareFunction &&
           _depthWriteEnabled == other.depthWriteEnabled &&
           _frontStencilPresent == other.frontStencilPresent &&
           _frontStencilCompareFunction == other.frontStencilCompareFunction &&
           _frontReadMask == other.frontReadMask &&
           _frontWriteMask == other.frontWriteMask &&
           _frontStencilFailureOperation == other.frontStencilFailureOperation &&
           _frontDepthFailureOperation == other.frontDepthFailureOperation &&
           _frontDepthStencilPassOperation == other.frontDepthStencilPassOperation &&
           _backStencilPresent == other.backStencilPresent &&
           _backStencilCompareFunction == other.backStencilCompareFunction &&
           _backReadMask == other.backReadMask &&
           _backWriteMask == other.backWriteMask &&
           _backStencilFailureOperation == other.backStencilFailureOperation &&
           _backDepthFailureOperation == other.backDepthFailureOperation &&
           _backDepthStencilPassOperation == other.backDepthStencilPassOperation;
}

- (NSUInteger)hash
{
    NSUInteger hash = 0;
    hash = hash * 31 + (NSUInteger)_depthCompareFunction;
    hash = hash * 31 + (NSUInteger)_depthWriteEnabled;
    hash = hash * 31 + (NSUInteger)_frontStencilPresent;
    hash = hash * 31 + (NSUInteger)_frontStencilCompareFunction;
    hash = hash * 31 + (NSUInteger)_frontReadMask;
    hash = hash * 31 + (NSUInteger)_frontWriteMask;
    hash = hash * 31 + (NSUInteger)_frontStencilFailureOperation;
    hash = hash * 31 + (NSUInteger)_frontDepthFailureOperation;
    hash = hash * 31 + (NSUInteger)_frontDepthStencilPassOperation;
    hash = hash * 31 + (NSUInteger)_backStencilPresent;
    hash = hash * 31 + (NSUInteger)_backStencilCompareFunction;
    hash = hash * 31 + (NSUInteger)_backReadMask;
    hash = hash * 31 + (NSUInteger)_backWriteMask;
    hash = hash * 31 + (NSUInteger)_backStencilFailureOperation;
    hash = hash * 31 + (NSUInteger)_backDepthFailureOperation;
    hash = hash * 31 + (NSUInteger)_backDepthStencilPassOperation;
    return hash;
}

- (id)copyWithZone:(NSZone *)zone
{
    MGLDepthStencilCacheKey *copy = [[MGLDepthStencilCacheKey allocWithZone:zone] init];
    copy.depthCompareFunction = _depthCompareFunction;
    copy.depthWriteEnabled = _depthWriteEnabled;
    copy.frontStencilPresent = _frontStencilPresent;
    copy.frontStencilCompareFunction = _frontStencilCompareFunction;
    copy.frontReadMask = _frontReadMask;
    copy.frontWriteMask = _frontWriteMask;
    copy.frontStencilFailureOperation = _frontStencilFailureOperation;
    copy.frontDepthFailureOperation = _frontDepthFailureOperation;
    copy.frontDepthStencilPassOperation = _frontDepthStencilPassOperation;
    copy.backStencilPresent = _backStencilPresent;
    copy.backStencilCompareFunction = _backStencilCompareFunction;
    copy.backReadMask = _backReadMask;
    copy.backWriteMask = _backWriteMask;
    copy.backStencilFailureOperation = _backStencilFailureOperation;
    copy.backDepthFailureOperation = _backDepthFailureOperation;
    copy.backDepthStencilPassOperation = _backDepthStencilPassOperation;
    return copy;
}

@end

/* v5 excludes either kind of incomplete render pipeline and isolates both
 * sanitizer builds and archive producers. The producer boundary prevents the
 * temporary A/B implementations from sharing mutable state; the archive-aware
 * PSO creation path below separately prevents repeated adds on cache hits. */
#if __has_feature(address_sanitizer)
static NSString * const kMGLPipelineArchiveBuildSchema = @"v5-asan";
#elif __has_feature(thread_sanitizer)
static NSString * const kMGLPipelineArchiveBuildSchema = @"v5-tsan";
#else
static NSString * const kMGLPipelineArchiveBuildSchema = @"v5";
#endif

static void MGLTouchLRU(NSMutableOrderedSet *lru, id key)
{
    if (!lru || !key) return;
    [lru removeObject:key];
    [lru addObject:key];
}

static NSUInteger MGLEvictLRU(NSMutableDictionary *cache,
                              NSMutableOrderedSet *lru,
                              NSUInteger count)
{
    NSUInteger removed = 0;
    while (removed < count && lru.count > 0) {
        id key = lru.firstObject;
        [lru removeObjectAtIndex:0];
        if ([cache objectForKey:key]) {
            [cache removeObjectForKey:key];
            removed++;
        }
    }
#if DEBUG
    NSCAssert(cache.count == lru.count, @"cache/LRU index count mismatch");
#endif
    return removed;
}

static NSString *MGLSafeArchivePathComponent(NSString *value)
{
    if (value.length == 0) return @"unknown";
    NSCharacterSet *unsafe = [[NSCharacterSet alphanumericCharacterSet] invertedSet];
    return [[value componentsSeparatedByCharactersInSet:unsafe] componentsJoinedByString:@"_"];
}

@implementation MGLPipelineCache

- (instancetype)initWithPSODedupEnabled:(BOOL)psoDedupEnabled
                depthStencilCacheEnabled:(BOOL)depthStencilCacheEnabled
                     binaryArchiveEnabled:(BOOL)binaryArchiveEnabled
{
    self = [super init];
    if (!self) return nil;

    _state.pipelineColor0Format = MTLPixelFormatInvalid;
    _state.pipelineDepthFormat = MTLPixelFormatInvalid;
    _state.pipelineStencilFormat = MTLPixelFormatInvalid;
    _state.pipelineStateCache = [[NSMutableDictionary alloc] initWithCapacity:64];
    _state.pipelineStateCacheLRU = [[NSMutableOrderedSet alloc] initWithCapacity:64];
    _state.pipelineDescriptorCache = [[NSMutableDictionary alloc] initWithCapacity:64];
    _state.pipelineDescriptorCacheLRU = [[NSMutableOrderedSet alloc] initWithCapacity:64];
    _state.psoDedupEnabled = psoDedupEnabled;
    _state.dsCacheEnabled = depthStencilCacheEnabled;
    _binaryArchiveRequested = binaryArchiveEnabled;
    if (depthStencilCacheEnabled) {
        _state.depthStencilStateCache = [NSMutableDictionary new];
        _state.depthStencilStateCacheLRU = [NSMutableOrderedSet new];
    }
    return self;
}

- (const MGLPipelineCacheState *)state
{
    return &_state;
}

- (BOOL)ensureCppOwnerCreated
{
    if (_cppOwner) return YES;
    if (!_device || !mglRenderCppGetDevice()) return NO;
    if (mglRenderCppCreatePipelineCacheOwner(
            _state.psoDedupEnabled ? 1 : 0,
            _state.dsCacheEnabled ? 1 : 0,
            _binaryArchiveRequested ? 1 : 0,
            &_cppOwner) != 0 || !_cppOwner) {
        _cppOwner = NULL;
        return NO;
    }

    for (NSUInteger index = 0; index < MAX_COLOR_ATTACHMENTS; ++index) {
        MGLRenderCppPipelineBlendState blend = {
            .source_rgb_factor =
                (uint32_t)_state.src_blend_rgb_factor[index],
            .destination_rgb_factor =
                (uint32_t)_state.dst_blend_rgb_factor[index],
            .source_alpha_factor =
                (uint32_t)_state.src_blend_alpha_factor[index],
            .destination_alpha_factor =
                (uint32_t)_state.dst_blend_alpha_factor[index],
            .rgb_operation =
                (uint32_t)_state.rgb_blend_operation[index],
            .alpha_operation =
                (uint32_t)_state.alpha_blend_operation[index],
            .color_write_mask = (uint32_t)_state.color_mask[index],
        };
        mglRenderCppSetPipelineBlendState(
            _cppOwner, (uint32_t)index, &blend);
    }
    MGLRenderCppPipelineActiveState active = {
        .pipeline_state = (__bridge void *)_state.pipelineState,
        .vertex_function = (__bridge void *)_state.pipelineVertexFunction,
        .fragment_function = (__bridge void *)_state.pipelineFragmentFunction,
        .color0_format = (uint32_t)_state.pipelineColor0Format,
        .depth_format = (uint32_t)_state.pipelineDepthFormat,
        .stencil_format = (uint32_t)_state.pipelineStencilFormat,
        .program_name = _state.pipelineProgramName,
    };
    mglRenderCppActivatePipelineState(_cppOwner, &active);
    return YES;
}

- (BOOL)ensureCppOwner
{
    return mglPipelineCacheUsesMetalCpp() && [self ensureCppOwnerCreated];
}

- (BOOL)isBinaryArchiveEnabled
{
    int enabled = _binaryArchiveRequested ? 1 : 0;
    if (_cppOwner) {
        mglRenderCppGetPipelineBinaryArchiveState(
            _cppOwner, &enabled, NULL);
    }
    return enabled != 0;
}

- (MGLMetalDeviceRef)device
{
    return _device;
}

- (void)setDevice:(MGLMetalDeviceRef)device
{
    if (_device != device) {
        mglRenderCppDestroyPipelineCacheOwner(&_cppOwner);
    }
    _device = device;
    if (_device) [self ensureCppOwnerCreated];
}

- (MGLMetalDepthStencilStateRef)depthStencilStateForDescriptor:
    (MTLDepthStencilDescriptor *)descriptor
{
    if (!descriptor || !_device) return nil;
    if ([self ensureCppOwner]) {
        MGLRenderCppDepthStencilDescriptorState descriptorState =
            mglPipelineCacheDepthStencilState(descriptor);
        void *statePtr = NULL;
        if (_state.dsCacheEnabled) {
            int created = 0;
            if (mglRenderCppGetOrCreateDepthStencilState(
                    _cppOwner, &descriptorState, &statePtr, &created) == 0 &&
                statePtr) {
                if (created) {
                    MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
                }
                return (__bridge MGLMetalDepthStencilStateRef)statePtr;
            }
            return nil;
        }
        if (mglRenderCppCreateDepthStencilStateFromState(
                &descriptorState, &statePtr) == 0 && statePtr) {
            MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
            return (__bridge_transfer MGLMetalDepthStencilStateRef)statePtr;
        }
        return nil;
    }
    if (!_state.dsCacheEnabled) {
        MGLMetalDepthStencilStateRef state =
            mglPipelineCacheCreateDepthStencilState(_device, descriptor);
        MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
        return state;
    }

    if (!_state.depthStencilCacheQueryKey) {
        _state.depthStencilCacheQueryKey = [MGLDepthStencilCacheKey new];
    }
    MGLDepthStencilCacheKey *key = _state.depthStencilCacheQueryKey;
    MGLPopulateDepthStencilKey(key, descriptor);

    /* Hit path uses the reusable query key (no per-lookup allocation).  The
     * DS cache is small (cap 64) and distinct depth/stencil states are few,
     * so we intentionally skip LRU-touch on hits — touching would require
     * copying the hash key, reintroducing the per-lookup alloc this avoids. */
    MGLMetalDepthStencilStateRef cached = _state.depthStencilStateCache[key];
    if (cached) {
        return cached;
    }

    MGLMetalDepthStencilStateRef state =
        mglPipelineCacheCreateDepthStencilState(_device, descriptor);
    MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
    if (state) {
        MGLDepthStencilCacheKey *cacheKey = [key copy];
        _state.depthStencilStateCache[cacheKey] = state;
        MGLTouchLRU(_state.depthStencilStateCacheLRU, cacheKey);
        if (_state.depthStencilStateCache.count > 64) {
            MGLEvictLRU(_state.depthStencilStateCache,
                        _state.depthStencilStateCacheLRU,
                        _state.depthStencilStateCache.count - 64);
        }
    }
    return state;
}

- (id)pipelineEntryForKey:(MGLPipelineCacheKey *)key
{
    if ([self ensureCppOwner] && key) {
        uint64_t words[MGL_PIPELINE_CACHE_KEY_WORDS];
        [key copyWords:words];
        MGLRenderCppPipelineActiveState cached = {0};
        if (mglRenderCppLookupPipeline(_cppOwner, words, &cached) != 1 ||
            !cached.pipeline_state) {
            return nil;
        }
        id vertexFunction = cached.vertex_function
            ? (__bridge id)cached.vertex_function : [NSNull null];
        id fragmentFunction = cached.fragment_function
            ? (__bridge id)cached.fragment_function : [NSNull null];
        return @{
            @"pipeline": (__bridge id)cached.pipeline_state,
            @"sig": @(words[5]),
            @"vsig": @(words[6]),
            @"vertexFunction": vertexFunction,
            @"fragmentFunction": fragmentFunction,
        };
    }
    return _state.pipelineStateCache[key];
}

- (MGLPipelineCacheKey *)pipelineQueryKeyForWords:
    (const uint64_t[MGL_PIPELINE_CACHE_KEY_WORDS])words
{
    if (!_state.pipelineCacheQueryKey) {
        _state.pipelineCacheQueryKey =
            [[MGLPipelineCacheKey alloc] initWithWords:words];
    } else {
        [_state.pipelineCacheQueryKey overwriteWords:words];
    }
    return _state.pipelineCacheQueryKey;
}

- (void)markPipelineEntryUsedForKey:(MGLPipelineCacheKey *)key
{
    if (_cppOwner) return;
    if (_state.pipelineStateCache[key]) {
        MGLTouchLRU(_state.pipelineStateCacheLRU, key);
    }
}

- (MTLRenderPipelineDescriptor *)pipelineDescriptorForKey:(MGLPipelineCacheKey *)key
{
    /* P4.2: descriptor cache 的 C++ 侧已改为 value-state 版
     * （pipelineDescriptorStateForWords:state:）；本方法只服务 gate-off 的
     * ObjC descriptor 字典。 */
    MTLRenderPipelineDescriptor *descriptor = _state.pipelineDescriptorCache[key];
    if (descriptor) MGLTouchLRU(_state.pipelineDescriptorCacheLRU, key);
    return descriptor;
}

- (NSUInteger)storePipelineEntry:(id)entry forKey:(MGLPipelineCacheKey *)key
{
    if (!entry || !key) return 0;
    if ([self ensureCppOwner]) {
        id pipeline = entry;
        id vertexFunction = nil;
        id fragmentFunction = nil;
        if ([entry isKindOfClass:[NSDictionary class]]) {
            NSDictionary *dictionary = (NSDictionary *)entry;
            pipeline = dictionary[@"pipeline"];
            vertexFunction = dictionary[@"vertexFunction"];
            fragmentFunction = dictionary[@"fragmentFunction"];
            if (vertexFunction == [NSNull null]) vertexFunction = nil;
            if (fragmentFunction == [NSNull null]) fragmentFunction = nil;
        }
        if (!pipeline) return 0;
        uint64_t words[MGL_PIPELINE_CACHE_KEY_WORDS];
        [key copyWords:words];
        MGLRenderCppPipelineActiveState state = {
            .pipeline_state = (__bridge void *)pipeline,
            .vertex_function = (__bridge void *)vertexFunction,
            .fragment_function = (__bridge void *)fragmentFunction,
        };
        uint32_t removed = 0;
        if (mglRenderCppStorePipeline(
                _cppOwner, words, &state, &removed) != 0) {
            return 0;
        }
        MGL_PERF_ADD(g_mglPipelineCacheEvictionsSinceSwap, removed);
        return (NSUInteger)removed;
    }
    NSUInteger removed = 0;
    BOOL replacing = _state.pipelineStateCache[key] != nil;
    if (!replacing && _state.pipelineStateCache.count >= 256) {
        NSUInteger evictCount = MAX((NSUInteger)1, _state.pipelineStateCache.count / 4);
        removed = MGLEvictLRU(_state.pipelineStateCache,
                              _state.pipelineStateCacheLRU,
                              evictCount);
        MGL_PERF_ADD(g_mglPipelineCacheEvictionsSinceSwap, removed);
    }
    _state.pipelineStateCache[key] = entry;
    MGLTouchLRU(_state.pipelineStateCacheLRU, key);
    return removed;
}

- (void)storePipelineDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                         forKey:(MGLPipelineCacheKey *)key
{
    /* P4.2: gate-off ObjC descriptor 字典（gate-on 走
     * storePipelineDescriptorState:forWords:）。 */
    if (!descriptor || !key) return;
    _state.pipelineDescriptorCache[key] = [descriptor copy];
    MGLTouchLRU(_state.pipelineDescriptorCacheLRU, key);
    if (_state.pipelineDescriptorCache.count > 128) {
        MGLEvictLRU(_state.pipelineDescriptorCache,
                    _state.pipelineDescriptorCacheLRU,
                    _state.pipelineDescriptorCache.count - 128);
    }
}

- (BOOL)lookupPipelineForWords:(const uint64_t *)words
                      pipeline:(MGLMetalRenderPipelineStateRef *)pipelineOut
                vertexFunction:(MGLMetalFunctionRef *)vertexFunctionOut
              fragmentFunction:(MGLMetalFunctionRef *)fragmentFunctionOut
{
    if (pipelineOut) *pipelineOut = nil;
    if (vertexFunctionOut) *vertexFunctionOut = nil;
    if (fragmentFunctionOut) *fragmentFunctionOut = nil;
    if (!words || !pipelineOut || !vertexFunctionOut ||
        !fragmentFunctionOut) {
        return NO;
    }
    if ([self ensureCppOwner]) {
        MGLRenderCppPipelineActiveState cached = {0};
        if (mglRenderCppLookupPipeline(_cppOwner, words, &cached) != 1 ||
            !cached.pipeline_state) {
            return NO;
        }
        *pipelineOut = (__bridge MGLMetalRenderPipelineStateRef)cached.pipeline_state;
        *vertexFunctionOut = (__bridge MGLMetalFunctionRef)cached.vertex_function;
        *fragmentFunctionOut =
            (__bridge MGLMetalFunctionRef)cached.fragment_function;
        return YES;
    }

    MGLPipelineCacheKey *key = [self pipelineQueryKeyForWords:words];
    id entry = _state.pipelineStateCache[key];
    if (!entry) return NO;
    if ([entry isKindOfClass:[NSDictionary class]]) {
        NSDictionary *dictionary = (NSDictionary *)entry;
        id pipeline = dictionary[@"pipeline"];
        id vertexFunction = dictionary[@"vertexFunction"];
        id fragmentFunction = dictionary[@"fragmentFunction"];
        if (!pipeline) return NO;
        *pipelineOut = pipeline;
        if (vertexFunction != [NSNull null]) {
            *vertexFunctionOut = vertexFunction;
        }
        if (fragmentFunction != [NSNull null]) {
            *fragmentFunctionOut = fragmentFunction;
        }
        return YES;
    }
    *pipelineOut = (MGLMetalRenderPipelineStateRef)entry;
    return YES;
}

- (MTLRenderPipelineDescriptor *)pipelineDescriptorForWords:
    (const uint64_t *)words
{
    /* P4.2: gate-off 专用（gate-on 用 pipelineDescriptorStateForWords:state:）。 */
    if (!words) return nil;
    return [self pipelineDescriptorForKey:
        [self pipelineQueryKeyForWords:words]];
}

- (NSUInteger)storePipeline:(MGLMetalRenderPipelineStateRef)pipeline
              vertexFunction:(MGLMetalFunctionRef)vertexFunction
            fragmentFunction:(MGLMetalFunctionRef)fragmentFunction
                    forWords:(const uint64_t *)words
{
    if (!pipeline || !words) return 0;
    if ([self ensureCppOwner]) {
        MGLRenderCppPipelineActiveState state = {
            .pipeline_state = (__bridge void *)pipeline,
            .vertex_function = (__bridge void *)vertexFunction,
            .fragment_function = (__bridge void *)fragmentFunction,
        };
        uint32_t removed = 0;
        if (mglRenderCppStorePipeline(
                _cppOwner, words, &state, &removed) != 0) {
            return 0;
        }
        MGL_PERF_ADD(g_mglPipelineCacheEvictionsSinceSwap, removed);
        return (NSUInteger)removed;
    }

    MGLPipelineCacheKey *key =
        [[MGLPipelineCacheKey alloc] initWithWords:words];
    id vertexValue = vertexFunction ?: [NSNull null];
    id fragmentValue = fragmentFunction ?: [NSNull null];
    NSDictionary *entry = @{
        @"pipeline": pipeline,
        @"sig": @(words[5]),
        @"vsig": @(words[6]),
        @"vertexFunction": vertexValue,
        @"fragmentFunction": fragmentValue,
    };
    return [self storePipelineEntry:entry forKey:key];
}

- (void)storePipelineDescriptor:(MTLRenderPipelineDescriptor *)descriptor
                       forWords:(const uint64_t *)words
{
    /* P4.2: gate-off 专用（gate-on 走 storePipelineDescriptorState:forWords:）。 */
    if (!descriptor || !words) return;
    MGLPipelineCacheKey *key =
        [[MGLPipelineCacheKey alloc] initWithWords:words];
    [self storePipelineDescriptor:descriptor forKey:key];
}

- (BOOL)pipelineDescriptorStateForWords:(const uint64_t *)words
                                  state:(MGLRenderCppPipelineDescriptorState *)stateOut
{
    if (!words || !stateOut) return NO;
    if ([self ensureCppOwner]) {
        if (mglRenderCppLookupPipelineDescriptorState(
                _cppOwner, words, stateOut) == 1) {
            return YES;
        }
    }
    return NO;
}

- (void)storePipelineDescriptorState:(const MGLRenderCppPipelineDescriptorState *)state
                            forWords:(const uint64_t *)words
{
    if (!state || !words) return;
    if ([self ensureCppOwner]) {
        mglRenderCppStorePipelineDescriptorState(
            _cppOwner, words, state);
    }
    /* gate-off 无 value-state cache：ObjC descriptor 字典由
     * storePipelineDescriptor:forWords: 维护。 */
}

- (BOOL)blendStateForAttachment:(NSUInteger)index
                            out:(MGLRenderCppPipelineBlendState *)outState
{
    if (index >= MAX_COLOR_ATTACHMENTS || !outState) return NO;
    if ([self ensureCppOwner]) {
        if (mglRenderCppGetPipelineBlendState(
                _cppOwner, (uint32_t)index, outState) == 0) {
            return YES;
        }
    }
    *outState = (MGLRenderCppPipelineBlendState){
        .source_rgb_factor =
            (uint32_t)_state.src_blend_rgb_factor[index],
        .destination_rgb_factor =
            (uint32_t)_state.dst_blend_rgb_factor[index],
        .source_alpha_factor =
            (uint32_t)_state.src_blend_alpha_factor[index],
        .destination_alpha_factor =
            (uint32_t)_state.dst_blend_alpha_factor[index],
        .rgb_operation =
            (uint32_t)_state.rgb_blend_operation[index],
        .alpha_operation =
            (uint32_t)_state.alpha_blend_operation[index],
        .color_write_mask = (uint32_t)_state.color_mask[index],
    };
    return YES;
}

- (NSURL *)binaryArchiveURL
{
    NSArray *caches = NSSearchPathForDirectoriesInDomains(NSCachesDirectory,
                                                          NSUserDomainMask, YES);
    NSString *baseDir = caches.firstObject ?: NSTemporaryDirectory();
    NSString *bundleID = NSBundle.mainBundle.bundleIdentifier;
    if (bundleID.length == 0) bundleID = NSProcessInfo.processInfo.processName;
    NSString *mglDir = [[baseDir stringByAppendingPathComponent:@"MGL"]
                        stringByAppendingPathComponent:MGLSafeArchivePathComponent(bundleID)];
    NSFileManager *fileManager = NSFileManager.defaultManager;
    if (![fileManager fileExistsAtPath:mglDir]) {
        [fileManager createDirectoryAtPath:mglDir
               withIntermediateDirectories:YES
                                attributes:nil
                                     error:NULL];
    }

    uint64_t registryID = 0;
    if (@available(macOS 11.0, *)) registryID = _device.registryID;
    NSString *deviceID = registryID != 0
        ? [NSString stringWithFormat:@"%016llx", (unsigned long long)registryID]
        : MGLSafeArchivePathComponent(_device.name);
    NSString *producer = mglPipelineCacheUsesMetalCpp() ? @"cpp" : @"objc";
    NSString *schema = [NSString stringWithFormat:@"%@-%@",
                        kMGLPipelineArchiveBuildSchema, producer];
    NSString *filename = [NSString stringWithFormat:@"pipeline-%@-%@.binaryarchive",
                          schema, deviceID];
    return [NSURL fileURLWithPath:[mglDir stringByAppendingPathComponent:filename]];
}

- (void)loadBinaryArchive
{
    if (!self.binaryArchiveEnabled || !_device ||
        ![self ensureCppOwnerCreated]) return;

    NSURL *archiveURL = [self binaryArchiveURL];
    NSString *archiveKey = archiveURL.path;
    NSFileManager *fileManager = NSFileManager.defaultManager;
    BOOL archiveExists = [fileManager fileExistsAtPath:archiveKey];
    int reused = 0;
    char message[512] = {0};
    int result = mglRenderCppLoadPipelineBinaryArchive(
        _cppOwner, archiveKey.UTF8String, (__bridge void *)archiveURL,
        archiveExists ? 1 : 0, &reused, message, sizeof(message));
    if (result != 0 && archiveExists) {
        NSError *removeError = nil;
        if (![fileManager removeItemAtURL:archiveURL error:&removeError]) {
            NSLog(@"MGL BINARY ARCHIVE: failed to remove incompatible archive: %@",
                  removeError.localizedDescription);
        }
        NSLog(@"MGL BINARY ARCHIVE: rebuilding incompatible archive: %s",
              message[0] ? message : "unknown error");
        archiveExists = NO;
        message[0] = '\0';
        result = mglRenderCppLoadPipelineBinaryArchive(
            _cppOwner, archiveKey.UTF8String, (__bridge void *)archiveURL,
            0, &reused, message, sizeof(message));
    }
    if (result == 0) {
        NSLog(@"MGL BINARY ARCHIVE: %@ %@",
              reused ? @"reused" : (archiveExists ? @"loaded" : @"created"),
              archiveURL.lastPathComponent);
    } else {
        NSLog(@"MGL BINARY ARCHIVE: unavailable, PSO compile will continue without it: %s",
              message[0] ? message : "unknown error");
    }
}

- (void)saveBinaryArchive
{
    int present = 0;
    if (!_cppOwner ||
        mglRenderCppGetPipelineBinaryArchiveState(
            _cppOwner, NULL, &present) != 0 || !present) return;

    NSURL *archiveURL = [self binaryArchiveURL];
    NSString *archiveKey = archiveURL.path;
    NSError *removeError = nil;
    char message[512] = {0};
    BOOL ok = mglRenderCppSerializePipelineBinaryArchive(
        _cppOwner, (__bridge void *)archiveURL,
        message, sizeof(message)) == 0;
    BOOL discarded = NO;
    if (!ok) {
        NSFileManager *fileManager = NSFileManager.defaultManager;
        discarded = ![fileManager fileExistsAtPath:archiveKey] ||
            [fileManager removeItemAtURL:archiveURL error:&removeError];
        mglRenderCppDiscardPipelineBinaryArchive(
            _cppOwner, archiveKey.UTF8String);
    }
    if (ok) {
        NSLog(@"MGL BINARY ARCHIVE: saved to %@", archiveURL.lastPathComponent);
    } else {
        NSString *description = message[0]
            ? [NSString stringWithUTF8String:message] : @"unknown error";
        if (discarded) {
            NSLog(@"MGL BINARY ARCHIVE: discarded unserializable archive: %@",
                  description);
        } else {
            NSLog(@"MGL BINARY ARCHIVE: serialize failed: %@; removal failed: %@",
                  description,
                  removeError.localizedDescription);
        }
    }
}

- (MGLMetalRenderPipelineStateRef)createRenderPipelineStateWithDescriptor:
    (MTLRenderPipelineDescriptor *)descriptor
    error:(NSError **)error
{
    if (error) *error = nil;
    if (!descriptor || !_device) {
        if (error) {
            *error = [NSError errorWithDomain:@"MGLPipelineCache"
                                         code:14
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                    @"Missing pipeline descriptor or Metal device"}];
        }
        return nil;
    }

    if ([self ensureCppOwnerCreated]) {
        void *pipeline = NULL;
        char message[512] = {0};
        int archiveHit = 0;
        int result = mglRenderCppCreateRenderPipelineStateWithArchiveOwner(
            _cppOwner, (__bridge void *)descriptor, &pipeline,
            &archiveHit, message, sizeof(message));
        (void)archiveHit;
        if (result == 0 && pipeline) {
            return (__bridge_transfer MGLMetalRenderPipelineStateRef)pipeline;
        }
        if (error) *error = mglPipelineCacheMetalCppError(message, 12);
        return nil;
    }

    NSError *compileError = nil;
    MGLMetalRenderPipelineStateRef pipeline =
        [_device newRenderPipelineStateWithDescriptor:descriptor
                                                error:&compileError];
    if (!pipeline) {
        if (error) *error = compileError;
        return nil;
    }

    return pipeline;
}

- (int)createRenderPipelineFromState:
    (const MGLRenderCppPipelineDescriptorState *)state
    vertexFunction:(void *)vertexFunction
    fragmentFunction:(void *)fragmentFunction
    pipelineOut:(void **)pipelineOut
    errorMessage:(char *)errorMessage
    errorCapacity:(size_t)errorCapacity
{
    if (![self ensureCppOwnerCreated]) return -1;
    return mglRenderCppCreateRenderPipelineFromStateWithArchiveOwner(
        _cppOwner, vertexFunction, fragmentFunction, state,
        pipelineOut, errorMessage, errorCapacity);
}

- (void)invalidatePipelineState
{
    if ([self ensureCppOwner]) {
        mglRenderCppInvalidatePipelineActiveState(_cppOwner);
    }
    _state.pipelineState = nil;
    _state.pipelineColor0Format = MTLPixelFormatInvalid;
    _state.pipelineDepthFormat = MTLPixelFormatInvalid;
    _state.pipelineStencilFormat = MTLPixelFormatInvalid;
    _state.pipelineProgramName = 0u;
    _state.pipelineVertexFunction = nil;
    _state.pipelineFragmentFunction = nil;
}

- (void)setPipelineState:(MGLMetalRenderPipelineStateRef)pipelineState
{
    if ([self ensureCppOwner]) {
        mglRenderCppSetPipelineActiveObject(
            _cppOwner, (__bridge void *)pipelineState);
    }
    _state.pipelineState = pipelineState;
}

- (void)activatePipelineState:(MGLMetalRenderPipelineStateRef)pipelineState
                 color0Format:(MTLPixelFormat)color0Format
                  depthFormat:(MTLPixelFormat)depthFormat
                stencilFormat:(MTLPixelFormat)stencilFormat
                  programName:(GLuint)programName
               vertexFunction:(MGLMetalFunctionRef)vertexFunction
             fragmentFunction:(MGLMetalFunctionRef)fragmentFunction
{
    if ([self ensureCppOwner]) {
        MGLRenderCppPipelineActiveState active = {
            .pipeline_state = (__bridge void *)pipelineState,
            .vertex_function = (__bridge void *)vertexFunction,
            .fragment_function = (__bridge void *)fragmentFunction,
            .color0_format = (uint32_t)color0Format,
            .depth_format = (uint32_t)depthFormat,
            .stencil_format = (uint32_t)stencilFormat,
            .program_name = programName,
        };
        mglRenderCppActivatePipelineState(_cppOwner, &active);
    }
    _state.pipelineState = pipelineState;
    _state.pipelineColor0Format = color0Format;
    _state.pipelineDepthFormat = depthFormat;
    _state.pipelineStencilFormat = stencilFormat;
    _state.pipelineProgramName = programName;
    _state.pipelineVertexFunction = vertexFunction;
    _state.pipelineFragmentFunction = fragmentFunction;
}

- (void)setBlendFactorsForAttachment:(NSUInteger)index
                        srcRgbFactor:(MTLBlendFactor)srcRgbFactor
                      srcAlphaFactor:(MTLBlendFactor)srcAlphaFactor
                        dstRgbFactor:(MTLBlendFactor)dstRgbFactor
                      dstAlphaFactor:(MTLBlendFactor)dstAlphaFactor
                        rgbOperation:(MTLBlendOperation)rgbOperation
                      alphaOperation:(MTLBlendOperation)alphaOperation
                           colorMask:(MTLColorWriteMask)colorMask
{
    if (index >= MAX_COLOR_ATTACHMENTS) return;
    if ([self ensureCppOwner]) {
        MGLRenderCppPipelineBlendState blend = {
            .source_rgb_factor = (uint32_t)srcRgbFactor,
            .destination_rgb_factor = (uint32_t)dstRgbFactor,
            .source_alpha_factor = (uint32_t)srcAlphaFactor,
            .destination_alpha_factor = (uint32_t)dstAlphaFactor,
            .rgb_operation = (uint32_t)rgbOperation,
            .alpha_operation = (uint32_t)alphaOperation,
            .color_write_mask = (uint32_t)colorMask,
        };
        mglRenderCppSetPipelineBlendState(
            _cppOwner, (uint32_t)index, &blend);
    }
    /* P4.2: gate-on 下 C++ owner 是 blend 状态的唯一权威（value-state 填充走
     * blendStateForAttachment:out: 的 owner-first 读取）；镜像只服务 gate-off
     * 的 ObjC descriptor 路径，gate-on 下不再写。 */
    if (!mglPipelineCacheUsesMetalCpp()) {
        _state.src_blend_rgb_factor[index] = srcRgbFactor;
        _state.src_blend_alpha_factor[index] = srcAlphaFactor;
        _state.dst_blend_rgb_factor[index] = dstRgbFactor;
        _state.dst_blend_alpha_factor[index] = dstAlphaFactor;
        _state.rgb_blend_operation[index] = rgbOperation;
        _state.alpha_blend_operation[index] = alphaOperation;
        _state.color_mask[index] = colorMask;
    }
}

- (void)disableBinaryArchive
{
    _binaryArchiveRequested = NO;
    if ([self ensureCppOwnerCreated]) {
        mglRenderCppDisablePipelineBinaryArchive(_cppOwner);
    }
}

- (void)resetCaches
{
    mglRenderCppResetPipelineCacheOwner(_cppOwner);
    _state.pipelineState = nil;
    _state.pipelineVertexFunction = nil;
    _state.pipelineFragmentFunction = nil;
    [_state.pipelineStateCache removeAllObjects];
    [_state.pipelineStateCacheLRU removeAllObjects];
    [_state.pipelineDescriptorCache removeAllObjects];
    [_state.pipelineDescriptorCacheLRU removeAllObjects];
    [_state.depthStencilStateCache removeAllObjects];
    [_state.depthStencilStateCacheLRU removeAllObjects];
}

- (void)shutdown
{
    [self resetCaches];
    _state.pipelineStateCache = nil;
    _state.pipelineStateCacheLRU = nil;
    _state.pipelineDescriptorCache = nil;
    _state.pipelineDescriptorCacheLRU = nil;
    _state.depthStencilStateCache = nil;
    _state.depthStencilStateCacheLRU = nil;
    _state.depthStencilCacheQueryKey = nil;
    _device = nil;
    mglRenderCppDestroyPipelineCacheOwner(&_cppOwner);
}

- (void)dealloc
{
    mglRenderCppDestroyPipelineCacheOwner(&_cppOwner);
}

@end
