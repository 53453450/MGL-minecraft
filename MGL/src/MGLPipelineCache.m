#import "MGLPipelineCache.h"

#import <os/lock.h>

#import "mgl_frame_activity.h"
#import "mgl_msl_compiler.h"

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

static os_unfair_lock s_binaryArchiveLock = OS_UNFAIR_LOCK_INIT;
static NSMutableDictionary<NSString *, id<MTLBinaryArchive>> *s_binaryArchives;

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
    _state.binaryArchiveEnabled = binaryArchiveEnabled;
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

- (id<MTLDepthStencilState>)depthStencilStateForDescriptor:
    (MTLDepthStencilDescriptor *)descriptor
{
    if (!descriptor || !_device) return nil;
    if (!_state.dsCacheEnabled) {
        id<MTLDepthStencilState> state = [_device newDepthStencilStateWithDescriptor:descriptor];
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
    id<MTLDepthStencilState> cached = _state.depthStencilStateCache[key];
    if (cached) {
        return cached;
    }

    id<MTLDepthStencilState> state = [_device newDepthStencilStateWithDescriptor:descriptor];
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
    if (_state.pipelineStateCache[key]) {
        MGLTouchLRU(_state.pipelineStateCacheLRU, key);
    }
}

- (MTLRenderPipelineDescriptor *)pipelineDescriptorForKey:(MGLPipelineCacheKey *)key
{
    MTLRenderPipelineDescriptor *descriptor = _state.pipelineDescriptorCache[key];
    if (descriptor) MGLTouchLRU(_state.pipelineDescriptorCacheLRU, key);
    return descriptor;
}

- (NSUInteger)storePipelineEntry:(id)entry forKey:(MGLPipelineCacheKey *)key
{
    if (!entry || !key) return 0;
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
    if (!descriptor || !key) return;
    _state.pipelineDescriptorCache[key] = [descriptor copy];
    MGLTouchLRU(_state.pipelineDescriptorCacheLRU, key);
    if (_state.pipelineDescriptorCache.count > 128) {
        MGLEvictLRU(_state.pipelineDescriptorCache,
                    _state.pipelineDescriptorCacheLRU,
                    _state.pipelineDescriptorCache.count - 128);
    }
}

- (void)initializeCompilerIfAvailableUnlessDisabled:(BOOL)disabled
{
#if MGL_HAS_MTL4_COMPILER
    if (!_device || _state.mtl4Compiler || disabled) return;
    if (@available(macOS 26.0, *)) {
        if (![_device respondsToSelector:@selector(newCompilerWithDescriptor:error:)]) return;
        NSError *error = nil;
        MTL4CompilerDescriptor *descriptor = [[MTL4CompilerDescriptor alloc] init];
        descriptor.label = @"MGL Metal 4 shader compiler";
        _state.mtl4Compiler = [_device newCompilerWithDescriptor:descriptor error:&error];
        if (_state.mtl4Compiler) {
            NSLog(@"MGL INFO: Metal 4 compiler enabled for shader libraries");
        } else if (error) {
            NSLog(@"MGL WARNING: Metal 4 compiler unavailable, falling back to MTLDevice library compile: %@",
                  error.localizedDescription);
        }
    }
#else
    (void)disabled;
#endif
}

- (id<MTLLibrary>)newMetalLibraryWithSource:(NSString *)source
                                      options:(MTLCompileOptions *)options
                                        label:(NSString *)label
                                        error:(NSError **)error
{
#if MGL_HAS_MTL4_COMPILER
    return mglCompileMSL(_device, _state.mtl4Compiler, source, options, label, error);
#else
    return mglCompileMSL(_device, nil, source, options, label, error);
#endif
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
    NSString *filename = [NSString stringWithFormat:@"pipeline-%@.binaryarchive", deviceID];
    return [NSURL fileURLWithPath:[mglDir stringByAppendingPathComponent:filename]];
}

- (void)loadBinaryArchive
{
    if (!_state.binaryArchiveEnabled || !_device) return;

    NSURL *archiveURL = [self binaryArchiveURL];
    NSString *archiveKey = archiveURL.path;
    os_unfair_lock_lock(&s_binaryArchiveLock);
    @try {
        id<MTLBinaryArchive> sharedArchive = s_binaryArchives[archiveKey];
        if (sharedArchive) {
            _state.binaryArchive = sharedArchive;
            return;
        }

        NSFileManager *fileManager = NSFileManager.defaultManager;
        BOOL archiveExists = [fileManager fileExistsAtPath:archiveKey];
        MTLBinaryArchiveDescriptor *descriptor = [[MTLBinaryArchiveDescriptor alloc] init];
        if (archiveExists) descriptor.url = archiveURL;

        NSError *loadError = nil;
        id<MTLBinaryArchive> archive = [_device newBinaryArchiveWithDescriptor:descriptor
                                                                         error:&loadError];
        if (!archive && archiveExists) {
            NSError *removeError = nil;
            if (![fileManager removeItemAtURL:archiveURL error:&removeError]) {
                NSLog(@"MGL BINARY ARCHIVE: failed to remove incompatible archive: %@",
                      removeError.localizedDescription);
            }
            NSLog(@"MGL BINARY ARCHIVE: rebuilding incompatible archive: %@",
                  loadError.localizedDescription);
            descriptor = [[MTLBinaryArchiveDescriptor alloc] init];
            loadError = nil;
            archive = [_device newBinaryArchiveWithDescriptor:descriptor error:&loadError];
            archiveExists = NO;
        }

        if (archive) {
            archive.label = @"MGL Pipeline Binary Archive";
            if (!s_binaryArchives) s_binaryArchives = [NSMutableDictionary new];
            s_binaryArchives[archiveKey] = archive;
            _state.binaryArchive = archive;
            NSLog(@"MGL BINARY ARCHIVE: %@ %@",
                  archiveExists ? @"loaded" : @"created", archiveURL.lastPathComponent);
        } else {
            NSLog(@"MGL BINARY ARCHIVE: unavailable, PSO compile will continue without it: %@",
                  loadError.localizedDescription);
        }
    } @catch (NSException *exception) {
        _state.binaryArchive = nil;
        NSLog(@"MGL BINARY ARCHIVE: load exception, continuing without archive: %@",
              exception.reason);
    } @finally {
        os_unfair_lock_unlock(&s_binaryArchiveLock);
    }
}

- (void)saveBinaryArchive
{
    if (!_state.binaryArchive) return;

    NSURL *archiveURL = [self binaryArchiveURL];
    NSError *serializeError = nil;
    BOOL ok = NO;
    os_unfair_lock_lock(&s_binaryArchiveLock);
    @try {
        ok = [_state.binaryArchive serializeToURL:archiveURL error:&serializeError];
    } @catch (NSException *exception) {
        NSLog(@"MGL BINARY ARCHIVE: serialize exception: %@", exception.reason);
    } @finally {
        os_unfair_lock_unlock(&s_binaryArchiveLock);
    }
    if (ok) {
        NSLog(@"MGL BINARY ARCHIVE: saved to %@", archiveURL.lastPathComponent);
    } else if (serializeError) {
        NSLog(@"MGL BINARY ARCHIVE: serialize failed: %@",
              serializeError.localizedDescription);
    }
}

- (void)applyBinaryArchiveToDescriptor:(MTLRenderPipelineDescriptor *)descriptor
{
    if (!_state.binaryArchiveEnabled || !_state.binaryArchive || !descriptor) return;
    descriptor.binaryArchives = @[_state.binaryArchive];
}

- (void)addPipelineToBinaryArchive:(MTLRenderPipelineDescriptor *)descriptor
{
    if (!_state.binaryArchiveEnabled || !_state.binaryArchive || !descriptor) return;
    NSError *addError = nil;
    os_unfair_lock_lock(&s_binaryArchiveLock);
    @try {
        [_state.binaryArchive addRenderPipelineFunctionsWithDescriptor:descriptor error:&addError];
    } @catch (NSException *exception) {
        NSLog(@"MGL BINARY ARCHIVE: addRenderPipeline exception: %@", exception.reason);
    } @finally {
        os_unfair_lock_unlock(&s_binaryArchiveLock);
    }
    if (addError) {
        NSLog(@"MGL BINARY ARCHIVE: addRenderPipeline warning: %@",
              addError.localizedDescription);
    }
}

- (void)invalidatePipelineState
{
    _state.pipelineState = nil;
    _state.pipelineColor0Format = MTLPixelFormatInvalid;
    _state.pipelineDepthFormat = MTLPixelFormatInvalid;
    _state.pipelineStencilFormat = MTLPixelFormatInvalid;
    _state.pipelineProgramName = 0u;
    _state.pipelineVertexFunction = nil;
    _state.pipelineFragmentFunction = nil;
}

- (void)setPipelineState:(id<MTLRenderPipelineState>)pipelineState
{
    _state.pipelineState = pipelineState;
}

- (void)activatePipelineState:(id<MTLRenderPipelineState>)pipelineState
                 color0Format:(MTLPixelFormat)color0Format
                  depthFormat:(MTLPixelFormat)depthFormat
                stencilFormat:(MTLPixelFormat)stencilFormat
                  programName:(GLuint)programName
               vertexFunction:(id<MTLFunction>)vertexFunction
             fragmentFunction:(id<MTLFunction>)fragmentFunction
{
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
    _state.src_blend_rgb_factor[index] = srcRgbFactor;
    _state.src_blend_alpha_factor[index] = srcAlphaFactor;
    _state.dst_blend_rgb_factor[index] = dstRgbFactor;
    _state.dst_blend_alpha_factor[index] = dstAlphaFactor;
    _state.rgb_blend_operation[index] = rgbOperation;
    _state.alpha_blend_operation[index] = alphaOperation;
    _state.color_mask[index] = colorMask;
}

- (void)disableBinaryArchive
{
    _state.binaryArchiveEnabled = NO;
}

- (void)resetCaches
{
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
    _state.binaryArchive = nil;
#if MGL_HAS_MTL4_COMPILER
    _state.mtl4Compiler = nil;
#endif
    _device = nil;
}

@end
