#import "MGLPipelineCache.h"

#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render_cpp.h"
#include "mgl_render_cpp_objc.h"
#include "mgl_air_loader.h"   /* MGLRenderCppPipelineDescriptorState */

@interface MGLPipelineCache ()
- (BOOL)ensureCppOwnerCreated;
- (BOOL)ensureCppOwner;
@end

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
    _state.psoDedupEnabled = psoDedupEnabled;
    _state.dsCacheEnabled = depthStencilCacheEnabled;
    _binaryArchiveRequested = binaryArchiveEnabled;
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
    return [self ensureCppOwnerCreated];
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
    if (![self ensureCppOwner]) return nil;
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
    if (![self ensureCppOwner]) return NO;
    MGLRenderCppPipelineActiveState cached = {0};
    if (mglRenderCppLookupPipeline(_cppOwner, words, &cached) != 1 ||
        !cached.pipeline_state) {
        return NO;
    }
    *pipelineOut =
        (__bridge MGLMetalRenderPipelineStateRef)cached.pipeline_state;
    *vertexFunctionOut =
        (__bridge MGLMetalFunctionRef)cached.vertex_function;
    *fragmentFunctionOut =
        (__bridge MGLMetalFunctionRef)cached.fragment_function;
    return YES;
}

- (NSUInteger)storePipeline:(MGLMetalRenderPipelineStateRef)pipeline
              vertexFunction:(MGLMetalFunctionRef)vertexFunction
            fragmentFunction:(MGLMetalFunctionRef)fragmentFunction
                    forWords:(const uint64_t *)words
{
    if (!pipeline || !words) return 0;
    if (![self ensureCppOwner]) return 0;
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

- (BOOL)pipelineDescriptorStateForWords:(const uint64_t *)words
                                  state:(MGLRenderCppPipelineDescriptorState *)stateOut
{
    if (!words || !stateOut) return NO;
    return [self ensureCppOwner] &&
        mglRenderCppLookupPipelineDescriptorState(
            _cppOwner, words, stateOut) == 1;
}

- (void)storePipelineDescriptorState:(const MGLRenderCppPipelineDescriptorState *)state
                            forWords:(const uint64_t *)words
{
    if (!state || !words) return;
    if (![self ensureCppOwner]) return;
    mglRenderCppStorePipelineDescriptorState(_cppOwner, words, state);
}

- (BOOL)blendStateForAttachment:(NSUInteger)index
                            out:(MGLRenderCppPipelineBlendState *)outState
{
    if (index >= MAX_COLOR_ATTACHMENTS || !outState) return NO;
    return [self ensureCppOwner] &&
        mglRenderCppGetPipelineBlendState(
            _cppOwner, (uint32_t)index, outState) == 0;
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
    NSString *schema = [NSString stringWithFormat:@"%@-cpp",
                        kMGLPipelineArchiveBuildSchema];
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

    if (![self ensureCppOwnerCreated]) return nil;
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
}

- (void)shutdown
{
    [self resetCaches];
    _device = nil;
    mglRenderCppDestroyPipelineCacheOwner(&_cppOwner);
}

- (void)dealloc
{
    mglRenderCppDestroyPipelineCacheOwner(&_cppOwner);
}

@end
