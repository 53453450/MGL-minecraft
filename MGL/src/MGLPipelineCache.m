/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#import "MGLPipelineCache.h"

#import "mgl_frame_activity.h"
#include "mgl_env_flag.h"
#include "mgl_render.h"
#include "mgl_air_loader.h"   /* MGLRenderPipelineDescriptorState */

@interface MGLPipelineCache ()
- (BOOL)ensureOwnerCreated;
- (BOOL)ensureOwner;
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

    _state.pipelineColor0Format = 0u;
    _state.pipelineDepthFormat = 0u;
    _state.pipelineStencilFormat = 0u;
    _state.psoDedupEnabled = psoDedupEnabled;
    _state.dsCacheEnabled = depthStencilCacheEnabled;
    _binaryArchiveRequested = binaryArchiveEnabled;
    return self;
}

- (const MGLPipelineCacheState *)state
{
    return &_state;
}

- (BOOL)ensureOwnerCreated
{
    if (_owner) return YES;
    if (!_device) return NO;
    if (mglRenderCreatePipelineCacheOwner(
            _state.psoDedupEnabled ? 1 : 0,
            _state.dsCacheEnabled ? 1 : 0,
            _binaryArchiveRequested ? 1 : 0,
            &_owner) != 0 || !_owner) {
        _owner = NULL;
        return NO;
    }

    MGLRenderPipelineActiveState active = {
        .pipeline_state = _state.pipelineState,
        .vertex_function = _state.pipelineVertexFunction,
        .fragment_function = _state.pipelineFragmentFunction,
        .color0_format = (uint32_t)_state.pipelineColor0Format,
        .depth_format = (uint32_t)_state.pipelineDepthFormat,
        .stencil_format = (uint32_t)_state.pipelineStencilFormat,
        .program_name = _state.pipelineProgramName,
    };
    mglRenderActivatePipelineState(_owner, &active);
    return YES;
}

- (BOOL)ensureOwner
{
    return [self ensureOwnerCreated];
}

- (BOOL)isBinaryArchiveEnabled
{
    int enabled = _binaryArchiveRequested ? 1 : 0;
    if (_owner) {
        mglRenderGetPipelineBinaryArchiveState(
            _owner, &enabled, NULL);
    }
    return enabled != 0;
}

- (id)device
{
    return (__bridge id)_device;
}

- (void)setDevice:(id)device
{
    void *opaqueDevice = (__bridge void *)device;
    if (_device != opaqueDevice) {
        mglRenderDestroyPipelineCacheOwner(&_owner);
    }
    _device = opaqueDevice;
    if (_device) [self ensureOwnerCreated];
}

- (id)depthStencilStateForValueState:
    (const MGLRenderDepthStencilDescriptorState *)descriptorState
{
    if (!descriptorState || !_device || ![self ensureOwner]) return nil;
    void *statePtr = NULL;
    if (_state.dsCacheEnabled) {
        int created = 0;
        if (mglRenderGetOrCreateDepthStencilState(
                _owner, descriptorState, &statePtr, &created) == 0 &&
            statePtr) {
            if (created) MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
            return (__bridge id)statePtr;
        }
        return nil;
    }
    if (mglRenderCreateDepthStencilStateFromState(
            descriptorState, &statePtr) == 0 && statePtr) {
        MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
        return (__bridge_transfer id)statePtr;
    }
    return nil;
}

- (BOOL)lookupPipelineForWords:(const uint64_t *)words
                      pipeline:(id *)pipelineOut
                vertexFunction:(id *)vertexFunctionOut
              fragmentFunction:(id *)fragmentFunctionOut
{
    if (pipelineOut) *pipelineOut = nil;
    if (vertexFunctionOut) *vertexFunctionOut = nil;
    if (fragmentFunctionOut) *fragmentFunctionOut = nil;
    if (!words || !pipelineOut || !vertexFunctionOut ||
        !fragmentFunctionOut) {
        return NO;
    }
    if (![self ensureOwner]) return NO;
    MGLRenderPipelineActiveState cached = {0};
    if (mglRenderLookupPipeline(_owner, words, &cached) != 1 ||
        !cached.pipeline_state) {
        return NO;
    }
    *pipelineOut = (__bridge id)cached.pipeline_state;
    *vertexFunctionOut = (__bridge id)cached.vertex_function;
    *fragmentFunctionOut = (__bridge id)cached.fragment_function;
    return YES;
}

- (NSUInteger)storePipeline:(id)pipeline
              vertexFunction:(id)vertexFunction
            fragmentFunction:(id)fragmentFunction
                    forWords:(const uint64_t *)words
{
    if (!pipeline || !words) return 0;
    if (![self ensureOwner]) return 0;
    MGLRenderPipelineActiveState state = {
        .pipeline_state = (__bridge void *)pipeline,
        .vertex_function = (__bridge void *)vertexFunction,
        .fragment_function = (__bridge void *)fragmentFunction,
    };
    uint32_t removed = 0;
    if (mglRenderStorePipeline(
            _owner, words, &state, &removed) != 0) {
        return 0;
    }
    MGL_PERF_ADD(g_mglPipelineCacheEvictionsSinceSwap, removed);
    return (NSUInteger)removed;
}

- (BOOL)pipelineDescriptorStateForWords:(const uint64_t *)words
                                  state:(MGLRenderPipelineDescriptorState *)stateOut
{
    if (!words || !stateOut) return NO;
    return [self ensureOwner] &&
        mglRenderLookupPipelineDescriptorState(
            _owner, words, stateOut) == 1;
}

- (void)storePipelineDescriptorState:(const MGLRenderPipelineDescriptorState *)state
                            forWords:(const uint64_t *)words
{
    if (!state || !words) return;
    if (![self ensureOwner]) return;
    mglRenderStorePipelineDescriptorState(_owner, words, state);
}

- (BOOL)blendStateForAttachment:(NSUInteger)index
                            out:(MGLRenderPipelineBlendState *)outState
{
    if (index >= MAX_COLOR_ATTACHMENTS || !outState) return NO;
    return [self ensureOwner] &&
        mglRenderGetPipelineBlendState(
            _owner, (uint32_t)index, outState) == 0;
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
    char deviceName[256] = {0};
    (void)mglRenderGetDeviceIdentity(_device,
                                        &registryID, deviceName,
                                        sizeof(deviceName));
    NSString *deviceNameString = deviceName[0]
        ? [NSString stringWithUTF8String:deviceName]
        : @"unknown";
    NSString *deviceID = registryID != 0
        ? [NSString stringWithFormat:@"%016llx", (unsigned long long)registryID]
        : MGLSafeArchivePathComponent(deviceNameString);
    NSString *schema = [NSString stringWithFormat:@"%@-cpp",
                        kMGLPipelineArchiveBuildSchema];
    NSString *filename = [NSString stringWithFormat:@"pipeline-%@-%@.binaryarchive",
                          schema, deviceID];
    return [NSURL fileURLWithPath:[mglDir stringByAppendingPathComponent:filename]];
}

- (void)loadBinaryArchive
{
    if (!self.binaryArchiveEnabled || !_device ||
        ![self ensureOwnerCreated]) return;

    NSURL *archiveURL = [self binaryArchiveURL];
    NSString *archiveKey = archiveURL.path;
    NSFileManager *fileManager = NSFileManager.defaultManager;
    BOOL archiveExists = [fileManager fileExistsAtPath:archiveKey];
    int reused = 0;
    char message[512] = {0};
    int result = mglRenderLoadPipelineBinaryArchive(
        _owner, archiveKey.UTF8String, (__bridge void *)archiveURL,
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
        result = mglRenderLoadPipelineBinaryArchive(
            _owner, archiveKey.UTF8String, (__bridge void *)archiveURL,
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
    if (!_owner ||
        mglRenderGetPipelineBinaryArchiveState(
            _owner, NULL, &present) != 0 || !present) return;

    NSURL *archiveURL = [self binaryArchiveURL];
    NSString *archiveKey = archiveURL.path;
    NSError *removeError = nil;
    char message[512] = {0};
    BOOL ok = mglRenderSerializePipelineBinaryArchive(
        _owner, (__bridge void *)archiveURL,
        message, sizeof(message)) == 0;
    BOOL discarded = NO;
    if (!ok) {
        NSFileManager *fileManager = NSFileManager.defaultManager;
        discarded = ![fileManager fileExistsAtPath:archiveKey] ||
            [fileManager removeItemAtURL:archiveURL error:&removeError];
        mglRenderDiscardPipelineBinaryArchive(
            _owner, archiveKey.UTF8String);
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

- (int)createRenderPipelineFromState:
    (const MGLRenderPipelineDescriptorState *)state
    vertexFunction:(void *)vertexFunction
    fragmentFunction:(void *)fragmentFunction
    pipelineOut:(void **)pipelineOut
    errorMessage:(char *)errorMessage
    errorCapacity:(size_t)errorCapacity
{
    if (![self ensureOwnerCreated]) return -1;
    return mglRenderCreateRenderPipelineFromStateWithArchiveOwner(
        _owner, vertexFunction, fragmentFunction, state,
        pipelineOut, errorMessage, errorCapacity);
}

- (void)invalidatePipelineState
{
    if ([self ensureOwner]) {
        mglRenderInvalidatePipelineActiveState(_owner);
    }
    _state.pipelineState = NULL;
    _state.pipelineColor0Format = 0u;
    _state.pipelineDepthFormat = 0u;
    _state.pipelineStencilFormat = 0u;
    _state.pipelineProgramName = 0u;
    _state.pipelineVertexFunction = NULL;
    _state.pipelineFragmentFunction = NULL;
}

- (void)setPipelineState:(id)pipelineState
{
    if ([self ensureOwner]) {
        mglRenderSetPipelineActiveObject(
            _owner, (__bridge void *)pipelineState);
    }
    _state.pipelineState = (__bridge void *)pipelineState;
}

- (void)activatePipelineState:(id)pipelineState
                 color0Format:(uint32_t)color0Format
                  depthFormat:(uint32_t)depthFormat
                stencilFormat:(uint32_t)stencilFormat
                  programName:(GLuint)programName
               vertexFunction:(id)vertexFunction
             fragmentFunction:(id)fragmentFunction
{
    if ([self ensureOwner]) {
        MGLRenderPipelineActiveState active = {
            .pipeline_state = (__bridge void *)pipelineState,
            .vertex_function = (__bridge void *)vertexFunction,
            .fragment_function = (__bridge void *)fragmentFunction,
            .color0_format = (uint32_t)color0Format,
            .depth_format = (uint32_t)depthFormat,
            .stencil_format = (uint32_t)stencilFormat,
            .program_name = programName,
        };
        mglRenderActivatePipelineState(_owner, &active);
    }
    _state.pipelineState = (__bridge void *)pipelineState;
    _state.pipelineColor0Format = (uint64_t)color0Format;
    _state.pipelineDepthFormat = (uint64_t)depthFormat;
    _state.pipelineStencilFormat = (uint64_t)stencilFormat;
    _state.pipelineProgramName = programName;
    _state.pipelineVertexFunction = (__bridge void *)vertexFunction;
    _state.pipelineFragmentFunction = (__bridge void *)fragmentFunction;
}

- (void)setBlendFactorsForAttachment:(NSUInteger)index
                        srcRgbFactor:(uint32_t)srcRgbFactor
                      srcAlphaFactor:(uint32_t)srcAlphaFactor
                        dstRgbFactor:(uint32_t)dstRgbFactor
                      dstAlphaFactor:(uint32_t)dstAlphaFactor
                        rgbOperation:(uint32_t)rgbOperation
                      alphaOperation:(uint32_t)alphaOperation
                           colorMask:(uint32_t)colorMask
{
    if (index >= MAX_COLOR_ATTACHMENTS) return;
    if ([self ensureOwner]) {
        MGLRenderPipelineBlendState blend = {
            .source_rgb_factor = (uint32_t)srcRgbFactor,
            .destination_rgb_factor = (uint32_t)dstRgbFactor,
            .source_alpha_factor = (uint32_t)srcAlphaFactor,
            .destination_alpha_factor = (uint32_t)dstAlphaFactor,
            .rgb_operation = (uint32_t)rgbOperation,
            .alpha_operation = (uint32_t)alphaOperation,
            .color_write_mask = (uint32_t)colorMask,
        };
        mglRenderSetPipelineBlendState(
            _owner, (uint32_t)index, &blend);
    }
}

- (void)disableBinaryArchive
{
    _binaryArchiveRequested = NO;
    if ([self ensureOwnerCreated]) {
        mglRenderDisablePipelineBinaryArchive(_owner);
    }
}

- (void)resetCaches
{
    mglRenderResetPipelineCacheOwner(_owner);
    _state.pipelineState = NULL;
    _state.pipelineVertexFunction = NULL;
    _state.pipelineFragmentFunction = NULL;
}

- (void)shutdown
{
    [self resetCaches];
    _device = NULL;
    mglRenderDestroyPipelineCacheOwner(&_owner);
}

- (void)dealloc
{
    mglRenderDestroyPipelineCacheOwner(&_owner);
}

@end
