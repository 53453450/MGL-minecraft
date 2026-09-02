/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Pipeline cache facade — migrated from MGLPipelineCache.m.
 * Foundation is used only for binary-archive filesystem paths.
 */

#import <Foundation/Foundation.h>

#include <string.h>

#include "mgl_pipeline_cache_facade.h"
#include "mgl_env_flag.h"
#include "mgl_frame_activity.h"
#include "mgl_render.h"

#if __has_feature(address_sanitizer)
static NSString *const kMGLPipelineArchiveBuildSchema = @"v5-asan";
#elif __has_feature(thread_sanitizer)
static NSString *const kMGLPipelineArchiveBuildSchema = @"v5-tsan";
#else
static NSString *const kMGLPipelineArchiveBuildSchema = @"v5";
#endif

static NSString *MGLSafeArchivePathComponent(NSString *value)
{
    if (value.length == 0) return @"unknown";
    NSCharacterSet *unsafe = [[NSCharacterSet alphanumericCharacterSet] invertedSet];
    return [[value componentsSeparatedByCharactersInSet:unsafe]
        componentsJoinedByString:@"_"];
}

static bool mglPipelineCacheEnsureOwnerCreated(MGLPipelineCacheState *state,
                                               void **owner, void *device,
                                               bool binaryArchiveRequested)
{
    if (!state || !owner || !device) return false;
    if (*owner) return true;
    if (mglRenderCreatePipelineCacheOwner(
            state->psoDedupEnabled ? 1 : 0, state->dsCacheEnabled ? 1 : 0,
            binaryArchiveRequested ? 1 : 0, owner) != 0 || !*owner) {
        *owner = NULL;
        return false;
    }
    MGLRenderPipelineActiveState active = {
        .pipeline_state = state->pipelineState,
        .vertex_function = state->pipelineVertexFunction,
        .fragment_function = state->pipelineFragmentFunction,
        .color0_format = (uint32_t)state->pipelineColor0Format,
        .depth_format = (uint32_t)state->pipelineDepthFormat,
        .stencil_format = (uint32_t)state->pipelineStencilFormat,
        .program_name = state->pipelineProgramName,
    };
    mglRenderActivatePipelineState(*owner, &active);
    return true;
}

void mglPipelineCacheInit(MGLPipelineCacheState *state, bool psoDedupEnabled,
                          bool depthStencilCacheEnabled,
                          bool binaryArchiveRequested)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->psoDedupEnabled = psoDedupEnabled;
    state->dsCacheEnabled = depthStencilCacheEnabled;
    (void)binaryArchiveRequested;
}

void mglPipelineCacheSetDevice(MGLPipelineCacheState *state, void **owner,
                               void *device, bool binaryArchiveRequested)
{
    if (!state || !owner) return;
    if (*owner) {
        mglRenderDestroyPipelineCacheOwner(owner);
    }
    if (device) {
        mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                           binaryArchiveRequested);
    }
}

bool mglPipelineCacheEnsureOwner(MGLPipelineCacheState *state, void **owner,
                                 void *device, bool binaryArchiveRequested)
{
    return mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                              binaryArchiveRequested);
}

bool mglPipelineCacheIsBinaryArchiveEnabled(MGLPipelineCacheState *state,
                                            void *owner,
                                            bool binaryArchiveRequested)
{
    int enabled = binaryArchiveRequested ? 1 : 0;
    if (owner) {
        mglRenderGetPipelineBinaryArchiveState(owner, &enabled, NULL);
    }
    (void)state;
    return enabled != 0;
}

void *mglPipelineCacheDepthStencilStateForValueState(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested,
    const MGLRenderDepthStencilDescriptorState *descriptorState)
{
    if (!descriptorState || !device ||
        !mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                            binaryArchiveRequested)) {
        return NULL;
    }
    void *statePtr = NULL;
    if (state->dsCacheEnabled) {
        int created = 0;
        if (mglRenderGetOrCreateDepthStencilState(*owner, descriptorState,
                                                  &statePtr, &created) == 0 &&
            statePtr) {
            if (created) MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
            return statePtr;
        }
        return NULL;
    }
    if (mglRenderCreateDepthStencilStateFromState(descriptorState, &statePtr) ==
            0 &&
        statePtr) {
        MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
        return statePtr;
    }
    return NULL;
}

bool mglPipelineCacheLookupPipeline(MGLPipelineCacheState *state, void **owner,
                                    void *device, bool binaryArchiveRequested,
                                    const uint64_t *words, void **pipelineOut,
                                    void **vertexFunctionOut,
                                    void **fragmentFunctionOut)
{
    if (pipelineOut) *pipelineOut = NULL;
    if (vertexFunctionOut) *vertexFunctionOut = NULL;
    if (fragmentFunctionOut) *fragmentFunctionOut = NULL;
    if (!words || !pipelineOut || !vertexFunctionOut || !fragmentFunctionOut ||
        !state) {
        return false;
    }
    if (!mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                            binaryArchiveRequested)) {
        return false;
    }
    MGLRenderPipelineActiveState cached = {0};
    if (mglRenderLookupPipeline(*owner, words, &cached) != 1 ||
        !cached.pipeline_state) {
        return false;
    }
    *pipelineOut = cached.pipeline_state;
    *vertexFunctionOut = cached.vertex_function;
    *fragmentFunctionOut = cached.fragment_function;
    return true;
}

uint32_t mglPipelineCacheStorePipeline(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested, void *pipeline, void *vertexFunction,
    void *fragmentFunction, const uint64_t *words)
{
    if (!pipeline || !words || !state) return 0;
    if (!mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                            binaryArchiveRequested)) {
        return 0;
    }
    MGLRenderPipelineActiveState pipelineState = {
        .pipeline_state = pipeline,
        .vertex_function = vertexFunction,
        .fragment_function = fragmentFunction,
    };
    uint32_t removed = 0;
    if (mglRenderStorePipeline(*owner, words, &pipelineState, &removed) != 0) {
        return 0;
    }
    MGL_PERF_ADD(g_mglPipelineCacheEvictionsSinceSwap, removed);
    return removed;
}

bool mglPipelineCachePipelineDescriptorStateForWords(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested, const uint64_t *words,
    MGLRenderPipelineDescriptorState *stateOut)
{
    if (!words || !stateOut || !state) return false;
    return mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                              binaryArchiveRequested) &&
           mglRenderLookupPipelineDescriptorState(*owner, words, stateOut) == 1;
}

void mglPipelineCacheStorePipelineDescriptorState(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested,
    const MGLRenderPipelineDescriptorState *descriptorState,
    const uint64_t *words)
{
    if (!descriptorState || !words || !state) return;
    if (!mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                            binaryArchiveRequested)) {
        return;
    }
    mglRenderStorePipelineDescriptorState(*owner, words, descriptorState);
}

bool mglPipelineCacheBlendStateForAttachment(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested, uint32_t index,
    MGLRenderPipelineBlendState *outState)
{
    if (index >= MAX_COLOR_ATTACHMENTS || !outState || !state) return false;
    return mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                              binaryArchiveRequested) &&
           mglRenderGetPipelineBlendState(*owner, index, outState) == 0;
}

static NSURL *mglPipelineCacheBinaryArchiveURL(void *device)
{
    NSArray *caches = NSSearchPathForDirectoriesInDomains(
        NSCachesDirectory, NSUserDomainMask, YES);
    NSString *baseDir = caches.firstObject ?: NSTemporaryDirectory();
    NSString *bundleID = NSBundle.mainBundle.bundleIdentifier;
    if (bundleID.length == 0) bundleID = NSProcessInfo.processInfo.processName;
    NSString *mglDir =
        [[baseDir stringByAppendingPathComponent:@"MGL"]
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
    (void)mglRenderGetDeviceIdentity(device, &registryID, deviceName,
                                     sizeof(deviceName));
    NSString *deviceNameString = deviceName[0]
        ? [NSString stringWithUTF8String:deviceName]
        : @"unknown";
    NSString *deviceID =
        registryID != 0
            ? [NSString stringWithFormat:@"%016llx",
                                         (unsigned long long)registryID]
            : MGLSafeArchivePathComponent(deviceNameString);
    NSString *schema =
        [NSString stringWithFormat:@"%@-cpp", kMGLPipelineArchiveBuildSchema];
    NSString *filename =
        [NSString stringWithFormat:@"pipeline-%@-%@.binaryarchive", schema,
                                   deviceID];
    return [NSURL fileURLWithPath:[mglDir stringByAppendingPathComponent:filename]];
}

void mglPipelineCacheLoadBinaryArchive(MGLPipelineCacheState *state,
                                       void **owner, void *device,
                                       bool *binaryArchiveRequested)
{
    if (!state || !owner || !device || !binaryArchiveRequested ||
        !*binaryArchiveRequested ||
        !mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                            *binaryArchiveRequested)) {
        return;
    }

    NSURL *archiveURL = mglPipelineCacheBinaryArchiveURL(device);
    NSString *archiveKey = archiveURL.path;
    NSFileManager *fileManager = NSFileManager.defaultManager;
    BOOL archiveExists = [fileManager fileExistsAtPath:archiveKey];
    int reused = 0;
    char message[512] = {0};
    int result = mglRenderLoadPipelineBinaryArchive(
        *owner, archiveKey.UTF8String, (__bridge void *)archiveURL,
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
            *owner, archiveKey.UTF8String, (__bridge void *)archiveURL, 0,
            &reused, message, sizeof(message));
    }
    if (result == 0) {
        NSLog(@"MGL BINARY ARCHIVE: %@ %@",
              reused ? @"reused" : (archiveExists ? @"loaded" : @"created"),
              archiveURL.lastPathComponent);
    } else {
        NSLog(@"MGL BINARY ARCHIVE: unavailable, PSO compile will continue "
              @"without it: %s",
              message[0] ? message : "unknown error");
    }
}

void mglPipelineCacheSaveBinaryArchive(MGLPipelineCacheState *state,
                                       void *owner, void *device)
{
    (void)state;
    (void)device;
    int present = 0;
    if (!owner ||
        mglRenderGetPipelineBinaryArchiveState(owner, NULL, &present) != 0 ||
        !present) {
        return;
    }

    NSURL *archiveURL = mglPipelineCacheBinaryArchiveURL(device);
    NSString *archiveKey = archiveURL.path;
    NSError *removeError = nil;
    char message[512] = {0};
    BOOL ok = mglRenderSerializePipelineBinaryArchive(
                  owner, (__bridge void *)archiveURL, message,
                  sizeof(message)) == 0;
    BOOL discarded = NO;
    if (!ok) {
        NSFileManager *fileManager = NSFileManager.defaultManager;
        discarded = ![fileManager fileExistsAtPath:archiveKey] ||
                    [fileManager removeItemAtURL:archiveURL error:&removeError];
        mglRenderDiscardPipelineBinaryArchive(owner, archiveKey.UTF8String);
    }
    if (ok) {
        NSLog(@"MGL BINARY ARCHIVE: saved to %@", archiveURL.lastPathComponent);
    } else {
        NSString *description = message[0]
            ? [NSString stringWithUTF8String:message]
            : @"unknown error";
        if (discarded) {
            NSLog(@"MGL BINARY ARCHIVE: discarded unserializable archive: %@",
                  description);
        } else {
            NSLog(@"MGL BINARY ARCHIVE: serialize failed: %@; removal failed: %@",
                  description, removeError.localizedDescription);
        }
    }
}

void mglPipelineCacheDisableBinaryArchive(MGLPipelineCacheState *state,
                                          void **owner, void *device,
                                          bool *binaryArchiveRequested)
{
    if (binaryArchiveRequested) *binaryArchiveRequested = false;
    if (state && owner && device &&
        mglPipelineCacheEnsureOwnerCreated(state, owner, device, false)) {
        mglRenderDisablePipelineBinaryArchive(*owner);
    }
}

int mglPipelineCacheCreateRenderPipelineFromState(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested,
    const MGLRenderPipelineDescriptorState *descriptorState,
    void *vertexFunction, void *fragmentFunction, void **pipelineOut,
    char *errorMessage, size_t errorCapacity)
{
    if (!mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                            binaryArchiveRequested)) {
        return -1;
    }
    return mglRenderCreateRenderPipelineFromStateWithArchiveOwner(
        *owner, vertexFunction, fragmentFunction, descriptorState, pipelineOut,
        errorMessage, errorCapacity);
}

void mglPipelineCacheInvalidatePipelineState(MGLPipelineCacheState *state,
                                             void **owner, void *device,
                                             bool binaryArchiveRequested)
{
    if (state && owner && device &&
        mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                           binaryArchiveRequested)) {
        mglRenderInvalidatePipelineActiveState(*owner);
    }
    if (!state) return;
    state->pipelineState = NULL;
    state->pipelineColor0Format = 0u;
    state->pipelineDepthFormat = 0u;
    state->pipelineStencilFormat = 0u;
    state->pipelineProgramName = 0u;
    state->pipelineVertexFunction = NULL;
    state->pipelineFragmentFunction = NULL;
}

void mglPipelineCacheSetPipelineState(MGLPipelineCacheState *state,
                                      void **owner, void *device,
                                      bool binaryArchiveRequested,
                                      void *pipelineState)
{
    if (state && owner && device &&
        mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                           binaryArchiveRequested)) {
        mglRenderSetPipelineActiveObject(*owner, pipelineState);
    }
    if (state) state->pipelineState = pipelineState;
}

void mglPipelineCacheActivatePipelineState(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested, void *pipelineState, uint32_t color0Format,
    uint32_t depthFormat, uint32_t stencilFormat, GLuint programName,
    void *vertexFunction, void *fragmentFunction)
{
    if (state && owner && device &&
        mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                           binaryArchiveRequested)) {
        MGLRenderPipelineActiveState active = {
            .pipeline_state = pipelineState,
            .vertex_function = vertexFunction,
            .fragment_function = fragmentFunction,
            .color0_format = color0Format,
            .depth_format = depthFormat,
            .stencil_format = stencilFormat,
            .program_name = programName,
        };
        mglRenderActivatePipelineState(*owner, &active);
    }
    if (!state) return;
    state->pipelineState = pipelineState;
    state->pipelineColor0Format = color0Format;
    state->pipelineDepthFormat = depthFormat;
    state->pipelineStencilFormat = stencilFormat;
    state->pipelineProgramName = programName;
    state->pipelineVertexFunction = vertexFunction;
    state->pipelineFragmentFunction = fragmentFunction;
}

void mglPipelineCacheSetBlendFactorsForAttachment(
    MGLPipelineCacheState *state, void **owner, void *device,
    bool binaryArchiveRequested, uint32_t index, uint32_t srcRgbFactor,
    uint32_t srcAlphaFactor, uint32_t dstRgbFactor, uint32_t dstAlphaFactor,
    uint32_t rgbOperation, uint32_t alphaOperation, uint32_t colorMask)
{
    if (index >= MAX_COLOR_ATTACHMENTS || !state) return;
    if (mglPipelineCacheEnsureOwnerCreated(state, owner, device,
                                           binaryArchiveRequested)) {
        MGLRenderPipelineBlendState blend = {
            .source_rgb_factor = srcRgbFactor,
            .destination_rgb_factor = dstRgbFactor,
            .source_alpha_factor = srcAlphaFactor,
            .destination_alpha_factor = dstAlphaFactor,
            .rgb_operation = rgbOperation,
            .alpha_operation = alphaOperation,
            .color_write_mask = colorMask,
        };
        mglRenderSetPipelineBlendState(*owner, index, &blend);
    }
}

void mglPipelineCacheResetCaches(MGLPipelineCacheState *state, void **owner)
{
    if (owner && *owner) {
        mglRenderResetPipelineCacheOwner(*owner);
    }
    if (!state) return;
    state->pipelineState = NULL;
    state->pipelineVertexFunction = NULL;
    state->pipelineFragmentFunction = NULL;
}

void mglPipelineCacheShutdown(MGLPipelineCacheState *state, void **owner)
{
    if (owner) {
        mglPipelineCacheResetCaches(state, owner);
        mglRenderDestroyPipelineCacheOwner(owner);
    }
}
