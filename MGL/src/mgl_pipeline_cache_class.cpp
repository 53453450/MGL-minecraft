/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_pipeline_cache_class.cpp - the MGLPipelineCache class, without a .m
 * (P0-1, log 205; T5 option (a)).
 *
 * The class was the last Objective-C object in MGL/ besides the platform shell
 * itself.  It is registered with the Objective-C runtime at load time instead:
 * objc_allocateClassPair / class_addIvar / class_addMethod /
 * objc_registerClassPair, with each method implemented as a plain C function and
 * the ivars reached through ivar_getOffset.  Message sends become objc_msgSend
 * through the typed mglPcSend<> helper, @"..." literals become CFSTR or
 * stringWithUTF8String:, and NSLog becomes fprintf.
 *
 * This is deliberately the *class* only: its callers (the shell's cache bridges)
 * keep sending it messages, which is what a runtime-registered class supports.
 *
 * The ivar order below must match the interface in MGLPipelineCache.h; the
 * offsets are resolved once, at registration (rule 64: the registration runs
 * from a constructor, nothing calls it).
 */

#include <objc/runtime.h>
#include <objc/message.h>

#include <CoreFoundation/CoreFoundation.h>

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "glm_limits.h"              /* MAX_COLOR_ATTACHMENTS */
#include "mgl_frame_activity.h"      /* MGL_PERF_INC / MGL_PERF_ADD */
#include "mgl_pipeline_cache_path.h" /* archive path (log 203) */
#include "mgl_pipeline_cache_state.h"
#include "mgl_render.h"
#include "mgl_types_state.h"

#ifndef MGL_PLATFORM_SHELL_SMOKE

/* BOOL is a plain byte in C; keep the Objective-C spelling readable. */
typedef signed char MGLPcBool;

/* Typed message send: the cast is what makes objc_msgSend's ABI work out. */
template <typename R, typename... Args>
static inline R mglPcSend(id object, SEL selector, Args... args)
{
    return ((R (*)(id, SEL, Args...))objc_msgSend)(object, selector, args...);
}

/* ---------------------------------------------------------------- runtime --- */

static Class mglPcClass = Nil;
static ptrdiff_t mglPcOffState = 0;
static ptrdiff_t mglPcOffDevice = 0;
static ptrdiff_t mglPcOffOwner = 0;
static ptrdiff_t mglPcOffArchiveRequested = 0;

static SEL mglPcSelInit = NULL;
static SEL mglPcSelEnsureOwner = NULL;
static SEL mglPcSelEnsureOwnerCreated = NULL;
static SEL mglPcSelIsBinaryArchiveEnabled = NULL;
static SEL mglPcSelBinaryArchivePath = NULL;
static SEL mglPcSelBinaryArchiveURL = NULL;
static SEL mglPcSelState = NULL;

static MGLPipelineCacheState *mglPcState(id self)
{
    return (MGLPipelineCacheState *)((char *)self + mglPcOffState);
}

static void *mglPcDevice(id self)
{
    return *(void **)((char *)self + mglPcOffDevice);
}

static void mglPcSetDeviceIvar(id self, void *device)
{
    *(void **)((char *)self + mglPcOffDevice) = device;
}

static void *mglPcOwnerSlot(id self)
{
    return (void *)((char *)self + mglPcOffOwner);
}

static void *mglPcOwner(id self)
{
    return *(void **)((char *)self + mglPcOffOwner);
}

static MGLPcBool mglPcArchiveRequested(id self)
{
    return *(MGLPcBool *)((char *)self + mglPcOffArchiveRequested);
}

static void mglPcSetArchiveRequested(id self, MGLPcBool value)
{
    *(MGLPcBool *)((char *)self + mglPcOffArchiveRequested) = value;
}

/* ------------------------------------------------------------- the class --- */

/* v5 excludes either kind of incomplete render pipeline and isolates both
 * sanitizer builds and archive producers. The producer boundary prevents the
 * temporary A/B implementations from sharing mutable state; the archive-aware
 * PSO creation path below separately prevents repeated adds on cache hits. */
#if __has_feature(address_sanitizer)
static const char *const kMGLPipelineArchiveBuildSchema = "v5-asan";
#elif __has_feature(thread_sanitizer)
static const char *const kMGLPipelineArchiveBuildSchema = "v5-tsan";
#else
static const char *const kMGLPipelineArchiveBuildSchema = "v5";
#endif

static id mglPcInit(id self, SEL _cmd, MGLPcBool psoDedupEnabled,
                    MGLPcBool depthStencilCacheEnabled,
                    MGLPcBool binaryArchiveEnabled)
{
    struct objc_super super = { self, class_getSuperclass(object_getClass(self)) };
    self = ((id (*)(struct objc_super *, SEL))objc_msgSendSuper)(
        &super, sel_registerName("init"));
    if (!self) return NULL;

    MGLPipelineCacheState *state = mglPcState(self);
    state->pipelineColor0Format = 0u;
    state->pipelineDepthFormat = 0u;
    state->pipelineStencilFormat = 0u;
    state->psoDedupEnabled = (uint8_t)(psoDedupEnabled ? 1 : 0);
    state->dsCacheEnabled = (uint8_t)(depthStencilCacheEnabled ? 1 : 0);
    mglPcSetArchiveRequested(self, binaryArchiveEnabled);
    return self;
}

static const MGLPipelineCacheState *mglPcGetState(id self, SEL _cmd)
{
    return mglPcState(self);
}

static MGLPcBool mglPcEnsureOwnerCreated(id self, SEL _cmd)
{
    void **ownerSlot = (void **)mglPcOwnerSlot(self);
    MGLPipelineCacheState *state = mglPcState(self);
    if (*ownerSlot) return 1;
    if (!mglPcDevice(self)) return 0;
    if (mglRenderCreatePipelineCacheOwner(
            state->psoDedupEnabled ? 1 : 0, state->dsCacheEnabled ? 1 : 0,
            mglPcArchiveRequested(self) ? 1 : 0, ownerSlot) != 0 ||
        !*ownerSlot) {
        *ownerSlot = NULL;
        return 0;
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
    mglRenderActivatePipelineState(*ownerSlot, &active);
    return 1;
}

static MGLPcBool mglPcEnsureOwner(id self, SEL _cmd)
{
    return mglPcEnsureOwnerCreated(self, _cmd);
}

static MGLPcBool mglPcIsBinaryArchiveEnabled(id self, SEL _cmd)
{
    int enabled = mglPcArchiveRequested(self) ? 1 : 0;
    void *owner = mglPcOwner(self);
    if (owner) {
        mglRenderGetPipelineBinaryArchiveState(owner, &enabled, NULL);
    }
    return enabled != 0;
}

static id mglPcGetDevice(id self, SEL _cmd)
{
    return (id)mglPcDevice(self);
}

static void mglPcSetDevice(id self, SEL _cmd, id device)
{
    void *opaqueDevice = (void *)device;
    void **ownerSlot = (void **)mglPcOwnerSlot(self);
    if (mglPcDevice(self) != opaqueDevice) {
        mglRenderDestroyPipelineCacheOwner(ownerSlot);
    }
    mglPcSetDeviceIvar(self, opaqueDevice);
    if (opaqueDevice) {
        (void)mglPcEnsureOwnerCreated(self, _cmd);
    }
}

static id mglPcDepthStencilStateForValueState(
    id self, SEL _cmd,
    const MGLRenderDepthStencilDescriptorState *descriptorState)
{
    if (!descriptorState || !mglPcDevice(self) ||
        !mglPcEnsureOwner(self, _cmd)) {
        return NULL;
    }
    void *owner = mglPcOwner(self);
    void *statePtr = NULL;
    if (mglPcState(self)->dsCacheEnabled) {
        int created = 0;
        if (mglRenderGetOrCreateDepthStencilState(owner, descriptorState,
                                                  &statePtr, &created) == 0 &&
            statePtr) {
            if (created) MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
            return (id)statePtr;
        }
        return NULL;
    }
    if (mglRenderCreateDepthStencilStateFromState(descriptorState, &statePtr) ==
            0 &&
        statePtr) {
        MGL_PERF_INC(g_mglDepthStencilStateCreatesSinceSwap);
        return (id)statePtr;
    }
    return NULL;
}

static MGLPcBool mglPcLookupPipelineForWords(id self, SEL _cmd,
                                             const uint64_t *words,
                                             id *pipelineOut,
                                             id *vertexFunctionOut,
                                             id *fragmentFunctionOut)
{
    if (pipelineOut) *pipelineOut = NULL;
    if (vertexFunctionOut) *vertexFunctionOut = NULL;
    if (fragmentFunctionOut) *fragmentFunctionOut = NULL;
    if (!words || !pipelineOut || !vertexFunctionOut || !fragmentFunctionOut) {
        return 0;
    }
    if (!mglPcEnsureOwner(self, _cmd)) return 0;
    MGLRenderPipelineActiveState cached = {0};
    if (mglRenderLookupPipeline(mglPcOwner(self), words, &cached) != 1 ||
        !cached.pipeline_state) {
        return 0;
    }
    *pipelineOut = (id)cached.pipeline_state;
    *vertexFunctionOut = (id)cached.vertex_function;
    *fragmentFunctionOut = (id)cached.fragment_function;
    return 1;
}

static unsigned long mglPcStorePipeline(id self, SEL _cmd, id pipeline,
                                        id vertexFunction, id fragmentFunction,
                                        const uint64_t *words)
{
    if (!pipeline || !words) return 0;
    if (!mglPcEnsureOwner(self, _cmd)) return 0;
    MGLRenderPipelineActiveState state = {
        .pipeline_state = (void *)pipeline,
        .vertex_function = (void *)vertexFunction,
        .fragment_function = (void *)fragmentFunction,
    };
    uint32_t removed = 0;
    if (mglRenderStorePipeline(mglPcOwner(self), words, &state, &removed) != 0) {
        return 0;
    }
    MGL_PERF_ADD(g_mglPipelineCacheEvictionsSinceSwap, removed);
    return (unsigned long)removed;
}

static MGLPcBool mglPcPipelineDescriptorStateForWords(
    id self, SEL _cmd, const uint64_t *words,
    MGLRenderPipelineDescriptorState *stateOut)
{
    if (!words || !stateOut) return 0;
    return mglPcEnsureOwner(self, _cmd) &&
           mglRenderLookupPipelineDescriptorState(mglPcOwner(self), words,
                                                  stateOut) == 1;
}

static void mglPcStorePipelineDescriptorState(
    id self, SEL _cmd, const MGLRenderPipelineDescriptorState *state,
    const uint64_t *words)
{
    if (!state || !words) return;
    if (!mglPcEnsureOwner(self, _cmd)) return;
    mglRenderStorePipelineDescriptorState(mglPcOwner(self), words, state);
}

static MGLPcBool mglPcBlendStateForAttachment(
    id self, SEL _cmd, unsigned long index, MGLRenderPipelineBlendState *outState)
{
    if (index >= MAX_COLOR_ATTACHMENTS || !outState) return 0;
    return mglPcEnsureOwner(self, _cmd) &&
           mglRenderGetPipelineBlendState(mglPcOwner(self), (uint32_t)index,
                                          outState) == 0;
}

/* The path itself is built in C (mgl_pipeline_cache_path.c); only the NSURL the
 * Metal-cpp archive API takes is still built here. */
static id mglPcBinaryArchivePath(id self, SEL _cmd)
{
    char path[PATH_MAX] = {0};
    if (mglPipelineCacheArchiveKey(mglPcDevice(self),
                                   kMGLPipelineArchiveBuildSchema, path,
                                   sizeof(path)) != 0) {
        return NULL;
    }
    return mglPcSend<id>((id)objc_getClass("NSString"),
                         sel_registerName("stringWithUTF8String:"), path);
}

static id mglPcBinaryArchiveURL(id self, SEL _cmd)
{
    id path = mglPcBinaryArchivePath(self, _cmd);
    if (!path) return NULL;
    return mglPcSend<id>((id)objc_getClass("NSURL"),
                         sel_registerName("fileURLWithPath:"), path);
}

static const char *mglPcLastPathComponent(id url)
{
    if (!url) return "";
    id component = mglPcSend<id>(url, sel_registerName("lastPathComponent"));
    const char *utf8 = component
                           ? mglPcSend<const char *>(component,
                                                     sel_registerName("UTF8String"))
                           : NULL;
    return utf8 ? utf8 : "";
}

static void mglPcLoadBinaryArchive(id self, SEL _cmd)
{
    if (!mglPcIsBinaryArchiveEnabled(self, _cmd) || !mglPcDevice(self) ||
        !mglPcEnsureOwnerCreated(self, _cmd)) {
        return;
    }

    id archiveURL = mglPcBinaryArchiveURL(self, _cmd);
    char archiveKey[PATH_MAX] = {0};
    (void)mglPipelineCacheArchiveKey(mglPcDevice(self),
                                     kMGLPipelineArchiveBuildSchema, archiveKey,
                                     sizeof(archiveKey));
    int archiveExists = mglPipelineCacheArchiveExists(archiveKey);
    int reused = 0;
    char message[512] = {0};
    void *owner = mglPcOwner(self);
    int result = mglRenderLoadPipelineBinaryArchive(
        owner, archiveKey, (void *)archiveURL, archiveExists ? 1 : 0, &reused,
        message, sizeof(message));
    if (result != 0 && archiveExists) {
        if (!mglPipelineCacheArchiveRemove(archiveKey)) {
            fprintf(stderr,
                    "MGL BINARY ARCHIVE: failed to remove incompatible archive: %s\n",
                    strerror(errno));
        }
        fprintf(stderr,
                "MGL BINARY ARCHIVE: rebuilding incompatible archive: %s\n",
                message[0] ? message : "unknown error");
        archiveExists = 0;
        message[0] = '\0';
        result = mglRenderLoadPipelineBinaryArchive(
            owner, archiveKey, (void *)archiveURL, 0, &reused, message,
            sizeof(message));
    }
    if (result == 0) {
        fprintf(stderr, "MGL BINARY ARCHIVE: %s %s\n",
                reused ? "reused" : (archiveExists ? "loaded" : "created"),
                mglPcLastPathComponent(archiveURL));
    } else {
        fprintf(stderr,
                "MGL BINARY ARCHIVE: unavailable, PSO compile will continue without it: %s\n",
                message[0] ? message : "unknown error");
    }
}

static void mglPcSaveBinaryArchive(id self, SEL _cmd)
{
    int present = 0;
    void *owner = mglPcOwner(self);
    if (!owner ||
        mglRenderGetPipelineBinaryArchiveState(owner, NULL, &present) != 0 ||
        !present) {
        return;
    }

    id archiveURL = mglPcBinaryArchiveURL(self, _cmd);
    char archiveKey[PATH_MAX] = {0};
    (void)mglPipelineCacheArchiveKey(mglPcDevice(self),
                                     kMGLPipelineArchiveBuildSchema, archiveKey,
                                     sizeof(archiveKey));
    char message[512] = {0};
    MGLPcBool ok =
        mglRenderSerializePipelineBinaryArchive(owner, (void *)archiveURL,
                                                message, sizeof(message)) == 0;
    MGLPcBool discarded = 0;
    if (!ok) {
        discarded = !mglPipelineCacheArchiveExists(archiveKey) ||
                    mglPipelineCacheArchiveRemove(archiveKey);
        mglRenderDiscardPipelineBinaryArchive(owner, archiveKey);
    }
    if (ok) {
        fprintf(stderr, "MGL BINARY ARCHIVE: saved to %s\n",
                mglPcLastPathComponent(archiveURL));
        return;
    }
    const char *rawDescription = message[0] ? message : "unknown error";
    id description =
        message[0]
            ? mglPcSend<id>((id)objc_getClass("NSString"),
                            sel_registerName("stringWithUTF8String:"), message)
            : (id)CFSTR("unknown error");
    const char *descriptionUtf8 =
        mglPcSend<const char *>(description, sel_registerName("UTF8String"));
    if (!descriptionUtf8) descriptionUtf8 = rawDescription;
    if (discarded) {
        fprintf(stderr,
                "MGL BINARY ARCHIVE: discarded unserializable archive: %s\n",
                descriptionUtf8);
    } else {
        fprintf(stderr,
                "MGL BINARY ARCHIVE: serialize failed: %s; removal failed: %s\n",
                descriptionUtf8, strerror(errno));
    }
}

static int mglPcCreateRenderPipelineFromState(
    id self, SEL _cmd, const MGLRenderPipelineDescriptorState *state,
    void *vertexFunction, void *fragmentFunction, void **pipelineOut,
    char *errorMessage, size_t errorCapacity)
{
    if (!mglPcEnsureOwnerCreated(self, _cmd)) return -1;
    return mglRenderCreateRenderPipelineFromStateWithArchiveOwner(
        mglPcOwner(self), vertexFunction, fragmentFunction, state, pipelineOut,
        errorMessage, errorCapacity);
}

static void mglPcInvalidatePipelineState(id self, SEL _cmd)
{
    if (mglPcEnsureOwner(self, _cmd)) {
        mglRenderInvalidatePipelineActiveState(mglPcOwner(self));
    }
    MGLPipelineCacheState *state = mglPcState(self);
    state->pipelineState = NULL;
    state->pipelineColor0Format = 0u;
    state->pipelineDepthFormat = 0u;
    state->pipelineStencilFormat = 0u;
    state->pipelineProgramName = 0u;
    state->pipelineVertexFunction = NULL;
    state->pipelineFragmentFunction = NULL;
}

static void mglPcSetPipelineState(id self, SEL _cmd, id pipelineState)
{
    if (mglPcEnsureOwner(self, _cmd)) {
        mglRenderSetPipelineActiveObject(mglPcOwner(self), (void *)pipelineState);
    }
    mglPcState(self)->pipelineState = (void *)pipelineState;
}

static void mglPcActivatePipelineState(id self, SEL _cmd, id pipelineState,
                                       uint32_t color0Format,
                                       uint32_t depthFormat,
                                       uint32_t stencilFormat,
                                       uint32_t programName, id vertexFunction,
                                       id fragmentFunction)
{
    if (mglPcEnsureOwner(self, _cmd)) {
        MGLRenderPipelineActiveState active = {
            .pipeline_state = (void *)pipelineState,
            .vertex_function = (void *)vertexFunction,
            .fragment_function = (void *)fragmentFunction,
            .color0_format = color0Format,
            .depth_format = depthFormat,
            .stencil_format = stencilFormat,
            .program_name = programName,
        };
        mglRenderActivatePipelineState(mglPcOwner(self), &active);
    }
    MGLPipelineCacheState *state = mglPcState(self);
    state->pipelineState = (void *)pipelineState;
    state->pipelineColor0Format = (uint64_t)color0Format;
    state->pipelineDepthFormat = (uint64_t)depthFormat;
    state->pipelineStencilFormat = (uint64_t)stencilFormat;
    state->pipelineProgramName = programName;
    state->pipelineVertexFunction = (void *)vertexFunction;
    state->pipelineFragmentFunction = (void *)fragmentFunction;
}

static void mglPcSetBlendFactorsForAttachment(
    id self, SEL _cmd, unsigned long index, uint32_t srcRgbFactor,
    uint32_t srcAlphaFactor, uint32_t dstRgbFactor, uint32_t dstAlphaFactor,
    uint32_t rgbOperation, uint32_t alphaOperation, uint32_t colorMask)
{
    if (index >= MAX_COLOR_ATTACHMENTS) return;
    if (mglPcEnsureOwner(self, _cmd)) {
        MGLRenderPipelineBlendState blend = {
            .source_rgb_factor = srcRgbFactor,
            .destination_rgb_factor = dstRgbFactor,
            .source_alpha_factor = srcAlphaFactor,
            .destination_alpha_factor = dstAlphaFactor,
            .rgb_operation = rgbOperation,
            .alpha_operation = alphaOperation,
            .color_write_mask = colorMask,
        };
        mglRenderSetPipelineBlendState(mglPcOwner(self), (uint32_t)index, &blend);
    }
}

static void mglPcDisableBinaryArchive(id self, SEL _cmd)
{
    mglPcSetArchiveRequested(self, 0);
    if (mglPcEnsureOwnerCreated(self, _cmd)) {
        mglRenderDisablePipelineBinaryArchive(mglPcOwner(self));
    }
}

static void mglPcResetCaches(id self, SEL _cmd)
{
    mglRenderResetPipelineCacheOwner(mglPcOwner(self));
    MGLPipelineCacheState *state = mglPcState(self);
    state->pipelineState = NULL;
    state->pipelineVertexFunction = NULL;
    state->pipelineFragmentFunction = NULL;
}

static void mglPcShutdown(id self, SEL _cmd)
{
    mglPcResetCaches(self, _cmd);
    mglPcSetDeviceIvar(self, NULL);
    mglRenderDestroyPipelineCacheOwner((void **)mglPcOwnerSlot(self));
}

static void mglPcDealloc(id self, SEL _cmd)
{
    mglRenderDestroyPipelineCacheOwner((void **)mglPcOwnerSlot(self));
    struct objc_super super = { self, class_getSuperclass(object_getClass(self)) };
    ((void (*)(struct objc_super *, SEL))objc_msgSendSuper)(
        &super, sel_registerName("dealloc"));
}

/* NSLog(@"MGL BINARY ARCHIVE: unavailable ...") style logging keeps its exact
 * text: the A/B oracle compares the stderr lines after stripping the NSLog
 * prefix, and fprintf produces the same payload. */

__attribute__((constructor))
static void mglPipelineCacheRegisterClass(void)
{
    Class cls = objc_allocateClassPair(objc_getClass("NSObject"),
                                       "MGLPipelineCache", 0);
    if (!cls) return;

    /* The ivar order is the interface's: MGLPipelineCache.h. */
    class_addIvar(cls, "_state", sizeof(MGLPipelineCacheState),
                  alignof(MGLPipelineCacheState), "?");
    class_addIvar(cls, "_cacheDevice", sizeof(void *), alignof(void *), "^v");
    class_addIvar(cls, "_owner", sizeof(void *), alignof(void *), "^v");
    class_addIvar(cls, "_binaryArchiveRequested", sizeof(MGLPcBool),
                  alignof(MGLPcBool), "c");

    class_addMethod(cls,
                    sel_registerName("initWithPSODedupEnabled:"
                                     "depthStencilCacheEnabled:"
                                     "binaryArchiveEnabled:"),
                    (IMP)mglPcInit, "@@:ccc");
    class_addMethod(cls, sel_registerName("state"), (IMP)mglPcGetState, "^?@:");
    class_addMethod(cls, sel_registerName("ensureOwnerCreated"),
                    (IMP)mglPcEnsureOwnerCreated, "c@:");
    class_addMethod(cls, sel_registerName("ensureOwner"), (IMP)mglPcEnsureOwner,
                    "c@:");
    class_addMethod(cls, sel_registerName("isBinaryArchiveEnabled"),
                    (IMP)mglPcIsBinaryArchiveEnabled, "c@:");
    class_addMethod(cls, sel_registerName("device"), (IMP)mglPcGetDevice, "@@:");
    class_addMethod(cls, sel_registerName("setDevice:"), (IMP)mglPcSetDevice,
                    "v@:@");
    class_addMethod(cls, sel_registerName("depthStencilStateForValueState:"),
                    (IMP)mglPcDepthStencilStateForValueState, "@@:^?");
    class_addMethod(cls,
                    sel_registerName("lookupPipelineForWords:pipeline:"
                                     "vertexFunction:fragmentFunction:"),
                    (IMP)mglPcLookupPipelineForWords, "c@:^Q^@^@^@");
    class_addMethod(cls,
                    sel_registerName("storePipeline:vertexFunction:"
                                     "fragmentFunction:forWords:"),
                    (IMP)mglPcStorePipeline, "Q@:@@@^Q");
    class_addMethod(cls, sel_registerName("pipelineDescriptorStateForWords:state:"),
                    (IMP)mglPcPipelineDescriptorStateForWords, "c@:^Q^?");
    class_addMethod(cls,
                    sel_registerName("storePipelineDescriptorState:forWords:"),
                    (IMP)mglPcStorePipelineDescriptorState, "v@:^?^Q");
    class_addMethod(cls, sel_registerName("blendStateForAttachment:out:"),
                    (IMP)mglPcBlendStateForAttachment, "c@:Q^?");
    class_addMethod(cls, sel_registerName("binaryArchivePath"),
                    (IMP)mglPcBinaryArchivePath, "@@:");
    class_addMethod(cls, sel_registerName("binaryArchiveURL"),
                    (IMP)mglPcBinaryArchiveURL, "@@:");
    class_addMethod(cls, sel_registerName("loadBinaryArchive"),
                    (IMP)mglPcLoadBinaryArchive, "v@:");
    class_addMethod(cls, sel_registerName("saveBinaryArchive"),
                    (IMP)mglPcSaveBinaryArchive, "v@:");
    class_addMethod(cls,
                    sel_registerName("createRenderPipelineFromState:"
                                     "vertexFunction:fragmentFunction:"
                                     "pipelineOut:errorMessage:errorCapacity:"),
                    (IMP)mglPcCreateRenderPipelineFromState, "i@:^?^^^@^**");
    class_addMethod(cls, sel_registerName("invalidatePipelineState"),
                    (IMP)mglPcInvalidatePipelineState, "v@:");
    class_addMethod(cls, sel_registerName("setPipelineState:"),
                    (IMP)mglPcSetPipelineState, "v@:@");
    class_addMethod(cls,
                    sel_registerName("activatePipelineState:color0Format:"
                                     "depthFormat:stencilFormat:programName:"
                                     "vertexFunction:fragmentFunction:"),
                    (IMP)mglPcActivatePipelineState, "v@:@IIII@@");
    class_addMethod(cls,
                    sel_registerName("setBlendFactorsForAttachment:"
                                     "srcRgbFactor:srcAlphaFactor:"
                                     "dstRgbFactor:dstAlphaFactor:"
                                     "rgbOperation:alphaOperation:colorMask:"),
                    (IMP)mglPcSetBlendFactorsForAttachment, "v@:QIIIIIII");
    class_addMethod(cls, sel_registerName("disableBinaryArchive"),
                    (IMP)mglPcDisableBinaryArchive, "v@:");
    class_addMethod(cls, sel_registerName("resetCaches"), (IMP)mglPcResetCaches,
                    "v@:");
    class_addMethod(cls, sel_registerName("shutdown"), (IMP)mglPcShutdown, "v@:");
    class_addMethod(cls, sel_registerName("dealloc"), (IMP)mglPcDealloc, "v@:");

    objc_registerClassPair(cls);
    mglPcClass = cls;
    mglPcOffState = ivar_getOffset(class_getInstanceVariable(cls, "_state"));
    mglPcOffDevice =
        ivar_getOffset(class_getInstanceVariable(cls, "_cacheDevice"));
    mglPcOffOwner = ivar_getOffset(class_getInstanceVariable(cls, "_owner"));
    mglPcOffArchiveRequested = ivar_getOffset(
        class_getInstanceVariable(cls, "_binaryArchiveRequested"));

    mglPcSelInit = sel_registerName("initWithPSODedupEnabled:"
                                    "depthStencilCacheEnabled:"
                                    "binaryArchiveEnabled:");
    mglPcSelEnsureOwner = sel_registerName("ensureOwner");
    mglPcSelEnsureOwnerCreated = sel_registerName("ensureOwnerCreated");
    mglPcSelIsBinaryArchiveEnabled = sel_registerName("isBinaryArchiveEnabled");
    mglPcSelBinaryArchivePath = sel_registerName("binaryArchivePath");
    mglPcSelBinaryArchiveURL = sel_registerName("binaryArchiveURL");
    mglPcSelState = sel_registerName("state");
}

#endif /* MGL_PLATFORM_SHELL_SMOKE */
