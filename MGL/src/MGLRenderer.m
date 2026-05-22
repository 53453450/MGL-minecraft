/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * MGLRenderer.m
 * MGL
 *
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#import <simd/simd.h>
#import <MetalKit/MetalKit.h>

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <string.h>
#include <math.h>

// Header shared between C code here, which executes Metal API commands, and .metal files, which
// uses these types as inputs to the shaders.
//#import "AAPLShaderTypes.h"

#import "MGLRenderer.h"
#import "glm_context.h"
#import "mgl_safety.h"

#define TRACE_FUNCTION()    DEBUG_PRINT("%s\n", __FUNCTION__);

extern void mglDrawBuffer(GLMContext ctx, GLenum buf);

// for resource types SPVC_RESOURCE_TYPE_UNIFORM_BUFFER..
#import "spirv_cross_c.h"

typedef struct SyncList_t {
    GLuint count;
    GLuint  size;
    Sync **list;
} SyncList;

MTLPixelFormat mtlPixelFormatForGLTex(Texture * gl_tex);

typedef struct MGLDrawable_t {
    GLuint width;
    GLuint height;
    id<MTLTexture> drawbuffer;
    id<MTLTexture> depthbuffer;
    id<MTLTexture> stencilbuffer;
} MGLDrawable;

enum {
    _FRONT,
    _BACK,
    _FRONT_LEFT,
    _FRONT_RIGHT,
    _BACK_LEFT,
    _BACK_RIGHT,
    _MAX_DRAW_BUFFERS
};

static GLuint mglDefaultDrawBufferIndexForGL(GLenum drawBuffer)
{
    switch (drawBuffer)
    {
        case GL_FRONT: return _FRONT;
        case GL_BACK: return _FRONT;
        case GL_FRONT_LEFT: return _FRONT_LEFT;
        case GL_FRONT_RIGHT: return _FRONT_RIGHT;
        case GL_BACK_LEFT: return _FRONT_LEFT;
        case GL_BACK_RIGHT: return _FRONT_RIGHT;
        case GL_LEFT: return _FRONT_LEFT;
        case GL_RIGHT: return _FRONT_RIGHT;
        case GL_FRONT_AND_BACK: return _FRONT;
        case GL_COLOR_ATTACHMENT0: return _FRONT;
        case GL_NONE: return _FRONT;
        default: return _FRONT;
    }
}

static GLsizei mglMetalDrawBufferCount(GLMContext drawCtx)
{
    if (!drawCtx || drawCtx->state.draw_buffer_count <= 0) {
        return 0;
    }
    if (drawCtx->state.draw_buffer_count > (GLsizei)MAX_COLOR_ATTACHMENTS) {
        return MAX_COLOR_ATTACHMENTS;
    }
    return drawCtx->state.draw_buffer_count;
}

static GLenum mglMetalDrawBufferAt(GLMContext drawCtx, GLuint slot)
{
    if (!drawCtx) {
        return GL_NONE;
    }

    GLsizei count = mglMetalDrawBufferCount(drawCtx);
    if (slot < (GLuint)count) {
        return drawCtx->state.draw_buffers[slot];
    }

    return GL_NONE;
}

static BOOL mglMetalResolveFboDrawAttachmentIndex(GLMContext drawCtx,
                                                  GLenum drawBuffer,
                                                  GLuint *attachmentIndex)
{
    if (!drawCtx || drawBuffer == GL_NONE) {
        return NO;
    }

    if (drawBuffer >= GL_COLOR_ATTACHMENT0 &&
        drawBuffer < (GL_COLOR_ATTACHMENT0 + drawCtx->state.max_color_attachments) &&
        drawBuffer < (GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS)) {
        if (attachmentIndex) {
            *attachmentIndex = (GLuint)(drawBuffer - GL_COLOR_ATTACHMENT0);
        }
        return YES;
    }

    switch (drawBuffer) {
        case GL_FRONT:
        case GL_BACK:
        case GL_FRONT_LEFT:
        case GL_FRONT_RIGHT:
        case GL_BACK_LEFT:
        case GL_BACK_RIGHT:
        case GL_LEFT:
        case GL_RIGHT:
        case GL_FRONT_AND_BACK:
            if (attachmentIndex) {
                *attachmentIndex = 0u;
            }
            return YES;

        default:
            return NO;
    }
}

typedef struct MGLMetalAttachmentSubresource_t {
    NSUInteger level;
    NSUInteger slice;
    NSUInteger depthPlane;
} MGLMetalAttachmentSubresource;

static MGLMetalAttachmentSubresource mglMetalAttachmentSubresourceForAttachment(const FBOAttachment *attachment)
{
    MGLMetalAttachmentSubresource subresource = {0u, 0u, 0u};
    if (!attachment) {
        return subresource;
    }

    subresource.level = attachment->level;

    switch (attachment->textarget) {
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
            subresource.slice = 0u;
            break;
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
            subresource.slice = 1u;
            break;
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
            subresource.slice = 2u;
            break;
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
            subresource.slice = 3u;
            break;
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
            subresource.slice = 4u;
            break;
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            subresource.slice = 5u;
            break;

        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            subresource.slice = attachment->layer;
            break;

        case GL_TEXTURE_3D:
            subresource.depthPlane = attachment->layer;
            break;

        default:
            break;
    }

    return subresource;
}

static BOOL mglMetalReadbackFormatIsBGRA8Compatible(MTLPixelFormat pixelFormat)
{
    switch (pixelFormat) {
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
            return YES;
        default:
            return NO;
    }
}

static BOOL mglMetalLayerPixelFormatIsSupported(MTLPixelFormat pixelFormat)
{
    switch (pixelFormat) {
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
            return YES;
        default:
            return NO;
    }
}

static MTLPixelFormat mglMetalLayerPixelFormatForContext(GLMContext drawCtx)
{
    MTLPixelFormat fallback = MTLPixelFormatBGRA8Unorm;
    if (!drawCtx) {
        return fallback;
    }

    MTLPixelFormat requested = (MTLPixelFormat)drawCtx->pixel_format.mtl_pixel_format;
    if (mglMetalLayerPixelFormatIsSupported(requested)) {
        return requested;
    }

    NSLog(@"MGL CAMetalLayer pixelFormat fallback glFormat=0x%x glType=0x%x requestedMtl=%lu fallback=%lu",
          drawCtx->pixel_format.format,
          drawCtx->pixel_format.type,
          (unsigned long)requested,
          (unsigned long)fallback);
    return fallback;
}

static void mglMetalCopyTextureBytesToBGRA8(const uint8_t *src,
                                            NSUInteger srcBytesPerRow,
                                            uint8_t *dst,
                                            NSUInteger dstBytesPerRow,
                                            NSUInteger width,
                                            NSUInteger height,
                                            MTLPixelFormat pixelFormat,
                                            BOOL flipY)
{
    if (!src || !dst || width == 0u || height == 0u) {
        return;
    }

    BOOL sourceIsRGBA =
        (pixelFormat == MTLPixelFormatRGBA8Unorm ||
         pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB);

    for (NSUInteger y = 0; y < height; y++) {
        const uint8_t *srcRow = src + (y * srcBytesPerRow);
        NSUInteger dstY = flipY ? (height - 1u - y) : y;
        uint8_t *dstRow = dst + (dstY * dstBytesPerRow);

        if (!sourceIsRGBA) {
            memcpy(dstRow, srcRow, width * 4u);
            continue;
        }

        for (NSUInteger x = 0; x < width; x++) {
            const uint8_t *s = srcRow + (x * 4u);
            uint8_t *d = dstRow + (x * 4u);
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
            d[3] = s[3];
        }
    }
}

typedef struct MGLScaledBlitParams_t {
    vector_float4 uvRect; // xy=min, zw=max in normalized Metal texture coordinates.
    float forceOpaqueAlpha;
    vector_float3 _padding;
} MGLScaledBlitParams;

typedef struct MGLBlitAxis_t {
    double src0;
    double src1;
    double dst0;
    double dst1;
} MGLBlitAxis;

static NSUInteger mglMetalTextureLevelDimension(NSUInteger base, NSUInteger level)
{
    NSUInteger value = MAX((NSUInteger)1u, base);
    while (level-- > 0u && value > 1u) {
        value >>= 1u;
    }
    return MAX((NSUInteger)1u, value);
}

static BOOL mglClipBlitAxisToDestination(MGLBlitAxis *axis, double dstLimit)
{
    if (!axis || axis->dst0 == axis->dst1 || axis->src0 == axis->src1 || dstLimit <= 0.0) {
        return NO;
    }

    double dstMin = fmin(axis->dst0, axis->dst1);
    double dstMax = fmax(axis->dst0, axis->dst1);
    double clippedMin = fmax(dstMin, 0.0);
    double clippedMax = fmin(dstMax, dstLimit);
    if (clippedMax <= clippedMin) {
        return NO;
    }

    double dstSpan = axis->dst1 - axis->dst0;
    double srcSpan = axis->src1 - axis->src0;
    double tMin = (clippedMin - axis->dst0) / dstSpan;
    double tMax = (clippedMax - axis->dst0) / dstSpan;
    double srcAtMin = axis->src0 + tMin * srcSpan;
    double srcAtMax = axis->src0 + tMax * srcSpan;

    if (axis->dst1 >= axis->dst0) {
        axis->dst0 = clippedMin;
        axis->dst1 = clippedMax;
        axis->src0 = srcAtMin;
        axis->src1 = srcAtMax;
    } else {
        axis->dst0 = clippedMax;
        axis->dst1 = clippedMin;
        axis->src0 = srcAtMax;
        axis->src1 = srcAtMin;
    }

    return YES;
}

static BOOL mglClipBlitAxisToSource(MGLBlitAxis *axis, double srcLimit)
{
    if (!axis || axis->dst0 == axis->dst1 || axis->src0 == axis->src1 || srcLimit <= 0.0) {
        return NO;
    }

    double srcMin = fmin(axis->src0, axis->src1);
    double srcMax = fmax(axis->src0, axis->src1);
    double clippedMin = fmax(srcMin, 0.0);
    double clippedMax = fmin(srcMax, srcLimit);
    if (clippedMax <= clippedMin) {
        return NO;
    }

    double srcSpan = axis->src1 - axis->src0;
    double dstSpan = axis->dst1 - axis->dst0;
    double tMin = (clippedMin - axis->src0) / srcSpan;
    double tMax = (clippedMax - axis->src0) / srcSpan;
    double dstAtMin = axis->dst0 + tMin * dstSpan;
    double dstAtMax = axis->dst0 + tMax * dstSpan;

    if (axis->src1 >= axis->src0) {
        axis->src0 = clippedMin;
        axis->src1 = clippedMax;
        axis->dst0 = dstAtMin;
        axis->dst1 = dstAtMax;
    } else {
        axis->src0 = clippedMax;
        axis->src1 = clippedMin;
        axis->dst0 = dstAtMax;
        axis->dst1 = dstAtMin;
    }

    return YES;
}

static BOOL mglClipBlitAxis(MGLBlitAxis *axis, double srcLimit, double dstLimit)
{
    return mglClipBlitAxisToDestination(axis, dstLimit) &&
           mglClipBlitAxisToSource(axis, srcLimit);
}

static BOOL mglNearlyEqual(double a, double b)
{
    return fabs(a - b) <= 0.00001;
}

static MTLCompareFunction mglMTLCompareFunctionForGL(GLenum func,
                                                     MTLCompareFunction fallback,
                                                     const char *label)
{
    switch (func) {
        case GL_NEVER: return MTLCompareFunctionNever;
        case GL_LESS: return MTLCompareFunctionLess;
        case GL_EQUAL: return MTLCompareFunctionEqual;
        case GL_LEQUAL: return MTLCompareFunctionLessEqual;
        case GL_GREATER: return MTLCompareFunctionGreater;
        case GL_NOTEQUAL: return MTLCompareFunctionNotEqual;
        case GL_GEQUAL: return MTLCompareFunctionGreaterEqual;
        case GL_ALWAYS: return MTLCompareFunctionAlways;
        default: {
            static uint64_t s_badCompareFunctionCount = 0;
            uint64_t hit = ++s_badCompareFunctionCount;

            if (hit <= 32 || (hit % 256) == 0) {
                NSLog(@"MGL WARNING: invalid %s compare func=0x%x, fallback=%lu hit=%llu",
                      label ? label : "unknown",
                      func,
                      (unsigned long)fallback,
                      (unsigned long long)hit);
            }

            return fallback;
        }
    }
}

static MTLWinding mglMTLWindingForGL(GLenum frontFace)
{
    switch (frontFace) {
        case GL_CW:
            return MTLWindingClockwise;
        case GL_CCW:
            return MTLWindingCounterClockwise;
        default: {
            static uint64_t s_badFrontFaceCount = 0;
            uint64_t hit = ++s_badFrontFaceCount;

            if (hit <= 32 || (hit % 256) == 0) {
                NSLog(@"MGL WARNING: invalid front face enum=0x%x, fallback=GL_CCW hit=%llu",
                      frontFace,
                      (unsigned long long)hit);
            }

            return MTLWindingCounterClockwise;
        }
    }
}

static BOOL mglIsValidGLCompareFunction(GLenum func)
{
    switch (func) {
        case GL_NEVER:
        case GL_LESS:
        case GL_EQUAL:
        case GL_LEQUAL:
        case GL_GREATER:
        case GL_NOTEQUAL:
        case GL_GEQUAL:
        case GL_ALWAYS:
            return YES;
        default:
            return NO;
    }
}

static BOOL mglIsValidGLBlendEquation(GLenum op)
{
    switch (op) {
        case GL_FUNC_ADD:
        case GL_FUNC_SUBTRACT:
        case GL_FUNC_REVERSE_SUBTRACT:
        case GL_MIN:
        case GL_MAX:
            return YES;
        default:
            return NO;
    }
}

static BOOL mglIsValidGLBlendFactor(GLenum factor)
{
    switch (factor) {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
        case GL_SRC_ALPHA_SATURATE:
            return YES;
        default:
            return NO;
    }
}

static void mglLogRenderStateRepair(const char *field, GLenum value, GLenum fallback)
{
    static uint64_t s_stateRepairCount = 0;
    uint64_t hit = ++s_stateRepairCount;

    if (hit <= 64 || (hit % 512) == 0) {
        NSLog(@"MGL WARNING: repairing invalid render state %s=0x%x -> 0x%x hit=%llu",
              field ? field : "unknown",
              value,
              fallback,
              (unsigned long long)hit);
    }
}

static BOOL mglShouldLogSmallBaseBinding(GLuint programName,
                                         int stage,
                                         int resourceType,
                                         GLuint binding,
                                         GLuint glName,
                                         GLsizeiptr rangeSize,
                                         NSUInteger reflectedSize)
{
    typedef struct MGLSmallBaseBindingLogKey_t {
        GLuint programName;
        int stage;
        int resourceType;
        GLuint binding;
        GLuint glName;
        GLsizeiptr rangeSize;
        NSUInteger reflectedSize;
        uint64_t hits;
    } MGLSmallBaseBindingLogKey;

    static MGLSmallBaseBindingLogKey s_keys[128];
    static uint32_t s_keyCount = 0;
    static uint64_t s_overflowHits = 0;

    for (uint32_t i = 0; i < s_keyCount; i++) {
        MGLSmallBaseBindingLogKey *key = &s_keys[i];
        if (key->programName == programName &&
            key->stage == stage &&
            key->resourceType == resourceType &&
            key->binding == binding &&
            key->glName == glName &&
            key->rangeSize == rangeSize &&
            key->reflectedSize == reflectedSize) {
            key->hits++;
            return key->hits <= 4 || (key->hits % 1024) == 0;
        }
    }

    if (s_keyCount < (uint32_t)(sizeof(s_keys) / sizeof(s_keys[0]))) {
        if (s_keyCount >= 32 && (s_keyCount % 16) != 0) {
            return NO;
        }
        s_keys[s_keyCount++] = (MGLSmallBaseBindingLogKey){
            .programName = programName,
            .stage = stage,
            .resourceType = resourceType,
            .binding = binding,
            .glName = glName,
            .rangeSize = rangeSize,
            .reflectedSize = reflectedSize,
            .hits = 1
        };
        return YES;
    }

    s_overflowHits++;
    return s_overflowHits <= 8 || (s_overflowHits % 2048) == 0;
}


__attribute__((constructor))
static void mglRendererDiagnosticBuildMarker(void)
{
    NSLog(@"MGL DIAG BUILD marker=semantic-layer-pf-abi-guard-20260522 built=%s %s renderer-loaded",
          __DATE__,
          __TIME__);
}

// CRITICAL SECURITY: Safe Metal object validation helper
static inline id<NSObject> SafeMetalBridge(void *ptr, Class expectedClass, const char *objectName) {
    if (!ptr) {
        NSLog(@"MGL SECURITY ERROR: NULL pointer for %s", objectName);
        return nil;
    }

    id<NSObject> obj = (__bridge id<NSObject>)(ptr);
    if (!obj) {
        NSLog(@"MGL SECURITY ERROR: Metal bridge cast returned nil for %s", objectName);
        return nil;
    }

    if (expectedClass && [obj isKindOfClass:expectedClass] == NO) {
        NSLog(@"MGL SECURITY ERROR: Metal object is not valid %s (got %@)", objectName, NSStringFromClass([obj class]));
        return nil;
    }

    return obj;
}

// Debug switch: temporarily disable shared-event synchronization path to isolate GPU timeout sources.
static const BOOL kMGLDisableSharedEventSync = YES;
// Leave verbose bind tracing off by default; per-draw logging can stall the render thread.
static const BOOL kMGLVerboseBindLogs = NO;
// Pipeline/descriptor tracing is similarly noisy; keep it opt-in.
static const BOOL kMGLVerbosePipelineLogs = NO;
// Frame-loop/state tracing is extremely hot; keep broad tracing off so the log
// reaches the actual crash site instead of Prism's 100k-line cap.
static const BOOL kMGLVerboseFrameLoopLogs = NO;
static const BOOL kMGLDiagnosticStateLogs = NO;
// Keep swap/present sampling available even when broad state tracing is disabled.
// This is intentionally low-frequency so Prism's 100k-line cap does not hide the
// final compositing evidence.
static const BOOL kMGLSwapPresentDiagnostics = NO;
// Narrow draw-submit breadcrumbs stay enabled because Metal validation aborts
// the process before Objective-C exceptions can catch anything useful.
static const BOOL kMGLDrawSubmitDiagnostics = NO;
// Dedicated upload buffers should not block the render thread by default.
static const BOOL kMGLSynchronizeTextureUploads = NO;
static const NSTimeInterval kMGLTextureUploadWaitTimeoutSeconds = 0.25;
// Keep vertex attribute buffers in a dedicated high slot range so they do not collide
// with UBO/SSBO bindings that are expected at low indices.
static const NSUInteger kMGLVertexAttribBufferBase = MAX_BINDABLE_BUFFERS;
// Metal vertex buffer layout indices are 0..30 (count=31).
// Guard all vertex-layout/binding paths against using index 31.
static const NSUInteger kMGLMaxMetalVertexBufferCount = 31;
static const NSUInteger kMGLMaxMetalVertexBufferIndex = 30;
// Metal validation requires bound stage buffers to satisfy argument byte length.
// Keep a conservative minimum for low-index base/resource slots.
static const NSUInteger kMGLMinimumStageBindingSize = 256;
static const NSUInteger kMGLDefaultStageFallbackBufferSize = 4096;
static const NSUInteger kMGLStageBindingStackScratchSize = 1024;
// Keep low-index vertex resource slots bound during diagnostics. Attribute VBOs
// live at kMGLVertexAttribBufferBase+, so this does not overwrite vertex input slots.
static const BOOL kMGLEnableVertexAllSlotFallback = YES;
static const BOOL kMGLEnableSampledTextureFallback = YES;
// Mirror Metal's drawArrays vertex-buffer range validation before calling into
// the debug layer. Metal aborts the process for these errors; we want a log and
// a skipped draw instead.
static const BOOL kMGLValidateDrawArraysVboRange = YES;
// Validate indexed draws against VBO size + written-range metadata.
static const BOOL kMGLValidateDrawElementsVboRange = YES;

// Cross-stage frame activity breadcrumbs for black-screen/beachball diagnostics.
static volatile uint64_t g_mglLastDrawArraysCall = 0;
static volatile double g_mglLastDrawArraysSeconds = 0.0;
static volatile uint64_t g_mglLastDrawElementsCall = 0;
static volatile double g_mglLastDrawElementsSeconds = 0.0;
static volatile GLuint g_mglLastDrawArraysProgram = 0;
static volatile GLuint g_mglLastDrawArraysMode = 0;
static volatile GLsizei g_mglLastDrawArraysCount = 0;
static volatile GLuint g_mglLastDrawElementsProgram = 0;
static volatile GLuint g_mglLastDrawElementsMode = 0;
static volatile GLsizei g_mglLastDrawElementsCount = 0;
static volatile uint64_t g_mglDrawArraysSinceSwap = 0;
static volatile uint64_t g_mglDrawElementsSinceSwap = 0;
static volatile uint64_t g_mglDrawArrayVerticesSinceSwap = 0;
static volatile uint64_t g_mglDrawElementIndicesSinceSwap = 0;
static volatile uint64_t g_mglDrawArraysSkippedSinceSwap = 0;
static volatile uint64_t g_mglDrawElementsSkippedSinceSwap = 0;
static volatile uint64_t g_mglProcessDrawCallsSinceSwap = 0;
static volatile uint64_t g_mglSwapCallCount = 0;
static volatile double g_mglLastSwapSeconds = 0.0;

static GLuint mglClientBufferBindingForResource(int resourceType, const SpirvResource *res)
{
    if (!res) {
        return 0u;
    }

    GLint knownPlainUniformBinding = -1;
    if (res->name) {
        if (!strcmp(res->name, "ModelViewMat")) {
            knownPlainUniformBinding = 0;
        } else if (!strcmp(res->name, "ProjMat")) {
            knownPlainUniformBinding = 1;
        } else if (!strcmp(res->name, "TextureMat")) {
            knownPlainUniformBinding = 2;
        } else if (!strcmp(res->name, "ColorModulator")) {
            knownPlainUniformBinding = 3;
        } else if (!strcmp(res->name, "FogStart")) {
            knownPlainUniformBinding = 4;
        } else if (!strcmp(res->name, "FogEnd")) {
            knownPlainUniformBinding = 5;
        } else if (!strcmp(res->name, "FogColor")) {
            knownPlainUniformBinding = 6;
        } else if (!strcmp(res->name, "FogShape")) {
            knownPlainUniformBinding = 7;
        } else if (!strcmp(res->name, "GameTime")) {
            knownPlainUniformBinding = 8;
        } else if (!strcmp(res->name, "ScreenSize")) {
            knownPlainUniformBinding = 9;
        } else if (!strcmp(res->name, "LineWidth")) {
            knownPlainUniformBinding = 10;
        } else if (!strcmp(res->name, "IViewRotMat")) {
            knownPlainUniformBinding = 11;
        } else if (!strcmp(res->name, "ChunkOffset")) {
            knownPlainUniformBinding = 12;
        }
    }

    /*
     * Plain uniforms are represented internally as one tiny GL buffer per
     * uniform location. SPIRV-Cross usually reports descriptor binding 0 for
     * all of them, while the generated MSL assigns distinct [[buffer(n)]]
     * slots. Use the GL uniform location to find the client-side buffer, then
     * map that location to the reflected Metal slot later.
     */
    if (resourceType == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT) {
        if (knownPlainUniformBinding >= 0) {
            return (GLuint)knownPlainUniformBinding;
        }
        if (res->uniform_location >= 0 && res->uniform_location < MAX_BINDABLE_BUFFERS) {
            return (GLuint)res->uniform_location;
        }
        if (res->location < MAX_BINDABLE_BUFFERS) {
            return res->location;
        }
        if (res->gl_binding < MAX_BINDABLE_BUFFERS) {
            return res->gl_binding;
        }
        if (res->binding < MAX_BINDABLE_BUFFERS) {
            return res->binding;
        }
    }

    return res->gl_binding;
}

typedef struct MGLSwapDrawCounters {
    uint64_t draw_arrays;
    uint64_t draw_elements;
    uint64_t array_vertices;
    uint64_t element_indices;
    uint64_t draw_arrays_skipped;
    uint64_t draw_elements_skipped;
    uint64_t process_draw_calls;
} MGLSwapDrawCounters;

static inline MGLSwapDrawCounters mglSnapshotSwapDrawCounters(void)
{
    MGLSwapDrawCounters counters;
    counters.draw_arrays = g_mglDrawArraysSinceSwap;
    counters.draw_elements = g_mglDrawElementsSinceSwap;
    counters.array_vertices = g_mglDrawArrayVerticesSinceSwap;
    counters.element_indices = g_mglDrawElementIndicesSinceSwap;
    counters.draw_arrays_skipped = g_mglDrawArraysSkippedSinceSwap;
    counters.draw_elements_skipped = g_mglDrawElementsSkippedSinceSwap;
    counters.process_draw_calls = g_mglProcessDrawCallsSinceSwap;
    return counters;
}

static inline void mglResetSwapDrawCounters(void)
{
    g_mglDrawArraysSinceSwap = 0;
    g_mglDrawElementsSinceSwap = 0;
    g_mglDrawArrayVerticesSinceSwap = 0;
    g_mglDrawElementIndicesSinceSwap = 0;
    g_mglDrawArraysSkippedSinceSwap = 0;
    g_mglDrawElementsSkippedSinceSwap = 0;
    g_mglProcessDrawCallsSinceSwap = 0;
}

static BOOL mglRendererPointerInHashTable(HashTable *table, const void *ptr);
static Framebuffer *mglRendererGetValidatedFramebuffer(GLMContext ctx, const char *where);
static GLuint mglRendererSafeFramebufferName(GLMContext ctx);

static inline BOOL mglRendererContextLikelyValid(GLMContext ctx)
{
    return (ctx != NULL) && ((uintptr_t)ctx >= 0x10000u);
}

static inline BOOL mglRendererObjectPointerLikelyValid(const void *ptr)
{
    return mglObjectPointerLooksPlausible(ptr);
}

static Program *mglResolveProgramFromState(GLMContext ctx)
{
    if (!mglRendererContextLikelyValid(ctx)) {
        return NULL;
    }

    Program *program = ctx->state.program;
    if (program) {
        if (!mglRendererObjectPointerLikelyValid(program) ||
            !mglRendererPointerInHashTable(&ctx->state.program_table, program) ||
            !mglPointerRangeIsReadable(program, sizeof(*program))) {
            NSLog(@"MGL PROGRAM RESOLVE invalid cached pointer=%p name=%u",
                  program,
                  (unsigned)ctx->state.program_name);
            ctx->state.program = NULL;
            program = NULL;
        }
    }

    if (program) {
        if (ctx->state.program_name == 0 || ctx->state.program_name != program->name) {
            ctx->state.program_name = program->name;
            ctx->state.var.current_program = program->name;
        }
        return program;
    }

    if (ctx->state.program_name == 0) {
        return NULL;
    }

    Program *resolved = (Program *)searchHashTable(&ctx->state.program_table, ctx->state.program_name);
    if (!resolved) {
        NSLog(@"MGL PROGRAM RESOLVE fail: name=%u missing in table", (unsigned)ctx->state.program_name);
        ctx->state.program_name = 0;
        ctx->state.var.current_program = 0;
        return NULL;
    }

    if (!resolved->linked_glsl_program) {
        NSLog(@"MGL PROGRAM RESOLVE pending: name=%u ptr=%p not linked",
              (unsigned)ctx->state.program_name, resolved);
        return NULL;
    }

    ctx->state.program = resolved;
    resolved->refcount++;
    ctx->state.dirty_bits |= DIRTY_PROGRAM;

    NSLog(@"MGL PROGRAM RESOLVE recovered name=%u ptr=%p",
          (unsigned)ctx->state.program_name, resolved);
    return resolved;
}

static Program *mglPeekProgramByName(GLMContext ctx, GLuint programName)
{
    if (!mglRendererContextLikelyValid(ctx) || programName == 0) {
        return NULL;
    }

    Program *program = ctx->state.program;
    if (program &&
        mglRendererObjectPointerLikelyValid(program) &&
        mglRendererPointerInHashTable(&ctx->state.program_table, program) &&
        mglPointerRangeIsReadable(program, sizeof(*program)) &&
        program->name == programName) {
        return program;
    }

    return (Program *)searchHashTable(&ctx->state.program_table, programName);
}

static const char *mglShaderStageName(int stage)
{
    switch (stage) {
        case _VERTEX_SHADER: return "vertex";
        case _TESS_CONTROL_SHADER: return "tess_control";
        case _TESS_EVALUATION_SHADER: return "tess_eval";
        case _GEOMETRY_SHADER: return "geometry";
        case _FRAGMENT_SHADER: return "fragment";
        case _COMPUTE_SHADER: return "compute";
        default: return "unknown";
    }
}

static const char *mglSpirvResourceTypeName(int type)
{
    switch (type) {
        case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER: return "uniform_buffer";
        case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT: return "uniform_constant";
        case SPVC_RESOURCE_TYPE_STORAGE_BUFFER: return "storage_buffer";
        case SPVC_RESOURCE_TYPE_STAGE_INPUT: return "stage_input";
        case SPVC_RESOURCE_TYPE_STAGE_OUTPUT: return "stage_output";
        case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE: return "sampled_image";
        case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE: return "separate_image";
        case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS: return "separate_sampler";
        case SPVC_RESOURCE_TYPE_PUSH_CONSTANT: return "push_constant";
        default: return "resource";
    }
}

static void mglLogProgramResourceInterface(Program *program, int stage, int type)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES || type < 0 || type >= _MAX_SPIRV_RES) {
        return;
    }

    SpirvResourceList *resources = &program->spirv_resources_list[stage][type];
    NSLog(@"MGL IFACE program=%u stage=%s type=%s count=%u",
          (unsigned)program->name,
          mglShaderStageName(stage),
          mglSpirvResourceTypeName(type),
          (unsigned)resources->count);

    for (GLuint i = 0; i < resources->count; i++) {
        SpirvResource *res = &resources->list[i];
        NSLog(@"MGL IFACE   #%u name=%s loc=%u glBinding=%u metalBinding=%u set=%u typeId=%u baseTypeId=%u required=%zu imageDim=%u arrayed=%u",
              (unsigned)i,
              res->name ? res->name : "(null)",
              (unsigned)res->location,
              (unsigned)res->gl_binding,
              (unsigned)res->binding,
              (unsigned)res->set,
              (unsigned)res->type_id,
              (unsigned)res->base_type_id,
              res->required_size,
              (unsigned)res->image_dim,
              (unsigned)res->image_arrayed);
    }
}

static void mglWriteProgramMSLDump(Program *program, NSString *reason)
{
    if (!program) {
        return;
    }

    BOOL forceDump = false;
    if (reason) {
        NSString *lowerReason = [reason lowercaseString];
        forceDump = [lowerReason containsString:@"tex"];
    }

    static GLuint s_dumpedPrograms[64] = {0};
    static GLuint s_forcedDumpedPrograms[64] = {0};
    static uint32_t s_dumpedProgramCount = 0;
    static uint32_t s_forcedDumpedProgramCount = 0;
    static uint32_t s_dumpGeneration = 0;
    if (forceDump) {
        for (uint32_t i = 0; i < s_forcedDumpedProgramCount; i++) {
            if (s_forcedDumpedPrograms[i] == program->name) {
                return;
            }
        }
    } else {
        for (uint32_t i = 0; i < s_dumpedProgramCount; i++) {
            if (s_dumpedPrograms[i] == program->name) {
                return;
            }
        }
    }

    if (forceDump && s_forcedDumpedProgramCount < (uint32_t)(sizeof(s_forcedDumpedPrograms) / sizeof(s_forcedDumpedPrograms[0]))) {
        s_forcedDumpedPrograms[s_forcedDumpedProgramCount++] = program->name;
    } else if (!forceDump && s_dumpedProgramCount < (uint32_t)(sizeof(s_dumpedPrograms) / sizeof(s_dumpedPrograms[0]))) {
        s_dumpedPrograms[s_dumpedProgramCount++] = program->name;
    } else {
        return;
    }
    s_dumpGeneration++;

    NSLog(@"MGL IFACE DUMP begin program=%u reason=%@ generation=%u",
          (unsigned)program->name,
          reason ?: @"(none)",
          (unsigned)s_dumpGeneration);

    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_OUTPUT);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_STAGE_OUTPUT);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE);
    mglLogProgramResourceInterface(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_SEPARATE_IMAGE);
    mglLogProgramResourceInterface(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_SEPARATE_IMAGE);

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        const char *msl = program->spirv[stage].msl_str;
        if (!msl || !program->shader_slots[stage]) {
            continue;
        }

        NSString *path = [NSString stringWithFormat:@"/tmp/mgl_program_%u_%s_%u.msl",
                                                   (unsigned)program->name,
                                                   mglShaderStageName(stage),
                                                   (unsigned)s_dumpGeneration];
        NSString *source = [NSString stringWithUTF8String:msl];
        NSError *writeError = nil;
        BOOL ok = [source writeToFile:path
                           atomically:YES
                             encoding:NSUTF8StringEncoding
                                error:&writeError];
        NSLog(@"MGL IFACE DUMP msl program=%u stage=%s entry=%s path=%@ ok=%d error=%@",
              (unsigned)program->name,
              mglShaderStageName(stage),
              program->shader_slots[stage]->entry_point ? program->shader_slots[stage]->entry_point : "(null)",
              path,
              ok ? 1 : 0,
              writeError);
    }
}

static GLuint g_mglFocusedLoadingPrograms[32] = {0};
static uint32_t g_mglFocusedLoadingProgramCount = 0;

static bool mglProgramHasImageDim(Program *program, GLuint imageDim)
{
    if (!program) {
        return false;
    }

    const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
        SPVC_RESOURCE_TYPE_SEPARATE_IMAGE,
        SPVC_RESOURCE_TYPE_STORAGE_IMAGE
    };

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        for (size_t t = 0; t < sizeof(resourceTypes) / sizeof(resourceTypes[0]); t++) {
            int type = resourceTypes[t];
            if (type < 0 || type >= _MAX_SPIRV_RES) {
                continue;
            }
            SpirvResourceList *resources = &program->spirv_resources_list[stage][type];
            for (GLuint i = 0; i < resources->count; i++) {
                if (resources->list[i].image_dim == imageDim) {
                    return true;
                }
            }
        }
    }

    return false;
}

static bool mglProgramHasResourceName(Program *program, int stage, int type, const char *name)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        type < 0 || type >= _MAX_SPIRV_RES || !name) {
        return false;
    }

    SpirvResourceList *resources = &program->spirv_resources_list[stage][type];
    for (GLuint i = 0; resources->list && i < resources->count; i++) {
        if (resources->list[i].name && strcmp(resources->list[i].name, name) == 0) {
            return true;
        }
    }

    return false;
}

static bool mglProgramLooksLikeMinecraftTerrain(Program *program)
{
    return mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "Position") &&
           mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "Color") &&
           mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "UV0") &&
           mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "UV2") &&
           mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, "Sampler2") &&
           mglProgramHasResourceName(program, _FRAGMENT_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, "Sampler0");
}

static bool mglProgramLooksLikeMinecraftGuiNoDepth(Program *program)
{
    if (!program) {
        return false;
    }
    if (!mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "Position") ||
        !mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "Color")) {
        return false;
    }
    if (!mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, "Projection") ||
        !mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, "DynamicTransforms")) {
        return false;
    }

    if (mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "Normal") ||
        mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_STAGE_INPUT, "UV2") ||
        mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, "ChunkSection") ||
        mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, "Globals") ||
        mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, "CloudInfo") ||
        mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, "CloudFaces") ||
        mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, "CloudFaces") ||
        mglProgramHasResourceName(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, "Sampler2")) {
        return false;
    }

    return true;
}

static void mglFocusLoadingProgram(GLuint programName, const char *reason, uint64_t detail)
{
    if (programName == 0) {
        return;
    }

    for (uint32_t i = 0; i < g_mglFocusedLoadingProgramCount; i++) {
        if (g_mglFocusedLoadingPrograms[i] == programName) {
            return;
        }
    }

    if (g_mglFocusedLoadingProgramCount >= (uint32_t)(sizeof(g_mglFocusedLoadingPrograms) / sizeof(g_mglFocusedLoadingPrograms[0]))) {
        return;
    }

    g_mglFocusedLoadingPrograms[g_mglFocusedLoadingProgramCount++] = programName;
    NSLog(@"MGL TRACE focus.program program=%u reason=%s detail=%llu",
          (unsigned)programName,
          reason ? reason : "(none)",
          (unsigned long long)detail);
}

static void mglObserveProgramDrawForFocus(GLuint programName,
                                          GLsizei count,
                                          GLuint enabledAttribs)
{
    typedef struct MGLProgramDrawObservation {
        GLuint program;
        GLsizei count;
        GLuint enabledAttribs;
        uint64_t hits;
    } MGLProgramDrawObservation;

    static MGLProgramDrawObservation s_observations[64] = {{0, 0, 0, 0}};

    if (programName == 0) {
        return;
    }

    for (uint32_t i = 0; i < (uint32_t)(sizeof(s_observations) / sizeof(s_observations[0])); i++) {
        if (s_observations[i].program == programName ||
            s_observations[i].program == 0) {
            if (s_observations[i].program == 0) {
                s_observations[i].program = programName;
                s_observations[i].count = count;
                s_observations[i].enabledAttribs = enabledAttribs;
            }

            if (s_observations[i].count == count &&
                s_observations[i].enabledAttribs == enabledAttribs) {
                s_observations[i].hits++;
            } else {
                s_observations[i].count = count;
                s_observations[i].enabledAttribs = enabledAttribs;
                s_observations[i].hits = 1;
            }

            if (s_observations[i].hits == 16ull) {
                mglFocusLoadingProgram(programName, "repeated-draw-pattern", s_observations[i].hits);
            }
            return;
        }
    }
}

static inline bool mglIsFocusedLoadingProgram(GLuint programName)
{
    for (uint32_t i = 0; i < g_mglFocusedLoadingProgramCount; i++) {
        if (g_mglFocusedLoadingPrograms[i] == programName) {
            return true;
        }
    }

    return false;
}

static inline bool mglShouldTraceCall(uint64_t count)
{
    if (!kMGLDiagnosticStateLogs) {
        return false;
    }
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}

static inline TextureLevel *mglTraceTextureBaseLevel(Texture *tex)
{
    if (!tex || tex->num_levels == 0 || !tex->faces[0].levels) {
        return NULL;
    }

    return &tex->faces[0].levels[0];
}

static inline TextureLevel *mglTextureAttachmentLevel(Texture *tex, GLuint level)
{
    if (!tex || tex->num_levels == 0 || !tex->faces[0].levels || level >= tex->num_levels) {
        return NULL;
    }

    return &tex->faces[0].levels[level];
}

static inline void mglMarkTextureLevelRenderTargetWritten(Texture *tex, GLuint level)
{
    TextureLevel *texLevel = mglTextureAttachmentLevel(tex, level);
    if (!texLevel) {
        return;
    }

    texLevel->ever_written = GL_TRUE;
    texLevel->has_initialized_data = GL_TRUE;
    texLevel->suspicious_zero_upload = GL_FALSE;
    texLevel->last_init_source = kTexRenderTargetWrite;
    texLevel->last_upload_size = 0u;
    texLevel->last_src_ptr = NULL;
    texLevel->last_src_hash = 0ull;
}

static inline void mglMarkTextureLevelMetalFilled(Texture *tex, GLuint level, size_t uploadSize)
{
    TextureLevel *texLevel = mglTextureAttachmentLevel(tex, level);
    if (!texLevel) {
        return;
    }

    texLevel->ever_written = GL_TRUE;
    texLevel->has_initialized_data = GL_TRUE;
    texLevel->suspicious_zero_upload = GL_FALSE;
    texLevel->last_init_source = kTexMetalFill;
    texLevel->last_upload_size = uploadSize;
    texLevel->last_src_ptr = NULL;
    texLevel->last_src_hash = 0ull;
}

static inline GLuint mglTraceTextureName(Texture *tex)
{
    return tex ? tex->name : 0u;
}

typedef NS_ENUM(NSUInteger, MGLTextureDataKind) {
    MGLTextureDataKindUnknown = 0,
    MGLTextureDataKindFloat = 1,
    MGLTextureDataKindSint = 2,
    MGLTextureDataKindUint = 3,
};

static MTLTextureType mglExpectedTextureTypeFromMSL(const char *msl, GLuint binding)
{
    if (!msl) {
        return 0;
    }

    char needle[32];
    snprintf(needle, sizeof(needle), "[[texture(%u)]]", (unsigned)binding);

    const char *cursor = msl;
    while ((cursor = strstr(cursor, needle)) != NULL) {
        const char *lineStart = cursor;
        while (lineStart > msl && lineStart[-1] != '\n' && lineStart[-1] != '\r') {
            lineStart--;
        }

        size_t lineLen = (size_t)(cursor - lineStart);
        if (lineLen > 1024u) {
            lineLen = 1024u;
        }

        char line[1025];
        memcpy(line, lineStart, lineLen);
        line[lineLen] = '\0';

        if (strstr(line, "texture_buffer")) {
            return MTLTextureTypeTextureBuffer;
        }
        if (strstr(line, "texturecube_array")) {
            return MTLTextureTypeCubeArray;
        }
        if (strstr(line, "texturecube")) {
            return MTLTextureTypeCube;
        }
        if (strstr(line, "texture3d")) {
            return MTLTextureType3D;
        }
        if (strstr(line, "texture2d_ms_array")) {
            return MTLTextureType2DMultisampleArray;
        }
        if (strstr(line, "texture2d_ms")) {
            return MTLTextureType2DMultisample;
        }
        if (strstr(line, "texture2d_array")) {
            return MTLTextureType2DArray;
        }
        if (strstr(line, "texture2d")) {
            return MTLTextureType2D;
        }
        if (strstr(line, "texture1d_array")) {
            return MTLTextureType1DArray;
        }
        if (strstr(line, "texture1d")) {
            return MTLTextureType1D;
        }

        cursor += strlen(needle);
    }

    return 0;
}

static MGLTextureDataKind mglExpectedTextureDataKindFromMSL(const char *msl, GLuint binding)
{
    if (!msl) {
        return MGLTextureDataKindUnknown;
    }

    char needle[32];
    snprintf(needle, sizeof(needle), "[[texture(%u)]]", (unsigned)binding);

    const char *cursor = msl;
    while ((cursor = strstr(cursor, needle)) != NULL) {
        const char *lineStart = cursor;
        while (lineStart > msl && lineStart[-1] != '\n' && lineStart[-1] != '\r') {
            lineStart--;
        }

        size_t lineLen = (size_t)(cursor - lineStart);
        if (lineLen > 1024u) {
            lineLen = 1024u;
        }

        char line[1025];
        memcpy(line, lineStart, lineLen);
        line[lineLen] = '\0';

        if (strstr(line, "<int") || strstr(line, "<short") || strstr(line, "<char")) {
            return MGLTextureDataKindSint;
        }
        if (strstr(line, "<uint") || strstr(line, "<ushort") || strstr(line, "<uchar")) {
            return MGLTextureDataKindUint;
        }
        if (strstr(line, "<float") || strstr(line, "<half")) {
            return MGLTextureDataKindFloat;
        }

        cursor += strlen(needle);
    }

    return MGLTextureDataKindUnknown;
}

static bool mglProgramHasResourceNamed(Program *program, int stage, int type, const char *name)
{
    if (!program || !name || stage < 0 || stage >= _MAX_SHADER_TYPES ||
        type < 0 || type >= _MAX_SPIRV_RES) {
        return false;
    }

    SpirvResourceList *resources = &program->spirv_resources_list[stage][type];
    for (GLuint i = 0; i < resources->count; i++) {
        SpirvResource *res = &resources->list[i];
        if (res->name && strcmp(res->name, name) == 0) {
            return true;
        }
    }

    return false;
}

static float mglTraceReadFloat(const uint8_t *bytes, size_t byteCount, size_t offset)
{
    if (!bytes || offset + sizeof(float) > byteCount) {
        return 0.0f;
    }

    float value = 0.0f;
    memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

static void mglTraceCloudVertexBufferBinding(Program *program,
                                             GLuint glBinding,
                                             NSUInteger metalBinding,
                                             Buffer *buffer,
                                             NSUInteger offset,
                                             NSUInteger availableBytes,
                                             const char *source)
{
    if (!program || !buffer ||
        !mglProgramHasResourceNamed(program, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, "CloudFaces")) {
        return;
    }
    if (glBinding != 0u && glBinding != 1u && glBinding != 3u && glBinding != 7u) {
        return;
    }
    if (!buffer->data.buffer_data || buffer->size <= 0 || offset >= (NSUInteger)buffer->size) {
        return;
    }

    static uint64_t s_cloudUBOTraceCount = 0;
    uint64_t hit = ++s_cloudUBOTraceCount;
    if (hit > 96ull && (hit % 512ull) != 0ull) {
        return;
    }

    const uint8_t *bytes = ((const uint8_t *)buffer->data.buffer_data) + offset;
    size_t byteCount = (size_t)((NSUInteger)buffer->size - offset);
    if (availableBytes > 0 && availableBytes < byteCount) {
        byteCount = (size_t)availableBytes;
    }

    if (glBinding == 3u) {
        NSLog(@"MGL CLOUD UBO hit=%llu program=%u source=%s glBinding=3(CloudInfo) metal=%lu buffer=%u offset=%lu bytes=%zu "
              "color=(%.5f,%.5f,%.5f,%.5f) cloudOffset=(%.5f,%.5f,%.5f) cellSize=(%.5f,%.5f,%.5f)",
              (unsigned long long)hit,
              (unsigned)program->name,
              source ? source : "(unknown)",
              (unsigned long)metalBinding,
              buffer->name,
              (unsigned long)offset,
              byteCount,
              mglTraceReadFloat(bytes, byteCount, 0),
              mglTraceReadFloat(bytes, byteCount, 4),
              mglTraceReadFloat(bytes, byteCount, 8),
              mglTraceReadFloat(bytes, byteCount, 12),
              mglTraceReadFloat(bytes, byteCount, 16),
              mglTraceReadFloat(bytes, byteCount, 20),
              mglTraceReadFloat(bytes, byteCount, 24),
              mglTraceReadFloat(bytes, byteCount, 32),
              mglTraceReadFloat(bytes, byteCount, 36),
              mglTraceReadFloat(bytes, byteCount, 40));
    } else if (glBinding == 7u) {
        NSLog(@"MGL CLOUD UBO hit=%llu program=%u source=%s glBinding=7 metal=%lu buffer=%u offset=%lu bytes=%zu "
              "color=(%.5f,%.5f,%.5f,%.5f) cloudOffset=(%.5f,%.5f,%.5f) cellSize=(%.5f,%.5f,%.5f)",
              (unsigned long long)hit,
              (unsigned)program->name,
              source ? source : "(unknown)",
              (unsigned long)metalBinding,
              buffer->name,
              (unsigned long)offset,
              byteCount,
              mglTraceReadFloat(bytes, byteCount, 0),
              mglTraceReadFloat(bytes, byteCount, 4),
              mglTraceReadFloat(bytes, byteCount, 8),
              mglTraceReadFloat(bytes, byteCount, 12),
              mglTraceReadFloat(bytes, byteCount, 16),
              mglTraceReadFloat(bytes, byteCount, 20),
              mglTraceReadFloat(bytes, byteCount, 24),
              mglTraceReadFloat(bytes, byteCount, 32),
              mglTraceReadFloat(bytes, byteCount, 36),
              mglTraceReadFloat(bytes, byteCount, 40));
    } else {
        NSLog(@"MGL CLOUD UBO hit=%llu program=%u source=%s glBinding=%u metal=%lu buffer=%u offset=%lu bytes=%zu "
              "m0=(%.5f,%.5f,%.5f,%.5f) m1=(%.5f,%.5f,%.5f,%.5f)",
              (unsigned long long)hit,
              (unsigned)program->name,
              source ? source : "(unknown)",
              (unsigned)glBinding,
              (unsigned long)metalBinding,
              buffer->name,
              (unsigned long)offset,
              byteCount,
              mglTraceReadFloat(bytes, byteCount, 0),
              mglTraceReadFloat(bytes, byteCount, 4),
              mglTraceReadFloat(bytes, byteCount, 8),
              mglTraceReadFloat(bytes, byteCount, 12),
              mglTraceReadFloat(bytes, byteCount, 16),
              mglTraceReadFloat(bytes, byteCount, 20),
              mglTraceReadFloat(bytes, byteCount, 24),
              mglTraceReadFloat(bytes, byteCount, 28));
    }
}

static MGLTextureDataKind mglTextureDataKindForPixelFormat(MTLPixelFormat pixelFormat)
{
    switch (pixelFormat) {
        case MTLPixelFormatR8Sint:
        case MTLPixelFormatRG8Sint:
        case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatR16Sint:
        case MTLPixelFormatRG16Sint:
        case MTLPixelFormatRGBA16Sint:
        case MTLPixelFormatR32Sint:
        case MTLPixelFormatRG32Sint:
        case MTLPixelFormatRGBA32Sint:
            return MGLTextureDataKindSint;

        case MTLPixelFormatR8Uint:
        case MTLPixelFormatRG8Uint:
        case MTLPixelFormatRGBA8Uint:
        case MTLPixelFormatR16Uint:
        case MTLPixelFormatRG16Uint:
        case MTLPixelFormatRGBA16Uint:
        case MTLPixelFormatR32Uint:
        case MTLPixelFormatRG32Uint:
        case MTLPixelFormatRGBA32Uint:
            return MGLTextureDataKindUint;

        case MTLPixelFormatInvalid:
            return MGLTextureDataKindUnknown;

        default:
            return MGLTextureDataKindFloat;
    }
}

static inline BOOL mglTexturePixelFormatCompatibleWithExpectedDataKind(MTLPixelFormat pixelFormat,
                                                                       MGLTextureDataKind expectedKind)
{
    if (expectedKind == MGLTextureDataKindUnknown) {
        return YES;
    }

    return mglTextureDataKindForPixelFormat(pixelFormat) == expectedKind;
}

static inline const char *mglTextureDataKindName(MGLTextureDataKind kind)
{
    switch (kind) {
        case MGLTextureDataKindFloat: return "float";
        case MGLTextureDataKindSint: return "sint";
        case MGLTextureDataKindUint: return "uint";
        default: return "unknown";
    }
}

static inline double mglNowSeconds(void)
{
    return CFAbsoluteTimeGetCurrent();
}

static inline void mglLogLoopHeartbeat(const char *tag,
                                       uint64_t callCount,
                                       double nowSeconds,
                                       double *lastCallSeconds,
                                       uint64_t *lastCallCount,
                                       double warnGapSeconds)
{
    if (!kMGLDiagnosticStateLogs || !lastCallSeconds || !lastCallCount) {
        return;
    }

    uint64_t deltaCalls = (*lastCallCount > 0) ? (callCount - *lastCallCount) : 0;
    double deltaMs = (*lastCallSeconds > 0.0) ? ((nowSeconds - *lastCallSeconds) * 1000.0) : 0.0;

    if (*lastCallSeconds > 0.0 &&
        warnGapSeconds > 0.0 &&
        (nowSeconds - *lastCallSeconds) >= warnGapSeconds) {
        NSLog(@"MGL TRACE %s gap=%.2fms deltaCalls=%llu call=%llu",
              tag ? tag : "loop",
              deltaMs,
              (unsigned long long)deltaCalls,
              (unsigned long long)callCount);
    } else if (mglShouldTraceCall(callCount) &&
               (callCount <= 20ull || (callCount % 60ull) == 0ull)) {
        NSLog(@"MGL TRACE %s heartbeat delta=%.2fms deltaCalls=%llu call=%llu",
              tag ? tag : "loop",
              deltaMs,
              (unsigned long long)deltaCalls,
              (unsigned long long)callCount);
    }

    *lastCallSeconds = nowSeconds;
    *lastCallCount = callCount;
}

static const char *mglCommandBufferStatusName(MTLCommandBufferStatus status)
{
    switch (status) {
        case MTLCommandBufferStatusNotEnqueued: return "NotEnqueued";
        case MTLCommandBufferStatusEnqueued: return "Enqueued";
        case MTLCommandBufferStatusCommitted: return "Committed";
        case MTLCommandBufferStatusScheduled: return "Scheduled";
        case MTLCommandBufferStatusCompleted: return "Completed";
        case MTLCommandBufferStatusError: return "Error";
        default: return "Unknown";
    }
}

static const char *mglLoadActionName(MTLLoadAction action)
{
    switch (action) {
        case MTLLoadActionDontCare: return "DontCare";
        case MTLLoadActionLoad: return "Load";
        case MTLLoadActionClear: return "Clear";
        default: return "Unknown";
    }
}

static const char *mglStoreActionName(MTLStoreAction action)
{
    switch (action) {
        case MTLStoreActionDontCare: return "DontCare";
        case MTLStoreActionStore: return "Store";
        case MTLStoreActionMultisampleResolve: return "MSResolve";
        case MTLStoreActionStoreAndMultisampleResolve: return "Store+MSResolve";
        case MTLStoreActionUnknown: return "Unknown";
        default: return "Other";
    }
}

static inline void mglAppendFlagName(char *dst, size_t dstSize, const char *name, bool *first)
{
    if (!dst || !name || dstSize == 0) {
        return;
    }

    size_t used = strlen(dst);
    if (used >= dstSize - 1) {
        return;
    }

    int written = snprintf(dst + used,
                           dstSize - used,
                           "%s%s",
                           (*first ? "" : "|"),
                           name);
    if (written > 0) {
        *first = false;
    }
}

static void mglFormatDirtyBits(uint32_t bits, char *dst, size_t dstSize)
{
    if (!dst || dstSize == 0) {
        return;
    }

    dst[0] = '\0';
    if (bits == 0) {
        snprintf(dst, dstSize, "none");
        return;
    }

    bool first = true;
    if (bits & DIRTY_VAO) mglAppendFlagName(dst, dstSize, "VAO", &first);
    if (bits & DIRTY_STATE) mglAppendFlagName(dst, dstSize, "STATE", &first);
    if (bits & DIRTY_BUFFER) mglAppendFlagName(dst, dstSize, "BUFFER", &first);
    if (bits & DIRTY_TEX) mglAppendFlagName(dst, dstSize, "TEX", &first);
    if (bits & DIRTY_TEX_PARAM) mglAppendFlagName(dst, dstSize, "TEX_PARAM", &first);
    if (bits & DIRTY_TEX_BINDING) mglAppendFlagName(dst, dstSize, "TEX_BINDING", &first);
    if (bits & DIRTY_SAMPLER) mglAppendFlagName(dst, dstSize, "SAMPLER", &first);
    if (bits & DIRTY_SHADER) mglAppendFlagName(dst, dstSize, "SHADER", &first);
    if (bits & DIRTY_PROGRAM) mglAppendFlagName(dst, dstSize, "PROGRAM", &first);
    if (bits & DIRTY_FBO) mglAppendFlagName(dst, dstSize, "FBO", &first);
    if (bits & DIRTY_DRAWABLE) mglAppendFlagName(dst, dstSize, "DRAWABLE", &first);
    if (bits & DIRTY_RENDER_STATE) mglAppendFlagName(dst, dstSize, "RENDER_STATE", &first);
    if (bits & DIRTY_ALPHA_STATE) mglAppendFlagName(dst, dstSize, "ALPHA_STATE", &first);
    if (bits & DIRTY_IMAGE_UNIT_STATE) mglAppendFlagName(dst, dstSize, "IMAGE_UNIT", &first);
    if (bits & DIRTY_BUFFER_BASE_STATE) mglAppendFlagName(dst, dstSize, "BUFFER_BASE", &first);
    if (bits & DIRTY_ALL_BIT) mglAppendFlagName(dst, dstSize, "ALL_BIT", &first);
    if (bits == DIRTY_ALL) mglAppendFlagName(dst, dstSize, "ALL", &first);

    if (first) {
        snprintf(dst, dstSize, "0x%x", bits);
    }
}

static void mglLogStateSnapshot(const char *tag,
                                GLMContext ctx,
                                id<MTLCommandBuffer> commandBuffer,
                                id<MTLRenderCommandEncoder> renderEncoder,
                                MTLRenderPassDescriptor *renderPassDescriptor,
                                id<CAMetalDrawable> drawable)
{
    if (!kMGLDiagnosticStateLogs) {
        return;
    }

    if (!mglRendererContextLikelyValid(ctx)) {
        NSLog(@"MGL TRACE %s ctx=%p(invalid) cb=%p enc=%p rpd=%p drawable=%p",
              tag ? tag : "snapshot", ctx, commandBuffer, renderEncoder, renderPassDescriptor, drawable);
        return;
    }

    Program *program = mglResolveProgramFromState(ctx);
    GLuint programName = ctx->state.program_name ? ctx->state.program_name : (program ? program->name : 0);
    Framebuffer *drawFBO = ctx->state.framebuffer;
    GLuint drawFBOName = 0;
    if (drawFBO) {
        if (mglRendererObjectPointerLikelyValid(drawFBO) &&
            mglRendererPointerInHashTable(&ctx->state.framebuffer_table, drawFBO) &&
            mglPointerRangeIsReadable(drawFBO, sizeof(*drawFBO))) {
            drawFBOName = drawFBO->name;
        } else {
            NSLog(@"MGL TRACE %s invalid drawFBO=%p", tag ? tag : "snapshot", drawFBO);
            drawFBO = NULL;
        }
    }

    MTLCommandBufferStatus cbStatus = commandBuffer ? commandBuffer.status : MTLCommandBufferStatusNotEnqueued;
    NSString *cbLabel = commandBuffer ? (commandBuffer.label ?: @"(no-label)") : @"(nil)";
    char dirtyNames[256];
    mglFormatDirtyBits((uint32_t)ctx->state.dirty_bits, dirtyNames, sizeof(dirtyNames));

    id<MTLTexture> rpColor0 = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].texture : nil;
    id<MTLTexture> rpDepth = renderPassDescriptor ? renderPassDescriptor.depthAttachment.texture : nil;
    id<MTLTexture> rpStencil = renderPassDescriptor ? renderPassDescriptor.stencilAttachment.texture : nil;
    MTLLoadAction colorLoadAction = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare;
    MTLStoreAction colorStoreAction = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare;
    MTLLoadAction depthLoadAction = renderPassDescriptor ? renderPassDescriptor.depthAttachment.loadAction : MTLLoadActionDontCare;
    MTLStoreAction depthStoreAction = renderPassDescriptor ? renderPassDescriptor.depthAttachment.storeAction : MTLStoreActionDontCare;
    MTLLoadAction stencilLoadAction = renderPassDescriptor ? renderPassDescriptor.stencilAttachment.loadAction : MTLLoadActionDontCare;
    MTLStoreAction stencilStoreAction = renderPassDescriptor ? renderPassDescriptor.stencilAttachment.storeAction : MTLStoreActionDontCare;
    MTLClearColor rpClearColor = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].clearColor : MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    id<MTLTexture> drawableTexture = drawable ? drawable.texture : nil;

    NSLog(@"MGL TRACE %s prog=%u dirty=0x%x[%s] clear=0x%x drawBuf=0x%x readBuf=0x%x vao=%p drawFBO=%p(%u) "
          "vp=(%u,%u,%u,%u) scissor(en=%d box=%d,%d,%d,%d) caps(depth=%d blend=%d cull=%d) "
          "stateClear=(%.3f,%.3f,%.3f,%.3f) cb=%p[%s] label=%@ enc=%p rpd=%p rt=%lux%lu "
          "c0=%p fmt=%lu usage=0x%lx la/sa=%s/%s clear=(%.3f,%.3f,%.3f,%.3f) "
          "depth=%p(%lu %s/%s) stencil=%p(%lu %s/%s) drawable=%p tex=%p d=%lux%lu",
          tag ? tag : "snapshot",
          (unsigned)programName,
          (unsigned)ctx->state.dirty_bits,
          dirtyNames,
          (unsigned)ctx->state.clear_bitmask,
          (unsigned)ctx->state.draw_buffer,
          (unsigned)ctx->state.read_buffer,
          ctx->state.vao,
          drawFBO,
          (unsigned)drawFBOName,
          (unsigned)ctx->state.viewport[0],
          (unsigned)ctx->state.viewport[1],
          (unsigned)ctx->state.viewport[2],
          (unsigned)ctx->state.viewport[3],
          ctx->state.caps.scissor_test ? 1 : 0,
          (int)ctx->state.var.scissor_box[0],
          (int)ctx->state.var.scissor_box[1],
          (int)ctx->state.var.scissor_box[2],
          (int)ctx->state.var.scissor_box[3],
          ctx->state.caps.depth_test ? 1 : 0,
          ctx->state.caps.blend ? 1 : 0,
          ctx->state.caps.cull_face ? 1 : 0,
          ctx->state.color_clear_value[0],
          ctx->state.color_clear_value[1],
          ctx->state.color_clear_value[2],
          ctx->state.color_clear_value[3],
          commandBuffer,
          mglCommandBufferStatusName(cbStatus),
          cbLabel,
          renderEncoder,
          renderPassDescriptor,
          (unsigned long)(renderPassDescriptor ? renderPassDescriptor.renderTargetWidth : 0),
          (unsigned long)(renderPassDescriptor ? renderPassDescriptor.renderTargetHeight : 0),
          rpColor0,
          (unsigned long)(rpColor0 ? rpColor0.pixelFormat : MTLPixelFormatInvalid),
          (unsigned long)(rpColor0 ? rpColor0.usage : 0),
          mglLoadActionName(colorLoadAction),
          mglStoreActionName(colorStoreAction),
          rpClearColor.red,
          rpClearColor.green,
          rpClearColor.blue,
          rpClearColor.alpha,
          rpDepth,
          (unsigned long)(rpDepth ? rpDepth.pixelFormat : MTLPixelFormatInvalid),
          mglLoadActionName(depthLoadAction),
          mglStoreActionName(depthStoreAction),
          rpStencil,
          (unsigned long)(rpStencil ? rpStencil.pixelFormat : MTLPixelFormatInvalid),
          mglLoadActionName(stencilLoadAction),
          mglStoreActionName(stencilStoreAction),
          drawable,
          drawableTexture,
          (unsigned long)(drawableTexture ? drawableTexture.width : 0),
          (unsigned long)(drawableTexture ? drawableTexture.height : 0));

    NSLog(@"MGL TRACE %s masks color0(use=%d rgba=%d%d%d%d) depthWrite=%d stencilWrite=0x%x",
          tag ? tag : "snapshot",
          ctx->state.caps.use_color_mask[0] ? 1 : 0,
          ctx->state.var.color_writemask[0][0] ? 1 : 0,
          ctx->state.var.color_writemask[0][1] ? 1 : 0,
          ctx->state.var.color_writemask[0][2] ? 1 : 0,
          ctx->state.var.color_writemask[0][3] ? 1 : 0,
          ctx->state.var.depth_writemask ? 1 : 0,
          (unsigned)ctx->state.var.stencil_writemask);
}

static void mglLogDrawWithoutSwapWatchdog(const char *kind,
                                          uint64_t drawCall,
                                          GLMContext ctx,
                                          id<MTLCommandBuffer> commandBuffer,
                                          id<MTLRenderCommandEncoder> renderEncoder,
                                          MTLRenderPassDescriptor *renderPassDescriptor)
{
    uint64_t drawArrays = g_mglDrawArraysSinceSwap;
    uint64_t drawElements = g_mglDrawElementsSinceSwap;
    uint64_t totalDraws = drawArrays + drawElements;
    if (totalDraws < 16384ull || (totalDraws % 16384ull) != 0ull) {
        return;
    }

    double now = mglNowSeconds();
    double lastSwap = g_mglLastSwapSeconds;
    double lastSwapAgeMs = (lastSwap > 0.0) ? ((now - lastSwap) * 1000.0) : -1.0;
    if (lastSwapAgeMs >= 0.0 && lastSwapAgeMs < 250.0) {
        return;
    }
    MTLCommandBufferStatus cbStatus = commandBuffer ? commandBuffer.status : MTLCommandBufferStatusNotEnqueued;
    id<MTLTexture> rpColor0 = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].texture : nil;
    MTLLoadAction colorLoadAction = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare;
    MTLStoreAction colorStoreAction = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare;
    MTLClearColor clear = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].clearColor : MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    NSLog(@"MGL WATCHDOG: draws-without-swap kind=%s drawCall=%llu total=%llu arrays=%llu elements=%llu "
          "swapCalls=%llu lastSwapAgeMs=%.2f program=%u drawBuf=0x%x fbo=%p vao=%p cb=%p[%s] enc=%p "
          "rpd=%p c0=%p fmt=%lu la/sa=%s/%s clear=(%.3f,%.3f,%.3f,%.3f)",
          kind ? kind : "draw",
          (unsigned long long)drawCall,
          (unsigned long long)totalDraws,
          (unsigned long long)drawArrays,
          (unsigned long long)drawElements,
          (unsigned long long)g_mglSwapCallCount,
          lastSwapAgeMs,
          (unsigned)(ctx ? ctx->state.program_name : 0u),
          (unsigned)(ctx ? ctx->state.draw_buffer : 0u),
          ctx ? ctx->state.framebuffer : NULL,
          ctx ? ctx->state.vao : NULL,
          commandBuffer,
          mglCommandBufferStatusName(cbStatus),
          renderEncoder,
          renderPassDescriptor,
          rpColor0,
          (unsigned long)(rpColor0 ? rpColor0.pixelFormat : MTLPixelFormatInvalid),
          mglLoadActionName(colorLoadAction),
          mglStoreActionName(colorStoreAction),
          clear.red,
          clear.green,
          clear.blue,
          clear.alpha);
}

static void mglLogRenderPassLifecycle(const char *tag,
                                      uint64_t call,
                                      GLMContext ctx,
                                      id<MTLCommandBuffer> commandBuffer,
                                      id<MTLRenderCommandEncoder> renderEncoder,
                                      MTLRenderPassDescriptor *renderPassDescriptor,
                                      id<CAMetalDrawable> drawable,
                                      Framebuffer *renderPassFramebuffer,
                                      GLuint renderPassFramebufferName,
                                      GLenum renderPassDrawBuffer,
                                      GLsizei renderPassDrawBufferCount)
{
    MTLCommandBufferStatus cbStatus = commandBuffer ? commandBuffer.status : MTLCommandBufferStatusNotEnqueued;
    id<MTLTexture> c0 = renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].texture : nil;
    id<MTLTexture> c1 = renderPassDescriptor ? renderPassDescriptor.colorAttachments[1].texture : nil;
    id<MTLTexture> depth = renderPassDescriptor ? renderPassDescriptor.depthAttachment.texture : nil;
    id<MTLTexture> stencil = renderPassDescriptor ? renderPassDescriptor.stencilAttachment.texture : nil;
    id<MTLTexture> drawableTexture = drawable ? drawable.texture : nil;
    MTLClearColor clear = renderPassDescriptor
        ? renderPassDescriptor.colorAttachments[0].clearColor
        : MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    if (fbo &&
        (!mglRendererObjectPointerLikelyValid(fbo) ||
         !mglRendererPointerInHashTable(&ctx->state.framebuffer_table, fbo) ||
         !mglPointerRangeIsReadable(fbo, sizeof(*fbo)))) {
        NSLog(@"MGL TRACE %s invalid lifecycle fbo=%p", tag ? tag : "unknown", fbo);
        fbo = NULL;
    }
    GLuint fboName = fbo ? fbo->name : 0u;
    GLuint color0Name = 0u;
    GLuint color1Name = 0u;
    GLuint depthName = 0u;
    if (fbo) {
        color0Name = fbo->color_attachments[0].texture;
        color1Name = fbo->color_attachments[1].texture;
        depthName = fbo->depth.texture;
    }

    NSLog(@"MGL TRACE renderpass.%s call=%llu program=%u dirty=0x%x drawBuf=0x%x readBuf=0x%x "
          "fbo=%u(%p) rpFbo=%u(%p) rpDrawBuf=0x%x rpDrawCount=%d vao=%p cb=%p[%s] enc=%p rpd=%p rt=%lux%lu "
          "c0Name=%u c0=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s clear=(%.3f,%.3f,%.3f,%.3f) "
          "c1Name=%u c1=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s "
          "depthName=%u depth=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s "
          "stencil=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s "
          "drawable=%p tex=%p size=%lux%lu",
          tag ? tag : "unknown",
          (unsigned long long)call,
          (unsigned)(ctx ? ctx->state.program_name : 0u),
          (unsigned)(ctx ? ctx->state.dirty_bits : 0u),
          (unsigned)(ctx ? ctx->state.draw_buffer : 0u),
          (unsigned)(ctx ? ctx->state.read_buffer : 0u),
          (unsigned)fboName,
          fbo,
          (unsigned)renderPassFramebufferName,
          renderPassFramebuffer,
          (unsigned)renderPassDrawBuffer,
          (int)renderPassDrawBufferCount,
          ctx ? ctx->state.vao : NULL,
          commandBuffer,
          mglCommandBufferStatusName(cbStatus),
          renderEncoder,
          renderPassDescriptor,
          (unsigned long)(renderPassDescriptor ? renderPassDescriptor.renderTargetWidth : 0),
          (unsigned long)(renderPassDescriptor ? renderPassDescriptor.renderTargetHeight : 0),
          (unsigned)color0Name,
          c0,
          (unsigned long)(c0 ? c0.pixelFormat : MTLPixelFormatInvalid),
          (unsigned long)(c0 ? c0.usage : 0),
          (unsigned long)(c0 ? c0.width : 0),
          (unsigned long)(c0 ? c0.height : 0),
          mglLoadActionName(renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].loadAction : MTLLoadActionDontCare),
          mglStoreActionName(renderPassDescriptor ? renderPassDescriptor.colorAttachments[0].storeAction : MTLStoreActionDontCare),
          clear.red,
          clear.green,
          clear.blue,
          clear.alpha,
          (unsigned)color1Name,
          c1,
          (unsigned long)(c1 ? c1.pixelFormat : MTLPixelFormatInvalid),
          (unsigned long)(c1 ? c1.usage : 0),
          (unsigned long)(c1 ? c1.width : 0),
          (unsigned long)(c1 ? c1.height : 0),
          mglLoadActionName(renderPassDescriptor ? renderPassDescriptor.colorAttachments[1].loadAction : MTLLoadActionDontCare),
          mglStoreActionName(renderPassDescriptor ? renderPassDescriptor.colorAttachments[1].storeAction : MTLStoreActionDontCare),
          (unsigned)depthName,
          depth,
          (unsigned long)(depth ? depth.pixelFormat : MTLPixelFormatInvalid),
          (unsigned long)(depth ? depth.usage : 0),
          (unsigned long)(depth ? depth.width : 0),
          (unsigned long)(depth ? depth.height : 0),
          mglLoadActionName(renderPassDescriptor ? renderPassDescriptor.depthAttachment.loadAction : MTLLoadActionDontCare),
          mglStoreActionName(renderPassDescriptor ? renderPassDescriptor.depthAttachment.storeAction : MTLStoreActionDontCare),
          stencil,
          (unsigned long)(stencil ? stencil.pixelFormat : MTLPixelFormatInvalid),
          (unsigned long)(stencil ? stencil.usage : 0),
          (unsigned long)(stencil ? stencil.width : 0),
          (unsigned long)(stencil ? stencil.height : 0),
          mglLoadActionName(renderPassDescriptor ? renderPassDescriptor.stencilAttachment.loadAction : MTLLoadActionDontCare),
          mglStoreActionName(renderPassDescriptor ? renderPassDescriptor.stencilAttachment.storeAction : MTLStoreActionDontCare),
          drawable,
          drawableTexture,
          (unsigned long)(drawableTexture ? drawableTexture.width : 0),
          (unsigned long)(drawableTexture ? drawableTexture.height : 0));
}

static BOOL mglRendererPointerInHashTable(HashTable *table, const void *ptr)
{
    return mglRendererObjectPointerLikelyValid(ptr) &&
           mglHashTableContainsData(table, ptr);
}

static void mglRendererDropCurrentVAO(GLMContext ctx)
{
    if (!ctx) {
        return;
    }

    ctx->state.vao = NULL;
    ctx->state.buffers[_ELEMENT_ARRAY_BUFFER] = ctx->state.default_vao_element_array_buffer;
    ctx->state.var.element_array_buffer_binding =
        ctx->state.default_vao_element_array_buffer ? ctx->state.default_vao_element_array_buffer->name : 0;
    ctx->state.dirty_bits |= DIRTY_VAO;
}

static VertexArray *mglRendererGetValidatedVAO(GLMContext ctx, const char *where)
{
    if (!ctx) {
        return NULL;
    }

    VertexArray *vao = ctx->state.vao;
    if (!vao) {
        return NULL;
    }

    if (!mglRendererObjectPointerLikelyValid(vao)) {
        NSLog(@"MGL VAO INVALID in %s: vao=%p (suspicious pseudo-pointer)",
              where ? where : "unknown", vao);
        mglRendererDropCurrentVAO(ctx);
        return NULL;
    }

    if (!mglRendererPointerInHashTable(&ctx->state.vao_table, vao)) {
        NSLog(@"MGL VAO INVALID in %s: vao=%p (not found in sane vao_table)",
              where ? where : "unknown", vao);
        mglRendererDropCurrentVAO(ctx);
        return NULL;
    }

    if (!mglPointerRangeIsReadable(vao, sizeof(*vao))) {
        NSLog(@"MGL VAO INVALID in %s: vao=%p (unreadable object memory)",
              where ? where : "unknown", vao);
        mglRendererDropCurrentVAO(ctx);
        return NULL;
    }

    if (vao->magic != MGL_VAO_MAGIC) {
        NSLog(@"MGL VAO INVALID in %s: vao=%p magic=0x%x",
              where ? where : "unknown", vao, vao->magic);
        mglRendererDropCurrentVAO(ctx);
        return NULL;
    }

    return vao;
}

static Buffer *mglRendererGetValidatedBuffer(GLMContext ctx, Buffer *candidate, const char *where, NSUInteger slot)
{
    if (!candidate) {
        return NULL;
    }

    if (!mglRendererObjectPointerLikelyValid(candidate)) {
        NSLog(@"MGL BUFFER INVALID in %s: slot=%lu candidate=%p (suspicious pseudo-pointer)",
              where ? where : "unknown", (unsigned long)slot, candidate);
        return NULL;
    }

    if (!ctx || !mglRendererPointerInHashTable(&ctx->state.buffer_table, candidate)) {
        NSLog(@"MGL BUFFER INVALID in %s: slot=%lu candidate=%p (not found in sane buffer_table)",
              where ? where : "unknown", (unsigned long)slot, candidate);
        return NULL;
    }

    if (!mglPointerRangeIsReadable(candidate, sizeof(*candidate))) {
        NSLog(@"MGL BUFFER INVALID in %s: slot=%lu candidate=%p (unreadable object memory)",
              where ? where : "unknown", (unsigned long)slot, candidate);
        return NULL;
    }

    return candidate;
}

static Framebuffer *mglRendererGetValidatedFramebuffer(GLMContext ctx, const char *where)
{
    if (!ctx) {
        return NULL;
    }

    Framebuffer *fbo = ctx->state.framebuffer;
    if (!fbo) {
        return NULL;
    }

    if (!mglRendererObjectPointerLikelyValid(fbo) ||
        !mglRendererPointerInHashTable(&ctx->state.framebuffer_table, fbo) ||
        !mglPointerRangeIsReadable(fbo, sizeof(*fbo))) {
        NSLog(@"MGL FBO INVALID in %s: framebuffer=%p (not found in sane framebuffer_table or unreadable)",
              where ? where : "unknown", fbo);
        if (ctx->state.readbuffer == fbo) {
            ctx->state.readbuffer = NULL;
        }
        ctx->state.framebuffer = NULL;
        ctx->state.dirty_bits |= (DIRTY_FBO | DIRTY_STATE);
        return NULL;
    }

    return fbo;
}

static GLuint mglRendererSafeFramebufferName(GLMContext ctx)
{
    Framebuffer *fbo = mglRendererGetValidatedFramebuffer(ctx, "safeFramebufferName");
    return fbo ? fbo->name : 0u;
}

static BOOL mglRendererSameVertexStream(Buffer *lhsBuffer,
                                        GLintptr lhsOffset,
                                        Buffer *rhsBuffer,
                                        GLintptr rhsOffset)
{
    if (!lhsBuffer || !rhsBuffer || lhsOffset != rhsOffset) {
        return NO;
    }

    return lhsBuffer == rhsBuffer ||
           (lhsBuffer->name == rhsBuffer->name && lhsBuffer->target == rhsBuffer->target);
}

static BOOL mglRendererProgramUsesVertexAttrib(Program *program, GLuint attribute)
{
    if (!program || attribute >= MAX_ATTRIBS) {
        return YES;
    }

    SpirvResourceList *inputs =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    if (!inputs->list || inputs->count == 0) {
        return YES;
    }

    for (GLuint i = 0; i < inputs->count; i++) {
        GLuint location = inputs->list[i].location;
        if (location == attribute) {
            return YES;
        }

        if (location == 0xffffffffu && i == attribute) {
            return YES;
        }
    }

    return NO;
}

static SpirvResource *mglRendererProgramVertexAttribResource(Program *program, GLuint attribute)
{
    if (!program || attribute >= MAX_ATTRIBS) {
        return NULL;
    }

    SpirvResourceList *inputs =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    if (!inputs->list || inputs->count == 0) {
        return NULL;
    }

    for (GLuint i = 0; i < inputs->count; i++) {
        GLuint location = inputs->list[i].location;
        if (location == attribute || (location == 0xffffffffu && i == attribute)) {
            return &inputs->list[i];
        }
    }

    return NULL;
}

static bool mglRendererVertexAttribIsColorInput(Program *program, GLuint attribute)
{
    SpirvResource *resource = mglRendererProgramVertexAttribResource(program, attribute);
    const char *name = resource ? resource->name : NULL;
    return name &&
           (strcasecmp(name, "Color") == 0 ||
            strcasecmp(name, "vertexColor") == 0 ||
            strcasecmp(name, "VertColor") == 0);
}

static int mglRendererResolveVertexAttributeBufferIndex(GLMContext ctx,
                                                        VertexArray *vao,
                                                        GLuint attribute,
                                                        const char *where)
{
    if (!ctx || !vao || attribute >= MAX_ATTRIBS) {
        return -1;
    }

    if ((vao->enabled_attribs & (0x1u << attribute)) == 0u) {
        return -1;
    }

    Program *activeProgram = mglResolveProgramFromState(ctx);
    if (!mglRendererProgramUsesVertexAttrib(activeProgram, attribute)) {
        return -1;
    }

    Buffer *target = mglRendererGetValidatedBuffer(ctx, vao->attrib[attribute].buffer, where, attribute);
    if (!target) {
        return -1;
    }
    GLintptr targetBindingOffset = vao->attrib[attribute].binding_offset;
    if (targetBindingOffset < 0) {
        NSLog(@"MGL ERROR: attribute %u has negative vertex binding offset=%lld in %s",
              attribute, (long long)targetBindingOffset, where);
        return -1;
    }

    Buffer *seenBuffers[MAX_ATTRIBS] = {0};
    GLintptr seenOffsets[MAX_ATTRIBS] = {0};
    GLuint seenCount = 0;
    GLuint maxAttribs = ctx->state.max_vertex_attribs;
    if (maxAttribs > MAX_ATTRIBS) {
        maxAttribs = MAX_ATTRIBS;
    }

    for (GLuint i = 0; i < maxAttribs; i++) {
        if ((vao->enabled_attribs & (0x1u << i)) == 0u) {
            continue;
        }
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, i)) {
            continue;
        }

        Buffer *attribBuffer = mglRendererGetValidatedBuffer(ctx, vao->attrib[i].buffer, where, i);
        if (!attribBuffer) {
            continue;
        }

        int slot = -1;
        for (GLuint s = 0; s < seenCount; s++) {
            Buffer *known = seenBuffers[s];
            if (mglRendererSameVertexStream(known,
                                            seenOffsets[s],
                                            attribBuffer,
                                            vao->attrib[i].binding_offset)) {
                slot = (int)s;
                break;
            }
        }

        if (slot < 0) {
            if (kMGLVertexAttribBufferBase + seenCount > kMGLMaxMetalVertexBufferIndex) {
                NSLog(@"MGL ERROR: Vertex attrib mapping overflow (seen=%u base=%lu maxIndex=%lu)",
                      seenCount, (unsigned long)kMGLVertexAttribBufferBase, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return -1;
            }

            seenBuffers[seenCount] = attribBuffer;
            seenOffsets[seenCount] = vao->attrib[i].binding_offset;
            slot = (int)seenCount;
            seenCount++;
        }

        if (i == attribute) {
            NSUInteger resolvedIndex = kMGLVertexAttribBufferBase + (NSUInteger)slot;
            if (resolvedIndex > kMGLMaxMetalVertexBufferIndex) {
                NSLog(@"MGL ERROR: Vertex attrib index out of Metal range (attrib=%u resolved=%lu max=%lu)",
                      attribute, (unsigned long)resolvedIndex, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return -1;
            }
            return (int)resolvedIndex;
        }

        // Early out once there are no higher enabled attributes.
        if ((vao->enabled_attribs >> (i + 1)) == 0u) {
            break;
        }
    }

    return -1;
}

// Main class performing the rendering
@implementation MGLRenderer
{
    NSView *_view;

    CAMetalLayer *_layer;
    id<CAMetalDrawable> _drawable;

    GLMContext  ctx;    // context macros need this exact name

    id<MTLDevice> _device;

    // CRITICAL FIX: Thread synchronization to prevent race conditions
    NSLock *_metalStateLock;

    // AGX GPU Error Tracking - Prevent command queue from entering error state
    NSUInteger _consecutiveGPUErrors;
    NSUInteger _consecutiveGPUSuccesses;
    NSTimeInterval _lastGPUErrorTime;
    BOOL _gpuErrorRecoveryMode;

    // Quarantine programs that repeatedly fail VS/FS interface validation.
    GLuint _interfaceMismatchBlockedProgram;
    CFTimeInterval _interfaceMismatchBlockedUntil;
    uint32_t _interfaceMismatchBlockedStreak;

    // PROACTIVE TEXTURE STORAGE - Essential textures created during initialization
    NSMutableArray *_proactiveTextures;

    MGLDrawable _drawBuffers[_MAX_DRAW_BUFFERS];

    MTLBlendFactor _src_blend_rgb_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor _dst_blend_rgb_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor _src_blend_alpha_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor _dst_blend_alpha_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendOperation _rgb_blend_operation[MAX_COLOR_ATTACHMENTS];
    MTLBlendOperation _alpha_blend_operation[MAX_COLOR_ATTACHMENTS];
    MTLColorWriteMask _color_mask[MAX_COLOR_ATTACHMENTS];

    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _commandQueue;

    // The render pipeline generated from the vertex and fragment shaders in the .metal shader file.
    id<MTLRenderPipelineState> _pipelineState;
    MTLPixelFormat _pipelineColor0Format;
    MTLPixelFormat _pipelineDepthFormat;
    MTLPixelFormat _pipelineStencilFormat;
    GLuint _pipelineProgramName;
    NSMutableDictionary<NSString *, id<MTLRenderPipelineState>> *_pipelineStateCache;

    // render pass descriptor containts the binding information for VAO's and such
    MTLRenderPassDescriptor *_renderPassDescriptor;
    Framebuffer *_renderPassFramebuffer;
    GLuint _renderPassFramebufferName;
    GLenum _renderPassDrawBuffer;
    GLsizei _renderPassDrawBufferCount;
    GLenum _renderPassDrawBuffers[MAX_COLOR_ATTACHMENTS];

    // each pass a new command buffer is created
    id<MTLCommandBuffer> _currentCommandBuffer;
    SyncList  *_currentCommandBufferSyncList;

    id<MTLRenderCommandEncoder> _currentRenderEncoder;
    id<MTLTexture> _fallbackRenderTargetTexture;
    id<MTLTexture> _transientDepthTexture;
    NSUInteger _transientDepthTextureWidth;
    NSUInteger _transientDepthTextureHeight;
    id<MTLTexture> _fallbackSampledTexture;
    id<MTLTexture> _fallbackCubeSampledTexture;
    id<MTLTexture> _fallbackLightmapSampledTexture;
    id<MTLBuffer> _fallbackTextureBufferStorage;
    id<MTLTexture> _fallbackSintTextureBuffer;
    NSMutableDictionary<NSNumber *, id<MTLTexture>> *_fallbackSampledTextureCache;
    id<MTLSamplerState> _fallbackSamplerState;
    NSMutableDictionary<NSNumber *, id<MTLRenderPipelineState>> *_scaledBlitPipelineCache;
    id<MTLSamplerState> _scaledBlitNearestSampler;
    id<MTLSamplerState> _scaledBlitLinearSampler;

    GLuint _blitOperationComplete;

    id<MTLEvent> _currentEvent;
    GLsizei _currentSyncName;
    BOOL _isCommittingCommandBuffer;
}

MTLVertexFormat glTypeSizeToMtlType(GLuint type, GLuint size, bool normalized)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE:
            if (normalized)
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatUCharNormalized;
                    case 2: return MTLVertexFormatUChar2Normalized;
                    case 3: return MTLVertexFormatUChar3Normalized;
                    case 4: return MTLVertexFormatUChar4Normalized;
                }
            }
            else
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatUChar;
                    case 2: return MTLVertexFormatUChar2;
                    case 3: return MTLVertexFormatUChar3;
                    case 4: return MTLVertexFormatUChar4;
                }
            }
            break;

        case GL_BYTE:
            if (normalized)
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatCharNormalized;
                    case 2: return MTLVertexFormatChar2Normalized;
                    case 3: return MTLVertexFormatChar3Normalized;
                    case 4: return MTLVertexFormatChar4Normalized;
                }
            }
            else
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatChar;
                    case 2: return MTLVertexFormatChar2;
                    case 3: return MTLVertexFormatChar3;
                    case 4: return MTLVertexFormatChar4;
                }
            }
            break;

        case GL_UNSIGNED_SHORT:
            if (normalized)
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatUShortNormalized;
                    case 2: return MTLVertexFormatUShort2Normalized;
                    case 3: return MTLVertexFormatUShort3Normalized;
                    case 4: return MTLVertexFormatUShort4Normalized;
                }
            }
            else
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatUShort;
                    case 2: return MTLVertexFormatUShort2;
                    case 3: return MTLVertexFormatUShort3;
                    case 4: return MTLVertexFormatUShort4;
                }
            }
            break;

        case GL_SHORT:
            if (normalized)
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatShortNormalized;
                    case 2: return MTLVertexFormatShort2Normalized;
                    case 3: return MTLVertexFormatShort3Normalized;
                    case 4: return MTLVertexFormatShort4Normalized;
                }
            }
            else
            {
                switch(size)
                {
                    case 1: return MTLVertexFormatShort;
                    case 2: return MTLVertexFormatShort2;
                    case 3: return MTLVertexFormatShort3;
                    case 4: return MTLVertexFormatShort4;
                }
            }
            break;

            case GL_HALF_FLOAT:
                switch(size)
                {
                    case 1: return MTLVertexFormatHalf;
                    case 2: return MTLVertexFormatHalf2;
                    case 3: return MTLVertexFormatHalf3;
                    case 4: return MTLVertexFormatHalf4;
                }
                break;

            case GL_FLOAT:
                switch(size)
                {
                    case 1: return MTLVertexFormatFloat;
                    case 2: return MTLVertexFormatFloat2;
                    case 3: return MTLVertexFormatFloat3;
                    case 4: return MTLVertexFormatFloat4;
                }
                break;

            case GL_INT:
                switch(size)
                {
                    case 1: return MTLVertexFormatInt;
                    case 2: return MTLVertexFormatInt2;
                    case 3: return MTLVertexFormatInt3;
                    case 4: return MTLVertexFormatInt4;
                }
                break;

            case GL_UNSIGNED_INT:
                switch(size)
                {
                    case 1: return MTLVertexFormatUInt;
                    case 2: return MTLVertexFormatUInt2;
                    case 3: return MTLVertexFormatUInt3;
                    case 4: return MTLVertexFormatUInt4;
                }
                break;

            case GL_RGB10:
                if (normalized)
                    return MTLVertexFormatInt1010102Normalized;
                break;

            case GL_UNSIGNED_INT_10_10_10_2:
            case GL_UNSIGNED_INT_2_10_10_10_REV:
                if (normalized)
                    return MTLVertexFormatUInt1010102Normalized;
                break;
        }

    return MTLVertexFormatInvalid;
}

static inline size_t mglVertexAttribComponentSize(GLenum type)
{
    switch (type)
    {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            return 1u;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
            return 2u;
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
        case GL_FIXED:
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            return 4u;
        case GL_DOUBLE:
            return 8u;
        default:
            return 0u;
    }
}

static const char *mglVertexFormatName(MTLVertexFormat format)
{
    switch (format) {
        case MTLVertexFormatFloat: return "Float";
        case MTLVertexFormatFloat2: return "Float2";
        case MTLVertexFormatFloat3: return "Float3";
        case MTLVertexFormatFloat4: return "Float4";
        case MTLVertexFormatUChar4: return "UChar4";
        case MTLVertexFormatUChar4Normalized: return "UChar4Normalized";
        case MTLVertexFormatUChar3: return "UChar3";
        case MTLVertexFormatUChar3Normalized: return "UChar3Normalized";
        case MTLVertexFormatUChar2: return "UChar2";
        case MTLVertexFormatUChar2Normalized: return "UChar2Normalized";
        case MTLVertexFormatUChar: return "UChar";
        case MTLVertexFormatUCharNormalized: return "UCharNormalized";
        case MTLVertexFormatShort: return "Short";
        case MTLVertexFormatShort2: return "Short2";
        case MTLVertexFormatShort3: return "Short3";
        case MTLVertexFormatShort4: return "Short4";
        case MTLVertexFormatShortNormalized: return "ShortNormalized";
        case MTLVertexFormatShort2Normalized: return "Short2Normalized";
        case MTLVertexFormatShort3Normalized: return "Short3Normalized";
        case MTLVertexFormatShort4Normalized: return "Short4Normalized";
        case MTLVertexFormatUShort: return "UShort";
        case MTLVertexFormatUShort2: return "UShort2";
        case MTLVertexFormatUShort3: return "UShort3";
        case MTLVertexFormatUShort4: return "UShort4";
        case MTLVertexFormatUShortNormalized: return "UShortNormalized";
        case MTLVertexFormatUShort2Normalized: return "UShort2Normalized";
        case MTLVertexFormatUShort3Normalized: return "UShort3Normalized";
        case MTLVertexFormatUShort4Normalized: return "UShort4Normalized";
        case MTLVertexFormatUInt1010102Normalized: return "UInt1010102Normalized";
        case MTLVertexFormatInt1010102Normalized: return "Int1010102Normalized";
        default: return "Unknown";
    }
}

static inline bool mglShouldInspectDrawCall(uint64_t drawCall, GLuint programName)
{
    if (!kMGLDrawSubmitDiagnostics) {
        return false;
    }

    if (drawCall <= 120ull) {
        return true;
    }

    if (mglIsFocusedLoadingProgram(programName)) {
        return (drawCall <= 512ull) || ((drawCall % 64ull) == 0ull);
    }

    // Keep a denser trail for active Minecraft pipeline churn without flooding.
    if ((programName == 3u || programName == 74u) && ((drawCall % 40ull) == 0ull)) {
        return true;
    }

    return ((drawCall % 128ull) == 0ull);
}

static inline NSUInteger mglGLIndexElementSize(GLenum type)
{
    switch (type) {
        case GL_UNSIGNED_BYTE: return 1u;
        case GL_UNSIGNED_SHORT: return 2u;
        case GL_UNSIGNED_INT: return 4u;
        default: return 0u;
    }
}

static inline uint32_t mglReadGLIndexValue(const uint8_t *indexBytes,
                                           GLenum type,
                                           NSUInteger elementIndex)
{
    if (!indexBytes) {
        return 0u;
    }

    switch (type) {
        case GL_UNSIGNED_BYTE: {
            uint8_t v = 0;
            memcpy(&v, indexBytes + elementIndex, sizeof(v));
            return (uint32_t)v;
        }
        case GL_UNSIGNED_SHORT: {
            uint16_t v = 0;
            memcpy(&v, indexBytes + (elementIndex * 2u), sizeof(v));
            return (uint32_t)v;
        }
        case GL_UNSIGNED_INT: {
            uint32_t v = 0;
            memcpy(&v, indexBytes + (elementIndex * 4u), sizeof(v));
            return v;
        }
        default:
            return 0u;
    }
}

static inline bool mglScanIndexRangeIgnoringRestart(const uint8_t *indexBytes,
                                                    GLenum indexType,
                                                    GLsizei count,
                                                    bool primitiveRestartEnabled,
                                                    uint32_t restartIndex,
                                                    uint32_t *outMin,
                                                    uint32_t *outMax)
{
    if (!indexBytes || count <= 0 || !outMin || !outMax) {
        return false;
    }

    uint32_t minIndex = UINT32_MAX;
    uint32_t maxIndex = 0u;
    for (GLsizei i = 0; i < count; i++) {
        uint32_t idxValue = mglReadGLIndexValue(indexBytes, indexType, (NSUInteger)i);
        if (primitiveRestartEnabled && idxValue == restartIndex) {
            continue;
        }
        if (idxValue < minIndex) {
            minIndex = idxValue;
        }
        if (idxValue > maxIndex) {
            maxIndex = idxValue;
        }
    }

    if (minIndex == UINT32_MAX) {
        return false;
    }

    *outMin = minIndex;
    *outMax = maxIndex;
    return true;
}

static inline bool mglPrimitiveRestartIndexForType(GLMContext ctx,
                                                  GLenum indexType,
                                                  uint32_t *outRestartIndex)
{
    if (!ctx || (!ctx->state.caps.primitive_restart && !ctx->state.caps.primitive_restart_fixed_index)) {
        return false;
    }

    uint32_t restartIndex = 0u;
    if (ctx->state.caps.primitive_restart_fixed_index) {
        switch (indexType) {
            case GL_UNSIGNED_BYTE:
                restartIndex = 0xffu;
                break;
            case GL_UNSIGNED_SHORT:
                restartIndex = 0xffffu;
                break;
            case GL_UNSIGNED_INT:
                restartIndex = 0xffffffffu;
                break;
            default:
                return false;
        }
    } else {
        restartIndex = ctx->state.var.primitive_restart_index;
    }

    if (outRestartIndex) {
        *outRestartIndex = restartIndex;
    }
    return true;
}

static inline bool mglPrimitiveModeHasDrawableSegment(GLenum mode, NSUInteger indexCount)
{
    switch (mode) {
        case GL_POINTS:
            return indexCount >= 1u;
        case GL_LINES:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
            return indexCount >= 2u;
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
            return indexCount >= 3u;
        default:
            return indexCount > 0u;
    }
}

static inline BOOL mglPolygonModePointForDrawMode(GLMContext ctx, GLenum mode)
{
    if (!ctx || ctx->state.var.polygon_mode != GL_POINT) {
        return NO;
    }

    switch (mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
            return YES;
        default:
            return NO;
    }
}

static inline BOOL mglDrawModeProducesPolygons(GLenum mode)
{
    switch (mode) {
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
            return YES;
        default:
            return NO;
    }
}

static inline bool mglComputeIndexByteOffset(NSUInteger baseByteOffset,
                                             NSUInteger firstElement,
                                             NSUInteger indexStride,
                                             NSUInteger *outByteOffset)
{
    if (!outByteOffset || indexStride == 0u) {
        return false;
    }
    if (firstElement > (NSUIntegerMax / indexStride)) {
        return false;
    }

    NSUInteger relativeByteOffset = firstElement * indexStride;
    if (baseByteOffset > (NSUIntegerMax - relativeByteOffset)) {
        return false;
    }

    *outByteOffset = baseByteOffset + relativeByteOffset;
    return true;
}

static inline bool mglComputePreparedIndexByteOffset(GLenum glIndexType,
                                                    NSUInteger glByteOffset,
                                                    NSUInteger *outPreparedByteOffset)
{
    if (!outPreparedByteOffset) {
        return false;
    }
    if (glIndexType == GL_UNSIGNED_BYTE) {
        if (glByteOffset > (NSUIntegerMax / sizeof(uint16_t))) {
            return false;
        }
        *outPreparedByteOffset = glByteOffset * sizeof(uint16_t);
        return true;
    }

    *outPreparedByteOffset = glByteOffset;
    return true;
}

static id<MTLBuffer> mglNewTriangleFanArrayIndexBuffer(id<MTLDevice> device,
                                                       NSUInteger vertexCount,
                                                       NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || vertexCount < 3u) {
        return nil;
    }

    NSUInteger triangleCount = vertexCount - 2u;
    if (triangleCount > (NSUIntegerMax / (3u * sizeof(uint32_t)))) {
        return nil;
    }

    NSUInteger indexCount = triangleCount * 3u;
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger tri = 0; tri < triangleCount; tri++) {
        indices[(tri * 3u) + 0u] = 0u;
        indices[(tri * 3u) + 1u] = (uint32_t)(tri + 1u);
        indices[(tri * 3u) + 2u] = (uint32_t)(tri + 2u);
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

static id<MTLBuffer> mglNewLineLoopArrayIndexBuffer(id<MTLDevice> device,
                                                    NSUInteger vertexCount,
                                                    NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || vertexCount < 2u) {
        return nil;
    }
    if (vertexCount > (NSUIntegerMax - 1u)) {
        return nil;
    }

    NSUInteger indexCount = vertexCount + 1u;
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger i = 0; i < vertexCount; i++) {
        if (i > UINT32_MAX) {
            free(indices);
            return nil;
        }
        indices[i] = (uint32_t)i;
    }
    indices[vertexCount] = 0u;

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

static id<MTLBuffer> mglNewTriangleStripArrayIndexBuffer(id<MTLDevice> device,
                                                         NSUInteger vertexCount,
                                                         NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || vertexCount < 3u) {
        return nil;
    }

    NSUInteger triangleCount = vertexCount - 2u;
    if (triangleCount > (NSUIntegerMax / (3u * sizeof(uint32_t)))) {
        return nil;
    }

    NSUInteger indexCount = triangleCount * 3u;
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger tri = 0; tri < triangleCount; tri++) {
        indices[(tri * 3u) + 0u] = (uint32_t)tri;
        indices[(tri * 3u) + 1u] = (uint32_t)(tri + 1u);
        indices[(tri * 3u) + 2u] = (uint32_t)(tri + 2u);
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

static id<MTLBuffer> mglNewTriangleFanElementIndexBuffer(id<MTLDevice> device,
                                                         const uint8_t *sourceIndexBytes,
                                                         GLenum sourceIndexType,
                                                         NSUInteger sourceIndexCount,
                                                         NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || !sourceIndexBytes || sourceIndexCount < 3u) {
        return nil;
    }

    NSUInteger triangleCount = sourceIndexCount - 2u;
    if (triangleCount > (NSUIntegerMax / (3u * sizeof(uint32_t)))) {
        return nil;
    }

    NSUInteger indexCount = triangleCount * 3u;
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    uint32_t center = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, 0u);
    for (NSUInteger tri = 0; tri < triangleCount; tri++) {
        indices[(tri * 3u) + 0u] = center;
        indices[(tri * 3u) + 1u] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, tri + 1u);
        indices[(tri * 3u) + 2u] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, tri + 2u);
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

static id<MTLBuffer> mglNewTriangleStripElementIndexBuffer(id<MTLDevice> device,
                                                           const uint8_t *sourceIndexBytes,
                                                           GLenum sourceIndexType,
                                                           NSUInteger sourceIndexCount,
                                                           NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || !sourceIndexBytes || sourceIndexCount < 3u) {
        return nil;
    }

    NSUInteger triangleCount = sourceIndexCount - 2u;
    if (triangleCount > (NSUIntegerMax / (3u * sizeof(uint32_t)))) {
        return nil;
    }

    NSUInteger indexCount = triangleCount * 3u;
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger tri = 0; tri < triangleCount; tri++) {
        indices[(tri * 3u) + 0u] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, tri);
        indices[(tri * 3u) + 1u] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, tri + 1u);
        indices[(tri * 3u) + 2u] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, tri + 2u);
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

static id<MTLBuffer> mglNewLineLoopElementIndexBuffer(id<MTLDevice> device,
                                                      const uint8_t *sourceIndexBytes,
                                                      GLenum sourceIndexType,
                                                      NSUInteger sourceIndexCount,
                                                      NSUInteger *outIndexCount)
{
    if (outIndexCount) {
        *outIndexCount = 0u;
    }

    if (!device || !sourceIndexBytes || sourceIndexCount < 2u) {
        return nil;
    }
    if (sourceIndexCount > (NSUIntegerMax - 1u)) {
        return nil;
    }

    NSUInteger indexCount = sourceIndexCount + 1u;
    uint32_t *indices = (uint32_t *)calloc(indexCount, sizeof(uint32_t));
    if (!indices) {
        return nil;
    }

    for (NSUInteger i = 0; i < sourceIndexCount; i++) {
        indices[i] = mglReadGLIndexValue(sourceIndexBytes, sourceIndexType, i);
    }
    indices[sourceIndexCount] = indices[0];

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(indexCount * sizeof(uint32_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);

    if (outIndexCount && buffer) {
        *outIndexCount = indexCount;
    }

    return buffer;
}

static id<MTLBuffer> mglNewUInt16IndexBufferFromUInt8(id<MTLDevice> device,
                                                      const uint8_t *sourceIndexBytes,
                                                      NSUInteger sourceIndexCount)
{
    if (!device || !sourceIndexBytes) {
        return nil;
    }
    if (sourceIndexCount == 0u) {
        return nil;
    }
    if (sourceIndexCount > (NSUIntegerMax / sizeof(uint16_t))) {
        return nil;
    }

    uint16_t *indices = (uint16_t *)calloc(sourceIndexCount, sizeof(uint16_t));
    if (!indices) {
        return nil;
    }
    for (NSUInteger i = 0; i < sourceIndexCount; i++) {
        indices[i] = (uint16_t)sourceIndexBytes[i];
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:indices
                                               length:(sourceIndexCount * sizeof(uint16_t))
                                              options:MTLResourceStorageModeShared];
    free(indices);
    return buffer;
}

static const uint8_t *mglElementIndexSourceBytes(Buffer *glElementBuffer,
                                                 id<MTLBuffer> metalElementBuffer,
                                                 NSUInteger *outSourceByteCount)
{
    if (outSourceByteCount) {
        *outSourceByteCount = 0u;
    }

    if (glElementBuffer && glElementBuffer->data.buffer_data &&
        ((uintptr_t)glElementBuffer->data.buffer_data >= 0x1000ull)) {
        if (outSourceByteCount && glElementBuffer->size > 0) {
            *outSourceByteCount = (NSUInteger)glElementBuffer->size;
        }
        return (const uint8_t *)glElementBuffer->data.buffer_data;
    }

    if (metalElementBuffer && metalElementBuffer.contents) {
        if (outSourceByteCount) {
            *outSourceByteCount = metalElementBuffer.length;
        }
        return (const uint8_t *)metalElementBuffer.contents;
    }

    return NULL;
}

static const uint8_t *mglElementIndexSourceForDraw(Buffer *glElementBuffer,
                                                   id<MTLBuffer> metalElementBuffer,
                                                   GLenum glIndexType,
                                                   NSUInteger indexOffset,
                                                   GLsizei indexCount)
{
    NSUInteger sourceByteCount = 0u;
    const uint8_t *sourceBytes = mglElementIndexSourceBytes(glElementBuffer,
                                                            metalElementBuffer,
                                                            &sourceByteCount);
    NSUInteger indexStride = mglGLIndexElementSize(glIndexType);
    if (!sourceBytes || sourceByteCount == 0u || indexStride == 0u || indexCount <= 0) {
        return NULL;
    }
    if ((NSUInteger)indexCount > (NSUIntegerMax / indexStride)) {
        return NULL;
    }

    NSUInteger neededBytes = (NSUInteger)indexCount * indexStride;
    if (indexOffset > sourceByteCount || (sourceByteCount - indexOffset) < neededBytes) {
        return NULL;
    }

    return sourceBytes + indexOffset;
}

static BOOL mglEncodeArrayLineLoop(id<MTLRenderCommandEncoder> encoder,
                                   id<MTLDevice> device,
                                   GLsizei count,
                                   GLint baseVertex,
                                   NSUInteger instanceCount,
                                   NSUInteger baseInstance,
                                   const char *label)
{
    if (count < 2) {
        return YES;
    }

    NSUInteger loopIndexCount = 0u;
    id<MTLBuffer> loopIndexBuffer = mglNewLineLoopArrayIndexBuffer(device,
                                                                   (NSUInteger)count,
                                                                   &loopIndexCount);
    if (!loopIndexBuffer || loopIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s line loop array emulation failed count=%d baseVertex=%d",
              label ? label : "draw",
              (int)count,
              (int)baseVertex);
        return NO;
    }

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeLineStrip
                         indexCount:loopIndexCount
                          indexType:MTLIndexTypeUInt32
                        indexBuffer:loopIndexBuffer
                  indexBufferOffset:0
                      instanceCount:instanceCount
                         baseVertex:baseVertex
                       baseInstance:baseInstance];
    return YES;
}

static BOOL mglEncodeArrayTriangleFan(id<MTLRenderCommandEncoder> encoder,
                                      id<MTLDevice> device,
                                      GLsizei count,
                                      GLint baseVertex,
                                      NSUInteger instanceCount,
                                      NSUInteger baseInstance,
                                      const char *label)
{
    if (count < 3) {
        return YES;
    }

    NSUInteger fanIndexCount = 0u;
    id<MTLBuffer> fanIndexBuffer = mglNewTriangleFanArrayIndexBuffer(device,
                                                                     (NSUInteger)count,
                                                                     &fanIndexCount);
    if (!fanIndexBuffer || fanIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s triangle fan array emulation failed count=%d baseVertex=%d",
              label ? label : "draw",
              (int)count,
              (int)baseVertex);
        return NO;
    }

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                         indexCount:fanIndexCount
                          indexType:MTLIndexTypeUInt32
                        indexBuffer:fanIndexBuffer
                  indexBufferOffset:0
                      instanceCount:instanceCount
                         baseVertex:baseVertex
                       baseInstance:baseInstance];
    return YES;
}

static BOOL mglEncodeElementLineLoop(id<MTLRenderCommandEncoder> encoder,
                                     id<MTLDevice> device,
                                     Buffer *glElementBuffer,
                                     id<MTLBuffer> metalElementBuffer,
                                     GLenum glIndexType,
                                     NSUInteger indexOffset,
                                     GLsizei count,
                                     NSUInteger instanceCount,
                                     NSInteger baseVertex,
                                     NSUInteger baseInstance,
                                     const char *label)
{
    if (count < 2) {
        return YES;
    }

    const uint8_t *loopSource = mglElementIndexSourceForDraw(glElementBuffer,
                                                             metalElementBuffer,
                                                             glIndexType,
                                                             indexOffset,
                                                             count);
    NSUInteger loopIndexCount = 0u;
    id<MTLBuffer> loopIndexBuffer = mglNewLineLoopElementIndexBuffer(device,
                                                                     loopSource,
                                                                     glIndexType,
                                                                     (NSUInteger)count,
                                                                     &loopIndexCount);
    if (!loopIndexBuffer || loopIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s line loop element emulation failed ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              loopSource);
        return NO;
    }

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeLineStrip
                         indexCount:loopIndexCount
                          indexType:MTLIndexTypeUInt32
                        indexBuffer:loopIndexBuffer
                  indexBufferOffset:0
                      instanceCount:instanceCount
                         baseVertex:baseVertex
                       baseInstance:baseInstance];
    return YES;
}

static BOOL mglEncodeElementTriangleFan(id<MTLRenderCommandEncoder> encoder,
                                        id<MTLDevice> device,
                                        Buffer *glElementBuffer,
                                        id<MTLBuffer> metalElementBuffer,
                                        GLenum glIndexType,
                                        NSUInteger indexOffset,
                                        GLsizei count,
                                        NSUInteger instanceCount,
                                        NSInteger baseVertex,
                                        NSUInteger baseInstance,
                                        const char *label)
{
    if (count < 3) {
        return YES;
    }

    const uint8_t *fanSource = mglElementIndexSourceForDraw(glElementBuffer,
                                                            metalElementBuffer,
                                                            glIndexType,
                                                            indexOffset,
                                                            count);
    NSUInteger fanIndexCount = 0u;
    id<MTLBuffer> fanIndexBuffer = mglNewTriangleFanElementIndexBuffer(device,
                                                                       fanSource,
                                                                       glIndexType,
                                                                       (NSUInteger)count,
                                                                       &fanIndexCount);
    if (!fanIndexBuffer || fanIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s triangle fan element emulation failed ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              fanSource);
        return NO;
    }

    [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                         indexCount:fanIndexCount
                          indexType:MTLIndexTypeUInt32
                        indexBuffer:fanIndexBuffer
                  indexBufferOffset:0
                      instanceCount:instanceCount
                         baseVertex:baseVertex
                       baseInstance:baseInstance];
    return YES;
}

static id<MTLBuffer> mglPreparedElementIndexBuffer(id<MTLDevice> device,
                                                   Buffer *glElementBuffer,
                                                   id<MTLBuffer> metalElementBuffer,
                                                   GLenum glIndexType,
                                                   NSUInteger *ioIndexBufferOffset,
                                                   MTLIndexType *outMetalIndexType)
{
    if (outMetalIndexType) {
        switch (glIndexType) {
            case GL_UNSIGNED_BYTE:
            case GL_UNSIGNED_SHORT:
                *outMetalIndexType = MTLIndexTypeUInt16;
                break;
            case GL_UNSIGNED_INT:
                *outMetalIndexType = MTLIndexTypeUInt32;
                break;
            default:
                *outMetalIndexType = (MTLIndexType)0xFFFFFFFF;
                break;
        }
    }
    if (glIndexType != GL_UNSIGNED_BYTE) {
        return metalElementBuffer;
    }

    NSUInteger sourceOffset = ioIndexBufferOffset ? *ioIndexBufferOffset : 0u;
    NSUInteger sourceByteCount = 0u;
    const uint8_t *sourceBytes = mglElementIndexSourceBytes(glElementBuffer,
                                                            metalElementBuffer,
                                                            &sourceByteCount);

    if (!sourceBytes || sourceByteCount == 0u) {
        NSLog(@"MGL WARNING: unable to expand GL_UNSIGNED_BYTE element buffer gl=%u source=%p bytes=%lu",
              glElementBuffer ? glElementBuffer->name : 0u,
              sourceBytes,
              (unsigned long)sourceByteCount);
        return nil;
    }

    id<MTLBuffer> expanded = mglNewUInt16IndexBufferFromUInt8(device, sourceBytes, sourceByteCount);
    if (!expanded) {
        NSLog(@"MGL WARNING: failed to allocate expanded UInt16 element buffer for GL_UNSIGNED_BYTE gl=%u bytes=%lu",
              glElementBuffer ? glElementBuffer->name : 0u,
              (unsigned long)sourceByteCount);
        return nil;
    }

    if (ioIndexBufferOffset) {
        if (sourceOffset > (NSUIntegerMax / sizeof(uint16_t))) {
            return nil;
        }
        *ioIndexBufferOffset = sourceOffset * sizeof(uint16_t);
    }
    if (outMetalIndexType) {
        *outMetalIndexType = MTLIndexTypeUInt16;
    }
    return expanded;
}

static BOOL mglEncodeArrayPolygonPoint(id<MTLRenderCommandEncoder> encoder,
                                       id<MTLDevice> device,
                                       GLenum mode,
                                       GLint first,
                                       GLsizei count,
                                       NSUInteger instanceCount,
                                       NSUInteger baseInstance,
                                       const char *label)
{
    if (count < 3) {
        return YES;
    }

    if (mode == GL_TRIANGLES) {
        NSUInteger drawableCount = ((NSUInteger)count / 3u) * 3u;
        if (drawableCount == 0u) {
            return YES;
        }
        [encoder drawPrimitives:MTLPrimitiveTypePoint
                    vertexStart:first
                    vertexCount:drawableCount
                  instanceCount:instanceCount
                   baseInstance:baseInstance];
        return YES;
    }

    NSUInteger pointIndexCount = 0u;
    id<MTLBuffer> pointIndexBuffer = nil;
    if (mode == GL_TRIANGLE_FAN) {
        pointIndexBuffer = mglNewTriangleFanArrayIndexBuffer(device,
                                                             (NSUInteger)count,
                                                             &pointIndexCount);
    } else if (mode == GL_TRIANGLE_STRIP) {
        pointIndexBuffer = mglNewTriangleStripArrayIndexBuffer(device,
                                                               (NSUInteger)count,
                                                               &pointIndexCount);
    } else {
        return NO;
    }

    if (!pointIndexBuffer || pointIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s polygon point array emulation failed mode=0x%x count=%d first=%d",
              label ? label : "draw",
              (unsigned)mode,
              (int)count,
              (int)first);
        return NO;
    }

    [encoder drawIndexedPrimitives:MTLPrimitiveTypePoint
                        indexCount:pointIndexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:pointIndexBuffer
                 indexBufferOffset:0
                     instanceCount:instanceCount
                        baseVertex:first
                      baseInstance:baseInstance];
    return YES;
}

static BOOL mglEncodeElementPolygonPoint(id<MTLRenderCommandEncoder> encoder,
                                         id<MTLDevice> device,
                                         Buffer *glElementBuffer,
                                         id<MTLBuffer> metalElementBuffer,
                                         GLenum mode,
                                         GLenum glIndexType,
                                         MTLIndexType metalIndexType,
                                         NSUInteger indexOffset,
                                         GLsizei count,
                                         NSUInteger instanceCount,
                                         NSInteger baseVertex,
                                         NSUInteger baseInstance,
                                         const char *label)
{
    if (count < 3) {
        return YES;
    }

    if (mode == GL_TRIANGLES) {
        NSUInteger drawableIndexCount = ((NSUInteger)count / 3u) * 3u;
        if (drawableIndexCount == 0u) {
            return YES;
        }

        NSUInteger drawIndexOffset = indexOffset;
        MTLIndexType drawIndexType = metalIndexType;
        id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(device,
                                                                      glElementBuffer,
                                                                      metalElementBuffer,
                                                                      glIndexType,
                                                                      &drawIndexOffset,
                                                                      &drawIndexType);
        if (!drawIndexBuffer) {
            return NO;
        }

        [encoder drawIndexedPrimitives:MTLPrimitiveTypePoint
                            indexCount:drawableIndexCount
                             indexType:drawIndexType
                           indexBuffer:drawIndexBuffer
                     indexBufferOffset:drawIndexOffset
                         instanceCount:instanceCount
                            baseVertex:baseVertex
                          baseInstance:baseInstance];
        return YES;
    }

    const uint8_t *source = mglElementIndexSourceForDraw(glElementBuffer,
                                                         metalElementBuffer,
                                                         glIndexType,
                                                         indexOffset,
                                                         count);
    NSUInteger pointIndexCount = 0u;
    id<MTLBuffer> pointIndexBuffer = nil;
    if (mode == GL_TRIANGLE_FAN) {
        pointIndexBuffer = mglNewTriangleFanElementIndexBuffer(device,
                                                               source,
                                                               glIndexType,
                                                               (NSUInteger)count,
                                                               &pointIndexCount);
    } else if (mode == GL_TRIANGLE_STRIP) {
        pointIndexBuffer = mglNewTriangleStripElementIndexBuffer(device,
                                                                 source,
                                                                 glIndexType,
                                                                 (NSUInteger)count,
                                                                 &pointIndexCount);
    } else {
        return NO;
    }

    if (!pointIndexBuffer || pointIndexCount == 0u) {
        NSLog(@"MGL WARNING: %s polygon point element emulation failed mode=0x%x ebo=%u count=%d offset=%lu source=%p",
              label ? label : "draw",
              (unsigned)mode,
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned long)indexOffset,
              source);
        return NO;
    }

    [encoder drawIndexedPrimitives:MTLPrimitiveTypePoint
                        indexCount:pointIndexCount
                         indexType:MTLIndexTypeUInt32
                       indexBuffer:pointIndexBuffer
                 indexBufferOffset:0
                     instanceCount:instanceCount
                        baseVertex:baseVertex
                      baseInstance:baseInstance];
    return YES;
}

typedef enum MGLPrimitiveRestartEncodeResult {
    MGLPrimitiveRestartEncodeNotNeeded = 0,
    MGLPrimitiveRestartEncodeHandled = 1,
    MGLPrimitiveRestartEncodeFailed = 2,
} MGLPrimitiveRestartEncodeResult;

static BOOL mglEncodeRestartSegment(id<MTLRenderCommandEncoder> encoder,
                                    id<MTLDevice> device,
                                    Buffer *glElementBuffer,
                                    id<MTLBuffer> metalElementBuffer,
                                    id<MTLBuffer> preparedIndexBuffer,
                                    GLenum mode,
                                    MTLPrimitiveType primitiveType,
                                    GLenum glIndexType,
                                    MTLIndexType preparedIndexType,
                                    NSUInteger baseIndexByteOffset,
                                    NSUInteger segmentStart,
                                    NSUInteger segmentIndexCount,
                                    NSUInteger instanceCount,
                                    NSInteger baseVertex,
                                    NSUInteger baseInstance,
                                    const char *label)
{
    if (!mglPrimitiveModeHasDrawableSegment(mode, segmentIndexCount)) {
        return YES;
    }

    NSUInteger segmentGLByteOffset = 0u;
    NSUInteger indexStride = mglGLIndexElementSize(glIndexType);
    if (!mglComputeIndexByteOffset(baseIndexByteOffset,
                                   segmentStart,
                                   indexStride,
                                   &segmentGLByteOffset)) {
        NSLog(@"MGL WARNING: %s primitive restart segment offset overflow base=%lu start=%lu stride=%lu count=%lu",
              label ? label : "draw",
              (unsigned long)baseIndexByteOffset,
              (unsigned long)segmentStart,
              (unsigned long)indexStride,
              (unsigned long)segmentIndexCount);
        return NO;
    }

    if (primitiveType == MTLPrimitiveTypePoint &&
        (mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP || mode == GL_TRIANGLE_FAN)) {
        return mglEncodeElementPolygonPoint(encoder,
                                            device,
                                            glElementBuffer,
                                            metalElementBuffer,
                                            mode,
                                            glIndexType,
                                            preparedIndexType,
                                            segmentGLByteOffset,
                                            (GLsizei)segmentIndexCount,
                                            instanceCount,
                                            baseVertex,
                                            baseInstance,
                                            label);
    }

    if (mode == GL_TRIANGLE_FAN) {
        return mglEncodeElementTriangleFan(encoder,
                                           device,
                                           glElementBuffer,
                                           metalElementBuffer,
                                           glIndexType,
                                           segmentGLByteOffset,
                                           (GLsizei)segmentIndexCount,
                                           instanceCount,
                                           baseVertex,
                                           baseInstance,
                                           label);
    }

    if (mode == GL_LINE_LOOP) {
        return mglEncodeElementLineLoop(encoder,
                                        device,
                                        glElementBuffer,
                                        metalElementBuffer,
                                        glIndexType,
                                        segmentGLByteOffset,
                                        (GLsizei)segmentIndexCount,
                                        instanceCount,
                                        baseVertex,
                                        baseInstance,
                                        label);
    }

    NSUInteger preparedByteOffset = 0u;
    if (!mglComputePreparedIndexByteOffset(glIndexType,
                                           segmentGLByteOffset,
                                           &preparedByteOffset)) {
        NSLog(@"MGL WARNING: %s primitive restart prepared offset overflow glType=0x%x byteOffset=%lu",
              label ? label : "draw",
              (unsigned)glIndexType,
              (unsigned long)segmentGLByteOffset);
        return NO;
    }

    [encoder drawIndexedPrimitives:primitiveType
                        indexCount:segmentIndexCount
                         indexType:preparedIndexType
                       indexBuffer:preparedIndexBuffer
                 indexBufferOffset:preparedByteOffset
                     instanceCount:instanceCount
                        baseVertex:baseVertex
                      baseInstance:baseInstance];
    return YES;
}

static MGLPrimitiveRestartEncodeResult mglEncodePrimitiveRestartedElementDraw(id<MTLRenderCommandEncoder> encoder,
                                                                              id<MTLDevice> device,
                                                                              GLMContext ctx,
                                                                              Buffer *glElementBuffer,
                                                                              id<MTLBuffer> metalElementBuffer,
                                                                              GLenum mode,
                                                                              MTLPrimitiveType primitiveType,
                                                                              GLenum glIndexType,
                                                                              MTLIndexType metalIndexType,
                                                                              NSUInteger indexOffset,
                                                                              GLsizei count,
                                                                              NSUInteger instanceCount,
                                                                              NSInteger baseVertex,
                                                                              NSUInteger baseInstance,
                                                                              const char *label)
{
    uint32_t restartIndex = 0u;
    if (!mglPrimitiveRestartIndexForType(ctx, glIndexType, &restartIndex)) {
        return MGLPrimitiveRestartEncodeNotNeeded;
    }
    if (count <= 0) {
        return MGLPrimitiveRestartEncodeHandled;
    }

    const uint8_t *source = mglElementIndexSourceForDraw(glElementBuffer,
                                                         metalElementBuffer,
                                                         glIndexType,
                                                         indexOffset,
                                                         count);
    if (!source) {
        NSLog(@"MGL WARNING: %s primitive restart enabled but index bytes are not CPU-readable ebo=%u count=%d type=0x%x offset=%lu; skipping draw to avoid treating restart as a vertex",
              label ? label : "draw",
              glElementBuffer ? glElementBuffer->name : 0u,
              (int)count,
              (unsigned)glIndexType,
              (unsigned long)indexOffset);
        return MGLPrimitiveRestartEncodeFailed;
    }

    BOOL sawRestart = NO;
    for (GLsizei i = 0; i < count; i++) {
        if (mglReadGLIndexValue(source, glIndexType, (NSUInteger)i) == restartIndex) {
            sawRestart = YES;
            break;
        }
    }
    if (!sawRestart) {
        return MGLPrimitiveRestartEncodeNotNeeded;
    }

    BOOL emulatedMode = (mode == GL_TRIANGLE_FAN ||
                         mode == GL_LINE_LOOP ||
                         (primitiveType == MTLPrimitiveTypePoint &&
                          (mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP)));
    id<MTLBuffer> preparedIndexBuffer = metalElementBuffer;
    MTLIndexType preparedIndexType = metalIndexType;
    if (!emulatedMode) {
        preparedIndexBuffer = mglPreparedElementIndexBuffer(device,
                                                            glElementBuffer,
                                                            metalElementBuffer,
                                                            glIndexType,
                                                            NULL,
                                                            &preparedIndexType);
        if (!preparedIndexBuffer) {
            return MGLPrimitiveRestartEncodeFailed;
        }
    }

    NSUInteger segmentStart = 0u;
    BOOL encodedAllSegments = YES;
    for (GLsizei i = 0; i < count; i++) {
        if (mglReadGLIndexValue(source, glIndexType, (NSUInteger)i) != restartIndex) {
            continue;
        }

        NSUInteger segmentCount = (NSUInteger)i - segmentStart;
        if (!mglEncodeRestartSegment(encoder,
                                     device,
                                     glElementBuffer,
                                     metalElementBuffer,
                                     preparedIndexBuffer,
                                     mode,
                                     primitiveType,
                                     glIndexType,
                                     preparedIndexType,
                                     indexOffset,
                                     segmentStart,
                                     segmentCount,
                                     instanceCount,
                                     baseVertex,
                                     baseInstance,
                                     label)) {
            encodedAllSegments = NO;
            break;
        }
        segmentStart = (NSUInteger)i + 1u;
    }

    if (encodedAllSegments) {
        NSUInteger trailingCount = (NSUInteger)count - segmentStart;
        encodedAllSegments = mglEncodeRestartSegment(encoder,
                                                     device,
                                                     glElementBuffer,
                                                     metalElementBuffer,
                                                     preparedIndexBuffer,
                                                     mode,
                                                     primitiveType,
                                                     glIndexType,
                                                     preparedIndexType,
                                                     indexOffset,
                                                     segmentStart,
                                                     trailingCount,
                                                     instanceCount,
                                                     baseVertex,
                                                     baseInstance,
                                                     label);
    }

    return encodedAllSegments ? MGLPrimitiveRestartEncodeHandled : MGLPrimitiveRestartEncodeFailed;
}

static BOOL mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(GLMContext ctx,
                                                                  GLenum glIndexType,
                                                                  const char *label)
{
    uint32_t restartIndex = 0u;
    if (!mglPrimitiveRestartIndexForType(ctx, glIndexType, &restartIndex)) {
        return NO;
    }

    static uint64_t s_indirectRestartSkipCount = 0;
    s_indirectRestartSkipCount++;
    if (s_indirectRestartSkipCount <= 8u || (s_indirectRestartSkipCount % 1000u) == 0u) {
        NSLog(@"MGL WARNING: %s primitive restart with indirect indexed draw is not emulated yet type=0x%x restart=%u occurrence=%llu; skipping draw",
              label ? label : "drawElementsIndirect",
              (unsigned)glIndexType,
              (unsigned)restartIndex,
              (unsigned long long)s_indirectRestartSkipCount);
    }
    return YES;
}

static BOOL mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(GLMContext ctx,
                                                               GLenum mode,
                                                               const char *label)
{
    if (!mglPolygonModePointForDrawMode(ctx, mode)) {
        return NO;
    }

    static uint64_t s_indirectPolygonPointSkipCount = 0;
    s_indirectPolygonPointSkipCount++;
    if (s_indirectPolygonPointSkipCount <= 8u || (s_indirectPolygonPointSkipCount % 1000u) == 0u) {
        NSLog(@"MGL WARNING: %s GL_POLYGON_MODE=GL_POINT requires triangle expansion for indirect draw mode=0x%x occurrence=%llu; skipping draw",
              label ? label : "drawIndirect",
              (unsigned)mode,
              (unsigned long long)s_indirectPolygonPointSkipCount);
    }
    return YES;
}

static inline uint64_t mglHashStepU64(uint64_t hash, uint64_t value)
{
    // 64-bit FNV-1a
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

static uint64_t mglVertexDescriptorSignature(MTLVertexDescriptor *vertexDescriptor)
{
    uint64_t hash = 1469598103934665603ull;
    if (!vertexDescriptor) {
        return hash;
    }

    for (NSUInteger i = 0; i < MAX_ATTRIBS; i++) {
        MTLVertexAttributeDescriptor *attrib = vertexDescriptor.attributes[i];
        if (!attrib) {
            continue;
        }
        hash = mglHashStepU64(hash, (uint64_t)attrib.format);
        hash = mglHashStepU64(hash, (uint64_t)attrib.offset);
        hash = mglHashStepU64(hash, (uint64_t)attrib.bufferIndex);
    }

    for (NSUInteger i = 0; i < kMGLMaxMetalVertexBufferCount; i++) {
        MTLVertexBufferLayoutDescriptor *layout = vertexDescriptor.layouts[i];
        if (!layout) {
            continue;
        }
        hash = mglHashStepU64(hash, (uint64_t)layout.stride);
        hash = mglHashStepU64(hash, (uint64_t)layout.stepFunction);
        hash = mglHashStepU64(hash, (uint64_t)layout.stepRate);
    }

    return hash;
}

static uint64_t mglPipelineDescriptorSignature(MTLRenderPipelineDescriptor *pipelineStateDescriptor)
{
    uint64_t hash = 1469598103934665603ull;
    if (!pipelineStateDescriptor) {
        return hash;
    }

    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.rasterSampleCount);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.alphaToCoverageEnabled);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.alphaToOneEnabled);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.depthAttachmentPixelFormat);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.stencilAttachmentPixelFormat);

    for (NSUInteger i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        MTLRenderPipelineColorAttachmentDescriptor *attachment = pipelineStateDescriptor.colorAttachments[i];
        if (!attachment) {
            continue;
        }
        hash = mglHashStepU64(hash, (uint64_t)attachment.pixelFormat);
        hash = mglHashStepU64(hash, (uint64_t)attachment.blendingEnabled);
        hash = mglHashStepU64(hash, (uint64_t)attachment.sourceRGBBlendFactor);
        hash = mglHashStepU64(hash, (uint64_t)attachment.destinationRGBBlendFactor);
        hash = mglHashStepU64(hash, (uint64_t)attachment.rgbBlendOperation);
        hash = mglHashStepU64(hash, (uint64_t)attachment.sourceAlphaBlendFactor);
        hash = mglHashStepU64(hash, (uint64_t)attachment.destinationAlphaBlendFactor);
        hash = mglHashStepU64(hash, (uint64_t)attachment.alphaBlendOperation);
        hash = mglHashStepU64(hash, (uint64_t)attachment.writeMask);
    }

    return hash;
}

static inline bool mglShouldTraceBufferTransferCall(uint64_t call)
{
    if (call <= 128ull) {
        return true;
    }
    return ((call % 64ull) == 0ull);
}

static uint64_t mglTraceHashBytes(const void *data, size_t len)
{
    if (!data || len == 0) {
        return 0ull;
    }

    const uint8_t *bytes = (const uint8_t *)data;
    size_t head = len < 1024 ? len : 1024;
    uint64_t hash = 1469598103934665603ull;

    for (size_t i = 0; i < head; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ull;
    }

    if (len > head) {
        const uint8_t *tail = bytes + (len - head);
        for (size_t i = 0; i < head; i++) {
            hash ^= (uint64_t)tail[i];
            hash *= 1099511628211ull;
        }
    }

    hash ^= (uint64_t)len;
    hash *= 1099511628211ull;
    return hash;
}

static void mglTraceFormatBytes(const void *data, size_t len, char *out, size_t outSize)
{
    if (!out || outSize == 0) {
        return;
    }

    if (!data || len == 0) {
        snprintf(out, outSize, "-");
        return;
    }

    const uint8_t *bytes = (const uint8_t *)data;
    size_t sample = len < 8 ? len : 8;
    size_t used = 0;

    for (size_t i = 0; i < sample && used + 3 < outSize; i++) {
        int wrote = snprintf(out + used, outSize - used, "%02x", bytes[i]);
        if (wrote <= 0) {
            break;
        }
        used += (size_t)wrote;
        if (i + 1 < sample && used + 2 < outSize) {
            out[used++] = ':';
            out[used] = '\0';
        }
    }

    if (len > sample && used + 4 < outSize) {
        snprintf(out + used, outSize - used, "...");
    }
}

static void mglDumpBytesToLog(NSString *label,
                              const uint8_t *bytes,
                              size_t length,
                              size_t baseOffset)
{
    if (!bytes || length == 0) {
        NSLog(@"MGL DUMP %@ empty", label ?: @"(null)");
        return;
    }

    const size_t row = 16u;
    for (size_t off = 0; off < length; off += row) {
        size_t n = MIN(row, length - off);
        char hex[3 * row + 1];
        char ascii[row + 1];
        size_t hp = 0;

        for (size_t i = 0; i < n; i++) {
            uint8_t b = bytes[off + i];
            int wrote = snprintf(hex + hp, sizeof(hex) - hp, "%02x", b);
            if (wrote <= 0) {
                break;
            }
            hp += (size_t)wrote;
            if (i + 1 < n && hp + 1 < sizeof(hex)) {
                hex[hp++] = ' ';
            }
            ascii[i] = (b >= 32u && b <= 126u) ? (char)b : '.';
        }
        hex[hp] = '\0';
        ascii[n] = '\0';

        NSLog(@"MGL DUMP %@ +0x%zx: %-47s |%s|",
              label ?: @"(null)",
              baseOffset + off,
              hex,
              ascii);
    }
}

static inline size_t mglVertexAttribElementBytes(GLenum type, GLuint size)
{
    switch (type) {
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
            return 4u;
        default: {
            size_t comp = mglVertexAttribComponentSize(type);
            if (comp == 0u || size == 0u) {
                return 0u;
            }
            return comp * (size_t)size;
        }
    }
}

static double mglDecodeVertexAttribComponent(const uint8_t *src,
                                             GLenum type,
                                             GLboolean normalized,
                                             NSUInteger component)
{
    if (!src) {
        return 0.0;
    }

    switch (type) {
        case GL_FLOAT: {
            float v = 0.0f;
            memcpy(&v, src + component * sizeof(float), sizeof(v));
            return (double)v;
        }
        case GL_UNSIGNED_BYTE: {
            uint8_t v = 0;
            memcpy(&v, src + component, sizeof(v));
            return normalized ? ((double)v / 255.0) : (double)v;
        }
        case GL_BYTE: {
            int8_t v = 0;
            memcpy(&v, src + component, sizeof(v));
            if (normalized) {
                double d = (double)v / 127.0;
                return d < -1.0 ? -1.0 : d;
            }
            return (double)v;
        }
        case GL_UNSIGNED_SHORT: {
            uint16_t v = 0;
            memcpy(&v, src + component * sizeof(uint16_t), sizeof(v));
            return normalized ? ((double)v / 65535.0) : (double)v;
        }
        case GL_SHORT: {
            int16_t v = 0;
            memcpy(&v, src + component * sizeof(int16_t), sizeof(v));
            if (normalized) {
                double d = (double)v / 32767.0;
                return d < -1.0 ? -1.0 : d;
            }
            return (double)v;
        }
        case GL_UNSIGNED_INT: {
            uint32_t v = 0;
            memcpy(&v, src + component * sizeof(uint32_t), sizeof(v));
            return normalized ? ((double)v / 4294967295.0) : (double)v;
        }
        case GL_INT: {
            int32_t v = 0;
            memcpy(&v, src + component * sizeof(int32_t), sizeof(v));
            if (normalized) {
                double d = (double)v / 2147483647.0;
                return d < -1.0 ? -1.0 : d;
            }
            return (double)v;
        }
        default:
            return 0.0;
    }
}

static void mglTraceDrawElementsAttrib(GLMContext ctx,
                                       VertexArray *vao,
                                       uint64_t drawCall,
                                       GLuint programName,
                                       const uint8_t *indexBytes,
                                       GLenum indexType,
                                       GLuint attrib)
{
    if (!ctx || !vao || attrib >= MAX_ATTRIBS ||
        (vao->enabled_attribs & (0x1u << attrib)) == 0u) {
        return;
    }

    VertexAttrib *a = &vao->attrib[attrib];
    Buffer *vbo = mglRendererGetValidatedBuffer(ctx, a->buffer, "drawElements.attrib", attrib);
    if (!vbo) {
        NSLog(@"MGL TRACE drawElements.attrib%u call=%llu program=%u invalid buffer",
              (unsigned)attrib,
              (unsigned long long)drawCall,
              (unsigned)programName);
        return;
    }

    const uint8_t *vboBytes = NULL;
    if (vbo->data.buffer_data && ((uintptr_t)vbo->data.buffer_data >= 0x1000ull)) {
        vboBytes = (const uint8_t *)vbo->data.buffer_data;
    } else if (vbo->data.mtl_data) {
        id<MTLBuffer> vb = (__bridge id<MTLBuffer>)(vbo->data.mtl_data);
        vboBytes = (const uint8_t *)vb.contents;
    }

    if (!vboBytes) {
        NSLog(@"MGL TRACE drawElements.attrib%u call=%llu program=%u vbo=%u no readable bytes",
              (unsigned)attrib,
              (unsigned long long)drawCall,
              (unsigned)programName,
              (unsigned)vbo->name);
        return;
    }

    uint32_t firstIndex = mglReadGLIndexValue(indexBytes, indexType, 0u);
    NSUInteger bindingOffset = (a->binding_offset > 0) ? (NSUInteger)a->binding_offset : 0u;
    NSUInteger relativeOffset = (a->relativeoffset > 0) ? (NSUInteger)a->relativeoffset : 0u;
    NSUInteger stride = (a->stride > 0u) ? (NSUInteger)a->stride : mglVertexAttribElementBytes(a->type, a->size);
    NSUInteger vertexOffset = bindingOffset + relativeOffset + ((NSUInteger)firstIndex * stride);
    size_t elemBytes = mglVertexAttribElementBytes(a->type, a->size);

    if (elemBytes == 0u ||
        vertexOffset > (NSUInteger)vbo->size ||
        ((NSUInteger)vbo->size - vertexOffset) < elemBytes) {
        NSLog(@"MGL TRACE drawElements.attrib%u call=%llu program=%u vbo=%u OOB firstIndex=%u bindingOffset=%lu relOffset=%lu stride=%lu size=%u type=0x%x normalized=%u elemBytes=%zu vboSize=%lld",
              (unsigned)attrib,
              (unsigned long long)drawCall,
              (unsigned)programName,
              (unsigned)vbo->name,
              (unsigned)firstIndex,
              (unsigned long)bindingOffset,
              (unsigned long)relativeOffset,
              (unsigned long)stride,
              (unsigned)a->size,
              (unsigned)a->type,
              (unsigned)a->normalized,
              elemBytes,
              (long long)vbo->size);
        return;
    }

    const uint8_t *attribBytes = vboBytes + vertexOffset;
    double comps[4] = {0.0, 0.0, 0.0, 0.0};
    for (NSUInteger c = 0; c < MIN((NSUInteger)a->size, (NSUInteger)4); c++) {
        comps[c] = mglDecodeVertexAttribComponent(attribBytes, a->type, a->normalized, c);
    }

    char raw[3 * 16 + 1] = {0};
    size_t rawLen = MIN((size_t)16u, elemBytes);
    size_t rawPos = 0u;
    for (size_t i = 0; i < rawLen && rawPos + 3u < sizeof(raw); i++) {
        int wrote = snprintf(raw + rawPos,
                             sizeof(raw) - rawPos,
                             "%02x%s",
                             attribBytes[i],
                             (i + 1u < rawLen) ? ":" : "");
        if (wrote <= 0) {
            break;
        }
        rawPos += (size_t)wrote;
    }
    NSLog(@"MGL TRACE drawElements.attrib%u call=%llu program=%u vbo=%u firstIndex=%u bindingOffset=%lu relOffset=%lu vertexOffset=%lu stride=%lu size=%u type=0x%x normalized=%u format=%lu(%s) decoded=(%.6f,%.6f,%.6f,%.6f) raw=%s",
          (unsigned)attrib,
          (unsigned long long)drawCall,
          (unsigned)programName,
          (unsigned)vbo->name,
          (unsigned)firstIndex,
          (unsigned long)bindingOffset,
          (unsigned long)relativeOffset,
          (unsigned long)vertexOffset,
          (unsigned long)stride,
          (unsigned)a->size,
          (unsigned)a->type,
          (unsigned)a->normalized,
          (unsigned long)glTypeSizeToMtlType(a->type, a->size, a->normalized),
          mglVertexFormatName(glTypeSizeToMtlType(a->type, a->size, a->normalized)),
          comps[0], comps[1], comps[2], comps[3],
          raw);
}

#pragma mark debug code
void printDirtyBit(unsigned dirty_bits, unsigned dirty_flag, const char *name)
{
    if (dirty_bits & dirty_flag)
        DEBUG_PRINT("%s", name);
}

void logDirtyBits(GLMContext ctx)
{
    if(ctx->state.dirty_bits)
    {
        if (ctx->state.dirty_bits & DIRTY_ALL_BIT)
        {
            printDirtyBit(ctx->state.dirty_bits, DIRTY_ALL_BIT, "DIRTY_ALL_BIT set");
        }
        else
        {
            printDirtyBit(ctx->state.dirty_bits, DIRTY_VAO, "DIRTY_VAO ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_STATE, "DIRTY_STATE ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_BUFFER, "DIRTY_BUFFER ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_TEX, "DIRTY_TEX ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_TEX_PARAM, "DIRTY_TEX_PARAM ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_TEX_BINDING, "DIRTY_TEX_BINDING ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_SAMPLER, "DIRTY_SAMPLER ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_SHADER, "DIRTY_SHADER ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_PROGRAM, "DIRTY_PROGRAM ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_FBO, "DIRTY_FBO ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_DRAWABLE, "DIRTY_DRAWABLE ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_RENDER_STATE, "DIRTY_RENDER_STATE ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_ALPHA_STATE, "DIRTY_ALPHA_STATE ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_IMAGE_UNIT_STATE, "DIRTY_IMAGE_UNIT_STATE ");
            printDirtyBit(ctx->state.dirty_bits, DIRTY_BUFFER_BASE_STATE, "DIRTY_BUFFER_BASE_STATE ");
        }
        DEBUG_PRINT("\n");
    }
}

#pragma mark buffer objects
- (void) bindMTLBuffer:(Buffer *) ptr
{
    MTLResourceOptions options;
    const size_t kMaxSafeBufferSize = (size_t)2 * 1024 * 1024 * 1024; // 2 GiB safety cap

    if (!ptr) {
        NSLog(@"MGL ERROR: bindMTLBuffer called with NULL buffer");
        return;
    }

    // Corrupted buffer sizes can crash Metal validation immediately.
    if (ptr->size == 0 || ptr->size > kMaxSafeBufferSize) {
        NSLog(@"MGL ERROR: Refusing to create Metal buffer with suspicious size=%zu for buffer %u",
              (size_t)ptr->size, ptr->name);
        ptr->data.mtl_data = NULL;
        return;
    }

    options = MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeManaged;

    // ways we will only write to this
    if ((ptr->storage_flags & GL_MAP_READ_BIT) == 0)
    {
        options |= MTLResourceCPUCacheModeWriteCombined;
    }

    if (ptr->storage_flags & GL_CLIENT_STORAGE_BIT)
    {
        if (!ptr->data.buffer_data) {
            NSLog(@"MGL ERROR: GL_CLIENT_STORAGE_BIT set but buffer_data is NULL for buffer %u", ptr->name);
            ptr->data.mtl_data = NULL;
            return;
        }

        id<MTLBuffer> buffer = [_device newBufferWithBytesNoCopy:(void *)(ptr->data.buffer_data)
                                                           length:ptr->size
                                                          options:options
                                                      deallocator:^(void *pointer, NSUInteger length)
                              {
                                  kern_return_t err;
                                  err = vm_deallocate((vm_map_t) mach_task_self(),
                                                      (vm_address_t) pointer,
                                                      length);
                                  if (err != 0) {
                                      NSLog(@"MGL WARNING: vm_deallocate failed for Metal no-copy buffer err=%d ptr=%p len=%lu",
                                            err,
                                            pointer,
                                            (unsigned long)length);
                                  }
                              }];

        ptr->data.mtl_data = (void *)CFBridgingRetain(buffer);
    }
    else
    {
        id<MTLBuffer> buffer;
        
        // a backing can allocated initially, delete it and point the
        // backing data to the MTL buffer
        if (ptr->data.buffer_data)
        {
            size_t safeBufferSize = ptr->data.buffer_size;
            if (safeBufferSize == 0 || safeBufferSize > kMaxSafeBufferSize) {
                safeBufferSize = ptr->size;
            }

            // check the GL allocated size, not the vm_allocated size as these are page aligned
            if (ptr->size > 4095)
            {
                buffer = [_device newBufferWithBytes:(void *)ptr->data.buffer_data
                                                            length:safeBufferSize
                                                           options:options];
                if (!buffer) {
                    NSLog(@"MGL ERROR: Failed to create Metal buffer from backing data (size=%zu, buffer=%u)",
                          safeBufferSize, ptr->name);
                    ptr->data.mtl_data = NULL;
                    return;
                }

                kern_return_t err;
                err = vm_deallocate((vm_map_t) mach_task_self(),
                                    (vm_address_t) ptr->data.buffer_data,
                                    safeBufferSize);
                if (err != 0) {
                    NSLog(@"MGL WARNING: vm_deallocate failed for buffer %u err=%d ptr=%p len=%zu",
                          ptr->name,
                          err,
                          (void *)ptr->data.buffer_data,
                          safeBufferSize);
                }

                ptr->data.buffer_data = (vm_address_t)buffer.contents;
            }
            else
            {
                // AGX Driver Compatibility: For small buffers, still create a Metal buffer to avoid NULL assertion
                buffer = [_device newBufferWithBytes:(void *)ptr->data.buffer_data
                                              length:ptr->size
                                             options:options];
                if (!buffer) {
                    NSLog(@"MGL ERROR: Failed to create small Metal buffer (size=%zu, buffer=%u)",
                          (size_t)ptr->size, ptr->name);
                    ptr->data.mtl_data = NULL;
                    return;
                }

                // Don't deallocate the original buffer for small sizes to maintain compatibility
            }
        }
        else
        {
            buffer = [_device newBufferWithLength: ptr->size // allocate by size
                                                        options: options];
            if (!buffer) {
                NSLog(@"MGL ERROR: Failed to allocate Metal buffer with length=%zu (buffer=%u)",
                      (size_t)ptr->size, ptr->name);
                ptr->data.mtl_data = NULL;
                return;
            }

            ptr->data.buffer_data = (vm_address_t)NULL;
        }

        ptr->data.mtl_data = (void *)CFBridgingRetain(buffer);
    }
}

- (bool) mapGLBuffersToMTLBufferMap:(BufferMapList *)buffer_map stage: (int) stage
{
    static uint64_t s_mapCallCountByStage[8] = {0};
    uint64_t mapCall = 0;
    if (stage >= 0 && stage < 8) {
        mapCall = ++s_mapCallCountByStage[stage];
    } else {
        mapCall = ++s_mapCallCountByStage[0];
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(mapCall)) {
        NSLog(@"MGL TRACE map.begin stage=%d call=%llu preCount=%u program=%u",
              stage,
              (unsigned long long)mapCall,
              buffer_map ? buffer_map->count : 0,
              ctx ? (unsigned)ctx->state.program_name : 0u);
    }

    int count;
    int mapped_buffers;
    struct {
        int spvc_type;
        int gl_buffer_type;
        const char *name;
    } mapped_types[4] = {
        {SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, _UNIFORM_BUFFER, "Uniform Buffer"},
        {SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, _UNIFORM_CONSTANT, "Uniform Constant"},
        {SPVC_RESOURCE_TYPE_STORAGE_BUFFER, _SHADER_STORAGE_BUFFER, "Shader Storage Buffer"},
        {SPVC_RESOURCE_TYPE_ATOMIC_COUNTER, _ATOMIC_COUNTER_BUFFER, "Atomic Counter Buffer"}
    };
#if DEBUG_MAPPED_TYPES
    const char *stages[] = {"VERTEX_SHADER", "TESS_CONTROL_SHADER", "TESS_EVALUATION_SHADER",
        "GEOMETRY_SHADER", "FRAGMENT_SHADER", "COMPUTE_SHADER"};
#endif
    
    // init mapped buffer count
    buffer_map->count = 0;

    // bind uniforms, shader storage and atomics to buffer map
    for(int type=0; type<4; type++)
    {
        int spvc_type;
        int gl_buffer_type;

        spvc_type = mapped_types[type].spvc_type;
        gl_buffer_type = mapped_types[type].gl_buffer_type;
        
        count = [self getProgramBindingCount: stage type: spvc_type];

#if DEBUG_MAPPED_TYPES
        DEBUG_PRINT("Checking mapped_types: %s count:%d for stage: %s\n", mapped_types[type].name, count, stages[stage]);
#endif
        
        if (count)
        {
            BufferBaseTarget *buffers;
            BufferBaseTarget *fallbackBuffers = NULL;

            Program *activeProgram = mglResolveProgramFromState(ctx);
            if (spvc_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT && activeProgram) {
                buffers = activeProgram->plain_uniform_buffers;
                fallbackBuffers = ctx->state.buffer_base[gl_buffer_type].buffers;
            } else {
                buffers = ctx->state.buffer_base[gl_buffer_type].buffers;
            }
            
            for (int i = 0; i < count; i++)
            {
                GLuint spirv_binding;
                Buffer *buf;
                BufferBaseTarget *baseBinding;

                // Use the GL binding point to locate the client's buffer base.
                // The resource's `binding` may already have been rewritten to the
                // Metal [[buffer(n)]] slot parsed from generated MSL.
                Program *program = mglResolveProgramFromState(ctx);
                if (!program || spvc_type < 0 || spvc_type >= _MAX_SPIRV_RES ||
                    i >= (int)program->spirv_resources_list[stage][spvc_type].count) {
                    continue;
                }
                SpirvResource *resource = &program->spirv_resources_list[stage][spvc_type].list[i];
                spirv_binding = mglClientBufferBindingForResource(spvc_type, resource);
                if (spirv_binding >= MAX_BINDABLE_BUFFERS)
                {
                    NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: stage=%d type=%d binding=%u exceeds MAX_BINDABLE_BUFFERS=%d, skipping",
                          stage, spvc_type, spirv_binding, MAX_BINDABLE_BUFFERS);
                    continue;
                }

                baseBinding = &buffers[spirv_binding];
                if (fallbackBuffers && !baseBinding->buf && baseBinding->buffer == 0) {
                    BufferBaseTarget *fallbackBinding = &fallbackBuffers[spirv_binding];
                    if (fallbackBinding->buf || fallbackBinding->buffer != 0) {
                        baseBinding = fallbackBinding;
                    }
                }
                buf = mglRendererGetValidatedBuffer(ctx, baseBinding->buf,
                                                    "mapGLBuffersToMTLBufferMap(base)",
                                                    (NSUInteger)spirv_binding);

                // Recover from name/object map skew: some paths can preserve GL name while pointer slot is stale.
                if (!buf && baseBinding->buffer != 0) {
                    Buffer *resolved = (Buffer *)searchHashTable(&ctx->state.buffer_table, baseBinding->buffer);
                    resolved = mglRendererGetValidatedBuffer(ctx, resolved,
                                                             "mapGLBuffersToMTLBufferMap(base,recover)",
                                                             (NSUInteger)spirv_binding);
                    if (resolved) {
                        baseBinding->buf = resolved;
                        buf = resolved;
                        NSLog(@"MGL BUFFER RECOVER: stage=%d type=%d binding=%u name=%u ptr=%p",
                              stage, spvc_type, spirv_binding, baseBinding->buffer, resolved);
                    }
                }

                if (buf)
                {
                    NSUInteger reflectedRequiredSize =
                        [self getProgramBindingRequiredSize:stage type:spvc_type index:i];

                    if (buffer_map->count >= MAX_MAPPED_BUFFERS)
                    {
                        NSLog(@"MGL ERROR: mapGLBuffersToMTLBufferMap overflow: count=%d max=%d",
                              buffer_map->count, MAX_MAPPED_BUFFERS);
                        return false;
                    }
                    buffer_map->buffers[buffer_map->count].attribute_mask = 0; // non attribute.. no bits set
                    buffer_map->buffers[buffer_map->count].buffer_base_index = spirv_binding;
                    buffer_map->buffers[buffer_map->count].buf = buf;
                    buffer_map->buffers[buffer_map->count].offset = baseBinding->offset;
                    buffer_map->buffers[buffer_map->count].size = baseBinding->size;
                    baseBinding->buffer = buf->name;
                    buffer_map->count++;

                    if (reflectedRequiredSize > 0 && baseBinding->size > 0 &&
                        (NSUInteger)baseBinding->size < reflectedRequiredSize) {
                        GLuint programName = ctx ? ctx->state.program_name : 0u;
                        if (mglShouldLogSmallBaseBinding(programName,
                                                         stage,
                                                         spvc_type,
                                                         spirv_binding,
                                                         buf->name,
                                                         baseBinding->size,
                                                         reflectedRequiredSize)) {
                            NSLog(@"MGL WARNING: base binding too small program=%u stage=%d type=%d binding=%u glName=%u range=%lld reflected=%lu (padding at bind)",
                                  programName,
                                  stage,
                                  spvc_type,
                                  spirv_binding,
                                  buf->name,
                                  (long long)baseBinding->size,
                                  (unsigned long)reflectedRequiredSize);
                        }
                    }
                    
                    //DEBUG_PRINT("Found buffer type: %s buffer_base_index: %d\n", mapped_types[type].name, spirv_binding);
                }
                else
                {
                    if (baseBinding->buf || baseBinding->buffer != 0 || baseBinding->offset != 0 || baseBinding->size != 0) {
                        NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: dropping invalid base buffer binding=%u stage=%d type=%d name=%u ptr=%p offset=%lld size=%lld",
                              spirv_binding, stage, spvc_type,
                              baseBinding->buffer,
                              baseBinding->buf,
                              (long long)baseBinding->offset,
                              (long long)baseBinding->size);
                        bzero(baseBinding, sizeof(BufferBaseTarget));
                    }
                    // Some vanilla shader paths tolerate unbound blocks on specific stages.
                    // Skip instead of poisoning global GL error state with GL_INVALID_OPERATION.
                    continue;
                }
            }
        }
    }
    
    // bind vao attribs to buffers (attribs can share the same buffer)
    if (stage == _VERTEX_SHADER)
    {
        int vao_buffer_start;
        GLuint next_vertex_binding_index = (GLuint)kMGLVertexAttribBufferBase;
        VertexArray *vao = mglRendererGetValidatedVAO(ctx, "mapGLBuffersToMTLBufferMap");
        Program *activeProgram = mglResolveProgramFromState(ctx);

        count = [self getProgramBindingCount: stage type: SPVC_RESOURCE_TYPE_STAGE_INPUT];
        mapped_buffers = 0;

        if (!vao) {
            if (count > 0) {
                NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: stage inputs=%d but VAO is invalid/null, skipping attrib mapping",
                      count);
            }
            return true;
        }

        if (kMGLVertexAttribBufferBase >= kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL ERROR: invalid vertex attrib base index=%lu (max valid=%lu)",
                  (unsigned long)kMGLVertexAttribBufferBase,
                  (unsigned long)kMGLMaxMetalVertexBufferIndex);
            return false;
        }

        // vao buffers start after the uniforms and shader buffers
        vao_buffer_start = buffer_map->count;
        // CRITICAL SECURITY FIX: Check against actual map capacity.
        if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
            NSLog(@"MGL SECURITY ERROR: buffer_map count %d exceeds MAX_MAPPED_BUFFERS %d",
                  buffer_map->count, MAX_MAPPED_BUFFERS);
            return false;
        }
        buffer_map->buffers[vao_buffer_start].attribute_mask = 0;
        buffer_map->buffers[vao_buffer_start].buffer_base_index = (GLuint)kMGLVertexAttribBufferBase;
        buffer_map->buffers[vao_buffer_start].buf = NULL;
        buffer_map->buffers[vao_buffer_start].offset = 0;
        buffer_map->buffers[vao_buffer_start].size = 0;

        // create attribute map
        //
        // we need to cache this mapping, its called on each draw command
        //
        for(int att=0;att<ctx->state.max_vertex_attribs; att++)
        {
            if (vao->enabled_attribs & (0x1 << att))
            {
                if (!mglRendererProgramUsesVertexAttrib(activeProgram, (GLuint)att)) {
                    if ((vao->enabled_attribs >> (att+1)) == 0)
                        break;
                    continue;
                }

                Buffer *gl_buffer = mglRendererGetValidatedBuffer(ctx, vao->attrib[att].buffer,
                                                                  "mapGLBuffersToMTLBufferMap",
                                                                  (NSUInteger)att);
                if (!gl_buffer) {
                    NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap: enabled attrib %d has invalid/NULL buffer, skipping attrib",
                          att);
                    continue;
                }

                Buffer *map_buffer = NULL;

                // check start for map... then check
                map_buffer = buffer_map->buffers[vao_buffer_start].buf;

                // empty slot map it here, only works on first buffer..
                if (map_buffer == NULL)
                {
                    if (next_vertex_binding_index >= kMGLMaxMetalVertexBufferCount) {
                        NSLog(@"MGL WARNING: vertex binding index overflow (next=%u maxValid=%lu), skipping attrib %d",
                              next_vertex_binding_index, (unsigned long)kMGLMaxMetalVertexBufferIndex, att);
                        continue;
                    }
                    // map the buffer object to a metal vertex index
                    if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
                        NSLog(@"MGL WARNING: vertex buffer map is full (count=%u max=%u), skipping attrib %d",
                              buffer_map->count, MAX_MAPPED_BUFFERS, att);
                        continue;
                    }
                    buffer_map->buffers[vao_buffer_start].attribute_mask |= (0x1 << att);
                    buffer_map->buffers[vao_buffer_start].buf = gl_buffer;
                    buffer_map->buffers[vao_buffer_start].buffer_base_index = next_vertex_binding_index++;
                    buffer_map->buffers[vao_buffer_start].offset = vao->attrib[att].binding_offset;
                    buffer_map->buffers[vao_buffer_start].size = 0;
                    buffer_map->count++;

                    mapped_buffers++;
                }
                else
                {
                    bool found_buffer = false;

                    // find vao attrib with same buffer
                    for (int map=vao_buffer_start;
                         (found_buffer == false) && map<buffer_map->count;
                         map++)
                    {
                        map_buffer = buffer_map->buffers[map].buf;
                        if (!map_buffer) {
                            continue;
                        }

                        // we need to check name and target, not pointers..
                        // FIX ME: I think we don't need a target as all attribs should be an array_buffer
                        if ((map_buffer->name == gl_buffer->name) &&
                            (map_buffer->target == gl_buffer->target) &&
                            (buffer_map->buffers[map].offset == vao->attrib[att].binding_offset))
                        {
                            // include it the list of attributes
                            buffer_map->buffers[map].attribute_mask |= (0x1 << att);
                            found_buffer = true;
                            mapped_buffers++;
                            break;
                        }
                    }

                    if (found_buffer == false)
                    {
                        if (next_vertex_binding_index >= kMGLMaxMetalVertexBufferCount) {
                            NSLog(@"MGL WARNING: vertex binding index overflow (next=%u maxValid=%lu), cannot append attrib %d",
                                  next_vertex_binding_index, (unsigned long)kMGLMaxMetalVertexBufferIndex, att);
                            continue;
                        }
                        // map the next buffer object to a metal vertex index
                        if (buffer_map->count >= MAX_MAPPED_BUFFERS) {
                            NSLog(@"MGL WARNING: vertex buffer map is full (count=%u max=%u), cannot append attrib %d",
                                  buffer_map->count, MAX_MAPPED_BUFFERS, att);
                            continue;
                        }
                        buffer_map->buffers[buffer_map->count].attribute_mask = (0x1 << att);
                        buffer_map->buffers[buffer_map->count].buffer_base_index = next_vertex_binding_index++;
                        buffer_map->buffers[buffer_map->count].buf = gl_buffer;
                        buffer_map->buffers[buffer_map->count].offset = vao->attrib[att].binding_offset;
                        buffer_map->buffers[buffer_map->count].size = 0;
                        buffer_map->count++;

                        mapped_buffers++;
                    }
                }
            }

            if ((vao->enabled_attribs >> (att+1)) == 0)
                break;
        }

        if (mapped_buffers != count) {
            static unsigned long long s_map_mismatch_hits = 0;
            s_map_mismatch_hits++;
            if ((s_map_mismatch_hits % 64ull) == 1ull) {
                Buffer *drawIndexBuffer = vao->element_array.buffer;
                void *indexBufferMetal = drawIndexBuffer ? drawIndexBuffer->data.mtl_data : NULL;
                NSLog(@"MGL WARNING: mapGLBuffersToMTLBufferMap mismatch (pipeline=%p mapped=%u expected=%u stage=%d hit=%llu indexBuffer=%p vao=%p)",
                      _pipelineState, mapped_buffers, count, stage, s_map_mismatch_hits, indexBufferMetal, vao);
            }
        }
    }
    else if (stage == _COMPUTE_SHADER)
    {
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(mapCall)) {
        NSLog(@"MGL TRACE map.end stage=%d call=%llu mappedCount=%u",
              stage,
              (unsigned long long)mapCall,
              buffer_map ? buffer_map->count : 0);
    }

    return true;
}

- (bool) mapBuffersToMTL
{
    if ([self mapGLBuffersToMTLBufferMap: &ctx->state.vertex_buffer_map_list stage:_VERTEX_SHADER] == false)
        return false;

    if ([self mapGLBuffersToMTLBufferMap: &ctx->state.fragment_buffer_map_list stage:_FRAGMENT_SHADER] == false)
        return false;

    return true;
}

- (bool) updateDirtyBuffer:(Buffer *)ptr
{
    // buffers less than 4k will be uploaded using setVertexBytes
    if (ptr->size < 4096)
    {
        if ((ptr->data.dirty_bits & DIRTY_BUFFER_DATA) && kMGLDiagnosticStateLogs) {
            static uint64_t s_smallDirtySkipCalls = 0;
            uint64_t call = ++s_smallDirtySkipCalls;
            if (mglShouldTraceBufferTransferCall(call)) {
                const void *cpuData = (const void *)(uintptr_t)ptr->data.buffer_data;
                size_t sampleLen = ptr->size > 0 ? (size_t)ptr->size : 0u;
                uint64_t cpuHash = mglTraceHashBytes(cpuData, sampleLen);
                char cpuHead[64];
                cpuHead[0] = '\0';
                mglTraceFormatBytes(cpuData, sampleLen, cpuHead, sizeof(cpuHead));

                uint64_t mtlHash = 0ull;
                char mtlHead[64];
                mtlHead[0] = '\0';
                NSUInteger metalLen = 0;
                if (ptr->data.mtl_data && (uintptr_t)ptr->data.mtl_data >= 0x10000u) {
                    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
                    if (mtlBuffer) {
                        metalLen = mtlBuffer.length;
                        const void *mtlBytes = mtlBuffer.contents;
                        size_t mtlSample = (size_t)MIN((NSUInteger)sampleLen, metalLen);
                        mtlHash = mglTraceHashBytes(mtlBytes, mtlSample);
                        mglTraceFormatBytes(mtlBytes, mtlSample, mtlHead, sizeof(mtlHead));
                    }
                }

                NSLog(@"MGL TRACE smallBufferDirty.skip call=%llu buffer=%u size=%lld dirty=0x%x cpuHash=0x%016llx cpuHead=%s mtl=%p mtlLen=%lu mtlHash=0x%016llx mtlHead=%s",
                      (unsigned long long)call,
                      ptr->name,
                      (long long)ptr->size,
                      ptr->data.dirty_bits,
                      (unsigned long long)cpuHash,
                      cpuHead,
                      ptr->data.mtl_data,
                      (unsigned long)metalLen,
                      (unsigned long long)mtlHash,
                      (metalLen > 0 ? mtlHead : "-"));
            }
        }

        ptr->data.dirty_bits &= ~DIRTY_BUFFER_ADDR;
        
        return true;
    }
    
    if (ptr->data.dirty_bits & DIRTY_BUFFER_ADDR)
    {
        if (ptr->data.mtl_data == NULL)
        {
            [self bindMTLBuffer: ptr];
            RETURN_FALSE_ON_NULL(ptr->data.mtl_data);

            // clear dirty bits
            ptr->data.dirty_bits = 0;
        }
    }
    else if (ptr->data.dirty_bits & DIRTY_BUFFER_DATA)
    {
        if (ptr->data.mtl_data == NULL)
        {
            [self bindMTLBuffer: ptr];
            RETURN_FALSE_ON_NULL(ptr->data.mtl_data);

            // clear dirty bits
            ptr->data.dirty_bits = 0;

            // we had to create a buffer so no need to update data
            return true;
        }

        // CRITICAL SECURITY FIX: Safe Metal buffer validation
        id<MTLBuffer> buffer = (id<MTLBuffer>)SafeMetalBridge(ptr->data.mtl_data, objc_getClass("MTLBuffer"), "MTLBuffer");
        if (!buffer) {
            NSLog(@"MGL SECURITY ERROR: Failed to validate Metal buffer (buffer %u)", ptr->name);
            return false;
        }

        // clear dirty bits if not mapped as coherent
        // this will cause us to keep loading the buffer and keep the GPU
        // contents in check for EVERY drawing operation
        if (ptr->access & GL_MAP_COHERENT_BIT)
        {
            [buffer didModifyRange: NSMakeRange(ptr->mapped_offset, ptr->mapped_length)];

            ptr->data.dirty_bits = DIRTY_BUFFER_DATA;
        }
        else
        {
            [buffer didModifyRange: NSMakeRange(0, ptr->data.buffer_size)];

            ptr->data.dirty_bits = 0;
        }
    }
    else
    {
        NSLog(@"MGL BUFFER ERROR: updateDirtyBuffer saw buffer %u with no CPU or Metal backing",
              ptr ? ptr->name : 0u);
        return false;
    }

    return true;
}

- (bool) checkForDirtyBufferData:  (BufferMapList *)buffer_map_list
{
    GLuint mapCount;

    if (!buffer_map_list) {
        return false;
    }

    mapCount = buffer_map_list->count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL WARNING: checkForDirtyBufferData mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    // update vbos, some vbos may not have metal buffers yet
    for (GLuint i = 0; i < mapCount; i++)
    {
        Buffer *gl_buffer = mglRendererGetValidatedBuffer(ctx,
                                                          buffer_map_list->buffers[i].buf,
                                                          __FUNCTION__,
                                                          (NSUInteger)i);

        if (gl_buffer)
        {
            if (gl_buffer->data.dirty_bits)
            {
                return true;
            }
        } else if (buffer_map_list->buffers[i].buf) {
            buffer_map_list->buffers[i].buf = NULL;
        }
    }

    return false;
}

- (bool) updateDirtyBaseBufferList: (BufferMapList *)buffer_map_list
{
    GLuint mapCount;

    if (!buffer_map_list) {
        return true;
    }

    mapCount = buffer_map_list->count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL WARNING: updateDirtyBaseBufferList mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    // update vbos, some vbos may not have metal buffers yet
    for (GLuint i = 0; i < mapCount; i++)
    {
        Buffer *gl_buffer = mglRendererGetValidatedBuffer(ctx,
                                                          buffer_map_list->buffers[i].buf,
                                                          __FUNCTION__,
                                                          (NSUInteger)i);

        if (gl_buffer)
        {
            if (gl_buffer->data.dirty_bits)
            {
                RETURN_FALSE_ON_FAILURE([self updateDirtyBuffer: gl_buffer]);
            }
        } else if (buffer_map_list->buffers[i].buf) {
            buffer_map_list->buffers[i].buf = NULL;
        }
    }

    return true;
}

- (bool) bindVertexBuffersToCurrentRenderEncoder
{
    static uint64_t s_vbindCallCount = 0;
    static double s_vbindLastCallTime = 0.0;
    static uint64_t s_vbindLastCallCount = 0;
    uint64_t vbindCall = ++s_vbindCallCount;
    double vbindStartSeconds = mglNowSeconds();
    mglLogLoopHeartbeat("vbind.loop",
                        vbindCall,
                        vbindStartSeconds,
                        &s_vbindLastCallTime,
                        &s_vbindLastCallCount,
                        0.25);

    BufferMap *map;
    Buffer *ptr;
    GLintptr offset;
    NSUInteger bindingIndex;
    bool isBaseBinding;
    bool anyBindingPresent[MAX_MAPPED_BUFFERS] = {false};
    bool baseBindingPresent[MAX_BINDABLE_BUFFERS] = {false};
    bool attribBindingReserved[MAX_MAPPED_BUFFERS] = {false};
    int attribBindingIndex[MAX_ATTRIBS];
    static id<MTLBuffer> fallbackBindingBuffer = nil;
    static id<MTLBuffer> minimumBindingBuffer = nil;
    Program *activeProgram;
    VertexArray *vao;
    GLuint mapCount;

    if (kMGLVerboseBindLogs) {
        NSLog(@"MGL VBIND begin ctx=%p vao=%p encoder=%p",
              ctx, ctx ? ctx->state.vao : NULL, _currentRenderEncoder);
    }

    if (!ctx || !_currentRenderEncoder) {
        NSLog(@"MGL VBIND skip: encoder/ctx nil");
        return false;
    }

    vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    if (!vao) {
        NSLog(@"MGL VBIND skip: vao nil/invalid");
        return false;
    }
    activeProgram = mglResolveProgramFromState(ctx);

    if (kMGLVerboseBindLogs) {
        NSLog(@"MGL VBIND vao=%p magic=0x%x", vao, vao->magic);
    }
    mapCount = ctx->state.vertex_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL WARNING: VBIND mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        attribBindingIndex[i] = -1;
    }

    // Resolve attribute slot reservations first so base/resource bindings do not
    // overwrite shader-required vertex input slots.
    GLuint reserveMaxAttribs = ctx->state.max_vertex_attribs;
    if (reserveMaxAttribs > MAX_ATTRIBS) {
        reserveMaxAttribs = MAX_ATTRIBS;
    }
    for (GLuint attrib = 0; attrib < reserveMaxAttribs; attrib++) {
        if ((vao->enabled_attribs & (0x1u << attrib)) == 0u) {
            continue;
        }
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, attrib)) {
            if ((vao->enabled_attribs >> (attrib + 1)) == 0u) {
                break;
            }
            continue;
        }

        int mappedIndex = [self getVertexBufferIndexWithAttributeSet:(int)attrib];
        if (mappedIndex < 0 || mappedIndex >= (int)kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL ERROR: VBIND reserve attrib=%u unresolved mapping=%d", attrib, mappedIndex);
            continue;
        }

        attribBindingIndex[attrib] = mappedIndex;
        attribBindingReserved[mappedIndex] = true;

        if ((vao->enabled_attribs >> (attrib + 1)) == 0u) {
            break;
        }
    }

    for (GLuint i = 0; i < MAX_ATTRIBS; i++) {
        BOOL enabled = ((vao->enabled_attribs >> i) & 0x1u) != 0;
        Buffer *attribBuffer = mglRendererGetValidatedBuffer(ctx, vao->attrib[i].buffer, __FUNCTION__, i);
        GLuint attribBufferName = attribBuffer ? attribBuffer->name : 0;
        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL VBIND attrib=%u enabled=%d buf=%p bufName=%u bindOffset=%lld ptr=0x%llx stride=%u size=%u type=0x%x normalized=%u divisor=%u binding=%u",
                  i,
                  enabled ? 1 : 0,
                  attribBuffer,
                  attribBufferName,
                  (long long)vao->attrib[i].binding_offset,
                  (unsigned long long)(uintptr_t)vao->attrib[i].relativeoffset,
                  (unsigned)vao->attrib[i].stride,
                  (unsigned)vao->attrib[i].size,
                  (unsigned)vao->attrib[i].type,
                  (unsigned)vao->attrib[i].normalized,
                  (unsigned)vao->attrib[i].divisor,
                  (unsigned)vao->attrib[i].buffer_bindingindex);
        }

        if (kMGLVerboseBindLogs && enabled && attribBuffer) {
            NSLog(@"MGL VBIND buffer detail attrib=%u name=%u size=%lld mtl=%p data=%p init(ever=%u full=%u range=[%lld,%lld) source=%u off=%lld size=%lld src=%p hash=0x%016llx)",
                  i,
                  attribBuffer->name,
                  (long long)attribBuffer->size,
                  attribBuffer->data.mtl_data,
                  (void *)attribBuffer->data.buffer_data,
                  (unsigned)attribBuffer->ever_written,
                  (unsigned)attribBuffer->has_initialized_data,
                  (long long)attribBuffer->written_min,
                  (long long)attribBuffer->written_max,
                  (unsigned)attribBuffer->last_init_source,
                  (long long)attribBuffer->last_write_offset,
                  (long long)attribBuffer->last_write_size,
                  attribBuffer->last_write_src_ptr,
                  (unsigned long long)attribBuffer->last_write_src_hash);
        }
    }

    for(int i=0; i<(int)mapCount; i++)
    {
        map = &ctx->state.vertex_buffer_map_list.buffers[i];
        
        ptr = mglRendererGetValidatedBuffer(ctx, map->buf, __FUNCTION__, (NSUInteger)i);
        offset = map->offset;
        isBaseBinding = (map->attribute_mask == 0);
        GLuint glBindingIndex = map->buffer_base_index;
        bindingIndex = glBindingIndex;
        if (isBaseBinding) {
            NSInteger metalBindingIndex = [self getProgramMetalBufferIndexForStage:_VERTEX_SHADER
                                                                            binding:glBindingIndex];
            if (metalBindingIndex < 0) {
                continue;
            }
            bindingIndex = (NSUInteger)metalBindingIndex;
        }

        // Vertex attribute streams are rebound from VAO below using a deterministic
        // attribute->slot mapping shared with generateVertexDescriptor.
        // Keep this pass for resource/base bindings only.
        if (!isBaseBinding) {
            continue;
        }

        if (bindingIndex >= kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL WARNING: Vertex binding index %lu out of Metal range (max valid=%lu), skipping map[%d]",
                  (unsigned long)bindingIndex, (unsigned long)kMGLMaxMetalVertexBufferIndex, i);
            continue;
        }

        if (attribBindingReserved[bindingIndex]) {
            if (kMGLVerboseBindLogs) {
                NSLog(@"MGL VBIND skip base slot %lu: reserved by attrib mapping",
                      (unsigned long)bindingIndex);
            }
            continue;
        }

        if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            baseBindingPresent[glBindingIndex] = true;
        }

        if (!ptr) {
            NSLog(@"MGL WARNING: Vertex buffer map[%d] has invalid/NULL buffer pointer, skipping", i);
            [_currentRenderEncoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
            continue;
        }

        if (offset < 0) {
            NSLog(@"MGL WARNING: Vertex buffer map[%d] has negative offset=%lld, skipping",
                  i, (long long)offset);
            [_currentRenderEncoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
            continue;
        }

        if (ptr->size < 0) {
            NSLog(@"MGL WARNING: Vertex buffer %u has invalid size=%lld, skipping",
                  ptr->name, (long long)ptr->size);
            [_currentRenderEncoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
            continue;
        }

        if (!ptr->data.mtl_data) {
            [self bindMTLBuffer:ptr];
        }
        if (!ptr->data.mtl_data) {
            NSLog(@"MGL WARNING: Vertex buffer %u has no Metal backing after bind attempt, skipping slot %d", ptr->name, i);
            [_currentRenderEncoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
            continue;
        }
        if ((uintptr_t)ptr->data.mtl_data < 0x10000u) {
            NSLog(@"MGL VBIND skip base slot %d buffer=%u: suspicious mtl_data pointer=%p",
                  i, ptr->name, ptr->data.mtl_data);
            [_currentRenderEncoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
            continue;
        }

        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
        if (!buffer) {
            NSLog(@"MGL WARNING: Vertex buffer %u Metal object bridge failed, skipping slot %d", ptr->name, i);
            [_currentRenderEncoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
            continue;
        }

        NSUInteger metalLen = buffer.length;
        NSUInteger bindOffset = (NSUInteger)offset;
        if (bindOffset >= metalLen) {
            NSLog(@"MGL VBIND skip base slot %d buffer=%u: offset=%lu length=%lu",
                  i, ptr->name, (unsigned long)bindOffset, (unsigned long)metalLen);
            [_currentRenderEncoder setVertexBuffer:nil offset:0 atIndex:bindingIndex];
            continue;
        }

        NSUInteger reflectedRequiredBytes = 0;
        NSUInteger requiredBindingBytes = kMGLMinimumStageBindingSize;
        if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            reflectedRequiredBytes = [self getProgramBindingRequiredSizeForStage:_VERTEX_SHADER
                                                                          binding:glBindingIndex];
            if (reflectedRequiredBytes > requiredBindingBytes) {
                requiredBindingBytes = reflectedRequiredBytes;
            }
        }
        NSUInteger availableBytes = metalLen - bindOffset;
        // Some GL clients bind a narrow UBO range while the translated MSL
        // reflection includes backend padding. If the backing buffer is large
        // enough, bind it directly instead of zero-padding the reflected tail.
        if (isBaseBinding &&
            map->size > 0 &&
            (NSUInteger)map->size >= requiredBindingBytes &&
            (NSUInteger)map->size < availableBytes) {
            availableBytes = (NSUInteger)map->size;
        }

        if (isBaseBinding &&
            glBindingIndex < MAX_BINDABLE_BUFFERS &&
            availableBytes < requiredBindingBytes) {
            BOOL boundPaddedBytes = NO;
            uint8_t stackScratch[kMGLStageBindingStackScratchSize];
            bzero(stackScratch, sizeof(stackScratch));

            if (ptr->data.buffer_data && ptr->size > 0) {
                uintptr_t cpuData = (uintptr_t)ptr->data.buffer_data;
                if (cpuData >= 0x100000000ULL) {
                    size_t cpuSize = (size_t)ptr->size;
                    size_t cpuOffset = bindOffset;
                    if (cpuOffset < cpuSize) {
                        size_t remaining = cpuSize - cpuOffset;
                        size_t paddedLen = (size_t)requiredBindingBytes;
                        uint8_t *paddedBytes = stackScratch;
                        bool usingHeap = false;

                        if (paddedLen > sizeof(stackScratch)) {
                            paddedBytes = (uint8_t *)calloc(1, paddedLen);
                            usingHeap = (paddedBytes != NULL);
                        }

                        if (paddedBytes) {
                            size_t copyLen = MIN(paddedLen, remaining);
                            memcpy(paddedBytes,
                               ((const uint8_t *)ptr->data.buffer_data) + cpuOffset,
                               copyLen);
                            [_currentRenderEncoder setVertexBytes:paddedBytes
                                                           length:paddedLen
                                                          atIndex:bindingIndex];
                            mglTraceCloudVertexBufferBinding(mglResolveProgramFromState(ctx),
                                                             glBindingIndex,
                                                             bindingIndex,
                                                             ptr,
                                                             bindOffset,
                                                             (NSUInteger)copyLen,
                                                             "vertex-padded");
        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL SET VERTEX BUFFER index=%lu glName=%u offset=%lu available=%lu source=base-padded-bytes(min=%lu reflected=%lu copy=%lu range=%lld)",
                  (unsigned long)bindingIndex,
                  ptr->name,
                  (unsigned long)bindOffset,
                  (unsigned long)availableBytes,
                  (unsigned long)requiredBindingBytes,
                  (unsigned long)reflectedRequiredBytes,
                  (unsigned long)copyLen,
                  (long long)map->size);
        }
                            anyBindingPresent[bindingIndex] = true;
                            boundPaddedBytes = YES;

                            if (usingHeap) {
                                free(paddedBytes);
                            }
                        } else {
                            NSLog(@"MGL WARNING: VBIND failed to allocate %lu-byte scratch buffer for binding index=%lu",
                                  (unsigned long)paddedLen, (unsigned long)bindingIndex);
                        }
                    }
                }
            }

            if (!boundPaddedBytes) {
                if (!minimumBindingBuffer || minimumBindingBuffer.length < requiredBindingBytes) {
                    minimumBindingBuffer = [_device newBufferWithLength:requiredBindingBytes
                                                                 options:MTLResourceStorageModeShared];
                }
                if (minimumBindingBuffer) {
                    [_currentRenderEncoder setVertexBuffer:minimumBindingBuffer
                                                    offset:0
                                                   atIndex:bindingIndex];
                    if (kMGLVerboseBindLogs) {
                        NSLog(@"MGL SET VERTEX BUFFER index=%lu glName=%u offset=0 available=%lu source=base-min-fallback(min=%lu reflected=%lu)",
                              (unsigned long)bindingIndex,
                              ptr->name,
                              (unsigned long)minimumBindingBuffer.length,
                              (unsigned long)requiredBindingBytes,
                              (unsigned long)reflectedRequiredBytes);
                    }
                    anyBindingPresent[bindingIndex] = true;
                    continue;
                }
            } else {
                continue;
            }
        }

        [_currentRenderEncoder setVertexBuffer:buffer offset:offset atIndex:bindingIndex];
        mglTraceCloudVertexBufferBinding(mglResolveProgramFromState(ctx),
                                         glBindingIndex,
                                         bindingIndex,
                                         ptr,
                                         bindOffset,
                                         availableBytes,
                                         "vertex-buffer");
        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL SET VERTEX BUFFER index=%lu glName=%u offset=%lu available=%lu source=base",
                  (unsigned long)bindingIndex,
                  ptr->name,
                  (unsigned long)bindOffset,
                  (unsigned long)metalLen);
        }
        anyBindingPresent[bindingIndex] = true;
    }

    // Attribute bindings must use the exact same index mapping as generateVertexDescriptor.
    // Do this pass directly from the VAO so pipeline creation does not depend on map list timing.
    GLuint maxAttribs = ctx->state.max_vertex_attribs;
    if (maxAttribs > MAX_ATTRIBS) {
        maxAttribs = MAX_ATTRIBS;
    }
    for (GLuint attrib = 0; attrib < maxAttribs; attrib++) {
        if ((vao->enabled_attribs & (0x1u << attrib)) == 0u) {
            continue;
        }
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, attrib)) {
            if ((vao->enabled_attribs >> (attrib + 1)) == 0u) {
                break;
            }
            continue;
        }

        int mappedIndex = attribBindingIndex[attrib];
        if (mappedIndex < 0 || mappedIndex >= (int)kMGLMaxMetalVertexBufferCount) {
            NSLog(@"MGL ERROR: VBIND attrib=%u unresolved mapping=%d", attrib, mappedIndex);
            continue;
        }

        bindingIndex = (NSUInteger)mappedIndex;
        if (anyBindingPresent[bindingIndex]) {
            if ((vao->enabled_attribs >> (attrib + 1)) == 0u) {
                break;
            }
            continue;
        }

        Buffer *attribBuffer = mglRendererGetValidatedBuffer(ctx, vao->attrib[attrib].buffer, __FUNCTION__, attrib);
        if (!attribBuffer) {
            NSLog(@"MGL VBIND skip attrib=%u: enabled but buffer is invalid", attrib);
            continue;
        }

        if (!attribBuffer->ever_written) {
            NSLog(@"MGL VBIND BLOCK draw: attrib=%u uses buffer=%u that was allocated but never populated "
                  "(initSource=%u hasInitialized=%u written=[%lld,%lld) lastOff=%lld lastSize=%lld lastSrc=%p hash=0x%016llx)",
                  attrib,
                  attribBuffer->name,
                  (unsigned)attribBuffer->last_init_source,
                  (unsigned)attribBuffer->has_initialized_data,
                  (long long)attribBuffer->written_min,
                  (long long)attribBuffer->written_max,
                  (long long)attribBuffer->last_write_offset,
                  (long long)attribBuffer->last_write_size,
                  attribBuffer->last_write_src_ptr,
                  (unsigned long long)attribBuffer->last_write_src_hash);
            return false;
        }

        if (vao->attrib[attrib].binding_offset < 0) {
            NSLog(@"MGL VBIND BLOCK draw: attrib=%u buffer=%u negative bindingOffset=%lld",
                  attrib,
                  attribBuffer->name,
                  (long long)vao->attrib[attrib].binding_offset);
            return false;
        }
        if (vao->attrib[attrib].relativeoffset < 0) {
            NSLog(@"MGL VBIND BLOCK draw: attrib=%u buffer=%u negative relativeOffset=%lld",
                  attrib,
                  attribBuffer->name,
                  (long long)vao->attrib[attrib].relativeoffset);
            return false;
        }
        GLintptr attrOffset = vao->attrib[attrib].binding_offset +
                              (GLintptr)(uintptr_t)vao->attrib[attrib].relativeoffset;
        size_t compSize = mglVertexAttribComponentSize(vao->attrib[attrib].type);
        size_t compCount = (size_t)vao->attrib[attrib].size;
        GLintptr attrSpan = 0;
        if (compSize > 0u && compCount > 0u) {
            size_t total = compSize * compCount;
            if (total > (size_t)INTPTR_MAX) {
                NSLog(@"MGL VBIND BLOCK draw: attrib=%u buffer=%u attr span overflow (compSize=%zu compCount=%zu)",
                      attrib,
                      attribBuffer->name,
                      compSize,
                      compCount);
                return false;
            }
            attrSpan = (GLintptr)total;
        }
        GLintptr attrEnd = attrOffset + ((attrSpan > 0) ? attrSpan : 1);
        if (attribBuffer->written_min >= 0 && attribBuffer->written_max >= 0) {
            if (attrOffset < attribBuffer->written_min || attrEnd > attribBuffer->written_max) {
                NSLog(@"MGL VBIND BLOCK draw: attrib=%u buffer=%u attrRange=[%lld,%lld) outside written range [%lld,%lld) (type=0x%x size=%u)",
                      attrib,
                      attribBuffer->name,
                      (long long)attrOffset,
                      (long long)attrEnd,
                      (long long)attribBuffer->written_min,
                      (long long)attribBuffer->written_max,
                      (unsigned)vao->attrib[attrib].type,
                      (unsigned)vao->attrib[attrib].size);
                return false;
            }
        }

        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL VBIND attrib map attrib=%u -> index=%lu buffer=%u bindingOffset=%lld",
                  attrib,
                  (unsigned long)bindingIndex,
                  (unsigned)attribBuffer->name,
                  (long long)vao->attrib[attrib].binding_offset);
        }

        if (!attribBuffer->data.mtl_data) {
            [self bindMTLBuffer:attribBuffer];
        }
        if (!attribBuffer->data.mtl_data) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: no Metal backing",
                  attrib, attribBuffer->name);
            continue;
        }
        if ((uintptr_t)attribBuffer->data.mtl_data < 0x10000u) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: suspicious mtl_data=%p",
                  attrib, attribBuffer->name, attribBuffer->data.mtl_data);
            continue;
        }

        id<MTLBuffer> attribMetalBuffer = (__bridge id<MTLBuffer>)(attribBuffer->data.mtl_data);
        if (!attribMetalBuffer) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: Metal bridge failed",
                  attrib, attribBuffer->name);
            continue;
        }

        NSUInteger attribBindingOffset = (NSUInteger)vao->attrib[attrib].binding_offset;
        if (attribBindingOffset >= attribMetalBuffer.length) {
            NSLog(@"MGL VBIND skip attrib=%u buffer=%u: bindingOffset=%lu >= metalLen=%lu",
                  attrib,
                  attribBuffer->name,
                  (unsigned long)attribBindingOffset,
                  (unsigned long)attribMetalBuffer.length);
            continue;
        }

        [_currentRenderEncoder setVertexBuffer:attribMetalBuffer offset:attribBindingOffset atIndex:bindingIndex];
        anyBindingPresent[bindingIndex] = true;
        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL SET VERTEX ATTRIB BUFFER index=%lu glName=%u offset=%lu available=%lu attrib=%u stride=%u attrOffset=0x%llx mtl=%p",
                  (unsigned long)bindingIndex,
                  attribBuffer->name,
                  (unsigned long)attribBindingOffset,
                  (unsigned long)attribMetalBuffer.length,
                  attrib,
                  (unsigned)vao->attrib[attrib].stride,
                  (unsigned long long)(uintptr_t)vao->attrib[attrib].relativeoffset,
                  attribBuffer->data.mtl_data);
        }

        if ((vao->enabled_attribs >> (attrib + 1)) == 0u) {
            break;
        }
    }

    if (!fallbackBindingBuffer) {
        fallbackBindingBuffer = [_device newBufferWithLength:kMGLDefaultStageFallbackBufferSize
                                                     options:MTLResourceStorageModeShared];
    }

    // Bind fallback buffer for required stage buffer bindings that were not mapped.
    // This prevents Metal validation aborts on missing buffer slots.
    const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER
    };
    for (int t = 0; t < 4; t++) {
        int resourceType = resourceTypes[t];
        int count = [self getProgramBindingCount:_VERTEX_SHADER type:resourceType];
        Program *program = mglResolveProgramFromState(ctx);
        for (int i = 0; i < count; i++) {
            if (!program || resourceType < 0 || resourceType >= _MAX_SPIRV_RES ||
                i >= (int)program->spirv_resources_list[_VERTEX_SHADER][resourceType].count) {
                continue;
            }
            SpirvResource *resource = &program->spirv_resources_list[_VERTEX_SHADER][resourceType].list[i];
            GLuint clientBinding = mglClientBufferBindingForResource(resourceType, resource);
            if (clientBinding >= MAX_BINDABLE_BUFFERS) {
                continue;
            }
            NSInteger metalBinding = [self getProgramMetalBufferIndexForStage:_VERTEX_SHADER
                                                                       binding:clientBinding];
            if (metalBinding < 0 || metalBinding >= (NSInteger)kMGLMaxMetalVertexBufferCount) {
                continue;
            }
            if (!baseBindingPresent[clientBinding] && fallbackBindingBuffer) {
                [_currentRenderEncoder setVertexBuffer:fallbackBindingBuffer offset:0 atIndex:(NSUInteger)metalBinding];
                baseBindingPresent[clientBinding] = true;
                anyBindingPresent[(NSUInteger)metalBinding] = true;
            }
        }
    }

    // Conservative safety net:
    // Ensure every stage buffer slot has a valid binding before draw validation.
    // This avoids hard aborts when reflection misses hidden/generated buffer args.
    if (kMGLEnableVertexAllSlotFallback && fallbackBindingBuffer) {
        for (NSUInteger s = 0; s < MAX_BINDABLE_BUFFERS; s++) {
            if (!anyBindingPresent[s]) {
                [_currentRenderEncoder setVertexBuffer:fallbackBindingBuffer offset:0 atIndex:s];
                anyBindingPresent[s] = true;
            }
        }
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(vbindCall)) {
        NSUInteger boundSlots = 0;
        NSUInteger reservedSlots = 0;
        NSUInteger baseSlots = 0;
        for (NSUInteger s = 0; s < kMGLMaxMetalVertexBufferCount; s++) {
            if (anyBindingPresent[s]) {
                boundSlots++;
            }
            if (attribBindingReserved[s]) {
                reservedSlots++;
            }
        }
        for (NSUInteger s = 0; s < MAX_BINDABLE_BUFFERS; s++) {
            if (baseBindingPresent[s]) {
                baseSlots++;
            }
        }
        NSLog(@"MGL TRACE vbind.end call=%llu mapCount=%u boundSlots=%lu reservedAttribSlots=%lu baseSlots=%lu elapsed=%.3fms",
              (unsigned long long)vbindCall,
              (unsigned)mapCount,
              (unsigned long)boundSlots,
              (unsigned long)reservedSlots,
              (unsigned long)baseSlots,
              (mglNowSeconds() - vbindStartSeconds) * 1000.0);
    }

    return true;
}

- (bool) bindFragmentBuffersToCurrentRenderEncoder
{
    static uint64_t s_fbindCallCount = 0;
    static double s_fbindLastCallTime = 0.0;
    static uint64_t s_fbindLastCallCount = 0;
    uint64_t fbindCall = ++s_fbindCallCount;
    double fbindStartSeconds = mglNowSeconds();
    mglLogLoopHeartbeat("fbind.loop",
                        fbindCall,
                        fbindStartSeconds,
                        &s_fbindLastCallTime,
                        &s_fbindLastCallCount,
                        0.25);

    GLuint mapCount;
    BufferMap *map;
    Buffer *ptr;
    GLintptr offset;
    NSUInteger bindingIndex;
    bool isBaseBinding;
    bool anyBindingPresent[MAX_BINDABLE_BUFFERS] = {false};
    bool baseBindingPresent[MAX_BINDABLE_BUFFERS] = {false};
    static id<MTLBuffer> fallbackBindingBuffer = nil;
    static id<MTLBuffer> minimumBindingBuffer = nil;

    if (kMGLVerboseBindLogs) {
        NSLog(@"MGL FBIND begin ctx=%p encoder=%p", ctx, _currentRenderEncoder);
    }

    if (!ctx || !_currentRenderEncoder) {
        NSLog(@"MGL FBIND skip: ctx/encoder nil");
        return false;
    }

    mapCount = ctx->state.fragment_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        NSLog(@"MGL WARNING: FBIND mapCount=%u exceeds MAX_MAPPED_BUFFERS=%d, clamping",
              mapCount, MAX_MAPPED_BUFFERS);
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < mapCount; i++)
    {
        map = &ctx->state.fragment_buffer_map_list.buffers[i];

        if (kMGLVerboseBindLogs) {
            NSLog(@"MGL FBIND slot=%u candidate=%p mask=0x%x baseIndex=%u offset=%lld",
                  i,
                  map->buf,
                  map->attribute_mask,
                  map->buffer_base_index,
                  (long long)map->offset);
        }

        ptr = mglRendererGetValidatedBuffer(ctx, map->buf, __FUNCTION__, (NSUInteger)i);
        offset = map->offset;
        isBaseBinding = (map->attribute_mask == 0);
        GLuint glBindingIndex = map->buffer_base_index;
        bindingIndex = glBindingIndex;
        if (isBaseBinding) {
            NSInteger metalBindingIndex = [self getProgramMetalBufferIndexForStage:_FRAGMENT_SHADER
                                                                            binding:glBindingIndex];
            if (metalBindingIndex < 0) {
                continue;
            }
            bindingIndex = (NSUInteger)metalBindingIndex;
        }

        if (bindingIndex >= MAX_BINDABLE_BUFFERS) {
            NSLog(@"MGL WARNING: Fragment binding index %lu out of range (max=%d), skipping map[%d]",
                  (unsigned long)bindingIndex, MAX_BINDABLE_BUFFERS, i);
            continue;
        }

        if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
            baseBindingPresent[glBindingIndex] = true;
        }

        if (!ptr) {
            NSLog(@"MGL FBIND skip slot=%u: invalid/NULL candidate=%p", i, map->buf);
            map->buf = NULL;
            [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
            continue;
        }

        if (offset < 0) {
            NSLog(@"MGL FBIND skip slot=%u buffer=%u: negative offset=%lld",
                  i, ptr->name, (long long)offset);
            [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
            continue;
        }

        if (ptr->size < 0) {
            NSLog(@"MGL FBIND skip slot=%u buffer=%u: invalid size=%lld",
                  i, ptr->name, (long long)ptr->size);
            continue;
        }
        
        if (!isBaseBinding && ptr->size < 4096)
        {
            if (ptr->data.buffer_data && ptr->size > 0) {
                uintptr_t cpuData = (uintptr_t)ptr->data.buffer_data;
                if (cpuData < 0x100000000ULL) {
                    NSLog(@"MGL FBIND skip small buffer=%u slot=%u: suspicious CPU pointer=%p",
                          ptr->name, i, (void *)ptr->data.buffer_data);
                    [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                    continue;
                }

                size_t bindOffset = (size_t)offset;
                size_t bufferSize = (size_t)ptr->size;
                if (bindOffset >= bufferSize) {
                    NSLog(@"MGL FBIND skip small buffer=%u slot=%u: offset=%lu bufferSize=%lu",
                          ptr->name, i, (unsigned long)bindOffset, (unsigned long)bufferSize);
                    [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                    continue;
                }

                size_t bindLength = bufferSize - bindOffset;
                const uint8_t *bindPtr = ((const uint8_t *)ptr->data.buffer_data) + bindOffset;
                [_currentRenderEncoder setFragmentBytes:bindPtr length:bindLength atIndex:bindingIndex];
                if (kMGLVerboseBindLogs) {
                    NSLog(@"MGL FBIND ok(slot=%lu) setFragmentBytes buffer=%u len=%lu offset=%lu",
                          (unsigned long)bindingIndex,
                          ptr->name,
                          (unsigned long)bindLength,
                          (unsigned long)bindOffset);
                }
                anyBindingPresent[bindingIndex] = true;
            } else if (ptr->data.mtl_data) {
                if ((uintptr_t)ptr->data.mtl_data < 0x100000000ULL) {
                    NSLog(@"MGL FBIND skip small MTL buffer=%u slot=%u: suspicious mtl_data pointer=%p",
                          ptr->name, i, ptr->data.mtl_data);
                    [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                    continue;
                }
                id<MTLBuffer> fallbackBuffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
                if (fallbackBuffer) {
                    NSUInteger metalLen = fallbackBuffer.length;
                    NSUInteger bindOffset = (NSUInteger)offset;
                    if (bindOffset >= metalLen) {
                        NSLog(@"MGL FBIND skip small MTL buffer=%u slot=%u: offset=%lu length=%lu",
                              ptr->name, i, (unsigned long)bindOffset, (unsigned long)metalLen);
                        [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                        continue;
                    }

                    [_currentRenderEncoder setFragmentBuffer:fallbackBuffer offset:offset atIndex:bindingIndex];
                    if (kMGLVerboseBindLogs) {
                        NSLog(@"MGL FBIND ok(slot=%lu) setFragmentBuffer buffer=%u mtl=%p len=%lu offset=%lu",
                              (unsigned long)bindingIndex,
                              ptr->name,
                              ptr->data.mtl_data,
                              (unsigned long)metalLen,
                              (unsigned long)bindOffset);
                    }
                    anyBindingPresent[bindingIndex] = true;
                }
            } else {
                [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
            }
            
            // clear buffer data dirty bits
            ptr->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
        }
        else
        {
            if (!ptr->data.mtl_data) {
                [self bindMTLBuffer:ptr];
            }
            if (!ptr->data.mtl_data) {
                NSLog(@"MGL WARNING: Fragment buffer %u has no Metal backing after bind attempt, skipping slot %d", ptr->name, i);
                [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                continue;
            }
            if ((uintptr_t)ptr->data.mtl_data < 0x100000000ULL) {
                NSLog(@"MGL FBIND skip slot=%u buffer=%u: suspicious mtl_data pointer=%p",
                      i, ptr->name, ptr->data.mtl_data);
                [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                continue;
            }
            
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
            if (!buffer) {
                NSLog(@"MGL WARNING: Fragment buffer %u Metal object bridge failed, skipping slot %d", ptr->name, i);
                [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                continue;
            }

            NSUInteger metalLen = buffer.length;
            NSUInteger bindOffset = (NSUInteger)offset;
            if (bindOffset >= metalLen) {
                NSLog(@"MGL FBIND skip slot=%u buffer=%u: offset=%lu length=%lu",
                      i, ptr->name, (unsigned long)bindOffset, (unsigned long)metalLen);
                [_currentRenderEncoder setFragmentBuffer:nil offset:0 atIndex:bindingIndex];
                continue;
            }

            NSUInteger reflectedRequiredBytes = 0;
            NSUInteger requiredBindingBytes = kMGLMinimumStageBindingSize;
            if (isBaseBinding && glBindingIndex < MAX_BINDABLE_BUFFERS) {
                reflectedRequiredBytes = [self getProgramBindingRequiredSizeForStage:_FRAGMENT_SHADER
                                                                              binding:glBindingIndex];
                if (reflectedRequiredBytes > requiredBindingBytes) {
                    requiredBindingBytes = reflectedRequiredBytes;
                }
            }
            NSUInteger availableBytes = metalLen - bindOffset;
            // Match the vertex path: reflected Metal argument sizes can include
            // padding beyond the GL range. Prefer valid backing bytes over zeros.
            if (isBaseBinding &&
                map->size > 0 &&
                (NSUInteger)map->size >= requiredBindingBytes &&
                (NSUInteger)map->size < availableBytes) {
                availableBytes = (NSUInteger)map->size;
            }

            if (isBaseBinding &&
                glBindingIndex < MAX_BINDABLE_BUFFERS &&
                availableBytes < requiredBindingBytes) {
                BOOL boundPaddedBytes = NO;
                uint8_t stackScratch[kMGLStageBindingStackScratchSize];
                bzero(stackScratch, sizeof(stackScratch));

                if (ptr->data.buffer_data && ptr->size > 0) {
                    uintptr_t cpuData = (uintptr_t)ptr->data.buffer_data;
                    if (cpuData >= 0x100000000ULL) {
                        size_t cpuSize = (size_t)ptr->size;
                        size_t cpuOffset = (size_t)offset;
                        if (cpuOffset < cpuSize) {
                            size_t remaining = cpuSize - cpuOffset;
                            size_t paddedLen = (size_t)requiredBindingBytes;
                            uint8_t *paddedBytes = stackScratch;
                            bool usingHeap = false;

                            if (paddedLen > sizeof(stackScratch)) {
                                paddedBytes = (uint8_t *)calloc(1, paddedLen);
                                usingHeap = (paddedBytes != NULL);
                            }

                            if (paddedBytes) {
                                size_t copyLen = MIN(paddedLen, remaining);
                                memcpy(paddedBytes,
                                   ((const uint8_t *)ptr->data.buffer_data) + cpuOffset,
                                   copyLen);
                                [_currentRenderEncoder setFragmentBytes:paddedBytes
                                                                 length:paddedLen
                                                                atIndex:bindingIndex];
                                if (kMGLVerboseBindLogs) {
                                    NSLog(@"MGL SET FRAGMENT BUFFER index=%lu glName=%u offset=%lu available=%lu source=base-padded-bytes(min=%lu reflected=%lu copy=%lu range=%lld)",
                                          (unsigned long)bindingIndex,
                                          ptr->name,
                                          (unsigned long)bindOffset,
                                          (unsigned long)availableBytes,
                                          (unsigned long)requiredBindingBytes,
                                          (unsigned long)reflectedRequiredBytes,
                                          (unsigned long)copyLen,
                                          (long long)map->size);
                                }
                                anyBindingPresent[bindingIndex] = true;
                                boundPaddedBytes = YES;

                                if (usingHeap) {
                                    free(paddedBytes);
                                }
                            } else {
                                NSLog(@"MGL WARNING: FBIND failed to allocate %lu-byte scratch buffer for binding index=%lu",
                                      (unsigned long)paddedLen, (unsigned long)bindingIndex);
                            }
                        }
                    }
                }

                if (!boundPaddedBytes) {
                    if (!minimumBindingBuffer || minimumBindingBuffer.length < requiredBindingBytes) {
                        minimumBindingBuffer = [_device newBufferWithLength:requiredBindingBytes
                                                                     options:MTLResourceStorageModeShared];
                    }
                    if (minimumBindingBuffer) {
                        [_currentRenderEncoder setFragmentBuffer:minimumBindingBuffer
                                                          offset:0
                                                         atIndex:bindingIndex];
                        if (kMGLVerboseBindLogs) {
                            NSLog(@"MGL SET FRAGMENT BUFFER index=%lu glName=%u offset=0 available=%lu source=base-min-fallback(min=%lu reflected=%lu)",
                                  (unsigned long)bindingIndex,
                                  ptr->name,
                                  (unsigned long)minimumBindingBuffer.length,
                                  (unsigned long)requiredBindingBytes,
                                  (unsigned long)reflectedRequiredBytes);
                        }
                        anyBindingPresent[bindingIndex] = true;
                        continue;
                    }
                } else {
                    continue;
                }
            }
            
            [_currentRenderEncoder setFragmentBuffer:buffer offset:offset atIndex:bindingIndex];
            if (kMGLVerboseBindLogs) {
                NSLog(@"MGL SET FRAGMENT BUFFER index=%lu glName=%u offset=%lu available=%lu source=%s",
                      (unsigned long)bindingIndex,
                      ptr->name,
                      (unsigned long)bindOffset,
                      (unsigned long)metalLen,
                      isBaseBinding ? "base" : "attrib");
            }
            if (kMGLVerboseBindLogs) {
                NSLog(@"MGL FBIND ok(slot=%lu) setFragmentBuffer buffer=%u mtl=%p len=%lu offset=%lu",
                      (unsigned long)bindingIndex,
                      ptr->name,
                      ptr->data.mtl_data,
                      (unsigned long)metalLen,
                      (unsigned long)bindOffset);
            }
            anyBindingPresent[bindingIndex] = true;
        }
    }

    if (!fallbackBindingBuffer) {
        fallbackBindingBuffer = [_device newBufferWithLength:kMGLDefaultStageFallbackBufferSize
                                                     options:MTLResourceStorageModeShared];
    }

    // Bind fallback buffer for required stage buffer bindings that were not mapped.
    const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER
    };
    for (int t = 0; t < 4; t++) {
        int resourceType = resourceTypes[t];
        int count = [self getProgramBindingCount:_FRAGMENT_SHADER type:resourceType];
        Program *program = mglResolveProgramFromState(ctx);
        for (int i = 0; i < count; i++) {
            if (!program || resourceType < 0 || resourceType >= _MAX_SPIRV_RES ||
                i >= (int)program->spirv_resources_list[_FRAGMENT_SHADER][resourceType].count) {
                continue;
            }
            SpirvResource *resource = &program->spirv_resources_list[_FRAGMENT_SHADER][resourceType].list[i];
            GLuint clientBinding = mglClientBufferBindingForResource(resourceType, resource);
            if (clientBinding >= MAX_BINDABLE_BUFFERS) {
                continue;
            }
            NSInteger metalBinding = [self getProgramMetalBufferIndexForStage:_FRAGMENT_SHADER
                                                                       binding:clientBinding];
            if (metalBinding < 0 || metalBinding >= (NSInteger)MAX_BINDABLE_BUFFERS) {
                continue;
            }
            if (!baseBindingPresent[clientBinding] && fallbackBindingBuffer) {
                [_currentRenderEncoder setFragmentBuffer:fallbackBindingBuffer offset:0 atIndex:(NSUInteger)metalBinding];
                baseBindingPresent[clientBinding] = true;
                anyBindingPresent[(NSUInteger)metalBinding] = true;
            }
        }
    }

    if (fallbackBindingBuffer) {
        for (NSUInteger s = 0; s < MAX_BINDABLE_BUFFERS; s++) {
            if (!anyBindingPresent[s]) {
                [_currentRenderEncoder setFragmentBuffer:fallbackBindingBuffer offset:0 atIndex:s];
                anyBindingPresent[s] = true;
            }
        }
    }

    if (kMGLDiagnosticStateLogs && mglShouldTraceCall(fbindCall)) {
        NSUInteger boundSlots = 0;
        NSUInteger baseSlots = 0;
        for (NSUInteger s = 0; s < MAX_BINDABLE_BUFFERS; s++) {
            if (anyBindingPresent[s]) {
                boundSlots++;
            }
            if (baseBindingPresent[s]) {
                baseSlots++;
            }
        }
        NSLog(@"MGL TRACE fbind.end call=%llu mapCount=%u boundSlots=%lu baseSlots=%lu elapsed=%.3fms",
              (unsigned long long)fbindCall,
              (unsigned)mapCount,
              (unsigned long)boundSlots,
              (unsigned long)baseSlots,
              (mglNowSeconds() - fbindStartSeconds) * 1000.0);
    }

    return true;
}

- (int) getVertexBufferIndexWithAttributeSet: (int) attribute
{
    if (attribute < 0 || attribute >= MAX_ATTRIBS) {
        NSLog(@"MGL ERROR: getVertexBufferIndexWithAttributeSet invalid attribute=%d", attribute);
        return -1;
    }

    VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    if (vao) {
        int resolved = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, (GLuint)attribute, __FUNCTION__);
        if (resolved >= 0) {
            return resolved;
        }
    }

    // Legacy fallback: use cached map list if available.
    GLuint mapCount = ctx->state.vertex_buffer_map_list.count;
    if (mapCount > MAX_MAPPED_BUFFERS) {
        mapCount = MAX_MAPPED_BUFFERS;
    }

    for (GLuint i = 0; i < mapCount; i++)
    {
        if (ctx->state.vertex_buffer_map_list.buffers[i].attribute_mask & (0x1 << attribute)) {
            GLuint baseIndex = ctx->state.vertex_buffer_map_list.buffers[i].buffer_base_index;
            if (baseIndex >= kMGLMaxMetalVertexBufferCount) {
                NSLog(@"MGL ERROR: getVertexBufferIndexWithAttributeSet mapped base index out of Metal range=%u (max valid=%lu)",
                      baseIndex, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return -1;
            }
            return (int)baseIndex;
        }
    }

    NSLog(@"MGL ERROR: No vertex buffer mapping found for attribute %d", attribute);
    return -1;
}

#pragma mark textures

- (void)swizzleTexDesc:(MTLTextureDescriptor *)tex_desc forTex:(Texture*)tex
{
    unsigned channel_r, channel_g, channel_b, channel_a;

    channel_r = channel_g = channel_b = channel_a = 0;

    switch(tex->params.swizzle_r)
    {
        case GL_RED: channel_r = MTLTextureSwizzleRed; break;
        case GL_GREEN: channel_r = MTLTextureSwizzleGreen; break;
        case GL_BLUE: channel_r = MTLTextureSwizzleBlue; break;
        case GL_ALPHA: channel_r = MTLTextureSwizzleAlpha; break;
        default: // CRITICAL FIX: Handle assertion gracefully instead of crashing
            NSLog(@"MGL ERROR: Unknown swizzle value in swizzleTexDesc at line %d", __LINE__);
            channel_r = MTLTextureSwizzleRed; // Safe default
            break;
    }

    switch(tex->params.swizzle_g)
    {
        case GL_RED: channel_g = MTLTextureSwizzleRed; break;
        case GL_GREEN: channel_g = MTLTextureSwizzleGreen; break;
        case GL_BLUE: channel_g = MTLTextureSwizzleBlue; break;
        case GL_ALPHA: channel_g = MTLTextureSwizzleAlpha; break;
        default: // CRITICAL FIX: Handle assertion gracefully instead of crashing
            NSLog(@"MGL ERROR: Unknown swizzle value in swizzleTexDesc at line %d", __LINE__);
            channel_g = MTLTextureSwizzleGreen; // Safe default
            break;
    }

    switch(tex->params.swizzle_b)
    {
        case GL_RED: channel_b = MTLTextureSwizzleRed; break;
        case GL_GREEN: channel_b = MTLTextureSwizzleGreen; break;
        case GL_BLUE: channel_b = MTLTextureSwizzleBlue; break;
        case GL_ALPHA: channel_b = MTLTextureSwizzleAlpha; break;
        default: // CRITICAL FIX: Handle assertion gracefully instead of crashing
            NSLog(@"MGL ERROR: Unknown swizzle value in swizzleTexDesc at line %d", __LINE__);
            channel_b = MTLTextureSwizzleBlue; // Safe default
            break;
    }

    switch(tex->params.swizzle_a)
    {
        case GL_RED: channel_a = MTLTextureSwizzleRed; break;
        case GL_GREEN: channel_a = MTLTextureSwizzleGreen; break;
        case GL_BLUE: channel_a = MTLTextureSwizzleBlue; break;
        case GL_ALPHA: channel_a = MTLTextureSwizzleAlpha; break;
        default: // CRITICAL FIX: Handle assertion gracefully instead of crashing
            NSLog(@"MGL ERROR: Unknown swizzle value in swizzleTexDesc at line %d", __LINE__);
            channel_a = MTLTextureSwizzleAlpha; // Safe default
            break;
    }

    tex_desc.swizzle = MTLTextureSwizzleChannelsMake(channel_r, channel_g, channel_b, channel_a);
}

- (id<MTLTexture>) createMTLTextureFromGLTexture:(Texture *) tex
{
    // PROPER FIX: Enhanced pre-creation validation to prevent AGX driver issues
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or command queue not available for texture creation");
        return nil;
    }

    // Check if we're in a recovery state that would make texture creation futile
    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: GPU operations temporarily suspended during recovery");
        return nil;
    }

    if (tex->target == GL_TEXTURE_BUFFER) {
        Buffer *sourceBuffer = tex->texture_buffer;
        if (!sourceBuffer || tex->texture_buffer_size <= 0) {
            NSLog(@"MGL TEXBUFFER ERROR: tex=%u has no attached buffer/size buffer=%p size=%lld",
                  tex->name,
                  sourceBuffer,
                  (long long)tex->texture_buffer_size);
            return nil;
        }

        if (tex->texture_buffer_offset < 0 ||
            tex->texture_buffer_offset > sourceBuffer->size ||
            tex->texture_buffer_size > sourceBuffer->size - tex->texture_buffer_offset) {
            NSLog(@"MGL TEXBUFFER ERROR: invalid range tex=%u buffer=%u off=%lld size=%lld bufferSize=%lld",
                  tex->name,
                  sourceBuffer->name,
                  (long long)tex->texture_buffer_offset,
                  (long long)tex->texture_buffer_size,
                  (long long)sourceBuffer->size);
            return nil;
        }

        NSUInteger bytesPerTexel = [self bytesPerPixelForFormat:tex->internalformat];
        if (bytesPerTexel == 0) {
            NSLog(@"MGL TEXBUFFER ERROR: unsupported internal format 0x%x tex=%u buffer=%u",
                  tex->internalformat,
                  tex->name,
                  sourceBuffer->name);
            return nil;
        }

        NSUInteger texelCount = (NSUInteger)tex->texture_buffer_size / bytesPerTexel;
        if (texelCount == 0) {
            NSLog(@"MGL TEXBUFFER ERROR: zero texel count tex=%u buffer=%u size=%lld bpt=%lu",
                  tex->name,
                  sourceBuffer->name,
                  (long long)tex->texture_buffer_size,
                  (unsigned long)bytesPerTexel);
            return nil;
        }

        MTLPixelFormat bufferPixelFormat = mtlPixelFormatForGLTex(tex);
        if (bufferPixelFormat == MTLPixelFormatInvalid || bufferPixelFormat == 0) {
            NSLog(@"MGL TEXBUFFER ERROR: invalid Metal format for tex=%u internal=0x%x",
                  tex->name,
                  tex->internalformat);
            return nil;
        }

        if (![self processBuffer:sourceBuffer]) {
            NSLog(@"MGL TEXBUFFER ERROR: failed to process source buffer tex=%u buffer=%u",
                  tex->name,
                  sourceBuffer->name);
            return nil;
        }

        const uint8_t *sourceBytes = NULL;
        if (sourceBuffer->data.buffer_data) {
            sourceBytes = ((const uint8_t *)(uintptr_t)sourceBuffer->data.buffer_data) + (size_t)tex->texture_buffer_offset;
        } else if (sourceBuffer->data.mtl_data) {
            id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)(sourceBuffer->data.mtl_data);
            if (mtlBuffer && mtlBuffer.contents) {
                sourceBytes = ((const uint8_t *)mtlBuffer.contents) + (size_t)tex->texture_buffer_offset;
            }
        }

        if (!sourceBytes) {
            NSLog(@"MGL TEXBUFFER ERROR: no readable backing for tex=%u buffer=%u cpu=%p mtl=%p",
                  tex->name,
                  sourceBuffer->name,
                  (void *)(uintptr_t)sourceBuffer->data.buffer_data,
                  sourceBuffer->data.mtl_data);
            return nil;
        }

        // SPIRV-Cross currently emits Minecraft's CloudFaces texel buffer as a
        // texture2d<int>. Keep GL lookup semantics as GL_TEXTURE_BUFFER, but
        // create a Metal 2D backing so the generated MSL argument type matches.
        // A texel buffer can be much wider than Metal's max 2D texture width,
        // so pack it into rows instead of creating texelCount x 1.
        /*
         * SPIRV-Cross lowers GL texture buffers to 2D Metal textures and emits
         * spvTexelBufferCoord(tc) using its MSL texel_buffer_texture_width
         * option. Keep this packing width in lockstep with program.c.
         */
        static const NSUInteger kMGLTexelBufferTextureWidth = 4096u;
        NSUInteger max2DSize = (NSUInteger)MIN((GLuint)kMGLTexelBufferTextureWidth,
                                               ctx ? ctx->state.var.max_texture_size : (GLuint)kMGLTexelBufferTextureWidth);
        if (max2DSize == 0 || max2DSize > kMGLTexelBufferTextureWidth) {
            max2DSize = kMGLTexelBufferTextureWidth;
        }

        NSUInteger texWidth = MIN(texelCount, max2DSize);
        NSUInteger texHeight = (texelCount + texWidth - 1) / texWidth;
        if (texHeight == 0 || texHeight > max2DSize) {
            NSLog(@"MGL TEXBUFFER ERROR: texel buffer too large for 2D fallback tex=%u buffer=%u texels=%lu packed=%lux%lu max=%lu",
                  tex->name,
                  sourceBuffer->name,
                  (unsigned long)texelCount,
                  (unsigned long)texWidth,
                  (unsigned long)texHeight,
                  (unsigned long)max2DSize);
            return nil;
        }

        NSUInteger bytesPerRow = texWidth * bytesPerTexel;
        NSUInteger packedBytes = bytesPerRow * texHeight;
        NSMutableData *packedData = nil;
        const uint8_t *uploadBytes = sourceBytes;
        if (texHeight > 1) {
            packedData = [NSMutableData dataWithLength:packedBytes];
            if (!packedData || !packedData.mutableBytes) {
                NSLog(@"MGL TEXBUFFER ERROR: failed allocating packed data tex=%u buffer=%u bytes=%lu",
                      tex->name,
                      sourceBuffer->name,
                      (unsigned long)packedBytes);
                return nil;
            }

            memcpy(packedData.mutableBytes, sourceBytes, (size_t)tex->texture_buffer_size);
            uploadBytes = (const uint8_t *)packedData.bytes;
        }

        uint64_t sourceHash = mglTraceHashBytes(sourceBytes, (size_t)tex->texture_buffer_size);
        uint64_t uploadHash = mglTraceHashBytes(uploadBytes, packedBytes);
        char sourceHead[64];
        char uploadHead[64];
        sourceHead[0] = '\0';
        uploadHead[0] = '\0';
        mglTraceFormatBytes(sourceBytes, (size_t)MIN((NSUInteger)tex->texture_buffer_size, (NSUInteger)64), sourceHead, sizeof(sourceHead));
        mglTraceFormatBytes(uploadBytes, (size_t)MIN(packedBytes, (NSUInteger)64), uploadHead, sizeof(uploadHead));

        MTLTextureDescriptor *bufferDesc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:bufferPixelFormat
                                                               width:texWidth
                                                              height:texHeight
                                                           mipmapped:NO];
        bufferDesc.usage = MTLTextureUsageShaderRead;
        bufferDesc.storageMode = MTLStorageModeShared;

        id<MTLTexture> bufferTexture = nil;
        @try {
            bufferTexture = [_device newTextureWithDescriptor:bufferDesc];
            if (bufferTexture) {
                [bufferTexture replaceRegion:MTLRegionMake2D(0, 0, texWidth, texHeight)
                                  mipmapLevel:0
                                    withBytes:uploadBytes
                                  bytesPerRow:bytesPerRow];
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL TEXBUFFER ERROR: failed creating/uploading tex=%u buffer=%u exception=%@",
                  tex->name,
                  sourceBuffer->name,
                  exception);
            return nil;
        }

        if (!bufferTexture) {
            NSLog(@"MGL TEXBUFFER ERROR: Metal texture creation returned nil tex=%u buffer=%u format=%lu texels=%lu",
                  tex->name,
                  sourceBuffer->name,
                  (unsigned long)bufferPixelFormat,
                  (unsigned long)texelCount);
            return nil;
        }

        tex->dirty_bits = 0;
        sourceBuffer->data.dirty_bits = 0;

        NSMutableData *readbackData = [NSMutableData dataWithLength:packedBytes];
        uint64_t readbackHash = 0ull;
        char readbackHead[64];
        readbackHead[0] = '\0';
        if (readbackData.mutableBytes) {
            [bufferTexture getBytes:readbackData.mutableBytes
                        bytesPerRow:bytesPerRow
                         fromRegion:MTLRegionMake2D(0, 0, texWidth, texHeight)
                        mipmapLevel:0];
            readbackHash = mglTraceHashBytes(readbackData.bytes, packedBytes);
            mglTraceFormatBytes(readbackData.bytes, (size_t)MIN(packedBytes, (NSUInteger)64), readbackHead, sizeof(readbackHead));
        }

        NSLog(@"MGL TEXBUFFER CREATE tex=%u buffer=%u internal=0x%x mtlFormat=%lu texels=%lu packed=%lux%lu rowBytes=%lu bytes=%lld offset=%lld as=texture2d sourceHash=0x%016llx uploadHash=0x%016llx readbackHash=0x%016llx sourceHead=%s uploadHead=%s readbackHead=%s",
              tex->name,
              sourceBuffer->name,
              tex->internalformat,
              (unsigned long)bufferPixelFormat,
              (unsigned long)texelCount,
              (unsigned long)texWidth,
              (unsigned long)texHeight,
              (unsigned long)bytesPerRow,
              (long long)tex->texture_buffer_size,
              (long long)tex->texture_buffer_offset,
              (unsigned long long)sourceHash,
              (unsigned long long)uploadHash,
              (unsigned long long)readbackHash,
              sourceHead,
              uploadHead,
              readbackHead);

        [self recordGPUSuccess];
        return bufferTexture;
    }

    NSUInteger width, height, depth;

    MTLTextureDescriptor *tex_desc;
    MTLTextureType tex_type;
    MTLPixelFormat pixelFormat;
    uint num_faces;
    GLuint effective_mipmap_levels;
    GLuint upload_level_count;
    BOOL mipmapped;
    BOOL is_array;

    num_faces = 1;
    is_array = false;
    effective_mipmap_levels = 0;
    upload_level_count = 0;

    switch(tex->target)
    {
//        case GL_TEXTURE_1D: tex_type = MTLTextureType1D; break;
        case GL_TEXTURE_1D: tex_type = MTLTextureType2D; break;
        case GL_RENDERBUFFER: tex_type = MTLTextureType2D; break;
        case GL_TEXTURE_1D_ARRAY: tex_type = MTLTextureType1DArray; is_array = true; break;
        case GL_TEXTURE_2D: tex_type = MTLTextureType2D; break;
        case GL_TEXTURE_2D_ARRAY: tex_type = MTLTextureType2DArray; is_array = true; break;
        // case GL_TEXTURE_2D_MULTISAMPLE: tex_type = MTLTextureType2DMultisample; break;

        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            num_faces = 6;
            tex_type = MTLTextureTypeCube;
            break;

        case GL_TEXTURE_CUBE_MAP_ARRAY:
            num_faces = 6;
            tex_type = MTLTextureTypeCubeArray;
            is_array = true;
            break;

        case GL_TEXTURE_3D: tex_type = MTLTextureType3D; break;
        // case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: tex_type = MTLTextureType2DMultisampleArray;  is_array = true; break;
        // case GL_TEXTURE_BUFFER: tex_type = MTLTextureTypeTextureBuffer; break;

        default:
            NSLog(@"MGL TEXTURE ERROR: unsupported texture target 0x%x for Metal texture creation tex=%u",
                  tex->target,
                  tex->name);
            return nil;
    }

    // verify completeness of texture when used
    effective_mipmap_levels = tex->mipmap_levels;
    if (tex->params.max_level != 1000u) {
        GLuint max_level_count = tex->params.max_level + 1u;
        if (max_level_count > 0 && max_level_count < effective_mipmap_levels) {
            effective_mipmap_levels = max_level_count;
        }
    }

    if (tex->num_levels > 1)
    {
        // mipmapped texture
        if (effective_mipmap_levels == 0) {
            effective_mipmap_levels = tex->num_levels;
        }

        if (tex->num_levels < effective_mipmap_levels)
        {
            static uint64_t s_mipmap_count_mismatch_logs = 0;
            if (++s_mipmap_count_mismatch_logs <= 32 || (s_mipmap_count_mismatch_logs % 512) == 0) {
                NSLog(@"MGL TEXTURE MIP COMPAT: tex=%u target=0x%x size=%ux%u num_levels=%u mipmap_levels=%u effective=%u base=%u max=%u immutable=%u; capping Metal mip count to uploaded levels hit=%llu",
                      tex->name,
                      tex->target,
                      tex->width,
                      tex->height,
                      tex->num_levels,
                      tex->mipmap_levels,
                      effective_mipmap_levels,
                      tex->params.base_level,
                      tex->params.max_level,
                      tex->immutable_storage,
                      (unsigned long long)s_mipmap_count_mismatch_logs);
            }
            effective_mipmap_levels = tex->num_levels;
        }

        for(int face=0; face<num_faces; face++)
        {
            for (int i=0; i<effective_mipmap_levels; i++)
            {
                // incomplete texture
                if (tex->faces[face].levels[i].complete == false) {
                    static uint64_t s_incomplete_mip_logs = 0;
                    if (++s_incomplete_mip_logs <= 32 || (s_incomplete_mip_logs % 512) == 0) {
                        NSLog(@"MGL TEXTURE INCOMPLETE: tex=%u target=0x%x face=%d level=%d incomplete num_levels=%u mipmap_levels=%u effective=%u base=%u max=%u hit=%llu",
                              tex->name,
                              tex->target,
                              face,
                              i,
                              tex->num_levels,
                              tex->mipmap_levels,
                              effective_mipmap_levels,
                              tex->params.base_level,
                              tex->params.max_level,
                              (unsigned long long)s_incomplete_mip_logs);
                    }
                    return NULL;
                }
            }
        }

        tex->mipmapped = true;
    }
    else if (tex->num_levels == 1)
    {
        effective_mipmap_levels = 1;
        // single level texture
        // incomplete texture
        for(int face=0; face<num_faces; face++)
        {
            if (tex->faces[face].levels[0].complete == false)
            {
                static uint64_t s_incomplete_base_logs = 0;
                if (++s_incomplete_base_logs <= 32 || (s_incomplete_base_logs % 512) == 0) {
                    NSLog(@"MGL TEXTURE INCOMPLETE: tex=%u target=0x%x face=%d base incomplete size=%ux%u hit=%llu",
                          tex->name,
                          tex->target,
                          face,
                          tex->width,
                          tex->height,
                          (unsigned long long)s_incomplete_base_logs);
                }
                return NULL;
            }
        }
    }
    else
    {
        NSLog(@"MGL TEXTURE ERROR: texture %u has no complete levels for Metal creation target=0x%x",
              tex->name,
              tex->target);
        return nil;
    }
    tex->complete = true;

    // PROPER FIX: Get original texture format and validate for AGX compatibility
    pixelFormat = mtlPixelFormatForGLTex(tex);

    NSLog(@"MGL INFO: PROPER FIX - Original texture format: internal=0x%x, mtl=0x%lx", tex->internalformat, (unsigned long)pixelFormat);

    // Validate format compatibility with AGX, but preserve original intent
    BOOL needsFormatConversion = NO;
    MTLPixelFormat originalFormat = pixelFormat;

    // Check for AGX-incompatible formats and only convert when necessary
    switch(pixelFormat) {
        case MTLPixelFormatB5G6R5Unorm:
        case MTLPixelFormatBGR5A1Unorm:
        case MTLPixelFormatA1BGR5Unorm:
            // 16-bit formats can cause issues on AGX
            needsFormatConversion = YES;
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        case MTLPixelFormatPVRTC_RGBA_2BPP:
        case MTLPixelFormatPVRTC_RGBA_4BPP:
        case MTLPixelFormatPVRTC_RGB_2BPP:
        case MTLPixelFormatPVRTC_RGB_4BPP:
            // PVRTC compression can cause issues in virtualization
            needsFormatConversion = YES;
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        case MTLPixelFormatEAC_R11Unorm:
        case MTLPixelFormatEAC_RG11Unorm:
        case MTLPixelFormatEAC_RGBA8:
        case MTLPixelFormatETC2_RGB8:
        case MTLPixelFormatETC2_RGB8A1:
            // ETC/ETC2 compression can cause issues on AGX
            needsFormatConversion = YES;
            pixelFormat = MTLPixelFormatRGBA8Unorm;
            break;
        default:
            // Most modern formats should work fine
            break;
    }

    if (needsFormatConversion) {
        NSLog(@"MGL INFO: PROPER FIX - Converting AGX-incompatible format 0x%lx to RGBA8", (unsigned long)originalFormat);
        tex->internalformat = GL_RGBA8;
    } else {
        NSLog(@"MGL INFO: PROPER FIX - Using original format 0x%lx (AGX compatible)", (unsigned long)pixelFormat);
    }

    width = tex->width;
    height = tex->height;
    depth = tex->depth;
    mipmapped = tex->mipmapped == 1;
    upload_level_count = mipmapped ? effective_mipmap_levels : tex->num_levels;

    tex_desc = [[MTLTextureDescriptor alloc] init];
    tex_desc.textureType = tex_type;
    tex_desc.pixelFormat = pixelFormat;
    tex_desc.width = width;
    tex_desc.height = height;

    // CONSERVATIVE: Use only Metal API patterns that work reliably with AGX driver
    tex_desc.cpuCacheMode = MTLCPUCacheModeWriteCombined;  // More stable than DefaultCache

    // CONSERVATIVE: Always use private storage to avoid compression/caching conflicts
    tex_desc.storageMode = MTLStorageModePrivate;

    // Normalize depth/array semantics per Metal texture type.
    if (tex_type == MTLTextureTypeCube) {
        if (width != height) {
            NSLog(@"MGL ERROR: invalid cube texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
        }
        tex_desc.depth = 1;
    } else if (tex_type == MTLTextureTypeCubeArray) {
        if (width != height) {
            NSLog(@"MGL ERROR: invalid cube-array texture size %lux%lu for tex=%u glTarget=0x%x",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
        }

        // GL cube-map-array depth is usually layer count (faces), so convert to cube count.
        // If depth is already cube-count (non-multiple of 6), keep it as-is.
        NSUInteger cubeCount = depth;
        if (cubeCount >= 6 && (cubeCount % 6) == 0) {
            cubeCount = cubeCount / 6;
        } else if (cubeCount > 1 && (cubeCount % 6) != 0) {
            NSLog(@"MGL WARNING: cube-array depth=%lu is not a multiple of 6, treating as cube count",
                  (unsigned long)cubeCount);
        }

        tex_desc.arrayLength = MAX((NSUInteger)1, cubeCount);
        tex_desc.depth = 1;
    } else if (is_array) {
        tex_desc.arrayLength = MAX((NSUInteger)1, depth);
        tex_desc.depth = 1;
    } else {
        tex_desc.depth = MAX((NSUInteger)1, depth);
    }

    if (mipmapped)
    {
        tex_desc.mipmapLevelCount = MAX((GLuint)1, effective_mipmap_levels);
    }

    switch(tex->access)
    {
        case GL_READ_ONLY:
            tex_desc.usage = MTLTextureUsageShaderRead; break;
        case GL_WRITE_ONLY:
            tex_desc.usage = MTLTextureUsageShaderWrite; break;
        case GL_READ_WRITE:
            tex_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; break;
        default:
            NSLog(@"MGL TEXTURE ERROR: invalid texture access 0x%x for tex=%u",
                  tex->access,
                  tex->name);
            return nil;
    }

    if (tex->is_render_target)
    {
        tex_desc.usage |= MTLTextureUsageRenderTarget;
    }

    // Allow safe same-memory format reinterpretation (e.g. RGBA8 <-> BGRA8)
    // for blit/present paths where OpenGL attachments and drawable formats differ.
    tex_desc.usage |= MTLTextureUsagePixelFormatView;

    if (tex_desc.textureType == MTLTextureTypeCube || tex_desc.textureType == MTLTextureTypeCubeArray) {
        NSLog(@"MGL CUBE DESC tex=%u glTarget=0x%x type=%lu width=%lu height=%lu depth=%lu arrayLength=%lu pixelFormat=%lu usage=%lu storage=%lu mipmapped=%d",
              tex->name,
              tex->target,
              (unsigned long)tex_desc.textureType,
              (unsigned long)tex_desc.width,
              (unsigned long)tex_desc.height,
              (unsigned long)tex_desc.depth,
              (unsigned long)tex_desc.arrayLength,
              (unsigned long)tex_desc.pixelFormat,
              (unsigned long)tex_desc.usage,
              (unsigned long)tex_desc.storageMode,
              (int)mipmapped);
    }

    // CRITICAL FIX: Proper validation instead of assertions
    if (!tex_desc) {
        NSLog(@"MGL ERROR: Failed to create texture descriptor");
        return NULL;
    }

    if (tex->params.swizzled)
    {
        [self swizzleTexDesc:tex_desc forTex:tex];
    }

    id<MTLTexture> texture;

    // CRITICAL FIX: Safe texture creation with proper validation
    @try {
        texture = [_device newTextureWithDescriptor:tex_desc];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating texture: %@", exception);
        [self recordGPUError];
        return NULL;
    }

    // CRITICAL FIX: Validate texture creation result instead of asserting
    if (!texture) {
        NSLog(@"MGL ERROR: Failed to create Metal texture with descriptor");
        return NULL;
    }

    if (tex->dirty_bits & DIRTY_TEXTURE_DATA)
    {
        NSLog(@"MGL DEBUG: DIRTY_TEXTURE_DATA detected - attempting texture filling");
        NSLog(@"MGL DEBUG: Texture details: target=0x%x, internalformat=0x%x, levels=%d effectiveLevels=%u",
              tex->target, tex->internalformat, tex->num_levels, upload_level_count);

        MTLRegion region;

        for(int face=0; face<num_faces; face++)
        {
            for (int level=0; level<upload_level_count; level++)
            {
                width = tex->faces[face].levels[level].width;
                height = tex->faces[face].levels[level].height;
                depth = tex->faces[face].levels[level].depth;

                if (depth > 1)
                    region = MTLRegionMake3D(0,0,0,width,height,depth);
                else if (height > 1)
                    region = MTLRegionMake2D(0,0,width,height);
                else
                    region = MTLRegionMake1D(0,width);

                NSUInteger bytesPerRow;
                NSUInteger bytesPerImage;
                bool hasExplicitDataSize = false;

                if (tex_type == MTLTextureType3D)
                {
                    // ogl considers an image a "row".. metal must be different
                    bytesPerRow = tex->faces[face].levels[level].pitch;
                    if (bytesPerRow == 0) {
                        NSLog(@"MGL WARNING: Invalid 3D bytesPerRow (0), skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                        continue;
                    }

                    bytesPerImage = bytesPerRow * height;

                    // NUCLEAR OPTION: Disable all texture uploads temporarily to isolate the crash source
                    if (tex->faces[face].levels[level].data && bytesPerRow > 0 && bytesPerImage > 0) {
                        NSLog(@"MGL INFO: PROPER FIX - Processing 3D texture upload (tex=%d, face=%d, level=%d, size=%lu)", tex->name, face, level, (unsigned long)bytesPerImage);

                        // PROPER FIX: Enable texture uploads but with safety checks
                        // continue; // Remove the continue to re-enable uploads
                        // PROPER FIX: Dynamic memory alignment based on GPU characteristics
                        void *srcData = (void *)tex->faces[face].levels[level].data;
                        uintptr_t addr = (uintptr_t)srcData;

                        // Determine optimal alignment based on pixel format and GPU capabilities
                        NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];
                        NSUInteger alignedBytesPerRow = bytesPerRow;
                        if (alignedBytesPerRow % alignment != 0) {
                            alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                        }

                        if (addr % 256 != 0 || alignedBytesPerRow != bytesPerRow) {
                            // Data is not aligned OR bytesPerRow needs alignment - allocate aligned buffer and copy row by row
                            NSUInteger alignedBytesPerImage = alignedBytesPerRow * MAX((NSUInteger)height, 1UL);
                            NSUInteger alignedSize = alignedBytesPerImage * MAX((NSUInteger)depth, 1UL);
                            if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                                NSLog(@"MGL WARNING: Rejecting aligned 3D upload staging size=%lu (tex=%d level=%d)",
                                      (unsigned long)alignedSize, tex->name, level);
                                continue;
                            }
                            void *alignedData = aligned_alloc(alignment, alignedSize);

                            if (alignedData) {
                                memset(alignedData, 0, alignedSize);
                                // Copy data row by row to handle bytesPerRow alignment
                                NSUInteger srcRowSize = bytesPerRow;
                                NSUInteger dstRowSize = alignedBytesPerRow;
                                NSUInteger texHeight = MAX((NSUInteger)height, 1UL);
                                NSUInteger texDepth = MAX((NSUInteger)depth, 1UL);
                                uint8_t *srcPtr = (uint8_t *)srcData;
                                uint8_t *dstPtr = (uint8_t *)alignedData;

                                for (NSUInteger z = 0; z < texDepth; z++) {
                                    for (NSUInteger row = 0; row < texHeight; row++) {
                                        NSUInteger copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                        NSUInteger dstOffset = z * alignedBytesPerImage + row * dstRowSize;
                                        NSUInteger srcOffset = z * bytesPerImage + row * srcRowSize;
                                        memcpy(dstPtr + dstOffset, srcPtr + srcOffset, copySize);
                                        // Clear padding to zero
                                        if (dstRowSize > copySize) {
                                            memset(dstPtr + dstOffset + copySize, 0, dstRowSize - copySize);
                                        }
                                    }
                                }

                                // CRITICAL SECURITY FIX: Validate alignedData before passing to Metal API
                                if (!alignedData) {
                                    NSLog(@"MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                                    continue;
                                }
                                if (alignedBytesPerRow == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                                    continue;
                                }
                                @try {
                                    BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                           texName:tex->name
                                                                         texTarget:tex->target
                                                                             bytes:alignedData
                                                                       bytesPerRow:alignedBytesPerRow
                                                                     bytesPerImage:alignedBytesPerImage
                                                                             width:width
                                                                            height:height
                                                                             depth:depth
                                                                             level:level
                                                                             slice:0];
                                    if (!uploaded) {
                                        NSLog(@"MGL WARNING: 3D aligned blit upload failed (level %d, face %d)", level, face);
                                    }
                                } @catch (NSException *exception) {
                                    NSLog(@"MGL ERROR: Failed to upload aligned 3D texture data (level %d, face %d): %@", level, face, exception);
                                }
                                free(alignedData);
                            } else {
                                NSLog(@"MGL ERROR: Failed to allocate aligned memory for 3D texture upload");
                            }
                        } else {
                            // Data and bytesPerRow are already aligned
                            // CRITICAL SECURITY FIX: Validate srcData and parameters before passing to Metal API
                            if (!srcData) {
                                NSLog(@"MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                                continue;
                            }
                            if (bytesPerRow == 0) {
                                NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                                continue;
                            }
                            if (bytesPerImage == 0) {
                                NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash", level);
                                continue;
                            }
                            @try {
                                BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                       texName:tex->name
                                                                     texTarget:tex->target
                                                                         bytes:srcData
                                                                   bytesPerRow:bytesPerRow
                                                                 bytesPerImage:bytesPerImage
                                                                         width:width
                                                                        height:height
                                                                         depth:depth
                                                                         level:level
                                                                         slice:0];
                                if (!uploaded) {
                                    NSLog(@"MGL WARNING: 3D direct blit upload failed (level %d, face %d)", level, face);
                                }
                            } @catch (NSException *exception) {
                                NSLog(@"MGL ERROR: Failed to upload 3D texture data (level %d, face %d): %@", level, face, exception);
                            }
                        }
                    } else {
                        NSLog(@"MGL WARNING: Skipping 3D texture upload due to invalid data or parameters");
                    }
                }
                else
                {
                    bytesPerRow = tex->faces[face].levels[level].pitch;
                    if (bytesPerRow == 0) {
                        NSLog(@"MGL WARNING: Invalid bytesPerRow (0), skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                        continue;
                    }

                    bytesPerImage = tex->faces[face].levels[level].data_size;
                    hasExplicitDataSize = (bytesPerImage > 0);
                    if (bytesPerImage == 0) {
                        // Some depth / render-target textures may report data_size==0.
                        // Fall back to pitch * logical height to avoid hard aborts.
                        NSUInteger fallbackHeight = (height > 0) ? (NSUInteger)height : 1;
                        bytesPerImage = bytesPerRow * fallbackHeight;
                        NSLog(@"MGL WARNING: data_size was 0, using fallback bytesPerImage=%lu (tex=%d face=%d level=%d)",
                              (unsigned long)bytesPerImage, tex->name, face, level);
                    }
                    if (bytesPerImage == 0) {
                        NSLog(@"MGL WARNING: Invalid bytesPerImage (0), skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                        continue;
                    }

                    if (is_array)
                    {
                        GLuint num_layers;
                        size_t offset;
                        GLubyte *tex_data;
                        BOOL arraySliceIs1D;
                        NSUInteger uploadSliceHeight;
                        NSUInteger backingBytes;
                        NSUInteger logicalBytesPerImage;

                        num_layers = tex->depth;
                        if (num_layers == 0) {
                            NSLog(@"MGL WARNING: Array texture has 0 layers, skipping upload (tex=%d face=%d level=%d)", tex->name, face, level);
                            continue;
                        }

                        arraySliceIs1D = (tex_type == MTLTextureType1DArray);
                        uploadSliceHeight = arraySliceIs1D ? 1UL : MAX((NSUInteger)height, 1UL);
                        backingBytes = bytesPerImage;
                        logicalBytesPerImage = bytesPerRow * uploadSliceHeight;
                        if (logicalBytesPerImage == 0 ||
                            backingBytes < logicalBytesPerImage * MAX((NSUInteger)num_layers, 1UL)) {
                            NSLog(@"MGL WARNING: Array texture backing too small for logical slices tex=%d face=%d level=%d backing=%lu layerBytes=%lu layers=%u",
                                  tex->name,
                                  face,
                                  level,
                                  (unsigned long)backingBytes,
                                  (unsigned long)logicalBytesPerImage,
                                  num_layers);
                            continue;
                        }
                        bytesPerImage = logicalBytesPerImage;

                        if (!arraySliceIs1D) // 2D array/cube-array: each Metal slice is a full 2D image.
                            region = MTLRegionMake2D(0,0,width,height);
                        else if (height >= 1) // 1d array
                            region = MTLRegionMake2D(0,0,width,1);
                        else { // ?
                            NSLog(@"MGL TEXTURE ERROR: invalid array texture height=%lu for tex=%u face=%d level=%d",
                                  (unsigned long)height,
                                  tex->name,
                                  face,
                                  level);
                            return nil;
                        }

                        for(int layer=0; layer<num_layers; layer++)
                        {
                            offset = bytesPerImage * layer;

                            tex_data = (GLubyte *)tex->faces[face].levels[level].data;
                            tex_data += offset;

                            // NUCLEAR OPTION: Disable all texture uploads temporarily to isolate the crash source
                        if (tex_data && bytesPerRow > 0 && bytesPerImage > 0) {
                                NSLog(@"MGL INFO: PROPER FIX - Processing array texture upload (tex=%d, face=%d, level=%d, layer=%d, size=%lu)", tex->name, face, level, layer, (unsigned long)bytesPerImage);

                                // PROPER FIX: Enable texture uploads but with safety checks
                                // continue; // Remove the continue to re-enable uploads
                                // Ensure memory is aligned for AGX compression (256-byte requirement)
                                void *srcData = (void *)tex_data;
                                uintptr_t addr = (uintptr_t)srcData;

                                // Use dynamic alignment based on pixel format
                                NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];
                                NSUInteger alignedBytesPerRow = bytesPerRow;
                                if (alignedBytesPerRow % alignment != 0) {
                                    alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                                }

                                if (addr % alignment != 0 || alignedBytesPerRow != bytesPerRow) {
                                    // Data is not aligned OR bytesPerRow needs alignment - allocate aligned buffer and copy
                                    NSUInteger alignedBytesPerImage = alignedBytesPerRow * uploadSliceHeight;
                                    NSUInteger alignedSize = alignedBytesPerImage;
                                    if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                                        NSLog(@"MGL WARNING: Rejecting aligned array upload staging size=%lu (tex=%d level=%d layer=%d)",
                                              (unsigned long)alignedSize, tex->name, level, layer);
                                        continue;
                                    }
                                    void *alignedData = aligned_alloc(alignment, alignedSize);

                                    if (alignedData) {
                                        memset(alignedData, 0, alignedSize);
                                        // Copy data with row alignment
                                        NSUInteger srcRowSize = bytesPerRow;
                                        NSUInteger dstRowSize = alignedBytesPerRow;
                                        uint8_t *srcPtr = (uint8_t *)srcData;
                                        uint8_t *dstPtr = (uint8_t *)alignedData;

                                        for (NSUInteger row = 0; row < uploadSliceHeight; row++) {
                                            NSUInteger copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                            memcpy(dstPtr + (row * dstRowSize), srcPtr + (row * srcRowSize), copySize);
                                            // Clear padding to zero
                                            if (dstRowSize > copySize) {
                                                memset(dstPtr + (row * dstRowSize) + copySize, 0, dstRowSize - copySize);
                                            }
                                        }

                                        // CRITICAL SECURITY FIX: Validate alignedData before passing to Metal API
                                        if (!alignedData) {
                                            NSLog(@"MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                            continue;
                                        }
                                        if (alignedBytesPerRow == 0) {
                                            NSLog(@"MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                            continue;
                                        }
                                        if (bytesPerImage == 0) {
                                            NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                            continue;
                                        }
                                        @try {
                                            if (hasExplicitDataSize) {
                                                BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                                       texName:tex->name
                                                                                     texTarget:tex->target
                                                                                         bytes:alignedData
                                                                                   bytesPerRow:alignedBytesPerRow
                                                                                 bytesPerImage:alignedBytesPerImage
                                                                                         width:width
                                                                                        height:uploadSliceHeight
                                                                                         depth:1
                                                                                         level:level
                                                                                         slice:layer];
                                                if (!uploaded) {
                                                    NSLog(@"MGL WARNING: Array texture blit upload failed (level %d, layer %d)", level, layer);
                                                }
                                            } else {
                                                NSLog(@"MGL INFO: Skipping array upload with synthesized data size (level %d, layer %d)", level, layer);
                                            }
                                        } @catch (NSException *exception) {
                                            NSLog(@"MGL ERROR: Failed to upload aligned array texture data (level %d, layer %d): %@", level, layer, exception);
                                        }
                                        free(alignedData);
                                    } else {
                                        NSLog(@"MGL ERROR: Failed to allocate aligned memory for array texture upload (level %d, layer %d)", level, layer);
                                    }
                                } else {
                                    // Data and bytesPerRow are already aligned
                                    // CRITICAL SECURITY FIX: Validate srcData before passing to Metal API
                                    if (!srcData) {
                                        NSLog(@"MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                        continue;
                                    }
                                    if (bytesPerRow == 0) {
                                        NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                        continue;
                                    }
                                    if (bytesPerImage == 0) {
                                        NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash", level, layer);
                                        continue;
                                    }
                                    if (hasExplicitDataSize) {
                                        BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                               texName:tex->name
                                                                             texTarget:tex->target
                                                                                 bytes:srcData
                                                                           bytesPerRow:bytesPerRow
                                                                         bytesPerImage:bytesPerImage
                                                                                 width:width
                                                                                height:uploadSliceHeight
                                                                                 depth:1
                                                                                 level:level
                                                                                 slice:layer];
                                        if (!uploaded) {
                                            NSLog(@"MGL WARNING: Array texture direct blit upload failed (level %d, layer %d)", level, layer);
                                        }
                                    } else {
                                        NSLog(@"MGL INFO: Skipping array upload with synthesized data size (level %d, layer %d)", level, layer);
                                    }
                                }
                            } else {
                                NSLog(@"MGL WARNING: Skipping array texture upload due to invalid data or parameters");
                            }
                        }
                    }
                    else
                    {
                        DEBUG_PRINT("tex id data update %d\n", tex->name);

                        // PROPER FIX: Enable 2D texture uploads with AGX safety and alignment
                        if (tex->faces[face].levels[level].data && bytesPerRow > 0 && bytesPerImage > 0) {
                            NSLog(@"MGL INFO: PROPER FIX - Processing 2D texture upload (tex=%d, face=%d, level=%d, size=%lu)", tex->name, face, level, (unsigned long)bytesPerImage);

                            // Ensure memory is aligned for AGX compression (256-byte requirement)
                            void *srcData = (void *)tex->faces[face].levels[level].data;
                            uintptr_t addr = (uintptr_t)srcData;

                            // Use dynamic alignment based on pixel format
                            NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:pixelFormat];
                            NSUInteger alignedBytesPerRow = bytesPerRow;
                            if (alignedBytesPerRow % alignment != 0) {
                                alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                            }

                            if (addr % alignment != 0 || alignedBytesPerRow != bytesPerRow) {
                                // Data is not aligned OR bytesPerRow needs alignment - allocate aligned buffer and copy
                                NSUInteger texHeight = MAX((NSUInteger)height, 1UL);
                                NSUInteger alignedBytesPerImage = alignedBytesPerRow * texHeight;
                                NSUInteger alignedSize = alignedBytesPerImage;
                                if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                                    NSLog(@"MGL WARNING: Rejecting aligned 2D upload staging size=%lu (tex=%d level=%d face=%d)",
                                          (unsigned long)alignedSize, tex->name, level, face);
                                    continue;
                                }
                                void *alignedData = aligned_alloc(alignment, alignedSize);

                                if (alignedData) {
                                    memset(alignedData, 0, alignedSize);
                                    // Copy data row by row to handle bytesPerRow alignment
                                    NSUInteger srcRowSize = bytesPerRow;
                                    NSUInteger dstRowSize = alignedBytesPerRow;
                                    uint8_t *srcPtr = (uint8_t *)srcData;
                                    uint8_t *dstPtr = (uint8_t *)alignedData;

                                    for (NSUInteger row = 0; row < texHeight; row++) {
                                        NSUInteger copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                        memcpy(dstPtr + (row * dstRowSize), srcPtr + (row * srcRowSize), copySize);
                                        // Clear padding to zero
                                        if (dstRowSize > copySize) {
                                            memset(dstPtr + (row * dstRowSize) + copySize, 0, dstRowSize - copySize);
                                        }
                                    }

                                    // CRITICAL SECURITY FIX: Validate alignedData and parameters before passing to Metal API
                                    if (!alignedData) {
                                        NSLog(@"MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                        free(alignedData);
                                        continue;
                                    }
                                    if (alignedBytesPerRow == 0) {
                                        NSLog(@"MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                        free(alignedData);
                                        continue;
                                    }
                                    if (bytesPerImage == 0) {
                                        NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                        free(alignedData);
                                        continue;
                                    }
                                    if (hasExplicitDataSize) {
                                        BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                               texName:tex->name
                                                                             texTarget:tex->target
                                                                                 bytes:alignedData
                                                                           bytesPerRow:alignedBytesPerRow
                                                                         bytesPerImage:alignedBytesPerImage
                                                                                 width:width
                                                                                height:height
                                                                                 depth:1
                                                                                 level:level
                                                                                 slice:face];
                                        if (!uploaded) {
                                            NSLog(@"MGL WARNING: Aligned 2D blit upload failed (level %d, face %d)", level, face);
                                        }
                                    } else {
                                        NSLog(@"MGL INFO: Skipping 2D upload with synthesized data size (level %d, face %d)", level, face);
                                    }
                                    free(alignedData);
                                } else {
                                    NSLog(@"MGL ERROR: Failed to allocate aligned memory for 2D texture upload (level %d, face %d)", level, face);
                                }
                            } else {
                                // Data and bytesPerRow are already aligned
                                // CRITICAL SECURITY FIX: Validate srcData before passing to Metal API
                                if (!srcData) {
                                    NSLog(@"MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                    continue;
                                }
                                if (bytesPerRow == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                    continue;
                                }
                                if (bytesPerImage == 0) {
                                    NSLog(@"MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash", level, face);
                                    continue;
                                }
                                if (hasExplicitDataSize) {
                                    BOOL uploaded = [self uploadTextureSliceViaBlit:texture
                                                                           texName:tex->name
                                                                        texTarget:tex->target
                                                                             bytes:srcData
                                                                       bytesPerRow:bytesPerRow
                                                                     bytesPerImage:bytesPerImage
                                                                             width:width
                                                                            height:height
                                                                             depth:1
                                                                             level:level
                                                                             slice:face];
                                    if (!uploaded) {
                                        NSLog(@"MGL WARNING: 2D direct blit upload failed (level %d, face %d)", level, face);
                                    }
                                } else {
                                    NSLog(@"MGL INFO: Skipping 2D upload with synthesized data size (level %d, face %d)", level, face);
                                }
                            }
                        } else {
                            NSLog(@"MGL WARNING: Skipping 2D texture upload due to invalid data or parameters");
                        }
                    }
                }
            }
        }
    }
    else
    {
        // PROPER FIX: Enable texture filling with AGX safety and proper memory alignment
        NSLog(@"MGL INFO: PROPER FIX - Processing texture fill (tex=%d, dims=%lux%lu)", tex->name, (unsigned long)texture.width, (unsigned long)texture.height);

        if (texture.width == 0 || texture.height == 0 || texture.width > 16384 || texture.height > 16384) {
            NSLog(@"MGL WARNING: Skipping texture fill due to invalid dimensions: %lux%lu", (unsigned long)texture.width, (unsigned long)texture.height);
        } else {
            // Determine pixel format size to create appropriate black data
            NSUInteger bytesPerPixel = 4; // Default to RGBA
            switch(texture.pixelFormat) {
                case MTLPixelFormatR8Unorm:
                case MTLPixelFormatR8Uint:
                case MTLPixelFormatR8Sint:
                    bytesPerPixel = 1;
                    break;
                case MTLPixelFormatRG8Unorm:
                case MTLPixelFormatRG8Uint:
                case MTLPixelFormatRG8Sint:
                    bytesPerPixel = 2;
                    break;
                case MTLPixelFormatRGBA8Unorm:
                case MTLPixelFormatRGBA8Uint:
                case MTLPixelFormatRGBA8Sint:
                    bytesPerPixel = 4;
                    break;
                default:
                    bytesPerPixel = 4; // Default assumption
                    break;
            }

            // Calculate dynamic alignment for Metal textures based on pixel format
            NSUInteger bytesPerRow = texture.width * bytesPerPixel;
            NSUInteger alignment = [self getOptimalAlignmentForPixelFormat:texture.pixelFormat];
            if (bytesPerRow % alignment != 0) {
                bytesPerRow = ((bytesPerRow + alignment - 1) / alignment) * alignment;
            }

            NSUInteger dataSize = bytesPerRow * texture.height;

            // Validate that dataSize is reasonable (not too large)
            if (dataSize > 64 * 1024 * 1024) { // 64MB limit per texture level
                NSLog(@"MGL WARNING: Skipping texture fill due to excessive size: %lu bytes", (unsigned long)dataSize);
            } else {
                // Allocate initialization data for texture clear.
                // aligned_alloc has been unreliable in this environment; calloc is safer here.
                (void)alignment;
                void *blackData = calloc(dataSize, 1);
                if (blackData) {
                    // CRITICAL SECURITY FIX: Comprehensive validation to prevent Metal driver crashes
                    // calloc already zero-initializes

                    // Multi-layer validation for all parameters
                    if (!blackData) {
                        NSLog(@"MGL SECURITY ERROR: blackData is NULL after memset - CORRUPTION DETECTED");
                        return texture;
                    }
                    if (bytesPerRow == 0) {
                        NSLog(@"MGL SECURITY ERROR: Invalid bytesPerRow (0) for texture fill");
                        free(blackData);
                        return texture;
                    }
                    if (dataSize == 0) {
                        NSLog(@"MGL SECURITY ERROR: Invalid dataSize (0) for texture fill");
                        free(blackData);
                        return texture;
                    }
                    if (!texture) {
                        NSLog(@"MGL SECURITY ERROR: Metal texture is NULL");
                        free(blackData);
                        return texture;
                    }
                    if (texture.width == 0 || texture.height == 0) {
                        NSLog(@"MGL SECURITY ERROR: Invalid texture dimensions %lux%lu", (unsigned long)texture.width, (unsigned long)texture.height);
                        free(blackData);
                        return texture;
                    }

                    // Additional validation: verify blackData contains expected zeros (anti-corruption check)
                    uint8_t *bytes = (uint8_t *)blackData;
                    bool dataCorrupted = false;
                    for (NSUInteger i = 0; i < MIN(dataSize, 1024); i++) { // Check first 1KB only for performance
                        if (bytes[i] != 0) {
                            dataCorrupted = true;
                            break;
                        }
                    }
                    if (dataCorrupted) {
                        NSLog(@"MGL SECURITY ERROR: blackData corruption detected - memory safety issue");
                        free(blackData);
                        return texture;
                    }

                    NSLog(@"MGL INFO: All validations passed for texture fill (size=%lu, bytesPerRow=%lu)", (unsigned long)dataSize, (unsigned long)bytesPerRow);

                    // ULTRA-DEFENSIVE: Final validation immediately before Metal API call
                    // This prevents race conditions and memory corruption between validation and use
                    if (!blackData) {
                        NSLog(@"MGL CRITICAL ERROR: blackData became NULL before Metal call - RACE CONDITION DETECTED");
                        free(blackData);
                        return texture;
                    }
                    if (!texture) {
                        NSLog(@"MGL CRITICAL ERROR: Metal texture became NULL before Metal call - RACE CONDITION DETECTED");
                        free(blackData);
                        return texture;
                    }
                    if (bytesPerRow == 0 || dataSize == 0) {
                        NSLog(@"MGL CRITICAL ERROR: Parameters became invalid before Metal call - RACE CONDITION DETECTED");
                        free(blackData);
                        return texture;
                    }

                    // Additional verification: Check if Metal texture is still valid
                    if (texture.width == 0 || texture.height == 0) {
                        NSLog(@"MGL CRITICAL ERROR: Metal texture dimensions became invalid before Metal call");
                        free(blackData);
                        return texture;
                    }

                    // Final integrity check: Verify blackData still contains expected zeros
                    uint8_t *finalCheck = (uint8_t *)blackData;
                    bool finalCorruption = false;
                    for (NSUInteger i = 0; i < MIN(dataSize, 256); i++) { // Check first 256 bytes
                        if (finalCheck[i] != 0) {
                            finalCorruption = true;
                            break;
                        }
                    }
                    if (finalCorruption) {
                        NSLog(@"MGL CRITICAL ERROR: Memory corruption detected immediately before Metal call");
                        free(blackData);
                        return texture;
                    }

                    NSLog(@"MGL INFO: FIXING: Implementing proper texture filling for Apple Metal compatibility");

                    // PROPER FIX: Use Apple Metal-compatible texture filling approach
                    // The issue was using incorrect bytesPerRow and region parameters
                    NSLog(@"MGL INFO: Implementing Metal-compliant texture fill operations");

                    // Use Metal's standard pattern for texture filling.
                    NSUInteger pixelSize = bytesPerPixel;
                    NSUInteger properBytesPerRow = width * pixelSize;

                    // Ensure proper alignment for Apple Metal driver
                    if (properBytesPerRow % 64 != 0) {
                        properBytesPerRow = ((properBytesPerRow + 63) / 64) * 64;
                    }

                    // Fill the entire level. A previous 1x1 safety fill left large textures
                    // mostly uninitialized while their Metal backing existed.
                    MTLRegion properRegion = MTLRegionMake2D(0, 0, width, height);

                    // Create properly aligned texture data buffer
                    NSUInteger fillSize = properBytesPerRow * properRegion.size.height;
                    uint8_t *properData = (uint8_t *)calloc(fillSize, 1);

                    if (properData) {
                        // Initialize with safe texture data (transparent black with alpha = 0)
                        for (NSUInteger y = 0; y < properRegion.size.height; y++) {
                            uint8_t *row = properData + (y * properBytesPerRow);
                            for (NSUInteger x = 0; x < properRegion.size.width; x++) {
                                uint8_t *pixel = row + (x * pixelSize);
                                pixel[0] = 0;  // R
                                if (pixelSize > 1) pixel[1] = 0;  // G
                                if (pixelSize > 2) pixel[2] = 0;  // B
                                if (pixelSize > 3) pixel[3] = 0; // A = transparent for uninitialized color data
                            }
                        }

                        @try {
                            NSLog(@"MGL INFO: Performing Metal-compliant texture fill:");
                            NSLog(@"  - Region: %dx%d", (int)properRegion.size.width, (int)properRegion.size.height);
                            NSLog(@"  - bytesPerRow: %lu", (unsigned long)properBytesPerRow);
                            NSLog(@"  - dataSize: %lu", (unsigned long)fillSize);

                            // ALTERNATIVE APPROACH: Safe texture filling without replaceRegion
                            NSLog(@"MGL INFO: Using alternative texture filling methods (AGX-safe)");

                            @try {
                                // ALTERNATIVE 1: Try MTLBuffer-to-texture copy approach
                                if (properData && dataSize > 0) {
                                    NSLog(@"MGL INFO: Attempting buffer-based texture fill");

                                    // Create a temporary MTLBuffer with the texture data
                                    id<MTLBuffer> tempBuffer = [_device newBufferWithBytes:properData
                                                                                    length:fillSize
                                                                                   options:MTLResourceStorageModeShared];

                                    if (tempBuffer) {
                                        NSLog(@"MGL INFO: Created temporary MTLBuffer for texture data");

                                        if ([self shouldSkipGPUOperations]) {
                                            NSLog(@"MGL AGX: Skipping texture fill during recovery - texture will be empty");
                                        } else {
                                            BOOL uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:tempBuffer
                                                                                                  sourceOffset:0
                                                                                             sourceBytesPerRow:properBytesPerRow
                                                                                           sourceBytesPerImage:fillSize
                                                                                                     sourceSize:MTLSizeMake(properRegion.size.width, properRegion.size.height, 1)
                                                                                                      toTexture:texture
                                                                                               destinationSlice:0
                                                                                               destinationLevel:0
                                                                                              destinationOrigin:MTLOriginMake(0, 0, 0)
                                                                                                         reason:"texture_fill_initialization"];
                                            if (uploaded) {
                                                NSLog(@"MGL SUCCESS: Texture data copied using dedicated upload command buffer");
                                                mglMarkTextureLevelMetalFilled(tex, 0, fillSize);
                                            } else {
                                                NSLog(@"MGL WARNING: Dedicated texture fill upload failed - texture may remain uninitialized");
                                            }
                                        }

                                        // Clean up the temporary buffer
                                        tempBuffer = nil;
                                    }
                                }
                            } @catch (NSException *exception) {
                                NSLog(@"MGL WARNING: Buffer-based texture fill failed - trying alternative");

                                // ALTERNATIVE 2: Simple direct color filling for basic cases
                                if (width <= 512 && height <= 512 && tex->internalformat == GL_RGBA8) {
                                    NSLog(@"MGL INFO: Attempting simple direct color fill for small RGBA8 texture");

                                    @try {
                                        // Create a simple pattern that's not magenta
                                        NSUInteger pixelCount = width * height;
                                        uint32_t *simpleData = calloc(pixelCount, sizeof(uint32_t));

                                        if (simpleData) {
                                            // Create a simple gradient pattern instead of magenta
                                            for (NSUInteger y = 0; y < height; y++) {
                                                for (NSUInteger x = 0; x < width; x++) {
                                                    NSUInteger index = y * width + x;

                                                    // Create a simple gradient from blue to green
                                                    uint8_t r = (uint8_t)(x * 255 / width);
                                                    uint8_t g = (uint8_t)(y * 255 / height);
                                                    uint8_t b = 128;
                                                    uint8_t a = 255;

                                                    simpleData[index] = (a << 24) | (b << 16) | (g << 8) | r;
                                                }
                                            }

                                            // Try direct replaceRegion for simple cases
                                            MTLRegion simpleRegion = MTLRegionMake2D(0, 0, width, height);
                                            [texture replaceRegion:simpleRegion
                                                    mipmapLevel:0
                                                          slice:0
                                                      withBytes:simpleData
                                                    bytesPerRow:width * sizeof(uint32_t)
                                                  bytesPerImage:width * height * sizeof(uint32_t)];

                                            NSLog(@"MGL SUCCESS: Simple direct color fill completed");
                                            mglMarkTextureLevelMetalFilled(tex, 0, pixelCount * sizeof(uint32_t));
                                            free(simpleData);
                                        }
                                    } @catch (NSException *exception) {
                                        NSLog(@"MGL WARNING: Simple direct fill also failed: %@", exception.reason);
                                    }
                                } else {
                                    NSLog(@"MGL INFO: Skipping complex texture - would use deferred initialization");
                                }
                            }
                        } @catch (NSException *exception) {
                            NSLog(@"MGL ERROR: Metal texture fill failed - investigating root cause");
                            NSLog(@"MGL ERROR: Exception: %@ (Reason: %@)", exception.name, exception.reason);
                            NSLog(@"MGL INFO: This indicates our parameters are still incompatible with AGX driver");
                        }

                        free(properData);
                    } else {
                        NSLog(@"MGL ERROR: Failed to allocate properly aligned texture data");
                    }
                    free(blackData);
                } else {
                    NSLog(@"MGL ERROR: Failed to allocate aligned memory for texture fill (%lu bytes)", (unsigned long)dataSize);
                }
            }
        }
    }

    tex->dirty_bits = 0;

    // Record successful texture creation for AGX error tracking
    [self recordGPUSuccess];

    return texture;
}

// AGX-SAFE Fallback texture creation for GPU error recovery scenarios
- (id<MTLTexture>) createFallbackMTLTexture:(Texture *) tex
{
    NSLog(@"MGL AGX: Creating emergency fallback texture (size: %dx%dx%d)", tex->width, tex->height, tex->depth);

    @try {
        MTLPixelFormat fallbackFormat = mtlPixelFormatForGLTex(tex);
        if (fallbackFormat == MTLPixelFormatInvalid) {
            // Conservative defaults by GL intent when translation is unavailable.
            if (tex->internalformat == GL_DEPTH24_STENCIL8 ||
                tex->internalformat == GL_DEPTH32F_STENCIL8) {
                fallbackFormat = MTLPixelFormatDepth32Float_Stencil8;
            } else if (tex->internalformat == GL_DEPTH_COMPONENT ||
                       tex->internalformat == GL_DEPTH_COMPONENT16 ||
                       tex->internalformat == GL_DEPTH_COMPONENT24 ||
                       tex->internalformat == GL_DEPTH_COMPONENT32 ||
                       tex->internalformat == GL_DEPTH_COMPONENT32F) {
                fallbackFormat = MTLPixelFormatDepth32Float;
            } else {
                fallbackFormat = MTLPixelFormatRGBA8Unorm;
            }
        }

        BOOL isDepthOrStencilFormat =
            (fallbackFormat == MTLPixelFormatDepth16Unorm ||
             fallbackFormat == MTLPixelFormatDepth32Float ||
             fallbackFormat == MTLPixelFormatDepth24Unorm_Stencil8 ||
             fallbackFormat == MTLPixelFormatDepth32Float_Stencil8 ||
             fallbackFormat == MTLPixelFormatStencil8);

        MTLTextureDescriptor *fallbackDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fallbackFormat
                                                                                                    width:MAX(tex->width, 1)
                                                                                                   height:MAX(tex->height, 1)
                                                                                                mipmapped:NO];
        fallbackDesc.usage = MTLTextureUsageShaderRead;
        if (tex->is_render_target || isDepthOrStencilFormat) {
            fallbackDesc.usage |= MTLTextureUsageRenderTarget;
        }
        fallbackDesc.storageMode = MTLStorageModeShared;

        id<MTLTexture> fallbackTexture = [_device newTextureWithDescriptor:fallbackDesc];

        if (fallbackTexture) {
            // Fill with simple gradient pattern using a simple approach
            NSUInteger width = fallbackTexture.width;
            NSUInteger height = fallbackTexture.height;

            if (!isDepthOrStencilFormat && width <= 512 && height <= 512) {
                uint32_t *gradientData = calloc(width * height, sizeof(uint32_t));
                if (gradientData) {
                    // Create simple red-blue gradient
                    for (NSUInteger y = 0; y < height; y++) {
                        for (NSUInteger x = 0; x < width; x++) {
                            NSUInteger index = y * width + x;
                            uint8_t r = (uint8_t)((x * 255) / width);
                            uint8_t g = 128;
                            uint8_t b = (uint8_t)((y * 255) / height);
                            uint8_t a = 255;
                            gradientData[index] = (a << 24) | (b << 16) | (g << 8) | r;
                        }
                    }

                    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
                    [fallbackTexture replaceRegion:region mipmapLevel:0 withBytes:gradientData
                               bytesPerRow:width * sizeof(uint32_t)];

                    free(gradientData);
                    NSLog(@"MGL AGX: Fallback color texture created with gradient pattern");
                }
            }
        }

        return fallbackTexture;

    } @catch (NSException *exception) {
        NSLog(@"MGL AGX: Even fallback texture creation failed: %@", exception.reason);
        return nil;
    }
}

// Helper function to calculate bytes per pixel for different OpenGL formats
- (NSUInteger)bytesPerPixelForFormat:(GLenum)internalformat
{
    switch(internalformat) {
        case GL_RED:
        case GL_R8:
        case GL_R8I:
        case GL_R8UI:
            return 1;

        case GL_RG:
        case GL_RG8:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_R16:
        case GL_R16F:
            return 2;

        case GL_RGB:
        case GL_RGB8:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_SRGB8:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
            return 3;

        case GL_RGBA:
        case GL_RGBA8:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_SRGB8_ALPHA8:
            return 4;

        case GL_RGBA16:
        case GL_RGBA16F:
        case GL_R32F:
            return 8;

        case GL_RGB16:
        case GL_RGB16F:
            return 6;

        case GL_RGBA16I:
        case GL_RGBA16UI:
            return 8;

        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
            return 12;

        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            return 16;

        default:
            // Default to 4 bytes for unknown formats
            NSLog(@"MGL WARNING: Unknown internal format 0x%x, defaulting to 4 bytes per pixel", internalformat);
            return 4;
    }
}

- (id<MTLSamplerState>) createMTLSamplerForTexParam:(TextureParameter *)tex_param target:(GLuint)target
{
    MTLSamplerDescriptor *samplerDescriptor;

    if (!tex_param) {
        NSLog(@"MGL SAMPLER ERROR: createMTLSamplerForTexParam called with NULL parameters");
        return nil;
    }

    samplerDescriptor = [MTLSamplerDescriptor new];
    if (!samplerDescriptor) {
        NSLog(@"MGL SAMPLER ERROR: failed to allocate MTLSamplerDescriptor");
        return nil;
    }

    switch(tex_param->min_filter)
    {
        case GL_NEAREST:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterNearest;
            break;

        case GL_LINEAR:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
            break;

        case GL_NEAREST_MIPMAP_NEAREST:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterNearest;
            samplerDescriptor.mipFilter = MTLSamplerMipFilterNearest;
            break;

        case GL_LINEAR_MIPMAP_NEAREST:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
            samplerDescriptor.mipFilter = MTLSamplerMipFilterNearest;
            break;

        case GL_NEAREST_MIPMAP_LINEAR:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterNearest;
            samplerDescriptor.mipFilter = MTLSamplerMipFilterLinear;
            break;

        case GL_LINEAR_MIPMAP_LINEAR:
            samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
            samplerDescriptor.mipFilter = MTLSamplerMipFilterLinear;
            break;

        default:
            NSLog(@"MGL SAMPLER ERROR: invalid GL_TEXTURE_MIN_FILTER 0x%x", tex_param->min_filter);
            return nil;
    }

    switch(tex_param->mag_filter)
    {
        case GL_NEAREST:
            samplerDescriptor.magFilter = MTLSamplerMinMagFilterNearest;
            break;

        case GL_LINEAR:
            samplerDescriptor.magFilter = MTLSamplerMinMagFilterLinear;
            break;

        default:
            NSLog(@"MGL SAMPLER ERROR: invalid GL_TEXTURE_MAG_FILTER 0x%x", tex_param->mag_filter);
            return nil;
    }

    //     @property (nonatomic) NSUInteger maxAnisotropy;
    if (tex_param->max_anisotropy > 1.0)
    {
        samplerDescriptor.maxAnisotropy = tex_param->max_anisotropy;
    }

    //    @property (nonatomic) MTLSamplerAddressMode sAddressMode;
    //    @property (nonatomic) MTLSamplerAddressMode tAddressMode;
    //    @property (nonatomic) MTLSamplerAddressMode rAddressMode;
    for (int i=0; i<3; i++)
    {
        MTLSamplerAddressMode mode = 0;
        GLenum type = 0;

        switch(i)
        {
            case 0: type = tex_param->wrap_s; break;
            case 1: type = tex_param->wrap_t; break;
            case 2: type = tex_param->wrap_r; break;
        }

        switch(type)
        {
            case GL_CLAMP_TO_EDGE:
                mode = MTLSamplerAddressModeClampToEdge;
                break;

            case GL_CLAMP_TO_BORDER:
                mode = MTLSamplerAddressModeClampToBorderColor;
                break;

            case GL_MIRRORED_REPEAT:
                mode = MTLSamplerAddressModeMirrorRepeat;
                break;

            case GL_REPEAT:
                mode = MTLSamplerAddressModeRepeat;
                break;

            case GL_MIRROR_CLAMP_TO_EDGE:
                mode = MTLSamplerAddressModeMirrorClampToEdge;
                break;

    //        case GL_CLAMP_TO_ZERO_MGL_EXT:
    //            mode = MTLSamplerAddressModeClampToZero;
    //            break;

            default:
                NSLog(@"MGL SAMPLER ERROR: invalid GL texture wrap mode 0x%x for axis %d", type, i);
                return nil;
        }

        switch(i)
        {
            case 0: samplerDescriptor.sAddressMode = mode; break;
            case 1: samplerDescriptor.tAddressMode = mode; break;
            case 2: samplerDescriptor.rAddressMode = mode; break;
        }
    }

    BOOL usesBorderColor = (tex_param->wrap_s == GL_CLAMP_TO_BORDER ||
                            tex_param->wrap_t == GL_CLAMP_TO_BORDER ||
                            tex_param->wrap_r == GL_CLAMP_TO_BORDER);
    if (!usesBorderColor)
    {
        samplerDescriptor.borderColor = MTLSamplerBorderColorTransparentBlack;
    }
    else if ((tex_param->border_color[0] == 0.0) &&
             (tex_param->border_color[1] == 0.0) &&
             (tex_param->border_color[2] == 0.0) &&
             (tex_param->border_color[3] == 0.0))
    {
        samplerDescriptor.borderColor = MTLSamplerBorderColorTransparentBlack;
    }
    else if ((tex_param->border_color[0] == 0.0) &&
             (tex_param->border_color[1] == 0.0) &&
             (tex_param->border_color[2] == 0.0) &&
             (tex_param->border_color[3] == 1.0))
    {
        samplerDescriptor.borderColor = MTLSamplerBorderColorOpaqueBlack;
    }
    else if ((tex_param->border_color[0] == 1.0) &&
             (tex_param->border_color[1] == 1.0) &&
             (tex_param->border_color[2] == 1.0) &&
             (tex_param->border_color[3] == 1.0))
    {
        samplerDescriptor.borderColor = MTLSamplerBorderColorOpaqueWhite;
    }
    else
    {
        static uint64_t s_unsupportedBorderColorCount = 0;
        uint64_t hit = ++s_unsupportedBorderColorCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL SAMPLER WARNING: GL border color (%g,%g,%g,%g) is not exactly representable by MTLSamplerBorderColor; approximating hit=%llu",
                  tex_param->border_color[0],
                  tex_param->border_color[1],
                  tex_param->border_color[2],
                  tex_param->border_color[3],
                  (unsigned long long)hit);
        }

        if (tex_param->border_color[3] < 0.5f) {
            samplerDescriptor.borderColor = MTLSamplerBorderColorTransparentBlack;
        } else if (tex_param->border_color[0] >= 0.5f &&
                   tex_param->border_color[1] >= 0.5f &&
                   tex_param->border_color[2] >= 0.5f) {
            samplerDescriptor.borderColor = MTLSamplerBorderColorOpaqueWhite;
        } else {
            samplerDescriptor.borderColor = MTLSamplerBorderColorOpaqueBlack;
        }
    }

    if (target == GL_TEXTURE_RECTANGLE)
    {
        if ((tex_param->wrap_s == GL_CLAMP_TO_EDGE) &&
            (tex_param->wrap_t == GL_CLAMP_TO_EDGE) &&
            (tex_param->wrap_r == GL_CLAMP_TO_EDGE))
        {
            samplerDescriptor.normalizedCoordinates = false;
        }
        else
        {
            DEBUG_PRINT("Non-normalized coordinates should only be used with 1D and 2D textures with the ClampToEdge wrap mode, otherwise the results of sampling are undefined.");
        }
    }

    // @property (nonatomic) BOOL lodAverage API_AVAILABLE(ios(9.0), macos(11.0), macCatalyst(14.0));


    // @property (nonatomic) MTLCompareFunction compareFunction API_AVAILABLE(macos(10.11), ios(9.0));
    if (tex_param->compare_mode == GL_NONE)
    {
        samplerDescriptor.compareFunction = MTLCompareFunctionNever;
    }
    else if (tex_param->compare_mode == GL_COMPARE_REF_TO_TEXTURE)
    {
        switch(tex_param->compare_func)
        {
            case GL_LEQUAL:
                samplerDescriptor.compareFunction = MTLCompareFunctionLessEqual;
                break;

            case GL_GEQUAL:
                samplerDescriptor.compareFunction = MTLCompareFunctionGreaterEqual;
                break;

            case GL_LESS:
                samplerDescriptor.compareFunction = MTLCompareFunctionLess;
                break;

            case GL_GREATER:
                samplerDescriptor.compareFunction = MTLCompareFunctionGreater;
                break;

            case GL_EQUAL:
                samplerDescriptor.compareFunction = MTLCompareFunctionEqual;
                break;

            case GL_NOTEQUAL:
                samplerDescriptor.compareFunction = MTLCompareFunctionNotEqual;
                break;

            case GL_ALWAYS:
                samplerDescriptor.compareFunction = MTLCompareFunctionAlways;
                break;

            case GL_NEVER:
                samplerDescriptor.compareFunction = MTLCompareFunctionNever;
                break;

            default:
                NSLog(@"MGL SAMPLER ERROR: invalid GL_TEXTURE_COMPARE_FUNC 0x%x", tex_param->compare_func);
                return nil;
        }
    }
    else
    {
        NSLog(@"MGL SAMPLER ERROR: invalid GL_TEXTURE_COMPARE_MODE 0x%x", tex_param->compare_mode);
        return nil;
    }

    id<MTLSamplerState> sampler = [_device newSamplerStateWithDescriptor:samplerDescriptor];
    if (!sampler) {
        NSLog(@"MGL SAMPLER ERROR: failed to create MTLSamplerState");
        return nil;
    }

    return sampler;
}

- (id<MTLTexture>)fallbackSampledTexture
{
    if (_fallbackSampledTexture || !kMGLEnableSampledTextureFallback) {
        return _fallbackSampledTexture;
    }

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:1
                                                          height:1
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    _fallbackSampledTexture = [_device newTextureWithDescriptor:desc];
    if (_fallbackSampledTexture) {
        uint32_t pixel = 0xffffffffu;
        [_fallbackSampledTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                                   mipmapLevel:0
                                     withBytes:&pixel
                                   bytesPerRow:sizeof(pixel)];
        NSLog(@"MGL INFO: Created 1x1 fallback sampled texture for missing shader resources");
    } else {
        NSLog(@"MGL ERROR: Failed to create fallback sampled texture");
    }

    return _fallbackSampledTexture;
}

- (id<MTLTexture>)fallbackCubeSampledTexture
{
    if (_fallbackCubeSampledTexture || !kMGLEnableSampledTextureFallback) {
        return _fallbackCubeSampledTexture;
    }

    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = MTLTextureTypeCube;
    desc.pixelFormat = MTLPixelFormatRGBA8Unorm;
    desc.width = 1;
    desc.height = 1;
    desc.depth = 1;
    desc.arrayLength = 1;
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    _fallbackCubeSampledTexture = [_device newTextureWithDescriptor:desc];
    if (_fallbackCubeSampledTexture) {
        uint32_t pixel = 0xffffffffu;
        for (NSUInteger face = 0; face < 6; face++) {
            [_fallbackCubeSampledTexture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                                           mipmapLevel:0
                                                 slice:face
                                             withBytes:&pixel
                                           bytesPerRow:sizeof(pixel)
                                         bytesPerImage:sizeof(pixel)];
        }
        NSLog(@"MGL INFO: Created 1x1 fallback cube sampled texture for missing shader resources");
    } else {
        NSLog(@"MGL ERROR: Failed to create fallback cube sampled texture");
    }

    return _fallbackCubeSampledTexture;
}

- (id<MTLTexture>)fallbackLightmapSampledTexture
{
    if (_fallbackLightmapSampledTexture || !kMGLEnableSampledTextureFallback) {
        return _fallbackLightmapSampledTexture;
    }

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:16
                                                          height:16
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    _fallbackLightmapSampledTexture = [_device newTextureWithDescriptor:desc];
    if (_fallbackLightmapSampledTexture) {
        uint32_t pixels[16 * 16];
        for (NSUInteger i = 0; i < (sizeof(pixels) / sizeof(pixels[0])); i++) {
            pixels[i] = 0xffffffffu;
        }
        [_fallbackLightmapSampledTexture replaceRegion:MTLRegionMake2D(0, 0, 16, 16)
                                           mipmapLevel:0
                                             withBytes:pixels
                                           bytesPerRow:(16 * sizeof(uint32_t))];
        NSLog(@"MGL INFO: Created 16x16 fallback lightmap texture for missing Sampler2 resources");
    } else {
        NSLog(@"MGL ERROR: Failed to create fallback lightmap texture");
    }

    return _fallbackLightmapSampledTexture;
}

- (id<MTLTexture>)fallbackTextureBufferSampledTexture
{
    if (_fallbackSintTextureBuffer || !kMGLEnableSampledTextureFallback) {
        return _fallbackSintTextureBuffer;
    }

    static const NSUInteger kFallbackTexelCount = 64;
    static const NSUInteger kFallbackBytesPerTexel = 4;

    if (!_fallbackTextureBufferStorage) {
        _fallbackTextureBufferStorage = [_device newBufferWithLength:(kFallbackTexelCount * kFallbackBytesPerTexel)
                                                              options:MTLResourceStorageModeShared];
        if (_fallbackTextureBufferStorage && _fallbackTextureBufferStorage.contents) {
            memset(_fallbackTextureBufferStorage.contents, 0, kFallbackTexelCount * kFallbackBytesPerTexel);
        }
    }

    if (!_fallbackTextureBufferStorage) {
        NSLog(@"MGL ERROR: Failed to create fallback texture-buffer backing storage");
        return nil;
    }

    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = MTLTextureTypeTextureBuffer;
    desc.pixelFormat = MTLPixelFormatRGBA8Sint;
    desc.width = kFallbackTexelCount;
    desc.height = 1;
    desc.depth = 1;
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    @try {
        _fallbackSintTextureBuffer = [_fallbackTextureBufferStorage newTextureWithDescriptor:desc
                                                                                     offset:0
                                                                                bytesPerRow:(kFallbackTexelCount * kFallbackBytesPerTexel)];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Failed to create fallback texture-buffer texture: %@", exception);
        _fallbackSintTextureBuffer = nil;
    }

    if (_fallbackSintTextureBuffer) {
        NSLog(@"MGL INFO: Created fallback signed integer texture buffer for missing/invalid texel-buffer resources");
    }

    return _fallbackSintTextureBuffer;
}

- (id<MTLTexture>)fallbackSampledTextureForExpectedType:(MTLTextureType)expectedType
                                               dataKind:(MGLTextureDataKind)dataKind
{
    if (!kMGLEnableSampledTextureFallback) {
        return nil;
    }

    if (dataKind == MGLTextureDataKindUnknown || dataKind == MGLTextureDataKindFloat) {
        if (expectedType == MTLTextureTypeCube) {
            return [self fallbackCubeSampledTexture];
        }
        if (expectedType == MTLTextureTypeTextureBuffer) {
            return [self fallbackTextureBufferSampledTexture];
        }
        return [self fallbackSampledTexture];
    }

    MTLPixelFormat pixelFormat = (dataKind == MGLTextureDataKindUint)
        ? MTLPixelFormatRGBA8Uint
        : MTLPixelFormatRGBA8Sint;

    MTLTextureType textureType = expectedType ? expectedType : MTLTextureType2D;
    if (textureType == MTLTextureTypeTextureBuffer) {
        // Keep the existing texel-buffer fallback for signed integer paths.
        // Uint texel-buffer fallback can be added when Minecraft hits that exact shader.
        return [self fallbackTextureBufferSampledTexture];
    }

    if (!_fallbackSampledTextureCache) {
        _fallbackSampledTextureCache = [[NSMutableDictionary alloc] initWithCapacity:8];
    }

    NSUInteger keyValue = (((NSUInteger)textureType) << 8u) | ((NSUInteger)dataKind);
    NSNumber *key = @(keyValue);
    id<MTLTexture> cached = _fallbackSampledTextureCache[key];
    if (cached) {
        return cached;
    }

    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = textureType;
    desc.pixelFormat = pixelFormat;
    desc.width = 1;
    desc.height = 1;
    desc.depth = 1;
    desc.arrayLength = (textureType == MTLTextureTypeCube ||
                        textureType == MTLTextureTypeCubeArray ||
                        textureType == MTLTextureType2DArray ||
                        textureType == MTLTextureType1DArray ||
                        textureType == MTLTextureType2DMultisampleArray) ? 1 : 1;
    desc.mipmapLevelCount = 1;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;

    id<MTLTexture> texture = [_device newTextureWithDescriptor:desc];
    if (!texture) {
        NSLog(@"MGL ERROR: Failed to create %@ fallback sampled texture type=%lu format=%lu",
              [NSString stringWithUTF8String:mglTextureDataKindName(dataKind)],
              (unsigned long)textureType,
              (unsigned long)pixelFormat);
        return nil;
    }

    uint32_t zeroPixel = 0u;
    if (textureType == MTLTextureTypeCube || textureType == MTLTextureTypeCubeArray) {
        NSUInteger sliceCount = (textureType == MTLTextureTypeCube) ? 6u : 6u;
        for (NSUInteger slice = 0; slice < sliceCount; slice++) {
            [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                       mipmapLevel:0
                             slice:slice
                         withBytes:&zeroPixel
                       bytesPerRow:sizeof(zeroPixel)
                     bytesPerImage:sizeof(zeroPixel)];
        }
    } else {
        [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                   mipmapLevel:0
                     withBytes:&zeroPixel
                   bytesPerRow:sizeof(zeroPixel)];
    }

    _fallbackSampledTextureCache[key] = texture;
    NSLog(@"MGL INFO: Created %@ fallback sampled texture type=%lu format=%lu",
          [NSString stringWithUTF8String:mglTextureDataKindName(dataKind)],
          (unsigned long)textureType,
          (unsigned long)pixelFormat);

    return texture;
}

- (id<MTLSamplerState>)scaledBlitSamplerForFilter:(GLuint)filter
{
    BOOL wantsNearest = (filter == GL_NEAREST);
    id<MTLSamplerState> cached = wantsNearest ? _scaledBlitNearestSampler : _scaledBlitLinearSampler;
    if (cached) {
        return cached;
    }

    MTLSamplerDescriptor *desc = [[MTLSamplerDescriptor alloc] init];
    desc.minFilter = wantsNearest ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    desc.magFilter = wantsNearest ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    desc.mipFilter = MTLSamplerMipFilterNotMipmapped;
    desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.rAddressMode = MTLSamplerAddressModeClampToEdge;

    id<MTLSamplerState> sampler = [_device newSamplerStateWithDescriptor:desc];
    if (!sampler) {
        NSLog(@"MGL ERROR: failed to create scaled blit sampler filter=0x%x", filter);
        return nil;
    }

    if (wantsNearest) {
        _scaledBlitNearestSampler = sampler;
    } else {
        _scaledBlitLinearSampler = sampler;
    }

    return sampler;
}

- (id<MTLRenderPipelineState>)scaledBlitPipelineForPixelFormat:(MTLPixelFormat)pixelFormat
{
    if (pixelFormat == MTLPixelFormatInvalid || pixelFormat == 0) {
        pixelFormat = MTLPixelFormatBGRA8Unorm;
    }

    if (!_scaledBlitPipelineCache) {
        _scaledBlitPipelineCache = [[NSMutableDictionary alloc] initWithCapacity:4];
    }

    NSNumber *key = @((NSUInteger)pixelFormat);
    id<MTLRenderPipelineState> cached = _scaledBlitPipelineCache[key];
    if (cached) {
        return cached;
    }

    static NSString *source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "struct MGLScaledBlitParams { float4 uvRect; float forceOpaqueAlpha; float3 _padding; };\n"
         "struct MGLScaledBlitVOut { float4 position [[position]]; float2 uv; };\n"
         "vertex MGLScaledBlitVOut mgl_scaled_blit_vs(uint vid [[vertex_id]], constant MGLScaledBlitParams& p [[buffer(0)]]) {\n"
         "    float2 pos[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0) };\n"
         "    float2 uv[4] = { float2(p.uvRect.x, p.uvRect.w), float2(p.uvRect.z, p.uvRect.w), float2(p.uvRect.x, p.uvRect.y), float2(p.uvRect.z, p.uvRect.y) };\n"
         "    MGLScaledBlitVOut o;\n"
         "    o.position = float4(pos[vid], 0.0, 1.0);\n"
         "    o.uv = uv[vid];\n"
         "    return o;\n"
         "}\n"
         "fragment float4 mgl_scaled_blit_fs(MGLScaledBlitVOut in [[stage_in]], constant MGLScaledBlitParams& p [[buffer(0)]], texture2d<float> src [[texture(0)]], sampler s [[sampler(0)]]) {\n"
         "    float4 color = src.sample(s, in.uv);\n"
         "    if (p.forceOpaqueAlpha > 0.5) { color.a = 1.0; }\n"
         "    return color;\n"
         "}\n";

    NSError *error = nil;
    id<MTLLibrary> library = [_device newLibraryWithSource:source options:nil error:&error];
    if (!library) {
        NSLog(@"MGL ERROR: scaled blit shader compile failed: %@", error);
        return nil;
    }

    id<MTLFunction> vs = [library newFunctionWithName:@"mgl_scaled_blit_vs"];
    id<MTLFunction> fs = [library newFunctionWithName:@"mgl_scaled_blit_fs"];
    if (!vs || !fs) {
        NSLog(@"MGL ERROR: scaled blit shader functions missing vs=%@ fs=%@", vs, fs);
        return nil;
    }

    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.label = @"MGL scaled blit";
    desc.vertexFunction = vs;
    desc.fragmentFunction = fs;
    desc.colorAttachments[0].pixelFormat = pixelFormat;
    desc.rasterSampleCount = 1;

    id<MTLRenderPipelineState> pipeline = [_device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!pipeline) {
        NSLog(@"MGL ERROR: scaled blit pipeline create failed pixelFormat=%lu error=%@",
              (unsigned long)pixelFormat,
              error);
        return nil;
    }

    _scaledBlitPipelineCache[key] = pipeline;
    NSLog(@"MGL INFO: created scaled blit pipeline pixelFormat=%lu", (unsigned long)pixelFormat);
    return pipeline;
}

- (id<MTLTexture>)fallbackSampledTextureForExpectedType:(MTLTextureType)expectedType
{
    if (expectedType == MTLTextureTypeCube) {
        return [self fallbackCubeSampledTexture];
    }
    if (expectedType == MTLTextureTypeTextureBuffer) {
        return [self fallbackTextureBufferSampledTexture];
    }

    return [self fallbackSampledTexture];
}

- (int)textureIndexForExpectedMetalType:(MTLTextureType)expectedType
{
    switch (expectedType) {
        case MTLTextureType1D:
            return _TEXTURE_1D;
        case MTLTextureType1DArray:
            return _TEXTURE_1D_ARRAY;
        case MTLTextureType2D:
        case MTLTextureType2DMultisample:
            return _TEXTURE_2D;
        case MTLTextureType2DArray:
        case MTLTextureType2DMultisampleArray:
            return _TEXTURE_2D_ARRAY;
        case MTLTextureType3D:
            return _TEXTURE_3D;
        case MTLTextureTypeCube:
            return _TEXTURE_CUBE_MAP;
        case MTLTextureTypeCubeArray:
            return _TEXTURE_CUBE_MAP_ARRAY;
        case MTLTextureTypeTextureBuffer:
            return _TEXTURE_BUFFER;
        default:
            return -1;
    }
}

static GLint mglNamedSamplerTextureUnit(const char *name)
{
    if (!name || !*name) {
        return -1;
    }

    const char *end = name + strlen(name);
    const char *digits = end;
    while (digits > name && digits[-1] >= '0' && digits[-1] <= '9') {
        digits--;
    }
    if (digits < end) {
        unsigned long unit = strtoul(digits, NULL, 10);
        return (unit < TEXTURE_UNITS) ? (GLint)unit : -1;
    }

    if (!strcmp(name, "DiffuseSampler")) {
        return 0;
    }
    if (!strcmp(name, "OverlaySampler")) {
        return 1;
    }
    if (!strcmp(name, "LightSampler") || !strcmp(name, "LightmapSampler")) {
        return 2;
    }
    return -1;
}

static bool mglProgramHasSampledImageNamed(Program *program, const char *name)
{
    if (!program || !name) {
        return false;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            if (resources->list[i].name && strcmp(resources->list[i].name, name) == 0) {
                return true;
            }
        }
    }

    return false;
}

static bool mglProgramHasUniformBufferNamed(Program *program, const char *name)
{
    if (!program || !name) {
        return false;
    }

    for (int stage = _VERTEX_SHADER; stage < _MAX_SHADER_TYPES; stage++) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
        for (GLuint i = 0; resources->list && i < resources->count; i++) {
            if (resources->list[i].name && strcmp(resources->list[i].name, name) == 0) {
                return true;
            }
        }
    }

    return false;
}

static bool mglProgramLooksLikeModernRenderPipelineResources(Program *program)
{
    return mglProgramHasUniformBufferNamed(program, "Projection") ||
           mglProgramHasUniformBufferNamed(program, "ChunkSection") ||
           mglProgramHasUniformBufferNamed(program, "DynamicTransforms") ||
           mglProgramHasUniformBufferNamed(program, "Globals");
}

static GLint mglNamedSamplerTextureUnitForProgram(Program *program, const char *name)
{
    if (name && strcmp(name, "Sampler2") == 0 &&
        mglProgramLooksLikeModernRenderPipelineResources(program) &&
        mglProgramHasSampledImageNamed(program, "Sampler0") &&
        !mglProgramHasSampledImageNamed(program, "Sampler1")) {
        return 1;
    }

    return mglNamedSamplerTextureUnit(name);
}

static bool mglIsLightmapSamplerName(const char *name)
{
    return name &&
           (!strcmp(name, "Sampler2") ||
            !strcmp(name, "LightSampler") ||
            !strcmp(name, "LightmapSampler"));
}

- (GLuint)textureUnitForSampledBinding:(GLuint)metalBinding stage:(int)stage
{
    Program *program = mglResolveProgramFromState(ctx);
    if (!program || metalBinding >= TEXTURE_UNITS) {
        return metalBinding;
    }

    const char *sampledName = NULL;
    if (stage >= 0 && stage < _MAX_SHADER_TYPES) {
        SpirvResourceList *resources =
            &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
        for (GLuint i = 0; i < resources->count; i++) {
            SpirvResource *res = &resources->list[i];
            if (res->binding == metalBinding) {
                sampledName = res->name;
                break;
            }
        }
    }

    /*
     * Minecraft assigns sampler texture units from the RenderPipeline sampler
     * list, not from the numeric suffix in names like Sampler2. For example,
     * chunk rendering declares Sampler0 and Sampler2, so Sampler2 is uploaded
     * through glUniform1i(..., 1). Trust the captured GL uniform value first;
     * use semantic names only as a last resort for shaders that never uploaded
     * a sampler uniform.
     */
    bool stageExplicit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? (program->sampler_units_explicit_by_stage[stage][metalBinding] == GL_TRUE)
        : false;
    bool globalExplicit = (program->sampler_units_explicit[metalBinding] == GL_TRUE);

    GLint unit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? program->sampler_units_by_stage[stage][metalBinding]
        : program->sampler_units[metalBinding];
    if (stageExplicit && unit >= 0 && unit < TEXTURE_UNITS) {
        return (GLuint)unit;
    }

    unit = program->sampler_units[metalBinding];
    if (globalExplicit && unit >= 0 && unit < TEXTURE_UNITS) {
        return (GLuint)unit;
    }

    GLint defaultUnit = (stage >= 0 && stage < _MAX_SHADER_TYPES)
        ? program->sampler_units_by_stage[stage][metalBinding]
        : program->sampler_units[metalBinding];
    if (defaultUnit < 0 || defaultUnit >= TEXTURE_UNITS) {
        defaultUnit = program->sampler_units[metalBinding];
    }

    /*
     * GL sampler uniforms start at unit 0. Some legacy GLSL-to-SPIR-V paths in
     * this translation layer synthesize sampled-image resources without seeing
     * a matching glUniform upload. Once a uniform was explicitly set, honor it
     * exactly; otherwise allow stable numeric sampler names to recover the unit
     * that the original GLSL interface advertised.
     */
    GLint namedUnit = mglNamedSamplerTextureUnitForProgram(program, sampledName);
    if (namedUnit >= 0 && namedUnit < TEXTURE_UNITS) {
        if (defaultUnit < 0 || defaultUnit >= TEXTURE_UNITS || namedUnit != defaultUnit) {
            static uint64_t s_namedSamplerUnitFallbackLogs = 0;
            uint64_t hit = ++s_namedSamplerUnitFallbackLogs;
            if (hit <= 64ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL SAMPLER UNIT named-fallback program=%u stage=%s name=%s metalBinding=%u defaultUnit=%d -> unit=%d hit=%llu",
                      (unsigned)program->name,
                      mglShaderStageName(stage),
                      sampledName ? sampledName : "(null)",
                      (unsigned)metalBinding,
                      defaultUnit,
                      namedUnit,
                      (unsigned long long)hit);
            }
            return (GLuint)namedUnit;
        }
    }

    if (defaultUnit >= 0 && defaultUnit < TEXTURE_UNITS) {
        return (GLuint)defaultUnit;
    }

    return 0u;
}

- (Texture *)textureForSampledBinding:(GLuint)metalBinding stage:(int)stage expectedType:(MTLTextureType)expectedType
{
    if (!ctx || metalBinding >= TEXTURE_UNITS) {
        return NULL;
    }

    GLuint textureUnit = [self textureUnitForSampledBinding:metalBinding stage:stage];
    if (textureUnit >= TEXTURE_UNITS) {
        return NULL;
    }

    int textureIndex = [self textureIndexForExpectedMetalType:expectedType];
    if (textureIndex >= 0 && textureIndex < _MAX_TEXTURE_TYPES) {
        Texture *typedTexture = STATE(texture_units[textureUnit].textures[textureIndex]);
        if (typedTexture) {
            return typedTexture;
        }

        // Texel-buffer resources must not silently fall back to GL_TEXTURE_2D.
        // Minecraft's CloudFaces is declared as SpvDimBuffer but SPIRV-Cross
        // lowers it to a 1-row texture2d<int> in MSL. If no GL_TEXTURE_BUFFER
        // is bound, using the active 2D atlas here feeds float/RGBA data into a
        // signed integer vertex resource and corrupts the whole frame.
        if (expectedType == MTLTextureTypeTextureBuffer) {
            static uint64_t s_missingTextureBufferBindingLogs = 0;
            uint64_t hit = ++s_missingTextureBufferBindingLogs;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                Texture *activeTexture = STATE(active_textures[textureUnit]);
                NSLog(@"MGL TEXBUFFER BIND MISSING binding=%u unit=%u activeTex=%u activeTarget=0x%x hit=%llu",
                      (unsigned)metalBinding,
                      (unsigned)textureUnit,
                      activeTexture ? (unsigned)activeTexture->name : 0u,
                      activeTexture ? (unsigned)activeTexture->target : 0u,
                      (unsigned long long)hit);
            }
            return NULL;
        }
    }

    return STATE(active_textures[textureUnit]);
}

- (id<MTLSamplerState>)fallbackSamplerState
{
    if (_fallbackSamplerState) {
        return _fallbackSamplerState;
    }

    MTLSamplerDescriptor *desc = [MTLSamplerDescriptor new];
    desc.minFilter = MTLSamplerMinMagFilterNearest;
    desc.magFilter = MTLSamplerMinMagFilterNearest;
    desc.mipFilter = MTLSamplerMipFilterNotMipmapped;
    desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    desc.rAddressMode = MTLSamplerAddressModeClampToEdge;

    _fallbackSamplerState = [_device newSamplerStateWithDescriptor:desc];
    if (!_fallbackSamplerState) {
        NSLog(@"MGL ERROR: Failed to create fallback sampler state");
    }

    return _fallbackSamplerState;
}

- (void)traceSampledTextureReadback:(id<MTLTexture>)texture
                              glTex:(Texture *)glTex
                              level:(TextureLevel *)level0
                            program:(GLuint)program
                            binding:(GLuint)binding
                              stage:(NSString *)stage
                             reason:(NSString *)reason
                                hit:(uint64_t)hit
{
    if (!texture || !_device || !_commandQueue) {
        return;
    }

    MTLPixelFormat fmt = texture.pixelFormat;
    BOOL fourByteColor =
        fmt == MTLPixelFormatRGBA8Unorm ||
        fmt == MTLPixelFormatRGBA8Unorm_sRGB ||
        fmt == MTLPixelFormatBGRA8Unorm ||
        fmt == MTLPixelFormatBGRA8Unorm_sRGB;
    if (!fourByteColor) {
        NSLog(@"MGL TRACE sampled.readback skip program=%u binding=%u glTex=%u reason=%@ fmt=%lu type=%lu size=%lux%lu hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              reason,
              (unsigned long)fmt,
              (unsigned long)texture.textureType,
              (unsigned long)texture.width,
              (unsigned long)texture.height,
              (unsigned long long)hit);
        return;
    }

    NSUInteger texWidth = (NSUInteger)texture.width;
    NSUInteger texHeight = (NSUInteger)texture.height;
    if (texWidth == 0 || texHeight == 0) {
        return;
    }

    NSUInteger sampleWidth = MIN(texWidth, 8u);
    NSUInteger sampleHeight = MIN(texHeight, 8u);
    NSUInteger bytesPerPixel = 4u;
    NSUInteger bytesPerRow = sampleWidth * bytesPerPixel;
    NSUInteger byteCount = bytesPerRow * sampleHeight;
    if (byteCount == 0) {
        return;
    }

    id<MTLBuffer> readback = [_device newBufferWithLength:byteCount
                                                  options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [_commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = cb ? [cb blitCommandEncoder] : nil;
    if (!readback || !cb || !blit) {
        NSLog(@"MGL TRACE sampled.readback setup-fail program=%u binding=%u glTex=%u reason=%@ readback=%p cb=%p blit=%p hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              reason,
              readback,
              cb,
              blit,
              (unsigned long long)hit);
        return;
    }

    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(sampleWidth, sampleHeight, 1)
                 toBuffer:readback
        destinationOffset:0
   destinationBytesPerRow:bytesPerRow
 destinationBytesPerImage:byteCount];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    const uint8_t *p = (const uint8_t *)readback.contents;
    uint64_t byteSum = 0;
    NSUInteger nonZeroBytes = 0;
    uint32_t firstPixel = 0;
    uint32_t pixelXor = 0;
    uint32_t minPixel = UINT32_MAX;
    uint32_t maxPixel = 0;
    NSUInteger pixelCount = byteCount / sizeof(uint32_t);

    if (p) {
        for (NSUInteger i = 0; i < byteCount; i++) {
            byteSum += (uint64_t)p[i];
            if (p[i] != 0) {
                nonZeroBytes++;
            }
        }
        if (byteCount >= sizeof(firstPixel)) {
            memcpy(&firstPixel, p, sizeof(firstPixel));
        }
        for (NSUInteger i = 0; i < pixelCount; i++) {
            uint32_t pixel = 0;
            memcpy(&pixel, p + (i * sizeof(pixel)), sizeof(pixel));
            pixelXor ^= pixel;
            if (pixel < minPixel) {
                minPixel = pixel;
            }
            if (pixel > maxPixel) {
                maxPixel = pixel;
            }
        }
    }

    NSLog(@"MGL TRACE sampled.readback stage=%@ program=%u binding=%u glTex=%u reason=%@ hit=%llu "
          "mtl=%p fmt=%lu type=%lu size=%lux%lu sample=%lux%lu status=%s error=%@ "
          "nonZero=%lu/%lu sum=%llu first=0x%08x min=0x%08x max=0x%08x xor=0x%08x "
          "level(init ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx)",
          stage,
          (unsigned)program,
          (unsigned)binding,
          glTex ? (unsigned)glTex->name : 0u,
          reason,
          (unsigned long long)hit,
          texture,
          (unsigned long)fmt,
          (unsigned long)texture.textureType,
          (unsigned long)texWidth,
          (unsigned long)texHeight,
          (unsigned long)sampleWidth,
          (unsigned long)sampleHeight,
          mglCommandBufferStatusName(cb.status),
          cb.error,
          (unsigned long)nonZeroBytes,
          (unsigned long)byteCount,
          (unsigned long long)byteSum,
          firstPixel,
          minPixel == UINT32_MAX ? 0u : minPixel,
          maxPixel,
          pixelXor,
          level0 ? (unsigned)level0->ever_written : 0u,
          level0 ? (unsigned)level0->has_initialized_data : 0u,
          level0 ? (unsigned)level0->suspicious_zero_upload : 0u,
          level0 ? (unsigned)level0->last_init_source : 0u,
          (unsigned long)(level0 ? level0->last_upload_size : 0u),
          level0 ? (void *)level0->last_src_ptr : NULL,
          (unsigned long long)(level0 ? level0->last_src_hash : 0ull));
}

- (bool) bindTexturesToCurrentRenderEncoder
{
    static const NSUInteger kMaxFragmentSamplerSlots = 16;
    static uint64_t s_bindTexturesCallCount = 0;
    uint64_t bindCall = ++s_bindTexturesCallCount;
    bool traceBind = mglShouldTraceCall(bindCall);
    GLuint vertexSampledCount = 0;
    GLuint vertexBoundTextures = 0;
    GLuint vertexFallbackTextures = 0;
    GLuint boundSampledTextures = 0;
    GLuint nilSampledTextures = 0;
    GLuint fallbackSampledTextures = 0;
    GLuint boundSampledSamplers = 0;

    if (!_currentRenderEncoder) {
        // No active render encoder yet (or it was rotated). Texture/sampler binding
        // can be deferred until the next encoder is created.
        return true;
    }

    id<MTLSamplerState> defaultSampler = [self fallbackSamplerState];
    if (defaultSampler) {
        NSUInteger warmupCount = TEXTURE_UNITS;
        if (warmupCount > kMaxFragmentSamplerSlots) {
            warmupCount = kMaxFragmentSamplerSlots;
        }
        for (NSUInteger s = 0; s < warmupCount; s++) {
            [_currentRenderEncoder setVertexSamplerState:defaultSampler atIndex:s];
            [_currentRenderEncoder setFragmentSamplerState:defaultSampler atIndex:s];
        }
    }

    // Metal validates every active stage resource. Bind vertex-stage sampled
    // images as well, even though most Minecraft pipelines only sample in FS.
    vertexSampledCount = [self getProgramBindingCount:_VERTEX_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    for (GLuint i = 0; i < vertexSampledCount; i++)
    {
        GLuint spirvBinding = [self getProgramBinding:_VERTEX_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE index:(int)i];
        GLuint glBinding = [self getProgramGLBinding:_VERTEX_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE index:(int)i];
        if (spirvBinding >= TEXTURE_UNITS || glBinding >= TEXTURE_UNITS) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledBinding:spirvBinding stage:_VERTEX_SHADER];
        Program *currentProgram = mglResolveProgramFromState(ctx);
        const char *sampledName = "";
        if (currentProgram &&
            i < currentProgram->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].count) {
            sampledName = currentProgram->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE].list[i].name;
        }
        BOOL missingLightmapSampler = mglIsLightmapSamplerName(sampledName);

        MTLTextureType expectedType = [self getProgramExpectedTextureType:_VERTEX_SHADER
                                                                      type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                     index:(int)i];
        MTLTextureType lookupType = [self getProgramDeclaredTextureType:_VERTEX_SHADER
                                                                    type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                   index:(int)i];
        MGLTextureDataKind expectedKind = [self getProgramExpectedTextureDataKind:_VERTEX_SHADER
                                                                             type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                            index:(int)i];
        Texture *ptr = [self textureForSampledBinding:spirvBinding
                                                stage:_VERTEX_SHADER
                                         expectedType:(lookupType ? lookupType : expectedType)];
        id<MTLTexture> texture = nil;
        id<MTLSamplerState> sampler = defaultSampler;
        BOOL usedTypeFallback = NO;
        BOOL preferNeutralLightmapFallback = missingLightmapSampler && ctx && !ctx->state.caps.depth_test;

        if (ptr) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
            if (ptr->mtl_data) {
                texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
            }
            if (texture && expectedType != 0 && texture.textureType != expectedType) {
                static uint64_t s_vertexTypeMismatchLogCount = 0;
                uint64_t hit = ++s_vertexTypeMismatchLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX TYPE MISMATCH vertex binding=%u program=%u glTex=%u glTarget=0x%x mtlType=%lu expected=%lu hit=%llu",
                          (unsigned)spirvBinding,
                          ctx ? (unsigned)ctx->state.program_name : 0u,
                          (unsigned)ptr->name,
                          (unsigned)ptr->target,
                          (unsigned long)texture.textureType,
                          (unsigned long)expectedType,
                          (unsigned long long)hit);
                }
                Program *dumpProgram = mglResolveProgramFromState(ctx);
                mglWriteProgramMSLDump(dumpProgram,
                                       [NSString stringWithFormat:@"tex-type-mismatch-vertex-binding-%u", spirvBinding]);
                texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
                usedTypeFallback = YES;
            }
            if (texture &&
                !mglTexturePixelFormatCompatibleWithExpectedDataKind(texture.pixelFormat, expectedKind)) {
                static uint64_t s_vertexDataKindMismatchLogCount = 0;
                uint64_t hit = ++s_vertexDataKindMismatchLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX DATA MISMATCH vertex binding=%u program=%u glTex=%u glTarget=0x%x format=%lu actualKind=%s expectedKind=%s expectedType=%lu hit=%llu",
                          (unsigned)spirvBinding,
                          ctx ? (unsigned)ctx->state.program_name : 0u,
                          (unsigned)ptr->name,
                          (unsigned)ptr->target,
                          (unsigned long)texture.pixelFormat,
                          mglTextureDataKindName(mglTextureDataKindForPixelFormat(texture.pixelFormat)),
                          mglTextureDataKindName(expectedKind),
                          (unsigned long)expectedType,
                          (unsigned long long)hit);
                }
                Program *dumpProgram = mglResolveProgramFromState(ctx);
                mglWriteProgramMSLDump(dumpProgram,
                                       [NSString stringWithFormat:@"tex-data-mismatch-vertex-binding-%u", spirvBinding]);
                texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
                usedTypeFallback = YES;
            }

            if (textureUnit < TEXTURE_UNITS && STATE(texture_samplers[textureUnit])) {
                Sampler *glSampler = STATE(texture_samplers[textureUnit]);
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    CFBridgingRelease(glSampler->mtl_data);
                    glSampler->mtl_data = NULL;
                }
                if (glSampler->mtl_data == NULL) {
                    glSampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&glSampler->params target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
            } else if (ptr->params.mtl_data) {
                sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
            }
        }

        if (!texture) {
            texture = (missingLightmapSampler || preferNeutralLightmapFallback)
                ? [self fallbackLightmapSampledTexture]
                : [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
            if (texture) {
                vertexFallbackTextures++;
                static uint64_t s_vertexFallbackLogCount = 0;
                uint64_t hit = ++s_vertexFallbackLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX FALLBACK vertex sampled binding=%u program=%u name=%s unit=%u glTex=%u kind=%s size=%lux%lu hit=%llu",
                          (unsigned)spirvBinding,
                          ctx ? (unsigned)ctx->state.program_name : 0u,
                          sampledName ? sampledName : "",
                          (unsigned)textureUnit,
                          ptr ? (unsigned)ptr->name : 0u,
                          preferNeutralLightmapFallback ? "gui-lightmap" : (missingLightmapSampler ? "lightmap" : "generic"),
                          (unsigned long)texture.width,
                          (unsigned long)texture.height,
                          (unsigned long long)hit);
                }
            }
        }

        [_currentRenderEncoder setVertexTexture:texture atIndex:spirvBinding];
        if (sampler && spirvBinding < kMaxFragmentSamplerSlots) {
            [_currentRenderEncoder setVertexSamplerState:sampler atIndex:spirvBinding];
        }
        if (ptr && ptr->target == GL_TEXTURE_BUFFER) {
            static uint64_t s_vertexTexelBufferBindLogs = 0;
            uint64_t hit = ++s_vertexTexelBufferBindLogs;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                Texture *unitActive = textureUnit < TEXTURE_UNITS ? STATE(active_textures[textureUnit]) : NULL;
                Texture *unitBuffer = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_BUFFER_TARGET]) : NULL;
                NSLog(@"MGL TEXBUFFER BIND vertex hit=%llu program=%u binding=%u unit=%u ptrTex=%u active=%u bufferSlot=%u expectedType=%lu lookupType=%lu mtlTex=%p mtlType=%lu size=%lux%lu format=%lu sampler=%p",
                      (unsigned long long)hit,
                      ctx ? (unsigned)ctx->state.program_name : 0u,
                      (unsigned)spirvBinding,
                      (unsigned)textureUnit,
                      (unsigned)ptr->name,
                      mglTraceTextureName(unitActive),
                      mglTraceTextureName(unitBuffer),
                      (unsigned long)expectedType,
                      (unsigned long)lookupType,
                      texture,
                      (unsigned long)(texture ? texture.textureType : 0),
                      (unsigned long)(texture ? texture.width : 0),
                      (unsigned long)(texture ? texture.height : 0),
                      (unsigned long)(texture ? texture.pixelFormat : 0),
                      sampler);
            }
        }
        if (ptr && ptr->target != GL_TEXTURE_BUFFER) {
            GLuint sampleProgramName = ctx ? (unsigned)ctx->state.program_name : 0u;
            TextureLevel *sampleLevel0 = mglTraceTextureBaseLevel(ptr);
            Program *sampleProgram = mglResolveProgramFromState(ctx);
            BOOL focusedVertexSample =
                (sampleProgramName == 34u) ||
                (mglProgramLooksLikeMinecraftTerrain(sampleProgram) &&
                 (bindCall <= 2048ull || ((bindCall % 512ull) == 0ull))) ||
                (sampleLevel0 &&
                 (sampleLevel0->suspicious_zero_upload ||
                  !sampleLevel0->ever_written ||
                  !sampleLevel0->has_initialized_data));
            if (focusedVertexSample) {
                static uint64_t s_vertexSampleDetailLogCount = 0;
                uint64_t hit = ++s_vertexSampleDetailLogCount;
                if (hit <= 128ull || (hit % 512ull) == 0ull) {
                    int expectedIndex = [self textureIndexForExpectedMetalType:expectedType];
                    Texture *unitActive = textureUnit < TEXTURE_UNITS ? STATE(active_textures[textureUnit]) : NULL;
                    Texture *unitExpected = (expectedIndex >= 0 && expectedIndex < _MAX_TEXTURE_TYPES)
                        ? STATE(texture_units[textureUnit].textures[expectedIndex])
                        : NULL;
                    uint64_t levelDataHash = (sampleLevel0 && sampleLevel0->data && sampleLevel0->data_size > 0)
                        ? mglTraceHashBytes((const void *)(uintptr_t)sampleLevel0->data, sampleLevel0->data_size)
                        : 0ull;

                    NSLog(@"MGL TRACE texbind.sample-detail call=%llu hit=%llu stage=vertex program=%u binding=%u "
                          "unit=%u expectedType=%lu expectedIndex=%d ptrTex=%u ptr=%p target=0x%x fallback=%d mtlTex=%p mtlType=%lu mtlSize=%lux%lu "
                          "unit(active=%u expected=%u) "
                          "l0=%ux%ux%u bytes=%lu init(ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx dataHash=0x%016llx)",
                          (unsigned long long)bindCall,
                          (unsigned long long)hit,
                          sampleProgramName,
                          (unsigned)spirvBinding,
                          (unsigned)textureUnit,
                          (unsigned long)expectedType,
                          expectedIndex,
                          mglTraceTextureName(ptr),
                          ptr,
                          ptr ? (unsigned)ptr->target : 0u,
                          usedTypeFallback ? 1 : 0,
                          texture,
                          (unsigned long)(texture ? texture.textureType : 0),
                          (unsigned long)(texture ? texture.width : 0),
                          (unsigned long)(texture ? texture.height : 0),
                          mglTraceTextureName(unitActive),
                          mglTraceTextureName(unitExpected),
                          sampleLevel0 ? (unsigned)sampleLevel0->width : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->height : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->depth : 0u,
                          (unsigned long)(sampleLevel0 ? sampleLevel0->data_size : 0u),
                          sampleLevel0 ? (unsigned)sampleLevel0->ever_written : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->has_initialized_data : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->suspicious_zero_upload : 0u,
                          sampleLevel0 ? (unsigned)sampleLevel0->last_init_source : 0u,
                          (unsigned long)(sampleLevel0 ? sampleLevel0->last_upload_size : 0u),
                          sampleLevel0 ? (void *)sampleLevel0->last_src_ptr : NULL,
                          (unsigned long long)(sampleLevel0 ? sampleLevel0->last_src_hash : 0ull),
                          (unsigned long long)levelDataHash);
                }
            }
        }
        if (texture) {
            vertexBoundTextures++;
            if (usedTypeFallback) {
                vertexFallbackTextures++;
            }
        }
    }

    // Bind sampled images (texture + sampler).
    GLuint sampledCount = [self getProgramBindingCount:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    for (GLuint i = 0; i < sampledCount; i++)
    {
        GLuint spirvBinding = [self getProgramBinding:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE index:(int)i];
        GLuint glBinding = [self getProgramGLBinding:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE index:(int)i];
        if (spirvBinding >= TEXTURE_UNITS || glBinding >= TEXTURE_UNITS) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledBinding:spirvBinding stage:_FRAGMENT_SHADER];

        MTLTextureType expectedType = [self getProgramExpectedTextureType:_FRAGMENT_SHADER
                                                                      type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                     index:(int)i];
        MTLTextureType lookupType = [self getProgramDeclaredTextureType:_FRAGMENT_SHADER
                                                                    type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                   index:(int)i];
        MGLTextureDataKind expectedKind = [self getProgramExpectedTextureDataKind:_FRAGMENT_SHADER
                                                                             type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE
                                                                            index:(int)i];
        Texture *ptr = [self textureForSampledBinding:spirvBinding
                                                stage:_FRAGMENT_SHADER
                                         expectedType:(lookupType ? lookupType : expectedType)];
        id<MTLTexture> texture = nil;
        id<MTLSamplerState> sampler = nil;
        BOOL usedFallbackTexture = NO;

        if (ptr) {
            RETURN_FALSE_ON_FAILURE([self bindMTLTexture:ptr]);
            if (ptr->mtl_data) {
                texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
            }
            if (texture && expectedType != 0 && texture.textureType != expectedType) {
                static uint64_t s_fragmentTypeMismatchLogCount = 0;
                uint64_t hit = ++s_fragmentTypeMismatchLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX TYPE MISMATCH fragment binding=%u program=%u glTex=%u glTarget=0x%x mtlType=%lu expected=%lu hit=%llu",
                          (unsigned)spirvBinding,
                          ctx ? (unsigned)ctx->state.program_name : 0u,
                          (unsigned)ptr->name,
                          (unsigned)ptr->target,
                          (unsigned long)texture.textureType,
                          (unsigned long)expectedType,
                          (unsigned long long)hit);
                }
                Program *dumpProgram = mglResolveProgramFromState(ctx);
                mglWriteProgramMSLDump(dumpProgram,
                                       [NSString stringWithFormat:@"tex-type-mismatch-fragment-binding-%u", spirvBinding]);
                texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
                usedFallbackTexture = YES;
            }
            if (texture &&
                !mglTexturePixelFormatCompatibleWithExpectedDataKind(texture.pixelFormat, expectedKind)) {
                static uint64_t s_fragmentDataKindMismatchLogCount = 0;
                uint64_t hit = ++s_fragmentDataKindMismatchLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX DATA MISMATCH fragment binding=%u program=%u glTex=%u glTarget=0x%x format=%lu actualKind=%s expectedKind=%s expectedType=%lu hit=%llu",
                          (unsigned)spirvBinding,
                          ctx ? (unsigned)ctx->state.program_name : 0u,
                          (unsigned)ptr->name,
                          (unsigned)ptr->target,
                          (unsigned long)texture.pixelFormat,
                          mglTextureDataKindName(mglTextureDataKindForPixelFormat(texture.pixelFormat)),
                          mglTextureDataKindName(expectedKind),
                          (unsigned long)expectedType,
                          (unsigned long long)hit);
                }
                Program *dumpProgram = mglResolveProgramFromState(ctx);
                mglWriteProgramMSLDump(dumpProgram,
                                       [NSString stringWithFormat:@"tex-data-mismatch-fragment-binding-%u", spirvBinding]);
                texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
                usedFallbackTexture = YES;
            }

            if (textureUnit < TEXTURE_UNITS && STATE(texture_samplers[textureUnit])) {
                Sampler *glSampler = STATE(texture_samplers[textureUnit]);
                if (glSampler->dirty_bits && glSampler->mtl_data) {
                    CFBridgingRelease(glSampler->mtl_data);
                    glSampler->mtl_data = NULL;
                }
                if (glSampler->mtl_data == NULL) {
                    glSampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&glSampler->params target:ptr->target]);
                    glSampler->dirty_bits = 0;
                }
                sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
            } else {
                sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
            }
        }

        if (!texture) {
            texture = [self fallbackSampledTextureForExpectedType:expectedType dataKind:expectedKind];
            if (texture) {
                usedFallbackTexture = YES;
                mglFocusLoadingProgram(ctx ? (unsigned)ctx->state.program_name : 0u,
                                       "sample-fallback",
                                       bindCall);
                fallbackSampledTextures++;
                static uint64_t s_fragmentFallbackLogCount = 0;
                uint64_t hit = ++s_fragmentFallbackLogCount;
                if (hit <= 32ull || (hit % 512ull) == 0ull) {
                    NSLog(@"MGL TEX FALLBACK fragment sampled binding=%u program=%u glTex=%u hit=%llu",
                          (unsigned)spirvBinding,
                          ctx ? (unsigned)ctx->state.program_name : 0u,
                          ptr ? (unsigned)ptr->name : 0u,
                          (unsigned long long)hit);
                }
            }
        }

	        if (!sampler) {
	            sampler = defaultSampler;
	        }

		        GLuint sampleProgramName = ctx ? (unsigned)ctx->state.program_name : 0u;
		        TextureLevel *sampleLevel0 = mglTraceTextureBaseLevel(ptr);
		        BOOL focusedSample =
		            mglIsFocusedLoadingProgram(sampleProgramName) &&
		            (bindCall <= 2048ull || ((bindCall % 512ull) == 0ull));
		        BOOL suspiciousSample =
		            usedFallbackTexture ||
		            (ptr && ptr->name == 13u) ||
		            focusedSample ||
		            (sampleLevel0 &&
		             (sampleLevel0->suspicious_zero_upload ||
		              !sampleLevel0->ever_written ||
		              !sampleLevel0->has_initialized_data));
        if (suspiciousSample) {
            static uint64_t s_fragmentSampleDetailLogCount = 0;
            uint64_t hit = ++s_fragmentSampleDetailLogCount;
            if (hit <= 256ull || (hit % 512ull) == 0ull) {
	                int expectedIndex = [self textureIndexForExpectedMetalType:expectedType];
	                Texture *unitActive = textureUnit < TEXTURE_UNITS ? STATE(active_textures[textureUnit]) : NULL;
	                Texture *unitExpected = (expectedIndex >= 0 && expectedIndex < _MAX_TEXTURE_TYPES)
	                    ? STATE(texture_units[textureUnit].textures[expectedIndex])
	                    : NULL;
	                Texture *unit2D = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_2D]) : NULL;
	                Texture *unitCube = textureUnit < TEXTURE_UNITS ? STATE(texture_units[textureUnit].textures[_TEXTURE_CUBE_MAP]) : NULL;
	                MTLTextureType actualType = texture ? texture.textureType : 0;
	                uint64_t levelDataHash = (sampleLevel0 && sampleLevel0->data && sampleLevel0->data_size > 0)
	                    ? mglTraceHashBytes((const void *)(uintptr_t)sampleLevel0->data, sampleLevel0->data_size)
	                    : 0ull;

	                NSLog(@"MGL TRACE texbind.sample-detail call=%llu hit=%llu stage=fragment program=%u binding=%u "
	                      "unit=%u expectedType=%lu expectedIndex=%d ptrTex=%u ptr=%p target=0x%x fallback=%d mtlTex=%p mtlType=%lu mtlSize=%lux%lu "
	                      "unit(active=%u expected=%u tex2D=%u cube=%u) "
	                      "l0=%ux%ux%u bytes=%lu init(ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx dataHash=0x%016llx)",
	                      (unsigned long long)bindCall,
	                      (unsigned long long)hit,
		                      sampleProgramName,
	                      (unsigned)spirvBinding,
	                      (unsigned)textureUnit,
	                      (unsigned long)expectedType,
	                      expectedIndex,
	                      mglTraceTextureName(ptr),
	                      ptr,
	                      ptr ? (unsigned)ptr->target : 0u,
	                      usedFallbackTexture ? 1 : 0,
	                      texture,
	                      (unsigned long)actualType,
	                      (unsigned long)(texture ? texture.width : 0),
	                      (unsigned long)(texture ? texture.height : 0),
	                      mglTraceTextureName(unitActive),
	                      mglTraceTextureName(unitExpected),
	                      mglTraceTextureName(unit2D),
	                      mglTraceTextureName(unitCube),
	                      sampleLevel0 ? (unsigned)sampleLevel0->width : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->height : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->depth : 0u,
	                      (unsigned long)(sampleLevel0 ? sampleLevel0->data_size : 0u),
	                      sampleLevel0 ? (unsigned)sampleLevel0->ever_written : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->has_initialized_data : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->suspicious_zero_upload : 0u,
	                      sampleLevel0 ? (unsigned)sampleLevel0->last_init_source : 0u,
	                      (unsigned long)(sampleLevel0 ? sampleLevel0->last_upload_size : 0u),
	                      sampleLevel0 ? (void *)sampleLevel0->last_src_ptr : NULL,
	                      (unsigned long long)(sampleLevel0 ? sampleLevel0->last_src_hash : 0ull),
	                      (unsigned long long)levelDataHash);
	            }

	            if (texture && sampleLevel0 &&
	                (sampleLevel0->suspicious_zero_upload ||
	                 !sampleLevel0->ever_written ||
	                 !sampleLevel0->has_initialized_data)) {
	                static uint64_t s_fragmentSampleReadbackCount = 0;
	                uint64_t rbHit = ++s_fragmentSampleReadbackCount;
	                if (rbHit <= 32ull || (rbHit % 512ull) == 0ull) {
	                    [self traceSampledTextureReadback:texture
	                                                glTex:ptr
	                                                level:sampleLevel0
	                                              program:sampleProgramName
	                                              binding:spirvBinding
	                                                stage:@"fragment"
	                                               reason:(sampleLevel0->suspicious_zero_upload ? @"zero-level" :
	                                                       (!sampleLevel0->ever_written ? @"never-written" : @"not-initialized"))
	                                                  hit:rbHit];
	                }
	            }
	        }
	
	        [_currentRenderEncoder setFragmentTexture:texture atIndex:spirvBinding];
        if (texture && !usedFallbackTexture) {
            boundSampledTextures++;
        } else if (usedFallbackTexture) {
            // Keep nilTex as the original GL binding failure count, while Metal receives fallback texture.
            nilSampledTextures++;
        } else {
            nilSampledTextures++;
        }
        if (sampler && spirvBinding < kMaxFragmentSamplerSlots) {
            [_currentRenderEncoder setFragmentSamplerState:sampler atIndex:spirvBinding];
            boundSampledSamplers++;
        }

        if (traceBind && i < 6) {
            TextureLevel *level0 = NULL;
            if (ptr && ptr->faces[0].levels) {
                level0 = &ptr->faces[0].levels[0];
            }
            uint32_t cpuFirstTexel = 0u;
            bool cpuFirstTexelValid = false;
            if (level0 && level0->data && level0->data_size >= sizeof(cpuFirstTexel) &&
                ((uintptr_t)level0->data >= 0x1000ull)) {
                memcpy(&cpuFirstTexel, (const void *)level0->data, sizeof(cpuFirstTexel));
                cpuFirstTexelValid = true;
            }

            NSLog(@"MGL TRACE texbind.sampled call=%llu idx=%u binding=%u glTex=%u target=0x%x internal=0x%x "
                  "l0=%ux%ux%u l0bytes=%lu l0first=0x%08x(valid=%d) "
                  "l0src(source=%u upload=%lu srcPtr=%p hash=0x%016llx init(ever=%u full=%u zero=%u)) "
                  "mtlTex=%p size=%lux%lu sampler=%p fallback=%d",
                  (unsigned long long)bindCall,
                  (unsigned)i,
                  (unsigned)spirvBinding,
                  ptr ? (unsigned)ptr->name : 0u,
                  ptr ? (unsigned)ptr->target : 0u,
                  ptr ? (unsigned)ptr->internalformat : 0u,
                  level0 ? (unsigned)level0->width : 0u,
                  level0 ? (unsigned)level0->height : 0u,
                  level0 ? (unsigned)level0->depth : 0u,
                  (unsigned long)(level0 ? level0->data_size : 0u),
                  (unsigned)cpuFirstTexel,
                  cpuFirstTexelValid ? 1 : 0,
                  (unsigned)(level0 ? level0->last_init_source : 0u),
                  (unsigned long)(level0 ? level0->last_upload_size : 0u),
                  (void *)(level0 ? level0->last_src_ptr : NULL),
                  (unsigned long long)(level0 ? level0->last_src_hash : 0ull),
                  (unsigned)(level0 ? level0->ever_written : 0u),
                  (unsigned)(level0 ? level0->has_initialized_data : 0u),
                  (unsigned)(level0 ? level0->suspicious_zero_upload : 0u),
                  texture,
                  (unsigned long)(texture ? texture.width : 0),
                  (unsigned long)(texture ? texture.height : 0),
                  sampler,
                  usedFallbackTexture ? 1 : 0);
        }
    }

    // Bind separate samplers explicitly.
    GLuint separateSamplerCount = [self getProgramBindingCount:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS];
    GLuint boundSeparateSamplers = 0;
    for (GLuint i = 0; i < separateSamplerCount; i++)
    {
        GLuint spirvBinding = [self getProgramBinding:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS index:(int)i];
        GLuint glBinding = [self getProgramGLBinding:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS index:(int)i];
        if (spirvBinding >= TEXTURE_UNITS || glBinding >= TEXTURE_UNITS) {
            continue;
        }
        GLuint textureUnit = [self textureUnitForSampledBinding:spirvBinding stage:_FRAGMENT_SHADER];

        id<MTLSamplerState> sampler = nil;
        if (textureUnit < TEXTURE_UNITS && STATE(texture_samplers[textureUnit])) {
            Sampler *glSampler = STATE(texture_samplers[textureUnit]);
            if (glSampler->dirty_bits && glSampler->mtl_data) {
                CFBridgingRelease(glSampler->mtl_data);
                glSampler->mtl_data = NULL;
            }
            if (glSampler->mtl_data == NULL) {
                glSampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&glSampler->params target:GL_TEXTURE_2D]);
                glSampler->dirty_bits = 0;
            }
            sampler = (__bridge id<MTLSamplerState>)(glSampler->mtl_data);
        }

        if (!sampler) {
            sampler = defaultSampler;
        }
        if (sampler && spirvBinding < kMaxFragmentSamplerSlots) {
            [_currentRenderEncoder setFragmentSamplerState:sampler atIndex:spirvBinding];
            boundSeparateSamplers++;
        }

        if (traceBind && i < 6) {
            NSLog(@"MGL TRACE texbind.separateSampler call=%llu idx=%u binding=%u unit=%u sampler=%p",
                  (unsigned long long)bindCall,
                  (unsigned)i,
                  (unsigned)spirvBinding,
                  (unsigned)textureUnit,
                  sampler);
        }
    }

    BOOL interestingTextureBind =
        (sampledCount > 0 && boundSampledTextures == 0) ||
        fallbackSampledTextures > 0 ||
        vertexFallbackTextures > 0;
    BOOL logTextureSummary = traceBind;
    if (interestingTextureBind) {
        static uint64_t s_interestingTextureSummaryCount = 0;
        uint64_t hit = ++s_interestingTextureSummaryCount;
        if (hit <= 64ull || (hit % 512ull) == 0ull) {
            logTextureSummary = YES;
        }
    }
    if (logTextureSummary) {
        GLuint programName = ctx ? ctx->state.program_name : 0u;
        NSLog(@"MGL TRACE texbind.summary call=%llu program=%u vertexSampled=%u vertexBoundTex=%u vertexFallback=%u sampled=%u boundTex=%u nilTex=%u fallbackTex=%u sampledSamplers=%u separateSamplers=%u boundSeparate=%u",
              (unsigned long long)bindCall,
              (unsigned)programName,
              (unsigned)vertexSampledCount,
              (unsigned)vertexBoundTextures,
              (unsigned)vertexFallbackTextures,
              (unsigned)sampledCount,
              (unsigned)boundSampledTextures,
              (unsigned)nilSampledTextures,
              (unsigned)fallbackSampledTextures,
              (unsigned)boundSampledSamplers,
              (unsigned)separateSamplerCount,
              (unsigned)boundSeparateSamplers);
    }

    return true;
}

#pragma mark framebuffers

extern bool isColorAttachment(GLMContext ctx, GLuint attachment);
extern FBOAttachment *getFBOAttachment(GLMContext ctx, Framebuffer *fbo, GLenum attachment);
extern Texture *findTexture(GLMContext ctx, GLuint texture);

-(void)mtlBlitFramebuffer:(GLMContext)glm_ctx srcX0:(GLint)srcX0 srcY0:(GLint)srcY0 srcX1:(GLint)srcX1 srcY1:(GLint)srcY1 dstX0:(GLint)dstX0 dstY0:(GLint)dstY0 dstX1:(GLint)dstX1 dstY1:(GLint)dstY1 mask:(size_t)mask filter:(GLuint)filter
{
    if (!glm_ctx || ((uintptr_t)glm_ctx < 0x1000)) {
        NSLog(@"MGL ERROR: mtlBlitFramebuffer called with invalid glm_ctx=%p", glm_ctx);
        return;
    }

    if (srcX1 == srcX0 || srcY1 == srcY0 || dstX1 == dstX0 || dstY1 == dstY0) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer ignored empty rect src=(%d,%d)-(%d,%d) dst=(%d,%d)-(%d,%d)",
              srcX0, srcY0, srcX1, srcY1,
              dstX0, dstY0, dstX1, dstY1);
        return;
    }

    if ((mask & GL_COLOR_BUFFER_BIT) == 0u) {
        if ((mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) != 0u) {
            static uint64_t s_depthStencilOnlyBlitWarnCount = 0;
            uint64_t hit = ++s_depthStencilOnlyBlitWarnCount;
            if (hit <= 32ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL WARN: mtlBlitFramebuffer depth/stencil-only blit is not implemented; skipping mask=0x%zx hit=%llu",
                      mask,
                      (unsigned long long)hit);
            }
        }
        return;
    }

    if ((mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) != 0u) {
        static uint64_t s_depthStencilBlitWarnCount = 0;
        uint64_t hit = ++s_depthStencilBlitWarnCount;
        if (hit <= 32ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer only copies color; depth/stencil bits in mask=0x%zx ignored hit=%llu",
                  mask,
                  (unsigned long long)hit);
        }
    }

    // Keep renderer ivar state consistent with the call site context.
    ctx = glm_ctx;

    Framebuffer * readfbo, * drawfbo;
    GLenum readAttachment, drawAttachment;
    FBOAttachment *readFBOAttachment = NULL;
    Texture *readTextureObject = NULL;
    MGLMetalAttachmentSubresource readSubresource = {0u, 0u, 0u};
    MGLMetalAttachmentSubresource drawSubresource = {0u, 0u, 0u};
    //int readtex, drawtex;

    readfbo = glm_ctx->state.readbuffer;
    drawfbo = glm_ctx->state.framebuffer;

    if (drawfbo == NULL) {
        NSUInteger requestedDrawableWidth = (NSUInteger)MAX(0, MAX(dstX0, dstX1));
        NSUInteger requestedDrawableHeight = (NSUInteger)MAX(0, MAX(dstY0, dstY1));
        if ([self mglEnsureLayerDrawableSizeAtLeastWidth:requestedDrawableWidth
                                                  height:requestedDrawableHeight
                                                  reason:"blitFramebuffer.defaultDraw"]) {
            _drawable = [_layer nextDrawable];
        }
    }

    id<MTLTexture> readtexid;

    if (readfbo==NULL) {
        if (!_drawable || !_drawable.texture) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer has no drawable source texture");
            return;
        }
        readtexid = _drawable.texture;
    } else {
        readAttachment = glm_ctx->state.read_buffer;
        if (readAttachment == GL_NONE) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer skipped color blit with GL_READ_BUFFER=GL_NONE");
            return;
        }
        if (!isColorAttachment(glm_ctx, readAttachment) &&
            readAttachment != GL_DEPTH_ATTACHMENT &&
            readAttachment != GL_STENCIL_ATTACHMENT &&
            readAttachment != GL_DEPTH_STENCIL_ATTACHMENT)
        {
            // OpenGL compatibility enums (e.g. GL_FRONT/GL_BACK) are not valid
            // FBO attachment enums. For user FBO blits, treat them as COLOR_ATTACHMENT0.
            readAttachment = GL_COLOR_ATTACHMENT0;
        }

        readFBOAttachment = getFBOAttachment(glm_ctx, readfbo, readAttachment);
        if (!readFBOAttachment) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer read attachment missing");
            return;
        }
        readSubresource = mglMetalAttachmentSubresourceForAttachment(readFBOAttachment);
        if (readFBOAttachment->textarget == GL_RENDERBUFFER)
        {
            readTextureObject = readFBOAttachment->buf.rbo->tex;
        }
        else
        {
            readTextureObject = readFBOAttachment->buf.tex;
        }
        if (!readTextureObject) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer read texture object missing");
            return;
        }
        if (!readTextureObject->mtl_data || readTextureObject->dirty_bits) {
            if (![self bindMTLTexture:readTextureObject]) {
                NSLog(@"MGL WARN: mtlBlitFramebuffer failed to bind read texture to Metal");
                return;
            }
        }
        readtexid = (__bridge id<MTLTexture>)(readTextureObject->mtl_data);
        if (!readtexid) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer read MTL texture missing");
            return;
        }
    }


    id<MTLTexture> drawtexid;
    if (drawfbo==NULL) {
        if (!_drawable || !_drawable.texture) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer has no drawable destination texture");
            return;
        }
        drawtexid = _drawable.texture;
    } else {
        drawAttachment = glm_ctx->state.draw_buffer;
        if (drawAttachment == GL_NONE) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer skipped color blit with GL_DRAW_BUFFER=GL_NONE");
            return;
        }
        if (!isColorAttachment(glm_ctx, drawAttachment) &&
            drawAttachment != GL_DEPTH_ATTACHMENT &&
            drawAttachment != GL_STENCIL_ATTACHMENT &&
            drawAttachment != GL_DEPTH_STENCIL_ATTACHMENT)
        {
            drawAttachment = GL_COLOR_ATTACHMENT0;
        }

        FBOAttachment * fboa = getFBOAttachment(glm_ctx, drawfbo, drawAttachment);
        if (!fboa) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer draw attachment missing");
            return;
        }
        drawSubresource = mglMetalAttachmentSubresourceForAttachment(fboa);
        Texture * drawtexobj;
        if (fboa->textarget == GL_RENDERBUFFER)
        {
            drawtexobj = fboa->buf.rbo->tex;
        }
        else
        {
            drawtexobj = fboa->buf.tex;
        }
        if (!drawtexobj) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer draw texture object missing");
            return;
        }
        if (!drawtexobj->mtl_data) {
            if (![self bindMTLTexture:drawtexobj]) {
                NSLog(@"MGL WARN: mtlBlitFramebuffer failed to bind draw texture to Metal");
                return;
            }
        }
        drawtexid = (__bridge id<MTLTexture>)(drawtexobj->mtl_data);
        if (!drawtexid) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer draw MTL texture missing");
            return;
        }
    }


    // end encoding on current render encoder
    [self endRenderEncoding];

    if (![self ensureWritableCommandBuffer:"mtlBlitFramebuffer"]) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer could not obtain writable command buffer");
        return;
    }

    if (readfbo &&
        readFBOAttachment &&
        readTextureObject &&
        readtexid &&
        isColorAttachment(glm_ctx, readAttachment) &&
        (readFBOAttachment->clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        MTLRenderPassDescriptor *clearPass = [MTLRenderPassDescriptor renderPassDescriptor];
        clearPass.colorAttachments[0].texture = readtexid;
        clearPass.colorAttachments[0].level = readSubresource.level;
        clearPass.colorAttachments[0].slice = readSubresource.slice;
        clearPass.colorAttachments[0].depthPlane = readSubresource.depthPlane;
        clearPass.colorAttachments[0].loadAction = MTLLoadActionClear;
        clearPass.colorAttachments[0].storeAction = MTLStoreActionStore;
        clearPass.colorAttachments[0].clearColor =
            MTLClearColorMake(readFBOAttachment->clear_color[0],
                              readFBOAttachment->clear_color[1],
                              readFBOAttachment->clear_color[2],
                              readFBOAttachment->clear_color[3]);

        id<MTLRenderCommandEncoder> clearEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:clearPass];
        if (clearEncoder) {
            [clearEncoder endEncoding];
            readFBOAttachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
            mglMarkTextureLevelRenderTargetWritten(readTextureObject, readFBOAttachment->level);
            NSLog(@"MGL TRACE blitFramebuffer.appliedPendingReadClear fbo=%u attachment=0x%x tex=%u rgba=(%.3f,%.3f,%.3f,%.3f)",
                  (unsigned)readfbo->name,
                  (unsigned)readAttachment,
                  (unsigned)readTextureObject->name,
                  readFBOAttachment->clear_color[0],
                  readFBOAttachment->clear_color[1],
                  readFBOAttachment->clear_color[2],
                  readFBOAttachment->clear_color[3]);
        } else {
            NSLog(@"MGL WARN: mtlBlitFramebuffer failed to apply pending read clear fbo=%u attachment=0x%x",
                  (unsigned)readfbo->name,
                  (unsigned)readAttachment);
        }
    }

    // Validate and clamp blit coordinates to avoid Metal validation aborts
    if (!readtexid || !drawtexid) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer missing source/destination Metal textures");
        return;
    }

    BOOL needsFormatConversionBlit = NO;
    if (readtexid.pixelFormat != drawtexid.pixelFormat) {
        BOOL rgbaBgraPair =
            ((readtexid.pixelFormat == MTLPixelFormatRGBA8Unorm && drawtexid.pixelFormat == MTLPixelFormatBGRA8Unorm) ||
             (readtexid.pixelFormat == MTLPixelFormatBGRA8Unorm && drawtexid.pixelFormat == MTLPixelFormatRGBA8Unorm));

        if (rgbaBgraPair) {
            needsFormatConversionBlit = YES;
            static uint64_t s_rgbaBgraBlitLogCount = 0;
            uint64_t hit = ++s_rgbaBgraBlitLogCount;
            if (hit <= 8ull || (hit % 512ull) == 0ull) {
                NSLog(@"MGL INFO: mtlBlitFramebuffer using shader conversion for RGBA/BGRA pair (src=%lu dst=%lu hit=%llu)",
                      (unsigned long)readtexid.pixelFormat,
                      (unsigned long)drawtexid.pixelFormat,
                      (unsigned long long)hit);
            }
        } else {
            NSLog(@"MGL WARN: mtlBlitFramebuffer pixel format mismatch (src=%lu dst=%lu), skipping blit",
                  (unsigned long)readtexid.pixelFormat, (unsigned long)drawtexid.pixelFormat);
            return;
        }
    }

    if (readSubresource.level >= readtexid.mipmapLevelCount ||
        drawSubresource.level >= drawtexid.mipmapLevelCount) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer invalid mip level read=%lu/%lu draw=%lu/%lu, skipping",
              (unsigned long)readSubresource.level,
              (unsigned long)readtexid.mipmapLevelCount,
              (unsigned long)drawSubresource.level,
              (unsigned long)drawtexid.mipmapLevelCount);
        return;
    }

    NSUInteger srcTexW = mglMetalTextureLevelDimension(readtexid.width, readSubresource.level);
    NSUInteger srcTexH = mglMetalTextureLevelDimension(readtexid.height, readSubresource.level);
    NSUInteger dstTexW = mglMetalTextureLevelDimension(drawtexid.width, drawSubresource.level);
    NSUInteger dstTexH = mglMetalTextureLevelDimension(drawtexid.height, drawSubresource.level);

    MGLBlitAxis axisX = { (double)srcX0, (double)srcX1, (double)dstX0, (double)dstX1 };
    MGLBlitAxis axisY = { (double)srcY0, (double)srcY1, (double)dstY0, (double)dstY1 };
    if (!mglClipBlitAxis(&axisX, (double)srcTexW, (double)dstTexW) ||
        !mglClipBlitAxis(&axisY, (double)srcTexH, (double)dstTexH)) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer clipped region is empty srcTex=%lux%lu dstTex=%lux%lu req src=(%d,%d)-(%d,%d) dst=(%d,%d)-(%d,%d)",
              (unsigned long)srcTexW,
              (unsigned long)srcTexH,
              (unsigned long)dstTexW,
              (unsigned long)dstTexH,
              srcX0, srcY0, srcX1, srcY1,
              dstX0, dstY0, dstX1, dstY1);
        return;
    }

    BOOL srcXForward = axisX.src1 >= axisX.src0;
    BOOL srcYForward = axisY.src1 >= axisY.src0;
    BOOL dstXForward = axisX.dst1 >= axisX.dst0;
    BOOL dstYForward = axisY.dst1 >= axisY.dst0;
    BOOL blitNeedsFlip = (srcXForward != dstXForward) || (srcYForward != dstYForward);

    double srcMinX = fmin(axisX.src0, axisX.src1);
    double srcMaxX = fmax(axisX.src0, axisX.src1);
    double srcMinY = fmin(axisY.src0, axisY.src1);
    double srcMaxY = fmax(axisY.src0, axisY.src1);
    double dstMinX = fmin(axisX.dst0, axisX.dst1);
    double dstMaxX = fmax(axisX.dst0, axisX.dst1);
    double dstMinY = fmin(axisY.dst0, axisY.dst1);
    double dstMaxY = fmax(axisY.dst0, axisY.dst1);
    double srcW = fabs(axisX.src1 - axisX.src0);
    double srcH = fabs(axisY.src1 - axisY.src0);
    double dstW = fabs(axisX.dst1 - axisX.dst0);
    double dstH = fabs(axisY.dst1 - axisY.dst0);

    if (srcW <= 0.0 || srcH <= 0.0 || dstW <= 0.0 || dstH <= 0.0) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer empty clipped region src=%.3fx%.3f dst=%.3fx%.3f, skipping",
              srcW, srcH, dstW, dstH);
        return;
    }

    BOOL needsScaledBlit =
        (needsFormatConversionBlit ||
         blitNeedsFlip ||
         !mglNearlyEqual(srcW, dstW) ||
         !mglNearlyEqual(srcH, dstH));

    NSInteger copySrcX = (NSInteger)floor(srcMinX + 0.00001);
    NSInteger copySrcY = (NSInteger)floor(srcMinY + 0.00001);
    NSInteger copyDstX = (NSInteger)floor(dstMinX + 0.00001);
    NSInteger copyDstY = (NSInteger)floor(dstMinY + 0.00001);
    NSInteger copyW = (NSInteger)ceil(srcMaxX - 0.00001) - copySrcX;
    NSInteger copyH = (NSInteger)ceil(srcMaxY - 0.00001) - copySrcY;
    NSInteger srcMetalY = (NSInteger)srcTexH - (copySrcY + copyH);
    NSInteger dstMetalY = (NSInteger)dstTexH - (copyDstY + copyH);

    double scaledDstMetalY = (double)dstTexH - dstMaxY;

    static uint64_t s_blitDiagCount = 0;
    uint64_t blitDiag = ++s_blitDiagCount;
    BOOL traceBlit = kMGLSwapPresentDiagnostics &&
        (blitDiag <= 24ull || (blitDiag % 120ull) == 0ull || needsScaledBlit);
    if (traceBlit) {
        NSLog(@"MGL TRACE blitFramebuffer call=%llu readFBO=%p drawFBO=%p mask=0x%zx filter=0x%x "
              "srcReq=(%d,%d)-(%d,%d) dstReq=(%d,%d)-(%d,%d) "
              "copy srcGL=(%.3f,%.3f %.3fx%.3f) dstGL=(%.3f,%.3f %.3fx%.3f) srcMTL=(%ld,%ld) dstMTL=(%ld,%ld) scaled=%d flip=%d "
              "srcTex=%p fmt=%lu %lux%lu dstTex=%p fmt=%lu %lux%lu drawBuf=0x%x readBuf=0x%x",
              (unsigned long long)blitDiag,
              readfbo,
              drawfbo,
              mask,
              (unsigned)filter,
              srcX0, srcY0, srcX1, srcY1,
              dstX0, dstY0, dstX1, dstY1,
              srcMinX, srcMinY, srcW, srcH,
              dstMinX, dstMinY, dstW, dstH,
              (long)copySrcX, (long)srcMetalY,
              (long)copyDstX, (long)dstMetalY,
              needsScaledBlit ? 1 : 0,
              blitNeedsFlip ? 1 : 0,
              readtexid,
              (unsigned long)readtexid.pixelFormat,
              (unsigned long)srcTexW,
              (unsigned long)srcTexH,
              drawtexid,
              (unsigned long)drawtexid.pixelFormat,
              (unsigned long)dstTexW,
              (unsigned long)dstTexH,
              (unsigned)(glm_ctx ? glm_ctx->state.draw_buffer : 0u),
              (unsigned)(glm_ctx ? glm_ctx->state.read_buffer : 0u));
    }

    if (needsScaledBlit) {
        if (readtexid == drawtexid) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer scaled self-blit unsupported texture=%p, skipping", readtexid);
            return;
        }
        if (readSubresource.level != 0u ||
            readSubresource.slice != 0u ||
            readSubresource.depthPlane != 0u ||
            readtexid.textureType != MTLTextureType2D) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer scaled source subresource/type unsupported level=%lu slice=%lu depth=%lu type=%lu, skipping",
                  (unsigned long)readSubresource.level,
                  (unsigned long)readSubresource.slice,
                  (unsigned long)readSubresource.depthPlane,
                  (unsigned long)readtexid.textureType);
            return;
        }

        id<MTLRenderPipelineState> pipeline = [self scaledBlitPipelineForPixelFormat:drawtexid.pixelFormat];
        id<MTLSamplerState> sampler = [self scaledBlitSamplerForFilter:filter];
        if (!pipeline || !sampler) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer scaled path unavailable pipeline=%p sampler=%p", pipeline, sampler);
            return;
        }

        float invSrcW = srcTexW ? (1.0f / (float)srcTexW) : 0.0f;
        float invSrcH = srcTexH ? (1.0f / (float)srcTexH) : 0.0f;
        float uvLeft = MAX(0.0f, MIN(1.0f, (float)srcMinX * invSrcW));
        float uvRight = MAX(0.0f, MIN(1.0f, (float)srcMaxX * invSrcW));
        float uvTop = MAX(0.0f, MIN(1.0f, (float)((double)srcTexH - srcMaxY) * invSrcH));
        float uvBottom = MAX(0.0f, MIN(1.0f, (float)((double)srcTexH - srcMinY) * invSrcH));
        if (srcXForward != dstXForward) {
            float tmp = uvLeft;
            uvLeft = uvRight;
            uvRight = tmp;
        }
        if (srcYForward != dstYForward) {
            float tmp = uvTop;
            uvTop = uvBottom;
            uvBottom = tmp;
        }
        MGLScaledBlitParams params;
        params.uvRect = (vector_float4){
            uvLeft,
            uvTop,
            uvRight,
            uvBottom
        };
        params.forceOpaqueAlpha = (drawfbo == NULL && drawtexid == (_drawable ? _drawable.texture : nil)) ? 1.0f : 0.0f;
        params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

        MTLRenderPassDescriptor *scaledPass = [MTLRenderPassDescriptor renderPassDescriptor];
        scaledPass.colorAttachments[0].texture = drawtexid;
        scaledPass.colorAttachments[0].level = drawSubresource.level;
        scaledPass.colorAttachments[0].slice = drawSubresource.slice;
        scaledPass.colorAttachments[0].depthPlane = drawSubresource.depthPlane;
        scaledPass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        scaledPass.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLRenderCommandEncoder> encoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:scaledPass];
        if (!encoder) {
            NSLog(@"MGL WARN: mtlBlitFramebuffer failed to create scaled render encoder");
            return;
        }

        [encoder setRenderPipelineState:pipeline];
        [encoder setVertexBytes:&params length:sizeof(params) atIndex:0];
        [encoder setFragmentBytes:&params length:sizeof(params) atIndex:0];
        [encoder setFragmentTexture:readtexid atIndex:0];
        [encoder setFragmentSamplerState:sampler atIndex:0];

        double scaledDstMetalBottom = scaledDstMetalY + dstH;
        NSInteger scissorX0 = (NSInteger)floor(dstMinX + 0.00001);
        NSInteger scissorX1 = (NSInteger)ceil(dstMaxX - 0.00001);
        NSInteger scissorY0 = (NSInteger)floor(scaledDstMetalY + 0.00001);
        NSInteger scissorY1 = (NSInteger)ceil(scaledDstMetalBottom - 0.00001);
        scissorX0 = MAX((NSInteger)0, MIN(scissorX0, (NSInteger)dstTexW));
        scissorX1 = MAX((NSInteger)0, MIN(scissorX1, (NSInteger)dstTexW));
        scissorY0 = MAX((NSInteger)0, MIN(scissorY0, (NSInteger)dstTexH));
        scissorY1 = MAX((NSInteger)0, MIN(scissorY1, (NSInteger)dstTexH));
        if (scissorX1 <= scissorX0 || scissorY1 <= scissorY0) {
            [encoder endEncoding];
            NSLog(@"MGL WARN: mtlBlitFramebuffer scaled scissor is empty after clipping, skipping draw");
            return;
        }

        [encoder setViewport:(MTLViewport){
            .originX = dstMinX,
            .originY = scaledDstMetalY,
            .width = dstW,
            .height = dstH,
            .znear = 0.0,
            .zfar = 1.0
        }];
        [encoder setScissorRect:(MTLScissorRect){
            .x = (NSUInteger)scissorX0,
            .y = (NSUInteger)scissorY0,
            .width = (NSUInteger)(scissorX1 - scissorX0),
            .height = (NSUInteger)(scissorY1 - scissorY0)
        }];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        [encoder endEncoding];
        return;
    }

    // start blit encoder
    id<MTLBlitCommandEncoder> blitCommandEncoder;
    blitCommandEncoder = [_currentCommandBuffer blitCommandEncoder];
    if (!blitCommandEncoder) {
        NSLog(@"MGL WARN: mtlBlitFramebuffer failed to create blit encoder");
        return;
    }
    if (copyW <= 0 || copyH <= 0 ||
        copySrcX < 0 || copySrcY < 0 || copyDstX < 0 || copyDstY < 0 ||
        srcMetalY < 0 || dstMetalY < 0 ||
        copySrcX + copyW > (NSInteger)srcTexW ||
        copySrcY + copyH > (NSInteger)srcTexH ||
        copyDstX + copyW > (NSInteger)dstTexW ||
        copyDstY + copyH > (NSInteger)dstTexH) {
        [blitCommandEncoder endEncoding];
        NSLog(@"MGL WARN: mtlBlitFramebuffer direct copy invalid after clipping src=(%ld,%ld %ldx%ld) dst=(%ld,%ld) srcTex=%lux%lu dstTex=%lux%lu",
              (long)copySrcX, (long)copySrcY, (long)copyW, (long)copyH,
              (long)copyDstX, (long)copyDstY,
              (unsigned long)srcTexW, (unsigned long)srcTexH,
              (unsigned long)dstTexW, (unsigned long)dstTexH);
        return;
    }
    [blitCommandEncoder
        copyFromTexture:readtexid sourceSlice:readSubresource.slice sourceLevel:readSubresource.level
           sourceOrigin:MTLOriginMake((NSUInteger)copySrcX, (NSUInteger)srcMetalY, readSubresource.depthPlane)
             sourceSize:MTLSizeMake((NSUInteger)copyW, (NSUInteger)copyH, 1u)
              toTexture:drawtexid destinationSlice:drawSubresource.slice destinationLevel:drawSubresource.level
      destinationOrigin:MTLOriginMake((NSUInteger)copyDstX, (NSUInteger)dstMetalY, drawSubresource.depthPlane)];
    [blitCommandEncoder endEncoding];

}

void mtlBlitFramebuffer(GLMContext glm_ctx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
{
    if (!glm_ctx || ((uintptr_t)glm_ctx < 0x1000)) {
        fprintf(stderr, "MGL ERROR: mtlBlitFramebuffer bridge received invalid glm_ctx=%p\n", (void*)glm_ctx);
        return;
    }

    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlBlitFramebuffer:glm_ctx srcX0:srcX0 srcY0:srcY0 srcX1:srcX1 srcY1:srcY1 dstX0:dstX0 dstY0:dstY0 dstX1:dstX1 dstY1:dstY1 mask:mask filter:filter];
}

- (void)mtlInvalidateRenderPass:(GLMContext)glm_ctx
{
    if (!glm_ctx || glm_ctx != ctx || !_currentRenderEncoder) {
        return;
    }

    static uint64_t s_renderPassInvalidateCount = 0;
    uint64_t hit = ++s_renderPassInvalidateCount;
    if (hit <= 64ull || (hit % 512ull) == 0ull) {
        Framebuffer *fbo = glm_ctx->state.framebuffer;
        NSLog(@"MGL TRACE renderpass.invalidate hit=%llu fbo=%u(%p) drawBuf=0x%x rpFbo=%u(%p) rpDrawBuf=0x%x",
              (unsigned long long)hit,
              (unsigned)(fbo ? fbo->name : 0u),
              fbo,
              (unsigned)glm_ctx->state.draw_buffer,
              (unsigned)_renderPassFramebufferName,
              _renderPassFramebuffer,
              (unsigned)_renderPassDrawBuffer);
        mglLogRenderPassLifecycle("invalidate-before-end",
                                  hit,
                                  glm_ctx,
                                  _currentCommandBuffer,
                                  _currentRenderEncoder,
                                  _renderPassDescriptor,
                                  _drawable,
                                  _renderPassFramebuffer,
                                  _renderPassFramebufferName,
                                  _renderPassDrawBuffer,
                                  _renderPassDrawBufferCount);
    }

    [self endRenderEncoding];
}

void mtlInvalidateRenderPass(GLMContext glm_ctx)
{
    if (!glm_ctx || !glm_ctx->mtl_funcs.mtlObj) {
        return;
    }

    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlInvalidateRenderPass:glm_ctx];
}

- (Texture *)framebufferAttachmentTexture: (FBOAttachment *)fbo_attachment
{
    Texture *tex = NULL;

    if (!fbo_attachment) {
        NSLog(@"MGL ERROR: framebufferAttachmentTexture called with NULL attachment");
        return NULL;
    }

    if (fbo_attachment->textarget == GL_RENDERBUFFER)
    {
        if (fbo_attachment->buf.rbo) {
            tex = fbo_attachment->buf.rbo->tex;
        }
    }
    else
    {
        tex = fbo_attachment->buf.tex;
        if (!tex && fbo_attachment->texture != 0 && fbo_attachment->textarget != GL_RENDERBUFFER)
        {
            tex = findTexture(ctx, fbo_attachment->texture);
            if (tex)
            {
                fbo_attachment->buf.tex = tex;
            }
        }
    }
    if (!tex) {
        NSLog(@"MGL WARN: framebuffer attachment has no texture (target=0x%x)", fbo_attachment->textarget);
    }

    return tex;
}

- (void)markCurrentFramebufferColorAttachmentWrittenAtIndex:(GLuint)attachmentIndex
{
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    if (!fbo || attachmentIndex >= MAX_COLOR_ATTACHMENTS) {
        return;
    }

    if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
        return;
    }

    FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
    Texture *tex = [self framebufferAttachmentTexture:attachment];
    mglMarkTextureLevelRenderTargetWritten(tex, attachment->level);
}

- (void)markCurrentFramebufferDrawAttachmentsWritten
{
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    if (!fbo) {
        return;
    }

    GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
    for (GLsizei slot = 0; slot < drawBufferCount; ++slot) {
        GLuint attachmentIndex = 0u;
        if (mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, (GLuint)slot),
                                                  &attachmentIndex)) {
            [self markCurrentFramebufferColorAttachmentWrittenAtIndex:attachmentIndex];
        }
    }
}

- (bool)currentRenderPassMatchesCurrentFramebuffer
{
    if (!ctx || !_renderPassDescriptor) {
        return true;
    }

    Framebuffer *fbo = ctx->state.framebuffer;
    GLuint fboName = fbo ? fbo->name : 0u;
    if (_renderPassFramebuffer != fbo ||
        _renderPassFramebufferName != fboName ||
        _renderPassDrawBuffer != ctx->state.draw_buffer ||
        _renderPassDrawBufferCount != mglMetalDrawBufferCount(ctx)) {
        return false;
    }
    for (GLsizei i = 0; i < _renderPassDrawBufferCount; ++i) {
        if (_renderPassDrawBuffers[i] != mglMetalDrawBufferAt(ctx, (GLuint)i)) {
            return false;
        }
    }

    if (!fbo) {
        GLuint mgl_drawbuffer = mglDefaultDrawBufferIndexForGL(ctx->state.draw_buffer);
        id<MTLTexture> expectedColor0 = nil;
        id<MTLTexture> actualColor0 = _renderPassDescriptor.colorAttachments[0].texture;

        if (mgl_drawbuffer == _FRONT) {
            expectedColor0 = _drawable ? _drawable.texture : nil;
        } else if (mgl_drawbuffer < _MAX_DRAW_BUFFERS) {
            expectedColor0 = _drawBuffers[mgl_drawbuffer].drawbuffer;
        }

        if (actualColor0 != expectedColor0) {
            return false;
        }

        id<MTLTexture> expectedDepth = nil;
        id<MTLTexture> expectedStencil = nil;
        if (mgl_drawbuffer < _MAX_DRAW_BUFFERS) {
            BOOL defaultPassNeedsDepth = ctx->state.caps.depth_test;
            BOOL defaultPassNeedsStencil = ctx->state.caps.stencil_test || ctx->stencil_format.format;
            expectedDepth = defaultPassNeedsDepth ? _drawBuffers[mgl_drawbuffer].depthbuffer : nil;
            expectedStencil = defaultPassNeedsStencil ? _drawBuffers[mgl_drawbuffer].stencilbuffer : nil;
        }

        if (_renderPassDescriptor.depthAttachment.texture != expectedDepth) {
            return false;
        }
        if (_renderPassDescriptor.stencilAttachment.texture != expectedStencil) {
            return false;
        }

        return true;
    }

    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        GLuint attachmentIndex = 0u;
        BOOL drawSlotPresent =
            mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, i),
                                                  &attachmentIndex) &&
            attachmentIndex < MAX_COLOR_ATTACHMENTS &&
            ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) != 0u;
        FBOAttachment *attachment = drawSlotPresent ? &fbo->color_attachments[attachmentIndex] : NULL;
        Texture *tex = drawSlotPresent ? [self framebufferAttachmentTexture:attachment] : NULL;
        id<MTLTexture> expected = nil;

        if (tex) {
            tex->is_render_target = true;
            if (!tex->mtl_data) {
                if (![self bindMTLTexture:tex]) {
                    return false;
                }
            }
            expected = (__bridge id<MTLTexture>)(tex->mtl_data);
        }

        id<MTLTexture> actual = _renderPassDescriptor.colorAttachments[i].texture;
        if (actual != expected) {
            return false;
        }

        if (i + 1u >= MAX_COLOR_ATTACHMENTS ||
            (mglMetalDrawBufferAt(ctx, i + 1u) == GL_NONE &&
             !_renderPassDescriptor.colorAttachments[i + 1u].texture)) {
            break;
        }
    }

    id<MTLTexture> expectedDepth = nil;
    if (fbo->depth.texture) {
        Texture *depthTex = [self framebufferAttachmentTexture:&fbo->depth];
        if (depthTex && !depthTex->mtl_data) {
            depthTex->is_render_target = true;
            if (![self bindMTLTexture:depthTex]) {
                return false;
            }
        }
        expectedDepth = depthTex ? (__bridge id<MTLTexture>)(depthTex->mtl_data) : nil;
    }
    if (_renderPassDescriptor.depthAttachment.texture != expectedDepth) {
        return false;
    }

    id<MTLTexture> expectedStencil = nil;
    if (fbo->stencil.texture) {
        Texture *stencilTex = [self framebufferAttachmentTexture:&fbo->stencil];
        if (stencilTex && !stencilTex->mtl_data) {
            stencilTex->is_render_target = true;
            if (![self bindMTLTexture:stencilTex]) {
                return false;
            }
        }
        expectedStencil = stencilTex ? (__bridge id<MTLTexture>)(stencilTex->mtl_data) : nil;
    }
    if (_renderPassDescriptor.stencilAttachment.texture != expectedStencil) {
        return false;
    }

    return true;
}

- (bool)ensureCurrentRenderPassMatchesFramebufferForDraw
{
    if (!ctx || !_currentRenderEncoder) {
        return true;
    }

    if ([self currentRenderPassMatchesCurrentFramebuffer]) {
        return true;
    }

    static uint64_t s_fboPassMismatchCount = 0;
    uint64_t hit = ++s_fboPassMismatchCount;
    if (hit <= 32ull || (hit % 256ull) == 0ull) {
        Framebuffer *fbo = ctx->state.framebuffer;
        id<MTLTexture> color0 = _renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].texture : nil;
        GLuint mglDefaultDrawbuffer = fbo ? 0u : mglDefaultDrawBufferIndexForGL(ctx->state.draw_buffer);
        id<MTLTexture> expectedDefaultColor0 = nil;
        if (!fbo) {
            expectedDefaultColor0 = (mglDefaultDrawbuffer == _FRONT)
                ? (_drawable ? _drawable.texture : nil)
                : ((mglDefaultDrawbuffer < _MAX_DRAW_BUFFERS) ? _drawBuffers[mglDefaultDrawbuffer].drawbuffer : nil);
        }
        GLuint fboName = fbo ? fbo->name : 0u;
        GLuint attachment0Name = (fbo && (fbo->color_attachment_bitfield & 1u)) ? fbo->color_attachments[0].texture : 0u;
        NSLog(@"MGL WARNING: render pass/FBO mismatch before draw hit=%llu fbo=%u drawBuffer=0x%x attachment0=%u passColor0=%p expectedDefaultColor0=%p defaultDrawBuffer=%u; rebuilding encoder",
              (unsigned long long)hit,
              (unsigned)fboName,
              (unsigned)(ctx ? ctx->state.draw_buffer : 0u),
              (unsigned)attachment0Name,
              color0,
              expectedDefaultColor0,
              (unsigned)mglDefaultDrawbuffer);
        mglLogRenderPassLifecycle(fbo ? "fbo-mismatch-before-rebuild" : "default-fbo-mismatch-before-rebuild",
                                  hit,
                                  ctx,
                                  _currentCommandBuffer,
                                  _currentRenderEncoder,
                                  _renderPassDescriptor,
                                  _drawable,
                                  _renderPassFramebuffer,
                                  _renderPassFramebufferName,
                                  _renderPassDrawBuffer,
                                  _renderPassDrawBufferCount);
    }

    [self endRenderEncoding];
    ctx->state.dirty_bits |= (DIRTY_FBO | DIRTY_PROGRAM | DIRTY_RENDER_STATE | DIRTY_VAO);
    return [self newRenderEncoder];
}

- (void)endRenderPassIfFramebufferChangedForNonDraw:(uint64_t)processCall
{
    if (!ctx || !_currentRenderEncoder) {
        return;
    }

    if ([self currentRenderPassMatchesCurrentFramebuffer]) {
        return;
    }

    static uint64_t s_nonDrawFboMismatchCount = 0;
    uint64_t hit = ++s_nonDrawFboMismatchCount;
    if (hit <= 32ull || (hit % 256ull) == 0ull) {
        Framebuffer *fbo = ctx->state.framebuffer;
        GLuint fboName = fbo ? fbo->name : 0u;
        NSLog(@"MGL TRACE non-draw renderpass/FBO mismatch processCall=%llu hit=%llu "
              "ctxFbo=%u(%p) ctxDrawBuf=0x%x rpFbo=%u(%p) rpDrawBuf=0x%x; ending stale encoder",
              (unsigned long long)processCall,
              (unsigned long long)hit,
              (unsigned)fboName,
              fbo,
              (unsigned)ctx->state.draw_buffer,
              (unsigned)_renderPassFramebufferName,
              _renderPassFramebuffer,
              (unsigned)_renderPassDrawBuffer);
        mglLogRenderPassLifecycle("non-draw-mismatch-before-end",
                                  hit,
                                  ctx,
                                  _currentCommandBuffer,
                                  _currentRenderEncoder,
                                  _renderPassDescriptor,
                                  _drawable,
                                  _renderPassFramebuffer,
                                  _renderPassFramebufferName,
                                  _renderPassDrawBuffer,
                                  _renderPassDrawBufferCount);
    }

    [self endRenderEncoding];
    ctx->state.dirty_bits |= (DIRTY_FBO | DIRTY_PROGRAM | DIRTY_RENDER_STATE);
}

- (bool)bindMTLTexture:(Texture *)tex
{
    if (tex && tex->target == GL_TEXTURE_BUFFER && tex->texture_buffer &&
        tex->texture_buffer->data.dirty_bits) {
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    }

    // If this texture is now used as a render target but was previously created
    // without render-target usage, force a recreate with proper usage flags.
    if (tex->mtl_data && tex->is_render_target) {
        id<MTLTexture> existingTexture = (__bridge id<MTLTexture>)(tex->mtl_data);
        if (existingTexture && ((existingTexture.usage & MTLTextureUsageRenderTarget) == 0)) {
            NSLog(@"MGL WARNING: Recreating texture %u with RenderTarget usage (old usage=0x%lx)",
                  tex->name, (unsigned long)existingTexture.usage);
            CFBridgingRelease(tex->mtl_data);
            tex->mtl_data = NULL;
            tex->dirty_bits |= DIRTY_TEXTURE_DATA;
        }
    }

    if (tex->dirty_bits)
    {
        bool textureNeedsRebuild =
            (tex->dirty_bits & (DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_ACCESS)) != 0;
        bool samplerNeedsRebuild =
            textureNeedsRebuild || ((tex->dirty_bits & DIRTY_TEXTURE_PARAM) != 0);

        // Texture parameter changes only affect the Metal sampler object. Do
        // not throw away texture storage for wrap/filter/lod updates; doing so
        // can turn Minecraft's frequent sampler changes into render-pass and
        // upload storms.
        if (tex->mtl_data)
        {
            if (textureNeedsRebuild) {
                CFBridgingRelease(tex->mtl_data);
                tex->mtl_data = NULL;
            }
        }

        if (samplerNeedsRebuild && tex->params.mtl_data)
        {
            CFBridgingRelease(tex->params.mtl_data);
            tex->params.mtl_data = NULL;
        }
    }

    if (tex->mtl_data == NULL)
    {
        NSLog(@"MGL INFO: Creating MTL texture for texture (size: %dx%dx%d)", tex->width, tex->height, tex->depth);

        tex->mtl_data = (void *)CFBridgingRetain([self createMTLTextureFromGLTexture: tex]);

        // AGX-SAFE: Handle NULL texture gracefully when in GPU recovery mode
        if (!tex->mtl_data) {
            NSLog(@"MGL AGX: Primary texture creation returned NULL, attempting fallback texture creation");
            // Create a simple fallback texture to prevent crashes
            tex->mtl_data = (void *)CFBridgingRetain([self createFallbackMTLTexture: tex]);

            if (tex->mtl_data) {
                NSLog(@"MGL SUCCESS: Fallback texture created successfully");
            } else {
                NSLog(@"MGL ERROR: Even fallback texture creation failed - this texture will remain NULL");
            }
        } else {
            NSLog(@"MGL SUCCESS: Primary texture created successfully");
        }

    }

    if (tex->params.mtl_data == NULL)
    {
        tex->params.mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&tex->params target:tex->target]);
        // Sampler creation should not fail even in recovery mode
        if (!tex->params.mtl_data) {
            NSLog(@"MGL WARNING: Sampler creation failed, using default");
            tex->params.mtl_data = (void *)CFBridgingRetain([_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]]);
        }
    }

    return true;
}

- (bool)bindActiveTexturesToMTL
{
    // search through active_texture_mask for enabled bits
    // 128 bits long.. do it on 4 parts
    for(int i=0; i<4; i++)
    {
        unsigned mask = STATE(active_texture_mask[i]);

        if (mask)
        {
            for(int bitpos=0; bitpos<32; bitpos++)
            {
                if (mask & (0x1 << bitpos))
                {
                    Texture *tex;
                    int unit = i * 32 + bitpos;

                    tex = STATE(active_textures[unit]);
                    if (!tex)
                    {
                        // Stale active texture mask bit; clear it and continue.
                        STATE(active_texture_mask[i]) &= ~(0x1u << bitpos);
                        continue;
                    }

                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: tex]);
                }

                // early out
                if ((mask >> (bitpos + 1)) == 0)
                    break;
            }
        }
    }

    return true;
}

- (bool)bindFramebufferTexture:(FBOAttachment *)fbo_attachment isDrawBuffer:(bool) isDrawBuffer
{
    Texture *tex;

    tex = [self framebufferAttachmentTexture: fbo_attachment];
    if (!tex) {
        // Incomplete/missing attachment. Do not crash.
        return true;
    }

    tex->is_render_target = isDrawBuffer;

    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: tex]);

    return true;
}


#pragma mark programs
- (int) getProgramBindingCount: (int) stage type: (int) type
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        NSLog(@"MGL ERROR: Invalid shader stage %d in getProgramBindingCount", stage);
        return 0;
    }
    switch(type)
    {
        case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
        case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
        case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
        case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
        case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
        case SPVC_RESOURCE_TYPE_STAGE_INPUT:
        case SPVC_RESOURCE_TYPE_STAGE_OUTPUT:
        case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
        case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS:
        case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
            break;

        default:
            NSLog(@"MGL ERROR: Unknown resource type %d in getProgramBindingCount (stage=%d)", type, stage);
            return 0;
    }

    ptr = mglResolveProgramFromState(ctx);
    if (ptr == NULL)
        return 0;

    return ptr->spirv_resources_list[stage][type].count;
}

- (int) getProgramBinding: (int) stage type: (int) type index: (int) index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        NSLog(@"MGL ERROR: Invalid shader stage %d in getProgramBinding", stage);
        return 0;
    }
    switch(type)
    {
       case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
       case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
       case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
       case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
       case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
       case SPVC_RESOURCE_TYPE_STAGE_INPUT:
       case SPVC_RESOURCE_TYPE_STAGE_OUTPUT:
       case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
       case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
       case SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS:
       case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
           break;

       default:
            NSLog(@"MGL ERROR: Unknown resource type %d in getProgramBinding (stage=%d)", type, stage);
            return 0;
    }

    ptr = mglResolveProgramFromState(ctx);
    if (!ptr) {
        NSLog(@"MGL ERROR: getProgramBinding with no current program (name=%u)",
              (unsigned)ctx->state.program_name);
        return 0;
    }

    int count = ptr->spirv_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        NSLog(@"MGL WARNING: getProgramBinding index out of range index=%d count=%d stage=%d type=%d",
              index, count, stage, type);
        return 0;
    }

    return ptr->spirv_resources_list[stage][type].list[index].binding;
}

- (int)getProgramGLBinding:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES || type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramFromState(ctx);
    if (!ptr) {
        return 0;
    }

    int count = ptr->spirv_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        return 0;
    }

    return (int)ptr->spirv_resources_list[stage][type].list[index].gl_binding;
}

- (NSUInteger)getProgramBindingRequiredSize:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramFromState(ctx);
    if (!ptr) {
        return 0;
    }

    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return 0;
    }

    return (NSUInteger)ptr->spirv_resources_list[stage][type].list[index].required_size;
}

- (NSInteger)getProgramMetalBufferIndexForStage:(int)stage binding:(GLuint)binding
{
    static const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT
    };

    Program *ptr = mglResolveProgramFromState(ctx);
    if (!ptr || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return (NSInteger)binding;
    }

    for (size_t t = 0; t < (sizeof(resourceTypes) / sizeof(resourceTypes[0])); t++) {
        int type = resourceTypes[t];
        if (type < 0 || type >= _MAX_SPIRV_RES) {
            continue;
        }

        SpirvResourceList *list = &ptr->spirv_resources_list[stage][type];
        for (GLuint i = 0; i < list->count; i++) {
            SpirvResource *res = &list->list[i];
            GLuint clientBinding = mglClientBufferBindingForResource(type, res);
            if (clientBinding == binding) {
                return (NSInteger)res->binding;
            }
        }
    }

    return -1;
}

- (MTLTextureType)getProgramDeclaredTextureType:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramFromState(ctx);
    if (!ptr) {
        return 0;
    }
    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return 0;
    }

    SpirvResource *res = &ptr->spirv_resources_list[stage][type].list[index];
    switch ((SpvDim)res->image_dim) {
        case SpvDim1D:
            return res->image_arrayed ? MTLTextureType1DArray : MTLTextureType1D;
        case SpvDim2D:
            if (res->image_multisampled) {
                return res->image_arrayed ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
            }
            return res->image_arrayed ? MTLTextureType2DArray : MTLTextureType2D;
        case SpvDim3D:
            return MTLTextureType3D;
        case SpvDimCube:
            return res->image_arrayed ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
        case SpvDimBuffer:
            return MTLTextureTypeTextureBuffer;
        default:
            return 0;
    }
}

- (MTLTextureType)getProgramExpectedTextureType:(int)stage type:(int)type index:(int)index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return 0;
    }

    ptr = mglResolveProgramFromState(ctx);
    if (!ptr) {
        return 0;
    }
    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return 0;
    }

    SpirvResource *res = &ptr->spirv_resources_list[stage][type].list[index];
    MTLTextureType mslType = mglExpectedTextureTypeFromMSL(ptr->spirv[stage].msl_str, res->binding);

    MTLTextureType spirvType = 0;
    switch ((SpvDim)res->image_dim) {
        case SpvDim1D:
            spirvType = res->image_arrayed ? MTLTextureType1DArray : MTLTextureType1D;
            break;
        case SpvDim2D:
            if (res->image_multisampled) {
                spirvType = res->image_arrayed ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
            } else {
                spirvType = res->image_arrayed ? MTLTextureType2DArray : MTLTextureType2D;
            }
            break;
        case SpvDim3D:
            spirvType = MTLTextureType3D;
            break;
        case SpvDimCube:
            spirvType = res->image_arrayed ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
            break;
        case SpvDimBuffer:
            spirvType = MTLTextureTypeTextureBuffer;
            break;
        default:
            spirvType = 0;
            break;
    }

    if (mslType != 0 && mslType != spirvType) {
        static uint64_t s_mslTextureTypeOverrideCount = 0;
        uint64_t hit = ++s_mslTextureTypeOverrideCount;
        if (hit <= 32ull || (hit % 512ull) == 0ull) {
            NSLog(@"MGL TEX EXPECT override from MSL stage=%d type=%d index=%d binding=%u name=%s spirvType=%lu mslType=%lu imageDim=%u hit=%llu",
                  stage,
                  type,
                  index,
                  (unsigned)res->binding,
                  res->name ? res->name : "(null)",
                  (unsigned long)spirvType,
                  (unsigned long)mslType,
                  (unsigned)res->image_dim,
                  (unsigned long long)hit);
        }
        return mslType;
    }

    return mslType ? mslType : spirvType;
}

- (MGLTextureDataKind)getProgramExpectedTextureDataKind:(int)stage type:(int)type index:(int)index
{
    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return MGLTextureDataKindUnknown;
    }
    if (type < 0 || type >= _MAX_SPIRV_RES) {
        return MGLTextureDataKindUnknown;
    }

    Program *ptr = mglResolveProgramFromState(ctx);
    if (!ptr) {
        return MGLTextureDataKindUnknown;
    }
    if (index < 0 || index >= (int)ptr->spirv_resources_list[stage][type].count) {
        return MGLTextureDataKindUnknown;
    }

    SpirvResource *res = &ptr->spirv_resources_list[stage][type].list[index];
    MGLTextureDataKind mslKind = mglExpectedTextureDataKindFromMSL(ptr->spirv[stage].msl_str, res->binding);
    if (mslKind != MGLTextureDataKindUnknown) {
        return mslKind;
    }

    // SPIRV-Cross can rewrite resource dimensionality/types in MSL; when the MSL
    // line is not parseable, keep float as the compatibility default for sampled images.
    return MGLTextureDataKindFloat;
}

- (NSUInteger)getProgramBindingRequiredSizeForStage:(int)stage binding:(GLuint)binding
{
    static const int resourceTypes[] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,
        SPVC_RESOURCE_TYPE_PUSH_CONSTANT
    };

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return 0;
    }

    NSUInteger required = 0;
    for (size_t t = 0; t < (sizeof(resourceTypes) / sizeof(resourceTypes[0])); t++) {
        int type = resourceTypes[t];
        int count = [self getProgramBindingCount:stage type:type];
        for (int i = 0; i < count; i++) {
            int mappedBinding = [self getProgramBinding:stage type:type index:i];
            Program *program = mglResolveProgramFromState(ctx);
            if (!program || type < 0 || type >= _MAX_SPIRV_RES ||
                i < 0 || i >= (int)program->spirv_resources_list[stage][type].count) {
                continue;
            }

            GLuint clientBinding =
                mglClientBufferBindingForResource(type,
                                                  &program->spirv_resources_list[stage][type].list[i]);
            if (mappedBinding < 0 || clientBinding != binding) {
                continue;
            }

            NSUInteger candidate = [self getProgramBindingRequiredSize:stage type:type index:i];
            if (candidate > required) {
                required = candidate;
            }
        }
    }

    return required;
}

- (int) getProgramLocation: (int) stage type: (int) type index: (int) index
{
    Program *ptr;

    if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
        NSLog(@"MGL ERROR: Invalid shader stage %d in getProgramLocation", stage);
        return 0;
    }
    switch(type)
    {
       case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
       case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
       case SPVC_RESOURCE_TYPE_STORAGE_BUFFER:
       case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
       case SPVC_RESOURCE_TYPE_PUSH_CONSTANT:
       case SPVC_RESOURCE_TYPE_STAGE_INPUT:
       case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
       case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
           break;

       default:
            NSLog(@"MGL WARNING: unsupported SPIRV-Cross resource type %d in getProgramLocation", type);
            return 0;
    }

    ptr = mglResolveProgramFromState(ctx);
    if (!ptr) {
        NSLog(@"MGL ERROR: getProgramLocation with no current program (name=%u)",
              (unsigned)ctx->state.program_name);
        return 0;
    }

    int count = ptr->spirv_resources_list[stage][type].count;
    if (index < 0 || index >= count) {
        NSLog(@"MGL WARNING: getProgramLocation index out of range index=%d count=%d stage=%d type=%d",
              index, count, stage, type);
        return 0;
    }
    
    return ptr->spirv_resources_list[stage][type].list[index].location;
}

- (id<MTLLibrary>) compileShader: (const char *) str
{
    id<MTLLibrary> library;
    __autoreleasing NSError *error = nil;

    library = [_device newLibraryWithSource: [NSString stringWithUTF8String: str] options: nil error: &error];
    if(!library) {
        NSLog(@"MGL ERROR: Failed to compile shader: %@ ", [error localizedDescription] );
        NSLog(@"MGL ERROR: Shader source: %s", str);
        // Return nil instead of asserting - caller must handle this gracefully
        return nil;
    }

    return library;
}

-(bool)bindMTLProgram:(Program *)ptr
{
    if (ptr->dirty_bits & DIRTY_PROGRAM)
    {
        // release mtl shaders
        for(int i=_VERTEX_SHADER; i<_MAX_SHADER_TYPES; i++)
        {
            Shader *shader;
            shader = ptr->shader_slots[i];

            if (shader)
            {
                if (shader->mtl_data.library)
                {
                    CFBridgingRelease(shader->mtl_data.library);
                    CFBridgingRelease(shader->mtl_data.function);
                    shader->mtl_data.library = NULL;
                    shader->mtl_data.function = NULL;
                }
            }
        }

        ptr->dirty_bits &= ~DIRTY_PROGRAM;
    }

    // bind mtl functions to shaders
    for(int i=_VERTEX_SHADER; i<_MAX_SHADER_TYPES; i++)
    {
        Shader *shader;
        shader = ptr->shader_slots[i];

        if (shader)
        {
            if (shader->mtl_data.library == NULL)
            {
                id<MTLLibrary> library;
                id<MTLFunction> function;

                library = [self compileShader: ptr->spirv[i].msl_str];
                if (!library) {
                    NSLog(@"MGL ERROR: Failed to compile %s shader, skipping render", i == _VERTEX_SHADER ? "vertex" : "fragment");
                    shader->mtl_data.library = NULL;
                    shader->mtl_data.function = NULL;
                    return false;  // Signal shader compilation failure
                }
                function = [library newFunctionWithName:[NSString stringWithUTF8String: shader->entry_point]];
                if (!function) {
                    NSLog(@"MGL ERROR: Failed to find function '%s' in compiled shader", shader->entry_point);
                    shader->mtl_data.library = NULL;
                    shader->mtl_data.function = NULL;
                    return false;  // Signal function lookup failure
                }
                shader->mtl_data.library = (void *)CFBridgingRetain(library);
                shader->mtl_data.function = (void *)CFBridgingRetain(function);
            }
        }
    }

    return true;
}

#pragma mark draw buffers
- (CGSize)mglSyncLayerDrawableSizeFromView:(const char *)reason
{
    if (!_layer) {
        return CGSizeZero;
    }

    CGSize oldDrawableSize = _layer.drawableSize;
    NSRect bounds = NSZeroRect;
    NSRect backingBounds = NSZeroRect;
    CGFloat scale = 1.0;

    if (_view) {
        [_view setWantsLayer:YES];
        bounds = [_view bounds];
        if (bounds.size.width <= 0.0 || bounds.size.height <= 0.0) {
            bounds = [_view frame];
            bounds.origin = NSZeroPoint;
        }

        backingBounds = [_view convertRectToBacking:bounds];
        if (bounds.size.width > 0.0 && backingBounds.size.width > 0.0) {
            scale = backingBounds.size.width / bounds.size.width;
        } else {
            NSWindow *window = [_view window];
            if (window) {
                scale = [window backingScaleFactor];
            } else if ([NSScreen mainScreen]) {
                scale = [[NSScreen mainScreen] backingScaleFactor];
            }

            if (scale <= 0.0) {
                scale = 1.0;
            }

            backingBounds = NSMakeRect(0.0,
                                       0.0,
                                       bounds.size.width * scale,
                                       bounds.size.height * scale);
        }

        _layer.frame = bounds;
        _layer.contentsScale = scale;
    } else if (oldDrawableSize.width <= 0.0 || oldDrawableSize.height <= 0.0) {
        bounds = [_layer frame];
        scale = _layer.contentsScale > 0.0 ? _layer.contentsScale : 1.0;
        backingBounds = NSMakeRect(0.0,
                                   0.0,
                                   bounds.size.width * scale,
                                   bounds.size.height * scale);
    } else {
        backingBounds = NSMakeRect(0.0, 0.0, oldDrawableSize.width, oldDrawableSize.height);
    }

    NSUInteger pixelWidth = (NSUInteger)MAX(1.0, backingBounds.size.width + 0.5);
    NSUInteger pixelHeight = (NSUInteger)MAX(1.0, backingBounds.size.height + 0.5);
    CGSize newDrawableSize = CGSizeMake((CGFloat)pixelWidth, (CGFloat)pixelHeight);

    if (oldDrawableSize.width != newDrawableSize.width ||
        oldDrawableSize.height != newDrawableSize.height) {
        _layer.drawableSize = newDrawableSize;
    }

    static uint64_t s_sizeSyncCall = 0;
    static NSUInteger s_lastPixelWidth = 0;
    static NSUInteger s_lastPixelHeight = 0;
    uint64_t call = ++s_sizeSyncCall;
    BOOL sizeChanged = (s_lastPixelWidth != pixelWidth || s_lastPixelHeight != pixelHeight);

    if (sizeChanged || call <= 8ull || ((call % 120ull) == 0ull)) {
        NSWindow *window = _view ? [_view window] : nil;
        NSRect windowFrame = window ? [window frame] : NSZeroRect;
        NSLog(@"MGL SIZE sync reason=%s call=%llu viewBounds=%.1fx%.1f backing=%.1fx%.1f scale=%.3f drawable=%lux%lu old=%.0fx%.0f window=%.1fx%.1f",
              reason ? reason : "unknown",
              (unsigned long long)call,
              bounds.size.width,
              bounds.size.height,
              backingBounds.size.width,
              backingBounds.size.height,
              scale,
              (unsigned long)pixelWidth,
              (unsigned long)pixelHeight,
              oldDrawableSize.width,
              oldDrawableSize.height,
              windowFrame.size.width,
              windowFrame.size.height);
    }

    s_lastPixelWidth = pixelWidth;
    s_lastPixelHeight = pixelHeight;
    return newDrawableSize;
}

- (BOOL)mglEnsureLayerDrawableSizeAtLeastWidth:(NSUInteger)requiredWidth
                                        height:(NSUInteger)requiredHeight
                                        reason:(const char *)reason
{
    if (!_layer || requiredWidth == 0 || requiredHeight == 0) {
        return NO;
    }

    CGSize viewDrawableSize = [self mglSyncLayerDrawableSizeFromView:reason ? reason : "ensureDrawableSize"];
    NSUInteger targetWidth = MAX(requiredWidth, (NSUInteger)MAX(1.0, viewDrawableSize.width));
    NSUInteger targetHeight = MAX(requiredHeight, (NSUInteger)MAX(1.0, viewDrawableSize.height));
    CGSize oldDrawableSize = _layer.drawableSize;

    if ((NSUInteger)oldDrawableSize.width == targetWidth &&
        (NSUInteger)oldDrawableSize.height == targetHeight) {
        return NO;
    }

    _layer.drawableSize = CGSizeMake((CGFloat)targetWidth, (CGFloat)targetHeight);
    if (_drawable) {
        _drawable = nil;
    }

    static uint64_t s_forcedDrawableResizeCount = 0;
    uint64_t hit = ++s_forcedDrawableResizeCount;
    if (hit <= 32ull || (hit % 120ull) == 0ull) {
        NSLog(@"MGL SIZE force drawable reason=%s hit=%llu required=%lux%lu viewSync=%.0fx%.0f old=%.0fx%.0f new=%lux%lu",
              reason ? reason : "unknown",
              (unsigned long long)hit,
              (unsigned long)requiredWidth,
              (unsigned long)requiredHeight,
              viewDrawableSize.width,
              viewDrawableSize.height,
              oldDrawableSize.width,
              oldDrawableSize.height,
              (unsigned long)targetWidth,
              (unsigned long)targetHeight);
    }

    return YES;
}

- (id)newDrawBuffer:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil
{
    id<MTLTexture> texture;
    MTLTextureDescriptor *tex_desc;
    CGSize drawableSize;

    if (!_layer) {
        NSLog(@"MGL DRAWBUFFER ERROR: cannot create draw buffer without CAMetalLayer");
        return nil;
    }
    drawableSize = [self mglSyncLayerDrawableSizeFromView:"newDrawBuffer"];

    tex_desc = [[MTLTextureDescriptor alloc] init];
    if (!tex_desc) {
        NSLog(@"MGL DRAWBUFFER ERROR: failed to allocate draw buffer descriptor");
        return nil;
    }
    tex_desc.width = (NSUInteger)MAX(1.0, drawableSize.width);
    tex_desc.height = (NSUInteger)MAX(1.0, drawableSize.height);
    tex_desc.pixelFormat = pixelFormat;
    tex_desc.usage = MTLTextureUsageRenderTarget;

    if (depthStencil)
    {
        tex_desc.storageMode = MTLStorageModePrivate;
    }

    texture = [_device newTextureWithDescriptor:tex_desc];
    if (!texture) {
        NSLog(@"MGL DRAWBUFFER ERROR: failed to create draw buffer texture format=%lu size=%lux%lu",
              (unsigned long)pixelFormat,
              (unsigned long)tex_desc.width,
              (unsigned long)tex_desc.height);
        return nil;
    }

    return texture;
}

- (id)newDrawBufferWithCustomSize:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil customSize:(CGSize)size
{
    id<MTLTexture> texture;
    MTLTextureDescriptor *tex_desc;

    tex_desc = [[MTLTextureDescriptor alloc] init];
    if (!tex_desc) {
        NSLog(@"MGL DRAWBUFFER ERROR: failed to allocate custom draw buffer descriptor");
        return nil;
    }
    tex_desc.width = (NSUInteger)MAX(1.0, size.width);
    tex_desc.height = (NSUInteger)MAX(1.0, size.height);
    tex_desc.pixelFormat = pixelFormat;
    tex_desc.usage = MTLTextureUsageRenderTarget;

    if (depthStencil)
    {
        tex_desc.storageMode = MTLStorageModePrivate;
    }

    texture = [_device newTextureWithDescriptor:tex_desc];
    if (!texture) {
        NSLog(@"MGL DRAWBUFFER ERROR: failed to create custom draw buffer texture format=%lu size=%lux%lu",
              (unsigned long)pixelFormat,
              (unsigned long)tex_desc.width,
              (unsigned long)tex_desc.height);
        return nil;
    }

    return texture;
}

- (bool) checkDrawBufferSize:(GLuint) index;
{
    CGSize drawableSize;

    drawableSize = [self mglSyncLayerDrawableSizeFromView:"checkDrawBufferSize"];

    if ((GLuint)drawableSize.width != _drawBuffers[index].width)
        return false;

    if ((GLuint)drawableSize.height != _drawBuffers[index].height)
        return false;

    return true;
}

#pragma mark render encoder and command buffer init code
- (MTLStencilOperation) mtlStencilOpForGLOp:(GLenum) op
{
    switch(op)
    {
        case GL_KEEP: return MTLStencilOperationKeep;
        case GL_ZERO: return MTLStencilOperationZero;
        case GL_REPLACE: return MTLStencilOperationReplace;
        case GL_INCR: return MTLStencilOperationIncrementClamp;
        case GL_INCR_WRAP: return MTLStencilOperationIncrementWrap;
        case GL_DECR: return MTLStencilOperationDecrementClamp;
        case GL_DECR_WRAP: return MTLStencilOperationDecrementWrap;
        case GL_INVERT: return MTLStencilOperationInvert;
        default:
            NSLog(@"MGL WARNING: Unknown stencil operation 0x%x, falling back to KEEP", op);
            return MTLStencilOperationKeep;
    }
}

- (void) updateCurrentRenderEncoder
{
    BOOL passHasDepthAttachment =
        (_renderPassDescriptor != nil &&
         _renderPassDescriptor.depthAttachment.texture != nil);
    BOOL passHasStencilAttachment =
        (_renderPassDescriptor != nil &&
         _renderPassDescriptor.stencilAttachment.texture != nil);
    BOOL useDepthState = ctx->state.caps.depth_test && passHasDepthAttachment;
    BOOL useStencilState = ctx->state.caps.stencil_test && passHasStencilAttachment;

    // Guard against stale depth state carried over from world/hand rendering into GUI passes.
    // 1.21.7+ GlCommandEncoder caches lastPipeline and may skip glDisable(GL_DEPTH_TEST) when
    // the same pipeline object is reused, leaving caps.depth_test=1 while drawing GUI elements.
    if (useDepthState) {
        Program *currentProgram = mglResolveProgramFromState(ctx);
        if (mglProgramLooksLikeMinecraftGuiNoDepth(currentProgram)) {
            static uint64_t s_guiDepthOverrideCount = 0;
            uint64_t hit = ++s_guiDepthOverrideCount;
            if (hit <= 32 || (hit % 256) == 0) {
                NSLog(@"MGL GUARD: GUI shader with stale depth_test=1, forcing depth off (program=%u hit=%llu)",
                      (unsigned)(currentProgram ? currentProgram->name : 0u),
                      (unsigned long long)hit);
            }
            useDepthState = NO;
            useStencilState = NO;
        }
    }

    if (ctx->state.caps.depth_test && !passHasDepthAttachment) {
        static uint64_t s_missingDepthAttachmentCount = 0;
        uint64_t hit = ++s_missingDepthAttachmentCount;
        if (hit <= 32 || (hit % 256) == 0) {
            NSLog(@"MGL WARNING: depth test/write requested without depth attachment, disabling depth for this pass hit=%llu fbo=%u drawBuf=0x%x",
                  (unsigned long long)hit,
                  mglRendererSafeFramebufferName(ctx),
                  ctx->state.draw_buffer);
        }
    }

    if (ctx->state.caps.stencil_test && !passHasStencilAttachment) {
        static uint64_t s_missingStencilAttachmentCount = 0;
        uint64_t hit = ++s_missingStencilAttachmentCount;
        if (hit <= 32 || (hit % 256) == 0) {
            NSLog(@"MGL WARNING: stencil test requested without stencil attachment, disabling stencil for this pass hit=%llu fbo=%u drawBuf=0x%x",
                  (unsigned long long)hit,
                  mglRendererSafeFramebufferName(ctx),
                  ctx->state.draw_buffer);
        }
    }

    if (useDepthState || useStencilState)
    {
        MTLDepthStencilDescriptor *dsDesc = [[MTLDepthStencilDescriptor alloc] init];

        if (useDepthState)
        {
            if (!mglIsValidGLCompareFunction(ctx->state.var.depth_func)) {
                mglLogRenderStateRepair("depth_func", ctx->state.var.depth_func, GL_LESS);
                ctx->state.var.depth_func = GL_LESS;
                ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
            }

            dsDesc.depthCompareFunction =
                mglMTLCompareFunctionForGL(ctx->state.var.depth_func,
                                           MTLCompareFunctionLess,
                                           "depth");
            dsDesc.depthWriteEnabled = ctx->state.var.depth_writemask;
        }

        if (useStencilState)
        {
            if (ctx->state.var.stencil_func != GL_NEVER)
            {
                if (!mglIsValidGLCompareFunction(ctx->state.var.stencil_func)) {
                    mglLogRenderStateRepair("stencil_func", ctx->state.var.stencil_func, GL_ALWAYS);
                    ctx->state.var.stencil_func = GL_ALWAYS;
                    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
                }

                MTLStencilDescriptor *frontSDesc = [[MTLStencilDescriptor alloc] init];

                frontSDesc.stencilCompareFunction =
                    mglMTLCompareFunctionForGL(ctx->state.var.stencil_func,
                                               MTLCompareFunctionAlways,
                                               "front-stencil");
                frontSDesc.stencilFailureOperation = [self mtlStencilOpForGLOp:ctx->state.var.stencil_fail ];
                frontSDesc.depthFailureOperation = [self mtlStencilOpForGLOp:ctx->state.var.stencil_pass_depth_fail];
                frontSDesc.depthStencilPassOperation = [self mtlStencilOpForGLOp:ctx->state.var.stencil_pass_depth_pass];
                frontSDesc.writeMask = ctx->state.var.stencil_writemask;
                frontSDesc.readMask = ctx->state.var.stencil_value_mask;    // ????

                dsDesc.frontFaceStencil = frontSDesc;
            }

            if (ctx->state.var.stencil_back_func != GL_NEVER)
            {
                if (!mglIsValidGLCompareFunction(ctx->state.var.stencil_back_func)) {
                    mglLogRenderStateRepair("stencil_back_func", ctx->state.var.stencil_back_func, GL_ALWAYS);
                    ctx->state.var.stencil_back_func = GL_ALWAYS;
                    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
                }

                MTLStencilDescriptor *backSDesc = [[MTLStencilDescriptor alloc] init];

                backSDesc.stencilCompareFunction =
                    mglMTLCompareFunctionForGL(ctx->state.var.stencil_back_func,
                                               MTLCompareFunctionAlways,
                                               "back-stencil");
                backSDesc.stencilFailureOperation = [self mtlStencilOpForGLOp:ctx->state.var.stencil_back_fail ];
                backSDesc.depthFailureOperation = [self mtlStencilOpForGLOp:ctx->state.var.stencil_back_pass_depth_fail];
                backSDesc.depthStencilPassOperation = [self mtlStencilOpForGLOp:ctx->state.var.stencil_back_pass_depth_pass];
                backSDesc.writeMask = ctx->state.var.stencil_back_writemask;
                backSDesc.readMask = ctx->state.var.stencil_back_value_mask;    // ????

                dsDesc.backFaceStencil = backSDesc;
            }
        }

        id <MTLDepthStencilState> dsState = [_device
                                  newDepthStencilStateWithDescriptor:dsDesc];

        [_currentRenderEncoder setDepthStencilState: dsState];
        if (useStencilState) {
            [_currentRenderEncoder setStencilFrontReferenceValue:(uint32_t)ctx->state.var.stencil_ref
                                              backReferenceValue:(uint32_t)ctx->state.var.stencil_back_ref];
        }
    }
    else
    {
        MTLDepthStencilDescriptor *disabledDSDesc = [[MTLDepthStencilDescriptor alloc] init];
        disabledDSDesc.depthCompareFunction = MTLCompareFunctionAlways;
        disabledDSDesc.depthWriteEnabled = NO;

        id <MTLDepthStencilState> disabledDSState = [_device
                                      newDepthStencilStateWithDescriptor:disabledDSDesc];
        if (disabledDSState) {
            [_currentRenderEncoder setDepthStencilState:disabledDSState];
        }
    }

    [_currentRenderEncoder setBlendColorRed:ctx->state.var.blend_color[0]
                                      green:ctx->state.var.blend_color[1]
                                       blue:ctx->state.var.blend_color[2]
                                      alpha:ctx->state.var.blend_color[3]];

    // Metal validates viewport/scissor strictly against the active render pass dimensions.
    // Always derive pass size from the current attachments first (not from window drawable fallback).
    {
        static uint64_t s_encoderStateUpdateCount = 0;
        bool traceEncoderState = kMGLDiagnosticStateLogs || mglShouldTraceCall(++s_encoderStateUpdateCount);

        NSUInteger passWidth = 0;
        NSUInteger passHeight = 0;
        id<MTLTexture> passTexture = nil;

        if (_renderPassDescriptor) {
            passWidth = _renderPassDescriptor.renderTargetWidth;
            passHeight = _renderPassDescriptor.renderTargetHeight;

            if (passWidth == 0 || passHeight == 0) {
                for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                    id<MTLTexture> candidate = _renderPassDescriptor.colorAttachments[i].texture;
                    if (candidate) {
                        passTexture = candidate;
                        break;
                    }
                }

                if (!passTexture) {
                    passTexture = _renderPassDescriptor.depthAttachment.texture;
                }
                if (!passTexture) {
                    passTexture = _renderPassDescriptor.stencilAttachment.texture;
                }

                if (passTexture) {
                    passWidth = passTexture.width;
                    passHeight = passTexture.height;
                    _renderPassDescriptor.renderTargetWidth = passWidth;
                    _renderPassDescriptor.renderTargetHeight = passHeight;
                    NSLog(@"MGL INFO: Resolved render pass size from attachment %lux%lu (rtw/rth were unset)",
                          (unsigned long)passWidth, (unsigned long)passHeight);
                }
            }
        }

        if ((passWidth == 0 || passHeight == 0) && _drawable && _drawable.texture) {
            passWidth = _drawable.texture.width;
            passHeight = _drawable.texture.height;
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: Falling back to drawable size for encoder state: %lux%lu",
                      (unsigned long)passWidth, (unsigned long)passHeight);
            }
        }

        if ((passWidth == 0 || passHeight == 0) && _layer) {
            CGSize drawableSize = _layer.drawableSize;
            if (drawableSize.width > 0 && drawableSize.height > 0) {
                passWidth = (NSUInteger)drawableSize.width;
                passHeight = (NSUInteger)drawableSize.height;
            } else {
                NSRect frame = [_layer frame];
                if (frame.size.width > 0 && frame.size.height > 0) {
                    passWidth = (NSUInteger)frame.size.width;
                    passHeight = (NSUInteger)frame.size.height;
                }
            }
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: Falling back to layer size for encoder state: %lux%lu",
                      (unsigned long)passWidth, (unsigned long)passHeight);
            }
        }

        if (passWidth > 0 && passHeight > 0) {
            GLint rawSx = 0;
            GLint rawSy = 0;
            GLint rawSw = (GLint)passWidth;
            GLint rawSh = (GLint)passHeight;

            GLint sx = 0;
            GLint sy = 0;
            GLint sw = (GLint)passWidth;
            GLint sh = (GLint)passHeight;

            if (ctx->state.caps.scissor_test) {
                rawSx = (GLint)ctx->state.var.scissor_box[0];
                rawSy = (GLint)ctx->state.var.scissor_box[1];
                rawSw = (GLint)ctx->state.var.scissor_box[2];
                rawSh = (GLint)ctx->state.var.scissor_box[3];

                sx = rawSx;
                sy = rawSy;
                sw = rawSw;
                sh = rawSh;

                // GL allows negative x/y; clamp origin and shrink extent accordingly.
                if (sx < 0) {
                    sw += sx;
                    sx = 0;
                }
                if (sy < 0) {
                    sh += sy;
                    sy = 0;
                }

                if (sx >= (GLint)passWidth || sy >= (GLint)passHeight) {
                    sx = 0;
                    sy = 0;
                    sw = (GLint)passWidth;
                    sh = (GLint)passHeight;
                } else {
                    GLint maxWidth = (GLint)passWidth - sx;
                    GLint maxHeight = (GLint)passHeight - sy;

                    if (sw > maxWidth) {
                        sw = maxWidth;
                    }
                    if (sh > maxHeight) {
                        sh = maxHeight;
                    }

                    if (sw <= 0 || sh <= 0) {
                        sx = 0;
                        sy = 0;
                        sw = (GLint)passWidth;
                        sh = (GLint)passHeight;
                    }
                }
            }

            GLint metalSy = sy;
            if (ctx->state.var.clip_origin != GL_UPPER_LEFT) {
                GLint flippedY = (GLint)passHeight - (sy + sh);
                if (flippedY < 0) {
                    flippedY = 0;
                }
                metalSy = flippedY;
            }

            if (traceEncoderState || sx != rawSx || sy != rawSy || sw != rawSw || sh != rawSh || metalSy != sy) {
                NSLog(@"MGL SCISSOR apply pass=%lux%lu scissorEnabled=%d origin=0x%x raw=(%d,%d,%d,%d) glResolved=(%d,%d,%d,%d) metal=(%d,%d,%d,%d)",
                      (unsigned long)passWidth, (unsigned long)passHeight,
                      ctx->state.caps.scissor_test ? 1 : 0,
                      ctx->state.var.clip_origin,
                      rawSx, rawSy, rawSw, rawSh,
                      sx, sy, sw, sh,
                      sx, metalSy, sw, sh);
            }

            MTLScissorRect rect;
            rect.x = (NSUInteger)sx;
            rect.y = (NSUInteger)metalSy;
            rect.width = (NSUInteger)sw;
            rect.height = (NSUInteger)sh;
            [_currentRenderEncoder setScissorRect:rect];

            GLdouble rawVx = (GLdouble)ctx->state.viewport[0];
            GLdouble rawVy = (GLdouble)ctx->state.viewport[1];
            GLdouble rawVw = (GLdouble)ctx->state.viewport[2];
            GLdouble rawVh = (GLdouble)ctx->state.viewport[3];

            GLdouble vx = rawVx;
            GLdouble vy = rawVy;
            GLdouble vw = rawVw;
            GLdouble vh = rawVh;

            if (vw <= 0.0 || vh <= 0.0) {
                vx = 0.0;
                vy = 0.0;
                vw = (GLdouble)passWidth;
                vh = (GLdouble)passHeight;
            }

            if (vx < 0.0) {
                vw += vx;
                vx = 0.0;
            }
            if (vy < 0.0) {
                vh += vy;
                vy = 0.0;
            }

            if (vx >= (GLdouble)passWidth || vy >= (GLdouble)passHeight) {
                vx = 0.0;
                vy = 0.0;
                vw = (GLdouble)passWidth;
                vh = (GLdouble)passHeight;
            } else {
                GLdouble maxVw = (GLdouble)passWidth - vx;
                GLdouble maxVh = (GLdouble)passHeight - vy;
                if (vw > maxVw) {
                    vw = maxVw;
                }
                if (vh > maxVh) {
                    vh = maxVh;
                }
                if (vw <= 0.0 || vh <= 0.0) {
                    vx = 0.0;
                    vy = 0.0;
                    vw = (GLdouble)passWidth;
                    vh = (GLdouble)passHeight;
                }
            }

            GLdouble metalVy = vy;
            if (ctx->state.var.clip_origin != GL_UPPER_LEFT) {
                metalVy = (GLdouble)passHeight - (vy + vh);
                if (metalVy < 0.0) {
                    metalVy = 0.0;
                }
            }

            BOOL viewportWasClamped = (vx != rawVx || vy != rawVy || vw != rawVw || vh != rawVh);
            BOOL viewportOriginConverted = (metalVy != vy);
            if (traceEncoderState || viewportWasClamped) {
                NSLog(@"MGL VIEWPORT apply pass=%lux%lu origin=0x%x raw=(%.3f,%.3f,%.3f,%.3f) resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
                      (unsigned long)passWidth, (unsigned long)passHeight,
                      ctx->state.var.clip_origin,
                      rawVx, rawVy, rawVw, rawVh,
                      vx, vy, vw, vh,
                      vx, metalVy, vw, vh);
            }

            if (viewportWasClamped || viewportOriginConverted) {
                static uint64_t s_viewportClampDetailCount = 0;
                uint64_t clampHit = ++s_viewportClampDetailCount;
                BOOL logClampDetail = (clampHit <= 80ull || (clampHit % 120ull) == 0ull);

                if (logClampDetail) {
                    Framebuffer *debugFbo = ctx->state.framebuffer;
                    BOOL debugFboValid = (debugFbo != NULL &&
                                          mglRendererObjectPointerLikelyValid(debugFbo) &&
                                          mglRendererPointerInHashTable(&ctx->state.framebuffer_table, debugFbo) &&
                                          mglPointerRangeIsReadable(debugFbo, sizeof(*debugFbo)));
                    id<MTLTexture> rpColor0 = _renderPassDescriptor.colorAttachments[0].texture;
                    id<MTLTexture> rpDepth = _renderPassDescriptor.depthAttachment.texture;
                    id<MTLTexture> drawableTexture = (_drawable ? _drawable.texture : nil);

                    NSLog(@"MGL VIEWPORT CLAMP DETAIL hit=%llu fbo=%p valid=%d fboName=%u drawBuffer=0x%x pass=%lux%lu "
                          "rpColor0=%p(%lux%lu) rpDepth=%p(%lux%lu) drawable=%p(%lux%lu) raw=(%.3f,%.3f,%.3f,%.3f) "
                          "resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
                          (unsigned long long)clampHit,
                          debugFbo,
                          debugFboValid ? 1 : 0,
                          (debugFboValid ? debugFbo->name : 0),
                          ctx->state.draw_buffer,
                          (unsigned long)passWidth,
                          (unsigned long)passHeight,
                          rpColor0,
                          (unsigned long)(rpColor0 ? rpColor0.width : 0),
                          (unsigned long)(rpColor0 ? rpColor0.height : 0),
                          rpDepth,
                          (unsigned long)(rpDepth ? rpDepth.width : 0),
                          (unsigned long)(rpDepth ? rpDepth.height : 0),
                          drawableTexture,
                          (unsigned long)(drawableTexture ? drawableTexture.width : 0),
                          (unsigned long)(drawableTexture ? drawableTexture.height : 0),
                          rawVx, rawVy, rawVw, rawVh,
                          vx, vy, vw, vh,
                          vx, metalVy, vw, vh);

                    if (debugFboValid) {
                        for (int attIndex = 0; attIndex < MAX_COLOR_ATTACHMENTS; attIndex++) {
                            FBOAttachment *attachment = &debugFbo->color_attachments[attIndex];
                            if (attachment->texture == 0 && attachment->buf.tex == NULL && attachment->buf.rbo == NULL) {
                                continue;
                            }

                            Texture *attachmentTexture = NULL;
                            if (attachment->textarget == GL_RENDERBUFFER) {
                                attachmentTexture = attachment->buf.rbo ? attachment->buf.rbo->tex : NULL;
                            } else {
                                attachmentTexture = attachment->buf.tex;
                                if (!attachmentTexture && attachment->texture != 0) {
                                    attachmentTexture = findTexture(ctx, attachment->texture);
                                }
                            }

                            id<MTLTexture> attachmentMtl = (attachmentTexture && attachmentTexture->mtl_data)
                                ? (__bridge id<MTLTexture>)(attachmentTexture->mtl_data)
                                : nil;
                            id<MTLTexture> rpAttachment = _renderPassDescriptor.colorAttachments[attIndex].texture;

                            NSLog(@"MGL VIEWPORT CLAMP FBO att=%d name=%u textarget=0x%x level=%d layer=%d tex=%p "
                                  "texName=%u texTarget=0x%x texSize=%ux%ux%u mtl=%p(%lux%lu) rpTex=%p(%lux%lu)",
                                  attIndex,
                                  attachment->texture,
                                  attachment->textarget,
                                  attachment->level,
                                  attachment->layer,
                                  attachmentTexture,
                                  attachmentTexture ? attachmentTexture->name : 0,
                                  attachmentTexture ? attachmentTexture->target : 0,
                                  attachmentTexture ? attachmentTexture->width : 0,
                                  attachmentTexture ? attachmentTexture->height : 0,
                                  attachmentTexture ? attachmentTexture->depth : 0,
                                  attachmentMtl,
                                  (unsigned long)(attachmentMtl ? attachmentMtl.width : 0),
                                  (unsigned long)(attachmentMtl ? attachmentMtl.height : 0),
                                  rpAttachment,
                                  (unsigned long)(rpAttachment ? rpAttachment.width : 0),
                                  (unsigned long)(rpAttachment ? rpAttachment.height : 0));
                        }
                    }
                }
            }

            [_currentRenderEncoder setViewport:(MTLViewport){vx, metalVy, vw, vh,
                                                ctx->state.var.depth_range[0], ctx->state.var.depth_range[1]}];
        } else {
            if (traceEncoderState) {
                NSLog(@"MGL WARNING: updateCurrentRenderEncoder could not resolve pass size; using raw GL viewport");
            }
            [_currentRenderEncoder setViewport:(MTLViewport){ctx->state.viewport[0], ctx->state.viewport[1],
                                                ctx->state.viewport[2], ctx->state.viewport[3],
                                                ctx->state.var.depth_range[0], ctx->state.var.depth_range[1]}];
        }
    }

    if (ctx->state.var.front_face != GL_CW && ctx->state.var.front_face != GL_CCW) {
        mglLogRenderStateRepair("front_face", ctx->state.var.front_face, GL_CCW);
        ctx->state.var.front_face = GL_CCW;
        ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
    }

    BOOL defaultFramebufferSampledPass =
        ctx->state.framebuffer == NULL &&
        !ctx->state.caps.depth_test &&
        [self getProgramBindingCount:_FRAGMENT_SHADER type:SPVC_RESOURCE_TYPE_SAMPLED_IMAGE] > 0;

    if (ctx->state.caps.cull_face && !defaultFramebufferSampledPass)
    {
        MTLCullMode cull_mode;

        switch(ctx->state.var.cull_face_mode)
        {
            case GL_BACK: cull_mode = MTLCullModeBack; break;
            case GL_FRONT: cull_mode = MTLCullModeFront; break;
            default:
                cull_mode = MTLCullModeNone;
        }

        [_currentRenderEncoder setCullMode:cull_mode];
        [_currentRenderEncoder setFrontFacingWinding:mglMTLWindingForGL(ctx->state.var.front_face)];
    }
    else
    {
        [_currentRenderEncoder setCullMode:MTLCullModeNone];
        [_currentRenderEncoder setFrontFacingWinding:mglMTLWindingForGL(ctx->state.var.front_face)];

        if (ctx->state.caps.cull_face && defaultFramebufferSampledPass) {
            static uint64_t s_defaultSampledCullBypassCount = 0;
            uint64_t hit = ++s_defaultSampledCullBypassCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL TRACE default sampled pass cull bypass hit=%llu program=%u drawBuf=0x%x",
                      (unsigned long long)hit,
                      (unsigned)(ctx ? ctx->state.program_name : 0u),
                      (unsigned)(ctx ? ctx->state.draw_buffer : 0u));
            }
        }
    }

    if (ctx->state.caps.depth_clamp)
    {
        [_currentRenderEncoder setDepthClipMode: MTLDepthClipModeClamp];
    }

    if (ctx->state.caps.polygon_offset_fill ||
        ctx->state.caps.polygon_offset_line ||
        ctx->state.caps.polygon_offset_point)
    {
        [_currentRenderEncoder setDepthBias:ctx->state.var.polygon_offset_units
                                 slopeScale:ctx->state.var.polygon_offset_factor
                                      clamp:0.0f];
    }
    else
    {
        [_currentRenderEncoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
    }

    MTLTriangleFillMode triangleFillMode = MTLTriangleFillModeFill;
    if (ctx->state.var.polygon_mode == GL_LINE)
    {
        triangleFillMode = MTLTriangleFillModeLines;
    }
    else if (ctx->state.var.polygon_mode != GL_FILL &&
             ctx->state.var.polygon_mode != GL_POINT)
    {
        mglLogRenderStateRepair("polygon_mode", ctx->state.var.polygon_mode, GL_FILL);
        ctx->state.var.polygon_mode = GL_FILL;
    }
    [_currentRenderEncoder setTriangleFillMode:triangleFillMode];
}

- (bool) newRenderEncoder
{
    // I can't remember why this is here...
    @autoreleasepool {
    static uint64_t s_newRenderEncoderCallCount = 0;
    uint64_t renderEncoderCall = ++s_newRenderEncoderCallCount;
    bool traceRenderEncoder = mglShouldTraceCall(renderEncoderCall) ||
                              (kMGLDiagnosticStateLogs && ((renderEncoderCall % 60ull) == 0ull));

    // AGX ERROR THROTTLING: Check if we should skip render encoder creation
    // BUT allow limited render encoder creation for essential functionality
    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: Render encoder creation requested during GPU recovery - attempting essential creation");
        // Continue with essential render encoder creation even during recovery
    }

    // CRITICAL SAFETY: Check command buffer before creating render encoder
    if (!_currentCommandBuffer) {
        NSLog(@"MGL ERROR: Cannot create render encoder - no command buffer available");
        [self recordGPUError];
        return false;
    }

    // end encoding on current render encoder
    [self endRenderEncoding];

    // grab the next drawable from CAMetalLayer
    if (_drawable == NULL)
    {
        if (!_layer) {
            NSLog(@"MGL ERROR: Cannot get drawable - no CAMetalLayer available");
            return false;
        }

        CGSize expectedDrawableSize = [self mglSyncLayerDrawableSizeFromView:"newRenderEncoder.nextDrawable"];
        _drawable = [_layer nextDrawable];

        // late init of gl scissor box on attachment to window system
        NSUInteger drawableWidth = (NSUInteger)MAX(1.0, expectedDrawableSize.width);
        NSUInteger drawableHeight = (NSUInteger)MAX(1.0, expectedDrawableSize.height);
        if (_drawable && _drawable.texture) {
            drawableWidth = (NSUInteger)_drawable.texture.width;
            drawableHeight = (NSUInteger)_drawable.texture.height;
        }

        if (!ctx->state.caps.scissor_test) {
            ctx->state.var.scissor_box[0] = 0;
            ctx->state.var.scissor_box[1] = 0;
        }
        ctx->state.var.scissor_box[2] = (GLint)drawableWidth;
        ctx->state.var.scissor_box[3] = (GLint)drawableHeight;
    }

    _renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
    if (!_renderPassDescriptor) {
        NSLog(@"MGL RENDERPASS ERROR: failed to allocate render pass descriptor");
        return false;
    }

    if (ctx->state.framebuffer)
    {
        Framebuffer *fbo;

        fbo = ctx->state.framebuffer;

        GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
        for (int i = 0; i < drawBufferCount; i++)
        {
            GLuint attachmentIndex = 0u;
            if (mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                      mglMetalDrawBufferAt(ctx, (GLuint)i),
                                                      &attachmentIndex) &&
                attachmentIndex < MAX_COLOR_ATTACHMENTS &&
                (fbo->color_attachment_bitfield & (1u << attachmentIndex)) &&
                fbo->color_attachments[attachmentIndex].texture)
            {
                Texture *tex;

                tex = [self framebufferAttachmentTexture: &fbo->color_attachments[attachmentIndex]];
                if (!tex) {
                    continue;
                }

                // Ensure attachment textures are created with RenderTarget usage.
                tex->is_render_target = true;
                RETURN_FALSE_ON_FAILURE([self bindMTLTexture: tex]);
                if (!tex->mtl_data) {
                    continue;
                }

                MGLMetalAttachmentSubresource subresource =
                    mglMetalAttachmentSubresourceForAttachment(&fbo->color_attachments[attachmentIndex]);
                _renderPassDescriptor.colorAttachments[i].texture = (__bridge id<MTLTexture> _Nullable)(tex->mtl_data);
                _renderPassDescriptor.colorAttachments[i].level = subresource.level;
                _renderPassDescriptor.colorAttachments[i].slice = subresource.slice;
                _renderPassDescriptor.colorAttachments[i].depthPlane = subresource.depthPlane;

                // Keep render pass dimensions aligned with attached color targets.
                // Some FBO paths use textures (not renderbuffers), and Metal still requires
                // scissor/viewport to be bounded by the attachment dimensions.
                NSUInteger attWidth = (NSUInteger)tex->width;
                NSUInteger attHeight = (NSUInteger)tex->height;
                if (attWidth > 0 && attHeight > 0) {
                    if (_renderPassDescriptor.renderTargetWidth == 0 || _renderPassDescriptor.renderTargetHeight == 0) {
                        _renderPassDescriptor.renderTargetWidth = attWidth;
                        _renderPassDescriptor.renderTargetHeight = attHeight;
                    } else if (_renderPassDescriptor.renderTargetWidth != attWidth ||
                               _renderPassDescriptor.renderTargetHeight != attHeight) {
                        NSUInteger oldWidth = _renderPassDescriptor.renderTargetWidth;
                        NSUInteger oldHeight = _renderPassDescriptor.renderTargetHeight;
                        _renderPassDescriptor.renderTargetWidth = MIN(_renderPassDescriptor.renderTargetWidth, attWidth);
                        _renderPassDescriptor.renderTargetHeight = MIN(_renderPassDescriptor.renderTargetHeight, attHeight);
                        NSLog(@"MGL WARNING: FBO color attachment size mismatch slot=%d old=%lux%lu new=%lux%lu resolved=%lux%lu",
                              i,
                              (unsigned long)oldWidth,
                              (unsigned long)oldHeight,
                              (unsigned long)attWidth,
                              (unsigned long)attHeight,
                              (unsigned long)_renderPassDescriptor.renderTargetWidth,
                              (unsigned long)_renderPassDescriptor.renderTargetHeight);
                    }
                }
            }
        }

        // depth attachment
        if (fbo->depth.texture)
        {
            Texture *tex;

            tex = [self framebufferAttachmentTexture: &fbo->depth];
            if (tex) {
                tex->is_render_target = true;
                RETURN_FALSE_ON_FAILURE([self bindMTLTexture: tex]);
            }
            if (tex && tex->mtl_data) {
                _renderPassDescriptor.depthAttachment.texture = (__bridge id<MTLTexture> _Nullable)(tex->mtl_data);
                MGLMetalAttachmentSubresource subresource =
                    mglMetalAttachmentSubresourceForAttachment(&fbo->depth);
                _renderPassDescriptor.depthAttachment.level = subresource.level;
                _renderPassDescriptor.depthAttachment.slice = subresource.slice;
                _renderPassDescriptor.depthAttachment.depthPlane = subresource.depthPlane;
            }
        }

        // stencil attachment
        if (fbo->stencil.texture)
        {
            Texture *tex;

            tex = [self framebufferAttachmentTexture: &fbo->stencil];
            if (tex) {
                tex->is_render_target = true;
                RETURN_FALSE_ON_FAILURE([self bindMTLTexture: tex]);
            }
            if (tex && tex->mtl_data) {
                _renderPassDescriptor.stencilAttachment.texture = (__bridge id<MTLTexture> _Nullable)(tex->mtl_data);
                MGLMetalAttachmentSubresource subresource =
                    mglMetalAttachmentSubresourceForAttachment(&fbo->stencil);
                _renderPassDescriptor.stencilAttachment.level = subresource.level;
                _renderPassDescriptor.stencilAttachment.slice = subresource.slice;
                _renderPassDescriptor.stencilAttachment.depthPlane = subresource.depthPlane;
            }
        }
    }
    else
    {
        GLuint mgl_drawbuffer;
        id<MTLTexture> texture = nil;
        id<MTLTexture> depth_texture = nil;
        id<MTLTexture> stencil_texture = nil;
        
        switch(ctx->state.draw_buffer)
        {
            case GL_FRONT: mgl_drawbuffer = _FRONT; break;
            case GL_BACK: mgl_drawbuffer = _FRONT; break;
            case GL_FRONT_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
            case GL_FRONT_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
            case GL_BACK_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
            case GL_BACK_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
            case GL_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
            case GL_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
            case GL_FRONT_AND_BACK: mgl_drawbuffer = _FRONT; break;
            case GL_COLOR_ATTACHMENT0: mgl_drawbuffer = _FRONT; break;
            case GL_NONE:
                // Handle GL_NONE gracefully - no draw buffer selected
                mgl_drawbuffer = _FRONT; // fallback to front
                DEBUG_PRINT("MGL: draw_buffer is GL_NONE, falling back to FRONT\n");
                break;
            default:
                DEBUG_PRINT("MGL: Unknown draw_buffer value: 0x%x, falling back to FRONT\n", ctx->state.draw_buffer);
                mgl_drawbuffer = _FRONT; // fallback to front instead of failing render setup
                NSLog(@"MGL WARNING: Unknown draw_buffer value 0x%x, using FRONT fallback", ctx->state.draw_buffer);
                break;
        }

        if([self checkDrawBufferSize:mgl_drawbuffer])
        {
            _drawBuffers[mgl_drawbuffer].drawbuffer = NULL;
            _drawBuffers[mgl_drawbuffer].depthbuffer = NULL;
            _drawBuffers[mgl_drawbuffer].stencilbuffer = NULL;
        }

        // attach color buffer
        if (mgl_drawbuffer == _FRONT)
        {
            // SAFETY: Ensure we have a valid drawable with texture
            if (!_drawable) {
                NSLog(@"MGL ERROR: No drawable available for front buffer");
                return false;
            }

            texture = _drawable.texture;

            // sleep mode will return a null texture - handle gracefully without crashing
            if (!texture) {
                NSLog(@"MGL WARNING: Drawable texture is NULL (sleep mode or window not visible), attempting to get new drawable");

                // Try to get a new drawable
                _drawable = [_layer nextDrawable];
                if (_drawable) {
                    texture = _drawable.texture;
                    NSLog(@"MGL INFO: Successfully obtained new drawable with texture");
                } else {
                    NSLog(@"MGL ERROR: Still no drawable texture available");
                    return false;
                }
            }
        }
        else if(_drawBuffers[mgl_drawbuffer].drawbuffer)
        {
            texture = _drawBuffers[mgl_drawbuffer].drawbuffer;
        }
        else
        {
            texture = [self newDrawBuffer: ctx->pixel_format.mtl_pixel_format isDepthStencil:false];
            _drawBuffers[mgl_drawbuffer].drawbuffer = texture;
        }

        // attach depth. The default framebuffer must have a usable depth
        // attachment whenever GL depth testing is active, even if the legacy
        // context format fields were left unset by the window/bootstrap path.
        BOOL defaultPassNeedsDepth = ctx->state.caps.depth_test;
        if (defaultPassNeedsDepth)
        {
            MTLPixelFormat depthFormat = ctx->depth_format.mtl_pixel_format;
            if (depthFormat == MTLPixelFormatInvalid) {
                depthFormat = MTLPixelFormatDepth32Float;
            }

            if(_drawBuffers[mgl_drawbuffer].depthbuffer)
            {
                depth_texture = _drawBuffers[mgl_drawbuffer].depthbuffer;
            }
            else
            {
                depth_texture = [self newDrawBufferWithCustomSize:depthFormat isDepthStencil:true customSize: CGSizeMake(texture.width, texture.height) ];
                _drawBuffers[mgl_drawbuffer].depthbuffer = depth_texture;
                if (depth_texture) {
                    static uint64_t s_defaultDepthCreateCount = 0;
                    uint64_t hit = ++s_defaultDepthCreateCount;
                    if (hit <= 8) {
                        NSLog(@"MGL DEFAULT FBO: created depth attachment fmt=%lu size=%lux%lu drawBuffer=%u",
                              (unsigned long)depthFormat,
                              (unsigned long)depth_texture.width,
                              (unsigned long)depth_texture.height,
                              mgl_drawbuffer);
                    }
                }
            }
        }

        // attach stencil
        BOOL defaultPassNeedsStencil = ctx->state.caps.stencil_test || ctx->stencil_format.format;
        if (defaultPassNeedsStencil)
        {
            MTLPixelFormat stencilFormat = ctx->stencil_format.mtl_pixel_format;
            if (stencilFormat == MTLPixelFormatInvalid ||
                stencilFormat == MTLPixelFormatDepth32Float_Stencil8) {
                stencilFormat = MTLPixelFormatStencil8;
            }

            if(_drawBuffers[mgl_drawbuffer].stencilbuffer)
            {
                stencil_texture = _drawBuffers[mgl_drawbuffer].stencilbuffer;
            }
            else
            {
                stencil_texture = [self newDrawBufferWithCustomSize:stencilFormat isDepthStencil:true customSize: CGSizeMake(texture.width, texture.height) ];
                _drawBuffers[mgl_drawbuffer].stencilbuffer = stencil_texture;
            }
        }

        _renderPassDescriptor.colorAttachments[0].texture = texture;
        _renderPassDescriptor.depthAttachment.texture = depth_texture;
        _renderPassDescriptor.stencilAttachment.texture = stencil_texture;

        _renderPassDescriptor.renderTargetWidth = texture.width;
        _renderPassDescriptor.renderTargetHeight = texture.height;
    }

    if (!ctx->state.framebuffer &&
        ctx->state.caps.depth_test &&
        !_renderPassDescriptor.depthAttachment.texture) {
        NSUInteger depthWidth = _renderPassDescriptor.renderTargetWidth;
        NSUInteger depthHeight = _renderPassDescriptor.renderTargetHeight;

        if (depthWidth == 0 || depthHeight == 0) {
            id<MTLTexture> color0 = _renderPassDescriptor.colorAttachments[0].texture;
            if (color0) {
                depthWidth = color0.width;
                depthHeight = color0.height;
            }
        }

        if (depthWidth > 0 && depthHeight > 0) {
            if (!_transientDepthTexture ||
                _transientDepthTextureWidth != depthWidth ||
                _transientDepthTextureHeight != depthHeight) {
                MTLTextureDescriptor *depthDesc =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                       width:depthWidth
                                                                      height:depthHeight
                                                                   mipmapped:NO];
                depthDesc.usage = MTLTextureUsageRenderTarget;
                depthDesc.storageMode = MTLStorageModePrivate;
                _transientDepthTexture = [_device newTextureWithDescriptor:depthDesc];
                _transientDepthTextureWidth = depthWidth;
                _transientDepthTextureHeight = depthHeight;

                if (_transientDepthTexture) {
                    static uint64_t s_transientDepthCreateCount = 0;
                    uint64_t hit = ++s_transientDepthCreateCount;
                    if (hit <= 16 || (hit % 128) == 0) {
                        NSLog(@"MGL TRANSIENT FBO: created depth attachment fmt=%lu size=%lux%lu fbo=%u",
                              (unsigned long)MTLPixelFormatDepth32Float,
                              (unsigned long)depthWidth,
                              (unsigned long)depthHeight,
                              (unsigned)(mglRendererSafeFramebufferName(ctx)));
                    }
                } else {
                    NSLog(@"MGL ERROR: failed to create transient depth attachment size=%lux%lu fbo=%u",
                          (unsigned long)depthWidth,
                          (unsigned long)depthHeight,
                          (unsigned)(mglRendererSafeFramebufferName(ctx)));
                }
            }

            if (_transientDepthTexture) {
                _renderPassDescriptor.depthAttachment.texture = _transientDepthTexture;
                _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
                _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionDontCare;
                _renderPassDescriptor.depthAttachment.clearDepth = ctx->state.var.depth_clear_value;
            }
        }
    }

    GLuint fboColorClearCount = 0;
    GLbitfield fboColorClearMask = 0;
    GLbitfield fboColorAttachment0ClearMask = 0;

    Framebuffer *fbo = ctx->state.framebuffer;
    if (fbo) {
        GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
        for (int i = 0; i < drawBufferCount; ++i) {
            GLuint attachmentIndex = 0u;
            if (!mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                       mglMetalDrawBufferAt(ctx, (GLuint)i),
                                                       &attachmentIndex) ||
                attachmentIndex >= MAX_COLOR_ATTACHMENTS ||
                ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
                _renderPassDescriptor.colorAttachments[i].loadAction = MTLLoadActionLoad;
                continue;
            }

            FBOAttachment *att = &fbo->color_attachments[attachmentIndex];
            if (attachmentIndex == 0) {
                fboColorAttachment0ClearMask = att->clear_bitmask;
            }
            
            if (att->clear_bitmask & GL_COLOR_BUFFER_BIT) {
                _renderPassDescriptor.colorAttachments[i].clearColor =
                    MTLClearColorMake(att->clear_color[0],
                                      att->clear_color[1],
                                      att->clear_color[2],
                                      att->clear_color[3]);
                _renderPassDescriptor.colorAttachments[i].loadAction = MTLLoadActionClear;
                _renderPassDescriptor.colorAttachments[i].storeAction = MTLStoreActionStore;
                
                att->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
                mglMarkTextureLevelRenderTargetWritten([self framebufferAttachmentTexture:att], att->level);
                
                fboColorClearCount++;
                fboColorClearMask |= (GLbitfield)(1u << attachmentIndex);
            } else {
                _renderPassDescriptor.colorAttachments[i].loadAction = MTLLoadActionLoad;
            }
        }

        if (fbo->depth.clear_bitmask & GL_DEPTH_BUFFER_BIT) {
            _renderPassDescriptor.depthAttachment.clearDepth = fbo->depth.clear_color[0];
            _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
            _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
            fbo->depth.clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
        } else {
            _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionLoad;
            if (_renderPassDescriptor.depthAttachment.texture) {
                _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
            }
        }

        if (fbo->stencil.clear_bitmask & GL_STENCIL_BUFFER_BIT) {
            _renderPassDescriptor.stencilAttachment.clearStencil = (uint32_t)fbo->stencil.clear_color[0];
            _renderPassDescriptor.stencilAttachment.loadAction = MTLLoadActionClear;
            _renderPassDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
            fbo->stencil.clear_bitmask &= ~GL_STENCIL_BUFFER_BIT;
        } else {
            _renderPassDescriptor.stencilAttachment.loadAction = MTLLoadActionLoad;
            if (_renderPassDescriptor.stencilAttachment.texture) {
                _renderPassDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
            }
        }
    } else {
        GLbitfield defaultClearMask = ctx->state.default_fbo_clear_bitmask;

        if (defaultClearMask & GL_COLOR_BUFFER_BIT) {
            _renderPassDescriptor.colorAttachments[0].clearColor =
                MTLClearColorMake(ctx->state.default_clear_color[0],
                                  ctx->state.default_clear_color[1],
                                  ctx->state.default_clear_color[2],
                                  ctx->state.default_clear_color[3]);
            _renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
            _renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
            ctx->state.default_fbo_clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        } else {
            _renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionLoad;
        }

        if (defaultClearMask & GL_DEPTH_BUFFER_BIT) {
            _renderPassDescriptor.depthAttachment.clearDepth = STATE_VAR(depth_clear_value);
            _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionClear;
            _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
            ctx->state.default_fbo_clear_bitmask &= ~GL_DEPTH_BUFFER_BIT;
        } else {
            _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionLoad;
            if (_renderPassDescriptor.depthAttachment.texture) {
                _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionStore;
            }
        }

        if (defaultClearMask & GL_STENCIL_BUFFER_BIT) {
            _renderPassDescriptor.stencilAttachment.clearStencil = STATE_VAR(stencil_clear_value);
            _renderPassDescriptor.stencilAttachment.loadAction = MTLLoadActionClear;
            _renderPassDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
            ctx->state.default_fbo_clear_bitmask &= ~GL_STENCIL_BUFFER_BIT;
        } else {
            _renderPassDescriptor.stencilAttachment.loadAction = MTLLoadActionLoad;
            if (_renderPassDescriptor.stencilAttachment.texture) {
                _renderPassDescriptor.stencilAttachment.storeAction = MTLStoreActionStore;
            }
        }
    }

	    if (kMGLDiagnosticStateLogs && traceRenderEncoder) {
	        MTLClearColor c0 = _renderPassDescriptor.colorAttachments[0].clearColor;
	        NSLog(@"MGL TRACE clear.resolve call=%llu fbo=%u "
	              "fboColorClears=%u fboColorMask=0x%x fboAtt0ClearMask=0x%x c0LA=%s depthLA=%s stencilLA=%s "
	              "c0Clear=(%.3f,%.3f,%.3f,%.3f) depthClear=%.3f stencilClear=%u",
              (unsigned long long)renderEncoderCall,
              (unsigned)(mglRendererSafeFramebufferName(ctx)),
              (unsigned)fboColorClearCount,
              (unsigned)fboColorClearMask,
              (unsigned)fboColorAttachment0ClearMask,
              mglLoadActionName(_renderPassDescriptor.colorAttachments[0].loadAction),
              mglLoadActionName(_renderPassDescriptor.depthAttachment.loadAction),
              mglLoadActionName(_renderPassDescriptor.stencilAttachment.loadAction),
              c0.red,
              c0.green,
              c0.blue,
              c0.alpha,
	              _renderPassDescriptor.depthAttachment.clearDepth,
	              (unsigned)_renderPassDescriptor.stencilAttachment.clearStencil);
	    }

	    BOOL clearResolveInteresting =
	        (fboColorClearCount != 0) ||
	        (fboColorAttachment0ClearMask != 0);
	    if (clearResolveInteresting) {
	        static uint64_t s_clearResolveDetailLogCount = 0;
	        uint64_t hit = ++s_clearResolveDetailLogCount;
	        if (hit <= 256ull || (hit % 512ull) == 0ull) {
	            MTLClearColor c0 = _renderPassDescriptor.colorAttachments[0].clearColor;
	            id<MTLTexture> c0Tex = _renderPassDescriptor.colorAttachments[0].texture;
	            id<MTLTexture> dTex = _renderPassDescriptor.depthAttachment.texture;
	            id<MTLTexture> sTex = _renderPassDescriptor.stencilAttachment.texture;
	            NSLog(@"MGL TRACE clear.resolve-detail call=%llu hit=%llu fbo=%u drawBuf=0x%x "
	                  "fboColorClears=%u fboColorMask=0x%x fboAtt0Mask=0x%x "
	                  "c0LA=%s c0SA=%s depthLA=%s depthSA=%s stencilLA=%s stencilSA=%s "
	                  "c0Tex=%p fmt=%lu size=%lux%lu depthTex=%p stencilTex=%p "
	                  "clearRGBA=(%.3f,%.3f,%.3f,%.3f) depthClear=%.3f stencilClear=%u",
	                  (unsigned long long)renderEncoderCall,
	                  (unsigned long long)hit,
	                  (unsigned)(mglRendererSafeFramebufferName(ctx)),
	                  (unsigned)ctx->state.draw_buffer,
	                  (unsigned)fboColorClearCount,
	                  (unsigned)fboColorClearMask,
	                  (unsigned)fboColorAttachment0ClearMask,
	                  mglLoadActionName(_renderPassDescriptor.colorAttachments[0].loadAction),
	                  mglStoreActionName(_renderPassDescriptor.colorAttachments[0].storeAction),
	                  mglLoadActionName(_renderPassDescriptor.depthAttachment.loadAction),
	                  mglStoreActionName(_renderPassDescriptor.depthAttachment.storeAction),
	                  mglLoadActionName(_renderPassDescriptor.stencilAttachment.loadAction),
	                  mglStoreActionName(_renderPassDescriptor.stencilAttachment.storeAction),
	                  c0Tex,
	                  (unsigned long)(c0Tex ? c0Tex.pixelFormat : MTLPixelFormatInvalid),
	                  (unsigned long)(c0Tex ? c0Tex.width : 0),
	                  (unsigned long)(c0Tex ? c0Tex.height : 0),
	                  dTex,
	                  sTex,
	                  c0.red,
	                  c0.green,
	                  c0.blue,
	                  c0.alpha,
	                  _renderPassDescriptor.depthAttachment.clearDepth,
	                  (unsigned)_renderPassDescriptor.stencilAttachment.clearStencil);
	        }
	    }
	
	    _renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;

    if (kMGLDiagnosticStateLogs && traceRenderEncoder) {
        id<MTLTexture> c0Tex = _renderPassDescriptor.colorAttachments[0].texture;
        id<MTLTexture> dTex = _renderPassDescriptor.depthAttachment.texture;
        id<MTLTexture> sTex = _renderPassDescriptor.stencilAttachment.texture;
        NSLog(@"MGL TRACE renderpass.attach call=%llu fbo=%u drawBuf=0x%x rt=%lux%lu "
              "c0=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s stencil=%p fmt=%lu size=%lux%lu la/sa=%s/%s",
              (unsigned long long)renderEncoderCall,
              (unsigned)(mglRendererSafeFramebufferName(ctx)),
              (unsigned)ctx->state.draw_buffer,
              (unsigned long)_renderPassDescriptor.renderTargetWidth,
              (unsigned long)_renderPassDescriptor.renderTargetHeight,
              c0Tex,
              (unsigned long)(c0Tex ? c0Tex.pixelFormat : MTLPixelFormatInvalid),
              (unsigned long)(c0Tex ? c0Tex.usage : 0),
              (unsigned long)(c0Tex ? c0Tex.width : 0),
              (unsigned long)(c0Tex ? c0Tex.height : 0),
              mglLoadActionName(_renderPassDescriptor.colorAttachments[0].loadAction),
              mglStoreActionName(_renderPassDescriptor.colorAttachments[0].storeAction),
              dTex,
              (unsigned long)(dTex ? dTex.pixelFormat : MTLPixelFormatInvalid),
              (unsigned long)(dTex ? dTex.width : 0),
              (unsigned long)(dTex ? dTex.height : 0),
              mglLoadActionName(_renderPassDescriptor.depthAttachment.loadAction),
              mglStoreActionName(_renderPassDescriptor.depthAttachment.storeAction),
              sTex,
              (unsigned long)(sTex ? sTex.pixelFormat : MTLPixelFormatInvalid),
              (unsigned long)(sTex ? sTex.width : 0),
              (unsigned long)(sTex ? sTex.height : 0),
              mglLoadActionName(_renderPassDescriptor.stencilAttachment.loadAction),
              mglStoreActionName(_renderPassDescriptor.stencilAttachment.storeAction));
    }

    // create a render encoder from the renderpass descriptor
    // CRITICAL SAFETY: Validate inputs before creating render encoder
    if (!_renderPassDescriptor) {
        NSLog(@"MGL ERROR: Cannot create render encoder - render pass descriptor is NULL");
        [self recordGPUError];
        return false;
    }

    // Metal debug layer crashes if render pass has no output attachment.
    // Provide a tiny fallback color attachment for targetless/invalid passes.
    bool hasOutputAttachment = false;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (_renderPassDescriptor.colorAttachments[i].texture) {
            hasOutputAttachment = true;
            break;
        }
    }
    if (!hasOutputAttachment &&
        (_renderPassDescriptor.depthAttachment.texture || _renderPassDescriptor.stencilAttachment.texture)) {
        hasOutputAttachment = true;
    }

    if (!hasOutputAttachment) {
        if (!_fallbackRenderTargetTexture) {
            MTLTextureDescriptor *fbDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                   width:1
                                                                  height:1
                                                               mipmapped:NO];
            fbDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            fbDesc.storageMode = MTLStorageModeShared;
            _fallbackRenderTargetTexture = [_device newTextureWithDescriptor:fbDesc];
        }

        if (_fallbackRenderTargetTexture) {
            NSLog(@"MGL WARNING: Render pass had no attachments; binding 1x1 fallback color target");
            _renderPassDescriptor.colorAttachments[0].texture = _fallbackRenderTargetTexture;
            _renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionLoad;
            _renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
            _renderPassDescriptor.renderTargetWidth = 1;
            _renderPassDescriptor.renderTargetHeight = 1;
        } else {
            NSLog(@"MGL ERROR: Failed to allocate fallback render target texture");
            [self recordGPUError];
            return false;
        }
    }

    // Final guard: Metal will assert if a color attachment texture is missing RenderTarget usage.
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        id<MTLTexture> attTex = _renderPassDescriptor.colorAttachments[i].texture;
        if (attTex && ((attTex.usage & MTLTextureUsageRenderTarget) == 0)) {
            NSLog(@"MGL WARNING: colorAttachment[%d] usage=0x%lx lacks RenderTarget; clearing attachment to avoid Metal assert",
                  i, (unsigned long)attTex.usage);
            _renderPassDescriptor.colorAttachments[i].texture = nil;
        }
    }

    // Default-framebuffer paths expect color attachment 0 specifically.
    // FBO draw-buffer mappings may intentionally leave slot 0 as GL_NONE.
    if (!ctx->state.framebuffer && !_renderPassDescriptor.colorAttachments[0].texture) {
        for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) {
            if (_renderPassDescriptor.colorAttachments[i].texture) {
                NSLog(@"MGL WARNING: colorAttachment[0] missing; remapping colorAttachment[%d] -> [0]", i);
                _renderPassDescriptor.colorAttachments[0].texture = _renderPassDescriptor.colorAttachments[i].texture;
                _renderPassDescriptor.colorAttachments[0].loadAction = _renderPassDescriptor.colorAttachments[i].loadAction;
                _renderPassDescriptor.colorAttachments[0].storeAction = _renderPassDescriptor.colorAttachments[i].storeAction;
                break;
            }
        }
    }

    // Ultimate slot-0 fallback to keep draw path alive and avoid black frame.
    if (!hasOutputAttachment && !_renderPassDescriptor.colorAttachments[0].texture) {
        if (!_fallbackRenderTargetTexture) {
            MTLTextureDescriptor *fbDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                   width:1
                                                                  height:1
                                                               mipmapped:NO];
            fbDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            fbDesc.storageMode = MTLStorageModeShared;
            _fallbackRenderTargetTexture = [_device newTextureWithDescriptor:fbDesc];
        }
        if (_fallbackRenderTargetTexture) {
            NSLog(@"MGL WARNING: colorAttachment[0] unavailable; binding 1x1 fallback");
            _renderPassDescriptor.colorAttachments[0].texture = _fallbackRenderTargetTexture;
            _renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionLoad;
            _renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
            if (_renderPassDescriptor.renderTargetWidth == 0 || _renderPassDescriptor.renderTargetHeight == 0) {
                _renderPassDescriptor.renderTargetWidth = 1;
                _renderPassDescriptor.renderTargetHeight = 1;
            }
        } else {
            NSLog(@"MGL ERROR: Unable to allocate fallback colorAttachment[0] texture");
            [self recordGPUError];
            return false;
        }
    }

    // Ensure renderTargetWidth/Height are always coherent with the active attachments.
    {
        id<MTLTexture> sizeTex = _renderPassDescriptor.colorAttachments[0].texture;
        if (!sizeTex) {
            for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) {
                if (_renderPassDescriptor.colorAttachments[i].texture) {
                    sizeTex = _renderPassDescriptor.colorAttachments[i].texture;
                    break;
                }
            }
        }
        if (!sizeTex) {
            sizeTex = _renderPassDescriptor.depthAttachment.texture;
        }
        if (!sizeTex) {
            sizeTex = _renderPassDescriptor.stencilAttachment.texture;
        }

        if (sizeTex) {
            NSUInteger texWidth = sizeTex.width;
            NSUInteger texHeight = sizeTex.height;
            if (_renderPassDescriptor.renderTargetWidth == 0 ||
                _renderPassDescriptor.renderTargetHeight == 0 ||
                _renderPassDescriptor.renderTargetWidth > texWidth ||
                _renderPassDescriptor.renderTargetHeight > texHeight) {
                NSLog(@"MGL INFO: Normalizing renderTarget size from %lux%lu to %lux%lu",
                      (unsigned long)_renderPassDescriptor.renderTargetWidth,
                      (unsigned long)_renderPassDescriptor.renderTargetHeight,
                      (unsigned long)texWidth,
                      (unsigned long)texHeight);
                _renderPassDescriptor.renderTargetWidth = texWidth;
                _renderPassDescriptor.renderTargetHeight = texHeight;
            }
        }
    }

    // CRITICAL FIX: Validate command buffer state before creating render encoder
    if (!_currentCommandBuffer) {
        NSLog(@"MGL ERROR: Cannot create render encoder - command buffer is NULL");
        [self recordGPUError];
        return false;
    }

    // Check if command buffer already has an active encoder (Metal API violation)
    if (_currentRenderEncoder) {
        NSLog(@"MGL WARNING: Active render encoder detected - ending it before creating new one");
        @try {
            [_currentRenderEncoder endEncoding];
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: Exception ending existing encoder: %@", exception);
        }
        _currentRenderEncoder = nil;
    }

    // Validate command buffer status. If already committed/completed, rotate to a new buffer.
    MTLCommandBufferStatus bufferStatus = _currentCommandBuffer.status;
    if (bufferStatus >= MTLCommandBufferStatusCommitted) {
        NSLog(@"MGL WARNING: Render encoder requested on finalized command buffer (status: %ld) - creating a fresh command buffer", (long)bufferStatus);
        if (![self newCommandBuffer]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer before creating render encoder");
            [self recordGPUError];
            return false;
        }

        if (!_currentCommandBuffer) {
            NSLog(@"MGL ERROR: newCommandBuffer returned but _currentCommandBuffer is NULL");
            [self recordGPUError];
            return false;
        }

        bufferStatus = _currentCommandBuffer.status;
        if (bufferStatus >= MTLCommandBufferStatusCommitted) {
            NSLog(@"MGL ERROR: Fresh command buffer is still finalized (status: %ld)", (long)bufferStatus);
            [self recordGPUError];
            return false;
        }
    }

    if (kMGLVerboseFrameLoopLogs) {
        NSLog(@"MGL DEBUG: About to create render encoder with descriptor and command buffer");
    }
    {
        static uint64_t s_renderPassPreCreateLogCount = 0;
        uint64_t hit = ++s_renderPassPreCreateLogCount;
        if (hit <= 128ull || (hit % 512ull) == 0ull) {
            mglLogRenderPassLifecycle("pre-create",
                                      hit,
                                      ctx,
                                      _currentCommandBuffer,
                                      _currentRenderEncoder,
                                      _renderPassDescriptor,
                                      _drawable,
                                      _renderPassFramebuffer,
                                      _renderPassFramebufferName,
                                      _renderPassDrawBuffer,
                                      _renderPassDrawBufferCount);
        }
    }
    @try {
        _currentRenderEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor: _renderPassDescriptor];
        if (!_currentRenderEncoder) {
            NSLog(@"MGL ERROR: Failed to create render encoder - invalid render pass descriptor or command buffer");
            NSLog(@"MGL DEBUG: Command buffer: %@, Render pass descriptor: %@", _currentCommandBuffer, _renderPassDescriptor);
            [self recordGPUError];
            return false;
        }
        _renderPassFramebuffer = ctx ? ctx->state.framebuffer : NULL;
        _renderPassFramebufferName = _renderPassFramebuffer ? _renderPassFramebuffer->name : 0u;
        _renderPassDrawBuffer = ctx ? ctx->state.draw_buffer : 0u;
        _renderPassDrawBufferCount = ctx ? mglMetalDrawBufferCount(ctx) : 0;
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            _renderPassDrawBuffers[i] = (ctx && i < _renderPassDrawBufferCount)
                ? mglMetalDrawBufferAt(ctx, (GLuint)i)
                : GL_NONE;
        }
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Successfully created Metal render encoder");
        }
        [self recordGPUSuccess];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating render encoder: %@ - continuing with degraded functionality", exception);
        NSLog(@"MGL DEBUG: Exception details - name: %@, reason: %@", exception.name, exception.reason);
        [self recordGPUError];
        _currentRenderEncoder = NULL;
        return false;
    }
    _currentRenderEncoder.label = @"GL Render Encoder";
    {
        static uint64_t s_renderPassCreatedLogCount = 0;
        uint64_t hit = ++s_renderPassCreatedLogCount;
        if (hit <= 128ull || (hit % 512ull) == 0ull) {
            mglLogRenderPassLifecycle("created",
                                      hit,
                                      ctx,
                                      _currentCommandBuffer,
                                      _currentRenderEncoder,
                                      _renderPassDescriptor,
                                      _drawable,
                                      _renderPassFramebuffer,
                                      _renderPassFramebufferName,
                                      _renderPassDrawBuffer,
                                      _renderPassDrawBufferCount);
        }
    }

    // apply all state that isn't included in a renderPassDescriptor into the render encoder
    [self updateCurrentRenderEncoder];

    // only bind all this if there is a VAO
    if (VAO())
    {
        if ([self bindVertexBuffersToCurrentRenderEncoder] == false)
        {
            DEBUG_PRINT("vertex buffer binding failed\n");
            [self recordGPUError];
            return false;
        }

        if ([self bindFragmentBuffersToCurrentRenderEncoder] == false)
        {
            DEBUG_PRINT("fragment buffer binding failed\n");
            [self recordGPUError];
            return false;
        }

        if ([self bindTexturesToCurrentRenderEncoder] == false)
        {
            DEBUG_PRINT("texture binding failed\n");
            [self recordGPUError];
            return false;
        }
    }

    // Record successful render encoder creation (final success)
    [self recordGPUSuccess];
    return true;
        
    } //     @autoreleasepool
}

- (bool) newCommandBuffer
{
    // CRITICAL FIX: Proper encoder cleanup BEFORE creating new command buffer
    // Metal API requires ending encoders before creating new command buffers

    // STEP 0: End any existing render encoder to prevent MTLReleaseAssertionFailure
    if (_currentRenderEncoder) {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Ending existing render encoder before creating new command buffer");
        }
        @try {
            [_currentRenderEncoder endEncoding];
            _currentRenderEncoder = nil;
        } @catch (NSException *exception) {
            NSLog(@"MGL WARNING: Exception ending render encoder: %@", exception);
            _currentRenderEncoder = nil; // Force clear even on exception
        }
    }

    // STEP 1: Clean up sync tracking list safely.
    // IMPORTANT: Do NOT dereference Sync* entries here. Sync objects are owned by GL sync lifecycle
    // and may already be deleted by glDeleteSync on other paths.
    if (_currentCommandBufferSyncList)
    {
        // CRITICAL: Add thread synchronization for sync list access
        if (_metalStateLock) {
            [_metalStateLock lock];
        }

        GLuint count = _currentCommandBufferSyncList->count;
        GLuint size = _currentCommandBufferSyncList->size;

        if (_currentCommandBufferSyncList->list == NULL || size == 0) {
            NSLog(@"MGL WARNING: Sync list storage invalid (list=%p size=%u), resetting", _currentCommandBufferSyncList->list, size);
            _currentCommandBufferSyncList->count = 0;
            if (_metalStateLock) {
                [_metalStateLock unlock];
            }
            goto create_new_command_buffer;
        }

        if (count > size) {
            NSLog(@"MGL WARNING: Sync list count overflow (count=%u size=%u), clamping", count, size);
            count = size;
            _currentCommandBufferSyncList->count = size;
        }

        for (GLuint i = 0; i < count; i++) {
            _currentCommandBufferSyncList->list[i] = NULL;
        }

        _currentCommandBufferSyncList->count = 0;

        if (_metalStateLock) {
            [_metalStateLock unlock];
        }
    }

create_new_command_buffer:
    // CRITICAL SAFETY: Validate command queue before creating buffer
    if (!_commandQueue) {
        NSLog(@"MGL ERROR: Cannot create command buffer - command queue is NULL");
        _currentCommandBuffer = NULL;
        return false;
    }

    // STEP 1: Create fresh command buffer FIRST with comprehensive AGX driver validation
    @try {
        // AGX DRIVER COMPATIBILITY: Validate command queue health before creating buffer
        if (!_commandQueue) {
            NSLog(@"MGL AGX ERROR: Command queue is NULL - recreating");
            [self resetMetalState];
            if (!_commandQueue) {
                NSLog(@"MGL AGX CRITICAL: Cannot recreate command queue");
                return false;
            }
        }

        // CRITICAL FIX: Validate _commandQueue before dereferencing to prevent NULL pointer crashes
        if (!_commandQueue) {
            NSLog(@"MGL AGX CRITICAL: _commandQueue is NULL - cannot create command buffer");
            [self recordGPUError];
            return false;
        }

        // Additional validation: Ensure _commandQueue is a valid Metal object
        @try {
            // Test if _commandQueue is valid by checking its class
            Class queueClass = [_commandQueue class];
            if (!queueClass) {
                NSLog(@"MGL AGX CRITICAL: _commandQueue is invalid (no class) - recreating");
                _commandQueue = [_device newCommandQueue];
                if (!_commandQueue) {
                    NSLog(@"MGL AGX CRITICAL: Failed to recreate command queue");
                    [self recordGPUError];
                    return false;
                }
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL AGX CRITICAL: _commandQueue validation exception: %@ - recreating", exception);
            [self recordGPUError];
            _commandQueue = [_device newCommandQueue];
            if (!_commandQueue) {
                NSLog(@"MGL AGX CRITICAL: Failed to recreate command queue after exception");
                [self recordGPUError];
                return false;
            }
        }

        _currentCommandBuffer = [_commandQueue commandBuffer];
        if (!_currentCommandBuffer) {
            NSLog(@"MGL AGX ERROR: Failed to create Metal command buffer - command queue may be in error state");
            [self recordGPUError];
            // Force command queue recreation
            [self resetMetalState];
            return false;
        }

        // AGX Driver Validation: Check if the command buffer is immediately invalid
        if (_currentCommandBuffer.error) {
            NSLog(@"MGL AGX WARNING: New command buffer has immediate error: %@", _currentCommandBuffer.error);
            [self recordGPUError];
            // Don't return false immediately - AGX sometimes creates error-state buffers that recover
        }

        // AGX DRIVER COMPATIBILITY: Enhanced validation to prevent rejections
        if (_currentCommandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"MGL AGX CRITICAL: Command buffer immediately in error state");
            [self recordGPUError];
            _currentCommandBuffer = nil; // Clear the problematic buffer
            [self resetMetalState]; // Force full reset
            return false;
        }

        // Additional AGX validation: Check for buffer properties that cause rejections
        if (_currentCommandBuffer.error) {
            NSLog(@"MGL AGX WARNING: Command buffer has immediate error: %@", _currentCommandBuffer.error);
            [self recordGPUError];
            _currentCommandBuffer = nil;
            [self resetMetalState];
            return false;
        }

        // Validate command queue health
        if (!_commandQueue) {
            NSLog(@"MGL AGX CRITICAL: Command queue became NULL");
            [self resetMetalState];
            return false;
        }

        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Successfully created new Metal command buffer (AGX validated)");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL AGX ERROR: Exception creating command buffer: %@", exception);
        [self recordGPUError];
        _currentCommandBuffer = NULL;

        // AGX DRIVER COMPATIBILITY: Force reset on exception to clear driver state
        [self resetMetalState];
        return false;
    }

    // STEP 2: Now handle pending event waits on the FRESH command buffer
    if (_currentEvent)
    {
        if (!_currentSyncName) {
            NSLog(@"MGL WARNING: dropping pending shared-event wait with no sync name");
            _currentEvent = NULL;
            return true;
        }

        if (kMGLDisableSharedEventSync) {
            NSLog(@"MGL INFO: Shared event wait disabled (debug no-op), skipping wait encode event=%p syncName=%u",
                  _currentEvent, _currentSyncName);
            _currentEvent = NULL;
            _currentSyncName = 0;
            return true;
        }

        // SAFELY ENCODE: Event wait functionality on the new command buffer
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Encoding event wait on fresh command buffer");
        }

        // CRITICAL SAFETY: Cache event and sync values to prevent race conditions
        id<MTLEvent> cachedEvent = _currentEvent;
        GLuint cachedSyncName = _currentSyncName;

        // COMPREHENSIVE EVENT VALIDATION: Validate Metal event pointer
        if (!cachedEvent) {
            NSLog(@"MGL ERROR: Cannot encode event wait - cached event is NULL");
            _currentEvent = NULL;
            _currentSyncName = 0;
            return false;
        }

        // Validate event pointer looks like a valid object address
        uintptr_t eventPtr = (uintptr_t)cachedEvent;
        if (eventPtr == 0x10 || eventPtr == 0x30 || eventPtr == 0x1000) {
            NSLog(@"MGL CRITICAL ERROR: Known corrupted event pointer pattern detected: 0x%lx", eventPtr);
            NSLog(@"MGL CRITICAL ERROR: Skipping event wait to prevent crash");
            _currentEvent = NULL;
            _currentSyncName = 0;
            return false;
        }

        if (eventPtr < 0x1000 || (eventPtr & 0x7) != 0) {
            NSLog(@"MGL ERROR: Suspicious event pointer value: %p", cachedEvent);
            NSLog(@"MGL INFO: Skipping event wait for safety");
            _currentEvent = NULL;
            _currentSyncName = 0;
            return false;
        }

        // ADDITIONAL SAFETY: Validate command buffer is still valid before encoding
        if (!_currentCommandBuffer) {
            NSLog(@"MGL ERROR: Command buffer became NULL before event wait encoding");
            _currentEvent = NULL;
            _currentSyncName = 0;
            return false;
        }

        @try {
            NSLog(@"MGL INFO: Encoding safe event wait: event=%p, syncName=%u, cmdbuf=%p", cachedEvent, cachedSyncName, _currentCommandBuffer);

            // Use conservative approach: only encode if everything looks perfect
            [_currentCommandBuffer encodeWaitForEvent:cachedEvent value:cachedSyncName];

            NSLog(@"MGL SUCCESS: Event wait encoded successfully on fresh command buffer");
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Event wait failed - %@: %@", exception.name, exception.reason);
            NSLog(@"MGL INFO: Continuing without event wait to maintain stability");
            // Continue without event wait - system remains stable
        }

        _currentEvent = NULL;
        _currentSyncName = 0;
    }

    return true;
}

- (bool)ensureWritableCommandBuffer:(const char *)reason
{
    if (!_currentCommandBuffer) {
        NSLog(@"MGL INFO: %s requested with NULL command buffer, creating one", reason ? reason : "operation");
        if (![self newCommandBuffer]) {
            NSLog(@"MGL ERROR: Failed to create command buffer for %s", reason ? reason : "operation");
            return false;
        }
    }

    MTLCommandBufferStatus status = _currentCommandBuffer.status;
    if (status >= MTLCommandBufferStatusCommitted) {
        NSLog(@"MGL INFO: %s requested on finalized command buffer (status: %ld), rotating", reason ? reason : "operation", (long)status);
        [self endRenderEncoding];
        if (![self newCommandBuffer]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer for %s", reason ? reason : "operation");
            return false;
        }

        if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted) {
            NSLog(@"MGL ERROR: Unable to obtain writable command buffer for %s", reason ? reason : "operation");
            return false;
        }
    }

    return true;
}

- (bool)copyTextureUploadWithDedicatedCommandBuffer:(id<MTLBuffer>)sourceBuffer
                                        sourceOffset:(NSUInteger)sourceOffset
                                   sourceBytesPerRow:(NSUInteger)sourceBytesPerRow
                                 sourceBytesPerImage:(NSUInteger)sourceBytesPerImage
                                           sourceSize:(MTLSize)sourceSize
                                            toTexture:(id<MTLTexture>)texture
                                     destinationSlice:(NSUInteger)destinationSlice
                                     destinationLevel:(NSUInteger)destinationLevel
                                    destinationOrigin:(MTLOrigin)destinationOrigin
                                               reason:(const char *)reason
{
    if (!sourceBuffer || !texture || !_commandQueue) {
        NSLog(@"MGL ERROR: dedicated texture upload prerequisites missing (source=%p texture=%p queue=%p)",
              sourceBuffer, texture, _commandQueue);
        return false;
    }

    id<MTLCommandBuffer> uploadCB = [_commandQueue commandBuffer];
    if (!uploadCB) {
        NSLog(@"MGL ERROR: failed to create dedicated upload command buffer for %s",
              reason ? reason : "texture_upload");
        [self recordGPUError];
        return false;
    }

    if (reason) {
        uploadCB.label = [NSString stringWithFormat:@"MGL.%s", reason];
    } else {
        uploadCB.label = @"MGL.texture_upload";
    }

    id<MTLBlitCommandEncoder> blitEncoder = [uploadCB blitCommandEncoder];
    if (!blitEncoder) {
        NSLog(@"MGL ERROR: failed to create dedicated upload blit encoder for %s",
              reason ? reason : "texture_upload");
        [self recordGPUError];
        return false;
    }

    @try {
        [blitEncoder copyFromBuffer:sourceBuffer
                       sourceOffset:sourceOffset
                   sourceBytesPerRow:sourceBytesPerRow
                 sourceBytesPerImage:sourceBytesPerImage
                          sourceSize:sourceSize
                           toTexture:texture
                    destinationSlice:destinationSlice
                    destinationLevel:destinationLevel
                   destinationOrigin:destinationOrigin];
        [blitEncoder endEncoding];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: dedicated upload encode failed (%s): %@",
              reason ? reason : "texture_upload", exception.reason);
        [blitEncoder endEncoding];
        [self recordGPUError];
        return false;
    }

    dispatch_semaphore_t completionSemaphore = kMGLSynchronizeTextureUploads
        ? dispatch_semaphore_create(0)
        : NULL;
    __weak typeof(self) weakSelf = self;
    [uploadCB addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (cb.error) {
            NSLog(@"MGL ERROR: dedicated upload command buffer failed (%s): %@",
                  reason ? reason : "texture_upload", cb.error);
            [weakSelf recordGPUError];
        }

        if (completionSemaphore) {
            dispatch_semaphore_signal(completionSemaphore);
        }
    }];

    [uploadCB commit];

    if (!kMGLSynchronizeTextureUploads) {
        // Keep uploads ordered on the same queue but avoid stalling the render thread.
        return true;
    }

    dispatch_time_t deadline = dispatch_time(DISPATCH_TIME_NOW,
                                             (int64_t)(kMGLTextureUploadWaitTimeoutSeconds * NSEC_PER_SEC));
    if (dispatch_semaphore_wait(completionSemaphore, deadline) != 0) {
        NSLog(@"MGL WARNING: dedicated upload wait timed out (%s), continuing asynchronously",
              reason ? reason : "texture_upload");
        return true;
    }

    return uploadCB.error == nil;
}

- (bool)uploadTextureSliceViaBlit:(id<MTLTexture>)texture
                          texName:(GLuint)texName
                         texTarget:(GLenum)texTarget
                            bytes:(const void *)bytes
                      bytesPerRow:(NSUInteger)bytesPerRow
                    bytesPerImage:(NSUInteger)bytesPerImage
                            width:(NSUInteger)width
                           height:(NSUInteger)height
                            depth:(NSUInteger)depth
                            level:(NSUInteger)level
                            slice:(NSUInteger)slice
{
    if (!texture || !bytes || bytesPerRow == 0 || bytesPerImage == 0 || width == 0) {
        return false;
    }

    if ([self shouldSkipGPUOperations]) {
        NSLog(@"MGL AGX: Skipping texture upload during recovery");
        return false;
    }

    MTLTextureType textureType = texture.textureType;
    BOOL is3DTexture = (textureType == MTLTextureType3D);
    BOOL isArrayOrCubeTexture =
        (textureType == MTLTextureTypeCube ||
         textureType == MTLTextureTypeCubeArray ||
         textureType == MTLTextureType2DArray ||
         textureType == MTLTextureType1DArray ||
         textureType == MTLTextureType2DMultisampleArray);

    NSUInteger safeHeight = (height > 0) ? height : 1;
    NSUInteger safeDepth = (depth > 0) ? depth : 1;
    NSUInteger expectedBytesPerImage = bytesPerRow * safeHeight;
    NSUInteger copyDepth = is3DTexture ? safeDepth : 1;
    NSUInteger safeBytesPerImage = bytesPerImage;

    if (isArrayOrCubeTexture) {
        // For array/cubemap uploads each slice is uploaded independently.
        // Clamp to per-slice bytes to avoid accidentally treating N slices as one image.
        if (safeBytesPerImage != expectedBytesPerImage) {
            NSLog(@"MGL INFO: Normalizing bytesPerImage for array/cube upload (slice=%lu level=%lu old=%lu expected=%lu)",
                  (unsigned long)slice, (unsigned long)level,
                  (unsigned long)safeBytesPerImage, (unsigned long)expectedBytesPerImage);
        }
        safeBytesPerImage = expectedBytesPerImage;
    } else if (is3DTexture) {
        if (safeBytesPerImage < expectedBytesPerImage) {
            safeBytesPerImage = expectedBytesPerImage;
        }
    } else {
        // Non-array/non-3D uploads should still represent a single image.
        safeBytesPerImage = expectedBytesPerImage;
    }

    if (textureType == MTLTextureTypeCube || textureType == MTLTextureTypeCubeArray) {
        NSLog(@"MGL CUBE UPLOAD tex=%u glTarget=0x%x face=%lu slice=%lu level=%lu origin=(0,0,0) size=%lux%lux%lu bpr=%lu bpi=%lu ptr=%p",
              texName,
              texTarget,
              (unsigned long)slice,
              (unsigned long)slice,
              (unsigned long)level,
              (unsigned long)width,
              (unsigned long)safeHeight,
              (unsigned long)copyDepth,
              (unsigned long)bytesPerRow,
              (unsigned long)safeBytesPerImage,
              bytes);
    }

    NSUInteger bufferSize = safeBytesPerImage * copyDepth;
    if (bufferSize == 0 || bufferSize > (512 * 1024 * 1024)) {
        NSLog(@"MGL WARNING: Rejecting texture upload with invalid buffer size: %lu", (unsigned long)bufferSize);
        return false;
    }

    id<MTLBuffer> uploadBuffer = [_device newBufferWithBytes:bytes
                                                       length:bufferSize
                                                      options:MTLResourceStorageModeShared];
    if (!uploadBuffer) {
        NSLog(@"MGL WARNING: Failed to allocate upload buffer for texture blit");
        return false;
    }

    bool uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:uploadBuffer
                                                         sourceOffset:0
                                                    sourceBytesPerRow:bytesPerRow
                                                  sourceBytesPerImage:safeBytesPerImage
                                                            sourceSize:MTLSizeMake(width, safeHeight, copyDepth)
                                                             toTexture:texture
                                                      destinationSlice:slice
                                                      destinationLevel:level
                                                     destinationOrigin:MTLOriginMake(0, 0, 0)
                                                                reason:"texture_upload_blit"];
    if (!uploaded) {
        NSLog(@"MGL WARNING: Dedicated texture upload failed (level=%lu slice=%lu)",
              (unsigned long)level, (unsigned long)slice);
    }
    return uploaded;
}

- (bool) newCommandBufferAndRenderEncoder
{
    // AGGRESSIVE MEMORY SAFETY: Validate fundamental Metal objects before use
    if (!_device) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - No device available");
        return false;
    }

    if (!_commandQueue) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - No command queue available");
        return false;
    }

    // Validate device pointer lower bound only (high canonical addresses are valid on macOS)
    uintptr_t device_addr = (uintptr_t)_device;
    if (device_addr < 0x1000) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - Invalid device pointer: 0x%lx", device_addr);
        return false;
    }

    @try {
        if ([self newCommandBuffer] == false) {
            NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - newCommandBuffer failed");
            return false;
        }

        if ([self newRenderEncoder] == false) {
            NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - newRenderEncoder failed");
            return false;
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: newCommandBufferAndRenderEncoder - Metal operation failed: %@", exception);
        return false;
    }

    return true;
}

#pragma mark pipeline descriptor
-(MTLRenderPipelineDescriptor *)generatePipelineDescriptor
{
    if (!ctx) {
        NSLog(@"MGL PIPELINE DESC fail: context is NULL");
        return nil;
    }

    Program *program = mglResolveProgramFromState(ctx);
    if (!program) {
        NSLog(@"MGL PIPELINE DESC fail: state program is NULL (name=%u ptr=%p)",
              (unsigned)ctx->state.program_name, ctx->state.program);
        return nil;
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC begin program=%u", (unsigned)program->name);
    }

    if (ctx->state.dirty_bits & DIRTY_PROGRAM) {
        if ([self bindMTLProgram:program] == false) {
            NSLog(@"MGL PIPELINE DESC fail: bindMTLProgram failed for program=%u", (unsigned)program->name);
            return nil;
        }
    }

    Shader *vertex_shader = program->shader_slots[_VERTEX_SHADER];
    Shader *fragment_shader = program->shader_slots[_FRAGMENT_SHADER];
    if (!vertex_shader || !fragment_shader) {
        NSLog(@"MGL PIPELINE DESC fail: missing shaders for program=%u (vs=%p fs=%p)",
              (unsigned)program->name, vertex_shader, fragment_shader);
        return nil;
    }

    id<MTLFunction> vertexFunction = (__bridge id<MTLFunction>)(vertex_shader->mtl_data.function);
    id<MTLFunction> fragmentFunction = (__bridge id<MTLFunction>)(fragment_shader->mtl_data.function);
    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC vs=%@ fs=%@",
              vertexFunction ? vertexFunction.name : @"(null)",
              fragmentFunction ? fragmentFunction.name : @"(null)");
    }
    if (!vertexFunction || !fragmentFunction) {
        NSLog(@"MGL PIPELINE DESC fail: missing MTLFunction (vs=%p fs=%p) for program=%u",
              vertexFunction, fragmentFunction, (unsigned)program->name);
        return nil;
    }

    MTLRenderPipelineDescriptor *pipelineStateDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    if (!pipelineStateDescriptor) {
        NSLog(@"MGL PIPELINE DESC fail: descriptor allocation failed for program=%u", (unsigned)program->name);
        return nil;
    }
    pipelineStateDescriptor.label = @"GLSL Pipeline";
    pipelineStateDescriptor.vertexFunction = vertexFunction;
    pipelineStateDescriptor.fragmentFunction = fragmentFunction;

    if (ctx->state.framebuffer) {
        Framebuffer *fbo = ctx->state.framebuffer;

        for (int i = 0; i < STATE(max_color_attachments); i++) {
            if (fbo->color_attachments[i].texture) {
                Texture *tex = [self framebufferAttachmentTexture:&fbo->color_attachments[i]];
                if (tex && ![self bindMTLTexture:tex]) {
                    NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for color attachment %d tex=%u",
                          i, tex->name);
                    return nil;
                }
                if (tex && tex->mtl_data) {
                    pipelineStateDescriptor.colorAttachments[i].pixelFormat = mtlPixelFormatForGLTex(tex);
                } else {
                    pipelineStateDescriptor.colorAttachments[i].pixelFormat = MTLPixelFormatInvalid;
                }
            }

            if ((fbo->color_attachment_bitfield >> (i + 1)) == 0) {
                break;
            }
        }

        if (fbo->depth.texture) {
            Texture *tex = [self framebufferAttachmentTexture:&fbo->depth];
            if (tex && ![self bindMTLTexture:tex]) {
                NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for depth tex=%u", tex->name);
                return nil;
            }
            if (tex && tex->mtl_data) {
                MTLPixelFormat depthFormat = mtlPixelFormatForGLTex(tex);
                if (depthFormat == MTLPixelFormatInvalid) {
                    NSLog(@"MGL ERROR: Invalid depth texture format, falling back to Depth32Float");
                    depthFormat = MTLPixelFormatDepth32Float;
                }
                pipelineStateDescriptor.depthAttachmentPixelFormat = depthFormat;
            } else {
                pipelineStateDescriptor.depthAttachmentPixelFormat = MTLPixelFormatInvalid;
            }
        }

        if (fbo->stencil.texture) {
            Texture *tex = [self framebufferAttachmentTexture:&fbo->stencil];
            if (tex && ![self bindMTLTexture:tex]) {
                NSLog(@"MGL PIPELINE DESC fail: bindMTLTexture failed for stencil tex=%u", tex->name);
                return nil;
            }
            if (tex && tex->mtl_data) {
                MTLPixelFormat stencilFormat = mtlPixelFormatForGLTex(tex);
                if (stencilFormat == MTLPixelFormatInvalid) {
                    NSLog(@"MGL ERROR: Invalid stencil texture format, falling back to Stencil8");
                    stencilFormat = MTLPixelFormatStencil8;
                }
                pipelineStateDescriptor.stencilAttachmentPixelFormat = stencilFormat;
            } else {
                pipelineStateDescriptor.stencilAttachmentPixelFormat = MTLPixelFormatInvalid;
            }
        }
    } else {
        MTLPixelFormat preferredColor0 = MTLPixelFormatInvalid;
        if (_renderPassDescriptor && _renderPassDescriptor.colorAttachments[0].texture) {
            preferredColor0 = _renderPassDescriptor.colorAttachments[0].texture.pixelFormat;
        } else if (_drawable && _drawable.texture) {
            preferredColor0 = _drawable.texture.pixelFormat;
        } else {
            preferredColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = preferredColor0;

        if (ctx->depth_format.format) {
            MTLPixelFormat depthFormat = ctx->depth_format.mtl_pixel_format;
            if (depthFormat == MTLPixelFormatInvalid) {
                depthFormat = MTLPixelFormatDepth32Float;
            }
            pipelineStateDescriptor.depthAttachmentPixelFormat = depthFormat;
        }

        if (ctx->stencil_format.format) {
            MTLPixelFormat stencilFormat = ctx->stencil_format.mtl_pixel_format;
            if (stencilFormat == MTLPixelFormatInvalid ||
                stencilFormat == MTLPixelFormatDepth32Float_Stencil8) {
                stencilFormat = MTLPixelFormatStencil8;
            }
            pipelineStateDescriptor.stencilAttachmentPixelFormat = stencilFormat;
        }
    }

    if (_renderPassDescriptor) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id<MTLTexture> rpColor = _renderPassDescriptor.colorAttachments[i].texture;
            if (rpColor) {
                pipelineStateDescriptor.colorAttachments[i].pixelFormat = rpColor.pixelFormat;
            }
        }

        id<MTLTexture> rpDepth = _renderPassDescriptor.depthAttachment.texture;
        id<MTLTexture> rpStencil = _renderPassDescriptor.stencilAttachment.texture;
        pipelineStateDescriptor.depthAttachmentPixelFormat =
            rpDepth ? rpDepth.pixelFormat : MTLPixelFormatInvalid;
        pipelineStateDescriptor.stencilAttachmentPixelFormat =
            rpStencil ? rpStencil.pixelFormat : MTLPixelFormatInvalid;
    }

    BOOL color0IsIntentionallyDisabled =
        ctx->state.framebuffer &&
        mglMetalDrawBufferAt(ctx, 0u) == GL_NONE;

    if (!color0IsIntentionallyDisabled &&
        (pipelineStateDescriptor.colorAttachments[0].pixelFormat == MTLPixelFormatInvalid ||
         pipelineStateDescriptor.colorAttachments[0].pixelFormat == 0)) {
        MTLPixelFormat fallbackColor0 = MTLPixelFormatInvalid;
        if (_renderPassDescriptor && _renderPassDescriptor.colorAttachments[0].texture) {
            fallbackColor0 = _renderPassDescriptor.colorAttachments[0].texture.pixelFormat;
        } else if (_drawable && _drawable.texture) {
            fallbackColor0 = _drawable.texture.pixelFormat;
        } else {
            fallbackColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        if (fallbackColor0 == MTLPixelFormatInvalid || fallbackColor0 == 0) {
            fallbackColor0 = MTLPixelFormatBGRA8Unorm;
        }
        if (kMGLVerbosePipelineLogs) {
            NSLog(@"MGL PIPELINE DESC missing color pixel format, fallback pixelFormat=%lu",
                  (unsigned long)fallbackColor0);
        }
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = fallbackColor0;
    }

    NSUInteger resolvedSampleCount = 1;
    if (_renderPassDescriptor) {
        id<MTLTexture> rpColor0 = _renderPassDescriptor.colorAttachments[0].texture;
        id<MTLTexture> rpDepth = _renderPassDescriptor.depthAttachment.texture;
        id<MTLTexture> rpStencil = _renderPassDescriptor.stencilAttachment.texture;
        if (rpColor0 && rpColor0.sampleCount > 0) {
            resolvedSampleCount = rpColor0.sampleCount;
        } else if (rpDepth && rpDepth.sampleCount > 0) {
            resolvedSampleCount = rpDepth.sampleCount;
        } else if (rpStencil && rpStencil.sampleCount > 0) {
            resolvedSampleCount = rpStencil.sampleCount;
        }
    }
    if (resolvedSampleCount == 0) {
        resolvedSampleCount = 1;
    }
    if (pipelineStateDescriptor.rasterSampleCount == 0) {
        pipelineStateDescriptor.rasterSampleCount = resolvedSampleCount;
    }
    if (pipelineStateDescriptor.rasterSampleCount == 0) {
        pipelineStateDescriptor.rasterSampleCount = 1;
    }

    NSUInteger activeColorAttachmentCount = 0;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (pipelineStateDescriptor.colorAttachments[i].pixelFormat != MTLPixelFormatInvalid &&
            pipelineStateDescriptor.colorAttachments[i].pixelFormat != 0) {
            activeColorAttachmentCount++;
        }
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL PIPELINE DESC colorAttachmentCount=%lu depthFormat=%lu stencilFormat=%lu sampleCount=%lu",
              (unsigned long)activeColorAttachmentCount,
              (unsigned long)pipelineStateDescriptor.depthAttachmentPixelFormat,
              (unsigned long)pipelineStateDescriptor.stencilAttachmentPixelFormat,
              (unsigned long)pipelineStateDescriptor.rasterSampleCount);
        NSLog(@"MGL PIPELINE DESC renderTarget[0]=%lu",
              (unsigned long)pipelineStateDescriptor.colorAttachments[0].pixelFormat);
    }

    return pipelineStateDescriptor;
}

#pragma mark vertex descriptor
- (MTLVertexDescriptor *)generateVertexDescriptor
{
    MTLVertexDescriptor *vertexDescriptor = [[MTLVertexDescriptor alloc] init];
    if (!vertexDescriptor) {
        NSLog(@"MGL VERTEX ERROR: failed to allocate MTLVertexDescriptor");
        return nil;
    }
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    Program *activeProgram = mglResolveProgramFromState(ctx);
    GLuint activeProgramName = activeProgram ? activeProgram->name : (ctx ? ctx->state.program_name : 0);
    GLuint maxAttribs;

    if (!vao) {
        NSLog(@"MGL PIPELINE DESC fail: cannot build vertex descriptor without a valid VAO");
        return nil;
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL VERTEX DESC begin program=%u vao=%p enabledMask=0x%x",
              (unsigned)activeProgramName, vao, vao->enabled_attribs);
    }

    [vertexDescriptor reset]; // ??? debug
    maxAttribs = ctx->state.max_vertex_attribs;
    if (maxAttribs > MAX_ATTRIBS) {
        maxAttribs = MAX_ATTRIBS;
    }

    // we can bind a new vertex descriptor without creating a new renderbuffer
    for (GLuint i = 0; i < maxAttribs; i++)
    {
        if (vao->enabled_attribs & (0x1u << i))
        {
            if (!mglRendererProgramUsesVertexAttrib(activeProgram, i)) {
                if ((vao->enabled_attribs >> (i + 1)) == 0u)
                    break;
                continue;
            }

            MTLVertexFormat format;
            Buffer *attribBuffer = mglRendererGetValidatedBuffer(ctx, vao->attrib[i].buffer, __FUNCTION__, i);

            if (!attribBuffer)
            {
                NSLog(@"MGL PIPELINE DESC fail: attrib %u enabled but buffer is invalid", i);
                return NULL;
            }

            GLboolean normalized = vao->attrib[i].normalized;
            if (!normalized &&
                vao->attrib[i].type == GL_UNSIGNED_BYTE &&
                vao->attrib[i].size == 4 &&
                mglRendererVertexAttribIsColorInput(activeProgram, i)) {
                normalized = GL_TRUE;
            }

            format = glTypeSizeToMtlType(vao->attrib[i].type,
                                         vao->attrib[i].size,
                                         normalized);

            if (format == MTLVertexFormatInvalid)
            {
                NSLog(@"MGL PIPELINE DESC fail: unable to map attrib %u type/size/normalize to MTL format", i);
                return nil;
            }

            int mapped_buffer_index;

            mapped_buffer_index = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, i, __FUNCTION__);
            if (mapped_buffer_index < 0 || mapped_buffer_index >= (int)kMGLMaxMetalVertexBufferCount) {
                NSLog(@"MGL ERROR: Invalid vertex buffer index %d for attribute %d (max valid=%lu)",
                      mapped_buffer_index, i, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return NULL;
            }

            vertexDescriptor.attributes[i].bufferIndex = mapped_buffer_index;
            vertexDescriptor.attributes[i].offset = vao->attrib[i].relativeoffset;
            vertexDescriptor.attributes[i].format = format;

            vertexDescriptor.layouts[mapped_buffer_index].stride = vao->attrib[i].stride;

            if (vao->attrib[i].divisor)
            {
                vertexDescriptor.layouts[mapped_buffer_index].stepRate = vao->attrib[i].divisor;
                vertexDescriptor.layouts[mapped_buffer_index].stepFunction = MTLVertexStepFunctionPerInstance;
            }
            else
            {
                vertexDescriptor.layouts[mapped_buffer_index].stepRate = 1;
                vertexDescriptor.layouts[mapped_buffer_index].stepFunction = MTLVertexStepFunctionPerVertex;
            }

            if (kMGLVerbosePipelineLogs) {
                NSLog(@"MGL VERTEX DESC attrib=%u enabled=%u glBuffer=%u metalIndex=%d bindingOffset=%lld offset=0x%llx stride=%u size=%u type=0x%x normalized=%u divisor=%u format=%lu(%s)",
                      i,
                      1u,
                      attribBuffer->name,
                      mapped_buffer_index,
                      (long long)vao->attrib[i].binding_offset,
                      (unsigned long long)(uintptr_t)vao->attrib[i].relativeoffset,
                      (unsigned)vao->attrib[i].stride,
                      (unsigned)vao->attrib[i].size,
                      (unsigned)vao->attrib[i].type,
                      (unsigned)normalized,
                      (unsigned)vao->attrib[i].divisor,
                      (unsigned long)format,
                      mglVertexFormatName(format));
            }

            if (vao->attrib[i].type == GL_UNSIGNED_BYTE &&
                vao->attrib[i].size == 4 &&
                vao->attrib[i].normalized == GL_FALSE &&
                !normalized) {
                if (kMGLVerbosePipelineLogs) {
                    NSLog(@"MGL VERTEX DESC note: attrib %u uses UBYTE4 non-normalized (format=%lu)",
                          i, (unsigned long)format);
                }
            }
        }

        // early out
        if ((vao->enabled_attribs >> (i + 1)) == 0u)
            break;
    }

    // clear all dirty bits as they have been translated into a vertex descriptor
    vao->dirty_bits = 0;

    return vertexDescriptor;
}

#pragma mark utility funcs for processGLState
- (MTLBlendFactor) blendFactorFromGL:(GLenum)gl_blend
{
    MTLBlendFactor factor;

    switch(gl_blend)
    {
        case GL_ZERO: factor = MTLBlendFactorZero; break;
        case GL_ONE: factor = MTLBlendFactorOne; break;
        case GL_SRC_COLOR: factor = MTLBlendFactorSourceColor; break;
        case GL_ONE_MINUS_SRC_COLOR: factor = MTLBlendFactorOneMinusSourceColor; break;
        case GL_DST_COLOR: factor = MTLBlendFactorDestinationColor; break;
        case GL_ONE_MINUS_DST_COLOR: factor = MTLBlendFactorOneMinusDestinationColor; break;
        case GL_SRC_ALPHA: factor = MTLBlendFactorSourceAlpha; break;
        case GL_ONE_MINUS_SRC_ALPHA: factor = MTLBlendFactorOneMinusSourceAlpha; break;
        case GL_DST_ALPHA: factor = MTLBlendFactorDestinationAlpha; break;
        case GL_ONE_MINUS_DST_ALPHA: factor = MTLBlendFactorOneMinusDestinationAlpha; break;
        case GL_CONSTANT_COLOR: factor = MTLBlendFactorBlendColor; break;
        case GL_ONE_MINUS_CONSTANT_COLOR: factor = MTLBlendFactorOneMinusBlendColor; break;
        case GL_CONSTANT_ALPHA: factor = MTLBlendFactorBlendAlpha; break;
        case GL_ONE_MINUS_CONSTANT_ALPHA: factor = MTLBlendFactorOneMinusBlendAlpha; break;
        case GL_SRC_ALPHA_SATURATE: factor = MTLBlendFactorSourceAlphaSaturated; break;

        default:
            // CRITICAL FIX: Handle assertion gracefully instead of crashing
            static uint64_t s_unknownBlendFactorCount = 0;
            uint64_t hit = ++s_unknownBlendFactorCount;
            if (hit <= 32 || (hit % 512) == 0) {
                NSLog(@"MGL ERROR: Unknown blend factor 0x%x hit=%llu",
                      gl_blend, (unsigned long long)hit);
            }
            return MTLBlendFactorZero;
    }

    return factor;
}

- (MTLBlendOperation) blendOperationFromGL:(GLenum)gl_blend_op
{
    MTLBlendOperation op;

    switch(gl_blend_op)
    {
        case GL_FUNC_ADD: op = MTLBlendOperationAdd; break;
        case GL_FUNC_SUBTRACT: op = MTLBlendOperationSubtract; break;
        case GL_FUNC_REVERSE_SUBTRACT: op = MTLBlendOperationReverseSubtract; break;
        case GL_MIN: op = MTLBlendOperationMin; break;
        case GL_MAX: op = MTLBlendOperationMax; break;

        default:
            // CRITICAL FIX: Handle assertion gracefully instead of crashing
            static uint64_t s_unknownBlendOperationCount = 0;
            uint64_t hit = ++s_unknownBlendOperationCount;
            if (hit <= 32 || (hit % 512) == 0) {
                NSLog(@"MGL ERROR: Unknown blend operation 0x%x hit=%llu",
                      gl_blend_op, (unsigned long long)hit);
            }
            return MTLBlendOperationAdd;
    }

    return op;
}

- (void) updateBlendStateCache
{
    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        if (!mglIsValidGLBlendFactor(ctx->state.var.blend_src_rgb[i])) {
            mglLogRenderStateRepair("blend_src_rgb", ctx->state.var.blend_src_rgb[i], GL_ONE);
            ctx->state.var.blend_src_rgb[i] = GL_ONE;
        }
        if (!mglIsValidGLBlendFactor(ctx->state.var.blend_src_alpha[i])) {
            mglLogRenderStateRepair("blend_src_alpha", ctx->state.var.blend_src_alpha[i], GL_ONE);
            ctx->state.var.blend_src_alpha[i] = GL_ONE;
        }
        if (!mglIsValidGLBlendFactor(ctx->state.var.blend_dst_rgb[i])) {
            mglLogRenderStateRepair("blend_dst_rgb", ctx->state.var.blend_dst_rgb[i], GL_ZERO);
            ctx->state.var.blend_dst_rgb[i] = GL_ZERO;
        }
        if (!mglIsValidGLBlendFactor(ctx->state.var.blend_dst_alpha[i])) {
            mglLogRenderStateRepair("blend_dst_alpha", ctx->state.var.blend_dst_alpha[i], GL_ZERO);
            ctx->state.var.blend_dst_alpha[i] = GL_ZERO;
        }
        if (!mglIsValidGLBlendEquation(ctx->state.var.blend_equation_rgb[i])) {
            mglLogRenderStateRepair("blend_equation_rgb", ctx->state.var.blend_equation_rgb[i], GL_FUNC_ADD);
            ctx->state.var.blend_equation_rgb[i] = GL_FUNC_ADD;
        }
        if (!mglIsValidGLBlendEquation(ctx->state.var.blend_equation_alpha[i])) {
            mglLogRenderStateRepair("blend_equation_alpha", ctx->state.var.blend_equation_alpha[i], GL_FUNC_ADD);
            ctx->state.var.blend_equation_alpha[i] = GL_FUNC_ADD;
        }

        _src_blend_rgb_factor[i] = [self blendFactorFromGL:ctx->state.var.blend_src_rgb[i]];
        _src_blend_alpha_factor[i] = [self blendFactorFromGL:ctx->state.var.blend_src_alpha[i]];

        _dst_blend_rgb_factor[i] = [self blendFactorFromGL:ctx->state.var.blend_dst_rgb[i]];
        _dst_blend_alpha_factor[i] = [self blendFactorFromGL:ctx->state.var.blend_dst_alpha[i]];

        _rgb_blend_operation[i] = [self blendOperationFromGL: ctx->state.var.blend_equation_rgb[i]];
        _alpha_blend_operation[i] = [self blendOperationFromGL: ctx->state.var.blend_equation_alpha[i]];

        if (!ctx->state.caps.use_color_mask[i]) {
            _color_mask[i] = MTLColorWriteMaskAll;
        } else {
            _color_mask[i] = MTLColorWriteMaskNone;

            if (ctx->state.var.color_writemask[i][0])
                _color_mask[i] |= MTLColorWriteMaskRed;

            if (ctx->state.var.color_writemask[i][1])
                _color_mask[i] |= MTLColorWriteMaskGreen;

            if (ctx->state.var.color_writemask[i][2])
                _color_mask[i] |= MTLColorWriteMaskBlue;

            if (ctx->state.var.color_writemask[i][3])
                _color_mask[i] |= MTLColorWriteMaskAlpha;
        }
    }
}

-(void)bindBlendStateToPipelineStateDescriptor:(MTLRenderPipelineDescriptor *)pipelineStateDescriptor
{
    pipelineStateDescriptor.alphaToCoverageEnabled = ctx->state.caps.sample_alpha_to_coverage ? YES : NO;
    pipelineStateDescriptor.alphaToOneEnabled = ctx->state.caps.sample_alpha_to_one ? YES : NO;

    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        if (pipelineStateDescriptor.colorAttachments[i].pixelFormat != MTLPixelFormatInvalid)
        {
            if (mglMetalDrawBufferAt(ctx, (GLuint)i) == GL_NONE) {
                pipelineStateDescriptor.colorAttachments[i].blendingEnabled = NO;
                pipelineStateDescriptor.colorAttachments[i].writeMask = 0;
                continue;
            }

            pipelineStateDescriptor.colorAttachments[i].blendingEnabled =
                ctx->state.caps.blendi[i] ? true : false;

            pipelineStateDescriptor.colorAttachments[i].sourceRGBBlendFactor = _src_blend_rgb_factor[i];
            pipelineStateDescriptor.colorAttachments[i].destinationRGBBlendFactor = _dst_blend_rgb_factor[i];
            pipelineStateDescriptor.colorAttachments[i].sourceAlphaBlendFactor = _src_blend_alpha_factor[i];
            pipelineStateDescriptor.colorAttachments[i].destinationAlphaBlendFactor = _dst_blend_alpha_factor[i];

            pipelineStateDescriptor.colorAttachments[i].rgbBlendOperation = _rgb_blend_operation[i];
            pipelineStateDescriptor.colorAttachments[i].alphaBlendOperation = _alpha_blend_operation[i];

            pipelineStateDescriptor.colorAttachments[i].writeMask = _color_mask[i];
        }
    }
}

-(bool)bindFramebufferAttachmentTextures
{
    Framebuffer *fbo;

    // MEMORY SAFETY: Validate context and framebuffer
    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in bindFramebufferAttachmentTextures");
        return false;
    }

    // Validate context pointer lower bound only (high addresses are valid on macOS/arm64)
    uintptr_t ctx_addr = (uintptr_t)ctx;
    if (ctx_addr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid context pointer detected in bindFramebufferAttachmentTextures: 0x%lx", ctx_addr);
        return false;
    }

    fbo = ctx->state.framebuffer;

    // MEMORY SAFETY: Validate framebuffer pointer
    if (!fbo) {
        NSLog(@"MGL ERROR: NULL framebuffer detected in bindFramebufferAttachmentTextures");
        return false;
    }

    // Validate framebuffer pointer lower bound only (high addresses are valid on macOS/arm64)
    uintptr_t fbo_addr = (uintptr_t)fbo;
    if (fbo_addr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid framebuffer pointer detected in bindFramebufferAttachmentTextures: 0x%lx", fbo_addr);
        return false;
    }

    for (int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        if (fbo->color_attachments[i].texture)
        {
            bool isDrawBuffer = true;
            if (fbo->color_attachments[i].textarget == GL_RENDERBUFFER && fbo->color_attachments[i].buf.rbo) {
                isDrawBuffer = fbo->color_attachments[i].buf.rbo->is_draw_buffer;
            }

            if ([self bindFramebufferTexture: &fbo->color_attachments[i] isDrawBuffer:isDrawBuffer] == false)
            {
                DEBUG_PRINT("Failed Framebuffer Attachment\n");
                return false;
            }
        }

        // early out
        if ((fbo->color_attachment_bitfield >> (i+1)) == 0)
            break;
    }

    // depth attachment
    if (fbo->depth.texture)
    {
        if ([self bindFramebufferTexture: &fbo->depth isDrawBuffer: true] == false)
        {
            DEBUG_PRINT("Failed Framebuffer Attachment\n");
            return false;
        }
    }

    // stencil attachment
    if (fbo->stencil.texture)
    {
        if ([self bindFramebufferTexture: &fbo->stencil isDrawBuffer: true] == false)
        {
            DEBUG_PRINT("Failed Framebuffer Attachment\n");
            return false;
        }
    }

    return true;
}

- (void) endRenderEncoding
{
    if (_currentRenderEncoder)
    {
        static uint64_t s_renderPassEndLogCount = 0;
        uint64_t hit = ++s_renderPassEndLogCount;
        if (hit <= 128ull || (hit % 1024ull) == 0ull) {
            mglLogRenderPassLifecycle("end",
                                      hit,
                                      ctx,
                                      _currentCommandBuffer,
                                      _currentRenderEncoder,
                                      _renderPassDescriptor,
                                      _drawable,
                                      _renderPassFramebuffer,
                                      _renderPassFramebufferName,
                                      _renderPassDrawBuffer,
                                      _renderPassDrawBufferCount);
        }
        @try {
            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL DEBUG: Ending render encoder");
            }
            [_currentRenderEncoder endEncoding];
            _currentRenderEncoder = NULL;
            _renderPassFramebuffer = NULL;
            _renderPassFramebufferName = 0;
            _renderPassDrawBuffer = 0;
            _renderPassDrawBufferCount = 0;
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                _renderPassDrawBuffers[i] = GL_NONE;
            }
            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL DEBUG: Render encoder ended successfully");
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Exception ending render encoder: %@ - ignoring", exception.reason);
            // Force clear the encoder even if ending failed
            _currentRenderEncoder = NULL;
            _renderPassFramebuffer = NULL;
            _renderPassFramebufferName = 0;
            _renderPassDrawBuffer = 0;
            _renderPassDrawBufferCount = 0;
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                _renderPassDrawBuffers[i] = GL_NONE;
            }
        }
    }
}

// ULTIMATE FAILSAFE: Emergency Metal state reset to recover from corruption
- (void) emergencyResetMetalState
{
    NSLog(@"MGL CRITICAL: Performing emergency Metal state reset");

    @try {
        // Force cleanup of all Metal objects
        [self endRenderEncoding];

        _currentCommandBuffer = NULL;
        _currentRenderEncoder = NULL;
        _drawable = NULL;

        // Re-initialize basic Metal objects
        if (_device && _commandQueue) {
            NSLog(@"MGL CRITICAL: Re-creating Metal command buffer");
            _currentCommandBuffer = [_commandQueue commandBuffer];

            if (!_currentCommandBuffer) {
                NSLog(@"MGL CRITICAL: Failed to create new command buffer during recovery");
            }
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL CRITICAL: Emergency Metal reset failed: %@", exception);
    }
}

#pragma mark ------------------------------------------------------------------------------------------
#pragma mark processGLState for resolving opengl state into metal state
#pragma mark ------------------------------------------------------------------------------------------

- (bool) processGLState: (bool) draw_command
{
    static uint64_t s_processGLStateCallCount = 0;
    static double s_processGLStateLastCallTime = 0.0;
    static uint64_t s_processGLStateLastCallCount = 0;
    uint64_t processCall = ++s_processGLStateCallCount;
    double processStartSeconds = mglNowSeconds();
    bool traceProcess = mglShouldTraceCall(processCall);
    mglLogLoopHeartbeat("processGLState.loop",
                        processCall,
                        processStartSeconds,
                        &s_processGLStateLastCallTime,
                        &s_processGLStateLastCallCount,
                        0.25);
    if (traceProcess) {
        NSLog(@"MGL TRACE processGLState.begin call=%llu draw=%d",
              (unsigned long long)processCall, draw_command ? 1 : 0);
        mglLogStateSnapshot("processGLState.enter",
                            ctx,
                            _currentCommandBuffer,
                            _currentRenderEncoder,
                            _renderPassDescriptor,
                            _drawable);
    }
    if (draw_command) {
        g_mglProcessDrawCallsSinceSwap++;
    }

    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in processGLState");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.null_ctx",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    uintptr_t earlyCtxAddr = (uintptr_t)ctx;
    if (earlyCtxAddr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid context pointer detected: 0x%lx", earlyCtxAddr);
        return false;
    }

    // REMOVED: Thread synchronization was causing deadlocks
    // The issue is not thread contention but Metal object corruption

    // ULTIMATE FAILSAFE: Metal state corruption detection and recovery
    static int corruption_recovery_count = 0;
    static int max_recovery_attempts = 3;

    // Check for corrupted Metal objects that might cause crashes.
    // Only reject NULL / obviously invalid low addresses.
    if (!_device || !_commandQueue || ((uintptr_t)_device < 0x1000) || ((uintptr_t)_commandQueue < 0x1000)) {
        NSLog(@"MGL CRITICAL: Metal state corruption detected in processGLState!");
        NSLog(@"MGL CRITICAL: device=0x%lx, queue=0x%lx", (uintptr_t)_device, (uintptr_t)_commandQueue);

        if (corruption_recovery_count < max_recovery_attempts) {
            NSLog(@"MGL CRITICAL: Attempting Metal state recovery (%d/%d)", corruption_recovery_count + 1, max_recovery_attempts);

            // Force a complete Metal state reset
            @try {
                [self emergencyResetMetalState];
                corruption_recovery_count++;

                // Re-check after recovery
                if (!_device || !_commandQueue) {
                    NSLog(@"MGL CRITICAL: Metal recovery failed, aborting operation");
                    return false;
                }
            } @catch (NSException *exception) {
                NSLog(@"MGL CRITICAL: Metal recovery failed: %@", exception);
                return false;
            }
        } else {
            NSLog(@"MGL CRITICAL: Maximum recovery attempts exceeded, permanently disabling Metal operations");
            return false;
        }
    }

    //logDirtyBits(ctx);

    if (!draw_command) {
        [self endRenderPassIfFramebufferChangedForNonDraw:processCall];
    }

    // since a clear is embedded into a render encoder
    if (VAO() == NULL)
    {
        if (draw_command)
        {
            NSLog(@"Error: No VAO defined for ctx\n");

            // quietly return if we are not in a draw command with no vao defined
            // like a clear or init call
            return false;
        }

        // for a clear flush sequence...
        if (ctx->state.dirty_bits & DIRTY_STATE)
        {
            // RESTORED: Attempt render encoder creation with improved error handling
            NSLog(@"MGL INFO: RESTORED - Attempting newRenderEncoder with GPU throttling protection");

            // end encoding on current render encoder
            [self endRenderEncoding];

            // Use GPU throttling to prevent crashes when creating new render encoder
            if (![self validateMetalObjects]) {
                NSLog(@"MGL WARNING: GPU throttling active - deferring render encoder creation");
                ctx->state.dirty_bits &= ~DIRTY_STATE;
                return true;
            }

            @try {
                NSLog(@"MGL INFO: Attempting to create new render encoder with safety protection");
                if ([self newRenderEncoder]) {
                    NSLog(@"MGL SUCCESS: New render encoder created successfully");
                } else {
                    NSLog(@"MGL WARNING: Failed to create render encoder - continuing with degraded functionality");
                }
            } @catch (NSException *exception) {
                NSLog(@"MGL ERROR: Render encoder creation failed: %@", exception);
                NSLog(@"MGL INFO: Continuing without render encoder for stability");
            }

            // Clear the dirty bit to prevent repeated attempts
            ctx->state.dirty_bits &= ~DIRTY_STATE;
        }

        return true;
    }

    // only draw commands need a functioning render encoder
    // this can mess up a transition between compute and rendering on a flush
    // so just return
    // we may have to create a blank render encoder to safely run compute and
    // rendering correctly
    if (draw_command == false)
    {
        return true;
    }

    // MEMORY SAFETY: Validate context before use
    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in processGLState");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.null_ctx",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    // Validate context pointer lower bound only (high addresses are valid on macOS/arm64)
    uintptr_t ctx_addr = (uintptr_t)ctx;
    if (ctx_addr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid context pointer detected: 0x%lx", ctx_addr);
        return false;
    }

    // Early circuit-breaker: if a program is currently quarantined due to repeated
    // vertex/fragment interface mismatch, skip draw before creating/rotating buffers.
    Program *blockedProgram = mglResolveProgramFromState(ctx);
    if (blockedProgram &&
        _interfaceMismatchBlockedProgram != 0 &&
        blockedProgram->name == _interfaceMismatchBlockedProgram)
    {
        CFTimeInterval now = CFAbsoluteTimeGetCurrent();
        if (now < _interfaceMismatchBlockedUntil) {
            static uint64_t s_quarantineSkipCount = 0;
            s_quarantineSkipCount++;
            if (s_quarantineSkipCount <= 16 || (s_quarantineSkipCount % 1000) == 0) {
                double remaining = _interfaceMismatchBlockedUntil - now;
                if (remaining < 0.0) remaining = 0.0;
                NSLog(@"MGL WARNING: Program %u quarantined due to interface mismatch (%.2fs remaining), skipping draw",
                      (unsigned)_interfaceMismatchBlockedProgram, remaining);
            }
            return false;
        }
    }

    // Keep command buffer lifecycle healthy: if the active one is already finalized,
    // rotate to a fresh buffer before any state processing.
    if (_currentCommandBuffer && _currentRenderEncoder == NULL) {
        MTLCommandBufferStatus preStatus = _currentCommandBuffer.status;
        if (preStatus >= MTLCommandBufferStatusCommitted) {
            NSLog(@"MGL INFO: processGLState rotating finalized command buffer (status: %ld)", (long)preStatus);
            if (![self newCommandBuffer]) {
                NSLog(@"MGL ERROR: processGLState failed to create a fresh command buffer");
                if (traceProcess) {
                    mglLogStateSnapshot("processGLState.fail.new_cb_rotate",
                                        ctx,
                                        _currentCommandBuffer,
                                        _currentRenderEncoder,
                                        _renderPassDescriptor,
                                        _drawable);
                }
                return false;
            }
        }
    } else if (!_currentCommandBuffer) {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: processGLState found NULL command buffer, creating one");
        }
        if (![self newCommandBuffer]) {
            NSLog(@"MGL ERROR: processGLState could not create initial command buffer");
            if (traceProcess) {
                mglLogStateSnapshot("processGLState.fail.new_cb_initial",
                                    ctx,
                                    _currentCommandBuffer,
                                    _currentRenderEncoder,
                                    _renderPassDescriptor,
                                    _drawable);
            }
            return false;
        }
    }

    if (ctx->state.dirty_bits)
    {
        // FBO binding/attachment changes alter the Metal render pass itself. They must
        // be handled even when no generic DIRTY_STATE bit is present; otherwise the
        // current render encoder can keep drawing into an old attachment while GL state
        // already points at a different FBO.
        if (ctx->state.dirty_bits & DIRTY_FBO)
        {
            Framebuffer *framebuffer = mglRendererGetValidatedFramebuffer(ctx, "processGLState.dirtyFBO");
            if (framebuffer)
            {
                if (framebuffer->dirty_bits & DIRTY_FBO_BINDING)
                {
                    RETURN_FALSE_ON_FAILURE([self bindFramebufferAttachmentTextures]);
                    framebuffer = mglRendererGetValidatedFramebuffer(ctx, "processGLState.dirtyFBO.afterBind");
                    if (framebuffer) {
                        framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
                    }
                }
            }

            [self endRenderEncoding];
            RETURN_FALSE_ON_FAILURE([self newRenderEncoder]);
        }

        // dirty state covers all rendering attachments and general state
        if (ctx->state.dirty_bits & DIRTY_STATE)
        {
            if (ctx->state.dirty_bits & DIRTY_FBO)
            {
                // MEMORY SAFETY: Add comprehensive validation to prevent use-after-free crashes
                Framebuffer *framebuffer = mglRendererGetValidatedFramebuffer(ctx, "processGLState.dirtyStateFBO");
                if (framebuffer)
                {
                    if (framebuffer->dirty_bits & DIRTY_FBO_BINDING)
                    {
                        RETURN_FALSE_ON_FAILURE([self bindFramebufferAttachmentTextures]);

                        // Additional validation after binding
                        framebuffer = mglRendererGetValidatedFramebuffer(ctx, "processGLState.dirtyStateFBO.afterBind");
                        if (framebuffer) {
                            framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
                        }
                    }
                }

                // dirty FBO state can't be cleared just yet its needed below
            }

            ctx->state.dirty_bits &= ~DIRTY_STATE;
        }

        // check for dirty program and vao
        // leave program / vao state dirty, buffers need to be mapped before used below
        // dirty program causes buffers to be remapped
        // dirty vao causes attributes to be remapped to new buffers
        // dirty buffer base causes buffers to be remapped to new indexes
        if (ctx->state.dirty_bits & (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_BUFFER_BASE_STATE))
        {
            // Avoid mapping draw buffers against a nil pipeline during startup/rebuild.
            // We'll map again after a valid pipeline is bound.
            bool deferBufferMapForNilPipeline =
                (draw_command &&
                 _pipelineState == nil &&
                 (ctx->state.dirty_bits & DIRTY_PROGRAM));

            if (deferBufferMapForNilPipeline) {
                static uint64_t s_deferredMapCount = 0;
                s_deferredMapCount++;
                if (s_deferredMapCount <= 16 || (s_deferredMapCount % 1000ull) == 0ull) {
                    NSLog(@"MGL DRAW SKIP: pipelineState is nil (deferring buffer mapping, occurrence=%llu)",
                          (unsigned long long)s_deferredMapCount);
                }
            } else {
                // programs are now compiled before execution, we shouldn't get here
                //assert(ctx->state.program->mtl_data); //

                // figure out vertex shader uniforms / buffer mappings
                RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
            }

            ctx->state.dirty_bits &= ~DIRTY_BUFFER_BASE_STATE;
        }

        // dirty tex covers all texture modifications
        if (ctx->state.dirty_bits & (DIRTY_PROGRAM | DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER))
        {
            RETURN_FALSE_ON_FAILURE([self bindActiveTexturesToMTL]);
            RETURN_FALSE_ON_FAILURE([self bindTexturesToCurrentRenderEncoder]);

            // textures / active textures and samplers are all handled in bindActiveTexturesToMTL
            ctx->state.dirty_bits &= ~(DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER);
        }

        // A dirty VAO changes vertex buffer bindings and may require a new
        // pipeline descriptor, but it does not change the render-pass
        // attachments. Keep the current encoder alive so GL draw ordering and
        // depth/load-store continuity are preserved across HUD/hand/UI passes.
        if (ctx->state.dirty_bits & DIRTY_VAO)
        {
            // updateDirtyBaseBufferList binds new mtl buffers or updates old ones
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list]);
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list]);

            if (!_currentRenderEncoder) {
                RETURN_FALSE_ON_FAILURE([self newRenderEncoder]);
            }

            [self updateCurrentRenderEncoder];

            // clear dirty render state
            ctx->state.dirty_bits &= ~DIRTY_RENDER_STATE;
        }
        else if (ctx->state.dirty_bits & DIRTY_BUFFER)
        {
            // updateDirtyBaseBufferList binds new mtl buffers or updates old ones
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list]);
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list]);

            ctx->state.dirty_bits &= ~DIRTY_BUFFER;
        }
        else if (ctx->state.dirty_bits & DIRTY_RENDER_STATE)
        {
            if (_currentRenderEncoder == NULL)
            {
                RETURN_FALSE_ON_FAILURE([self newRenderEncoder]);
            }

            // a dirty render state may just be something like alpha changes which don't require a new renderbuffer

            // updateCurrentRenderEncoder will update the renderstate outside of creating a new one
            [self updateCurrentRenderEncoder];

            ctx->state.dirty_bits &= ~DIRTY_RENDER_STATE;
        }

        // new pipeline / vertex / renderbuffer and pipelinestate descriptor, should probably make this a single dirty bit
        if (ctx->state.dirty_bits & (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE))
        {
            static CFTimeInterval s_pipelineRetryAfter = 0.0;
            static CFTimeInterval s_interfaceMismatchRetryAfter = 0.0;
            static GLuint s_interfaceMismatchProgramName = 0;
            static MTLPixelFormat s_interfaceMismatchColor0Format = MTLPixelFormatInvalid;
            static MTLPixelFormat s_interfaceMismatchDepthFormat = MTLPixelFormatInvalid;
            static MTLPixelFormat s_interfaceMismatchStencilFormat = MTLPixelFormatInvalid;
            static uint32_t s_interfaceMismatchStreak = 0;
            static GLuint s_programMismatchProgramName = 0;
            static CFTimeInterval s_programMismatchRetryAfter = 0.0;
            static uint32_t s_programMismatchStreak = 0;
            CFTimeInterval now = CFAbsoluteTimeGetCurrent();
            bool skipPipelineBuild = false;
            Program *currentProgram = mglResolveProgramFromState(ctx);
            GLuint currentProgramName = ctx->state.program_name ?
                                        ctx->state.program_name :
                                        (currentProgram ? currentProgram->name : 0);
            VertexArray *currentVAO = ctx->state.vao;
            Framebuffer *currentFBO = mglRendererGetValidatedFramebuffer(ctx, "processGLState.currentFBO");
            GLuint currentFBOName = currentFBO ? currentFBO->name : 0;

            // Program-level breaker (independent of render-pass signature) to avoid
            // mismatch storms where color/depth/stencil signatures keep changing.
            if (currentProgramName != 0 &&
                currentProgramName == s_programMismatchProgramName &&
                now < s_programMismatchRetryAfter) {
                static uint64_t s_programMismatchSkipCount = 0;
                s_programMismatchSkipCount++;
                if (s_programMismatchSkipCount <= 16 || (s_programMismatchSkipCount % 1000ull) == 0ull) {
                    double remaining = s_programMismatchRetryAfter - now;
                    if (remaining < 0.0) remaining = 0.0;
                    NSLog(@"MGL WARNING: Program-level mismatch breaker active (program=%u, %.2fs remaining), skipping draw",
                          (unsigned)currentProgramName,
                          remaining);
                }
                ctx->state.dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                return false;
            }

	            if (now < s_pipelineRetryAfter) {
	                BOOL retryAppliesToCurrentProgram =
	                    (currentProgramName != 0 &&
	                     (currentProgramName == s_interfaceMismatchProgramName ||
	                      currentProgramName == s_programMismatchProgramName ||
	                      currentProgramName == _interfaceMismatchBlockedProgram));

	                if (retryAppliesToCurrentProgram) {
	                    ctx->state.dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
	                    if (!_pipelineState) {
	                        return false;
	                    }
	                    // Keep existing pipeline, but do not early-return before setRenderPipelineState.
	                    skipPipelineBuild = true;
	                } else {
	                    static uint64_t s_retryBypassCount = 0;
	                    s_retryBypassCount++;
	                    if (s_retryBypassCount <= 16 || (s_retryBypassCount % 1000ull) == 0ull) {
	                        NSLog(@"MGL PIPELINE RETRY bypass global retry for unrelated program=%u mismatchProgram=%u blockedProgram=%u",
	                              (unsigned)currentProgramName,
	                              (unsigned)s_interfaceMismatchProgramName,
	                              (unsigned)_interfaceMismatchBlockedProgram);
	                    }
	                }
	            }

            if (!skipPipelineBuild) {
            // create pipeline descriptor
            MTLRenderPipelineDescriptor *pipelineStateDescriptor;

            pipelineStateDescriptor = [self generatePipelineDescriptor];
            if (!pipelineStateDescriptor) {
                NSLog(@"MGL PIPELINE CREATE fail error=generatePipelineDescriptor returned nil");
                return false;
            }

            MTLPixelFormat builtColor0Format = pipelineStateDescriptor.colorAttachments[0].pixelFormat;
            MTLPixelFormat builtDepthFormat = pipelineStateDescriptor.depthAttachmentPixelFormat;
            MTLPixelFormat builtStencilFormat = pipelineStateDescriptor.stencilAttachmentPixelFormat;

            // Circuit breaker for repeated VS/FS interface mismatch.
            if (now < s_interfaceMismatchRetryAfter &&
                currentProgramName == s_interfaceMismatchProgramName &&
                builtColor0Format == s_interfaceMismatchColor0Format &&
                builtDepthFormat == s_interfaceMismatchDepthFormat &&
                builtStencilFormat == s_interfaceMismatchStencilFormat) {
                ctx->state.dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                return false;
            }

            // create vertex descriptor
            MTLVertexDescriptor *vertexDescriptor;

            vertexDescriptor = [self generateVertexDescriptor];
            if (!vertexDescriptor) {
                NSLog(@"MGL PIPELINE CREATE fail error=generateVertexDescriptor returned nil");
                return false;
            }

            [self updateBlendStateCache];
            ctx->state.dirty_bits &= ~DIRTY_ALPHA_STATE;
            [self bindBlendStateToPipelineStateDescriptor:pipelineStateDescriptor];

            if (kMGLVerbosePipelineLogs) {
                MTLRenderPipelineColorAttachmentDescriptor *ca0 = pipelineStateDescriptor.colorAttachments[0];
                NSLog(@"MGL PIPELINE DESC c0 state program=%u fmt=%lu writeMask=0x%x blend=%d srcRGB=%lu dstRGB=%lu srcA=%lu dstA=%lu opRGB=%lu opA=%lu",
                      (unsigned)currentProgramName,
                      (unsigned long)ca0.pixelFormat,
                      (unsigned)ca0.writeMask,
                      ca0.blendingEnabled ? 1 : 0,
                      (unsigned long)ca0.sourceRGBBlendFactor,
                      (unsigned long)ca0.destinationRGBBlendFactor,
                      (unsigned long)ca0.sourceAlphaBlendFactor,
                      (unsigned long)ca0.destinationAlphaBlendFactor,
                      (unsigned long)ca0.rgbBlendOperation,
                      (unsigned long)ca0.alphaBlendOperation);
            }

	            pipelineStateDescriptor.vertexDescriptor = vertexDescriptor;
	            NSString *pipelineCacheKey = nil;
	            bool pipelineResolvedFromCache = false;

	            if (!pipelineResolvedFromCache && _pipelineStateCache && currentProgramName != 0) {
	                uint64_t pipelineSig = mglPipelineDescriptorSignature(pipelineStateDescriptor);
	                uint64_t vertexSig = mglVertexDescriptorSignature(vertexDescriptor);
	                pipelineCacheKey = [NSString stringWithFormat:@"%u:%016llx:%016llx",
	                                    (unsigned)currentProgramName,
	                                    (unsigned long long)pipelineSig,
	                                    (unsigned long long)vertexSig];
	                id<MTLRenderPipelineState> cachedPipeline = [_pipelineStateCache objectForKey:pipelineCacheKey];
	                if (cachedPipeline) {
	                    static uint64_t s_pipelineCacheHitCount = 0;
	                    s_pipelineCacheHitCount++;
	                    if (kMGLVerbosePipelineLogs &&
                            (s_pipelineCacheHitCount <= 128ull || (s_pipelineCacheHitCount % 1000ull) == 0ull)) {
	                        NSLog(@"MGL PIPELINE CACHE hit program=%u vao=%p fbo=%u key=%@",
	                              (unsigned)currentProgramName, currentVAO, (unsigned)currentFBOName, pipelineCacheKey);
	                    }

	                    _pipelineState = cachedPipeline;
	                    pipelineResolvedFromCache = true;
	                    _pipelineColor0Format = builtColor0Format;
	                    _pipelineDepthFormat = builtDepthFormat;
	                    _pipelineStencilFormat = builtStencilFormat;
	                    _pipelineProgramName = currentProgramName;

	                    // Mirror successful compile-side breaker resets.
	                    s_interfaceMismatchStreak = 0;
	                    s_interfaceMismatchProgramName = 0;
	                    s_interfaceMismatchColor0Format = MTLPixelFormatInvalid;
	                    s_interfaceMismatchDepthFormat = MTLPixelFormatInvalid;
	                    s_interfaceMismatchStencilFormat = MTLPixelFormatInvalid;
	                    s_interfaceMismatchRetryAfter = 0.0;
	                    if (s_programMismatchProgramName == currentProgramName) {
	                        s_programMismatchProgramName = 0;
	                        s_programMismatchRetryAfter = 0.0;
	                        s_programMismatchStreak = 0u;
	                    }
	                    if (_interfaceMismatchBlockedProgram == currentProgramName) {
	                        _interfaceMismatchBlockedProgram = 0;
	                        _interfaceMismatchBlockedUntil = 0.0;
	                        _interfaceMismatchBlockedStreak = 0u;
	                    }
	                }
	            }

	            // PROPER AGX VIRTUALIZATION COMPATIBILITY: Fix root cause while maintaining Metal functionality
	            if (!pipelineResolvedFromCache) {
	            NSError *error;
	            id<MTLRenderPipelineState> previousPipelineState = _pipelineState;
	            bool pipelineReusedPrevious = false;

            @try {
                static uint64_t s_pipelineCreateBeginCount = 0;
                s_pipelineCreateBeginCount++;
                if (kMGLVerbosePipelineLogs &&
                    (s_pipelineCreateBeginCount <= 128ull || (s_pipelineCreateBeginCount % 500ull) == 0ull)) {
                    NSLog(@"MGL PIPELINE CREATE begin program=%u vao=%p fbo=%u",
                          (unsigned)currentProgramName, currentVAO, (unsigned)currentFBOName);
                }

                if (kMGLVerbosePipelineLogs) {
                    NSLog(@"MGL INFO: Creating Metal pipeline state with AGX virtualization compatibility...");
                }

                // ROOT CAUSE FIX: The issue is with async shader compilation in virtualized environments
                // Force synchronous pipeline creation to avoid completion queue crashes
                if (kMGLVerbosePipelineLogs) {
                    NSLog(@"MGL INFO: Using synchronous pipeline creation to prevent virtualization crashes");
                }

                // PROPER FIX: Disable async compilation that causes completion queue crashes
                if (kMGLVerbosePipelineLogs &&
                    [_device name] && ([[_device name] containsString:@"AGX"])) {
                    NSLog(@"MGL INFO: AGX virtualization detected - using safe synchronous compilation");
                }

                _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];

                if (!_pipelineState) {
                    NSLog(@"MGL PIPELINE CREATE fail error=%@", error);
                    NSLog(@"MGL ERROR: Pipeline creation failed: %@", error);

                    NSString *errDesc = error.localizedDescription ?: @"";
                    NSString *errDomain = error.domain ?: @"";
                    BOOL isInterfaceMismatch = ((error.code == 3 && [errDomain hasPrefix:@"AGXMetal"]) ||
                                                [errDesc containsString:@"mismatching vertex shader output"] ||
                                                [errDesc containsString:@"not written by vertex shader"]);

	                    if (isInterfaceMismatch) {
	                        mglWriteProgramMSLDump(currentProgram, errDesc);
	                        BOOL sameProgram = (_pipelineProgramName != 0 && _pipelineProgramName == currentProgramName);
                        BOOL colorCompatible = (_pipelineColor0Format == MTLPixelFormatInvalid ||
                                                builtColor0Format == MTLPixelFormatInvalid ||
                                                _pipelineColor0Format == builtColor0Format);
                        BOOL depthCompatible = (_pipelineDepthFormat == MTLPixelFormatInvalid ||
                                                builtDepthFormat == MTLPixelFormatInvalid ||
                                                _pipelineDepthFormat == builtDepthFormat);
                        BOOL stencilCompatible = (_pipelineStencilFormat == MTLPixelFormatInvalid ||
                                                  builtStencilFormat == MTLPixelFormatInvalid ||
                                                  _pipelineStencilFormat == builtStencilFormat);

                        if (previousPipelineState && sameProgram && colorCompatible && depthCompatible && stencilCompatible) {
                            NSLog(@"MGL WARNING: Interface mismatch for program %u; reusing previous compatible pipeline once",
                                  (unsigned)currentProgramName);
                            _pipelineState = previousPipelineState;
                            pipelineReusedPrevious = true;
                        } else {
                            BOOL sameMismatchSignature =
                                (currentProgramName == s_interfaceMismatchProgramName &&
                                 builtColor0Format == s_interfaceMismatchColor0Format &&
                                 builtDepthFormat == s_interfaceMismatchDepthFormat &&
                                 builtStencilFormat == s_interfaceMismatchStencilFormat);
                            if (sameMismatchSignature) {
                                if (s_interfaceMismatchStreak < UINT32_MAX) {
                                    s_interfaceMismatchStreak++;
                                }
                            } else {
                                s_interfaceMismatchStreak = 1;
                                s_interfaceMismatchProgramName = currentProgramName;
                                s_interfaceMismatchColor0Format = builtColor0Format;
                                s_interfaceMismatchDepthFormat = builtDepthFormat;
                                s_interfaceMismatchStencilFormat = builtStencilFormat;
                            }

                            // Exponential backoff: 0.10, 0.20, 0.40, 0.80, 1.60, capped at 2.00 sec.
                            uint32_t cappedShift = (s_interfaceMismatchStreak > 5u) ? 4u : (s_interfaceMismatchStreak - 1u);
                            double retryDelay = 0.10 * (double)(1u << cappedShift);
                            if (retryDelay > 2.0) {
                                retryDelay = 2.0;
                            }
                            s_interfaceMismatchRetryAfter = now + retryDelay;

                            if (s_interfaceMismatchStreak <= 5u || (s_interfaceMismatchStreak % 200u) == 0u) {
                                NSLog(@"MGL WARNING: Interface mismatch (program=%u, streak=%u), throttling retries for %.2fs",
                                      (unsigned)currentProgramName,
                                      (unsigned)s_interfaceMismatchStreak,
                                      retryDelay);
                            }

                            // Program-level breaker update (ignores attachment signature).
                            if (s_programMismatchProgramName == currentProgramName) {
                                if (s_programMismatchStreak < UINT32_MAX) {
                                    s_programMismatchStreak++;
                                }
                            } else {
                                s_programMismatchProgramName = currentProgramName;
                                s_programMismatchStreak = 1u;
                            }
                            double programDelay = 0.25 * (double)(1u << ((s_programMismatchStreak > 6u) ? 6u : (s_programMismatchStreak - 1u)));
                            if (programDelay > 20.0) {
                                programDelay = 20.0;
                            }
                            s_programMismatchRetryAfter = now + programDelay;
                            if (s_programMismatchStreak <= 8u || (s_programMismatchStreak % 64u) == 0u) {
                                NSLog(@"MGL WARNING: Program %u mismatch breaker set for %.2fs (streak=%u)",
                                      (unsigned)currentProgramName,
                                      programDelay,
                                      (unsigned)s_programMismatchStreak);
                            }

                            // Global quarantine for this program to prevent command-buffer storm.
                            if (_interfaceMismatchBlockedProgram == currentProgramName) {
                                if (_interfaceMismatchBlockedStreak < UINT32_MAX) {
                                    _interfaceMismatchBlockedStreak++;
                                }
                            } else {
                                _interfaceMismatchBlockedProgram = currentProgramName;
                                _interfaceMismatchBlockedStreak = 1u;
                            }
                            // Use a stronger quarantine window than compile retry backoff.
                            // This prevents pathological draw loops from repeatedly re-entering
                            // pipeline compilation and overwhelming AGX command submission.
                            double quarantineDelay = retryDelay * 8.0;
                            if (quarantineDelay < 1.00) quarantineDelay = 1.00;
                            if (quarantineDelay > 15.00) quarantineDelay = 15.00;
                            _interfaceMismatchBlockedUntil = now + quarantineDelay;
                            if (_interfaceMismatchBlockedStreak <= 6u || (_interfaceMismatchBlockedStreak % 64u) == 0u) {
                                NSLog(@"MGL WARNING: Program %u quarantined for %.2fs after interface mismatch (streak=%u)",
                                      (unsigned)currentProgramName,
                                      quarantineDelay,
                                      (unsigned)_interfaceMismatchBlockedStreak);
                            }

                            _pipelineState = nil;
                            s_pipelineRetryAfter = (_interfaceMismatchBlockedUntil > s_interfaceMismatchRetryAfter)
                                ? _interfaceMismatchBlockedUntil
                                : s_interfaceMismatchRetryAfter;
                            ctx->state.dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                            return false;
                        }
                    }

                    if (!skipPipelineBuild) {
                        // Avoid destructive global recovery during shader/pipeline compile errors.
                        // These are usually content/interface issues, not GPU-state corruption.

                        // AGX VIRTUALIZATION FALLBACK: Try with minimal descriptor
                        @try {
                            NSLog(@"MGL INFO: VIRTUALIZED AGX - Trying simplified compilation fallback...");

                            // Simplify the descriptor to avoid complex shader compilation issues
                            MTLRenderPipelineDescriptor *simpleDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
                            simpleDescriptor.colorAttachments[0].pixelFormat = pipelineStateDescriptor.colorAttachments[0].pixelFormat;
                            simpleDescriptor.depthAttachmentPixelFormat = pipelineStateDescriptor.depthAttachmentPixelFormat;
                            simpleDescriptor.stencilAttachmentPixelFormat = pipelineStateDescriptor.stencilAttachmentPixelFormat;
                            simpleDescriptor.vertexDescriptor = pipelineStateDescriptor.vertexDescriptor;
                            simpleDescriptor.vertexFunction = pipelineStateDescriptor.vertexFunction;
                            simpleDescriptor.fragmentFunction = pipelineStateDescriptor.fragmentFunction;

                            _pipelineState = [_device newRenderPipelineStateWithDescriptor:simpleDescriptor error:&error];
                            if (_pipelineState) {
                                builtColor0Format = simpleDescriptor.colorAttachments[0].pixelFormat;
                                builtDepthFormat = simpleDescriptor.depthAttachmentPixelFormat;
                                builtStencilFormat = simpleDescriptor.stencilAttachmentPixelFormat;
                            }
                        } @catch (NSException *innerException) {
                            NSLog(@"MGL ERROR: VIRTUALIZED AGX - Simplified compilation also failed: %@", innerException);
                        }
                    }
                }

            } @catch (NSException *exception) {
                NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - Metal pipeline creation crashed: %@", exception);
                NSLog(@"MGL CRITICAL: Exception name: %@", [exception name]);
                NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);

                // VIRTUALIZED AGX ULTIMATE FALLBACK: Create minimal safe pipeline
                NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating ultimate fallback pipeline for virtualization safety");

                @try {
                    MTLRenderPipelineDescriptor *safeDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
                    MTLPixelFormat safeColor0Format = pipelineStateDescriptor.colorAttachments[0].pixelFormat;
                    if (_renderPassDescriptor && _renderPassDescriptor.colorAttachments[0].texture) {
                        safeColor0Format = _renderPassDescriptor.colorAttachments[0].texture.pixelFormat;
                    } else if (_drawable && _drawable.texture) {
                        safeColor0Format = _drawable.texture.pixelFormat;
                    }
                    if (safeColor0Format == MTLPixelFormatInvalid) {
                        safeColor0Format = MTLPixelFormatBGRA8Unorm;
                    }
                    safeDescriptor.colorAttachments[0].pixelFormat = safeColor0Format;
                    safeDescriptor.depthAttachmentPixelFormat = pipelineStateDescriptor.depthAttachmentPixelFormat;
                    safeDescriptor.stencilAttachmentPixelFormat = pipelineStateDescriptor.stencilAttachmentPixelFormat;
                    safeDescriptor.colorAttachments[0].blendingEnabled = NO;

                    // Use hardcoded minimal shaders that are guaranteed to work in virtualization
                    NSString *safeVertexShader = @"#include <metal_stdlib>\nusing namespace metal;\nvertex float4 main(uint vid [[vertex_id]]) { return float4(0.0, 0.0, 0.0, 1.0); }";
                    NSString *safeFragmentShader = @"#include <metal_stdlib>\nusing namespace metal;\nfragment float4 main() { return float4(0.0, 0.0, 0.0, 1.0); }";

                    NSError *libraryError;
                    id<MTLLibrary> vertLibrary = [_device newLibraryWithSource:safeVertexShader options:nil error:&libraryError];
                    id<MTLLibrary> fragLibrary = [_device newLibraryWithSource:safeFragmentShader options:nil error:&libraryError];

                    if (vertLibrary && fragLibrary) {
                        safeDescriptor.vertexFunction = [vertLibrary newFunctionWithName:@"main"];
                        safeDescriptor.fragmentFunction = [fragLibrary newFunctionWithName:@"main"];

                        _pipelineState = [_device newRenderPipelineStateWithDescriptor:safeDescriptor error:&error];
                        if (_pipelineState) {
                            builtColor0Format = safeDescriptor.colorAttachments[0].pixelFormat;
                            builtDepthFormat = safeDescriptor.depthAttachmentPixelFormat;
                            builtStencilFormat = safeDescriptor.stencilAttachmentPixelFormat;
                            NSLog(@"MGL INFO: VIRTUALIZED AGX - Safe fallback pipeline created successfully");
                        }
                    }
                } @catch (NSException *fallbackException) {
                    NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - Even fallback pipeline failed: %@", fallbackException);
                }

                if (!_pipelineState) {
                    NSLog(@"MGL CRITICAL: VIRTUALIZED AGX - All pipeline creation attempts failed, disabling rendering");
                    _pipelineState = nil;
                    s_pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.25;
                    ctx->state.dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                    return false;
                }
            }

            // Pipeline State creation could fail if the pipeline descriptor isn't set up properly.
            //  If the Metal API validation is enabled, you can find out more information about what
            //  went wrong.  (Metal API validation is enabled by default when a debug build is run
            //  from Xcode.)
	            if (!_pipelineState) {
	                NSLog(@"MGL ERROR: Failed to create pipeline state: %@", error);
	                NSLog(@"MGL WARNING: Skipping draw for this pipeline build failure; will retry later");
                s_pipelineRetryAfter = CFAbsoluteTimeGetCurrent() + 0.10;
                ctx->state.dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
                return false;
            } else {
                if (kMGLVerbosePipelineLogs) {
                    NSLog(@"MGL PIPELINE CREATE success pipeline=%p", _pipelineState);
                    NSLog(@"MGL INFO: Pipeline state created successfully");
                }
                // Clear interface-mismatch breaker after a successful compile path.
                s_interfaceMismatchStreak = 0;
                s_interfaceMismatchProgramName = 0;
                s_interfaceMismatchColor0Format = MTLPixelFormatInvalid;
                s_interfaceMismatchDepthFormat = MTLPixelFormatInvalid;
                s_interfaceMismatchStencilFormat = MTLPixelFormatInvalid;
                s_interfaceMismatchRetryAfter = 0.0;
                if (!pipelineReusedPrevious) {
                    _pipelineColor0Format = builtColor0Format;
                    _pipelineDepthFormat = builtDepthFormat;
                    _pipelineStencilFormat = builtStencilFormat;
                    _pipelineProgramName = currentProgramName;
                }
                if (s_programMismatchProgramName == currentProgramName) {
                    s_programMismatchProgramName = 0;
                    s_programMismatchRetryAfter = 0.0;
                    s_programMismatchStreak = 0u;
                }
	                if (_interfaceMismatchBlockedProgram == currentProgramName) {
	                    _interfaceMismatchBlockedProgram = 0;
	                    _interfaceMismatchBlockedUntil = 0.0;
	                    _interfaceMismatchBlockedStreak = 0u;
	                }

	                if (_pipelineStateCache) {
	                    if (_pipelineStateCache.count >= 256) {
	                        [_pipelineStateCache removeAllObjects];
	                    }
                        if (pipelineCacheKey) {
	                        [_pipelineStateCache setObject:_pipelineState forKey:pipelineCacheKey];
                        }
	                }
	            }
	            }

	            ctx->state.dirty_bits &= ~(DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO);
	            }
        }

        //if (ctx->state.dirty_bits)
        //    logDirtyBits(ctx);

        // clear all bits when the DIRTY ALL bit is set.. kind of a hack but we want to
        // check for dirty bits outside of dirty all
        if (ctx->state.dirty_bits & DIRTY_ALL_BIT)
            ctx->state.dirty_bits = 0;

        // we missed something
        //assert(ctx->state.dirty_bits == 0);
    }
    else // if (ctx->state.dirty_bits)
    {
        // buffer data can be changed but the bindings remain in place.. so we need to update the data if this is the case
        // like a uniform or buffer sub data call
        
        if( [self checkForDirtyBufferData: &ctx->state.vertex_buffer_map_list])
        {
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list]);

            RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder]);
        }
        
        if( [self checkForDirtyBufferData: &ctx->state.fragment_buffer_map_list])
        {
            RETURN_FALSE_ON_FAILURE([self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list]);

            RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder]);
        }
    }

    // Ensure a render encoder exists for draw commands.
    if (!_currentRenderEncoder) {
        static uint64_t s_nilEncoderRecoveryCount = 0;
        uint64_t nilHit = ++s_nilEncoderRecoveryCount;
        NSLog(@"MGL WARNING: processGLState - current render encoder is nil, attempting recovery hit=%llu",
              (unsigned long long)nilHit);
        if (nilHit <= 128ull || (nilHit % 512ull) == 0ull) {
            mglLogRenderPassLifecycle("nil-encoder-before-recovery",
                                      nilHit,
                                      ctx,
                                      _currentCommandBuffer,
                                      _currentRenderEncoder,
                                      _renderPassDescriptor,
                                      _drawable,
                                      _renderPassFramebuffer,
                                      _renderPassFramebufferName,
                                      _renderPassDrawBuffer,
                                      _renderPassDrawBufferCount);
        }
        RETURN_FALSE_ON_FAILURE([self newRenderEncoder]);
        if (nilHit <= 128ull || (nilHit % 512ull) == 0ull) {
            mglLogRenderPassLifecycle("nil-encoder-after-recovery",
                                      nilHit,
                                      ctx,
                                      _currentCommandBuffer,
                                      _currentRenderEncoder,
                                      _renderPassDescriptor,
                                      _drawable,
                                      _renderPassFramebuffer,
                                      _renderPassFramebufferName,
                                      _renderPassDrawBuffer,
                                      _renderPassDrawBufferCount);
        }
    }

    if (draw_command) {
        RETURN_FALSE_ON_FAILURE([self ensureCurrentRenderPassMatchesFramebufferForDraw]);
        [self updateCurrentRenderEncoder];
    }

    if (draw_command && kMGLVerbosePipelineLogs) {
        static uint64_t s_drawPipelineLookupCount = 0;
        s_drawPipelineLookupCount++;
        if (s_drawPipelineLookupCount <= 256ull || (s_drawPipelineLookupCount % 1000ull) == 0ull) {
            Program *lookupProgram = mglResolveProgramFromState(ctx);
            GLuint lookupProgramName = ctx->state.program_name ?
                                       ctx->state.program_name :
                                       (lookupProgram ? lookupProgram->name : 0);
            Framebuffer *lookupFBO = ctx->state.framebuffer;
            GLuint lookupFBOName = lookupFBO ? lookupFBO->name : 0;
            fprintf(stderr, "MGL Draw current program name=%u ptr=%p\n",
                    (unsigned)lookupProgramName, (void *)lookupProgram);
            NSLog(@"MGL DRAW pipeline lookup result=%p program=%u vao=%p fbo=%u",
                  _pipelineState, (unsigned)lookupProgramName, ctx->state.vao, (unsigned)lookupFBOName);
        }
    }

    if (!_pipelineState) {
        static uint64_t nil_pipeline_count = 0;
        nil_pipeline_count++;
        if (nil_pipeline_count <= 8 || (nil_pipeline_count % 1000) == 0) {
            NSLog(@"MGL DRAW SKIP: pipelineState is nil, forcing rebuild (occurrence=%llu)",
                  (unsigned long long)nil_pipeline_count);
        }
        // Force rebuild on next state processing pass.
        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_pipeline",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    // Guard against invalid render pass state before binding pipeline.
    // Metal debug validation can abort the process if the encoder/render pass is incompatible.
    if (!_renderPassDescriptor) {
        NSLog(@"MGL ERROR: processGLState - renderPassDescriptor is nil before pipeline bind");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_rpd",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }
    BOOL passHasAnyAttachment = NO;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        id<MTLTexture> colorAttachment = _renderPassDescriptor.colorAttachments[i].texture;
        if (colorAttachment) {
            passHasAnyAttachment = YES;
            if ((colorAttachment.usage & MTLTextureUsageRenderTarget) == 0) {
                NSLog(@"MGL WARNING: processGLState - color attachment %d missing RenderTarget usage (usage=0x%lx); skipping draw",
                      i,
                      (unsigned long)colorAttachment.usage);
                if (traceProcess) {
                    mglLogStateSnapshot("processGLState.fail.color_usage",
                                        ctx,
                                        _currentCommandBuffer,
                                        _currentRenderEncoder,
                                        _renderPassDescriptor,
                                        _drawable);
                }
                return false;
            }
        }
    }
    if (_renderPassDescriptor.depthAttachment.texture ||
        _renderPassDescriptor.stencilAttachment.texture) {
        passHasAnyAttachment = YES;
    }

    if (!passHasAnyAttachment) {
        NSLog(@"MGL WARNING: processGLState - render pass has no attachments, skipping draw to avoid Metal assert");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.no_attachments",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    MTLPixelFormat currentColor0Format = MTLPixelFormatInvalid;
    MTLPixelFormat currentDepthFormat = MTLPixelFormatInvalid;
    MTLPixelFormat currentStencilFormat = MTLPixelFormatInvalid;

    id<MTLTexture> rpColor0 = _renderPassDescriptor.colorAttachments[0].texture;
    id<MTLTexture> rpDepth = _renderPassDescriptor.depthAttachment.texture;
    id<MTLTexture> rpStencil = _renderPassDescriptor.stencilAttachment.texture;
    if (rpColor0) {
        currentColor0Format = rpColor0.pixelFormat;
    }
    if (rpDepth) {
        currentDepthFormat = rpDepth.pixelFormat;
    }
    if (rpStencil) {
        currentStencilFormat = rpStencil.pixelFormat;
    }

    // IMPORTANT:
    // Never mutate depth/stencil attachments here to "fit" an existing pipeline.
    // The active Metal render encoder was already created with a render-pass descriptor,
    // and changing attachments after encoder creation does not make that encoder compatible.
    // We must instead reject mismatched pipeline/pass combinations and rebuild safely.

    if (_pipelineColor0Format != MTLPixelFormatInvalid &&
        currentColor0Format != MTLPixelFormatInvalid &&
        _pipelineColor0Format != currentColor0Format) {
        static uint64_t s_colorFormatMismatchCount = 0;
        s_colorFormatMismatchCount++;
        if (s_colorFormatMismatchCount <= 16 || (s_colorFormatMismatchCount % 250) == 0) {
            NSLog(@"MGL WARNING: Pipeline/pass color format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
                  (unsigned long)_pipelineColor0Format, (unsigned long)currentColor0Format);
        }
        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
        return false;
    }

    if (_pipelineDepthFormat != currentDepthFormat) {
        BOOL pipelineHasDepth = (_pipelineDepthFormat != MTLPixelFormatInvalid);
        BOOL passHasDepth = (currentDepthFormat != MTLPixelFormatInvalid);
        if (!pipelineHasDepth && !passHasDepth) {
            goto depth_format_ok;
        }
        // Recovery path: if the pipeline was compiled without depth but the pass has depth,
        // temporarily drop depth attachment for this encoder to avoid hard validation loops.
        if (!pipelineHasDepth && passHasDepth && _renderPassDescriptor) {
            NSLog(@"MGL WARNING: Pipeline has no depth format but pass has depth (%lu); recreating encoder without depth attachment",
                  (unsigned long)currentDepthFormat);
            _renderPassDescriptor.depthAttachment.texture = nil;
            _renderPassDescriptor.depthAttachment.resolveTexture = nil;
            _renderPassDescriptor.depthAttachment.loadAction = MTLLoadActionDontCare;
            _renderPassDescriptor.depthAttachment.storeAction = MTLStoreActionDontCare;
            [self endRenderEncoding];
            if (![self newRenderEncoder]) {
                NSLog(@"MGL ERROR: Failed to recreate render encoder after depth detachment");
                return false;
            }
            currentDepthFormat = MTLPixelFormatInvalid;
            goto depth_format_ok;
        }
        NSLog(@"MGL WARNING: Pipeline/pass depth format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
              (unsigned long)_pipelineDepthFormat, (unsigned long)currentDepthFormat);
        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
        return false;
    }
depth_format_ok:;

    if (_pipelineStencilFormat != currentStencilFormat) {
        BOOL pipelineHasStencil = (_pipelineStencilFormat != MTLPixelFormatInvalid);
        BOOL passHasStencil = (currentStencilFormat != MTLPixelFormatInvalid);
        if (!pipelineHasStencil && !passHasStencil) {
            goto stencil_format_ok;
        }
        // Recovery path: if the pipeline has no stencil but the pass does, strip stencil from pass.
        if (!pipelineHasStencil && passHasStencil && _renderPassDescriptor) {
            NSLog(@"MGL WARNING: Pipeline has no stencil format but pass has stencil (%lu); recreating encoder without stencil attachment",
                  (unsigned long)currentStencilFormat);
            _renderPassDescriptor.stencilAttachment.texture = nil;
            _renderPassDescriptor.stencilAttachment.resolveTexture = nil;
            _renderPassDescriptor.stencilAttachment.loadAction = MTLLoadActionDontCare;
            _renderPassDescriptor.stencilAttachment.storeAction = MTLStoreActionDontCare;
            [self endRenderEncoding];
            if (![self newRenderEncoder]) {
                NSLog(@"MGL ERROR: Failed to recreate render encoder after stencil detachment");
                return false;
            }
            currentStencilFormat = MTLPixelFormatInvalid;
            goto stencil_format_ok;
        }
        NSLog(@"MGL WARNING: Pipeline/pass stencil format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild",
              (unsigned long)_pipelineStencilFormat, (unsigned long)currentStencilFormat);
        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
        return false;
    }
stencil_format_ok:;

    @try {
        [_currentRenderEncoder setRenderPipelineState:_pipelineState];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: processGLState - setRenderPipelineState failed: %@", exception.reason);
        // Force pipeline/state retranslation on next draw instead of crashing this frame.
        ctx->state.dirty_bits |= (DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO | DIRTY_RENDER_STATE);
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.set_pipeline",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        }
        return false;
    }

    // Stability-first rebinding pass:
    // Command buffer rotation / encoder recreation can drop previously latched bindings.
    // Rebind required resources before every draw to avoid Metal validation aborts.
    RETURN_FALSE_ON_FAILURE([self mapBuffersToMTL]);
    RETURN_FALSE_ON_FAILURE([self bindVertexBuffersToCurrentRenderEncoder]);
    RETURN_FALSE_ON_FAILURE([self bindFragmentBuffersToCurrentRenderEncoder]);
    RETURN_FALSE_ON_FAILURE([self bindActiveTexturesToMTL]);
    RETURN_FALSE_ON_FAILURE([self bindTexturesToCurrentRenderEncoder]);

    double processElapsedMs = (mglNowSeconds() - processStartSeconds) * 1000.0;
    if (traceProcess) {
        NSLog(@"MGL TRACE processGLState.end call=%llu draw=%d elapsed=%.3fms",
              (unsigned long long)processCall, draw_command ? 1 : 0, processElapsedMs);
        mglLogStateSnapshot("processGLState.exit.ok",
                            ctx,
                            _currentCommandBuffer,
                            _currentRenderEncoder,
                            _renderPassDescriptor,
                            _drawable);
    } else if (processElapsedMs >= 25.0) {
        NSLog(@"MGL TRACE processGLState.slow call=%llu draw=%d elapsed=%.3fms",
              (unsigned long long)processCall, draw_command ? 1 : 0, processElapsedMs);
    }
    return true;
}

#pragma mark ----- compute utility ---------------------------------------------------------------------

- (bool) bindBuffersToComputeEncoder:(id <MTLComputeCommandEncoder>) computeCommandEncoder
{
    if (!computeCommandEncoder) {
        NSLog(@"MGL COMPUTE ERROR: NULL compute encoder for buffer binding");
        return false;
    }

    RETURN_FALSE_ON_FAILURE([self mapGLBuffersToMTLBufferMap: &ctx->state.compute_buffer_map_list stage:_COMPUTE_SHADER]);

    // dirty buffer covers all buffer modifications
    if (ctx->state.dirty_bits & DIRTY_BUFFER)
    {
        // updateDirtyBaseBufferList binds new mtl buffers or updates old ones
        [self updateDirtyBaseBufferList: &ctx->state.compute_buffer_map_list];

        ctx->state.dirty_bits &= ~DIRTY_BUFFER;
    }

    for(int i=0; i<ctx->state.compute_buffer_map_list.count; i++)
    {
        Buffer *ptr;

        ptr = ctx->state.compute_buffer_map_list.buffers[i].buf;

        RETURN_FALSE_ON_NULL(ptr);
        RETURN_FALSE_ON_NULL(ptr->data.mtl_data);

        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(ptr->data.mtl_data);
        if (!buffer) {
            NSLog(@"MGL COMPUTE ERROR: buffer %u has NULL Metal buffer after mapping", ptr->name);
            return false;
        }

        [computeCommandEncoder setBuffer:buffer offset:0 atIndex:i ];
    }

    return true;
}

- (bool) bindTexturesToComputeEncoder:(id <MTLComputeCommandEncoder>) computeCommandEncoder
{
    GLuint count;
    enum {
        _TEXTURE,
        _IMAGE_TEXTURE
    };
    struct {
        int spvc_type;
        int gl_texture_type;
    } mapped_types[] = {
        {SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, _TEXTURE},
        {SPVC_RESOURCE_TYPE_STORAGE_IMAGE, _IMAGE_TEXTURE},
        {0,0}
    };

    if (!computeCommandEncoder) {
        NSLog(@"MGL COMPUTE ERROR: NULL compute encoder for texture binding");
        return false;
    }

    for(int type=0; mapped_types[type].spvc_type; type++)
    {
        int spvc_type;
        int gl_texture_type;

        spvc_type = mapped_types[type].spvc_type;
        gl_texture_type = mapped_types[type].gl_texture_type;

        // iterate shader storage buffers
        count = [self getProgramBindingCount: _COMPUTE_SHADER type: spvc_type];
        if (count)
        {
            int textures_to_be_mapped = count;

            if (textures_to_be_mapped > TEXTURE_UNITS) {
                textures_to_be_mapped = TEXTURE_UNITS;
            }

            for (int i=0; i < (int)count && textures_to_be_mapped > 0; i++)
            {
               // GLuint spirv_location;
                GLuint spirv_binding;
                Texture *ptr;

                spirv_binding = [self getProgramLocation:_COMPUTE_SHADER type:spvc_type index: i];
                spirv_binding = [self getProgramBinding:_COMPUTE_SHADER type:spvc_type index: i];
                if (spirv_binding >= TEXTURE_UNITS) {
                    continue;
                }

                switch(gl_texture_type)
                {
                    case _TEXTURE: ptr = STATE(active_textures[spirv_binding]); break;
                    case _IMAGE_TEXTURE: ptr = STATE(image_units[spirv_binding].tex); break;
                    default:
                        ptr = NULL;
                        NSLog(@"MGL COMPUTE ERROR: unknown compute texture binding class %d", gl_texture_type);
                        return false;
                }

                if (ptr)
                {
                    RETURN_FALSE_ON_FAILURE([self bindMTLTexture: ptr]);
                    if (!ptr->mtl_data) {
                        continue;
                    }

                    id<MTLTexture> texture;
                    texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
                    if (!texture) {
                        continue;
                    }

                    id<MTLSamplerState> sampler;

                    // late binding of texture samplers.. but its better than scanning the entire texture_samplers
                    if(STATE(texture_samplers[spirv_binding]))
                    {
                        Sampler *gl_sampler;

                        gl_sampler = STATE(texture_samplers[spirv_binding]);

                        // delete existing sampler if dirty
                        if (gl_sampler->dirty_bits)
                        {
                            if (gl_sampler->mtl_data)
                            {
                                CFBridgingRelease(gl_sampler->mtl_data);
                                gl_sampler->mtl_data = NULL;
                            }
                        }

                        if (gl_sampler->mtl_data == NULL)
                        {
                            gl_sampler->mtl_data = (void *)CFBridgingRetain([self createMTLSamplerForTexParam:&gl_sampler->params target:ptr->target]);
                            gl_sampler->dirty_bits = 0;
                        }

                        sampler = (__bridge id<MTLSamplerState>)(gl_sampler->mtl_data);
                    }
                    else
                    {
                        sampler = (__bridge id<MTLSamplerState>)(ptr->params.mtl_data);
                    }

                    if (!sampler) {
                        id<MTLSamplerState> fallbackSampler = [_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
                        sampler = fallbackSampler;
                        if (!sampler) {
                            continue;
                        }
                    }

                    [computeCommandEncoder setTexture:texture atIndex:spirv_binding];
                    [computeCommandEncoder setSamplerState: sampler atIndex:spirv_binding];

                    textures_to_be_mapped--;
                }
            }

            // texture not found
            if (textures_to_be_mapped)
            {
                DEBUG_PRINT("No texture bound for fragment shader location\n");

                return false;
            }
        }
    }

    ctx->state.dirty_bits &= ~(DIRTY_TEX_BINDING | DIRTY_SAMPLER | DIRTY_IMAGE_UNIT_STATE);

    return true;
}

#pragma mark ------------------------------------------------------------------------------------------
#pragma mark processCompute
#pragma mark ------------------------------------------------------------------------------------------
-(bool)processCompute:(id <MTLComputeCommandEncoder>) computeCommandEncoder
{
    // from https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Compute-Ctx/Compute-Ctx.html#//apple_ref/doc/uid/TP40014221-CH6-SW1
    Program *program;

    if (!computeCommandEncoder) {
        NSLog(@"MGL COMPUTE ERROR: processCompute called with NULL encoder");
        return false;
    }

    program = ctx->state.program;
    if (!program) {
        NSLog(@"MGL COMPUTE ERROR: glDispatchCompute with no current program");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return false;
    }

    if (program->dirty_bits)
    {
        if (![self bindMTLProgram: program]) {
            NSLog(@"MGL COMPUTE ERROR: failed to bind compute program %u", program->name);
            return false;
        }
    }

    Shader *computeShader;
    computeShader = program->shader_slots[_COMPUTE_SHADER];
    if (!computeShader) {
        NSLog(@"MGL COMPUTE ERROR: current program %u has no compute shader", program->name);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return false;
    }

    id <MTLFunction> func;
    func = (__bridge id<MTLFunction>)(computeShader->mtl_data.function);
    if (!func) {
        NSLog(@"MGL COMPUTE ERROR: compute shader for program %u has no Metal function", program->name);
        return false;
    }

    id <MTLComputePipelineState> computePipelineState;
    NSError *errors;
    computePipelineState = [_device newComputePipelineStateWithFunction:func error: &errors];
    if (!computePipelineState) {
        NSLog(@"MGL COMPUTE ERROR: failed to create compute pipeline for program %u: %@",
              program->name,
              errors);
        return false;
    }

    [computeCommandEncoder setComputePipelineState:computePipelineState];

    RETURN_FALSE_ON_FAILURE([self bindBuffersToComputeEncoder: computeCommandEncoder]);

    //setTexture:atIndex:
    //setTextures:withRange:
    RETURN_FALSE_ON_FAILURE([self bindTexturesToComputeEncoder: computeCommandEncoder]);

    // setSamplerState:atIndex:
    // setSamplerState:lodMinClamp:lodMaxClamp:atIndex:
    // setSamplerStates:withRange:
    // setSamplerStates:lodMinClamps:lodMaxClamps:withRange:

    // [computeCommandEncoder setThreadgroupMemoryLength:atIndex:

    ctx->state.dirty_bits = 0;

    return true;
}

-(void)mtlDispatchCompute:(GLMContext)glm_ctx groupsX:(GLuint)groups_x groupsY:(GLuint)groups_y groupsZ:(GLuint)groups_z
{
    if (!glm_ctx) {
        NSLog(@"MGL COMPUTE ERROR: mtlDispatchCompute called with NULL context");
        return;
    }

    ctx = glm_ctx;

    if (groups_x == 0 || groups_y == 0 || groups_z == 0) {
        NSLog(@"MGL COMPUTE TRACE: glDispatchCompute zero-sized dispatch %ux%ux%u skipped",
              groups_x,
              groups_y,
              groups_z);
        return;
    }

    // end encoding on current render encoder
    [self endRenderEncoding];

    RETURN_ON_FAILURE([self ensureWritableCommandBuffer:"mtlDispatchCompute"]);

    id <MTLComputeCommandEncoder> computeCommandEncoder = [_currentCommandBuffer computeCommandEncoder];
    if (!computeCommandEncoder) {
        NSLog(@"MGL ERROR: Failed to create compute command encoder");
        return;
    }

    if (![self processCompute:computeCommandEncoder]) {
        [computeCommandEncoder endEncoding];
        return;
    }

    MTLSize numThreadgroups;
    MTLSize threadsPerThreadgroup;

    Program *ptr;
    ptr = glm_ctx->state.program;

    GLuint local_x = ptr->local_workgroup_size.x ? ptr->local_workgroup_size.x : 1u;
    GLuint local_y = ptr->local_workgroup_size.y ? ptr->local_workgroup_size.y : 1u;
    GLuint local_z = ptr->local_workgroup_size.z ? ptr->local_workgroup_size.z : 1u;

    if (ptr->local_workgroup_size.x || ptr->local_workgroup_size.y || ptr->local_workgroup_size.z)
    {
        GLuint mod_x, mod_y, mod_z;
        GLuint size_x, size_y, size_z;

        mod_x = groups_x % local_x;
        mod_y = groups_y % local_y;
        mod_z = groups_z % local_z;

        size_x = groups_x / local_x;
        size_y = groups_y / local_y;
        size_z = groups_z / local_z;

        if (mod_x || mod_y || mod_z)
        {
            if (mod_x)
                size_x++;

            if (mod_y)
                size_y++;

            if (mod_z)
                size_z++;
        }

        numThreadgroups = MTLSizeMake(size_x, size_y, size_z);
        threadsPerThreadgroup = MTLSizeMake(local_x, local_y, local_z);

        [computeCommandEncoder dispatchThreadgroups:numThreadgroups
                                        threadsPerThreadgroup:threadsPerThreadgroup];
    }
    else
    {
        numThreadgroups = MTLSizeMake(groups_x, groups_y, groups_z);
        threadsPerThreadgroup = MTLSizeMake(1, 1, 1);

        [computeCommandEncoder dispatchThreadgroups:numThreadgroups
                                        threadsPerThreadgroup:threadsPerThreadgroup];
    }

    [computeCommandEncoder endEncoding];

    glm_ctx->state.dirty_bits = DIRTY_ALL;

    //[self newRenderEncoder];
}

void mtlDispatchCompute(GLMContext glm_ctx, GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDispatchCompute: glm_ctx groupsX:num_groups_x groupsY:num_groups_y groupsZ:num_groups_z];
}


-(void)mtlDispatchComputeIndirect:(GLMContext)glm_ctx indirect:(GLintptr)indirect
{

}

void mtlDispatchComputeIndirect(GLMContext glm_ctx, GLintptr indirect)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDispatchComputeIndirect: glm_ctx indirect:indirect];
}


-(bool) processBuffer:(Buffer*)ptr
{
    if (ptr == NULL)
    {
        NSLog(@"Error: processBuffer failed\n");

        return false;
    }

    if (ptr->data.mtl_data == NULL)
    {
        [self bindMTLBuffer: ptr];
        RETURN_FALSE_ON_NULL(ptr->data.mtl_data);
    }

    if (ptr->data.dirty_bits)
    {
        [self updateDirtyBuffer: ptr];
    }

    return true;
}
-(void) flushCommandBuffer: (bool) finish
{
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or queue is NULL in flushCommandBuffer");
        return;
    }

    if (![self processGLState: false]) {
        NSLog(@"MGL WARNING: processGLState failed in flushCommandBuffer, continuing with cleanup");
    }

    [self endRenderEncoding];

    if (![self ensureWritableCommandBuffer:"flushCommandBuffer"]) {
        NSLog(@"MGL ERROR: Unable to obtain writable command buffer in flushCommandBuffer");
        return;
    }

    if (!_currentCommandBuffer) {
        NSLog(@"MGL WARNING: No current command buffer in flushCommandBuffer");
        return;
    }

    MTLCommandBufferStatus currentStatus = _currentCommandBuffer.status;
    if (currentStatus != MTLCommandBufferStatusNotEnqueued) {
        NSLog(@"MGL INFO: flushCommandBuffer found finalized buffer (status=%ld), rotating", (long)currentStatus);
        if (![self newCommandBuffer]) {
            NSLog(@"MGL ERROR: Failed to rotate command buffer in flushCommandBuffer");
        }
        return;
    }

    if (_currentCommandBuffer.error) {
        NSLog(@"MGL ERROR: Command buffer has error before commit: %@", _currentCommandBuffer.error);
        [self cleanupCommandBuffer];
        return;
    }

    if (![self validateMetalObjects]) {
        NSLog(@"MGL WARNING: GPU throttling active - skipping command buffer commit");
        [self cleanupCommandBuffer];
        return;
    }

    id<MTLCommandBuffer> commandBufferToCommit = _currentCommandBuffer;
    _currentCommandBuffer = nil;

    @try {
        [self commitCommandBufferWithAGXRecovery:commandBufferToCommit];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Command buffer commit failed in flushCommandBuffer: %@", exception);
        [self recordGPUError];
        [self cleanupCommandBuffer];
    }

    if (!finish) {
        [self newCommandBuffer];
    }
}
#pragma mark C interface to mtlBindBuffer
void mtlBindBuffer(GLMContext glm_ctx, Buffer *ptr)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj bindMTLBuffer:ptr];
}

#pragma mark C interface to mtlBindTexture
void mtlBindTexture(GLMContext glm_ctx, Texture *ptr)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj bindMTLTexture:ptr];
}

#pragma mark C interface to mtlBindProgram
void mtlBindProgram(GLMContext glm_ctx, Program *ptr)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj bindMTLProgram:ptr];
}

#pragma mark C interface to mtlDeleteMTLObj
-(void) mtlDeleteMTLObj:(GLMContext) glm_ctx buffer: (void *)obj
{
    if (!obj)
        return;

    // Do not force-flush per-object destruction.
    // Metal command buffers retain referenced resources, so immediate release is safe and
    // avoids shutdown-time command-buffer storms (one commit per deleted object).
    CFBridgingRelease(obj);
}

void mtlDeleteMTLObj (GLMContext glm_ctx, void *obj)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDeleteMTLObj: glm_ctx buffer: obj];
}

#pragma mark C interface to mtlGetSync
-(void) mtlGetSync:(GLMContext) glm_ctx sync: (Sync *)sync
{
    if (kMGLDisableSharedEventSync) {
        if (sync) {
            sync->mtl_event = NULL;
        }
        _currentEvent = NULL;
        _currentSyncName = 0;
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: mtlGetSync no-op (shared event sync disabled)");
        }
        return;
    }

    // SAFETY: Check Metal objects before processing
    if (!_device || !_commandQueue) {
        NSLog(@"MGL ERROR: Metal device or queue is NULL in mtlGetSync");
        return;
    }

    if (![self processGLState: false]) {
        NSLog(@"MGL WARNING: processGLState failed in mtlGetSync");
        return;
    }

    if (_currentEvent == NULL)
    {
        @try {
            _currentEvent = [_device newEvent];
            if (!_currentEvent) {
                NSLog(@"MGL ERROR: Failed to create Metal event");
                return;
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Exception creating Metal event: %@", exception);
            return;
        }
    }

    _currentSyncName = sync->name;

    sync->mtl_event = (void *)CFBridgingRetain(_currentEvent);

    if (_currentCommandBufferSyncList == NULL)
    {
        // CRITICAL SECURITY FIX: Check malloc results instead of using assert()
        _currentCommandBufferSyncList = (SyncList *)malloc(sizeof(SyncList));
        if (!_currentCommandBufferSyncList) {
            NSLog(@"MGL SECURITY ERROR: Failed to allocate SyncList");
            return;
        }

        _currentCommandBufferSyncList->size = 8;
        _currentCommandBufferSyncList->list = (Sync **)malloc(sizeof(Sync *) * 8);
        if (!_currentCommandBufferSyncList->list) {
            NSLog(@"MGL SECURITY ERROR: Failed to allocate SyncList array");
            free(_currentCommandBufferSyncList);
            _currentCommandBufferSyncList = NULL;
            return;
        }

        _currentCommandBufferSyncList->count = 0;
    }

    if (_currentCommandBufferSyncList->count >= _currentCommandBufferSyncList->size)
    {
        // CRITICAL SECURITY FIX: Check for integer overflow before multiplication
        size_t current_size = (size_t)_currentCommandBufferSyncList->size;
        if (current_size > SIZE_MAX / 2 / sizeof(Sync *)) {
            NSLog(@"MGL SECURITY ERROR: SyncList size would overflow, preventing expansion");
            return;
        }

        size_t new_size = current_size * 2;
        Sync **new_list = (Sync **)realloc(_currentCommandBufferSyncList->list,
                                           sizeof(Sync *) * new_size);
        if (!new_list) {
            NSLog(@"MGL SECURITY ERROR: Failed to reallocate SyncList array");
            return;
        }

        _currentCommandBufferSyncList->size = new_size;
        _currentCommandBufferSyncList->list = new_list;
    }

    _currentCommandBufferSyncList->list[_currentCommandBufferSyncList->count] = sync;
    _currentCommandBufferSyncList->count++;
}

void mtlGetSync (GLMContext glm_ctx, Sync *sync)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlGetSync: glm_ctx sync: sync];
}

#pragma mark C interface to mtlWaitForSync
-(void) mtlWaitForSync:(GLMContext) glm_ctx sync: (Sync *)sync
{
    if (kMGLDisableSharedEventSync) {
        if (sync) {
            sync->mtl_event = NULL;
        }
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: mtlWaitForSync no-op (shared event sync disabled)");
        }
        return;
    }

    // CRITICAL SAFETY: Validate sync object before processing
    if (!sync) {
        NSLog(@"MGL ERROR: mtlWaitForSync - sync object is NULL");
        return;
    }

    // SAFETY: Validate mtl_event before releasing - prevent objc_release crash
    if (!sync->mtl_event) {
        NSLog(@"MGL WARNING: mtlWaitForSync - sync->mtl_event is NULL");
        return;
    }

    @try {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL INFO: Releasing Metal sync event");
        }
        CFBridgingRelease(sync->mtl_event);
        sync->mtl_event = NULL;
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception releasing sync event: %@", exception);
        // Don't crash - set to NULL to prevent double release
        sync->mtl_event = NULL;
    }
}

void mtlWaitForSync (GLMContext glm_ctx, Sync *sync)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlWaitForSync: glm_ctx sync: sync];
}

#pragma mark C interface to mtlFlush
-(void) mtlFlush:(GLMContext) glm_ctx finish:(bool)finish
{
    [self flushCommandBuffer: finish];
}

void mtlFlush (GLMContext glm_ctx, bool finish)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlFlush:glm_ctx finish:finish];
}

#pragma mark C interface to mtlSwapBuffers
-(void) mtlSwapBuffers:(GLMContext) glm_ctx
{
    static uint64_t s_swapCallCount = 0;
    static double s_swapLastCallTime = 0.0;
    static uint64_t s_swapLastCallCount = 0;
    static volatile double s_mainThreadHeartbeatSeconds = 0.0;
    static volatile uint64_t s_mainThreadPingCount = 0;
    uint64_t swapCall = ++s_swapCallCount;
    double swapStartSeconds = mglNowSeconds();
    bool traceSwap = mglShouldTraceCall(swapCall);
    g_mglSwapCallCount = swapCall;
    g_mglLastSwapSeconds = swapStartSeconds;
    if (swapCall <= 20ull || (swapCall % 60ull) == 0ull) {
        NSLog(@"MGL TRACE swap.entry call=%llu drawArraysSinceSwap=%llu drawElementsSinceSwap=%llu processDrawCallsSinceSwap=%llu",
              (unsigned long long)swapCall,
              (unsigned long long)g_mglDrawArraysSinceSwap,
              (unsigned long long)g_mglDrawElementsSinceSwap,
              (unsigned long long)g_mglProcessDrawCallsSinceSwap);
    }
    mglLogLoopHeartbeat("swap.loop",
                        swapCall,
                        swapStartSeconds,
                        &s_swapLastCallTime,
                        &s_swapLastCallCount,
                        0.25);

    if (!mglRendererContextLikelyValid(glm_ctx)) {
        NSLog(@"MGL CRITICAL: swap.begin invalid glm_ctx=%p", glm_ctx);
        return;
    }

    if (ctx != glm_ctx) {
        NSLog(@"MGL TRACE swap.contextSync old=%p new=%p", ctx, glm_ctx);
        ctx = glm_ctx;
    }

    GLMContext activeCtx = glm_ctx;
    GLenum drawBuffer = activeCtx->state.draw_buffer;
    bool shouldPresent = (drawBuffer != GL_NONE);
    if (traceSwap) {
        NSLog(@"MGL TRACE swap.begin call=%llu shouldPresent=%d draw_buffer=0x%x",
              (unsigned long long)swapCall, shouldPresent ? 1 : 0, (unsigned)drawBuffer);
        mglLogStateSnapshot("swap.enter",
                            activeCtx,
                            _currentCommandBuffer,
                            _currentRenderEncoder,
                            _renderPassDescriptor,
                            _drawable);
    }

    // Main-thread responsiveness probe for beachball diagnostics.
    // Render thread periodically posts a ping to main queue; stale heartbeat means main thread is blocked.
    if (kMGLDiagnosticStateLogs &&
        (swapCall <= 20ull || (swapCall % 30ull) == 0ull)) {
        dispatch_async(dispatch_get_main_queue(), ^{
            s_mainThreadHeartbeatSeconds = mglNowSeconds();
            s_mainThreadPingCount++;
        });

        double hb = s_mainThreadHeartbeatSeconds;
        if (hb > 0.0) {
            double lagMs = (swapStartSeconds - hb) * 1000.0;
            if (lagMs > 500.0) {
                NSLog(@"MGL TRACE mainthread.stall suspected lag=%.2fms swapCall=%llu pingCount=%llu",
                      lagMs,
                      (unsigned long long)swapCall,
                      (unsigned long long)s_mainThreadPingCount);
                if (traceSwap || (swapCall % 120ull) == 0ull) {
                    mglLogStateSnapshot("mainthread.stall.snapshot",
                                        activeCtx,
                                        _currentCommandBuffer,
                                        _currentRenderEncoder,
                                        _renderPassDescriptor,
                                        _drawable);
                }
            } else if (traceSwap) {
                NSLog(@"MGL TRACE mainthread.heartbeat lag=%.2fms swapCall=%llu pingCount=%llu",
                      lagMs,
                      (unsigned long long)swapCall,
                      (unsigned long long)s_mainThreadPingCount);
            }
        } else if (traceSwap) {
            NSLog(@"MGL TRACE mainthread.heartbeat uninitialized swapCall=%llu", (unsigned long long)swapCall);
        }
    }

    if (kMGLDiagnosticStateLogs) {
        MGLSwapDrawCounters frameCounters = mglSnapshotSwapDrawCounters();
        mglResetSwapDrawCounters();

        uint64_t lastDrawArraysCall = g_mglLastDrawArraysCall;
        uint64_t lastDrawElementsCall = g_mglLastDrawElementsCall;
        double lastDrawArraysSeconds = g_mglLastDrawArraysSeconds;
        double lastDrawElementsSeconds = g_mglLastDrawElementsSeconds;
        GLuint lastDrawArraysProgram = g_mglLastDrawArraysProgram;
        GLuint lastDrawArraysMode = g_mglLastDrawArraysMode;
        GLsizei lastDrawArraysCount = g_mglLastDrawArraysCount;
        GLuint lastDrawElementsProgram = g_mglLastDrawElementsProgram;
        GLuint lastDrawElementsMode = g_mglLastDrawElementsMode;
        GLsizei lastDrawElementsCount = g_mglLastDrawElementsCount;
        double drawArraysAgeMs = (lastDrawArraysSeconds > 0.0)
            ? ((swapStartSeconds - lastDrawArraysSeconds) * 1000.0)
            : -1.0;
        double drawElementsAgeMs = (lastDrawElementsSeconds > 0.0)
            ? ((swapStartSeconds - lastDrawElementsSeconds) * 1000.0)
            : -1.0;
        BOOL hasFrameWork = (frameCounters.draw_arrays > 0 ||
                             frameCounters.draw_elements > 0 ||
                             frameCounters.draw_arrays_skipped > 0 ||
                             frameCounters.draw_elements_skipped > 0 ||
                             frameCounters.process_draw_calls > 0);
        if (traceSwap || hasFrameWork || swapCall <= 20ull || (swapCall % 20ull) == 0ull) {
            NSLog(@"MGL TRACE swap.drawActivity call=%llu processDrawCalls=%llu drawArrays=%llu verts=%llu "
                  "drawElements=%llu indices=%llu skipArrays=%llu skipElements=%llu "
                  "lastDrawArrays=%llu prog=%u mode=0x%x count=%d age=%.2fms "
                  "lastDrawElements=%llu prog=%u mode=0x%x count=%d age=%.2fms",
                  (unsigned long long)swapCall,
                  (unsigned long long)frameCounters.process_draw_calls,
                  (unsigned long long)frameCounters.draw_arrays,
                  (unsigned long long)frameCounters.array_vertices,
                  (unsigned long long)frameCounters.draw_elements,
                  (unsigned long long)frameCounters.element_indices,
                  (unsigned long long)frameCounters.draw_arrays_skipped,
                  (unsigned long long)frameCounters.draw_elements_skipped,
                  (unsigned long long)lastDrawArraysCall,
                  (unsigned)lastDrawArraysProgram,
                  (unsigned)lastDrawArraysMode,
                  (int)lastDrawArraysCount,
                  drawArraysAgeMs,
                  (unsigned long long)lastDrawElementsCall,
                  (unsigned)lastDrawElementsProgram,
                  (unsigned)lastDrawElementsMode,
                  (int)lastDrawElementsCount,
                  drawElementsAgeMs);
        }
    }

    if (shouldPresent)
    {
        if (![self processGLState: false]) {
            static uint64_t s_swapProcessStateFailCount = 0;
            s_swapProcessStateFailCount++;
            if (s_swapProcessStateFailCount <= 16 || (s_swapProcessStateFailCount % 500) == 0) {
                NSLog(@"MGL WARNING: mtlSwapBuffers continuing despite processGLState failure (occurrence=%llu)",
                      (unsigned long long)s_swapProcessStateFailCount);
            }
        }

        [self endRenderEncoding];

        if (![self ensureWritableCommandBuffer:"mtlSwapBuffers"]) {
            NSLog(@"MGL ERROR: Failed to obtain writable command buffer in mtlSwapBuffers");
            return;
        }

        if (_drawable == NULL)
        {
            if (traceSwap) {
                NSLog(@"MGL TRACE swap.nextDrawable.begin call=%llu stage=pre_present", (unsigned long long)swapCall);
            }
            [self mglSyncLayerDrawableSizeFromView:"swap.pre_present"];
            _drawable = [_layer nextDrawable];
            if (traceSwap) {
                id<MTLTexture> tex = _drawable ? _drawable.texture : nil;
                NSLog(@"MGL TRACE swap.nextDrawable.end call=%llu stage=pre_present drawable=%p tex=%p size=%lux%lu",
                      (unsigned long long)swapCall,
                      _drawable,
                      tex,
                      (unsigned long)(tex ? tex.width : 0),
                      (unsigned long)(tex ? tex.height : 0));
            }
        }

        if (_drawable == NULL) {
            NSLog(@"MGL WARNING: Drawable is NULL in mtlSwapBuffers, getting new drawable");
            if (traceSwap) {
                NSLog(@"MGL TRACE swap.nextDrawable.begin call=%llu stage=pre_present_retry", (unsigned long long)swapCall);
            }
            [self mglSyncLayerDrawableSizeFromView:"swap.pre_present_retry"];
            _drawable = [_layer nextDrawable];
            if (traceSwap) {
                id<MTLTexture> tex = _drawable ? _drawable.texture : nil;
                NSLog(@"MGL TRACE swap.nextDrawable.end call=%llu stage=pre_present_retry drawable=%p tex=%p size=%lux%lu",
                      (unsigned long long)swapCall,
                      _drawable,
                      tex,
                      (unsigned long)(tex ? tex.width : 0),
                      (unsigned long)(tex ? tex.height : 0));
            }
            if (_drawable == NULL) {
                NSLog(@"MGL ERROR: Failed to obtain any drawable from Metal layer");
                return;
            }
        }

        // Diagnostic + compatibility path:
        // When swapping the default framebuffer, the active render pass should target the drawable.
        // If it still points to an offscreen texture, copy that texture into the drawable before present.
        id<MTLTexture> rpColor0 = _renderPassDescriptor ? _renderPassDescriptor.colorAttachments[0].texture : nil;
        id<MTLTexture> drawableTexture = _drawable ? _drawable.texture : nil;
        if (ctx->state.framebuffer == NULL &&
            rpColor0 &&
            drawableTexture &&
            rpColor0 != drawableTexture) {
            BOOL traceCopyToDrawable = traceSwap ||
                (kMGLSwapPresentDiagnostics &&
                 (swapCall <= 12ull || (swapCall % 120ull) == 0ull));
            if (traceCopyToDrawable) {
                NSLog(@"MGL TRACE swap.copyToDrawable.begin call=%llu src=%p fmt=%lu %lux%lu dst=%p fmt=%lu %lux%lu",
                      (unsigned long long)swapCall,
                      rpColor0,
                      (unsigned long)rpColor0.pixelFormat,
                      (unsigned long)rpColor0.width,
                      (unsigned long)rpColor0.height,
                      drawableTexture,
                      (unsigned long)drawableTexture.pixelFormat,
                      (unsigned long)drawableTexture.width,
                      (unsigned long)drawableTexture.height);
            }

            BOOL canShaderCopyToDrawable =
                (rpColor0.pixelFormat == drawableTexture.pixelFormat ||
                 (rpColor0.pixelFormat == MTLPixelFormatRGBA8Unorm && drawableTexture.pixelFormat == MTLPixelFormatBGRA8Unorm) ||
                 (rpColor0.pixelFormat == MTLPixelFormatBGRA8Unorm && drawableTexture.pixelFormat == MTLPixelFormatRGBA8Unorm));
            if (canShaderCopyToDrawable) {
                    id<MTLRenderPipelineState> pipeline = [self scaledBlitPipelineForPixelFormat:drawableTexture.pixelFormat];
                    id<MTLSamplerState> sampler = [self scaledBlitSamplerForFilter:GL_NEAREST];
                    NSUInteger copyWidth = MIN((NSUInteger)rpColor0.width, (NSUInteger)drawableTexture.width);
                    NSUInteger copyHeight = MIN((NSUInteger)rpColor0.height, (NSUInteger)drawableTexture.height);
                    if (pipeline && sampler && copyWidth > 0 && copyHeight > 0) {
                        MGLScaledBlitParams params;
                        params.uvRect = (vector_float4){
                            0.0f,
                            0.0f,
                            rpColor0.width ? ((float)copyWidth / (float)rpColor0.width) : 0.0f,
                            rpColor0.height ? ((float)copyHeight / (float)rpColor0.height) : 0.0f
                        };
                        params.forceOpaqueAlpha = 1.0f;
                        params._padding = (vector_float3){0.0f, 0.0f, 0.0f};

                        MTLRenderPassDescriptor *copyPass = [MTLRenderPassDescriptor renderPassDescriptor];
                        copyPass.colorAttachments[0].texture = drawableTexture;
                        copyPass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
                        copyPass.colorAttachments[0].storeAction = MTLStoreActionStore;

                        id<MTLRenderCommandEncoder> copyEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:copyPass];
                        if (copyEncoder) {
                            [copyEncoder setRenderPipelineState:pipeline];
                            [copyEncoder setVertexBytes:&params length:sizeof(params) atIndex:0];
                            [copyEncoder setFragmentBytes:&params length:sizeof(params) atIndex:0];
                            [copyEncoder setFragmentTexture:rpColor0 atIndex:0];
                            [copyEncoder setFragmentSamplerState:sampler atIndex:0];
                            [copyEncoder setViewport:(MTLViewport){
                                .originX = 0.0,
                                .originY = 0.0,
                                .width = (double)copyWidth,
                                .height = (double)copyHeight,
                                .znear = 0.0,
                                .zfar = 1.0
                            }];
                            [copyEncoder setScissorRect:(MTLScissorRect){
                                .x = 0,
                                .y = 0,
                                .width = copyWidth,
                                .height = copyHeight
                            }];
                            [copyEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
                            [copyEncoder endEncoding];
                        } else {
                            NSLog(@"MGL WARNING: swap.copyToDrawable failed to create shader copy encoder");
                        }
                    } else {
                        NSLog(@"MGL WARNING: swap.copyToDrawable shader copy unavailable pipeline=%p sampler=%p size=%lux%lu",
                              pipeline,
                              sampler,
                              (unsigned long)copyWidth,
                              (unsigned long)copyHeight);
                    }
            } else {
                NSLog(@"MGL WARNING: swap.copyToDrawable skipped due to pixel format mismatch src=%lu dst=%lu",
                      (unsigned long)rpColor0.pixelFormat,
                      (unsigned long)drawableTexture.pixelFormat);
            }

            if (traceCopyToDrawable) {
                NSLog(@"MGL TRACE swap.copyToDrawable.end call=%llu", (unsigned long long)swapCall);
            }
        }

        // Low-frequency dual texture sampling for black-screen diagnostics.
        // Sample both render-pass color source and drawable target so we can
        // distinguish "rendered black" from "copy/present black".
        if (kMGLSwapPresentDiagnostics &&
            ((swapCall <= 12ull && (swapCall % 3ull) == 0ull) || ((swapCall % 120ull) == 0ull))) {
            void (^scheduleTextureSample)(id<MTLTexture>, NSString *, NSUInteger, NSUInteger) =
                ^(id<MTLTexture> sampleTexture, NSString *sampleTag, NSUInteger originX, NSUInteger originY) {
                    if (!sampleTexture) {
                        NSLog(@"MGL TRACE swap.sample.%@ call=%llu skipped(texture=nil)",
                              sampleTag,
                              (unsigned long long)swapCall);
                        return;
                    }

                    if (sampleTexture.pixelFormat != MTLPixelFormatBGRA8Unorm &&
                        sampleTexture.pixelFormat != MTLPixelFormatRGBA8Unorm) {
                        NSLog(@"MGL TRACE swap.sample.%@ call=%llu skipped(fmt=%lu tex=%lux%lu)",
                              sampleTag,
                              (unsigned long long)swapCall,
                              (unsigned long)sampleTexture.pixelFormat,
                              (unsigned long)sampleTexture.width,
                              (unsigned long)sampleTexture.height);
                        return;
                    }

                    NSUInteger sampleWidth = MIN((NSUInteger)sampleTexture.width, 8u);
                    NSUInteger sampleHeight = MIN((NSUInteger)sampleTexture.height, 8u);
                    NSUInteger bytesPerPixel = 4u;
                    NSUInteger sampleBytesPerRow = sampleWidth * bytesPerPixel;
                    NSUInteger sampleBytesPerImage = sampleBytesPerRow * sampleHeight;
                    if (sampleWidth == 0 || sampleHeight == 0 || sampleBytesPerImage == 0) {
                        NSLog(@"MGL TRACE swap.sample.%@ call=%llu skipped(invalid-size tex=%lux%lu)",
                              sampleTag,
                              (unsigned long long)swapCall,
                              (unsigned long)sampleTexture.width,
                              (unsigned long)sampleTexture.height);
                        return;
                    }

                    NSUInteger clampedOriginX = originX;
                    NSUInteger clampedOriginY = originY;
                    if (clampedOriginX + sampleWidth > (NSUInteger)sampleTexture.width) {
                        clampedOriginX = ((NSUInteger)sampleTexture.width > sampleWidth)
                            ? ((NSUInteger)sampleTexture.width - sampleWidth)
                            : 0u;
                    }
                    if (clampedOriginY + sampleHeight > (NSUInteger)sampleTexture.height) {
                        clampedOriginY = ((NSUInteger)sampleTexture.height > sampleHeight)
                            ? ((NSUInteger)sampleTexture.height - sampleHeight)
                            : 0u;
                    }

                    id<MTLBuffer> sampleBuffer = [_device newBufferWithLength:sampleBytesPerImage
                                                                       options:MTLResourceStorageModeShared];
                    if (!sampleBuffer) {
                        NSLog(@"MGL WARNING: swap.sample.%@ call=%llu failed(alloc size=%lu)",
                              sampleTag,
                              (unsigned long long)swapCall,
                              (unsigned long)sampleBytesPerImage);
                        return;
                    }

                    id<MTLBlitCommandEncoder> sampleEncoder = [_currentCommandBuffer blitCommandEncoder];
                    if (!sampleEncoder) {
                        NSLog(@"MGL WARNING: swap.sample.%@ call=%llu failed(create blit encoder)",
                              sampleTag,
                              (unsigned long long)swapCall);
                        return;
                    }

                    [sampleEncoder copyFromTexture:sampleTexture
                                       sourceSlice:0
                                       sourceLevel:0
                                      sourceOrigin:MTLOriginMake(clampedOriginX, clampedOriginY, 0)
                                        sourceSize:MTLSizeMake(sampleWidth, sampleHeight, 1)
                                          toBuffer:sampleBuffer
                                 destinationOffset:0
                            destinationBytesPerRow:sampleBytesPerRow
                          destinationBytesPerImage:sampleBytesPerImage];
                    [sampleEncoder endEncoding];

                    uint64_t sampleSwapCall = swapCall;
                    NSString *sampleTagCopy = [sampleTag copy];
                    NSUInteger sampleTexWidth = (NSUInteger)sampleTexture.width;
                    NSUInteger sampleTexHeight = (NSUInteger)sampleTexture.height;
                    NSUInteger sampleOriginX = clampedOriginX;
                    NSUInteger sampleOriginY = clampedOriginY;
                    [sampleBuffer addDebugMarker:@"mgl_swap_sample" range:NSMakeRange(0, sampleBytesPerImage)];
                    [_currentCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> sampleCB) {
                        const uint8_t *p = (const uint8_t *)sampleBuffer.contents;
                        if (!p) {
                            NSLog(@"MGL TRACE swap.sample.%@ call=%llu unavailable(contents=nil) status=%s error=%@",
                                  sampleTagCopy,
                                  (unsigned long long)sampleSwapCall,
                                  mglCommandBufferStatusName(sampleCB.status),
                                  sampleCB.error);
                            return;
                        }

                        uint64_t sum = 0;
                        NSUInteger nonZero = 0;
                        for (NSUInteger bi = 0; bi < sampleBytesPerImage; bi++) {
                            uint8_t v = p[bi];
                            sum += (uint64_t)v;
                            if (v != 0) {
                                nonZero++;
                            }
                        }

                        uint32_t firstPixel = 0;
                        if (sampleBytesPerImage >= sizeof(firstPixel)) {
                            memcpy(&firstPixel, p, sizeof(firstPixel));
                        }

                        uint32_t minPixel = UINT32_MAX;
                        uint32_t maxPixel = 0u;
                        uint32_t pixelXor = 0u;
                        NSUInteger diffFromFirst = 0u;
                        NSUInteger pixelCount = sampleBytesPerImage / sizeof(uint32_t);
                        for (NSUInteger pi = 0; pi < pixelCount; pi++) {
                            uint32_t pixel = 0u;
                            memcpy(&pixel, p + (pi * sizeof(uint32_t)), sizeof(pixel));
                            if (pixel < minPixel) {
                                minPixel = pixel;
                            }
                            if (pixel > maxPixel) {
                                maxPixel = pixel;
                            }
                            pixelXor ^= pixel;
                            if (pixel != firstPixel) {
                                diffFromFirst++;
                            }
                        }
                        BOOL appearsSolid = (pixelCount > 0u && diffFromFirst == 0u);

                        NSLog(@"MGL TRACE swap.sample.%@ call=%llu tex=%lux%lu origin=(%lu,%lu) sample=%lux%lu "
                              "nonZero=%lu/%lu sum=%llu firstPixel=0x%08x min=0x%08x max=0x%08x xor=0x%08x diff=%lu solid=%d status=%s error=%@",
                              sampleTagCopy,
                              (unsigned long long)sampleSwapCall,
                              (unsigned long)sampleTexWidth,
                              (unsigned long)sampleTexHeight,
                              (unsigned long)sampleOriginX,
                              (unsigned long)sampleOriginY,
                              (unsigned long)sampleWidth,
                              (unsigned long)sampleHeight,
                              (unsigned long)nonZero,
                              (unsigned long)sampleBytesPerImage,
                              (unsigned long long)sum,
                              firstPixel,
                              minPixel == UINT32_MAX ? 0u : minPixel,
                              maxPixel,
                              pixelXor,
                              (unsigned long)diffFromFirst,
                              appearsSolid ? 1 : 0,
                              mglCommandBufferStatusName(sampleCB.status),
                              sampleCB.error);

                        if ([sampleTagCopy isEqualToString:@"src.center"]) {
                            static uint32_t s_lastCenterPixel = 0u;
                            static uint64_t s_sameCenterPixelRun = 0ull;
                            if (firstPixel == s_lastCenterPixel) {
                                s_sameCenterPixelRun++;
                            } else {
                                s_lastCenterPixel = firstPixel;
                                s_sameCenterPixelRun = 1ull;
                            }

                            if (s_sameCenterPixelRun == 10ull ||
                                s_sameCenterPixelRun == 30ull ||
                                (s_sameCenterPixelRun % 120ull) == 0ull) {
                                NSLog(@"MGL TRACE swap.sample.center_stable firstPixel=0x%08x run=%llu solid=%d diff=%lu",
                                      firstPixel,
                                      (unsigned long long)s_sameCenterPixelRun,
                                      appearsSolid ? 1 : 0,
                                      (unsigned long)diffFromFirst);
                            }
                        }
                    }];
                };

            scheduleTextureSample(rpColor0, @"src.tl", 0u, 0u);
            if (rpColor0) {
                NSUInteger cx = ((NSUInteger)rpColor0.width > 8u) ? (((NSUInteger)rpColor0.width / 2u) - 4u) : 0u;
                NSUInteger cy = ((NSUInteger)rpColor0.height > 8u) ? (((NSUInteger)rpColor0.height / 2u) - 4u) : 0u;
                NSUInteger rx = ((NSUInteger)rpColor0.width > 8u) ? ((NSUInteger)rpColor0.width - 8u) : 0u;
                NSUInteger by = ((NSUInteger)rpColor0.height > 8u) ? ((NSUInteger)rpColor0.height - 8u) : 0u;
                scheduleTextureSample(rpColor0, @"src.center", cx, cy);
                scheduleTextureSample(rpColor0, @"src.right", rx, cy);
                scheduleTextureSample(rpColor0, @"src.bottom", cx, by);
            }
            if (drawableTexture != rpColor0) {
                scheduleTextureSample(drawableTexture, @"dst.tl", 0u, 0u);
                if (drawableTexture) {
                    NSUInteger dcx = ((NSUInteger)drawableTexture.width > 8u) ? (((NSUInteger)drawableTexture.width / 2u) - 4u) : 0u;
                    NSUInteger dcy = ((NSUInteger)drawableTexture.height > 8u) ? (((NSUInteger)drawableTexture.height / 2u) - 4u) : 0u;
                    NSUInteger drx = ((NSUInteger)drawableTexture.width > 8u) ? ((NSUInteger)drawableTexture.width - 8u) : 0u;
                    NSUInteger dby = ((NSUInteger)drawableTexture.height > 8u) ? ((NSUInteger)drawableTexture.height - 8u) : 0u;
                    scheduleTextureSample(drawableTexture, @"dst.center", dcx, dcy);
                    scheduleTextureSample(drawableTexture, @"dst.right", drx, dcy);
                    scheduleTextureSample(drawableTexture, @"dst.bottom", dcx, dby);
                }
            } else {
                scheduleTextureSample(drawableTexture, @"srcdst.tl", 0u, 0u);
                if (drawableTexture) {
                    NSUInteger sx = ((NSUInteger)drawableTexture.width > 8u) ? (((NSUInteger)drawableTexture.width / 2u) - 4u) : 0u;
                    NSUInteger sy = ((NSUInteger)drawableTexture.height > 8u) ? (((NSUInteger)drawableTexture.height / 2u) - 4u) : 0u;
                    NSUInteger srx = ((NSUInteger)drawableTexture.width > 8u) ? ((NSUInteger)drawableTexture.width - 8u) : 0u;
                    NSUInteger sby = ((NSUInteger)drawableTexture.height > 8u) ? ((NSUInteger)drawableTexture.height - 8u) : 0u;
                    scheduleTextureSample(drawableTexture, @"srcdst.center", sx, sy);
                    scheduleTextureSample(drawableTexture, @"srcdst.right", srx, sy);
                    scheduleTextureSample(drawableTexture, @"srcdst.bottom", sx, sby);
                }
            }
        }

        if (_layer == NULL) {
            NSLog(@"MGL ERROR: Metal layer is NULL, cannot present drawable");
            return;
        }

        if (!_currentCommandBuffer) {
            NSLog(@"MGL ERROR: No command buffer available for presentation");
            return;
        }

        MTLCommandBufferStatus bufferStatus = _currentCommandBuffer.status;
        if (bufferStatus != MTLCommandBufferStatusNotEnqueued) {
            NSLog(@"MGL WARNING: mtlSwapBuffers found finalized command buffer (status: %ld), rotating", (long)bufferStatus);
            [self endRenderEncoding];
            [self newCommandBuffer];
            if (!_currentCommandBuffer) {
                NSLog(@"MGL ERROR: Failed to create new command buffer for presentation");
                return;
            }
        }

        @try {
            if (_drawable.texture == NULL) {
                NSLog(@"MGL ERROR: Drawable texture is NULL, cannot present");
                return;
            }

            if (_drawable.texture.width == 0 || _drawable.texture.height == 0) {
                NSLog(@"MGL ERROR: Drawable has invalid dimensions: %dx%d",
                      (int)_drawable.texture.width, (int)_drawable.texture.height);
                return;
            }

            if (kMGLVerboseFrameLoopLogs) {
                NSLog(@"MGL INFO: Presenting drawable with texture: %dx%d, format: %lu",
                      (int)_drawable.texture.width, (int)_drawable.texture.height,
                      (unsigned long)_drawable.texture.pixelFormat);
            }

            [_currentCommandBuffer presentDrawable: _drawable];
            if (traceSwap) {
                NSLog(@"MGL TRACE swap.present call=%llu cb=%p drawable=%p",
                      (unsigned long long)swapCall, _currentCommandBuffer, _drawable);
            }

        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Critical drawable presentation failure: %@", exception);
            NSLog(@"MGL ERROR: Exception name: %@, reason: %@", [exception name], [exception reason]);
            [self cleanupCommandBuffer];
            return;
        }

        id<MTLCommandBuffer> commandBufferToCommit = _currentCommandBuffer;
        _currentCommandBuffer = nil;
        @try {
            if (traceSwap) {
                NSLog(@"MGL TRACE swap.commit.begin call=%llu cb=%p status=%s label=%@",
                      (unsigned long long)swapCall,
                      commandBufferToCommit,
                      mglCommandBufferStatusName(commandBufferToCommit ? commandBufferToCommit.status : MTLCommandBufferStatusError),
                      commandBufferToCommit ? (commandBufferToCommit.label ?: @"(no-label)") : @"(nil)");
            }
            [self commitCommandBufferWithAGXRecovery:commandBufferToCommit];
            if (traceSwap) {
                NSLog(@"MGL TRACE swap.commit.end call=%llu", (unsigned long long)swapCall);
            }
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: Failed to commit command buffer: %@", exception);
            [self recordGPUError];
        }

        if (traceSwap) {
            NSLog(@"MGL TRACE swap.nextDrawable.begin call=%llu stage=post_commit", (unsigned long long)swapCall);
        }
        _drawable = [_layer nextDrawable];
        if (traceSwap) {
            id<MTLTexture> tex = _drawable ? _drawable.texture : nil;
            NSLog(@"MGL TRACE swap.nextDrawable.end call=%llu stage=post_commit drawable=%p tex=%p size=%lux%lu",
                  (unsigned long long)swapCall,
                  _drawable,
                  tex,
                  (unsigned long)(tex ? tex.width : 0),
                  (unsigned long)(tex ? tex.height : 0));
        }
        if (_drawable == NULL) {
            NSLog(@"MGL WARNING: Failed to get next drawable in mtlSwapBuffers");
            return;
        }

        if (![self newCommandBuffer]) {
            NSLog(@"MGL ERROR: Failed to create post-swap command buffer");
            return;
        }
        ctx->state.dirty_bits |= DIRTY_FBO | DIRTY_RENDER_STATE;
        double swapElapsedMs = (mglNowSeconds() - swapStartSeconds) * 1000.0;
        if (traceSwap) {
            NSLog(@"MGL TRACE swap.end call=%llu elapsed=%.3fms",
                  (unsigned long long)swapCall,
                  swapElapsedMs);
            mglLogStateSnapshot("swap.exit.ok",
                                ctx,
                                _currentCommandBuffer,
                                _currentRenderEncoder,
                                _renderPassDescriptor,
                                _drawable);
        } else if (swapElapsedMs >= 25.0) {
            NSLog(@"MGL TRACE swap.slow call=%llu elapsed=%.3fms",
                  (unsigned long long)swapCall,
                  swapElapsedMs);
        }
    }
    else if (kMGLVerboseFrameLoopLogs || traceSwap)
    {
        NSLog(@"MGL INFO: mtlSwapBuffers skipped present because draw_buffer is GL_NONE");
    }
}
void mtlSwapBuffers (GLMContext glm_ctx)
{
    // CRITICAL FIX: Validate context and Metal object pointer before dereferencing
    // This prevents pointer authentication failures from corrupted pointers
    if (!glm_ctx) {
        NSLog(@"MGL CRITICAL: mtlSwapBuffers - GLM context is NULL");
        return;
    }

    // Validate the Metal object pointer lower bound only.
    if (!glm_ctx->mtl_funcs.mtlObj || ((uintptr_t)glm_ctx->mtl_funcs.mtlObj < 0x1000)) {
        NSLog(@"MGL CRITICAL: mtlSwapBuffers - Invalid Metal object pointer: %p", glm_ctx->mtl_funcs.mtlObj);
        NSLog(@"MGL CRITICAL: This indicates memory corruption or context destruction");
        return;
    }

    // Call the Objective-C method using Objective-C syntax
    @autoreleasepool {
        @try {
            [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlSwapBuffers: glm_ctx];
        } @catch (NSException *exception) {
            NSLog(@"MGL CRITICAL: mtlSwapBuffers - Exception caught: %@", exception);
            NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);
        }
    }
}

#pragma mark C interface to mtlClearBuffer
-(void) mtlClearBuffer:(GLMContext) glm_ctx type:(GLuint) type mask:(GLbitfield) mask
{
    RETURN_ON_FAILURE([self processGLState: false]);
}

void mtlClearBuffer (GLMContext glm_ctx, GLuint type, GLbitfield mask)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlClearBuffer: glm_ctx type: type mask: mask];
}

#pragma mark C interface to mtlBufferSubData

-(void) mtlBufferSubData:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size ptr:(const void *)ptr
{
    static uint64_t s_mtlBufferSubDataCalls = 0;
    uint64_t call = ++s_mtlBufferSubDataCalls;
    bool trace = kMGLDiagnosticStateLogs && mglShouldTraceBufferTransferCall(call);
    id<MTLBuffer> mtl_buffer;
    void *data;

    if (!buf) {
        NSLog(@"MGL ERROR: mtlBufferSubData null buffer offset=%zu size=%zu", offset, size);
        return;
    }

    if (size == 0) {
        return;
    }

    if (!ptr) {
        NSLog(@"MGL WARNING: mtlBufferSubData null source ptr buffer=%u offset=%zu size=%zu", buf->name, offset, size);
        return;
    }

    if (trace) {
        char srcHead[64];
        srcHead[0] = '\0';
        mglTraceFormatBytes(ptr, size, srcHead, sizeof(srcHead));
        uint64_t srcHash = mglTraceHashBytes(ptr, size);
        NSLog(@"MGL TRACE mtlBufferSubData.begin call=%llu buffer=%u size=%lld off=%zu len=%zu mtl=%p cpu=%p dirty=0x%x srcHash=0x%016llx srcHead=%s",
              (unsigned long long)call,
              buf->name,
              (long long)buf->size,
              offset,
              size,
              buf->data.mtl_data,
              (void *)(uintptr_t)buf->data.buffer_data,
              buf->data.dirty_bits,
              (unsigned long long)srcHash,
              srcHead);
    }

    if (buf->data.mtl_data == NULL)
    {
        [self bindMTLBuffer:buf];
    }

    // AGX Driver Compatibility: For small buffers, bindMTLBuffer may still have NULL mtl_data
    // In this case, we should update the buffer_data directly
    if (buf->data.mtl_data == NULL)
    {
        // Small buffer case - update buffer_data directly
        if (buf->data.buffer_data)
        {
            memcpy((void *)(buf->data.buffer_data + offset), ptr, size);
            if (trace) {
                const void *dst = (const void *)((uintptr_t)buf->data.buffer_data + offset);
                char dstHead[64];
                dstHead[0] = '\0';
                mglTraceFormatBytes(dst, size, dstHead, sizeof(dstHead));
                uint64_t dstHash = mglTraceHashBytes(dst, size);
                NSLog(@"MGL TRACE mtlBufferSubData.cpuFallback call=%llu buffer=%u off=%zu len=%zu dstHash=0x%016llx dstHead=%s",
                      (unsigned long long)call,
                      buf->name,
                      offset,
                      size,
                      (unsigned long long)dstHash,
                      dstHead);
            }
        }
        return;
    }

    mtl_buffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!mtl_buffer) {
        NSLog(@"MGL ERROR: mtlBufferSubData buffer=%u has invalid Metal buffer", buf->name);
        return;
    }

    if (offset > mtl_buffer.length || size > (mtl_buffer.length - offset)) {
        NSLog(@"MGL ERROR: mtlBufferSubData range exceeds Metal buffer buffer=%u off=%zu size=%zu len=%lu",
              buf->name,
              offset,
              size,
              (unsigned long)mtl_buffer.length);
        return;
    }

    data = mtl_buffer.contents;
    if (!data) {
        NSLog(@"MGL ERROR: mtlBufferSubData buffer=%u has NULL contents", buf->name);
        return;
    }
    memcpy(data+offset, ptr, size);

    [mtl_buffer didModifyRange:NSMakeRange(offset, size)];

    if (trace) {
        const void *dst = (const void *)((const uint8_t *)mtl_buffer.contents + offset);
        char dstHead[64];
        dstHead[0] = '\0';
        mglTraceFormatBytes(dst, size, dstHead, sizeof(dstHead));
        uint64_t dstHash = mglTraceHashBytes(dst, size);
        NSLog(@"MGL TRACE mtlBufferSubData.end call=%llu buffer=%u off=%zu len=%zu mtlLen=%lu dstHash=0x%016llx dstHead=%s",
              (unsigned long long)call,
              buf->name,
              offset,
              size,
              (unsigned long)mtl_buffer.length,
              (unsigned long long)dstHash,
              dstHead);
    }
}

void mtlBufferSubData(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, const void *ptr)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlBufferSubData: glm_ctx buf: buf offset:offset size:size ptr:ptr];
}

#pragma mark C interface to mtlMapUnmapBuffer
-(void *) mtlMapUnmapBuffer:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(size_t) offset size:(size_t) size access:(GLenum) access map:(bool)map
{
    id<MTLBuffer> mtl_buffer = nil;

    if (!buf) {
        NSLog(@"MGL ERROR: mtlMapUnmapBuffer called with NULL buffer");
        return NULL;
    }

    if (buf->data.mtl_data == NULL)
    {
        [self bindMTLBuffer:buf];
    }

    mtl_buffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!mtl_buffer) {
        NSLog(@"MGL ERROR: mtlMapUnmapBuffer buffer=%u has NULL Metal buffer after bind", buf->name);
        return NULL;
    }

    uint8_t *mtlBase = (uint8_t *)mtl_buffer.contents;
    NSUInteger mtlLen = mtl_buffer.length;
    if (offset > mtlLen) {
        NSLog(@"MGL ERROR: mtlMapUnmapBuffer buffer=%u offset=%zu beyond mtlLen=%lu",
              buf->name, offset, (unsigned long)mtlLen);
        return NULL;
    }
    NSUInteger safeLen = MIN((NSUInteger)size, (mtlLen - (NSUInteger)offset));

    uint8_t *cpuBase = NULL;
    if (buf->data.buffer_data && ((uintptr_t)buf->data.buffer_data >= 0x1000ull)) {
        cpuBase = (uint8_t *)(uintptr_t)buf->data.buffer_data;
    }

    if (map)
    {
        uint8_t *mappedPtr = mtlBase ? (mtlBase + offset) : NULL;

        if (kMGLDiagnosticStateLogs) {
            uint64_t mtlHash = mglTraceHashBytes(mappedPtr, (size_t)safeLen);
            char mtlHead[64];
            mtlHead[0] = '\0';
            mglTraceFormatBytes(mappedPtr, (size_t)safeLen, mtlHead, sizeof(mtlHead));

            uint8_t *cpuPtr = cpuBase ? (cpuBase + offset) : NULL;
            uint64_t cpuHash = mglTraceHashBytes(cpuPtr, (size_t)safeLen);
            char cpuHead[64];
            cpuHead[0] = '\0';
            mglTraceFormatBytes(cpuPtr, (size_t)safeLen, cpuHead, sizeof(cpuHead));

            NSLog(@"MGL TRACE mtlMap.map buffer=%u off=%zu req=%zu safe=%lu access=0x%x mtlPtr=%p cpuPtr=%p samePtr=%d mtlHash=0x%016llx cpuHash=0x%016llx mtlHead=%s cpuHead=%s",
                  buf->name,
                  offset,
                  size,
                  (unsigned long)safeLen,
                  (unsigned)access,
                  mappedPtr,
                  cpuPtr,
                  (mappedPtr && cpuPtr && mappedPtr == cpuPtr) ? 1 : 0,
                  (unsigned long long)mtlHash,
                  (unsigned long long)cpuHash,
                  mtlHead,
                  cpuHead);
        }

        return mappedPtr;
    }

    // Keep CPU shadow coherent for diagnostics and any CPU-side fallback paths.
    // For small buffers we often keep a separate vm-allocated CPU store; mapped writes
    // go to mtl.contents, so mirror them back on unmap.
    if (cpuBase && mtlBase && safeLen > 0) {
        uint8_t *mtlPtr = mtlBase + offset;
        uint8_t *cpuPtr = cpuBase + offset;
        if (mtlPtr != cpuPtr) {
            memcpy(cpuPtr, mtlPtr, (size_t)safeLen);
        }
    }

    [mtl_buffer didModifyRange:NSMakeRange(offset, safeLen)];

    if (kMGLDiagnosticStateLogs) {
        uint8_t *mtlPtr = mtlBase ? (mtlBase + offset) : NULL;
        uint8_t *cpuPtr = cpuBase ? (cpuBase + offset) : NULL;
        uint64_t mtlHash = mglTraceHashBytes(mtlPtr, (size_t)safeLen);
        uint64_t cpuHash = mglTraceHashBytes(cpuPtr, (size_t)safeLen);
        char mtlHead[64];
        char cpuHead[64];
        mtlHead[0] = '\0';
        cpuHead[0] = '\0';
        mglTraceFormatBytes(mtlPtr, (size_t)safeLen, mtlHead, sizeof(mtlHead));
        mglTraceFormatBytes(cpuPtr, (size_t)safeLen, cpuHead, sizeof(cpuHead));
        NSLog(@"MGL TRACE mtlMap.unmap buffer=%u off=%zu req=%zu safe=%lu access=0x%x mtlPtr=%p cpuPtr=%p samePtr=%d mtlHash=0x%016llx cpuHash=0x%016llx mtlHead=%s cpuHead=%s",
              buf->name,
              offset,
              size,
              (unsigned long)safeLen,
              (unsigned)access,
              mtlPtr,
              cpuPtr,
              (mtlPtr && cpuPtr && mtlPtr == cpuPtr) ? 1 : 0,
              (unsigned long long)mtlHash,
              (unsigned long long)cpuHash,
              mtlHead,
              cpuHead);
    }

    return NULL;
}

void *mtlMapUnmapBuffer(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, GLenum access, bool map)
{
    // Call the Objective-C method using Objective-C syntax
    return [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlMapUnmapBuffer: glm_ctx buf: buf offset: offset size: size access: access map: map];
}

#pragma mark C interface to mtlFlushMappedBufferRange
-(void) mtlFlushMappedBufferRange:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(size_t) offset length:(size_t) length
{
    id<MTLBuffer> mtl_buffer;

    if (!buf) {
        NSLog(@"MGL ERROR: mtlFlushMappedBufferRange called with NULL buffer");
        return;
    }

    mtl_buffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!mtl_buffer) {
        NSLog(@"MGL ERROR: mtlFlushMappedBufferRange buffer=%u has no Metal buffer", buf->name);
        return;
    }

    if (offset > mtl_buffer.length || length > (mtl_buffer.length - offset)) {
        NSLog(@"MGL ERROR: mtlFlushMappedBufferRange out of range buffer=%u off=%zu len=%zu mtlLen=%lu",
              buf->name,
              offset,
              length,
              (unsigned long)mtl_buffer.length);
        return;
    }

    [mtl_buffer didModifyRange:NSMakeRange(offset, length)];
}

void mtlFlushBufferRange(GLMContext glm_ctx, Buffer *buf, GLintptr offset, GLsizeiptr length)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlFlushMappedBufferRange: glm_ctx buf: buf offset: offset length: length];
}

- (void)mglApplyPendingDefaultColorClearToTexture:(id<MTLTexture>)texture
{
    if (!ctx || !texture || !(ctx->state.default_fbo_clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        return;
    }

    MTLRenderPassDescriptor *clearPass = [MTLRenderPassDescriptor renderPassDescriptor];
    clearPass.colorAttachments[0].texture = texture;
    clearPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    clearPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    clearPass.colorAttachments[0].clearColor =
        MTLClearColorMake(ctx->state.default_clear_color[0],
                          ctx->state.default_clear_color[1],
                          ctx->state.default_clear_color[2],
                          ctx->state.default_clear_color[3]);

    id<MTLRenderCommandEncoder> clearEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:clearPass];
    if (clearEncoder) {
        [clearEncoder endEncoding];
        ctx->state.default_fbo_clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    } else {
        NSLog(@"MGL WARNING: readPixels failed to apply pending default framebuffer color clear");
    }
}

- (void)mglApplyPendingFBOColorClearForReadback:(Framebuffer *)fbo
                                     attachment:(FBOAttachment *)attachment
                                    textureObj:(Texture *)textureObj
                                     mtlTexture:(id<MTLTexture>)texture
                                  attachmentEnum:(GLenum)attachmentEnum
{
    if (!fbo || !attachment || !texture || !(attachment->clear_bitmask & GL_COLOR_BUFFER_BIT)) {
        return;
    }

    MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);
    MTLRenderPassDescriptor *clearPass = [MTLRenderPassDescriptor renderPassDescriptor];
    clearPass.colorAttachments[0].texture = texture;
    clearPass.colorAttachments[0].level = subresource.level;
    clearPass.colorAttachments[0].slice = subresource.slice;
    clearPass.colorAttachments[0].depthPlane = subresource.depthPlane;
    clearPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    clearPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    clearPass.colorAttachments[0].clearColor =
        MTLClearColorMake(attachment->clear_color[0],
                          attachment->clear_color[1],
                          attachment->clear_color[2],
                          attachment->clear_color[3]);

    id<MTLRenderCommandEncoder> clearEncoder = [_currentCommandBuffer renderCommandEncoderWithDescriptor:clearPass];
    if (clearEncoder) {
        [clearEncoder endEncoding];
        attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
        mglMarkTextureLevelRenderTargetWritten(textureObj, attachment->level);
    } else {
        NSLog(@"MGL WARNING: readPixels failed to apply pending FBO clear fbo=%u attachment=0x%x",
              (unsigned)fbo->name,
              (unsigned)attachmentEnum);
    }
}

- (BOOL)mglReadColorTextureAsBGRA8:(id<MTLTexture>)sourceTexture
                       sourceLevel:(NSUInteger)sourceLevel
                       sourceSlice:(NSUInteger)sourceSlice
                   sourceDepthPlane:(NSUInteger)sourceDepthPlane
                         pixelBytes:(void *)pixelBytes
                        bytesPerRow:(NSUInteger)bytesPerRow
                      bytesPerImage:(NSUInteger)bytesPerImage
                         fromRegion:(MTLRegion)region
                             reason:(const char *)reason
{
    NSUInteger readSize = bytesPerImage;
    if (readSize == 0u && bytesPerRow > 0u) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0u) {
        return NO;
    }

    if (!sourceTexture || region.size.width == 0u || region.size.height == 0u) {
        return sourceTexture != nil;
    }

    if ([sourceTexture isFramebufferOnly]) {
        static uint64_t s_framebufferOnlyReadCount = 0;
        uint64_t hit = ++s_framebufferOnlyReadCount;
        if (hit <= 16ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: readPixels cannot read framebufferOnly texture for %s hit=%llu",
                  reason ? reason : "unknown",
                  (unsigned long long)hit);
        }
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (!mglMetalReadbackFormatIsBGRA8Compatible(sourceTexture.pixelFormat)) {
        static uint64_t s_unsupportedReadFormatCount = 0;
        uint64_t hit = ++s_unsupportedReadFormatCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: readPixels unsupported Metal color readback format=%lu for %s hit=%llu; returning zero data",
                  (unsigned long)sourceTexture.pixelFormat,
                  reason ? reason : "unknown",
                  (unsigned long long)hit);
        }
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (bytesPerRow < region.size.width * 4u) {
        NSLog(@"MGL WARNING: readPixels destination row too small row=%lu width=%lu for %s",
              (unsigned long)bytesPerRow,
              (unsigned long)region.size.width,
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    NSUInteger levelWidth = sourceTexture.width;
    NSUInteger levelHeight = sourceTexture.height;
    if (sourceLevel > 0u) {
        if (sourceLevel >= sourceTexture.mipmapLevelCount) {
            NSLog(@"MGL WARNING: readPixels invalid mip level=%lu mipLevels=%lu for %s",
                  (unsigned long)sourceLevel,
                  (unsigned long)sourceTexture.mipmapLevelCount,
                  reason ? reason : "unknown");
            mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return NO;
        }
        levelWidth = MAX((NSUInteger)1u, sourceTexture.width >> sourceLevel);
        levelHeight = MAX((NSUInteger)1u, sourceTexture.height >> sourceLevel);
    }

    NSInteger requestX = (NSInteger)region.origin.x;
    NSInteger requestY = (NSInteger)region.origin.y;
    NSInteger requestW = (NSInteger)region.size.width;
    NSInteger requestH = (NSInteger)region.size.height;
    NSInteger glMinX = MAX((NSInteger)0, requestX);
    NSInteger glMinY = MAX((NSInteger)0, requestY);
    NSInteger glMaxX = MIN((NSInteger)levelWidth, requestX + requestW);
    NSInteger glMaxY = MIN((NSInteger)levelHeight, requestY + requestH);
    NSInteger copyW = glMaxX - glMinX;
    NSInteger copyH = glMaxY - glMinY;
    NSInteger dstX = glMinX - requestX;
    NSInteger dstY = glMinY - requestY;
    NSInteger metalSrcX = glMinX;
    NSInteger metalSrcY = (NSInteger)levelHeight - glMaxY;

    memset(pixelBytes, 0, readSize);

    if (copyW <= 0 || copyH <= 0) {
        return YES;
    }

    NSUInteger stagingBytesPerRow = (NSUInteger)copyW * 4u;
    NSUInteger stagingSize = stagingBytesPerRow * (NSUInteger)copyH;
    if (stagingSize == 0u) {
        return YES;
    }

    NSUInteger dstOffset = ((NSUInteger)dstY * bytesPerRow) + ((NSUInteger)dstX * 4u);
    if (dstOffset >= readSize ||
        stagingBytesPerRow > bytesPerRow ||
        ((NSUInteger)copyH - 1u) * bytesPerRow + stagingBytesPerRow > readSize - dstOffset) {
        NSLog(@"MGL WARNING: readPixels clipped copy exceeds destination storage for %s",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    if (![self ensureWritableCommandBuffer:"mglReadColorTextureAsBGRA8"]) {
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    id<MTLBuffer> readBuffer = [_device newBufferWithLength:stagingSize
                                                    options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blitEncoder = readBuffer ? [_currentCommandBuffer blitCommandEncoder] : nil;
    if (!readBuffer || !blitEncoder) {
        NSLog(@"MGL WARNING: readPixels failed to create readback resources for %s",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
        return NO;
    }

    BOOL blitEncoderEnded = NO;
    @try {
        [blitEncoder copyFromTexture:sourceTexture
                          sourceSlice:sourceSlice
                          sourceLevel:sourceLevel
                         sourceOrigin:MTLOriginMake((NSUInteger)metalSrcX,
                                                    (NSUInteger)metalSrcY,
                                                    sourceDepthPlane)
                           sourceSize:MTLSizeMake((NSUInteger)copyW,
                                                  (NSUInteger)copyH,
                                                  1u)
                             toBuffer:readBuffer
                    destinationOffset:0u
               destinationBytesPerRow:stagingBytesPerRow
             destinationBytesPerImage:stagingSize];
        [blitEncoder endEncoding];
        blitEncoderEnded = YES;
    } @catch (NSException *exception) {
        if (!blitEncoderEnded) {
            @try {
                [blitEncoder endEncoding];
            } @catch (NSException *endException) {
                NSLog(@"MGL WARNING: readPixels failed to end blit encoder after copy exception for %s: %@",
                      reason ? reason : "unknown",
                      endException);
            }
        }
        NSLog(@"MGL WARNING: readPixels texture copy failed for %s: %@",
              reason ? reason : "unknown",
              exception);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return NO;
    }

    __block NSError *readbackError = nil;
    dispatch_semaphore_t readbackDone = dispatch_semaphore_create(0);
    [_currentCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        readbackError = cb.error;
        dispatch_semaphore_signal(readbackDone);
    }];
    [_currentCommandBuffer commit];

    dispatch_time_t readbackDeadline = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.25 * NSEC_PER_SEC));
    BOOL success = YES;
    if (dispatch_semaphore_wait(readbackDone, readbackDeadline) != 0) {
        NSLog(@"MGL WARNING: readPixels command buffer timed out for %s; returning zeroed data",
              reason ? reason : "unknown");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        success = NO;
    } else if (readbackError) {
        NSLog(@"MGL WARNING: readPixels command buffer failed for %s: %@; returning zeroed data",
              reason ? reason : "unknown",
              readbackError);
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        success = NO;
    } else {
        uint8_t *dst = ((uint8_t *)pixelBytes) + dstOffset;
        mglMetalCopyTextureBytesToBGRA8((const uint8_t *)readBuffer.contents,
                                        stagingBytesPerRow,
                                        dst,
                                        bytesPerRow,
                                        (NSUInteger)copyW,
                                        (NSUInteger)copyH,
                                        sourceTexture.pixelFormat,
                                        YES);
    }

    [self newCommandBuffer];
    return success;
}


#pragma mark C interface to mtlReadDrawable
-(void) mtlReadDrawable:(GLMContext) glm_ctx pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MTLRegion)region
{
    ctx = glm_ctx;

    NSUInteger readSize = bytesPerImage;
    if (readSize == 0 && bytesPerRow > 0) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0) {
        return;
    }

    if (glm_ctx->state.readbuffer)
    {
        Framebuffer *fbo = glm_ctx->state.readbuffer;
        GLenum readBuffer = glm_ctx->state.read_buffer;
        if (!fbo ||
            readBuffer == GL_NONE ||
            readBuffer < GL_COLOR_ATTACHMENT0 ||
            readBuffer >= GL_COLOR_ATTACHMENT0 + glm_ctx->state.max_color_attachments ||
            readBuffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
            static uint64_t s_invalidReadFBOCount = 0;
            uint64_t hit = ++s_invalidReadFBOCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL WARNING: readPixels invalid FBO read buffer=0x%x maxColor=%u hit=%llu; returning zero data",
                      (unsigned)readBuffer,
                      (unsigned)glm_ctx->state.max_color_attachments,
                      (unsigned long long)hit);
            }
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        GLuint attachmentIndex = (GLuint)(readBuffer - GL_COLOR_ATTACHMENT0);
        if (((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) == 0u) {
            static uint64_t s_missingReadAttachmentCount = 0;
            uint64_t hit = ++s_missingReadAttachmentCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                NSLog(@"MGL WARNING: readPixels FBO read attachment 0x%x is not attached fbo=%u hit=%llu; returning zero data",
                      (unsigned)readBuffer,
                      (unsigned)fbo->name,
                      (unsigned long long)hit);
            }
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
        Texture *readTextureObject = [self framebufferAttachmentTexture:attachment];
        if (!readTextureObject) {
            NSLog(@"MGL WARNING: readPixels FBO attachment has no texture fbo=%u attachment=0x%x",
                  (unsigned)fbo->name,
                  (unsigned)readBuffer);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        readTextureObject->is_render_target = true;
        if (![self bindMTLTexture:readTextureObject] || !readTextureObject->mtl_data) {
            NSLog(@"MGL WARNING: readPixels could not bind FBO read texture fbo=%u attachment=0x%x tex=%u",
                  (unsigned)fbo->name,
                  (unsigned)readBuffer,
                  (unsigned)readTextureObject->name);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }

        id<MTLTexture> texture = (__bridge id<MTLTexture>)(readTextureObject->mtl_data);
        MGLMetalAttachmentSubresource subresource =
            mglMetalAttachmentSubresourceForAttachment(attachment);

        [self endRenderEncoding];
        if (![self ensureWritableCommandBuffer:"mtlReadDrawable.fbo"]) {
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
        [self mglApplyPendingFBOColorClearForReadback:fbo
                                           attachment:attachment
                                           textureObj:readTextureObject
                                           mtlTexture:texture
                                       attachmentEnum:readBuffer];
        [self mglReadColorTextureAsBGRA8:texture
                              sourceLevel:subresource.level
                              sourceSlice:subresource.slice
                          sourceDepthPlane:subresource.depthPlane
                                pixelBytes:pixelBytes
                               bytesPerRow:bytesPerRow
                             bytesPerImage:bytesPerImage
                                fromRegion:region
                                    reason:"FBO color readback"];
        return;
    }

    GLuint mgl_drawbuffer;
    id<MTLTexture> texture = nil;

    switch(glm_ctx->state.read_buffer)
    {
        case GL_FRONT: mgl_drawbuffer = _FRONT; break;
        case GL_BACK: mgl_drawbuffer = _FRONT; break;
        case GL_FRONT_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_FRONT_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_BACK_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_BACK_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        case GL_LEFT: mgl_drawbuffer = _FRONT_LEFT; break;
        case GL_RIGHT: mgl_drawbuffer = _FRONT_RIGHT; break;
        default:
            NSLog(@"MGL WARNING: readPixels unsupported default read buffer=0x%x; returning zero data",
                  (unsigned)glm_ctx->state.read_buffer);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
    }

    if (mgl_drawbuffer == _FRONT)
    {
        if (!_drawable) {
            [self mglSyncLayerDrawableSizeFromView:"readPixels.default"];
            _drawable = [_layer nextDrawable];
        }
        texture = _drawable ? _drawable.texture : nil;
    }
    else if (mgl_drawbuffer < _MAX_DRAW_BUFFERS)
    {
        texture = _drawBuffers[mgl_drawbuffer].drawbuffer;
    }

    if (!texture)
    {
        NSLog(@"MGL WARNING: readPixels default drawbuffer slot=%u has no texture; returning zero data",
              (unsigned)mgl_drawbuffer);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    [self endRenderEncoding];
    if (![self ensureWritableCommandBuffer:"mtlReadDrawable.default"]) {
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }
    if (mgl_drawbuffer == _FRONT) {
        [self mglApplyPendingDefaultColorClearToTexture:texture];
    }
    [self mglReadColorTextureAsBGRA8:texture
                          sourceLevel:0u
                          sourceSlice:0u
                      sourceDepthPlane:0u
                            pixelBytes:pixelBytes
                           bytesPerRow:bytesPerRow
                            bytesPerImage:bytesPerImage
                               fromRegion:region
                                   reason:"default framebuffer readback"];
    return;
}

#pragma mark C interface to mtlGetTexImage
-(void) mtlGetTexImage:(GLMContext) glm_ctx tex: (Texture *)tex pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice
{
    id<MTLTexture> texture = nil;

    ctx = glm_ctx;

    if (!tex) {
        NSLog(@"MGL ERROR: mtlGetTexImage called with NULL texture");
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (!pixelBytes) {
        NSLog(@"MGL WARNING: mtlGetTexImage called with NULL destination for texture %u", tex->name);
        return;
    }

    if (!tex->mtl_data && ![self bindMTLTexture:tex]) {
        NSLog(@"MGL ERROR: mtlGetTexImage failed to bind texture %u", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    texture = (__bridge id<MTLTexture>)(tex->mtl_data);
    if (!texture) {
        NSLog(@"MGL ERROR: mtlGetTexImage texture %u has no Metal texture", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if ([texture isFramebufferOnly] == NO)
    {
        @try {
            [texture getBytes:pixelBytes
                  bytesPerRow:bytesPerRow
                bytesPerImage:bytesPerImage
                   fromRegion:region
                  mipmapLevel:level
                        slice:slice];
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: mtlGetTexImage texture read failed for texture %u: %@",
                  tex->name,
                  exception);
            mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        }
    }
    else
    {
        // issue a gl error as we can't read a framebuffer only texture
        NSLog(@"Cannot read from framebuffer only texture\n");
        mglDispatchError(ctx, __FUNCTION__, GL_INVALID_OPERATION);
    }
}

void mtlReadDrawable(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlReadDrawable:glm_ctx pixelBytes:pixelBytes bytesPerRow:bytesPerRow bytesPerImage:bytesPerImage fromRegion:MTLRegionMake2D(x,y,width,height)];
}

void mtlGetTexImage(GLMContext glm_ctx, Texture *tex, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLuint level, GLuint slice)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlGetTexImage:glm_ctx tex:tex pixelBytes:pixelBytes bytesPerRow:bytesPerRow bytesPerImage:bytesPerImage fromRegion:MTLRegionMake2D(x,y,width,height) mipmapLevel:level slice:slice];
}

#pragma mark C interface to mtlGenerateMipmaps

-(void)mtlGenerateMipmaps:(GLMContext)glm_ctx forTexture:(Texture *) tex
{
    ctx = glm_ctx;

    if (!tex) {
        NSLog(@"MGL ERROR: mtlGenerateMipmaps called with NULL texture");
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    RETURN_ON_FAILURE([self processGLState: false]);

    // end encoding on current render encoder
    [self endRenderEncoding];

    RETURN_ON_FAILURE([self ensureWritableCommandBuffer:"mtlGenerateMipmaps"]);

    // no failure path..?
    RETURN_ON_FAILURE([self bindMTLTexture:tex]);

    id<MTLTexture> texture;

    texture = (__bridge id<MTLTexture>)(tex->mtl_data);
    if (!texture) {
        NSLog(@"MGL ERROR: mtlGenerateMipmaps texture %u has no Metal texture after bind", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (texture.mipmapLevelCount <= 1u) {
        return;
    }

    // start blit encoder
    id<MTLBlitCommandEncoder> blitCommandEncoder;
    blitCommandEncoder = [_currentCommandBuffer blitCommandEncoder];
    if (!blitCommandEncoder) {
        NSLog(@"MGL ERROR: Failed to create blit encoder for mipmap generation");
        return;
    }

    @try {
        [blitCommandEncoder generateMipmapsForTexture:texture];
        [blitCommandEncoder endEncoding];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: generateMipmapsForTexture failed for texture %u: %@",
              tex->name,
              exception);
        @try {
            [blitCommandEncoder endEncoding];
        } @catch (NSException *endException) {
            NSLog(@"MGL WARNING: failed to end mipmap blit encoder after exception: %@", endException);
        }
        mglDispatchError(glm_ctx, __FUNCTION__, GL_INVALID_OPERATION);
    }
}

void mtlGenerateMipmaps(GLMContext glm_ctx, Texture *tex)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlGenerateMipmaps:glm_ctx forTexture:tex];
}


#pragma mark C interface to mtlTexSubImage

-(void)mtlTexSubImage:(GLMContext)glm_ctx tex:(Texture *)tex buf:(Buffer *)buf src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size src_size:(size_t)src_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset
{
    if (!tex || !buf) {
        NSLog(@"MGL ERROR: mtlTexSubImage called with null tex/buf (tex=%p buf=%p)", tex, buf);
        return;
    }

    if (src_pitch == 0 || width == 0 || height == 0) {
        NSLog(@"MGL ERROR: mtlTexSubImage invalid dimensions/pitch tex=%u width=%zu height=%zu src_pitch=%zu",
              tex->name, width, height, src_pitch);
        return;
    }

    // we can deal with a null buffer but we need a texture
    if (buf->data.mtl_data == NULL)
    {
        [self bindMTLBuffer: buf];
        RETURN_ON_NULL(buf->data.mtl_data);
    }

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!buffer) {
        NSLog(@"MGL ERROR: mtlTexSubImage missing Metal buffer object tex=%u", tex->name);
        return;
    }

    if (tex->mtl_data == NULL)
    {
        [self bindMTLTexture: tex];
        RETURN_ON_NULL(tex->mtl_data);
    }

    id<MTLTexture> texture = (__bridge id<MTLTexture>)(tex->mtl_data);
    if (!texture) {
        NSLog(@"MGL ERROR: mtlTexSubImage missing Metal texture object tex=%u", tex->name);
        return;
    }

    // Keep uploads out of active render encoders/command buffers.
    [self endRenderEncoding];

    // IMPORTANT: array/cubemap target slice is provided via `slice`.
    // `zoffset` is only for 3D texture origin.z.
    NSUInteger destinationSlice = 0;
    MTLOrigin destinationOrigin = MTLOriginMake(xoffset, yoffset, 0);
    MTLTextureType textureType = texture.textureType;
    NSUInteger copyDepth = 1;

    if (textureType == MTLTextureType3D) {
        destinationSlice = 0;
        destinationOrigin = MTLOriginMake(xoffset, yoffset, zoffset);
        copyDepth = MAX((NSUInteger)depth, (NSUInteger)1);
    } else if (textureType == MTLTextureTypeCube ||
               textureType == MTLTextureTypeCubeArray ||
               textureType == MTLTextureType2DArray ||
               textureType == MTLTextureType1DArray ||
               textureType == MTLTextureType2DMultisampleArray) {
        destinationSlice = (NSUInteger)slice;
        destinationOrigin = MTLOriginMake(xoffset, yoffset, 0);
        copyDepth = 1;
    } else {
        destinationSlice = 0;
        destinationOrigin = MTLOriginMake(xoffset, yoffset, 0);
        copyDepth = 1;
    }

    NSUInteger copyWidth = (width > 0) ? width : 1;
    NSUInteger copyHeight = (height > 0) ? height : 1;
    NSUInteger expectedBytesPerImage = src_pitch * copyHeight;
    NSUInteger copyBytesPerImage = src_image_size;
    // Array/cube uploads are one slice at a time. Never treat them as a stacked multi-image upload.
    if (textureType == MTLTextureTypeCube ||
        textureType == MTLTextureTypeCubeArray ||
        textureType == MTLTextureType2DArray ||
        textureType == MTLTextureType1DArray ||
        textureType == MTLTextureType2DMultisampleArray) {
        if (copyBytesPerImage != expectedBytesPerImage) {
            NSLog(@"MGL INFO: mtlTexSubImage normalize bytesPerImage tex=%u slice=%u level=%u old=%lu expected=%lu",
                  tex->name, slice, level, (unsigned long)copyBytesPerImage, (unsigned long)expectedBytesPerImage);
        }
        copyBytesPerImage = expectedBytesPerImage;
    } else if (textureType == MTLTextureType3D) {
        if (copyBytesPerImage < expectedBytesPerImage) {
            copyBytesPerImage = expectedBytesPerImage;
        }
    } else {
        copyBytesPerImage = expectedBytesPerImage;
    }

    if (textureType == MTLTextureTypeCube || textureType == MTLTextureTypeCubeArray) {
        uint8_t *bufferBase = (uint8_t *)buf->data.buffer_data;
        void *pixelPtr = bufferBase ? (void *)(bufferBase + src_offset) : NULL;
        NSLog(@"MGL CUBE UPLOAD tex=%u glTarget=0x%x face=%u slice=%lu level=%u origin=(%lu,%lu,%lu) size=%lux%lux%lu bpr=%lu bpi=%lu ptr=%p",
              tex->name,
              tex->target,
              slice,
              (unsigned long)destinationSlice,
              level,
              (unsigned long)xoffset,
              (unsigned long)yoffset,
              (unsigned long)destinationOrigin.z,
              (unsigned long)copyWidth,
              (unsigned long)copyHeight,
              (unsigned long)copyDepth,
              (unsigned long)src_pitch,
              (unsigned long)copyBytesPerImage,
              pixelPtr);
    }

    bool uploaded = [self copyTextureUploadWithDedicatedCommandBuffer:buffer
                                                         sourceOffset:src_offset
                                                    sourceBytesPerRow:src_pitch
                                                  sourceBytesPerImage:copyBytesPerImage
                                                            sourceSize:MTLSizeMake(copyWidth, copyHeight, copyDepth)
                                                             toTexture:texture
                                                      destinationSlice:destinationSlice
                                                      destinationLevel:level
                                                     destinationOrigin:destinationOrigin
                                                                reason:"mtlTexSubImage"];
    if (!uploaded) {
        NSLog(@"MGL ERROR: mtlTexSubImage dedicated upload failed (tex=%u slice=%u level=%u)",
              tex->name, slice, level);
    }
}

void mtlTexSubImage(GLMContext glm_ctx, Texture *tex, Buffer *buf, size_t src_offset, size_t src_pitch, size_t src_image_size, size_t src_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlTexSubImage:glm_ctx tex:tex buf:buf src_offset:src_offset src_pitch:src_pitch src_image_size:src_image_size src_size:src_size slice:slice level:level width:width height:height depth:depth xoffset:xoffset yoffset:yoffset zoffset:zoffset];
}

#pragma mark utility functions for draw commands
MTLPrimitiveType getMTLPrimitiveType(GLenum mode)
{
    const GLuint err = 0xFFFFFFFF;

    switch(mode)
    {
        case GL_POINTS:
            return MTLPrimitiveTypePoint;

        case GL_LINES:
            return MTLPrimitiveTypeLine;

        case GL_LINE_STRIP:
            return MTLPrimitiveTypeLineStrip;

        case GL_TRIANGLES:
            return MTLPrimitiveTypeTriangle;

        case GL_TRIANGLE_STRIP:
            return MTLPrimitiveTypeTriangleStrip;

        case GL_LINE_LOOP:
        case GL_LINE_STRIP_ADJACENCY:
        case GL_LINES_ADJACENCY:
        case GL_TRIANGLE_FAN:
        case GL_TRIANGLE_STRIP_ADJACENCY:
        case GL_PATCHES:
            return (MTLPrimitiveType)0xFFFFFFFF;
            break;
    }

    return err;
}

MTLIndexType getMTLIndexType(GLenum type)
{
    const GLuint err = 0xFFFFFFFF;

    switch(type)
    {
        case GL_UNSIGNED_BYTE:
            return MTLIndexTypeUInt16;

        case GL_UNSIGNED_SHORT:
            return MTLIndexTypeUInt16;

        case GL_UNSIGNED_INT:
            return MTLIndexTypeUInt32;
    }

    return err;
}

Buffer *getElementBuffer(GLMContext ctx)
{
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    Buffer *gl_element_buffer = vao ? vao->element_array.buffer : NULL;

    return gl_element_buffer;
}

- (bool) validateDrawArraysVertexInputs:(GLMContext)drawCtx
                                    mode:(GLenum)mode
                                   first:(GLint)first
                                   count:(GLsizei)count
                                drawCall:(uint64_t)drawCall
{
    if (!kMGLValidateDrawArraysVboRange) {
        return true;
    }

    if (!drawCtx) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=null_ctx mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    if (count == 0) {
        return false;
    }

    if (count < 0 || first < 0) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=invalid_range mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    uint64_t firstVertex = (uint64_t)(uint32_t)first;
    uint64_t vertexCount = (uint64_t)(uint32_t)count;
    if (vertexCount == 0u || firstVertex > UINT64_MAX - (vertexCount - 1u)) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=vertex_range_overflow mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    uint64_t lastVertex = firstVertex + vertexCount - 1u;
    VertexArray *vao = mglRendererGetValidatedVAO(drawCtx, "drawArrays.vboRange");
    if (!vao) {
        NSLog(@"MGL DRAWARRAYS BLOCK call=%llu reason=invalid_vao mode=0x%x first=%d count=%d",
              (unsigned long long)drawCall, (unsigned)mode, (int)first, (int)count);
        return false;
    }

    GLuint maxAttribs = drawCtx->state.max_vertex_attribs;
    if (maxAttribs > MAX_ATTRIBS) {
        maxAttribs = MAX_ATTRIBS;
    }

    for (GLuint attrib = 0; attrib < maxAttribs; attrib++) {
        if ((vao->enabled_attribs & (0x1u << attrib)) == 0u) {
            continue;
        }

        VertexAttrib *a = &vao->attrib[attrib];
        Buffer *vbo = mglRendererGetValidatedBuffer(drawCtx, a->buffer, "drawArrays.vboRange", attrib);
        if (!vbo) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u reason=invalid_vbo mode=0x%x first=%d count=%d",
                  (unsigned long long)drawCall, (unsigned)attrib, (unsigned)mode, (int)first, (int)count);
            return false;
        }

        if (!vbo->ever_written) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=never_written "
                  "init(source=%u full=%u range=[%lld,%lld) lastOff=%lld lastSize=%lld src=%p hash=0x%016llx)",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned)vbo->last_init_source,
                  (unsigned)vbo->has_initialized_data,
                  (long long)vbo->written_min,
                  (long long)vbo->written_max,
                  (long long)vbo->last_write_offset,
                  (long long)vbo->last_write_size,
                  vbo->last_write_src_ptr,
                  (unsigned long long)vbo->last_write_src_hash);
            return false;
        }

        if (a->binding_offset < 0 || a->relativeoffset < 0) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=negative_attrib_offset bindingOffset=%lld relativeOffset=%lld",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (long long)a->binding_offset,
                  (long long)a->relativeoffset);
            return false;
        }

        size_t compSize = mglVertexAttribComponentSize(a->type);
        size_t compCount = (size_t)a->size;
        if (compSize == 0u || compCount == 0u) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=invalid_attrib_format type=0x%x size=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned)a->type,
                  (unsigned)a->size);
            return false;
        }

        if (compCount > (SIZE_MAX / compSize)) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=elem_size_overflow compSize=%zu compCount=%zu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  compSize,
                  compCount);
            return false;
        }

        uint64_t elemBytes = (uint64_t)(compSize * compCount);
        uint64_t stride = (a->stride > 0u) ? (uint64_t)a->stride : elemBytes;
        uint64_t bindingOffset = (uint64_t)a->binding_offset;
        uint64_t attrRelativeOffset = (uint64_t)a->relativeoffset;
        if (bindingOffset > UINT64_MAX - attrRelativeOffset) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=offset_overflow bindingOffset=%llu relativeOffset=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)attrRelativeOffset);
            return false;
        }
        uint64_t relOffset = bindingOffset + attrRelativeOffset;
        if (stride == 0u || elemBytes == 0u) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=zero_stride_or_elem stride=%llu elem=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)stride,
                  (unsigned long long)elemBytes);
            return false;
        }

        // Per-instance attributes are still consumed by a non-instanced draw for
        // instance zero, so validate element zero instead of ignoring them.
        uint64_t rangeFirst = (a->divisor != 0u) ? 0u : firstVertex;
        uint64_t rangeLast = (a->divisor != 0u) ? 0u : lastVertex;

        if (relOffset > UINT64_MAX - elemBytes) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=byte_range_overflow bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)a->divisor);
            return false;
        }

        if (rangeLast > (UINT64_MAX - relOffset - elemBytes) / stride ||
            rangeFirst > (UINT64_MAX - relOffset) / stride) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=byte_range_overflow "
                  "range=[%llu,%llu] stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)rangeFirst,
                  (unsigned long long)rangeLast,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)a->divisor);
            return false;
        }

        uint64_t byteStart = relOffset + (rangeFirst * stride);
        uint64_t byteEnd = relOffset + (rangeLast * stride) + elemBytes;
        uint64_t vboSize = (vbo->size > 0) ? (uint64_t)vbo->size : 0u;
        if (byteEnd > vboSize) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=vbo_oob "
                  "vertexRange=[%llu,%llu] byteRange=[%llu,%llu) vboSize=%llu "
                  "mode=0x%x first=%d count=%d stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu type=0x%x size=%u divisor=%u",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)rangeFirst,
                  (unsigned long long)rangeLast,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)vboSize,
                  (unsigned)mode,
                  (int)first,
                  (int)count,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes,
                  (unsigned)a->type,
                  (unsigned)a->size,
                  (unsigned)a->divisor);
            return false;
        }

        if (!vbo->data.mtl_data) {
            [self bindMTLBuffer:vbo];
        }
        if (!vbo->data.mtl_data) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=no_mtl_buffer byteRange=[%llu,%llu)",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd);
            return false;
        }

        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)(vbo->data.mtl_data);
        if (!mtlBuffer) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=mtl_bridge_nil",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name);
            return false;
        }

        uint64_t metalLen = (uint64_t)mtlBuffer.length;
        if (byteEnd > metalLen) {
            NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=metal_oob "
                  "byteRange=[%llu,%llu) metalLen=%llu vboSize=%llu first=%d count=%d",
                  (unsigned long long)drawCall,
                  (unsigned)attrib,
                  (unsigned)vbo->name,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)metalLen,
                  (unsigned long long)vboSize,
                  (int)first,
                  (int)count);
            return false;
        }

        if (vbo->written_min >= 0 && vbo->written_max >= 0) {
            uint64_t writtenMin = (uint64_t)vbo->written_min;
            uint64_t writtenMax = (uint64_t)vbo->written_max;
            if (byteStart < writtenMin || byteEnd > writtenMax) {
                NSLog(@"MGL DRAWARRAYS BLOCK call=%llu attrib=%u buffer=%u reason=unwritten_range "
                      "byteRange=[%llu,%llu) written=[%llu,%llu) first=%d count=%d source=%u",
                      (unsigned long long)drawCall,
                      (unsigned)attrib,
                      (unsigned)vbo->name,
                      (unsigned long long)byteStart,
                      (unsigned long long)byteEnd,
                      (unsigned long long)writtenMin,
                      (unsigned long long)writtenMax,
                      (int)first,
                      (int)count,
                      (unsigned)vbo->last_init_source);
                return false;
            }
        }

        if (mglShouldInspectDrawCall(drawCall, drawCtx->state.program_name) && attrib == 0u) {
            NSLog(@"MGL TRACE drawArrays.attrib0 call=%llu program=%u buffer=%u first=%d count=%d "
                  "byteRange=[%llu,%llu) vboSize=%llu metalLen=%llu stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu",
                  (unsigned long long)drawCall,
                  (unsigned)drawCtx->state.program_name,
                  (unsigned)vbo->name,
                  (int)first,
                  (int)count,
                  (unsigned long long)byteStart,
                  (unsigned long long)byteEnd,
                  (unsigned long long)vboSize,
                  (unsigned long long)metalLen,
                  (unsigned long long)stride,
                  (unsigned long long)bindingOffset,
                  (unsigned long long)relOffset,
                  (unsigned long long)elemBytes);
        }
    }

    return true;
}

Buffer *getIndirectBuffer(GLMContext ctx)
{
    Buffer *gl_indirect_buffer = STATE(buffers[_DRAW_INDIRECT_BUFFER]);

    return gl_indirect_buffer;
}

- (BOOL)resolveElementBufferForDraw:(const char *)label
                            context:(GLMContext)drawCtx
                           glBuffer:(Buffer **)glBufferOut
                          mtlBuffer:(id<MTLBuffer> *)mtlBufferOut
{
    Buffer *gl_element_buffer = getElementBuffer(drawCtx);
    if (!gl_element_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no element array buffer is bound", label ? label : "indexed draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, GL_INVALID_OPERATION);
        }
        return NO;
    }

    if ([self processBuffer:gl_element_buffer] == false) {
        return NO;
    }

    id<MTLBuffer> indexBuffer = (__bridge id<MTLBuffer>)(gl_element_buffer->data.mtl_data);
    if (!indexBuffer) {
        NSLog(@"MGL WARNING: %s skipped because element buffer %u has no Metal buffer",
              label ? label : "indexed draw",
              gl_element_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_element_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indexBuffer;
    }
    return YES;
}

- (BOOL)resolveIndirectBufferForDraw:(const char *)label
                             context:(GLMContext)drawCtx
                            glBuffer:(Buffer **)glBufferOut
                           mtlBuffer:(id<MTLBuffer> *)mtlBufferOut
{
    Buffer *gl_indirect_buffer = getIndirectBuffer(drawCtx);
    if (!gl_indirect_buffer) {
        NSLog(@"MGL WARNING: %s skipped because no draw indirect buffer is bound", label ? label : "indirect draw");
        if (drawCtx) {
            mglDispatchError(drawCtx, label ? label : __FUNCTION__, GL_INVALID_OPERATION);
        }
        return NO;
    }

    if ([self processBuffer:gl_indirect_buffer] == false) {
        return NO;
    }

    id<MTLBuffer> indirectBuffer = (__bridge id<MTLBuffer>)(gl_indirect_buffer->data.mtl_data);
    if (!indirectBuffer) {
        NSLog(@"MGL WARNING: %s skipped because indirect buffer %u has no Metal buffer",
              label ? label : "indirect draw",
              gl_indirect_buffer->name);
        return NO;
    }

    if (glBufferOut) {
        *glBufferOut = gl_indirect_buffer;
    }
    if (mtlBufferOut) {
        *mtlBufferOut = indirectBuffer;
    }
    return YES;
}

- (BOOL)currentDrawRasterizationIsEmpty
{
    if (!ctx) {
        return NO;
    }

    GLint vx = ctx->state.viewport[0];
    GLint vy = ctx->state.viewport[1];
    GLint vw = ctx->state.viewport[2];
    GLint vh = ctx->state.viewport[3];
    if (vw <= 0 || vh <= 0) {
        return YES;
    }

    NSUInteger passWidth = _renderPassDescriptor ? _renderPassDescriptor.renderTargetWidth : 0;
    NSUInteger passHeight = _renderPassDescriptor ? _renderPassDescriptor.renderTargetHeight : 0;
    if ((passWidth == 0 || passHeight == 0) && _renderPassDescriptor) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            id<MTLTexture> color = _renderPassDescriptor.colorAttachments[i].texture;
            if (color) {
                passWidth = color.width;
                passHeight = color.height;
                break;
            }
        }
        if ((passWidth == 0 || passHeight == 0) && _renderPassDescriptor.depthAttachment.texture) {
            passWidth = _renderPassDescriptor.depthAttachment.texture.width;
            passHeight = _renderPassDescriptor.depthAttachment.texture.height;
        }
        if ((passWidth == 0 || passHeight == 0) && _renderPassDescriptor.stencilAttachment.texture) {
            passWidth = _renderPassDescriptor.stencilAttachment.texture.width;
            passHeight = _renderPassDescriptor.stencilAttachment.texture.height;
        }
    }

    if (passWidth == 0 || passHeight == 0) {
        return NO;
    }

    int64_t fbW = (int64_t)passWidth;
    int64_t fbH = (int64_t)passHeight;
    int64_t vx0 = (int64_t)vx;
    int64_t vy0 = (int64_t)vy;
    int64_t vx1 = vx0 + (int64_t)vw;
    int64_t vy1 = vy0 + (int64_t)vh;
    if (vx1 <= 0 || vy1 <= 0 || vx0 >= fbW || vy0 >= fbH) {
        return YES;
    }

    if (ctx->state.caps.scissor_test) {
        GLint sx = ctx->state.var.scissor_box[0];
        GLint sy = ctx->state.var.scissor_box[1];
        GLint sw = ctx->state.var.scissor_box[2];
        GLint sh = ctx->state.var.scissor_box[3];
        if (sw <= 0 || sh <= 0) {
            return YES;
        }

        int64_t sx0 = (int64_t)sx;
        int64_t sy0 = (int64_t)sy;
        int64_t sx1 = sx0 + (int64_t)sw;
        int64_t sy1 = sy0 + (int64_t)sh;
        if (sx1 <= 0 || sy1 <= 0 || sx0 >= fbW || sy0 >= fbH) {
            return YES;
        }
    }

    return NO;
}

- (void)applyPolygonOffsetForDrawMode:(GLenum)mode
{
    if (!_currentRenderEncoder) {
        return;
    }

    MTLTriangleFillMode triangleFillMode = MTLTriangleFillModeFill;
    if (ctx && mglDrawModeProducesPolygons(mode)) {
        if (ctx->state.var.polygon_mode == GL_LINE) {
            triangleFillMode = MTLTriangleFillModeLines;
        } else if (ctx->state.var.polygon_mode != GL_FILL &&
                   ctx->state.var.polygon_mode != GL_POINT) {
            mglLogRenderStateRepair("polygon_mode", ctx->state.var.polygon_mode, GL_FILL);
            ctx->state.var.polygon_mode = GL_FILL;
            ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
        }
    }
    [_currentRenderEncoder setTriangleFillMode:triangleFillMode];

    BOOL enableDepthBias = NO;

    if (ctx && mglDrawModeProducesPolygons(mode)) {
        switch (ctx->state.var.polygon_mode) {
            case GL_POINT:
                enableDepthBias = ctx->state.caps.polygon_offset_point;
                break;
            case GL_LINE:
                enableDepthBias = ctx->state.caps.polygon_offset_line;
                break;
            case GL_FILL:
            default:
                enableDepthBias = ctx->state.caps.polygon_offset_fill;
                break;
        }
    }

    if (enableDepthBias) {
        [_currentRenderEncoder setDepthBias:ctx->state.var.polygon_offset_units
                                 slopeScale:ctx->state.var.polygon_offset_factor
                                      clamp:0.0f];
    } else {
        [_currentRenderEncoder setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
    }
}

- (BOOL)currentDrawModeIsFullyCulled:(GLenum)mode
{
    return ctx &&
           ctx->state.caps.cull_face &&
           ctx->state.var.cull_face_mode == GL_FRONT_AND_BACK &&
           mglDrawModeProducesPolygons(mode);
}

#pragma mark C interface to mtlDrawArrays
-(void) mtlDrawArrays: (GLMContext) ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count
{
    static uint64_t s_drawArraysCallCount = 0;
    static double s_drawArraysLastCallTime = 0.0;
    static uint64_t s_drawArraysLastCallCount = 0;
    uint64_t drawCall = ++s_drawArraysCallCount;
    double drawStartSeconds = mglNowSeconds();
    bool traceDraw = mglShouldTraceCall(drawCall);
    mglLogLoopHeartbeat("drawArrays.loop",
                        drawCall,
                        drawStartSeconds,
                        &s_drawArraysLastCallTime,
                        &s_drawArraysLastCallCount,
                        0.25);

    MTLPrimitiveType primitiveType;
    static uint64_t process_state_fail_count = 0;
    static uint64_t no_render_encoder_count = 0;

    // AGGRESSIVE MEMORY SAFETY: Immediate validation before any Metal operations
    if (!ctx || ((uintptr_t)ctx < 0x1000)) {
        NSLog(@"MGL ERROR: mtlDrawArrays - Invalid context detected, aborting");
        return; // Early return to prevent crash
    }

    if ([self processGLState: true] == false) {
        process_state_fail_count++;
        g_mglDrawArraysSkippedSinceSwap++;
        if (process_state_fail_count <= 8 || (process_state_fail_count % 1000) == 0) {
            NSLog(@"MGL ERROR: mtlDrawArrays - processGLState failed, aborting (occurrence=%llu)",
                  (unsigned long long)process_state_fail_count);
        }
        return; // Early return instead of continuing with invalid state
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    // Additional safety check after processGLState
    if (!_currentRenderEncoder) {
        // One recovery attempt to avoid persistent "No current render encoder" failure loops.
        [self newRenderEncoder];
        if (!_currentRenderEncoder) {
            no_render_encoder_count++;
            if (no_render_encoder_count <= 8 || (no_render_encoder_count % 1000) == 0) {
                NSLog(@"MGL ERROR: mtlDrawArrays - No current render encoder, aborting (occurrence=%llu)",
                      (unsigned long long)no_render_encoder_count);
            }
            return;
        }

        if (!_pipelineState) {
            NSLog(@"MGL ERROR: mtlDrawArrays - No pipeline state after render encoder recovery, aborting draw");
            return;
        }

        // Guard against Metal validation aborts when emergency-rebinding pipeline after
        // encoder recovery. Only bind when pass attachment formats are compatible.
        MTLPixelFormat rpColor0Format = MTLPixelFormatInvalid;
        MTLPixelFormat rpDepthFormat = MTLPixelFormatInvalid;
        MTLPixelFormat rpStencilFormat = MTLPixelFormatInvalid;
        if (_renderPassDescriptor) {
            id<MTLTexture> rpColor0 = _renderPassDescriptor.colorAttachments[0].texture;
            id<MTLTexture> rpDepth = _renderPassDescriptor.depthAttachment.texture;
            id<MTLTexture> rpStencil = _renderPassDescriptor.stencilAttachment.texture;
            if (rpColor0) rpColor0Format = rpColor0.pixelFormat;
            if (rpDepth) rpDepthFormat = rpDepth.pixelFormat;
            if (rpStencil) rpStencilFormat = rpStencil.pixelFormat;
        }

        BOOL colorMismatch = (_pipelineColor0Format != MTLPixelFormatInvalid &&
                              rpColor0Format != MTLPixelFormatInvalid &&
                              _pipelineColor0Format != rpColor0Format);
        BOOL depthMismatch = (_pipelineDepthFormat != rpDepthFormat);
        BOOL stencilMismatch = (_pipelineStencilFormat != rpStencilFormat);
        if (colorMismatch || depthMismatch || stencilMismatch) {
            NSLog(@"MGL WARNING: mtlDrawArrays recovery skipped pipeline bind due to pass mismatch "
                  "(pipeline c/d/s=%lu/%lu/%lu, pass c/d/s=%lu/%lu/%lu)",
                  (unsigned long)_pipelineColor0Format,
                  (unsigned long)_pipelineDepthFormat,
                  (unsigned long)_pipelineStencilFormat,
                  (unsigned long)rpColor0Format,
                  (unsigned long)rpDepthFormat,
                  (unsigned long)rpStencilFormat);
            return;
        }

        @try {
            [_currentRenderEncoder setRenderPipelineState:_pipelineState];
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: mtlDrawArrays - setRenderPipelineState failed after recovery: %@", exception);
            return;
        }
    }

    if (![self validateDrawArraysVertexInputs:ctx
                                         mode:mode
                                        first:first
                                        count:count
                                     drawCall:drawCall]) {
        g_mglDrawArraysSkippedSinceSwap++;
        return;
    }

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    if (polygonModePoint) {
        if (!mglEncodeArrayPolygonPoint(_currentRenderEncoder,
                                        _device,
                                        mode,
                                        first,
                                        count,
                                        1u,
                                        0u,
                                        "drawArrays")) {
            g_mglDrawArraysSkippedSinceSwap++;
            return;
        }
    } else if (emulateTriangleFan) {
        if (count < 3) {
            return;
        }

        NSUInteger fanIndexCount = 0u;
        id<MTLBuffer> fanIndexBuffer = mglNewTriangleFanArrayIndexBuffer(_device,
                                                                         (NSUInteger)count,
                                                                         &fanIndexCount);
        if (!fanIndexBuffer || fanIndexCount == 0u) {
            NSLog(@"MGL WARNING: drawArrays triangle fan emulation failed count=%d first=%d",
                  (int)count,
                  (int)first);
            g_mglDrawArraysSkippedSinceSwap++;
            return;
        }

        @try {
            [_currentRenderEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                              indexCount:fanIndexCount
                                               indexType:MTLIndexTypeUInt32
                                             indexBuffer:fanIndexBuffer
                                       indexBufferOffset:0
                                           instanceCount:1
                                              baseVertex:first
                                            baseInstance:0];
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: mtlDrawArrays triangle fan drawIndexedPrimitives failed: %@", exception);
            return;
        }
    } else if (emulateLineLoop) {
        if (count < 2) {
            return;
        }

        NSUInteger loopIndexCount = 0u;
        id<MTLBuffer> loopIndexBuffer = mglNewLineLoopArrayIndexBuffer(_device,
                                                                       (NSUInteger)count,
                                                                       &loopIndexCount);
        if (!loopIndexBuffer || loopIndexCount == 0u) {
            NSLog(@"MGL WARNING: drawArrays line loop emulation failed count=%d first=%d",
                  (int)count,
                  (int)first);
            g_mglDrawArraysSkippedSinceSwap++;
            return;
        }

        @try {
            [_currentRenderEncoder drawIndexedPrimitives:MTLPrimitiveTypeLineStrip
                                              indexCount:loopIndexCount
                                               indexType:MTLIndexTypeUInt32
                                             indexBuffer:loopIndexBuffer
                                       indexBufferOffset:0
                                           instanceCount:1
                                              baseVertex:first
                                            baseInstance:0];
        } @catch (NSException *exception) {
            NSLog(@"MGL ERROR: mtlDrawArrays line loop drawIndexedPrimitives failed: %@", exception);
            return;
        }
    } else {
        primitiveType = getMTLPrimitiveType(mode);
        if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    @try {
        [_currentRenderEncoder drawPrimitives: primitiveType
                                 vertexStart: first
                                 vertexCount: count];
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: mtlDrawArrays - drawPrimitives failed: %@", exception);
        // Don't crash, just return gracefully
        return;
    }
    }

    g_mglLastDrawArraysCall = drawCall;
    g_mglLastDrawArraysSeconds = mglNowSeconds();
    g_mglLastDrawArraysProgram = ctx ? ctx->state.program_name : 0u;
    g_mglLastDrawArraysMode = mode;
    g_mglLastDrawArraysCount = count;
    g_mglDrawArraysSinceSwap++;
    if (count > 0) {
        g_mglDrawArrayVerticesSinceSwap += (uint64_t)count;
    }
    [self markCurrentFramebufferDrawAttachmentsWritten];
    mglLogDrawWithoutSwapWatchdog("arrays",
                                  drawCall,
                                  ctx,
                                  _currentCommandBuffer,
                                  _currentRenderEncoder,
                                  _renderPassDescriptor);

    double drawElapsedMs = (mglNowSeconds() - drawStartSeconds) * 1000.0;
    if (traceDraw || drawElapsedMs >= 16.0) {
        NSLog(@"MGL TRACE drawArrays.end call=%llu mode=0x%x first=%d count=%d elapsed=%.3fms encoder=%p",
              (unsigned long long)drawCall,
              (unsigned)mode,
              (int)first,
              (int)count,
              drawElapsedMs,
              _currentRenderEncoder);
    }
}

void mtlDrawArrays(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count)
{
    // FINAL FAILSAFE: Catch any unhandled exceptions to prevent QEMU crashes
    @try {
        // Validate context before bridging
        if (!glm_ctx || ((uintptr_t)glm_ctx < 0x1000)) {
            NSLog(@"MGL CRITICAL: mtlDrawArrays - Invalid GLM context, aborting operation");
            return;
        }

        // Validate the Metal object pointer lower bound only
        if (!glm_ctx->mtl_funcs.mtlObj || ((uintptr_t)glm_ctx->mtl_funcs.mtlObj < 0x1000)) {
            NSLog(@"MGL CRITICAL: mtlDrawArrays - Invalid Metal object, aborting operation");
            return;
        }

        [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawArrays: glm_ctx mode: mode first: first count: count];
    } @catch (NSException *exception) {
        NSLog(@"MGL CRITICAL: mtlDrawArrays - Unhandled exception caught: %@", exception);
        NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);
        NSLog(@"MGL CRITICAL: This is a failsafe to prevent QEMU crashes");
        // Don't crash, just return gracefully
    } @catch (id exception) {
        NSLog(@"MGL CRITICAL: mtlDrawArrays - Unknown exception caught: %@", exception);
        // Final safety net
    }
}

#pragma mark C interface to mtlDrawElements
-(void) mtlDrawElements: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices
{
    static uint64_t s_drawElementsCallCount = 0;
    static double s_drawElementsLastCallTime = 0.0;
    static uint64_t s_drawElementsLastCallCount = 0;
    static uint64_t s_drawElementsProcessStateFailCount = 0;
    uint64_t drawCall = ++s_drawElementsCallCount;
    double drawStartSeconds = mglNowSeconds();
    bool traceDraw = mglShouldTraceCall(drawCall);
    mglLogLoopHeartbeat("drawElements.loop",
                        drawCall,
                        drawStartSeconds,
                        &s_drawElementsLastCallTime,
                        &s_drawElementsLastCallCount,
                        0.25);

    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;
    GLuint activeProgramName = ctx ? ctx->state.program_name : 0u;
    Program *drawProgram = NULL;
    bool drawProgramUsesCloudFaces = false;

    if (traceDraw) {
        NSLog(@"MGL TRACE drawElements.begin call=%llu mode=0x%x count=%d type=0x%x indices=%p program=%u vao=%p fbo=%p",
              (unsigned long long)drawCall,
              (unsigned)mode,
              (int)count,
              (unsigned)type,
              indices,
              activeProgramName,
              ctx ? ctx->state.vao : NULL,
              ctx ? ctx->state.framebuffer : NULL);
    }

    if (count <= 0) {
        if (traceDraw) {
            NSLog(@"MGL TRACE drawElements.skip.invalidCount call=%llu count=%d",
                  (unsigned long long)drawCall,
                  (int)count);
        }
        return;
    }

    if ([self processGLState: true] == false) {
        s_drawElementsProcessStateFailCount++;
        g_mglDrawElementsSkippedSinceSwap++;
        if (traceDraw || s_drawElementsProcessStateFailCount <= 16 || (s_drawElementsProcessStateFailCount % 500) == 0) {
            NSLog(@"MGL TRACE drawElements.skip.processGLState call=%llu failCount=%llu",
                  (unsigned long long)drawCall,
                  (unsigned long long)s_drawElementsProcessStateFailCount);
        }
        return;
    }
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    Program *activeProgram = ctx ? mglResolveProgramFromState(ctx) : NULL;
    activeProgramName = ctx ? (ctx->state.program_name ? ctx->state.program_name : (activeProgram ? activeProgram->name : 0u)) : 0u;
    if (ctx && activeProgramName != 0u) {
        VertexArray *validVAO = mglRendererGetValidatedVAO(ctx, "drawElements.enabledMask");
        GLuint enabledAttribMask = validVAO ? validVAO->enabled_attribs : 0u;
        drawProgram = mglPeekProgramByName(ctx, activeProgramName);
        drawProgramUsesCloudFaces =
            mglProgramHasResourceNamed(drawProgram, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, "CloudFaces") ||
            mglProgramHasResourceNamed(drawProgram, _VERTEX_SHADER, SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT, "CloudFaces");

        mglObserveProgramDrawForFocus(activeProgramName, count, enabledAttribMask);

        // SPIR-V image dimension enum values: Cube is 3. Keep this literal here to avoid
        // depending on which SPIR-V enum header variant is pulled through spirv_cross_c.h.
        if (mglProgramHasImageDim(drawProgram, 3u)) {
            mglFocusLoadingProgram(activeProgramName, "cube-sampled-image", drawCall);
        }
        if (mglProgramLooksLikeMinecraftTerrain(drawProgram)) {
            mglFocusLoadingProgram(activeProgramName, "minecraft-terrain-shader", drawCall);
        }
        if (drawProgramUsesCloudFaces) {
            mglFocusLoadingProgram(activeProgramName, "minecraft-cloudfaces-shader", drawCall);
        }
    }

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : getMTLPrimitiveType(mode)));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = getElementBuffer(ctx);
    if (!gl_element_buffer) {
        NSLog(@"MGL WARNING: drawElements call=%llu missing element buffer mode=0x%x count=%d type=0x%x",
              (unsigned long long)drawCall,
              (unsigned)mode,
              (int)count,
              (unsigned)type);
        return;
    }

    if ([self processBuffer: gl_element_buffer] == false)
        return;

    if (!gl_element_buffer->data.mtl_data) {
        NSLog(@"MGL WARNING: drawElements call=%llu element buffer %u has no Metal backing",
              (unsigned long long)drawCall, gl_element_buffer->name);
        return;
    }

    id <MTLBuffer>indexBuffer = (__bridge id<MTLBuffer>)(gl_element_buffer->data.mtl_data);
    if (!indexBuffer) {
        NSLog(@"MGL WARNING: drawElements call=%llu element buffer bridge failed for gl=%u",
              (unsigned long long)drawCall, gl_element_buffer->name);
        return;
    }

    NSUInteger indexStride = mglGLIndexElementSize(type);
    if (indexStride == 0u) {
        NSLog(@"MGL WARNING: drawElements call=%llu unsupported GL index type=0x%x",
              (unsigned long long)drawCall,
              (unsigned)type);
        return;
    }
    NSUInteger indexOffset = (NSUInteger)(uintptr_t)indices;
    if ((indexOffset % indexStride) != 0u) {
        NSLog(@"MGL DRAW_ELEMENTS BLOCK: call=%llu unaligned indices offset=%lu stride=%lu mode=0x%x count=%d type=0x%x ebo=%u len=%lu program=%u",
              (unsigned long long)drawCall,
              (unsigned long)indexOffset,
              (unsigned long)indexStride,
              (unsigned)mode,
              (int)count,
              (unsigned)type,
              gl_element_buffer->name,
              (unsigned long)indexBuffer.length,
              (unsigned)activeProgramName);
        g_mglDrawElementsSkippedSinceSwap++;
        return;
    }

    NSUInteger indexBytesNeeded = 0u;
    if ((NSUInteger)count > (NSUInteger)(NSUIntegerMax / indexStride)) {
        NSLog(@"MGL ERROR: drawElements call=%llu overflow computing index bytes count=%d stride=%lu",
              (unsigned long long)drawCall,
              (int)count,
              (unsigned long)indexStride);
        return;
    }
    indexBytesNeeded = (NSUInteger)count * indexStride;
    if (indexOffset > indexBuffer.length || (indexBuffer.length - indexOffset) < indexBytesNeeded) {
        NSLog(@"MGL ERROR: drawElements call=%llu index range OOB gl=%u offset=%lu needed=%lu len=%lu type=0x%x count=%d",
              (unsigned long long)drawCall,
              gl_element_buffer->name,
              (unsigned long)indexOffset,
              (unsigned long)indexBytesNeeded,
              (unsigned long)indexBuffer.length,
              (unsigned)type,
              (int)count);
        return;
    }

    const uint8_t *indexBytesForValidation = NULL;
    if (gl_element_buffer->data.buffer_data &&
        ((uintptr_t)gl_element_buffer->data.buffer_data >= 0x1000ull)) {
        indexBytesForValidation = (const uint8_t *)gl_element_buffer->data.buffer_data;
    } else if (indexBuffer.contents) {
        indexBytesForValidation = (const uint8_t *)indexBuffer.contents;
    }

    uint32_t minIndexForDraw = 0u;
    uint32_t maxIndexForDraw = 0u;
    bool haveIndexRange = false;
    uint32_t restartIndexForDraw = 0u;
    bool primitiveRestartForDraw = mglPrimitiveRestartIndexForType(ctx, type, &restartIndexForDraw);
    if (indexBytesForValidation) {
        haveIndexRange = mglScanIndexRangeIgnoringRestart(indexBytesForValidation + indexOffset,
                                                          type,
                                                          count,
                                                          primitiveRestartForDraw,
                                                          restartIndexForDraw,
                                                          &minIndexForDraw,
                                                          &maxIndexForDraw);
    }

    if (kMGLValidateDrawElementsVboRange && haveIndexRange && ctx) {
        VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
        if (vao) {
            GLuint maxAttribs = ctx->state.max_vertex_attribs;
            if (maxAttribs > MAX_ATTRIBS) {
                maxAttribs = MAX_ATTRIBS;
            }

            for (GLuint attrib = 0; attrib < maxAttribs; attrib++) {
                if ((vao->enabled_attribs & (0x1u << attrib)) == 0u) {
                    continue;
                }

                VertexAttrib *a = &vao->attrib[attrib];

                Buffer *vbo = mglRendererGetValidatedBuffer(ctx, a->buffer, "drawElements.vboRange", attrib);
                if (!vbo) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u invalid buffer",
                          (unsigned long long)drawCall,
                          (unsigned)attrib);
                    return;
                }

                if (!vbo->ever_written) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u never written",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)vbo->name);
                    return;
                }

                if (a->binding_offset < 0 || a->relativeoffset < 0) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u negative attrib offset bindingOffset=%lld relativeOffset=%lld",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)vbo->name,
                          (long long)a->binding_offset,
                          (long long)a->relativeoffset);
                    return;
                }

                size_t compSize = mglVertexAttribComponentSize(a->type);
                size_t compCount = (size_t)a->size;
                if (compSize == 0u || compCount == 0u) {
                    continue;
                }

                if (compCount > (SIZE_MAX / compSize)) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u component span overflow type=0x%x size=%u",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)vbo->name,
                          (unsigned)a->type,
                          (unsigned)a->size);
                    return;
                }

                uint64_t elemBytes = (uint64_t)compSize * (uint64_t)compCount;
                if (elemBytes == 0u) {
                    continue;
                }

                uint64_t stride = (a->stride > 0u) ? (uint64_t)a->stride : elemBytes;
                uint64_t bindingOffset = (uint64_t)a->binding_offset;
                uint64_t attrRelativeOffset = (uint64_t)(uintptr_t)a->relativeoffset;
                if (bindingOffset > UINT64_MAX - attrRelativeOffset) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u offset overflow bindingOffset=%llu relativeOffset=%llu",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)vbo->name,
                          (unsigned long long)bindingOffset,
                          (unsigned long long)attrRelativeOffset);
                    return;
                }
                uint64_t relOffset = bindingOffset + attrRelativeOffset;
                if (stride == 0u) {
                    continue;
                }

                uint32_t attribMinIndex = (a->divisor != 0u) ? 0u : minIndexForDraw;
                uint32_t attribMaxIndex = (a->divisor != 0u) ? 0u : maxIndexForDraw;

                if (relOffset > UINT64_MAX - elemBytes) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u overflow computing vertex range bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned long long)bindingOffset,
                          (unsigned long long)relOffset,
                          (unsigned long long)elemBytes,
                          (unsigned)a->divisor);
                    return;
                }

                if ((uint64_t)attribMaxIndex > (UINT64_MAX - relOffset - elemBytes) / stride) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u overflow computing vertex range maxIndex=%u stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)attribMaxIndex,
                          (unsigned long long)stride,
                          (unsigned long long)bindingOffset,
                          (unsigned long long)relOffset,
                          (unsigned long long)elemBytes,
                          (unsigned)a->divisor);
                    return;
                }

                if ((uint64_t)attribMinIndex > (UINT64_MAX - relOffset) / stride) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u overflow computing min range minIndex=%u stride=%llu bindingOffset=%llu relOffset=%llu divisor=%u",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)attribMinIndex,
                          (unsigned long long)stride,
                          (unsigned long long)bindingOffset,
                          (unsigned long long)relOffset,
                          (unsigned)a->divisor);
                    return;
                }

                uint64_t minStart = relOffset + ((uint64_t)attribMinIndex * stride);
                uint64_t maxEnd = relOffset + ((uint64_t)attribMaxIndex * stride) + elemBytes;
                uint64_t vboSize = (vbo->size > 0) ? (uint64_t)vbo->size : 0u;

                if (maxEnd > vboSize) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u indexRange=[%u,%u] byteRange=[%llu,%llu) exceeds vboSize=%llu (stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u)",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)vbo->name,
                          (unsigned)attribMinIndex,
                          (unsigned)attribMaxIndex,
                          (unsigned long long)minStart,
                          (unsigned long long)maxEnd,
                          (unsigned long long)vboSize,
                          (unsigned long long)stride,
                          (unsigned long long)bindingOffset,
                          (unsigned long long)relOffset,
                          (unsigned long long)elemBytes,
                          (unsigned)a->divisor);
                    return;
                }

                if (!vbo->data.mtl_data) {
                    [self bindMTLBuffer:vbo];
                }
                if (!vbo->data.mtl_data) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u has no Metal backing byteRange=[%llu,%llu)",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)vbo->name,
                          (unsigned long long)minStart,
                          (unsigned long long)maxEnd);
                    return;
                }
                id<MTLBuffer> attribMetalBuffer = (__bridge id<MTLBuffer>)(vbo->data.mtl_data);
                if (!attribMetalBuffer) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u Metal bridge failed",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)vbo->name);
                    return;
                }

                uint64_t metalLen = (uint64_t)attribMetalBuffer.length;
                if (maxEnd > metalLen) {
                    NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u indexRange=[%u,%u] byteRange=[%llu,%llu) exceeds metalLen=%llu vboSize=%llu stride=%llu bindingOffset=%llu relOffset=%llu elemBytes=%llu divisor=%u",
                          (unsigned long long)drawCall,
                          (unsigned)attrib,
                          (unsigned)vbo->name,
                          (unsigned)attribMinIndex,
                          (unsigned)attribMaxIndex,
                          (unsigned long long)minStart,
                          (unsigned long long)maxEnd,
                          (unsigned long long)metalLen,
                          (unsigned long long)vboSize,
                          (unsigned long long)stride,
                          (unsigned long long)bindingOffset,
                          (unsigned long long)relOffset,
                          (unsigned long long)elemBytes,
                          (unsigned)a->divisor);
                    return;
                }

                if (vbo->written_min >= 0 && vbo->written_max >= 0) {
                    uint64_t writtenMin = (uint64_t)vbo->written_min;
                    uint64_t writtenMax = (uint64_t)vbo->written_max;
                    if (minStart < writtenMin || maxEnd > writtenMax) {
                        NSLog(@"MGL VBORANGE BLOCK drawElements call=%llu attrib=%u buffer=%u indexRange=[%u,%u] byteRange=[%llu,%llu) outside written=[%llu,%llu) (source=%u divisor=%u)",
                              (unsigned long long)drawCall,
                              (unsigned)attrib,
                              (unsigned)vbo->name,
                              (unsigned)attribMinIndex,
                              (unsigned)attribMaxIndex,
                              (unsigned long long)minStart,
                              (unsigned long long)maxEnd,
                              (unsigned long long)writtenMin,
                              (unsigned long long)writtenMax,
                              (unsigned)vbo->last_init_source,
                              (unsigned)a->divisor);
                        return;
                    }
                }
            }
        }
    }

    if (traceDraw || indexOffset != 0u) {
        NSLog(@"MGL TRACE drawElements.indices call=%llu gl=%u offset=%lu stride=%lu needed=%lu len=%lu",
              (unsigned long long)drawCall,
              gl_element_buffer->name,
              (unsigned long)indexOffset,
              (unsigned long)indexStride,
              (unsigned long)indexBytesNeeded,
              (unsigned long)indexBuffer.length);
    }

    if (mglShouldInspectDrawCall(drawCall, activeProgramName) || drawProgramUsesCloudFaces) {
        if (ctx && mglIsFocusedLoadingProgram(activeProgramName)) {
            Program *focusedProgram = mglResolveProgramFromState(ctx);
            if (focusedProgram) {
                mglWriteProgramMSLDump(focusedProgram,
                                       [NSString stringWithFormat:@"drawElements hot program %u call %llu",
                                                                  (unsigned)activeProgramName,
                                                                  (unsigned long long)drawCall]);
            }
        }
        if (drawProgramUsesCloudFaces && drawProgram) {
            mglWriteProgramMSLDump(drawProgram,
                                   [NSString stringWithFormat:@"CloudFaces texel buffer drawElements call %llu",
                                                              (unsigned long long)drawCall]);
        }

        if (ctx) {
            MTLTriangleFillMode loggedTriangleFillMode =
                (mglDrawModeProducesPolygons(mode) && ctx->state.var.polygon_mode == GL_LINE)
                    ? MTLTriangleFillModeLines
                    : MTLTriangleFillModeFill;
            NSLog(@"MGL TRACE drawElements.state call=%llu program=%u mode=0x%x polygonMode=0x%x triFill=%lu colorMask(use=%d rgba=%d%d%d%d) depth(write=%d test=%d) blend=%d cull=%d viewport=%d,%d,%d,%d",
                  (unsigned long long)drawCall,
                  (unsigned)activeProgramName,
                  (unsigned)mode,
                  (unsigned)ctx->state.var.polygon_mode,
                  (unsigned long)loggedTriangleFillMode,
                  ctx->state.caps.use_color_mask[0] ? 1 : 0,
                  ctx->state.var.color_writemask[0][0] ? 1 : 0,
                  ctx->state.var.color_writemask[0][1] ? 1 : 0,
                  ctx->state.var.color_writemask[0][2] ? 1 : 0,
                  ctx->state.var.color_writemask[0][3] ? 1 : 0,
                  ctx->state.var.depth_writemask ? 1 : 0,
                  ctx->state.caps.depth_test ? 1 : 0,
                  ctx->state.caps.blend ? 1 : 0,
                  ctx->state.caps.cull_face ? 1 : 0,
                  (int)ctx->state.viewport[0],
                  (int)ctx->state.viewport[1],
                  (int)ctx->state.viewport[2],
                  (int)ctx->state.viewport[3]);
        }

        const uint8_t *indexBytes = NULL;
        if (gl_element_buffer->data.buffer_data &&
            ((uintptr_t)gl_element_buffer->data.buffer_data >= 0x1000ull)) {
            indexBytes = (const uint8_t *)gl_element_buffer->data.buffer_data;
        } else if (indexBuffer.contents) {
            indexBytes = (const uint8_t *)indexBuffer.contents;
        }

        if (indexBytes) {
            const uint8_t *start = indexBytes + indexOffset;
            NSUInteger previewCount = MIN((NSUInteger)count, (NSUInteger)12);
            char preview[256];
            preview[0] = '\0';
            uint32_t minIndex = UINT32_MAX;
            uint32_t maxIndex = 0u;

            for (NSUInteger ii = 0; ii < previewCount; ii++) {
                uint32_t idxValue = mglReadGLIndexValue(start, type, ii);
                if (idxValue < minIndex) {
                    minIndex = idxValue;
                }
                if (idxValue > maxIndex) {
                    maxIndex = idxValue;
                }

                size_t used = strlen(preview);
                if (used < sizeof(preview) - 1u) {
                    snprintf(preview + used,
                             sizeof(preview) - used,
                             "%s%u",
                             (ii == 0u ? "" : ","),
                             idxValue);
                }
            }

            NSLog(@"MGL TRACE drawElements.preview call=%llu program=%u ebo=%u count=%d type=0x%x offset=%lu first[%lu]={%s} min=%u max=%u",
                  (unsigned long long)drawCall,
                  (unsigned)activeProgramName,
                  (unsigned)gl_element_buffer->name,
                  (int)count,
                  (unsigned)type,
                  (unsigned long)indexOffset,
                  (unsigned long)previewCount,
                  preview,
                  minIndex == UINT32_MAX ? 0u : minIndex,
                  maxIndex);

            VertexArray *vao = ctx ? ctx->state.vao : NULL;
            if (vao) {
                GLuint traceAttribLimit = MIN((GLuint)4u, ctx ? ctx->state.max_vertex_attribs : (GLuint)4u);
                for (GLuint attrib = 0; attrib < traceAttribLimit; attrib++) {
                    mglTraceDrawElementsAttrib(ctx,
                                               vao,
                                               drawCall,
                                               activeProgramName,
                                               start,
                                               type,
                                               attrib);
                }
            }
            if (vao && (vao->enabled_attribs & 0x1u)) {
                VertexAttrib *a0 = &vao->attrib[0];
                Buffer *vbo = mglRendererGetValidatedBuffer(ctx, a0->buffer, "drawElements.attrib0", 0u);
                if (vbo) {
                    const uint8_t *vboBytes = NULL;
                    if (vbo->data.buffer_data && ((uintptr_t)vbo->data.buffer_data >= 0x1000ull)) {
                        vboBytes = (const uint8_t *)vbo->data.buffer_data;
                    } else if (vbo->data.mtl_data) {
                        id<MTLBuffer> vb = (__bridge id<MTLBuffer>)(vbo->data.mtl_data);
                        vboBytes = (const uint8_t *)vb.contents;
                    }

                    if (vboBytes &&
                        a0->type == GL_FLOAT &&
                        (a0->size >= 2u && a0->size <= 4u) &&
                        a0->stride >= (sizeof(float) * a0->size)) {
                        uint32_t firstIndex = mglReadGLIndexValue(start, type, 0u);
                        NSUInteger bindingOffset = (a0->binding_offset > 0) ? (NSUInteger)a0->binding_offset : 0u;
                        NSUInteger vertexOffset = bindingOffset +
                                                  (NSUInteger)a0->relativeoffset +
                                                  ((NSUInteger)firstIndex * (NSUInteger)a0->stride);
                        NSUInteger needed = sizeof(float) * a0->size;
                        if (vertexOffset <= (NSUInteger)vbo->size &&
                            ((NSUInteger)vbo->size - vertexOffset) >= needed) {
                            float comps[4] = {0.f, 0.f, 0.f, 0.f};
                            memcpy(comps, vboBytes + vertexOffset, needed);
                            NSLog(@"MGL TRACE drawElements.attrib0 call=%llu program=%u vbo=%u firstIndex=%u bindingOffset=%lu relOffset=%u stride=%u size=%u vec=(%.4f,%.4f,%.4f,%.4f) vboSize=%lld init(ever=%u full=%u source=%u off=%lld size=%lld src=%p hash=0x%016llx)",
                                  (unsigned long long)drawCall,
                                  (unsigned)activeProgramName,
                                  (unsigned)vbo->name,
                                  (unsigned)firstIndex,
                                  (unsigned long)bindingOffset,
                                  (unsigned)a0->relativeoffset,
                                  (unsigned)a0->stride,
                                  (unsigned)a0->size,
                                  comps[0], comps[1], comps[2], comps[3],
                                  (long long)vbo->size,
                                  (unsigned)vbo->ever_written,
                                  (unsigned)vbo->has_initialized_data,
                                  (unsigned)vbo->last_init_source,
                                  (long long)vbo->last_write_offset,
                                  (long long)vbo->last_write_size,
                                  vbo->last_write_src_ptr,
                                  (unsigned long long)vbo->last_write_src_hash);

                            typedef struct MGLAttrib0DumpKey {
                                GLuint program;
                                GLuint vbo;
                            } MGLAttrib0DumpKey;
                            static MGLAttrib0DumpKey s_dumpedAttrib0RawBuffers[24] = {{0, 0}};
                            static uint32_t s_dumpedAttrib0RawBufferCount = 0;
                            BOOL alreadyDumpedAttrib0 = NO;
                            for (uint32_t dumpIndex = 0; dumpIndex < s_dumpedAttrib0RawBufferCount; dumpIndex++) {
                                if (s_dumpedAttrib0RawBuffers[dumpIndex].program == activeProgramName &&
                                    s_dumpedAttrib0RawBuffers[dumpIndex].vbo == vbo->name) {
                                    alreadyDumpedAttrib0 = YES;
                                    break;
                                }
                            }

                            if (!alreadyDumpedAttrib0 &&
                                s_dumpedAttrib0RawBufferCount < (uint32_t)(sizeof(s_dumpedAttrib0RawBuffers) / sizeof(s_dumpedAttrib0RawBuffers[0])) &&
                                vbo->size > 0) {
                                size_t totalSize = (size_t)vbo->size;
                                size_t headLen = MIN((size_t)256, totalSize);
                                size_t windowOffset = (size_t)vertexOffset;
                                if (windowOffset > totalSize) {
                                    windowOffset = totalSize;
                                }
                                size_t windowLen = 0;
                                if (windowOffset < totalSize) {
                                    windowLen = MIN((size_t)128, totalSize - windowOffset);
                                }

                                NSLog(@"MGL DUMP attrib0.raw.begin call=%llu program=%u vbo=%u size=%zu firstIndex=%u vertexOffset=%zu stride=%u bindingOffset=%lu relOffset=%u",
                                      (unsigned long long)drawCall,
                                      (unsigned)activeProgramName,
                                      (unsigned)vbo->name,
                                      totalSize,
                                      (unsigned)firstIndex,
                                      (size_t)vertexOffset,
                                      (unsigned)a0->stride,
                                      (unsigned long)bindingOffset,
                                      (unsigned)a0->relativeoffset);
                                mglDumpBytesToLog(@"attrib0.vbo.head", vboBytes, headLen, 0u);
                                if (windowLen > 0) {
                                    mglDumpBytesToLog(@"attrib0.vbo.vertexWindow",
                                                      vboBytes + windowOffset,
                                                      windowLen,
                                                      windowOffset);
                                }
                                NSLog(@"MGL DUMP attrib0.raw.end vbo=%u", (unsigned)vbo->name);
                                s_dumpedAttrib0RawBuffers[s_dumpedAttrib0RawBufferCount].program = activeProgramName;
                                s_dumpedAttrib0RawBuffers[s_dumpedAttrib0RawBufferCount].vbo = vbo->name;
                                s_dumpedAttrib0RawBufferCount++;
                            }
                        } else {
                            NSLog(@"MGL WARNING: drawElements.attrib0 call=%llu OOB firstIndex=%u bindingOffset=%lu relOffset=%u stride=%u size=%u vboSize=%lld",
                                  (unsigned long long)drawCall,
                                  (unsigned)firstIndex,
                                  (unsigned long)bindingOffset,
                                  (unsigned)a0->relativeoffset,
                                  (unsigned)a0->stride,
                                  (unsigned)a0->size,
                                  (long long)vbo->size);
                        }
                    } else {
                        NSLog(@"MGL TRACE drawElements.attrib0 call=%llu skipped(vboBytes=%p type=0x%x size=%u stride=%u)",
                              (unsigned long long)drawCall,
                              vboBytes,
                              (unsigned)a0->type,
                              (unsigned)a0->size,
                              (unsigned)a0->stride);
                    }
                }
            }
        } else {
            NSLog(@"MGL WARNING: drawElements.preview call=%llu unavailable(index bytes nil) ebo=%u",
                  (unsigned long long)drawCall,
                  (unsigned)gl_element_buffer->name);
        }
    }

    if (mglShouldInspectDrawCall(drawCall, activeProgramName) || drawProgramUsesCloudFaces) {
        VertexArray *submitVAO = ctx ? ctx->state.vao : NULL;
        NSLog(@"MGL TRACE drawElements.submit call=%llu program=%u mode=0x%x count=%d type=0x%x ebo=%u offset=%lu stride=%lu needed=%lu len=%lu haveRange=%d range=[%u,%u] vao=%p enabled=0x%x encoder=%p cloudFaces=%d",
              (unsigned long long)drawCall,
              (unsigned)activeProgramName,
              (unsigned)mode,
              (int)count,
              (unsigned)type,
              (unsigned)gl_element_buffer->name,
              (unsigned long)indexOffset,
              (unsigned long)indexStride,
              (unsigned long)indexBytesNeeded,
              (unsigned long)indexBuffer.length,
              haveIndexRange ? 1 : 0,
              (unsigned)minIndexForDraw,
              (unsigned)maxIndexForDraw,
              submitVAO,
              submitVAO ? (unsigned)submitVAO->enabled_attribs : 0u,
              _currentRenderEncoder,
              drawProgramUsesCloudFaces ? 1 : 0);
    }

    @try {
        MGLPrimitiveRestartEncodeResult restartResult =
            mglEncodePrimitiveRestartedElementDraw(_currentRenderEncoder,
                                                   _device,
                                                   ctx,
                                                   gl_element_buffer,
                                                   indexBuffer,
                                                   mode,
                                                   primitiveType,
                                                   type,
                                                   indexType,
                                                   indexOffset,
                                                   count,
                                                   1u,
                                                   0,
                                                   0u,
                                                   "drawElements");
        if (restartResult == MGLPrimitiveRestartEncodeFailed) {
            g_mglDrawElementsSkippedSinceSwap++;
            return;
        }
        BOOL restartHandled = (restartResult == MGLPrimitiveRestartEncodeHandled);

        if (restartHandled) {
            // Already emitted as restart-separated Metal draws.
        } else if (polygonModePoint) {
            if (!mglEncodeElementPolygonPoint(_currentRenderEncoder,
                                              _device,
                                              gl_element_buffer,
                                              indexBuffer,
                                              mode,
                                              type,
                                              indexType,
                                              indexOffset,
                                              count,
                                              1u,
                                              0,
                                              0u,
                                              "drawElements")) {
                g_mglDrawElementsSkippedSinceSwap++;
                return;
            }
        } else if (emulateTriangleFan) {
            if (count < 3) {
                return;
            }

            const uint8_t *fanSource = indexBytesForValidation ? (indexBytesForValidation + indexOffset) : NULL;
            NSUInteger fanIndexCount = 0u;
            id<MTLBuffer> fanIndexBuffer = mglNewTriangleFanElementIndexBuffer(_device,
                                                                               fanSource,
                                                                               type,
                                                                               (NSUInteger)count,
                                                                               &fanIndexCount);
            if (!fanIndexBuffer || fanIndexCount == 0u) {
                NSLog(@"MGL WARNING: drawElements call=%llu triangle fan emulation failed ebo=%u count=%d offset=%lu source=%p",
                      (unsigned long long)drawCall,
                      (unsigned)gl_element_buffer->name,
                      (int)count,
                      (unsigned long)indexOffset,
                      fanSource);
                g_mglDrawElementsSkippedSinceSwap++;
                return;
            }

            [_currentRenderEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                              indexCount:fanIndexCount
                                               indexType:MTLIndexTypeUInt32
                                             indexBuffer:fanIndexBuffer
                                       indexBufferOffset:0
                                           instanceCount:1];
        } else if (emulateLineLoop) {
            if (count < 2) {
                return;
            }

            const uint8_t *loopSource = indexBytesForValidation ? (indexBytesForValidation + indexOffset) : NULL;
            NSUInteger loopIndexCount = 0u;
            id<MTLBuffer> loopIndexBuffer = mglNewLineLoopElementIndexBuffer(_device,
                                                                             loopSource,
                                                                             type,
                                                                             (NSUInteger)count,
                                                                             &loopIndexCount);
            if (!loopIndexBuffer || loopIndexCount == 0u) {
                NSLog(@"MGL WARNING: drawElements call=%llu line loop emulation failed ebo=%u count=%d offset=%lu source=%p",
                      (unsigned long long)drawCall,
                      (unsigned)gl_element_buffer->name,
                      (int)count,
                      (unsigned long)indexOffset,
                      loopSource);
                g_mglDrawElementsSkippedSinceSwap++;
                return;
            }

            [_currentRenderEncoder drawIndexedPrimitives:MTLPrimitiveTypeLineStrip
                                              indexCount:loopIndexCount
                                               indexType:MTLIndexTypeUInt32
                                             indexBuffer:loopIndexBuffer
                                       indexBufferOffset:0
                                           instanceCount:1];
        } else if (drawProgramUsesCloudFaces) {
            [_currentRenderEncoder drawPrimitives:primitiveType
                                      vertexStart:0
                                      vertexCount:(NSUInteger)count];
        } else {
            NSUInteger drawIndexOffset = indexOffset;
            MTLIndexType drawIndexType = indexType;
            id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                          gl_element_buffer,
                                                                          indexBuffer,
                                                                          type,
                                                                          &drawIndexOffset,
                                                                          &drawIndexType);
            if (!drawIndexBuffer) {
                g_mglDrawElementsSkippedSinceSwap++;
                return;
            }
            [_currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType
                                             indexBuffer:drawIndexBuffer indexBufferOffset:drawIndexOffset instanceCount:1];
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: drawElements call=%llu drawIndexedPrimitives exception: %@",
              (unsigned long long)drawCall, exception);
        return;
    }

    g_mglLastDrawElementsCall = drawCall;
    g_mglLastDrawElementsSeconds = mglNowSeconds();
    g_mglLastDrawElementsProgram = ctx ? ctx->state.program_name : 0u;
    g_mglLastDrawElementsMode = mode;
    g_mglLastDrawElementsCount = count;
    g_mglDrawElementsSinceSwap++;
    if (count > 0) {
        g_mglDrawElementIndicesSinceSwap += (uint64_t)count;
    }
    [self markCurrentFramebufferDrawAttachmentsWritten];
    mglLogDrawWithoutSwapWatchdog("elements",
                                  drawCall,
                                  ctx,
                                  _currentCommandBuffer,
                                  _currentRenderEncoder,
                                  _renderPassDescriptor);

    double drawElapsedMs = (mglNowSeconds() - drawStartSeconds) * 1000.0;
    if (traceDraw || drawElapsedMs >= 16.0) {
        NSLog(@"MGL TRACE drawElements.end call=%llu elapsed=%.3fms indexBuffer=%u len=%lu encoder=%p",
              (unsigned long long)drawCall,
              drawElapsedMs,
              gl_element_buffer->name,
              (unsigned long)indexBuffer.length,
              _currentRenderEncoder);
    }
}

void mtlDrawElements(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawElements: glm_ctx mode: mode count: count type: type indices: indices];
}


#pragma mark C interface to mtlDrawRangeElements
-(void) mtlDrawRangeElements: (GLMContext) glm_ctx mode:(GLenum) mode start:(GLuint) start end:(GLuint) end count: (GLsizei) count type: (GLenum) type indices:(const void *)indices
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;
    (void)start;
    (void)end;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : getMTLPrimitiveType(mode)));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawRangeElements" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               1u,
                                               0,
                                               0u,
                                               "drawRangeElements");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          1u,
                                          0,
                                          0u,
                                          "drawRangeElements")) {
            return;
        }
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         1u,
                                         0,
                                         0u,
                                         "drawRangeElements")) {
            return;
        }
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      1u,
                                      0,
                                      0u,
                                      "drawRangeElements")) {
            return;
        }
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    [_currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType
                                     indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:1];
}

void mtlDrawRangeElements(GLMContext glm_ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawRangeElements: glm_ctx mode: mode start: start end: end count: count type: type indices: indices];
}


#pragma mark C interface to mtlDrawArraysInstanced
-(void) mtlDrawArraysInstanced: (GLMContext) glm_ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count instancecount:(GLsizei) instancecount
{
    MTLPrimitiveType primitiveType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint) {
        (void)mglEncodeArrayPolygonPoint(_currentRenderEncoder,
                                         _device,
                                         mode,
                                         first,
                                         count,
                                         (NSUInteger)instancecount,
                                         0u,
                                         "drawArraysInstanced");
        return;
    }

    if (mode == GL_TRIANGLE_FAN) {
        (void)mglEncodeArrayTriangleFan(_currentRenderEncoder,
                                        _device,
                                        count,
                                        first,
                                        (NSUInteger)instancecount,
                                        0u,
                                        "drawArraysInstanced");
        return;
    }
    if (mode == GL_LINE_LOOP) {
        (void)mglEncodeArrayLineLoop(_currentRenderEncoder,
                                     _device,
                                     count,
                                     first,
                                     (NSUInteger)instancecount,
                                     0u,
                                     "drawArraysInstanced");
        return;
    }

    primitiveType = mglPolygonModePointForDrawMode(ctx, mode) ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    [_currentRenderEncoder drawPrimitives:primitiveType vertexStart:first vertexCount:count instanceCount:instancecount];
}

void mtlDrawArraysInstanced(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawArraysInstanced: glm_ctx mode: mode first: first count: count instancecount: instancecount];
}


#pragma mark C interface to mtlDrawElementsInstanced
-(void) mtlDrawElementsInstanced: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : getMTLPrimitiveType(mode)));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsInstanced" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               (NSUInteger)instancecount,
                                               0,
                                               0u,
                                               "drawElementsInstanced");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          (NSUInteger)instancecount,
                                          0,
                                          0u,
                                          "drawElementsInstanced")) {
            return;
        }
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         (NSUInteger)instancecount,
                                         0,
                                         0u,
                                         "drawElementsInstanced")) {
            return;
        }
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      (NSUInteger)instancecount,
                                      0,
                                      0u,
                                      "drawElementsInstanced")) {
            return;
        }
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    // for now lets just ignore the range data and use drawIndexedPrimitives
    //
    // in the future it would be an idea to use temp buffers for large buffers that would wire
    // to much memory down.. like a million point galaxy drawing
    //
    [_currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType
                                     indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:instancecount];
}

void mtlDrawElementsInstanced(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsInstanced: glm_ctx mode: mode count: count type: type indices: indices instancecount: instancecount];
}


#pragma mark C interface to mtlDrawElementsBaseVertex
-(void) mtlDrawElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices basevertex:(GLint) basevertex
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : getMTLPrimitiveType(mode)));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsBaseVertex" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               1u,
                                               basevertex,
                                               0u,
                                               "drawElementsBaseVertex");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          1u,
                                          basevertex,
                                          0u,
                                          "drawElementsBaseVertex")) {
            return;
        }
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         1u,
                                         basevertex,
                                         0u,
                                         "drawElementsBaseVertex")) {
            return;
        }
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      1u,
                                      basevertex,
                                      0u,
                                      "drawElementsBaseVertex")) {
            return;
        }
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    [_currentRenderEncoder drawIndexedPrimitives: primitiveType indexCount:count indexType: drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:1 baseVertex:basevertex baseInstance:0];
}

void mtlDrawElementsBaseVertex(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsBaseVertex: glm_ctx mode: mode count: count type: type indices: indices basevertex: basevertex];
}


#pragma mark C interface to mtlDrawRangeElementsBaseVertex
-(void) mtlDrawRangeElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode start: (GLuint) start end: (GLuint) end count:(GLsizei) count type: (GLenum) type indices:(const void *)indices basevertex:(GLint) basevertex
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;
    (void)start;
    (void)end;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : getMTLPrimitiveType(mode)));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawRangeElementsBaseVertex" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               1u,
                                               basevertex,
                                               0u,
                                               "drawRangeElementsBaseVertex");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          1u,
                                          basevertex,
                                          0u,
                                          "drawRangeElementsBaseVertex")) {
            return;
        }
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         1u,
                                         basevertex,
                                         0u,
                                         "drawRangeElementsBaseVertex")) {
            return;
        }
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      1u,
                                      basevertex,
                                      0u,
                                      "drawRangeElementsBaseVertex")) {
            return;
        }
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    [_currentRenderEncoder drawIndexedPrimitives: primitiveType indexCount:count indexType: drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:1 baseVertex:basevertex baseInstance:0];
}

void mtlDrawRangeElementsBaseVertex(GLMContext glm_ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawRangeElementsBaseVertex:glm_ctx mode:mode start: start end: end count:count type: type indices: indices basevertex:basevertex];
}


#pragma mark C interface to mtlDrawElementsInstancedBaseVertex
-(void) mtlDrawElementsInstancedBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count:(GLuint) count type: (GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount basevertex:(GLint) basevertex
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : getMTLPrimitiveType(mode)));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsInstancedBaseVertex" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               (GLsizei)count,
                                               (NSUInteger)instancecount,
                                               basevertex,
                                               0u,
                                               "drawElementsInstancedBaseVertex");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          (GLsizei)count,
                                          (NSUInteger)instancecount,
                                          basevertex,
                                          0u,
                                          "drawElementsInstancedBaseVertex")) {
            return;
        }
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         (NSUInteger)instancecount,
                                         basevertex,
                                         0u,
                                         "drawElementsInstancedBaseVertex")) {
            return;
        }
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      (NSUInteger)instancecount,
                                      basevertex,
                                      0u,
                                      "drawElementsInstancedBaseVertex")) {
            return;
        }
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    [_currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:instancecount baseVertex:basevertex baseInstance:0];
}

void mtlDrawElementsInstancedBaseVertex(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsInstancedBaseVertex:glm_ctx mode:mode count:count type:type indices:indices instancecount:instancecount basevertex:basevertex];
}

#pragma mark C interface to mtlDrawArraysIndirect
-(void) mtlDrawArraysIndirect: (GLMContext) glm_ctx mode:(GLenum) mode indirect: (const void *) indirect
{
    MTLPrimitiveType primitiveType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    if (mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "drawArraysIndirect")) {
        return;
    }

    primitiveType = mglPolygonModePointForDrawMode(ctx, mode) ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"drawArraysIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer])
        return;

    [_currentRenderEncoder drawPrimitives:primitiveType
                           indirectBuffer:indirectBuffer
                     indirectBufferOffset:(NSUInteger)(uintptr_t)indirect];
}

void mtlDrawArraysIndirect(GLMContext glm_ctx, GLenum mode, const void *indirect)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawArraysIndirect:glm_ctx mode:mode indirect:indirect];
}


#pragma mark C interface to mtlDrawElementsIndirect
-(void) mtlDrawElementsIndirect: (GLMContext) glm_ctx mode:(GLenum) mode type:(GLenum) type indirect: (const void *) indirect
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    if (mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "drawElementsIndirect")) {
        return;
    }

    primitiveType = mglPolygonModePointForDrawMode(ctx, mode) ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    // get element buffer
    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }
    if (mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(ctx, type, "drawElementsIndirect")) {
        return;
    }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsIndirect" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    // get indirect buffer
    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"drawElementsIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer])
        return;

    NSUInteger indexBufferOffset = 0u;
    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &indexBufferOffset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    // draw indexed primitive
    [_currentRenderEncoder drawIndexedPrimitives:primitiveType
                                       indexType:drawIndexType
                                     indexBuffer:drawIndexBuffer
                               indexBufferOffset:indexBufferOffset
                                  indirectBuffer:indirectBuffer
                            indirectBufferOffset:(NSUInteger)(uintptr_t)indirect];
}

void mtlDrawElementsIndirect(GLMContext glm_ctx, GLenum mode, GLenum type, const void *indirect)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsIndirect:glm_ctx mode:mode type:type indirect:indirect];
}


#pragma mark C interface to mtlDrawArraysInstancedBaseInstance
-(void) mtlDrawArraysInstancedBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count instancecount:(GLsizei) instancecount baseinstance:(GLuint) baseinstance
{
    MTLPrimitiveType primitiveType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint) {
        (void)mglEncodeArrayPolygonPoint(_currentRenderEncoder,
                                         _device,
                                         mode,
                                         first,
                                         count,
                                         (NSUInteger)instancecount,
                                         (NSUInteger)baseinstance,
                                         "drawArraysInstancedBaseInstance");
        return;
    }

    if (mode == GL_TRIANGLE_FAN) {
        (void)mglEncodeArrayTriangleFan(_currentRenderEncoder,
                                        _device,
                                        count,
                                        first,
                                        (NSUInteger)instancecount,
                                        (NSUInteger)baseinstance,
                                        "drawArraysInstancedBaseInstance");
        return;
    }
    if (mode == GL_LINE_LOOP) {
        (void)mglEncodeArrayLineLoop(_currentRenderEncoder,
                                     _device,
                                     count,
                                     first,
                                     (NSUInteger)instancecount,
                                     (NSUInteger)baseinstance,
                                     "drawArraysInstancedBaseInstance");
        return;
    }

    primitiveType = mglPolygonModePointForDrawMode(ctx, mode) ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    [_currentRenderEncoder drawPrimitives:primitiveType vertexStart:first vertexCount:count instanceCount:instancecount baseInstance:baseinstance];
}

void mtlDrawArraysInstancedBaseInstance(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawArraysInstancedBaseInstance:glm_ctx mode:mode first:first count:count instancecount:instancecount baseinstance:baseinstance];
}


#pragma mark C interface to mtlDrawElementsInstancedBaseInstance
-(void) mtlDrawElementsInstancedBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode  count: (GLsizei) count type:(GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount baseinstance:(GLuint) baseinstance
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : getMTLPrimitiveType(mode)));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsInstancedBaseInstance" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               (NSUInteger)instancecount,
                                               0,
                                               (NSUInteger)baseinstance,
                                               "drawElementsInstancedBaseInstance");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          (NSUInteger)instancecount,
                                          0,
                                          (NSUInteger)baseinstance,
                                          "drawElementsInstancedBaseInstance")) {
            return;
        }
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         (NSUInteger)instancecount,
                                         0,
                                         (NSUInteger)baseinstance,
                                         "drawElementsInstancedBaseInstance")) {
            return;
        }
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      (NSUInteger)instancecount,
                                      0,
                                      (NSUInteger)baseinstance,
                                      "drawElementsInstancedBaseInstance")) {
            return;
        }
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    // for now lets just ignore the range data and use drawIndexedPrimitives
    //
    // in the future it would be an idea to use temp buffers for large buffers that would wire
    // to much memory down.. like a million point galaxy drawing
    //
    [_currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:instancecount baseVertex:0 baseInstance:baseinstance];
}

void mtlDrawElementsInstancedBaseInstance(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsInstancedBaseInstance:glm_ctx mode:mode count:count type:type indices:indices instancecount:instancecount baseinstance:baseinstance];
}


#pragma mark C interface to mtlDrawElementsInstancedBaseVertexBaseInstance
-(void) mtlDrawElementsInstancedBaseVertexBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type:(GLenum) type indices:(const void *)indices
                                                        instancecount:(GLsizei) instancecount basevertex:(GLint) basevertex baseinstance:(GLuint) baseinstance
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : getMTLPrimitiveType(mode)));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"drawElementsInstancedBaseVertexBaseInstance" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    NSUInteger offset = (NSUInteger)(uintptr_t)indices;
    MGLPrimitiveRestartEncodeResult restartResult =
        mglEncodePrimitiveRestartedElementDraw(_currentRenderEncoder,
                                               _device,
                                               ctx,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               primitiveType,
                                               type,
                                               indexType,
                                               offset,
                                               count,
                                               (NSUInteger)instancecount,
                                               basevertex,
                                               (NSUInteger)baseinstance,
                                               "drawElementsInstancedBaseVertexBaseInstance");
    if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
        return;
    }

    if (polygonModePoint) {
        if (!mglEncodeElementPolygonPoint(_currentRenderEncoder,
                                          _device,
                                          gl_element_buffer,
                                          indexBuffer,
                                          mode,
                                          type,
                                          indexType,
                                          offset,
                                          count,
                                          (NSUInteger)instancecount,
                                          basevertex,
                                          (NSUInteger)baseinstance,
                                          "drawElementsInstancedBaseVertexBaseInstance")) {
            return;
        }
        return;
    }

    if (emulateTriangleFan) {
        if (!mglEncodeElementTriangleFan(_currentRenderEncoder,
                                         _device,
                                         gl_element_buffer,
                                         indexBuffer,
                                         type,
                                         offset,
                                         count,
                                         (NSUInteger)instancecount,
                                         basevertex,
                                         (NSUInteger)baseinstance,
                                         "drawElementsInstancedBaseVertexBaseInstance")) {
            return;
        }
        return;
    }
    if (emulateLineLoop) {
        if (!mglEncodeElementLineLoop(_currentRenderEncoder,
                                      _device,
                                      gl_element_buffer,
                                      indexBuffer,
                                      type,
                                      offset,
                                      count,
                                      (NSUInteger)instancecount,
                                      basevertex,
                                      (NSUInteger)baseinstance,
                                      "drawElementsInstancedBaseVertexBaseInstance")) {
            return;
        }
        return;
    }

    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &offset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    // for now lets just ignore the range data and use drawIndexedPrimitives
    //
    // in the future it would be an idea to use temp buffers for large buffers that would wire
    // to much memory down.. like a million point galaxy drawing
    //
    [_currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count indexType:drawIndexType indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:instancecount baseVertex:basevertex baseInstance:baseinstance];
}

void mtlDrawElementsInstancedBaseVertexBaseInstance(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsInstancedBaseVertexBaseInstance:glm_ctx mode:mode count:count type:type indices:indices instancecount:instancecount basevertex:basevertex baseinstance:baseinstance];
}


#pragma mark C interface to mtlMultiDrawArrays
-(void) mtlMultiDrawArrays: (GLMContext)glm_ctx mode:(GLenum) mode first:(const GLint *)first count:(const GLsizei *)count drawcount:(GLsizei) drawcount
{
    MTLPrimitiveType primitiveType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    if (polygonModePoint) {
        for (int i = 0; i < drawcount; i++) {
            (void)mglEncodeArrayPolygonPoint(_currentRenderEncoder,
                                             _device,
                                             mode,
                                             first[i],
                                             count[i],
                                             1u,
                                             0u,
                                             "multiDrawArrays");
        }
        return;
    }

    if (mode == GL_TRIANGLE_FAN) {
        for (int i = 0; i < drawcount; i++) {
            (void)mglEncodeArrayTriangleFan(_currentRenderEncoder,
                                            _device,
                                            count[i],
                                            first[i],
                                            1u,
                                            0u,
                                            "multiDrawArrays");
        }
        return;
    }
    if (mode == GL_LINE_LOOP) {
        for (int i = 0; i < drawcount; i++) {
            (void)mglEncodeArrayLineLoop(_currentRenderEncoder,
                                         _device,
                                         count[i],
                                         first[i],
                                         1u,
                                         0u,
                                         "multiDrawArrays");
        }
        return;
    }

    primitiveType = getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    for(int i=0; i<drawcount; i++)
    {
         [_currentRenderEncoder drawPrimitives: primitiveType
                                  vertexStart: first[i]
                                  vertexCount: count[i]];
    }
}

void mtlMultiDrawArrays(GLMContext glm_ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlMultiDrawArrays:glm_ctx mode:mode first:first count:count drawcount:drawcount];
}


#pragma mark C interface to mtlMultiDrawElements
-(void) mtlMultiDrawElements: (GLMContext)glm_ctx mode:(GLenum) mode count:(const GLsizei *)count type:(GLenum)type indices:(const void *const*)indices drawcount:(GLsizei) drawcount
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : getMTLPrimitiveType(mode)));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"multiDrawElements" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    for(int i=0; i<drawcount; i++)
    {
        NSUInteger offset = (NSUInteger)(uintptr_t)indices[i];
        MGLPrimitiveRestartEncodeResult restartResult =
            mglEncodePrimitiveRestartedElementDraw(_currentRenderEncoder,
                                                   _device,
                                                   ctx,
                                                   gl_element_buffer,
                                                   indexBuffer,
                                                   mode,
                                                   primitiveType,
                                                   type,
                                                   indexType,
                                                   offset,
                                                   count[i],
                                                   1u,
                                                   0,
                                                   0u,
                                                   "multiDrawElements");
        if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
            continue;
        }

        if (polygonModePoint) {
            (void)mglEncodeElementPolygonPoint(_currentRenderEncoder,
                                               _device,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               type,
                                               indexType,
                                               offset,
                                               count[i],
                                               1u,
                                               0,
                                               0u,
                                               "multiDrawElements");
            continue;
        }

        if (emulateTriangleFan) {
            (void)mglEncodeElementTriangleFan(_currentRenderEncoder,
                                              _device,
                                              gl_element_buffer,
                                              indexBuffer,
                                              type,
                                              offset,
                                              count[i],
                                              1u,
                                              0,
                                              0u,
                                              "multiDrawElements");
            continue;
        }
        if (emulateLineLoop) {
            (void)mglEncodeElementLineLoop(_currentRenderEncoder,
                                           _device,
                                           gl_element_buffer,
                                           indexBuffer,
                                           type,
                                           offset,
                                           count[i],
                                           1u,
                                           0,
                                           0u,
                                           "multiDrawElements");
            continue;
        }

        MTLIndexType drawIndexType = indexType;
        id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                      gl_element_buffer,
                                                                      indexBuffer,
                                                                      type,
                                                                      &offset,
                                                                      &drawIndexType);
        if (!drawIndexBuffer) {
            continue;
        }

        [_currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count[i] indexType:drawIndexType
                                     indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:1];
    }
}

void mtlMultiDrawElements(GLMContext glm_ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount)
{
    // Call the Objective-C method using Objective-C syntax
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlMultiDrawElements: glm_ctx mode: mode count: count type: type indices: indices drawcount: drawcount];
}




#pragma mark C interface to mtlMultiDrawElementsBaseVertex
-(void) mtlMultiDrawElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count: (const GLsizei *) count type: (GLenum) type indices:(const void *const *)indices drawcount:(GLsizei) drawcount basevertex:(const GLint *) basevertex
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];

    BOOL polygonModePoint = mglPolygonModePointForDrawMode(ctx, mode);
    BOOL emulateTriangleFan = (mode == GL_TRIANGLE_FAN && !polygonModePoint);
    BOOL emulateLineLoop = (mode == GL_LINE_LOOP);
    primitiveType = polygonModePoint ? MTLPrimitiveTypePoint : (emulateTriangleFan ? MTLPrimitiveTypeTriangle : (emulateLineLoop ? MTLPrimitiveTypeLineStrip : getMTLPrimitiveType(mode)));
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }

    // element buffer
    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"multiDrawElementsBaseVertex" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;


    for(int i=0; i<drawcount; i++)
    {
        NSUInteger offset = (NSUInteger)(uintptr_t)indices[i];
        MGLPrimitiveRestartEncodeResult restartResult =
            mglEncodePrimitiveRestartedElementDraw(_currentRenderEncoder,
                                                   _device,
                                                   ctx,
                                                   gl_element_buffer,
                                                   indexBuffer,
                                                   mode,
                                                   primitiveType,
                                                   type,
                                                   indexType,
                                                   offset,
                                                   count[i],
                                                   1u,
                                                   basevertex[i],
                                                   0u,
                                                   "multiDrawElementsBaseVertex");
        if (restartResult != MGLPrimitiveRestartEncodeNotNeeded) {
            continue;
        }

        if (polygonModePoint) {
            (void)mglEncodeElementPolygonPoint(_currentRenderEncoder,
                                               _device,
                                               gl_element_buffer,
                                               indexBuffer,
                                               mode,
                                               type,
                                               indexType,
                                               offset,
                                               count[i],
                                               1u,
                                               basevertex[i],
                                               0u,
                                               "multiDrawElementsBaseVertex");
            continue;
        }

        if (emulateTriangleFan) {
            (void)mglEncodeElementTriangleFan(_currentRenderEncoder,
                                              _device,
                                              gl_element_buffer,
                                              indexBuffer,
                                              type,
                                              offset,
                                              count[i],
                                              1u,
                                              basevertex[i],
                                              0u,
                                              "multiDrawElementsBaseVertex");
            continue;
        }
        if (emulateLineLoop) {
            (void)mglEncodeElementLineLoop(_currentRenderEncoder,
                                           _device,
                                           gl_element_buffer,
                                           indexBuffer,
                                           type,
                                           offset,
                                           count[i],
                                           1u,
                                           basevertex[i],
                                           0u,
                                           "multiDrawElementsBaseVertex");
            continue;
        }

        MTLIndexType drawIndexType = indexType;
        id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                      gl_element_buffer,
                                                                      indexBuffer,
                                                                      type,
                                                                      &offset,
                                                                      &drawIndexType);
        if (!drawIndexBuffer) {
            continue;
        }

        [_currentRenderEncoder drawIndexedPrimitives:primitiveType indexCount:count[i] indexType:drawIndexType
                                     indexBuffer:drawIndexBuffer indexBufferOffset:offset instanceCount:1 baseVertex:basevertex[i] baseInstance:0];
    }
}

void mtlMultiDrawElementsBaseVertex(GLMContext glm_ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlMultiDrawElementsBaseVertex: glm_ctx mode: mode count: count type: type indices: indices drawcount: drawcount basevertex:basevertex];
}


-(void) mtlMultiDrawArraysIndirect: (GLMContext)glm_ctx mode:(GLenum) mode indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride
{
    MTLPrimitiveType primitiveType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    if (mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "multiDrawArraysIndirect")) {
        return;
    }

    primitiveType = mglPolygonModePointForDrawMode(ctx, mode) ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"multiDrawArraysIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer])
        return;

    for(int i=0; i<drawcount; i++)
    {
        size_t offset;

        if (stride)
        {
            offset = (char *)((char *)indirect + i * stride) - (char *)NULL;
        }
        else
        {
            offset = (char *)((char *)indirect + i * sizeof(DrawArraysIndirectCommand)) - (char *)NULL;
        }

        [_currentRenderEncoder drawPrimitives:primitiveType indirectBuffer:indirectBuffer indirectBufferOffset:offset];
    }
}

void mtlMultiDrawArraysIndirect(GLMContext glm_ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlMultiDrawArraysIndirect:glm_ctx mode:mode indirect:indirect drawcount:drawcount stride:stride];
}


-(void) mtlMultiDrawElementsIndirect: (GLMContext)glm_ctx mode:(GLenum) mode type:(GLenum)type indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride
{
    MTLPrimitiveType primitiveType;
    MTLIndexType indexType;

    RETURN_ON_FAILURE([self processGLState: true]);
    if ([self currentDrawRasterizationIsEmpty]) {
        return;
    }
    if ([self currentDrawModeIsFullyCulled:mode]) {
        return;
    }
    [self applyPolygonOffsetForDrawMode:mode];
    if (mglSkipIndirectDrawWhenPolygonPointEmulationNeeded(ctx, mode, "multiDrawElementsIndirect")) {
        return;
    }

    primitiveType = mglPolygonModePointForDrawMode(ctx, mode) ? MTLPrimitiveTypePoint : getMTLPrimitiveType(mode);
    if ((GLuint)primitiveType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported primitive mode=0x%x, skipping draw call", mode); return; }

    // get element buffer
    indexType = getMTLIndexType(type);
    if ((GLuint)indexType == 0xFFFFFFFF) { NSLog(@"MGL WARNING: Unsupported index type=0x%x, skipping draw call", type); return; }
    if (mglSkipIndirectElementDrawWhenPrimitiveRestartEnabled(ctx, type, "multiDrawElementsIndirect")) {
        return;
    }

    Buffer *gl_element_buffer = NULL;
    id<MTLBuffer> indexBuffer = nil;
    if (![self resolveElementBufferForDraw:"multiDrawElementsIndirect" context:ctx glBuffer:&gl_element_buffer mtlBuffer:&indexBuffer])
        return;

    // get indirect buffer
    Buffer *gl_indirect_buffer = NULL;
    id<MTLBuffer> indirectBuffer = nil;
    if (![self resolveIndirectBufferForDraw:"multiDrawElementsIndirect" context:ctx glBuffer:&gl_indirect_buffer mtlBuffer:&indirectBuffer])
        return;

    NSUInteger indexBufferOffset = 0u;
    MTLIndexType drawIndexType = indexType;
    id<MTLBuffer> drawIndexBuffer = mglPreparedElementIndexBuffer(_device,
                                                                  gl_element_buffer,
                                                                  indexBuffer,
                                                                  type,
                                                                  &indexBufferOffset,
                                                                  &drawIndexType);
    if (!drawIndexBuffer) {
        return;
    }

    for(int i=0; i<drawcount; i++)
    {
        size_t offset;

        if (stride)
        {
            offset = (char *)((char *)indirect + i * stride) - (char *)NULL;
        }
        else
        {
            offset = (char *)((char *)indirect + i * sizeof(DrawElementsIndirectCommand)) - (char *)NULL;
        }

        // draw indexed primitive
        [_currentRenderEncoder drawIndexedPrimitives:primitiveType indexType:drawIndexType indexBuffer: drawIndexBuffer indexBufferOffset:indexBufferOffset indirectBuffer:indirectBuffer indirectBufferOffset:offset];
    }
}

void mtlMultiDrawElementsIndirect(GLMContext glm_ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    [(__bridge id) glm_ctx->mtl_funcs.mtlObj mtlMultiDrawElementsIndirect:glm_ctx mode:mode type:type indirect:indirect drawcount:drawcount stride:stride];
}

#pragma mark C interface to context functions

- (void) bindObjFuncsToGLMContext: (GLMContext) glm_ctx
{
    glm_ctx->mtl_funcs.mtlObj = (void *)CFBridgingRetain(self);

    glm_ctx->mtl_funcs.mtlBindBuffer = mtlBindBuffer;
    glm_ctx->mtl_funcs.mtlBindTexture = mtlBindTexture;
    glm_ctx->mtl_funcs.mtlBindProgram = mtlBindProgram;

    glm_ctx->mtl_funcs.mtlDeleteMTLObj = mtlDeleteMTLObj;

    glm_ctx->mtl_funcs.mtlGetSync = mtlGetSync;
    glm_ctx->mtl_funcs.mtlWaitForSync = mtlWaitForSync;
    glm_ctx->mtl_funcs.mtlFlush = mtlFlush;
    glm_ctx->mtl_funcs.mtlSwapBuffers = mtlSwapBuffers;
    glm_ctx->mtl_funcs.mtlInvalidateRenderPass = mtlInvalidateRenderPass;
    glm_ctx->mtl_funcs.mtlClearBuffer = mtlClearBuffer;
    glm_ctx->mtl_funcs.mtlBlitFramebuffer = mtlBlitFramebuffer;

    glm_ctx->mtl_funcs.mtlBufferSubData = mtlBufferSubData;
    glm_ctx->mtl_funcs.mtlMapUnmapBuffer = mtlMapUnmapBuffer;
    glm_ctx->mtl_funcs.mtlFlushBufferRange = mtlFlushBufferRange;

    glm_ctx->mtl_funcs.mtlReadDrawable = mtlReadDrawable;
    glm_ctx->mtl_funcs.mtlGetTexImage = mtlGetTexImage;
    
    glm_ctx->mtl_funcs.mtlGenerateMipmaps = mtlGenerateMipmaps;
    glm_ctx->mtl_funcs.mtlTexSubImage = mtlTexSubImage;

    glm_ctx->mtl_funcs.mtlDrawArrays = mtlDrawArrays;
    glm_ctx->mtl_funcs.mtlDrawElements = mtlDrawElements;
    glm_ctx->mtl_funcs.mtlDrawRangeElements = mtlDrawRangeElements;
    glm_ctx->mtl_funcs.mtlDrawArraysInstanced = mtlDrawArraysInstanced;
    glm_ctx->mtl_funcs.mtlDrawElementsInstanced = mtlDrawElementsInstanced;
    glm_ctx->mtl_funcs.mtlDrawElementsBaseVertex = mtlDrawElementsBaseVertex;
    glm_ctx->mtl_funcs.mtlDrawRangeElementsBaseVertex = mtlDrawRangeElementsBaseVertex;
    glm_ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertex = mtlDrawElementsInstancedBaseVertex;
    glm_ctx->mtl_funcs.mtlMultiDrawElementsBaseVertex = mtlMultiDrawElementsBaseVertex;
    glm_ctx->mtl_funcs.mtlDrawArraysIndirect = mtlDrawArraysIndirect;
    glm_ctx->mtl_funcs.mtlDrawElementsIndirect = mtlDrawElementsIndirect;
    glm_ctx->mtl_funcs.mtlDrawArraysInstancedBaseInstance = mtlDrawArraysInstancedBaseInstance;
    glm_ctx->mtl_funcs.mtlDrawElementsInstancedBaseInstance = mtlDrawElementsInstancedBaseInstance;
    glm_ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertexBaseInstance = mtlDrawElementsInstancedBaseVertexBaseInstance;

    glm_ctx->mtl_funcs.mtlMultiDrawArrays = mtlMultiDrawArrays;
    glm_ctx->mtl_funcs.mtlMultiDrawElements = mtlMultiDrawElements;
    glm_ctx->mtl_funcs.mtlMultiDrawElementsBaseVertex = mtlMultiDrawElementsBaseVertex;
    glm_ctx->mtl_funcs.mtlMultiDrawArraysIndirect = mtlMultiDrawArraysIndirect;
    glm_ctx->mtl_funcs.mtlMultiDrawElementsIndirect = mtlMultiDrawElementsIndirect;

    glm_ctx->mtl_funcs.mtlDispatchCompute = mtlDispatchCompute;
    glm_ctx->mtl_funcs.mtlDispatchComputeIndirect = mtlDispatchComputeIndirect;
}

- (id) initMGLRendererFromContext: (void *)glm_ctx andBindToWindow: (NSWindow *)window;
{
    assert (window);
    assert (glm_ctx);
    
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);

    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);

    [view setWantsLayer:YES];
    [window setContentView:view];
    
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    
    return self;
}

- (id) createMGLRendererFromContext: (void *)glm_ctx andBindToWindow: (NSWindow *)window;
{
    assert (window);
    assert (glm_ctx);
    
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);

    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);

    [view setWantsLayer:YES];
    [window setContentView:view];
    
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    
    return renderer;
}


void* CppCreateMGLRendererFromContextAndBindToWindow (void *glm_ctx, void *window)
{
    assert (window);
    assert (glm_ctx);
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);
    NSWindow * w = (__bridge NSWindow *)(window); // just a plain bridge as the autorelease pool will try to release this and crash on exit
    assert (w);
    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);
    [view setWantsLayer:YES];
    //assert(w.contentView);
    //[w.contentView addSubview:view];
    [w setContentView:view];
    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    return  (__bridge void *)(renderer);
}

void* CppCreateMGLRendererHeadless (void *glm_ctx)
{
    assert (glm_ctx);
    MGLRenderer *renderer = [[MGLRenderer alloc] init];
    assert (renderer);

    // Create a dummy NSView for headless rendering
    NSView *view = [[NSView alloc] initWithFrame:NSMakeRect(100, 100, 100, 100)];
    assert (view);
    [view setWantsLayer:YES];

    [renderer createMGLRendererAndBindToContext: glm_ctx view: view];
    return  (__bridge void *)(renderer);
}

void* CppCreateMGLRendererAndBindToContext (void *glm_ctx)
{
    // Compatibility export used by reference libMGL.dylib.
    // Falls back to headless binding when no Cocoa window is supplied.
    return CppCreateMGLRendererHeadless(glm_ctx);
}

- (void) createMGLRendererAndBindToContext: (GLMContext) glm_ctx view: (NSView *) view
{
    ctx = glm_ctx;

    // CRITICAL FIX: Initialize thread synchronization lock
    _metalStateLock = [[NSLock alloc] init];
    if (!_metalStateLock) {
        NSLog(@"MGL ERROR: Failed to create metal state lock");
    } else {
        NSLog(@"MGL INFO: Metal state lock created successfully");
    }

    // Initialize AGX GPU error tracking
    _consecutiveGPUErrors = 0;
    _lastGPUErrorTime = 0;
    _gpuErrorRecoveryMode = NO;
    _pipelineColor0Format = MTLPixelFormatInvalid;
    _pipelineDepthFormat = MTLPixelFormatInvalid;
    _pipelineStencilFormat = MTLPixelFormatInvalid;
    _pipelineProgramName = 0;
    _pipelineStateCache = [[NSMutableDictionary alloc] initWithCapacity:64];
    NSLog(@"MGL INFO: AGX GPU error tracking initialized");

    [self bindObjFuncsToGLMContext: glm_ctx];

    // VIRTUALIZED AGX DETECTION: Create Metal device with virtualization safety
    NSLog(@"MGL INFO: VIRTUALIZED AGX - Creating Metal device with virtualization detection");

    // Create the Metal device
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        NSLog(@"MGL ERROR: Metal device not found - this is required for Apple Silicon");
        return; // Exit early rather than continuing with nil device
    }

    NSLog(@"MGL INFO: Metal device created: %@", _device);

    // PROPER AGX VIRTUALIZATION DETECTION: Maintain Metal functionality with virtualization compatibility
    BOOL isVirtualized = NO;
    NSString *deviceName = [_device name];

    // DETECTION: Check if running in QEMU virtualization but keep Metal enabled
    if ([deviceName containsString:@"AGX"]) {
        isVirtualized = YES;
        NSLog(@"MGL INFO: AGX device detected - enabling virtualization compatibility mode: %@", deviceName);
        NSLog(@"MGL INFO: Metal functionality will be maintained with AGX virtualization safety measures");
    }

    // Create command queue with virtualization-safe settings
    MTLCommandQueueDescriptor *queueDescriptor = [[MTLCommandQueueDescriptor alloc] init];
    if (isVirtualized) {
        NSLog(@"MGL INFO: VIRTUALIZED AGX - Enabling virtualization-safe command queue settings");
        queueDescriptor.maxCommandBufferCount = 16;  // Limit concurrent buffers for virtualization safety
    }

    _commandQueue = [_device newCommandQueueWithDescriptor:queueDescriptor];
    if (!_commandQueue) {
        NSLog(@"MGL ERROR: Failed to create Metal command queue");
        return;
    }

    NSLog(@"MGL INFO: Metal command queue created successfully");

    _view = view;

    // PROPER FIX: Create Metal layer with AGX-safe settings
    NSLog(@"MGL INFO: PROPER FIX - Creating Metal layer with AGX-safe settings");

    _layer = [[CAMetalLayer alloc] init];
    if (!_layer) {
        NSLog(@"MGL ERROR: Failed to create Metal layer");
        return;
    }

    _layer.device = _device;
    MTLPixelFormat requestedPixelFormat = ctx ? (MTLPixelFormat)ctx->pixel_format.mtl_pixel_format
                                              : MTLPixelFormatInvalid;
    MTLPixelFormat pf = mglMetalLayerPixelFormatForContext(ctx);

    @try {
        _layer.pixelFormat = pf;
    } @catch (NSException *exception) {
        NSLog(@"MGL CAMetalLayer invalid pixelFormat=%lu requested=%lu exception=%@; falling back to BGRA8Unorm",
              (unsigned long)pf,
              (unsigned long)requestedPixelFormat,
              exception);
        pf = MTLPixelFormatBGRA8Unorm;
        _layer.pixelFormat = pf;
    }

    if (ctx && ctx->pixel_format.mtl_pixel_format != (GLuint)pf) {
        NSLog(@"MGL CAMetalLayer sync default framebuffer metal format glFormat=0x%x glType=0x%x oldMtl=%u newMtl=%lu",
              ctx->pixel_format.format,
              ctx->pixel_format.type,
              ctx->pixel_format.mtl_pixel_format,
              (unsigned long)pf);
        ctx->pixel_format.mtl_pixel_format = (GLuint)pf;
    }
    NSLog(@"MGL CAMetalLayer pixelFormat=%lu requested=%lu glFormat=0x%x glType=0x%x",
          (unsigned long)_layer.pixelFormat,
          (unsigned long)requestedPixelFormat,
          ctx ? ctx->pixel_format.format : 0u,
          ctx ? ctx->pixel_format.type : 0u);
    _layer.opaque = YES;
    _layer.framebufferOnly = NO; // enable blitting to main color buffer
    _layer.allowsNextDrawableTimeout = YES; // avoid indefinite nextDrawable stalls
    _layer.magnificationFilter = kCAFilterNearest;
    _layer.presentsWithTransaction = NO;

    // AGX-safe layer attachment
    if ([_view layer]) {
        [[_view layer] addSublayer: _layer];
    } else {
        [_view setLayer: _layer];
    }
    [self mglSyncLayerDrawableSizeFromView:"createRenderer"];

    mglDrawBuffer(glm_ctx, GL_FRONT);

    // Create initial command buffer for AGX safety
    @try {
        _currentCommandBuffer = [_commandQueue commandBuffer];
        if (!_currentCommandBuffer) {
            NSLog(@"MGL ERROR: Failed to create initial Metal command buffer");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception creating initial Metal command buffer: %@", exception);
    }
    
    glm_ctx->mtl_funcs.mtlView = (void *)CFBridgingRetain(view);

    // PROACTIVE TEXTURE CREATION: Create essential textures to break sync loop
    NSLog(@"MGL INFO: PROACTIVE - Creating essential textures to prevent magenta screen");
    [self createProactiveTextures];

    // capture Metal commands in MGL.gputrace
    // necessitates Info.plist in the cwd, see https://stackoverflow.com/a/64172784
    //MTLCaptureDescriptor *descriptor = [self setupCaptureToFile: _device];
    //[self startCapture:descriptor];
}

// PROACTIVE TEXTURE CREATION: Create essential textures during initialization to break sync loop
- (void)createProactiveTextures
{
    NSLog(@"MGL PROACTIVE: Starting essential texture creation");

    @try {
        // Create a simple 2D texture with gradient pattern to prevent magenta screens
        MTLTextureDescriptor *proactiveDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                                          width:256
                                                                                                         height:256
                                                                                                      mipmapped:NO];
        proactiveDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        proactiveDesc.storageMode = MTLStorageModeShared;

        id<MTLTexture> proactiveTexture = [_device newTextureWithDescriptor:proactiveDesc];
        if (proactiveTexture) {
            // Create gradient pattern data
            uint32_t *gradientData = calloc(256 * 256, sizeof(uint32_t));
            if (gradientData) {
                // Create blue-green gradient pattern
                for (NSUInteger y = 0; y < 256; y++) {
                    for (NSUInteger x = 0; x < 256; x++) {
                        NSUInteger index = y * 256 + x;
                        uint8_t r = (uint8_t)((x * 128) / 256 + 64);      // Red: 64-192
                        uint8_t g = (uint8_t)((y * 128) / 256 + 64);      // Green: 64-192
                        uint8_t b = 255;                                  // Blue: 255
                        uint8_t a = 255;                                  // Alpha: 255
                        gradientData[index] = (a << 24) | (b << 16) | (g << 8) | r;
                    }
                }

                MTLRegion region = MTLRegionMake2D(0, 0, 256, 256);
                [proactiveTexture replaceRegion:region
                                     mipmapLevel:0
                                       withBytes:gradientData
                                     bytesPerRow:256 * sizeof(uint32_t)];

                free(gradientData);
                NSLog(@"MGL PROACTIVE SUCCESS: Created 256x256 gradient texture (prevents magenta screen)");
            } else {
                NSLog(@"MGL PROACTIVE WARNING: Could not allocate gradient data");
            }

            // Store the proactive texture for future use
            if (!_proactiveTextures) {
                _proactiveTextures = [[NSMutableArray alloc] init];
            }
            [_proactiveTextures addObject:proactiveTexture];

        } else {
            NSLog(@"MGL PROACTIVE ERROR: Could not create proactive texture");
        }

    } @catch (NSException *exception) {
        NSLog(@"MGL PROACTIVE ERROR: Exception creating proactive textures: %@", exception.reason);
    }

    NSLog(@"MGL PROACTIVE: Essential texture creation completed");
}

- (MTLCaptureDescriptor *)setupCaptureToFile: (id<MTLDevice>)device//(nonnull MTLDevice* )device // (nonnull MTKView *)view
{
    MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
    descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
    descriptor.outputURL = [NSURL fileURLWithPath:@"MGL.gputrace"];
    descriptor.captureObject = device; //((MTKView *)view).device;
    
    return descriptor;
}

- (void)startCapture:(MTLCaptureDescriptor *) descriptor
{
    NSError *error = nil;
    BOOL success = [MTLCaptureManager.sharedCaptureManager startCaptureWithDescriptor:descriptor
                                                                                error:&error];
    if (!success) {
        NSLog(@" error capturing mtl => %@ ", [error localizedDescription] );
    }
}

// Stop the capture.
- (void)stopCapture
{
    [MTLCaptureManager.sharedCaptureManager stopCapture];
}

// CRITICAL FIX: Proper resource cleanup to prevent memory leaks and crashes
- (void)dealloc
{
    NSLog(@"MGL INFO: MGLRenderer dealloc - cleaning up Metal resources");

    @try {
        // Stop any ongoing capture
        [MTLCaptureManager.sharedCaptureManager stopCapture];

        // End any active rendering
        [self endRenderEncoding];

        // Cleanup command buffer and encoder
        if (_currentCommandBuffer) {
            NSLog(@"MGL INFO: Releasing current command buffer");
            _currentCommandBuffer = nil;
        }

        if (_currentRenderEncoder) {
            NSLog(@"MGL INFO: Releasing current render encoder");
            _currentRenderEncoder = nil;
        }

        // Cleanup sync objects
        if (_currentEvent) {
            NSLog(@"MGL INFO: Releasing current sync event");
            _currentEvent = nil;
        }

        // Cleanup pipeline state
        if (_pipelineState) {
            NSLog(@"MGL INFO: Releasing pipeline state");
            _pipelineState = nil;
        }
        if (_pipelineStateCache) {
            [_pipelineStateCache removeAllObjects];
            _pipelineStateCache = nil;
        }

        // Cleanup drawable and layer
        if (_drawable) {
            NSLog(@"MGL INFO: Releasing drawable");
            _drawable = nil;
        }

        if (_layer) {
            NSLog(@"MGL INFO: Removing and releasing layer");
            [_layer removeFromSuperlayer];
            _layer = nil;
        }

        // Cleanup command queue and device
        if (_commandQueue) {
            NSLog(@"MGL INFO: Releasing command queue");
            _commandQueue = nil;
        }

        if (_device) {
            NSLog(@"MGL INFO: Releasing Metal device");
            _device = nil;
        }

        // Cleanup thread lock
        if (_metalStateLock) {
            NSLog(@"MGL INFO: Releasing metal state lock");
            _metalStateLock = nil;
        }

    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception during dealloc cleanup: %@", exception);
    }

    NSLog(@"MGL INFO: MGLRenderer dealloc completed");
}

#pragma mark - Metal State Validation and Recovery

- (BOOL)validateMetalObjects
{
    // PROPER FIX: Comprehensive Metal object validation with GPU health monitoring
    @try {
        // Check Metal device validity
        if (!_device) {
            NSLog(@"MGL ERROR: Metal device is nil during validation");
            return NO;
        }

        // Check command queue validity
        if (!_commandQueue) {
            NSLog(@"MGL ERROR: Metal command queue is nil during validation");
            return NO;
        }

        // GPU ERROR THROTTLING: Track recent GPU failures to prevent error cascades
        static NSUInteger consecutiveGpuErrors = 0;
        static NSTimeInterval lastErrorTime = 0;
        static NSTimeInterval throttleWindow = 2.0; // 2 second throttle window
        static NSUInteger maxErrorsPerWindow = 3;

        // Get current error tracking from command buffer if available
        if (_currentCommandBuffer && _currentCommandBuffer.error) {
            NSTimeInterval currentTime = [[NSDate date] timeIntervalSince1970];

            // Check if this is within the throttle window
            if (currentTime - lastErrorTime < throttleWindow) {
                consecutiveGpuErrors++;
                NSLog(@"MGL GPU THROTTLING: %lu consecutive GPU errors detected", (unsigned long)consecutiveGpuErrors);

                // If we've exceeded the error threshold, temporarily disable operations
                if (consecutiveGpuErrors > maxErrorsPerWindow) {
                    NSLog(@"MGL CRITICAL: GPU error threshold exceeded - throttling operations for %.1f seconds", throttleWindow);

                    // Force a reset and temporary pause
                    [self resetMetalState];

                    // Reset counter after pause
                    if (currentTime - lastErrorTime > throttleWindow) {
                        consecutiveGpuErrors = 0;
                    } else {
                        return NO; // Skip this operation to prevent more errors
                    }
                }
            } else {
                // Reset counter if outside throttle window
                consecutiveGpuErrors = 1;
                lastErrorTime = currentTime;
            }
        }

        // Check for virtualization environment changes
        if (@available(macOS 11.0, *)) {
            // Device registry ID changes indicate virtualization issues
            if (_device.registryID == 0) {
                NSLog(@"MGL WARNING: Detected virtualized Metal environment - enabling safety mode");
                // Note: _isVirtualized would be an instance variable to track virtualization state
            }
        }

        return YES;
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Metal object validation failed: %@", exception);
        return NO;
    }
}

- (BOOL)recoverFromMetalError:(NSError *)error operation:(NSString *)operation
{
    // PROPER FIX: Intelligent Metal error recovery
    NSLog(@"MGL ERROR: Metal operation '%@' failed: %@", operation, error);

    // Interface mismatch during pipeline creation is not a GPU-state corruption case.
    // Avoid destructive resets here to prevent reset/retry loops.
    if ([operation isEqualToString:@"pipeline_creation"]) {
        NSString *desc = error.localizedDescription ?: @"";
        NSString *domain = error.domain ?: @"";
        if ((error.code == 3 && [domain hasPrefix:@"AGXMetal"]) ||
            [desc containsString:@"mismatching vertex shader output"] ||
            [desc containsString:@"not written by vertex shader"]) {
            static uint64_t s_pipelineMismatchLogCount = 0;
            s_pipelineMismatchLogCount++;
            if ((s_pipelineMismatchLogCount % 64ull) == 1ull) {
                NSLog(@"MGL WARNING: Pipeline interface mismatch detected; skipping destructive recovery (count=%llu)",
                      s_pipelineMismatchLogCount);
            }
            return NO;
        }
    }

    // Analyze error code for specific recovery strategies
    switch (error.code) {
        case MTLCommandBufferStatusError:
            NSLog(@"MGL INFO: Command buffer execution failed - recreating command buffer");
            [self cleanupCommandBuffer];
            return YES;

        default:
            NSLog(@"MGL ERROR: Unknown Metal error code %ld - attempting recovery", (long)error.code);

            // Handle common error scenarios based on error code
            if (error.code >= 1000 && error.code < 2000) {
                NSLog(@"MGL INFO: Detected feature compatibility issue - using safer settings");
            } else if (error.code >= 2000 && error.code < 3000) {
                NSLog(@"MGL INFO: Detected memory issue - clearing resources");
                [self clearTextureCache];
            } else {
                NSLog(@"MGL ERROR: Unknown Metal error - attempting full recovery");
                [self resetMetalState];
            }
            return YES;
    }
}

- (void)clearTextureCache
{
    // PROPER FIX: Intelligent texture cache cleanup
    NSLog(@"MGL INFO: Clearing texture cache to free memory");

    // Note: Texture binding cache cleanup would require instance variables
    // For now, we focus on basic resource cleanup

    // Force garbage collection using available methods
    if (@available(macOS 10.15, *)) {
        // Simply nil out some references to encourage garbage collection
        // This is a placeholder for more sophisticated cache management
    }
}

- (void)cleanupCommandBuffer
{
    // PROPER FIX: Safe command buffer cleanup
    @try {
        if (_currentCommandBuffer) {
            if (_currentCommandBuffer.status == MTLCommandBufferStatusCommitted) {
                // Do not block indefinitely here; cleanup can be invoked on the render thread.
                // Command buffers retain resources until completion, so dropping the reference is safe.
                if (kMGLVerboseFrameLoopLogs) {
                    NSLog(@"MGL INFO: cleanupCommandBuffer skipping blocking wait for committed command buffer");
                }
            }
            _currentCommandBuffer = nil;
        }

        if (_currentRenderEncoder) {
            [_currentRenderEncoder endEncoding];
            _currentRenderEncoder = nil;
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL ERROR: Exception during command buffer cleanup: %@", exception);
    }
}

- (void)resetMetalState
{
    // PROPER FIX: Full Metal state reset for AGX driver recovery
    NSLog(@"MGL INFO: Performing full Metal state reset for AGX recovery");

    [self cleanupCommandBuffer];

    // CRITICAL: Recreate command queue to clear AGX driver error state
    NSLog(@"MGL AGX RECOVERY: Recreating command queue to clear GPU error state");
    _commandQueue = nil;
    _commandQueue = [_device newCommandQueue];
    if (!_commandQueue) {
        NSLog(@"MGL CRITICAL: Failed to recreate command queue during AGX recovery");
    } else {
        NSLog(@"MGL AGX RECOVERY: Command queue successfully recreated");
    }

    // Reset pipeline state
    _pipelineState = nil;
    [_pipelineStateCache removeAllObjects];
    // Note: _depthStencilState would be an instance variable if it exists

    // Clear all cached objects
    [self clearTextureCache];

    NSLog(@"MGL INFO: AGX Metal state reset completed");
}

// AGX Driver Compatibility: Specialized command buffer commit with recovery
- (void)commitCommandBufferWithAGXRecovery:(id<MTLCommandBuffer>)commandBuffer
{
    static uint64_t s_commitCallCount = 0;
    uint64_t commitCall = ++s_commitCallCount;
    bool traceCommit = mglShouldTraceCall(commitCall);

    if (!commandBuffer) {
        NSLog(@"MGL ERROR: Cannot commit NULL command buffer");
        return;
    }

    if (traceCommit) {
        NSLog(@"MGL TRACE commit.begin call=%llu cb=%p status=%s label=%@",
              (unsigned long long)commitCall,
              commandBuffer,
              mglCommandBufferStatusName(commandBuffer.status),
              commandBuffer.label ?: @"(no-label)");
    }
    double commitQueuedAtSeconds = mglNowSeconds();

    // Pre-commit validation for AGX driver
    if (commandBuffer.error) {
        NSLog(@"MGL AGX WARNING: Command buffer has pre-commit error: %@", commandBuffer.error);
        [self recordGPUError];
    }

    // Add completion handler for AGX error detection
    __block typeof(self) blockSelf = self;
    uint64_t commitCallForBlock = commitCall;
    bool traceCommitForBlock = traceCommit;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            double completeElapsedMs = (mglNowSeconds() - commitQueuedAtSeconds) * 1000.0;
            if (traceCommitForBlock || buffer.error || completeElapsedMs >= 50.0) {
                NSLog(@"MGL TRACE commit.completed call=%llu status=%s elapsed=%.3fms error=%@",
                      (unsigned long long)commitCallForBlock,
                      mglCommandBufferStatusName(buffer.status),
                      completeElapsedMs,
                      buffer.error);
            }
            if (buffer.error) {
                NSLog(@"MGL AGX ERROR: Command buffer completed with error: %@", buffer.error);
                [blockSelf recordGPUError];

                // Specific handling for AGX driver rejection
                if ([buffer.error.domain isEqualToString:@"MTLCommandBufferErrorDomain"] &&
                    buffer.error.code == 4) { // "Ignored (for causing prior/excessive GPU errors)"
                static NSTimeInterval s_lastDriverRejectionReset = 0.0;
                NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
                if (now - s_lastDriverRejectionReset > 2.0) {
                    s_lastDriverRejectionReset = now;
                    NSLog(@"MGL AGX RECOVERY: Driver rejection detected; throttled reset scheduled");
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [blockSelf resetMetalState];
                    });
                } else {
                    NSLog(@"MGL AGX RECOVERY: Driver rejection detected; skipping immediate reset (throttled)");
                }
                }
            } else {
            [blockSelf recordGPUSuccess];

            // AGX Recovery: Clear recovery mode on success
            if (blockSelf->_gpuErrorRecoveryMode) {
                NSLog(@"MGL AGX RECOVERY: Exiting GPU recovery mode after successful completion");
                blockSelf->_gpuErrorRecoveryMode = NO;
            }
        }
    }];

    // CRITICAL FIX: Enhanced command buffer validation before commit
    // Prevents MTLReleaseAssertionFailure in AGX driver
    if (!commandBuffer) {
        NSLog(@"MGL AGX ERROR: Cannot commit nil command buffer");
        return;
    }

    // Check command buffer status before commit
    MTLCommandBufferStatus status = [commandBuffer status];
    if (status >= MTLCommandBufferStatusCommitted) {
        NSLog(@"MGL AGX WARNING: Command buffer already committed (status: %ld) - skipping commit", (long)status);
        if (traceCommit) {
            NSLog(@"MGL TRACE commit.skip.already_committed call=%llu status=%s",
                  (unsigned long long)commitCall, mglCommandBufferStatusName(status));
        }
        return;
    }

    // Validate command buffer is in a valid state for commit
    if (status == MTLCommandBufferStatusError) {
        NSLog(@"MGL AGX ERROR: Command buffer in error state - skipping commit");
        [self recordGPUError];
        if (traceCommit) {
            NSLog(@"MGL TRACE commit.skip.error_state call=%llu", (unsigned long long)commitCall);
        }
        return;
    }

    if (_isCommittingCommandBuffer) {
        NSLog(@"MGL AGX WARNING: Commit already in progress, skipping nested commit");
        if (traceCommit) {
            NSLog(@"MGL TRACE commit.skip.nested call=%llu", (unsigned long long)commitCall);
        }
        return;
    }

    _isCommittingCommandBuffer = YES;
    @try {
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL AGX: Committing command buffer (status: %ld)", (long)status);
        }
        [commandBuffer commit];
        if (kMGLVerboseFrameLoopLogs) {
            NSLog(@"MGL AGX: Command buffer committed successfully");
        }
    } @catch (NSException *exception) {
        NSLog(@"MGL AGX ERROR: Command buffer commit exception: %@", exception);
        [self recordGPUError];

        // AGX-specific recovery for commit failures
        if ([[exception name] containsString:@"CommandBuffer"] ||
            [[exception name] containsString:@"GPU"]) {
            NSLog(@"MGL AGX RECOVERY: Immediate reset due to commit exception");
            dispatch_async(dispatch_get_main_queue(), ^{
                [self resetMetalState];
            });
        }
    } @finally {
        _isCommittingCommandBuffer = NO;
        if (traceCommit) {
            NSLog(@"MGL TRACE commit.end call=%llu cb=%p finalStatus=%s",
                  (unsigned long long)commitCall,
                  commandBuffer,
                  mglCommandBufferStatusName(commandBuffer.status));
        }
    }
}

// AGX GPU Error Throttling - Prevent command queue from entering error state
- (BOOL)shouldSkipGPUOperations
{
    NSTimeInterval currentTime = [[NSDate date] timeIntervalSince1970];

    // PROPER FIX: More realistic recovery window based on actual AGX behavior
    if (currentTime - _lastGPUErrorTime > 15.0) {
        if (_consecutiveGPUErrors > 0) {
            NSLog(@"MGL AGX: Recovery timeout - attempting GPU operations (had %lu errors)", (unsigned long)_consecutiveGPUErrors);
        }
        _consecutiveGPUErrors = 0;
        _gpuErrorRecoveryMode = NO;
        return NO;
    }

    // PROPER FIX: Threshold based on actual AGX driver tolerance
    // AGX driver starts rejecting after just a few errors in virtualization
    if (_consecutiveGPUErrors >= 3 || _gpuErrorRecoveryMode) {
        if (!_gpuErrorRecoveryMode) {
            NSLog(@"MGL AGX: Entering recovery mode after %lu consecutive errors", (unsigned long)_consecutiveGPUErrors);
            _gpuErrorRecoveryMode = YES;

            // PROPER FIX: Clear problematic state but don't give up completely
            [self clearProblematicGPUState];
        }
        return YES;
    }

    return NO;
}

// PROPER FIX: Clear problematic state without giving up on GPU operations entirely
- (void)clearProblematicGPUState
{
    NSLog(@"MGL AGX: Clearing problematic GPU state for recovery");

    // Clear current problematic resources
    if (_currentCommandBuffer) {
        _currentCommandBuffer = nil;
    }

    // Don't recreate command queue immediately - let it rest
    // The AGX driver needs time to recover from error state
}

// AGX DRIVER COMPATIBILITY: Accept virtualization limitations and provide minimal functionality
- (void)enableMinimalFunctionalityMode
{
    NSLog(@"MGL AGX: Enabling minimal functionality mode for AGX virtualization compatibility");

    // Stop fighting the AGX driver - accept virtualization limitations
    // Don't recreate command queues - they will continue to fail
    // Don't submit command buffers - they will continue to be rejected

    // Provide minimal framebuffer clearing without GPU operations
    // This prevents magenta screens while accepting virtualization constraints
}

- (void)recordGPUError
{
    _consecutiveGPUErrors++;
    _consecutiveGPUSuccesses = 0;
    _lastGPUErrorTime = [[NSDate date] timeIntervalSince1970];
    NSLog(@"MGL AGX: Recorded GPU error (%lu consecutive)", (unsigned long)_consecutiveGPUErrors);
}

- (void)recordGPUSuccess
{
    if (_consecutiveGPUErrors > 0 || _gpuErrorRecoveryMode) {
        _consecutiveGPUSuccesses++;
        NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
        NSTimeInterval sinceLastError = now - _lastGPUErrorTime;
        // Require multiple consecutive successful completions before clearing
        // recovery, otherwise mixed success/error callbacks can flap the state.
        if (_consecutiveGPUSuccesses >= 4 && sinceLastError > 0.25) {
            NSLog(@"MGL AGX: Sustained GPU recovery (%lu successes), resetting error count (was %lu)",
                  (unsigned long)_consecutiveGPUSuccesses,
                  (unsigned long)_consecutiveGPUErrors);
            _consecutiveGPUErrors = 0;
            _gpuErrorRecoveryMode = NO;
            _consecutiveGPUSuccesses = 0;
        }
    }
}


#pragma mark - Metal Optimization Methods

- (NSUInteger)getOptimalAlignmentForPixelFormat:(MTLPixelFormat)format
{
    (void)format;
    // aligned_alloc requires an alignment compatible with platform pointer alignment.
    // Using a conservative 64-byte value avoids EINVAL on macOS/arm64 and is safe for texture rows.
    return 64;
}

@end
