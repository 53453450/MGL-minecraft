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
 * mgl_metal_bridge.m
 * MGL
 *
 * C -> Objective-C bridge layer extracted from MGLRenderer.m.
 *
 * Each function is a plain C entry point (matching the function pointer
 * signature in `struct GLMMetalFuncs`) that forwards into an MGLRenderer
 * Objective-C method through the bridged Metal object:
 *     [mglBridgeTarget(glm_ctx, __func__) <selector>]
 *
 * The nil/NULL guards and @try/@catch safety nets are preserved verbatim
 * from MGLRenderer.m so behaviour is unchanged.
 */

#include "glm_context.h"
#include "mgl_metal_bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/* Fix #21: Named sentinel for pointer-validity checks.  Any pointer below
 * this address is treated as invalid (catches NULL and low-page dereferences
 * that would crash on dereference).  Replaces the previous magic 0x1000
 * literals scattered through the bridge. */
static const uintptr_t kMGLMinValidPointer = 0x1000;

/* Helper that validates glm_ctx and the bridged Metal object, returning the
 * Objective-C target or nil on failure.  mtlObj is stored as `void *` in
 * GLMMetalFuncs, so the bridge casts it to `id<MGLMetalBridgeTarget>` to give
 * the compiler the selector and return-type information it needs (especially
 * under ARC).  The protocol is declared in mgl_metal_bridge.h (Fix #14). */
static id<MGLMetalBridgeTarget> mglBridgeTarget(GLMContext glm_ctx, const char *function)
{
    /* Fix #21: use the named sentinel instead of a magic 0x1000 literal. */
    if (!glm_ctx || ((uintptr_t)glm_ctx < kMGLMinValidPointer)) {
        NSLog(@"MGL ERROR: %s received invalid GLM context %p",
              function ? function : "mglBridgeTarget", glm_ctx);
        return nil;
    }
    if (!glm_ctx->mtl_funcs.mtlObj || ((uintptr_t)glm_ctx->mtl_funcs.mtlObj < kMGLMinValidPointer)) {
        NSLog(@"MGL ERROR: %s received invalid Metal object %p",
              function ? function : "mglBridgeTarget", glm_ctx->mtl_funcs.mtlObj);
        return nil;
    }
    return (__bridge id<MGLMetalBridgeTarget>)glm_ctx->mtl_funcs.mtlObj;
}

#pragma mark - Bind / Delete

void mtlBindBuffer(GLMContext glm_ctx, Buffer *ptr) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target bindMTLBuffer:ptr];
}

void mtlBindTexture(GLMContext glm_ctx, Texture *ptr) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target bindMTLTexture:ptr];
}

void mtlBindProgram(GLMContext glm_ctx, Program *ptr) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target bindMTLProgram:ptr];
}

void mtlDeleteMTLObj(GLMContext glm_ctx, void *obj) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlDeleteMTLObj: glm_ctx buffer: obj];
}

#pragma mark - Sync

void mtlGetSync(GLMContext glm_ctx, Sync *sync) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlGetSync: glm_ctx sync: sync];
}

void mtlWaitForSync(GLMContext glm_ctx, Sync *sync) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlWaitForSync: glm_ctx sync: sync];
}

GLenum mtlGetSyncStatus(GLMContext glm_ctx, Sync *sync) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return 0;
    }
    return [target mtlGetSyncStatus: glm_ctx sync: sync];
}

void mtlReleaseSync(GLMContext glm_ctx, Sync *sync) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlReleaseSync: glm_ctx sync: sync];
}

#pragma mark - Flush / Swap

void mtlFlushDrawBuffer(GLMContext glm_ctx) {
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) return;
    @autoreleasepool {
        @try {
            [target flushDrawBuffer:glm_ctx];
        } @catch (NSException *e) {
            NSLog(@"MGL ERROR: mtlFlushDrawBuffer exception: %@", e);
        }
    }
}

void mtlFlush(GLMContext glm_ctx, bool finish) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlFlush:glm_ctx finish:finish];
    }
}

void mtlSwapBuffers(GLMContext glm_ctx) {
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        @try {
            [target mtlSwapBuffers: glm_ctx];
        } @catch (NSException *exception) {
            NSLog(@"MGL CRITICAL: mtlSwapBuffers - Exception caught: %@", exception);
            NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);
        }
    }
}

#pragma mark - Clear / Blit / Invalidate

void mtlClearBuffer(GLMContext glm_ctx, GLuint type, GLbitfield mask) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlClearBuffer: glm_ctx type: type mask: mask];
}

void mtlBlitFramebuffer(GLMContext glm_ctx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) {
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlBlitFramebuffer:glm_ctx srcX0:srcX0 srcY0:srcY0 srcX1:srcX1 srcY1:srcY1 dstX0:dstX0 dstY0:dstY0 dstX1:dstX1 dstY1:dstY1 mask:mask filter:filter];
}

void mtlInvalidateRenderPass(GLMContext glm_ctx) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlInvalidateRenderPass:glm_ctx];
}

#pragma mark - Buffer

void mtlBufferSubData(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, const void *ptr) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlBufferSubData: glm_ctx buf: buf offset:offset size:size ptr:ptr];
}

void *mtlMapUnmapBuffer(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, GLenum access, bool map) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return NULL;
    }
    return [target mtlMapUnmapBuffer: glm_ctx buf: buf offset: offset size: size access: access map: map];
}

void mtlFlushBufferRange(GLMContext glm_ctx, Buffer *buf, GLintptr offset, GLsizeiptr length) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlFlushMappedBufferRange: glm_ctx buf: buf offset: offset length: length];
}

#pragma mark - Readback

void mtlReadDrawable(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlReadDrawable:glm_ctx pixelBytes:pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)width, (NSUInteger)height)];
}

void mtlReadIntegerPixels(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlReadIntegerPixels:glm_ctx pixelBytes:pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)width, (NSUInteger)height) format:format type:type];
}

void mtlReadDepthPixels(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlReadDepthPixels: glm_ctx pixelBytes:pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)width, (NSUInteger)height)];
}

void mtlGetTexImage(GLMContext glm_ctx, Texture *tex, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLuint level, GLuint slice) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlGetTexImage:glm_ctx tex:tex pixelBytes:pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:MTLRegionMake2D((NSUInteger)x, (NSUInteger)y, (NSUInteger)width, (NSUInteger)height) format:format type:type mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice];
}

#pragma mark - Texture

void mtlCopyTexSubImage(GLMContext glm_ctx, Texture *tex, GLuint slice, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlCopyTexSubImage:glm_ctx tex:tex slice:(NSUInteger)slice mipmapLevel:(NSUInteger)level xoffset:(NSInteger)xoffset yoffset:(NSInteger)yoffset x:(NSInteger)x y:(NSInteger)y width:(NSUInteger)width height:(NSUInteger)height];
}

void mtlGenerateMipmaps(GLMContext glm_ctx, Texture *tex) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlGenerateMipmaps:glm_ctx forTexture:tex];
}

void mtlCopyImageSubData(GLMContext glm_ctx, Texture *srcTex, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, Texture *dstTex, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei width, GLsizei height, GLsizei depth) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlCopyImageSubData:glm_ctx srcTexture:srcTex srcLevel:srcLevel srcX:srcX srcY:srcY srcZ:srcZ dstTexture:dstTex dstLevel:dstLevel dstX:dstX dstY:dstY dstZ:dstZ width:width height:height depth:depth];
}

void mtlTexSubImage(GLMContext glm_ctx, Texture *tex, Buffer *buf, size_t src_offset, size_t src_pitch, size_t src_image_size, size_t src_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlTexSubImage:glm_ctx tex:tex buf:buf src_offset:src_offset src_pitch:src_pitch src_image_size:src_image_size src_size:src_size slice:slice level:level width:width height:height depth:depth xoffset:xoffset yoffset:yoffset zoffset:zoffset];
}

bool mtlTexSubImageBytes(GLMContext glm_ctx, Texture *tex, const void *bytes, size_t bytes_size, size_t src_offset, size_t src_pitch, size_t src_image_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return false;
    }
    return [target mtlTexSubImageBytes:glm_ctx tex:tex bytes:bytes bytesSize:bytes_size src_offset:src_offset src_pitch:src_pitch src_image_size:src_image_size slice:slice level:level width:width height:height depth:depth xoffset:xoffset yoffset:yoffset zoffset:zoffset];
}

#pragma mark - Compute

void mtlDispatchCompute(GLMContext glm_ctx, GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlDispatchCompute: glm_ctx groupsX:num_groups_x groupsY:num_groups_y groupsZ:num_groups_z];
}

void mtlDispatchComputeIndirect(GLMContext glm_ctx, GLintptr indirect) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlDispatchComputeIndirect: glm_ctx indirect:indirect];
}

#pragma mark - Query

void mtlBeginTimerQuery(GLMContext glm_ctx) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlBeginTimerQuery: glm_ctx];
}

GLuint64 mtlEndTimerQuery(GLMContext glm_ctx) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return 0;
    }
    return [target mtlEndTimerQuery: glm_ctx];
}

GLuint64 mtlGetGPUTimestamp(GLMContext glm_ctx) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return 0;
    }
    return [target mtlGetGPUTimestamp: glm_ctx];
}

void mtlBeginSampleQuery(GLMContext glm_ctx) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlBeginSampleQuery: glm_ctx];
}

GLuint64 mtlEndSampleQuery(GLMContext glm_ctx) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return 0;
    }
    return [target mtlEndSampleQuery: glm_ctx];
}

#pragma mark - Draw

void mtlDrawArrays(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count) {
    @autoreleasepool {
        @try {
            /* Fix #21: use the named sentinel instead of a magic 0x1000 literal.
             * (This function already had inline NULL guards, so Fix #4 does not
             * add a separate mglBridgeTarget() guard here.) */
            if (!glm_ctx || ((uintptr_t)glm_ctx < kMGLMinValidPointer)) {
                NSLog(@"MGL CRITICAL: mtlDrawArrays - Invalid GLM context, aborting operation");
                return;
            }
            if (!glm_ctx->mtl_funcs.mtlObj || ((uintptr_t)glm_ctx->mtl_funcs.mtlObj < kMGLMinValidPointer)) {
                NSLog(@"MGL CRITICAL: mtlDrawArrays - Invalid Metal object, aborting operation");
                return;
            }
            [mglBridgeTarget(glm_ctx, __func__) mtlDrawArrays: glm_ctx mode: mode first: first count: count];
        } @catch (NSException *exception) {
            NSLog(@"MGL CRITICAL: mtlDrawArrays - Unhandled exception caught: %@", exception);
            NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);
        }
    }
    /* Fix #19: removed the broad `@catch (id exception)` block that was
     * swallowing non-NSException objects; the NSException handler above is
     * sufficient for the documented failure modes. */
}

void mtlDrawElements(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawElements: glm_ctx mode: mode count: count type: type indices: indices];
    }
}

void mtlDrawRangeElements(GLMContext glm_ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawRangeElements: glm_ctx mode: mode start: start end: end count: count type: type indices: indices];
    }
}

void mtlDrawArraysInstanced(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawArraysInstanced: glm_ctx mode: mode first: first count: count instancecount: instancecount];
    }
}

void mtlDrawElementsInstanced(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawElementsInstanced: glm_ctx mode: mode count: count type: type indices: indices instancecount: instancecount];
    }
}

void mtlDrawElementsBaseVertex(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawElementsBaseVertex: glm_ctx mode: mode count: count type: type indices: indices basevertex: basevertex];
    }
}

void mtlDrawRangeElementsBaseVertex(GLMContext glm_ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawRangeElementsBaseVertex:glm_ctx mode:mode start: start end: end count:count type: type indices: indices basevertex:basevertex];
    }
}

void mtlDrawElementsInstancedBaseVertex(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawElementsInstancedBaseVertex:glm_ctx mode:mode count:count type:type indices:indices instancecount:instancecount basevertex:basevertex];
    }
}

void mtlDrawArraysIndirect(GLMContext glm_ctx, GLenum mode, const void *indirect) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawArraysIndirect:glm_ctx mode:mode indirect:indirect];
    }
}

void mtlDrawElementsIndirect(GLMContext glm_ctx, GLenum mode, GLenum type, const void *indirect) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawElementsIndirect:glm_ctx mode:mode type:type indirect:indirect];
    }
}

void mtlDrawArraysInstancedBaseInstance(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawArraysInstancedBaseInstance:glm_ctx mode:mode first:first count:count instancecount:instancecount baseinstance:baseinstance];
    }
}

void mtlDrawElementsInstancedBaseInstance(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawElementsInstancedBaseInstance:glm_ctx mode:mode count:count type:type indices:indices instancecount:instancecount baseinstance:baseinstance];
    }
}

void mtlDrawElementsInstancedBaseVertexBaseInstance(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    @autoreleasepool {
        [target mtlDrawElementsInstancedBaseVertexBaseInstance:glm_ctx mode:mode count:count type:type indices:indices instancecount:instancecount basevertex:basevertex baseinstance:baseinstance];
    }
}

void mtlMultiDrawArrays(GLMContext glm_ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlMultiDrawArrays:glm_ctx mode:mode first:first count:count drawcount:drawcount];
}

void mtlMultiDrawElements(GLMContext glm_ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlMultiDrawElements: glm_ctx mode: mode count: count type: type indices: indices drawcount: drawcount];
}

void mtlMultiDrawElementsBaseVertex(GLMContext glm_ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlMultiDrawElementsBaseVertex: glm_ctx mode: mode count: count type: type indices: indices drawcount: drawcount basevertex:basevertex];
}

void mtlMultiDrawArraysIndirect(GLMContext glm_ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlMultiDrawArraysIndirect:glm_ctx mode:mode indirect:indirect drawcount:drawcount stride:stride];
}

void mtlMultiDrawElementsIndirect(GLMContext glm_ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride) {
    /* Fix #4: NULL-guard the bridged target before dispatching. */
    id<MGLMetalBridgeTarget> target = mglBridgeTarget(glm_ctx, __func__);
    if (!target) {
        return;
    }
    [target mtlMultiDrawElementsIndirect:glm_ctx mode:mode type:type indirect:indirect drawcount:drawcount stride:stride];
}
