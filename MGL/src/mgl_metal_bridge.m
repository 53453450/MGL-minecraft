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
 *     [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj <selector>]
 *
 * The nil/NULL guards and @try/@catch safety nets are preserved verbatim
 * from MGLRenderer.m so behaviour is unchanged.
 */

#include "glm_context.h"
#include "mgl_metal_bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/* Protocol describing the MGLRenderer Objective-C methods invoked by the
 * bridge functions below.  mtlObj is stored as `void *` in GLMMetalFuncs, so
 * the bridge casts it to `id<MGLMetalBridgeTarget>` to give the compiler the
 * selector and return-type information it needs (especially under ARC). */
@protocol MGLMetalBridgeTarget
- (void)bindMTLBuffer:(Buffer *)ptr;
- (void)bindMTLTexture:(Texture *)ptr;
- (void)bindMTLProgram:(Program *)ptr;
- (void)mtlDeleteMTLObj:(GLMContext)glm_ctx buffer:(void *)obj;
- (void)mtlGetSync:(GLMContext)glm_ctx sync:(Sync *)sync;
- (void)mtlWaitForSync:(GLMContext)glm_ctx sync:(Sync *)sync;
- (GLenum)mtlGetSyncStatus:(GLMContext)glm_ctx sync:(Sync *)sync;
- (void)mtlReleaseSync:(GLMContext)glm_ctx sync:(Sync *)sync;
- (void)flushDrawBuffer:(GLMContext)glm_ctx;
- (void)mtlFlush:(GLMContext)glm_ctx finish:(bool)finish;
- (void)mtlSwapBuffers:(GLMContext)glm_ctx;
- (void)mtlClearBuffer:(GLMContext)glm_ctx type:(GLuint)type mask:(GLbitfield)mask;
- (void)mtlBlitFramebuffer:(GLMContext)glm_ctx srcX0:(GLint)srcX0 srcY0:(GLint)srcY0 srcX1:(GLint)srcX1 srcY1:(GLint)srcY1 dstX0:(GLint)dstX0 dstY0:(GLint)dstY0 dstX1:(GLint)dstX1 dstY1:(GLint)dstY1 mask:(GLbitfield)mask filter:(GLenum)filter;
- (void)mtlInvalidateRenderPass:(GLMContext)glm_ctx;
- (void)mtlBufferSubData:(GLMContext)glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size ptr:(const void *)ptr;
- (void *)mtlMapUnmapBuffer:(GLMContext)glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size access:(GLenum)access map:(bool)map;
- (void)mtlFlushMappedBufferRange:(GLMContext)glm_ctx buf:(Buffer *)buf offset:(GLintptr)offset length:(GLsizeiptr)length;
- (void)mtlReadDrawable:(GLMContext)glm_ctx pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MTLRegion)region;
- (void)mtlReadIntegerPixels:(GLMContext)glm_ctx pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MTLRegion)region format:(GLenum)format type:(GLenum)type;
- (void)mtlReadDepthPixels:(GLMContext)glm_ctx pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MTLRegion)region;
- (void)mtlGetTexImage:(GLMContext)glm_ctx tex:(Texture *)tex pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MTLRegion)region format:(GLenum)format type:(GLenum)type mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice;
- (void)mtlCopyTexSubImage:(GLMContext)glm_ctx tex:(Texture *)tex slice:(NSUInteger)slice mipmapLevel:(NSUInteger)level xoffset:(NSInteger)xoffset yoffset:(NSInteger)yoffset x:(NSInteger)x y:(NSInteger)y width:(NSUInteger)width height:(NSUInteger)height;
- (void)mtlGenerateMipmaps:(GLMContext)glm_ctx forTexture:(Texture *)tex;
- (void)mtlCopyImageSubData:(GLMContext)glm_ctx srcTexture:(Texture *)srcTex srcLevel:(GLint)srcLevel srcX:(GLint)srcX srcY:(GLint)srcY srcZ:(GLint)srcZ dstTexture:(Texture *)dstTex dstLevel:(GLint)dstLevel dstX:(GLint)dstX dstY:(GLint)dstY dstZ:(GLint)dstZ width:(GLsizei)width height:(GLsizei)height depth:(GLsizei)depth;
- (void)mtlTexSubImage:(GLMContext)glm_ctx tex:(Texture *)tex buf:(Buffer *)buf src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size src_size:(size_t)src_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset;
- (bool)mtlTexSubImageBytes:(GLMContext)glm_ctx tex:(Texture *)tex bytes:(const void *)bytes bytesSize:(size_t)bytes_size src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset;
- (void)mtlDispatchCompute:(GLMContext)glm_ctx groupsX:(GLuint)groups_x groupsY:(GLuint)groups_y groupsZ:(GLuint)groups_z;
- (void)mtlDispatchComputeIndirect:(GLMContext)glm_ctx indirect:(GLintptr)indirect;
- (void)mtlBeginTimerQuery:(GLMContext)glm_ctx;
- (GLuint64)mtlEndTimerQuery:(GLMContext)glm_ctx;
- (GLuint64)mtlGetGPUTimestamp:(GLMContext)glm_ctx;
- (void)mtlBeginSampleQuery:(GLMContext)glm_ctx;
- (GLuint64)mtlEndSampleQuery:(GLMContext)glm_ctx;
- (void)mtlDrawArrays:(GLMContext)glm_ctx mode:(GLenum)mode first:(GLint)first count:(GLsizei)count;
- (void)mtlDrawElements:(GLMContext)glm_ctx mode:(GLenum)mode count:(GLsizei)count type:(GLenum)type indices:(const void *)indices;
- (void)mtlDrawRangeElements:(GLMContext)glm_ctx mode:(GLenum)mode start:(GLuint)start end:(GLuint)end count:(GLsizei)count type:(GLenum)type indices:(const void *)indices;
- (void)mtlDrawArraysInstanced:(GLMContext)glm_ctx mode:(GLenum)mode first:(GLint)first count:(GLsizei)count instancecount:(GLsizei)instancecount;
- (void)mtlDrawElementsInstanced:(GLMContext)glm_ctx mode:(GLenum)mode count:(GLsizei)count type:(GLenum)type indices:(const void *)indices instancecount:(GLsizei)instancecount;
- (void)mtlDrawElementsBaseVertex:(GLMContext)glm_ctx mode:(GLenum)mode count:(GLsizei)count type:(GLenum)type indices:(const void *)indices basevertex:(GLint)basevertex;
- (void)mtlDrawRangeElementsBaseVertex:(GLMContext)glm_ctx mode:(GLenum)mode start:(GLuint)start end:(GLuint)end count:(GLsizei)count type:(GLenum)type indices:(const void *)indices basevertex:(GLint)basevertex;
- (void)mtlDrawElementsInstancedBaseVertex:(GLMContext)glm_ctx mode:(GLenum)mode count:(GLsizei)count type:(GLenum)type indices:(const void *)indices instancecount:(GLsizei)instancecount basevertex:(GLint)basevertex;
- (void)mtlDrawArraysIndirect:(GLMContext)glm_ctx mode:(GLenum)mode indirect:(const void *)indirect;
- (void)mtlDrawElementsIndirect:(GLMContext)glm_ctx mode:(GLenum)mode type:(GLenum)type indirect:(const void *)indirect;
- (void)mtlDrawArraysInstancedBaseInstance:(GLMContext)glm_ctx mode:(GLenum)mode first:(GLint)first count:(GLsizei)count instancecount:(GLsizei)instancecount baseinstance:(GLuint)baseinstance;
- (void)mtlDrawElementsInstancedBaseInstance:(GLMContext)glm_ctx mode:(GLenum)mode count:(GLsizei)count type:(GLenum)type indices:(const void *)indices instancecount:(GLsizei)instancecount baseinstance:(GLuint)baseinstance;
- (void)mtlDrawElementsInstancedBaseVertexBaseInstance:(GLMContext)glm_ctx mode:(GLenum)mode count:(GLsizei)count type:(GLenum)type indices:(const void *)indices instancecount:(GLsizei)instancecount basevertex:(GLint)basevertex baseinstance:(GLuint)baseinstance;
- (void)mtlMultiDrawArrays:(GLMContext)glm_ctx mode:(GLenum)mode first:(const GLint *)first count:(const GLsizei *)count drawcount:(GLsizei)drawcount;
- (void)mtlMultiDrawElements:(GLMContext)glm_ctx mode:(GLenum)mode count:(const GLsizei *)count type:(GLenum)type indices:(const void *const*)indices drawcount:(GLsizei)drawcount;
- (void)mtlMultiDrawElementsBaseVertex:(GLMContext)glm_ctx mode:(GLenum)mode count:(const GLsizei *)count type:(GLenum)type indices:(const void *const*)indices drawcount:(GLsizei)drawcount basevertex:(const GLint *)basevertex;
- (void)mtlMultiDrawArraysIndirect:(GLMContext)glm_ctx mode:(GLenum)mode indirect:(const void *)indirect drawcount:(GLsizei)drawcount stride:(GLsizei)stride;
- (void)mtlMultiDrawElementsIndirect:(GLMContext)glm_ctx mode:(GLenum)mode type:(GLenum)type indirect:(const void *)indirect drawcount:(GLsizei)drawcount stride:(GLsizei)stride;
@end

#pragma mark - Bind / Delete

void mtlBindBuffer(GLMContext glm_ctx, Buffer *ptr) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj bindMTLBuffer:ptr];
}

void mtlBindTexture(GLMContext glm_ctx, Texture *ptr) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj bindMTLTexture:ptr];
}

void mtlBindProgram(GLMContext glm_ctx, Program *ptr) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj bindMTLProgram:ptr];
}

void mtlDeleteMTLObj(GLMContext glm_ctx, void *obj) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDeleteMTLObj: glm_ctx buffer: obj];
}

#pragma mark - Sync

void mtlGetSync(GLMContext glm_ctx, Sync *sync) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlGetSync: glm_ctx sync: sync];
}

void mtlWaitForSync(GLMContext glm_ctx, Sync *sync) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlWaitForSync: glm_ctx sync: sync];
}

GLenum mtlGetSyncStatus(GLMContext glm_ctx, Sync *sync) {
    return [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlGetSyncStatus: glm_ctx sync: sync];
}

void mtlReleaseSync(GLMContext glm_ctx, Sync *sync) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlReleaseSync: glm_ctx sync: sync];
}

#pragma mark - Flush / Swap

void mtlFlushDrawBuffer(GLMContext glm_ctx) {
    if (!glm_ctx || !glm_ctx->mtl_funcs.mtlObj) return;
    @try {
        [(__bridge id<MGLMetalBridgeTarget>)glm_ctx->mtl_funcs.mtlObj flushDrawBuffer:glm_ctx];
    } @catch (NSException *e) {
        NSLog(@"MGL ERROR: mtlFlushDrawBuffer exception: %@", e);
    }
}

void mtlFlush(GLMContext glm_ctx, bool finish) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlFlush:glm_ctx finish:finish];
}

void mtlSwapBuffers(GLMContext glm_ctx) {
    if (!glm_ctx) {
        NSLog(@"MGL CRITICAL: mtlSwapBuffers - GLM context is NULL");
        return;
    }
    if (!glm_ctx->mtl_funcs.mtlObj || ((uintptr_t)glm_ctx->mtl_funcs.mtlObj < 0x1000)) {
        NSLog(@"MGL CRITICAL: mtlSwapBuffers - Invalid Metal object pointer: %p", glm_ctx->mtl_funcs.mtlObj);
        NSLog(@"MGL CRITICAL: This indicates memory corruption or context destruction");
        return;
    }
    @autoreleasepool {
        @try {
            [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlSwapBuffers: glm_ctx];
        } @catch (NSException *exception) {
            NSLog(@"MGL CRITICAL: mtlSwapBuffers - Exception caught: %@", exception);
            NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);
        }
    }
}

#pragma mark - Clear / Blit / Invalidate

void mtlClearBuffer(GLMContext glm_ctx, GLuint type, GLbitfield mask) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlClearBuffer: glm_ctx type: type mask: mask];
}

void mtlBlitFramebuffer(GLMContext glm_ctx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter) {
    if (!glm_ctx || ((uintptr_t)glm_ctx < 0x1000)) {
        fprintf(stderr, "MGL ERROR: mtlBlitFramebuffer bridge received invalid glm_ctx=%p\n", (void*)glm_ctx);
        return;
    }
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlBlitFramebuffer:glm_ctx srcX0:srcX0 srcY0:srcY0 srcX1:srcX1 srcY1:srcY1 dstX0:dstX0 dstY0:dstY0 dstX1:dstX1 dstY1:dstY1 mask:mask filter:filter];
}

void mtlInvalidateRenderPass(GLMContext glm_ctx) {
    if (!glm_ctx || !glm_ctx->mtl_funcs.mtlObj) {
        return;
    }
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlInvalidateRenderPass:glm_ctx];
}

#pragma mark - Buffer

void mtlBufferSubData(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, const void *ptr) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlBufferSubData: glm_ctx buf: buf offset:offset size:size ptr:ptr];
}

void *mtlMapUnmapBuffer(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, GLenum access, bool map) {
    return [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlMapUnmapBuffer: glm_ctx buf: buf offset: offset size: size access: access map: map];
}

void mtlFlushBufferRange(GLMContext glm_ctx, Buffer *buf, GLintptr offset, GLsizeiptr length) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlFlushMappedBufferRange: glm_ctx buf: buf offset: offset length: length];
}

#pragma mark - Readback

void mtlReadDrawable(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlReadDrawable:glm_ctx pixelBytes:pixelBytes bytesPerRow:bytesPerRow bytesPerImage:bytesPerImage fromRegion:MTLRegionMake2D(x,y,width,height)];
}

void mtlReadIntegerPixels(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlReadIntegerPixels:glm_ctx pixelBytes:pixelBytes bytesPerRow:bytesPerRow bytesPerImage:bytesPerImage fromRegion:MTLRegionMake2D(x,y,width,height) format:format type:type];
}

void mtlReadDepthPixels(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlReadDepthPixels: glm_ctx pixelBytes:pixelBytes bytesPerRow:bytesPerRow bytesPerImage:bytesPerImage fromRegion:MTLRegionMake2D(x,y,width,height)];
}

void mtlGetTexImage(GLMContext glm_ctx, Texture *tex, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLuint level, GLuint slice) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlGetTexImage:glm_ctx tex:tex pixelBytes:pixelBytes bytesPerRow:bytesPerRow bytesPerImage:bytesPerImage fromRegion:MTLRegionMake2D(x,y,width,height) format:format type:type mipmapLevel:level slice:slice];
}

#pragma mark - Texture

void mtlCopyTexSubImage(GLMContext glm_ctx, Texture *tex, GLuint slice, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlCopyTexSubImage:glm_ctx tex:tex slice:slice mipmapLevel:(NSUInteger)level xoffset:xoffset yoffset:yoffset x:x y:y width:(NSUInteger)width height:(NSUInteger)height];
}

void mtlGenerateMipmaps(GLMContext glm_ctx, Texture *tex) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlGenerateMipmaps:glm_ctx forTexture:tex];
}

void mtlCopyImageSubData(GLMContext glm_ctx, Texture *srcTex, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, Texture *dstTex, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei width, GLsizei height, GLsizei depth) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlCopyImageSubData:glm_ctx srcTexture:srcTex srcLevel:srcLevel srcX:srcX srcY:srcY srcZ:srcZ dstTexture:dstTex dstLevel:dstLevel dstX:dstX dstY:dstY dstZ:dstZ width:width height:height depth:depth];
}

void mtlTexSubImage(GLMContext glm_ctx, Texture *tex, Buffer *buf, size_t src_offset, size_t src_pitch, size_t src_image_size, size_t src_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlTexSubImage:glm_ctx tex:tex buf:buf src_offset:src_offset src_pitch:src_pitch src_image_size:src_image_size src_size:src_size slice:slice level:level width:width height:height depth:depth xoffset:xoffset yoffset:yoffset zoffset:zoffset];
}

bool mtlTexSubImageBytes(GLMContext glm_ctx, Texture *tex, const void *bytes, size_t bytes_size, size_t src_offset, size_t src_pitch, size_t src_image_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset) {
    return [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlTexSubImageBytes:glm_ctx tex:tex bytes:bytes bytesSize:bytes_size src_offset:src_offset src_pitch:src_pitch src_image_size:src_image_size slice:slice level:level width:width height:height depth:depth xoffset:xoffset yoffset:yoffset zoffset:zoffset];
}

#pragma mark - Compute

void mtlDispatchCompute(GLMContext glm_ctx, GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDispatchCompute: glm_ctx groupsX:num_groups_x groupsY:num_groups_y groupsZ:num_groups_z];
}

void mtlDispatchComputeIndirect(GLMContext glm_ctx, GLintptr indirect) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDispatchComputeIndirect: glm_ctx indirect:indirect];
}

#pragma mark - Query

void mtlBeginTimerQuery(GLMContext glm_ctx) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlBeginTimerQuery: glm_ctx];
}

GLuint64 mtlEndTimerQuery(GLMContext glm_ctx) {
    return [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlEndTimerQuery: glm_ctx];
}

GLuint64 mtlGetGPUTimestamp(GLMContext glm_ctx) {
    return [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlGetGPUTimestamp: glm_ctx];
}

void mtlBeginSampleQuery(GLMContext glm_ctx) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlBeginSampleQuery: glm_ctx];
}

GLuint64 mtlEndSampleQuery(GLMContext glm_ctx) {
    return [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlEndSampleQuery: glm_ctx];
}

#pragma mark - Draw

void mtlDrawArrays(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count) {
    @try {
        if (!glm_ctx || ((uintptr_t)glm_ctx < 0x1000)) {
            NSLog(@"MGL CRITICAL: mtlDrawArrays - Invalid GLM context, aborting operation");
            return;
        }
        if (!glm_ctx->mtl_funcs.mtlObj || ((uintptr_t)glm_ctx->mtl_funcs.mtlObj < 0x1000)) {
            NSLog(@"MGL CRITICAL: mtlDrawArrays - Invalid Metal object, aborting operation");
            return;
        }
        [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawArrays: glm_ctx mode: mode first: first count: count];
    } @catch (NSException *exception) {
        NSLog(@"MGL CRITICAL: mtlDrawArrays - Unhandled exception caught: %@", exception);
        NSLog(@"MGL CRITICAL: Exception reason: %@", [exception reason]);
    } @catch (id exception) {
        NSLog(@"MGL CRITICAL: mtlDrawArrays - Unknown exception caught: %@", exception);
    }
}

void mtlDrawElements(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawElements: glm_ctx mode: mode count: count type: type indices: indices];
}

void mtlDrawRangeElements(GLMContext glm_ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawRangeElements: glm_ctx mode: mode start: start end: end count: count type: type indices: indices];
}

void mtlDrawArraysInstanced(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawArraysInstanced: glm_ctx mode: mode first: first count: count instancecount: instancecount];
}

void mtlDrawElementsInstanced(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsInstanced: glm_ctx mode: mode count: count type: type indices: indices instancecount: instancecount];
}

void mtlDrawElementsBaseVertex(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsBaseVertex: glm_ctx mode: mode count: count type: type indices: indices basevertex: basevertex];
}

void mtlDrawRangeElementsBaseVertex(GLMContext glm_ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawRangeElementsBaseVertex:glm_ctx mode:mode start: start end: end count:count type: type indices: indices basevertex:basevertex];
}

void mtlDrawElementsInstancedBaseVertex(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsInstancedBaseVertex:glm_ctx mode:mode count:count type:type indices:indices instancecount:instancecount basevertex:basevertex];
}

void mtlDrawArraysIndirect(GLMContext glm_ctx, GLenum mode, const void *indirect) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawArraysIndirect:glm_ctx mode:mode indirect:indirect];
}

void mtlDrawElementsIndirect(GLMContext glm_ctx, GLenum mode, GLenum type, const void *indirect) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsIndirect:glm_ctx mode:mode type:type indirect:indirect];
}

void mtlDrawArraysInstancedBaseInstance(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawArraysInstancedBaseInstance:glm_ctx mode:mode first:first count:count instancecount:instancecount baseinstance:baseinstance];
}

void mtlDrawElementsInstancedBaseInstance(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsInstancedBaseInstance:glm_ctx mode:mode count:count type:type indices:indices instancecount:instancecount baseinstance:baseinstance];
}

void mtlDrawElementsInstancedBaseVertexBaseInstance(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlDrawElementsInstancedBaseVertexBaseInstance:glm_ctx mode:mode count:count type:type indices:indices instancecount:instancecount basevertex:basevertex baseinstance:baseinstance];
}

void mtlMultiDrawArrays(GLMContext glm_ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlMultiDrawArrays:glm_ctx mode:mode first:first count:count drawcount:drawcount];
}

void mtlMultiDrawElements(GLMContext glm_ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlMultiDrawElements: glm_ctx mode: mode count: count type: type indices: indices drawcount: drawcount];
}

void mtlMultiDrawElementsBaseVertex(GLMContext glm_ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlMultiDrawElementsBaseVertex: glm_ctx mode: mode count: count type: type indices: indices drawcount: drawcount basevertex:basevertex];
}

void mtlMultiDrawArraysIndirect(GLMContext glm_ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlMultiDrawArraysIndirect:glm_ctx mode:mode indirect:indirect drawcount:drawcount stride:stride];
}

void mtlMultiDrawElementsIndirect(GLMContext glm_ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride) {
    [(__bridge id<MGLMetalBridgeTarget>) glm_ctx->mtl_funcs.mtlObj mtlMultiDrawElementsIndirect:glm_ctx mode:mode type:type indirect:indirect drawcount:drawcount stride:stride];
}
