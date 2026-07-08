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
 * mgl_metal_bridge.h
 * MGL
 *
 * C bridge functions extracted from MGLRenderer.m.
 *
 * These are pure C entry points that forward into MGLRenderer's Objective-C
 * methods via the bridged Metal object stored in
 * glm_ctx->mtl_funcs.mtlObj (accessed as `(__bridge id) mtlObj`).
 *
 * The function signatures match the function pointer fields declared in
 * `struct GLMMetalFuncs` (glm_context.h) and are assigned into that struct
 * so the rest of MGL can invoke Metal work without depending on ObjC.
 */

#ifndef MGL_METAL_BRIDGE_H
#define MGL_METAL_BRIDGE_H

#include "GL/glcorearb.h"

#include <stddef.h>
#include <stdbool.h>

/* GLMContext, Buffer, Texture, Program and Sync are defined here.  Including
 * glm_context.h also transitively pulls in glcorearb.h via glm_dispatch.h. */
#include "glm_context.h"

/* Fix #14: Move @protocol MGLMetalBridgeTarget into the header so that
 * MGLRenderer can formally conform to it.  The protocol is the single source
 * of truth for the Objective-C selectors invoked through the bridge.
 * Fix #3: Method signatures use the canonical GL types (GLintptr, GLsizeiptr,
 * GLsizei, GLbitfield, ...) so the bridge ABI matches the C entry points. */
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MGLMetalBridgeTarget
- (void)bindMTLBuffer:(Buffer *)ptr;
- (bool)bindMTLTexture:(Texture *)ptr;
- (bool)bindMTLProgram:(Program *)ptr;
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
#endif /* __OBJC__ */

#ifdef __cplusplus
extern "C" {
#endif

/* Bind / Delete */
void mtlBindBuffer(GLMContext glm_ctx, Buffer *ptr);
void mtlBindTexture(GLMContext glm_ctx, Texture *ptr);
void mtlBindProgram(GLMContext glm_ctx, Program *ptr);
void mtlDeleteMTLObj(GLMContext glm_ctx, void *obj);

/* Sync */
void mtlGetSync(GLMContext glm_ctx, Sync *sync);
void mtlWaitForSync(GLMContext glm_ctx, Sync *sync);
GLenum mtlGetSyncStatus(GLMContext glm_ctx, Sync *sync);
void mtlReleaseSync(GLMContext glm_ctx, Sync *sync);

/* Flush / Swap */
void mtlFlushDrawBuffer(GLMContext glm_ctx);
void mtlFlush(GLMContext glm_ctx, bool finish);
void mtlSwapBuffers(GLMContext glm_ctx);

/* Clear / Blit / Invalidate */
void mtlClearBuffer(GLMContext glm_ctx, GLuint type, GLbitfield mask);
void mtlBlitFramebuffer(GLMContext glm_ctx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);
void mtlInvalidateRenderPass(GLMContext glm_ctx);

/* Buffer */
void mtlBufferSubData(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, const void *ptr);
void *mtlMapUnmapBuffer(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, GLenum access, bool map);
void mtlFlushBufferRange(GLMContext glm_ctx, Buffer *buf, GLintptr offset, GLsizeiptr length);

/* Readback */
void mtlReadDrawable(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height);
void mtlReadIntegerPixels(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type);
void mtlReadDepthPixels(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height);
void mtlGetTexImage(GLMContext glm_ctx, Texture *tex, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLuint level, GLuint slice);

/* Texture */
void mtlCopyTexSubImage(GLMContext glm_ctx, Texture *tex, GLuint slice, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
void mtlGenerateMipmaps(GLMContext glm_ctx, Texture *tex);
void mtlCopyImageSubData(GLMContext glm_ctx, Texture *srcTex, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, Texture *dstTex, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei width, GLsizei height, GLsizei depth);
void mtlTexSubImage(GLMContext glm_ctx, Texture *tex, Buffer *buf, size_t src_offset, size_t src_pitch, size_t src_image_size, size_t src_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset);
bool mtlTexSubImageBytes(GLMContext glm_ctx, Texture *tex, const void *bytes, size_t bytes_size, size_t src_offset, size_t src_pitch, size_t src_image_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset);

/* Compute */
void mtlDispatchCompute(GLMContext glm_ctx, GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
void mtlDispatchComputeIndirect(GLMContext glm_ctx, GLintptr indirect);

/* Query */
void mtlBeginTimerQuery(GLMContext glm_ctx);
GLuint64 mtlEndTimerQuery(GLMContext glm_ctx);
GLuint64 mtlGetGPUTimestamp(GLMContext glm_ctx);
void mtlBeginSampleQuery(GLMContext glm_ctx);
GLuint64 mtlEndSampleQuery(GLMContext glm_ctx);

/* Draw */
void mtlDrawArrays(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count);
void mtlDrawElements(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices);
void mtlDrawRangeElements(GLMContext glm_ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices);
void mtlDrawArraysInstanced(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
void mtlDrawElementsInstanced(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount);
void mtlDrawElementsBaseVertex(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex);
void mtlDrawRangeElementsBaseVertex(GLMContext glm_ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex);
void mtlDrawElementsInstancedBaseVertex(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex);
void mtlDrawArraysIndirect(GLMContext glm_ctx, GLenum mode, const void *indirect);
void mtlDrawElementsIndirect(GLMContext glm_ctx, GLenum mode, GLenum type, const void *indirect);
void mtlDrawArraysInstancedBaseInstance(GLMContext glm_ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance);
void mtlDrawElementsInstancedBaseInstance(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance);
void mtlDrawElementsInstancedBaseVertexBaseInstance(GLMContext glm_ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance);
void mtlMultiDrawArrays(GLMContext glm_ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount);
void mtlMultiDrawElements(GLMContext glm_ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount);
void mtlMultiDrawElementsBaseVertex(GLMContext glm_ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex);
void mtlMultiDrawArraysIndirect(GLMContext glm_ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride);
void mtlMultiDrawElementsIndirect(GLMContext glm_ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride);

#ifdef __cplusplus
}
#endif

#endif /* MGL_METAL_BRIDGE_H */
