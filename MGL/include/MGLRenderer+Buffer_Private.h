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
 * MGLRenderer+Buffer_Private.h
 * MGL
 *
 * Private method declarations and C helpers for the Buffer category
 * (MGLRenderer+Buffer.m).  Imports MGLRenderer.h for the MGLRenderer interface;
 * the category file itself imports MGLRenderer_Private.h for ivar access
 * and shared types.
 */

#ifndef MGLRenderer_Buffer_Private_h
#define MGLRenderer_Buffer_Private_h

#import "MGLRenderer.h"

/* Trace sampling helper for buffer-transfer call counters.  Defined here as
 * static inline so both MGLRenderer.m (mtlBufferSubData path) and
 * MGLRenderer+Buffer.m can use it across translation units. */
static inline bool mglShouldTraceBufferTransferCall(uint64_t call)
{
    if (call <= 128ull) {
        return true;
    }
    return ((call % 64ull) == 0ull);
}

/* Snapshot helpers defined in MGLRenderer+Buffer.m.  Non-static because
 * MGLRenderer.m (swap-diagnostics path) also calls them across translation
 * units. */
BOOL mglSnapshotSharedDirtyBuffer(id<MTLDevice> device,
                                  Buffer *ptr,
                                  id<MTLBuffer> *bufferPtr);
BOOL mglSnapshotSharedBufferRange(id<MTLDevice> device,
                                  Buffer *ptr,
                                  id<MTLBuffer> *bufferPtr,
                                  NSUInteger offset,
                                  NSUInteger length);

@interface MGLRenderer (Buffer)

/* mapGLBuffersToMTLBufferMap:stage: helpers */
- (bool)mapShaderBufferResourcesToBufferMap:(BufferMapList *)buffer_map stage:(int)stage;
- (bool)mapVertexAttributeBuffersToBufferMap:(BufferMapList *)buffer_map
                                         vao:(VertexArray *)vao
                            stageInputCount:(int)count
                                       stage:(int)stage;

/* Public entry points (called from MGLRenderer.m, +Compute.m, +RenderPass.m) */
- (bool)mapGLBuffersToMTLBufferMap:(BufferMapList *)buffer_map stage:(int)stage;
- (bool)mapBuffersToMTL;
- (bool)updateDirtyBuffer:(Buffer *)ptr;
- (bool)checkForDirtyBufferData:(BufferMapList *)buffer_map_list;
- (bool)updateDirtyBaseBufferList:(BufferMapList *)buffer_map_list;
- (int)getVertexBufferIndexWithAttributeSet:(int)attribute;

/* Vertex attribute conversion helpers (called from MGLRenderer+Draw.m) */
- (id<MTLBuffer>)floatVertexBufferForDoubleAttrib:(Buffer *)sourceBuffer
                                         resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                             size:(GLuint)componentCount
                                         outStride:(NSUInteger *)outStride;
- (id<MTLBuffer>)floatVertexBufferForIntAttrib:(Buffer *)sourceBuffer
                                      resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                          size:(GLuint)componentCount
                                    normalized:(GLboolean)normalized
                                          type:(GLenum)type
                                     outStride:(NSUInteger *)outStride;
- (id<MTLBuffer>)integerVertexBufferForAttrib:(Buffer *)sourceBuffer
                                     resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                         size:(GLuint)componentCount
                                       srcType:(GLenum)srcType
                                     dstIsInt:(BOOL)dstIsInt
                                    outStride:(NSUInteger *)outStride;

/* GL_FIXED: each component is a 32-bit signed integer representing a 16.16
 * fixed-point value (actual value = raw / 65536.0). size ranges 1-4; each
 * component is converted independently to float. Output is float[size]. */
- (id<MTLBuffer>)floatVertexBufferForFixedAttrib:(Buffer *)sourceBuffer
                                         resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                             size:(GLuint)componentCount
                                        outStride:(NSUInteger *)outStride;

/* GL_UNSIGNED_INT_10_10_10_2: 1 uint32 packed as RGBA.
 * Non-REV bit layout: R[22-31] G[12-21] B[2-11] A[0-1].
 * Converted to float4(R/1023.0, G/1023.0, B/1023.0, A/3.0). */
- (id<MTLBuffer>)floatVertexBufferForPacked1010102Attrib:(Buffer *)sourceBuffer
                                                  resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                                 outStride:(NSUInteger *)outStride;

/* GL_UNSIGNED_INT_10F_11F_11F_REV: 1 uint32 packed as RGB float.
 * REV bit layout: R[0-10] G[11-21] B[22-31].
 * R/G are 11-bit float, B is 10-bit float (unsigned). Converted to float3. */
- (id<MTLBuffer>)floatVertexBufferForPacked10f11f11fAttrib:(Buffer *)sourceBuffer
                                                     resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                                    outStride:(NSUInteger *)outStride;

@end

#endif /* MGLRenderer_Buffer_Private_h */
