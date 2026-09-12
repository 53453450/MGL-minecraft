/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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
BOOL mglSnapshotSharedDirtyBuffer(Buffer *ptr, id *bufferPtr);
BOOL mglSnapshotSharedBufferRange(Buffer *ptr,
                                  id *bufferPtr,
                                  NSUInteger offset,
                                  NSUInteger length);

/* Copy-on-write snapshot pool: frame-generation gates for reusing snapshot
 * MTLBuffers after the GPU has finished reading them.  All pool entry points
 * run under METAL_LOCK; only mglRecordFrameCompleted runs on the Metal
 * completion thread. */
uint64_t mglCompletedFrameGeneration(void);
uint64_t mglAdvanceFrameGeneration(void);
void mglRecordFrameCompleted(uint64_t generation);
void mglNoteBufferEncoded(Buffer *buf);

@interface MGLRenderer (Buffer)

/* mapGLBuffersToMTLBufferMap:stage: helpers */
- (bool)mapShaderBufferResourcesToBufferMap:(BufferMapList *)buffer_map stage:(int)stage;
/* Attribute-buffer mapping is planned in C
 * (mglRenderPlanVertexAttribBuffers, mgl_buffer_plan.h); the candidate mask is
 * gathered in mapGLBuffersToMTLBufferMap:stage:. */

/* Public entry points (called from MGLRenderer.m, +Compute.m, +RenderPass.m) */
- (bool)mapGLBuffersToMTLBufferMap:(BufferMapList *)buffer_map stage:(int)stage;
- (bool)mapBuffersToMTL;
- (bool)updateDirtyBuffer:(Buffer *)ptr;
- (bool)checkForDirtyBufferData:(BufferMapList *)buffer_map_list;
- (bool)updateDirtyBaseBufferList:(BufferMapList *)buffer_map_list;
- (int)getVertexBufferIndexWithAttributeSet:(int)attribute;

/* Vertex attribute conversion (called from BindingState). */
- (id)convertedVertexBufferForAttribKind:(int)attribKind
                                  source:(Buffer *)sourceBuffer
                                resolved:(const MGLResolvedVertexAttribBinding *)resolved
                                    size:(GLuint)componentCount
                                    type:(GLenum)type
                              normalized:(GLboolean)normalized
                               dstIsInt:(BOOL)dstIsInt
                               outStride:(NSUInteger *)outStride;

@end

#endif /* MGLRenderer_Buffer_Private_h */
