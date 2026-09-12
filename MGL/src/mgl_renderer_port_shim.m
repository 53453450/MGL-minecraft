/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_renderer_port_shim.m — the ObjC side of the C port surface (T4).
 *
 * Every function here is a three-line wrapper: it casts the void *renderer
 * handle back to MGLRenderer and calls the entry point the batch path needs.
 * They exist so C translation units (mgl_batch_issue_encode.c today, its
 * neighbours next) do not have to be Objective-C to drive the renderer, and so
 * that the remaining ObjC surface lives in ONE file instead of being spread
 * over the TUs that will themselves be converted.
 *
 * When an implementation file is converted, its wrapper moves into it (as
 * mglRendererAttachmentTextureFor did) and this shim shrinks -- the goal is for
 * it to disappear, or to be the single platform-shell TU the zero-ObjC target
 * allows.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#import "MGLRenderer+BatchPorts_Private.h"
#import "MGLRenderer+RenderPass_Private.h"
#include "mgl_renderer_ports.h"
#include "mgl_renderer_backend.h"

void *mglRendererMdiScratchBufferPort(void *renderer, uint64_t length,
                                      uint64_t *offset_out)
{
    NSUInteger offset = 0u;
    id buffer = [(__bridge MGLRenderer *)renderer
        mdiArgumentScratchBufferWithLength:(NSUInteger)length
                                    offset:&offset];
    if (offset_out) {
        *offset_out = (uint64_t)offset;
    }
    return (__bridge void *)buffer;
}

int mglRendererResolveElementBufferPort(void *renderer, const void *command,
                                        const char *label, GLMContext ctx,
                                        Buffer **gl_buffer_out,
                                        void **mtl_buffer_out)
{
    Buffer *gl_buffer = NULL;
    id mtl_buffer = nil;
    if (![(__bridge MGLRenderer *)renderer
            resolveElementBufferForCommand:(const MGLDrawCommand *)command
                                    label:label
                                  context:ctx
                                 glBuffer:&gl_buffer
                                mtlBuffer:&mtl_buffer]) {
        return 0;
    }
    if (gl_buffer_out) {
        *gl_buffer_out = gl_buffer;
    }
    if (mtl_buffer_out) {
        *mtl_buffer_out = (__bridge void *)mtl_buffer;
    }
    return 1;
}

void mglRendererTraceReplayCommandPort(void *renderer, void *batch,
                                       void *command, GLMContext ctx,
                                       uint64_t flush_id, uint32_t batch_index,
                                       uint32_t command_index,
                                       const char *phase, const char *reason)
{
    [(__bridge MGLRenderer *)renderer traceReplayCommand:(MGLDrawBatch *)batch
                                                 command:(MGLDrawCommand *)command
                                                 context:ctx
                                                 flushId:flush_id
                                              batchIndex:batch_index
                                            commandIndex:command_index
                                                   phase:phase
                                                  reason:reason];
}

int mglRendererTryReplaySimpleBatchPort(void *renderer, void *batch,
                                        GLMContext ctx,
                                        const void *encode_context)
{
    return [(__bridge MGLRenderer *)renderer
               tryReplaySimpleBatch:(MGLDrawBatch *)batch
                            context:ctx
                      encodeContext:(const MGLEncodeContext *)encode_context]
               ? 1
               : 0;
}

int mglRendererApplyDynamicBindingsPort(void *renderer, const void *command,
                                        GLMContext ctx, void *encode_context)
{
    return [(__bridge MGLRenderer *)renderer
               applyDynamicBindingsForCommand:(const MGLDrawCommand *)command
                                      context:ctx
                                encodeContext:(MGLEncodeContext *)encode_context]
               ? 1
               : 0;
}

int mglRendererApplySamplerSnapshotPort(void *renderer, const void *command,
                                        GLMContext ctx,
                                        const void *encode_context)
{
    return [(__bridge MGLRenderer *)renderer
               applySamplerSnapshotForCommand:(const MGLDrawCommand *)command
                                      context:ctx
                                encodeContext:(const MGLEncodeContext *)encode_context]
               ? 1
               : 0;
}

int mglRendererCaptureCullArrayPort(void *renderer, GLMContext ctx, int32_t first,
                                    int32_t count, int32_t instance_count,
                                    uint32_t base_instance)
{
    return [(__bridge MGLRenderer *)renderer
               captureAIRCullDistancesForArrayDraw:ctx
                                             first:(GLint)first
                                             count:(GLsizei)count
                                     instanceCount:(GLsizei)instance_count
                                      baseInstance:(GLuint)base_instance]
               ? 1
               : 0;
}

int mglRendererCaptureCullElementPort(void *renderer, GLMContext ctx,
                                      const uint8_t *index_bytes,
                                      uint32_t index_type, int32_t count,
                                      int32_t base_vertex,
                                      int32_t instance_count,
                                      uint32_t base_instance)
{
    return [(__bridge MGLRenderer *)renderer
               captureAIRCullDistancesForElementDraw:ctx
                                          indexBytes:index_bytes
                                           indexType:(GLenum)index_type
                                               count:(GLsizei)count
                                          baseVertex:(GLint)base_vertex
                                       instanceCount:(GLsizei)instance_count
                                        baseInstance:(GLuint)base_instance]
               ? 1
               : 0;
}

int mglRendererProcessGLStatePort(void *renderer, int draw_command)
{
    return [(__bridge MGLRenderer *)renderer processGLState:draw_command ? true : false]
               ? 1
               : 0;
}

uint64_t mglRendererBatchTraceFlushIdPort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return mglRendererRenderPassManager(r).state->traceReplayFlushId;
}

uint32_t mglRendererBatchTraceBatchIndexPort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return mglRendererRenderPassManager(r).state->traceReplayBatchIndex;
}

void *mglRendererCurrentRenderEncoderOwnerPort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return mglRendererRenderPassManager(r).state->currentRenderEncoderOwner;
}
