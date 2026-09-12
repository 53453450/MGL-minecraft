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
#include "mgl_batch_mtl_encode.h"  /* mgl_batch_mtl_create_icb */

void *mglRendererMdiScratchBufferPort(void *renderer, uint64_t length,
                                      uint64_t *offset_out)
{
    /* Body of the former -[MGLRenderer mdiArgumentScratchBufferWithLength:
     * offset:]; the render pass manager owns the ring buffer. */
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    if (!r) {
        return NULL;
    }
    NSUInteger offset = 0u;
    id buffer = (__bridge id)[mglRendererRenderPassManager(r)
        mdiArgumentScratchBufferWithDevice:mglRendererBackendGetDevice(r->_backend)
                                    length:(NSUInteger)length
                                    offset:&offset];
    if (offset_out) {
        *offset_out = (uint64_t)offset;
    }
    return (__bridge void *)buffer;
}

int mglRendererProcessBufferPort(void *renderer, void *buffer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return (r && buffer && [r processBuffer:(Buffer *)buffer]) ? 1 : 0;
}

void *mglRendererCreateIndirectCommandBufferPort(void *renderer, int indexed,
                                                 uint64_t count,
                                                 int *failed_out)
{
    (void)renderer;
    if (failed_out) {
        *failed_out = 0;
    }
    /* The @try/@catch is the reason this one stays ObjC for now: Metal raises
     * when an indirect command buffer cannot be allocated, and that has to
     * become a NULL result the replay path can fall back from. */
    @try {
        return mgl_batch_mtl_create_icb(indexed, count);
    } @catch (NSException *ex) {
        static uint64_t s_hit = 0;
        uint64_t hit = ++s_hit;
        if (hit <= 8ull || (hit % 256ull) == 0ull) {
            NSLog(@"MGL WARNING: ICB creation failed, falling back: %@", ex);
        }
        if (failed_out) {
            *failed_out = 1;
        }
        return NULL;
    }
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

MGLFragmentTextureTraceBinding *mglRendererFragmentTraceBindingsPort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? &r->_resourceFallback.fragmentTextureTraceBindings[0] : NULL;
}

void *mglRendererPipelineStatePort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? r->_pipelineCache.state->pipelineState : NULL;
}

uint32_t mglRendererPipelineProgramNamePort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? (uint32_t)r->_pipelineCache.state->pipelineProgramName : 0u;
}

uint32_t mglRendererRenderPassFramebufferNamePort(void *renderer)
{
    MGLRenderer *r = (__bridge MGLRenderer *)renderer;
    return r ? (uint32_t)mglRendererRenderPassManager(r).state->renderPassFramebufferName : 0u;
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
