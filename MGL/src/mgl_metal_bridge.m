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
 * C callback adapter layer extracted from MGLRenderer.m.
 *
 * Each function is a plain C entry point (matching the function pointer
 * signature in `struct GLMMetalFuncs`) that constructs C ABI value-state and
 * enters the opaque callback runtime.  No renderer selector is sent here.
 */

#include "glm_context.h"
#include "mgl_metal_bridge.h"
#include "mgl_render_cpp.h"

int mglMetalBridgeGetCallbackCensus(
    GLMContext glm_ctx,
    MGLMetalCallbackCensus *census_out)
{
    if (!glm_ctx || !census_out) return -1;

    MGLMetalCallbackCensus census = {0};
#define MGL_MTL_FUNC_CENSUS(field, cname, ret, args) \
    do { \
        ++census.total; \
        if (!glm_ctx->mtl_funcs.field) { \
            ++census.null_entries; \
        } else if (glm_ctx->mtl_funcs.field == cname) { \
            ++census.legacy; \
        } else { \
            ++census.non_legacy; \
        } \
    } while (0);
    MGL_MTL_FUNC_LIST(MGL_MTL_FUNC_CENSUS)
#undef MGL_MTL_FUNC_CENSUS

    *census_out = census;
    return census.total == MGL_MTL_FUNC_COUNT ? 0 : -1;
}

#pragma mark - Pure C callback adapters

static uint64_t mglInvokeLegacy(GLMContext ctx,
                                MGLRenderCppLegacyCallbackArgs args)
{
    return mglRenderCppInvokeLegacyCallback(ctx, &args);
}

void mtlBindBuffer(GLMContext c, Buffer *v) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_BIND_BUFFER,.buffer=v}); }
void mtlBindTexture(GLMContext c, Texture *v) { mglRenderCppInvokeBindTextureCallback(c, v); }
void mtlBindProgram(GLMContext c, Program *v) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_BIND_PROGRAM,.program=v}); }
void mtlDeleteMTLObj(GLMContext c, void *v) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_DELETE_OBJECT,.object=v}); }
void mtlReleaseBufferMetalData(GLMContext c, Buffer *v) { if (v && v->data.mtl_data) { mtlDeleteMTLObj(c, v->data.mtl_data); v->data.mtl_data = NULL; } }
void mtlGetSync(GLMContext c, Sync *v) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_GET_SYNC,.sync=v}); }
void mtlWaitForSync(GLMContext c, Sync *v) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_WAIT_SYNC,.sync=v}); }
GLenum mtlGetSyncStatus(GLMContext c, Sync *v) { return (GLenum)mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_GET_SYNC_STATUS,.sync=v}); }
void mtlReleaseSync(GLMContext c, Sync *v) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_RELEASE_SYNC,.sync=v}); }
void mtlFlushDrawBuffer(GLMContext c) { mglRenderCppInvokeFlushDrawBufferCallback(c); }
void mtlFlush(GLMContext c, bool v) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_FLUSH,.flag=v}); }
void mtlSwapBuffers(GLMContext c) { mglRenderCppInvokeSwapBuffersCallback(c); }
void mtlClearBuffer(GLMContext c, GLuint t, GLbitfield m) { mglRenderCppInvokeClearBufferCallback(c, t, m); }
void mtlBlitFramebuffer(GLMContext c, GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLbitfield m, GLenum f) { mglRenderCppInvokeBlitFramebufferCallback(c, sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, m, f); }
void mtlInvalidateRenderPass(GLMContext c) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_INVALIDATE_RENDER_PASS}); }
void mtlBufferSubData(GLMContext c, Buffer *b, size_t o, size_t s, const void *p) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_BUFFER_SUB_DATA,.buffer=b,.bytes=p,.offset=o,.size=s}); }
void *mtlMapUnmapBuffer(GLMContext c, Buffer *b, size_t o, size_t s, GLenum a, bool m) { return (void *)(uintptr_t)mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_MAP_UNMAP_BUFFER,.buffer=b,.offset=o,.size=s,.value=a,.flag=m}); }
void mtlReadBackBuffer(GLMContext c, Buffer *b, size_t o, size_t s) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_READ_BACK_BUFFER,.buffer=b,.offset=o,.size=s}); }
void mtlFlushBufferRange(GLMContext c, Buffer *b, GLintptr o, GLsizeiptr l) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_FLUSH_BUFFER_RANGE,.buffer=b,.signed_offset=o,.signed_length=l}); }

static int mglInvokeResource(GLMContext c, MGLRenderCppResourceCallbackArgs a) { return mglRenderCppInvokeResourceCallback(c, &a); }
void mtlReadDrawable(GLMContext c, void *p, GLuint r, GLuint i, GLint x, GLint y, GLsizei w, GLsizei h) { mglInvokeResource(c, (MGLRenderCppResourceCallbackArgs){.kind=MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_DRAWABLE,.pixel_bytes=p,.width=(size_t)w,.height=(size_t)h,.bytes_per_row=r,.bytes_per_image=i,.x=x,.y=y}); }
void mtlReadIntegerPixels(GLMContext c, void *p, GLuint r, GLuint i, GLint x, GLint y, GLsizei w, GLsizei h, GLenum f, GLenum t) { mglInvokeResource(c, (MGLRenderCppResourceCallbackArgs){.kind=MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_INTEGER_PIXELS,.pixel_bytes=p,.width=(size_t)w,.height=(size_t)h,.bytes_per_row=r,.bytes_per_image=i,.format=f,.type=t,.x=x,.y=y}); }
void mtlReadDepthPixels(GLMContext c, void *p, GLuint r, GLuint i, GLint x, GLint y, GLsizei w, GLsizei h) { mglInvokeResource(c, (MGLRenderCppResourceCallbackArgs){.kind=MGL_RENDER_CPP_RESOURCE_CALLBACK_READ_DEPTH_PIXELS,.pixel_bytes=p,.width=(size_t)w,.height=(size_t)h,.bytes_per_row=r,.bytes_per_image=i,.x=x,.y=y}); }
void mtlGetTexImage(GLMContext c, Texture *v, void *p, GLuint r, GLuint i, GLint x, GLint y, GLsizei w, GLsizei h, GLenum f, GLenum t, GLuint l, GLuint s) { mglInvokeResource(c, (MGLRenderCppResourceCallbackArgs){.kind=MGL_RENDER_CPP_RESOURCE_CALLBACK_GET_TEX_IMAGE,.texture=v,.pixel_bytes=p,.width=(size_t)w,.height=(size_t)h,.bytes_per_row=r,.bytes_per_image=i,.format=f,.type=t,.slice=s,.level=l,.x=x,.y=y}); }
void mtlGenerateMipmaps(GLMContext c, Texture *v) { mglInvokeResource(c, (MGLRenderCppResourceCallbackArgs){.kind=MGL_RENDER_CPP_RESOURCE_CALLBACK_GENERATE_MIPMAPS,.texture=v}); }
void mtlTexSubImage(GLMContext c, Texture *v, Buffer *b, size_t o, size_t p, size_t i, size_t s, GLuint sl, GLuint l, size_t w, size_t h, size_t d, size_t x, size_t y, size_t z) { mglInvokeResource(c, (MGLRenderCppResourceCallbackArgs){.kind=MGL_RENDER_CPP_RESOURCE_CALLBACK_TEX_SUB_IMAGE,.texture=v,.buffer=b,.source_offset=o,.source_pitch=p,.source_image_size=i,.source_size=s,.width=w,.height=h,.depth=d,.x_offset=x,.y_offset=y,.z_offset=z,.slice=sl,.level=l}); }
bool mtlTexSubImageBytes(GLMContext c, Texture *v, const void *b, size_t bs, size_t o, size_t p, size_t i, GLuint sl, GLuint l, size_t w, size_t h, size_t d, size_t x, size_t y, size_t z) { return mglInvokeResource(c, (MGLRenderCppResourceCallbackArgs){.kind=MGL_RENDER_CPP_RESOURCE_CALLBACK_TEX_SUB_IMAGE_BYTES,.texture=v,.bytes=b,.bytes_size=bs,.source_offset=o,.source_pitch=p,.source_image_size=i,.width=w,.height=h,.depth=d,.x_offset=x,.y_offset=y,.z_offset=z,.slice=sl,.level=l}) != 0; }
void mtlCopyTexSubImage(GLMContext c, Texture *v, GLuint s, GLint l, GLint xo, GLint yo, GLint x, GLint y, GLsizei w, GLsizei h) { mglInvokeResource(c, (MGLRenderCppResourceCallbackArgs){.kind=MGL_RENDER_CPP_RESOURCE_CALLBACK_COPY_TEX_SUB_IMAGE,.texture=v,.width=(size_t)w,.height=(size_t)h,.x_offset=(size_t)xo,.y_offset=(size_t)yo,.slice=s,.level=(uint32_t)l,.x=x,.y=y}); }
void mtlCopyImageSubData(GLMContext c, Texture *s, GLint sl, GLint sx, GLint sy, GLint sz, Texture *d, GLint dl, GLint dx, GLint dy, GLint dz, GLsizei w, GLsizei h, GLsizei dp) { mglInvokeResource(c, (MGLRenderCppResourceCallbackArgs){.kind=MGL_RENDER_CPP_RESOURCE_CALLBACK_COPY_IMAGE_SUB_DATA,.source_texture=s,.destination_texture=d,.width=(size_t)w,.height=(size_t)h,.depth=(size_t)dp,.source_level=sl,.source_x=sx,.source_y=sy,.source_z=sz,.destination_level=dl,.destination_x=dx,.destination_y=dy,.destination_z=dz}); }

void mtlDispatchCompute(GLMContext c, GLuint x, GLuint y, GLuint z) { mglRenderCppInvokeComputeCallback(c, x, y, z); }
void mtlDispatchComputeIndirect(GLMContext c, GLintptr i) { mglRenderCppInvokeComputeIndirectCallback(c, i); }
void mtlBeginTimerQuery(GLMContext c) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_BEGIN_TIMER_QUERY}); }
GLuint64 mtlEndTimerQuery(GLMContext c) { return mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_END_TIMER_QUERY}); }
GLuint64 mtlGetGPUTimestamp(GLMContext c) { return mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_GET_GPU_TIMESTAMP}); }
void mtlBeginSampleQuery(GLMContext c, GLenum t) { mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_BEGIN_SAMPLE_QUERY,.value=t}); }
GLuint64 mtlEndSampleQuery(GLMContext c) { return mglInvokeLegacy(c, (MGLRenderCppLegacyCallbackArgs){.kind=MGL_RENDER_CPP_LEGACY_CALLBACK_END_SAMPLE_QUERY}); }

static void mglInvokeDraw(GLMContext c, MGLRenderCppDrawCallbackArgs a) { mglRenderCppInvokeDrawCallback(c, &a); }
void mtlDrawArrays(GLMContext c, GLenum m, GLint f, GLsizei n) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS,.mode=m,.first=f,.count=n}); }
void mtlDrawElements(GLMContext c, GLenum m, GLsizei n, GLenum t, const void *i) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS,.mode=m,.type=t,.count=n,.indices_or_indirect=i}); }
void mtlDrawRangeElements(GLMContext c, GLenum m, GLuint s, GLuint e, GLsizei n, GLenum t, const void *i) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_RANGE_ELEMENTS,.mode=m,.type=t,.start=s,.end=e,.count=n,.indices_or_indirect=i}); }
void mtlDrawArraysInstanced(GLMContext c, GLenum m, GLint f, GLsizei n, GLsizei ic) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS_INSTANCED,.mode=m,.first=f,.count=n,.instance_count=ic}); }
void mtlDrawElementsInstanced(GLMContext c, GLenum m, GLsizei n, GLenum t, const void *i, GLsizei ic) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED,.mode=m,.type=t,.count=n,.instance_count=ic,.indices_or_indirect=i}); }
void mtlDrawElementsBaseVertex(GLMContext c, GLenum m, GLsizei n, GLenum t, const void *i, GLint b) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_BASE_VERTEX,.mode=m,.type=t,.count=n,.base_vertex=b,.indices_or_indirect=i}); }
void mtlDrawRangeElementsBaseVertex(GLMContext c, GLenum m, GLuint s, GLuint e, GLsizei n, GLenum t, const void *i, GLint b) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_RANGE_ELEMENTS_BASE_VERTEX,.mode=m,.type=t,.start=s,.end=e,.count=n,.base_vertex=b,.indices_or_indirect=i}); }
void mtlDrawElementsInstancedBaseVertex(GLMContext c, GLenum m, GLsizei n, GLenum t, const void *i, GLsizei ic, GLint b) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED_BASE_VERTEX,.mode=m,.type=t,.count=n,.instance_count=ic,.base_vertex=b,.indices_or_indirect=i}); }
void mtlDrawArraysIndirect(GLMContext c, GLenum m, const void *i) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS_INDIRECT,.mode=m,.indices_or_indirect=i}); }
void mtlDrawElementsIndirect(GLMContext c, GLenum m, GLenum t, const void *i) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INDIRECT,.mode=m,.type=t,.indices_or_indirect=i}); }
void mtlDrawArraysInstancedBaseInstance(GLMContext c, GLenum m, GLint f, GLsizei n, GLsizei ic, GLuint bi) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ARRAYS_INSTANCED_BASE_INSTANCE,.mode=m,.first=f,.count=n,.instance_count=ic,.base_instance=bi}); }
void mtlDrawElementsInstancedBaseInstance(GLMContext c, GLenum m, GLsizei n, GLenum t, const void *i, GLsizei ic, GLuint bi) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED_BASE_INSTANCE,.mode=m,.type=t,.count=n,.instance_count=ic,.base_instance=bi,.indices_or_indirect=i}); }
void mtlDrawElementsInstancedBaseVertexBaseInstance(GLMContext c, GLenum m, GLsizei n, GLenum t, const void *i, GLsizei ic, GLint bv, GLuint bi) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE,.mode=m,.type=t,.count=n,.instance_count=ic,.base_vertex=bv,.base_instance=bi,.indices_or_indirect=i}); }
void mtlMultiDrawArrays(GLMContext c, GLenum m, const GLint *f, const GLsizei *n, GLsizei dc) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ARRAYS,.mode=m,.draw_count=dc,.firsts=f,.counts=n}); }
void mtlMultiDrawElements(GLMContext c, GLenum m, const GLsizei *n, GLenum t, const void *const*i, GLsizei dc) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ELEMENTS,.mode=m,.type=t,.draw_count=dc,.indices_or_indirect=i,.counts=n}); }
void mtlMultiDrawElementsBaseVertex(GLMContext c, GLenum m, const GLsizei *n, GLenum t, const void *const*i, GLsizei dc, const GLint *b) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ELEMENTS_BASE_VERTEX,.mode=m,.type=t,.draw_count=dc,.indices_or_indirect=i,.counts=n,.base_vertices=b}); }
void mtlMultiDrawArraysIndirect(GLMContext c, GLenum m, const void *i, GLsizei dc, GLsizei s) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ARRAYS_INDIRECT,.mode=m,.draw_count=dc,.stride=s,.indices_or_indirect=i}); }
void mtlMultiDrawElementsIndirect(GLMContext c, GLenum m, GLenum t, const void *i, GLsizei dc, GLsizei s) { mglInvokeDraw(c, (MGLRenderCppDrawCallbackArgs){.kind=MGL_RENDER_CPP_DRAW_CALLBACK_MULTI_ELEMENTS_INDIRECT,.mode=m,.type=t,.draw_count=dc,.stride=s,.indices_or_indirect=i}); }
