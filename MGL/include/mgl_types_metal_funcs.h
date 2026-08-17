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
 * mgl_types_metal_funcs.h
 * MGL
 *
 * GLMMetalFuncs function-pointer table split from glm_context.h.
 * Decouple the Metal backend callback table from the god header.
 * The struct only uses GLMContext (opaque pointer), Buffer*, Texture*,
 * Program*, and Sync* — all available via the mgl_types_*.h headers —
 * so it can be safely separated.
 */

#ifndef mgl_types_metal_funcs_h
#define mgl_types_metal_funcs_h

#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_program.h"
#include "mgl_types_sync.h"

/* Forward-declared so this header is self-contained.  glm_context.h
 * provides the same typedef; C allows the redundant declaration. */
typedef struct GLMContextRec_t *GLMContext;

struct GLMMetalFuncs {
    void *mtlObj;
    void *mtlView;
/* ==== Single source of truth for the MGL Metal function-pointer table ====
 * MGL_MTL_FUNC_LIST defines every callback once.  It generates both the
 * struct fields below and the assignment block in
 * MGLRenderer+Lifecycle.m (bindObjFuncsToGLMContext).  Each entry is
 * (field name, C bridge function name, return type, argument list).  The
 * two names differ only for release_buffer_metal_data.  Adding a new
 * callback touches this list plus the bridge declaration/definition and
 * the ObjC method. */
#define MGL_MTL_FUNC_LIST(M) \
    M(mtlBindBuffer, mtlBindBuffer, void, (GLMContext glm_ctx, Buffer *ptr)) \
    M(mtlBindTexture, mtlBindTexture, void, (GLMContext glm_ctx, Texture *ptr)) \
    M(mtlBindProgram, mtlBindProgram, void, (GLMContext glm_ctx, Program *ptr)) \
    M(mtlDeleteMTLObj, mtlDeleteMTLObj, void, (GLMContext glm_ctx, void *obj)) \
    M(release_buffer_metal_data, mtlReleaseBufferMetalData, void, (GLMContext glm_ctx, Buffer *buffer)) \
    M(mtlGetSync, mtlGetSync, void, (GLMContext glm_ctx, Sync *sync)) \
    M(mtlWaitForSync, mtlWaitForSync, void, (GLMContext glm_ctx, Sync *sync)) \
    M(mtlGetSyncStatus, mtlGetSyncStatus, GLenum, (GLMContext glm_ctx, Sync *sync)) \
    M(mtlReleaseSync, mtlReleaseSync, void, (GLMContext glm_ctx, Sync *sync)) \
    M(mtlFlush, mtlFlush, void, (GLMContext glm_ctx, bool finish)) \
    M(mtlSwapBuffers, mtlSwapBuffers, void, (GLMContext glm_ctx)) \
    M(mtlFlushDrawBuffer, mtlFlushDrawBuffer, void, (GLMContext glm_ctx)) \
    M(mtlInvalidateRenderPass, mtlInvalidateRenderPass, void, (GLMContext glm_ctx)) \
    M(mtlClearBuffer, mtlClearBuffer, void, (GLMContext glm_ctx, GLuint type, GLbitfield mask)) \
    M(mtlBlitFramebuffer, mtlBlitFramebuffer, void, (GLMContext ctx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)) \
    M(mtlBufferSubData, mtlBufferSubData, void, (GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, const void *ptr)) \
    M(mtlMapUnmapBuffer, mtlMapUnmapBuffer, void *, (GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, GLenum access, bool map)) \
    M(mtlReadBackBuffer, mtlReadBackBuffer, void, (GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size)) \
    M(mtlFlushBufferRange, mtlFlushBufferRange, void, (GLMContext glm_ctx, Buffer *buf, GLintptr offset, GLsizeiptr length)) \
    M(mtlReadDrawable, mtlReadDrawable, void, (GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height)) \
    M(mtlReadIntegerPixels, mtlReadIntegerPixels, void, (GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type)) \
    M(mtlReadDepthPixels, mtlReadDepthPixels, void, (GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height)) \
    M(mtlGetTexImage, mtlGetTexImage, void, (GLMContext glm_ctx, Texture *tex, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLuint level, GLuint slice)) \
    M(mtlGenerateMipmaps, mtlGenerateMipmaps, void, (GLMContext glm_ctx, Texture *tex)) \
    M(mtlTexSubImage, mtlTexSubImage, void, (GLMContext glm_ctx, Texture *tex, Buffer *buf, size_t src_offset, size_t src_pitch, size_t src_image_size, size_t src_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset)) \
    M(mtlTexSubImageBytes, mtlTexSubImageBytes, bool, (GLMContext glm_ctx, Texture *tex, const void *bytes, size_t bytes_size, size_t src_offset, size_t src_pitch, size_t src_image_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset)) \
    M(mtlCopyTexSubImage, mtlCopyTexSubImage, void, (GLMContext glm_ctx, Texture *tex, GLuint slice, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)) \
    M(mtlCopyImageSubData, mtlCopyImageSubData, void, (GLMContext glm_ctx, Texture *srcTex, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, Texture *dstTex, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei width, GLsizei height, GLsizei depth)) \
    M(mtlDrawArrays, mtlDrawArrays, void, (GLMContext ctx, GLenum mode, GLint first, GLsizei count)) \
    M(mtlDrawElements, mtlDrawElements, void, (GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices)) \
    M(mtlDrawRangeElements, mtlDrawRangeElements, void, (GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices)) \
    M(mtlDrawArraysInstanced, mtlDrawArraysInstanced, void, (GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount)) \
    M(mtlDrawElementsInstanced, mtlDrawElementsInstanced, void, (GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount)) \
    M(mtlDrawElementsBaseVertex, mtlDrawElementsBaseVertex, void, (GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex)) \
    M(mtlDrawRangeElementsBaseVertex, mtlDrawRangeElementsBaseVertex, void, (GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex)) \
    M(mtlDrawElementsInstancedBaseVertex, mtlDrawElementsInstancedBaseVertex, void, (GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex)) \
    M(mtlDrawArraysIndirect, mtlDrawArraysIndirect, void, (GLMContext ctx, GLenum mode, const void *indirect)) \
    M(mtlDrawElementsIndirect, mtlDrawElementsIndirect, void, (GLMContext ctx, GLenum mode, GLenum type, const void *indirect)) \
    M(mtlDrawArraysInstancedBaseInstance, mtlDrawArraysInstancedBaseInstance, void, (GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)) \
    M(mtlDrawElementsInstancedBaseInstance, mtlDrawElementsInstancedBaseInstance, void, (GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance)) \
    M(mtlDrawElementsInstancedBaseVertexBaseInstance, mtlDrawElementsInstancedBaseVertexBaseInstance, void, (GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)) \
    M(mtlMultiDrawArrays, mtlMultiDrawArrays, void, (GLMContext ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount)) \
    M(mtlMultiDrawElements, mtlMultiDrawElements, void, (GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount)) \
    M(mtlMultiDrawElementsBaseVertex, mtlMultiDrawElementsBaseVertex, void, (GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex)) \
    M(mtlMultiDrawArraysIndirect, mtlMultiDrawArraysIndirect, void, (GLMContext ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride)) \
    M(mtlMultiDrawElementsIndirect, mtlMultiDrawElementsIndirect, void, (GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride)) \
    M(mtlDispatchCompute, mtlDispatchCompute, void, (GLMContext ctx, GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)) \
    M(mtlDispatchComputeIndirect, mtlDispatchComputeIndirect, void, (GLMContext ctx, GLintptr indirect)) \
    M(mtlBeginSampleQuery, mtlBeginSampleQuery, void, (GLMContext ctx, GLenum target)) \
    M(mtlEndSampleQuery, mtlEndSampleQuery, GLuint64, (GLMContext ctx)) \
    M(mtlBeginTimerQuery, mtlBeginTimerQuery, void, (GLMContext ctx)) \
    M(mtlEndTimerQuery, mtlEndTimerQuery, GLuint64, (GLMContext ctx)) \
    M(mtlGetGPUTimestamp, mtlGetGPUTimestamp, GLuint64, (GLMContext ctx)) \

#define MGL_MTL_FUNC_STRUCT(field, cname, ret, args) ret (*field) args;
MGL_MTL_FUNC_LIST(MGL_MTL_FUNC_STRUCT)
#undef MGL_MTL_FUNC_STRUCT
} ;

enum {
#define MGL_MTL_FUNC_COUNT_ONE(field, cname, ret, args) + 1
    MGL_MTL_FUNC_COUNT = 0 MGL_MTL_FUNC_LIST(MGL_MTL_FUNC_COUNT_ONE)
#undef MGL_MTL_FUNC_COUNT_ONE
};

#endif /* mgl_types_metal_funcs_h */
