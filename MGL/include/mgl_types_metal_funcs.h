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

    void (*mtlBindBuffer)(GLMContext glm_ctx, Buffer *ptr);
    void (*mtlBindTexture)(GLMContext glm_ctx, Texture *ptr);
    void (*mtlBindProgram)(GLMContext glm_ctx, Program *ptr);

    void (*mtlDeleteMTLObj)(GLMContext glm_ctx, void *obj);

    /* Release and clear the Metal object stored in a transient Buffer's
     * mtl_data slot without exposing Metal ownership to the GL layer. */
    void (*release_buffer_metal_data)(GLMContext glm_ctx, Buffer *buffer);

    void (*mtlGetSync)(GLMContext glm_ctx, Sync *sync);
    void (*mtlWaitForSync)(GLMContext glm_ctx, Sync *sync);
    /* Non-blocking status query: returns GL_SIGNALED (CB completed or no CB) or
     * GL_UNSIGNALED. Used by mglGetSynciv / mglClientWaitSync polling. */
    GLenum (*mtlGetSyncStatus)(GLMContext glm_ctx, Sync *sync);
    /* Non-blocking release of the fence's retained Metal resources (CB + event)
     * without waiting for GPU completion. Used by mglDeleteSync. */
    void (*mtlReleaseSync)(GLMContext glm_ctx, Sync *sync);

    void (*mtlFlush)(GLMContext glm_ctx, bool finish);
    void (*mtlSwapBuffers)(GLMContext glm_ctx);
    void (*mtlFlushDrawBuffer)(GLMContext glm_ctx);
    void (*mtlInvalidateRenderPass)(GLMContext glm_ctx);

    void (*mtlClearBuffer)(GLMContext glm_ctx, GLuint type, GLbitfield mask);
    void (*mtlBlitFramebuffer)(GLMContext ctx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);


    void (*mtlBufferSubData)(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, const void *ptr);
    void *(*mtlMapUnmapBuffer)(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, GLenum access, bool map);
    void (*mtlFlushBufferRange)(GLMContext glm_ctx, Buffer *buf, GLintptr offset, GLsizeiptr length);

    void (*mtlReadDrawable)(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height);
    void (*mtlReadIntegerPixels)(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type);
    void (*mtlReadDepthPixels)(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height);
    void (*mtlGetTexImage)(GLMContext glm_ctx, Texture *tex, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLuint level, GLuint slice);

    void (*mtlGenerateMipmaps)(GLMContext glm_ctx, Texture *tex);
    void (*mtlTexSubImage)(GLMContext glm_ctx, Texture *tex, Buffer *buf, size_t src_offset, size_t src_pitch, size_t src_image_size, size_t src_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset);
    bool (*mtlTexSubImageBytes)(GLMContext glm_ctx, Texture *tex, const void *bytes, size_t bytes_size, size_t src_offset, size_t src_pitch, size_t src_image_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset);
    void (*mtlCopyTexSubImage)(GLMContext glm_ctx, Texture *tex, GLuint slice, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
    void (*mtlCopyImageSubData)(GLMContext glm_ctx, Texture *srcTex, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, Texture *dstTex, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei width, GLsizei height, GLsizei depth);

    // draw arrays / elements
    void (*mtlDrawArrays)(GLMContext ctx, GLenum mode, GLint first, GLsizei count);
    void (*mtlDrawElements)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices);
    void (*mtlDrawRangeElements)(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices);

    // draw arrays / elements instanced
    void (*mtlDrawArraysInstanced)(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
    void (*mtlDrawElementsInstanced)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount);

    // draw arrays / elements base vertex
    void (*mtlDrawElementsBaseVertex)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex);
    void (*mtlDrawRangeElementsBaseVertex)(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex);
    void (*mtlDrawElementsInstancedBaseVertex)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex);

    // draw arrays / elements intanced base vertex
    void (*mtlDrawArraysIndirect)(GLMContext ctx, GLenum mode, const void *indirect);
    void (*mtlDrawElementsIndirect)(GLMContext ctx, GLenum mode, GLenum type, const void *indirect);

    void (*mtlDrawArraysInstancedBaseInstance)(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance);
    void (*mtlDrawElementsInstancedBaseInstance)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance);
    // ?? running out of names here.
    void (*mtlDrawElementsInstancedBaseVertexBaseInstance)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance);

    // multi calls of many of the above
    void (*mtlMultiDrawArrays)(GLMContext ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount);
    void (*mtlMultiDrawElements)(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount);
    void (*mtlMultiDrawElementsBaseVertex)(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex);

    void (*mtlMultiDrawArraysIndirect)(GLMContext ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride);
    void (*mtlMultiDrawElementsIndirect)(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride);


    void (*mtlDispatchCompute)(GLMContext ctx, GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
    void (*mtlDispatchComputeIndirect)(GLMContext ctx, GLintptr indirect);

    /* Occlusion query support via Metal visibility result buffer.
     * mtlBeginSampleQuery marks that the next render pass should enable
     * visibility result mode. mtlEndSampleQuery flushes the current pass,
     * reads back the result, and returns the sample-pass count (0 if no
     * samples passed). */
    void (*mtlBeginSampleQuery)(GLMContext ctx);
    GLuint64 (*mtlEndSampleQuery)(GLMContext ctx);

    /* GPU timer query support (GL_TIME_ELAPSED / GL_TIMESTAMP).
     * mtlBeginTimerQuery flushes pending GPU work and samples the GPU
     * timestamp into an internal ivar.
     * mtlEndTimerQuery flushes pending work, samples the GPU timestamp
     * again, and returns the elapsed GPU nanoseconds (end - begin).
     * mtlGetGPUTimestamp returns the current GPU timestamp in nanoseconds
     * (used by glQueryCounter(GL_TIMESTAMP)). */
    void (*mtlBeginTimerQuery)(GLMContext ctx);
    GLuint64 (*mtlEndTimerQuery)(GLMContext ctx);
    GLuint64 (*mtlGetGPUTimestamp)(GLMContext ctx);
} ;

#endif /* mgl_types_metal_funcs_h */
