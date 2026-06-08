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
 * draw_command.h
 * MGL
 *
 */

#ifndef draw_command_h
#define draw_command_h

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct GLMContextRec_t;
typedef struct GLMContextRec_t *GLMContext;

#define MGL_MAX_DRAWS_PER_BATCH   1024
#define MGL_MAX_BATCHES           64
#define MGL_MDI_MIN_BATCH_SIZE    2
#define MGL_MAX_PENDING_BUFFER_RANGES 4096
#define MGL_MAX_PENDING_TEXTURE_WRITES 256
#define MGL_MAX_PENDING_TEXTURE_READS 512

typedef enum {
    MGL_CMD_DRAW_ARRAYS = 0,
    MGL_CMD_DRAW_ELEMENTS,
    MGL_CMD_DRAW_ARRAYS_INSTANCED,
    MGL_CMD_DRAW_ELEMENTS_INSTANCED,
    MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX,
    MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX,
    MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE,
    MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE,
    MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE,
} MGLDrawCommandType;

typedef struct {
    MGLDrawCommandType type;
    GLenum   mode;
    GLint    first;
    GLsizei  count;
    GLsizei  instanceCount;
    GLint    baseVertex;
    GLuint   baseInstance;
    GLenum   indexType;
    GLuint   indexBufferOffset;
    void    *elementBuffer;
} MGLDrawCommand;

typedef struct {
    uint32_t program_name;
    uint32_t program_pipeline_name;
    uint32_t vertex_program_name;
    uint32_t fragment_program_name;
    uint32_t vao_name;
    uint32_t fbo_name;
    int16_t  viewport[4];
    int16_t  scissor[4];
    uint8_t  scissor_enabled;
    uint8_t  primitive_type;
    uint16_t caps_flags;
    uint64_t texture_hash;
    uint64_t render_state_hash;
} MGLStateKey;

typedef struct {
    MGLStateKey     key;
    uint32_t        command_count;
    MGLDrawCommand *commands;
    void           *state_snapshot;
    void           *vao_snapshot;
    void           *source_vao;
    void           *retained_program;
    void           *retained_vertex_program;
    void           *retained_fragment_program;
    void           *stream_vertex_buffer;
    void           *stream_index_buffer;
    size_t          stream_vertex_bytes;
    size_t          stream_index_bytes;
    size_t          stream_index_count;
    size_t          stream_vertex_stride;
    uint64_t        stream_layout_hash;
    bool            mdi_compatible;
    bool            uses_elements;
    bool            stream_merged;
} MGLDrawBatch;

typedef struct {
    void     *buffer;
    uint64_t  start;
    uint64_t  end;
} MGLBufferReadRange;

typedef struct {
    MGLDrawBatch batches[MGL_MAX_BATCHES];
    uint32_t     batch_count;
    uint32_t     total_commands;
    void        *mdi_scratch_buffer;
    size_t       mdi_scratch_capacity;
    uint32_t     array_cmd_count;
    uint32_t     element_cmd_count;
    MGLBufferReadRange buffer_read_ranges[MGL_MAX_PENDING_BUFFER_RANGES];
    uint32_t     buffer_read_range_count;
    bool         buffer_read_range_overflow;
    void        *texture_write_objects[MGL_MAX_PENDING_TEXTURE_WRITES];
    uint32_t     texture_write_count;
    bool         texture_write_overflow;
    void        *texture_read_objects[MGL_MAX_PENDING_TEXTURE_READS];
    uint32_t     texture_read_count;
    bool         texture_read_overflow;
} MGLCommandBuffer;

void mglInitCommandBuffer(MGLCommandBuffer *cb);
void mglResetCommandBuffer(MGLCommandBuffer *cb);
void mglResetCommandBufferForContext(GLMContext ctx, MGLCommandBuffer *cb);
void mglComputeStateKey(GLMContext ctx, GLenum mode, bool uses_elements, MGLStateKey *out);
bool mglStateKeysEqual(const MGLStateKey *a, const MGLStateKey *b);
void mglAppendDrawCommand(GLMContext ctx, const MGLDrawCommand *cmd);
void mglFlushCommandBuffer(GLMContext ctx);
void mglFlushPendingDraws(GLMContext ctx);
bool mglPendingDrawsReadBufferRange(GLMContext ctx, void *buffer, int64_t offset, int64_t size);
bool mglPendingDrawsWriteTexture(GLMContext ctx, void *texture);
bool mglPendingDrawsReadTexture(GLMContext ctx, void *texture);
void mglFlushPendingDrawsForBuffer(GLMContext ctx, void *buffer);
void mglFlushPendingDrawsForBufferRange(GLMContext ctx, void *buffer, int64_t offset, int64_t size);
void mglFlushPendingDrawsForVertexArray(GLMContext ctx, void *vao);
void mglFlushPendingDrawsForTexture(GLMContext ctx, void *texture);
void mglFlushPendingDrawsBeforeTextureWrite(GLMContext ctx, void *texture);
void mglFlushPendingDrawsForActiveTextures(GLMContext ctx);

#ifdef __cplusplus
}
#endif

#endif /* draw_command_h */
