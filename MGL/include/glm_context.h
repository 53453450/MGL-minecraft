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
 * glm_context.h
 * MGL
 *
 */

#ifndef glm_context_h
#define glm_context_h

#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include <mach/vm_types.h>
#include "glm_dispatch.h"

#include "draw_command.h"
#include "hash_table.h"

// defines above set sizes in glm_params
#include "glm_params.h"

#ifdef DEBUG
#define DEBUG_LEVEL 3
#endif

#if defined(DEBUG_LEVEL) && DEBUG_LEVEL > 0
 #define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d: " fmt, \
    __func__, __LINE__, ##args)
#else
 #define DEBUG_PRINT(fmt, args...) /* Don't do anything in release builds */
#endif


// macros because I get tired of write if this and that then return
#define RETURN_ON_FAILURE(_expr_) if (_expr_ == false) { fprintf(stderr, "failure %s:%d\n",__FUNCTION__,__LINE__); return; }
#define RETURN_FALSE_ON_FAILURE(_expr_) if (_expr_ == false) { fprintf(stderr, "failure %s:%d\n",__FUNCTION__,__LINE__); return false; }
#define RETURN_FALSE_ON_NULL(_expr_) if (_expr_ == NULL) { fprintf(stderr, "failure %s:%d\n",__FUNCTION__,__LINE__); return false; }
#define RETURN_NULL_ON_FAILURE(_expr_) if (_expr_ == false) { fprintf(stderr, "failure %s:%d\n",__FUNCTION__,__LINE__); return NULL; }
#define RETURN_ON_NULL(_expr_) if (_expr_ == NULL) { fprintf(stderr, "failure %s:%d\n",__FUNCTION__,__LINE__); return; }

/* STATE() / STATE_VAR() / VAO() redirect through ctx->active_state, which
 * always points at the embedded &ctx->state.  The indirection lets the Metal
 * encoding layer share one access path with the C GL layer during batch
 * replay. */
#define STATE(_VAR_)     ctx->active_state->_VAR_
#define STATE_VAR(_VAR_) ctx->active_state->var._VAR_

#define VAO()   ctx->active_state->vao
#define VAO_STATE(_val_)   ctx->active_state->vao->_val_
#define VAO_ATTRIB_STATE(_index_) ctx->active_state->vao->attrib[_index_]

void mglDispatchError(GLMContext ctx, const char *func, GLenum type);

#define ERROR_RETURN(_type_) do { mglDispatchError(ctx, __FUNCTION__, (_type_)); } while(0)
#define ERROR_RETURN_VALUE(_type_, _val_) do { mglDispatchError(ctx, __FUNCTION__, (_type_)); return (_val_); } while(0)
#define ERROR_CHECK_RETURN(_expr_, _type_) do { if ((_expr_) == false) { mglDispatchError(ctx, __FUNCTION__, (_type_)); return; } } while(0)
#define ERROR_CHECK_RETURN_VALUE(_expr_, _type_, _val_) do { if ((_expr_) == false) { mglDispatchError(ctx, __FUNCTION__, (_type_)); return (_val_); } } while(0)

// Type definitions (pulled in from the split headers)
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_vertex.h"
#include "mgl_types_program.h"
#include "mgl_types_framebuffer.h"
#include "mgl_types_sync.h"
#include "mgl_types_state.h"
#include "mgl_renderer_backend.h"

static_assert(_TEXTURE_BUFFER == _TEXTURE_BUFFER_TARGET, "_TEXTURE_BUFFER != _TEXTURE_BUFFER_TARGET");

static_assert(TEXTURE_UNITS == 128, "active_texture_mask relies on this");

/* Query objects and active slots are context-local (GL 4.6 §2.1 / §4.2).
 * Do not place them in shareable GLMState / ShareGroup tables (A02). */
#define MGL_QUERY_TARGET_SLOT_COUNT 18u
#define MGL_QUERY_MAX_INDEX 4u

typedef struct GLMContextRec_t *GLMContext;

typedef struct GLMContextRec_t {
    GLuint      context_flags;

#ifdef MGL_GL_CORE
    struct GLMDispatchTable dispatch;
#endif

#ifdef MGL_GL_ES
    struct GLM_ES_DispatchTable dispatch;
#endif

    GLMState    state;
    /* Pointer to the currently active GLMState.  Defaults to &state; batch
     * replay may later redirect to replay_state once remaining ctx->state
     * readers migrate (ARCHITECTURE_AUDIT R3). */
    GLMState   *active_state;
    GLboolean   assert_on_error;

    PixelFormat pixel_format;
    PixelFormat depth_format;
    PixelFormat stencil_format;
    GLboolean   default_framebuffer_srgb_capable;
    GLuint      default_framebuffer_linear_mtl_pixel_format;
    GLuint      default_framebuffer_srgb_mtl_pixel_format;

    BufferData  *temp_element_buffer;

    MGLCommandBuffer draw_command_buffer;
    bool            draw_defer_enabled;
    bool            sync_strict;

    /* Bump-allocator arena for batch snapshot allocations (Task 4).
     * NULL when MGL_ARENA_SNAPSHOT is not enabled; otherwise points to the
     * MGLRenderer-owned MGLBatchArena ivar.  Accessed from draw_command.c. */
    MGLBatchArena  *batch_arena;

    /* Context-owned query registry (ARCHITECTURE_AUDIT A02). */
    HashTable   query_table;
    GLuint      active_query_by_target[MGL_QUERY_TARGET_SLOT_COUNT][MGL_QUERY_MAX_INDEX];
    GLuint64    query_timestamp_counter;

    /* GL 4.6 Core chapter 20 Debug Output — context-owned ring, not snapshot. */
#define MGL_DEBUG_LOG_CAP 16
#define MGL_DEBUG_MSG_MAX 1024
    GLDEBUGPROC debug_callback;
    const void *debug_callback_user;
    GLboolean   debug_output;
    GLuint      debug_log_count;
    GLuint      debug_log_head;
    struct {
        GLenum source;
        GLenum type;
        GLenum severity;
        GLuint id;
        GLsizei length;
        char msg[MGL_DEBUG_MSG_MAX];
    } debug_log[MGL_DEBUG_LOG_CAP];

    /* Renderer roots. The backend owns Metal state; the context retains the
     * platform renderer shell until backend teardown is complete. */
    void *renderer_backend;
    void *platform_renderer_shell;

    void (* error_func)(GLMContext ctx, const char *func, GLenum type);

    /* Trailing replay workspace (R3).  Kept last so inserting it does not
     * shift earlier GLMContextRec field offsets for incremental rebuilds. */
    GLMState replay_state;
} GLMContextRec;


GLMContext createGLMContext(GLenum format, GLenum type,
                            GLenum depth_format, GLenum depth_type,
                            GLenum stencil_format, GLenum stencil_type);

void MGLsetDefaultFramebufferSRGBCapable(GLMContext ctx, GLboolean capable);
void mgl_lazy_init(void);
GLboolean mglShouldSkipConditionalRender(GLMContext ctx);
void mglRecordActiveSampleQueryDraw(GLMContext ctx);

void MGLsetCurrentContext(GLMContext ctx);
void destroyGLMContext(GLMContext ctx);
/* Free context-local query objects; called from destroyGLMContext. */
void mglDestroyContextQueries(GLMContext ctx);

#include "mgl_context_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

GLuint sizeForFormatType(GLenum format, GLenum type);
GLuint bicountForFormatType(GLenum format, GLenum type, GLenum component);
GLMContext MGLgetCurrentContext(void);
void MGLget(GLMContext ctx, GLenum param, GLuint *data);
bool pixelConvertToInternalFormat(GLMContext ctx, GLenum internalformat, GLenum format, GLenum type, const void *src, void *dst, size_t len);

bool createTextureLevel(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLboolean is_array, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels, GLboolean proxy);
void mglInvalidateTextureBaseLevelView(GLMContext ctx, Texture *tex);

Framebuffer *findFrameBuffer(GLMContext ctx, GLuint framebuffer);
GLboolean mglFramebufferPrimaryColorSize(GLMContext ctx, Framebuffer *fbo, GLuint *outWidth, GLuint *outHeight);
void mglSetViewportToFramebufferSize(GLMContext ctx, Framebuffer *fbo);
void mglAssignDrawFramebuffer(GLMContext ctx, Framebuffer *fbo);


#ifdef __cplusplus
};
#endif

#endif /* glm_context_h */
