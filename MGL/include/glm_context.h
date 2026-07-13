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
#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>

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
#define RETURN_ON_FAILURE(_expr_) if (_expr_ == false) { printf("failure %s:%d\n",__FUNCTION__,__LINE__); return; }
#define RETURN_FALSE_ON_FAILURE(_expr_) if (_expr_ == false) { printf("failure %s:%d\n",__FUNCTION__,__LINE__); return false; }
#define RETURN_FALSE_ON_NULL(_expr_) if (_expr_ == NULL) { printf("failure %s:%d\n",__FUNCTION__,__LINE__); return false; }
#define RETURN_NULL_ON_FAILURE(_expr_) if (_expr_ == false) { printf("failure %s:%d\n",__FUNCTION__,__LINE__); return NULL; }
#define RETURN_ON_NULL(_expr_) if (_expr_ == NULL) { printf("failure %s:%d\n",__FUNCTION__,__LINE__); return; }

/* STATE() / STATE_VAR() / VAO() redirect through ctx->active_state so that
 * parallel-encoding workers can point at a per-worker GLMState snapshot
 * instead of the shared ctx->state.  In non-parallel mode active_state
 * always equals &ctx->state, so behaviour is identical. */
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

// 类型定义（从拆分的头文件引入）
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_vertex.h"
#include "mgl_types_program.h"
#include "mgl_types_framebuffer.h"
#include "mgl_types_sync.h"
#include "mgl_types_state.h"
#include "mgl_types_metal_funcs.h"

static_assert(_TEXTURE_BUFFER == _TEXTURE_BUFFER_TARGET, "_TEXTURE_BUFFER != _TEXTURE_BUFFER_TARGET");

static_assert(TEXTURE_UNITS == 128, "active_texture_mask relies on this");

typedef struct GLMContextRec_t *GLMContext;

/* GLMMetalFuncs moved to mgl_types_metal_funcs.h (P2-3). */

typedef struct GLMContextRec_t {
    GLuint      context_flags;

#ifdef MGL_GL_CORE
    struct GLMDispatchTable dispatch;
#endif

#ifdef MGL_GL_ES
    struct GLM_ES_DispatchTable dispatch;
#endif

    struct GLMMetalFuncs mtl_funcs;

    GLMState    state;
    /* Pointer to the currently active GLMState.  In non-parallel mode this
     * always points to the embedded state above.  During parallel batch
     * encoding each worker redirects this to its own per-worker
     * GLMState copy, so that ctx->active_state->* accesses are thread-safe.
     * STATE() / STATE_VAR() / VAO() macros and all direct accesses in the
     * Metal encoding layer go through this pointer. */
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

    void (* error_func)(GLMContext ctx, const char *func, GLenum type);
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

#ifndef MGL_CONTEXT_ENUMS_DEFINED
#define MGL_CONTEXT_ENUMS_DEFINED
enum {
    MGL_PIXEL_FORMAT,
    MGL_PIXEL_TYPE,
    MGL_DEPTH_FORMAT,
    MGL_DEPTH_TYPE,
    MGL_STENCIL_FORMAT,
    MGL_STENCIL_TYPE,
    MGL_CONTEXT_FLAGS
};
#endif /* MGL_CONTEXT_ENUMS_DEFINED */

#ifdef __cplusplus
extern "C" {
#endif

GLuint sizeForFormatType(GLenum format, GLenum type);
GLuint bicountForFormatType(GLenum format, GLenum type, GLenum component);
GLMContext MGLgetCurrentContext(void);
void MGLget(GLMContext ctx, GLenum param, GLuint *data);
bool pixelConvertToInternalFormat(GLMContext ctx, GLenum internalformat, GLenum format, GLenum type, const void *src, void *dst, size_t len);

bool createTextureLevel(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLboolean is_array, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels, GLboolean proxy);

Framebuffer *findFrameBuffer(GLMContext ctx, GLuint framebuffer);
GLboolean mglFramebufferPrimaryColorSize(GLMContext ctx, Framebuffer *fbo, GLuint *outWidth, GLuint *outHeight);
void mglSetViewportToFramebufferSize(GLMContext ctx, Framebuffer *fbo);
void mglAssignDrawFramebuffer(GLMContext ctx, Framebuffer *fbo);


#ifdef __cplusplus
};
#endif

#endif /* glm_context_h */
