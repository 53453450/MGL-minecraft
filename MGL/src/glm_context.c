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
 * glm_context.c
 * MGL
 *
 */


#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>

#include <stdint.h>

#include <assert.h>
#include <CoreFoundation/CoreFoundation.h>

#include "mgl_metal_ref.h"
#include <mach/mach_init.h>
#include <mach/vm_map.h>

#include "glm_context.h"
#include "mgl_renderer_backend.h"
#include "vertex_arrays.h"
#include "buffers.h"
#include "shaders.h"
#include "MGLRenderer.h"
#include "error.h"
#include "mgl_safety.h"

extern void getMacOSDefaults(GLMContext glm_ctx);
extern void init_dispatch(GLMContext ctx);
extern void invalidateTexture(GLMContext ctx, Texture *tex);
extern void mglFreeProgram(GLMContext ctx, Program *ptr);
#include "mgl_trace_log.h"

static _Thread_local GLMContext _ctx = NULL;
static _Thread_local GLboolean _ctx_explicitly_unbound = GL_FALSE;

int mglContextHasReadyRendererBackend(GLMContext ctx)
{
    const uintptr_t minimum_valid_pointer = 0x1000u;
    return ctx && (uintptr_t)ctx >= minimum_valid_pointer &&
           ctx->renderer_backend &&
           (uintptr_t)ctx->renderer_backend >= minimum_valid_pointer &&
           mglRendererBackendIsReady(
               (MGLRendererBackendHandle *)ctx->renderer_backend) == 1;
}

enum {
    kMGLMGLPixelFormatInvalid = 0,
    kMGLMGLPixelFormatRGBA8Unorm = 70,
    kMGLMGLPixelFormatRGBA8Unorm_sRGB = 71,
    kMGLMGLPixelFormatBGRA8Unorm = 80,
    kMGLMGLPixelFormatBGRA8Unorm_sRGB = 81
};

/* Declared in MGLRenderer.m */
extern void* CppCreateMGLRendererHeadless(void *glm_ctx);

/* Initialize MGL on-demand (not at library load time).
 * Loading via dlopen must never crash if runtime dependencies are not ready.
 */
static void mgl_auto_init(void) {
    if (_ctx == NULL && !_ctx_explicitly_unbound) {
        GLMContext ctx = createGLMContext(GL_RGBA, GL_UNSIGNED_BYTE,
                                          GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                          GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE);
        _ctx = ctx;
        if (!CppCreateMGLRendererHeadless(ctx)) {
            fprintf(stderr, "MGL: Failed to initialize headless Metal renderer\n");
            _ctx = NULL;
            destroyGLMContext(ctx);
            return;
        }
        fprintf(stderr, "MGL: Initialized headless Metal renderer\n");
    }
}

/* Lazy-initialize MGL context on first GL API call if auto-init didn't run */
void mgl_lazy_init(void) {
    if (_ctx != NULL) {
        static int s_validate_current_context = -1;
        if (s_validate_current_context < 0) {
            const char *env = getenv("MGL_VALIDATE_CURRENT_CONTEXT");
            s_validate_current_context =
                (env && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
        }
        if (!s_validate_current_context) {
            return;
        }

        // If `_ctx` ever gets corrupted (e.g. memory stomp), avoid dereferencing it.
        if (!mglPointerRangeIsReadable(_ctx, sizeof(*_ctx))) {
            fprintf(stderr, "MGL ERROR: current context pointer looks corrupted (%p); reinitializing\n", (void *)_ctx);
            _ctx = NULL;
        } else {
            return;
        }
    }

    if (_ctx == NULL) {
        mgl_auto_init();
    }
}

GLMContext mglGetContext(void)
{
    return _ctx;
}

static GLuint mglLinearDefaultFramebufferFormat(GLuint mtl_format)
{
    switch (mtl_format) {
        case kMGLMGLPixelFormatRGBA8Unorm_sRGB:
            return kMGLMGLPixelFormatRGBA8Unorm;
        case kMGLMGLPixelFormatBGRA8Unorm_sRGB:
            return kMGLMGLPixelFormatBGRA8Unorm;
        default:
            return mtl_format;
    }
}

static GLuint mglSRGBDefaultFramebufferFormat(GLuint mtl_format)
{
    switch (mtl_format) {
        case kMGLMGLPixelFormatRGBA8Unorm:
        case kMGLMGLPixelFormatRGBA8Unorm_sRGB:
            return kMGLMGLPixelFormatRGBA8Unorm_sRGB;
        case kMGLMGLPixelFormatBGRA8Unorm:
        case kMGLMGLPixelFormatBGRA8Unorm_sRGB:
            return kMGLMGLPixelFormatBGRA8Unorm_sRGB;
        default:
            return mtl_format;
    }
}

static GLuint mglDefaultFramebufferFormatForGLFormatType(GLenum format, GLenum type)
{
    GLuint mtl_format = mtlPixelFormatForGLFormatType(format, type);
    if (mtl_format != kMGLMGLPixelFormatInvalid) {
        return mtl_format;
    }

    switch (type) {
        case GL_UNSIGNED_BYTE:
            if (format == GL_BGRA) {
                return kMGLMGLPixelFormatBGRA8Unorm;
            }
            if (format == GL_RGBA) {
                return kMGLMGLPixelFormatRGBA8Unorm;
            }
            break;
        case GL_UNSIGNED_INT_8_8_8_8:
            if (format == GL_RGBA || format == GL_BGRA) {
                return kMGLMGLPixelFormatRGBA8Unorm;
            }
            break;
        case GL_UNSIGNED_INT_8_8_8_8_REV:
            if (format == GL_RGBA || format == GL_BGRA) {
                return kMGLMGLPixelFormatBGRA8Unorm;
            }
            break;
        default:
            break;
    }

    return mtl_format;
}

void MGLsetDefaultFramebufferSRGBCapable(GLMContext ctx, GLboolean capable)
{
    if (!ctx) {
        return;
    }

    if (ctx->default_framebuffer_linear_mtl_pixel_format == 0u ||
        ctx->default_framebuffer_linear_mtl_pixel_format == kMGLMGLPixelFormatInvalid) {
        ctx->default_framebuffer_linear_mtl_pixel_format =
            mglLinearDefaultFramebufferFormat(ctx->pixel_format.mtl_pixel_format);
    }
    if (ctx->default_framebuffer_srgb_mtl_pixel_format == 0u ||
        ctx->default_framebuffer_srgb_mtl_pixel_format == kMGLMGLPixelFormatInvalid) {
        ctx->default_framebuffer_srgb_mtl_pixel_format =
            mglSRGBDefaultFramebufferFormat(ctx->default_framebuffer_linear_mtl_pixel_format);
    }

    ctx->default_framebuffer_srgb_capable = capable ? GL_TRUE : GL_FALSE;
    ctx->pixel_format.mtl_pixel_format = ctx->default_framebuffer_srgb_capable
        ? ctx->default_framebuffer_srgb_mtl_pixel_format
        : ctx->default_framebuffer_linear_mtl_pixel_format;
    mglMarkStateDirtyBits(&ctx->state, DIRTY_FBO | DIRTY_RENDER_STATE | DIRTY_DRAWABLE);
}

GLMContext createGLMContext(GLenum format, GLenum type,
                            GLenum depth_format, GLenum depth_type,
                            GLenum stencil_format, GLenum stencil_type)
{
    GLMContext ctx = (GLMContext)malloc(sizeof(GLMContextRec));
    GLMContext save = _ctx;

    if (!ctx) {
        /* OOM: no GLMContext exists yet, so ctx->error_func cannot be called.
         * The caller (likely the GL entry-point dispatcher or auto-init path)
         * has no context to dispatch an error into — surface the failure via
         * the trace log and return NULL so the caller can degrade gracefully. */
        mglTraceLogExternal("MGL OOM: malloc(sizeof(GLMContextRec)) failed in createGLMContext");
        return NULL;
    }

    bzero((void *)ctx, sizeof(GLMContextRec));

    /* active_state defaults to the embedded state; batch replay redirects
     * MGL_STATE reads through it while a snapshot is installed. */
    ctx->active_state = &ctx->state;

    _ctx = ctx;

    if ((format == 0) && (type == 0))
    {
        format = GL_BGRA;
        type = GL_UNSIGNED_INT_8_8_8_8_REV;
    }

    ctx->pixel_format.format = format;
    ctx->pixel_format.type = type;
    ctx->pixel_format.mtl_pixel_format = mglDefaultFramebufferFormatForGLFormatType(format, type);
    ctx->default_framebuffer_linear_mtl_pixel_format =
        mglLinearDefaultFramebufferFormat(ctx->pixel_format.mtl_pixel_format);
    ctx->default_framebuffer_srgb_mtl_pixel_format =
        mglSRGBDefaultFramebufferFormat(ctx->default_framebuffer_linear_mtl_pixel_format);

    if (depth_format)
    {
        ctx->depth_format.format = depth_format;
        ctx->depth_format.type = depth_type;
        ctx->depth_format.mtl_pixel_format = mtlPixelFormatForGLFormatType(depth_format, depth_type);
    }

    if (stencil_format)
    {
        ctx->stencil_format.format = stencil_format;
        ctx->stencil_format.type = stencil_type;
        ctx->stencil_format.mtl_pixel_format = mtlPixelFormatForGLFormatType(stencil_format, stencil_type);
    }

    // use a CGL context to read guestimates of gl params for installed GPU
    getMacOSDefaults(ctx);

    if (STATE(max_color_attachments) == 0 ||
        STATE(max_color_attachments) > MAX_COLOR_ATTACHMENTS ||
        STATE(max_color_attachments) == 0x01010101u)
    {
        fprintf(stderr,
                "MGL WARNING: GL_MAX_COLOR_ATTACHMENTS state value suspicious (%u), using fallback %u\n",
                STATE(max_color_attachments),
                MAX_COLOR_ATTACHMENTS);
        STATE(max_color_attachments) = MAX_COLOR_ATTACHMENTS;
    }
    STATE(var.max_color_attachments) = STATE(max_color_attachments);

    if (STATE(var.max_draw_buffers) == 0 ||
        STATE(var.max_draw_buffers) > MAX_COLOR_ATTACHMENTS ||
        STATE(var.max_draw_buffers) == 0x01010101u)
    {
        fprintf(stderr,
                "MGL WARNING: GL_MAX_DRAW_BUFFERS state value suspicious (%u), using fallback %u\n",
                STATE(var.max_draw_buffers),
                MAX_COLOR_ATTACHMENTS);
        STATE(var.max_draw_buffers) = MAX_COLOR_ATTACHMENTS;
    }

    if (STATE(var.max_clip_distances) == 0 ||
        STATE(var.max_clip_distances) > MAX_CLIP_DISTANCES ||
        STATE(var.max_clip_distances) == 0x01010101u)
    {
        fprintf(stderr,
                "MGL WARNING: GL_MAX_CLIP_DISTANCES state value suspicious (%u), using fallback %u\n",
                STATE(var.max_clip_distances),
                MAX_CLIP_DISTANCES);
        STATE(var.max_clip_distances) = MAX_CLIP_DISTANCES;
    }
    STATE(var.max_clip_planes) = STATE(var.max_clip_distances);

    if (STATE(max_color_attachments) > MAX_COLOR_ATTACHMENTS) {
        fprintf(stderr,
                "MGL WARNING: max_color_attachments %u exceeds backend cap %u; clamping\n",
                STATE(max_color_attachments),
                MAX_COLOR_ATTACHMENTS);
        STATE(max_color_attachments) = MAX_COLOR_ATTACHMENTS;
        STATE(var.max_color_attachments) = MAX_COLOR_ATTACHMENTS;
    }
    if (STATE(max_vertex_attribs) > MAX_ATTRIBS) {
        fprintf(stderr,
                "MGL WARNING: max_vertex_attribs %u exceeds backend cap %u; clamping\n",
                STATE(max_vertex_attribs),
                MAX_ATTRIBS);
        STATE(max_vertex_attribs) = MAX_ATTRIBS;
        STATE(var.max_vertex_attribs) = MAX_ATTRIBS;
    }

    /*
     * The current Metal backend does not allocate true multisample textures or
     * renderbuffers.  Report the OpenGL 4.6 minimum limits so that CTS limit
     * tests pass; actual multisample rendering may be silently downgraded to
     * single-sample by the backend.
     */
    if (STATE(var.max_compute_texture_image_units) == 0 ||
        STATE(var.max_compute_texture_image_units) == 0x01010101u ||
        STATE(var.max_compute_texture_image_units) > 16u) {
        STATE(var.max_compute_texture_image_units) = 16u;
    }
    if (STATE(var.max_sample_mask_words) < 1) {
        STATE(var.max_sample_mask_words) = 1;
    }
    if (STATE(var.max_color_texture_samples) < 1) {
        STATE(var.max_color_texture_samples) = 4;
    }
    if (STATE(var.max_depth_texture_samples) < 1) {
        STATE(var.max_depth_texture_samples) = 4;
    }
    if (STATE(var.max_integer_samples) < 1) {
        STATE(var.max_integer_samples) = 4;
    }
    if (STATE(var.max_framebuffer_samples) < 4) {
        STATE(var.max_framebuffer_samples) = 4;
    }
    if (STATE(var.max_samples) < 4) {
        STATE(var.max_samples) = 4;
    }

    /* Ensure compute limits meet OpenGL 4.6 minimums */
    if (STATE(var.max_compute_uniform_components) < 1024) {
        STATE(var.max_compute_uniform_components) = 1024;
    }
    if (STATE(var.max_compute_atomic_counters) < 8) {
        STATE(var.max_compute_atomic_counters) = 8;
    }
    if (STATE(var.max_compute_atomic_counter_buffers) < 8) {
        STATE(var.max_compute_atomic_counter_buffers) = 8;
    }
    if (STATE(var.max_fragment_atomic_counters) < 8) {
        STATE(var.max_fragment_atomic_counters) = 8;
    }
    if (STATE(var.max_combined_atomic_counters) < 8) {
        STATE(var.max_combined_atomic_counters) = 8;
    }
    if (STATE(var.max_geometry_atomic_counters) < 8) {
        STATE(var.max_geometry_atomic_counters) = 8;
    }
    if (STATE(var.max_geometry_atomic_counter_buffers) < 8) {
        STATE(var.max_geometry_atomic_counter_buffers) = 8;
    }

    /* Ensure max_element_index meets minimum */
    if (STATE(var.max_element_index) == 0 ||
        STATE(var.max_element_index) < 0xFFFFFFFFu) {
        STATE(var.max_element_index) = 0xFFFFFFFFu;
    }

    /* Ensure max_label_length meets minimum */
    if (STATE(var.max_label_length) < 256) {
        STATE(var.max_label_length) = 256;
    }

    // For this Metal backend, default framebuffer rendering targets the current drawable.
    // Keep legacy default as FRONT to avoid routing GL_BACK to an internal offscreen buffer.
    STATE(draw_buffer) = GL_FRONT;
    STATE(draw_buffer_count) = 1;
    STATE(draw_buffers[0]) = GL_FRONT;
    STATE(default_draw_buffer) = GL_FRONT;
    STATE(default_draw_buffer_count) = 1;
    STATE(default_draw_buffers[0]) = GL_FRONT;
    for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        STATE(draw_buffers[i]) = GL_NONE;
        STATE(default_draw_buffers[i]) = GL_NONE;
    }
    STATE(read_buffer) = GL_FRONT;
    STATE(default_read_buffer) = GL_FRONT;
    STATE(active_texture) = 0;

    STATE(pack.swap_bytes) = false;
    STATE(pack.lsb_first) = false;
    STATE(pack.row_length) = 0;
    STATE(pack.image_height) = 0;
    STATE(pack.skip_rows) = 0;
    STATE(pack.skip_pixels) = 0;
    STATE(pack.skip_images) = 0;
    STATE(pack.alignment) = 4;

    STATE(unpack.swap_bytes) = false;
    STATE(unpack.lsb_first) = false;
    STATE(unpack.row_length) = 0;
    STATE(unpack.image_height) = 0;
    STATE(unpack.skip_rows) = 0;
    STATE(unpack.skip_pixels) = 0;
    STATE(unpack.skip_images) = 0;
    STATE(unpack.alignment) = 4;

    STATE(caps.blend) = false;
    STATE(caps.line_smooth) = false;
    STATE(caps.polygon_smooth) = false;
    STATE(caps.cull_face) = false;
    STATE(caps.depth_test) = false;
    STATE(caps.stencil_test) = false;
    STATE(caps.dither) = true;
    STATE(caps.scissor_test) = false;
    STATE(caps.color_logic_op) = false;
    STATE(caps.polygon_offset_point) = false;
    STATE(caps.polygon_offset_line) = false;
    STATE(caps.polygon_offset_fill) = false;
    STATE(caps.index_logic_op) = false;
    STATE(caps.multisample) = true;
    STATE(caps.sample_alpha_to_coverage) = false;
    STATE(caps.sample_alpha_to_one) = false;
    STATE(caps.sample_coverage) = false;
    STATE(caps.rasterizer_discard) = false;
    STATE(caps.framebuffer_srgb) = false;
    STATE(caps.primitive_restart) = false;
    STATE(caps.depth_clamp) = false;
    STATE(caps.texture_cube_map_seamless) = false;
    STATE(caps.sample_mask) = false;
    STATE(caps.sample_shading) = false;
    STATE(var.sample_coverage_value) = 1.0f;
    STATE(var.sample_coverage_invert) = GL_FALSE;
    STATE(caps.primitive_restart_fixed_index) = false;
    STATE(caps.debug_output_synchronous) = false;
    STATE(caps.debug_output) = false;
    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        STATE(caps.blendi[i]) = false;
    }
    for (int i = 0; i < MGL_MAX_VIEWPORTS; i++)
    {
        STATE(caps.scissor_testi[i]) = false;
    }

    STATE(var.cull_face_mode) = GL_BACK;
    STATE(var.front_face) = GL_CCW;

    STATE(hints.line_smooth_hint) = GL_DONT_CARE;
    STATE(hints.polygon_smooth_hint) = GL_DONT_CARE;
    STATE(hints.texture_compression_hint) = GL_DONT_CARE;
    STATE(hints.fragment_shader_derivative_hint) = GL_DONT_CARE;

    STATE(var.line_width) = 1.0f;
    STATE(var.point_size) = 1.0f;
    STATE(var.polygon_mode) = GL_FILL;
    STATE(var.primitive_restart_index) = 0u;
    STATE(var.provoking_vertex) = GL_LAST_VERTEX_CONVENTION;
    STATE(var.polygon_offset_factor) = 0.0f;
    STATE(var.polygon_offset_units) = 0.0f;

    // Viewport and scissor should match the drawable dimensions.
    // getMacOSDefaults() already queried GL_VIEWPORT/GL_SCISSOR_BOX from the
    // system GL; validate and use those values instead of hardcoding 1024x768.
    {
        GLint vpW = STATE(viewport[2]);
        GLint vpH = STATE(viewport[3]);
        if (vpW <= 0 || vpH <= 0 || vpW > 32768 || vpH > 32768)
        {
            vpW = 1024;
            vpH = 768;
            STATE(viewport[0]) = 0;
            STATE(viewport[1]) = 0;
            STATE(viewport[2]) = vpW;
            STATE(viewport[3]) = vpH;
        }
        STATE(var.scissor_box[0]) = 0;
        STATE(var.scissor_box[1]) = 0;
        STATE(var.scissor_box[2]) = vpW;
        STATE(var.scissor_box[3]) = vpH;
        for (int i = 0; i < MGL_MAX_VIEWPORTS; i++)
        {
            STATE(viewport_array[i][0]) = 0.0f;
            STATE(viewport_array[i][1]) = 0.0f;
            STATE(viewport_array[i][2]) = (GLfloat)vpW;
            STATE(viewport_array[i][3]) = (GLfloat)vpH;
            STATE(scissor_box_array[i][0]) = 0;
            STATE(scissor_box_array[i][1]) = 0;
            STATE(scissor_box_array[i][2]) = vpW;
            STATE(scissor_box_array[i][3]) = vpH;
        }
        STATE(viewport_array_set) = GL_FALSE;
        for (int i = 0; i < MGL_MAX_VIEWPORTS; i++)
        {
            STATE(depth_range_array[i][0]) = 0.0;
            STATE(depth_range_array[i][1]) = 1.0;
        }
    }

    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        STATE(var.blend_src_rgb[i]) = GL_ONE;
        STATE(var.blend_src_alpha[i]) = GL_ONE;
        STATE(var.blend_dst_rgb[i]) = GL_ZERO;
        STATE(var.blend_dst_alpha[i]) = GL_ZERO;
        STATE(var.blend_equation_rgb[i]) = GL_FUNC_ADD;
        STATE(var.blend_equation_alpha[i]) = GL_FUNC_ADD;
    }

    STATE(var.depth_func) = GL_LESS;
    STATE(var.depth_writemask) = GL_TRUE;
    STATE(var.depth_clear_value) = 1.0;
    STATE(var.clip_origin) = GL_LOWER_LEFT;
    STATE(var.clip_depth_mode) = GL_NEGATIVE_ONE_TO_ONE;

    // GL_COLOR_CLEAR_VALUE defaults to (0, 0, 0, 0) per GL 4.6 spec
    STATE(color_clear_value[0]) = 0.0f;
    STATE(color_clear_value[1]) = 0.0f;
    STATE(color_clear_value[2]) = 0.0f;
    STATE(color_clear_value[3]) = 0.0f;

    // Initialize default FBO clear state
    STATE(default_fbo_clear_bitmask) = 0;
    STATE(default_clear_color[0]) = 0.0f;
    STATE(default_clear_color[1]) = 0.0f;
    STATE(default_clear_color[2]) = 0.0f;
    STATE(default_clear_color[3]) = 1.0f;

    for (int i = 0; i < MAX_ATTRIBS; i++)
    {
        STATE(current_vertex_attrib[i].f[0]) = 0.0f;
        STATE(current_vertex_attrib[i].f[1]) = 0.0f;
        STATE(current_vertex_attrib[i].f[2]) = 0.0f;
        STATE(current_vertex_attrib[i].f[3]) = 1.0f;
        STATE(current_vertex_attrib[i].i[0]) = 0;
        STATE(current_vertex_attrib[i].i[1]) = 0;
        STATE(current_vertex_attrib[i].i[2]) = 0;
        STATE(current_vertex_attrib[i].i[3]) = 1;
        STATE(current_vertex_attrib[i].u[0]) = 0u;
        STATE(current_vertex_attrib[i].u[1]) = 0u;
        STATE(current_vertex_attrib[i].u[2]) = 0u;
        STATE(current_vertex_attrib[i].u[3]) = 1u;
        STATE(current_vertex_attrib[i].d[0]) = 0.0;
        STATE(current_vertex_attrib[i].d[1]) = 0.0;
        STATE(current_vertex_attrib[i].d[2]) = 0.0;
        STATE(current_vertex_attrib[i].d[3]) = 1.0;
        STATE(current_vertex_attrib[i].type) = GL_FLOAT;
        STATE(current_vertex_attrib[i].integer) = GL_FALSE;
        STATE(current_vertex_attrib[i].long_attribute) = GL_FALSE;
    }

    STATE(var.logic_op) = GL_COPY;
    STATE(var.logic_op_mode) = GL_COPY;
    STATE(var.stencil_func) = GL_ALWAYS;

    STATE(var.stencil_fail) = GL_KEEP;
    STATE(var.stencil_pass_depth_fail) = GL_KEEP;
    STATE(var.stencil_pass_depth_pass) = GL_KEEP;

    for(int i=0; i<MAX_CLIP_DISTANCES; i++)
    {
        STATE(caps.clip_distances[i]) = false;
    }

    STATE(var.stencil_fail) = GL_KEEP;
    STATE(var.stencil_pass_depth_fail) = GL_KEEP;
    STATE(var.stencil_pass_depth_pass) = GL_KEEP;
    STATE(var.stencil_back_fail) = GL_KEEP;
    STATE(var.stencil_fail) = GL_KEEP;
    STATE(var.stencil_back_pass_depth_fail) = GL_KEEP;
    STATE(var.stencil_back_pass_depth_pass) = GL_KEEP;

    STATE(var.stencil_func) = GL_ALWAYS;
    STATE(var.stencil_ref) = 0;
    STATE(var.stencil_value_mask) = 0xFFFFFFFF;
    STATE(var.stencil_writemask) = 0xFFFFFFFF;

    STATE(var.stencil_back_func) = GL_ALWAYS;
    STATE(var.stencil_back_ref) = 0;
    STATE(var.stencil_back_value_mask) = 0xFFFFFFFF;
    STATE(var.stencil_back_writemask) = 0xFFFFFFFF;

    STATE(var.max_compute_work_group_invocations) = 1024;

    STATE(var.max_compute_work_group_count[0]) = 65535;
    STATE(var.max_compute_work_group_count[1]) = 65535;
    STATE(var.max_compute_work_group_count[2]) = 65535;

    STATE(var.max_compute_work_group_size[0]) = 1024;
    STATE(var.max_compute_work_group_size[1]) = 1024;
    STATE(var.max_compute_work_group_size[2]) = 256;

    for(int attachment=0; attachment<MAX_COLOR_ATTACHMENTS; attachment++)
    {
        STATE(caps.use_color_mask[attachment]) = false;

        for(int i=0; i<4; i++)
            STATE(var.color_writemask[attachment][i]) = GL_TRUE;
    }


    STATE(var.cull_face_mode) = GL_BACK;

    STATE(sync_name) = 1;
    STATE(program_name) = 0;

    STATE(dirty_bits) = DIRTY_ALL;

    /* Initialize hash cache dirty flags to 1 (needs initial computation) */
    STATE(texture_dirty) = 1;
    STATE(vertex_layout_dirty) = 1;
    STATE(render_state_dirty) = 1;
    STATE(uniform_buffer_dirty) = 1;
    STATE(cached_texture_hash) = 0;
    STATE(cached_vertex_layout_hash) = 0;
    STATE(cached_render_state_hash) = 0;
    STATE(cached_uniform_buffer_hash) = 0;

    initHashTable(&STATE(vao_table), 32);
    initHashTable(&STATE(buffer_table), 32);
    initHashTable(&STATE(texture_table), 32);
    initHashTable(&STATE(shader_table), 32);
    initHashTable(&STATE(program_table), 32);
    initHashTable(&STATE(program_pipeline_table), 32);
    initHashTable(&STATE(transform_feedback_table), 32);
    initHashTable(&STATE(renderbuffer_table), 32);
    initHashTable(&STATE(framebuffer_table), 32);
    initHashTable(&STATE(sampler_table), 32);
    initHashTable(&STATE(sync_table), 32);
    
    init_dispatch(ctx);

    ctx->assert_on_error = GL_TRUE;
    ctx->error_func = error_func;

    ctx->temp_element_buffer = NULL;
    
    _ctx = save;

    mglInitCommandBuffer(&ctx->draw_command_buffer);
    ctx->draw_defer_enabled = (getenv("MGL_DISABLE_DRAW_DEFER") == NULL);
    /* MGL_SYNC_STRICT: when enabled, every sync boundary takes the most
     * conservative path (full flush + commit + waitUntilCompleted) for
     * regression triage and correctness-baseline comparison */
    ctx->sync_strict = (getenv("MGL_SYNC_STRICT") != NULL);

    return ctx;
}

void MGLsetCurrentContext(GLMContext ctx)
{
    _ctx = ctx;
    _ctx_explicitly_unbound = (ctx == NULL) ? GL_TRUE : GL_FALSE;
}

GLMContext MGLgetCurrentContext(void)
{
    return _ctx;
}

void MGLget(GLMContext ctx, GLenum param, GLuint *data)
{
    if (ctx == NULL)
        ctx = _ctx;
    
    if (ctx == NULL)
        return;
    
    switch(param)
    {
        case MGL_PIXEL_FORMAT: *data = ctx->pixel_format.format; break;
        case MGL_PIXEL_TYPE: *data = ctx->pixel_format.type; break;
        case MGL_DEPTH_FORMAT: *data = ctx->depth_format.format; break;
        case MGL_DEPTH_TYPE: *data = ctx->depth_format.type; break;
        case MGL_STENCIL_FORMAT: *data = ctx->stencil_format.format; break;
        case MGL_STENCIL_TYPE: *data = ctx->stencil_format.type; break;
        case MGL_CONTEXT_FLAGS: *data = ctx->context_flags; break;
        default:
            fprintf(stderr, "MGL WARNING: MGLget unknown param 0x%x\n", param);
            *data = 0;
            break;
    }
}

void MGLswapBuffers(GLMContext ctx)
{
    static uint64_t s_mglSwapBuffersCalls = 0;
    uint64_t call = ++s_mglSwapBuffersCalls;

    if (ctx == NULL)
        ctx = _ctx;

    if (ctx == NULL) {
        if (call <= 20 || (call % 60) == 0) {
            mglTraceLogExternal("SWAP_ENTRY call=%llu ctx=NULL",
                                (unsigned long long)call);
        }
        return;
    }

    if (call <= 20 || (call % 60) == 0) {
        mglTraceLogExternal("SWAP_ENTRY call=%llu ctx=%p mtlSwap=%p drawBuf=0x%x fbo=%p program=%u",
                            (unsigned long long)call,
                            (void *)ctx,
                            (void *)mglRendererSwapBuffers,
                            (unsigned)ctx->state.draw_buffer,
                            (void *)ctx->state.framebuffer,
                            (unsigned)ctx->state.program_name);
    }

    mglRendererSwapBuffers(ctx);
}

static void mglDestroyContextBuffer(GLuint name, void *data, void *user)
{
    (void)name;
    (void)user;
    Buffer *buffer = (Buffer *)data;

    if (!buffer) {
        return;
    }

    mglReleaseBufferStorage(buffer);

    free(buffer);
}

static void mglDestroyContextTexture(GLuint name, void *data, void *user)
{
    (void)name;
    GLMContext ctx = (GLMContext)user;
    Texture *texture = (Texture *)data;

    if (!texture) {
        return;
    }

    invalidateTexture(ctx, texture);
    free(texture);
}

static void mglDestroyContextShader(GLuint name, void *data, void *user)
{
    (void)name;
    GLMContext ctx = (GLMContext)user;
    Shader *shader = (Shader *)data;

    if (!shader) {
        return;
    }

    shader->refcount = 0;
    mglFreeShader(ctx, shader);
}

static void mglDestroyContextProgram(GLuint name, void *data, void *user)
{
    GLMContext ctx = (GLMContext)user;
    Program *program = (Program *)data;

    if (!program) {
        return;
    }

    /* mglFreeProgram requires the name to be detached first.  foreach does
     * not rehash on deletion, so removing the current slot is safe and keeps
     * teardown on the same lifecycle path as glDeleteProgram. */
    deleteHashElement(&ctx->state.program_table, name);
    mglFreeProgram(ctx, program);
}

static void mglDestroyContextSampler(GLuint name, void *data, void *user)
{
    (void)name;
    (void)user;
    Sampler *sampler = (Sampler *)data;

    if (!sampler) {
        return;
    }

    mglSafeReleaseMetalObj((void **)&sampler->mtl_data);

    free(sampler);
}

static void mglDestroyContextRenderbuffer(GLuint name, void *data, void *user)
{
    (void)name;
    GLMContext ctx = (GLMContext)user;
    Renderbuffer *renderbuffer = (Renderbuffer *)data;

    if (renderbuffer && renderbuffer->tex) {
        invalidateTexture(ctx, renderbuffer->tex);
        free(renderbuffer->tex);
        renderbuffer->tex = NULL;
    }

    free(renderbuffer);
}

static void mglDestroyContextFramebuffer(GLuint name, void *data, void *user)
{
    (void)name;
    (void)user;
    Framebuffer *framebuffer = (Framebuffer *)data;

    free(framebuffer);
}

static void mglDestroyContextVertexArray(GLuint name, void *data, void *user)
{
    (void)name;
    (void)user;
    VertexArray *vao = (VertexArray *)data;

    if (vao) {
        vao->magic = 0;
    }
    free(vao);
}

static void mglDestroyContextProgramPipeline(GLuint name, void *data, void *user)
{
    (void)name;
    (void)user;
    ProgramPipeline *pipeline = (ProgramPipeline *)data;

    free(pipeline);
}

static void mglDestroyContextTransformFeedback(GLuint name, void *data, void *user)
{
    (void)name;
    (void)user;
    TransformFeedback *tf = (TransformFeedback *)data;

    free(tf);
}

/* release Sync objects left in sync_table at context destroy time.
 * Mirrors mglDeleteSync's release logic — non-blocking path preferred,
 * blocking wait as fallback. */
static void mglDestroyContextSync(GLuint name, void *data, void *user)
{
    (void)name;
    GLMContext ctx = (GLMContext)user;
    Sync *sync = (Sync *)data;

    if (!sync) {
        return;
    }

    if (ctx) {
        mglRendererReleaseSync(ctx, sync);
    }

    free(sync);
}

// CRITICAL FIX: Proper context destruction to prevent memory leaks
void destroyGLMContext(GLMContext ctx)
{
    if (ctx == NULL)
        return;

    fprintf(stderr, "MGL INFO: Destroying GLMContext\n");

    GLMContext save = _ctx;
    _ctx = ctx;
    _ctx_explicitly_unbound = GL_FALSE;

    mglFlushPendingDraws(ctx);
    mglResetCommandBufferForContext(ctx, &ctx->draw_command_buffer);

    mglHashTableForEach(&ctx->state.program_table, mglDestroyContextProgram, ctx);
    mglHashTableForEach(&ctx->state.shader_table, mglDestroyContextShader, ctx);
    mglHashTableForEach(&ctx->state.texture_table, mglDestroyContextTexture, ctx);
    mglHashTableForEach(&ctx->state.buffer_table, mglDestroyContextBuffer, ctx);
    mglHashTableForEach(&ctx->state.sampler_table, mglDestroyContextSampler, ctx);
    mglHashTableForEach(&ctx->state.renderbuffer_table, mglDestroyContextRenderbuffer, ctx);
    mglHashTableForEach(&ctx->state.framebuffer_table, mglDestroyContextFramebuffer, ctx);
    mglHashTableForEach(&ctx->state.vao_table, mglDestroyContextVertexArray, ctx);
    mglHashTableForEach(&ctx->state.program_pipeline_table, mglDestroyContextProgramPipeline, ctx);
    mglHashTableForEach(&ctx->state.transform_feedback_table, mglDestroyContextTransformFeedback, ctx);
    mglHashTableForEach(&ctx->state.sync_table, mglDestroyContextSync, ctx);

    mglHashTableClearEntries(&ctx->state.program_table);
    mglHashTableClearEntries(&ctx->state.shader_table);
    mglHashTableClearEntries(&ctx->state.texture_table);
    mglHashTableClearEntries(&ctx->state.buffer_table);
    mglHashTableClearEntries(&ctx->state.sampler_table);
    mglHashTableClearEntries(&ctx->state.renderbuffer_table);
    mglHashTableClearEntries(&ctx->state.framebuffer_table);
    mglHashTableClearEntries(&ctx->state.vao_table);
    mglHashTableClearEntries(&ctx->state.program_pipeline_table);
    mglHashTableClearEntries(&ctx->state.transform_feedback_table);
    mglHashTableClearEntries(&ctx->state.sync_table);

    // CRITICAL FIX: Use hash-table owned cleanup to avoid freeing non-owned/corrupted pointers.
    #define MGL_FREE_HASH_TABLE(_tbl_) destroyHashTable(&(_tbl_))

    // 1. Basic cleanup of programs and shaders (major memory leaks)
    MGL_FREE_HASH_TABLE(ctx->state.program_table);
    MGL_FREE_HASH_TABLE(ctx->state.shader_table);

    // 2. Basic cleanup of textures (major memory leaks)
    MGL_FREE_HASH_TABLE(ctx->state.texture_table);

    // 3. Basic cleanup of buffers (major memory leaks)
    MGL_FREE_HASH_TABLE(ctx->state.buffer_table);

    // CRITICAL FIX: Basic cleanup for remaining hash tables to prevent major memory leaks
    MGL_FREE_HASH_TABLE(ctx->state.renderbuffer_table);
    MGL_FREE_HASH_TABLE(ctx->state.framebuffer_table);
    MGL_FREE_HASH_TABLE(ctx->state.vao_table);
    MGL_FREE_HASH_TABLE(ctx->state.sampler_table);
    MGL_FREE_HASH_TABLE(ctx->state.program_pipeline_table);
    MGL_FREE_HASH_TABLE(ctx->state.transform_feedback_table);
    MGL_FREE_HASH_TABLE(ctx->state.sync_table);

    #undef MGL_FREE_HASH_TABLE

    mglRendererBackendDestroy(
        (MGLRendererBackendHandle **)&ctx->renderer_backend);
    if (ctx->platform_renderer_shell) {
        CFRelease(ctx->platform_renderer_shell);
        ctx->platform_renderer_shell = NULL;
    }

    if (save == ctx) {
        _ctx = NULL;
        _ctx_explicitly_unbound = GL_TRUE;
    } else {
        _ctx = save;
        _ctx_explicitly_unbound = (save == NULL) ? GL_TRUE : GL_FALSE;
    }

    printf("MGL INFO: Context cleanup completed successfully\n");
    free(ctx);
}

// CRITICAL FIX: Library destructor for proper cleanup.
// project_memory hard constraint: mgl_auto_cleanup must call destroyGLMContext
// when _ctx != NULL to prevent Metal object leaks in dlopen/dlclose scenarios.
// The backend operation context retains MGLRenderer. Clearing _ctx before the
// call lets destroyGLMContext's save/restore logic leave the TLS slot clean.
__attribute__((destructor))
static void mgl_auto_cleanup(void)
{
    if (_ctx != NULL) {
        fprintf(stderr, "MGL INFO: Auto-cleanup - destroying GLMContext\n");

        GLMContext ctx_to_destroy = _ctx;
        _ctx = NULL;
        _ctx_explicitly_unbound = GL_TRUE;
        destroyGLMContext(ctx_to_destroy);

        fprintf(stderr, "MGL INFO: Auto-cleanup completed\n");
    }
}
