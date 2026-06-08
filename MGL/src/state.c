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
 * state.c
 * MGL
 *
 */

#include "mgl.h"
#include "glm_context.h"
#include "mgl_safety.h"

#define ENABLE_CAP(_cap_)   ctx->state.caps._cap_ = true; break
#define DISABLE_CAP(_cap_)   ctx->state.caps._cap_ = false; break

static Framebuffer *mglGetSafeDrawFramebuffer(GLMContext ctx, const char *where);

static void mglSetAllBlendEnables(GLMContext ctx, GLboolean enabled)
{
    if (!ctx)
        return;

    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        ctx->state.caps.blendi[i] = enabled ? GL_TRUE : GL_FALSE;
    }
    ctx->state.caps.blend = enabled ? GL_TRUE : GL_FALSE;
}

static GLuint mglEffectiveMaxViewports(GLMContext ctx)
{
    GLuint max_viewports = ctx ? ctx->state.var.max_viewports : 1u;
    if (max_viewports == 0u || max_viewports > MGL_MAX_VIEWPORTS)
        max_viewports = MGL_MAX_VIEWPORTS;
    return max_viewports ? max_viewports : 1u;
}

static void mglSetAllScissorEnables(GLMContext ctx, GLboolean enabled)
{
    if (!ctx)
        return;

    for (GLuint i = 0; i < MGL_MAX_VIEWPORTS; i++)
    {
        ctx->state.caps.scissor_testi[i] = enabled ? GL_TRUE : GL_FALSE;
    }
    ctx->state.caps.scissor_test = enabled ? GL_TRUE : GL_FALSE;
}

static void mglUpdateGlobalScissorEnableFromIndexZero(GLMContext ctx)
{
    if (!ctx)
        return;

    ctx->state.caps.scissor_test = ctx->state.caps.scissor_testi[0] ? GL_TRUE : GL_FALSE;
}

static void mglLogMinecraftOffscreenViewport(GLMContext ctx,
                                             GLint x,
                                             GLint y,
                                             GLsizei width,
                                             GLsizei height)
{
    if (!ctx || width <= 0 || height <= 0)
        return;

    if (!((width == 392 && height == 560) ||
          (width == 1024 && height == 1024) ||
          (width <= 560 && height <= 1024 && ctx->state.framebuffer != NULL)))
        return;

    static uint64_t s_offscreenViewportLogCount = 0;
    uint64_t hit = ++s_offscreenViewportLogCount;
    if (hit > 96ull && (hit % 512ull) != 0ull)
        return;

    Framebuffer *fbo = mglGetSafeDrawFramebuffer(ctx, "Viewport.offscreen");
    FBOAttachment *color0 = fbo ? &fbo->color_attachments[0] : NULL;
    Texture *colorTex = NULL;
    if (color0) {
        colorTex = (color0->textarget == GL_RENDERBUFFER && color0->buf.rbo)
            ? color0->buf.rbo->tex
            : color0->buf.tex;
    }

    Texture *depthTex = NULL;
    if (fbo) {
        depthTex = (fbo->depth.textarget == GL_RENDERBUFFER && fbo->depth.buf.rbo)
            ? fbo->depth.buf.rbo->tex
            : fbo->depth.buf.tex;
    }

    fprintf(stderr,
            "MGL VIEWPORT CALL offscreen hit=%llu fbo=%u drawBuf=0x%x viewport=(%d,%d,%d,%d) "
            "color0=%u(%ux%u target=0x%x) depth=%u(%ux%u target=0x%x) dirty=0x%x pendingDraws=%u\n",
            (unsigned long long)hit,
            fbo ? fbo->name : 0u,
            ctx->state.draw_buffer,
            x, y, width, height,
            colorTex ? colorTex->name : 0u,
            colorTex ? colorTex->width : 0u,
            colorTex ? colorTex->height : 0u,
            colorTex ? colorTex->target : 0u,
            depthTex ? depthTex->name : 0u,
            depthTex ? depthTex->width : 0u,
            depthTex ? depthTex->height : 0u,
            depthTex ? depthTex->target : 0u,
            ctx->state.dirty_bits,
            ctx->draw_command_buffer.total_commands);
}

static Framebuffer *mglGetSafeDrawFramebuffer(GLMContext ctx, const char *where)
{
    Framebuffer *fbo;

    if (!ctx)
        return NULL;

    fbo = ctx->state.framebuffer;
    if (!fbo)
        return NULL;

    if (!mglObjectPointerLooksPlausible(fbo) ||
        !mglHashTableContainsData(&ctx->state.framebuffer_table, fbo) ||
        !mglPointerRangeIsReadable(fbo, sizeof(*fbo)))
    {
        fprintf(stderr, "MGL WARNING: %s dropping invalid draw framebuffer pointer %p\n",
                where ? where : "state",
                (void *)fbo);
        if (ctx->state.readbuffer == fbo)
            ctx->state.readbuffer = NULL;
        ctx->state.framebuffer = NULL;
        return NULL;
    }

    return fbo;
}

static void mglRecomputeGlobalBlendEnable(GLMContext ctx)
{
    GLboolean enabled = GL_FALSE;

    if (!ctx)
        return;

    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        if (ctx->state.caps.blendi[i])
        {
            enabled = GL_TRUE;
            break;
        }
    }

    ctx->state.caps.blend = enabled;
}

static GLboolean mglClipDistanceIndex(GLMContext ctx, GLenum cap, GLuint *index)
{
    if (cap < GL_CLIP_DISTANCE0 || cap > GL_CLIP_DISTANCE7)
        return GL_FALSE;

    GLuint idx = (GLuint)(cap - GL_CLIP_DISTANCE0);
    GLuint limit = ctx ? ctx->state.var.max_clip_distances : MAX_CLIP_DISTANCES;
    if (limit == 0 || limit > MAX_CLIP_DISTANCES)
        limit = MAX_CLIP_DISTANCES;
    if (idx >= limit)
        return GL_FALSE;

    if (index)
        *index = idx;
    return GL_TRUE;
}

void mglDisable(GLMContext ctx, GLenum cap)
{
    GLuint clipIndex = 0;

    if (mglClipDistanceIndex(ctx, cap, &clipIndex))
    {
        ctx->state.caps.clip_distances[clipIndex] = GL_FALSE;
        ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_RENDER_STATE;
        return;
    }

    switch(cap)
    {
        case GL_BLEND:
            mglSetAllBlendEnables(ctx, GL_FALSE);
            break;
        case GL_LINE_SMOOTH: DISABLE_CAP(line_smooth);
        case GL_POLYGON_SMOOTH: DISABLE_CAP(polygon_smooth);
        case GL_CULL_FACE: DISABLE_CAP(cull_face);
        case GL_DEPTH_TEST: DISABLE_CAP(depth_test);
        case GL_STENCIL_TEST: DISABLE_CAP(stencil_test);
        case GL_DITHER: DISABLE_CAP(dither);
        case GL_SCISSOR_TEST:
            mglSetAllScissorEnables(ctx, GL_FALSE);
            break;
        case GL_COLOR_LOGIC_OP: DISABLE_CAP(color_logic_op);
        case GL_POLYGON_OFFSET_POINT: DISABLE_CAP(polygon_offset_point);
        case GL_POLYGON_OFFSET_LINE: DISABLE_CAP(polygon_offset_line);
        case GL_POLYGON_OFFSET_FILL: DISABLE_CAP(polygon_offset_fill);
        case GL_MULTISAMPLE: DISABLE_CAP(multisample);
        case GL_SAMPLE_ALPHA_TO_COVERAGE: DISABLE_CAP(sample_alpha_to_coverage);
        case GL_SAMPLE_ALPHA_TO_ONE: DISABLE_CAP(sample_alpha_to_one);
        case GL_SAMPLE_COVERAGE: DISABLE_CAP(sample_coverage);
        case GL_RASTERIZER_DISCARD: DISABLE_CAP(rasterizer_discard);
        case GL_FRAMEBUFFER_SRGB: DISABLE_CAP(framebuffer_srgb);
        case GL_PRIMITIVE_RESTART: DISABLE_CAP(primitive_restart);
        case GL_DEPTH_CLAMP: DISABLE_CAP(depth_clamp);
        case GL_TEXTURE_CUBE_MAP_SEAMLESS: DISABLE_CAP(texture_cube_map_seamless);
        case GL_SAMPLE_MASK: DISABLE_CAP(sample_mask);
        case GL_SAMPLE_SHADING: DISABLE_CAP(sample_shading);
        case GL_PRIMITIVE_RESTART_FIXED_INDEX: DISABLE_CAP(primitive_restart_fixed_index);
        case GL_DEBUG_OUTPUT_SYNCHRONOUS: DISABLE_CAP(debug_output_synchronous);
        case GL_DEBUG_OUTPUT: DISABLE_CAP(debug_output);
        case GL_PROGRAM_POINT_SIZE: DISABLE_CAP(program_point_size);
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_RENDER_STATE;
    if (cap == GL_BLEND ||
        cap == GL_COLOR_LOGIC_OP ||
        cap == GL_SAMPLE_ALPHA_TO_COVERAGE ||
        cap == GL_SAMPLE_ALPHA_TO_ONE) {
        ctx->state.dirty_bits |= DIRTY_ALPHA_STATE;
    }
    if (cap == GL_DEPTH_TEST || cap == GL_STENCIL_TEST || cap == GL_FRAMEBUFFER_SRGB) {
        ctx->state.dirty_bits |= DIRTY_FBO;
        Framebuffer *fbo = mglGetSafeDrawFramebuffer(ctx, __FUNCTION__);
        if (fbo) {
            fbo->dirty_bits |= DIRTY_FBO_BINDING;
        }
    }
}

void mglEnable(GLMContext ctx, GLenum cap)
{
    GLuint clipIndex = 0;

    if (mglClipDistanceIndex(ctx, cap, &clipIndex))
    {
        ctx->state.caps.clip_distances[clipIndex] = GL_TRUE;
        ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_RENDER_STATE;
        return;
    }

    switch(cap)
    {
        case GL_BLEND:
            mglSetAllBlendEnables(ctx, GL_TRUE);
            break;
        case GL_LINE_SMOOTH: ENABLE_CAP(line_smooth);
        case GL_POLYGON_SMOOTH: ENABLE_CAP(polygon_smooth);
        case GL_CULL_FACE: ENABLE_CAP(cull_face);
        case GL_DEPTH_TEST: ENABLE_CAP(depth_test);
        case GL_STENCIL_TEST: ENABLE_CAP(stencil_test);
        case GL_DITHER: ENABLE_CAP(dither);
        case GL_SCISSOR_TEST:
            mglSetAllScissorEnables(ctx, GL_TRUE);
            break;
        case GL_COLOR_LOGIC_OP: ENABLE_CAP(color_logic_op);
        case GL_POLYGON_OFFSET_POINT: ENABLE_CAP(polygon_offset_point);
        case GL_POLYGON_OFFSET_LINE: ENABLE_CAP(polygon_offset_line);
        case GL_POLYGON_OFFSET_FILL: ENABLE_CAP(polygon_offset_fill);
        case GL_PROGRAM_POINT_SIZE: ENABLE_CAP(program_point_size);
        case GL_MULTISAMPLE: ENABLE_CAP(multisample);
        case GL_SAMPLE_ALPHA_TO_COVERAGE: ENABLE_CAP(sample_alpha_to_coverage);
        case GL_SAMPLE_ALPHA_TO_ONE: ENABLE_CAP(sample_alpha_to_one);
        case GL_SAMPLE_COVERAGE: ENABLE_CAP(sample_coverage);
        case GL_RASTERIZER_DISCARD: ENABLE_CAP(rasterizer_discard);
        case GL_FRAMEBUFFER_SRGB: ENABLE_CAP(framebuffer_srgb);
        case GL_PRIMITIVE_RESTART: ENABLE_CAP(primitive_restart);
        case GL_DEPTH_CLAMP: ENABLE_CAP(depth_clamp);
        case GL_TEXTURE_CUBE_MAP_SEAMLESS: ENABLE_CAP(texture_cube_map_seamless);
        case GL_SAMPLE_MASK: ENABLE_CAP(sample_mask);
        case GL_SAMPLE_SHADING: ENABLE_CAP(sample_shading);
        case GL_PRIMITIVE_RESTART_FIXED_INDEX: ENABLE_CAP(primitive_restart_fixed_index);
        case GL_DEBUG_OUTPUT_SYNCHRONOUS: ENABLE_CAP(debug_output_synchronous);
        case GL_DEBUG_OUTPUT: ENABLE_CAP(debug_output);
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_RENDER_STATE;
    if (cap == GL_BLEND ||
        cap == GL_COLOR_LOGIC_OP ||
        cap == GL_SAMPLE_ALPHA_TO_COVERAGE ||
        cap == GL_SAMPLE_ALPHA_TO_ONE) {
        ctx->state.dirty_bits |= DIRTY_ALPHA_STATE;
    }
    if (cap == GL_DEPTH_TEST || cap == GL_STENCIL_TEST || cap == GL_FRAMEBUFFER_SRGB) {
        ctx->state.dirty_bits |= DIRTY_FBO;
        Framebuffer *fbo = mglGetSafeDrawFramebuffer(ctx, __FUNCTION__);
        if (fbo) {
            fbo->dirty_bits |= DIRTY_FBO_BINDING;
        }
    }
}

void mglCullFace(GLMContext ctx, GLenum mode)
{
    switch(mode)
    {
        case GL_FRONT:
        case GL_BACK:
        case GL_FRONT_AND_BACK:
            ctx->state.var.cull_face_mode = mode;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglFrontFace(GLMContext ctx, GLenum mode)
{
    switch(mode)
    {
        case GL_CW:
        case GL_CCW:
            ctx->state.var.front_face = mode;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

#define HINT(_target_) ctx->state.hints._target_ = mode; break;
void mglHint(GLMContext ctx, GLenum target, GLenum mode)
{
    switch(target)
    {
        case GL_LINE_SMOOTH_HINT: HINT(line_smooth_hint);
        case GL_POLYGON_SMOOTH_HINT: HINT(polygon_smooth_hint)
        case GL_TEXTURE_COMPRESSION_HINT: HINT(texture_compression_hint);
        case GL_FRAGMENT_SHADER_DERIVATIVE_HINT: HINT(fragment_shader_derivative_hint);
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
}

void mglLineWidth(GLMContext ctx, GLfloat width)
{
    ERROR_CHECK_RETURN(width > 0.0f, GL_INVALID_VALUE);

    ctx->state.var.line_width = width;

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglPointSize(GLMContext ctx, GLfloat size)
{
    ERROR_CHECK_RETURN(size > 0.0f, GL_INVALID_VALUE);

    ctx->state.var.point_size = size;

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglPolygonMode(GLMContext ctx, GLenum face, GLenum mode)
{
    if (face != GL_FRONT_AND_BACK)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    switch(mode)
    {
        case GL_POINT:
        case GL_LINE:
        case GL_FILL:
            ctx->state.var.polygon_mode = mode;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglScissor(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height)
{
    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height >= 0, GL_INVALID_VALUE);

    ctx->state.var.scissor_box[0] = x;
    ctx->state.var.scissor_box[1] = y;
    ctx->state.var.scissor_box[2] = width;
    ctx->state.var.scissor_box[3] = height;
    ctx->state.scissor_box_array[0][0] = x;
    ctx->state.scissor_box_array[0][1] = y;
    ctx->state.scissor_box_array[0][2] = width;
    ctx->state.scissor_box_array[0][3] = height;

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglLogicOp(GLMContext ctx, GLenum opcode)
{
    switch(opcode)
    {
        case GL_CLEAR:
        case GL_SET:
        case GL_COPY:
        case GL_COPY_INVERTED:
        case GL_NOOP:
        case GL_AND:
        case GL_NAND:
        case GL_OR:
        case GL_NOR:
        case GL_XOR:
        case GL_EQUIV:
        case GL_AND_REVERSE:
        case GL_AND_INVERTED:
        case GL_OR_REVERSE:
        case GL_OR_INVERTED:
            ctx->state.var.logic_op_mode = opcode;
            ctx->state.var.logic_op = opcode;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglStencilFunc(GLMContext ctx, GLenum func, GLint ref, GLuint mask)
{
    switch(func)
    {
        case GL_LEQUAL:
        case GL_GEQUAL:
        case GL_LESS:
        case GL_GREATER:
        case GL_EQUAL:
        case GL_NOTEQUAL:
        case GL_ALWAYS:
        case GL_NEVER:
            ctx->state.var.stencil_func = func;
            ctx->state.var.stencil_back_func = func;
            ctx->state.var.stencil_ref = ref;
            ctx->state.var.stencil_back_ref = ref;
            ctx->state.var.stencil_value_mask = mask;
            ctx->state.var.stencil_back_value_mask = mask;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

static bool validStencilOpSeparate(GLMContext ctx, GLenum op)
{
    switch(op)
    {
        case GL_KEEP:
        case GL_ZERO:
        case GL_REPLACE:
        case GL_INCR:
        case GL_INCR_WRAP:
        case GL_DECR:
        case GL_DECR_WRAP:
        case GL_INVERT:
            return true;
    }

    return false;
}

void mglStencilOp(GLMContext ctx, GLenum fail, GLenum zfail, GLenum zpass)
{
    ERROR_CHECK_RETURN(validStencilOpSeparate(ctx, fail), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(validStencilOpSeparate(ctx, zfail), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(validStencilOpSeparate(ctx, zpass), GL_INVALID_ENUM);

    ctx->state.var.stencil_fail = fail;
    ctx->state.var.stencil_pass_depth_fail = zfail;
    ctx->state.var.stencil_pass_depth_pass = zpass;
    ctx->state.var.stencil_back_fail = fail;
    ctx->state.var.stencil_back_pass_depth_fail = zfail;
    ctx->state.var.stencil_back_pass_depth_pass = zpass;

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}


void mglStencilMask(GLMContext ctx, GLuint mask)
{
    ctx->state.var.stencil_writemask = mask;
    ctx->state.var.stencil_back_writemask = mask;

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglColorMask(GLMContext ctx, GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha)
{
    if (!ctx) {
        return;
    }

    red = red ? GL_TRUE : GL_FALSE;
    green = green ? GL_TRUE : GL_FALSE;
    blue = blue ? GL_TRUE : GL_FALSE;
    alpha = alpha ? GL_TRUE : GL_FALSE;

    if (red == GL_FALSE || green == GL_FALSE || blue == GL_FALSE || alpha == GL_FALSE)
    {
        for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
        {
            STATE(caps.use_color_mask[i]) = true;

            ctx->state.var.color_writemask[i][0] = red;
            ctx->state.var.color_writemask[i][1] = green;
            ctx->state.var.color_writemask[i][2] = blue;
            ctx->state.var.color_writemask[i][3] = alpha;

        }
    }
    else
    {
        for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
        {
            STATE(caps.use_color_mask[i]) = false;

            ctx->state.var.color_writemask[i][0] = GL_TRUE;
            ctx->state.var.color_writemask[i][1] = GL_TRUE;
            ctx->state.var.color_writemask[i][2] = GL_TRUE;
            ctx->state.var.color_writemask[i][3] = GL_TRUE;
        }
    }

    // Metal color write masks are baked into MTLRenderPipelineState, not a
    // dynamic encoder state. Rebuild the pipeline whenever GL_COLOR_WRITEMASK
    // changes; otherwise a depth-only pass can poison later color draws.
    ctx->state.dirty_bits |= DIRTY_RENDER_STATE | DIRTY_ALPHA_STATE;
}

void mglDepthMask(GLMContext ctx, GLboolean flag)
{
    ctx->state.var.depth_writemask = flag ? GL_TRUE : GL_FALSE;

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglStencilOpSeparate(GLMContext ctx, GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass)
{
    ERROR_CHECK_RETURN(validStencilOpSeparate(ctx, sfail), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(validStencilOpSeparate(ctx, dpfail), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(validStencilOpSeparate(ctx, dppass), GL_INVALID_ENUM);

    switch(face)
    {
        case GL_FRONT:
            ctx->state.var.stencil_fail = sfail;
            ctx->state.var.stencil_pass_depth_fail = dpfail;
            ctx->state.var.stencil_pass_depth_pass = dppass;
            break;

        case GL_BACK:
            ctx->state.var.stencil_back_fail = sfail;
            ctx->state.var.stencil_back_pass_depth_fail = dpfail;
            ctx->state.var.stencil_back_pass_depth_pass = dppass;
            break;

        case GL_FRONT_AND_BACK:
            ctx->state.var.stencil_fail = sfail;
            ctx->state.var.stencil_pass_depth_fail = dpfail;
            ctx->state.var.stencil_pass_depth_pass = dppass;
            ctx->state.var.stencil_back_fail = sfail;
            ctx->state.var.stencil_back_pass_depth_fail = dpfail;
            ctx->state.var.stencil_back_pass_depth_pass = dppass;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglStencilFuncSeparate(GLMContext ctx, GLenum face, GLenum func, GLint ref, GLuint mask)
{
    switch(func)
    {
        case GL_LEQUAL:
        case GL_GEQUAL:
        case GL_LESS:
        case GL_GREATER:
        case GL_EQUAL:
        case GL_NOTEQUAL:
        case GL_ALWAYS:
        case GL_NEVER:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    switch(face)
    {
        case GL_FRONT:
            ctx->state.var.stencil_func = func;
            ctx->state.var.stencil_ref = ref;
            ctx->state.var.stencil_value_mask = mask;
            break;

        case GL_BACK:
            ctx->state.var.stencil_back_func = func;
            ctx->state.var.stencil_back_ref = ref;
            ctx->state.var.stencil_back_value_mask = mask;
            break;

        case GL_FRONT_AND_BACK:
            ctx->state.var.stencil_func = func;
            ctx->state.var.stencil_ref = ref;
            ctx->state.var.stencil_value_mask = mask;
            ctx->state.var.stencil_back_func = func;
            ctx->state.var.stencil_back_ref = ref;
            ctx->state.var.stencil_back_value_mask = mask;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglStencilMaskSeparate(GLMContext ctx, GLenum face, GLuint mask)
{
    switch(face)
    {
        case GL_FRONT:
            ctx->state.var.stencil_writemask = mask;
            break;

        case GL_BACK:
            ctx->state.var.stencil_back_writemask = mask;
            break;

        case GL_FRONT_AND_BACK:
            ctx->state.var.stencil_writemask = mask;
            ctx->state.var.stencil_back_writemask = mask;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglDepthFunc(GLMContext ctx, GLenum func)
{
    switch(func)
    {
        case GL_LEQUAL:
        case GL_GEQUAL:
        case GL_LESS:
        case GL_GREATER:
        case GL_EQUAL:
        case GL_NOTEQUAL:
        case GL_ALWAYS:
        case GL_NEVER:
            ctx->state.var.depth_func = func;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

static GLdouble _clamp(GLdouble a)
{
    if (a < 0.0)
        a = 0.0;
    else if (a > 1.0)
        a = 1.0;

    return a;
}

void mglDepthRange(GLMContext ctx, GLdouble n, GLdouble f)
{
    n = _clamp(n);
    f = _clamp(f);

    ctx->state.var.depth_range[0] = n;
    ctx->state.var.depth_range[1] = f;
    ctx->state.depth_range_array[0][0] = n;
    ctx->state.depth_range_array[0][1] = f;

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

void mglViewport(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height)
{
    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height >= 0, GL_INVALID_VALUE);

    if (ctx &&
        (ctx->state.viewport[0] != x ||
         ctx->state.viewport[1] != y ||
         ctx->state.viewport[2] != width ||
         ctx->state.viewport[3] != height)) {
        mglFlushPendingDraws(ctx);
    }

    ctx->state.viewport[0] = x;
    ctx->state.viewport[1] = y;
    ctx->state.viewport[2] = width;
    ctx->state.viewport[3] = height;
    ctx->state.viewport_array[0][0] = (GLfloat)x;
    ctx->state.viewport_array[0][1] = (GLfloat)y;
    ctx->state.viewport_array[0][2] = (GLfloat)width;
    ctx->state.viewport_array[0][3] = (GLfloat)height;

    mglLogMinecraftOffscreenViewport(ctx, x, y, width, height);

    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
}

#define RET_VAR(_VAR_, _DEFAULT_)  return (ctx->state.var._VAR_ == _DEFAULT_)
#define RET_CAP(_CAP_)  return ctx->state.caps._CAP_

GLboolean mglIsEnabled(GLMContext ctx, GLenum cap)
{
    GLuint clipIndex = 0;
    if (mglClipDistanceIndex(ctx, cap, &clipIndex))
    {
        return ctx->state.caps.clip_distances[clipIndex];
    }

    switch(cap)
    {
        case GL_BLEND: RET_CAP(blend);
        case GL_COLOR_LOGIC_OP: RET_CAP(color_logic_op);
        case GL_CULL_FACE: RET_CAP(cull_face);
        case GL_DEPTH_CLAMP: RET_CAP(depth_clamp);
        case GL_DEBUG_OUTPUT: RET_CAP(debug_output);
        case GL_DEBUG_OUTPUT_SYNCHRONOUS: RET_CAP(debug_output_synchronous);
        case GL_DEPTH_TEST: RET_CAP(depth_test);
        case GL_DITHER: RET_CAP(dither);
        case GL_FRAMEBUFFER_SRGB: RET_CAP(framebuffer_srgb);
        case GL_LINE_SMOOTH: RET_CAP(line_smooth);
        case GL_MULTISAMPLE: RET_CAP(multisample);
        case GL_POLYGON_SMOOTH: RET_CAP(polygon_smooth);
        case GL_POLYGON_OFFSET_FILL: RET_CAP(polygon_offset_fill);
        case GL_POLYGON_OFFSET_LINE: RET_CAP(polygon_offset_line);
        case GL_POLYGON_OFFSET_POINT: RET_CAP(polygon_offset_point);
        case GL_PROGRAM_POINT_SIZE: RET_CAP(program_point_size);
        case GL_PRIMITIVE_RESTART: RET_CAP(primitive_restart);
        case GL_PRIMITIVE_RESTART_FIXED_INDEX: RET_CAP(primitive_restart_fixed_index);
        case GL_SAMPLE_ALPHA_TO_COVERAGE: RET_CAP(sample_alpha_to_coverage);
        case GL_SAMPLE_ALPHA_TO_ONE: RET_CAP(sample_alpha_to_one);
        case GL_SAMPLE_COVERAGE: RET_CAP(sample_coverage);
        case GL_SAMPLE_MASK: RET_CAP(sample_mask);
        case GL_SCISSOR_TEST: RET_CAP(scissor_test);
        case GL_STENCIL_TEST: RET_CAP(stencil_test);
        case GL_TEXTURE_CUBE_MAP_SEAMLESS: RET_CAP(texture_cube_map_seamless);

        default:
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, GL_FALSE);
    }

    return false;
}

void mglEnablei(GLMContext ctx, GLenum target, GLuint index)
{
    if (!ctx)
        return;

    if (target == GL_BLEND)
    {
        if (index < MAX_COLOR_ATTACHMENTS)
        {
            ctx->state.caps.blendi[index] = GL_TRUE;
            mglRecomputeGlobalBlendEnable(ctx);
            ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;

            return;
        }

        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (target == GL_SCISSOR_TEST)
    {
        if (index < mglEffectiveMaxViewports(ctx))
        {
            ctx->state.caps.scissor_testi[index] = GL_TRUE;
            if (index == 0)
                mglUpdateGlobalScissorEnableFromIndexZero(ctx);
            ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_RENDER_STATE;
            return;
        }

        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglDisablei(GLMContext ctx, GLenum target, GLuint index)
{
    if (!ctx)
        return;

    if (target == GL_BLEND)
    {
        if (index < MAX_COLOR_ATTACHMENTS)
        {
            ctx->state.caps.blendi[index] = GL_FALSE;
            mglRecomputeGlobalBlendEnable(ctx);
            ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;

            return;
        }

        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (target == GL_SCISSOR_TEST)
    {
        if (index < mglEffectiveMaxViewports(ctx))
        {
            ctx->state.caps.scissor_testi[index] = GL_FALSE;
            if (index == 0)
                mglUpdateGlobalScissorEnableFromIndexZero(ctx);
            ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_RENDER_STATE;
            return;
        }

        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    ERROR_RETURN(GL_INVALID_ENUM);
}

GLboolean mglIsEnabledi(GLMContext ctx, GLenum target, GLuint index)
{
    if (!ctx)
        return GL_FALSE;

    if (target == GL_BLEND)
    {
        if (index < MAX_COLOR_ATTACHMENTS)
        {
            return ctx->state.caps.blendi[index];
        }

        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if (target == GL_SCISSOR_TEST)
    {
        if (index < mglEffectiveMaxViewports(ctx))
        {
            return ctx->state.caps.scissor_testi[index];
        }

        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
}

void mglClearDepthf(GLMContext ctx, GLfloat d)
{
    ctx->state.var.depth_clear_value = _clamp(d);
    ctx->state.dirty_bits |= DIRTY_STATE;
}

void mglBlendColor(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    ctx->state.var.blend_color[0] = red;
    ctx->state.var.blend_color[1] = green;
    ctx->state.var.blend_color[2] = blue;
    ctx->state.var.blend_color[3] = alpha;

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
}

void mglBlendEquation(GLMContext ctx, GLenum mode)
{
    switch(mode)
    {
        case GL_FUNC_ADD:
        case GL_FUNC_SUBTRACT:
        case GL_FUNC_REVERSE_SUBTRACT:
        case GL_MIN:
        case GL_MAX:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        ctx->state.var.blend_equation_rgb[i] = mode;
        ctx->state.var.blend_equation_alpha[i] = mode;
    }

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
}

void mglBlendEquationi(GLMContext ctx, GLuint buf, GLenum mode)
{
    switch(mode)
    {
        case GL_FUNC_ADD:
        case GL_FUNC_SUBTRACT:
        case GL_FUNC_REVERSE_SUBTRACT:
        case GL_MIN:
        case GL_MAX:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ERROR_CHECK_RETURN(buf >=0 && buf < MAX_COLOR_ATTACHMENTS, GL_INVALID_VALUE);

    ctx->state.var.blend_equation_rgb[buf] = mode;
    ctx->state.var.blend_equation_alpha[buf] = mode;

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
}

void mglBlendEquationSeparatei(GLMContext ctx, GLuint buf, GLenum modeRGB, GLenum modeAlpha)
{
    switch(modeRGB)
    {
        case GL_FUNC_ADD:
        case GL_FUNC_SUBTRACT:
        case GL_FUNC_REVERSE_SUBTRACT:
        case GL_MIN:
        case GL_MAX:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch(modeAlpha)
    {
        case GL_FUNC_ADD:
        case GL_FUNC_SUBTRACT:
        case GL_FUNC_REVERSE_SUBTRACT:
        case GL_MIN:
        case GL_MAX:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ERROR_CHECK_RETURN(buf >= 0 && buf < MAX_COLOR_ATTACHMENTS, GL_INVALID_VALUE);

    ctx->state.var.blend_equation_rgb[buf] = modeRGB;
    ctx->state.var.blend_equation_alpha[buf] = modeAlpha;

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
}

void mglBlendFunc(GLMContext ctx, GLenum sfactor, GLenum dfactor)
{
    switch(sfactor)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
        case GL_SRC_ALPHA_SATURATE:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch(dfactor)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        ctx->state.var.blend_src_rgb[i] = sfactor;
        ctx->state.var.blend_src_alpha[i] = sfactor;
        ctx->state.var.blend_dst_rgb[i] = dfactor;
        ctx->state.var.blend_dst_alpha[i] = dfactor;
    }

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
}

void mglBlendFunci(GLMContext ctx, GLuint buf, GLenum sfactor, GLenum dfactor)
{
    switch(sfactor)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
        case GL_SRC_ALPHA_SATURATE:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch(dfactor)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ERROR_CHECK_RETURN(buf >=0 && buf < MAX_COLOR_ATTACHMENTS, GL_INVALID_VALUE);

    ctx->state.var.blend_src_rgb[buf] = sfactor;
    ctx->state.var.blend_src_alpha[buf] = sfactor;
    ctx->state.var.blend_dst_rgb[buf] = dfactor;
    ctx->state.var.blend_dst_alpha[buf] = dfactor;

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
}

void mglBlendFuncSeparatei(GLMContext ctx, GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha)
{
    switch(srcRGB)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
        case GL_SRC_ALPHA_SATURATE:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch(dstRGB)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch(srcAlpha)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
        case GL_SRC_ALPHA_SATURATE:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch(dstAlpha)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    ERROR_CHECK_RETURN(buf < MAX_COLOR_ATTACHMENTS, GL_INVALID_VALUE);

    ctx->state.var.blend_src_rgb[buf] = srcRGB;
    ctx->state.var.blend_dst_rgb[buf] = dstRGB;
    ctx->state.var.blend_src_alpha[buf] = srcAlpha;
    ctx->state.var.blend_dst_alpha[buf] = dstAlpha;
    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
}

void mglBlendEquationSeparate(GLMContext ctx, GLenum modeRGB, GLenum modeAlpha)
{
    switch(modeRGB)
    {
        case GL_FUNC_ADD:
        case GL_FUNC_SUBTRACT:
        case GL_FUNC_REVERSE_SUBTRACT:
        case GL_MIN:
        case GL_MAX:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch(modeAlpha)
    {
        case GL_FUNC_ADD:
        case GL_FUNC_SUBTRACT:
        case GL_FUNC_REVERSE_SUBTRACT:
        case GL_MIN:
        case GL_MAX:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        ctx->state.var.blend_equation_rgb[i] = modeRGB;
        ctx->state.var.blend_equation_alpha[i] = modeAlpha;
    }

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
}


void mglGetPointerv(GLMContext ctx, GLenum pname, void **params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch (pname)
    {
        case GL_DEBUG_CALLBACK_FUNCTION:
        case GL_DEBUG_CALLBACK_USER_PARAM:
            *params = NULL;
            return;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}

void mglPolygonOffset(GLMContext ctx, GLfloat factor, GLfloat units)
{
    if (!ctx)
        return;

    ctx->state.var.polygon_offset_factor = factor;
    ctx->state.var.polygon_offset_units = units;
    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_RENDER_STATE;
}

void mglBlendFuncSeparate(GLMContext ctx, GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha)
{
    switch(sfactorRGB)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
        case GL_SRC_ALPHA_SATURATE:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch(dfactorRGB)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch(sfactorAlpha)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
        case GL_SRC_ALPHA_SATURATE:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch(dfactorAlpha)
    {
        case GL_ZERO:
        case GL_ONE:
        case GL_SRC_COLOR:
        case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR:
        case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA:
        case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA:
        case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR:
        case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA:
        case GL_ONE_MINUS_CONSTANT_ALPHA:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        ctx->state.var.blend_src_rgb[i] = sfactorRGB;
        ctx->state.var.blend_dst_rgb[i] = dfactorRGB;
        ctx->state.var.blend_src_alpha[i] = sfactorAlpha;
        ctx->state.var.blend_dst_alpha[i] = dfactorAlpha;
    }

    ctx->state.dirty_bits |= DIRTY_STATE | DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
}

void mglPointParameterf(GLMContext ctx, GLenum pname, GLfloat param)
{
    switch (pname)
    {
        case GL_POINT_FADE_THRESHOLD_SIZE:
        case 0x8126: // GL_POINT_SIZE_MIN
        case 0x8127: // GL_POINT_SIZE_MAX
            if (param < 0.0f)
                ERROR_RETURN(GL_INVALID_VALUE);
            return;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}

void mglPointParameterfv(GLMContext ctx, GLenum pname, const GLfloat *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch (pname)
    {
        case 0x8129: // GL_POINT_DISTANCE_ATTENUATION
            return;
        default:
            mglPointParameterf(ctx, pname, *params);
            return;
    }
}

void mglPointParameteri(GLMContext ctx, GLenum pname, GLint param)
{
    switch (pname)
    {
        case GL_POINT_FADE_THRESHOLD_SIZE:
        case 0x8126: // GL_POINT_SIZE_MIN
        case 0x8127: // GL_POINT_SIZE_MAX
            if (param < 0)
                ERROR_RETURN(GL_INVALID_VALUE);
            return;
        case GL_POINT_SPRITE_COORD_ORIGIN:
            if (param != GL_LOWER_LEFT && param != GL_UPPER_LEFT)
                ERROR_RETURN(GL_INVALID_ENUM);
            return;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}

void mglPointParameteriv(GLMContext ctx, GLenum pname, const GLint *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch (pname)
    {
        case 0x8129: // GL_POINT_DISTANCE_ATTENUATION
            return;
        default:
            mglPointParameteri(ctx, pname, *params);
            return;
    }
}
