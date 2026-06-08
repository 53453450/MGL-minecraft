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
 * get.c
 * MGL
 *
 */

#include <inttypes.h>
#include <limits.h>

#include "glm_context.h"
#include "mgl_extensions.h"
#include "mgl_safety.h"
#include "pixel_utils.h"

void mglGetIntegeri_v(GLMContext ctx, GLenum target, GLuint index, GLint *data);

static const char *kMglExtensions[] = {
    "GL_ARB_vertex_array_object",
    "GL_ARB_framebuffer_object",
    "GL_ARB_texture_storage",
    "GL_ARB_sampler_objects",
    "GL_ARB_uniform_buffer_object",
    "GL_ARB_draw_buffers",
    "GL_ARB_debug_output",
    "GL_KHR_debug",
    "GL_ARB_texture_buffer_object",
    "GL_ARB_texture_buffer_range",
    "GL_ARB_buffer_storage",
    "GL_ARB_direct_state_access"
};
static_assert((sizeof(kMglExtensions) / sizeof(kMglExtensions[0])) == MGL_NUM_EXTENSIONS,
              "MGL_NUM_EXTENSIONS must match kMglExtensions");

static GLint mglClampGetInteger64ToInt(GLint64 value, const char *label)
{
    if (value > INT_MAX) {
        fprintf(stderr,
                "MGL WARNING: %s value %" PRId64 " exceeds GLint range; clamping for glGetIntegeri_v\n",
                label ? label : "indexed get",
                value);
        return INT_MAX;
    }
    if (value < INT_MIN) {
        fprintf(stderr,
                "MGL WARNING: %s value %" PRId64 " below GLint range; clamping for glGetIntegeri_v\n",
                label ? label : "indexed get",
                value);
        return INT_MIN;
    }
    return (GLint)value;
}

GLsizei mglSafeMaxTextureSize(GLMContext ctx)
{
    const GLsizei kFallback = 16384;
    GLint maxTex = 0;

    if (!ctx) {
        return kFallback;
    }

    maxTex = ctx->state.var.max_texture_size;
    if (maxTex == 0x01010101 || maxTex <= 1024 || maxTex > 32768) {
        fprintf(stderr,
                "MGL WARNING: GL_MAX_TEXTURE_SIZE state value suspicious (%u), using safe fallback %d\n",
                (unsigned)maxTex, (int)kFallback);
        // Self-heal corrupted/uninitialized state so repeated queries stay stable.
        ctx->state.var.max_texture_size = kFallback;
        return kFallback;
    }

    return (GLsizei)maxTex;
}

static GLsizei mglSafeMaxTextureBufferSize(GLMContext ctx)
{
    const GLsizei kFallback = 1 << 20; // texels, conservative but useful for Minecraft cloud buffers
    GLuint value = ctx ? ctx->state.var.max_texture_buffer_size : 0u;

    if (value == 0u || value == 0x01010101u || value > (1u << 28)) {
        if (ctx) {
            fprintf(stderr,
                    "MGL WARNING: GL_MAX_TEXTURE_BUFFER_SIZE state value suspicious (%u), using safe fallback %d\n",
                    value,
                    (int)kFallback);
            ctx->state.var.max_texture_buffer_size = (GLuint)kFallback;
        }
        return kFallback;
    }

    return (GLsizei)value;
}

static GLsizei mglSafeTextureBufferOffsetAlignment(GLMContext ctx)
{
    const GLsizei kFallback = 16;
    GLuint value = ctx ? ctx->state.var.texture_buffer_offset_alignment : 0u;

    if (value == 0u || value == 0x01010101u || value > 4096u) {
        if (ctx) {
            fprintf(stderr,
                    "MGL WARNING: GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT state value suspicious (%u), using safe fallback %d\n",
                    value,
                    (int)kFallback);
            ctx->state.var.texture_buffer_offset_alignment = (GLuint)kFallback;
        }
        return kFallback;
    }

    return (GLsizei)value;
}

static GLuint mglSafeMaxViewports(GLMContext ctx)
{
    GLuint value = ctx ? ctx->state.var.max_viewports : 1u;
    if (value == 0u || value > MGL_MAX_VIEWPORTS) {
        value = MGL_MAX_VIEWPORTS;
    }
    return value ? value : 1u;
}

static GLuint mglSafeMaxVertexAttribBindings(GLMContext ctx)
{
    GLuint value = ctx ? ctx->state.var.max_vertex_attrib_bindings : 0u;
    if (value < MAX_ATTRIBS || value == 0x01010101u || value > MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
        value = MGL_MAX_VERTEX_ATTRIB_BINDINGS;
        if (ctx) {
            ctx->state.var.max_vertex_attrib_bindings = value;
        }
    }
    return value;
}

static GLuint mglSafeMaxVertexAttribRelativeOffset(GLMContext ctx)
{
    const GLuint kFallback = 2047u;
    GLuint value = ctx ? ctx->state.var.max_vertex_attrib_relative_offset : 0u;
    if (value < kFallback || value == 0x01010101u) {
        value = kFallback;
        if (ctx) {
            ctx->state.var.max_vertex_attrib_relative_offset = value;
        }
    }
    return value;
}

static GLuint mglSafeMaxVertexAttribStride(GLMContext ctx)
{
    (void)ctx;
    return 2048u;
}

static GLuint mglCurrentDrawFramebufferBinding(GLMContext ctx)
{
    GLuint name = (ctx && ctx->state.framebuffer) ? ctx->state.framebuffer->name : 0u;
    if (ctx) {
        ctx->state.var.draw_framebuffer_binding = name;
    }
    return name;
}

static GLuint mglCurrentReadFramebufferBinding(GLMContext ctx)
{
    GLuint name = (ctx && ctx->state.readbuffer) ? ctx->state.readbuffer->name : 0u;
    if (ctx) {
        ctx->state.var.read_framebuffer_binding = name;
    }
    return name;
}

// these cast a void ptr to a type and value
#define RET_BOOL(__value__) *((GLboolean *)data) = (GLboolean)__value__; break;
#define RET_INT(__value__) *((GLint *)data) = (GLint)__value__; break;
#define RET_FLOAT(__value__) *((GLfloat *)data) = (GLfloat)__value__; break;
#define RET_DOUBLE(__value__) *((GLdouble *)data) = (GLdouble)__value__; break;

enum {
    kBool, kInt, kFloat, kDouble
};

static void mglReturnPolygonMode(GLMContext ctx, GLuint type, void *data)
{
    GLuint mode = ctx ? ctx->state.var.polygon_mode : GL_FILL;

    switch(type) {
        case kBool:
            ((GLboolean *)data)[0] = (GLboolean)mode;
            ((GLboolean *)data)[1] = (GLboolean)mode;
            break;
        case kInt:
            ((GLint *)data)[0] = (GLint)mode;
            ((GLint *)data)[1] = (GLint)mode;
            break;
        case kFloat:
            ((GLfloat *)data)[0] = (GLfloat)mode;
            ((GLfloat *)data)[1] = (GLfloat)mode;
            break;
        case kDouble:
            ((GLdouble *)data)[0] = (GLdouble)mode;
            ((GLdouble *)data)[1] = (GLdouble)mode;
            break;
    }
}

// set value based on type from ctx->state.var
#define RET_TYPE_VAR(__TYPE__, __VALUE__) \
switch(type) {  \
case kBool: RET_BOOL(ctx->state.var.__VALUE__);   \
    case kInt: RET_INT(ctx->state.var.__VALUE__)    \
    case kFloat: RET_FLOAT(ctx->state.var.__VALUE__)    \
    case kDouble: RET_DOUBLE(ctx->state.var.__VALUE__)    \
}

// set count values based on type
#define RET_TYPE_VAR_COUNT(__TYPE__, __VALUE__, __COUNT__) \
for(int i=0, counts[]={1,4,4,8};i<__COUNT__; data+=counts[__TYPE__], i++) \
    switch(type) {  \
        case kBool: RET_BOOL(ctx->state.var.__VALUE__[i])    \
        case kInt: RET_INT(ctx->state.var.__VALUE__[i])    \
        case kFloat: RET_FLOAT(ctx->state.var.__VALUE__[i])    \
        case kDouble: RET_DOUBLE(ctx->state.var.__VALUE__[i])    \
}

// set value based on type from ctx->state not ctx->state.var
#define RET_TYPE(__TYPE__, __VALUE__) \
switch(type) {  \
    case kBool: RET_BOOL(ctx->state.__VALUE__)    \
    case kInt: RET_INT(ctx->state.__VALUE__)    \
    case kFloat: RET_FLOAT(ctx->state.__VALUE__)    \
    case kDouble: RET_DOUBLE(ctx->state.__VALUE__)    \
}

// set count values based on type
#define RET_TYPE_COUNT(__TYPE__, __VALUE__, __COUNT__) \
for(int i=0, counts[]={1,4,4,8};i<__COUNT__; data+=counts[__TYPE__], i++) \
    switch(type) {  \
        case kBool: RET_BOOL(ctx->state.__VALUE__[i])    \
        case kInt: RET_INT(ctx->state.__VALUE__[i])    \
        case kFloat: RET_FLOAT(ctx->state.__VALUE__[i])    \
        case kDouble: RET_DOUBLE(ctx->state.__VALUE__[i])    \
}

static void mglGet(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    if (pname >= GL_DRAW_BUFFER0 &&
        pname < (GL_DRAW_BUFFER0 + MAX_COLOR_ATTACHMENTS))
    {
        GLuint index = pname - GL_DRAW_BUFFER0;
        GLuint maxDrawBuffers = ctx->state.var.max_draw_buffers;
        if (maxDrawBuffers == 0 || maxDrawBuffers > MAX_COLOR_ATTACHMENTS) {
            maxDrawBuffers = MAX_COLOR_ATTACHMENTS;
        }
        if (index >= maxDrawBuffers) {
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
        }

        GLenum value = (index < (GLuint)ctx->state.draw_buffer_count)
            ? ctx->state.draw_buffers[index]
            : GL_NONE;
        switch(type) {
            case kBool: RET_BOOL(value);
            case kInt: RET_INT(value);
            case kFloat: RET_FLOAT(value);
            case kDouble: RET_DOUBLE(value);
        }
    }

    switch(pname)
    {
        case 0x0B11: RET_TYPE_VAR(type, point_size); break; // GL_POINT_SIZE
        case 0x0B12: RET_TYPE_VAR(type, point_size_range); break; // GL_POINT_SIZE_RANGE
        case 0x0B13: RET_TYPE_VAR(type, point_size_granularity); break; // GL_POINT_SIZE_GRANULARITY
        case 0x0B21: RET_TYPE_VAR(type, line_width); break; // GL_LINE_WIDTH
        case 0x0B22: RET_TYPE_VAR(type, line_width_range); break; // GL_LINE_WIDTH_RANGE
        case 0x0B23: RET_TYPE_VAR(type, line_width_granularity); break; // GL_LINE_WIDTH_GRANULARITY
        case 0x0B40: mglReturnPolygonMode(ctx, type, data); break; // GL_POLYGON_MODE
        case 0x0B45: RET_TYPE_VAR(type, cull_face_mode); break; // GL_CULL_FACE_MODE
        case 0x0B46: RET_TYPE_VAR(type, front_face); break; // GL_FRONT_FACE

        case 0x0B70: RET_TYPE_VAR_COUNT(type, depth_range, 2); break; // GL_DEPTH_RANGE

        case 0x0B72: RET_TYPE_VAR(type, depth_writemask); break; // GL_DEPTH_WRITEMASK
        case 0x0B73: RET_TYPE_VAR(type, depth_clear_value); break; // GL_DEPTH_CLEAR_VALUE
        case 0x0B74: RET_TYPE_VAR(type, depth_func); break; // GL_DEPTH_FUNC
        case 0x0B91: RET_TYPE_VAR(type, stencil_clear_value); break; // GL_STENCIL_CLEAR_VALUE
        case 0x0B92: RET_TYPE_VAR(type, stencil_func); break; // GL_STENCIL_FUNC
        case 0x0B93: RET_TYPE_VAR(type, stencil_value_mask); break; // GL_STENCIL_VALUE_MASK
        case 0x0B94: RET_TYPE_VAR(type, stencil_fail); break; // GL_STENCIL_FAIL
        case 0x0B95: RET_TYPE_VAR(type, stencil_pass_depth_fail); break; // GL_STENCIL_PASS_DEPTH_FAIL
        case 0x0B96: RET_TYPE_VAR(type, stencil_pass_depth_pass); break; // GL_STENCIL_PASS_DEPTH_PASS
        case 0x0B97: RET_TYPE_VAR(type, stencil_ref); break; // GL_STENCIL_REF
        case 0x0B98: RET_TYPE_VAR(type, stencil_writemask); break; // GL_STENCIL_WRITEMASK

        case 0x0BA2: RET_TYPE_COUNT(type, viewport, 4); break; // GL_VIEWPORT

        case 0x0BE0: RET_TYPE_VAR(type, blend_dst_rgb[0]); break; // GL_BLEND_DST
        case 0x0BE1: RET_TYPE_VAR(type, blend_src_rgb[0]); break; // GL_BLEND_SRC
        case 0x0BE2: // GL_BLEND
            switch(type) {
                case kBool: RET_BOOL(ctx->state.caps.blend);
                case kInt: RET_INT(ctx->state.caps.blend);
                case kFloat: RET_FLOAT(ctx->state.caps.blend);
                case kDouble: RET_DOUBLE(ctx->state.caps.blend);
            }
            break;

        case 0x0BF0: RET_TYPE_VAR(type, logic_op_mode); break; // GL_LOGIC_OP_MODE
        case 0x0C01: RET_TYPE(type, draw_buffer); break; // GL_DRAW_BUFFER
        case 0x0C02: RET_TYPE(type, read_buffer); break; // GL_READ_BUFFER

        case 0x0C10: RET_TYPE_VAR_COUNT(type, scissor_box, 4); break; // GL_SCISSOR_BOX

        case 0x0C22: RET_TYPE_COUNT(type, color_clear_value, 4); break; // GL_COLOR_CLEAR_VALUE

        case 0x0C23: RET_TYPE_VAR_COUNT(type, color_writemask[0], 4); break; // GL_COLOR_WRITEMASK

        case 0x0D33: { // GL_MAX_TEXTURE_SIZE
            GLsizei maxTex = mglSafeMaxTextureSize(ctx);
            switch(type) {
                case kBool: RET_BOOL(maxTex);
                case kInt: RET_INT(maxTex);
                case kFloat: RET_FLOAT(maxTex);
                case kDouble: RET_DOUBLE(maxTex);
            }
            break;
        }
        case 0x0D3A: RET_TYPE_VAR(type, max_viewport_dims); break; // GL_MAX_VIEWPORT_DIMS
        case 0x0D50: RET_TYPE_VAR(type, subpixel_bits); break; // GL_SUBPIXEL_BITS
        case 0x0B00: RET_TYPE_VAR(type, current_color); break; // GL_CURRENT_COLOR
        case 0x0B01: RET_TYPE_VAR(type, current_index); break; // GL_CURRENT_INDEX
        case 0x0B02: RET_TYPE_VAR(type, current_normal); break; // GL_CURRENT_NORMAL
        case 0x0B04: RET_TYPE_VAR(type, current_raster_color); break; // GL_CURRENT_RASTER_COLOR
        case 0x0B05: RET_TYPE_VAR(type, current_raster_index); break; // GL_CURRENT_RASTER_INDEX
        case 0x0B06: RET_TYPE_VAR(type, current_raster_texture_coords); break; // GL_CURRENT_RASTER_TEXTURE_COORDS
        case 0x0B07: RET_TYPE_VAR(type, current_raster_position); break; // GL_CURRENT_RASTER_POSITION
        case 0x0B08: RET_TYPE_VAR(type, current_raster_position_valid); break; // GL_CURRENT_RASTER_POSITION_VALID
        case 0x0B09: RET_TYPE_VAR(type, current_raster_distance); break; // GL_CURRENT_RASTER_DISTANCE
        case 0x0B25: RET_TYPE_VAR(type, line_stipple_pattern); break; // GL_LINE_STIPPLE_PATTERN
        case 0x0B26: RET_TYPE_VAR(type, line_stipple_repeat); break; // GL_LINE_STIPPLE_REPEAT
        case 0x0B30: RET_TYPE_VAR(type, list_mode); break; // GL_LIST_MODE
        case 0x0B31: RET_TYPE_VAR(type, max_list_nesting); break; // GL_MAX_LIST_NESTING
        case 0x0B32: RET_TYPE_VAR(type, list_base); break; // GL_LIST_BASE
        case 0x0B33: RET_TYPE_VAR(type, list_index); break; // GL_LIST_INDEX
        case 0x0B43: RET_TYPE_VAR(type, edge_flag); break; // GL_EDGE_FLAG
        case 0x0B54: RET_TYPE_VAR(type, shade_model); break; // GL_SHADE_MODEL
        case 0x0B55: RET_TYPE_VAR(type, color_material_face); break; // GL_COLOR_MATERIAL_FACE
        case 0x0B56: RET_TYPE_VAR(type, color_material_parameter); break; // GL_COLOR_MATERIAL_PARAMETER
        case 0x0B80: RET_TYPE_VAR(type, accum_clear_value); break; // GL_ACCUM_CLEAR_VALUE
        case 0x0BA0: RET_TYPE_VAR(type, matrix_mode); break; // GL_MATRIX_MODE
        case 0x0BA3: RET_TYPE_VAR(type, modelview_stack_depth); break; // GL_MODELVIEW_STACK_DEPTH
        case 0x0BA4: RET_TYPE_VAR(type, projection_stack_depth); break; // GL_PROJECTION_STACK_DEPTH
        case 0x0BA5: RET_TYPE_VAR(type, texture_stack_depth); break; // GL_TEXTURE_STACK_DEPTH
        case 0x0BA6: RET_TYPE_VAR(type, modelview_matrix); break; // GL_MODELVIEW_MATRIX
        case 0x0BA7: RET_TYPE_VAR(type, projection_matrix); break; // GL_PROJECTION_MATRIX
        case 0x0BB0: RET_TYPE_VAR(type, attrib_stack_depth); break; // GL_ATTRIB_STACK_DEPTH
        case 0x0BC1: RET_TYPE_VAR(type, alpha_test_func); break; // GL_ALPHA_TEST_FUNC
        case 0x0BC2: RET_TYPE_VAR(type, alpha_test_ref); break; // GL_ALPHA_TEST_REF
        case 0x0BF1: RET_TYPE_VAR(type, logic_op); break; // GL_LOGIC_OP
        case 0x0C00: RET_TYPE_VAR(type, aux_buffers); break; // GL_AUX_BUFFERS
        case 0x0C20: RET_TYPE_VAR(type, index_clear_value); break; // GL_INDEX_CLEAR_VALUE
        case 0x0C21: RET_TYPE_VAR(type, index_writemask); break; // GL_INDEX_WRITEMASK
        case 0x0C30: RET_TYPE_VAR(type, index_mode); break; // GL_INDEX_MODE
        case 0x0C31: RET_TYPE_VAR(type, rgba_mode); break; // GL_RGBA_MODE
        case 0x0C40: RET_TYPE_VAR(type, render_mode); break; // GL_RENDER_MODE
        case 0x0CB0: RET_TYPE_VAR(type, pixel_map_i_to_i_size); break; // GL_PIXEL_MAP_I_TO_I_SIZE
        case 0x0CB1: RET_TYPE_VAR(type, pixel_map_s_to_s_size); break; // GL_PIXEL_MAP_S_TO_S_SIZE
        case 0x0CB2: RET_TYPE_VAR(type, pixel_map_i_to_r_size); break; // GL_PIXEL_MAP_I_TO_R_SIZE
        case 0x0CB3: RET_TYPE_VAR(type, pixel_map_i_to_g_size); break; // GL_PIXEL_MAP_I_TO_G_SIZE
        case 0x0CB4: RET_TYPE_VAR(type, pixel_map_i_to_b_size); break; // GL_PIXEL_MAP_I_TO_B_SIZE
        case 0x0CB5: RET_TYPE_VAR(type, pixel_map_i_to_a_size); break; // GL_PIXEL_MAP_I_TO_A_SIZE
        case 0x0CB6: RET_TYPE_VAR(type, pixel_map_r_to_r_size); break; // GL_PIXEL_MAP_R_TO_R_SIZE
        case 0x0CB7: RET_TYPE_VAR(type, pixel_map_g_to_g_size); break; // GL_PIXEL_MAP_G_TO_G_SIZE
        case 0x0CB8: RET_TYPE_VAR(type, pixel_map_b_to_b_size); break; // GL_PIXEL_MAP_B_TO_B_SIZE
        case 0x0CB9: RET_TYPE_VAR(type, pixel_map_a_to_a_size); break; // GL_PIXEL_MAP_A_TO_A_SIZE
        case GL_UNPACK_SWAP_BYTES: RET_TYPE(type, unpack.swap_bytes); break;
        case GL_UNPACK_LSB_FIRST: RET_TYPE(type, unpack.lsb_first); break;
        case GL_UNPACK_ROW_LENGTH: RET_TYPE(type, unpack.row_length); break;
        case GL_UNPACK_SKIP_ROWS: RET_TYPE(type, unpack.skip_rows); break;
        case GL_UNPACK_SKIP_PIXELS: RET_TYPE(type, unpack.skip_pixels); break;
        case GL_UNPACK_ALIGNMENT: RET_TYPE(type, unpack.alignment); break;
        case GL_PACK_SWAP_BYTES: RET_TYPE(type, pack.swap_bytes); break;
        case GL_PACK_LSB_FIRST: RET_TYPE(type, pack.lsb_first); break;
        case GL_PACK_ROW_LENGTH: RET_TYPE(type, pack.row_length); break;
        case GL_PACK_SKIP_ROWS: RET_TYPE(type, pack.skip_rows); break;
        case GL_PACK_SKIP_PIXELS: RET_TYPE(type, pack.skip_pixels); break;
        case GL_PACK_ALIGNMENT: RET_TYPE(type, pack.alignment); break;
        case 0x0D16: RET_TYPE_VAR(type, zoom_x); break; // GL_ZOOM_X
        case 0x0D17: RET_TYPE_VAR(type, zoom_y); break; // GL_ZOOM_Y
        case 0x0D30: RET_TYPE_VAR(type, max_eval_order); break; // GL_MAX_EVAL_ORDER
        case 0x0D31: RET_TYPE_VAR(type, max_lights); break; // GL_MAX_LIGHTS
        case 0x0D32: RET_TYPE_VAR(type, max_clip_planes); break; // GL_MAX_CLIP_PLANES
        case 0x0D34: RET_TYPE_VAR(type, max_pixel_map_table); break; // GL_MAX_PIXEL_MAP_TABLE
        case 0x0D35: RET_TYPE_VAR(type, max_attrib_stack_depth); break; // GL_MAX_ATTRIB_STACK_DEPTH
        case 0x0D36: RET_TYPE_VAR(type, max_modelview_stack_depth); break; // GL_MAX_MODELVIEW_STACK_DEPTH
        case 0x0D37: RET_TYPE_VAR(type, max_name_stack_depth); break; // GL_MAX_NAME_STACK_DEPTH
        case 0x0D38: RET_TYPE_VAR(type, max_projection_stack_depth); break; // GL_MAX_PROJECTION_STACK_DEPTH
        case 0x0D39: RET_TYPE_VAR(type, max_texture_stack_depth); break; // GL_MAX_TEXTURE_STACK_DEPTH
        case 0x0D51: RET_TYPE_VAR(type, index_bits); break; // GL_INDEX_BITS
        case 0x0D52: RET_TYPE_VAR(type, red_bits); break; // GL_RED_BITS
        case 0x0D53: RET_TYPE_VAR(type, green_bits); break; // GL_GREEN_BITS
        case 0x0D54: RET_TYPE_VAR(type, blue_bits); break; // GL_BLUE_BITS
        case 0x0D55: RET_TYPE_VAR(type, alpha_bits); break; // GL_ALPHA_BITS
        case 0x0D56: RET_TYPE_VAR(type, depth_bits); break; // GL_DEPTH_BITS
        case 0x0D57: RET_TYPE_VAR(type, stencil_bits); break; // GL_STENCIL_BITS
        case 0x0D58: RET_TYPE_VAR(type, accum_red_bits); break; // GL_ACCUM_RED_BITS
        case 0x0D59: RET_TYPE_VAR(type, accum_green_bits); break; // GL_ACCUM_GREEN_BITS
        case 0x0D5A: RET_TYPE_VAR(type, accum_blue_bits); break; // GL_ACCUM_BLUE_BITS
        case 0x0D5B: RET_TYPE_VAR(type, accum_alpha_bits); break; // GL_ACCUM_ALPHA_BITS
        case 0x0D70: RET_TYPE_VAR(type, name_stack_depth); break; // GL_NAME_STACK_DEPTH
        case 0x0DD0: RET_TYPE_VAR(type, map1_grid_domain); break; // GL_MAP1_GRID_DOMAIN
        case 0x0DD1: RET_TYPE_VAR(type, map1_grid_segments); break; // GL_MAP1_GRID_SEGMENTS
        case 0x0DD2: RET_TYPE_VAR(type, map2_grid_domain); break; // GL_MAP2_GRID_DOMAIN
        case 0x0DD3: RET_TYPE_VAR(type, map2_grid_segments); break; // GL_MAP2_GRID_SEGMENTS
        case 0x2A00: RET_TYPE_VAR(type, polygon_offset_units); break; // GL_POLYGON_OFFSET_UNITS
        case 0x8038: RET_TYPE_VAR(type, polygon_offset_factor); break; // GL_POLYGON_OFFSET_FACTOR
        case 0x8068: RET_TYPE_VAR(type, texture_binding_1d); break; // GL_TEXTURE_BINDING_1D
        case 0x8069: RET_TYPE_VAR(type, texture_binding_2d); break; // GL_TEXTURE_BINDING_2D
        case 0x0BB1: RET_TYPE_VAR(type, client_attrib_stack_depth); break; // GL_CLIENT_ATTRIB_STACK_DEPTH
        case 0x0D3B: RET_TYPE_VAR(type, max_client_attrib_stack_depth); break; // GL_MAX_CLIENT_ATTRIB_STACK_DEPTH
        case 0x0DF1: RET_TYPE_VAR(type, feedback_buffer_size); break; // GL_FEEDBACK_BUFFER_SIZE
        case 0x0DF2: RET_TYPE_VAR(type, feedback_buffer_type); break; // GL_FEEDBACK_BUFFER_TYPE
        case 0x0DF4: RET_TYPE_VAR(type, selection_buffer_size); break; // GL_SELECTION_BUFFER_SIZE
        case 0x807A: RET_TYPE_VAR(type, vertex_array_size); break; // GL_VERTEX_ARRAY_SIZE
        case 0x807B: RET_TYPE_VAR(type, vertex_array_type); break; // GL_VERTEX_ARRAY_TYPE
        case 0x807C: RET_TYPE_VAR(type, vertex_array_stride); break; // GL_VERTEX_ARRAY_STRIDE
        case 0x807E: RET_TYPE_VAR(type, normal_array_type); break; // GL_NORMAL_ARRAY_TYPE
        case 0x807F: RET_TYPE_VAR(type, normal_array_stride); break; // GL_NORMAL_ARRAY_STRIDE
        case 0x8081: RET_TYPE_VAR(type, color_array_size); break; // GL_COLOR_ARRAY_SIZE
        case 0x8082: RET_TYPE_VAR(type, color_array_type); break; // GL_COLOR_ARRAY_TYPE
        case 0x8083: RET_TYPE_VAR(type, color_array_stride); break; // GL_COLOR_ARRAY_STRIDE
        case 0x8085: RET_TYPE_VAR(type, index_array_type); break; // GL_INDEX_ARRAY_TYPE
        case 0x8086: RET_TYPE_VAR(type, index_array_stride); break; // GL_INDEX_ARRAY_STRIDE
        case 0x8088: RET_TYPE_VAR(type, texture_coord_array_size); break; // GL_TEXTURE_COORD_ARRAY_SIZE
        case 0x8089: RET_TYPE_VAR(type, texture_coord_array_type); break; // GL_TEXTURE_COORD_ARRAY_TYPE
        case 0x808A: RET_TYPE_VAR(type, texture_coord_array_stride); break; // GL_TEXTURE_COORD_ARRAY_STRIDE
        case 0x808C: RET_TYPE_VAR(type, edge_flag_array_stride); break; // GL_EDGE_FLAG_ARRAY_STRIDE
        case 0x806A: RET_TYPE_VAR(type, texture_binding_3d); break; // GL_TEXTURE_BINDING_3D
        case 0x8073: RET_TYPE_VAR(type, max_3d_texture_size); break; // GL_MAX_3D_TEXTURE_SIZE
        case 0x80E8: RET_TYPE_VAR(type, max_elements_vertices); break; // GL_MAX_ELEMENTS_VERTICES
        case 0x80E9: RET_TYPE_VAR(type, max_elements_indices); break; // GL_MAX_ELEMENTS_INDICES
        case 0x846E: RET_TYPE_VAR(type, aliased_line_width_range); break; // GL_ALIASED_LINE_WIDTH_RANGE
        case 0x846D: RET_TYPE_VAR(type, aliased_point_size_range); break; // GL_ALIASED_POINT_SIZE_RANGE
        case 0x84E0: { // GL_ACTIVE_TEXTURE
            GLenum activeTexture = GL_TEXTURE0 + (ctx ? ctx->state.active_texture : 0u);
            switch(type) {
                case kBool: RET_BOOL(activeTexture)
                case kInt: RET_INT(activeTexture)
                case kFloat: RET_FLOAT(activeTexture)
                case kDouble: RET_DOUBLE(activeTexture)
            }
            break;
        }
        case 0x80AA: RET_TYPE_VAR(type, sample_coverage_value); break; // GL_SAMPLE_COVERAGE_VALUE
        case 0x80AB: RET_TYPE_VAR(type, sample_coverage_invert); break; // GL_SAMPLE_COVERAGE_INVERT
        case 0x8514: RET_TYPE_VAR(type, texture_binding_cube_map); break; // GL_TEXTURE_BINDING_CUBE_MAP
        case 0x851C: RET_TYPE_VAR(type, max_cube_map_texture_size); break; // GL_MAX_CUBE_MAP_TEXTURE_SIZE
        case 0x86A2: RET_TYPE_VAR(type, num_compressed_texture_formats); break; // GL_NUM_COMPRESSED_TEXTURE_FORMATS
        case 0x86A3: RET_TYPE_VAR(type, compressed_texture_formats); break; // GL_COMPRESSED_TEXTURE_FORMATS

        case 0x80C8: RET_TYPE_VAR(type, blend_dst_rgb[0]); break; // GL_BLEND_DST_RGB
        case 0x80C9: RET_TYPE_VAR(type, blend_src_rgb[0]); break; // GL_BLEND_SRC_RGB
        case 0x80CA: RET_TYPE_VAR(type, blend_dst_alpha[0]); break; // GL_BLEND_DST_ALPHA
        case 0x80CB: RET_TYPE_VAR(type, blend_src_alpha[0]); break; // GL_BLEND_SRC_ALPHA

        case 0x84FD: RET_TYPE_VAR(type, max_texture_lod_bias); break; // GL_MAX_TEXTURE_LOD_BIAS

        case 0x8005: RET_TYPE_VAR_COUNT(type, blend_color,4); break; // GL_BLEND_COLOR

        case 0x8894: RET_TYPE_VAR(type, array_buffer_binding); break; // GL_ARRAY_BUFFER_BINDING
        case 0x8895: { // GL_ELEMENT_ARRAY_BUFFER_BINDING
            GLuint ebo = 0;
            VertexArray *vao = ctx->state.vao;
            if (vao &&
                mglObjectPointerLooksPlausible(vao) &&
                mglHashTableContainsData(&ctx->state.vao_table, vao) &&
                mglPointerRangeIsReadable(vao, sizeof(*vao)) &&
                vao->element_array.buffer) {
                ebo = vao->element_array.buffer->name;
            } else if (ctx->state.default_vao_element_array_buffer) {
                ebo = ctx->state.default_vao_element_array_buffer->name;
            }
            switch(type) {
                case kBool: RET_BOOL(ebo);
                case kInt: RET_INT(ebo);
                case kFloat: RET_FLOAT(ebo);
                case kDouble: RET_DOUBLE(ebo);
            }
            break;
        }
        case 0x8009: RET_TYPE_VAR(type, blend_equation_rgb[0]); break; // GL_BLEND_EQUATION_RGB
        case 0x8800: RET_TYPE_VAR(type, stencil_back_func); break; // GL_STENCIL_BACK_FUNC
        case 0x8801: RET_TYPE_VAR(type, stencil_back_fail); break; // GL_STENCIL_BACK_FAIL
        case 0x8802: RET_TYPE_VAR(type, stencil_back_pass_depth_fail); break; // GL_STENCIL_BACK_PASS_DEPTH_FAIL
        case 0x8803: RET_TYPE_VAR(type, stencil_back_pass_depth_pass); break; // GL_STENCIL_BACK_PASS_DEPTH_PASS
        case 0x8824: RET_TYPE_VAR(type, max_draw_buffers); break; // GL_MAX_DRAW_BUFFERS
        case 0x8CDF: RET_TYPE_VAR(type, max_draw_buffers); break; // GL_MAX_COLOR_ATTACHMENTS

        case 0x883D: RET_TYPE_VAR(type, blend_equation_alpha[0]); break; // GL_BLEND_EQUATION_ALPHA

        case 0x8869: RET_TYPE(type, max_vertex_attribs); break; // GL_MAX_VERTEX_ATTRIBS
        case 0x8872: RET_TYPE_VAR(type, max_texture_image_units); break; // GL_MAX_TEXTURE_IMAGE_UNITS
        case 0x8B49: RET_TYPE_VAR(type, max_fragment_uniform_components); break; // GL_MAX_FRAGMENT_UNIFORM_COMPONENTS
        case 0x8B4A: RET_TYPE_VAR(type, max_vertex_uniform_components); break; // GL_MAX_VERTEX_UNIFORM_COMPONENTS
        case 0x8B4B: RET_TYPE_VAR(type, max_varying_floats); break; // GL_MAX_VARYING_FLOATS
        case 0x8B4C: RET_TYPE_VAR(type, max_vertex_texture_image_units); break; // GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS
        case 0x8B4D: RET_TYPE_VAR(type, max_combined_texture_image_units); break; // GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS
        case 0x8B8D: RET_TYPE_VAR(type, current_program); break; // GL_CURRENT_PROGRAM
        case 0x8CA3: RET_TYPE_VAR(type, stencil_back_ref); break; // GL_STENCIL_BACK_REF
        case 0x8CA4: RET_TYPE_VAR(type, stencil_back_value_mask); break; // GL_STENCIL_BACK_VALUE_MASK
        case 0x8CA5: RET_TYPE_VAR(type, stencil_back_writemask); break; // GL_STENCIL_BACK_WRITEMASK
        case 0x88ED: RET_TYPE_VAR(type, pixel_pack_buffer_binding); break; // GL_PIXEL_PACK_BUFFER_BINDING
        case 0x88EF: RET_TYPE_VAR(type, pixel_unpack_buffer_binding); break; // GL_PIXEL_UNPACK_BUFFER_BINDING
        case GL_PACK_SKIP_IMAGES: RET_TYPE(type, pack.skip_images); break;
        case GL_PACK_IMAGE_HEIGHT: RET_TYPE(type, pack.image_height); break;
        case GL_UNPACK_SKIP_IMAGES: RET_TYPE(type, unpack.skip_images); break;
        case GL_UNPACK_IMAGE_HEIGHT: RET_TYPE(type, unpack.image_height); break;
        case 0x821B: RET_TYPE_VAR(type, major_version); break; // GL_MAJOR_VERSION
        case 0x821C: RET_TYPE_VAR(type, minor_version); break; // GL_MINOR_VERSION
        case 0x821D: // GL_NUM_EXTENSIONS
            switch(type) {
                case kBool: RET_BOOL(MGL_NUM_EXTENSIONS);
                case kInt: RET_INT(MGL_NUM_EXTENSIONS);
                case kFloat: RET_FLOAT(MGL_NUM_EXTENSIONS);
                case kDouble: RET_DOUBLE(MGL_NUM_EXTENSIONS);
            }
            break;
        case 0x821E: RET_TYPE_VAR(type, context_flags); break; // GL_CONTEXT_FLAGS
        case 0x88FF: RET_TYPE_VAR(type, max_array_texture_layers); break; // GL_MAX_ARRAY_TEXTURE_LAYERS
        case 0x8904: RET_TYPE_VAR(type, min_program_texel_offset); break; // GL_MIN_PROGRAM_TEXEL_OFFSET
        case 0x8905: RET_TYPE_VAR(type, max_program_texel_offset); break; // GL_MAX_PROGRAM_TEXEL_OFFSET
        case 0x8C1C: RET_TYPE_VAR(type, texture_binding_1d_array); break; // GL_TEXTURE_BINDING_1D_ARRAY
        case 0x8C1D: RET_TYPE_VAR(type, texture_binding_2d_array); break; // GL_TEXTURE_BINDING_2D_ARRAY
        case 0x84E8: RET_TYPE_VAR(type, max_renderbuffer_size); break; // GL_MAX_RENDERBUFFER_SIZE
        case 0x8CA6: // GL_DRAW_FRAMEBUFFER_BINDING / GL_FRAMEBUFFER_BINDING
            switch(type) {
                case kBool: RET_BOOL(mglCurrentDrawFramebufferBinding(ctx));
                case kInt: RET_INT(mglCurrentDrawFramebufferBinding(ctx));
                case kFloat: RET_FLOAT(mglCurrentDrawFramebufferBinding(ctx));
                case kDouble: RET_DOUBLE(mglCurrentDrawFramebufferBinding(ctx));
            }
            break;
        case 0x8CA7: RET_TYPE_VAR(type, renderbuffer_binding); break; // GL_RENDERBUFFER_BINDING
        case 0x8CAA: // GL_READ_FRAMEBUFFER_BINDING
            switch(type) {
                case kBool: RET_BOOL(mglCurrentReadFramebufferBinding(ctx));
                case kInt: RET_INT(mglCurrentReadFramebufferBinding(ctx));
                case kFloat: RET_FLOAT(mglCurrentReadFramebufferBinding(ctx));
                case kDouble: RET_DOUBLE(mglCurrentReadFramebufferBinding(ctx));
            }
            break;
        case 0x85B5: RET_TYPE_VAR(type, vertex_array_binding); break; // GL_VERTEX_ARRAY_BINDING
        case 0x8C2B: // GL_MAX_TEXTURE_BUFFER_SIZE
            switch(type) {
                case kBool: RET_BOOL(mglSafeMaxTextureBufferSize(ctx));
                case kInt: RET_INT(mglSafeMaxTextureBufferSize(ctx));
                case kFloat: RET_FLOAT(mglSafeMaxTextureBufferSize(ctx));
                case kDouble: RET_DOUBLE(mglSafeMaxTextureBufferSize(ctx));
            }
            break;
        case 0x8C2C: RET_TYPE_VAR(type, texture_binding_buffer); break; // GL_TEXTURE_BINDING_BUFFER
        case 0x84F6: RET_TYPE_VAR(type, texture_binding_rectangle); break; // GL_TEXTURE_BINDING_RECTANGLE
        case 0x84F8: RET_TYPE_VAR(type, max_rectangle_texture_size); break; // GL_MAX_RECTANGLE_TEXTURE_SIZE
        case 0x8F9E: RET_TYPE_VAR(type, primitive_restart_index); break; // GL_PRIMITIVE_RESTART_INDEX
        case 0x8A28: RET_TYPE_VAR(type, uniform_buffer_binding); break; // GL_UNIFORM_BUFFER_BINDING
        case 0x8A29: RET_TYPE_VAR(type, uniform_buffer_start); break; // GL_UNIFORM_BUFFER_START
        case 0x8A2A: RET_TYPE_VAR(type, uniform_buffer_size); break; // GL_UNIFORM_BUFFER_SIZE
        case 0x8A2B: RET_TYPE_VAR(type, max_vertex_uniform_blocks); break; // GL_MAX_VERTEX_UNIFORM_BLOCKS
        case 0x8A2C: RET_TYPE_VAR(type, max_geometry_uniform_blocks); break; // GL_MAX_GEOMETRY_UNIFORM_BLOCKS
        case 0x8A2D: RET_TYPE_VAR(type, max_fragment_uniform_blocks); break; // GL_MAX_FRAGMENT_UNIFORM_BLOCKS
        case 0x8A2E: RET_TYPE_VAR(type, max_combined_uniform_blocks); break; // GL_MAX_COMBINED_UNIFORM_BLOCKS
        case 0x8A2F: RET_TYPE_VAR(type, max_uniform_buffer_bindings); break; // GL_MAX_UNIFORM_BUFFER_BINDINGS
        case 0x8A30: RET_TYPE_VAR(type, max_uniform_block_size); break; // GL_MAX_UNIFORM_BLOCK_SIZE
        case 0x8A31: RET_TYPE_VAR(type, max_combined_vertex_uniform_components); break; // GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS
        case 0x8A32: RET_TYPE_VAR(type, max_combined_geometry_uniform_components); break; // GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS
        case 0x8A33: RET_TYPE_VAR(type, max_combined_fragment_uniform_components); break; // GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS
        case 0x8A34: RET_TYPE_VAR(type, uniform_buffer_offset_alignment); break; // GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT
        case 0x8C29: RET_TYPE_VAR(type, max_geometry_texture_image_units); break; // GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS
        case 0x8DDF: RET_TYPE_VAR(type, max_geometry_uniform_components); break; // GL_MAX_GEOMETRY_UNIFORM_COMPONENTS
        case 0x9122: RET_TYPE_VAR(type, max_vertex_output_components); break; // GL_MAX_VERTEX_OUTPUT_COMPONENTS
        case 0x9123: RET_TYPE_VAR(type, max_geometry_input_components); break; // GL_MAX_GEOMETRY_INPUT_COMPONENTS
        case 0x9124: RET_TYPE_VAR(type, max_geometry_output_components); break; // GL_MAX_GEOMETRY_OUTPUT_COMPONENTS
        case 0x9125: RET_TYPE_VAR(type, max_fragment_input_components); break; // GL_MAX_FRAGMENT_INPUT_COMPONENTS
        case 0x9126: RET_TYPE_VAR(type, context_profile_mask); break; // GL_CONTEXT_PROFILE_MASK
        case 0x8E4F: RET_TYPE_VAR(type, provoking_vertex); break; // GL_PROVOKING_VERTEX
        case 0x9111: RET_TYPE_VAR(type, max_server_wait_timeout); break; // GL_MAX_SERVER_WAIT_TIMEOUT
        case 0x8D57: RET_TYPE_VAR(type, max_framebuffer_samples); break; // GL_MAX_SAMPLES
        case 0x8E59: RET_TYPE_VAR(type, max_sample_mask_words); break; // GL_MAX_SAMPLE_MASK_WORDS
        case 0x9104: RET_TYPE_VAR(type, texture_binding_2d_multisample); break; // GL_TEXTURE_BINDING_2D_MULTISAMPLE
        case 0x9105: RET_TYPE_VAR(type, texture_binding_2d_multisample_array); break; // GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY
        case 0x910E: RET_TYPE_VAR(type, max_color_texture_samples); break; // GL_MAX_COLOR_TEXTURE_SAMPLES
        case 0x910F: RET_TYPE_VAR(type, max_depth_texture_samples); break; // GL_MAX_DEPTH_TEXTURE_SAMPLES
        case 0x9110: RET_TYPE_VAR(type, max_integer_samples); break; // GL_MAX_INTEGER_SAMPLES
        case 0x88FC: RET_TYPE_VAR(type, max_dual_source_draw_buffers); break; // GL_MAX_DUAL_SOURCE_DRAW_BUFFERS
        case 0x8919: RET_TYPE_VAR(type, sampler_binding); break; // GL_SAMPLER_BINDING
        case 0x8E89: RET_TYPE_VAR(type, max_tess_control_uniform_blocks); break; // GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS
        case 0x8E8A: RET_TYPE_VAR(type, max_tess_evaluation_uniform_blocks); break; // GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS
        case 0x8DFA: RET_TYPE_VAR(type, shader_compiler); break; // GL_SHADER_COMPILER
        case 0x8DF8: RET_TYPE_VAR(type, shader_binary_formats); break; // GL_SHADER_BINARY_FORMATS
        case 0x8DF9: RET_TYPE_VAR(type, num_shader_binary_formats); break; // GL_NUM_SHADER_BINARY_FORMATS
        case 0x8DFB: RET_TYPE_VAR(type, max_vertex_uniform_vectors); break; // GL_MAX_VERTEX_UNIFORM_VECTORS
        case 0x8DFC: RET_TYPE_VAR(type, max_varying_vectors); break; // GL_MAX_VARYING_VECTORS
        case 0x8DFD: RET_TYPE_VAR(type, max_fragment_uniform_vectors); break; // GL_MAX_FRAGMENT_UNIFORM_VECTORS
        case 0x87FE: RET_TYPE_VAR(type, num_program_binary_formats); break; // GL_NUM_PROGRAM_BINARY_FORMATS
        case 0x87FF: RET_TYPE_VAR(type, program_binary_formats); break; // GL_PROGRAM_BINARY_FORMATS
        case 0x825A: RET_TYPE_VAR(type, program_pipeline_binding); break; // GL_PROGRAM_PIPELINE_BINDING
        case 0x825B: RET_TYPE_VAR(type, max_viewports); break; // GL_MAX_VIEWPORTS
        case 0x825C: RET_TYPE_VAR(type, viewport_subpixel_bits); break; // GL_VIEWPORT_SUBPIXEL_BITS
        case 0x825D: RET_TYPE_VAR(type, viewport_bounds_range); break; // GL_VIEWPORT_BOUNDS_RANGE
        case 0x825E: RET_TYPE_VAR(type, layer_provoking_vertex); break; // GL_LAYER_PROVOKING_VERTEX
        case 0x825F: RET_TYPE_VAR(type, viewport_index_provoking_vertex); break; // GL_VIEWPORT_INDEX_PROVOKING_VERTEX
        case GL_CLIP_ORIGIN: RET_TYPE_VAR(type, clip_origin); break;
        case GL_CLIP_DEPTH_MODE: RET_TYPE_VAR(type, clip_depth_mode); break;
        case 0x90BC: RET_TYPE_VAR(type, min_map_buffer_alignment); break; // GL_MIN_MAP_BUFFER_ALIGNMENT
        case 0x92D2: RET_TYPE_VAR(type, max_vertex_atomic_counters); break; // GL_MAX_VERTEX_ATOMIC_COUNTERS
        case 0x92D3: RET_TYPE_VAR(type, max_tess_control_atomic_counters); break; // GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS
        case 0x92D4: RET_TYPE_VAR(type, max_tess_evaluation_atomic_counters); break; // GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS
        case 0x92D5: RET_TYPE_VAR(type, max_geometry_atomic_counters); break; // GL_MAX_GEOMETRY_ATOMIC_COUNTERS
        case 0x92D6: RET_TYPE_VAR(type, max_fragment_atomic_counters); break; // GL_MAX_FRAGMENT_ATOMIC_COUNTERS
        case 0x92D7: RET_TYPE_VAR(type, max_combined_atomic_counters); break; // GL_MAX_COMBINED_ATOMIC_COUNTERS
        case 0x8D6B: RET_TYPE_VAR(type, max_element_index); break; // GL_MAX_ELEMENT_INDEX
        case 0x91BB: RET_TYPE_VAR(type, max_compute_uniform_blocks); break; // GL_MAX_COMPUTE_UNIFORM_BLOCKS
        case 0x91BC: RET_TYPE_VAR(type, max_compute_texture_image_units); break; // GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS
        case 0x8263: RET_TYPE_VAR(type, max_compute_uniform_components); break; // GL_MAX_COMPUTE_UNIFORM_COMPONENTS
        case 0x8264: RET_TYPE_VAR(type, max_compute_atomic_counter_buffers); break; // GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS
        case 0x8265: RET_TYPE_VAR(type, max_compute_atomic_counters); break; // GL_MAX_COMPUTE_ATOMIC_COUNTERS
        case 0x8266: RET_TYPE_VAR(type, max_combined_compute_uniform_components); break; // GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS
        case 0x90EB: RET_TYPE_VAR(type, max_compute_work_group_invocations); break; // GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS
        case 0x91BE: RET_TYPE_VAR(type, max_compute_work_group_count[0]); break; // GL_MAX_COMPUTE_WORK_GROUP_COUNT
        case 0x91BF: RET_TYPE_VAR(type, max_compute_work_group_size[0]); break; // GL_MAX_COMPUTE_WORK_GROUP_SIZE
        case 0x90EF: RET_TYPE_VAR(type, dispatch_indirect_buffer_binding); break; // GL_DISPATCH_INDIRECT_BUFFER_BINDING
        case 0x826C: RET_TYPE_VAR(type, max_debug_group_stack_depth); break; // GL_MAX_DEBUG_GROUP_STACK_DEPTH
        case 0x826D: RET_TYPE_VAR(type, debug_group_stack_depth); break; // GL_DEBUG_GROUP_STACK_DEPTH
        case 0x82E8: RET_TYPE_VAR(type, max_label_length); break; // GL_MAX_LABEL_LENGTH
        case 0x826E: RET_TYPE_VAR(type, max_uniform_locations); break; // GL_MAX_UNIFORM_LOCATIONS
        case 0x9315: RET_TYPE_VAR(type, max_framebuffer_width); break; // GL_MAX_FRAMEBUFFER_WIDTH
        case 0x9316: RET_TYPE_VAR(type, max_framebuffer_height); break; // GL_MAX_FRAMEBUFFER_HEIGHT
        case 0x9317: RET_TYPE_VAR(type, max_framebuffer_layers); break; // GL_MAX_FRAMEBUFFER_LAYERS
        case 0x9318: RET_TYPE_VAR(type, max_framebuffer_samples); break; // GL_MAX_FRAMEBUFFER_SAMPLES
        case 0x90D3: RET_TYPE_VAR(type, shader_storage_buffer_binding); break; // GL_SHADER_STORAGE_BUFFER_BINDING
        case 0x90D4: RET_TYPE_VAR(type, shader_storage_buffer_start); break; // GL_SHADER_STORAGE_BUFFER_START
        case 0x90D5: RET_TYPE_VAR(type, shader_storage_buffer_size); break; // GL_SHADER_STORAGE_BUFFER_SIZE
        case 0x90D6: RET_TYPE_VAR(type, max_vertex_shader_storage_blocks); break; // GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS
        case 0x90D7: RET_TYPE_VAR(type, max_geometry_shader_storage_blocks); break; // GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS
        case 0x90D8: RET_TYPE_VAR(type, max_tess_control_shader_storage_blocks); break; // GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS
        case 0x90D9: RET_TYPE_VAR(type, max_tess_evaluation_shader_storage_blocks); break; // GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS
        case 0x90DA: RET_TYPE_VAR(type, max_fragment_shader_storage_blocks); break; // GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS
        case 0x90DB: RET_TYPE_VAR(type, max_compute_shader_storage_blocks); break; // GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS
        case 0x90DC: RET_TYPE_VAR(type, max_combined_shader_storage_blocks); break; // GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS
        case 0x90DD: RET_TYPE_VAR(type, max_shader_storage_buffer_bindings); break; // GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS
        case 0x90DF: RET_TYPE_VAR(type, shader_storage_buffer_offset_alignment); break; // GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT
        case 0x919F: // GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT
            switch(type) {
                case kBool: RET_BOOL(mglSafeTextureBufferOffsetAlignment(ctx));
                case kInt: RET_INT(mglSafeTextureBufferOffsetAlignment(ctx));
                case kFloat: RET_FLOAT(mglSafeTextureBufferOffsetAlignment(ctx));
                case kDouble: RET_DOUBLE(mglSafeTextureBufferOffsetAlignment(ctx));
            }
            break;
        case 0x82D6: RET_TYPE_VAR(type, vertex_binding_divisor); break; // GL_VERTEX_BINDING_DIVISOR
        case 0x82D7: RET_TYPE_VAR(type, vertex_binding_offset); break; // GL_VERTEX_BINDING_OFFSET
        case 0x82D8: RET_TYPE_VAR(type, vertex_binding_stride); break; // GL_VERTEX_BINDING_STRIDE
        case 0x82D9: // GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET
            switch(type) {
                case kBool: RET_BOOL(mglSafeMaxVertexAttribRelativeOffset(ctx));
                case kInt: RET_INT(mglSafeMaxVertexAttribRelativeOffset(ctx));
                case kFloat: RET_FLOAT(mglSafeMaxVertexAttribRelativeOffset(ctx));
                case kDouble: RET_DOUBLE(mglSafeMaxVertexAttribRelativeOffset(ctx));
            }
            break;
        case 0x82DA: // GL_MAX_VERTEX_ATTRIB_BINDINGS
            switch(type) {
                case kBool: RET_BOOL(mglSafeMaxVertexAttribBindings(ctx));
                case kInt: RET_INT(mglSafeMaxVertexAttribBindings(ctx));
                case kFloat: RET_FLOAT(mglSafeMaxVertexAttribBindings(ctx));
                case kDouble: RET_DOUBLE(mglSafeMaxVertexAttribBindings(ctx));
            }
            break;
        case 0x82E5: // GL_MAX_VERTEX_ATTRIB_STRIDE
            switch(type) {
                case kBool: RET_BOOL(mglSafeMaxVertexAttribStride(ctx));
                case kInt: RET_INT(mglSafeMaxVertexAttribStride(ctx));
                case kFloat: RET_FLOAT(mglSafeMaxVertexAttribStride(ctx));
                case kDouble: RET_DOUBLE(mglSafeMaxVertexAttribStride(ctx));
            }
            break;
    }
}

void mglGetBooleanv(GLMContext ctx, GLenum pname, GLboolean *data)
{
    if (!data) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglGet(ctx, pname, kBool, (void *)data);
}

void mglGetDoublev(GLMContext ctx, GLenum pname, GLdouble *data)
{
    if (!data) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglGet(ctx, pname, kDouble, (void *)data);
}

void mglGetFloatv(GLMContext ctx, GLenum pname, GLfloat *data)
{
    if (!data) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglGet(ctx, pname, kFloat, (void *)data);
}

void mglGetIntegerv(GLMContext ctx, GLenum pname, GLint *data)
{
    if (!data) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglGet(ctx, pname, kInt, (void *)data);
}

const GLubyte *mglGetString(GLMContext ctx, GLenum name)
{
    switch(name)
    {
        case GL_VENDOR:
            return (const GLubyte *)"Mike Larson";

        case GL_RENDERER:
            return (const GLubyte *)"MGL";

        case GL_VERSION:
            return (const GLubyte *)"4.6.0";

        case GL_SHADING_LANGUAGE_VERSION:
            return (const GLubyte *)"4.6";

        default:
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, NULL);
    }

    return NULL;
}

static GLsizei mglGetParameterCount(GLenum pname)
{
    switch (pname) {
        case GL_VIEWPORT:
        case GL_SCISSOR_BOX:
        case GL_COLOR_CLEAR_VALUE:
        case GL_COLOR_WRITEMASK:
            return 4;
        case GL_DEPTH_RANGE:
        case GL_POLYGON_MODE:
        case GL_ALIASED_LINE_WIDTH_RANGE:
        case 0x846D: // GL_ALIASED_POINT_SIZE_RANGE is not exposed by this core header.
        case GL_SMOOTH_LINE_WIDTH_RANGE:
        case GL_SMOOTH_POINT_SIZE_RANGE:
        case GL_MAX_VIEWPORT_DIMS:
        case GL_VIEWPORT_BOUNDS_RANGE:
            return 2;
        default:
            return 1;
    }
}

void mglGetInteger64v(GLMContext ctx, GLenum pname, GLint64 *data)
{
    if (!data) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    GLint tmp[16] = {0};
    GLsizei count = mglGetParameterCount(pname);
    if (count > (GLsizei)(sizeof(tmp) / sizeof(tmp[0]))) {
        count = (GLsizei)(sizeof(tmp) / sizeof(tmp[0]));
    }

    mglGet(ctx, pname, kInt, tmp);
    for (GLsizei i = 0; i < count; ++i) {
        data[i] = (GLint64)tmp[i];
    }
}

void mglGetInteger64i_v(GLMContext ctx, GLenum target, GLuint index, GLint64 *data)
{
    if (!data) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!ctx)
        return;

    switch (target) {
        case GL_VIEWPORT:
            if (index >= mglSafeMaxViewports(ctx)) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            for (int i = 0; i < 4; i++) {
                data[i] = (GLint64)ctx->state.viewport_array[index][i];
            }
            return;

        case GL_SCISSOR_BOX:
            if (index >= mglSafeMaxViewports(ctx)) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            for (int i = 0; i < 4; i++) {
                data[i] = (GLint64)ctx->state.scissor_box_array[index][i];
            }
            return;

        case GL_DEPTH_RANGE:
            if (index >= mglSafeMaxViewports(ctx)) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            data[0] = (GLint64)ctx->state.depth_range_array[index][0];
            data[1] = (GLint64)ctx->state.depth_range_array[index][1];
            return;

        case GL_SCISSOR_TEST:
            if (index >= mglSafeMaxViewports(ctx)) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            *data = ctx->state.caps.scissor_testi[index] ? GL_TRUE : GL_FALSE;
            return;

        case GL_UNIFORM_BUFFER_BINDING:
        case GL_UNIFORM_BUFFER_START:
        case GL_UNIFORM_BUFFER_SIZE:
        case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING:
        case GL_TRANSFORM_FEEDBACK_BUFFER_START:
        case GL_TRANSFORM_FEEDBACK_BUFFER_SIZE:
        case GL_SHADER_STORAGE_BUFFER_BINDING:
        case GL_SHADER_STORAGE_BUFFER_START:
        case GL_SHADER_STORAGE_BUFFER_SIZE:
        case GL_ATOMIC_COUNTER_BUFFER_BINDING:
        case GL_ATOMIC_COUNTER_BUFFER_START:
        case GL_ATOMIC_COUNTER_BUFFER_SIZE:
        {
            if (index >= MAX_BINDABLE_BUFFERS) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }

            GLuint bufferIndex = _UNIFORM_BUFFER;
            switch (target) {
                case GL_UNIFORM_BUFFER_BINDING:
                case GL_UNIFORM_BUFFER_START:
                case GL_UNIFORM_BUFFER_SIZE:
                    bufferIndex = _UNIFORM_BUFFER;
                    break;
                case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING:
                case GL_TRANSFORM_FEEDBACK_BUFFER_START:
                case GL_TRANSFORM_FEEDBACK_BUFFER_SIZE:
                    bufferIndex = _TRANSFORM_FEEDBACK_BUFFER;
                    break;
                case GL_SHADER_STORAGE_BUFFER_BINDING:
                case GL_SHADER_STORAGE_BUFFER_START:
                case GL_SHADER_STORAGE_BUFFER_SIZE:
                    bufferIndex = _SHADER_STORAGE_BUFFER;
                    break;
                case GL_ATOMIC_COUNTER_BUFFER_BINDING:
                case GL_ATOMIC_COUNTER_BUFFER_START:
                case GL_ATOMIC_COUNTER_BUFFER_SIZE:
                    bufferIndex = _ATOMIC_COUNTER_BUFFER;
                    break;
            }

            BufferBaseTarget *binding = &ctx->state.buffer_base[bufferIndex].buffers[index];
            switch (target) {
                case GL_UNIFORM_BUFFER_BINDING:
                case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING:
                case GL_SHADER_STORAGE_BUFFER_BINDING:
                case GL_ATOMIC_COUNTER_BUFFER_BINDING:
                    *data = (GLint64)binding->buffer;
                    break;
                case GL_UNIFORM_BUFFER_START:
                case GL_TRANSFORM_FEEDBACK_BUFFER_START:
                case GL_SHADER_STORAGE_BUFFER_START:
                case GL_ATOMIC_COUNTER_BUFFER_START:
                    *data = (GLint64)binding->offset;
                    break;
                default:
                    *data = (GLint64)binding->size;
                    break;
            }
            return;
        }

        case GL_VERTEX_BINDING_BUFFER:
        case GL_VERTEX_BINDING_OFFSET:
        case GL_VERTEX_BINDING_STRIDE:
        case GL_VERTEX_BINDING_DIVISOR:
        {
            GLuint maxBindings = mglSafeMaxVertexAttribBindings(ctx);
            if (index >= maxBindings || index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            VertexArray *vao = ctx->state.vao;
            if (!vao) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }

            BufferBinding *binding = &vao->bindings[index];
            switch (target) {
                case GL_VERTEX_BINDING_BUFFER:
                    *data = binding->buffer ? (GLint64)binding->buffer->name : 0;
                    break;
                case GL_VERTEX_BINDING_OFFSET:
                    *data = (GLint64)binding->offset;
                    break;
                case GL_VERTEX_BINDING_STRIDE:
                    *data = (GLint64)binding->stride;
                    break;
                case GL_VERTEX_BINDING_DIVISOR:
                    *data = (GLint64)binding->divisor;
                    break;
            }
            return;
        }
    }

    GLint tmp[4] = {0};
    GLsizei count = (target == GL_COLOR_WRITEMASK) ? 4 : 1;
    mglGetIntegeri_v(ctx, target, index, tmp);
    for (GLsizei i = 0; i < count; i++) {
        data[i] = (GLint64)tmp[i];
    }
}

const GLubyte  *mglGetStringi(GLMContext ctx, GLenum name, GLuint index)
{
    if (!ctx)
        return NULL;

    if (name != GL_EXTENSIONS)
    {
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, NULL);
    }

    if (index >= MGL_NUM_EXTENSIONS)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, NULL);
    }

    return (const GLubyte *)kMglExtensions[index];
}

void mglGetIntegeri_v(GLMContext ctx, GLenum target, GLuint index, GLint *data)
{
    if (!data) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!ctx)
        return;

    switch(target)
    {
        case GL_VIEWPORT:
            if (index >= mglSafeMaxViewports(ctx)) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            for (int i = 0; i < 4; i++) {
                data[i] = (GLint)ctx->state.viewport_array[index][i];
            }
            break;

        case GL_SCISSOR_BOX:
            if (index >= mglSafeMaxViewports(ctx)) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            for (int i = 0; i < 4; i++) {
                data[i] = ctx->state.scissor_box_array[index][i];
            }
            break;

        case GL_DEPTH_RANGE:
            if (index >= mglSafeMaxViewports(ctx)) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            data[0] = (GLint)ctx->state.depth_range_array[index][0];
            data[1] = (GLint)ctx->state.depth_range_array[index][1];
            break;

        case GL_SCISSOR_TEST:
            if (index >= mglSafeMaxViewports(ctx)) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            *data = ctx->state.caps.scissor_testi[index] ? GL_TRUE : GL_FALSE;
            break;

        case GL_UNIFORM_BUFFER_BINDING:
        case GL_UNIFORM_BUFFER_START:
        case GL_UNIFORM_BUFFER_SIZE:
        case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING:
        case GL_TRANSFORM_FEEDBACK_BUFFER_START:
        case GL_TRANSFORM_FEEDBACK_BUFFER_SIZE:
        case GL_SHADER_STORAGE_BUFFER_BINDING:
        case GL_SHADER_STORAGE_BUFFER_START:
        case GL_SHADER_STORAGE_BUFFER_SIZE:
        case GL_ATOMIC_COUNTER_BUFFER_BINDING:
        case GL_ATOMIC_COUNTER_BUFFER_START:
        case GL_ATOMIC_COUNTER_BUFFER_SIZE:
        {
            if (index >= MAX_BINDABLE_BUFFERS) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }

            GLuint bufferIndex = _UNIFORM_BUFFER;
            switch (target) {
                case GL_UNIFORM_BUFFER_BINDING:
                case GL_UNIFORM_BUFFER_START:
                case GL_UNIFORM_BUFFER_SIZE:
                    bufferIndex = _UNIFORM_BUFFER;
                    break;
                case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING:
                case GL_TRANSFORM_FEEDBACK_BUFFER_START:
                case GL_TRANSFORM_FEEDBACK_BUFFER_SIZE:
                    bufferIndex = _TRANSFORM_FEEDBACK_BUFFER;
                    break;
                case GL_SHADER_STORAGE_BUFFER_BINDING:
                case GL_SHADER_STORAGE_BUFFER_START:
                case GL_SHADER_STORAGE_BUFFER_SIZE:
                    bufferIndex = _SHADER_STORAGE_BUFFER;
                    break;
                case GL_ATOMIC_COUNTER_BUFFER_BINDING:
                case GL_ATOMIC_COUNTER_BUFFER_START:
                case GL_ATOMIC_COUNTER_BUFFER_SIZE:
                    bufferIndex = _ATOMIC_COUNTER_BUFFER;
                    break;
            }

            BufferBaseTarget *binding = &ctx->state.buffer_base[bufferIndex].buffers[index];
            switch (target) {
                case GL_UNIFORM_BUFFER_BINDING:
                case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING:
                case GL_SHADER_STORAGE_BUFFER_BINDING:
                case GL_ATOMIC_COUNTER_BUFFER_BINDING:
                    *data = (GLint)binding->buffer;
                    break;
                case GL_UNIFORM_BUFFER_START:
                case GL_TRANSFORM_FEEDBACK_BUFFER_START:
                case GL_SHADER_STORAGE_BUFFER_START:
                case GL_ATOMIC_COUNTER_BUFFER_START:
                    *data = mglClampGetInteger64ToInt((GLint64)binding->offset, "indexed buffer offset");
                    break;
                default:
                    *data = mglClampGetInteger64ToInt((GLint64)binding->size, "indexed buffer size");
                    break;
            }
            break;
        }

        case GL_VERTEX_BINDING_BUFFER:
        case GL_VERTEX_BINDING_OFFSET:
        case GL_VERTEX_BINDING_STRIDE:
        case GL_VERTEX_BINDING_DIVISOR:
        {
            GLuint maxBindings = mglSafeMaxVertexAttribBindings(ctx);
            if (index >= maxBindings || index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            VertexArray *vao = ctx->state.vao;
            if (!vao) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }

            BufferBinding *binding = &vao->bindings[index];
            switch (target) {
                case GL_VERTEX_BINDING_BUFFER:
                    *data = binding->buffer ? (GLint)binding->buffer->name : 0;
                    break;
                case GL_VERTEX_BINDING_OFFSET:
                    *data = mglClampGetInteger64ToInt((GLint64)binding->offset, "vertex binding offset");
                    break;
                case GL_VERTEX_BINDING_STRIDE:
                    *data = (GLint)binding->stride;
                    break;
                case GL_VERTEX_BINDING_DIVISOR:
                    *data = (GLint)binding->divisor;
                    break;
            }
            break;
        }

        case GL_DRAW_BUFFER:
            if (index < MAX_COLOR_ATTACHMENTS)
            {
                *data = (index < (GLuint)ctx->state.draw_buffer_count)
                    ? ctx->state.draw_buffers[index]
                    : GL_NONE;
            }
            else
            {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
            break;

        case GL_BLEND:
            if (index < MAX_COLOR_ATTACHMENTS)
            {
                *data = ctx->state.caps.blendi[index] ? GL_TRUE : GL_FALSE;
            }
            else
            {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
            break;

        case GL_COLOR_WRITEMASK:
            if (index < MAX_COLOR_ATTACHMENTS)
            {
                data[0] = ctx->state.var.color_writemask[index][0] ? GL_TRUE : GL_FALSE;
                data[1] = ctx->state.var.color_writemask[index][1] ? GL_TRUE : GL_FALSE;
                data[2] = ctx->state.var.color_writemask[index][2] ? GL_TRUE : GL_FALSE;
                data[3] = ctx->state.var.color_writemask[index][3] ? GL_TRUE : GL_FALSE;
            }
            else
            {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
            break;

        case GL_BLEND_SRC_RGB:
            if (index < MAX_COLOR_ATTACHMENTS)
                *data = ctx->state.var.blend_src_rgb[index];
            else
                ERROR_RETURN(GL_INVALID_VALUE);
            break;

        case GL_BLEND_SRC_ALPHA:
            if (index < MAX_COLOR_ATTACHMENTS)
                *data = ctx->state.var.blend_src_alpha[index];
            else
                ERROR_RETURN(GL_INVALID_VALUE);
            break;

        case GL_BLEND_DST_RGB:
            if (index < MAX_COLOR_ATTACHMENTS)
                *data = ctx->state.var.blend_dst_rgb[index];
            else
                ERROR_RETURN(GL_INVALID_VALUE);
            break;

        case GL_BLEND_DST_ALPHA:
            if (index < MAX_COLOR_ATTACHMENTS)
                *data = ctx->state.var.blend_dst_alpha[index];
            else
                ERROR_RETURN(GL_INVALID_VALUE);
            break;

        case GL_BLEND_EQUATION_RGB:
            if (index < MAX_COLOR_ATTACHMENTS)
                *data = ctx->state.var.blend_equation_rgb[index];
            else
                ERROR_RETURN(GL_INVALID_VALUE);
            break;

        case GL_BLEND_EQUATION_ALPHA:
            if (index < MAX_COLOR_ATTACHMENTS)
                *data = ctx->state.var.blend_equation_alpha[index];
            else
                ERROR_RETURN(GL_INVALID_VALUE);
            break;

        case GL_MAX_COMPUTE_WORK_GROUP_COUNT:
            if (index < 3)
            {
                *data = ctx->state.var.max_compute_work_group_count[index];
            }
            else
            {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
            break;

        case GL_MAX_COMPUTE_WORK_GROUP_SIZE:
            if (index < 3)
            {
                *data = ctx->state.var.max_compute_work_group_size[index];
            }
            else
            {
                ERROR_RETURN(GL_INVALID_VALUE);
            }
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }
}

static GLenum mglCanonicalInternalFormat(GLenum internalformat)
{
    switch (internalformat) {
        case GL_RED: return GL_R8;
        case GL_RG: return GL_RG8;
        case GL_RGB: return GL_RGB8;
        case GL_RGBA: return GL_RGBA8;
        case GL_SRGB: return GL_SRGB8;
        case GL_SRGB_ALPHA: return GL_SRGB8_ALPHA8;
        case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT24;
        case GL_DEPTH_STENCIL: return GL_DEPTH24_STENCIL8;
        default: return internalformat;
    }
}

static GLboolean mglInternalformatTargetValid(GLenum target)
{
    switch (target) {
        case GL_RENDERBUFFER:
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_BUFFER:
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean mglInternalFormatIsCompressed(GLenum internalformat)
{
    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_COMPRESSED_RED:
        case GL_COMPRESSED_RG:
        case GL_COMPRESSED_RGB:
        case GL_COMPRESSED_RGBA:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
        case GL_COMPRESSED_RGB8_ETC2:
        case GL_COMPRESSED_SRGB8_ETC2:
        case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_RGBA8_ETC2_EAC:
        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
        case GL_COMPRESSED_R11_EAC:
        case GL_COMPRESSED_SIGNED_R11_EAC:
        case GL_COMPRESSED_RG11_EAC:
        case GL_COMPRESSED_SIGNED_RG11_EAC:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean mglInternalFormatHasDepth(GLenum internalformat)
{
    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean mglInternalFormatHasStencil(GLenum internalformat)
{
    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
        case GL_STENCIL_INDEX1:
        case GL_STENCIL_INDEX4:
        case GL_STENCIL_INDEX8:
        case GL_STENCIL_INDEX16:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean mglInternalFormatIsSignedInteger(GLenum internalformat)
{
    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_R8I:
        case GL_R16I:
        case GL_R32I:
        case GL_RG8I:
        case GL_RG16I:
        case GL_RG32I:
        case GL_RGB8I:
        case GL_RGB16I:
        case GL_RGB32I:
        case GL_RGBA8I:
        case GL_RGBA16I:
        case GL_RGBA32I:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean mglInternalFormatIsUnsignedInteger(GLenum internalformat)
{
    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_R8UI:
        case GL_R16UI:
        case GL_R32UI:
        case GL_RG8UI:
        case GL_RG16UI:
        case GL_RG32UI:
        case GL_RGB8UI:
        case GL_RGB16UI:
        case GL_RGB32UI:
        case GL_RGBA8UI:
        case GL_RGBA16UI:
        case GL_RGBA32UI:
        case GL_RGB10_A2UI:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean mglInternalFormatIsInteger(GLenum internalformat)
{
    return mglInternalFormatIsSignedInteger(internalformat) ||
           mglInternalFormatIsUnsignedInteger(internalformat);
}

static GLboolean mglInternalFormatIsSignedNormalized(GLenum internalformat)
{
    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_R8_SNORM:
        case GL_R16_SNORM:
        case GL_RG8_SNORM:
        case GL_RG16_SNORM:
        case GL_RGB8_SNORM:
        case GL_RGB16_SNORM:
        case GL_RGBA8_SNORM:
        case GL_RGBA16_SNORM:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_R11_EAC:
        case GL_COMPRESSED_SIGNED_RG11_EAC:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean mglInternalFormatIsFloat(GLenum internalformat)
{
    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_R16F:
        case GL_R32F:
        case GL_RG16F:
        case GL_RG32F:
        case GL_RGB16F:
        case GL_RGB32F:
        case GL_RGBA16F:
        case GL_RGBA32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH32F_STENCIL8:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLboolean mglInternalFormatIsSRGB(GLenum internalformat)
{
    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_SRGB8:
        case GL_SRGB8_ALPHA8:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB8_ETC2:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLenum mglInternalFormatBaseFormat(GLenum internalformat)
{
    GLenum storage = mglCanonicalInternalFormat(internalformat);
    if (mglInternalFormatHasDepth(storage) && mglInternalFormatHasStencil(storage)) {
        return GL_DEPTH_STENCIL;
    }
    if (mglInternalFormatHasDepth(storage)) {
        return GL_DEPTH_COMPONENT;
    }
    if (mglInternalFormatHasStencil(storage)) {
        return GL_STENCIL_INDEX;
    }

    switch (storage) {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_COMPRESSED_RED:
        case GL_COMPRESSED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
        case GL_COMPRESSED_R11_EAC:
        case GL_COMPRESSED_SIGNED_R11_EAC:
            return GL_RED;

        case GL_R8I:
        case GL_R16I:
        case GL_R32I:
        case GL_R8UI:
        case GL_R16UI:
        case GL_R32UI:
            return GL_RED_INTEGER;

        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_RG16F:
        case GL_RG32F:
        case GL_COMPRESSED_RG:
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
        case GL_COMPRESSED_RG11_EAC:
        case GL_COMPRESSED_SIGNED_RG11_EAC:
            return GL_RG;

        case GL_RG8I:
        case GL_RG16I:
        case GL_RG32I:
        case GL_RG8UI:
        case GL_RG16UI:
        case GL_RG32UI:
            return GL_RG_INTEGER;

        case GL_R3_G3_B2:
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB8:
        case GL_RGB8_SNORM:
        case GL_RGB10:
        case GL_RGB12:
        case GL_RGB16:
        case GL_RGB16_SNORM:
        case GL_SRGB8:
        case GL_RGB16F:
        case GL_RGB32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_RGB565:
        case GL_COMPRESSED_RGB:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
        case GL_COMPRESSED_RGB8_ETC2:
        case GL_COMPRESSED_SRGB8_ETC2:
            return GL_RGB;

        case GL_RGB8I:
        case GL_RGB16I:
        case GL_RGB32I:
        case GL_RGB8UI:
        case GL_RGB16UI:
        case GL_RGB32UI:
            return GL_RGB_INTEGER;

        case GL_RGBA2:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_RGBA8:
        case GL_RGBA8_SNORM:
        case GL_RGB10_A2:
        case GL_RGBA12:
        case GL_RGBA16:
        case GL_RGBA16_SNORM:
        case GL_SRGB8_ALPHA8:
        case GL_RGBA16F:
        case GL_RGBA32F:
        case GL_COMPRESSED_RGBA:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_RGBA8_ETC2_EAC:
        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
            return GL_RGBA;

        case GL_RGBA8I:
        case GL_RGBA16I:
        case GL_RGBA32I:
        case GL_RGBA8UI:
        case GL_RGBA16UI:
        case GL_RGBA32UI:
        case GL_RGB10_A2UI:
            return GL_RGBA_INTEGER;

        default:
            return GL_NONE;
    }
}

static GLboolean mglInternalFormatIsColor(GLenum internalformat)
{
    GLenum base = mglInternalFormatBaseFormat(internalformat);
    return base == GL_RED || base == GL_RG || base == GL_RGB || base == GL_RGBA ||
           base == GL_RED_INTEGER || base == GL_RG_INTEGER ||
           base == GL_RGB_INTEGER || base == GL_RGBA_INTEGER;
}

static GLuint mglInternalFormatComponentBits(GLenum internalformat, GLenum component)
{
    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R8I:
        case GL_R8UI:
            return component == GL_RED ? 8u : 0u;
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R16I:
        case GL_R16UI:
            return component == GL_RED ? 16u : 0u;
        case GL_R32F:
        case GL_R32I:
        case GL_R32UI:
            return component == GL_RED ? 32u : 0u;

        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RG8I:
        case GL_RG8UI:
            return (component == GL_RED || component == GL_GREEN) ? 8u : 0u;
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_RG16F:
        case GL_RG16I:
        case GL_RG16UI:
            return (component == GL_RED || component == GL_GREEN) ? 16u : 0u;
        case GL_RG32F:
        case GL_RG32I:
        case GL_RG32UI:
            return (component == GL_RED || component == GL_GREEN) ? 32u : 0u;

        case GL_R3_G3_B2:
            switch (component) {
                case GL_RED: return 3u;
                case GL_GREEN: return 3u;
                case GL_BLUE: return 2u;
                default: return 0u;
            }
        case GL_RGB4:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 4u : 0u;
        case GL_RGB5:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 5u : 0u;
        case GL_RGB565:
            switch (component) {
                case GL_RED: return 5u;
                case GL_GREEN: return 6u;
                case GL_BLUE: return 5u;
                default: return 0u;
            }
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB8_SNORM:
        case GL_RGB8I:
        case GL_RGB8UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 8u : 0u;
        case GL_RGB10:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 10u : 0u;
        case GL_RGB12:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 12u : 0u;
        case GL_RGB16:
        case GL_RGB16_SNORM:
        case GL_RGB16F:
        case GL_RGB16I:
        case GL_RGB16UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 16u : 0u;
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 32u : 0u;

        case GL_RGBA2:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE || component == GL_ALPHA) ? 2u : 0u;
        case GL_RGBA4:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE || component == GL_ALPHA) ? 4u : 0u;
        case GL_RGB5_A1:
            switch (component) {
                case GL_RED:
                case GL_GREEN:
                case GL_BLUE: return 5u;
                case GL_ALPHA: return 1u;
                default: return 0u;
            }
        case GL_RGBA8:
        case GL_SRGB8_ALPHA8:
        case GL_RGBA8_SNORM:
        case GL_RGBA8I:
        case GL_RGBA8UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE || component == GL_ALPHA) ? 8u : 0u;
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
            switch (component) {
                case GL_RED:
                case GL_GREEN:
                case GL_BLUE: return 10u;
                case GL_ALPHA: return 2u;
                default: return 0u;
            }
        case GL_RGBA12:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE || component == GL_ALPHA) ? 12u : 0u;
        case GL_RGBA16:
        case GL_RGBA16_SNORM:
        case GL_RGBA16F:
        case GL_RGBA16I:
        case GL_RGBA16UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE || component == GL_ALPHA) ? 16u : 0u;
        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE || component == GL_ALPHA) ? 32u : 0u;

        case GL_R11F_G11F_B10F:
            switch (component) {
                case GL_RED:
                case GL_GREEN: return 11u;
                case GL_BLUE: return 10u;
                default: return 0u;
            }
        case GL_RGB9_E5:
            return (component == GL_RED || component == GL_GREEN || component == GL_BLUE) ? 9u : 0u;

        case GL_DEPTH_COMPONENT16:
            return component == GL_DEPTH ? 16u : 0u;
        case GL_DEPTH_COMPONENT24:
            return component == GL_DEPTH ? 24u : 0u;
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            return component == GL_DEPTH ? 32u : 0u;
        case GL_DEPTH24_STENCIL8:
            return component == GL_DEPTH ? 24u : (component == GL_STENCIL ? 8u : 0u);
        case GL_DEPTH32F_STENCIL8:
            return component == GL_DEPTH ? 32u : (component == GL_STENCIL ? 8u : 0u);
        case GL_STENCIL_INDEX1:
            return component == GL_STENCIL ? 1u : 0u;
        case GL_STENCIL_INDEX4:
            return component == GL_STENCIL ? 4u : 0u;
        case GL_STENCIL_INDEX8:
            return component == GL_STENCIL ? 8u : 0u;
        case GL_STENCIL_INDEX16:
            return component == GL_STENCIL ? 16u : 0u;

        default:
            if (mglInternalFormatIsCompressed(internalformat)) {
                return 0u;
            }
            return bitcountForInternalFormat(mglCanonicalInternalFormat(internalformat), component);
    }
}

static GLenum mglInternalFormatComponentType(GLenum internalformat, GLenum component)
{
    if (mglInternalFormatComponentBits(internalformat, component) == 0u) {
        return GL_NONE;
    }
    if (component == GL_STENCIL) {
        return GL_UNSIGNED_INT;
    }
    if (mglInternalFormatIsSignedInteger(internalformat)) {
        return GL_INT;
    }
    if (mglInternalFormatIsUnsignedInteger(internalformat)) {
        return GL_UNSIGNED_INT;
    }
    if (mglInternalFormatIsFloat(internalformat)) {
        return GL_FLOAT;
    }
    if (mglInternalFormatIsSignedNormalized(internalformat)) {
        return GL_SIGNED_NORMALIZED;
    }
    return GL_UNSIGNED_NORMALIZED;
}

static GLenum mglInternalFormatImageType(GLenum internalformat)
{
    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_R8I:
        case GL_RG8I:
        case GL_RGB8I:
        case GL_RGBA8I:
            return GL_BYTE;
        case GL_R8UI:
        case GL_RG8UI:
        case GL_RGB8UI:
        case GL_RGBA8UI:
        case GL_R8:
        case GL_RG8:
        case GL_RGB8:
        case GL_RGBA8:
        case GL_SRGB8:
        case GL_SRGB8_ALPHA8:
        case GL_R8_SNORM:
        case GL_RG8_SNORM:
        case GL_RGB8_SNORM:
        case GL_RGBA8_SNORM:
        case GL_STENCIL_INDEX8:
            return GL_UNSIGNED_BYTE;
        case GL_R16I:
        case GL_RG16I:
        case GL_RGB16I:
        case GL_RGBA16I:
            return GL_SHORT;
        case GL_R16UI:
        case GL_RG16UI:
        case GL_RGB16UI:
        case GL_RGBA16UI:
        case GL_R16:
        case GL_RG16:
        case GL_RGB16:
        case GL_RGBA16:
        case GL_R16_SNORM:
        case GL_RG16_SNORM:
        case GL_RGB16_SNORM:
        case GL_RGBA16_SNORM:
        case GL_DEPTH_COMPONENT16:
        case GL_STENCIL_INDEX16:
            return GL_UNSIGNED_SHORT;
        case GL_R32I:
        case GL_RG32I:
        case GL_RGB32I:
        case GL_RGBA32I:
            return GL_INT;
        case GL_R32UI:
        case GL_RG32UI:
        case GL_RGB32UI:
        case GL_RGBA32UI:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
            return GL_UNSIGNED_INT;
        case GL_R16F:
        case GL_RG16F:
        case GL_RGB16F:
        case GL_RGBA16F:
            return GL_HALF_FLOAT;
        case GL_R32F:
        case GL_RG32F:
        case GL_RGB32F:
        case GL_RGBA32F:
        case GL_DEPTH_COMPONENT32F:
            return GL_FLOAT;
        case GL_R3_G3_B2:
            return GL_UNSIGNED_BYTE_3_3_2;
        case GL_RGB565:
            return GL_UNSIGNED_SHORT_5_6_5;
        case GL_RGBA4:
            return GL_UNSIGNED_SHORT_4_4_4_4;
        case GL_RGB5_A1:
            return GL_UNSIGNED_SHORT_5_5_5_1;
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
            return GL_UNSIGNED_INT_2_10_10_10_REV;
        case GL_R11F_G11F_B10F:
            return GL_UNSIGNED_INT_10F_11F_11F_REV;
        case GL_RGB9_E5:
            return GL_UNSIGNED_INT_5_9_9_9_REV;
        case GL_DEPTH24_STENCIL8:
            return GL_UNSIGNED_INT_24_8;
        case GL_DEPTH32F_STENCIL8:
            return GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
        default:
            return GL_UNSIGNED_BYTE;
    }
}

static void mglInternalFormatCompressedBlock(GLenum internalformat, GLint64 *width, GLint64 *height, GLint64 *size)
{
    *width = 0;
    *height = 0;
    *size = 0;

    switch (mglCanonicalInternalFormat(internalformat)) {
        case GL_COMPRESSED_RGB:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_RED_RGTC1:
        case GL_COMPRESSED_SIGNED_RED_RGTC1:
        case GL_COMPRESSED_RGB8_ETC2:
        case GL_COMPRESSED_SRGB8_ETC2:
        case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
        case GL_COMPRESSED_R11_EAC:
        case GL_COMPRESSED_SIGNED_R11_EAC:
            *width = 4;
            *height = 4;
            *size = 8;
            break;
        case GL_COMPRESSED_RG:
        case GL_COMPRESSED_RGBA:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_RG_RGTC2:
        case GL_COMPRESSED_SIGNED_RG_RGTC2:
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
        case GL_COMPRESSED_RGBA8_ETC2_EAC:
        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
        case GL_COMPRESSED_RG11_EAC:
        case GL_COMPRESSED_SIGNED_RG11_EAC:
            *width = 4;
            *height = 4;
            *size = 16;
            break;
        default:
            break;
    }
}

static GLboolean mglInternalFormatIsRenderable(GLenum internalformat)
{
    GLenum storage = mglCanonicalInternalFormat(internalformat);
    if (mglInternalFormatIsCompressed(storage)) {
        return GL_FALSE;
    }
    if (!mglInternalFormatIsColor(storage) &&
        !mglInternalFormatHasDepth(storage) &&
        !mglInternalFormatHasStencil(storage)) {
        return GL_FALSE;
    }
    return mtlFormatForGLInternalFormat(storage) != MTLPixelFormatInvalid;
}

static GLboolean mglInternalFormatTargetSupported(GLenum target, GLenum internalformat)
{
    GLenum storage = mglCanonicalInternalFormat(internalformat);
    if (!validInternalFormat(internalformat) ||
        !mglInternalformatTargetValid(target) ||
        mtlFormatForGLInternalFormat(storage) == MTLPixelFormatInvalid) {
        return GL_FALSE;
    }

    switch (target) {
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return GL_FALSE;
        case GL_TEXTURE_BUFFER:
            return mglInternalFormatIsColor(storage) && !mglInternalFormatIsCompressed(storage);
        case GL_RENDERBUFFER:
            return mglInternalFormatIsRenderable(storage);
        default:
            return GL_TRUE;
    }
}

static GLboolean mglInternalFormatTargetLayered(GLenum target)
{
    switch (target) {
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return GL_TRUE;
        default:
            return GL_FALSE;
    }
}

static GLint64 mglPositiveStateOrFallback(GLuint value, GLint64 fallback)
{
    if (value == 0u || value == 0x01010101u || value > (GLuint)INT_MAX) {
        return fallback;
    }
    return (GLint64)value;
}

static GLint64 mglInternalFormatMaxWidth(GLMContext ctx, GLenum target)
{
    switch (target) {
        case GL_RENDERBUFFER:
            return mglPositiveStateOrFallback(ctx ? ctx->state.var.max_renderbuffer_size : 0u, 16384);
        case GL_TEXTURE_BUFFER:
            return mglSafeMaxTextureBufferSize(ctx);
        case GL_TEXTURE_RECTANGLE:
            return mglPositiveStateOrFallback(ctx ? ctx->state.var.max_rectangle_texture_size : 0u, 16384);
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            return mglPositiveStateOrFallback(ctx ? ctx->state.var.max_cube_map_texture_size : 0u, 16384);
        case GL_TEXTURE_3D:
            return mglPositiveStateOrFallback(ctx ? ctx->state.var.max_3d_texture_size : 0u, 2048);
        default:
            return mglSafeMaxTextureSize(ctx);
    }
}

static GLint64 mglInternalFormatMaxHeight(GLMContext ctx, GLenum target)
{
    switch (target) {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_BUFFER:
            return 1;
        case GL_TEXTURE_3D:
            return mglPositiveStateOrFallback(ctx ? ctx->state.var.max_3d_texture_size : 0u, 2048);
        default:
            return mglInternalFormatMaxWidth(ctx, target);
    }
}

static GLint64 mglInternalFormatMaxDepth(GLMContext ctx, GLenum target)
{
    return target == GL_TEXTURE_3D
        ? mglPositiveStateOrFallback(ctx ? ctx->state.var.max_3d_texture_size : 0u, 2048)
        : 1;
}

static GLint64 mglInternalFormatMaxLayers(GLMContext ctx, GLenum target)
{
    switch (target) {
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return mglPositiveStateOrFallback(ctx ? ctx->state.var.max_array_texture_layers : 0u, 2048);
        case GL_TEXTURE_CUBE_MAP:
            return 6;
        default:
            return 1;
    }
}

static GLint64 mglSaturatingMul64(GLint64 a, GLint64 b)
{
    if (a <= 0 || b <= 0) {
        return 0;
    }
    if (a > INT64_MAX / b) {
        return INT64_MAX;
    }
    return a * b;
}

static GLenum mglInternalFormatViewClass(GLenum internalformat)
{
    GLenum storage = mglCanonicalInternalFormat(internalformat);
    if (mglInternalFormatIsCompressed(storage)) {
        switch (storage) {
            case GL_COMPRESSED_RED_RGTC1:
            case GL_COMPRESSED_SIGNED_RED_RGTC1:
                return GL_VIEW_CLASS_RGTC1_RED;
            case GL_COMPRESSED_RG_RGTC2:
            case GL_COMPRESSED_SIGNED_RG_RGTC2:
                return GL_VIEW_CLASS_RGTC2_RG;
            case GL_COMPRESSED_RGBA_BPTC_UNORM:
            case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
                return GL_VIEW_CLASS_BPTC_UNORM;
            case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
            case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
                return GL_VIEW_CLASS_BPTC_FLOAT;
            default:
                return GL_NONE;
        }
    }

    GLuint bits =
        mglInternalFormatComponentBits(storage, GL_RED) +
        mglInternalFormatComponentBits(storage, GL_GREEN) +
        mglInternalFormatComponentBits(storage, GL_BLUE) +
        mglInternalFormatComponentBits(storage, GL_ALPHA) +
        mglInternalFormatComponentBits(storage, GL_DEPTH) +
        mglInternalFormatComponentBits(storage, GL_STENCIL);

    if (bits <= 8u) return GL_VIEW_CLASS_8_BITS;
    if (bits <= 16u) return GL_VIEW_CLASS_16_BITS;
    if (bits <= 24u) return GL_VIEW_CLASS_24_BITS;
    if (bits <= 32u) return GL_VIEW_CLASS_32_BITS;
    if (bits <= 48u) return GL_VIEW_CLASS_48_BITS;
    if (bits <= 64u) return GL_VIEW_CLASS_64_BITS;
    if (bits <= 96u) return GL_VIEW_CLASS_96_BITS;
    return GL_VIEW_CLASS_128_BITS;
}

static GLint64 mglInternalFormatSupportValue(GLboolean supported)
{
    return supported ? GL_FULL_SUPPORT : GL_NONE;
}

static GLsizei mglGetInternalformatValues(GLMContext ctx, GLenum target, GLenum internalformat, GLenum pname, GLint64 *params, GLsizei count)
{
    if (!mglInternalformatTargetValid(target)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return -1;
    }
    if (!validInternalFormat(internalformat)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return -1;
    }

    GLenum storage = mglCanonicalInternalFormat(internalformat);
    GLboolean supported = mglInternalFormatTargetSupported(target, internalformat);
    GLint64 value = 0;

    switch (pname) {
        case GL_INTERNALFORMAT_SUPPORTED:
            value = supported ? GL_TRUE : GL_FALSE;
            break;
        case GL_INTERNALFORMAT_PREFERRED:
            value = supported ? storage : GL_NONE;
            break;
        case GL_INTERNALFORMAT_RED_SIZE:
            value = supported ? mglInternalFormatComponentBits(storage, GL_RED) : 0;
            break;
        case GL_INTERNALFORMAT_GREEN_SIZE:
            value = supported ? mglInternalFormatComponentBits(storage, GL_GREEN) : 0;
            break;
        case GL_INTERNALFORMAT_BLUE_SIZE:
            value = supported ? mglInternalFormatComponentBits(storage, GL_BLUE) : 0;
            break;
        case GL_INTERNALFORMAT_ALPHA_SIZE:
            value = supported ? mglInternalFormatComponentBits(storage, GL_ALPHA) : 0;
            break;
        case GL_INTERNALFORMAT_DEPTH_SIZE:
            value = supported ? mglInternalFormatComponentBits(storage, GL_DEPTH) : 0;
            break;
        case GL_INTERNALFORMAT_STENCIL_SIZE:
            value = supported ? mglInternalFormatComponentBits(storage, GL_STENCIL) : 0;
            break;
        case GL_INTERNALFORMAT_SHARED_SIZE:
            value = supported && storage == GL_RGB9_E5 ? 5 : 0;
            break;
        case GL_INTERNALFORMAT_RED_TYPE:
            value = supported ? mglInternalFormatComponentType(storage, GL_RED) : GL_NONE;
            break;
        case GL_INTERNALFORMAT_GREEN_TYPE:
            value = supported ? mglInternalFormatComponentType(storage, GL_GREEN) : GL_NONE;
            break;
        case GL_INTERNALFORMAT_BLUE_TYPE:
            value = supported ? mglInternalFormatComponentType(storage, GL_BLUE) : GL_NONE;
            break;
        case GL_INTERNALFORMAT_ALPHA_TYPE:
            value = supported ? mglInternalFormatComponentType(storage, GL_ALPHA) : GL_NONE;
            break;
        case GL_INTERNALFORMAT_DEPTH_TYPE:
            value = supported ? mglInternalFormatComponentType(storage, GL_DEPTH) : GL_NONE;
            break;
        case GL_INTERNALFORMAT_STENCIL_TYPE:
            value = supported ? mglInternalFormatComponentType(storage, GL_STENCIL) : GL_NONE;
            break;
        case GL_MAX_WIDTH:
            value = supported ? mglInternalFormatMaxWidth(ctx, target) : 0;
            break;
        case GL_MAX_HEIGHT:
            value = supported ? mglInternalFormatMaxHeight(ctx, target) : 0;
            break;
        case GL_MAX_DEPTH:
            value = supported ? mglInternalFormatMaxDepth(ctx, target) : 0;
            break;
        case GL_MAX_LAYERS:
            value = supported ? mglInternalFormatMaxLayers(ctx, target) : 0;
            break;
        case GL_MAX_COMBINED_DIMENSIONS:
            value = supported
                ? mglSaturatingMul64(
                    mglSaturatingMul64(mglInternalFormatMaxWidth(ctx, target),
                                       mglInternalFormatMaxHeight(ctx, target)),
                    mglSaturatingMul64(mglInternalFormatMaxDepth(ctx, target),
                                       mglInternalFormatMaxLayers(ctx, target)))
                : 0;
            break;
        case GL_COLOR_COMPONENTS:
            value = supported && mglInternalFormatIsColor(storage) ? GL_TRUE : GL_FALSE;
            break;
        case GL_DEPTH_COMPONENTS:
            value = supported && mglInternalFormatHasDepth(storage) ? GL_TRUE : GL_FALSE;
            break;
        case GL_STENCIL_COMPONENTS:
            value = supported && mglInternalFormatHasStencil(storage) ? GL_TRUE : GL_FALSE;
            break;
        case GL_COLOR_RENDERABLE:
            value = mglInternalFormatSupportValue(supported && mglInternalFormatIsColor(storage) && mglInternalFormatIsRenderable(storage));
            break;
        case GL_DEPTH_RENDERABLE:
            value = mglInternalFormatSupportValue(supported && mglInternalFormatHasDepth(storage) && mglInternalFormatIsRenderable(storage));
            break;
        case GL_STENCIL_RENDERABLE:
            value = mglInternalFormatSupportValue(supported && mglInternalFormatHasStencil(storage) && mglInternalFormatIsRenderable(storage));
            break;
        case GL_FRAMEBUFFER_RENDERABLE:
            value = mglInternalFormatSupportValue(supported && mglInternalFormatIsRenderable(storage));
            break;
        case GL_FRAMEBUFFER_RENDERABLE_LAYERED:
            value = mglInternalFormatSupportValue(supported && mglInternalFormatIsRenderable(storage) && mglInternalFormatTargetLayered(target));
            break;
        case GL_FRAMEBUFFER_BLEND:
            value = mglInternalFormatSupportValue(supported && mglInternalFormatIsRenderable(storage) &&
                                                  mglInternalFormatIsColor(storage) &&
                                                  !mglInternalFormatIsInteger(storage));
            break;
        case GL_READ_PIXELS:
            value = mglInternalFormatSupportValue(supported && mglInternalFormatIsRenderable(storage) &&
                                                  mglInternalFormatIsColor(storage));
            break;
        case GL_READ_PIXELS_FORMAT:
        case GL_TEXTURE_IMAGE_FORMAT:
        case GL_GET_TEXTURE_IMAGE_FORMAT:
        case GL_IMAGE_PIXEL_FORMAT:
            value = supported ? mglInternalFormatBaseFormat(storage) : GL_NONE;
            break;
        case GL_READ_PIXELS_TYPE:
        case GL_TEXTURE_IMAGE_TYPE:
        case GL_GET_TEXTURE_IMAGE_TYPE:
        case GL_IMAGE_PIXEL_TYPE:
            value = supported ? mglInternalFormatImageType(storage) : GL_NONE;
            break;
        case GL_MIPMAP:
        case GL_MANUAL_GENERATE_MIPMAP:
            value = mglInternalFormatSupportValue(supported &&
                                                  target != GL_RENDERBUFFER &&
                                                  target != GL_TEXTURE_BUFFER &&
                                                  target != GL_TEXTURE_RECTANGLE);
            break;
        case GL_AUTO_GENERATE_MIPMAP:
            value = GL_NONE;
            break;
        case GL_COLOR_ENCODING:
            value = supported ? (mglInternalFormatIsSRGB(storage) ? GL_SRGB : GL_LINEAR) : GL_NONE;
            break;
        case GL_SRGB_READ:
        case GL_SRGB_WRITE:
            value = mglInternalFormatSupportValue(supported && mglInternalFormatIsSRGB(storage));
            break;
        case GL_FILTER:
        case GL_TEXTURE_GATHER:
            value = mglInternalFormatSupportValue(supported &&
                                                  !mglInternalFormatIsInteger(storage) &&
                                                  !mglInternalFormatHasStencil(storage));
            break;
        case GL_VERTEX_TEXTURE:
        case GL_TESS_CONTROL_TEXTURE:
        case GL_TESS_EVALUATION_TEXTURE:
        case GL_GEOMETRY_TEXTURE:
        case GL_FRAGMENT_TEXTURE:
            value = mglInternalFormatSupportValue(supported && target != GL_RENDERBUFFER);
            break;
        case GL_COMPUTE_TEXTURE:
        case GL_SHADER_IMAGE_LOAD:
        case GL_SHADER_IMAGE_STORE:
        case GL_SHADER_IMAGE_ATOMIC:
            value = GL_NONE;
            break;
        case GL_TEXTURE_SHADOW:
        case GL_TEXTURE_GATHER_SHADOW:
            value = mglInternalFormatSupportValue(supported && mglInternalFormatHasDepth(storage));
            break;
        case GL_IMAGE_TEXEL_SIZE:
            value = supported && !mglInternalFormatIsCompressed(storage) ? sizeForInternalFormat(storage, 0, 0) : 0;
            break;
        case GL_IMAGE_COMPATIBILITY_CLASS:
            value = GL_NONE;
            break;
        case GL_TEXTURE_COMPRESSED_BLOCK_WIDTH:
        case GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT:
        case GL_TEXTURE_COMPRESSED_BLOCK_SIZE:
        {
            GLint64 blockWidth = 0;
            GLint64 blockHeight = 0;
            GLint64 blockSize = 0;
            if (supported) {
                mglInternalFormatCompressedBlock(storage, &blockWidth, &blockHeight, &blockSize);
            }
            value = (pname == GL_TEXTURE_COMPRESSED_BLOCK_WIDTH) ? blockWidth :
                    (pname == GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT) ? blockHeight : blockSize;
            break;
        }
        case GL_CLEAR_BUFFER:
            value = mglInternalFormatSupportValue(supported && mglInternalFormatIsRenderable(storage));
            break;
        case GL_TEXTURE_VIEW:
            value = GL_NONE;
            break;
        case GL_VIEW_COMPATIBILITY_CLASS:
            value = supported ? mglInternalFormatViewClass(storage) : GL_NONE;
            break;
        case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST:
        case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST:
        case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE:
        case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE:
            value = mglInternalFormatSupportValue(supported && target != GL_RENDERBUFFER);
            break;
        case GL_NUM_SAMPLE_COUNTS:
            value = supported ? 1 : 0;
            break;
        case GL_SAMPLES:
            if (supported && count > 0) {
                params[0] = 0;
            }
            return supported ? 1 : 0;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return -1;
    }

    if (count > 0) {
        params[0] = value;
    }
    return 1;
}

void mglGetInternalformati64v(GLMContext ctx, GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint64 *params)
{
    if (count < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (count == 0) {
        return;
    }
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mglGetInternalformatValues(ctx, target, internalformat, pname, params, count);
}

void mglGetInternalformativ(GLMContext ctx, GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint *params)
{
    if (count < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (count == 0) {
        return;
    }
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    GLint64 values[16] = {0};
    GLsizei maxCount = count < (GLsizei)(sizeof(values) / sizeof(values[0]))
        ? count
        : (GLsizei)(sizeof(values) / sizeof(values[0]));
    GLsizei written = mglGetInternalformatValues(ctx, target, internalformat, pname, values, maxCount);
    if (written < 0) {
        return;
    }
    if (written > maxCount) {
        written = maxCount;
    }

    for (GLsizei i = 0; i < written; ++i) {
        if (values[i] > INT_MAX) {
            params[i] = INT_MAX;
        } else if (values[i] < INT_MIN) {
            params[i] = INT_MIN;
        } else {
            params[i] = (GLint)values[i];
        }
    }
}
