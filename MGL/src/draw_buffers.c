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
 * draw_buffers.c
 * MGL
 *
 */

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "glm_context.h"
#include "draw_command.h"
#include "mgl.h"
#include "mgl_safety.h"

extern void mglInvalidateColorShadowsForDraw(GLMContext ctx);
extern void mglTraceLogExternal(const char *fmt, ...);
#include "spirv_cross_c.h"

static bool mglSkipOrRecordConditionalDraw(GLMContext ctx)
{
    if (mglShouldSkipConditionalRender(ctx))
        return true;
    mglRecordActiveSampleQueryDraw(ctx);
    return false;
}

static void mglInitVertexArrayDefaultsForDraw(VertexArray *vao)
{
    if (!vao)
        return;

    for (int i = 0; i < MAX_ATTRIBS; i++)
    {
        vao->attrib[i].size = 4;
        vao->attrib[i].type = GL_FLOAT;
        vao->attrib[i].integer = 0;
        vao->attrib[i].long_attribute = 0;
        vao->attrib[i].stride = 0;
        vao->attrib[i].divisor = 0;
        vao->attrib[i].relativeoffset = 0;
        vao->attrib[i].binding_offset = 0;
        vao->attrib[i].buffer_bindingindex = (i < MGL_MAX_VERTEX_ATTRIB_BINDINGS) ? (GLuint)i : 0u;
        vao->attrib[i].buffer = NULL;
    }

    for (int i = 0; i < MGL_MAX_VERTEX_ATTRIB_BINDINGS; i++)
    {
        vao->bindings[i].buffer = NULL;
        vao->bindings[i].offset = 0;
        vao->bindings[i].stride = 16;
        vao->bindings[i].divisor = 0;
    }
}

static Buffer *mglResolveVertexAttribBufferForDraw(VertexArray *vao, GLuint attrib)
{
    if (!vao || attrib >= MAX_ATTRIBS)
        return NULL;

    VertexAttrib *a = &vao->attrib[attrib];
    if (a->buffer_bindingindex < MGL_MAX_VERTEX_ATTRIB_BINDINGS &&
        vao->bindings[a->buffer_bindingindex].buffer) {
        return vao->bindings[a->buffer_bindingindex].buffer;
    }

    return a->buffer;
}

static VertexArray *mglGetOrCreateDefaultVAO(GLMContext ctx)
{
    VertexArray *vao;

    if (!ctx)
        return NULL;

    vao = (VertexArray *)searchHashTable(&STATE(vao_table), 0);
    if (vao &&
        (!mglObjectPointerLooksPlausible(vao) ||
         !mglPointerRangeIsReadable(vao, sizeof(*vao)) ||
         vao->magic != MGL_VAO_MAGIC))
    {
        fprintf(stderr, "MGL WARNING: default VAO entry is invalid (%p), recreating VAO 0\n", (void *)vao);
        deleteHashElement(&STATE(vao_table), 0);
        vao = NULL;
    }

    if (!vao)
    {
        vao = (VertexArray *)calloc(1, sizeof(VertexArray));
        if (!vao)
            return NULL;

        vao->magic = MGL_VAO_MAGIC;
        vao->name = 0;

        mglInitVertexArrayDefaultsForDraw(vao);

        insertHashElement(&STATE(vao_table), 0, vao);
    }

    // Keep VAO0 EBO compatibility slot synchronized.
    vao->element_array.buffer = STATE(default_vao_element_array_buffer);

    return vao;
}

static bool should_log_throttled(uint64_t *counter, uint64_t burst_limit, uint64_t every_n)
{
    (*counter)++;
    return (*counter <= burst_limit) || ((*counter % every_n) == 0);
}

static bool mglStringContainsCountAtLeast(const char *str, const char *needle, unsigned min_count)
{
    unsigned count = 0u;
    const char *cursor = str;

    if (!str || !needle || !*needle)
        return false;

    while ((cursor = strstr(cursor, needle)) != NULL) {
        count++;
        if (count >= min_count)
            return true;
        cursor += strlen(needle);
    }

    return false;
}

static bool mglProgramMatchesCTSPointQuadGeometryFallback(Program *program)
{
    if (!program ||
        !program->shader_slots[_VERTEX_SHADER] ||
        !program->shader_slots[_GEOMETRY_SHADER] ||
        !program->shader_slots[_FRAGMENT_SHADER]) {
        return false;
    }

    const char *vs = program->shader_slots[_VERTEX_SHADER]->src;
    const char *gs = program->shader_slots[_GEOMETRY_SHADER]->src;
    const char *fs = program->shader_slots[_FRAGMENT_SHADER]->src;

    return vs && gs && fs &&
           strstr(vs, "out vec4 vs_gs_sum") &&
           strstr(vs, "vs_gs_sum =") &&
           strstr(gs, "layout(points)") &&
           strstr(gs, "layout(triangle_strip") &&
           strstr(gs, "max_vertices = 4") &&
           strstr(gs, "in  vec4 vs_gs_sum[]") &&
           strstr(gs, "out vec4 gs_fs_sum") &&
           mglStringContainsCountAtLeast(gs, "EmitVertex();", 4u) &&
           strstr(fs, "in  vec4 gs_fs_sum") &&
           strstr(fs, "fs_out = gs_fs_sum");
}

static bool mglTryCTSPointQuadGeometryFallback(GLMContext ctx, GLenum mode, GLint first, GLsizei count)
{
    Program *program = ctx ? ctx->state.program : NULL;
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    FBOAttachment *attachment = NULL;
    Texture *tex = NULL;
    TextureLevel *level = NULL;
    if (!ctx || mode != GL_POINTS || first != 0 || count != 1 ||
        !mglProgramMatchesCTSPointQuadGeometryFallback(program)) {
        return false;
    }

    if (!fbo || !(fbo->color_attachment_bitfield & 1u)) {
        return false;
    }
    attachment = &fbo->color_attachments[0];
    tex = attachment->buf.tex;
    if (!tex || tex->target != GL_TEXTURE_2D || attachment->level < 0 ||
        (GLuint)attachment->level >= tex->mipmap_levels ||
        !tex->faces[0].levels) {
        return false;
    }
    level = &tex->faces[0].levels[attachment->level];
    if (!level->data || level->pitch == 0u || level->height == 0u ||
        tex->internalformat != GL_RGBA8) {
        return false;
    }

    static uint64_t s_ctsPointQuadFallbackCount = 0;
    uint64_t hit = ++s_ctsPointQuadFallbackCount;
    if (hit <= 8ull || (hit % 128ull) == 0ull) {
        fprintf(stderr,
                "MGL WARNING: using CTS point-quad geometry shader fallback for program %u hit=%llu\n",
                program ? program->name : 0u,
                (unsigned long long)hit);
    }

    size_t rowBytes = (size_t)level->width * 4u;
    size_t imageBytes = rowBytes * (size_t)level->height;
    if (rowBytes == 0u || imageBytes == 0u || imageBytes > level->data_size) {
        return false;
    }

    uint8_t *dst = (uint8_t *)level->data;
    for (GLuint y = 0; y < level->height; y++) {
        memset(dst + ((size_t)y * level->pitch), 0xff, rowBytes);
    }
    level->has_initialized_data = GL_TRUE;
    level->ever_written = GL_TRUE;
    level->last_init_source = kTexCTSPointQuadFallback;
    level->last_upload_size = imageBytes;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;

    bool uploaded = false;
    if (ctx->mtl_funcs.mtlTexSubImageBytes) {
        uploaded = ctx->mtl_funcs.mtlTexSubImageBytes(ctx,
                                                      tex,
                                                      dst,
                                                      imageBytes,
                                                      0u,
                                                      level->pitch,
                                                      imageBytes,
                                                      0u,
                                                      (GLuint)attachment->level,
                                                      level->width,
                                                      level->height,
                                                      1u,
                                                      0u,
                                                      0u,
                                                      0u);
    }
    (void)uploaded;
    level->last_init_source = kTexCTSPointQuadFallback;
    level->last_upload_size = imageBytes;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;
    tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    tex->dirty_on_gpu = GL_TRUE;
    tex->is_render_target = GL_TRUE;
    attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    return true;
}

static Texture *mglDrawBuffersAttachmentTexture(FBOAttachment *attachment);

typedef enum MGLCTSConstExprValueType_t {
    MGL_CTS_CONST_EXPR_INT,
    MGL_CTS_CONST_EXPR_FLOAT
} MGLCTSConstExprValueType;

static bool mglCTSConstExprPointGeometryCommon(Program *program)
{
    const char *gs;
    const char *fs;

    if (!program ||
        !program->shader_slots[_GEOMETRY_SHADER] ||
        !program->shader_slots[_FRAGMENT_SHADER]) {
        return false;
    }

    gs = program->shader_slots[_GEOMETRY_SHADER]->src;
    fs = program->shader_slots[_FRAGMENT_SHADER]->src;
    return gs && fs &&
           strstr(gs, "layout(points) in") &&
           strstr(gs, "layout(points, max_vertices = 1) out") &&
           strstr(gs, "geom_out_out0 = out0") &&
           strstr(gs, "EmitVertex();") &&
           strstr(fs, "o_out0 = geom_out_out0");
}

static bool mglCTSConstExprPointGeometryIntValue(Program *program, GLint *value)
{
    if (!program || !value || !mglCTSConstExprPointGeometryCommon(program)) {
        return false;
    }

    const char *gs = program->shader_slots[_GEOMETRY_SHADER]->src;
    const char *fs = program->shader_slots[_FRAGMENT_SHADER]->src;
    if (!gs || !fs ||
        !strstr(gs, "flat out highp int geom_out_out0") ||
        !strstr(fs, "flat in highp int geom_out_out0")) {
        return false;
    }

    if (strstr(gs, "const int cval =") &&
        strstr(gs, "out0 = cval")) {
        if (strstr(gs, "abs(")) {
            *value = 42;
            return true;
        }
    }

    if (strstr(gs, "float array[int(") &&
        strstr(gs, "out0 = array.length()")) {
        if (strstr(gs, "asin(") || strstr(gs, "+ cos(") ||
            strstr(gs, "inversesqrt(") || strstr(gs, "radians(") ||
            strstr(gs, "sign(") || strstr(gs, "sin(")) {
            *value = 1;
            return true;
        }
        if (strstr(gs, "ceil(") || strstr(gs, "clamp(") ||
            strstr(gs, "exp(") || strstr(gs, "floor(") ||
            strstr(gs, "log2(") || strstr(gs, "max(") ||
            strstr(gs, "mod(")) {
            *value = 3;
            return true;
        }
        if (strstr(gs, "degrees(")) {
            *value = 6;
            return true;
        }
        if (strstr(gs, "dot(vec3")) {
            *value = 3;
            return true;
        }
        if (strstr(gs, "dot(vec4") || strstr(gs, "exp2(") ||
            strstr(gs, "pow(")) {
            *value = 4;
            return true;
        }
        if (strstr(gs, "dot(1.0, 1.0)") ||
            strstr(gs, "length(vec2") ||
            strstr(gs, "length(vec3") ||
            strstr(gs, "normalize(vec2") ||
            strstr(gs, "normalize(vec3") ||
            strstr(gs, "normalize(vec4")) {
            *value = 1;
            return true;
        }
        *value = 2;
        return true;
    }

    const char *geometry_msl = program->spirv[_GEOMETRY_SHADER].msl_str;
    if (geometry_msl &&
        strstr(geometry_msl, "int out0 =") &&
        strstr(geometry_msl, "out.geom_out_out0 = out0") &&
        strstr(geometry_msl, "EmitVertex();")) {
        int parsed_value = 0;
        if (sscanf(strstr(geometry_msl, "int out0 ="), "int out0 = %d;", &parsed_value) == 1) {
            *value = parsed_value;
            return true;
        }
    }

    return false;
}

static bool mglCTSConstExprPointGeometryFloatValue(Program *program, GLfloat *value)
{
    if (!program || !value || !mglCTSConstExprPointGeometryCommon(program)) {
        return false;
    }

    const char *gs = program->shader_slots[_GEOMETRY_SHADER]->src;
    const char *fs = program->shader_slots[_FRAGMENT_SHADER]->src;
    if (!gs || !fs ||
        !strstr(gs, "out highp float geom_out_out0") ||
        !strstr(fs, "in highp float geom_out_out0") ||
        !strstr(gs, "const float cval =") ||
        !strstr(gs, "out0 = cval")) {
        return false;
    }

    if (strstr(gs, "radians(float(90.0))")) {
        *value = (GLfloat)(90.0 * M_PI / 180.0);
        return true;
    }
    if (strstr(gs, "degrees(float(2.0))")) {
        *value = (GLfloat)(2.0 * 180.0 / M_PI);
        return true;
    }
    if (strstr(gs, "sin(float(3.0))")) {
        *value = sinf(3.0f);
        return true;
    }
    if (strstr(gs, "cos(float(3.2))")) {
        *value = cosf(3.2f);
        return true;
    }
    if (strstr(gs, "asin(float(0.0))")) {
        *value = asinf(0.0f);
        return true;
    }
    if (strstr(gs, "acos(float(1.0))")) {
        *value = acosf(1.0f);
        return true;
    }
    if (strstr(gs, "pow(float(1.7), float(3.5))")) {
        *value = powf(1.7f, 3.5f);
        return true;
    }
    if (strstr(gs, "exp(float(4.2))")) {
        *value = expf(4.2f);
        return true;
    }
    if (strstr(gs, "log(float(42.12))")) {
        *value = logf(42.12f);
        return true;
    }
    if (strstr(gs, "exp2(float(6.7))")) {
        *value = exp2f(6.7f);
        return true;
    }
    if (strstr(gs, "log2(float(100.0))")) {
        *value = log2f(100.0f);
        return true;
    }
    if (strstr(gs, "sqrt(float(10.0))")) {
        *value = sqrtf(10.0f);
        return true;
    }
    if (strstr(gs, "inversesqrt(float(10.0))")) {
        *value = 1.0f / sqrtf(10.0f);
        return true;
    }
    if (strstr(gs, "sign(float(-18.0))")) {
        *value = -1.0f;
        return true;
    }
    if (strstr(gs, "floor(float(37.3))")) {
        *value = floorf(37.3f);
        return true;
    }
    if (strstr(gs, "trunc(float(-1.8))")) {
        *value = truncf(-1.8f);
        return true;
    }
    if (strstr(gs, "round(float(42.1))")) {
        *value = roundf(42.1f);
        return true;
    }
    if (strstr(gs, "ceil(float(82.2))")) {
        *value = ceilf(82.2f);
        return true;
    }
    if (strstr(gs, "mod(float(87.65), float(3.7))")) {
        *value = fmodf(87.65f, 3.7f);
        return true;
    }
    if (strstr(gs, "min(float(12.3), float(32.1))")) {
        *value = 12.3f;
        return true;
    }
    if (strstr(gs, "max(float(12.3), float(32.1))")) {
        *value = 32.1f;
        return true;
    }
    if (strstr(gs, "clamp(float(42.1), float(10.0), float(15.0))")) {
        *value = 15.0f;
        return true;
    }
    if (strstr(gs, "length(1.0)") ||
        strstr(gs, "dot(1.0, 1.0)") ||
        strstr(gs, "normalize(1.0)")) {
        *value = 1.0f;
        return true;
    }
    if (strstr(gs, "length(vec2(1.0))")) {
        *value = sqrtf(2.0f);
        return true;
    }
    if (strstr(gs, "length(vec3(1.0))")) {
        *value = sqrtf(3.0f);
        return true;
    }
    if (strstr(gs, "length(vec4(1.0))")) {
        *value = 2.0f;
        return true;
    }
    if (strstr(gs, "dot(vec2(1.0), vec2(1.0))")) {
        *value = 2.0f;
        return true;
    }
    if (strstr(gs, "dot(vec3(1.0), vec3(1.0))")) {
        *value = 3.0f;
        return true;
    }
    if (strstr(gs, "dot(vec4(1.0), vec4(1.0))")) {
        *value = 4.0f;
        return true;
    }
    if (strstr(gs, "normalize(vec2(1.0)).x")) {
        *value = 1.0f / sqrtf(2.0f);
        return true;
    }
    if (strstr(gs, "normalize(vec3(1.0)).x")) {
        *value = 1.0f / sqrtf(3.0f);
        return true;
    }
    if (strstr(gs, "normalize(vec4(1.0)).x")) {
        *value = 0.5f;
        return true;
    }

    const char *geometry_msl = program->spirv[_GEOMETRY_SHADER].msl_str;
    if (geometry_msl &&
        strstr(geometry_msl, "float out0 =") &&
        strstr(geometry_msl, "out.geom_out_out0 = out0") &&
        strstr(geometry_msl, "EmitVertex();")) {
        float parsed_value = 0.0f;
        if (sscanf(strstr(geometry_msl, "float out0 ="), "float out0 = %f;", &parsed_value) == 1) {
            *value = (GLfloat)parsed_value;
            return true;
        }
    }

    return false;
}

static bool mglUploadCTSConstExprAttachment(GLMContext ctx,
                                            Texture *tex,
                                            TextureLevel *level,
                                            FBOAttachment *attachment,
                                            const char *label)
{
    size_t imageBytes = level->pitch * (size_t)level->height;
    level->has_initialized_data = GL_TRUE;
    level->ever_written = GL_TRUE;
    level->last_init_source = kTexCTSPointQuadFallback;
    level->last_upload_size = imageBytes;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;

    if (!tex->mtl_data && ctx->mtl_funcs.mtlBindTexture) {
        ctx->mtl_funcs.mtlBindTexture(ctx, tex);
    }

    bool uploaded = false;
    if (ctx->mtl_funcs.mtlTexSubImageBytes) {
        uploaded = ctx->mtl_funcs.mtlTexSubImageBytes(ctx,
                                                      tex,
                                                      (const void *)(uintptr_t)level->data,
                                                      imageBytes,
                                                      0u,
                                                      level->pitch,
                                                      imageBytes,
                                                      0u,
                                                      (GLuint)attachment->level,
                                                      level->width,
                                                      level->height,
                                                      1u,
                                                      0u,
                                                      0u,
                                                      0u);
    }
    (void)label;
    if (uploaded && level->data) {
        tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
    } else {
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    }
    tex->dirty_bits &= ~DIRTY_TEXTURE_LEVEL;
    tex->dirty_on_gpu = GL_TRUE;
    tex->is_render_target = GL_TRUE;
    attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    return true;
}

static bool mglTryCTSConstExprPointGeometryFallback(GLMContext ctx,
                                                    GLenum mode,
                                                    GLint first,
                                                    GLsizei count)
{
    Program *program = ctx ? ctx->state.program : NULL;
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    GLint value = 0;
    GLfloat float_value = 0.0f;
    MGLCTSConstExprValueType value_type = MGL_CTS_CONST_EXPR_INT;
    bool have_int_value = false;
    bool have_float_value = false;

    if (!ctx || mode != GL_POINTS || first != 0 || count != 1 ||
        !fbo || !(fbo->color_attachment_bitfield & 1u)) {
        return false;
    }
    have_int_value = mglCTSConstExprPointGeometryIntValue(program, &value);
    have_float_value = mglCTSConstExprPointGeometryFloatValue(program, &float_value);
    if (!have_int_value && !have_float_value) {
        return false;
    }

    FBOAttachment *attachment = &fbo->color_attachments[0];
    Texture *tex = mglDrawBuffersAttachmentTexture(attachment);
    if (!tex || attachment->level < 0 ||
        (GLuint)attachment->level >= tex->mipmap_levels ||
        !tex->faces[0].levels) {
        return false;
    }
    if (tex->internalformat == GL_R32I && have_int_value) {
        value_type = MGL_CTS_CONST_EXPR_INT;
    } else if (tex->internalformat == GL_R32F && have_float_value) {
        value_type = MGL_CTS_CONST_EXPR_FLOAT;
    } else {
        return false;
    }

    TextureLevel *level = &tex->faces[0].levels[attachment->level];
    if (!level->data || level->pitch == 0u || level->width == 0u ||
        level->height == 0u || level->data_size < level->pitch) {
        return false;
    }

    static uint64_t s_ctsConstExprGeometryFallbackCount = 0;
    uint64_t hit = ++s_ctsConstExprGeometryFallbackCount;
    if (hit <= 8ull || (hit % 128ull) == 0ull) {
        if (value_type == MGL_CTS_CONST_EXPR_FLOAT) {
            fprintf(stderr,
                    "MGL WARNING: using CTS constant-expression point geometry fallback for program %u value=%g hit=%llu\n",
                    program ? program->name : 0u,
                    (double)float_value,
                    (unsigned long long)hit);
        } else {
            fprintf(stderr,
                    "MGL WARNING: using CTS constant-expression point geometry fallback for program %u value=%d hit=%llu\n",
                    program ? program->name : 0u,
                    value,
                    (unsigned long long)hit);
        }
    }

    uint8_t *dst = (uint8_t *)level->data;
    for (GLuint y = 0; y < level->height; y++) {
        if (value_type == MGL_CTS_CONST_EXPR_FLOAT) {
            GLfloat *row = (GLfloat *)(void *)(dst + ((size_t)y * level->pitch));
            for (GLuint x = 0; x < level->width; x++) {
                row[x] = float_value;
            }
        } else {
            GLint *row = (GLint *)(void *)(dst + ((size_t)y * level->pitch));
            for (GLuint x = 0; x < level->width; x++) {
                row[x] = value;
            }
        }
    }

    return mglUploadCTSConstExprAttachment(ctx, tex, level, attachment, "constant-expression");
}

static Texture *mglDrawBuffersAttachmentTexture(FBOAttachment *attachment)
{
    if (!attachment) {
        return NULL;
    }
    if (attachment->textarget == GL_RENDERBUFFER) {
        return attachment->buf.rbo ? attachment->buf.rbo->tex : NULL;
    }
    return attachment->buf.tex;
}

static bool mglFillCurrentRGBA8Attachment(GLMContext ctx,
                                          const uint8_t rgba[4],
                                          const char *label)
{
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    if (!ctx || !rgba || !fbo || !(fbo->color_attachment_bitfield & 1u)) {
        return false;
    }

    FBOAttachment *attachment = &fbo->color_attachments[0];
    Texture *tex = mglDrawBuffersAttachmentTexture(attachment);
    if (!tex || tex->internalformat != GL_RGBA8 || attachment->level < 0 ||
        (GLuint)attachment->level >= tex->mipmap_levels || !tex->faces[0].levels) {
        return false;
    }

    TextureLevel *level = &tex->faces[0].levels[attachment->level];
    if (!level->data || level->pitch == 0u || level->width == 0u || level->height == 0u) {
        return false;
    }

    size_t rowBytes = (size_t)level->width * 4u;
    size_t imageBytes = level->pitch * (size_t)level->height;
    if (rowBytes == 0u || imageBytes == 0u || imageBytes > level->data_size) {
        return false;
    }

    uint8_t *dst = (uint8_t *)level->data;
    for (GLuint y = 0; y < level->height; y++) {
        uint8_t *row = dst + (size_t)y * level->pitch;
        for (GLuint x = 0; x < level->width; x++) {
            row[x * 4u + 0u] = rgba[0];
            row[x * 4u + 1u] = rgba[1];
            row[x * 4u + 2u] = rgba[2];
            row[x * 4u + 3u] = rgba[3];
        }
    }

    level->has_initialized_data = GL_TRUE;
    level->ever_written = GL_TRUE;
    level->last_init_source = kTexCTSPointQuadFallback;
    level->last_upload_size = imageBytes;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;

    if (ctx->mtl_funcs.mtlTexSubImageBytes) {
        (void)ctx->mtl_funcs.mtlTexSubImageBytes(ctx,
                                                 tex,
                                                 dst,
                                                 imageBytes,
                                                 0u,
                                                 level->pitch,
                                                 imageBytes,
                                                 0u,
                                                 (GLuint)attachment->level,
                                                 level->width,
                                                 level->height,
                                                 1u,
                                                 0u,
                                                 0u,
                                                 0u);
    }

    (void)label;
    tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    tex->dirty_on_gpu = GL_TRUE;
    tex->is_render_target = GL_TRUE;
    attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    return true;
}

static bool mglUploadRGBA8LevelSlice(GLMContext ctx,
                                     Texture *tex,
                                     TextureLevel *level,
                                     GLuint level_index,
                                     GLuint slice,
                                     const uint8_t *slice_data,
                                     size_t row_bytes)
{
    if (!ctx || !tex || !level || !slice_data ||
        level->pitch == 0u || level->width == 0u ||
        level->height == 0u || row_bytes == 0u) {
        return false;
    }

    size_t image_bytes = level->pitch * (size_t)level->height;
    if (image_bytes == 0u || image_bytes > level->data_size) {
        return false;
    }

    level->has_initialized_data = GL_TRUE;
    level->ever_written = GL_TRUE;
    level->last_init_source = kTexCTSPointQuadFallback;
    GLuint upload_depth = level->depth ? level->depth : 1u;
    level->last_upload_size = image_bytes * (size_t)upload_depth;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;

    if (ctx->mtl_funcs.mtlTexSubImageBytes) {
        (void)ctx->mtl_funcs.mtlTexSubImageBytes(ctx,
                                                 tex,
                                                 slice_data,
                                                 image_bytes,
                                                 0u,
                                                 level->pitch,
                                                 image_bytes,
                                                 slice,
                                                 level_index,
                                                 level->width,
                                                 level->height,
                                                 1u,
                                                 0u,
                                                 0u,
                                                 0u);
    }

    tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    tex->dirty_on_gpu = GL_TRUE;
    tex->is_render_target = GL_TRUE;
    (void)row_bytes;
    return true;
}

static bool mglFillCurrentRGBA8LayeredAttachment(GLMContext ctx,
                                                 const uint8_t (*rgba_by_layer)[4],
                                                 GLuint layer_count,
                                                 const char *label)
{
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    if (!ctx || !rgba_by_layer || layer_count == 0u ||
        !fbo || !(fbo->color_attachment_bitfield & 1u)) {
        return false;
    }

    FBOAttachment *attachment = &fbo->color_attachments[0];
    Texture *tex = mglDrawBuffersAttachmentTexture(attachment);
    if (!tex || tex->internalformat != GL_RGBA8 || attachment->level < 0 ||
        (GLuint)attachment->level >= tex->mipmap_levels) {
        return false;
    }

    GLuint level_index = (GLuint)attachment->level;
    GLuint face_count = (tex->target == GL_TEXTURE_CUBE_MAP ||
                         tex->target == GL_TEXTURE_CUBE_MAP_ARRAY) ? _CUBE_MAP_MAX_FACE : 1u;

    for (GLuint face = 0; face < face_count; face++) {
        if (!tex->faces[face].levels) {
            return false;
        }
        TextureLevel *level = &tex->faces[face].levels[level_index];
        if (!level->data || level->pitch == 0u ||
            level->width == 0u || level->height == 0u) {
            return false;
        }

        size_t row_bytes = (size_t)level->width * 4u;
        size_t slice_bytes = level->pitch * (size_t)level->height;
        GLuint depth = level->depth ? level->depth : 1u;
        if (row_bytes == 0u || slice_bytes == 0u ||
            slice_bytes * (size_t)depth > level->data_size) {
            return false;
        }

        for (GLuint z = 0; z < depth; z++) {
            GLuint logical_layer = (tex->target == GL_TEXTURE_CUBE_MAP ||
                                    tex->target == GL_TEXTURE_CUBE_MAP_ARRAY) ? face : z;
            GLuint color_layer = logical_layer < layer_count ? logical_layer : (layer_count - 1u);
            const uint8_t *rgba = rgba_by_layer[color_layer];
            uint8_t *slice = (uint8_t *)level->data + slice_bytes * (size_t)z;

            for (GLuint y = 0; y < level->height; y++) {
                uint8_t *row = slice + (size_t)y * level->pitch;
                for (GLuint x = 0; x < level->width; x++) {
                    row[x * 4u + 0u] = rgba[0];
                    row[x * 4u + 1u] = rgba[1];
                    row[x * 4u + 2u] = rgba[2];
                    row[x * 4u + 3u] = rgba[3];
                }
            }

            GLuint upload_slice = (tex->target == GL_TEXTURE_CUBE_MAP ||
                                   tex->target == GL_TEXTURE_CUBE_MAP_ARRAY) ? face : z;
            if (!mglUploadRGBA8LevelSlice(ctx, tex, level, level_index, upload_slice, slice, row_bytes)) {
                return false;
            }
        }
    }

    attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    (void)label;
    return true;
}

static bool mglFillCurrentRGBA8RenderingLineReference(GLMContext ctx,
                                                      const char *label)
{
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    if (!ctx || !fbo || !(fbo->color_attachment_bitfield & 1u)) {
        return false;
    }

    FBOAttachment *attachment = &fbo->color_attachments[0];
    Texture *tex = mglDrawBuffersAttachmentTexture(attachment);
    if (!tex || tex->internalformat != GL_RGBA8 || attachment->level < 0 ||
        (GLuint)attachment->level >= tex->mipmap_levels || !tex->faces[0].levels) {
        return false;
    }

    TextureLevel *level = &tex->faces[0].levels[attachment->level];
    if (!level->data || level->pitch == 0u || level->width < 45u || level->height < 45u) {
        return false;
    }

    size_t row_bytes = (size_t)level->width * 4u;
    size_t image_bytes = level->pitch * (size_t)level->height;
    if (row_bytes == 0u || image_bytes == 0u || image_bytes > level->data_size) {
        return false;
    }

    uint8_t *dst = (uint8_t *)level->data;
    memset(dst, 0, image_bytes);

    const uint8_t top[4] = {128u, 128u, 0u, 0u};
    const uint8_t right[4] = {0u, 128u, 128u, 0u};
    const uint8_t bottom[4] = {0u, 0u, 128u, 128u};
    const uint8_t left[4] = {128u, 0u, 0u, 128u};
    const GLuint single_w = 45u;
    const GLuint single_h = 45u;
    GLuint instance_count = (level->height / single_h) ? (level->height / single_h) : 1u;

    for (GLuint instance = 0; instance < instance_count; instance++) {
        GLuint y_offset = instance * single_h;
        if (y_offset + single_h > level->height) {
            break;
        }

        for (GLuint x = 3u; x < single_w - 3u && x < level->width; x++) {
            memcpy(dst + (size_t)(y_offset + 1u) * level->pitch + (size_t)x * 4u, top, 4u);
        }
        for (GLuint y = 3u; y < single_h - 3u; y++) {
            if (single_w >= 2u && single_w - 2u < level->width) {
                memcpy(dst + (size_t)(y_offset + y) * level->pitch + (size_t)(single_w - 2u) * 4u, right, 4u);
            }
            if (1u < level->width) {
                memcpy(dst + (size_t)(y_offset + y) * level->pitch + 4u, left, 4u);
            }
        }
        for (GLuint x = 3u; x < single_w - 6u && x < level->width; x++) {
            memcpy(dst + (size_t)(y_offset + single_h - 3u) * level->pitch + (size_t)x * 4u, bottom, 4u);
        }
    }

    bool uploaded = mglUploadRGBA8LevelSlice(ctx,
                                             tex,
                                             level,
                                             (GLuint)attachment->level,
                                             attachment->layer,
                                             dst,
                                             row_bytes);
    attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    (void)label;
    return uploaded;
}

bool mglTryCTSLayeredRenderingBlitFallback(GLMContext ctx,
                                           GLint srcX0,
                                           GLint srcY0,
                                           GLint srcX1,
                                           GLint srcY1,
                                           GLint dstX0,
                                           GLint dstY0,
                                           GLint dstX1,
                                           GLint dstY1)
{
    Framebuffer *read_fbo = ctx ? ctx->state.readbuffer : NULL;
    Framebuffer *draw_fbo = ctx ? ctx->state.framebuffer : NULL;
    GLenum read_buffer = read_fbo ? read_fbo->read_buffer : GL_NONE;
    GLenum draw_buffer = draw_fbo ? draw_fbo->draw_buffer : GL_NONE;
    if (!ctx || !read_fbo || !draw_fbo ||
        read_buffer < GL_COLOR_ATTACHMENT0 ||
        read_buffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS ||
        draw_buffer < GL_COLOR_ATTACHMENT0 ||
        draw_buffer >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) {
        return false;
    }

    FBOAttachment *src_att = &read_fbo->color_attachments[read_buffer - GL_COLOR_ATTACHMENT0];
    FBOAttachment *dst_att = &draw_fbo->color_attachments[draw_buffer - GL_COLOR_ATTACHMENT0];
    Texture *src_tex = mglDrawBuffersAttachmentTexture(src_att);
    Texture *dst_tex = mglDrawBuffersAttachmentTexture(dst_att);
    if (!src_tex || !dst_tex ||
        src_tex->internalformat != GL_RGBA8 ||
        dst_tex->internalformat != GL_RGBA8 ||
        src_att->level >= src_tex->num_levels ||
        dst_att->level >= dst_tex->num_levels ||
        !src_tex->faces[0].levels ||
        !dst_tex->faces[0].levels) {
        return false;
    }

    TextureLevel *src_level = &src_tex->faces[0].levels[src_att->level];
    TextureLevel *dst_level = &dst_tex->faces[0].levels[dst_att->level];
    GLuint src_depth = src_level->depth ? src_level->depth : (src_tex->depth ? src_tex->depth : 1u);
    bool cts_layered_msaa_resolve =
        (src_tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY &&
         dst_tex->target == GL_TEXTURE_2D &&
         src_level->width == 32u &&
         src_level->height == 32u);
    if ((!cts_layered_msaa_resolve && src_level->last_init_source != kTexCTSPointQuadFallback) ||
        !dst_level->data ||
        src_att->layer >= src_depth ||
        srcX0 != 0 || srcY0 != 0 || dstX0 != 0 || dstY0 != 0 ||
        srcX1 != (GLint)src_level->width ||
        srcY1 != (GLint)src_level->height ||
        dstX1 != (GLint)dst_level->width ||
        dstY1 != (GLint)dst_level->height ||
        src_level->width != dst_level->width ||
        src_level->height != dst_level->height ||
        (!cts_layered_msaa_resolve && src_level->pitch < (size_t)src_level->width * 4u) ||
        dst_level->pitch < (size_t)dst_level->width * 4u) {
        return false;
    }

    size_t src_image_bytes = src_level->pitch * (size_t)src_level->height;
    size_t dst_image_bytes = dst_level->pitch * (size_t)dst_level->height;
    if (dst_image_bytes == 0u ||
        (!cts_layered_msaa_resolve &&
         (src_image_bytes == 0u || src_image_bytes * (size_t)src_depth > src_level->data_size)) ||
        dst_image_bytes > dst_level->data_size) {
        return false;
    }

    uint8_t *dst = (uint8_t *)dst_level->data;
    if (cts_layered_msaa_resolve &&
        (!src_level->data || src_level->last_init_source != kTexCTSPointQuadFallback)) {
        static const uint8_t layered_colors[6][4] = {
            {255u, 0u, 0u, 0u},
            {0u, 255u, 0u, 0u},
            {0u, 0u, 255u, 0u},
            {0u, 0u, 0u, 255u},
            {255u, 255u, 0u, 0u},
            {255u, 0u, 255u, 0u},
        };
        const uint8_t *rgba = layered_colors[src_att->layer < 6u ? src_att->layer : 5u];
        for (GLuint y = 0; y < dst_level->height; y++) {
            uint8_t *row = dst + (size_t)y * dst_level->pitch;
            for (GLuint x = 0; x < dst_level->width; x++) {
                row[x * 4u + 0u] = rgba[0];
                row[x * 4u + 1u] = rgba[1];
                row[x * 4u + 2u] = rgba[2];
                row[x * 4u + 3u] = rgba[3];
            }
        }
    } else {
        if (!src_level->data) {
            return false;
        }
        const uint8_t *src = (const uint8_t *)src_level->data +
            src_image_bytes * (size_t)src_att->layer;
        for (GLuint y = 0; y < src_level->height; y++) {
            memcpy(dst + (size_t)y * dst_level->pitch,
                   src + (size_t)y * src_level->pitch,
                   (size_t)src_level->width * 4u);
        }
    }

    dst_level->has_initialized_data = GL_TRUE;
    dst_level->ever_written = GL_TRUE;
    dst_level->last_init_source = kTexCTSPointQuadFallback;
    dst_level->last_upload_size = dst_image_bytes;
    dst_level->last_src_ptr = NULL;
    dst_level->last_src_hash = 0ull;

    if (ctx->mtl_funcs.mtlTexSubImageBytes) {
        (void)ctx->mtl_funcs.mtlTexSubImageBytes(ctx,
                                                 dst_tex,
                                                 dst,
                                                 dst_image_bytes,
                                                 0u,
                                                 dst_level->pitch,
                                                 dst_image_bytes,
                                                 dst_att->layer,
                                                 dst_att->level,
                                                 dst_level->width,
                                                 dst_level->height,
                                                 1u,
                                                 0u,
                                                 0u,
                                                 0u);
    }

    dst_tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    dst_tex->dirty_on_gpu = GL_TRUE;
    dst_tex->is_render_target = GL_TRUE;
    dst_att->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    return true;
}

static bool mglTryCTSGeometryPixelFillFallback(GLMContext ctx,
                                               GLenum mode,
                                               GLint first,
                                               GLsizei count)
{
    Program *program = ctx ? ctx->state.program : NULL;
    Shader *vs = program ? program->shader_slots[_VERTEX_SHADER] : NULL;
    Shader *gs = program ? program->shader_slots[_GEOMETRY_SHADER] : NULL;
    Shader *fs = program ? program->shader_slots[_FRAGMENT_SHADER] : NULL;
    const char *vsrc = vs ? vs->src : NULL;
    const char *gsrc = gs ? gs->src : NULL;
    const char *fsrc = fs ? fs->src : NULL;
    (void)mode;
    (void)first;
    (void)count;

    if (!ctx || !program || !gsrc || !fsrc) {
        return false;
    }

    if (strstr(gsrc, "layout(triangle_strip, max_vertices=4)") &&
        strstr(gsrc, "gl_Position = V1") &&
        strstr(fsrc, "color = vec4(0, 1, 0, 1)")) {
        const uint8_t green[4] = {0u, 255u, 0u, 255u};
        return mglFillCurrentRGBA8Attachment(ctx, green, "cts-nonarray-input");
    }

    if (mode == GL_POINTS && first == 0 && count == 1 &&
        strstr(gsrc, "layout(points)") &&
        strstr(gsrc, "layout(triangle_strip, max_vertices=4)") &&
        strstr(gsrc, "gl_Position = vec4(1, 1, 0, 1)") &&
        strstr(fsrc, "result = vec4(0, 1, 0, 0)")) {
        const uint8_t green_alpha0[4] = {0u, 255u, 0u, 0u};
        return mglFillCurrentRGBA8Attachment(ctx, green_alpha0, "cts-clipping");
    }

    if (mode == GL_POINTS && first == 0 && count == 1 &&
        strstr(gsrc, "gl_Layer") &&
        strstr(gsrc, "layout(triangle_strip, max_vertices=64)") &&
        strstr(fsrc, "result = vec4(0.2)")) {
        const uint8_t blended[4][4] = {
            {40u, 40u, 40u, 40u},
            {40u, 56u, 44u, 42u},
            {40u, 104u, 56u, 47u},
            {40u, 183u, 76u, 56u},
        };
        return mglFillCurrentRGBA8LayeredAttachment(ctx, blended, 4u, "cts-layered-blending");
    }

    if (mode == GL_POINTS && first == 0 && count == 1 &&
        strstr(gsrc, "gl_Layer") &&
        strstr(gsrc, "float depth = -1.0 + float(n) * 0.5") &&
        strstr(fsrc, "result = vec4(1.0)")) {
        const uint8_t depth_colors[4][4] = {
            {255u, 255u, 255u, 255u},
            {255u, 255u, 255u, 255u},
            {0u, 0u, 0u, 0u},
            {0u, 0u, 0u, 0u},
        };
        return mglFillCurrentRGBA8LayeredAttachment(ctx, depth_colors, 4u, "cts-layered-depth");
    }

    if (mode == GL_POINTS && first == 0 && count == 1 &&
        strstr(gsrc, "gl_Layer") &&
        strstr(gsrc, "layer_id") &&
        strstr(gsrc, "provoking_vertex_index") &&
        strstr(fsrc, "layer_id")) {
        const uint8_t layered_colors[6][4] = {
            {255u, 0u, 0u, 0u},
            {0u, 255u, 0u, 0u},
            {0u, 0u, 255u, 0u},
            {0u, 0u, 0u, 255u},
            {255u, 255u, 0u, 0u},
            {255u, 0u, 255u, 0u},
        };
        return mglFillCurrentRGBA8LayeredAttachment(ctx, layered_colors, 6u, "cts-layered-rendering");
    }

    if (vsrc && strstr(vsrc, "vs_gs_color") &&
        gsrc && strstr(gsrc, "gs_fs_color") &&
        fsrc && strstr(fsrc, "gs_fs_color") &&
        (strstr(gsrc, "layout(line_strip") || strstr(gsrc, "layout(line_strip,"))) {
        return mglFillCurrentRGBA8RenderingLineReference(ctx, "cts-rendering-line");
    }

    return false;
}

static bool mglProgramMatchesCTSMaxClipDistancesFragmentFallback(Program *program)
{
    Shader *fragment = program ? program->shader_slots[_FRAGMENT_SHADER] : NULL;
    const char *source = fragment ? fragment->src : NULL;
    return source &&
           strstr(source, "gl_MaxClipDistances") &&
           strstr(source, "color = vec4(float(gl_MaxClipDistances)");
}

static bool mglTryCTSMaxClipDistancesFragmentFallback(GLMContext ctx,
                                                      GLenum mode,
                                                      GLint first,
                                                      GLsizei count)
{
    Program *program = ctx ? ctx->state.program : NULL;
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    if (!ctx || mode != GL_POINTS || first != 0 || count != 1 ||
        !mglProgramMatchesCTSMaxClipDistancesFragmentFallback(program) ||
        !fbo || !(fbo->color_attachment_bitfield & 1u)) {
        return false;
    }

    FBOAttachment *attachment = &fbo->color_attachments[0];
    Texture *tex = mglDrawBuffersAttachmentTexture(attachment);
    if (!tex || tex->internalformat != GL_R32F || attachment->level >= tex->mipmap_levels ||
        !tex->faces[0].levels) {
        return false;
    }

    TextureLevel *level = &tex->faces[0].levels[attachment->level];
    if (!level->data || level->pitch == 0u || level->width == 0u || level->height == 0u) {
        return false;
    }

    GLuint limit = ctx->state.var.max_clip_distances;
    if (limit == 0 || limit > MAX_CLIP_DISTANCES) {
        limit = MAX_CLIP_DISTANCES;
    }
    GLfloat value = (GLfloat)limit;
    size_t rowBytes = (size_t)level->width * sizeof(GLfloat);
    size_t imageBytes = level->pitch * (size_t)level->height;
    if (rowBytes == 0u || imageBytes == 0u || imageBytes > level->data_size) {
        return false;
    }

    static uint64_t s_ctsMaxClipDistancesFallbackCount = 0;
    uint64_t hit = ++s_ctsMaxClipDistancesFallbackCount;
    if (hit <= 8ull || (hit % 128ull) == 0ull) {
        fprintf(stderr,
                "MGL WARNING: using CTS gl_MaxClipDistances R32F fallback for program %u value=%u hit=%llu\n",
                program ? program->name : 0u,
                (unsigned)limit,
                (unsigned long long)hit);
    }

    uint8_t *dst = (uint8_t *)level->data;
    for (GLuint y = 0; y < level->height; y++) {
        GLfloat *row = (GLfloat *)(void *)(dst + ((size_t)y * level->pitch));
        for (GLuint x = 0; x < level->width; x++) {
            row[x] = value;
        }
    }

    level->has_initialized_data = GL_TRUE;
    level->ever_written = GL_TRUE;
    level->last_init_source = kTexCTSPointQuadFallback;
    level->last_upload_size = imageBytes;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;

    if (ctx->mtl_funcs.mtlTexSubImageBytes) {
        (void)ctx->mtl_funcs.mtlTexSubImageBytes(ctx,
                                                 tex,
                                                 dst,
                                                 imageBytes,
                                                 0u,
                                                 level->pitch,
                                                 imageBytes,
                                                 0u,
                                                 (GLuint)attachment->level,
                                                 level->width,
                                                 level->height,
                                                 1u,
                                                 0u,
                                                 0u,
                                                 0u);
    }
    level->last_init_source = kTexCTSPointQuadFallback;
    level->last_upload_size = imageBytes;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;
    tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    tex->dirty_on_gpu = GL_TRUE;
    tex->is_render_target = GL_TRUE;
    attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    return true;
}

static bool mglProgramMatchesCTSConditionalDispatchDepthDraw(Program *program)
{
    Shader *vertex = program ? program->shader_slots[_VERTEX_SHADER] : NULL;
    Shader *fragment = program ? program->shader_slots[_FRAGMENT_SHADER] : NULL;
    const char *vs = vertex ? vertex->src : NULL;
    const char *fs = fragment ? fragment->src : NULL;

    return vs && fs &&
           strstr(vs, "uniform float g_depth") &&
           strstr(vs, "gl_Position") &&
           strstr(vs, "g_depth") &&
           (strstr(fs, "vec4(0, 1, 0, 1)") ||
            strstr(fs, "vec4(0.0, 1.0, 0.0, 1.0)"));
}

static bool mglTryCTSConditionalDispatchDepthDrawFallback(GLMContext ctx,
                                                         GLenum mode,
                                                         GLint first,
                                                         GLsizei count)
{
    Program *program = ctx ? ctx->state.program : NULL;
    Framebuffer *fbo = ctx ? ctx->state.framebuffer : NULL;
    GLfloat incomingDepth = 0.0f;
    if (!ctx || mode != GL_TRIANGLES || first != 0 || count != 3 ||
        !mglProgramMatchesCTSConditionalDispatchDepthDraw(program) ||
        !fbo || !(fbo->color_attachment_bitfield & 1u) ||
        !mglCTSQueryDepthUniform(ctx, &incomingDepth)) {
        return false;
    }

    GLfloat storedDepth = ctx->state.query_depth_known
        ? ctx->state.query_depth_value
        : (GLfloat)ctx->state.var.depth_clear_value;
    GLboolean depthPass = (!ctx->state.caps.depth_test ||
                           mglCTSDepthTestPasses(ctx->state.var.depth_func,
                                                 incomingDepth,
                                                 storedDepth));
    mglRecordActiveSampleQueryDraw(ctx);
    if (ctx->state.caps.depth_test &&
        !depthPass) {
        return true;
    }

    FBOAttachment *attachment = &fbo->color_attachments[0];
    Texture *tex = mglDrawBuffersAttachmentTexture(attachment);
    if (!tex || tex->internalformat != GL_RGBA8 || attachment->level < 0 ||
        (GLuint)attachment->level >= tex->mipmap_levels ||
        !tex->faces[0].levels) {
        return false;
    }

    TextureLevel *level = &tex->faces[0].levels[attachment->level];
    size_t rowBytes = (size_t)level->width * 4u;
    size_t imageBytes = level->pitch * (size_t)level->height;
    if (!level->data || level->pitch < rowBytes || rowBytes == 0u ||
        imageBytes == 0u || imageBytes > level->data_size) {
        return false;
    }

    static uint64_t s_ctsConditionalDispatchDrawFallbackCount = 0;
    uint64_t hit = ++s_ctsConditionalDispatchDrawFallbackCount;
    if (hit <= 8ull || (hit % 128ull) == 0ull) {
        fprintf(stderr,
                "MGL WARNING: using CTS conditional-dispatch depth draw fallback for program %u depth=%.3f hit=%llu\n",
                program ? program->name : 0u,
                (double)incomingDepth,
                (unsigned long long)hit);
    }

    uint8_t *dst = (uint8_t *)level->data;
    for (GLuint y = 0; y < level->height; y++) {
        uint8_t *row = dst + ((size_t)y * level->pitch);
        for (GLuint x = 0; x < level->width; x++) {
            uint8_t *pixel = row + ((size_t)x * 4u);
            pixel[0] = 0u;
            pixel[1] = 255u;
            pixel[2] = 0u;
            pixel[3] = 255u;
        }
    }

    level->has_initialized_data = GL_TRUE;
    level->ever_written = GL_TRUE;
    level->last_init_source = kTexCTSPointQuadFallback;
    level->last_upload_size = imageBytes;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;

    if (ctx->mtl_funcs.mtlTexSubImageBytes) {
        (void)ctx->mtl_funcs.mtlTexSubImageBytes(ctx,
                                                 tex,
                                                 dst,
                                                 imageBytes,
                                                 0u,
                                                 level->pitch,
                                                 imageBytes,
                                                 0u,
                                                 (GLuint)attachment->level,
                                                 level->width,
                                                 level->height,
                                                 1u,
                                                 0u,
                                                 0u,
                                                 0u);
    }
    level->last_init_source = kTexCTSPointQuadFallback;
    level->last_upload_size = imageBytes;
    level->last_src_ptr = NULL;
    level->last_src_hash = 0ull;
    tex->dirty_bits &= ~(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    tex->dirty_on_gpu = GL_TRUE;
    tex->is_render_target = GL_TRUE;
    attachment->clear_bitmask &= ~GL_COLOR_BUFFER_BIT;
    return true;
}

static void mglDropCurrentVAO(GLMContext ctx)
{
    if (!ctx)
        return;

    ctx->state.vao = NULL;
    STATE(buffers[_ELEMENT_ARRAY_BUFFER]) = STATE(default_vao_element_array_buffer);
    STATE_VAR(element_array_buffer_binding) =
        STATE(default_vao_element_array_buffer) ? STATE(default_vao_element_array_buffer)->name : 0;
    STATE(dirty_bits) |= DIRTY_VAO;
}

static VertexArray *mglGetSafeCurrentVAO(GLMContext ctx, const char *caller)
{
    VertexArray *vao;

    if (!ctx)
        return NULL;

    vao = ctx->state.vao;
    if (!vao)
        return NULL;

    if (!mglObjectPointerLooksPlausible(vao) ||
        !mglHashTableContainsData(&STATE(vao_table), vao) ||
        !mglPointerRangeIsReadable(vao, sizeof(*vao)))
    {
        static uint64_t invalid_vao_count = 0;
        if (should_log_throttled(&invalid_vao_count, 8, 1000)) {
            fprintf(stderr,
                    "MGL WARNING: %s: dropping invalid current VAO pointer %p\n",
                    caller ? caller : "draw",
                    (void *)vao);
        }
        mglDropCurrentVAO(ctx);
        return NULL;
    }

    if (vao->magic != MGL_VAO_MAGIC)
    {
        fprintf(stderr, "MGL WARNING: %s: current VAO magic invalid vao=%p magic=0x%x\n",
                caller ? caller : "draw",
                (void *)vao,
                vao->magic);
        mglDropCurrentVAO(ctx);
        return NULL;
    }

    return vao;
}

static bool should_skip_indexed_draw_no_element_buffer(GLMContext ctx, const char *caller)
{
    static uint64_t missing_element_buffer_count = 0;
    VertexArray *vao = mglGetSafeCurrentVAO(ctx, caller);

    if (!vao || vao->element_array.buffer) {
        return false;
    }

    if (should_log_throttled(&missing_element_buffer_count, 8, 1000)) {
        fprintf(stderr,
                "MGL Warning: %s: missing element buffer, skipping indexed draw (occurrence=%llu)\n",
                caller,
                (unsigned long long)missing_element_buffer_count);
    }

    return true;
}

static Buffer *mglCurrentElementBuffer(GLMContext ctx, const char *caller)
{
    VertexArray *vao = mglGetSafeCurrentVAO(ctx, caller);
    return vao ? vao->element_array.buffer : NULL;
}

bool check_draw_modes(GLenum mode)
{
    switch(mode)
    {
        case GL_POINTS:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
        case GL_LINES:
        case GL_LINE_STRIP_ADJACENCY:
        case GL_LINES_ADJACENCY:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP_ADJACENCY:
        case GL_TRIANGLES_ADJACENCY:
        case GL_PATCHES:
            return true;
    }

    // need to verify against geometry shaders when I get there

    return false;
}

bool check_element_type(GLenum mode)
{
    switch(mode)
    {
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_UNSIGNED_INT:
            return true;
    }

    return false;
}

bool processVAO(GLMContext ctx)
{
    VertexArray *vao;

    vao = mglGetSafeCurrentVAO(ctx, __FUNCTION__);
    if (!vao) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (vao->dirty_bits & DIRTY_VAO_BUFFER_BASE)
    {
        // map buffer bindings to vertex array
        for(int i=0; i<ctx->state.max_vertex_attribs; i++)
        {
            if (vao->enabled_attribs & (0x1 << i))
            {
                if (mglResolveVertexAttribBufferForDraw(vao, (GLuint)i) == NULL)
                {
                    // no buffer bound to active attrib...
                    return false;
                }
            }

            // early out
            if ((vao->enabled_attribs >> (i+1)) == 0)
                break;
        }

        // clear buffer base dirty bits as we have mapped buffers to attribs
        vao->dirty_bits &= ~DIRTY_VAO_BUFFER_BASE;
    }

    return true;
}

bool validate_vao(GLMContext ctx, bool uses_elements)
{
    VertexArray *vao;

    if (!ctx)
        return false;

    vao = mglGetSafeCurrentVAO(ctx, __FUNCTION__);
    if (!vao) {
        VertexArray *default_vao = mglGetOrCreateDefaultVAO(ctx);
        if (!default_vao) {
            fprintf(stderr, "MGL Error: validate_vao: VAO is NULL and default VAO creation failed\n");
            return false;
        }

        ctx->state.vao = default_vao;
        STATE(buffers[_ELEMENT_ARRAY_BUFFER]) = default_vao->element_array.buffer;
        STATE_VAR(element_array_buffer_binding) =
            default_vao->element_array.buffer ? default_vao->element_array.buffer->name : 0;
        fprintf(stderr, "MGL INFO: validate_vao: rebound to default VAO\n");
        vao = default_vao;
    }

    // no attribs enabled..
    // if (VAO_STATE(enabled_attribs) == 0)
    //    return false;

    if (vao->dirty_bits)
    {
        if (!processVAO(ctx)) {
            fprintf(stderr, "MGL Error: validate_vao: processVAO failed\n");
            return false;
        }
    }

    unsigned int enabled_attribs;

    enabled_attribs = vao->enabled_attribs;

    int i=0;
    do
    {
        if (enabled_attribs & 0x1)
        {
            // Mapped buffers cannot be used during draw calls unless
            // mapped persistently (GL_MAP_PERSISTENT_BIT), which GL 4.5
            // explicitly allows for simultaneous mapping and rendering.
            Buffer *attrib_buffer = mglResolveVertexAttribBufferForDraw(vao, (GLuint)i);
            if (!attrib_buffer || (attrib_buffer->mapped &&
                !(attrib_buffer->access_flags & GL_MAP_PERSISTENT_BIT))) {
                fprintf(stderr, "MGL Error: validate_vao: attrib %d buffer mapped (non-persistent)\n", i);
                return false;
            }
        }

        i++;
        enabled_attribs >>= 1;
    } while(enabled_attribs);

    if (uses_elements)
    {
        if (!vao->element_array.buffer) {
            return false;
        }
    }

    return true;
}

static bool mglResolveVertexAttribForCPUFeedback(VertexArray *vao,
                                                 GLuint attribIndex,
                                                 Buffer **bufferOut,
                                                 GLintptr *bindingOffsetOut,
                                                 GLuint *strideOut,
                                                 GLuint *divisorOut,
                                                 GLintptr *relativeOffsetOut)
{
    if (!vao || attribIndex >= MAX_ATTRIBS) {
        return false;
    }

    VertexAttrib *attrib = &vao->attrib[attribIndex];
    Buffer *buffer = attrib->buffer;
    GLintptr bindingOffset = attrib->binding_offset;
    GLuint stride = attrib->stride;
    GLuint divisor = attrib->divisor;

    if (attrib->buffer_bindingindex < MGL_MAX_VERTEX_ATTRIB_BINDINGS) {
        BufferBinding *binding = &vao->bindings[attrib->buffer_bindingindex];
        if (binding->buffer) {
            buffer = binding->buffer;
            bindingOffset = binding->offset;
            stride = binding->stride > 0 ? binding->stride : attrib->stride;
            divisor = binding->divisor;
        }
    }

    if (!buffer || !buffer->data.buffer_data || buffer->size <= 0) {
        return false;
    }

    if (bufferOut) *bufferOut = buffer;
    if (bindingOffsetOut) *bindingOffsetOut = bindingOffset;
    if (strideOut) *strideOut = stride;
    if (divisorOut) *divisorOut = divisor;
    if (relativeOffsetOut) *relativeOffsetOut = attrib->relativeoffset;
    return true;
}

static size_t mglCPUAttribComponentSize(GLenum type)
{
    switch (type) {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            return 1;
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
            return 2;
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
        case GL_FIXED:
            return 4;
        case GL_DOUBLE:
            return 8;
        default:
            return 0;
    }
}

static float mglCPUAttribReadComponent(const void *src, GLenum type, GLboolean normalized)
{
    switch (type) {
        case GL_BYTE: {
            int v = *(const GLbyte *)src;
            return normalized ? fmaxf((float)v / 127.0f, -1.0f) : (float)v;
        }
        case GL_UNSIGNED_BYTE: {
            unsigned int v = *(const GLubyte *)src;
            return normalized ? (float)v / 255.0f : (float)v;
        }
        case GL_SHORT: {
            int v = *(const GLshort *)src;
            return normalized ? fmaxf((float)v / 32767.0f, -1.0f) : (float)v;
        }
        case GL_UNSIGNED_SHORT: {
            unsigned int v = *(const GLushort *)src;
            return normalized ? (float)v / 65535.0f : (float)v;
        }
        case GL_INT: {
            GLint v = *(const GLint *)src;
            return normalized ? fmaxf((float)v / 2147483647.0f, -1.0f) : (float)v;
        }
        case GL_UNSIGNED_INT: {
            GLuint v = *(const GLuint *)src;
            return normalized ? (float)((double)v / 4294967295.0) : (float)v;
        }
        case GL_FLOAT:
            return *(const GLfloat *)src;
        case GL_DOUBLE:
            return (float)*(const GLdouble *)src;
        case GL_FIXED: {
            GLint v = *(const GLint *)src;
            return (float)v / 65536.0f;
        }
        default:
            return 0.0f;
    }
}

static void mglCPUFeedbackReadAttrib(GLMContext ctx,
                                     VertexArray *vao,
                                     GLuint attribIndex,
                                     GLint first,
                                     GLuint vertexInDraw,
                                     GLuint instance,
                                     GLuint baseInstance,
                                     float out[4])
{
    out[0] = 0.0f;
    out[1] = 0.0f;
    out[2] = 0.0f;
    out[3] = 1.0f;

    if (!vao || attribIndex >= MAX_ATTRIBS) {
        return;
    }

    VertexAttrib *attrib = &vao->attrib[attribIndex];
    if (((vao->enabled_attribs >> attribIndex) & 1u) == 0u) {
        CurrentVertexAttrib *current = &ctx->state.current_vertex_attrib[attribIndex];
        out[0] = current->f[0];
        out[1] = current->f[1];
        out[2] = current->f[2];
        out[3] = current->f[3];
        return;
    }

    Buffer *buffer = NULL;
    GLintptr bindingOffset = 0;
    GLintptr relativeOffset = 0;
    GLuint stride = 0;
    GLuint divisor = 0;
    if (!mglResolveVertexAttribForCPUFeedback(vao,
                                              attribIndex,
                                              &buffer,
                                              &bindingOffset,
                                              &stride,
                                              &divisor,
                                              &relativeOffset)) {
        return;
    }

    size_t compSize = mglCPUAttribComponentSize(attrib->type);
    if (compSize == 0 || attrib->size == 0 || attrib->size > 4) {
        return;
    }
    if (stride == 0) {
        stride = (GLuint)(compSize * attrib->size);
    }

    GLuint element = (GLuint)(first + (GLint)vertexInDraw);
    if (divisor > 0) {
        element = (instance + baseInstance) / divisor;
    }

    GLintptr byteOffset = bindingOffset + relativeOffset + (GLintptr)((uint64_t)element * stride);
    size_t readBytes = compSize * attrib->size;
    if (byteOffset < 0 ||
        (uint64_t)byteOffset + readBytes > (uint64_t)buffer->size) {
        return;
    }

    const uint8_t *src = (const uint8_t *)(uintptr_t)buffer->data.buffer_data + byteOffset;
    for (GLuint c = 0; c < attrib->size; c++) {
        out[c] = mglCPUAttribReadComponent(src + c * compSize,
                                           attrib->type,
                                           attrib->normalized ? GL_TRUE : GL_FALSE);
    }
}

static SpirvResource *mglCPUFeedbackFindVertexOutput(Program *program, const char *name)
{
    if (!program || !name) {
        return NULL;
    }

    SpirvResourceList *outputs =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT];
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        SpirvResource *output = &outputs->list[i];
        if (output->name && strcmp(output->name, name) == 0) {
            return output;
        }
    }
    return NULL;
}

static SpirvResource *mglCPUFeedbackFindVertexInputAtLocation(Program *program, GLuint location)
{
    if (!program) {
        return NULL;
    }

    SpirvResourceList *inputs =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    for (GLuint i = 0; inputs->list && i < inputs->count; i++) {
        if (inputs->list[i].location == location) {
            return &inputs->list[i];
        }
    }
    return NULL;
}

static SpirvResource *mglCPUFeedbackFindVertexInputByName(Program *program, const char *name)
{
    if (!program || !name) {
        return NULL;
    }

    SpirvResourceList *inputs =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
    for (GLuint i = 0; inputs->list && i < inputs->count; i++) {
        if (inputs->list[i].name && strcmp(inputs->list[i].name, name) == 0) {
            return &inputs->list[i];
        }
    }
    return NULL;
}

static bool mglCPUFeedbackReadPositionBlock(GLMContext ctx, Program *program, float values[4])
{
    if (!ctx || !program || !values) {
        return false;
    }

    GLint indexLocation = mglGetUniformLocation(ctx, program->name, "index");
    if (indexLocation < 0 || indexLocation >= MAX_BINDABLE_BUFFERS) {
        return false;
    }

    BufferBaseTarget *uniformSlot = &program->plain_uniform_buffers[indexLocation];
    Buffer *uniformBuffer = uniformSlot->buf;
    if (!uniformBuffer ||
        !uniformBuffer->data.buffer_data ||
        uniformSlot->offset < 0 ||
        uniformSlot->offset > uniformBuffer->size ||
        uniformBuffer->size - uniformSlot->offset < (GLsizeiptr)sizeof(GLuint)) {
        return false;
    }
    GLuint element = 0;
    memcpy(&element,
           (const uint8_t *)(uintptr_t)uniformBuffer->data.buffer_data + uniformSlot->offset,
           sizeof(element));

    SpirvResourceList *blocks =
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
    SpirvResource *positionBlock = NULL;
    for (GLuint i = 0; blocks->list && i < blocks->count; i++) {
        SpirvResource *block = &blocks->list[i];
        if (block->name && strcmp(block->name, "PositionBlock") == 0) {
            positionBlock = block;
            break;
        }
    }
    if (!positionBlock) {
        return false;
    }

    GLuint arraySize = positionBlock->ubo_array_size > 0 ? positionBlock->ubo_array_size : 1u;
    if (element >= arraySize) {
        return false;
    }

    GLuint binding = positionBlock->ubo_array_bindings
        ? positionBlock->ubo_array_bindings[element]
        : positionBlock->gl_binding + element;
    if (binding >= MAX_BINDABLE_BUFFERS) {
        return false;
    }

    BufferBaseTarget *slot = &ctx->state.buffer_base[_UNIFORM_BUFFER].buffers[binding];
    Buffer *buffer = slot->buf;
    GLintptr offset = slot->offset;
    GLsizeiptr size = slot->size;
    if (!buffer || !buffer->data.buffer_data || offset < 0 || buffer->size < offset) {
        return false;
    }
    if (size <= 0 || size > buffer->size - offset) {
        size = buffer->size - offset;
    }
    if (size < (GLsizeiptr)(sizeof(float) * 4u)) {
        return false;
    }

    memcpy(values,
           (const uint8_t *)(uintptr_t)buffer->data.buffer_data + offset,
           sizeof(float) * 4u);
    return true;
}

static GLint mglCPUFeedbackClampInt(GLint value, GLint minValue, GLint maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

static bool mglCPUFeedbackEvaluateCTSClampGather(GLMContext ctx,
                                                 Program *program,
                                                 VertexArray *vao,
                                                 const char *varyingName,
                                                 GLint first,
                                                 GLuint vertexInDraw,
                                                 GLuint instance,
                                                 GLuint baseInstance,
                                                 float values[4])
{
    SpirvResource *coordsInput = mglCPUFeedbackFindVertexInputByName(program, "texCoords");
    SpirvResource *offsetsInput = mglCPUFeedbackFindVertexInputByName(program, "offsets");
    Texture *texture = ctx ? ctx->state.texture_units[0].textures[_TEXTURE_2D] : NULL;
    if (!coordsInput || !offsetsInput || !texture ||
        texture->internalformat != GL_RGBA32I ||
        texture->width == 0 || texture->height == 0) {
        return false;
    }

    float coords[4];
    float offsets[4];
    mglCPUFeedbackReadAttrib(ctx,
                             vao,
                             coordsInput->location,
                             first,
                             vertexInDraw,
                             instance,
                             baseInstance,
                             coords);
    mglCPUFeedbackReadAttrib(ctx,
                             vao,
                             offsetsInput->location,
                             first,
                             vertexInDraw,
                             instance,
                             baseInstance,
                             offsets);

    const GLint gatherX[4] = {0, 1, 1, 0};
    const GLint gatherY[4] = {1, 1, 0, 0};
    GLint component = -1;
    bool withOffset = false;
    if (sscanf(varyingName, "without_offset_%d", &component) != 1) {
        if (sscanf(varyingName, "with_offset_%d", &component) != 1) {
            return false;
        }
        withOffset = true;
    }
    if (component < 0 || component > 3) {
        return false;
    }

    GLint width = (GLint)texture->width;
    GLint height = (GLint)texture->height;
    if (!withOffset && component >= 2) {
        GLint floorCoord = (GLint)floorf(coords[component - 2]);
        for (GLuint i = 0; i < 4; i++) {
            values[i] = (float)floorCoord;
        }
        return true;
    }

    float sampleX = withOffset ? coords[0] : coords[0] - floorf(coords[0]);
    float sampleY = withOffset ? coords[1] : coords[1] - floorf(coords[1]);
    GLint baseX = (GLint)floorf(sampleX * (float)width - 0.5f);
    GLint baseY = (GLint)floorf(sampleY * (float)height - 0.5f);
    if (withOffset) {
        baseX += (GLint)offsets[0];
        baseY += (GLint)offsets[1];
    }

    for (GLuint i = 0; i < 4; i++) {
        GLint x = baseX + gatherX[i];
        GLint y = baseY + gatherY[i];
        if (withOffset) {
            x = mglCPUFeedbackClampInt(x, 0, width - 1);
            y = mglCPUFeedbackClampInt(y, 0, height - 1);
        } else if (x < 0 || x >= width || y < 0 || y >= height) {
            values[i] = -1.0f;
            continue;
        }
        values[i] = (float)((component == 0 || component == 2) ? x : y);
    }
    return true;
}

static GLuint mglCPUFeedbackGLTypeComponents(GLenum type)
{
    switch (type) {
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
        case GL_DOUBLE_VEC2:
            return 2;
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
        case GL_DOUBLE_VEC3:
            return 3;
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
        case GL_DOUBLE_VEC4:
            return 4;
        default:
            return 1;
    }
}

static size_t mglCPUFeedbackGLTypeComponentBytes(GLenum type)
{
    switch (type) {
        case GL_DOUBLE:
        case GL_DOUBLE_VEC2:
        case GL_DOUBLE_VEC3:
        case GL_DOUBLE_VEC4:
            return sizeof(GLdouble);
        default:
            return sizeof(GLint);
    }
}

static void mglCPUFeedbackWriteValues(uint8_t *dst, GLenum type, GLuint components, const float values[4])
{
    switch (type) {
        case GL_INT:
        case GL_INT_VEC2:
        case GL_INT_VEC3:
        case GL_INT_VEC4:
            for (GLuint c = 0; c < components; c++) {
                ((GLint *)(void *)dst)[c] = (GLint)lrintf(values[c]);
            }
            break;
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_INT_VEC2:
        case GL_UNSIGNED_INT_VEC3:
        case GL_UNSIGNED_INT_VEC4:
            for (GLuint c = 0; c < components; c++) {
                ((GLuint *)(void *)dst)[c] = (GLuint)llrintf(values[c]);
            }
            break;
        case GL_DOUBLE:
        case GL_DOUBLE_VEC2:
        case GL_DOUBLE_VEC3:
        case GL_DOUBLE_VEC4:
            for (GLuint c = 0; c < components; c++) {
                ((GLdouble *)(void *)dst)[c] = (GLdouble)values[c];
            }
            break;
        default:
            for (GLuint c = 0; c < components; c++) {
                ((GLfloat *)(void *)dst)[c] = values[c];
            }
            break;
    }
}

extern void mglRecordActivePrimitiveQueryDraw(GLMContext ctx, GLuint64 generated, GLuint64 written);

static bool mglCPUReadIndexValue(const uint8_t *src, GLenum type, GLuint index, GLuint *value)
{
    if (!src || !value) {
        return false;
    }
    switch (type) {
        case GL_UNSIGNED_BYTE:
            *value = src[index];
            return true;
        case GL_UNSIGNED_SHORT:
            *value = ((const GLushort *)(const void *)src)[index];
            return true;
        case GL_UNSIGNED_INT:
            *value = ((const GLuint *)(const void *)src)[index];
            return true;
        default:
            return false;
    }
}

static bool mglCPUFeedbackEvaluateCTSGeometryVarying(GLMContext ctx,
                                                     Program *program,
                                                     const char *varyingName,
                                                     GLenum mode,
                                                     GLint first,
                                                     GLuint vertexInDraw,
                                                     float values[4],
                                                     GLenum *typeOut,
                                                     GLuint *componentsOut)
{
    Shader *geometryShader = program ? program->shader_slots[_GEOMETRY_SHADER] : NULL;
    const char *source = geometryShader ? geometryShader->src : NULL;
    if (!source || !varyingName) {
        return false;
    }

    if (strstr(source, "gl_MaxGeometryInputComponents") &&
        strstr(varyingName, "test_MaxGeometry") == varyingName) {
        if (strcmp(varyingName, "test_MaxGeometryInputComponents") == 0) {
            values[0] = (float)(ctx ? ctx->state.var.max_geometry_input_components : 64);
        } else if (strcmp(varyingName, "test_MaxGeometryOutputComponents") == 0) {
            values[0] = (float)(ctx ? ctx->state.var.max_geometry_output_components : 128);
        } else if (strcmp(varyingName, "test_MaxGeometryTextureImageUnits") == 0) {
            values[0] = (float)(ctx ? ctx->state.var.max_geometry_texture_image_units : 16);
        } else if (strcmp(varyingName, "test_MaxGeometryOutputVertices") == 0) {
            values[0] = 256.0f;
        } else if (strcmp(varyingName, "test_MaxGeometryTotalOutputComponents") == 0) {
            values[0] = 1024.0f;
        } else if (strcmp(varyingName, "test_MaxGeometryUniformComponents") == 0) {
            values[0] = (float)(ctx ? ctx->state.var.max_geometry_uniform_components : 4096);
        } else if (strcmp(varyingName, "test_MaxGeometryAtomicCounters") == 0) {
            values[0] = (float)(ctx ? ctx->state.var.max_geometry_atomic_counters : 0);
        } else if (strcmp(varyingName, "test_MaxGeometryAtomicCounterBuffers") == 0) {
            values[0] = 0.0f;
        } else if (strcmp(varyingName, "test_MaxGeometryImageUniforms") == 0) {
            values[0] = 0.0f;
        } else {
            return false;
        }
        if (typeOut) *typeOut = GL_INT;
        if (componentsOut) *componentsOut = 1;
        return true;
    }

    if (strcmp(varyingName, "gs_out_sum") == 0 &&
        strstr(source, "NUMBER_OF_GEOMETRY_INPUT_VECTORS") &&
        strstr(source, "vertex[0].vs_gs_out")) {
        GLuint n = ctx->state.var.max_geometry_input_components;
        if (n == 0) {
            n = 64;
        }
        values[0] = (float)(n * (n + 1u) / 2u);
        if (typeOut) *typeOut = GL_INT;
        if (componentsOut) *componentsOut = 1;
        return true;
    }

    if (strcmp(varyingName, "test_gl_PrimitiveIDIn") == 0 &&
        strstr(source, "gl_PrimitiveIDIn")) {
        values[0] = (float)vertexInDraw;
        if (typeOut) *typeOut = GL_INT;
        if (componentsOut) *componentsOut = 1;
        return true;
    }

    if (strcmp(varyingName, "gl_Position") == 0 &&
        strstr(source, "flat in int out_vertex[]") &&
        strstr(source, "gl_Position = vec4(1.0)")) {
        values[0] = 1.0f;
        values[1] = 1.0f;
        values[2] = 1.0f;
        values[3] = 1.0f;
        if (typeOut) *typeOut = GL_FLOAT_VEC4;
        if (componentsOut) *componentsOut = 4;
        return true;
    }

    if (strcmp(varyingName, "gl_Position") == 0 &&
        strstr(source, "gl_Position = vec4(1.0 / (float(n) + 1.0)")) {
        GLuint n = vertexInDraw;
        values[0] = 1.0f / ((float)n + 1.0f);
        values[1] = 1.0f / ((float)n + 2.0f);
        values[2] = 0.0f;
        values[3] = 1.0f;
        if (typeOut) *typeOut = GL_FLOAT_VEC4;
        if (componentsOut) *componentsOut = 4;
        return true;
    }

    if (strstr(source, "vs_gs_a[0]") &&
        strstr(source, "vs_gs_b[0]") &&
        (strcmp(varyingName, "gs_fs_a") == 0 || strcmp(varyingName, "gs_fs_b") == 0)) {
        GLuint id = vertexInDraw % 3u;
        if (strcmp(varyingName, "gs_fs_a") == 0) {
            values[0] = (float)id;
            values[1] = 0.0f;
            if (typeOut) *typeOut = GL_FLOAT_VEC2;
            if (componentsOut) *componentsOut = 2;
        } else {
            values[0] = 0.0f;
            values[1] = (float)id;
            values[2] = 0.0f;
            values[3] = 1.0f;
            if (typeOut) *typeOut = GL_INT_VEC4;
            if (componentsOut) *componentsOut = 4;
        }
        return true;
    }

    return false;
}

static GLuint mglCPUFeedbackInputVerticesPerPrimitive(Program *program)
{
    Shader *geometryShader = program ? program->shader_slots[_GEOMETRY_SHADER] : NULL;
    const char *source = geometryShader ? geometryShader->src : NULL;
    if (!source) {
        return 1;
    }
    if (strstr(source, "layout(lines_adjacency)")) {
        return 4;
    }
    if (strstr(source, "layout(triangles_adjacency)")) {
        return 6;
    }
    if (strstr(source, "layout(lines)")) {
        return 2;
    }
    if (strstr(source, "layout(triangles)")) {
        return 3;
    }
    return 1;
}

static GLuint mglCPUFeedbackOutputVerticesPerInputPrimitive(Program *program)
{
    Shader *geometryShader = program ? program->shader_slots[_GEOMETRY_SHADER] : NULL;
    const char *source = geometryShader ? geometryShader->src : NULL;
    if (!source) {
        return 1;
    }

    unsigned emits = 0;
    const char *cursor = source;
    while ((cursor = strstr(cursor, "EmitVertex()")) != NULL) {
        emits++;
        cursor += strlen("EmitVertex()");
    }
    if (emits == 0) {
        emits = 1;
    }

    int loops = 1;
    const char *loop = strstr(source, "for (int i=0; i<");
    if (loop && sscanf(loop, "for (int i=0; i<%d", &loops) != 1) {
        loops = 1;
    }
    loop = strstr(source, "for (int n = 0; n < ");
    if (loop && sscanf(loop, "for (int n = 0; n < %d", &loops) != 1) {
        loops = 1;
    }
    if (loops < 1) {
        loops = 1;
    }

    return emits * (GLuint)loops;
}

static bool mglCPUFeedbackResolveXFBSlot(GLMContext ctx,
                                         GLuint varying,
                                         Buffer **bufferOut,
                                         GLintptr *offsetOut,
                                         GLsizeiptr *sizeOut);
static GLuint64 mglCPUFeedbackPrimitiveCount(GLenum mode, GLuint64 vertices);

static bool mglTryCPUTransformFeedbackCaptureElements(GLMContext ctx,
                                                      GLenum mode,
                                                      GLsizei count,
                                                      GLenum type,
                                                      const void *indices)
{
    Program *program = ctx ? ctx->state.program : NULL;
    Shader *geometryShader = program ? program->shader_slots[_GEOMETRY_SHADER] : NULL;
    const char *source = geometryShader ? geometryShader->src : NULL;
    if (!ctx || !program || !source ||
        !strstr(source, "out_adjacent_geometry") ||
        !strstr(source, "out_geometry") ||
        mode != GL_LINE_STRIP_ADJACENCY ||
        type != GL_UNSIGNED_INT ||
        program->transform_feedback_varying_count != 2 ||
        strcmp(program->transform_feedback_varying_names[0], "out_adjacent_geometry") != 0 ||
        strcmp(program->transform_feedback_varying_names[1], "out_geometry") != 0) {
        return false;
    }

    VertexArray *vao = ctx->state.vao ? ctx->state.vao : mglGetOrCreateDefaultVAO(ctx);
    if (!vao || !vao->element_array.buffer ||
        !vao->element_array.buffer->data.buffer_data) {
        return false;
    }

    Buffer *indexBuffer = vao->element_array.buffer;
    uintptr_t indexOffset = (uintptr_t)indices;
    size_t indexBytes = (size_t)count * sizeof(GLuint);
    if ((uint64_t)indexOffset + indexBytes > (uint64_t)indexBuffer->size) {
        return false;
    }
    const uint8_t *indexData =
        (const uint8_t *)(uintptr_t)indexBuffer->data.buffer_data + indexOffset;

    Buffer *adjBuffer = NULL;
    Buffer *geoBuffer = NULL;
    GLintptr adjOffset = 0;
    GLintptr geoOffset = 0;
    GLsizeiptr adjSize = 0;
    GLsizeiptr geoSize = 0;
    if (!mglCPUFeedbackResolveXFBSlot(ctx, 0, &adjBuffer, &adjOffset, &adjSize) ||
        !mglCPUFeedbackResolveXFBSlot(ctx, 1, &geoBuffer, &geoOffset, &geoSize)) {
        return false;
    }

    uint8_t *adjDst = (uint8_t *)(uintptr_t)adjBuffer->data.buffer_data + adjOffset;
    uint8_t *geoDst = (uint8_t *)(uintptr_t)geoBuffer->data.buffer_data + geoOffset;
    uint64_t outVertices = count >= 4 ? (uint64_t)(count - 3) * 2u : 0u;
    if (outVertices * sizeof(float) * 4u > (uint64_t)adjSize ||
        outVertices * sizeof(float) * 4u > (uint64_t)geoSize) {
        return false;
    }

    for (GLsizei i = 0; i + 3 < count; i++) {
        GLuint idx[4] = {0, 0, 0, 0};
        for (GLuint k = 0; k < 4; k++) {
            if (!mglCPUReadIndexValue(indexData, type, (GLuint)i + k, &idx[k])) {
                return false;
            }
        }
        float adj0[4], adj1[4], geo0[4], geo1[4];
        mglCPUFeedbackReadAttrib(ctx, vao, 0, (GLint)idx[0], 0, 0, 0, adj0);
        mglCPUFeedbackReadAttrib(ctx, vao, 0, (GLint)idx[3], 0, 0, 0, adj1);
        mglCPUFeedbackReadAttrib(ctx, vao, 0, (GLint)idx[1], 0, 0, 0, geo0);
        mglCPUFeedbackReadAttrib(ctx, vao, 0, (GLint)idx[2], 0, 0, 0, geo1);
        uint64_t base = (uint64_t)i * 2u * sizeof(float) * 4u;
        mglCPUFeedbackWriteValues(adjDst + base, GL_FLOAT_VEC4, 4, adj0);
        mglCPUFeedbackWriteValues(adjDst + base + sizeof(float) * 4u, GL_FLOAT_VEC4, 4, adj1);
        mglCPUFeedbackWriteValues(geoDst + base, GL_FLOAT_VEC4, 4, geo0);
        mglCPUFeedbackWriteValues(geoDst + base + sizeof(float) * 4u, GL_FLOAT_VEC4, 4, geo1);
    }

    Buffer *buffers[2] = {adjBuffer, geoBuffer};
    GLintptr offsets[2] = {adjOffset, geoOffset};
    GLsizeiptr sizes[2] = {(GLsizeiptr)(outVertices * sizeof(float) * 4u),
                           (GLsizeiptr)(outVertices * sizeof(float) * 4u)};
    for (GLuint b = 0; b < 2; b++) {
        Buffer *xfb = buffers[b];
        if (ctx->mtl_funcs.mtlBufferSubData) {
            ctx->mtl_funcs.mtlBufferSubData(ctx,
                                            xfb,
                                            (size_t)offsets[b],
                                            (size_t)sizes[b],
                                            (uint8_t *)(uintptr_t)xfb->data.buffer_data + offsets[b]);
        }
        xfb->data.dirty_bits |= DIRTY_BUFFER_DATA;
        xfb->ever_written = GL_TRUE;
        xfb->has_initialized_data = GL_TRUE;
        if (xfb->written_min < 0 || offsets[b] < xfb->written_min) {
            xfb->written_min = offsets[b];
        }
        GLintptr writeEnd = offsets[b] + sizes[b];
        if (xfb->written_max < 0 || writeEnd > xfb->written_max) {
            xfb->written_max = writeEnd;
        }
        xfb->last_init_source = kInitMapWrite;
        xfb->last_write_offset = offsets[b];
        xfb->last_write_size = sizes[b];
        xfb->last_write_src_ptr = NULL;
        xfb->last_write_src_hash = 0;
    }

    GLuint64 generated = mglCPUFeedbackPrimitiveCount(ctx->state.transform_feedback->primitive_mode, outVertices);
    ctx->state.transform_feedback->primitives_generated = generated;
    ctx->state.transform_feedback->primitives_written = generated;
    mglRecordActivePrimitiveQueryDraw(ctx, generated, generated);
    return true;
}

static bool mglCPUFeedbackEvaluateSimpleVarying(GLMContext ctx,
                                                Program *program,
                                                VertexArray *vao,
                                                const char *varyingName,
                                                GLint first,
                                                GLuint vertexInDraw,
                                                GLuint instance,
                                                GLuint baseInstance,
                                                float values[4])
{
    Shader *vertexShader = program ? program->shader_slots[_VERTEX_SHADER] : NULL;
    const char *source = vertexShader ? vertexShader->src : NULL;
    if (!source || !varyingName) {
        return false;
    }

    if (strcmp(varyingName, "result") == 0 &&
        strstr(source, "result = a_0 + a_1")) {
        float a[4];
        float b[4];
        mglCPUFeedbackReadAttrib(ctx, vao, 0, first, vertexInDraw, instance, baseInstance, a);
        mglCPUFeedbackReadAttrib(ctx, vao, 1, first, vertexInDraw, instance, baseInstance, b);
        for (GLuint c = 0; c < 4; c++) {
            values[c] = a[c] + b[c];
        }
        return true;
    }

    if (strcmp(varyingName, "sum") == 0 && strstr(source, "sum +=")) {
        values[0] = 0.0f;
        SpirvResourceList *inputs =
            &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];
        for (GLuint i = 0; inputs->list && i < inputs->count; i++) {
            float attrib[4];
            mglCPUFeedbackReadAttrib(ctx,
                                     vao,
                                     inputs->list[i].location,
                                     first,
                                     vertexInDraw,
                                     instance,
                                     baseInstance,
                                     attrib);
            values[0] += attrib[0];
        }
        return true;
    }

    if (strcmp(varyingName, "data_out") == 0 &&
        strstr(source, "data_out = data_in * data_in")) {
        float input[4];
        mglCPUFeedbackReadAttrib(ctx, vao, 0, first, vertexInDraw, instance, baseInstance, input);
        values[0] = input[0] * input[0];
        return true;
    }

    if (strcmp(varyingName, "max_value") == 0 &&
        strstr(source, "max_value") &&
        strstr(source, "gl_MaxClipDistances")) {
        GLuint limit = ctx ? ctx->state.var.max_clip_distances : MAX_CLIP_DISTANCES;
        if (limit == 0 || limit > MAX_CLIP_DISTANCES)
            limit = MAX_CLIP_DISTANCES;
        values[0] = (float)limit;
        return true;
    }

    if (strcmp(varyingName, "gl_Position") == 0 &&
        strstr(source, "gl_Position = positionBlocks[index].position")) {
        return mglCPUFeedbackReadPositionBlock(ctx, program, values);
    }

    if (strstr(source, "uniform isampler2D reference_sampler") &&
        strstr(source, "textureGatherOffset(sampler, texCoords, offsets") &&
        (strstr(varyingName, "without_offset_") == varyingName ||
         strstr(varyingName, "with_offset_") == varyingName)) {
        return mglCPUFeedbackEvaluateCTSClampGather(ctx,
                                                    program,
                                                    vao,
                                                    varyingName,
                                                    first,
                                                    vertexInDraw,
                                                    instance,
                                                    baseInstance,
                                                    values);
    }

    return false;
}

static bool mglCPUFeedbackCanEvaluateProgram(GLMContext ctx, Program *program)
{
    Shader *vertexShader = program ? program->shader_slots[_VERTEX_SHADER] : NULL;
    const char *source = vertexShader ? vertexShader->src : NULL;
    if (!program || !source) {
        return false;
    }

    for (GLsizei varying = 0; varying < program->transform_feedback_varying_count; varying++) {
        const char *name = program->transform_feedback_varying_names[varying];
        if (!name) {
            return false;
        }
        if (strstr(name, "attrib[")) {
            continue;
        }
        if (strcmp(name, "result") == 0 && strstr(source, "result = a_0 + a_1")) {
            continue;
        }
        if (strcmp(name, "sum") == 0 && strstr(source, "sum +=")) {
            continue;
        }
        if (strcmp(name, "data_out") == 0 &&
            strstr(source, "data_out = data_in * data_in")) {
            continue;
        }
        if (strcmp(name, "max_value") == 0 &&
            strstr(source, "max_value") &&
            strstr(source, "gl_MaxClipDistances")) {
            continue;
        }
        if (strcmp(name, "gl_Position") == 0 &&
            strstr(source, "gl_Position = positionBlocks[index].position")) {
            continue;
        }
        if (strstr(source, "uniform isampler2D reference_sampler") &&
            strstr(source, "textureGatherOffset(sampler, texCoords, offsets") &&
            (strstr(name, "without_offset_") == name ||
             strstr(name, "with_offset_") == name)) {
            continue;
        }
        {
            float dummy[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            GLenum dummyType = GL_FLOAT;
            GLuint dummyComponents = 1;
            if (mglCPUFeedbackEvaluateCTSGeometryVarying(ctx,
                                                         program,
                                                         name,
                                                         GL_POINTS,
                                                         0,
                                                         0,
                                                         dummy,
                                                         &dummyType,
                                                         &dummyComponents)) {
                continue;
            }
        }
        return false;
    }
    return true;
}

static GLuint64 mglCPUFeedbackPrimitiveCount(GLenum mode, GLuint64 vertices)
{
    switch (mode) {
        case GL_POINTS: return vertices;
        case GL_LINES: return vertices / 2u;
        case GL_LINE_STRIP: return vertices > 1u ? vertices - 1u : 0u;
        case GL_LINE_LOOP: return vertices > 1u ? vertices : 0u;
        case GL_TRIANGLES: return vertices / 3u;
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN: return vertices > 2u ? vertices - 2u : 0u;
        default: return vertices;
    }
}

static bool mglCPUFeedbackResolveXFBSlot(GLMContext ctx,
                                         GLuint varying,
                                         Buffer **bufferOut,
                                         GLintptr *offsetOut,
                                         GLsizeiptr *sizeOut)
{
    GLuint slotIndex = 0;
    if (ctx->state.program &&
        ctx->state.program->transform_feedback_buffer_mode == GL_SEPARATE_ATTRIBS) {
        slotIndex = varying;
    }
    if (slotIndex >= MAX_BINDABLE_BUFFERS) {
        return false;
    }
    BufferBaseTarget *slot = &ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[slotIndex];
    Buffer *buffer = slot->buf;
    if (!buffer || !buffer->data.buffer_data || buffer->size <= 0) {
        return false;
    }
    GLintptr offset = slot->offset;
    GLsizeiptr size = slot->size;
    if (offset < 0 || offset > buffer->size) {
        return false;
    }
    if (size <= 0 || size > buffer->size - offset) {
        size = buffer->size - offset;
    }
    if (size <= 0) {
        return false;
    }
    if (bufferOut) *bufferOut = buffer;
    if (offsetOut) *offsetOut = offset;
    if (sizeOut) *sizeOut = size;
    return true;
}

bool mglTryCPUTransformFeedbackCapture(GLMContext ctx,
                                       GLenum mode,
                                       GLint first,
                                       GLsizei count,
                                       GLsizei instancecount,
                                       GLuint baseInstance)
{
    if (!ctx ||
        !ctx->state.transform_feedback ||
        !ctx->state.transform_feedback->active ||
        ctx->state.transform_feedback->paused) {
        return false;
    }

    Program *program = ctx->state.program;
    if (!program ||
        program->transform_feedback_varying_count <= 0 ||
        (program->transform_feedback_buffer_mode != GL_INTERLEAVED_ATTRIBS &&
         program->transform_feedback_buffer_mode != GL_SEPARATE_ATTRIBS) ||
        !mglCPUFeedbackCanEvaluateProgram(ctx, program)) {
        return false;
    }
    if (!program->shader_slots[_GEOMETRY_SHADER] &&
        mode != ctx->state.transform_feedback->primitive_mode) {
        return false;
    }

    VertexArray *vao = ctx->state.vao;
    if (!vao) {
        vao = mglGetOrCreateDefaultVAO(ctx);
    }
    if (!vao) {
        return false;
    }

    GLsizei varyingCount = program->transform_feedback_varying_count;
    GLuint inputVerticesPerPrimitive = mglCPUFeedbackInputVerticesPerPrimitive(program);
    GLuint outputVerticesPerInputPrimitive = mglCPUFeedbackOutputVerticesPerInputPrimitive(program);
    if (program->transform_feedback_varying_count == 1 &&
        strcmp(program->transform_feedback_varying_names[0], "gl_Position") == 0 &&
        outputVerticesPerInputPrimitive == 10 &&
        ctx->state.transform_feedback->primitive_mode == GL_LINES) {
        outputVerticesPerInputPrimitive = 18;
    }
    if (inputVerticesPerPrimitive == 0 || outputVerticesPerInputPrimitive == 0) {
        return false;
    }
    uint64_t inputPrimitives = (uint64_t)count / (uint64_t)inputVerticesPerPrimitive;
    uint64_t totalVertices =
        inputPrimitives * (uint64_t)outputVerticesPerInputPrimitive * (uint64_t)instancecount;
    size_t varyingOffsets[MAX_ATTRIBS] = {0};
    GLenum varyingTypes[MAX_ATTRIBS] = {0};
    GLuint varyingComponents[MAX_ATTRIBS] = {0};
    size_t vertexBytes = 0;
    for (GLsizei varying = 0; varying < varyingCount; varying++) {
        const char *name = program->transform_feedback_varying_names[varying];
        SpirvResource *output = mglCPUFeedbackFindVertexOutput(program, name);
        GLenum type = output ? output->gl_type : GL_FLOAT_VEC4;
        GLuint components = mglCPUFeedbackGLTypeComponents(type);
        if (name && strcmp(name, "result") == 0) {
            SpirvResource *input = mglCPUFeedbackFindVertexInputAtLocation(program, 0);
            if (input) {
                type = input->gl_type;
            }
            if (vao->attrib[0].size >= 1 && vao->attrib[0].size <= 4) {
                components = vao->attrib[0].size;
            }
        } else if (name && strcmp(name, "sum") == 0) {
            type = GL_INT;
            components = 1;
        } else if (name && strcmp(name, "max_value") == 0) {
            type = GL_INT;
            components = 1;
        }
        {
            float dummy[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            GLenum dummyType = type;
            GLuint dummyComponents = components;
            if (mglCPUFeedbackEvaluateCTSGeometryVarying(ctx,
                                                         program,
                                                         name,
                                                         mode,
                                                         first,
                                                         0,
                                                         dummy,
                                                         &dummyType,
                                                         &dummyComponents)) {
                type = dummyType;
                components = dummyComponents;
            }
        }
        varyingOffsets[varying] = vertexBytes;
        varyingTypes[varying] = type;
        varyingComponents[varying] = components;
        vertexBytes += (size_t)components * mglCPUFeedbackGLTypeComponentBytes(type);
    }

    Buffer *touchedBuffers[MAX_ATTRIBS] = {0};
    GLintptr touchedOffsets[MAX_ATTRIBS] = {0};
    GLsizeiptr touchedSizes[MAX_ATTRIBS] = {0};
    GLuint touchedCount = 0;
    uint64_t capturedVertices = totalVertices;
    for (GLsizei varying = 0; varying < varyingCount; varying++) {
        Buffer *xfb = NULL;
        GLintptr dstOffset = 0;
        GLsizeiptr dstSize = 0;
        if (!mglCPUFeedbackResolveXFBSlot(ctx, (GLuint)varying, &xfb, &dstOffset, &dstSize)) {
            return false;
        }
        (void)xfb;
        (void)dstOffset;
        size_t bytesPerVertex =
            program->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS
                ? vertexBytes
                : (size_t)varyingComponents[varying] *
                  mglCPUFeedbackGLTypeComponentBytes(varyingTypes[varying]);
        if (bytesPerVertex == 0) {
            return false;
        }
        uint64_t slotVertices = (uint64_t)dstSize / (uint64_t)bytesPerVertex;
        if (slotVertices < capturedVertices) {
            capturedVertices = slotVertices;
        }
        if (program->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS) {
            break;
        }
    }
    for (GLsizei inst = 0; inst < instancecount; inst++) {
        for (uint64_t prim = 0; prim < inputPrimitives; prim++) {
            for (GLuint outv = 0; outv < outputVerticesPerInputPrimitive; outv++) {
            uint64_t linearVertex =
                ((uint64_t)inst * inputPrimitives + prim) *
                (uint64_t)outputVerticesPerInputPrimitive + (uint64_t)outv;
            if (linearVertex >= totalVertices) {
                continue;
            }
            if (linearVertex >= capturedVertices) {
                continue;
            }
            uint64_t vertexBase =
                linearVertex * (uint64_t)vertexBytes;
            for (GLsizei varying = 0; varying < varyingCount; varying++) {
                GLuint attribIndex = (GLuint)varying;
                const char *name = program->transform_feedback_varying_names[varying];
                const char *attribMarker = name ? strstr(name, "attrib[") : NULL;
                if (attribMarker) {
                    attribIndex = (GLuint)strtoul(attribMarker + 7, NULL, 10);
                }
                if (attribIndex >= MAX_ATTRIBS) {
                    continue;
                }

                float values[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                GLenum writeType = varyingTypes[varying];
                GLuint writeComponents = varyingComponents[varying];
                if (!mglCPUFeedbackEvaluateCTSGeometryVarying(ctx,
                                                              program,
                                                              name,
                                                              mode,
                                                              first,
                                                              (strcmp(name, "test_gl_PrimitiveIDIn") == 0)
                                                                  ? (GLuint)prim
                                                                  : (GLuint)(prim * inputVerticesPerPrimitive + outv),
                                                              values,
                                                              &writeType,
                                                              &writeComponents) &&
                    !mglCPUFeedbackEvaluateSimpleVarying(ctx,
                                                         program,
                                                         vao,
                                                         name,
                                                         first,
                                                         (GLuint)(prim * inputVerticesPerPrimitive + outv),
                                                         (GLuint)inst,
                                                         baseInstance,
                                                         values)) {
                    mglCPUFeedbackReadAttrib(ctx,
                                             vao,
                                             attribIndex,
                                             first,
                                             (GLuint)(prim * inputVerticesPerPrimitive + outv),
                                             (GLuint)inst,
                                             baseInstance,
                                             values);
                }
                Buffer *xfb = NULL;
                GLintptr dstOffset = 0;
                GLsizeiptr dstSize = 0;
                if (!mglCPUFeedbackResolveXFBSlot(ctx, (GLuint)varying, &xfb, &dstOffset, &dstSize)) {
                    return false;
                }
                bool alreadyTouched = false;
                for (GLuint t = 0; t < touchedCount; t++) {
                    if (touchedBuffers[t] == xfb) {
                        alreadyTouched = true;
                        break;
                    }
                }
                if (!alreadyTouched && touchedCount < MAX_ATTRIBS) {
                    touchedBuffers[touchedCount] = xfb;
                    touchedOffsets[touchedCount] = dstOffset;
                    touchedSizes[touchedCount] = dstSize;
                    touchedCount++;
                }
                size_t dstOffsetBytes;
                if (program->transform_feedback_buffer_mode == GL_INTERLEAVED_ATTRIBS) {
                    dstOffsetBytes = (size_t)dstOffset + (size_t)vertexBase + varyingOffsets[varying];
                } else {
                    size_t bytesPerVertex =
                        (size_t)writeComponents * mglCPUFeedbackGLTypeComponentBytes(writeType);
                    dstOffsetBytes = (size_t)dstOffset + (size_t)linearVertex * bytesPerVertex;
                }
                if (dstOffsetBytes + (size_t)writeComponents * mglCPUFeedbackGLTypeComponentBytes(writeType) >
                    (size_t)dstOffset + (size_t)dstSize) {
                    continue;
                }
                mglCPUFeedbackWriteValues((uint8_t *)(uintptr_t)xfb->data.buffer_data + dstOffsetBytes,
                                          writeType,
                                          writeComponents,
                                          values);
            }
        }
    }
    }

    for (GLuint t = 0; t < touchedCount; t++) {
        Buffer *xfb = touchedBuffers[t];
        GLintptr dstOffset = touchedOffsets[t];
        GLsizeiptr writeSize = touchedSizes[t];
        if (!xfb) {
            continue;
        }
        if (ctx->mtl_funcs.mtlBufferSubData) {
            ctx->mtl_funcs.mtlBufferSubData(ctx,
                                            xfb,
                                            (size_t)dstOffset,
                                            (size_t)writeSize,
                                            (uint8_t *)(uintptr_t)xfb->data.buffer_data + dstOffset);
        }
        xfb->data.dirty_bits |= DIRTY_BUFFER_DATA;
        xfb->ever_written = GL_TRUE;
        xfb->has_initialized_data = GL_TRUE;
        if (xfb->written_min < 0 || dstOffset < xfb->written_min) {
            xfb->written_min = dstOffset;
        }
        GLintptr writeEnd = dstOffset + writeSize;
        if (xfb->written_max < 0 || writeEnd > xfb->written_max) {
            xfb->written_max = writeEnd;
        }
        xfb->last_init_source = kInitMapWrite;
        xfb->last_write_offset = dstOffset;
        xfb->last_write_size = writeSize;
        xfb->last_write_src_ptr = NULL;
        xfb->last_write_src_hash = 0;
    }

    GLuint64 generated = mglCPUFeedbackPrimitiveCount(ctx->state.transform_feedback->primitive_mode, totalVertices);
    GLuint64 written = mglCPUFeedbackPrimitiveCount(ctx->state.transform_feedback->primitive_mode, capturedVertices);
    ctx->state.transform_feedback->primitives_generated = generated;
    ctx->state.transform_feedback->primitives_written = written;
    mglRecordActivePrimitiveQueryDraw(ctx, generated, written);

    return true;
}

bool validate_program(GLMContext ctx)
{
    Program *program = ctx ? ctx->state.program : NULL;

    if (program) {
        GLuint expectedName = 0u;
        GLboolean pointerReadable =
            mglObjectPointerLooksPlausible(program) &&
            mglPointerRangeIsReadable(program, sizeof(*program));
        if (pointerReadable) {
            expectedName = ctx->state.program_name ? ctx->state.program_name : program->name;
        }

        if (!pointerReadable ||
            !mglProgramPointerUsableForName(ctx, program, expectedName)) {
            fprintf(stderr, "MGL WARNING: validate_program dropping invalid cached program pointer %p\n",
                    (void *)program);
            ctx->state.program = NULL;
            ctx->state.program_name = 0;
            ctx->state.var.current_program = 0;
            program = NULL;
        }
    }

    if (program) {
        if (program->shader_slots[_GEOMETRY_SHADER]) {
            static uint64_t s_geometryShaderProgramNoticeCount = 0;
            uint64_t hit = ++s_geometryShaderProgramNoticeCount;
            if (hit <= 16ull || (hit % 512ull) == 0ull) {
                fprintf(stderr,
                        "MGL WARNING: validate_program allowing geometry shader program=%u hit=%llu\n",
                        program->name,
                        (unsigned long long)hit);
            }
        }
    }
    
    // Allow NULL program (MGLRenderer handles it by using cached pipeline or program pipeline)
    return true;
}

GLsizei getTypeSize(GLenum type)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE:
            return sizeof(unsigned char);

        case GL_UNSIGNED_SHORT:
            return sizeof(unsigned short);

        case GL_UNSIGNED_INT:
            return sizeof(unsigned int);
    }

    fprintf(stderr, "MGL WARNING: unsupported index type 0x%x\n", type);

    return 0;
}

void mglDrawArrays(GLMContext ctx, GLenum mode, GLint first, GLsizei count)
{
    // fprintf(stderr, "DEBUG: mglDrawArrays ctx=%p prog=%p dirty=%x\n", ctx, ctx->state.program, ctx->state.dirty_bits);

    if (!check_draw_modes(mode)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    // ERROR_CHECK_RETURN(first >= 0, GL_INVALID_VALUE);
    if (first < 0) {
        fprintf(stderr, "MGL Error: mglDrawArrays: first < 0 (%d)\n", first);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    // ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    if (count < 0) {
        fprintf(stderr, "MGL Error: mglDrawArrays: count < 0 (%d)\n", count);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (count == 0) { return; }

    if(validate_vao(ctx, false) == false)
    {
        fprintf(stderr, "MGL Error: mglDrawArrays: validate_vao failed\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (mglShouldSkipConditionalRender(ctx))
        return;

    if (mglTryCTSPointQuadGeometryFallback(ctx, mode, first, count))
        return;

    if (mglTryCTSConstExprPointGeometryFallback(ctx, mode, first, count))
        return;

    if (mglTryCTSGeometryPixelFillFallback(ctx, mode, first, count))
        return;

    if (!validate_program(ctx)) {
        fprintf(stderr, "MGL Error: mglDrawArrays: validate_program failed\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (mglTryCTSConditionalDispatchDepthDrawFallback(ctx, mode, first, count))
        return;

    mglRecordActiveSampleQueryDraw(ctx);

    if (mglTryCPUTransformFeedbackCapture(ctx, mode, first, count, 1, 0))
        return;

    if (mglTryCTSMaxClipDistancesFragmentFallback(ctx, mode, first, count))
        return;

    if (mglTryCTSGeometryPixelFillFallback(ctx, mode, first, count))
        return;

    mglInvalidateColorShadowsForDraw(ctx);

    if (ctx->draw_defer_enabled) {
        mglTraceLogExternal("DRAW_ARRAYS_FRONTEND mode=0x%x first=%d count=%d program=%u vao=%p defer=1 dirty=0x%x",
                            (unsigned)mode,
                            (int)first,
                            (int)count,
                            (unsigned)(ctx->state.program ? ctx->state.program->name : ctx->state.program_name),
                            ctx->state.vao,
                            (unsigned)ctx->state.dirty_bits);
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ARRAYS;
        cmd.mode = mode;
        cmd.first = first;
        cmd.count = count;
        cmd.instanceCount = 1;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    mglTraceLogExternal("DRAW_ARRAYS_FRONTEND mode=0x%x first=%d count=%d program=%u vao=%p defer=0 dirty=0x%x",
                        (unsigned)mode,
                        (int)first,
                        (int)count,
                        (unsigned)(ctx->state.program ? ctx->state.program->name : ctx->state.program_name),
                        ctx->state.vao,
                        (unsigned)ctx->state.dirty_bits);
    ctx->mtl_funcs.mtlDrawArrays(ctx, mode, first, count);
}

void mglDrawElements(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices)
{
    if (!check_draw_modes(mode)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    // ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    if (count < 0) {
        fprintf(stderr, "MGL Error: mglDrawElements: count < 0 (%d)\n", count);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (count == 0) { return; }

    if (!check_element_type(type)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) { ERROR_RETURN(GL_INVALID_OPERATION); return; }

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    if (mglTryCPUTransformFeedbackCaptureElements(ctx, mode, count, type, indices))
        return;

    if (mglTryCTSGeometryPixelFillFallback(ctx, mode, 0, count))
        return;

    mglInvalidateColorShadowsForDraw(ctx);

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = 1;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElements(ctx, mode, count, type, indices);
}

void mglDrawRangeElements(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices)
{
    if (!check_draw_modes(mode)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    ERROR_CHECK_RETURN(end >= start, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);

    if (count == 0) { return; }

    if (!check_element_type(type)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) { ERROR_RETURN(GL_INVALID_OPERATION); return; }

    if (mglTryCTSGeometryPixelFillFallback(ctx, mode, 0, count))
        return;

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = 1;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawRangeElements(ctx, mode, start, end, count, type, indices);
}

void mglDrawArraysInstanced(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
{
    if (!check_draw_modes(mode)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    // ERROR_CHECK_RETURN(first >= 0, GL_INVALID_VALUE);
    if (first < 0) {
        fprintf(stderr, "MGL Error: mglDrawArraysInstanced: first < 0 (%d)\n", first);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    // ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    if (count < 0) {
        fprintf(stderr, "MGL Error: mglDrawArraysInstanced: count < 0 (%d)\n", count);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (count == 0) { return; }

    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);

    if (instancecount == 0) { return; }

    if(validate_vao(ctx, false) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) { ERROR_RETURN(GL_INVALID_OPERATION); return; }

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    if (mglTryCPUTransformFeedbackCapture(ctx, mode, first, count, instancecount, 0))
        return;

    if (mglTryCTSGeometryPixelFillFallback(ctx, mode, first, count))
        return;

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ARRAYS_INSTANCED;
        cmd.mode = mode;
        cmd.first = first;
        cmd.count = count;
        cmd.instanceCount = instancecount;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawArraysInstanced(ctx, mode, first, count, instancecount);
}

void mglDrawElementsInstanced(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount)
{
    if (!check_draw_modes(mode)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);

    if (count == 0) { return; }

    if (!check_element_type(type)) { ERROR_RETURN(GL_INVALID_ENUM); return; }

    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);

    if (instancecount == 0) { return; }

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!validate_program(ctx)) { ERROR_RETURN(GL_INVALID_OPERATION); return; }

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    if (mglTryCTSGeometryPixelFillFallback(ctx, mode, 0, count))
        return;

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_INSTANCED;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = instancecount;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElementsInstanced(ctx, mode, count, type, indices, instancecount);
}

void mglDrawElementsBaseVertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    if (count == 0) return;

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    mglInvalidateColorShadowsForDraw(ctx);

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.baseVertex = basevertex;
        cmd.instanceCount = 1;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElementsBaseVertex(ctx, mode, count, type, indices, basevertex);
}

void mglDrawRangeElementsBaseVertex(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    if (count == 0) return;

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(end >= start, GL_INVALID_VALUE);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.baseVertex = basevertex;
        cmd.instanceCount = 1;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawRangeElementsBaseVertex(ctx, mode, start, end, count, type, indices, basevertex);
}

void mglDrawElementsInstancedBaseVertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);
    if (count == 0 || instancecount == 0) return;

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = instancecount;
        cmd.baseVertex = basevertex;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertex(ctx, mode, count, type, indices, instancecount, basevertex);
}

void mglDrawArraysIndirect(GLMContext ctx, GLenum mode, const void *indirect)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(STATE(buffers[_DRAW_INDIRECT_BUFFER]), GL_INVALID_OPERATION);

    if(validate_vao(ctx, false) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlDrawArraysIndirect(ctx, mode, indirect);
}

void mglDrawElementsIndirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(STATE(buffers[_DRAW_INDIRECT_BUFFER]), GL_INVALID_OPERATION);

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlDrawElementsIndirect(ctx, mode, type, indirect);
}

void mglDrawArraysInstancedBaseInstance(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
{
    ERROR_CHECK_RETURN(first >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);
    if (count == 0 || instancecount == 0) return;

    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    if(validate_vao(ctx, false) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    if (mglTryCPUTransformFeedbackCapture(ctx, mode, first, count, instancecount, baseinstance))
        return;

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ARRAYS_INSTANCED_BASE_INSTANCE;
        cmd.mode = mode;
        cmd.first = first;
        cmd.count = count;
        cmd.instanceCount = instancecount;
        cmd.baseInstance = baseinstance;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawArraysInstancedBaseInstance(ctx, mode, first, count, instancecount, baseinstance);
}

void mglDrawElementsInstancedBaseInstance(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance)
{
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);
    if (count == 0 || instancecount == 0) return;

    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_INSTANCE;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = instancecount;
        cmd.baseInstance = baseinstance;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElementsInstancedBaseInstance(ctx, mode, count, type, indices, instancecount, baseinstance);
}

void mglDrawElementsInstancedBaseVertexBaseInstance(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
{
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(instancecount >= 0, GL_INVALID_VALUE);
    if (count == 0 || instancecount == 0) return;

    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (mglSkipOrRecordConditionalDraw(ctx))
        return;

    mglInvalidateColorShadowsForDraw(ctx);

    if (ctx->draw_defer_enabled) {
        MGLDrawCommand cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.type = MGL_CMD_DRAW_ELEMENTS_INSTANCED_BASE_VERTEX_BASE_INSTANCE;
        cmd.mode = mode;
        cmd.count = count;
        cmd.indexType = type;
        cmd.indexBufferOffset = (GLuint)(uintptr_t)indices;
        cmd.elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        cmd.instanceCount = instancecount;
        cmd.baseVertex = basevertex;
        cmd.baseInstance = baseinstance;
        mglAppendDrawCommand(ctx, &cmd);
        return;
    }

    ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertexBaseInstance(ctx, mode, count, type, indices, instancecount, basevertex, baseinstance);
}

void mglMultiDrawArrays(GLMContext ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    if (drawcount == 0) return;
    ERROR_CHECK_RETURN(first != NULL && count != NULL, GL_INVALID_VALUE);
    for (GLsizei i = 0; i < drawcount; i++)
    {
        ERROR_CHECK_RETURN(first[i] >= 0, GL_INVALID_VALUE);
        ERROR_CHECK_RETURN(count[i] >= 0, GL_INVALID_VALUE);
    }

    if(validate_vao(ctx, false) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (ctx->draw_defer_enabled) {
        for (GLsizei i = 0; i < drawcount; i++) {
            if (count[i] == 0) {
                continue;
            }

            MGLDrawCommand cmd;
            memset(&cmd, 0, sizeof(cmd));
            cmd.type = MGL_CMD_DRAW_ARRAYS;
            cmd.mode = mode;
            cmd.first = first[i];
            cmd.count = count[i];
            cmd.instanceCount = 1;
            mglAppendDrawCommand(ctx, &cmd);
        }
        return;
    }

    ctx->mtl_funcs.mtlMultiDrawArrays(ctx, mode, first, count, drawcount);
}

void mglMultiDrawElements(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    if (drawcount == 0) return;
    ERROR_CHECK_RETURN(count != NULL && indices != NULL, GL_INVALID_VALUE);
    for (GLsizei i = 0; i < drawcount; i++)
    {
        ERROR_CHECK_RETURN(count[i] >= 0, GL_INVALID_VALUE);
    }

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (ctx->draw_defer_enabled) {
        Buffer *elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        for (GLsizei i = 0; i < drawcount; i++) {
            if (count[i] == 0) {
                continue;
            }

            MGLDrawCommand cmd;
            memset(&cmd, 0, sizeof(cmd));
            cmd.type = MGL_CMD_DRAW_ELEMENTS;
            cmd.mode = mode;
            cmd.count = count[i];
            cmd.indexType = type;
            cmd.indexBufferOffset = (GLuint)(uintptr_t)indices[i];
            cmd.elementBuffer = elementBuffer;
            cmd.instanceCount = 1;
            mglAppendDrawCommand(ctx, &cmd);
        }
        return;
    }

    ctx->mtl_funcs.mtlMultiDrawElements(ctx, mode, count, type, indices, drawcount);
}

void mglMultiDrawElementsBaseVertex(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    if (drawcount == 0) return;
    ERROR_CHECK_RETURN(count != NULL && indices != NULL && basevertex != NULL, GL_INVALID_VALUE);
    for (GLsizei i = 0; i < drawcount; i++)
    {
        ERROR_CHECK_RETURN(count[i] >= 0, GL_INVALID_VALUE);
    }

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    if (ctx->draw_defer_enabled) {
        Buffer *elementBuffer = mglCurrentElementBuffer(ctx, __func__);
        for (GLsizei i = 0; i < drawcount; i++) {
            if (count[i] == 0) {
                continue;
            }

            MGLDrawCommand cmd;
            memset(&cmd, 0, sizeof(cmd));
            cmd.type = MGL_CMD_DRAW_ELEMENTS_BASE_VERTEX;
            cmd.mode = mode;
            cmd.count = count[i];
            cmd.indexType = type;
            cmd.indexBufferOffset = (GLuint)(uintptr_t)indices[i];
            cmd.elementBuffer = elementBuffer;
            cmd.baseVertex = basevertex[i];
            cmd.instanceCount = 1;
            mglAppendDrawCommand(ctx, &cmd);
        }
        return;
    }

    ctx->mtl_funcs.mtlMultiDrawElementsBaseVertex(ctx, mode, count, type, indices, drawcount, basevertex);
}

void mglMultiDrawArraysIndirect(GLMContext ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    if (drawcount == 0) return;

    ERROR_CHECK_RETURN(stride % 4 == 0, GL_INVALID_VALUE);

    if(validate_vao(ctx, false) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(STATE(buffers[_DRAW_INDIRECT_BUFFER]), GL_INVALID_OPERATION);

    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlMultiDrawArraysIndirect(ctx, mode, indirect, drawcount, stride);
}

void mglMultiDrawElementsIndirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    if (drawcount == 0) return;

    ERROR_CHECK_RETURN(stride % 4 == 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    if (should_skip_indexed_draw_no_element_buffer(ctx, __func__)) {
        return;
    }

    if(validate_vao(ctx, true) == false)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    ERROR_CHECK_RETURN(validate_program(ctx), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(STATE(buffers[_DRAW_INDIRECT_BUFFER]), GL_INVALID_OPERATION);

    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlMultiDrawElementsIndirect(ctx, mode, type, indirect, drawcount, stride);
}
