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
 * textures.c
 * MGL
 *
 */

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <mach/mach_time.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>

#include <Accelerate/Accelerate.h>

#include "pixel_utils.h"
#include "utils.h"
#include "glm_context.h"

extern void *getBufferData(GLMContext ctx, Buffer *ptr);
extern GLsizei mglSafeMaxTextureSize(GLMContext ctx);
extern GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);

#ifndef MGL_VERBOSE_TEXTURE_UPLOAD_LOGS
#define MGL_VERBOSE_TEXTURE_UPLOAD_LOGS 0
#endif

#ifndef MGL_VERBOSE_TEXTURE_BIND_LOGS
#define MGL_VERBOSE_TEXTURE_BIND_LOGS 0
#endif

bool texSubImage(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels);
void invalidateTexture(GLMContext ctx, Texture *tex);

static inline double mglTextureNowMs(void)
{
    static mach_timebase_info_data_t s_timebase = {0, 0};
    if (s_timebase.denom == 0) {
        (void)mach_timebase_info(&s_timebase);
    }

    uint64_t t = mach_absolute_time();
    long double ns = (long double)t * (long double)s_timebase.numer / (long double)s_timebase.denom;
    return (double)(ns / 1000000.0L);
}

static void mglDumpBytesToStderr(const char *label,
                                 const uint8_t *bytes,
                                 size_t length,
                                 size_t base_offset)
{
    if (!label) {
        label = "dump";
    }

    if (!bytes || length == 0) {
        fprintf(stderr, "MGL DUMP %s empty\n", label);
        return;
    }

    const size_t row = 16u;
    for (size_t off = 0; off < length; off += row) {
        size_t n = (length - off) < row ? (length - off) : row;
        char hex[3 * 16 + 1];
        char ascii[16 + 1];
        size_t hp = 0;

        for (size_t i = 0; i < n; i++) {
            uint8_t b = bytes[off + i];
            int wrote = snprintf(hex + hp, sizeof(hex) - hp, "%02x", b);
            if (wrote <= 0) {
                break;
            }
            hp += (size_t)wrote;
            if (i + 1 < n && hp + 1 < sizeof(hex)) {
                hex[hp++] = ' ';
            }
            ascii[i] = (b >= 32u && b <= 126u) ? (char)b : '.';
        }
        hex[hp] = '\0';
        ascii[n] = '\0';

        fprintf(stderr,
                "MGL DUMP %s +0x%zx: %-47s |%s|\n",
                label,
                base_offset + off,
                hex,
                ascii);
    }
}

static uint64_t mglHashBytesSampled(const void *data, size_t len)
{
    if (!data || len == 0) {
        return 0ull;
    }

    const uint8_t *bytes = (const uint8_t *)data;
    size_t head = len < 1024u ? len : 1024u;
    uint64_t hash = 1469598103934665603ull;

    for (size_t i = 0; i < head; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ull;
    }

    if (len > head) {
        const uint8_t *tail = bytes + (len - head);
        for (size_t i = 0; i < head; i++) {
            hash ^= (uint64_t)tail[i];
            hash *= 1099511628211ull;
        }
    }

    hash ^= (uint64_t)len;
    hash *= 1099511628211ull;
    return hash;
}

static bool mglLooksAllZero(const uint8_t *bytes, size_t len)
{
    if (!bytes || len == 0) {
        return false;
    }

    for (size_t i = 0; i < len; i++) {
        if (bytes[i] != 0u) {
            return false;
        }
    }
    return true;
}

static bool mglShouldTraceTextureUpload(Texture *tex,
                                        GLuint unpack_name,
                                        GLsizei width,
                                        GLsizei height,
                                        GLsizei depth,
                                        size_t required_bytes)
{
    if (MGL_VERBOSE_TEXTURE_UPLOAD_LOGS) {
        return true;
    }

    if (tex && tex->name == 13u) {
        return true;
    }

    if (required_bytes >= (1024u * 1024u)) {
        return true;
    }

    if (width >= 512 && height >= 512) {
        return true;
    }

    if (depth > 1 && width >= 128 && height >= 128) {
        return true;
    }

    if (unpack_name != 0u) {
        static unsigned s_pbo_trace_count = 0u;
        if (s_pbo_trace_count < 64u) {
            s_pbo_trace_count++;
            return true;
        }
    }

    return false;
}

static bool mglMulSizeT(size_t a, size_t b, size_t *out)
{
    if (!out) {
        return false;
    }
    if (a == 0u || b == 0u) {
        *out = 0u;
        return true;
    }
    if (a > (SIZE_MAX / b)) {
        return false;
    }
    *out = a * b;
    return true;
}

static bool mglAddSizeT(size_t a, size_t b, size_t *out)
{
    if (!out) {
        return false;
    }
    if (a > (SIZE_MAX - b)) {
        return false;
    }
    *out = a + b;
    return true;
}

static void mglHandleProxyTexImageQuery(GLMContext ctx,
                                        GLenum target,
                                        GLint level,
                                        GLint internalformat,
                                        GLsizei width,
                                        GLsizei height,
                                        GLsizei depth,
                                        GLint border)
{
    GLuint target_index = textureIndexFromTarget(ctx, target);
    ProxyTextureQueryState *proxy_state = NULL;
    GLsizei maxSize = mglSafeMaxTextureSize(ctx);
    bool require_level_zero = (target == GL_PROXY_TEXTURE_RECTANGLE);
    bool require_square = (target == GL_PROXY_TEXTURE_CUBE_MAP || target == GL_PROXY_TEXTURE_CUBE_MAP_ARRAY);
    bool ok = (level >= 0) &&
              (border == 0) &&
              (width > 0) &&
              (height > 0) &&
              (depth > 0) &&
              (width <= maxSize) &&
              (height <= maxSize) &&
              (depth <= maxSize) &&
              (!require_level_zero || level == 0) &&
              (!require_square || width == height);

    if (target_index >= _MAX_TEXTURE_TYPES) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    proxy_state = &STATE(proxy_texture_query[target_index]);
    proxy_state->width = ok ? width : 0;
    proxy_state->height = ok ? height : 0;
    proxy_state->depth = ok ? depth : 0;
    proxy_state->internalformat = ok ? internalformat : 0;

    if (MGL_VERBOSE_TEXTURE_UPLOAD_LOGS) {
        fprintf(stderr,
                "MGL PROXY TEX query target=0x%x ok=%d req=%dx%dx%d level=%d border=%d max=%d\n",
                target,
                ok ? 1 : 0,
                width,
                height,
                depth,
                level,
                border,
                maxSize);
    }

    // Proxy probe should not leave a GL error behind.
    STATE(error) = GL_NO_ERROR;
}

static bool mglResolveTexSubImageSource(GLMContext ctx,
                                        Texture *tex,
                                        GLuint face,
                                        GLint level,
                                        GLint xoffset,
                                        GLint yoffset,
                                        GLint zoffset,
                                        GLsizei width,
                                        GLsizei height,
                                        GLsizei depth,
                                        GLenum format,
                                        GLenum type,
                                        const void *pixels_raw,
                                        size_t skip_offset_bytes,
                                        size_t required_bytes,
                                        bool trace_upload,
                                        const uint8_t **resolved_src_out,
                                        Buffer **unpack_buf_out)
{
    Buffer *unpack_buf = STATE(buffers[_PIXEL_UNPACK_BUFFER]);
    GLuint unpack_name = unpack_buf ? unpack_buf->name : 0u;
    const char *source_class = unpack_buf ? "PBO" : "CPU";
    const uint8_t *resolved_src = NULL;
    uintptr_t raw_value = (uintptr_t)pixels_raw;
    uint64_t src_hash = 0ull;

    if (unpack_buf) {
        if (unpack_buf->mapped) {
            fprintf(stderr, "MGL ERROR: texSubImage source resolve: unpack buffer %u is mapped\n", unpack_name);
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }

        const uint8_t *pbo_data = (const uint8_t *)getBufferData(ctx, unpack_buf);
        if (!pbo_data) {
            fprintf(stderr, "MGL ERROR: texSubImage source resolve: unpack buffer %u has NULL data\n", unpack_name);
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }

        if (raw_value > unpack_buf->size) {
            fprintf(stderr,
                    "MGL ERROR: texSubImage source resolve: PBO offset overflow unpack=%u off=%" PRIuPTR " size=%lld\n",
                    unpack_name,
                    raw_value,
                    (long long)unpack_buf->size);
            ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
        }

        size_t pbo_size = (size_t)unpack_buf->size;
        size_t raw_off = (size_t)raw_value;
        size_t effective_off = 0u;
        size_t end_off = 0u;

        if (!mglAddSizeT(raw_off, skip_offset_bytes, &effective_off)) {
            fprintf(stderr,
                    "MGL ERROR: texSubImage source resolve: PBO offset addition overflow unpack=%u rawOff=%zu skipOff=%zu\n",
                    unpack_name,
                    raw_off,
                    skip_offset_bytes);
            ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
        }

        if (required_bytes > 0u) {
            if (!mglAddSizeT(effective_off, required_bytes, &end_off)) {
                fprintf(stderr,
                        "MGL ERROR: texSubImage source resolve: PBO required range overflow unpack=%u off=%zu required=%zu\n",
                        unpack_name,
                        effective_off,
                        required_bytes);
                ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
            }
            if (effective_off > pbo_size || end_off > pbo_size) {
                fprintf(stderr,
                        "MGL ERROR: texSubImage source resolve: PBO range overflow unpack=%u rawOff=%zu skipOff=%zu effectiveOff=%zu required=%zu pboSize=%zu\n",
                        unpack_name,
                        raw_off,
                        skip_offset_bytes,
                        effective_off,
                        required_bytes,
                        pbo_size);
                ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
            }
        }

        resolved_src = pbo_data + effective_off;
    } else {
        if (pixels_raw) {
            uintptr_t effective_raw = raw_value + (uintptr_t)skip_offset_bytes;
            if (effective_raw < raw_value) {
                fprintf(stderr,
                        "MGL ERROR: texSubImage source resolve: CPU pointer overflow raw=%p skipOff=%zu\n",
                        pixels_raw,
                        skip_offset_bytes);
                ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
            }
            resolved_src = (const uint8_t *)effective_raw;
        } else {
            resolved_src = NULL;
        }
    }

    if (resolved_src) {
        src_hash = mglHashBytesSampled(resolved_src, required_bytes);
    }

    if (trace_upload) {
        fprintf(stderr,
                "MGL TRACE TexSubImage.source tex=%u target=0x%x face=%u level=%d fmt=0x%x type=0x%x "
                "dims=%dx%dx%d off=(%d,%d,%d) unpackBufferName=%u pixelsRaw=%p resolvedSrcPtr=%p "
                "sourceClass=%s rowLength=%d alignment=%d skipPixels=%d skipRows=%d skipImages=%d skipOffsetBytes=%zu requiredBytes=%zu srcHash=0x%016" PRIx64 "\n",
                tex ? tex->name : 0u,
                tex ? tex->target : 0u,
                face,
                level,
                format,
                type,
                width,
                height,
                depth,
                xoffset,
                yoffset,
                zoffset,
                unpack_name,
                pixels_raw,
                resolved_src,
                source_class,
                ctx->state.unpack.row_length,
                ctx->state.unpack.alignment,
                ctx->state.unpack.skip_pixels,
                ctx->state.unpack.skip_rows,
                ctx->state.unpack.skip_images,
                skip_offset_bytes,
                required_bytes,
                src_hash);
    }

    if (resolved_src) {
        size_t dump_len = required_bytes;
        if (dump_len > 32u) {
            dump_len = 32u;
        }
        if (dump_len == 0u) {
            dump_len = 32u;
        }

        if (trace_upload) {
            mglDumpBytesToStderr("TexSubImage.source.head32", resolved_src, dump_len, 0u);
        }

        size_t zero_probe_len = required_bytes;
        if (zero_probe_len > 256u) {
            zero_probe_len = 256u;
        }
        if (trace_upload && zero_probe_len >= 64u && mglLooksAllZero(resolved_src, zero_probe_len)) {
            fprintf(stderr,
                    "MGL WARNING: TexSubImage source resolved but first %zu bytes are all zero "
                    "(tex=%u target=0x%x unpack=%u raw=%p resolved=%p)\n",
                    zero_probe_len,
                    tex ? tex->name : 0u,
                    tex ? tex->target : 0u,
                    unpack_name,
                    pixels_raw,
                    resolved_src);
        }
    }

    if (resolved_src_out) {
        *resolved_src_out = resolved_src;
    }
    if (unpack_buf_out) {
        *unpack_buf_out = unpack_buf;
    }
    return true;
}

GLuint textureIndexFromTarget(GLMContext ctx, GLenum target)
{
    (void)ctx;

    switch(target)
    {
        case GL_PROXY_TEXTURE_1D:
        case GL_TEXTURE_BUFFER: return _TEXTURE_BUFFER_TARGET;
        case GL_TEXTURE_1D: return _TEXTURE_1D;
        case GL_PROXY_TEXTURE_2D:
        case GL_TEXTURE_2D: return _TEXTURE_2D;
        case GL_PROXY_TEXTURE_3D:
        case GL_TEXTURE_3D: return _TEXTURE_3D;
        case GL_PROXY_TEXTURE_RECTANGLE:
        case GL_TEXTURE_RECTANGLE: return _TEXTURE_RECTANGLE;
        case GL_PROXY_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_1D_ARRAY: return _TEXTURE_1D_ARRAY;
        case GL_PROXY_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_2D_ARRAY: return _TEXTURE_2D_ARRAY;
        case GL_PROXY_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
        case GL_TEXTURE_CUBE_MAP: return _TEXTURE_CUBE_MAP;
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY: return _TEXTURE_CUBE_MAP_ARRAY;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE: return _TEXTURE_2D_MULTISAMPLE;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: return _TEXTURE_2D_MULTISAMPLE_ARRAY;
        case GL_RENDERBUFFER: return _RENDERBUFFER;

        default:
            return _MAX_TEXTURE_TYPES;
    }
}

Texture *currentTexture(GLMContext ctx, GLuint index)
{
    GLuint active_texture;

    active_texture = STATE(active_texture);

    return STATE(texture_units[active_texture].textures[index]);
}

Texture *newTexObj(GLMContext ctx, GLenum target)
{
    Texture *ptr;
    GLuint index;

    index = textureIndexFromTarget(ctx, target);
    if (index == _MAX_TEXTURE_TYPES)
    {
        STATE(error) = GL_INVALID_ENUM;
        return NULL;
    }

    ptr = (Texture *)malloc(sizeof(Texture));
    // CRITICAL SECURITY FIX: Check malloc result instead of using assert()
    if (!ptr) {
        fprintf(stderr, "MGL SECURITY ERROR: Failed to allocate memory for texture\n");
        STATE(error) = GL_OUT_OF_MEMORY;
        return NULL;
    }

    bzero(ptr, sizeof(Texture));

    ptr->name = TEX_OBJ_RES_NAME;
    ptr->target = target;
    ptr->index = index;

    float black_color[] = {0,0,0,0};

    ptr->params.depth_stencil_mode = GL_DEPTH_COMPONENT;
    ptr->params.base_level = 0;
    memcpy(ptr->params.border_color, black_color, 4 * sizeof(float));
    ptr->params.compare_func = GL_NEVER;
    ptr->params.compare_mode = GL_ALWAYS;
    ptr->params.lod_bias = 0.0;
    ptr->params.min_filter = GL_NEAREST;
    ptr->params.mag_filter = GL_NEAREST;
    ptr->params.max_anisotropy = 0.0;
    ptr->params.min_lod = -1000;
    ptr->params.max_lod = 1000;
    ptr->params.max_level = 1000;
    ptr->params.swizzle_r = GL_RED;
    ptr->params.swizzle_g = GL_GREEN;
    ptr->params.swizzle_b = GL_BLUE;
    ptr->params.swizzle_a = GL_ALPHA;
    ptr->params.wrap_s = GL_REPEAT;
    ptr->params.wrap_t = GL_REPEAT;
    ptr->params.wrap_r = GL_REPEAT;

    return ptr;
}

Texture *newTexture(GLMContext ctx, GLenum target, GLuint texture)
{
    Texture *ptr;
    GLuint index;

    if (!ctx || texture == 0)
    {
        if (ctx) {
            STATE(error) = GL_INVALID_VALUE;
        }
        fprintf(stderr,
                "MGL ERROR: newTexture refused invalid name=%u target=0x%x ctx=%p\n",
                texture,
                target,
                (void *)ctx);
        return NULL;
    }

    index = textureIndexFromTarget(ctx, target);
    if (index == _MAX_TEXTURE_TYPES)
    {
        STATE(error) = GL_INVALID_ENUM;
        return NULL;
    }

    ptr = newTexObj(ctx, target);

    ptr->name = texture;

    return ptr;
}

static Texture *getTexture(GLMContext ctx, GLenum target, GLuint texture)
{
    Texture *ptr;

    if (!ctx || texture == 0)
        return NULL;

    ptr = (Texture *)searchHashTable(&STATE(texture_table), texture);

    if (!ptr)
    {
        ptr = newTexture(ctx, target, texture);
        if (!ptr)
            return NULL;

        insertHashElement(&STATE(texture_table), texture, ptr);
    }

    return ptr;
}

static int isTexture(GLMContext ctx, GLuint texture)
{
    Texture *ptr;

    if (!ctx || texture == 0)
        return 0;

    ptr = (Texture *)searchHashTable(&STATE(texture_table), texture);

    if (ptr)
        return 1;

    return 0;
}

Texture *findTexture(GLMContext ctx, GLuint texture)
{
    Texture *ptr;

    if (!ctx || texture == 0)
        return NULL;

    ptr = (Texture *)searchHashTable(&STATE(texture_table), texture);

    return ptr;
}

Texture *getTex(GLMContext ctx, GLuint name, GLenum target)
{
    GLuint index;
    Texture *ptr;

    if (!ctx) {
        return NULL;
    }

    if (name == 0)
    {
        index = textureIndexFromTarget(ctx, target);
        if (index == _MAX_TEXTURE_TYPES)
        {
            STATE(error) = GL_INVALID_ENUM;
            return NULL;
        }

        ptr = currentTexture(ctx, index);
        
        // Create default texture if none exists for this target
        if (!ptr) {
            GLuint active_texture = STATE(active_texture);
            ptr = newTexObj(ctx, target);
            if (!ptr) {
                fprintf(stderr,
                        "MGL ERROR: getTex failed to create default texture target=0x%x activeUnit=%u\n",
                        target,
                        active_texture);
                return NULL;
            }
            STATE(texture_units[active_texture].textures[index]) = ptr;
            fprintf(stderr, "MGL: Created default texture for target 0x%x\n", target);
        }
    }
    else
    {
        ptr = findTexture(ctx, name);
        if (!ptr) {
            fprintf(stderr,
                    "MGL ERROR: getTex failed to resolve texture name=%u target=0x%x\n",
                    name,
                    target);
            STATE(error) = GL_INVALID_OPERATION;
            return NULL;
        }
        
        target = ptr->target;

        index = textureIndexFromTarget(ctx, target);
        if (index == _MAX_TEXTURE_TYPES)
        {
            STATE(error) = GL_INVALID_ENUM;
            return NULL;
        }
    }

    return ptr;
}

bool checkInternalFormatForMetal(GLMContext ctx, GLuint internalformat)
{
    // see if we can actually use this internal format
    GLenum mtl_format;
    mtl_format = mtlFormatForGLInternalFormat(internalformat);

    if (mtl_format == MTLPixelFormatInvalid)
    {
        // Only warn once per format to reduce log spam during capability probing
        static unsigned warned_formats[64] = {0};
        static int warned_count = 0;
        int already_warned = 0;
        for (int i = 0; i < warned_count && i < 64; i++) {
            if (warned_formats[i] == internalformat) { already_warned = 1; break; }
        }
        if (!already_warned && warned_count < 64) {
            warned_formats[warned_count++] = internalformat;
            // Only warn for standard GL format ranges (not internal Mesa/Gallium enums)
            // Skip 0x2xxx (GL get parameters), 0x8Dxx-0x9xxx (internal enums)
            if (internalformat >= 0x8040 && internalformat < 0x8D70) {
                fprintf(stderr, "MGL: checkInternalFormatForMetal - internalformat 0x%x has no Metal equivalent\n", internalformat);
            }
        }
        return false;
    }

    return true;
}


#pragma mark basic tex calls bind / delete / gen...
void mglGenTextures(GLMContext ctx, GLsizei n, GLuint *textures)
{
    static uint64_t s_gen_textures_calls = 0u;
    uint64_t call_id = ++s_gen_textures_calls;

    if (!ctx || n <= 0 || !textures) {
        fprintf(stderr,
                "MGL TRACE GenTextures.skip call=%llu ctx=%p n=%d textures=%p\n",
                (unsigned long long)call_id,
                (void *)ctx,
                (int)n,
                (void *)textures);
        return;
    }

    assert(textures);

    while(n--)
    {
        GLuint name = getNewName(&STATE(texture_table));
        *textures++ = name;
        fprintf(stderr,
                "MGL TRACE GenTextures call=%llu generated=%u currentName=%u tableCount=%zu tableCap=%zu\n",
                (unsigned long long)call_id,
                name,
                STATE(texture_table).current_name,
                STATE(texture_table).count,
                STATE(texture_table).size);

        // TEX_OBJ_RES_NAME has special name.. skip it
        if (STATE(texture_table.current_name) == TEX_OBJ_RES_NAME)
            getNewName(&STATE(texture_table));
    }
}

void mglCreateTextures(GLMContext ctx, GLenum target, GLsizei n, GLuint *textures)
{
    mglGenTextures(ctx, n, textures);

    while(n--)
    {
        // create a texture object
        GLuint name = *textures++;
        if (!getTexture(ctx, target, name))
        {
            fprintf(stderr, "MGL Error: mglCreateTextures: failed to create texture %u for target 0x%x\n",
                    (unsigned)name, (unsigned)target);
            STATE(error) = GL_INVALID_ENUM;
            return;
        }
    }
}

void mglBindTexture(GLMContext ctx, GLenum target, GLuint texture)
{
    GLuint active_texture;
    GLint index;
    Texture *ptr;

    if (MGL_VERBOSE_TEXTURE_BIND_LOGS) {
        fprintf(stderr,
                "MGL TRACE BindTexture target=0x%x texture=%u activeUnit=%u ctx=%p\n",
                target,
                texture,
                ctx ? ctx->state.active_texture : 0u,
                (void *)ctx);
    }

    index = textureIndexFromTarget(ctx, target);
    if (index == _MAX_TEXTURE_TYPES)
    {
        fprintf(stderr, "MGL Error: mglBindTexture: invalid target 0x%x\n", (unsigned)target);
        STATE(error) = GL_INVALID_ENUM;
        return;
    }

    if (texture)
    {
        ptr = getTexture(ctx, target, texture);
        if (!ptr) {
            fprintf(stderr,
                    "MGL Error: mglBindTexture failed to resolve/create texture=%u target=0x%x\n",
                    texture,
                    target);
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }
    }
    else
    {
        ptr = NULL;
    }

    active_texture = STATE(active_texture);

    GLuint mask_index = active_texture / 32;
    GLuint mask = (0x1 << active_texture);

    if (ptr)
    {
        STATE(active_texture_mask[mask_index]) |= mask;
    }
    else
    {
        STATE(active_texture_mask[mask_index]) &= ~mask;
    }

    STATE(active_textures[active_texture]) = ptr;
    STATE(texture_units[active_texture].textures[index]) = ptr;
    STATE(dirty_bits) |= DIRTY_TEX;
}

void mglBindImageTexture(GLMContext ctx, GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum internalformat)
{
    Texture *ptr;

    // ERROR_CHECK_RETURN(unit < TEXTURE_UNITS, GL_INVALID_VALUE);
    if (unit >= TEXTURE_UNITS) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: unit >= TEXTURE_UNITS (%d)\n", unit);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    ptr = getTex(ctx, texture, 0);

    // ERROR_CHECK_RETURN(ptr, GL_INVALID_VALUE);
    if (!ptr) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: texture %d not found\n", texture);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    // ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);
    if (level < 0) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: level < 0 (%d)\n", level);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    // ERROR_CHECK_RETURN(layered >= 0, GL_INVALID_VALUE);
    if (layered < 0) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: layered < 0 (%d)\n", layered);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    switch(access)
    {
        case GL_READ_ONLY:
        case GL_WRITE_ONLY:
        case GL_READ_WRITE:
            break;

        default:
            fprintf(stderr, "MGL Error: mglBindImageTexture: invalid access 0x%x\n", access);
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    // ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_ENUM);
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: invalid internalformat 0x%x\n", internalformat);
        ERROR_RETURN(GL_INVALID_ENUM);
    }

    // ERROR_CHECK_RETURN(ptr->internalformat == internalformat, GL_INVALID_VALUE);
    if (ptr->internalformat != internalformat) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: internalformat mismatch (tex=0x%x req=0x%x)\n", ptr->internalformat, internalformat);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    // ERROR_CHECK_RETURN(level < ptr->num_levels, GL_INVALID_VALUE);
    if (level >= ptr->num_levels) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: level >= num_levels (%d >= %d)\n", level, ptr->num_levels);
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    
    ImageUnit unit_params;

    if (ptr->access != access)
    {
        ptr->dirty_bits |= DIRTY_TEXTURE_ACCESS;
        ptr->access = access;
    }

    unit_params.texture = texture;
    unit_params.level = level;
    unit_params.layered = layered;
    unit_params.layer = layer;
    unit_params.access = access;
    unit_params.internalformat = internalformat;
    unit_params.tex = ptr;

    ctx->state.image_units[unit] = unit_params;

    ctx->state.dirty_bits |= DIRTY_IMAGE_UNIT_STATE;
}

void mglDeleteTextures(GLMContext ctx, GLsizei n, const GLuint *textures)
{
    if (!ctx || n <= 0 || !textures)
        return;

    while(n--)
    {
        GLuint name;

        name = *textures++;
        if (name == 0)
            continue;

        Texture *tex;

        tex = findTexture(ctx, name);

        if(tex)
        {
            for(int i=0; i<TEXTURE_UNITS; i++)
            {
                if(ctx->state.active_textures[i] == tex)
                {
                    ctx->state.active_textures[i] = NULL;
                    ctx->state.texture_units[i].textures[tex->index] = NULL;
                    ctx->state.active_texture_mask[i / 32] &= ~(1u << (i % 32));

                    ctx->state.dirty_bits |= DIRTY_TEX_BINDING;
                }
            }

            for(int i=0; i<TEXTURE_UNITS; i++)
            {
                if(ctx->state.image_units[i].texture == name)
                {
                    bzero(&ctx->state.image_units[i], sizeof(ImageUnit));

                    ctx->state.dirty_bits |= DIRTY_IMAGE_UNIT_STATE;
                }
            }

            invalidateTexture(ctx, tex);
            deleteHashElement(&STATE(texture_table), name);
            free(tex);
        }
    }
}

GLboolean mglIsTexture(GLMContext ctx, GLuint texture)
{
    return isTexture(ctx, texture);
}

void mglInvalidateTexImage(GLMContext ctx, GLuint texture, GLint level)
{
    // Stub - invalidation is just a hint, safe to ignore
    fprintf(stderr, "MGL: glInvalidateTexImage called (stub) - texture=%u level=%d\n", texture, level);
}

void mglInvalidateTexSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth)
{
    // Stub - invalidation is just a hint, safe to ignore
    fprintf(stderr, "MGL: glInvalidateTexSubImage called (stub)\n");
}

void mglBindImageTextures(GLMContext ctx, GLuint first, GLsizei count, const GLuint *textures)
{
    fprintf(stderr, "MGL: glBindImageTextures called first=%u count=%d\n", first, count);
    // Bind multiple image textures
    for (GLsizei i = 0; i < count; i++) {
        GLuint tex_name = textures ? textures[i] : 0;
        if (tex_name != 0) {
            Texture *tex = findTexture(ctx, tex_name);
            if (tex) {
                mglBindImageTexture(ctx, first + i, tex_name, 0, GL_FALSE, 0, tex->access ? tex->access : GL_READ_ONLY, tex->internalformat);
            }
        }
    }
}

void mglClientActiveTexture(GLMContext ctx, GLenum texture)
{
    // Legacy function for fixed-function pipeline, safe to ignore in modern GL
    fprintf(stderr, "MGL: glClientActiveTexture called (stub) - legacy function\n");
}

void mglActiveTexture(GLMContext ctx, GLenum texture)
{
    GLuint unit;

    if (texture < GL_TEXTURE0)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
    }

    unit = (GLuint)(texture - GL_TEXTURE0);

    if (unit >= TEXTURE_UNITS || unit >= STATE_VAR(max_combined_texture_image_units))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
    }

    STATE(active_texture) = unit;
    ctx->state.dirty_bits |= DIRTY_TEX_BINDING;
}

void mglBindTextures(GLMContext ctx, GLuint first, GLsizei count, const GLuint *textures)
{
    for (int i=0; i < count; i++)
    {
        GLuint texture;

        if (textures == NULL)
        {
            texture = 0;
        }
        else
        {
            texture = textures[i];
        }

        STATE(active_texture) = first + i - GL_TEXTURE0;

        if (texture != 0)
        {
            Texture *ptr;
            GLenum target;

            ptr = findTexture(ctx, textures[i]);
            assert(ptr);

            target = ptr->target;

            mglBindTexture(ctx, target, textures[i]);
        }
        else
        {
            for(GLenum target=0; target<_MAX_TEXTURE_TYPES; target++)
            {
                mglBindTexture(ctx, target, 0);
            }
        }
    }

}

void mglBindTextureUnit(GLMContext ctx, GLuint unit, GLuint texture)
{
    // bind texture without changing the current active unit
    // my guess

    Texture *ptr;
    GLenum target;

    ptr = findTexture(ctx, texture);
    assert(ptr);

    unit = unit - GL_TEXTURE0;
    assert(unit < TEXTURE_UNITS);

    target = ptr->target;

    STATE(texture_units[unit].textures[target]) = ptr;
}

void generateMipmaps(GLMContext ctx, GLuint texture, GLenum target)
{
    Texture *ptr;

    ptr = getTex(ctx, texture, target);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    // level 0 needs to be filled out for mipmap geneation
    ERROR_CHECK_RETURN(ptr->faces[0].levels[0].complete, GL_INVALID_OPERATION);

    ptr->mipmapped = true;
    ptr->genmipmaps = true;

    ptr->dirty_bits |= DIRTY_TEXTURE_LEVEL;

    ctx->mtl_funcs.mtlGenerateMipmaps(ctx, ptr);
}

void mglGenerateMipmap(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    generateMipmaps(ctx, 0, target);
}

void mglGenerateTextureMipmap(GLMContext ctx, GLuint texture)
{
    generateMipmaps(ctx, texture, 0);
}

static size_t page_size_align(size_t size)
{
    if (size & (4096-1))
    {
        size_t pad_size = 0;

        pad_size = 4096 - (size & (4096-1));

        size += pad_size;
    }

    return size;
}

void invalidateTexture(GLMContext ctx, Texture *tex)
{
    if (tex->mtl_data)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, tex->mtl_data);
    }

    for(int face=0; face<_CUBE_MAP_MAX_FACE; face++)
    {
        for(int i=0; i<tex->num_levels; i++)
        {
            if (tex->faces[face].levels[i].complete)
            {
                if (tex->faces[face].levels[i].data)
                {
                    vm_deallocate(mach_host_self(), tex->faces[face].levels[i].data, tex->faces[face].levels[i].data);
                }
            }
        }
    }

    for(int i=0; i<6; i++)
    {
        if (tex->faces[i].levels)
            free(tex->faces[i].levels);
    }

    bzero(tex, sizeof(Texture));
}

void initBaseTexLevel(GLMContext ctx, Texture *tex, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    tex->mipmapped = 0;
    tex->mipmap_levels = ilog2(MAX(width, height)) + 1;

    for(int face=0; face<_CUBE_MAP_MAX_FACE; face++)
    {
        // CRITICAL SECURITY FIX: Prevent integer overflow in mipmap allocation
        if (tex->mipmap_levels > SIZE_MAX / sizeof(TextureLevel)) {
            fprintf(stderr, "MGL SECURITY ERROR: Mipmap levels %d would cause allocation overflow\n", tex->mipmap_levels);
            // CRITICAL FIX: Handle gracefully instead of crashing
            STATE(error) = GL_OUT_OF_MEMORY;
            return;
        }

        tex->faces[face].levels = (TextureLevel *)calloc(tex->mipmap_levels, sizeof(TextureLevel));
        if (!tex->faces[face].levels) {
            fprintf(stderr, "MGL SECURITY ERROR: calloc failed for face %d with %d levels\n", face, tex->mipmap_levels);
            // CRITICAL FIX: Handle gracefully instead of crashing
            STATE(error) = GL_OUT_OF_MEMORY;
            return;
        }
    }

    tex->internalformat = internalformat;
    tex->width = width;
    tex->height = height;
    tex->depth = depth;
    tex->complete = false;

    for(int face=0; face<_CUBE_MAP_MAX_FACE; face++)
    {
        for(int i=0; i<tex->mipmap_levels; i++)
        {
            tex->faces[face].levels[i].complete = false;
            tex->faces[face].levels[i].has_initialized_data = GL_FALSE;
            tex->faces[face].levels[i].ever_written = GL_FALSE;
            tex->faces[face].levels[i].suspicious_zero_upload = GL_FALSE;
        }
    }
}

bool checkTexLevelParams(GLMContext ctx, Texture *tex, GLint level, GLuint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type)
{
    GLuint base_width, base_height;

    if (level >= tex->mipmap_levels)
    {
        return false;
    }

    if (tex->target == GL_TEXTURE_2D)
    {
        if (level != 0)
        {
            base_width = tex->width;
            base_height = tex->height;

            while(level--)
            {
                base_width >>= 1;
                base_height >>= 1;
            }

            if (width != base_width || height != base_height)
            {
                return false;
            }
        }
    }

    if (internalformat)
    {
        // internal formats don't jive
        if (internalformat != tex->internalformat)
        {
            return false;
        }
    }
    else
    {
        GLuint temp_internalformat;

        // check if we are expected to convert data
        temp_internalformat = internalFormatForGLFormatType(format, type);

        if (temp_internalformat != tex->internalformat)
        {
            return false;
        }
    }

    if (checkInternalFormatForMetal(ctx, internalformat))
    {
        return false;
    }

    return true;
}


bool verifyInternalFormatAndFormatType(GLMContext ctx, GLint internalformat, GLenum format, GLenum type)
{
    switch(internalformat)
    {
        // unsized formats
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        case GL_RED:
        case GL_RG:
        case GL_RGB:
        case GL_RGBA:
            break;

        // sized formats
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_R3_G3_B2:
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB8:
        case GL_RGB8_SNORM:
        case GL_RGB10:
        case GL_RGB12:
        case GL_RGB16_SNORM:
        case GL_RGBA2:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_RGBA8:
        case GL_RGBA8_SNORM:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_RGBA12:
        case GL_RGBA16:
        case GL_SRGB8:
        case GL_SRGB8_ALPHA8:
        case GL_R16F:
        case GL_RG16F:
        case GL_RGB16F:
        case GL_RGBA16F:
        case GL_R32F:
        case GL_RG32F:
        case GL_RGB32F:
        case GL_RGBA32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA32I:
        case GL_RGBA32UI:
        // Missing SNORM/UI formats used by virgl
        case 0x9014: // GL_ALPHA8_SNORM
        case 0x9016: // GL_LUMINANCE8_ALPHA8_SNORM
        case 0x9018: // GL_ALPHA16_SNORM
        case 0x901a: // GL_LUMINANCE16_ALPHA16_SNORM
        case 0x8d7e: // GL_ALPHA8UI_EXT
            break;

        // compressed types
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
            break;

        // Legacy alpha/luminance formats (deprecated but still used)
        case 0x803c: // GL_ALPHA8
        case 0x803e: // GL_ALPHA16
        case 0x8040: // GL_LUMINANCE8
        case 0x8042: // GL_LUMINANCE16
        case 0x8045: // GL_LUMINANCE8_ALPHA8
        case 0x8048: // GL_LUMINANCE16_ALPHA16
        case 0x8816: // GL_ALPHA16F_ARB
        case 0x8818: // GL_LUMINANCE16F_ARB
        case 0x8819: // GL_LUMINANCE_ALPHA16F_ARB
        case 0x881c: // GL_ALPHA32F_ARB
        case 0x881e: // GL_LUMINANCE32F_ARB
        case 0x881f: // GL_LUMINANCE_ALPHA32F_ARB
            break;

        // ASTC compressed formats
        case 0x93b0: // GL_COMPRESSED_RGBA_ASTC_4x4_KHR
        case 0x93b1: // GL_COMPRESSED_RGBA_ASTC_5x4_KHR
        case 0x93b2: // GL_COMPRESSED_RGBA_ASTC_5x5_KHR
        case 0x93b3: // GL_COMPRESSED_RGBA_ASTC_6x5_KHR
        case 0x93b4: // GL_COMPRESSED_RGBA_ASTC_6x6_KHR
        case 0x93b5: // GL_COMPRESSED_RGBA_ASTC_8x5_KHR
        case 0x93b6: // GL_COMPRESSED_RGBA_ASTC_8x6_KHR
        case 0x93b7: // GL_COMPRESSED_RGBA_ASTC_8x8_KHR
        case 0x93b8: // GL_COMPRESSED_RGBA_ASTC_10x5_KHR
        case 0x93b9: // GL_COMPRESSED_RGBA_ASTC_10x6_KHR
        case 0x93ba: // GL_COMPRESSED_RGBA_ASTC_10x8_KHR
        case 0x93bb: // GL_COMPRESSED_RGBA_ASTC_10x10_KHR
        case 0x93bc: // GL_COMPRESSED_RGBA_ASTC_12x10_KHR
        case 0x93bd: // GL_COMPRESSED_RGBA_ASTC_12x12_KHR
        case 0x93d0: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR
        case 0x93d1: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR
        case 0x93d2: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR
        case 0x93d3: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR
        case 0x93d4: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR
        case 0x93d5: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR
        case 0x93d6: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR
        case 0x93d7: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR
        case 0x93d8: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR
        case 0x93d9: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR
        case 0x93da: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR
        case 0x93db: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR
        case 0x93dc: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR
        case 0x93dd: // GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR
            break;

        // ETC2/EAC compressed formats
        case 0x9270: // GL_COMPRESSED_R11_EAC
        case 0x9271: // GL_COMPRESSED_SIGNED_R11_EAC
        case 0x9272: // GL_COMPRESSED_RG11_EAC
        case 0x9273: // GL_COMPRESSED_SIGNED_RG11_EAC
        case 0x9274: // GL_COMPRESSED_RGB8_ETC2
        case 0x9275: // GL_COMPRESSED_SRGB8_ETC2
        case 0x9276: // GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2
        case 0x9277: // GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2
        case 0x9278: // GL_COMPRESSED_RGBA8_ETC2_EAC
        case 0x9279: // GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC
            break;

        // S3TC/DXT compressed formats
        case 0x83f0: // GL_COMPRESSED_RGB_S3TC_DXT1_EXT
        case 0x83f1: // GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
        case 0x83f2: // GL_COMPRESSED_RGBA_S3TC_DXT3_EXT
        case 0x83f3: // GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
        case 0x8c4c: // GL_COMPRESSED_SRGB_S3TC_DXT1_EXT
        case 0x8c4d: // GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT
        case 0x8c4e: // GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT
        case 0x8c4f: // GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT
            break;

        // Additional integer formats (alternate enum values used by some implementations)
        case 0x8d72: // alternate GL_RGBA8I
        case 0x8d75: // alternate GL_RGB8I
        case 0x8d78: // alternate GL_RGBA8UI
        case 0x8d7a: // alternate GL_RGB8UI
        case 0x8d7b: // GL_ALPHA8I_EXT
        // case 0x8d7e: // alternate GL_RGBA32UI - Duplicate of GL_ALPHA8UI_EXT
        case 0x8d80: // alternate GL_RGB32UI
        case 0x8d81: // GL_ALPHA32I_EXT
        case 0x8d84: // alternate GL_RGBA16I
        case 0x8d86: // alternate GL_RGB16I
        case 0x8d87: // GL_ALPHA16I_EXT
        case 0x8d8a: // alternate GL_RGBA32I
        case 0x8d8c: // alternate GL_RGB32I
        case 0x8d8d: // GL_ALPHA32I_EXT
        case 0x8d90: // alternate GL_RGBA16UI
        case 0x8d92: // alternate GL_RGB16UI
        case 0x8d93: // GL_ALPHA16UI_EXT
            break;

        // SNORM formats
        case 0x8f9b: // GL_SIGNED_NORMALIZED
        case 0x8fbd: // GL_RGB10_A2UI (alternate)
        case 0x8fbe: // GL_RGBA16_SNORM
            break;

        // Depth/stencil special formats
        // case 0x9014: // GL_DEPTH_COMPONENT16_NONLINEAR_NV - Duplicate of GL_ALPHA8_SNORM
        // case 0x9016: // GL_TEXTURE_2D_MULTISAMPLE - Duplicate of GL_LUMINANCE8_ALPHA8_SNORM
        // case 0x9018: // GL_TEXTURE_2D_MULTISAMPLE_ARRAY - Duplicate of GL_ALPHA16_SNORM
        // case 0x901a: // GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY - Duplicate of GL_LUMINANCE16_ALPHA16_SNORM
            break;

        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            ERROR_CHECK_RETURN_VALUE(format == GL_DEPTH_COMPONENT, GL_INVALID_OPERATION, false);
            break;
            
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            ERROR_CHECK_RETURN_VALUE(format == GL_DEPTH_STENCIL, GL_INVALID_OPERATION, false);
            break;
            
        case GL_STENCIL_INDEX8:
            ERROR_CHECK_RETURN_VALUE(format == GL_STENCIL_INDEX, GL_INVALID_OPERATION, false);
            break;
            
        case GL_RGB565:
            break;

        default:
            // Log warning but don't error - many formats work even if not explicitly listed
            fprintf(stderr, "MGL WARNING: verifyInternalFormat unknown internalformat 0x%x\n", internalformat);
            break;
    }

    switch(format)
    {
        case GL_RED:
        case GL_RG:
        case GL_RGB:
        case GL_BGR:
        case GL_RGBA:
        case GL_BGRA:
        case GL_RED_INTEGER:
        case GL_RG_INTEGER:
        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
        case GL_STENCIL_INDEX:
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        // Legacy formats (deprecated but still used by virglrenderer)
        case 0x1906: // GL_ALPHA
        case 0x1909: // GL_LUMINANCE
        case 0x190a: // GL_LUMINANCE_ALPHA
        case 0x8000: // GL_COLOR_INDEX (legacy)
        case 0x8d97: // GL_ALPHA_INTEGER
        case 0x8d9c: // GL_LUMINANCE_INTEGER_EXT
        case 0x8d9d: // GL_LUMINANCE_ALPHA_INTEGER_EXT
            break;

        default:
            // Allow unknown formats with warning - virglrenderer may use nonstandard values
            fprintf(stderr, "MGL WARNING: verifyFormat unknown format 0x%x, allowing\n", format);
            break;
    }

    switch(type)
    {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
            break;

        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
            ERROR_CHECK_RETURN_VALUE(format == GL_RGB,GL_INVALID_OPERATION, false);
            break;

        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            // Allow RGBA, BGRA, and integer variants (RGBA_INTEGER, RGB_INTEGER, BGRA_INTEGER)
            ERROR_CHECK_RETURN_VALUE((format == GL_RGBA || format == GL_BGRA || 
                                      format == GL_RGBA_INTEGER || format == GL_RGB_INTEGER ||
                                      format == GL_BGRA_INTEGER), GL_INVALID_OPERATION, false);
            break;
            
        case GL_UNSIGNED_INT_24_8:
            ERROR_CHECK_RETURN_VALUE(format == GL_DEPTH_STENCIL, GL_INVALID_OPERATION, false);
            break;
            
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            ERROR_CHECK_RETURN_VALUE(format == GL_DEPTH_STENCIL, GL_INVALID_OPERATION, false);
            break;
        
        // Packed float types for special formats
        case 0x8c3b: // GL_UNSIGNED_INT_10F_11F_11F_REV
        case 0x8c3e: // GL_UNSIGNED_INT_5_9_9_9_REV
            break;
            
        default:
            fprintf(stderr, "MGL WARNING: verifyInternalFormat unknown type 0x%x\n", type);
            break;
    }

    return true;
}


void unpackTexture(GLMContext ctx, Texture *tex, GLuint face, GLuint level, void *src_data, void *dst_data, size_t src_pitch, size_t pixel_size, size_t xoffset, size_t yoffset, size_t zoffset, size_t width, size_t height, size_t depth)
{
    GLubyte *src, *dst;
    size_t dst_pitch;

    src = (GLubyte *)src_data;
    dst = (GLubyte *)dst_data;

    assert(tex);
    dst_pitch = tex->faces[face].levels[level].pitch;
    assert(dst_pitch);

    if (xoffset || yoffset || zoffset)
    {
        size_t xoffset_bytes = xoffset * pixel_size; // num pixels
        size_t yoffset_bytes = yoffset * dst_pitch; // num lines (rows * bytes_per_row)
        size_t zoffset_bytes = zoffset * dst_pitch * height; // num planes

        dst += xoffset_bytes;
        dst += yoffset_bytes;
        dst += zoffset_bytes;
    }

    if (depth > 1)
    {
        // 3d texture
        for(int y=0; y<depth; y++)
        {
            memcpy(dst, src, dst_pitch);
            src += src_pitch;
            dst += dst_pitch;
        }
    }
    else if (height > 1)
    {
        // 2d texture
        size_t copy_size = width * pixel_size;
        
        for(int y=0; y<height; y++)
        {
            memcpy(dst, src, copy_size);
            src += src_pitch;
            dst += dst_pitch;
        }
    }
    else
    {
        // 1d texture
        memcpy(dst, src, width * pixel_size);
    }
}

#pragma mark texImage 1D/2D/3D
// Forward declaration
bool texSubImage(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels);

bool createTextureLevel(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLboolean is_array, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels, GLboolean proxy)
{
    // all the levels are created on a tex storage call.. if we get here we should just assert
    if (tex->immutable_storage)
    {
        // Compatibility: Treat glTexImage* on immutable texture as glTexSubImage*
        // This allows guests to update content using glTexImage* which is common in some drivers
        // We pass 0 for offsets. texSubImage will handle validation.
        if (pixels == NULL) {
            // Allocation-only call against immutable storage: nothing to upload.
            return true;
        }
        return texSubImage(ctx, tex, face, level, 0, 0, 0, width, height, depth, format, type, pixels);
    }
    
    if (level == 0)
    {
        if (internalformat == 0)
        {
            internalformat = internalFormatForGLFormatType(format, type);

            if (internalformat == 0)
            {
                ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
            }
        }
        else if (pixels)
        {
            GLuint temp_format;

            // check if format type can be copied directly to the internal format
            temp_format = internalFormatForGLFormatType(format, type);

            // MGL doesn't support pixel format conversion
            // If mismatch, use the format that matches the incoming data
            if (temp_format != internalformat)
            {
                internalformat = temp_format;
            }
        }

        // see if we can actually use this internal format
        if (checkInternalFormatForMetal(ctx, internalformat) == false)
        {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }

        if (tex->mipmap_levels == 0)
        {
            // uninitialized tex
            initBaseTexLevel(ctx, tex, internalformat, width, height, depth);
        }
        else if (width != tex->width || height != tex->height || internalformat != tex->internalformat)
        {
            // invalidate texture because the base level width / height / internal format are being changed...
            invalidateTexture(ctx, tex);

            initBaseTexLevel(ctx, tex, internalformat, width, height, depth);
        }
    }
    else if (checkTexLevelParams(ctx, tex, level, internalformat, width, height, depth, format, type) == false)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    tex->num_levels = MAX(tex->num_levels, level + 1);
    tex->faces[face].levels[level].width = width;
    tex->faces[face].levels[level].height = height;
    tex->faces[face].levels[level].depth = depth;

    // Proxy textures are capability probes: validate and store metadata only.
    // Do not allocate backing storage or upload data.
    if (proxy)
    {
        tex->is_array = is_array;
        tex->faces[face].levels[level].pitch = 0;
        tex->faces[face].levels[level].data_size = 0;
        tex->faces[face].levels[level].data = 0;
        tex->faces[face].levels[level].has_initialized_data = GL_FALSE;
        tex->faces[face].levels[level].ever_written = GL_FALSE;
        tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;
        tex->faces[face].levels[level].last_init_source = kTexInitNone;
        tex->faces[face].levels[level].last_upload_size = 0u;
        tex->faces[face].levels[level].last_src_ptr = NULL;
        tex->faces[face].levels[level].last_src_hash = 0ull;
        tex->faces[face].levels[level].complete = true;
        tex->complete = true;
        return true;
    }

    kern_return_t err;
    vm_address_t texture_data;
    size_t pixel_size;
    size_t internal_size;
    size_t texture_size;
    size_t src_pitch;

    pixel_size = sizeForInternalFormat(internalformat, format, type);
    ERROR_CHECK_RETURN_VALUE(pixel_size, GL_INVALID_ENUM, false);

    assert(width);
    assert(height);
    assert(depth);

    tex->faces[face].levels[level].pitch = pixel_size * width;

    if (depth > 1)
    {
        // 3d texture
        internal_size = pixel_size * width * height * depth;
    }
    else if (height > 1)
    {
        // 2d texture
        internal_size = pixel_size * width * height;
    }
    else
    {
        // 1d texture
        internal_size = pixel_size * width;
    }

    texture_size = page_size_align(internal_size);
    assert(texture_size);

    switch(mtlFormatForGLInternalFormat(internalformat))
    {
        case MTLPixelFormatDepth16Unorm:
        case MTLPixelFormatDepth32Float:
        case MTLPixelFormatDepth24Unorm_Stencil8:
        case MTLPixelFormatDepth32Float_Stencil8:
            tex->mtl_requires_private_storage = true;
            break;

        default:
            tex->mtl_requires_private_storage = false;
            break;
    }

    if (tex->mtl_requires_private_storage == false)
    {
        // Allocate directly from VM
        err = vm_allocate((vm_map_t) mach_task_self(),
                          (vm_address_t*) &texture_data,
                          texture_size,
                          VM_FLAGS_ANYWHERE);
        assert(err == 0);
        assert(texture_data);

        tex->faces[face].levels[level].data_size = texture_size;
        tex->faces[face].levels[level].data = (vm_address_t)texture_data;

        if (pixels)
        {
            GLsizei src_size;

            src_size = width * sizeForFormatType(format, type);

            if (ctx->state.unpack.row_length)
            {
                size_t alignment;

                if (ctx->state.unpack.row_length < width) {
                    ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
                }

                alignment = ctx->state.unpack.alignment;
                if (alignment)
                {
                    /* row_length is in pixels, so multiply by pixel_size to get bytes per row */
                    size_t row_bytes = ctx->state.unpack.row_length * pixel_size;
                    if (row_bytes >= alignment)
                    {
                        src_pitch = row_bytes;
                        assert(src_pitch);
                    }
                    else if (depth > 1)
                    {
                        // 3d texture
                        src_pitch = alignment / src_size;

                        src_pitch = src_pitch * src_size * ctx->state.unpack.row_length * height;

                        src_pitch = src_pitch / alignment;
                        assert(src_pitch);
                    }
                    else
                    {
                        src_pitch = alignment / src_size;

                        src_pitch = src_pitch * src_size * ctx->state.unpack.row_length;

                        src_pitch = src_pitch / alignment;
                        assert(src_pitch);
                    }
                }
                else
                {
                    src_pitch = src_size * ctx->state.unpack.row_length;
                    assert(src_pitch);
                }
            }
            else
            {
                src_pitch = tex->faces[face].levels[level].pitch;
                assert(src_pitch);
            }

            // unpack from pixel buffer
            if (STATE(buffers[_PIXEL_UNPACK_BUFFER]))
            {
                Buffer *ptr;
                size_t offset;
                size_t pbo_row_copy_bytes = 0u;
                size_t pbo_row_tail_bytes = 0u;
                size_t pbo_image_pitch_bytes = 0u;
                size_t pbo_slice_tail_bytes = 0u;
                size_t pbo_required_bytes = 0u;

                ptr = STATE(buffers[_PIXEL_UNPACK_BUFFER]);

                ERROR_CHECK_RETURN(ptr->mapped == false, GL_INVALID_OPERATION);

                GLubyte *buffer_data;
                buffer_data = getBufferData(ctx, ptr);
                if (!buffer_data) {
                    fprintf(stderr, "MGL ERROR: createTextureLevel unpack buffer has NULL data\n");
                    ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
                }

                // if a pixel buffer is the src, pixels is the offset
                offset = (size_t)pixels;
                if (offset > (size_t)ptr->size) {
                    fprintf(stderr,
                            "MGL ERROR: createTextureLevel unpack buffer offset overflow tex=%u face=%u level=%d "
                            "offset=%zu bufferSize=%zu\n",
                            tex->name,
                            face,
                            level,
                            offset,
                            (size_t)ptr->size);
                    ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
                }

                if (!mglMulSizeT((size_t)MAX(width, 1), pixel_size, &pbo_row_copy_bytes)) {
                    fprintf(stderr,
                            "MGL ERROR: createTextureLevel unpack buffer size computation overflow tex=%u face=%u level=%d "
                            "rowCopy width=%d pixelSize=%zu\n",
                            tex->name,
                            face,
                            level,
                            width,
                            pixel_size);
                    ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
                }

                pbo_required_bytes = pbo_row_copy_bytes;

                if (height > 1) {
                    if (!mglMulSizeT(src_pitch, (size_t)(height - 1), &pbo_row_tail_bytes) ||
                        !mglAddSizeT(pbo_required_bytes, pbo_row_tail_bytes, &pbo_required_bytes)) {
                        fprintf(stderr,
                                "MGL ERROR: createTextureLevel unpack row-tail overflow tex=%u face=%u level=%d "
                                "srcPitch=%zu height=%d\n",
                                tex->name,
                                face,
                                level,
                                src_pitch,
                                height);
                        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
                    }
                }

                if (depth > 1) {
                    if (!mglMulSizeT(src_pitch, (size_t)MAX(height, 1), &pbo_image_pitch_bytes) ||
                        !mglMulSizeT(pbo_image_pitch_bytes, (size_t)(depth - 1), &pbo_slice_tail_bytes) ||
                        !mglAddSizeT(pbo_required_bytes, pbo_slice_tail_bytes, &pbo_required_bytes)) {
                        fprintf(stderr,
                                "MGL ERROR: createTextureLevel unpack slice-tail overflow tex=%u face=%u level=%d "
                                "srcPitch=%zu dims=%dx%dx%d\n",
                                tex->name,
                                face,
                                level,
                                src_pitch,
                                width,
                                height,
                                depth);
                        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
                    }
                }

                if (pbo_required_bytes > 0u) {
                    size_t end_off = 0u;
                    if (!mglAddSizeT(offset, pbo_required_bytes, &end_off) || end_off > (size_t)ptr->size) {
                        fprintf(stderr,
                                "MGL ERROR: createTextureLevel unpack buffer range overflow tex=%u face=%u level=%d "
                                "offset=%zu required=%zu bufferSize=%zu\n",
                                tex->name,
                                face,
                                level,
                                offset,
                                pbo_required_bytes,
                                (size_t)ptr->size);
                        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
                    }
                }

                pixels = &buffer_data[offset];
            }

            unpackTexture(ctx, tex, face, level, (void *)pixels, (void *)texture_data, src_pitch, pixel_size, 0, 0, 0, width, height, depth);

            size_t level_upload_bytes = 0u;
            size_t level_image_bytes = 0u;
            size_t upload_depth = (size_t)MAX(depth, 1);
            size_t zero_probe_len = 0u;
            if (mglMulSizeT(src_pitch, (size_t)MAX(height, 1), &level_image_bytes) &&
                mglMulSizeT(level_image_bytes, upload_depth, &level_upload_bytes)) {
                tex->faces[face].levels[level].last_upload_size = level_upload_bytes;
                tex->faces[face].levels[level].last_src_hash = mglHashBytesSampled(pixels, level_upload_bytes);
            } else {
                tex->faces[face].levels[level].last_upload_size = 0u;
                tex->faces[face].levels[level].last_src_hash = 0ull;
            }
            tex->faces[face].levels[level].last_init_source = kTexImageCopy;
            tex->faces[face].levels[level].last_src_ptr = pixels;
            tex->faces[face].levels[level].ever_written = GL_TRUE;
            tex->faces[face].levels[level].has_initialized_data = GL_TRUE;
            tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;

            if (level_upload_bytes > 0u && pixels) {
                zero_probe_len = level_upload_bytes > 256u ? 256u : level_upload_bytes;
                if (zero_probe_len >= 64u && mglLooksAllZero((const uint8_t *)pixels, zero_probe_len)) {
                    tex->faces[face].levels[level].suspicious_zero_upload = GL_TRUE;
                    tex->faces[face].levels[level].has_initialized_data = GL_FALSE;
                    fprintf(stderr,
                            "MGL WARNING: createTextureLevel upload looks all-zero tex=%u face=%u level=%d probe=%zu bytes src=%p\n",
                            tex->name,
                            face,
                            level,
                            zero_probe_len,
                            pixels);
                }
            }

            tex->dirty_bits |= DIRTY_TEXTURE_DATA;
        };
    }

    tex->faces[face].levels[level].complete = true;
    if (!pixels) {
        tex->faces[face].levels[level].has_initialized_data = GL_FALSE;
        tex->faces[face].levels[level].ever_written = GL_FALSE;
        tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;
        tex->faces[face].levels[level].last_init_source = (format != 0 && type != 0) ? kTexImageNull : kTexInitNone;
        tex->faces[face].levels[level].last_upload_size = 0u;
        tex->faces[face].levels[level].last_src_ptr = NULL;
        tex->faces[face].levels[level].last_src_hash = 0ull;
    }

    tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
    STATE(dirty_bits) |= DIRTY_TEX;

    return true;
}

void mglTexImage1D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;
    bool proxy;

    proxy = false;

    switch(target)
    {
        case GL_TEXTURE_1D:
            break;

        case GL_PROXY_TEXTURE_1D:
            proxy = true;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);

    // verifyFormatType sets the error
    ERROR_CHECK_RETURN(verifyInternalFormatAndFormatType(ctx, internalformat, format, type), 0);

    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(border == 0, GL_INVALID_VALUE);

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, level, internalformat, width, 1, 1, border);
        return;
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    tex->access = GL_READ_ONLY;

    createTextureLevel(ctx, tex, 0, level, false, internalformat, width, 1, 1, format, type, (void *)pixels, proxy);
}

void mglTexImage2D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;
    GLuint face;
    GLboolean is_array;
    GLboolean proxy;
    bool created_ok;

    if (MGL_VERBOSE_TEXTURE_UPLOAD_LOGS) {
        fprintf(stderr,
                "MGL: mglTexImage2D called - target=0x%x, level=%d, internalformat=0x%x, width=%d, height=%d, border=%d, format=0x%x, type=0x%x, pixels=%p\n",
                target, level, internalformat, width, height, border, format, type, pixels);
    }

    face = 0;
    is_array = false;
    proxy = false;

    switch(target)
    {
        case GL_TEXTURE_2D:
            break;

        case GL_PROXY_TEXTURE_2D:
        case GL_PROXY_TEXTURE_CUBE_MAP:
            proxy = true;
            break;

        case GL_PROXY_TEXTURE_1D_ARRAY:
            is_array = true;
            proxy = true;
            break;

        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            face = target - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
            break;

        case GL_PROXY_TEXTURE_RECTANGLE:
            proxy = true;
            ERROR_CHECK_RETURN(level==0, GL_INVALID_OPERATION);
            break;

        case GL_TEXTURE_RECTANGLE:
            ERROR_CHECK_RETURN(level==0, GL_INVALID_OPERATION);
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);

    // verifyFormatType sets the error
    ERROR_CHECK_RETURN(verifyInternalFormatAndFormatType(ctx, internalformat, format, type), 0);

    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(border == 0, GL_INVALID_VALUE);

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, level, internalformat, width, height, 1, border);
        return;
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    tex->access = GL_READ_ONLY;

    if (pixels == NULL)
    {
        created_ok = createTextureLevel(ctx, tex, face, level, is_array, internalformat, width, height, 1, format, type, NULL, proxy);
        if (created_ok)
        {
            tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
            tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
            STATE(dirty_bits) |= DIRTY_TEX;
            ctx->state.error = GL_NO_ERROR;
        }

        if (MGL_VERBOSE_TEXTURE_UPLOAD_LOGS) {
            fprintf(stderr,
                    "MGL TexImage2D allocate-only tex=%u target=0x%x %dx%d pixels=NULL\n",
                    tex->name, target, width, height);
        }
        return;
    }

    created_ok = createTextureLevel(ctx, tex, face, level, is_array, internalformat, width, height, 1, format, type, (void *)pixels, proxy);
    if (created_ok)
    {
        /* Clear stale validation errors on successful upload to avoid false-positive 1282. */
        ctx->state.error = GL_NO_ERROR;
    }
}

void mglTexImage2DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    // Multisample textures are used by virglrenderer for capability probing.
    // Apple Silicon handles MSAA differently - we silently succeed to allow
    // the rendering pipeline to proceed without MSAA.
    fprintf(stderr, "MGL: mglTexImage2DMultisample (stub) - target=0x%x samples=%d internalformat=0x%x %dx%d\n",
            target, samples, internalformat, width, height);
    (void)ctx; (void)target; (void)samples; (void)internalformat;
    (void)width; (void)height; (void)fixedsamplelocations;
    // Don't set error - allow probing to "succeed" so virglrenderer continues
}

void mglTexImage3D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;
    GLboolean is_array;
    GLboolean proxy;

    is_array = false;
    proxy = false;

    switch(target)
    {
        case GL_TEXTURE_3D:
            break;

        case GL_PROXY_TEXTURE_3D:
            proxy = true;
            break;

        case GL_TEXTURE_2D_ARRAY:
            is_array = true;
            break;

        case GL_PROXY_TEXTURE_2D_ARRAY:
            is_array = true;
            proxy = true;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);

    // verifyFormatType sets the error
    ERROR_CHECK_RETURN(verifyInternalFormatAndFormatType(ctx, internalformat, format, type), 0);

    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(depth >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(border == 0, GL_INVALID_VALUE);

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, level, internalformat, width, height, depth, border);
        return;
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    tex->access = GL_READ_ONLY;

    createTextureLevel(ctx, tex, 0, level, is_array, internalformat, width, height, depth, format, type, (void *)pixels, proxy);
}

void mglTexImage3DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    // Multisample array textures - silently succeed like 2D multisample
    fprintf(stderr, "MGL: mglTexImage3DMultisample (stub) - target=0x%x samples=%d internalformat=0x%x %dx%dx%d\n",
            target, samples, internalformat, width, height, depth);
    (void)ctx; (void)target; (void)samples; (void)internalformat;
    (void)width; (void)height; (void)depth; (void)fixedsamplelocations;
    // Don't set error
}

#pragma mark texSubImage
bool texSubImage(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels)
{
    static uint64_t s_tex_sub_image_calls = 0u;
    static int s_dumped_tex13_upload_source = 0;
    uint64_t call_id = ++s_tex_sub_image_calls;
    double start_ms = mglTextureNowMs();
    const void *pixels_raw = pixels;
    const uint8_t *resolved_src = NULL;
    Buffer *resolved_unpack_buf = NULL;
    uint64_t resolved_src_hash = 0ull;
    bool suspicious_zero_upload = false;
    size_t suspicious_probe_len = 0u;
    Buffer *initial_unpack_buf = STATE(buffers[_PIXEL_UNPACK_BUFFER]);
    GLuint initial_unpack_name = initial_unpack_buf ? initial_unpack_buf->name : 0u;
    bool trace_upload = mglShouldTraceTextureUpload(tex,
                                                    initial_unpack_name,
                                                    width,
                                                    height,
                                                    depth,
                                                    0u);
    // Debug: Log large texture uploads (VM framebuffer size)
    if (MGL_VERBOSE_TEXTURE_UPLOAD_LOGS && width >= 640 && height >= 400) {
        fprintf(stderr, "MGL DEBUG: texSubImage tex_id=%u face=%u level=%d %dx%dx%d at (%d,%d,%d) pixels=%p\n",
                tex ? tex->name : 0, face, level, width, height, depth, xoffset, yoffset, zoffset, pixels);
    }

    if (trace_upload) {
        fprintf(stderr,
                "MGL TRACE texSubImage.begin call=%" PRIu64 " tex=%u target=0x%x face=%u level=%d off=(%d,%d,%d) dims=%dx%dx%d fmt=0x%x type=0x%x pixelsRaw=%p\n",
                call_id,
                tex ? tex->name : 0u,
                tex ? tex->target : 0u,
                face,
                level,
                xoffset,
                yoffset,
                zoffset,
                width,
                height,
                depth,
                format,
                type,
                pixels_raw);
    }
    
    // ERROR_CHECK_RETURN_VALUE(tex != NULL, GL_INVALID_OPERATION, false);
    if (tex == NULL) {
        fprintf(stderr, "MGL Error: texSubImage: tex is NULL\n");
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (tex->target == 0) {
        fprintf(stderr,
                "MGL ERROR: texSubImage called with invalid texture object tex=%p target=0x%x\n",
                (void *)tex,
                tex->target);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (face >= _CUBE_MAP_MAX_FACE) {
        fprintf(stderr, "MGL ERROR: texSubImage invalid face=%u\n", face);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if (width <= 0 || height <= 0 || depth <= 0) {
        fprintf(stderr,
                "MGL texSubImage skip invalid size tex=%u %dx%dx%d\n",
                tex->name, width, height, depth);
        return true;
    }

    // ERROR_CHECK_RETURN_VALUE(level <= tex->num_levels, GL_INVALID_OPERATION, false);
    if (level >= (GLint)tex->num_levels) {
        fprintf(stderr, "MGL Error: texSubImage: level %d >= num_levels %d\n", level, tex->num_levels);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }
    
    if (!tex->faces[face].levels) {
        fprintf(stderr, "MGL Error: texSubImage: levels is NULL\n");
        ERROR_CHECK_RETURN_VALUE(false, GL_INVALID_OPERATION, false);
    }
    
    // ERROR_CHECK_RETURN_VALUE(tex->faces[face].levels[level].complete, GL_INVALID_OPERATION, false);
    if (!tex->faces[face].levels[level].complete) {
        fprintf(stderr, "MGL Error: texSubImage: level %d not complete\n", level);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (!pixels && !STATE(buffers[_PIXEL_UNPACK_BUFFER]))
    {
        fprintf(stderr,
                "MGL texSubImage skip upload: pixels=NULL tex=%u target=0x%x %dx%d\n",
                tex->name, tex->target, width, height);
        return true;
    }

    size_t pixel_size;
    size_t row_length_pixels;
    size_t src_pitch;
    size_t src_image_rows;
    size_t src_image_size;
    size_t skip_pixels_bytes;
    size_t skip_rows_bytes;
    size_t skip_images_bytes;
    size_t skip_offset_bytes;

    pixel_size = sizeForFormatType(format, type);
    ERROR_CHECK_RETURN_VALUE(pixel_size > 0u, GL_INVALID_ENUM, false);

    if (ctx->state.unpack.row_length < 0 ||
        ctx->state.unpack.image_height < 0 ||
        ctx->state.unpack.skip_pixels < 0 ||
        ctx->state.unpack.skip_rows < 0 ||
        ctx->state.unpack.skip_images < 0) {
        fprintf(stderr,
                "MGL ERROR: texSubImage invalid negative unpack state rowLength=%d imageHeight=%d skipPixels=%d skipRows=%d skipImages=%d\n",
                ctx->state.unpack.row_length,
                ctx->state.unpack.image_height,
                ctx->state.unpack.skip_pixels,
                ctx->state.unpack.skip_rows,
                ctx->state.unpack.skip_images);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    row_length_pixels = ctx->state.unpack.row_length > 0 ?
                        (size_t)ctx->state.unpack.row_length :
                        (size_t)width;
    if (row_length_pixels < (size_t)width) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if (!mglMulSizeT(row_length_pixels, pixel_size, &src_pitch)) {
        fprintf(stderr,
                "MGL ERROR: texSubImage src pitch overflow rowLength=%zu pixelSize=%zu\n",
                row_length_pixels,
                pixel_size);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    size_t alignment = (size_t)(ctx->state.unpack.alignment > 0 ? ctx->state.unpack.alignment : 1);
    size_t align_rem = src_pitch % alignment;
    if (align_rem) {
        size_t pad = alignment - align_rem;
        if (!mglAddSizeT(src_pitch, pad, &src_pitch)) {
            fprintf(stderr,
                    "MGL ERROR: texSubImage alignment overflow srcPitch=%zu alignment=%zu\n",
                    src_pitch,
                    alignment);
            ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
        }
    }

    src_image_rows = ctx->state.unpack.image_height > 0 ?
                     (size_t)ctx->state.unpack.image_height :
                     (size_t)height;
    if (src_image_rows < (size_t)height) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if (!mglMulSizeT(src_pitch, src_image_rows, &src_image_size)) {
        fprintf(stderr,
                "MGL ERROR: texSubImage image size overflow srcPitch=%zu imageRows=%zu\n",
                src_pitch,
                src_image_rows);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if (!mglMulSizeT((size_t)ctx->state.unpack.skip_pixels, pixel_size, &skip_pixels_bytes) ||
        !mglMulSizeT((size_t)ctx->state.unpack.skip_rows, src_pitch, &skip_rows_bytes) ||
        !mglMulSizeT((size_t)ctx->state.unpack.skip_images, src_image_size, &skip_images_bytes) ||
        !mglAddSizeT(skip_pixels_bytes, skip_rows_bytes, &skip_offset_bytes) ||
        !mglAddSizeT(skip_offset_bytes, skip_images_bytes, &skip_offset_bytes)) {
        fprintf(stderr,
                "MGL ERROR: texSubImage skip offset overflow skipPixels=%d skipRows=%d skipImages=%d srcPitch=%zu imageSize=%zu pixelSize=%zu\n",
                ctx->state.unpack.skip_pixels,
                ctx->state.unpack.skip_rows,
                ctx->state.unpack.skip_images,
                src_pitch,
                src_image_size,
                pixel_size);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    size_t required_bytes = 0u;
    size_t image_bytes = 0u;
    size_t src_depth = (size_t)MAX(depth, 1);
    if (!mglMulSizeT(src_pitch, (size_t)MAX(height, 1), &image_bytes)) {
        fprintf(stderr,
                "MGL ERROR: texSubImage image byte computation overflow tex=%u dims=%dx%dx%d srcPitch=%zu\n",
                tex->name,
                width,
                height,
                depth,
                src_pitch);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if (src_depth <= 1u) {
        required_bytes = image_bytes;
    } else {
        size_t trailing_images = src_depth - 1u;
        size_t trailing_bytes = 0u;
        if (!mglMulSizeT(src_image_size, trailing_images, &trailing_bytes) ||
            !mglAddSizeT(trailing_bytes, image_bytes, &required_bytes)) {
            fprintf(stderr,
                    "MGL ERROR: texSubImage required byte computation overflow tex=%u dims=%dx%dx%d srcImageSize=%zu srcPitch=%zu\n",
                    tex->name,
                    width,
                    height,
                    depth,
                    src_image_size,
                    src_pitch);
            ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
        }
    }

    if (!trace_upload) {
        trace_upload = mglShouldTraceTextureUpload(tex,
                                                   initial_unpack_name,
                                                   width,
                                                   height,
                                                   depth,
                                                   required_bytes);
    }

    if (!mglResolveTexSubImageSource(ctx,
                                     tex,
                                     face,
                                     level,
                                     xoffset,
                                     yoffset,
                                     zoffset,
                                     width,
                                     height,
                                     depth,
                                     format,
                                     type,
                                     pixels_raw,
                                     skip_offset_bytes,
                                     required_bytes,
                                     trace_upload,
                                     &resolved_src,
                                     &resolved_unpack_buf)) {
        return false;
    }

    if (!resolved_src) {
        fprintf(stderr,
                "MGL texSubImage skip upload: resolved source is NULL tex=%u target=0x%x %dx%dx%d\n",
                tex->name,
                tex->target,
                width,
                height,
                depth);
        return true;
    }

    resolved_src_hash = mglHashBytesSampled(resolved_src, required_bytes);
    if (required_bytes > 0u) {
        suspicious_probe_len = required_bytes > 256u ? 256u : required_bytes;
    }
    if (suspicious_probe_len >= 64u && mglLooksAllZero(resolved_src, suspicious_probe_len)) {
        static uint64_t s_zero_upload_warning_count = 0u;
        uint64_t zero_warning_id = ++s_zero_upload_warning_count;
        suspicious_zero_upload = true;
        if (trace_upload || zero_warning_id <= 32u || (zero_warning_id % 512u) == 0u) {
            fprintf(stderr,
                    "MGL WARNING: texSubImage source appears all-zero tex=%u face=%u level=%d probe=%zu bytes required=%zu src=%p warn=%" PRIu64 "\n",
                    tex->name,
                    face,
                    level,
                    suspicious_probe_len,
                    required_bytes,
                    resolved_src,
                    zero_warning_id);
        }
    }

    if (tex->name == 13u && s_dumped_tex13_upload_source == 0 && resolved_src) {
        size_t row_bytes = src_pitch > 0 ? src_pitch : (size_t)(width * pixel_size);
        size_t level_image_bytes = row_bytes * (size_t)MAX(height, 1);
        size_t dump_len = level_image_bytes;
        if (dump_len > 256u) {
            dump_len = 256u;
        }

        fprintf(stderr,
                "MGL DUMP tex13.texSubImage.src.begin tex=%u face=%u level=%d target=0x%x fmt=0x%x type=0x%x "
                "dims=%dx%dx%d off=(%d,%d,%d) pixelSize=%zu srcPitch=%zu dumpLen=%zu ptr=%p\n",
                tex->name,
                face,
                level,
                tex->target,
                format,
                type,
                width,
                height,
                depth,
                xoffset,
                yoffset,
                zoffset,
                pixel_size,
                src_pitch,
                dump_len,
                resolved_src);

        mglDumpBytesToStderr("tex13.texSubImage.src", resolved_src, dump_len, 0u);
        fprintf(stderr, "MGL DUMP tex13.texSubImage.src.end tex=%u\n", tex->name);
        s_dumped_tex13_upload_source = 1;
    }

    void *texture_data;
    TextureLevel *lvl = &tex->faces[face].levels[level];
    bool full_level_write = (!suspicious_zero_upload &&
                             xoffset == 0 &&
                             yoffset == 0 &&
                             zoffset == 0 &&
                             width >= (GLsizei)lvl->width &&
                             height >= (GLsizei)lvl->height &&
                             depth >= (GLsizei)lvl->depth);

    texture_data = (void *)tex->faces[face].levels[level].data;
    if (!texture_data) {
        fprintf(stderr,
                "MGL ERROR: texSubImage texture_data is NULL tex=%u face=%u level=%d target=0x%x\n",
                tex->name,
                face,
                level,
                tex->target);
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }
    
    unpackTexture(ctx, tex, face, level, (void *)resolved_src, texture_data, src_pitch, pixel_size, xoffset, yoffset, zoffset, width, height, depth);

    uint64_t dst_hash = mglHashBytesSampled(texture_data, required_bytes);
    if (trace_upload) {
        fprintf(stderr,
                "MGL TRACE texSubImage.afterUnpack call=%" PRIu64 " tex=%u face=%u level=%d requiredBytes=%zu srcHash=0x%016" PRIx64 " dstHash=0x%016" PRIx64 " elapsed=%.3fms\n",
                call_id,
                tex->name,
                face,
                level,
                required_bytes,
                resolved_src_hash,
                dst_hash,
                mglTextureNowMs() - start_ms);
    }

    // use a blit command to update data
    do
    {
        Buffer *buf;

        buf = resolved_unpack_buf;

        if (buf == NULL)
            continue;

        if (tex->mtl_data == NULL)
            continue;

        size_t src_offset;
        size_t src_image_size;
        size_t src_size;

        src_offset = (size_t)0;

        src_image_size = src_pitch * height;

        src_size = src_image_size * depth;

        // Preserve cube-map / array target slice information. zoffset is for 3D origin, not array/cube slice.
        ctx->mtl_funcs.mtlTexSubImage(ctx, tex, buf, src_offset, src_pitch, src_image_size, src_size, face, level, width, height, depth, xoffset, yoffset, zoffset);
        lvl->ever_written = GL_TRUE;
        lvl->suspicious_zero_upload = suspicious_zero_upload ? GL_TRUE : GL_FALSE;
        lvl->has_initialized_data = full_level_write ? GL_TRUE : GL_FALSE;
        lvl->last_init_source = kTexSubImagePBO;
        lvl->last_upload_size = required_bytes;
        lvl->last_src_ptr = resolved_src;
        lvl->last_src_hash = resolved_src_hash;

        if (trace_upload) {
            fprintf(stderr,
                    "MGL TRACE texSubImage.end call=%" PRIu64 " tex=%u face=%u level=%d upload=PBO ok=1 elapsed=%.3fms\n",
                    call_id,
                    tex->name,
                    face,
                    level,
                    mglTextureNowMs() - start_ms);
        }

        return true;
    } while(false);

    // use process gl to upload texture data
    tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    lvl->ever_written = GL_TRUE;
    lvl->suspicious_zero_upload = suspicious_zero_upload ? GL_TRUE : GL_FALSE;
    lvl->has_initialized_data = full_level_write ? GL_TRUE : GL_FALSE;
    lvl->last_init_source = resolved_unpack_buf ? kTexSubImagePBO : kTexSubImageCPU;
    lvl->last_upload_size = required_bytes;
    lvl->last_src_ptr = resolved_src;
    lvl->last_src_hash = resolved_src_hash;

    if (trace_upload) {
        fprintf(stderr,
                "MGL TRACE texSubImage.end call=%" PRIu64 " tex=%u face=%u level=%d upload=DEFER ok=1 elapsed=%.3fms\n",
                call_id,
                tex->name,
                face,
                level,
                mglTextureNowMs() - start_ms);
    }
    
    return true;
}

#pragma mark texSubImage1D
void texSubImage1D(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    ERROR_CHECK_RETURN(level >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(verifyInternalFormatAndFormatType(ctx, tex->internalformat, format, type), 0);

    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(width + xoffset <= tex->width, GL_INVALID_VALUE);

    texSubImage(ctx, tex, face, level, xoffset, 0, 0, width, 1, 1, format, type, (void *)pixels);
}

void mglTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;

    switch(target)
    {
        case GL_TEXTURE_1D:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

    texSubImage1D(ctx, tex, 0, level, xoffset, width, format, type, pixels);
}

void mglTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;

    tex = getTex(ctx, texture, 0);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

   texSubImage1D(ctx, tex, 0, level, xoffset, width, format, type, pixels);
}

#pragma mark texSubImage2D
bool texSubImage2D(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    TextureLevel *lvl = NULL;

    ERROR_CHECK_RETURN_VALUE(face < _CUBE_MAP_MAX_FACE, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(level >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(tex != NULL, GL_INVALID_OPERATION, false);
    ERROR_CHECK_RETURN_VALUE(level < (GLint)tex->num_levels, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(tex->faces[face].levels != NULL, GL_INVALID_OPERATION, false);

    lvl = &tex->faces[face].levels[level];
    ERROR_CHECK_RETURN_VALUE(lvl->complete, GL_INVALID_OPERATION, false);
    ERROR_CHECK_RETURN_VALUE(verifyInternalFormatAndFormatType(ctx, tex->internalformat, format, type), 0, false);

    ERROR_CHECK_RETURN_VALUE(width >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(height >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(xoffset >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(yoffset >= 0, GL_INVALID_VALUE, false);

    ERROR_CHECK_RETURN_VALUE(width + xoffset <= (GLsizei)lvl->width, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(height + yoffset <= (GLsizei)lvl->height, GL_INVALID_VALUE, false);

    return texSubImage(ctx, tex, face, level, xoffset, yoffset, 0, width, height, 1, format, type, (void *)pixels);
}

void mglTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    static uint64_t s_tex_sub_image2d_calls = 0u;
    uint64_t call_id = ++s_tex_sub_image2d_calls;
    double start_ms = mglTextureNowMs();
    Texture *tex;
    GLuint face;
    bool updated_ok;
    Buffer *unpack_buf = STATE(buffers[_PIXEL_UNPACK_BUFFER]);
    GLuint unpack_name = unpack_buf ? unpack_buf->name : 0u;
    bool trace_call = MGL_VERBOSE_TEXTURE_UPLOAD_LOGS ||
                      unpack_name != 0u ||
                      (width >= 512 && height >= 512);

    if (trace_call) {
        fprintf(stderr,
                "MGL TRACE mglTexSubImage2D.entry call=%" PRIu64 " target=0x%x level=%d off=(%d,%d) size=%dx%d format=0x%x type=0x%x "
                "unpackBufferName=%u pixelsRaw=%p rowLength=%d alignment=%d skipPixels=%d skipRows=%d skipImages=%d\n",
                call_id,
                target,
                level,
                xoffset,
                yoffset,
                width,
                height,
                format,
                type,
                unpack_name,
                pixels,
                ctx->state.unpack.row_length,
                ctx->state.unpack.alignment,
                ctx->state.unpack.skip_pixels,
                ctx->state.unpack.skip_rows,
                ctx->state.unpack.skip_images);
    }

    face = 0;

    switch(target)
    {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_RECTANGLE:
            break;

        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            face = target - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    {
        GLuint active_unit = STATE(active_texture);
        GLuint tex_index = textureIndexFromTarget(ctx, target);
        Texture *bound_tex = NULL;
        if (tex_index < _MAX_TEXTURE_TYPES) {
            bound_tex = STATE(texture_units[active_unit].textures[tex_index]);
        }
        if (bound_tex && bound_tex->name == 13u) {
            trace_call = true;
        }

        if (trace_call) {
            fprintf(stderr,
                    "MGL TRACE mglTexSubImage2D.bound call=%" PRIu64 " activeUnit=%u target=0x%x texIndex=%u boundTex=%p boundName=%u boundTarget=0x%x\n",
                    call_id,
                    active_unit,
                    target,
                    tex_index,
                    (void *)bound_tex,
                    bound_tex ? bound_tex->name : 0u,
                    bound_tex ? bound_tex->target : 0u);
        }
    }

    tex = getTex(ctx, 0, target);

    if (!tex) {
        fprintf(stderr,
                "MGL ERROR: mglTexSubImage2D getTex returned NULL call=%" PRIu64 " target=0x%x level=%d\n",
                call_id,
                target,
                level);
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    updated_ok = texSubImage2D(ctx, tex, face, level, xoffset, yoffset, width, height, format, type, pixels);

    if (trace_call || (tex && tex->name == 13u) || !updated_ok) {
        fprintf(stderr,
                "MGL TRACE mglTexSubImage2D.exit call=%" PRIu64 " tex=%u face=%u level=%d ok=%d elapsed=%.3fms\n",
                call_id,
                tex ? tex->name : 0u,
                face,
                level,
                updated_ok ? 1 : 0,
                mglTextureNowMs() - start_ms);
    }
}

void mglTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;
    bool updated_ok;

    tex = getTex(ctx, texture, 0);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

    updated_ok = texSubImage2D(ctx, tex, 0, level, xoffset, yoffset, width, height, format, type, pixels);
    if (!updated_ok) {
        return;
    }
}

#pragma mark texSubImage3D
bool texSubImage3D(GLMContext ctx, Texture *tex, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    TextureLevel *lvl = NULL;

    ERROR_CHECK_RETURN_VALUE(level >= 0, GL_INVALID_VALUE, false);

    ERROR_CHECK_RETURN_VALUE(tex, GL_INVALID_OPERATION, false);
    ERROR_CHECK_RETURN_VALUE(level < (GLint)tex->num_levels, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(tex->faces[0].levels != NULL, GL_INVALID_OPERATION, false);
    lvl = &tex->faces[0].levels[level];
    ERROR_CHECK_RETURN_VALUE(lvl->complete, GL_INVALID_OPERATION, false);

    ERROR_CHECK_RETURN_VALUE(verifyInternalFormatAndFormatType(ctx, tex->internalformat, format, type), 0, false);

    ERROR_CHECK_RETURN_VALUE(width >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(height >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(depth >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(xoffset >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(yoffset >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(zoffset >= 0, GL_INVALID_VALUE, false);

    ERROR_CHECK_RETURN_VALUE(width + xoffset <= (GLsizei)lvl->width, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(height + yoffset <= (GLsizei)lvl->height, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(depth + zoffset <= (GLsizei)lvl->depth, GL_INVALID_VALUE, false);

    return texSubImage(ctx, tex, 0, level, xoffset, yoffset, zoffset, width, height, depth, format, type, (void *)pixels);
}

void mglTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;

    switch(target)
    {
        case GL_TEXTURE_3D:
        case GL_TEXTURE_2D_ARRAY:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

    if (!texSubImage3D(ctx, tex, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels)) {
        return;
    }
}

void mglTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex;

    tex = getTex(ctx, texture, 0);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

    if (!texSubImage3D(ctx, tex, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels)) {
        return;
    }
}

#pragma mark TexStorage

void texStorage(GLMContext ctx, Texture *tex, GLuint faces, GLsizei levels, GLboolean is_array, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean proxy)
{
    tex->access = GL_READ_ONLY;

    for(int face=0; face<faces; face++)
    {
        GLuint level_width, level_height;

        level_width = width;
        level_height = height;

        for(int level=0; level<levels; level++)
        {
            createTextureLevel(ctx, tex, face, level, is_array, internalformat, level_width, level_height, depth, 0, 0, NULL, proxy);

            level_width >>= 1;
            level_height >>= 1;
            
            // Mipmap dimensions must be at least 1
            if (level_width == 0) level_width = 1;
            if (level_height == 0) level_height = 1;
        }
    }

    // mark it immutable
    tex->immutable_storage = BUFFER_IMMUTABLE_STORAGE_FLAG;

    // bind it to metal
    ctx->mtl_funcs.mtlBindTexture(ctx, tex);

    ERROR_CHECK_RETURN(tex->mtl_data, GL_OUT_OF_MEMORY);
}

void mglTexStorage1D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width)
{
    Texture *tex;
    GLboolean proxy;

    proxy = false;

    switch(target)
    {
        case GL_TEXTURE_1D:
            break;

        case GL_PROXY_TEXTURE_1D:
            proxy = true;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, width, 1, 1, 0);
        return;
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

    texStorage(ctx, tex, 1, levels, false, internalformat, width, 1, 1, proxy);
}

void mglTextureStorage1D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width)
{
    Texture *tex;

    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);

    tex = getTex(ctx, texture, 0);

    ERROR_CHECK_RETURN(tex != NULL, GL_INVALID_OPERATION);

    texStorage(ctx, tex, 1, levels, false, internalformat, width, 1, 1, false);
}

void mglTexStorage2D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
{
    Texture *tex;
    GLboolean is_array;
    GLboolean proxy;
    GLuint num_faces;

    is_array = false;
    proxy = false;
    num_faces = 1;

    switch(target)
    {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            break;

        case GL_PROXY_TEXTURE_2D:
        case GL_PROXY_TEXTURE_RECTANGLE:
            proxy = true;
            break;

        case GL_TEXTURE_CUBE_MAP:
            num_faces = 6;
            proxy = false;
            break;

        case GL_PROXY_TEXTURE_CUBE_MAP:
            num_faces = 6;
            proxy = true;
            break;

        case GL_TEXTURE_1D_ARRAY:
            is_array = true;
            break;

        case GL_PROXY_TEXTURE_1D_ARRAY:
            is_array = true;
            proxy = true;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height > 0, GL_INVALID_VALUE);

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, width, height, 1, 0);
        return;
    }

    tex = getTex(ctx, 0, target);
    
    fprintf(stderr, "MGL: mglTexStorage2D target=0x%x levels=%d internalformat=0x%x %dx%d tex=%p\n",
            target, levels, internalformat, width, height, tex);
    fflush(stderr);

    texStorage(ctx, tex, num_faces, levels, is_array, internalformat, width, height, 1, proxy);
    ctx->state.error = GL_NO_ERROR;
}


void mglTextureStorage2D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
{
    Texture *tex;

    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height > 0, GL_INVALID_VALUE);

    tex = getTex(ctx, texture, 0);

    texStorage(ctx, tex, 1, levels, false, internalformat, width, height, 1, false);
}

void mglTextureStorage2DMultisample(GLMContext ctx, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    fprintf(stderr, "MGL WARNING: glTextureStorage2DMultisample called (stub) - MSAA not fully supported\n");
    // Fall back to non-MSAA storage
    mglTextureStorage2D(ctx, texture, 1, internalformat, width, height);
}

void mglTexStorage3D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    Texture *tex;
    GLboolean is_array;
    GLboolean proxy;

    is_array = false;
    proxy = false;

    switch(target)
    {
        case GL_TEXTURE_3D:
            break;

        case GL_PROXY_TEXTURE_3D:
            proxy = true;
            break;

        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            is_array = true;
            break;

        case GL_PROXY_TEXTURE_2D_ARRAY:
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY: // keep proxy case explicit (no duplicate GL_TEXTURE_CUBE_MAP_ARRAY here)
            is_array = true;
            proxy = true;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(checkMaxLevels(levels, width, height, depth), GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(depth > 0, GL_INVALID_VALUE);

    if (proxy)
    {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, width, height, depth, 0);
        return;
    }

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    texStorage(ctx, tex, 1, levels, is_array, internalformat, width, height, depth, proxy);
}

void mglTextureStorage3D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    Texture *tex;

    ERROR_CHECK_RETURN(levels > 0, GL_INVALID_VALUE);

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(width > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height > 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(depth > 0, GL_INVALID_VALUE);

    tex = getTex(ctx, texture, 0);

    createTextureLevel(ctx, tex, 0, 0, false, internalformat, width, height, depth, 0, 0, NULL, false);
}

void mglTextureStorage3DMultisample(GLMContext ctx, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    fprintf(stderr, "MGL WARNING: glTextureStorage3DMultisample called (stub) - MSAA not fully supported\n");
    // Fall back to non-MSAA storage
    mglTextureStorage3D(ctx, texture, 1, internalformat, width, height, depth);
}


#pragma mark clear tex image
void mglClearTexImage(GLMContext ctx, GLuint texture, GLint level, GLenum format, GLenum type, const void *data)
{
    fprintf(stderr, "MGL: glClearTexImage called - texture=%u level=%d\n", texture, level);
    
    Texture *tex = getTex(ctx, texture, 0);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    // For now, use texSubImage to clear - fill with the clear data
    GLsizei width = tex->width >> level;
    GLsizei height = tex->height >> level;
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    
    // If data is NULL, clear to zero
    if (data == NULL) {
        size_t pixel_size = sizeForFormatType(format, type);

        // CRITICAL SECURITY FIX: Prevent integer overflow in texture clear allocation
        if (width > SIZE_MAX / height / pixel_size) {
            fprintf(stderr, "MGL SECURITY ERROR: Texture clear allocation would overflow: %dx%dx%zu\n", width, height, pixel_size);
            STATE(error) = GL_OUT_OF_MEMORY;
            return;
        }

        size_t size = width * height * pixel_size;
        void *clear_data = calloc(1, size);
        if (clear_data) {
            texSubImage(ctx, tex, 0, level, 0, 0, 0, width, height, 1, format, type, clear_data);
            free(clear_data);
        }
    } else {
        // Fill entire texture with the provided clear value
        size_t pixel_size = sizeForFormatType(format, type);

        // CRITICAL SECURITY FIX: Prevent integer overflow in texture fill allocation
        if (width > SIZE_MAX / height / pixel_size) {
            fprintf(stderr, "MGL SECURITY ERROR: Texture fill allocation would overflow: %dx%dx%zu\n", width, height, pixel_size);
            STATE(error) = GL_OUT_OF_MEMORY;
            return;
        }

        size_t size = width * height * pixel_size;
        void *fill_data = malloc(size);
        if (fill_data) {
            // Replicate the clear value across the entire buffer
            for (size_t i = 0; i < width * height; i++) {
                memcpy((char*)fill_data + i * pixel_size, data, pixel_size);
            }
            texSubImage(ctx, tex, 0, level, 0, 0, 0, width, height, 1, format, type, fill_data);
            free(fill_data);
        }
    }
}

void mglClearTexSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *data)
{
    fprintf(stderr, "MGL: glClearTexSubImage called - texture=%u %dx%dx%d at (%d,%d,%d)\n",
            texture, width, height, depth, xoffset, yoffset, zoffset);
    
    Texture *tex = getTex(ctx, texture, 0);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    size_t pixel_size = sizeForFormatType(format, type);

    // CRITICAL SECURITY FIX: Prevent integer overflow in texture subimage allocation
    if (width > SIZE_MAX / height / depth / pixel_size) {
        fprintf(stderr, "MGL SECURITY ERROR: Texture subimage allocation would overflow: %dx%dx%dx%zu\n", width, height, depth, pixel_size);
        STATE(error) = GL_OUT_OF_MEMORY;
        return;
    }

    size_t size = width * height * depth * pixel_size;
    
    if (data == NULL) {
        void *clear_data = calloc(1, size);
        if (clear_data) {
            texSubImage(ctx, tex, 0, level, xoffset, yoffset, zoffset, width, height, depth, format, type, clear_data);
            free(clear_data);
        }
    } else {
        void *fill_data = malloc(size);
        if (fill_data) {
            for (size_t i = 0; i < width * height * depth; i++) {
                memcpy((char*)fill_data + i * pixel_size, data, pixel_size);
            }
            texSubImage(ctx, tex, 0, level, xoffset, yoffset, zoffset, width, height, depth, format, type, fill_data);
            free(fill_data);
        }
    }
}

#pragma mark compressed tex image
void mglCompressedTexImage3D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexImage3D called (stub) - compressed textures not supported\n");
}

void mglCompressedTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexImage2D called (stub) - compressed textures not supported\n");
}

void mglCompressedTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexImage1D called (stub) - compressed textures not supported\n");
}

void mglCompressedTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexSubImage3D called (stub) - compressed textures not supported\n");
}

void mglCompressedTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexSubImage2D called (stub) - compressed textures not supported\n");
}

void mglCompressedTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexSubImage1D called (stub) - compressed textures not supported\n");
}

#pragma mark copy tex
void mglCopyTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border)
{
    // Stub - not commonly used
    fprintf(stderr, "MGL WARNING: glCopyTexImage1D called (stub)\n");
}

void mglCopyTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border)
{
    fprintf(stderr, "MGL: glCopyTexImage2D called - target=0x%x level=%d %dx%d\n", target, level, width, height);
    
    // Get or create texture
    Texture *tex = getTex(ctx, 0, target);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    // Initialize texture storage if needed
    if (tex->mipmap_levels == 0) {
        initBaseTexLevel(ctx, tex, internalformat, width, height, 1);
    }
    
    // Copy from framebuffer to texture
    ctx->mtl_funcs.mtlCopyTexSubImage(ctx, tex, level, 0, 0, x, y, width, height);
}

void mglCopyTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    fprintf(stderr, "MGL: glCopyTexSubImage1D called (stub)\n");
    // Stub - 1D textures rarely used
}

void mglCopyTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    fprintf(stderr, "MGL: glCopyTexSubImage2D called - target=0x%x %dx%d at (%d,%d) from (%d,%d)\n",
            target, width, height, xoffset, yoffset, x, y);
    
    // Get the bound texture
    Texture *tex = getTex(ctx, 0, target);
    if (!tex) {
        fprintf(stderr, "MGL ERROR: glCopyTexSubImage2D - no texture bound\n");
        return;
    }
    
    // This copies from the current read framebuffer to the texture
    // For now, use the Metal blit function
    ctx->mtl_funcs.mtlCopyTexSubImage(ctx, tex, level, xoffset, yoffset, x, y, width, height);
}

void mglCopyTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    fprintf(stderr, "MGL: glCopyTexSubImage3D called - stub, only 2D copy supported\n");
    // For now just do 2D copy, ignoring zoffset
    Texture *tex = getTex(ctx, 0, target);
    if (tex) {
        ctx->mtl_funcs.mtlCopyTexSubImage(ctx, tex, level, xoffset, yoffset, x, y, width, height);
    }
}

void mglCopyTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    fprintf(stderr, "MGL: glCopyTextureSubImage1D called (stub)\n");
}

void mglCopyTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    fprintf(stderr, "MGL: glCopyTextureSubImage2D called - texture=%u %dx%d\n", texture, width, height);
    Texture *tex = getTex(ctx, texture, 0);
    if (tex) {
        ctx->mtl_funcs.mtlCopyTexSubImage(ctx, tex, level, xoffset, yoffset, x, y, width, height);
    }
}

void mglCopyTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    fprintf(stderr, "MGL: glCopyTextureSubImage3D called (stub)\n");
    Texture *tex = getTex(ctx, texture, 0);
    if (tex) {
        ctx->mtl_funcs.mtlCopyTexSubImage(ctx, tex, level, xoffset, yoffset, x, y, width, height);
    }
}

#pragma mark get tex image

void mglGetTexImage(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, void *pixels)
{
    fprintf(stderr, "MGL: glGetTexImage called - target=0x%x level=%d format=0x%x type=0x%x\n",
            target, level, format, type);
    
    Texture *tex = getTex(ctx, 0, target);
    if (!tex) {
        fprintf(stderr, "MGL ERROR: glGetTexImage - no texture bound\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    if (!tex->mtl_data) {
        fprintf(stderr, "MGL ERROR: glGetTexImage - texture has no Metal data\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    if (level >= tex->num_levels) {
        fprintf(stderr, "MGL ERROR: glGetTexImage - invalid level %d (max %d)\n", level, tex->num_levels);
        ERROR_RETURN(GL_INVALID_VALUE);
    }
    
    // Calculate dimensions for this level
    GLsizei width = tex->width >> level;
    GLsizei height = tex->height >> level;
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    
    // Calculate bytes per row
    size_t pixel_size = sizeForFormatType(format, type);
    GLuint bytesPerRow = width * pixel_size;
    GLuint bytesPerImage = bytesPerRow * height;
    
    fprintf(stderr, "MGL: glGetTexImage - reading %dx%d, bytesPerRow=%u\n", width, height, bytesPerRow);
    
    // Use the Metal function to read the texture
    ctx->mtl_funcs.mtlGetTexImage(ctx, tex, pixels, bytesPerRow, bytesPerImage, 0, 0, width, height, level, 0);
}

void mglGetTextureImage(GLMContext ctx, GLuint texture, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL: glGetTextureImage called - texture=%u level=%d\n", texture, level);
    
    Texture *tex = getTex(ctx, texture, 0);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    if (!tex->mtl_data) {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    GLsizei width = tex->width >> level;
    GLsizei height = tex->height >> level;
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    
    size_t pixel_size = sizeForFormatType(format, type);
    GLuint bytesPerRow = width * pixel_size;
    GLuint bytesPerImage = bytesPerRow * height;
    
    if (bytesPerImage > bufSize) {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    ctx->mtl_funcs.mtlGetTexImage(ctx, tex, pixels, bytesPerRow, bytesPerImage, 0, 0, width, height, level, 0);
}

void mglGetTextureSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL: glGetTextureSubImage called - texture=%u\n", texture);
    
    Texture *tex = getTex(ctx, texture, 0);
    if (!tex || !tex->mtl_data) {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
    
    size_t pixel_size = sizeForFormatType(format, type);
    GLuint bytesPerRow = width * pixel_size;
    GLuint bytesPerImage = bytesPerRow * height;
    
    ctx->mtl_funcs.mtlGetTexImage(ctx, tex, pixels, bytesPerRow, bytesPerImage, xoffset, yoffset, width, height, level, zoffset);
}

void mglGetCompressedTexImage(GLMContext ctx, GLenum target, GLint level, void *img)
{
    fprintf(stderr, "MGL WARNING: glGetCompressedTexImage called (stub) - compressed textures not supported\n");
}

void mglGetnCompressedTexImage(GLMContext ctx, GLenum target, GLint lod, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL WARNING: glGetnCompressedTexImage called (stub) - compressed textures not supported\n");
}

void mglGetCompressedTextureSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL WARNING: glGetCompressedTextureSubImage called (stub) - compressed textures not supported\n");
}

void mglTextureView(GLMContext ctx, GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers)
{
    fprintf(stderr, "MGL WARNING: glTextureView called (stub) - texture views not supported\n");
}

void mglTextureBuffer(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer)
{
    fprintf(stderr, "MGL WARNING: glTextureBuffer called (stub)\n");
}

void mglTextureBufferRange(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    fprintf(stderr, "MGL WARNING: glTextureBufferRange called (stub)\n");
}

void mglCompressedTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data)
{
    fprintf(stderr, "MGL WARNING: glCompressedTextureSubImage1D called (stub)\n");
}

void mglCompressedTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    fprintf(stderr, "MGL WARNING: glCompressedTextureSubImage2D called (stub)\n");
}

void mglCompressedTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data)
{
    fprintf(stderr, "MGL WARNING: glCompressedTextureSubImage3D called (stub)\n");
}

void mglGetCompressedTextureImage(GLMContext ctx, GLuint texture, GLint level, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL WARNING: glGetCompressedTextureImage called (stub)\n");
}

void mglGetTextureLevelParameteriv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLint *params)
{
    Texture *tex = getTex(ctx, texture, 0);
    if (!tex) {
        *params = 0;
        return;
    }
    
    GLsizei width = tex->width >> level;
    GLsizei height = tex->height >> level;
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    
    switch (pname) {
        case GL_TEXTURE_WIDTH:
            *params = width;
            break;
        case GL_TEXTURE_HEIGHT:
            *params = height;
            break;
        case GL_TEXTURE_DEPTH:
            *params = tex->depth >> level;
            if (*params < 1) *params = 1;
            break;
        case GL_TEXTURE_INTERNAL_FORMAT:
            *params = tex->internalformat;
            break;
        default:
            fprintf(stderr, "MGL: glGetTextureLevelParameteriv pname=0x%x not implemented\n", pname);
            *params = 0;
            break;
    }
}

void mglGetTextureLevelParameterfv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLfloat *params)
{
    GLint iparams;
    mglGetTextureLevelParameteriv(ctx, texture, level, pname, &iparams);
    *params = (GLfloat)iparams;
}

void mglGetTextureParameterfv(GLMContext ctx, GLuint texture, GLenum pname, GLfloat *params)
{
    fprintf(stderr, "MGL: glGetTextureParameterfv called (stub) pname=0x%x\n", pname);
    *params = 0.0f;
}

void mglGetTextureParameterIiv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params)
{
    fprintf(stderr, "MGL: glGetTextureParameterIiv called (stub) pname=0x%x\n", pname);
    *params = 0;
}

void mglGetTextureParameterIuiv(GLMContext ctx, GLuint texture, GLenum pname, GLuint *params)
{
    fprintf(stderr, "MGL: glGetTextureParameterIuiv called (stub) pname=0x%x\n", pname);
    *params = 0;
}

void mglGetTextureParameteriv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params)
{
    fprintf(stderr, "MGL: glGetTextureParameteriv called (stub) pname=0x%x\n", pname);
    *params = 0;
}

void mglGetTexParameterIiv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    fprintf(stderr, "MGL: glGetTexParameterIiv called (stub) pname=0x%x\n", pname);
    *params = 0;
}

void mglGetTexParameterIuiv(GLMContext ctx, GLenum target, GLenum pname, GLuint *params)
{
    fprintf(stderr, "MGL: glGetTexParameterIuiv called (stub) pname=0x%x\n", pname);
    *params = 0;
}

void mglSampleCoverage(GLMContext ctx, GLfloat value, GLboolean invert)
{
    // Stub - sample coverage is a hint for multisampling
    fprintf(stderr, "MGL: glSampleCoverage called (stub) value=%f invert=%d\n", value, invert);
}
