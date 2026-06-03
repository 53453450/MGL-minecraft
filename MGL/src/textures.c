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

#include <errno.h>
#include <execinfo.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <Accelerate/Accelerate.h>

#include "pixel_utils.h"
#include "utils.h"
#include "glm_context.h"

extern void *getBufferData(GLMContext ctx, Buffer *ptr);
extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);
extern GLsizei mglSafeMaxTextureSize(GLMContext ctx);
extern GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);
extern bool getParam(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *iparam, GLfloat *fparam);

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

static void mglDumpByteWindowToStderr(const char *label,
                                      const uint8_t *bytes,
                                      size_t total_length,
                                      size_t requested_offset,
                                      size_t window_length)
{
    if (!bytes || total_length == 0 || window_length == 0) {
        fprintf(stderr,
                "MGL DUMP %s unavailable base=%p total=%zu window=%zu\n",
                label ? label : "window",
                bytes,
                total_length,
                window_length);
        return;
    }

    size_t offset = requested_offset;
    if (offset >= total_length) {
        offset = total_length - 1u;
    }

    if (offset + window_length > total_length) {
        window_length = total_length - offset;
    }

    mglDumpBytesToStderr(label, bytes + offset, window_length, offset);
}

static uint64_t mglHashBytesSampled(const void *data, size_t len);
static bool mglMulSizeT(size_t a, size_t b, size_t *out);
static bool mglAddSizeT(size_t a, size_t b, size_t *out);

typedef struct MGLTextureUnpackLayout_t {
    size_t pixel_size;
    size_t row_copy_bytes;
    size_t row_length_pixels;
    size_t src_pitch;
    size_t src_image_rows;
    size_t src_image_size;
    size_t skip_offset_bytes;
    size_t required_bytes;
    size_t compact_upload_bytes;
} MGLTextureUnpackLayout;

typedef struct MGLTexturePackLayout_t {
    size_t pixel_size;
    size_t row_copy_bytes;
    size_t row_length_pixels;
    size_t dst_pitch;
    size_t dst_image_rows;
    size_t dst_image_size;
    size_t skip_offset_bytes;
    size_t write_span_bytes;
    size_t required_bytes;
} MGLTexturePackLayout;

static void mglDumpTextureUploadRowSamples(const char *prefix,
                                           const uint8_t *bytes,
                                           size_t total_length,
                                           size_t pitch,
                                           size_t row_bytes,
                                           GLsizei width,
                                           GLsizei height,
                                           GLsizei depth,
                                           size_t pixel_size)
{
    if (!prefix) {
        prefix = "texSubImage.zeroProbe.row";
    }

    if (!bytes || total_length == 0u || pitch == 0u || row_bytes == 0u ||
        width <= 0 || height <= 0 || depth <= 0) {
        fprintf(stderr,
                "MGL DUMP %s.rows unavailable base=%p total=%zu pitch=%zu rowBytes=%zu dims=%dx%dx%d pixelSize=%zu\n",
                prefix,
                bytes,
                total_length,
                pitch,
                row_bytes,
                width,
                height,
                depth,
                pixel_size);
        return;
    }

    const size_t h = (size_t)height;
    const size_t d = (size_t)depth;
    const size_t plane_pitch = pitch * h;
    const size_t planes[2] = {0u, d > 1u ? (d / 2u) : 0u};
    const size_t rows[7] = {
        0u,
        h > 1u ? 1u : 0u,
        h > 2u ? 2u : 0u,
        h / 2u,
        h > 3u ? h - 3u : 0u,
        h > 2u ? h - 2u : 0u,
        h > 1u ? h - 1u : 0u
    };

    for (size_t pi = 0u; pi < (d > 1u ? 2u : 1u); pi++) {
        size_t z = planes[pi];
        if (z >= d) {
            continue;
        }

        for (size_t ri = 0u; ri < 7u; ri++) {
            size_t y = rows[ri];
            if (y >= h) {
                continue;
            }

            bool duplicate = false;
            for (size_t prev = 0u; prev < ri; prev++) {
                if (rows[prev] == y) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }

            size_t offset = z * plane_pitch + y * pitch;
            if (offset >= total_length) {
                fprintf(stderr,
                        "MGL DUMP %s.row z=%zu y=%zu offset=%zu outside total=%zu\n",
                        prefix,
                        z,
                        y,
                        offset,
                        total_length);
                continue;
            }

            size_t available = total_length - offset;
            size_t scan_len = row_bytes < available ? row_bytes : available;
            size_t nonzero = 0u;
            for (size_t i = 0u; i < scan_len; i++) {
                if (bytes[offset + i] != 0u) {
                    nonzero++;
                }
            }

            uint32_t first_pixel = 0u;
            size_t first_pixel_bytes = pixel_size < sizeof(first_pixel) ? pixel_size : sizeof(first_pixel);
            if (first_pixel_bytes > 0u && available >= first_pixel_bytes) {
                memcpy(&first_pixel, bytes + offset, first_pixel_bytes);
            }

            fprintf(stderr,
                    "MGL DUMP %s.row z=%zu y=%zu offset=%zu pitch=%zu rowBytes=%zu scan=%zu nonZero=%zu hash=0x%016" PRIx64 " firstPixel=0x%08x\n",
                    prefix,
                    z,
                    y,
                    offset,
                    pitch,
                    row_bytes,
                    scan_len,
                    nonzero,
                    mglHashBytesSampled(bytes + offset, scan_len),
                    first_pixel);

            char label[128];
            snprintf(label, sizeof(label), "%s.row.z%zu.y%zu.first64", prefix, z, y);
            mglDumpByteWindowToStderr(label, bytes, total_length, offset, 64u);
        }
    }
}

static void mglDumpTextureUploadSamples(Texture *tex,
                                        GLuint face,
                                        GLint level,
                                        const uint8_t *src,
                                        size_t src_total,
                                        size_t src_pitch,
                                        const uint8_t *dst,
                                        size_t dst_total,
                                        size_t dst_pitch,
                                        size_t pixel_size,
                                        GLsizei width,
                                        GLsizei height,
                                        GLsizei depth)
{
    size_t sample_len = 64u;
    size_t src_center = 0u;
    size_t dst_center = 0u;
    size_t src_tail = src_total > sample_len ? src_total - sample_len : 0u;
    size_t dst_tail = dst_total > sample_len ? dst_total - sample_len : 0u;

    if (width > 0 && height > 0 && pixel_size > 0) {
        size_t cx = (size_t)width / 2u;
        size_t cy = (size_t)height / 2u;
        size_t cz = depth > 1 ? (size_t)depth / 2u : 0u;
        size_t src_plane = src_pitch * (size_t)MAX(height, 1);
        size_t dst_plane = dst_pitch * (size_t)MAX(height, 1);

        src_center = (cz * src_plane) + (cy * src_pitch) + (cx * pixel_size);
        dst_center = (cz * dst_plane) + (cy * dst_pitch) + (cx * pixel_size);
    }

    fprintf(stderr,
            "MGL DUMP texSubImage.zeroProbe.begin tex=%u face=%u level=%d dims=%dx%dx%d "
            "src=%p srcTotal=%zu srcPitch=%zu dst=%p dstTotal=%zu dstPitch=%zu pixelSize=%zu\n",
            tex ? tex->name : 0u,
            face,
            level,
            width,
            height,
            depth,
            src,
            src_total,
            src_pitch,
            dst,
            dst_total,
            dst_pitch,
            pixel_size);

    mglDumpByteWindowToStderr("texSubImage.zeroProbe.src.first64", src, src_total, 0u, sample_len);
    mglDumpByteWindowToStderr("texSubImage.zeroProbe.src.center64", src, src_total, src_center, sample_len);
    mglDumpByteWindowToStderr("texSubImage.zeroProbe.src.tail64", src, src_total, src_tail, sample_len);
    mglDumpByteWindowToStderr("texSubImage.zeroProbe.dst.first64", dst, dst_total, 0u, sample_len);
    mglDumpByteWindowToStderr("texSubImage.zeroProbe.dst.center64", dst, dst_total, dst_center, sample_len);
    mglDumpByteWindowToStderr("texSubImage.zeroProbe.dst.tail64", dst, dst_total, dst_tail, sample_len);

    size_t row_bytes = 0u;
    if (width > 0 && pixel_size > 0u) {
        row_bytes = (size_t)width * pixel_size;
    }
    if (row_bytes > 0u) {
        mglDumpTextureUploadRowSamples("texSubImage.zeroProbe.src",
                                       src,
                                       src_total,
                                       src_pitch,
                                       row_bytes,
                                       width,
                                       height,
                                       depth,
                                       pixel_size);
        mglDumpTextureUploadRowSamples("texSubImage.zeroProbe.dst",
                                       dst,
                                       dst_total,
                                       dst_pitch,
                                       row_bytes,
                                       width,
                                       height,
                                       depth,
                                       pixel_size);
    }

    fprintf(stderr,
            "MGL DUMP texSubImage.zeroProbe.end tex=%u\n",
            tex ? tex->name : 0u);
}

static const char *mglTextureInitSourceName(GLuint source)
{
    switch ((MGLTexLevelInitSource)source) {
        case kTexInitNone: return "none";
        case kTexImageNull: return "TexImage(NULL)";
        case kTexImageCopy: return "TexImage(copy)";
        case kTexImagePBO: return "TexImage(PBO)";
        case kTexSubImageCPU: return "TexSubImage(CPU)";
        case kTexSubImagePBO: return "TexSubImage(PBO)";
        case kTexRenderTargetWrite: return "RenderTarget(write)";
        case kTexMetalFill: return "Metal(fill)";
        default: return "unknown";
    }
}

static const char *mglBufferInitSourceName(MGLBufferInitSource source)
{
    switch (source) {
        case kInitNone: return "none";
        case kInitBufferDataNull: return "BufferData(NULL)";
        case kInitBufferDataCopy: return "BufferData(copy)";
        case kInitBufferSubData: return "BufferSubData";
        case kInitCopyBufferSubData: return "CopyBufferSubData";
        case kInitMapWrite: return "MapWrite";
        default: return "unknown";
    }
}

static void mglDumpNativeBacktraceToStderr(const char *tag, size_t max_frames)
{
    const char *safe_tag = tag ? tag : "backtrace";
    void *frames[64];
    int limit = (int)(max_frames > 64u ? 64u : max_frames);
    if (limit <= 0) {
        limit = 1;
    }

    int count = backtrace(frames, limit);
    char **symbols = backtrace_symbols(frames, count);

    fprintf(stderr,
            "MGL TRACE %s nativeBacktrace frames=%d\n",
            safe_tag,
            count);

    for (int i = 0; i < count; i++) {
        fprintf(stderr,
                "MGL TRACE %s bt[%02d]=%s\n",
                safe_tag,
                i,
                symbols ? symbols[i] : "(symbol unavailable)");
    }

    if (symbols) {
        free(symbols);
    }
}

static void mglRequestJavaThreadDumpForZeroCpuUpload(Texture *tex,
                                                     GLuint face,
                                                     GLint level,
                                                     GLsizei width,
                                                     GLsizei height,
                                                     GLsizei depth,
                                                     uint64_t warning_id)
{
    static int s_requested_512_zero_cpu_thread_dump = 0;

    /*
     * HotSpot treats SIGQUIT as "print all Java thread stacks" instead of a
     * fatal signal.  This gives us symbolic Minecraft/LWJGL frames for the JIT
     * addresses shown by native backtrace_symbols().
     */
    if (s_requested_512_zero_cpu_thread_dump ||
        width != 512 ||
        height != 512 ||
        depth != 1) {
        return;
    }

    if (getenv("MGL_DISABLE_ZERO_UPLOAD_JAVA_STACK")) {
        fprintf(stderr,
                "MGL TRACE texSubImage.zeroCPU javaThreadDump skipped by MGL_DISABLE_ZERO_UPLOAD_JAVA_STACK tex=%u warn=%" PRIu64 "\n",
                tex ? tex->name : 0u,
                warning_id);
        s_requested_512_zero_cpu_thread_dump = 1;
        return;
    }

    s_requested_512_zero_cpu_thread_dump = 1;
    errno = 0;
    int rc = kill(getpid(), SIGQUIT);

    fprintf(stderr,
            "MGL TRACE texSubImage.zeroCPU javaThreadDump request rc=%d errno=%d (%s) tex=%u face=%u level=%d dims=%dx%dx%d warn=%" PRIu64 "\n",
            rc,
            errno,
            strerror(errno),
            tex ? tex->name : 0u,
            face,
            level,
            width,
            height,
            depth,
            warning_id);
}

static void mglDumpTexSubImageZeroCpuResourceTag(GLMContext ctx,
                                                 Texture *tex,
                                                 TextureLevel *lvl,
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
                                                 const uint8_t *resolved_src,
                                                 Buffer *resolved_unpack_buf,
                                                 size_t required_bytes,
                                                 size_t compact_upload_bytes,
                                                 size_t src_pitch,
                                                 size_t compact_upload_row_bytes,
                                                 size_t pixel_size,
                                                 uint64_t warning_id)
{
    GLuint active_unit = ctx ? ctx->state.active_texture : 0u;
    Texture *active_tex = NULL;
    Texture *bound_2d = NULL;
    Texture *bound_cube = NULL;
    Texture *bound_2d_array = NULL;
    Sampler *bound_sampler = NULL;
    Buffer *unpack = NULL;
    unsigned mask_word = 0u;
    GLuint program_name = 0u;

    if (ctx) {
        program_name = ctx->state.program_name;
        unpack = ctx->state.buffers[_PIXEL_UNPACK_BUFFER];
        if (active_unit < TEXTURE_UNITS) {
            active_tex = ctx->state.active_textures[active_unit];
            bound_2d = ctx->state.texture_units[active_unit].textures[_TEXTURE_2D];
            bound_cube = ctx->state.texture_units[active_unit].textures[_TEXTURE_CUBE_MAP];
            bound_2d_array = ctx->state.texture_units[active_unit].textures[_TEXTURE_2D_ARRAY];
            bound_sampler = ctx->state.texture_samplers[active_unit];
            mask_word = ctx->state.active_texture_mask[active_unit / 32u];
        }
    }

    fprintf(stderr,
            "MGL ZERO CPU UPLOAD resource warn=%" PRIu64 " tex=%u texPtr=%p target=0x%x index=%u face=%u level=%d "
            "label=\"%s\" dims=%dx%dx%d off=(%d,%d,%d) fmt=0x%x type=0x%x internal=0x%x base=%ux%ux%u levels=%u complete=%d "
            "mtl=%p pixelsRaw=%p resolvedSrc=%p resolvedUnpack=%u\n",
            warning_id,
            tex ? tex->name : 0u,
            (void *)tex,
            tex ? tex->target : 0u,
            tex ? tex->index : 0u,
            face,
            level,
            (tex && tex->debug_label[0] != '\0') ? tex->debug_label : "(none)",
            width,
            height,
            depth,
            xoffset,
            yoffset,
            zoffset,
            format,
            type,
            tex ? tex->internalformat : 0u,
            tex ? tex->width : 0u,
            tex ? tex->height : 0u,
            tex ? tex->depth : 0u,
            tex ? tex->num_levels : 0u,
            tex ? tex->complete : 0,
            tex ? tex->mtl_data : NULL,
            pixels_raw,
            resolved_src,
            resolved_unpack_buf ? resolved_unpack_buf->name : 0u);

    fprintf(stderr,
            "MGL ZERO CPU UPLOAD state warn=%" PRIu64 " program=%u activeUnit=%u activeTex=%u tex2D=%u cube=%u tex2DArray=%u "
            "sampler=%u maskWord=0x%x unpackBuffer=%u unpackPtr=%p unpackSize=%lld unpackWritten=%d unpackRange=[%lld,%lld) "
            "unpackSource=%s rowLength=%d imageHeight=%d alignment=%d skipPixels=%d skipRows=%d skipImages=%d "
            "required=%zu compact=%zu srcPitch=%zu compactRow=%zu pixelSize=%zu\n",
            warning_id,
            program_name,
            active_unit,
            active_tex ? active_tex->name : 0u,
            bound_2d ? bound_2d->name : 0u,
            bound_cube ? bound_cube->name : 0u,
            bound_2d_array ? bound_2d_array->name : 0u,
            bound_sampler ? bound_sampler->name : 0u,
            mask_word,
            unpack ? unpack->name : 0u,
            (void *)unpack,
            unpack ? (long long)unpack->size : 0ll,
            unpack ? unpack->ever_written : 0,
            unpack ? (long long)unpack->written_min : -1ll,
            unpack ? (long long)unpack->written_max : -1ll,
            unpack ? mglBufferInitSourceName(unpack->last_init_source) : "none",
            ctx ? ctx->state.unpack.row_length : 0,
            ctx ? ctx->state.unpack.image_height : 0,
            ctx ? ctx->state.unpack.alignment : 0,
            ctx ? ctx->state.unpack.skip_pixels : 0,
            ctx ? ctx->state.unpack.skip_rows : 0,
            ctx ? ctx->state.unpack.skip_images : 0,
            required_bytes,
            compact_upload_bytes,
            src_pitch,
            compact_upload_row_bytes,
            pixel_size);

    if (lvl) {
        fprintf(stderr,
                "MGL ZERO CPU UPLOAD level warn=%" PRIu64 " tex=%u face=%u level=%d levelComplete=%d levelSize=%ux%ux%u "
                "pitch=%zu dataSize=%zu data=%p ever=%d initialized=%d suspicious=%d lastSource=%s lastUpload=%zu "
                "lastSrc=%p lastHash=0x%016" PRIx64 "\n",
                warning_id,
                tex ? tex->name : 0u,
                face,
                level,
                lvl->complete,
                lvl->width,
                lvl->height,
                lvl->depth,
                lvl->pitch,
                lvl->data_size,
                (void *)(uintptr_t)lvl->data,
                lvl->ever_written,
                lvl->has_initialized_data,
                lvl->suspicious_zero_upload,
                mglTextureInitSourceName(lvl->last_init_source),
                lvl->last_upload_size,
                lvl->last_src_ptr,
                lvl->last_src_hash);
    }

    if (ctx && tex) {
        unsigned printed = 0u;
        for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
            Texture *unit_active = ctx->state.active_textures[unit];
            Texture *unit_2d = ctx->state.texture_units[unit].textures[_TEXTURE_2D];
            Texture *unit_cube = ctx->state.texture_units[unit].textures[_TEXTURE_CUBE_MAP];
            Texture *unit_2d_array = ctx->state.texture_units[unit].textures[_TEXTURE_2D_ARRAY];
            if (unit_active == tex || unit_2d == tex || unit_cube == tex || unit_2d_array == tex) {
                fprintf(stderr,
                        "MGL ZERO CPU UPLOAD boundUnit warn=%" PRIu64 " unit=%u active=%u tex2D=%u cube=%u tex2DArray=%u sampler=%u\n",
                        warning_id,
                        unit,
                        unit_active ? unit_active->name : 0u,
                        unit_2d ? unit_2d->name : 0u,
                        unit_cube ? unit_cube->name : 0u,
                        unit_2d_array ? unit_2d_array->name : 0u,
                        ctx->state.texture_samplers[unit] ? ctx->state.texture_samplers[unit]->name : 0u);
                printed++;
                if (printed >= 8u) {
                    fprintf(stderr,
                            "MGL ZERO CPU UPLOAD boundUnit warn=%" PRIu64 " truncated after %u matching units\n",
                            warning_id,
                            printed);
                    break;
                }
            }
        }
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

static bool mglLooksAllZeroSampled(const uint8_t *bytes, size_t len)
{
    size_t probe;
    size_t mid;
    size_t tail;

    if (!bytes || len == 0) {
        return false;
    }

    probe = len < 256u ? len : 256u;
    if (probe < 64u) {
        return false;
    }

    if (!mglLooksAllZero(bytes, probe)) {
        return false;
    }

    if (len <= probe) {
        return true;
    }

    mid = len / 2u;
    if (mid + probe > len) {
        mid = len - probe;
    }
    if (!mglLooksAllZero(bytes + mid, probe)) {
        return false;
    }

    tail = len - probe;
    if (tail != 0u && tail != mid && !mglLooksAllZero(bytes + tail, probe)) {
        return false;
    }

    return true;
}

static bool mglTextureRectByteRange(TextureLevel *level,
                                    size_t pixel_size,
                                    size_t xoffset,
                                    size_t yoffset,
                                    size_t zoffset,
                                    size_t width,
                                    size_t height,
                                    size_t depth,
                                    size_t *byte_offset_out,
                                    size_t *byte_span_out)
{
    if (!level || !level->data || pixel_size == 0u ||
        width == 0u || height == 0u || depth == 0u ||
        level->pitch == 0u || level->height == 0u) {
        return false;
    }

    size_t row_bytes = 0u;
    size_t x_bytes = 0u;
    size_t y_bytes = 0u;
    size_t image_pitch = 0u;
    size_t z_bytes = 0u;
    size_t xy_bytes = 0u;
    size_t base_offset = 0u;
    size_t trailing_rows = height - 1u;
    size_t trailing_slices = depth - 1u;
    size_t trailing_row_bytes = 0u;
    size_t trailing_slice_bytes = 0u;
    size_t trailing_bytes = 0u;

    if (!mglMulSizeT(width, pixel_size, &row_bytes) ||
        !mglMulSizeT(xoffset, pixel_size, &x_bytes) ||
        !mglMulSizeT(yoffset, level->pitch, &y_bytes) ||
        !mglMulSizeT(level->pitch, (size_t)level->height, &image_pitch) ||
        !mglMulSizeT(zoffset, image_pitch, &z_bytes) ||
        !mglAddSizeT(x_bytes, y_bytes, &xy_bytes) ||
        !mglAddSizeT(xy_bytes, z_bytes, &base_offset) ||
        !mglMulSizeT(trailing_rows, level->pitch, &trailing_row_bytes) ||
        !mglMulSizeT(trailing_slices, image_pitch, &trailing_slice_bytes) ||
        !mglAddSizeT(trailing_row_bytes, trailing_slice_bytes, &trailing_bytes) ||
        !mglAddSizeT(trailing_bytes, row_bytes, &trailing_bytes)) {
        return false;
    }

    if (base_offset > level->data_size || trailing_bytes > level->data_size - base_offset) {
        return false;
    }

    if (byte_offset_out) {
        *byte_offset_out = base_offset;
    }
    if (byte_span_out) {
        *byte_span_out = trailing_bytes;
    }
    return true;
}

static uint64_t mglHashTextureRect(const uint8_t *base,
                                   size_t dst_pitch,
                                   size_t dst_image_pitch,
                                   size_t row_bytes,
                                   size_t height,
                                   size_t depth)
{
    if (!base || row_bytes == 0u || height == 0u || depth == 0u) {
        return 0ull;
    }

    uint64_t hash = 1469598103934665603ull;
    size_t total = 0u;

    for (size_t z = 0u; z < depth; z++) {
        const uint8_t *slice = base + z * dst_image_pitch;
        for (size_t y = 0u; y < height; y++) {
            const uint8_t *row = slice + y * dst_pitch;
            for (size_t i = 0u; i < row_bytes; i++) {
                hash ^= (uint64_t)row[i];
                hash *= 1099511628211ull;
            }
            total += row_bytes;
        }
    }

    hash ^= (uint64_t)total;
    hash *= 1099511628211ull;
    return hash;
}

static bool mglTextureRectLooksAllZero(const uint8_t *base,
                                       size_t dst_pitch,
                                       size_t dst_image_pitch,
                                       size_t row_bytes,
                                       size_t height,
                                       size_t depth)
{
    if (!base || row_bytes == 0u || height == 0u || depth == 0u) {
        return false;
    }

    for (size_t z = 0u; z < depth; z++) {
        const uint8_t *slice = base + z * dst_image_pitch;
        for (size_t y = 0u; y < height; y++) {
            const uint8_t *row = slice + y * dst_pitch;
            if (!mglLooksAllZero(row, row_bytes)) {
                return false;
            }
        }
    }

    return true;
}

static bool mglFindFirstNonZeroByte(const uint8_t *bytes, size_t len, size_t *offset_out, uint8_t *value_out)
{
    if (!bytes || len == 0) {
        return false;
    }

    for (size_t i = 0; i < len; i++) {
        if (bytes[i] != 0u) {
            if (offset_out) {
                *offset_out = i;
            }
            if (value_out) {
                *value_out = bytes[i];
            }
            return true;
        }
    }

    return false;
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

static bool mglComputeTexturePackLayout(GLMContext ctx,
                                        GLsizei width,
                                        GLsizei height,
                                        GLsizei depth,
                                        size_t pixel_size,
                                        const char *op,
                                        MGLTexturePackLayout *layout)
{
    if (!ctx || !layout || pixel_size == 0u || width <= 0 || height <= 0 || depth <= 0) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    memset(layout, 0, sizeof(*layout));
    layout->pixel_size = pixel_size;

    if (ctx->state.pack.row_length < 0 ||
        ctx->state.pack.image_height < 0 ||
        ctx->state.pack.skip_pixels < 0 ||
        ctx->state.pack.skip_rows < 0 ||
        ctx->state.pack.skip_images < 0) {
        fprintf(stderr,
                "MGL ERROR: %s invalid negative pack state rowLength=%d imageHeight=%d skipPixels=%d skipRows=%d skipImages=%d\n",
                op ? op : "texture readback",
                ctx->state.pack.row_length,
                ctx->state.pack.image_height,
                ctx->state.pack.skip_pixels,
                ctx->state.pack.skip_rows,
                ctx->state.pack.skip_images);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    layout->row_length_pixels = ctx->state.pack.row_length > 0 ?
                                (size_t)ctx->state.pack.row_length :
                                (size_t)width;

    if (!mglMulSizeT((size_t)width, pixel_size, &layout->row_copy_bytes) ||
        !mglMulSizeT(layout->row_length_pixels, pixel_size, &layout->dst_pitch)) {
        fprintf(stderr,
                "MGL ERROR: %s pack row computation overflow width=%d rowLength=%zu pixelSize=%zu\n",
                op ? op : "texture readback",
                width,
                layout->row_length_pixels,
                pixel_size);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    size_t alignment = (size_t)(ctx->state.pack.alignment > 0 ? ctx->state.pack.alignment : 1);
    size_t align_rem = layout->dst_pitch % alignment;
    if (align_rem) {
        size_t pad = alignment - align_rem;
        if (!mglAddSizeT(layout->dst_pitch, pad, &layout->dst_pitch)) {
            fprintf(stderr,
                    "MGL ERROR: %s pack row alignment overflow dstPitch=%zu alignment=%zu\n",
                    op ? op : "texture readback",
                    layout->dst_pitch,
                    alignment);
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }
    }

    layout->dst_image_rows = ctx->state.pack.image_height > 0 ?
                             (size_t)ctx->state.pack.image_height :
                             (size_t)height;
    if (!mglMulSizeT(layout->dst_pitch, layout->dst_image_rows, &layout->dst_image_size)) {
        fprintf(stderr,
                "MGL ERROR: %s pack image stride overflow dstPitch=%zu imageRows=%zu\n",
                op ? op : "texture readback",
                layout->dst_pitch,
                layout->dst_image_rows);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    size_t skip_pixels_bytes = 0u;
    size_t skip_rows_bytes = 0u;
    size_t skip_images_bytes = 0u;
    GLint skip_images = depth > 1 ? ctx->state.pack.skip_images : 0;
    if (!mglMulSizeT((size_t)ctx->state.pack.skip_pixels, pixel_size, &skip_pixels_bytes) ||
        !mglMulSizeT((size_t)ctx->state.pack.skip_rows, layout->dst_pitch, &skip_rows_bytes) ||
        !mglMulSizeT((size_t)skip_images, layout->dst_image_size, &skip_images_bytes) ||
        !mglAddSizeT(skip_pixels_bytes, skip_rows_bytes, &layout->skip_offset_bytes) ||
        !mglAddSizeT(layout->skip_offset_bytes, skip_images_bytes, &layout->skip_offset_bytes)) {
        fprintf(stderr,
                "MGL ERROR: %s pack skip computation overflow skipPixels=%d skipRows=%d skipImages=%d dstPitch=%zu imageSize=%zu pixelSize=%zu\n",
                op ? op : "texture readback",
                ctx->state.pack.skip_pixels,
                ctx->state.pack.skip_rows,
                skip_images,
                layout->dst_pitch,
                layout->dst_image_size,
                pixel_size);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    size_t image_span = layout->row_copy_bytes;
    if (height > 1) {
        size_t trailing_row_bytes = 0u;
        if (!mglMulSizeT(layout->dst_pitch, (size_t)(height - 1), &trailing_row_bytes) ||
            !mglAddSizeT(image_span, trailing_row_bytes, &image_span)) {
            fprintf(stderr,
                    "MGL ERROR: %s pack image span overflow dstPitch=%zu height=%d rowBytes=%zu\n",
                    op ? op : "texture readback",
                    layout->dst_pitch,
                    height,
                    layout->row_copy_bytes);
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }
    }

    layout->write_span_bytes = image_span;
    if (depth > 1) {
        size_t trailing_image_bytes = 0u;
        if (!mglMulSizeT(layout->dst_image_size, (size_t)(depth - 1), &trailing_image_bytes) ||
            !mglAddSizeT(layout->write_span_bytes, trailing_image_bytes, &layout->write_span_bytes)) {
            fprintf(stderr,
                    "MGL ERROR: %s pack depth span overflow imageSize=%zu depth=%d imageSpan=%zu\n",
                    op ? op : "texture readback",
                    layout->dst_image_size,
                    depth,
                    image_span);
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }
    }

    if (!mglAddSizeT(layout->skip_offset_bytes, layout->write_span_bytes, &layout->required_bytes)) {
        fprintf(stderr,
                "MGL ERROR: %s pack required byte overflow skip=%zu span=%zu\n",
                op ? op : "texture readback",
                layout->skip_offset_bytes,
                layout->write_span_bytes);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    return true;
}

static bool mglComputeTextureUnpackLayout(GLMContext ctx,
                                          GLsizei width,
                                          GLsizei height,
                                          GLsizei depth,
                                          size_t pixel_size,
                                          const char *op,
                                          MGLTextureUnpackLayout *layout)
{
    if (!ctx || !layout || pixel_size == 0u || width <= 0 || height <= 0 || depth <= 0) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    memset(layout, 0, sizeof(*layout));
    layout->pixel_size = pixel_size;

    if (ctx->state.unpack.row_length < 0 ||
        ctx->state.unpack.image_height < 0 ||
        ctx->state.unpack.skip_pixels < 0 ||
        ctx->state.unpack.skip_rows < 0 ||
        ctx->state.unpack.skip_images < 0) {
        fprintf(stderr,
                "MGL ERROR: %s invalid negative unpack state rowLength=%d imageHeight=%d skipPixels=%d skipRows=%d skipImages=%d\n",
                op ? op : "texture upload",
                ctx->state.unpack.row_length,
                ctx->state.unpack.image_height,
                ctx->state.unpack.skip_pixels,
                ctx->state.unpack.skip_rows,
                ctx->state.unpack.skip_images);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    layout->row_length_pixels = ctx->state.unpack.row_length > 0 ?
                                (size_t)ctx->state.unpack.row_length :
                                (size_t)width;

    if (!mglMulSizeT((size_t)width, pixel_size, &layout->row_copy_bytes) ||
        !mglMulSizeT(layout->row_length_pixels, pixel_size, &layout->src_pitch)) {
        fprintf(stderr,
                "MGL ERROR: %s unpack row computation overflow width=%d rowLength=%zu pixelSize=%zu\n",
                op ? op : "texture upload",
                width,
                layout->row_length_pixels,
                pixel_size);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    size_t alignment = (size_t)(ctx->state.unpack.alignment > 0 ? ctx->state.unpack.alignment : 1);
    size_t align_rem = layout->src_pitch % alignment;
    if (align_rem) {
        size_t pad = alignment - align_rem;
        if (!mglAddSizeT(layout->src_pitch, pad, &layout->src_pitch)) {
            fprintf(stderr,
                    "MGL ERROR: %s unpack row alignment overflow srcPitch=%zu alignment=%zu\n",
                    op ? op : "texture upload",
                    layout->src_pitch,
                    alignment);
            ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
        }
    }

    layout->src_image_rows = ctx->state.unpack.image_height > 0 ?
                             (size_t)ctx->state.unpack.image_height :
                             (size_t)height;

    if (!mglMulSizeT(layout->src_pitch, layout->src_image_rows, &layout->src_image_size)) {
        fprintf(stderr,
                "MGL ERROR: %s unpack image stride overflow srcPitch=%zu imageRows=%zu\n",
                op ? op : "texture upload",
                layout->src_pitch,
                layout->src_image_rows);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    size_t skip_pixels_bytes = 0u;
    size_t skip_rows_bytes = 0u;
    size_t skip_images_bytes = 0u;
    GLint skip_images = depth > 1 ? ctx->state.unpack.skip_images : 0;
    if (!mglMulSizeT((size_t)ctx->state.unpack.skip_pixels, pixel_size, &skip_pixels_bytes) ||
        !mglMulSizeT((size_t)ctx->state.unpack.skip_rows, layout->src_pitch, &skip_rows_bytes) ||
        !mglMulSizeT((size_t)skip_images, layout->src_image_size, &skip_images_bytes) ||
        !mglAddSizeT(skip_pixels_bytes, skip_rows_bytes, &layout->skip_offset_bytes) ||
        !mglAddSizeT(layout->skip_offset_bytes, skip_images_bytes, &layout->skip_offset_bytes)) {
        fprintf(stderr,
                "MGL ERROR: %s unpack skip computation overflow skipPixels=%d skipRows=%d skipImages=%d srcPitch=%zu imageSize=%zu pixelSize=%zu\n",
                op ? op : "texture upload",
                ctx->state.unpack.skip_pixels,
                ctx->state.unpack.skip_rows,
                skip_images,
                layout->src_pitch,
                layout->src_image_size,
                pixel_size);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    size_t image_required_bytes = layout->row_copy_bytes;
    if (height > 1) {
        size_t trailing_row_bytes = 0u;
        if (!mglMulSizeT(layout->src_pitch, (size_t)(height - 1), &trailing_row_bytes) ||
            !mglAddSizeT(image_required_bytes, trailing_row_bytes, &image_required_bytes)) {
            fprintf(stderr,
                    "MGL ERROR: %s unpack image span overflow srcPitch=%zu height=%d rowBytes=%zu\n",
                    op ? op : "texture upload",
                    layout->src_pitch,
                    height,
                    layout->row_copy_bytes);
            ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
        }
    }

    layout->required_bytes = image_required_bytes;
    if (depth > 1) {
        size_t trailing_image_bytes = 0u;
        if (!mglMulSizeT(layout->src_image_size, (size_t)(depth - 1), &trailing_image_bytes) ||
            !mglAddSizeT(layout->required_bytes, trailing_image_bytes, &layout->required_bytes)) {
            fprintf(stderr,
                    "MGL ERROR: %s unpack depth span overflow srcImageSize=%zu depth=%d imageBytes=%zu\n",
                    op ? op : "texture upload",
                    layout->src_image_size,
                    depth,
                    image_required_bytes);
            ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
        }
    }

    if (!mglMulSizeT(layout->row_copy_bytes, (size_t)height, &layout->compact_upload_bytes) ||
        !mglMulSizeT(layout->compact_upload_bytes, (size_t)depth, &layout->compact_upload_bytes)) {
        fprintf(stderr,
                "MGL ERROR: %s compact upload byte computation overflow dims=%dx%dx%d rowBytes=%zu\n",
                op ? op : "texture upload",
                width,
                height,
                depth,
                layout->row_copy_bytes);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

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

    if (MGL_VERBOSE_TEXTURE_UPLOAD_LOGS || target == GL_PROXY_TEXTURE_2D) {
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
    bool source_range_is_bounded = false;

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
        source_range_is_bounded = true;
    } else {
        if (pixels_raw) {
            if (raw_value < 4096u) {
                fprintf(stderr,
                        "MGL ERROR: texSubImage source resolve: CPU source pointer looks like offset/raw integer raw=%p skipOff=%zu\n",
                        pixels_raw,
                        skip_offset_bytes);
                ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
            }
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

    /*
     * Only PBO-backed uploads give us a known readable range.  For CPU pointers
     * we cannot prove the mapped allocation is at least required_bytes long;
     * probing head/mid/tail for diagnostics can SIGBUS before the real upload
     * path gets a chance to validate/copy row-by-row.
     */
    if (resolved_src && source_range_is_bounded) {
        src_hash = mglHashBytesSampled(resolved_src, required_bytes);
    }

    if (trace_upload) {
        fprintf(stderr,
                "MGL TRACE TexSubImage.source tex=%u target=0x%x face=%u level=%d fmt=0x%x type=0x%x "
                "label=\"%s\" dims=%dx%dx%d off=(%d,%d,%d) unpackBufferName=%u pixelsRaw=%p resolvedSrcPtr=%p "
                "sourceClass=%s rowLength=%d alignment=%d skipPixels=%d skipRows=%d skipImages=%d skipOffsetBytes=%zu requiredBytes=%zu srcHash=0x%016" PRIx64 "\n",
                tex ? tex->name : 0u,
                tex ? tex->target : 0u,
                face,
                level,
                format,
                type,
                (tex && tex->debug_label[0] != '\0') ? tex->debug_label : "(none)",
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

    if (resolved_src && source_range_is_bounded) {
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

        if (trace_upload && mglLooksAllZeroSampled(resolved_src, required_bytes)) {
            size_t first_nonzero = 0u;
            uint8_t first_value = 0u;
            bool has_nonzero = mglFindFirstNonZeroByte(resolved_src, required_bytes, &first_nonzero, &first_value);
            fprintf(stderr,
                    "MGL WARNING: TexSubImage source sampled head/mid/tail chunks are all zero "
                    "(tex=%u target=0x%x unpack=%u raw=%p resolved=%p required=%zu fullZero=%d firstNonZero=0x%zx value=0x%02x)\n",
                    tex ? tex->name : 0u,
                    tex ? tex->target : 0u,
                    unpack_name,
                    pixels_raw,
                    resolved_src,
                    required_bytes,
                    has_nonzero ? 0 : 1,
                    has_nonzero ? first_nonzero : 0u,
                    has_nonzero ? first_value : 0u);
            if (has_nonzero) {
                size_t dump_offset = first_nonzero;
                size_t dump_available = required_bytes - first_nonzero;
                if (dump_available > 64u) {
                    dump_available = 64u;
                }
                mglDumpBytesToStderr("TexSubImage.source.firstNonZero", resolved_src + dump_offset, dump_available, dump_offset);
            }
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

static bool mglIsTextureObjectTarget(GLenum target)
{
    switch(target)
    {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        case GL_TEXTURE_BUFFER:
            return true;
        default:
            return false;
    }
}

static GLenum mglCanonicalTextureObjectTarget(GLenum target)
{
    switch(target)
    {
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
        case GL_PROXY_TEXTURE_CUBE_MAP:
            return GL_TEXTURE_CUBE_MAP;
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY:
            return GL_TEXTURE_CUBE_MAP_ARRAY;
        case GL_PROXY_TEXTURE_1D:
            return GL_TEXTURE_1D;
        case GL_PROXY_TEXTURE_2D:
            return GL_TEXTURE_2D;
        case GL_PROXY_TEXTURE_3D:
            return GL_TEXTURE_3D;
        case GL_PROXY_TEXTURE_RECTANGLE:
            return GL_TEXTURE_RECTANGLE;
        case GL_PROXY_TEXTURE_1D_ARRAY:
            return GL_TEXTURE_1D_ARRAY;
        case GL_PROXY_TEXTURE_2D_ARRAY:
            return GL_TEXTURE_2D_ARRAY;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE:
            return GL_TEXTURE_2D_MULTISAMPLE;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
        default:
            return target;
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
    GLenum object_target;

    object_target = mglCanonicalTextureObjectTarget(target);
    index = textureIndexFromTarget(ctx, object_target);
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
    ptr->target = object_target;
    ptr->index = index;

    float black_color[] = {0,0,0,0};

    ptr->params.depth_stencil_mode = GL_DEPTH_COMPONENT;
    ptr->params.base_level = 0;
    memcpy(ptr->params.border_color, black_color, 4 * sizeof(float));
    ptr->params.compare_func = GL_LEQUAL;
    ptr->params.compare_mode = GL_NONE;
    ptr->params.lod_bias = 0.0;
    ptr->params.min_filter = GL_NEAREST_MIPMAP_LINEAR;
    ptr->params.mag_filter = GL_LINEAR;
    ptr->params.max_anisotropy = 1.0;
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
    if (!ptr)
        return NULL;

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
    else if (ptr->target != target)
    {
        fprintf(stderr,
                "MGL ERROR: texture name %u was first bound/created as target 0x%x, cannot bind/create as 0x%x\n",
                texture,
                ptr->target,
                target);
        STATE(error) = GL_INVALID_OPERATION;
        return NULL;
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

static inline GLuint mglTraceTextureNameC(Texture *tex)
{
    return tex ? tex->name : 0u;
}

static bool mglTextureFormatLooksDepthOrStencil(GLenum internalformat)
{
    switch (internalformat) {
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
        case GL_STENCIL_INDEX:
        case GL_STENCIL_INDEX8:
            return true;
        default:
            return false;
    }
}

static bool mglTextureIsSampleableColor2D(Texture *tex)
{
    if (!tex ||
        tex->target != GL_TEXTURE_2D ||
        tex->index != _TEXTURE_2D ||
        tex->is_render_target ||
        tex->internalformat == 0 ||
        mglTextureFormatLooksDepthOrStencil(tex->internalformat) ||
        tex->num_levels == 0 ||
        !tex->faces[0].levels) {
        return false;
    }

    TextureLevel *level0 = &tex->faces[0].levels[0];
    return level0->complete &&
           (level0->ever_written || level0->has_initialized_data);
}

static void mglRecordLastSampled2DTexture(GLMContext ctx, GLuint unit, Texture *tex)
{
    if (!ctx || unit >= TEXTURE_UNITS) {
        return;
    }

    if (mglTextureIsSampleableColor2D(tex)) {
        STATE(last_sampled_2d_textures[unit]) = tex;
    }
}

static void mglRecordBoundSampled2DTextureIfReady(GLMContext ctx, Texture *tex)
{
    if (!ctx || !tex) {
        return;
    }

    for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
        if (STATE(texture_units[unit].textures[_TEXTURE_2D]) == tex) {
            mglRecordLastSampled2DTexture(ctx, unit, tex);
        }
    }
}

void mglClearLastSampled2DTextureIfMatches(GLMContext ctx, Texture *tex)
{
    if (!ctx || !tex) {
        return;
    }

    for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
        if (STATE(last_sampled_2d_textures[unit]) == tex) {
            STATE(last_sampled_2d_textures[unit]) = NULL;
        }
    }
}

static GLboolean mglTextureUnitHasAnyBinding(GLMContext ctx, GLuint unit)
{
    if (!ctx || unit >= TEXTURE_UNITS) {
        return GL_FALSE;
    }

    for (GLuint i = 0; i < _MAX_TEXTURE_TYPES; i++) {
        if (STATE(texture_units[unit].textures[i])) {
            return GL_TRUE;
        }
    }

    return GL_FALSE;
}

static Texture *mglChooseTextureUnitActiveBinding(GLMContext ctx, GLuint unit)
{
    if (!ctx || unit >= TEXTURE_UNITS) {
        return NULL;
    }

    Texture *active = STATE(active_textures[unit]);
    if (active) {
        for (GLuint i = 0; i < _MAX_TEXTURE_TYPES; i++) {
            if (STATE(texture_units[unit].textures[i]) == active) {
                return active;
            }
        }
    }

    static const GLuint fallback_order[] = {
        _TEXTURE_2D,
        _TEXTURE_2D_ARRAY,
        _TEXTURE_CUBE_MAP,
        _TEXTURE_3D,
        _TEXTURE_1D,
        _TEXTURE_1D_ARRAY,
        _TEXTURE_RECTANGLE,
        _TEXTURE_CUBE_MAP_ARRAY,
        _TEXTURE_BUFFER_TARGET,
        _TEXTURE_2D_MULTISAMPLE,
        _TEXTURE_2D_MULTISAMPLE_ARRAY,
        _RENDERBUFFER
    };

    for (size_t i = 0; i < sizeof(fallback_order) / sizeof(fallback_order[0]); i++) {
        Texture *tex = STATE(texture_units[unit].textures[fallback_order[i]]);
        if (tex) {
            return tex;
        }
    }

    return NULL;
}

static void mglUpdateTextureUnitActiveMask(GLMContext ctx, GLuint unit)
{
    if (!ctx || unit >= TEXTURE_UNITS) {
        return;
    }

    Texture *active = mglChooseTextureUnitActiveBinding(ctx, unit);
    GLuint mask_index = unit / 32u;
    GLuint mask = 1u << (unit % 32u);

    STATE(active_textures[unit]) = active;
    if (mglTextureUnitHasAnyBinding(ctx, unit)) {
        STATE(active_texture_mask[mask_index]) |= mask;
    } else {
        STATE(active_texture_mask[mask_index]) &= ~mask;
    }
}

static void mglTraceTextureUnitState(GLMContext ctx,
                                     const char *api,
                                     GLuint unit,
                                     GLenum target,
                                     GLuint texture,
                                     Texture *bound)
{
    static uint64_t s_texture_unit_trace_count = 0;

    if (!ctx || unit >= TEXTURE_UNITS) {
        return;
    }

    Texture *unit_active = STATE(active_textures[unit]);
    Texture *unit_buffer = STATE(texture_units[unit].textures[_TEXTURE_BUFFER_TARGET]);
    Texture *unit_2d = STATE(texture_units[unit].textures[_TEXTURE_2D]);
    Texture *unit_cube = STATE(texture_units[unit].textures[_TEXTURE_CUBE_MAP]);
    Texture *observed = bound ? bound : unit_active;
    GLenum observed_format = observed ? observed->internalformat : 0u;
    bool depth_or_stencil = mglTextureFormatLooksDepthOrStencil(observed_format);
    bool dynamic_render_target_texture =
        (texture >= 50u && texture <= 100u) ||
        (mglTraceTextureNameC(unit_active) >= 50u && mglTraceTextureNameC(unit_active) <= 100u) ||
        (mglTraceTextureNameC(unit_2d) >= 50u && mglTraceTextureNameC(unit_2d) <= 100u);

    bool interesting =
        unit == 0 ||
        target == GL_TEXTURE_BUFFER ||
        texture == 0 ||
        texture == 10 ||
        texture == 13 ||
        texture == 4231 ||
        dynamic_render_target_texture ||
        depth_or_stencil ||
        mglTraceTextureNameC(unit_active) == 4231 ||
        mglTraceTextureNameC(unit_buffer) != 0 ||
        mglTraceTextureNameC(unit_2d) == 4231 ||
        mglTraceTextureNameC(unit_cube) == 10;

    if (!interesting) {
        return;
    }

    uint64_t hit = ++s_texture_unit_trace_count;
    if (!dynamic_render_target_texture && !depth_or_stencil &&
        hit > 512 && (hit % 512) != 0) {
        return;
    }

    fprintf(stderr,
            "MGL TRACE TexUnit.%s hit=%llu unit=%u activeUnit=%u target=0x%x texture=%u bound=%u "
            "state(active=%u texBuffer=%u tex2D=%u cube=%u maskWord=0x%x internal=0x%x depthStencil=%d dynamicRT=%d)\n",
            api ? api : "?",
            (unsigned long long)hit,
            unit,
            STATE(active_texture),
            target,
            texture,
            mglTraceTextureNameC(bound),
            mglTraceTextureNameC(unit_active),
            mglTraceTextureNameC(unit_buffer),
            mglTraceTextureNameC(unit_2d),
            mglTraceTextureNameC(unit_cube),
            STATE(active_texture_mask[unit / 32]),
            observed_format,
            depth_or_stencil ? 1 : 0,
            dynamic_render_target_texture ? 1 : 0);
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

    if (!ctx)
        return;

    if (n < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (n == 0)
        return;

    if (!textures) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (MGL_VERBOSE_TEXTURE_BIND_LOGS) {
        fprintf(stderr,
                "MGL TRACE GenTextures call=%llu ctx=%p n=%d textures=%p\n",
                (unsigned long long)call_id,
                (void *)ctx,
                (int)n,
                (void *)textures);
    }

    while(n--)
    {
        GLuint name = getNewName(&STATE(texture_table));
        *textures++ = name;
        if (MGL_VERBOSE_TEXTURE_BIND_LOGS || call_id <= 32ull || (call_id % 4096ull) == 0ull) {
            fprintf(stderr,
                    "MGL TRACE GenTextures call=%llu generated=%u currentName=%u tableCount=%zu tableCap=%zu\n",
                    (unsigned long long)call_id,
                    name,
                    STATE(texture_table).current_name,
                    STATE(texture_table).count,
                    STATE(texture_table).size);
        }

        // TEX_OBJ_RES_NAME has special name.. skip it
        if (STATE(texture_table.current_name) == TEX_OBJ_RES_NAME)
            getNewName(&STATE(texture_table));
    }
}

void mglCreateTextures(GLMContext ctx, GLenum target, GLsizei n, GLuint *textures)
{
    if (!ctx)
        return;

    if (!mglIsTextureObjectTarget(target))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (n < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (n == 0)
        return;

    if (!textures)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

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

    if (!ctx)
        return;

    if (MGL_VERBOSE_TEXTURE_BIND_LOGS) {
        fprintf(stderr,
                "MGL TRACE BindTexture target=0x%x texture=%u activeUnit=%u ctx=%p\n",
                target,
                texture,
                ctx ? ctx->state.active_texture : 0u,
                (void *)ctx);
    }

    if (!mglIsTextureObjectTarget(target))
    {
        fprintf(stderr, "MGL Error: mglBindTexture: invalid target 0x%x\n", (unsigned)target);
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    index = textureIndexFromTarget(ctx, target);
    if (index == _MAX_TEXTURE_TYPES)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
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
    if (active_texture >= TEXTURE_UNITS) {
        fprintf(stderr,
                "MGL ERROR: mglBindTexture active unit out of range unit=%u target=0x%x texture=%u\n",
                active_texture,
                target,
                texture);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    Texture *old_typed_ptr = STATE(texture_units[active_texture].textures[index]);
    Texture *old_active_ptr = STATE(active_textures[active_texture]);
    if (old_typed_ptr != ptr || (ptr && old_active_ptr != ptr) ||
        (ptr && mglPendingDrawsWriteTexture(ctx, ptr))) {
        mglFlushPendingDraws(ctx);
    }

    STATE(texture_units[active_texture].textures[index]) = ptr;
    if (ptr) {
        STATE(active_textures[active_texture]) = ptr;
    }
    mglRecordLastSampled2DTexture(ctx, active_texture, ptr);
    mglUpdateTextureUnitActiveMask(ctx, active_texture);
    STATE(dirty_bits) |= DIRTY_TEX | DIRTY_TEX_BINDING;

    mglTraceTextureUnitState(ctx, ptr ? "BindTexture" : "BindTexture.unbindTarget", active_texture, target, texture, STATE(active_textures[active_texture]));
}

void mglBindImageTexture(GLMContext ctx, GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum internalformat)
{
    Texture *ptr;

    if (unit >= TEXTURE_UNITS) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: unit >= TEXTURE_UNITS (%d)\n", unit);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (texture == 0u) {
        bzero(&ctx->state.image_units[unit], sizeof(ImageUnit));
        ctx->state.dirty_bits |= DIRTY_IMAGE_UNIT_STATE;
        return;
    }

    ptr = getTex(ctx, texture, 0);

    if (!ptr) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: texture %d not found\n", texture);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (level < 0) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: level < 0 (%d)\n", level);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!layered && layer < 0) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: layer < 0 (%d)\n", layer);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
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
            return;
    }

    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: invalid internalformat 0x%x\n", internalformat);
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (ptr->internalformat != internalformat) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: internalformat mismatch (tex=0x%x req=0x%x)\n", ptr->internalformat, internalformat);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (level >= (GLint)ptr->num_levels) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: level >= num_levels (%d >= %d)\n", level, ptr->num_levels);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!ptr->faces[0].levels || !ptr->faces[0].levels[level].complete) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: incomplete level %d for texture %u\n", level, texture);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!layered && layer >= (GLint)ptr->faces[0].levels[level].depth) {
        fprintf(stderr, "MGL Error: mglBindImageTexture: layer %d out of range depth=%u\n",
                layer,
                ptr->faces[0].levels[level].depth);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
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

    mglFlushPendingDraws(ctx);

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
                GLboolean cleared_unit = GL_FALSE;

                if(ctx->state.active_textures[i] == tex) {
                    ctx->state.active_textures[i] = NULL;
                    cleared_unit = GL_TRUE;
                }
                if(ctx->state.last_sampled_2d_textures[i] == tex) {
                    ctx->state.last_sampled_2d_textures[i] = NULL;
                    cleared_unit = GL_TRUE;
                }

                for (int target_index = 0; target_index < _MAX_TEXTURE_TYPES; target_index++) {
                    if (ctx->state.texture_units[i].textures[target_index] == tex) {
                        ctx->state.texture_units[i].textures[target_index] = NULL;
                        cleared_unit = GL_TRUE;
                    }
                }

                if (cleared_unit) {
                    mglUpdateTextureUnitActiveMask(ctx, (GLuint)i);
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

            deleteHashElement(&STATE(texture_table), name);
            invalidateTexture(ctx, tex);
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
    Texture *tex = findTexture(ctx, texture);

    if (!tex) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (level < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (tex->num_levels > 0 && level >= (GLint)tex->num_levels) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    /* Invalidation is a hint; after validation it is legal to leave storage as-is. */
}

void mglInvalidateTexSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth)
{
    Texture *tex = findTexture(ctx, texture);

    if (!tex) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (level < 0 || xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width < 0 || height < 0 || depth < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (tex->num_levels > 0 && level >= (GLint)tex->num_levels) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (tex->faces[0].levels && level < (GLint)tex->num_levels) {
        TextureLevel *lvl = &tex->faces[0].levels[level];
        if (lvl->complete &&
            (xoffset > (GLint)lvl->width ||
             yoffset > (GLint)lvl->height ||
             zoffset > (GLint)lvl->depth ||
             width > (GLsizei)(lvl->width - (GLuint)xoffset) ||
             height > (GLsizei)(lvl->height - (GLuint)yoffset) ||
             depth > (GLsizei)(lvl->depth - (GLuint)zoffset))) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
    }

    /* Invalidation is a hint; after validation it is legal to leave storage as-is. */
}

void mglBindImageTextures(GLMContext ctx, GLuint first, GLsizei count, const GLuint *textures)
{
    if (!ctx) {
        return;
    }
    if (count < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (count == 0) {
        return;
    }
    if (first >= TEXTURE_UNITS || (GLuint)count > TEXTURE_UNITS - first) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < count; i++) {
        GLuint tex_name = textures ? textures[i] : 0u;
        if (tex_name == 0u) {
            continue;
        }

        Texture *tex = findTexture(ctx, tex_name);
        if (!tex || tex->num_levels == 0 || !tex->faces[0].levels ||
            !tex->faces[0].levels[0].complete) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    for (GLsizei i = 0; i < count; i++) {
        GLuint tex_name = textures ? textures[i] : 0u;
        if (tex_name == 0u) {
            bzero(&ctx->state.image_units[first + i], sizeof(ImageUnit));
            continue;
        }

        Texture *tex = findTexture(ctx, tex_name);
        mglBindImageTexture(ctx,
                            first + i,
                            tex_name,
                            0,
                            GL_FALSE,
                            0,
                            tex->access ? tex->access : GL_READ_ONLY,
                            tex->internalformat);
    }

    ctx->state.dirty_bits |= DIRTY_IMAGE_UNIT_STATE;
}

void mglClientActiveTexture(GLMContext ctx, GLenum texture)
{
    if (texture < GL_TEXTURE0 ||
        (GLuint)(texture - GL_TEXTURE0) >= STATE_VAR(max_combined_texture_image_units)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    /* Legacy fixed-function client texture state is otherwise unused by MGL. */
}

void mglActiveTexture(GLMContext ctx, GLenum texture)
{
    GLuint unit;

    if (texture < GL_TEXTURE0)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    unit = (GLuint)(texture - GL_TEXTURE0);

    if (unit >= TEXTURE_UNITS || unit >= STATE_VAR(max_combined_texture_image_units))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    STATE(active_texture) = unit;
    ctx->state.dirty_bits |= DIRTY_TEX_BINDING;
    mglTraceTextureUnitState(ctx, "ActiveTexture", unit, 0, 0, STATE(active_textures[unit]));
}

void mglBindTextures(GLMContext ctx, GLuint first, GLsizei count, const GLuint *textures)
{
    GLuint old_active_texture;

    if (!ctx || count <= 0) {
        return;
    }

    if (first >= TEXTURE_UNITS || (GLuint)count > TEXTURE_UNITS - first) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    old_active_texture = STATE(active_texture);

    for (int i=0; i < count; i++)
    {
        GLuint texture;
        GLuint unit = first + (GLuint)i;

        if (textures == NULL)
        {
            texture = 0;
        }
        else
        {
            texture = textures[i];
        }

        if (texture != 0)
        {
            Texture *ptr;
            GLuint index;

            ptr = findTexture(ctx, texture);
            if (!ptr) {
                fprintf(stderr,
                        "MGL ERROR: mglBindTextures unknown texture=%u unit=%u first=%u count=%d\n",
                        texture,
                        unit,
                        first,
                        count);
                ERROR_RETURN(GL_INVALID_OPERATION);
                STATE(active_texture) = old_active_texture;
                return;
            }

            index = ptr->index;
            if (index >= _MAX_TEXTURE_TYPES) {
                ERROR_RETURN(GL_INVALID_OPERATION);
                STATE(active_texture) = old_active_texture;
                return;
            }

            Texture *old_typed_ptr = STATE(texture_units[unit].textures[index]);
            Texture *old_active_ptr = STATE(active_textures[unit]);
            if (old_typed_ptr != ptr || old_active_ptr != ptr ||
                mglPendingDrawsWriteTexture(ctx, ptr)) {
                mglFlushPendingDraws(ctx);
            }

            STATE(texture_units[unit].textures[index]) = ptr;
            STATE(active_textures[unit]) = ptr;
            mglRecordLastSampled2DTexture(ctx, unit, ptr);
            mglUpdateTextureUnitActiveMask(ctx, unit);
            mglTraceTextureUnitState(ctx, "BindTextures", unit, ptr->target, texture, ptr);
        }
        else
        {
            GLboolean had_binding = GL_FALSE;
            for(GLuint index=0; index<_MAX_TEXTURE_TYPES; index++)
            {
                if (STATE(texture_units[unit].textures[index]) != NULL) {
                    had_binding = GL_TRUE;
                }
            }
            if (STATE(active_textures[unit]) != NULL) {
                had_binding = GL_TRUE;
            }
            if (had_binding) {
                mglFlushPendingDraws(ctx);
            }
            for(GLuint index=0; index<_MAX_TEXTURE_TYPES; index++)
            {
                STATE(texture_units[unit].textures[index]) = NULL;
            }
            STATE(active_textures[unit]) = NULL;
            mglUpdateTextureUnitActiveMask(ctx, unit);
            mglTraceTextureUnitState(ctx, "BindTextures.unbind", unit, 0, 0, NULL);
        }
    }

    STATE(active_texture) = old_active_texture;
    STATE(dirty_bits) |= DIRTY_TEX | DIRTY_TEX_BINDING;
}

void mglBindTextureUnit(GLMContext ctx, GLuint unit, GLuint texture)
{
    Texture *ptr;
    GLuint index;

    if (!ctx) {
        return;
    }

    if (unit >= TEXTURE_UNITS) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (texture == 0) {
        GLboolean had_binding = GL_FALSE;
        for (index = 0; index < _MAX_TEXTURE_TYPES; index++) {
            if (STATE(texture_units[unit].textures[index]) != NULL) {
                had_binding = GL_TRUE;
            }
        }
        if (STATE(active_textures[unit]) != NULL) {
            had_binding = GL_TRUE;
        }
        if (had_binding) {
            mglFlushPendingDraws(ctx);
        }
        for (index = 0; index < _MAX_TEXTURE_TYPES; index++) {
            STATE(texture_units[unit].textures[index]) = NULL;
        }
        STATE(active_textures[unit]) = NULL;
        mglUpdateTextureUnitActiveMask(ctx, unit);
        STATE(dirty_bits) |= DIRTY_TEX | DIRTY_TEX_BINDING;
        mglTraceTextureUnitState(ctx, "BindTextureUnit.unbind", unit, 0, 0, NULL);
        return;
    }

    ptr = findTexture(ctx, texture);
    if (!ptr) {
        fprintf(stderr,
                "MGL ERROR: mglBindTextureUnit unknown texture=%u unit=%u\n",
                texture,
                unit);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    index = ptr->index;
    if (index >= _MAX_TEXTURE_TYPES) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    Texture *old_typed_ptr = STATE(texture_units[unit].textures[index]);
    Texture *old_active_ptr = STATE(active_textures[unit]);
    if (old_typed_ptr != ptr || old_active_ptr != ptr ||
        mglPendingDrawsWriteTexture(ctx, ptr)) {
        mglFlushPendingDraws(ctx);
    }

    STATE(texture_units[unit].textures[index]) = ptr;
    STATE(active_textures[unit]) = ptr;
    mglRecordLastSampled2DTexture(ctx, unit, ptr);
    mglUpdateTextureUnitActiveMask(ctx, unit);
    STATE(dirty_bits) |= DIRTY_TEX | DIRTY_TEX_BINDING;
    mglTraceTextureUnitState(ctx, "BindTextureUnit", unit, ptr->target, texture, ptr);
}

void generateMipmaps(GLMContext ctx, GLuint texture, GLenum target)
{
    Texture *ptr;
    GLboolean needs_storage_update;

    ptr = getTex(ctx, texture, target);

    ERROR_CHECK_RETURN(ptr, GL_INVALID_OPERATION);

    // level 0 needs to be filled out for mipmap geneation
    ERROR_CHECK_RETURN(ptr->faces[0].levels[0].complete, GL_INVALID_OPERATION);

    mglFlushPendingDraws(ctx);

    needs_storage_update = (!ptr->mtl_data || !ptr->mipmapped || !ptr->genmipmaps);

    ptr->mipmapped = true;
    ptr->genmipmaps = true;

    if (needs_storage_update) {
        ptr->dirty_bits |= DIRTY_TEXTURE_LEVEL;
    }

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
    if (!ctx || !tex)
        return;

    GLuint old_name = tex->name;
    GLenum old_target = tex->target;
    GLuint old_index = tex->index;
    GLuint level_count = tex->mipmap_levels ? tex->mipmap_levels : tex->num_levels;

    if (level_count > 1024u) {
        fprintf(stderr,
                "MGL WARNING: invalidateTexture suspicious level count tex=%u target=0x%x levels=%u mipmapLevels=%u numLevels=%u; clamping cleanup\n",
                tex->name,
                tex->target,
                level_count,
                tex->mipmap_levels,
                tex->num_levels);
        level_count = 1024u;
    }

    fprintf(stderr,
            "MGL TRACE invalidateTexture tex=%u target=0x%x index=%u levels=%u mtl=%p\n",
            tex->name,
            tex->target,
            tex->index,
            level_count,
            tex->mtl_data);

    mglClearLastSampled2DTextureIfMatches(ctx, tex);

    if (tex->mtl_data)
    {
        if (ctx->mtl_funcs.mtlDeleteMTLObj) {
            ctx->mtl_funcs.mtlDeleteMTLObj(ctx, tex->mtl_data);
        } else {
            fprintf(stderr,
                    "MGL WARNING: invalidateTexture cannot release Metal texture tex=%u because mtlDeleteMTLObj is NULL\n",
                    tex->name);
        }
        tex->mtl_data = NULL;
    }

    if (tex->params.mtl_data)
    {
        if (ctx->mtl_funcs.mtlDeleteMTLObj) {
            ctx->mtl_funcs.mtlDeleteMTLObj(ctx, tex->params.mtl_data);
        } else {
            fprintf(stderr,
                    "MGL WARNING: invalidateTexture cannot release Metal sampler params tex=%u because mtlDeleteMTLObj is NULL\n",
                    tex->name);
        }
        tex->params.mtl_data = NULL;
    }

    for(int face=0; face<_CUBE_MAP_MAX_FACE; face++)
    {
        TextureLevel *levels = tex->faces[face].levels;
        if (!levels)
            continue;

        for(GLuint i=0; i<level_count; i++)
        {
            TextureLevel *lvl = &levels[i];
            if (lvl->data)
            {
                size_t dealloc_size = lvl->data_size;
                if (dealloc_size == 0u && lvl->pitch > 0u && lvl->height > 0u && lvl->depth > 0u) {
                    size_t rows_size = lvl->pitch * (size_t)lvl->height;
                    dealloc_size = page_size_align(rows_size * (size_t)lvl->depth);
                }

                if (dealloc_size > 0u) {
                    kern_return_t kr = vm_deallocate((vm_map_t)mach_task_self(),
                                                     lvl->data,
                                                     dealloc_size);
                    if (kr != KERN_SUCCESS) {
                        fprintf(stderr,
                                "MGL WARNING: invalidateTexture vm_deallocate failed tex=%u face=%d level=%u data=%p size=%zu kr=%d\n",
                                old_name,
                                face,
                                i,
                                (void *)(uintptr_t)lvl->data,
                                dealloc_size,
                                kr);
                    }
                } else {
                    fprintf(stderr,
                            "MGL WARNING: invalidateTexture skipping CPU texture storage release with unknown size tex=%u face=%d level=%u data=%p\n",
                            old_name,
                            face,
                            i,
                            (void *)(uintptr_t)lvl->data);
                }

                lvl->data = 0;
                lvl->data_size = 0;
            }
        }
    }

    for(int i=0; i<_CUBE_MAP_MAX_FACE; i++)
    {
        free(tex->faces[i].levels);
        tex->faces[i].levels = NULL;
    }

    tex->dirty_bits = 0;
    tex->dirty_on_gpu = 0;
    tex->is_render_target = GL_FALSE;
    tex->immutable_storage = GL_FALSE;
    tex->mipmapped = 0;
    tex->genmipmaps = GL_FALSE;
    tex->mtl_requires_private_storage = GL_FALSE;
    tex->internalformat = 0;
    tex->width = 0;
    tex->height = 0;
    tex->depth = 0;
    tex->is_array = GL_FALSE;
    tex->complete = GL_FALSE;
    tex->num_levels = 0;
    tex->mipmap_levels = 0;
    tex->texture_buffer = NULL;
    tex->texture_buffer_offset = 0;
    tex->texture_buffer_size = 0;
    tex->debug_label[0] = '\0';

    /* Keep object identity intact for glTexImage redefinition paths. */
    tex->name = old_name;
    tex->target = old_target;
    tex->index = old_index;
}

void initBaseTexLevel(GLMContext ctx, Texture *tex, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    tex->mipmapped = 0;
    tex->mipmap_levels = ilog2(MAX(width, height)) + 1;

    for(int face=0; face<_CUBE_MAP_MAX_FACE; face++)
    {
        // CRITICAL SECURITY FIX: Prevent integer overflow in mipmap allocation
        if ((size_t)tex->mipmap_levels > SIZE_MAX / sizeof(TextureLevel)) {
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

static bool ensureTextureLevelCapacity(GLMContext ctx, Texture *tex, GLuint required_levels)
{
    TextureLevel *new_levels[_CUBE_MAP_MAX_FACE] = {0};
    GLuint new_capacity;

    if (!ctx || !tex || required_levels == 0)
        return false;

    if (required_levels <= tex->mipmap_levels) {
        for (int face = 0; face < _CUBE_MAP_MAX_FACE; face++) {
            if (!tex->faces[face].levels) {
                break;
            }
            if (face == _CUBE_MAP_MAX_FACE - 1) {
                return true;
            }
        }
    }

    new_capacity = MAX(required_levels, tex->mipmap_levels);

    if ((size_t)new_capacity > SIZE_MAX / sizeof(TextureLevel)) {
        fprintf(stderr,
                "MGL ERROR: texture level grow overflow tex=%u required=%u old=%u\n",
                tex->name,
                new_capacity,
                tex->mipmap_levels);
        STATE(error) = GL_OUT_OF_MEMORY;
        return false;
    }

    for (int face = 0; face < _CUBE_MAP_MAX_FACE; face++) {
        new_levels[face] = (TextureLevel *)calloc(new_capacity, sizeof(TextureLevel));
        if (!new_levels[face]) {
            for (int i = 0; i < face; i++) {
                free(new_levels[i]);
            }
            STATE(error) = GL_OUT_OF_MEMORY;
            return false;
        }

        if (tex->faces[face].levels && tex->mipmap_levels > 0) {
            memcpy(new_levels[face],
                   tex->faces[face].levels,
                   tex->mipmap_levels * sizeof(TextureLevel));
        }
    }

    for (int face = 0; face < _CUBE_MAP_MAX_FACE; face++) {
        free(tex->faces[face].levels);
        tex->faces[face].levels = new_levels[face];
    }

    fprintf(stderr,
            "MGL TRACE texture level capacity grow tex=%u old=%u new=%u base=%ux%u target=0x%x\n",
            tex->name,
            tex->mipmap_levels,
            new_capacity,
            tex->width,
            tex->height,
            tex->target);

    tex->mipmap_levels = new_capacity;
    return true;
}

bool checkTexLevelParams(GLMContext ctx, Texture *tex, GLint level, GLuint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type)
{
    GLuint base_width, base_height;

    if (!tex || tex->mipmap_levels == 0)
    {
        fprintf(stderr,
                "MGL ERROR: checkTexLevelParams before base level tex=%p level=%d size=%dx%dx%d\n",
                (void *)tex,
                level,
                width,
                height,
                depth);
        return false;
    }

    if (tex->target == GL_TEXTURE_2D)
    {
        if (level != 0)
        {
            GLint check_level = level;
            base_width = tex->width;
            base_height = tex->height;

            while(check_level--)
            {
                base_width = MAX(base_width >> 1, 1u);
                base_height = MAX(base_height >> 1, 1u);
            }

            if (width != base_width || height != base_height)
            {
                fprintf(stderr,
                        "MGL ERROR: checkTexLevelParams size mismatch tex=%u level=%d got=%dx%d expected=%ux%u base=%ux%u\n",
                        tex->name,
                        level,
                        width,
                        height,
                        base_width,
                        base_height,
                        tex->width,
                        tex->height);
                return false;
            }
        }
    }

    if (internalformat)
    {
        // internal formats don't jive
        if (internalformat != tex->internalformat)
        {
            fprintf(stderr,
                    "MGL ERROR: checkTexLevelParams internalformat mismatch tex=%u level=%d got=0x%x expected=0x%x\n",
                    tex->name,
                    level,
                    internalformat,
                    tex->internalformat);
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
            fprintf(stderr,
                    "MGL ERROR: checkTexLevelParams format/type mismatch tex=%u level=%d derived=0x%x expected=0x%x format=0x%x type=0x%x\n",
                    tex->name,
                    level,
                    temp_internalformat,
                    tex->internalformat,
                    format,
                    type);
            return false;
        }
    }

    if (checkInternalFormatForMetal(ctx, tex->internalformat) == false)
    {
        fprintf(stderr,
                "MGL ERROR: checkTexLevelParams unsupported internalformat=0x%x level=%d size=%dx%dx%d\n",
                tex->internalformat,
                level,
                width,
                height,
                depth);
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


void unpackTexture(GLMContext ctx, Texture *tex, GLuint face, GLuint level, void *src_data, void *dst_data, size_t src_pitch, size_t src_image_pitch, size_t pixel_size, size_t xoffset, size_t yoffset, size_t zoffset, size_t width, size_t height, size_t depth)
{
    GLubyte *src, *dst;
    size_t dst_pitch;

    if (!tex || face >= _CUBE_MAP_MAX_FACE || !tex->faces[face].levels || level >= tex->mipmap_levels)
    {
        fprintf(stderr,
                "MGL WARNING: unpackTexture skipped invalid texture level tex=%p face=%u level=%u levels=%u\n",
                (void *)tex,
                face,
                level,
                tex ? tex->mipmap_levels : 0u);
        return;
    }

    if (!src_data || !dst_data || pixel_size == 0u || width == 0u || height == 0u || depth == 0u)
    {
        fprintf(stderr,
                "MGL WARNING: unpackTexture skipped invalid copy args tex=%u src=%p dst=%p pixelSize=%zu size=%zux%zux%zu\n",
                tex->name,
                src_data,
                dst_data,
                pixel_size,
                width,
                height,
                depth);
        return;
    }

    src = (GLubyte *)src_data;
    dst = (GLubyte *)dst_data;

    dst_pitch = tex->faces[face].levels[level].pitch;
    if (dst_pitch == 0u)
    {
        fprintf(stderr,
                "MGL WARNING: unpackTexture skipped level with zero destination pitch tex=%u face=%u level=%u\n",
                tex->name,
                face,
                level);
        return;
    }

    if (xoffset || yoffset || zoffset)
    {
        size_t xoffset_bytes = xoffset * pixel_size; // num pixels
        size_t yoffset_bytes = yoffset * dst_pitch; // num lines (rows * bytes_per_row)
        size_t level_height = tex->faces[face].levels[level].height;
        size_t zoffset_bytes = zoffset * dst_pitch * level_height; // num planes

        dst += xoffset_bytes;
        dst += yoffset_bytes;
        dst += zoffset_bytes;
    }

    if (depth > 1)
    {
        // 3d texture
        size_t copy_size = width * pixel_size;
        size_t dst_image_pitch = dst_pitch * tex->faces[face].levels[level].height;
        if (src_image_pitch == 0u) {
            src_image_pitch = src_pitch * height;
        }

        for(size_t z=0; z<depth; z++)
        {
            GLubyte *src_slice = src + z * src_image_pitch;
            GLubyte *dst_slice = dst + z * dst_image_pitch;
            for(size_t y=0; y<height; y++)
            {
                memcpy(dst_slice, src_slice, copy_size);
                src_slice += src_pitch;
                dst_slice += dst_pitch;
            }
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
    if (!tex)
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (level < 0 || width < 0 || height < 0 || depth < 0)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if (!proxy) {
        mglFlushPendingDraws(ctx);
    }

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

    /*
     * Minecraft/LWJGL can submit a zero-sized tail mip (for example 16x16
     * level 5 -> 0x0) while walking a mip chain. GL accepts non-negative
     * dimensions, but there is no Metal/CPU storage to create for a zero-sized
     * image. Treat this as a successful no-op instead of turning it into a
     * repeated INVALID_OPERATION from checkTexLevelParams().
     */
    if (level > 0 && (width == 0 || height == 0 || depth == 0))
    {
        static uint64_t s_zero_tail_mip_logs = 0;
        uint64_t hit = ++s_zero_tail_mip_logs;

        if (hit <= 64ull || (hit % 1024ull) == 0ull)
        {
            fprintf(stderr,
                    "MGL TRACE createTextureLevel skip zero-sized tail mip tex=%u target=0x%x face=%u level=%d size=%dx%dx%d base=%ux%ux%u numLevels=%u mipmapLevels=%u hit=%llu\n",
                    tex ? tex->name : 0u,
                    tex ? tex->target : 0u,
                    face,
                    level,
                    width,
                    height,
                    depth,
                    tex ? tex->width : 0u,
                    tex ? tex->height : 0u,
                    tex ? tex->depth : 0u,
                    tex ? tex->num_levels : 0u,
                    tex ? tex->mipmap_levels : 0u,
                    (unsigned long long)hit);
        }

        return true;
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

    if (!ensureTextureLevelCapacity(ctx, tex, (GLuint)level + 1u))
    {
        fprintf(stderr,
                "MGL ERROR: createTextureLevel failed to grow levels tex=%u face=%u level=%d currentLevels=%u size=%dx%dx%d\n",
                tex ? tex->name : 0u,
                face,
                level,
                tex ? tex->mipmap_levels : 0u,
                width,
                height,
                depth);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
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

    if (width == 0 || height == 0 || depth == 0)
    {
        fprintf(stderr,
                "MGL TRACE createTextureLevel zero-sized image tex=%u target=0x%x face=%u level=%d size=%dx%dx%d\n",
                tex->name,
                tex->target,
                face,
                level,
                width,
                height,
                depth);
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
        tex->faces[face].levels[level].complete = false;
        tex->complete = false;
        tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
        STATE(dirty_bits) |= DIRTY_TEX;
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
    if (texture_size == 0u)
    {
        fprintf(stderr,
                "MGL ERROR: createTextureLevel computed zero allocation tex=%u level=%d internal=0x%x size=%dx%dx%d pixelSize=%zu\n",
                tex->name,
                level,
                internalformat,
                width,
                height,
                depth,
                pixel_size);
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

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

    bool has_upload_source = (pixels != NULL) || (STATE(buffers[_PIXEL_UNPACK_BUFFER]) != NULL);

    if (tex->mtl_requires_private_storage == false)
    {
        // Allocate directly from VM
        err = vm_allocate((vm_map_t) mach_task_self(),
                          (vm_address_t*) &texture_data,
                          texture_size,
                          VM_FLAGS_ANYWHERE);
        if (err != 0 || !texture_data)
        {
            fprintf(stderr,
                    "MGL ERROR: createTextureLevel vm_allocate failed err=%d bytes=%zu tex=%u level=%d\n",
                    err,
                    texture_size,
                    tex->name,
                    level);
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }

        tex->faces[face].levels[level].data_size = texture_size;
        tex->faces[face].levels[level].data = (vm_address_t)texture_data;

        if (has_upload_source)
        {
            size_t source_pixel_size = sizeForFormatType(format, type);
            MGLTextureUnpackLayout unpack_layout;
            const uint8_t *resolved_src = NULL;
            Buffer *resolved_unpack_buf = NULL;

            ERROR_CHECK_RETURN_VALUE(source_pixel_size > 0u, GL_INVALID_ENUM, false);

            if (!mglComputeTextureUnpackLayout(ctx,
                                               width,
                                               height,
                                               depth,
                                               source_pixel_size,
                                               "createTextureLevel",
                                               &unpack_layout)) {
                return false;
            }

            if (!mglResolveTexSubImageSource(ctx,
                                             tex,
                                             face,
                                             level,
                                             0,
                                             0,
                                             0,
                                             width,
                                             height,
                                             depth,
                                             format,
                                             type,
                                             pixels,
                                             unpack_layout.skip_offset_bytes,
                                             unpack_layout.required_bytes,
                                             false,
                                             &resolved_src,
                                             &resolved_unpack_buf)) {
                return false;
            }

            if (!resolved_src) {
                fprintf(stderr,
                        "MGL createTextureLevel skip upload: resolved source is NULL tex=%u target=0x%x %dx%dx%d\n",
                        tex->name,
                        tex->target,
                        width,
                        height,
                        depth);
                has_upload_source = false;
            } else {
                src_pitch = unpack_layout.src_pitch;
                unpackTexture(ctx,
                              tex,
                              face,
                              level,
                              (void *)resolved_src,
                              (void *)texture_data,
                              src_pitch,
                              unpack_layout.src_image_size,
                              source_pixel_size,
                              0,
                              0,
                              0,
                              width,
                              height,
                              depth);

                tex->faces[face].levels[level].last_upload_size = unpack_layout.required_bytes;
                tex->faces[face].levels[level].last_src_hash = mglHashBytesSampled(resolved_src, unpack_layout.required_bytes);
            }

            if (resolved_src) {
                tex->faces[face].levels[level].last_init_source = resolved_unpack_buf ? kTexImagePBO : kTexImageCopy;
                tex->faces[face].levels[level].last_src_ptr = resolved_src;
                tex->faces[face].levels[level].ever_written = GL_TRUE;
                tex->faces[face].levels[level].has_initialized_data = GL_TRUE;
                tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;
            }

            if (unpack_layout.required_bytes > 0u && resolved_src) {
                if (mglLooksAllZeroSampled(resolved_src, unpack_layout.required_bytes)) {
                    size_t first_nonzero = 0u;
                    uint8_t first_value = 0u;
                    bool has_nonzero = mglFindFirstNonZeroByte(resolved_src,
                                                               unpack_layout.required_bytes,
                                                               &first_nonzero,
                                                               &first_value);
                    if (!has_nonzero) {
                        tex->faces[face].levels[level].suspicious_zero_upload = GL_FALSE;
                        tex->faces[face].levels[level].has_initialized_data = GL_TRUE;
                    }
                    fprintf(stderr,
                            "MGL WARNING: createTextureLevel upload sampled head/mid/tail all-zero tex=%u face=%u level=%d bytes=%zu src=%p fullZero=%d firstNonZero=0x%zx value=0x%02x\n",
                            tex->name,
                            face,
                            level,
                            unpack_layout.required_bytes,
                            resolved_src,
                            has_nonzero ? 0 : 1,
                            has_nonzero ? first_nonzero : 0u,
                            has_nonzero ? first_value : 0u);
                }
            }

            if (resolved_src) {
                tex->dirty_bits |= DIRTY_TEXTURE_DATA;
            }
        };
    }

    tex->faces[face].levels[level].complete = true;
    if (!has_upload_source) {
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

    if (!proxy) {
        mglRecordBoundSampled2DTextureIfReady(ctx, tex);
    }

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

    ERROR_CHECK_RETURN(width >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(height >= 0, GL_INVALID_VALUE);

    if (proxy)
    {
        if (border != 0) {
            STATE(error) = GL_INVALID_VALUE;
            mglHandleProxyTexImageQuery(ctx, target, level, internalformat, 0, 0, 0, border);
            return;
        }
        mglHandleProxyTexImageQuery(ctx, target, level, internalformat, width, height, 1, border);
        return;
    }

    // verifyFormatType sets the error
    ERROR_CHECK_RETURN(verifyInternalFormatAndFormatType(ctx, internalformat, format, type), 0);

    ERROR_CHECK_RETURN(border == 0, GL_INVALID_VALUE);

    tex = getTex(ctx, 0, target);

    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);

    tex->access = GL_READ_ONLY;

    if (pixels == NULL && !STATE(buffers[_PIXEL_UNPACK_BUFFER]))
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
    (void)fixedsamplelocations;

    if (target != GL_TEXTURE_2D_MULTISAMPLE &&
        target != GL_PROXY_TEXTURE_2D_MULTISAMPLE) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (samples < 1 || width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (target == GL_PROXY_TEXTURE_2D_MULTISAMPLE) {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, 0, 0, 0, 0);
        return;
    }

    fprintf(stderr,
            "MGL WARNING: glTexImage2DMultisample unsupported target=0x%x samples=%d internalformat=0x%x %dx%d\n",
            target, samples, internalformat, width, height);
    ERROR_RETURN(GL_INVALID_OPERATION);
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
    (void)fixedsamplelocations;

    if (target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY &&
        target != GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (samples < 1 || width < 0 || height < 0 || depth < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (target == GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        mglHandleProxyTexImageQuery(ctx, target, 0, internalformat, 0, 0, 0, 0);
        return;
    }

    fprintf(stderr,
            "MGL WARNING: glTexImage3DMultisample unsupported target=0x%x samples=%d internalformat=0x%x %dx%dx%d\n",
            target, samples, internalformat, width, height, depth);
    ERROR_RETURN(GL_INVALID_OPERATION);
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
    Buffer *initial_unpack_buf = STATE(buffers[_PIXEL_UNPACK_BUFFER]);
    GLuint initial_unpack_name = initial_unpack_buf ? initial_unpack_buf->name : 0u;
    bool trace_upload = mglShouldTraceTextureUpload(tex,
                                                    initial_unpack_name,
                                                    width,
                                                    height,
                                                    depth,
                                                    0u);
    mglFlushPendingDraws(ctx);

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
    size_t src_pitch;
    size_t src_image_size;
    size_t skip_offset_bytes;
    size_t required_bytes;
    MGLTextureUnpackLayout unpack_layout;

    pixel_size = sizeForFormatType(format, type);
    ERROR_CHECK_RETURN_VALUE(pixel_size > 0u, GL_INVALID_ENUM, false);

    if (!mglComputeTextureUnpackLayout(ctx,
                                       width,
                                       height,
                                       depth,
                                       pixel_size,
                                       "texSubImage",
                                       &unpack_layout)) {
        return false;
    }

    src_pitch = unpack_layout.src_pitch;
    src_image_size = unpack_layout.src_image_size;
    skip_offset_bytes = unpack_layout.skip_offset_bytes;
    required_bytes = unpack_layout.required_bytes;

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

    if (resolved_unpack_buf) {
        resolved_src_hash = mglHashBytesSampled(resolved_src, required_bytes);
    } else {
        /*
         * CPU uploads can point at a direct/native buffer whose exact mapped
         * extent is unknown to us. Avoid diagnostic scans here; the actual
         * row-wise unpack below is the first safe consumer, and the destination
         * texture backing store can be inspected after that.
         */
        resolved_src_hash = 0u;
    }

    if (resolved_unpack_buf && mglLooksAllZeroSampled(resolved_src, required_bytes)) {
        static uint64_t s_zero_upload_warning_count = 0u;
        uint64_t zero_warning_id = ++s_zero_upload_warning_count;
        size_t first_nonzero = 0u;
        uint8_t first_value = 0u;
        bool has_nonzero = mglFindFirstNonZeroByte(resolved_src,
                                                   required_bytes,
                                                   &first_nonzero,
                                                   &first_value);
        if (trace_upload || zero_warning_id <= 32u || (zero_warning_id % 512u) == 0u) {
            fprintf(stderr,
                    "MGL WARNING: texSubImage source sampled head/mid/tail all-zero tex=%u face=%u level=%d required=%zu src=%p fullZero=%d firstNonZero=0x%zx value=0x%02x warn=%" PRIu64 "\n",
                    tex->name,
                    face,
                    level,
                    required_bytes,
                    resolved_src,
                    has_nonzero ? 0 : 1,
                    has_nonzero ? first_nonzero : 0u,
                    has_nonzero ? first_value : 0u,
                    zero_warning_id);
            if (has_nonzero && trace_upload) {
                size_t dump_offset = first_nonzero;
                size_t dump_available = required_bytes - first_nonzero;
                if (dump_available > 64u) {
                    dump_available = 64u;
                }
                mglDumpBytesToStderr("texSubImage.source.firstNonZero", resolved_src + dump_offset, dump_available, dump_offset);
            }
        }
    }

    if (tex->name == 13u && s_dumped_tex13_upload_source == 0 && resolved_src && resolved_unpack_buf) {
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
    size_t compact_upload_bytes = 0u;
    size_t compact_upload_row_bytes = 0u;
    if (!mglMulSizeT((size_t)width, pixel_size, &compact_upload_row_bytes) ||
        !mglMulSizeT(compact_upload_row_bytes, (size_t)MAX(height, 1), &compact_upload_bytes) ||
        !mglMulSizeT(compact_upload_bytes, (size_t)MAX(depth, 1), &compact_upload_bytes)) {
        fprintf(stderr,
                "MGL ERROR: texSubImage compact byte computation overflow tex=%u dims=%dx%dx%d pixelSize=%zu\n",
                tex->name,
                width,
                height,
                depth,
                pixel_size);
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

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
    
    unpackTexture(ctx, tex, face, level, (void *)resolved_src, texture_data, src_pitch, src_image_size, pixel_size, xoffset, yoffset, zoffset, width, height, depth);

    size_t upload_dst_offset = 0u;
    size_t upload_dst_span = 0u;
    const uint8_t *upload_dst = (const uint8_t *)texture_data;
    size_t dst_pitch = lvl->pitch;
    size_t dst_image_pitch = dst_pitch * (size_t)MAX(lvl->height, 1);
    bool upload_rect_valid = mglTextureRectByteRange(lvl,
                                                     pixel_size,
                                                     (size_t)xoffset,
                                                     (size_t)yoffset,
                                                     (size_t)zoffset,
                                                     (size_t)width,
                                                     (size_t)height,
                                                     (size_t)depth,
                                                     &upload_dst_offset,
                                                     &upload_dst_span);
    if (upload_rect_valid) {
        upload_dst = (const uint8_t *)texture_data + upload_dst_offset;
    }

    uint64_t dst_hash = upload_rect_valid
        ? mglHashTextureRect(upload_dst,
                             dst_pitch,
                             dst_image_pitch,
                             compact_upload_row_bytes,
                             (size_t)MAX(height, 1),
                             (size_t)MAX(depth, 1))
        : mglHashBytesSampled(texture_data, compact_upload_bytes);
    bool upload_rect_zero = upload_rect_valid
        ? mglTextureRectLooksAllZero(upload_dst,
                                     dst_pitch,
                                     dst_image_pitch,
                                     compact_upload_row_bytes,
                                     (size_t)MAX(height, 1),
                                     (size_t)MAX(depth, 1))
        : mglLooksAllZeroSampled((const uint8_t *)texture_data, compact_upload_bytes);
    /*
     * Tiny transparent updates are normal for atlases and mip tails. Keep the
     * zero-upload probe for large CPU uploads where an all-zero rectangle would
     * indicate a real unpack/source lifetime problem.
     */
    bool suspicious_zero_cpu_upload =
        !resolved_unpack_buf &&
        upload_rect_zero &&
        (trace_upload || compact_upload_bytes >= (256u * 1024u));
    if (suspicious_zero_cpu_upload) {
        static uint64_t s_cpu_zero_upload_warning_count = 0u;
        uint64_t zero_warning_id = ++s_cpu_zero_upload_warning_count;
        if (trace_upload || zero_warning_id <= 32u || (zero_warning_id % 512u) == 0u) {
            fprintf(stderr,
                    "MGL WARNING: texSubImage CPU upload produced all-zero uploaded rect tex=%u label=\"%s\" face=%u level=%d required=%zu src=%p dst=%p rectOffset=%zu rectSpan=%zu warn=%" PRIu64 "\n",
                    tex->name,
                    tex->debug_label[0] != '\0' ? tex->debug_label : "(none)",
                    face,
                    level,
                    compact_upload_bytes,
                    resolved_src,
                    upload_dst,
                    upload_dst_offset,
                    upload_dst_span,
                    zero_warning_id);

            mglDumpTexSubImageZeroCpuResourceTag(ctx,
                                                 tex,
                                                 lvl,
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
                                                 resolved_src,
                                                 resolved_unpack_buf,
                                                 required_bytes,
                                                 compact_upload_bytes,
                                                 src_pitch,
                                                 compact_upload_row_bytes,
                                                 pixel_size,
                                                 zero_warning_id);

            if (trace_upload || zero_warning_id <= 8u) {
                mglDumpNativeBacktraceToStderr("texSubImage.zeroCPU", 32u);
            }

            mglRequestJavaThreadDumpForZeroCpuUpload(tex,
                                                     face,
                                                     level,
                                                     width,
                                                     height,
                                                     depth,
                                                     zero_warning_id);

            /*
             * At this point unpackTexture has already consumed the CPU pointer
             * row-by-row without faulting.  Dump only three small windows from
             * source and destination so the next log can prove whether the
             * incoming CPU image is really zero or our unpack path zeroed it.
             */
            size_t dst_total = upload_rect_valid ? upload_dst_span : lvl->data_size;
            size_t dump_dst_pitch = lvl->pitch;
            if (dst_total == 0u) {
                dst_total = compact_upload_bytes;
            }
            if (dump_dst_pitch == 0u) {
                dump_dst_pitch = compact_upload_row_bytes;
            }
            mglDumpTextureUploadSamples(tex,
                                        face,
                                        level,
                                        (const uint8_t *)resolved_src,
                                        required_bytes,
                                        src_pitch,
                                        upload_rect_valid ? upload_dst : (const uint8_t *)texture_data,
                                        dst_total,
                                        dump_dst_pitch,
                                        pixel_size,
                                        width,
                                        height,
                                        depth);
        }
    }

    if (tex->name == 13u && s_dumped_tex13_upload_source == 0) {
        size_t dump_len = compact_upload_bytes;
        if (dump_len > 256u) {
            dump_len = 256u;
        }

        fprintf(stderr,
                "MGL DUMP tex13.texSubImage.dst.begin tex=%u face=%u level=%d target=0x%x fmt=0x%x type=0x%x "
                "dims=%dx%dx%d off=(%d,%d,%d) pixelSize=%zu srcPitch=%zu dumpLen=%zu src=%p dst=%p srcClass=%s\n",
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
                resolved_src,
                texture_data,
                resolved_unpack_buf ? "PBO" : "CPU");

        mglDumpBytesToStderr("tex13.texSubImage.dst", (const uint8_t *)texture_data, dump_len, 0u);
        fprintf(stderr, "MGL DUMP tex13.texSubImage.dst.end tex=%u\n", tex->name);
        s_dumped_tex13_upload_source = 1;
    }

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

        if (src_pitch < unpack_layout.row_copy_bytes) {
            continue;
        }

        size_t metal_min_image_size = 0u;
        if (!mglMulSizeT(src_pitch, (size_t)height, &metal_min_image_size) ||
            src_image_size < metal_min_image_size) {
            continue;
        }

        size_t src_offset;
        size_t src_size;

        const uint8_t *pbo_data = (const uint8_t *)getBufferData(ctx, buf);
        if (!pbo_data || resolved_src < pbo_data) {
            fprintf(stderr,
                    "MGL ERROR: texSubImage PBO direct upload has invalid source base tex=%u unpack=%u base=%p resolved=%p\n",
                    tex->name,
                    buf ? buf->name : 0u,
                    pbo_data,
                    resolved_src);
            break;
        }

        src_offset = (size_t)(resolved_src - pbo_data);
        if (src_offset > (size_t)buf->size) {
            fprintf(stderr,
                    "MGL ERROR: texSubImage PBO direct upload source offset out of range tex=%u unpack=%u offset=%zu size=%lld\n",
                    tex->name,
                    buf ? buf->name : 0u,
                    src_offset,
                    (long long)buf->size);
            break;
        }

        src_size = required_bytes;

        // Preserve cube-map / array target slice information. zoffset is for 3D origin, not array/cube slice.
        ctx->mtl_funcs.mtlTexSubImage(ctx, tex, buf, src_offset, src_pitch, src_image_size, src_size, face, level, width, height, depth, xoffset, yoffset, zoffset);
        lvl->ever_written = GL_TRUE;
        lvl->suspicious_zero_upload = GL_FALSE;
        lvl->has_initialized_data = GL_TRUE;
        lvl->last_init_source = kTexSubImagePBO;
        lvl->last_upload_size = required_bytes;
        lvl->last_src_ptr = resolved_src;
        lvl->last_src_hash = dst_hash;
        mglRecordBoundSampled2DTextureIfReady(ctx, tex);

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
    lvl->suspicious_zero_upload = GL_FALSE;
    lvl->has_initialized_data = GL_TRUE;
    lvl->last_init_source = resolved_unpack_buf ? kTexSubImagePBO : kTexSubImageCPU;
    lvl->last_upload_size = required_bytes;
    lvl->last_src_ptr = resolved_src;
    lvl->last_src_hash = dst_hash;
    mglRecordBoundSampled2DTextureIfReady(ctx, tex);

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

    /*
     * TexStorage declares the exact immutable mip count. initBaseTexLevel()
     * derives a full chain from the base size, which is correct for legacy
     * TexImage completeness but too strict for Minecraft atlases such as
     * 2048x2048x4. Keep the allocated level arrays, but make completeness and
     * the Metal descriptor use the GL-declared storage level count.
     */
    if (levels > 0) {
        tex->mipmap_levels = (GLuint)levels;
        tex->num_levels = MAX(tex->num_levels, (GLuint)levels);
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
    Texture *tex;
    (void)fixedsamplelocations;

    if (samples < 1 || width <= 0 || height <= 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    tex = getTex(ctx, texture, 0);
    if (!tex || tex->target != GL_TEXTURE_2D_MULTISAMPLE) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    fprintf(stderr,
            "MGL WARNING: glTextureStorage2DMultisample unsupported texture=%u samples=%d internalformat=0x%x %dx%d\n",
            texture, samples, internalformat, width, height);
    ERROR_RETURN(GL_INVALID_OPERATION);
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
    Texture *tex;
    (void)fixedsamplelocations;

    if (samples < 1 || width <= 0 || height <= 0 || depth <= 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    tex = getTex(ctx, texture, 0);
    if (!tex || tex->target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    fprintf(stderr,
            "MGL WARNING: glTextureStorage3DMultisample unsupported texture=%u samples=%d internalformat=0x%x %dx%dx%d\n",
            texture, samples, internalformat, width, height, depth);
    ERROR_RETURN(GL_INVALID_OPERATION);
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
    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglCompressedTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexImage2D called (stub) - compressed textures not supported\n");
    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglCompressedTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexImage1D called (stub) - compressed textures not supported\n");
    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglCompressedTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexSubImage3D called (stub) - compressed textures not supported\n");
    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglCompressedTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexSubImage2D called (stub) - compressed textures not supported\n");
    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglCompressedTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data)
{
    // Stub - compressed textures not fully supported yet
    fprintf(stderr, "MGL WARNING: glCompressedTexSubImage1D called (stub) - compressed textures not supported\n");
    ERROR_RETURN(GL_INVALID_ENUM);
}

#pragma mark copy tex
static bool mglCopyTex2DFaceForTarget(GLenum target, GLuint *face_out)
{
    if (face_out) {
        *face_out = 0u;
    }

    switch (target) {
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            return true;

        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            if (face_out) {
                *face_out = (GLuint)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X);
            }
            return true;

        default:
            return false;
    }
}

void mglCopyTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border)
{
    // Stub - not commonly used
    fprintf(stderr, "MGL WARNING: glCopyTexImage1D called (stub)\n");
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglCopyTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border)
{
    fprintf(stderr, "MGL: glCopyTexImage2D called - target=0x%x level=%d %dx%d\n", target, level, width, height);
    GLuint face = 0u;

    if (!mglCopyTex2DFaceForTarget(target, &face)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (level < 0 || width < 0 || height < 0 || border != 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (target == GL_TEXTURE_RECTANGLE && level != 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (!checkInternalFormatForMetal(ctx, internalformat)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    // Get or create texture
    Texture *tex = getTex(ctx, 0, target);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!createTextureLevel(ctx,
                            tex,
                            face,
                            level,
                            false,
                            internalformat,
                            width,
                            height,
                            1,
                            GL_BGRA,
                            GL_UNSIGNED_BYTE,
                            NULL,
                            false)) {
        return;
    }

    if (!ctx->mtl_funcs.mtlCopyTexSubImage) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    // Copy from framebuffer to texture
    ctx->mtl_funcs.mtlCopyTexSubImage(ctx, tex, face, level, 0, 0, x, y, width, height);
}

void mglCopyTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    fprintf(stderr, "MGL: glCopyTexSubImage1D called (stub)\n");
    // Stub - 1D textures rarely used
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglCopyTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    fprintf(stderr, "MGL: glCopyTexSubImage2D called - target=0x%x %dx%d at (%d,%d) from (%d,%d)\n",
            target, width, height, xoffset, yoffset, x, y);
    GLuint face = 0u;

    if (!mglCopyTex2DFaceForTarget(target, &face)) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    if (level < 0 || xoffset < 0 || yoffset < 0 || width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (target == GL_TEXTURE_RECTANGLE && level != 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (width == 0 || height == 0) {
        return;
    }

    // Get the bound texture
    Texture *tex = getTex(ctx, 0, target);
    if (!tex) {
        fprintf(stderr, "MGL ERROR: glCopyTexSubImage2D - no texture bound\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (level >= (GLint)tex->num_levels ||
        !tex->faces[face].levels ||
        !tex->faces[face].levels[level].complete) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    TextureLevel *lvl = &tex->faces[face].levels[level];
    if ((GLuint)xoffset > lvl->width ||
        (GLuint)yoffset > lvl->height ||
        (GLuint)width > lvl->width - (GLuint)xoffset ||
        (GLuint)height > lvl->height - (GLuint)yoffset) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!ctx->mtl_funcs.mtlCopyTexSubImage) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    // This copies from the current read framebuffer to the texture
    ctx->mtl_funcs.mtlCopyTexSubImage(ctx, tex, face, level, xoffset, yoffset, x, y, width, height);
}

void mglCopyTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    fprintf(stderr, "MGL WARNING: glCopyTexSubImage3D unsupported; refusing to ignore zoffset=%d\n", zoffset);
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglCopyTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    fprintf(stderr, "MGL: glCopyTextureSubImage1D called (stub)\n");
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglCopyTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    fprintf(stderr, "MGL: glCopyTextureSubImage2D called - texture=%u %dx%d\n", texture, width, height);
    if (texture == 0 || level < 0 || xoffset < 0 || yoffset < 0 || width < 0 || height < 0) {
        ERROR_RETURN(texture == 0 ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
        return;
    }
    if (width == 0 || height == 0) {
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (tex) {
        if (level >= (GLint)tex->num_levels ||
            !tex->faces[0].levels ||
            !tex->faces[0].levels[level].complete) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
        TextureLevel *lvl = &tex->faces[0].levels[level];
        if ((GLuint)xoffset > lvl->width ||
            (GLuint)yoffset > lvl->height ||
            (GLuint)width > lvl->width - (GLuint)xoffset ||
            (GLuint)height > lvl->height - (GLuint)yoffset) {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
        if (!ctx->mtl_funcs.mtlCopyTexSubImage) {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
        ctx->mtl_funcs.mtlCopyTexSubImage(ctx, tex, 0, level, xoffset, yoffset, x, y, width, height);
    } else {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }
}

void mglCopyTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    fprintf(stderr, "MGL WARNING: glCopyTextureSubImage3D unsupported; refusing to ignore zoffset=%d\n", zoffset);
    ERROR_RETURN(GL_INVALID_OPERATION);
}

#pragma mark get tex image

void mglGetTexImage(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, void *pixels)
{
    fprintf(stderr, "MGL: glGetTexImage called - target=0x%x level=%d format=0x%x type=0x%x\n",
            target, level, format, type);
    GLuint slice = 0;

    switch (target) {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            break;
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            slice = (GLuint)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X);
            break;
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            fprintf(stderr, "MGL WARNING: glGetTexImage layered/3D readback target=0x%x is unsupported\n", target);
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
    if (STATE(buffers[_PIXEL_PACK_BUFFER])) {
        fprintf(stderr, "MGL WARNING: glGetTexImage with GL_PIXEL_PACK_BUFFER is unsupported\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!pixels) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!tex) {
        fprintf(stderr, "MGL ERROR: glGetTexImage - no texture bound\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    if (!tex->mtl_data) {
        fprintf(stderr, "MGL ERROR: glGetTexImage - texture has no Metal data\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    if (level < 0 || level >= (GLint)tex->num_levels) {
        fprintf(stderr, "MGL ERROR: glGetTexImage - invalid level %d (max %d)\n", level, tex->num_levels);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (sizeForFormatType(format, type) == 0u) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    
    // Calculate dimensions for this level
    GLsizei width = tex->width >> level;
    GLsizei height = tex->height >> level;
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    
    size_t pixel_size = sizeForFormatType(format, type);
    MGLTexturePackLayout pack_layout;
    if (!mglComputeTexturePackLayout(ctx,
                                     width,
                                     height,
                                     1,
                                     pixel_size,
                                     "glGetTexImage",
                                     &pack_layout)) {
        return;
    }
    if (pack_layout.dst_pitch > UINT_MAX ||
        pack_layout.dst_image_size > UINT_MAX) {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }

    fprintf(stderr, "MGL: glGetTexImage - reading %dx%d, bytesPerRow=%zu\n", width, height, pack_layout.dst_pitch);
    
    // Use the Metal function to read the texture
    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlGetTexImage(ctx,
                                  tex,
                                  (uint8_t *)pixels + pack_layout.skip_offset_bytes,
                                  (GLuint)pack_layout.dst_pitch,
                                  (GLuint)pack_layout.dst_image_size,
                                  0,
                                  0,
                                  width,
                                  height,
                                  format,
                                  type,
                                  level,
                                  slice);
}

void mglGetTextureImage(GLMContext ctx, GLuint texture, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL: glGetTextureImage called - texture=%u level=%d\n", texture, level);
    if (bufSize < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (STATE(buffers[_PIXEL_PACK_BUFFER])) {
        fprintf(stderr, "MGL WARNING: glGetTextureImage with GL_PIXEL_PACK_BUFFER is unsupported\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!pixels && bufSize > 0) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    Texture *tex = getTex(ctx, texture, 0);
    if (!tex) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    if (!tex->mtl_data) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (tex->target == GL_TEXTURE_3D ||
        tex->target == GL_TEXTURE_1D_ARRAY ||
        tex->target == GL_TEXTURE_2D_ARRAY ||
        tex->target == GL_TEXTURE_CUBE_MAP ||
        tex->target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        fprintf(stderr, "MGL WARNING: glGetTextureImage layered/3D readback texture=%u target=0x%x is unsupported\n",
                texture,
                tex->target);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (level < 0 || level >= (GLint)tex->num_levels) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    
    GLsizei width = tex->width >> level;
    GLsizei height = tex->height >> level;
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    
    size_t pixel_size = sizeForFormatType(format, type);
    if (pixel_size == 0u) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    MGLTexturePackLayout pack_layout;
    if (!mglComputeTexturePackLayout(ctx,
                                     width,
                                     height,
                                     1,
                                     pixel_size,
                                     "glGetTextureImage",
                                     &pack_layout)) {
        return;
    }

    if (pack_layout.required_bytes > (size_t)bufSize ||
        pack_layout.dst_pitch > UINT_MAX ||
        pack_layout.dst_image_size > UINT_MAX) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlGetTexImage(ctx,
                                  tex,
                                  (uint8_t *)pixels + pack_layout.skip_offset_bytes,
                                  (GLuint)pack_layout.dst_pitch,
                                  (GLuint)pack_layout.dst_image_size,
                                  0,
                                  0,
                                  width,
                                  height,
                                  format,
                                  type,
                                  level,
                                  0);
}

void mglGetTextureSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL: glGetTextureSubImage called - texture=%u\n", texture);
    if (bufSize < 0 || level < 0 || xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width < 0 || height < 0 || depth < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (width == 0 || height == 0 || depth == 0) {
        return;
    }
    if (STATE(buffers[_PIXEL_PACK_BUFFER])) {
        fprintf(stderr, "MGL WARNING: glGetTextureSubImage with GL_PIXEL_PACK_BUFFER is unsupported\n");
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (!pixels) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    Texture *tex = getTex(ctx, texture, 0);
    if (!tex || !tex->mtl_data) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (level >= (GLint)tex->num_levels || !tex->faces[0].levels) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    if (tex->target == GL_TEXTURE_3D) {
        fprintf(stderr, "MGL WARNING: glGetTextureSubImage 3D readback texture=%u is unsupported\n", texture);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (depth != 1) {
        fprintf(stderr, "MGL WARNING: glGetTextureSubImage multi-slice readback texture=%u depth=%d is unsupported\n",
                texture,
                depth);
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    TextureLevel *lvl = &tex->faces[0].levels[level];
    if (!lvl->complete ||
        xoffset > (GLint)lvl->width ||
        yoffset > (GLint)lvl->height ||
        zoffset > (GLint)lvl->depth ||
        width > (GLsizei)(lvl->width - (GLuint)xoffset) ||
        height > (GLsizei)(lvl->height - (GLuint)yoffset) ||
        depth > (GLsizei)(lvl->depth - (GLuint)zoffset)) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    
    size_t pixel_size = sizeForFormatType(format, type);
    if (pixel_size == 0u) {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }
    MGLTexturePackLayout pack_layout;
    if (!mglComputeTexturePackLayout(ctx,
                                     width,
                                     height,
                                     depth,
                                     pixel_size,
                                     "glGetTextureSubImage",
                                     &pack_layout)) {
        return;
    }
    if (pack_layout.required_bytes > (size_t)bufSize ||
        pack_layout.dst_pitch > UINT_MAX ||
        pack_layout.dst_image_size > UINT_MAX) {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    
    mglFlushCommandBuffer(ctx);
    ctx->mtl_funcs.mtlGetTexImage(ctx,
                                  tex,
                                  (uint8_t *)pixels + pack_layout.skip_offset_bytes,
                                  (GLuint)pack_layout.dst_pitch,
                                  (GLuint)pack_layout.dst_image_size,
                                  xoffset,
                                  yoffset,
                                  width,
                                  height,
                                  format,
                                  type,
                                  level,
                                  zoffset);
}

void mglGetCompressedTexImage(GLMContext ctx, GLenum target, GLint level, void *img)
{
    fprintf(stderr, "MGL WARNING: glGetCompressedTexImage called (stub) - compressed textures not supported\n");
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnCompressedTexImage(GLMContext ctx, GLenum target, GLint lod, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL WARNING: glGetnCompressedTexImage called (stub) - compressed textures not supported\n");
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetCompressedTextureSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL WARNING: glGetCompressedTextureSubImage called (stub) - compressed textures not supported\n");
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTextureView(GLMContext ctx, GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers)
{
    fprintf(stderr, "MGL WARNING: glTextureView called (stub) - texture views not supported\n");
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTextureBufferRange(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size);

void mglTextureBuffer(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer)
{
    mglTextureBufferRange(ctx, texture, internalformat, buffer, 0, 0);
}

void mglTextureBufferRange(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    Texture *tex;
    Buffer *buf = NULL;
    size_t bytes_per_texel;
    GLsizeiptr attach_size;

    ERROR_CHECK_RETURN(texture != 0, GL_INVALID_OPERATION);

    tex = findTexture(ctx, texture);
    ERROR_CHECK_RETURN(tex, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(tex->target == GL_TEXTURE_BUFFER, GL_INVALID_OPERATION);

    mglFlushPendingDraws(ctx);

    if (buffer == 0)
    {
        if (tex->mtl_data && ctx->mtl_funcs.mtlDeleteMTLObj)
        {
            ctx->mtl_funcs.mtlDeleteMTLObj(ctx, tex->mtl_data);
        }

        tex->texture_buffer = NULL;
        tex->texture_buffer_offset = 0;
        tex->texture_buffer_size = 0;
        tex->internalformat = internalformat;
        tex->width = 0;
        tex->height = 1;
        tex->depth = 1;
        tex->complete = GL_FALSE;
        tex->mtl_data = NULL;
        tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
        STATE(dirty_bits) |= DIRTY_TEX | DIRTY_TEX_BINDING;
        return;
    }

    ERROR_CHECK_RETURN(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_OPERATION);
    bytes_per_texel = sizeForInternalFormat(internalformat, 0, 0);
    ERROR_CHECK_RETURN(bytes_per_texel != 0, GL_INVALID_ENUM);

    buf = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN(buf, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(buf->size >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(offset >= 0, GL_INVALID_VALUE);

    if (size == 0)
    {
        attach_size = buf->size - offset;
    }
    else
    {
        attach_size = size;
    }

    ERROR_CHECK_RETURN(attach_size >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(offset <= buf->size, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(attach_size <= buf->size - offset, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(((size_t)attach_size / bytes_per_texel) > 0, GL_INVALID_VALUE);

    /*
     * Some render paths re-specify the same texel-buffer attachment before
     * each draw. GL treats that as the same association, so keep the existing
     * Metal texture alive and let bindMTLTexture refresh it only when the
     * source buffer itself is dirty.
     */
    if (tex->texture_buffer == buf &&
        tex->texture_buffer_offset == offset &&
        tex->texture_buffer_size == attach_size &&
        tex->internalformat == internalformat &&
        tex->complete == GL_TRUE)
    {
        tex->dirty_bits &= ~DIRTY_TEXTURE_LEVEL;
        STATE(dirty_bits) |= DIRTY_TEX | DIRTY_TEX_BINDING;

        static uint64_t s_texBufferUnchangedLogs = 0;
        uint64_t hit = ++s_texBufferUnchangedLogs;
        if (hit <= 32ull || (hit % 512ull) == 0ull) {
            fprintf(stderr,
                    "MGL TRACE TexBuffer unchanged hit=%llu texture=%u internal=0x%x buffer=%u offset=%lld size=%lld texels=%u bpt=%zu dirty=0x%x bufferDirty=0x%x\n",
                    (unsigned long long)hit,
                    texture,
                    internalformat,
                    buffer,
                    (long long)offset,
                    (long long)attach_size,
                    tex->width,
                    bytes_per_texel,
                    tex->dirty_bits,
                    buf->data.dirty_bits);
        }
        return;
    }

    if (tex->mtl_data && ctx->mtl_funcs.mtlDeleteMTLObj)
    {
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, tex->mtl_data);
        tex->mtl_data = NULL;
    }

    tex->texture_buffer = buf;
    tex->texture_buffer_offset = offset;
    tex->texture_buffer_size = attach_size;
    tex->internalformat = internalformat;
    tex->width = (GLuint)((size_t)attach_size / bytes_per_texel);
    tex->height = 1;
    tex->depth = 1;
    tex->is_array = GL_FALSE;
    tex->complete = GL_TRUE;
    tex->num_levels = 1;
    tex->mipmap_levels = 1;
    tex->dirty_bits |= DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA;
    STATE(dirty_bits) |= DIRTY_TEX | DIRTY_TEX_BINDING;

    fprintf(stderr,
            "MGL TRACE TexBuffer texture=%u internal=0x%x buffer=%u offset=%lld size=%lld texels=%u bpt=%zu\n",
            texture,
            internalformat,
            buffer,
            (long long)offset,
            (long long)attach_size,
            tex->width,
            bytes_per_texel);
}

void mglCompressedTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data)
{
    fprintf(stderr, "MGL WARNING: glCompressedTextureSubImage1D called (stub)\n");
    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglCompressedTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    fprintf(stderr, "MGL WARNING: glCompressedTextureSubImage2D called (stub)\n");
    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglCompressedTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data)
{
    fprintf(stderr, "MGL WARNING: glCompressedTextureSubImage3D called (stub)\n");
    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglGetCompressedTextureImage(GLMContext ctx, GLuint texture, GLint level, GLsizei bufSize, void *pixels)
{
    fprintf(stderr, "MGL WARNING: glGetCompressedTextureImage called (stub)\n");
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetTextureLevelParameteriv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (texture == 0 || level < 0) {
        ERROR_RETURN(texture == 0 ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex) {
        *params = 0;
        return;
    }

    if (level >= (GLint)tex->num_levels || !tex->faces[0].levels) {
        switch (pname) {
            case GL_TEXTURE_WIDTH:
            case GL_TEXTURE_HEIGHT:
            case GL_TEXTURE_DEPTH:
            case GL_TEXTURE_INTERNAL_FORMAT:
                *params = 0;
                return;
            default:
                break;
        }
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
        case GL_TEXTURE_RED_SIZE:
            *params = bitcountForInternalFormat(tex->internalformat, GL_RED);
            break;
        case GL_TEXTURE_GREEN_SIZE:
            *params = bitcountForInternalFormat(tex->internalformat, GL_GREEN);
            break;
        case GL_TEXTURE_BLUE_SIZE:
            *params = bitcountForInternalFormat(tex->internalformat, GL_BLUE);
            break;
        case GL_TEXTURE_ALPHA_SIZE:
            *params = bitcountForInternalFormat(tex->internalformat, GL_ALPHA);
            break;
        case GL_TEXTURE_DEPTH_SIZE:
            *params = bitcountForInternalFormat(tex->internalformat, GL_DEPTH_COMPONENT);
            break;
        case GL_TEXTURE_STENCIL_SIZE:
            *params = bitcountForInternalFormat(tex->internalformat, GL_STENCIL_INDEX);
            break;
        case GL_TEXTURE_COMPRESSED:
            *params = GL_FALSE;
            break;
        case GL_TEXTURE_COMPRESSED_IMAGE_SIZE:
            *params = 0;
            break;
        case GL_TEXTURE_SAMPLES:
            *params = 0;
            break;
        case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS:
            *params = GL_TRUE;
            break;
        case GL_TEXTURE_SHARED_SIZE:
            *params = (tex->internalformat == GL_RGB9_E5) ? 5 : 0;
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}

void mglGetTextureLevelParameterfv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLfloat *params)
{
    if (!params)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    GLint iparams;
    mglGetTextureLevelParameteriv(ctx, texture, level, pname, &iparams);
    *params = (GLfloat)iparams;
}

static bool mglTextureParameterGetiv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLint *params)
{
    if (!tex_params || !params)
        return false;

    switch (pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; i++)
                params[i] = (GLint)tex_params->border_color[i];
            return true;
        case GL_TEXTURE_SWIZZLE_RGBA:
            params[0] = tex_params->swizzle_r;
            params[1] = tex_params->swizzle_g;
            params[2] = tex_params->swizzle_b;
            params[3] = tex_params->swizzle_a;
            return true;
        case GL_TEXTURE_MIN_LOD:
            *params = (GLint)tex_params->min_lod;
            return true;
        case GL_TEXTURE_MAX_LOD:
            *params = (GLint)tex_params->max_lod;
            return true;
        case GL_TEXTURE_LOD_BIAS:
            *params = (GLint)tex_params->lod_bias;
            return true;
        case GL_TEXTURE_MAX_ANISOTROPY:
            *params = (GLint)tex_params->max_anisotropy;
            return true;
        default:
        {
            GLfloat fparam = 0.0f;
            return getParam(ctx, tex_params, pname, params, &fparam);
        }
    }
}

static bool mglTextureParameterGetfv(GLMContext ctx, TextureParameter *tex_params, GLenum pname, GLfloat *params)
{
    if (!tex_params || !params)
        return false;

    switch (pname)
    {
        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; i++)
                params[i] = tex_params->border_color[i];
            return true;
        case GL_TEXTURE_SWIZZLE_RGBA:
            params[0] = (GLfloat)tex_params->swizzle_r;
            params[1] = (GLfloat)tex_params->swizzle_g;
            params[2] = (GLfloat)tex_params->swizzle_b;
            params[3] = (GLfloat)tex_params->swizzle_a;
            return true;
        case GL_TEXTURE_MIN_LOD:
            *params = tex_params->min_lod;
            return true;
        case GL_TEXTURE_MAX_LOD:
            *params = tex_params->max_lod;
            return true;
        case GL_TEXTURE_LOD_BIAS:
            *params = tex_params->lod_bias;
            return true;
        case GL_TEXTURE_MAX_ANISOTROPY:
            *params = tex_params->max_anisotropy;
            return true;
        default:
        {
            GLint iparam = 0;
            if (getParam(ctx, tex_params, pname, &iparam, params)) {
                *params = (GLfloat)iparam;
                return true;
            }
            return false;
        }
    }
}

static bool mglTextureParameterGetIiv(TextureParameter *tex_params, GLenum pname, GLint *params)
{
    if (!tex_params || !params)
        return false;

    if (pname == GL_TEXTURE_BORDER_COLOR) {
        for (int i = 0; i < 4; i++)
            params[i] = tex_params->border_color_i[i];
        return true;
    }

    return false;
}

static bool mglTextureParameterGetIuiv(TextureParameter *tex_params, GLenum pname, GLuint *params)
{
    if (!tex_params || !params)
        return false;

    if (pname == GL_TEXTURE_BORDER_COLOR) {
        for (int i = 0; i < 4; i++)
            params[i] = tex_params->border_color_ui[i];
        return true;
    }

    return false;
}

void mglGetTextureParameterfv(GLMContext ctx, GLuint texture, GLenum pname, GLfloat *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex)
        return;

    if (!mglTextureParameterGetfv(ctx, &tex->params, pname, params))
        ERROR_RETURN(GL_INVALID_ENUM);
}

void mglGetTextureParameterIiv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex)
        return;

    if (mglTextureParameterGetIiv(&tex->params, pname, params))
        return;

    if (!mglTextureParameterGetiv(ctx, &tex->params, pname, params))
        ERROR_RETURN(GL_INVALID_ENUM);
}

void mglGetTextureParameterIuiv(GLMContext ctx, GLuint texture, GLenum pname, GLuint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex)
        return;

    if (mglTextureParameterGetIuiv(&tex->params, pname, params))
        return;

    GLint iparam = 0;
    if (mglTextureParameterGetiv(ctx, &tex->params, pname, &iparam)) {
        *params = (GLuint)iparam;
        return;
    }

    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglGetTextureParameteriv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, texture, 0);
    if (!tex)
        return;

    if (!mglTextureParameterGetiv(ctx, &tex->params, pname, params))
        ERROR_RETURN(GL_INVALID_ENUM);
}

void mglGetTexParameterIiv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!tex)
        return;

    if (mglTextureParameterGetIiv(&tex->params, pname, params))
        return;

    if (!mglTextureParameterGetiv(ctx, &tex->params, pname, params))
        ERROR_RETURN(GL_INVALID_ENUM);
}

void mglGetTexParameterIuiv(GLMContext ctx, GLenum target, GLenum pname, GLuint *params)
{
    if (!params) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!tex)
        return;

    if (mglTextureParameterGetIuiv(&tex->params, pname, params))
        return;

    GLint iparam = 0;
    if (mglTextureParameterGetiv(ctx, &tex->params, pname, &iparam)) {
        *params = (GLuint)iparam;
        return;
    }

    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglSampleCoverage(GLMContext ctx, GLfloat value, GLboolean invert)
{
    if (!ctx)
        return;

    STATE(var.sample_coverage_value) = clamp(value, 0.0f, 1.0f);
    STATE(var.sample_coverage_invert) = invert ? GL_TRUE : GL_FALSE;
    STATE(dirty_bits) |= DIRTY_RENDER_STATE;
}
