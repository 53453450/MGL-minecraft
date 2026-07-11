/*
 * mgl_texture_debug.c
 * MGL
 *
 * Debug / utility helpers extracted from textures.c as part of the
 * God Object decomposition (Task 8).  See mgl_texture_debug.h for the
 * public interface.
 */

#include "mgl_texture_debug.h"

#include <execinfo.h>
#include <mach/mach_time.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <limits.h>

#include "mgl_pixel_format.h"
#include "mgl_trace_log.h"
#include "utils.h"

#ifndef MGL_VERBOSE_TEXTURE_UPLOAD_LOGS
#define MGL_VERBOSE_TEXTURE_UPLOAD_LOGS 0
#endif

#ifndef MGL_VERBOSE_TEXTURE_BIND_LOGS
#define MGL_VERBOSE_TEXTURE_BIND_LOGS 0
#endif

void mglDumpBytesToStderr(const char *label,
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

void mglDumpByteWindowToStderr(const char *label,
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

void mglDumpTextureUploadRowSamples(const char *prefix,
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

void mglDumpTextureUploadSamples(Texture *tex,
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

const char *mglTextureInitSourceName(GLuint source)
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

const char *mglBufferInitSourceName(MGLBufferInitSource source)
{
    switch (source) {
        case kInitNone: return "none";
        case kInitBufferDataNull: return "BufferData(NULL)";
        case kInitBufferDataCopy: return "BufferData(copy)";
        case kInitBufferSubData: return "BufferSubData";
        case kInitCopyBufferSubData: return "CopyBufferSubData";
        case kInitReadPixels: return "ReadPixels";
        case kInitMapWrite: return "MapWrite";
        default: return "unknown";
    }
}

void mglDumpNativeBacktraceToStderr(const char *tag, size_t max_frames)
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

void mglRequestJavaThreadDumpForZeroCpuUpload(Texture *tex,
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

void mglDumpTexSubImageZeroCpuResourceTag(GLMContext ctx,
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

uint64_t mglHashBytesSampled(const void *data, size_t len)
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

bool mglLooksAllZero(const uint8_t *bytes, size_t len)
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

bool mglLooksAllZeroSampled(const uint8_t *bytes, size_t len)
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

bool mglTextureRectByteRange(TextureLevel *level,
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

uint64_t mglHashTextureRect(const uint8_t *base,
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

bool mglTextureRectLooksAllZero(const uint8_t *base,
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

bool mglShouldTraceTextureUpload(Texture *tex,
                                        GLuint unpack_name,
                                        GLsizei width,
                                        GLsizei height,
                                        GLsizei depth,
                                        size_t required_bytes)
{
    if (MGL_VERBOSE_TEXTURE_UPLOAD_LOGS) {
        return true;
    }

    if (!mglTraceLogIsEnabled()) {
        return false;
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

bool mglMulSizeT(size_t a, size_t b, size_t *out)
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

bool mglAddSizeT(size_t a, size_t b, size_t *out)
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

void mglTraceTextureUnitState(GLMContext ctx,
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

    if (MGL_VERBOSE_TEXTURE_BIND_LOGS) {
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
    } else {
        mglTraceLogExternal("TEX_UNIT_%s hit=%llu unit=%u activeUnit=%u target=0x%x texture=%u bound=%u state(active=%u texBuffer=%u tex2D=%u cube=%u maskWord=0x%x internal=0x%x depthStencil=%d dynamicRT=%d)",
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
}

/* page_size_align is defined in buffers.c; declared in mgl_texture_debug.h. */
