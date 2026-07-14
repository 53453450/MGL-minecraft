/*
 * mgl_texture_transfer.c
 * MGL
 *
 * CPU upload / download / transfer helper functions extracted from
 * textures.c as part of the God Object decomposition (Task 9).  See
 * mgl_texture_transfer.h for the public interface.
 */

#include "mgl_texture_transfer.h"

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_pixel_format.h"
#include "mgl_texture_debug.h"
#include "pixel_utils.h"
#include "utils.h"

/* mglTexLevelInternalFormatCompressed is defined in tex_param.c but not
 * declared in any public header.  mgl_pixel_format.c has the same extern. */
extern bool mglTexLevelInternalFormatCompressed(GLint internalformat);

#ifndef MGL_VERBOSE_TEXTURE_UPLOAD_LOGS
#define MGL_VERBOSE_TEXTURE_UPLOAD_LOGS 0
#endif

/* ---------------------------------------------------------------------------
 * Forward declarations — functions defined in textures.c that the extracted
 * transfer helpers call.  These are non-static in textures.c.
 * ------------------------------------------------------------------------- */

Texture *getTex(GLMContext ctx, GLuint name, GLenum target);
void initBaseTexLevel(GLMContext ctx, Texture *tex, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth);
void invalidateTexture(GLMContext ctx, Texture *tex);
bool verifyInternalFormatAndFormatType(GLMContext ctx, GLint internalformat, GLenum format, GLenum type);
GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);
void mglReleaseGLSampledTextureCopy(GLMContext ctx, Texture *tex, const char *reason);
bool ensureTextureLevelCapacity(GLMContext ctx, Texture *tex, GLuint required_levels);

/* Externs for helpers declared in other translation units. */
extern void *getBufferData(GLMContext ctx, Buffer *ptr);
extern GLsizei mglSafeMaxTextureSize(GLMContext ctx);
extern size_t mglPixelTypeDatumBytes(GLenum type);

bool mglComputeTexturePackLayout(GLMContext ctx,
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
    GLint skip_images = ctx->state.pack.skip_images;
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

bool mglConvertTextureRectToCPU(GLenum internalformat,
                                       TextureLevel *lvl,
                                       GLint xoffset,
                                       GLint yoffset,
                                       GLint zoffset,
                                       GLsizei width,
                                       GLsizei height,
                                       GLsizei depth,
                                       GLenum format,
                                       GLenum type,
                                       const uint8_t *src_base,
                                       size_t src_pitch,
                                       size_t src_image_size,
                                       bool swap_bytes)
{
    if (!lvl || !lvl->data || !src_base ||
        xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width <= 0 || height <= 0 || depth <= 0 ||
        lvl->width == 0u || lvl->height == 0u || lvl->pitch == 0u) {
        return false;
    }

    size_t storage_pixel_size = lvl->pitch / (size_t)lvl->width;
    size_t src_pixel_size = sizeForFormatType(format, type);
    if (storage_pixel_size == 0u || src_pixel_size == 0u ||
        storage_pixel_size * (size_t)lvl->width > lvl->pitch ||
        (GLuint)xoffset > lvl->width ||
        (GLuint)yoffset > lvl->height ||
        (GLuint)zoffset > lvl->depth ||
        (GLuint)width > lvl->width - (GLuint)xoffset ||
        (GLuint)height > lvl->height - (GLuint)yoffset ||
        (GLuint)depth > lvl->depth - (GLuint)zoffset) {
        return false;
    }

    /* Fast path: when the external format/type produces the exact same bit
     * layout as the internal format's CPU storage, do a raw memcpy.  This
     * avoids precision loss from per-component unpack/repack through double
     * (e.g. R11F_G11F_B10F mantissa bits being cleared) and is the critical
     * performance path for common uncompressed formats (RGBA8+UB, RGBA32F, etc). */
    if (storage_pixel_size == src_pixel_size && !swap_bytes &&
        (mglIsIdentityPackedFormat(internalformat, format, type) ||
         mglIsIdentityUncompressedFormat(internalformat, format, type))) {
        size_t dst_img_size = 0u;
        if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &dst_img_size)) {
            return false;
        }
        if (src_image_size == 0u && !mglMulSizeT(src_pitch, (size_t)height, &src_image_size)) {
            return false;
        }
        size_t row_bytes = (size_t)width * storage_pixel_size;
        uint8_t *dst_base = (uint8_t *)lvl->data +
                            ((size_t)zoffset * dst_img_size) +
                            ((size_t)yoffset * lvl->pitch) +
                            ((size_t)xoffset * storage_pixel_size);
        for (GLsizei z = 0; z < depth; z++) {
            const uint8_t *src_slice = src_base + ((size_t)z * src_image_size);
            uint8_t *dst_slice = dst_base + ((size_t)z * dst_img_size);
            for (GLsizei y = 0; y < height; y++) {
                memcpy(dst_slice + ((size_t)y * lvl->pitch),
                       src_slice + ((size_t)y * src_pitch),
                       row_bytes);
            }
        }
        return true;
    }

    /* Fast path: BGRA→RGBA byte swap.  When the external format is BGR/BGRA
     * and the internal format is RGB/RGBA with matching 8-bit components,
     * the only conversion needed is a per-pixel B↔R byte swap (offset 0↔2). */
    if (storage_pixel_size == src_pixel_size && !swap_bytes &&
        mglIsBGRByteSwapFormat(internalformat, format, type)) {
        size_t dst_img_size = 0u;
        if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &dst_img_size)) {
            return false;
        }
        if (src_image_size == 0u && !mglMulSizeT(src_pitch, (size_t)height, &src_image_size)) {
            return false;
        }
        size_t row_bytes = (size_t)width * storage_pixel_size;
        uint8_t *dst_base = (uint8_t *)lvl->data +
                            ((size_t)zoffset * dst_img_size) +
                            ((size_t)yoffset * lvl->pitch) +
                            ((size_t)xoffset * storage_pixel_size);
        for (GLsizei z = 0; z < depth; z++) {
            const uint8_t *src_slice = src_base + ((size_t)z * src_image_size);
            uint8_t *dst_slice = dst_base + ((size_t)z * dst_img_size);
            for (GLsizei y = 0; y < height; y++) {
                const uint8_t *src_row = src_slice + ((size_t)y * src_pitch);
                uint8_t *dst_row = dst_slice + ((size_t)y * lvl->pitch);
                memcpy(dst_row, src_row, row_bytes);
                for (GLsizei x = 0; x < width; x++) {
                    uint8_t *px = dst_row + ((size_t)x * storage_pixel_size);
                    uint8_t tmp = px[0];
                    px[0] = px[2];
                    px[2] = tmp;
                }
            }
        }
        return true;
    }

    MGLCPUPixelLayout cpu_layout;
    if (!mglBuildCPUPixelLayout(internalformat, storage_pixel_size, &cpu_layout)) {
        /* GL_RGB9_E5 has a shared exponent that cannot be handled by the
         * per-component layout model.  Handle it with a dedicated path. */
        if (internalformat == GL_RGB9_E5 && storage_pixel_size == 4u) {
            bool integer_fmt = mglExternalFormatIsInteger(format);
            size_t dst_img_size = 0u;
            if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &dst_img_size)) {
                return false;
            }
            if (src_image_size == 0u && !mglMulSizeT(src_pitch, (size_t)height, &src_image_size)) {
                return false;
            }
            uint8_t *dst_base = (uint8_t *)lvl->data +
                                ((size_t)zoffset * dst_img_size) +
                                ((size_t)yoffset * lvl->pitch) +
                                ((size_t)xoffset * storage_pixel_size);
            size_t elem_size = mglPixelTypeDatumBytes(type);
            for (GLsizei z = 0; z < depth; z++) {
                const uint8_t *src_slice = src_base + ((size_t)z * src_image_size);
                uint8_t *dst_slice = dst_base + ((size_t)z * dst_img_size);
                for (GLsizei y = 0; y < height; y++) {
                    const uint8_t *src_row = src_slice + ((size_t)y * src_pitch);
                    uint8_t *dst_row = dst_slice + ((size_t)y * lvl->pitch);
                    for (GLsizei x = 0; x < width; x++) {
                        const uint8_t *src_pixel = src_row + ((size_t)x * src_pixel_size);
                        uint8_t *dst_pixel = dst_row + ((size_t)x * storage_pixel_size);
                        uint8_t swapped_pixel[32];
                        const uint8_t *read_pixel = src_pixel;
                        if (swap_bytes && elem_size > 1u && src_pixel_size <= sizeof(swapped_pixel)) {
                            memcpy(swapped_pixel, src_pixel, src_pixel_size);
                            mglSwapPixelBytes(swapped_pixel, src_pixel_size, elem_size);
                            read_pixel = swapped_pixel;
                        }
                        memset(dst_pixel, 0, storage_pixel_size);
                        /* If external type is already GL_UNSIGNED_INT_5_9_9_9_REV,
                         * the data is in native shared-exp format; copy directly. */
                        if (type == GL_UNSIGNED_INT_5_9_9_9_REV && !integer_fmt) {
                            uint32_t packed = (uint32_t)mglReadUnsignedLE(read_pixel, sizeof(uint32_t));
                            mglWriteUnsignedLE(dst_pixel, sizeof(uint32_t), packed);
                        } else {
                            /* Map internal R/G/B to external source indices (handles BGR/BGRA). */
                            int sr = mglExternalSourceIndexForComponent(format, 0);
                            int sg = mglExternalSourceIndexForComponent(format, 1);
                            int sb = mglExternalSourceIndexForComponent(format, 2);
                            GLuint rr = (sr >= 0) ? (GLuint)sr : 0u;
                            GLuint rg = (sg >= 0) ? (GLuint)sg : 1u;
                            GLuint rb = (sb >= 0) ? (GLuint)sb : 2u;
                            double r = mglReadExternalComponent(read_pixel, type, sr, integer_fmt, rr);
                            double g = mglReadExternalComponent(read_pixel, type, sg, integer_fmt, rg);
                            double b = mglReadExternalComponent(read_pixel, type, sb, integer_fmt, rb);
                            uint32_t packed = mglPackRGBToSharedExp(r, g, b);
                            mglWriteUnsignedLE(dst_pixel, sizeof(uint32_t), packed);
                        }
                    }
                }
            }
            return true;
        }

        /* Depth/stencil combined formats: convert external data to
         * canonical CPU shadow storage.
         * GL_DEPTH32F_STENCIL8: 5 bytes (float depth + uint8 stencil)
         * GL_DEPTH24_STENCIL8: 4 bytes (uint32: 24-bit depth high [31:8], 8-bit stencil low [7:0]) */
        if (mglInternalFormatIsCombinedDepthStencil(internalformat) &&
            (format == GL_DEPTH_STENCIL || format == GL_DEPTH_COMPONENT)) {
            bool integer_fmt = mglExternalFormatIsInteger(format);
            size_t dst_img_size = 0u;
            if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &dst_img_size)) {
                return false;
            }
            if (src_image_size == 0u && !mglMulSizeT(src_pitch, (size_t)height, &src_image_size)) {
                return false;
            }
            uint8_t *dst_base_ds = (uint8_t *)lvl->data +
                                   ((size_t)zoffset * dst_img_size) +
                                   ((size_t)yoffset * lvl->pitch) +
                                   ((size_t)xoffset * storage_pixel_size);
            size_t elem_size = mglPixelTypeDatumBytes(type);
            for (GLsizei z = 0; z < depth; z++) {
                const uint8_t *src_slice = src_base + ((size_t)z * src_image_size);
                uint8_t *dst_slice = dst_base_ds + ((size_t)z * dst_img_size);
                for (GLsizei y = 0; y < height; y++) {
                    const uint8_t *src_row = src_slice + ((size_t)y * src_pitch);
                    uint8_t *dst_row = dst_slice + ((size_t)y * lvl->pitch);
                    for (GLsizei x = 0; x < width; x++) {
                        const uint8_t *src_pixel = src_row + ((size_t)x * src_pixel_size);
                        uint8_t *dst_pixel = dst_row + ((size_t)x * storage_pixel_size);
                        uint8_t swapped_pixel[32];
                        const uint8_t *read_pixel = src_pixel;
                        if (swap_bytes && elem_size > 1u && src_pixel_size <= sizeof(swapped_pixel)) {
                            memcpy(swapped_pixel, src_pixel, src_pixel_size);
                            mglSwapPixelBytes(swapped_pixel, src_pixel_size, elem_size);
                            read_pixel = swapped_pixel;
                        }
                        memset(dst_pixel, 0, storage_pixel_size);
                        GLfloat depthVal = 0.0f;
                        uint8_t stencilVal = 0u;
                        if (format == GL_DEPTH_STENCIL) {
                            if (type == GL_UNSIGNED_INT_24_8) {
                                uint32_t packed;
                                memcpy(&packed, read_pixel, sizeof(uint32_t));
                                depthVal = (GLfloat)(packed >> 8) / 16777215.0f;
                                stencilVal = (uint8_t)(packed & 0xffu);
                            } else if (type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV) {
                                memcpy(&depthVal, read_pixel, sizeof(float));
                                uint32_t s;
                                memcpy(&s, read_pixel + 4, sizeof(uint32_t));
                                stencilVal = (uint8_t)(s & 0xffu);
                            }
                        } else { /* GL_DEPTH_COMPONENT */
                            double d = mglReadExternalComponent(read_pixel, type, 0, integer_fmt, 0);
                            depthVal = (GLfloat)d;
                        }
                        if (internalformat == GL_DEPTH32F_STENCIL8) {
                            memcpy(dst_pixel, &depthVal, sizeof(float));
                            if (storage_pixel_size >= 5u)
                                dst_pixel[4] = stencilVal;
                        } else { /* GL_DEPTH24_STENCIL8 */
                            uint32_t packed = ((uint32_t)(depthVal * 16777215.0f + 0.5f) << 8) |
                                              (uint32_t)stencilVal;
                            memcpy(dst_pixel, &packed, sizeof(uint32_t));
                        }
                    }
                }
            }
            return true;
        }
        return false;
    }

    bool integer_format = mglExternalFormatIsInteger(format);
    size_t dst_image_size = 0u;
    if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &dst_image_size)) {
        return false;
    }
    if (src_image_size == 0u && !mglMulSizeT(src_pitch, (size_t)height, &src_image_size)) {
        return false;
    }

    uint8_t *dst_start = (uint8_t *)lvl->data +
                         ((size_t)zoffset * dst_image_size) +
                         ((size_t)yoffset * lvl->pitch) +
                         ((size_t)xoffset * storage_pixel_size);

    size_t element_size = mglPixelTypeDatumBytes(type);

    for (GLsizei z = 0; z < depth; z++) {
        const uint8_t *src_slice = src_base + ((size_t)z * src_image_size);
        uint8_t *dst_slice = dst_start + ((size_t)z * dst_image_size);
        for (GLsizei y = 0; y < height; y++) {
            const uint8_t *src_row = src_slice + ((size_t)y * src_pitch);
            uint8_t *dst_row = dst_slice + ((size_t)y * lvl->pitch);
            for (GLsizei x = 0; x < width; x++) {
                const uint8_t *src_pixel = src_row + ((size_t)x * src_pixel_size);
                uint8_t *dst_pixel = dst_row + ((size_t)x * storage_pixel_size);
                uint8_t swapped_pixel[32];
                const uint8_t *read_pixel = src_pixel;
                if (swap_bytes && element_size > 1u && src_pixel_size <= sizeof(swapped_pixel)) {
                    memcpy(swapped_pixel, src_pixel, src_pixel_size);
                    mglSwapPixelBytes(swapped_pixel, src_pixel_size, element_size);
                    read_pixel = swapped_pixel;
                }
                memset(dst_pixel, 0, storage_pixel_size);
                for (GLuint component = 0; component < cpu_layout.component_count; component++) {
                    int src_index = mglExternalSourceIndexForComponent(format, component);
                    GLuint read_component = (src_index >= 0) ? (GLuint)src_index : component;
                    double value = mglReadExternalComponent(read_pixel, type, src_index, integer_format, read_component);
                    mglStoreInternalComponent(dst_pixel, &cpu_layout.components[component], value);
                }
            }
        }
    }

    return true;
}

bool mglFillTextureRectCPU(GLenum internalformat,
                                  TextureLevel *lvl,
                                  GLint xoffset,
                                  GLint yoffset,
                                  GLint zoffset,
                                  GLsizei width,
                                  GLsizei height,
                                  GLsizei depth,
                                  GLenum format,
                                  GLenum type,
                                  const void *data)
{
    size_t src_pixel_size = sizeForFormatType(format, type);
    if (src_pixel_size == 0u) {
        return false;
    }

    if (!lvl || !lvl->data ||
        xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width <= 0 || height <= 0 || depth <= 0 ||
        lvl->width == 0u || lvl->height == 0u || lvl->pitch == 0u) {
        return false;
    }

    size_t storage_pixel_size = lvl->pitch / (size_t)lvl->width;
    if (storage_pixel_size == 0u || storage_pixel_size > 64u ||
        storage_pixel_size * (size_t)lvl->width > lvl->pitch ||
        (GLuint)xoffset > lvl->width ||
        (GLuint)yoffset > lvl->height ||
        (GLuint)zoffset > lvl->depth ||
        (GLuint)width > lvl->width - (GLuint)xoffset ||
        (GLuint)height > lvl->height - (GLuint)yoffset ||
        (GLuint)depth > lvl->depth - (GLuint)zoffset) {
        return false;
    }

    /* Fast path for identity formats: the source pixel has the same byte
     * layout as the storage, so we can skip the per-component double
     * conversion and use the source data directly as the clear value. */
    if (storage_pixel_size == src_pixel_size &&
        (mglIsIdentityPackedFormat(internalformat, format, type) ||
         mglIsIdentityUncompressedFormat(internalformat, format, type))) {
        uint8_t zero_pixel[64];
        const uint8_t *clear_src = (const uint8_t *)data;
        if (!clear_src) {
            memset(zero_pixel, 0, storage_pixel_size);
            clear_src = zero_pixel;
        }
        size_t image_pitch = 0u;
        if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &image_pitch)) {
            return false;
        }
        uint8_t *base = (uint8_t *)lvl->data +
                        ((size_t)zoffset * image_pitch) +
                        ((size_t)yoffset * lvl->pitch) +
                        ((size_t)xoffset * storage_pixel_size);
        for (GLsizei z = 0; z < depth; z++) {
            uint8_t *slice = base + ((size_t)z * image_pitch);
            for (GLsizei y = 0; y < height; y++) {
                uint8_t *row = slice + ((size_t)y * lvl->pitch);
                for (GLsizei x = 0; x < width; x++) {
                    memcpy(row + ((size_t)x * storage_pixel_size), clear_src, storage_pixel_size);
                }
            }
        }
        return true;
    }

    /* Fast path for BGRA→RGBA swap: swap B↔R in the clear pixel, then replicate. */
    if (storage_pixel_size == src_pixel_size &&
        mglIsBGRByteSwapFormat(internalformat, format, type)) {
        uint8_t zero_pixel[64];
        uint8_t swap_pixel[64];
        const uint8_t *clear_src = (const uint8_t *)data;
        if (!clear_src) {
            memset(zero_pixel, 0, storage_pixel_size);
            clear_src = zero_pixel;
        }
        memcpy(swap_pixel, clear_src, storage_pixel_size);
        uint8_t tmp = swap_pixel[0];
        swap_pixel[0] = swap_pixel[2];
        swap_pixel[2] = tmp;
        size_t image_pitch = 0u;
        if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &image_pitch)) {
            return false;
        }
        uint8_t *base = (uint8_t *)lvl->data +
                        ((size_t)zoffset * image_pitch) +
                        ((size_t)yoffset * lvl->pitch) +
                        ((size_t)xoffset * storage_pixel_size);
        for (GLsizei z = 0; z < depth; z++) {
            uint8_t *slice = base + ((size_t)z * image_pitch);
            for (GLsizei y = 0; y < height; y++) {
                uint8_t *row = slice + ((size_t)y * lvl->pitch);
                for (GLsizei x = 0; x < width; x++) {
                    memcpy(row + ((size_t)x * storage_pixel_size), swap_pixel, storage_pixel_size);
                }
            }
        }
        return true;
    }

    MGLCPUPixelLayout cpu_layout;
    if (!mglBuildCPUPixelLayout(internalformat, storage_pixel_size, &cpu_layout)) {
        /* GL_RGB9_E5 has a shared exponent that cannot be handled by the
         * per-component layout model.  Handle it with a dedicated path. */
        if (internalformat == GL_RGB9_E5 && storage_pixel_size == 4u) {
            bool integer_fmt = mglExternalFormatIsInteger(format);
            uint8_t zero_pixel[64];
            const uint8_t *clear_src = (const uint8_t *)data;
            if (!clear_src) {
                memset(zero_pixel, 0, src_pixel_size);
                clear_src = zero_pixel;
            }
            uint32_t packed;
            if (type == GL_UNSIGNED_INT_5_9_9_9_REV && !integer_fmt) {
                /* Already in native shared-exp format. */
                packed = (uint32_t)mglReadUnsignedLE(clear_src, sizeof(uint32_t));
            } else {
                /* Map internal R/G/B to external source indices (handles BGR). */
                int sr = mglExternalSourceIndexForComponent(format, 0);
                int sg = mglExternalSourceIndexForComponent(format, 1);
                int sb = mglExternalSourceIndexForComponent(format, 2);
                GLuint rr = (sr >= 0) ? (GLuint)sr : 0u;
                GLuint rg = (sg >= 0) ? (GLuint)sg : 1u;
                GLuint rb = (sb >= 0) ? (GLuint)sb : 2u;
                double r = mglReadExternalComponent(clear_src, type, sr, integer_fmt, rr);
                double g = mglReadExternalComponent(clear_src, type, sg, integer_fmt, rg);
                double b = mglReadExternalComponent(clear_src, type, sb, integer_fmt, rb);
                packed = mglPackRGBToSharedExp(r, g, b);
            }
            uint8_t clear_pixel[4];
            mglWriteUnsignedLE(clear_pixel, sizeof(uint32_t), packed);
            size_t image_pitch = 0u;
            if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &image_pitch)) {
                return false;
            }
            uint8_t *base = (uint8_t *)lvl->data +
                            ((size_t)zoffset * image_pitch) +
                            ((size_t)yoffset * lvl->pitch) +
                            ((size_t)xoffset * storage_pixel_size);
            for (GLsizei z = 0; z < depth; z++) {
                uint8_t *slice = base + ((size_t)z * image_pitch);
                for (GLsizei y = 0; y < height; y++) {
                    uint8_t *row = slice + ((size_t)y * lvl->pitch);
                    for (GLsizei x = 0; x < width; x++) {
                        memcpy(row + ((size_t)x * storage_pixel_size),
                               clear_pixel, storage_pixel_size);
                    }
                }
            }
            return true;
        }
        return false;
    }

    uint8_t zero_pixel[64];
    const uint8_t *src = (const uint8_t *)data;
    if (!src) {
        if (src_pixel_size > sizeof(zero_pixel)) {
            return false;
        }
        memset(zero_pixel, 0, src_pixel_size);
        src = zero_pixel;
    }

    bool integer_format = mglExternalFormatIsInteger(format);
    uint8_t clear_pixel[64];
    memset(clear_pixel, 0, storage_pixel_size);
    for (GLuint component = 0; component < cpu_layout.component_count; component++) {
        int src_index = mglExternalSourceIndexForComponent(format, component);
        double value = mglReadExternalComponent(src, type, src_index, integer_format, component);
        mglStoreInternalComponent(clear_pixel, &cpu_layout.components[component], value);
    }

    size_t image_pitch = 0u;
    if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &image_pitch)) {
        return false;
    }

    uint8_t *base = (uint8_t *)lvl->data +
                    ((size_t)zoffset * image_pitch) +
                    ((size_t)yoffset * lvl->pitch) +
                    ((size_t)xoffset * storage_pixel_size);
    for (GLsizei z = 0; z < depth; z++) {
        uint8_t *slice = base + ((size_t)z * image_pitch);
        for (GLsizei y = 0; y < height; y++) {
            uint8_t *row = slice + ((size_t)y * lvl->pitch);
            for (GLsizei x = 0; x < width; x++) {
                memcpy(row + ((size_t)x * storage_pixel_size), clear_pixel, storage_pixel_size);
            }
        }
    }

    return true;
}

bool mglTextureHasCompressedInternalFormat(Texture *tex)
{
    return tex && (mglTexLevelInternalFormatCompressed(tex->internalformat) ||
                   (tex->compressed_internalformat != 0u &&
                    mglTexLevelInternalFormatCompressed(tex->compressed_internalformat)));
}

bool mglCopyTextureRectFromCPU(GLenum internalformat,
                                      TextureLevel *lvl,
                                      GLint xoffset,
                                      GLint yoffset,
                                      GLint zoffset,
                                      GLsizei width,
                                      GLsizei height,
                                      GLsizei depth,
                                      GLenum format,
                                      GLenum type,
                                      const MGLTexturePackLayout *pack_layout,
                                      void *pixels,
                                      bool swap_bytes)
{
    if (!lvl || !pack_layout || !pixels || !lvl->data ||
        xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width <= 0 || height <= 0 || depth <= 0 ||
        pack_layout->pixel_size == 0u ||
        pack_layout->row_copy_bytes == 0u ||
        lvl->width == 0u || lvl->height == 0u || lvl->pitch == 0u) {
        return false;
    }

    size_t storage_pixel_size = lvl->pitch / (size_t)lvl->width;
    if (storage_pixel_size == 0u ||
        storage_pixel_size * (size_t)lvl->width > lvl->pitch ||
        (GLuint)xoffset > lvl->width ||
        (GLuint)yoffset > lvl->height ||
        (GLuint)zoffset > lvl->depth ||
        (GLuint)width > lvl->width - (GLuint)xoffset ||
        (GLuint)height > lvl->height - (GLuint)yoffset ||
        (GLuint)depth > lvl->depth - (GLuint)zoffset) {
        return false;
    }

    /* Fast path: when the external format/type produces the exact same bit
     * layout as the internal format's CPU storage, do a raw memcpy.  This
     * avoids precision loss from per-component unpack/repack through double
     * (e.g. R11F_G11F_B10F mantissa bits being cleared) and is the critical
     * performance path for common uncompressed formats (RGBA8+UB, RGBA32F, etc). */
    if (storage_pixel_size == pack_layout->pixel_size && !swap_bytes &&
        (mglIsIdentityPackedFormat(internalformat, format, type) ||
         mglIsIdentityUncompressedFormat(internalformat, format, type))) {
        size_t src_img_size = 0u;
        size_t src_base_off = 0u;
        size_t byte_span_tmp = 0u;
        if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &src_img_size) ||
            !mglTextureRectByteRange(lvl, storage_pixel_size,
                                     (size_t)xoffset, (size_t)yoffset, (size_t)zoffset,
                                     (size_t)width, (size_t)height, (size_t)depth,
                                     &src_base_off, &byte_span_tmp)) {
            return false;
        }
        const uint8_t *src_base = (const uint8_t *)lvl->data + src_base_off;
        uint8_t *dst_base = (uint8_t *)pixels + pack_layout->skip_offset_bytes;
        size_t row_bytes = (size_t)width * storage_pixel_size;
        for (GLsizei z = 0; z < depth; z++) {
            const uint8_t *src_slice = src_base + ((size_t)z * src_img_size);
            uint8_t *dst_slice = dst_base + ((size_t)z * pack_layout->dst_image_size);
            for (GLsizei y = 0; y < height; y++) {
                memcpy(dst_slice + ((size_t)y * pack_layout->dst_pitch),
                       src_slice + ((size_t)y * lvl->pitch),
                       row_bytes);
            }
        }
        return true;
    }

    /* Fast path: RGBA→BGRA byte swap.  When the external format is BGR/BGRA
     * and the internal format is RGB/RGBA with matching 8-bit components,
     * the only conversion needed is a per-pixel B↔R byte swap (offset 0↔2). */
    if (storage_pixel_size == pack_layout->pixel_size && !swap_bytes &&
        mglIsBGRByteSwapFormat(internalformat, format, type)) {
        size_t src_img_size = 0u;
        size_t src_base_off = 0u;
        size_t byte_span_tmp = 0u;
        if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &src_img_size) ||
            !mglTextureRectByteRange(lvl, storage_pixel_size,
                                     (size_t)xoffset, (size_t)yoffset, (size_t)zoffset,
                                     (size_t)width, (size_t)height, (size_t)depth,
                                     &src_base_off, &byte_span_tmp)) {
            return false;
        }
        const uint8_t *src_base = (const uint8_t *)lvl->data + src_base_off;
        uint8_t *dst_base = (uint8_t *)pixels + pack_layout->skip_offset_bytes;
        size_t row_bytes = (size_t)width * storage_pixel_size;
        for (GLsizei z = 0; z < depth; z++) {
            const uint8_t *src_slice = src_base + ((size_t)z * src_img_size);
            uint8_t *dst_slice = dst_base + ((size_t)z * pack_layout->dst_image_size);
            for (GLsizei y = 0; y < height; y++) {
                const uint8_t *src_row = src_slice + ((size_t)y * lvl->pitch);
                uint8_t *dst_row = dst_slice + ((size_t)y * pack_layout->dst_pitch);
                memcpy(dst_row, src_row, row_bytes);
                for (GLsizei x = 0; x < width; x++) {
                    uint8_t *px = dst_row + ((size_t)x * storage_pixel_size);
                    uint8_t tmp = px[0];
                    px[0] = px[2];
                    px[2] = tmp;
                }
            }
        }
        return true;
    }

    MGLCPUPixelLayout cpu_layout;
    if (!mglBuildCPUPixelLayout(internalformat, storage_pixel_size, &cpu_layout)) {
        /* GL_RGB9_E5 has a shared exponent that cannot be handled by the
         * per-component layout model.  Handle it with a dedicated path. */
        if (internalformat == GL_RGB9_E5 && storage_pixel_size == 4u) {
            bool integer_fmt = mglExternalFormatIsInteger(format);
            size_t src_img_size = 0u;
            size_t src_base_off = 0u;
            size_t byte_span_tmp = 0u;
            if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &src_img_size) ||
                !mglTextureRectByteRange(lvl, storage_pixel_size,
                                         (size_t)xoffset, (size_t)yoffset, (size_t)zoffset,
                                         (size_t)width, (size_t)height, (size_t)depth,
                                         &src_base_off, &byte_span_tmp)) {
                return false;
            }
            const uint8_t *src_base_se = (const uint8_t *)lvl->data + src_base_off;
            uint8_t *dst_base_se = (uint8_t *)pixels + pack_layout->skip_offset_bytes;
            size_t elem_size = mglPixelTypeDatumBytes(type);
            for (GLsizei z = 0; z < depth; z++) {
                const uint8_t *src_slice = src_base_se + ((size_t)z * src_img_size);
                uint8_t *dst_slice = dst_base_se + ((size_t)z * pack_layout->dst_image_size);
                for (GLsizei y = 0; y < height; y++) {
                    const uint8_t *src_row = src_slice + ((size_t)y * lvl->pitch);
                    uint8_t *dst_row = dst_slice + ((size_t)y * pack_layout->dst_pitch);
                    for (GLsizei x = 0; x < width; x++) {
                        const uint8_t *src_pixel = src_row + ((size_t)x * storage_pixel_size);
                        uint8_t *dst_pixel = dst_row + ((size_t)x * pack_layout->pixel_size);
                        memset(dst_pixel, 0, pack_layout->pixel_size);
                        uint32_t packed = (uint32_t)mglReadUnsignedLE(src_pixel, sizeof(uint32_t));
                        /* If external type is GL_UNSIGNED_INT_5_9_9_9_REV,
                         * the data is already in native format; copy directly. */
                        if (type == GL_UNSIGNED_INT_5_9_9_9_REV && !integer_fmt) {
                            mglWriteUnsignedLE(dst_pixel, sizeof(uint32_t), packed);
                            if (swap_bytes) {
                                mglSwapPixelBytes(dst_pixel, sizeof(uint32_t), sizeof(uint32_t));
                            }
                            continue;
                        }
                        double r, g, b;
                        mglUnpackSharedExp(packed, &r, &g, &b);
                        /* Map GL_RGB components to external format order.
                         * Skip components that are not present in the external
                         * format (e.g. GL_RED only has R, so G and B are skipped). */
                        for (int c = 0; c < 3; c++) {
                            int dst_idx = mglExternalSourceIndexForComponent(format, (GLuint)c);
                            if (dst_idx < 0) continue;
                            double val = (c == 0) ? r : (c == 1) ? g : b;
                            mglWriteExternalComponent(dst_pixel, type, dst_idx, integer_fmt, val);
                        }
                        if (swap_bytes && elem_size > 1u) {
                            mglSwapPixelBytes(dst_pixel, pack_layout->pixel_size, elem_size);
                        }
                    }
                }
            }
            return true;
        }

        /* Depth/stencil combined formats: convert from canonical CPU
         * shadow storage to the requested external format.
         * GL_DEPTH32F_STENCIL8: 5 bytes (float depth + uint8 stencil)
         * GL_DEPTH24_STENCIL8: 4 bytes (uint32: 24-bit depth high [31:8], 8-bit stencil low [7:0]) */
        if (mglInternalFormatIsCombinedDepthStencil(internalformat) &&
            (format == GL_DEPTH_STENCIL || format == GL_DEPTH_COMPONENT ||
             format == GL_STENCIL_INDEX)) {
            bool integer_fmt = mglExternalFormatIsInteger(format);
            size_t src_img_size = 0u;
            size_t src_base_off = 0u;
            size_t byte_span_tmp = 0u;
            if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &src_img_size) ||
                !mglTextureRectByteRange(lvl, storage_pixel_size,
                                         (size_t)xoffset, (size_t)yoffset, (size_t)zoffset,
                                         (size_t)width, (size_t)height, (size_t)depth,
                                         &src_base_off, &byte_span_tmp)) {
                return false;
            }
            const uint8_t *src_base_ds = (const uint8_t *)lvl->data + src_base_off;
            uint8_t *dst_base_ds = (uint8_t *)pixels + pack_layout->skip_offset_bytes;
            size_t elem_size = mglPixelTypeDatumBytes(type);
            for (GLsizei z = 0; z < depth; z++) {
                const uint8_t *src_slice = src_base_ds + ((size_t)z * src_img_size);
                uint8_t *dst_slice = dst_base_ds + ((size_t)z * pack_layout->dst_image_size);
                for (GLsizei y = 0; y < height; y++) {
                    const uint8_t *src_row = src_slice + ((size_t)y * lvl->pitch);
                    uint8_t *dst_row = dst_slice + ((size_t)y * pack_layout->dst_pitch);
                    for (GLsizei x = 0; x < width; x++) {
                        const uint8_t *src_pixel = src_row + ((size_t)x * storage_pixel_size);
                        uint8_t *dst_pixel = dst_row + ((size_t)x * pack_layout->pixel_size);
                        memset(dst_pixel, 0, pack_layout->pixel_size);
                        GLfloat depthVal = 0.0f;
                        uint8_t stencilVal = 0u;
                        if (internalformat == GL_DEPTH32F_STENCIL8) {
                            memcpy(&depthVal, src_pixel, sizeof(float));
                            if (storage_pixel_size >= 5u)
                                stencilVal = src_pixel[4];
                        } else { /* GL_DEPTH24_STENCIL8 */
                            uint32_t packed;
                            memcpy(&packed, src_pixel, sizeof(uint32_t));
                            depthVal = (GLfloat)(packed >> 8) / 16777215.0f;
                            stencilVal = (uint8_t)(packed & 0xffu);
                        }
                        if (format == GL_DEPTH_STENCIL) {
                            if (type == GL_UNSIGNED_INT_24_8) {
                                uint32_t packed = ((uint32_t)(depthVal * 16777215.0f + 0.5f) << 8) |
                                                  (uint32_t)stencilVal;
                                memcpy(dst_pixel, &packed, sizeof(uint32_t));
                            } else if (type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV) {
                                memcpy(dst_pixel, &depthVal, sizeof(float));
                                uint32_t s = (uint32_t)stencilVal;
                                memcpy(dst_pixel + 4, &s, sizeof(uint32_t));
                            }
                        } else if (format == GL_DEPTH_COMPONENT) {
                            mglWriteExternalComponent(dst_pixel, type, 0, integer_fmt, (double)depthVal);
                        } else { /* GL_STENCIL_INDEX */
                            mglWriteExternalComponent(dst_pixel, type, 0, true, (double)stencilVal);
                        }
                        if (swap_bytes && elem_size > 1u) {
                            mglSwapPixelBytes(dst_pixel, pack_layout->pixel_size, elem_size);
                        }
                    }
                }
            }
            return true;
        }
        return false;
    }

    bool integer_format = mglExternalFormatIsInteger(format);
    size_t src_image_size = 0u;
    size_t src_base_offset = 0u;
    size_t byte_span = 0u;
    if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &src_image_size) ||
        !mglTextureRectByteRange(lvl,
                                 storage_pixel_size,
                                 (size_t)xoffset,
                                 (size_t)yoffset,
                                 (size_t)zoffset,
                                 (size_t)width,
                                 (size_t)height,
                                 (size_t)depth,
                                 &src_base_offset,
                                 &byte_span)) {
        return false;
    }

    const uint8_t *src_base = (const uint8_t *)lvl->data + src_base_offset;
    uint8_t *dst_base = (uint8_t *)pixels + pack_layout->skip_offset_bytes;

    for (GLsizei z = 0; z < depth; z++) {
        const uint8_t *src_slice = src_base + ((size_t)z * src_image_size);
        uint8_t *dst_slice = dst_base + ((size_t)z * pack_layout->dst_image_size);
        for (GLsizei y = 0; y < height; y++) {
            const uint8_t *src_row = src_slice + ((size_t)y * lvl->pitch);
            uint8_t *dst_row = dst_slice + ((size_t)y * pack_layout->dst_pitch);
            for (GLsizei x = 0; x < width; x++) {
                const uint8_t *src_pixel = src_row + ((size_t)x * storage_pixel_size);
                uint8_t *dst_pixel = dst_row + ((size_t)x * pack_layout->pixel_size);
                memset(dst_pixel, 0, pack_layout->pixel_size);

                /* GL_UNSIGNED_INT_5_9_9_9_REV (GL_RGB9_E5) packs 3 RGB
                 * mantissas into a single 32-bit word with one shared
                 * 5-bit exponent. The per-component path cannot handle this
                 * because each mantissa must use the same exponent (computed
                 * from the max of the 3 values). Handle it specially. */
                if (type == GL_UNSIGNED_INT_5_9_9_9_REV &&
                    !integer_format &&
                    pack_layout->pixel_size >= sizeof(uint32_t)) {
                    double r = (cpu_layout.component_count > 0u)
                        ? mglLoadInternalComponent(src_pixel, &cpu_layout.components[0])
                        : 0.0;
                    double g = (cpu_layout.component_count > 1u)
                        ? mglLoadInternalComponent(src_pixel, &cpu_layout.components[1])
                        : 0.0;
                    double b = (cpu_layout.component_count > 2u)
                        ? mglLoadInternalComponent(src_pixel, &cpu_layout.components[2])
                        : 0.0;
                    uint32_t packed = mglPackRGBToSharedExp(r, g, b);
                    mglWriteUnsignedLE(dst_pixel, sizeof(uint32_t), packed);
                    if (swap_bytes) {
                        mglSwapPixelBytes(dst_pixel, sizeof(uint32_t), sizeof(uint32_t));
                    }
                    continue;
                }

                for (GLuint component = 0; component < 4u; component++) {
                    int dst_index = mglExternalSourceIndexForComponent(format, component);
                    if (dst_index < 0 || component >= cpu_layout.component_count) {
                        continue;
                    }
                    double value = mglLoadInternalComponent(src_pixel, &cpu_layout.components[component]);
                    mglWriteExternalComponent(dst_pixel, type, dst_index, integer_format, value);
                }

                if (swap_bytes) {
                    size_t element_size = mglPixelTypeDatumBytes(type);
                    if (element_size > 1u) {
                        mglSwapPixelBytes(dst_pixel, pack_layout->pixel_size, element_size);
                    }
                }
            }
        }
    }

    return true;
}

bool mglCopyTextureLevelToPackBuffer(TextureLevel *lvl,
                                            GLenum internalformat,
                                            GLsizei width,
                                            GLsizei height,
                                            GLsizei depth,
                                            GLenum format,
                                            GLenum type,
                                            const MGLTexturePackLayout *pack_layout,
                                            void *pixels,
                                            bool swap_bytes)
{
    if (mglCopyTextureRectFromCPU(internalformat,
                                  lvl,
                                  0,
                                  0,
                                  0,
                                  width,
                                  height,
                                  depth,
                                  format,
                                  type,
                                  pack_layout,
                                  pixels,
                                  swap_bytes)) {
        return true;
    }

    if (!lvl || !pack_layout || !pixels || !lvl->data ||
        width <= 0 || height <= 0 || depth <= 0 ||
        pack_layout->pixel_size == 0u ||
        pack_layout->row_copy_bytes == 0u ||
        lvl->pitch < pack_layout->row_copy_bytes ||
        lvl->height == 0u) {
        return false;
    }

    size_t src_image_size = 0u;
    size_t last_src_offset = 0u;
    size_t last_src_span = 0u;
    if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &src_image_size) ||
        !mglMulSizeT(src_image_size, (size_t)(depth - 1), &last_src_offset) ||
        !mglMulSizeT(lvl->pitch, (size_t)(height - 1), &last_src_span) ||
        !mglAddSizeT(last_src_offset, last_src_span, &last_src_offset) ||
        !mglAddSizeT(last_src_offset, pack_layout->row_copy_bytes, &last_src_span) ||
        last_src_span > lvl->data_size) {
        return false;
    }

    const uint8_t *src_base = (const uint8_t *)lvl->data;
    uint8_t *dst_base = (uint8_t *)pixels + pack_layout->skip_offset_bytes;

    for (GLsizei z = 0; z < depth; z++) {
        const uint8_t *src_slice = src_base + ((size_t)z * src_image_size);
        uint8_t *dst_slice = dst_base + ((size_t)z * pack_layout->dst_image_size);
        for (GLsizei y = 0; y < height; y++) {
            memcpy(dst_slice + ((size_t)y * pack_layout->dst_pitch),
                   src_slice + ((size_t)y * lvl->pitch),
                   pack_layout->row_copy_bytes);
        }
    }

    return true;
}

bool mglCopyTextureSubRectToPackBuffer(TextureLevel *lvl,
                                              GLenum internalformat,
                                              GLint xoffset,
                                              GLint yoffset,
                                              GLint zoffset,
                                              GLsizei width,
                                              GLsizei height,
                                              GLsizei depth,
                                              GLenum format,
                                              GLenum type,
                                              const MGLTexturePackLayout *pack_layout,
                                              void *pixels,
                                              bool swap_bytes)
{
    if (mglCopyTextureRectFromCPU(internalformat,
                                  lvl,
                                  xoffset,
                                  yoffset,
                                  zoffset,
                                  width,
                                  height,
                                  depth,
                                  format,
                                  type,
                                  pack_layout,
                                  pixels,
                                  swap_bytes)) {
        return true;
    }

    if (!lvl || !pack_layout || !pixels || !lvl->data ||
        width <= 0 || height <= 0 || depth <= 0 ||
        xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        pack_layout->pixel_size == 0u ||
        pack_layout->row_copy_bytes == 0u ||
        lvl->pitch < ((size_t)xoffset * pack_layout->pixel_size) + pack_layout->row_copy_bytes ||
        lvl->height == 0u) {
        return false;
    }

    size_t src_image_size = 0u;
    size_t src_base_offset = 0u;
    size_t byte_span = 0u;
    if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &src_image_size) ||
        !mglTextureRectByteRange(lvl,
                                 pack_layout->pixel_size,
                                 (size_t)xoffset,
                                 (size_t)yoffset,
                                 (size_t)zoffset,
                                 (size_t)width,
                                 (size_t)height,
                                 (size_t)depth,
                                 &src_base_offset,
                                 &byte_span)) {
        return false;
    }

    const uint8_t *src_base = (const uint8_t *)lvl->data + src_base_offset;
    uint8_t *dst_base = (uint8_t *)pixels + pack_layout->skip_offset_bytes;

    for (GLsizei z = 0; z < depth; z++) {
        const uint8_t *src_slice = src_base + ((size_t)z * src_image_size);
        uint8_t *dst_slice = dst_base + ((size_t)z * pack_layout->dst_image_size);
        for (GLsizei y = 0; y < height; y++) {
            memcpy(dst_slice + ((size_t)y * pack_layout->dst_pitch),
                   src_slice + ((size_t)y * lvl->pitch),
                   pack_layout->row_copy_bytes);
        }
    }

    return true;
}

bool mglClearTextureLevelCPU(TextureLevel *lvl,
                                    GLenum internalformat,
                                    GLint xoffset,
                                    GLint yoffset,
                                    GLint zoffset,
                                    GLsizei width,
                                    GLsizei height,
                                    GLsizei depth,
                                    GLenum format,
                                    GLenum type,
                                    const void *data)
{
    if (width == 0 || height == 0 || depth == 0) {
        return true;
    }

    if (!lvl || !lvl->data || !lvl->complete ||
        xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width < 0 || height < 0 || depth < 0 ||
        (GLuint)xoffset > lvl->width ||
        (GLuint)yoffset > lvl->height ||
        (GLuint)zoffset > lvl->depth ||
        (GLuint)width > lvl->width - (GLuint)xoffset ||
        (GLuint)height > lvl->height - (GLuint)yoffset ||
        (GLuint)depth > lvl->depth - (GLuint)zoffset ||
        lvl->width == 0u || lvl->height == 0u || lvl->pitch == 0u) {
        return false;
    }

    size_t clear_pixel_size = sizeForFormatType(format, type);
    if (clear_pixel_size == 0u) {
        return false;
    }

    size_t storage_pixel_size = lvl->pitch / (size_t)lvl->width;
    if (storage_pixel_size == 0u ||
        storage_pixel_size > 64u ||
        storage_pixel_size * (size_t)lvl->width > lvl->pitch) {
        return false;
    }

    if (mglFillTextureRectCPU(internalformat, lvl, xoffset, yoffset, zoffset, width, height, depth, format, type, data)) {
        lvl->ever_written = GL_TRUE;
        lvl->has_initialized_data = GL_TRUE;
        lvl->suspicious_zero_upload = GL_FALSE;
        lvl->metal_data_authoritative = GL_FALSE;
        lvl->last_init_source = kTexSubImageCPU;
        lvl->last_upload_size = storage_pixel_size * (size_t)width * (size_t)height * (size_t)depth;
        lvl->last_src_ptr = data;
        lvl->last_src_hash = data ? mglHashBytesSampled(data, clear_pixel_size) : 0ull;
        return true;
    }

    uint8_t clear_pixel[64];
    memset(clear_pixel, 0, storage_pixel_size);
    if (data) {
        size_t copy_bytes = clear_pixel_size < storage_pixel_size ? clear_pixel_size : storage_pixel_size;
        memcpy(clear_pixel, data, copy_bytes);
    }

    if (format == GL_RED || format == GL_RG || format == GL_RGB ||
        format == GL_BGR || format == GL_RED_INTEGER || format == GL_RG_INTEGER ||
        format == GL_RGB_INTEGER || format == GL_BGR_INTEGER) {
        mglStoreDefaultAlpha(clear_pixel, storage_pixel_size, type);
    }

    size_t image_pitch = 0u;
    if (!mglMulSizeT(lvl->pitch, (size_t)lvl->height, &image_pitch)) {
        return false;
    }

    uint8_t *base = (uint8_t *)lvl->data +
                    ((size_t)zoffset * image_pitch) +
                    ((size_t)yoffset * lvl->pitch) +
                    ((size_t)xoffset * storage_pixel_size);
    for (GLsizei z = 0; z < depth; z++) {
        uint8_t *slice = base + ((size_t)z * image_pitch);
        for (GLsizei y = 0; y < height; y++) {
            uint8_t *row = slice + ((size_t)y * lvl->pitch);
            for (GLsizei x = 0; x < width; x++) {
                memcpy(row + ((size_t)x * storage_pixel_size), clear_pixel, storage_pixel_size);
            }
        }
    }

    lvl->ever_written = GL_TRUE;
    lvl->has_initialized_data = GL_TRUE;
    lvl->suspicious_zero_upload = GL_FALSE;
    lvl->metal_data_authoritative = GL_FALSE;
    lvl->last_init_source = kTexSubImageCPU;
    lvl->last_upload_size = storage_pixel_size * (size_t)width * (size_t)height * (size_t)depth;
    lvl->last_src_ptr = data;
    lvl->last_src_hash = data ? mglHashBytesSampled(data, clear_pixel_size) : 0ull;
    return true;
}

bool mglComputeTextureUnpackLayout(GLMContext ctx,
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
    GLint skip_images = ctx->state.unpack.skip_images;
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

void mglHandleProxyTexImageQuery(GLMContext ctx,
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

bool mglResolveTexSubImageSource(GLMContext ctx,
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
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }

        size_t pbo_size = (size_t)unpack_buf->size;
        size_t raw_off = (size_t)raw_value;
        size_t effective_off = 0u;
        size_t end_off = 0u;

        /* D: when a PBO is bound, `pixels` is a byte offset that must be evenly
         * divisible by the size in bytes of a single datum indicated by `type`. */
        size_t datum_bytes = mglPixelTypeDatumBytes(type);
        if (datum_bytes > 0u && (raw_off % datum_bytes) != 0u) {
            fprintf(stderr,
                    "MGL ERROR: texSubImage source resolve: PBO offset not divisible by datum unpack=%u off=%zu datum=%zu type=0x%x\n",
                    unpack_name,
                    raw_off,
                    datum_bytes,
                    type);
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }

        if (!mglAddSizeT(raw_off, skip_offset_bytes, &effective_off)) {
            fprintf(stderr,
                    "MGL ERROR: texSubImage source resolve: PBO offset addition overflow unpack=%u rawOff=%zu skipOff=%zu\n",
                    unpack_name,
                    raw_off,
                    skip_offset_bytes);
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }

        if (required_bytes > 0u) {
            if (!mglAddSizeT(effective_off, required_bytes, &end_off)) {
                fprintf(stderr,
                        "MGL ERROR: texSubImage source resolve: PBO required range overflow unpack=%u off=%zu required=%zu\n",
                        unpack_name,
                        effective_off,
                        required_bytes);
                ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
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
                ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
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
bool mglVerifyInternalFormatAndFormatTypeForCall(GLMContext ctx, GLint internalformat, GLenum format, GLenum type)
{
    GLuint old_error_count = ctx ? ctx->state.error_count : 0u;
    GLenum old_error = ctx ? ctx->state.error : GL_NO_ERROR;

    if (verifyInternalFormatAndFormatType(ctx, internalformat, format, type)) {
        return true;
    }

    if (ctx &&
        ctx->state.error_count == old_error_count &&
        ctx->state.error == old_error) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    return false;
}
bool mglStoreCompressedTextureImage(GLMContext ctx,
                                           GLenum target,
                                           GLint level,
                                           GLenum internalformat,
                                           GLsizei width,
                                           GLsizei height,
                                           GLsizei depth,
                                           GLint border,
                                           GLsizei imageSize,
                                           const void *data)
{
    if (level < 0 || width < 0 || height < 0 || depth < 0 || border != 0 || imageSize < 0) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }
    if (!mglTexLevelInternalFormatCompressed(internalformat)) {
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!tex) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    GLuint face = 0u;
    if (target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X &&
        target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z) {
        face = (GLuint)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X);
    }

    if (level == 0) {
        if (tex->mipmap_levels == 0) {
            initBaseTexLevel(ctx, tex, internalformat, width, height, depth);
            if (STATE(error) != GL_NO_ERROR) {
                return false;
            }
            tex->compressed_internalformat = internalformat;
            /* Compressed textures created via glCompressedTexImage* are
             * sampler-readable by default, not image-binding targets; mirror
             * the GL_READ_ONLY default glTexImage* uses (textures.c:5883) so
             * the Metal-texture-creation path's switch(tex->access) doesn't
             * fall through to return nil. */
            tex->access = GL_READ_ONLY;
        } else if (tex->width != (GLuint)width ||
                   tex->height != (GLuint)height ||
                   tex->depth != (GLuint)depth ||
                   tex->internalformat != internalformat) {
            invalidateTexture(ctx, tex);
            initBaseTexLevel(ctx, tex, internalformat, width, height, depth);
            if (STATE(error) != GL_NO_ERROR) {
                return false;
            }
            tex->compressed_internalformat = internalformat;
            tex->access = GL_READ_ONLY;
        }
    } else if (tex->mipmap_levels == 0 || tex->internalformat != internalformat) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    if (!ensureTextureLevelCapacity(ctx, tex, (GLuint)level + 1u)) {
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
    }

    TextureLevel *lvl = &tex->faces[face].levels[level];
    if (lvl->data && lvl->data_size > 0u) {
        vm_deallocate((vm_map_t)mach_task_self(), lvl->data, lvl->data_size);
        lvl->data = 0;
        lvl->data_size = 0u;
    }

    /*
     * When GL_PIXEL_UNPACK_BUFFER is bound, `data` is a byte offset into that
     * buffer, not a CPU pointer.  Resolve it (rejecting mapped / out-of-range
     * sources with a real GL error) before memcpy/hash, otherwise a small offset
     * dereferences an unmapped address and segfaults.  No PBO bound: data is a
     * CPU pointer (may be NULL) and is used as-is.
     */
    const uint8_t *resolved_src = (const uint8_t *)data;
    size_t resolved_src_available = SIZE_MAX;
    Buffer *unpack_buf = STATE(buffers[_PIXEL_UNPACK_BUFFER]);
    if (unpack_buf) {
        const uint8_t *pbo_data = (const uint8_t *)getBufferData(ctx, unpack_buf);
        if (unpack_buf->mapped || !pbo_data) {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }
        uintptr_t raw_off = (uintptr_t)data;
        if (raw_off > (uintptr_t)unpack_buf->size ||
            (size_t)imageSize > (size_t)unpack_buf->size - raw_off) {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }
        resolved_src_available = (size_t)unpack_buf->size - raw_off;
        resolved_src = pbo_data + raw_off;
    }

    if (imageSize > 0) {
        /* Repack block-compressed data into the destination (tightly packed)
         * layout required by Metal when GL_UNPACK_COMPRESSED_BLOCK_WIDTH and
         * GL_UNPACK_COMPRESSED_BLOCK_SIZE enable compressed pixel storage and
         * the GL unpack state specifies a non-tight source layout.  Per GL 4.6
         * section 8.4.4 the source stride per block row is
         *   ceil(row_length / block_w) * block_bytes
         * and per source image is
         *   block_rows_per_image * src_blocks_per_row * block_bytes
         * where block_rows_per_image is ceil(image_height / block_h) (or
         * ceil(height / block_h) when UNPACK_IMAGE_HEIGHT is 0).  When layout
         * is already tight, fall through to a plain memcpy of imageSize bytes
         * (matches historic behaviour). */
        GLint user_cbw = ctx->state.unpack.compressed_block_width;
        GLint user_cbh = ctx->state.unpack.compressed_block_height;
        GLint user_cbd = ctx->state.unpack.compressed_block_depth;
        GLint user_cbs = ctx->state.unpack.compressed_block_size;
        bool compressed_pixel_store_active = (user_cbw > 0 && user_cbs > 0);
        GLuint ubw = 0, ubh = 0, ubs = 0, ubd = 1;
        GLint row_length  = ctx->state.unpack.row_length;
        GLint image_height = (depth > 1) ? ctx->state.unpack.image_height : 0;
        GLint skip_p = ctx->state.unpack.skip_pixels;
        GLint skip_r = (height > 1) ? ctx->state.unpack.skip_rows : 0;
        GLint skip_i = (depth > 1) ? ctx->state.unpack.skip_images : 0;

        if (compressed_pixel_store_active) {
            GLuint fmt_bh = 0, fmt_bd = 1;
            (void)mglCompressedBlockInfoOf(internalformat, NULL, &fmt_bh, &fmt_bd, NULL);
            ubw = (GLuint)user_cbw;
            ubh = (user_cbh > 0) ? (GLuint)user_cbh : fmt_bh;
            ubd = (user_cbd > 0) ? (GLuint)user_cbd : fmt_bd;
            ubs = (GLuint)user_cbs;
            if (height <= 1 && ubh == 0) {
                ubh = 1;
            }
            if (depth <= 1 && ubd == 0) {
                ubd = 1;
            }
            if (ubw == 0 || ubs == 0 ||
                (height > 1 && ubh == 0) ||
                (depth > 1 && ubd == 0)) {
                ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
            }
            if ((skip_p % user_cbw) != 0 ||
                (height > 1 && (skip_r % (GLint)ubh) != 0) ||
                (depth > 1 && (skip_i % (GLint)ubd) != 0)) {
                ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
            }
        }

        bool needs_repack = compressed_pixel_store_active &&
                            ((row_length > 0 && (GLuint)row_length  != (GLuint)width)  ||
                             (image_height > 0 && (GLuint)image_height != (GLuint)height) ||
                             skip_p > 0 || skip_r > 0 || skip_i > 0 ||
                             depth > 1);

        bool do_repack = needs_repack && ubw != 0 && ubh != 0 && ubd != 0 && ubs != 0;
        GLuint dst_blocks_per_row = 0, dst_block_rows = 0, dst_block_depths = 0;
        GLuint src_blocks_per_row = 0, src_rows_per_image = 0;
        size_t src_row_bytes = 0, src_image_bytes = 0, skip_offset = 0;
        size_t dst_row_bytes = 0, dst_image_bytes = 0, dst_total_bytes = 0;
        size_t alloc_size = (size_t)imageSize;

        if (compressed_pixel_store_active && !do_repack) {
            if (!mglMulSizeT((size_t)((width  + (GLint)ubw - 1) / (GLint)ubw), (size_t)ubs, &dst_row_bytes) ||
                !mglMulSizeT(dst_row_bytes, (size_t)((height + (GLint)ubh - 1) / (GLint)ubh), &dst_image_bytes) ||
                !mglMulSizeT(dst_image_bytes, (size_t)((depth  + (GLint)ubd - 1) / (GLint)ubd), &dst_total_bytes)) {
                ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
            }
            if ((size_t)imageSize != dst_total_bytes) {
                ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
            }
        }

        if (do_repack) {
            dst_blocks_per_row = ((GLuint)(width  + (GLint)ubw - 1) / ubw);
            dst_block_rows    = ((GLuint)(height + (GLint)ubh - 1) / ubh);
            dst_block_depths  = ((GLuint)(depth  + (GLint)ubd - 1) / ubd);

            src_blocks_per_row = (row_length > 0)
                ? ((GLuint)(row_length + (GLint)ubw - 1) / ubw)
                : dst_blocks_per_row;
            src_rows_per_image = (image_height > 0)
                ? ((GLuint)(image_height + (GLint)ubh - 1) / ubh)
                : dst_block_rows;

            GLuint skip_block_rows = (GLuint)(skip_r / (GLint)ubh);
            GLuint skip_block_cols = (GLuint)(skip_p / (GLint)ubw);
            GLuint skip_block_imgs = (GLuint)(skip_i / (GLint)ubd);

            size_t skip_imgs_bytes = 0, skip_rows_bytes = 0, skip_cols_bytes = 0;
            size_t trailing_img_bytes = 0, trailing_row_bytes = 0, source_span = 0;
            if (!mglMulSizeT((size_t)src_blocks_per_row, (size_t)ubs, &src_row_bytes) ||
                !mglMulSizeT(src_row_bytes, (size_t)src_rows_per_image, &src_image_bytes) ||
                !mglMulSizeT((size_t)skip_block_imgs, src_image_bytes, &skip_imgs_bytes) ||
                !mglMulSizeT((size_t)skip_block_rows, src_row_bytes, &skip_rows_bytes) ||
                !mglMulSizeT((size_t)skip_block_cols, (size_t)ubs, &skip_cols_bytes) ||
                !mglAddSizeT(skip_imgs_bytes, skip_rows_bytes, &skip_offset) ||
                !mglAddSizeT(skip_offset, skip_cols_bytes, &skip_offset) ||
                !mglMulSizeT((size_t)dst_blocks_per_row, (size_t)ubs, &dst_row_bytes) ||
                !mglMulSizeT(dst_row_bytes, (size_t)dst_block_rows, &dst_image_bytes) ||
                !mglMulSizeT(dst_image_bytes, (size_t)dst_block_depths, &dst_total_bytes) ||
                !mglMulSizeT((size_t)(dst_block_depths ? dst_block_depths - 1u : 0u), src_image_bytes, &trailing_img_bytes) ||
                !mglMulSizeT((size_t)(dst_block_rows ? dst_block_rows - 1u : 0u), src_row_bytes, &trailing_row_bytes) ||
                !mglAddSizeT(skip_offset, trailing_img_bytes, &source_span) ||
                !mglAddSizeT(source_span, trailing_row_bytes, &source_span) ||
                !mglAddSizeT(source_span, dst_row_bytes, &source_span)) {
                ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
            }
            if (dst_total_bytes == 0u || (size_t)imageSize != dst_total_bytes) {
                ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
            }
            if (resolved_src && source_span > resolved_src_available) {
                ERROR_RETURN_VALUE(unpack_buf ? GL_INVALID_OPERATION : GL_INVALID_VALUE, false);
            }
            alloc_size = dst_total_bytes;
        }

        vm_address_t compressed_data = 0;
        kern_return_t kr = vm_allocate((vm_map_t)mach_task_self(),
                                       &compressed_data,
                                       alloc_size,
                                       VM_FLAGS_ANYWHERE);
        if (kr != KERN_SUCCESS || !compressed_data) {
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }

        if (!resolved_src) {
            memset((void *)(uintptr_t)compressed_data, 0, alloc_size);
        } else if (!do_repack) {
            memcpy((void *)(uintptr_t)compressed_data, resolved_src, alloc_size);
        } else {
            uint8_t *dst = (uint8_t *)(uintptr_t)compressed_data;
            const uint8_t *src_base = resolved_src + skip_offset;

            for (GLuint di = 0; di < dst_block_depths; di++) {
                const uint8_t *src_img = src_base + (size_t)di * src_image_bytes;
                uint8_t *dst_img = dst + (size_t)di * dst_image_bytes;
                for (GLuint dr = 0; dr < dst_block_rows; dr++) {
                    const uint8_t *src_row = src_img + (size_t)dr * src_row_bytes;
                    uint8_t *dst_row = dst_img + (size_t)dr * dst_row_bytes;
                    memcpy(dst_row, src_row, dst_row_bytes);
                }
            }
        }
        lvl->data = compressed_data;
        lvl->data_size = alloc_size;
    }

    lvl->width = (GLuint)width;
    lvl->height = (GLuint)height;
    lvl->depth = (GLuint)depth;
    /* For block-compressed formats, pitch is the byte stride per row of blocks
     * (ceil(width/block_w) * bytes_per_block); the Metal upload path requires
     * a non-zero bytesPerRow to actually copy data into the compressed
     * MTLPixelFormat texture.  For unknown/uncompressed formats this falls
     * back to 0 (the historic behaviour). */
    lvl->pitch = mglCompressedBytesPerRowOf(internalformat, width);
    lvl->mtl_format = 0u;
    lvl->complete = true;
    lvl->has_initialized_data = resolved_src ? GL_TRUE : GL_FALSE;
    lvl->ever_written = resolved_src ? GL_TRUE : GL_FALSE;
    lvl->suspicious_zero_upload = GL_FALSE;
    lvl->metal_data_authoritative = GL_FALSE;
    lvl->last_init_source = (resolved_src && unpack_buf) ? kTexImagePBO
                                                         : (resolved_src ? kTexImageCopy : kTexImageNull);
    lvl->last_upload_size = (size_t)imageSize;
    lvl->last_src_ptr = resolved_src;
    lvl->last_src_hash = resolved_src ? mglHashBytesSampled(resolved_src, (size_t)imageSize) : 0ull;

    tex->num_levels = MAX(tex->num_levels, (GLuint)level + 1u);
    tex->complete = GL_TRUE;
    tex->dirty_bits |= DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA;
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);
    return true;
}
bool mglCompressedSubImageUpdate(GLMContext ctx,
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
                                        GLsizei imageSize,
                                        const void *data)
{
    if (!tex) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }
    if (level < 0 || xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width < 0 || height < 0 || depth < 0 || imageSize < 0) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }
    if (!mglTexLevelInternalFormatCompressed(format)) {
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
    }
    /* G (1D): block formats that require a height are not valid for a 1-D texture. */
    if (tex->target == GL_TEXTURE_1D && mglCompressedFormatRequiresHeight(format)) {
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
    }
    if (face >= _CUBE_MAP_MAX_FACE ||
        level >= (GLint)tex->num_levels ||
        !tex->faces[face].levels ||
        !tex->faces[face].levels[level].complete) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    TextureLevel *lvl = &tex->faces[face].levels[level];
    if ((GLuint)xoffset > lvl->width ||
        (GLuint)yoffset > lvl->height ||
        (GLuint)zoffset > lvl->depth ||
        (GLuint)width > lvl->width - (GLuint)xoffset ||
        (GLuint)height > lvl->height - (GLuint)yoffset ||
        (GLuint)depth > lvl->depth - (GLuint)zoffset) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    /* I: imageSize must be consistent with the compressed image data the
     * implementation reports for this level.  MGL stores the generic compressed
     * internalformats uncompressed and reports GL_TEXTURE_COMPRESSED_IMAGE_SIZE
     * as the (page-aligned) level data_size, so a CompressedTexSubImage update
     * whose imageSize differs from that reported size is inconsistent. */
    if ((size_t)imageSize != lvl->data_size) {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    /* J: CompressedTexSubImage is not allowed on TEXTURE_RECTANGLE textures. */
    if (tex->target == GL_TEXTURE_RECTANGLE) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    /*
     * When GL_PIXEL_UNPACK_BUFFER is bound, `data` is a byte offset into that
     * buffer, not a CPU pointer.  Resolve it (and reject mapped / out-of-range
     * sources with a real GL error) before any deref, otherwise hashing a small
     * offset like 0x400 segfaults.  No PBO bound: data is a CPU pointer (may be
     * NULL) and is used as-is, matching the historical behaviour.
     */
    const uint8_t *resolved_src = (const uint8_t *)data;
    Buffer *unpack_buf = STATE(buffers[_PIXEL_UNPACK_BUFFER]);
    if (unpack_buf) {
        const uint8_t *pbo_data = (const uint8_t *)getBufferData(ctx, unpack_buf);
        if (unpack_buf->mapped || !pbo_data) {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }
        uintptr_t raw_off = (uintptr_t)data;
        if (raw_off > (uintptr_t)unpack_buf->size ||
            (size_t)imageSize > (size_t)unpack_buf->size - raw_off) {
            ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
        }
        resolved_src = pbo_data + raw_off;
    }

    /* H: the update format must match the texture's stored internalformat. */
    if (format != (GLenum)tex->internalformat) {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, false);
    }

    /* G: the six generic compressed formats are never a valid CompressedTexSubImage
     * format.  Checked last so that dimension / imageSize / rectangle / PBO errors
     * (which CTS probes with these same generic values) take precedence. */
    if (mglIsGenericCompressedFormat(format)) {
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
    }

    lvl->has_initialized_data = GL_TRUE;
    lvl->ever_written = GL_TRUE;
    lvl->suspicious_zero_upload = GL_FALSE;
    lvl->metal_data_authoritative = GL_FALSE;
    lvl->last_init_source = unpack_buf ? kTexSubImagePBO : kTexSubImageCPU;
    lvl->last_upload_size = (size_t)imageSize;
    lvl->last_src_ptr = resolved_src;
    lvl->last_src_hash = resolved_src ? mglHashBytesSampled(resolved_src, (size_t)imageSize) : 0ull;
    tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    mglReleaseGLSampledTextureCopy(ctx, tex, "compressedTexSubImage");
    mglMarkStateDirtyBits(ctx->active_state, DIRTY_TEX);
    return true;
}
bool mglCopyTextureSubImageValidate(GLMContext ctx,
                                           Texture *tex,
                                           GLint level,
                                           GLint xoffset,
                                           GLint yoffset,
                                           GLint zoffset,
                                           GLsizei width,
                                           GLsizei height)
{
    if (!tex ||
        level < 0 ||
        level >= (GLint)tex->num_levels ||
        !tex->faces[0].levels ||
        !tex->faces[0].levels[level].complete) {
        return false;
    }

    TextureLevel *lvl = &tex->faces[0].levels[level];
    if (xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        width < 0 || height < 0) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return false;
    }

    if ((GLuint)xoffset > lvl->width ||
        (GLuint)yoffset > lvl->height ||
        (GLuint)zoffset > lvl->depth ||
        (GLuint)width > lvl->width - (GLuint)xoffset ||
        (GLuint)height > lvl->height - (GLuint)yoffset) {
        ERROR_RETURN(GL_INVALID_VALUE);
        return false;
    }

    return true;
}
