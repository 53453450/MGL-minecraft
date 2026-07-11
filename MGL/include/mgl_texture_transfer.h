/*
 * mgl_texture_transfer.h
 * MGL
 *
 * CPU upload / download / transfer helper functions extracted from
 * textures.c as part of the God Object decomposition (Task 9).  These
 * functions handle pixel pack/unpack layout computation, CPU-side texture
 * rect conversion / fill / copy / clear, compressed texture image storage,
 * texSubImage source resolution and validation.
 */

#ifndef mgl_texture_transfer_h
#define mgl_texture_transfer_h

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <mach/vm_types.h>

#include "glm_context.h"

/* ---------------------------------------------------------------------------
 * Layout descriptors shared between the pack (readback) and unpack (upload)
 * paths.  Moved here from textures.c so both textures.c and
 * mgl_texture_transfer.c can use them.
 * ------------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------------
 * Pack / unpack layout computation.
 * ------------------------------------------------------------------------- */

bool mglComputeTexturePackLayout(GLMContext ctx,
                                 GLsizei width,
                                 GLsizei height,
                                 GLsizei depth,
                                 size_t pixel_size,
                                 const char *op,
                                 MGLTexturePackLayout *layout);

bool mglComputeTextureUnpackLayout(GLMContext ctx,
                                   GLsizei width,
                                   GLsizei height,
                                   GLsizei depth,
                                   size_t pixel_size,
                                   const char *op,
                                   MGLTextureUnpackLayout *layout);

/* ---------------------------------------------------------------------------
 * CPU-side texture rect conversion / fill / copy / clear.
 * ------------------------------------------------------------------------- */

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
                                bool swap_bytes);

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
                           const void *data);

bool mglTextureHasCompressedInternalFormat(Texture *tex);

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
                               bool swap_bytes);

bool mglCopyTextureLevelToPackBuffer(TextureLevel *lvl,
                                     GLenum internalformat,
                                     GLsizei width,
                                     GLsizei height,
                                     GLsizei depth,
                                     GLenum format,
                                     GLenum type,
                                     const MGLTexturePackLayout *pack_layout,
                                     void *pixels,
                                     bool swap_bytes);

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
                                       bool swap_bytes);

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
                             const void *data);

/* ---------------------------------------------------------------------------
 * Proxy texture queries, source resolution and validation.
 * ------------------------------------------------------------------------- */

void mglHandleProxyTexImageQuery(GLMContext ctx,
                                 GLenum target,
                                 GLint level,
                                 GLint internalformat,
                                 GLsizei width,
                                 GLsizei height,
                                 GLsizei depth,
                                 GLint border);

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
                                 Buffer **unpack_buf_out);

bool mglVerifyInternalFormatAndFormatTypeForCall(GLMContext ctx,
                                                 GLint internalformat,
                                                 GLenum format,
                                                 GLenum type);

/* ---------------------------------------------------------------------------
 * Compressed texture image storage and sub-image updates.
 * ------------------------------------------------------------------------- */

bool mglStoreCompressedTextureImage(GLMContext ctx,
                                    GLenum target,
                                    GLint level,
                                    GLenum internalformat,
                                    GLsizei width,
                                    GLsizei height,
                                    GLsizei depth,
                                    GLint border,
                                    GLsizei imageSize,
                                    const void *data);

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
                                 const void *data);

bool mglCopyTextureSubImageValidate(GLMContext ctx,
                                    Texture *tex,
                                    GLint level,
                                    GLint xoffset,
                                    GLint yoffset,
                                    GLint zoffset,
                                    GLsizei width,
                                    GLsizei height);

/* Functions that remain defined in textures.c but are called by
 * mgl_texture_transfer.c.  Declared here so both translation units share
 * a single declaration. */
bool ensureTextureLevelCapacity(GLMContext ctx, Texture *tex, GLuint required_levels);
void mglReleaseGLSampledTextureCopy(GLMContext ctx, Texture *tex, const char *reason);

#endif /* mgl_texture_transfer_h */
