/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_blit_drivers.h — two self-contained blit paths moved out of
 * MGLRenderer+Blit.m (P0-1, log 135).
 *
 *   -blitFramebufferResolveMsaaSource:…  -> mglBlitResolveMsaaSource
 *   -copyImageSubDataCpuToCpu:…          -> mglBlitCopyImageSubDataCpuToCpu
 *
 * Both are leaves of the blit family (they call no other Objective-C method of
 * the file), so their only callers switch to these C entries without adding a
 * port.
 */

#ifndef MGL_BLIT_DRIVERS_H
#define MGL_BLIT_DRIVERS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "glm_context.h"        /* GLMContext, GLint/GLsizei */
#include "mgl_region_value.h"    /* MGLRegionValue */
#include "mgl_sync.h"           /* MGLMetalAttachmentSubresource */
#include "mgl_types_texture.h"  /* Texture */

#ifdef __cplusplus
extern "C" {
#endif

/* Resolve a multisample (native or emulated) source into a fresh single-sample
 * texture and rewrite *read_texid_ptr / *read_subresource_ptr.  Returns false
 * when no resolve could be encoded (the method's NO). */
bool mglBlitResolveMsaaSource(void *renderer, void **read_texid_ptr,
                              void *draw_texid,
                              MGLMetalAttachmentSubresource *read_subresource_ptr,
                              size_t src_tex_w, size_t src_tex_h,
                              Texture *read_texture_object,
                              int *out_did_msaa_resolve);

/* Read a texture region back through a blit encoder into `bytes`.
 * Was -readTextureRegionViaBlit:region:slice:level:bytes:bytesPerRow:
 * bytesPerImage:reason:.  Returns true when the readback landed. */
bool mglBlitReadTextureRegion(void *renderer, void *texture,
                              MGLRegionValue region, size_t slice, size_t level,
                              void *bytes, size_t bytes_per_row,
                              size_t bytes_per_image, const char *reason);

/* Texture-to-texture copyTexSubImage blit (source = the current read
 * framebuffer attachment).  Was -mtlCopyTexSubImageViaTextureBlit:tex:… */
bool mglBlitCopyTexSubImageViaTextureBlit(
    void *renderer, GLMContext glm_ctx, Texture *tex, void *dest_texture,
    size_t slice, size_t level, int64_t xoffset, int64_t yoffset, int64_t x,
    int64_t y, size_t width, size_t height);

/* Raw CPU-to-CPU copy between matching-format textures that both have CPU
 * data.  Returns true when the copy succeeded (the caller then returns). */
bool mglBlitCopyImageSubDataCpuToCpu(
    void *renderer, GLMContext glm_ctx, Texture *src_tex, void *src_texture,
    uint32_t src_type, GLint src_level, GLint src_x, GLint src_y, GLint src_z,
    Texture *dst_tex, void *dst_texture, uint32_t dst_type, GLint dst_level,
    GLint dst_x, GLint dst_y, GLint dst_z, GLsizei width, GLsizei height,
    GLsizei depth);

/* Metal-to-Metal format-conversion copy for copyImageSubData (source and
 * destination have different Metal pixel formats: read the source through
 * getBytes, write the destination through replaceRegion).  Was
 * -copyImageSubDataFormatConversion:….  Returns true when the path was taken
 * (the caller then returns) — the method's YES/NO. */
bool mglBlitCopyImageSubDataFormatConversion(
    void *renderer, GLMContext glm_ctx, Texture *src_tex, void *src_texture,
    uint32_t src_type, GLint src_level, GLint src_x, GLint src_y, GLint src_z,
    Texture *dst_tex, void *dst_texture, uint32_t dst_type, GLint dst_level,
    GLint dst_x, GLint dst_y, GLint dst_z, GLsizei width, GLsizei height,
    GLsizei depth);

/* Buffer-mediated 3D-destination fallback for copyImageSubData (works around
 * the AGX "slice OOB" assertions).  Was -copyImageSubData3DFallback:….
 * Returns true when the fallback handled the copy, false to fall through to
 * the blit path — the method's YES/NO. */
bool mglBlitCopyImageSubData3DFallback(
    void *renderer, GLMContext glm_ctx, Texture *src_tex, void *src_texture,
    uint32_t src_type, GLint src_level, GLint src_x, GLint src_y, GLint src_z,
    Texture *dst_tex, void *dst_texture, uint32_t dst_type, GLint dst_level,
    GLint dst_x, GLint dst_y, GLint dst_z, GLsizei width, GLsizei height,
    GLsizei depth);

/* Post-blit CPU readback for copyImageSubData: read the blitted region back
 * from the destination Metal texture so the CPU data stays authoritative.
 * Was -copyImageSubDataPostBlitReadback:dstTexture:….  Returns true when the
 * readback landed (the method's YES). */
bool mglBlitCopyImageSubDataPostBlitReadback(
    void *renderer, Texture *dst_tex, void *dst_texture, uint32_t dst_type,
    GLint dst_level, GLint dst_x, GLint dst_y, GLint dst_z, GLsizei width,
    GLsizei height, GLsizei depth);

/* The whole glCopyImageSubData dispatch: 3D-destination workaround, CPU-to-CPU,
 * format conversion, then the blit encoder path plus the post-blit readback.
 * Was -(void)mtlCopyImageSubData:srcTexture:… (the caller in
 * MGLRenderer+Texture.m used to send it as a message). */
void mglBlitCopyImageSubData(void *renderer, GLMContext glm_ctx,
                             Texture *src_tex, GLint src_level, GLint src_x,
                             GLint src_y, GLint src_z, Texture *dst_tex,
                             GLint dst_level, GLint dst_x, GLint dst_y,
                             GLint dst_z, GLsizei width, GLsizei height,
                             GLsizei depth);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BLIT_DRIVERS_H */
