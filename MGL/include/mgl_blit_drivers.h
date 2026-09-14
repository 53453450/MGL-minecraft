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

/* Raw CPU-to-CPU copy between matching-format textures that both have CPU
 * data.  Returns true when the copy succeeded (the caller then returns). */
bool mglBlitCopyImageSubDataCpuToCpu(
    void *renderer, GLMContext glm_ctx, Texture *src_tex, void *src_texture,
    uint32_t src_type, GLint src_level, GLint src_x, GLint src_y, GLint src_z,
    Texture *dst_tex, void *dst_texture, uint32_t dst_type, GLint dst_level,
    GLint dst_x, GLint dst_y, GLint dst_z, GLsizei width, GLsizei height,
    GLsizei depth);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BLIT_DRIVERS_H */
