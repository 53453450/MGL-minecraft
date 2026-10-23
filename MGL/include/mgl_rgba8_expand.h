/* SPDX-License-Identifier: LGPL-3.0-only */
#ifndef MGL_RGBA8_EXPAND_H
#define MGL_RGBA8_EXPAND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Expands one source pixel of `internal_format` into RGBA8 (alpha defaulting
 * per format).  Callers own both buffers. */
void mglExpandInternalFormatToRGBA8(uint32_t internal_format,
                                    const uint8_t *src_pixel,
                                    uint32_t src_pixel_bytes,
                                    uint8_t out[4]);

#ifdef __cplusplus
}
#endif

#endif
