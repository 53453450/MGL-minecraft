/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/* mgl_region_value.h
 *
 * Canonical, C++-safe definitions of the region / origin / size value structs
 * (MGLSizeValue / MGLOriginValue / MGLRegionValue) and their pure-C
 * constructors.
 *
 * These types were previously duplicated inside the ObjC private headers
 * (MGLRenderer_Private.h, MGLRenderer+Texture_Private.h) and the constructor
 * cluster was duplicated as file-scope static helpers in MGLRenderer+Texture.m
 * (mglTextureOrigin / mglTextureSize / mglTextureRegion1D|2D|3D) and
 * MGLRenderer+Blit.m (mglBlitSize / mglBlitOrigin / mglBlitRegion2D|3D).  They
 * are pure C (no Metal, no ObjC), so they now live in a C++ TU.
 *
 * Making the types available from a C++-safe header is also a prerequisite for
 * the O3 / O4 plan layers (mgl_render_pass_plan.*, mgl_blit_plan.*, ...), which
 * must be able to build MGLRegionValue without pulling in an ObjC header.
 */

#ifndef MGL_REGION_VALUE_H
#define MGL_REGION_VALUE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MGL_VALUE_GEOMETRY_TYPES
#define MGL_VALUE_GEOMETRY_TYPES 1
typedef struct MGLSizeValue_t { uint64_t width, height, depth; } MGLSizeValue;
typedef struct MGLOriginValue_t { int64_t x, y, z; } MGLOriginValue;
typedef struct MGLRegionValue_t { MGLOriginValue origin; MGLSizeValue size; } MGLRegionValue;
#endif

/* Canonical constructors (defined out-of-line in mgl_region_value.cpp). */
MGLOriginValue mglRegionOrigin(uint64_t x, uint64_t y, uint64_t z);
MGLSizeValue   mglRegionSize(uint64_t width, uint64_t height, uint64_t depth);
MGLRegionValue mglRegion1D(uint64_t x, uint64_t width);
MGLRegionValue mglRegion2D(uint64_t x, uint64_t y, uint64_t width, uint64_t height);
MGLRegionValue mglRegion3D(uint64_t x, uint64_t y, uint64_t z,
                            uint64_t width, uint64_t height, uint64_t depth);

/* Back-compat aliases for the duplicated +Texture / +Blit clusters, now
 * consolidated into the canonical constructors above (O4 dedup sink). */
static inline MGLOriginValue mglTextureOrigin(uint64_t x, uint64_t y, uint64_t z)
{ return mglRegionOrigin(x, y, z); }
static inline MGLSizeValue mglTextureSize(uint64_t width, uint64_t height, uint64_t depth)
{ return mglRegionSize(width, height, depth); }
static inline MGLRegionValue mglTextureRegion1D(uint64_t x, uint64_t width)
{ return mglRegion1D(x, width); }
static inline MGLRegionValue mglTextureRegion2D(uint64_t x, uint64_t y, uint64_t width, uint64_t height)
{ return mglRegion2D(x, y, width, height); }
static inline MGLRegionValue mglTextureRegion3D(uint64_t x, uint64_t y, uint64_t z,
                                                uint64_t width, uint64_t height, uint64_t depth)
{ return mglRegion3D(x, y, z, width, height, depth); }

static inline MGLSizeValue mglBlitSize(uint64_t width, uint64_t height, uint64_t depth)
{ return mglRegionSize(width, height, depth); }
static inline MGLOriginValue mglBlitOrigin(uint64_t x, uint64_t y, uint64_t z)
{ return mglRegionOrigin(x, y, z); }
static inline MGLRegionValue mglBlitRegion2D(uint64_t x, uint64_t y, uint64_t width, uint64_t height)
{ return mglRegion2D(x, y, width, height); }
static inline MGLRegionValue mglBlitRegion3D(uint64_t x, uint64_t y, uint64_t z,
                                             uint64_t width, uint64_t height, uint64_t depth)
{ return mglRegion3D(x, y, z, width, height, depth); }

#ifdef __cplusplus
}
#endif

#endif /* MGL_REGION_VALUE_H */
