/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/* mgl_region_value.cpp
 *
 * Pure-C canonical constructors for the region / origin / size value structs.
 * Sinked out of the duplicated static helpers in MGLRenderer+Texture.m and
 * MGLRenderer+Blit.m (O4 dedup sink).  No Metal, no ObjC — this TU is compiled
 * as plain C++ and exposes only the C ABI declared in mgl_region_value.h.
 */

#include "mgl_region_value.h"

MGLOriginValue mglRegionOrigin(uint64_t x, uint64_t y, uint64_t z)
{
    return (MGLOriginValue){(int64_t)x, (int64_t)y, (int64_t)z};
}

MGLSizeValue mglRegionSize(uint64_t width, uint64_t height, uint64_t depth)
{
    return (MGLSizeValue){width, height, depth};
}

MGLRegionValue mglRegion1D(uint64_t x, uint64_t width)
{
    return (MGLRegionValue){mglRegionOrigin(x, 0, 0), mglRegionSize(width, 1, 1)};
}

MGLRegionValue mglRegion2D(uint64_t x, uint64_t y, uint64_t width, uint64_t height)
{
    return (MGLRegionValue){mglRegionOrigin(x, y, 0), mglRegionSize(width, height, 1)};
}

MGLRegionValue mglRegion3D(uint64_t x, uint64_t y, uint64_t z,
                            uint64_t width, uint64_t height, uint64_t depth)
{
    return (MGLRegionValue){mglRegionOrigin(x, y, z), mglRegionSize(width, height, depth)};
}
