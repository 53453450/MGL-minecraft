/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#include "mgl_coordinate.h"
#include "mgl_raster_orientation_plan.h"  /* mglFBOGLOrderEnabled */

MGLYFlipDecision mglDecideYFlipForSampledRT(Texture *tex)
{
    /* GL-order render targets store the rows GL expects; sampling needs no
     * flipped copy. */
    if (!tex || !tex->is_render_target || mglFBOGLOrderEnabled()) {
        return MGL_YFLIP_USE_ORIGINAL;
    }
    return MGL_YFLIP_USE_SAMPLED_COPY;
}
