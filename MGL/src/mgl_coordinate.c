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

MGLYFlipDecision mglDecideYFlipForSampledRT(Texture *tex)
{
    if (!tex || !tex->is_render_target) {
        return MGL_YFLIP_USE_ORIGINAL;
    }
    return MGL_YFLIP_USE_SAMPLED_COPY;
}
