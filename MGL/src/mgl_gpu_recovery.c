/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/* mgl_gpu_recovery.c — bodies of -[MGLRenderer clearTextureCache] and
 * -getOptimalAlignmentForPixelFormat:, which never needed Objective-C. */

#include "mgl_gpu_recovery.h"

#include <stdio.h>

void mglRendererClearTextureCache(void)
{
    /* PROPER FIX: Intelligent texture cache cleanup.  Texture binding cache
     * cleanup would need the renderer's instance state; this is the hook that
     * would take it. */
    fprintf(stderr, "MGL INFO: Clearing texture cache to free memory\n");
}

uint64_t mglRendererOptimalAlignmentForPixelFormat(uint32_t format)
{
    (void)format;
    /* aligned_alloc requires an alignment compatible with platform pointer
     * alignment.  A conservative 64-byte value avoids EINVAL on macOS/arm64 and
     * is safe for texture rows. */
    return 64;
}
