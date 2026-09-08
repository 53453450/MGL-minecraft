/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * Shared static helpers for batch encode ObjC ports (not a cluster .m).
 */
#ifndef MGL_BATCH_ENCODE_SHARED_H
#define MGL_BATCH_ENCODE_SHARED_H

#include "mgl_batch_issue.h"
#include "mgl_render.h"

#include <stdint.h>

static inline int mgl_batch_encode_map_scratch(void *buf, uint64_t off,
                                              uint64_t need, void **out)
{
    void *contents = NULL;
    uint64_t length = 0;
    if (mglRenderGetBufferContents(buf, &contents, &length) != 0 ||
        !mgl_batch_issue_scratch_range_ok(off, need, length) || !contents) {
        return 0;
    }
    if (out) {
        *out = (uint8_t *)contents + off;
    }
    return 1;
}

#endif /* MGL_BATCH_ENCODE_SHARED_H */
