/*
 * mgl_state_log.m
 * MGL
 *
 * Implementation of the GL State Logging Subsystem.
 * See mgl_state_log.h for the API contract.
 */

#import "mgl_state_log.h"

#include <stdio.h>
#include <string.h>

void mglAppendFlagName(char *dst, size_t dstSize, size_t *used,
                       const char *name, bool *first)
{
    if (!dst || !name || !used || dstSize == 0) {
        return;
    }

    /* Use caller-tracked *used instead of strlen(dst) to avoid O(n^2)
     * accumulation across repeated appends.  *used includes the NUL
     * terminator (1 for an empty string). */
    size_t pos = (*used > 0) ? *used - 1 : 0;
    if (pos >= dstSize - 1) {
        return;
    }

    int written = snprintf(dst + pos,
                           dstSize - pos,
                           "%s%s",
                           (*first ? "" : "|"),
                           name);
    if (written > 0) {
        *first = false;
        *used = pos + (size_t)written + 1;
    }
}

void mglFormatDirtyBits(uint32_t bits, char *dst, size_t dstSize)
{
    if (!dst || dstSize == 0) {
        return;
    }

    dst[0] = '\0';
    if (bits == 0) {
        snprintf(dst, dstSize, "none");
        return;
    }

    bool first = true;
    size_t used = 1;  /* track length to avoid O(n^2) strlen */
    if (bits & DIRTY_VAO) mglAppendFlagName(dst, dstSize, &used, "VAO", &first);
    if (bits & DIRTY_STATE) mglAppendFlagName(dst, dstSize, &used, "STATE", &first);
    if (bits & DIRTY_BUFFER) mglAppendFlagName(dst, dstSize, &used, "BUFFER", &first);
    if (bits & DIRTY_TEX) mglAppendFlagName(dst, dstSize, &used, "TEX", &first);
    if (bits & DIRTY_TEX_PARAM) mglAppendFlagName(dst, dstSize, &used, "TEX_PARAM", &first);
    if (bits & DIRTY_TEX_BINDING) mglAppendFlagName(dst, dstSize, &used, "TEX_BINDING", &first);
    if (bits & DIRTY_SAMPLER) mglAppendFlagName(dst, dstSize, &used, "SAMPLER", &first);
    if (bits & DIRTY_SHADER) mglAppendFlagName(dst, dstSize, &used, "SHADER", &first);
    if (bits & DIRTY_PROGRAM) mglAppendFlagName(dst, dstSize, &used, "PROGRAM", &first);
    if (bits & DIRTY_FBO) mglAppendFlagName(dst, dstSize, &used, "FBO", &first);
    if (bits & DIRTY_DRAWABLE) mglAppendFlagName(dst, dstSize, &used, "DRAWABLE", &first);
    if (bits & DIRTY_RENDER_STATE) mglAppendFlagName(dst, dstSize, &used, "RENDER_STATE", &first);
    if (bits & DIRTY_ALPHA_STATE) mglAppendFlagName(dst, dstSize, &used, "ALPHA_STATE", &first);
    if (bits & DIRTY_IMAGE_UNIT_STATE) mglAppendFlagName(dst, dstSize, &used, "IMAGE_UNIT", &first);
    if (bits & DIRTY_BUFFER_BASE_STATE) mglAppendFlagName(dst, dstSize, &used, "BUFFER_BASE", &first);
    if (bits & DIRTY_ALL_BIT) mglAppendFlagName(dst, dstSize, &used, "ALL_BIT", &first);
    if (bits == DIRTY_ALL) mglAppendFlagName(dst, dstSize, &used, "ALL", &first);

    if (first) {
        snprintf(dst, dstSize, "0x%x", bits);
    }
}
