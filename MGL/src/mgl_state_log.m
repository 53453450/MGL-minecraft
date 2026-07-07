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

void mglAppendFlagName(char *dst, size_t dstSize, const char *name, bool *first)
{
    if (!dst || !name || dstSize == 0) {
        return;
    }

    size_t used = strlen(dst);
    if (used >= dstSize - 1) {
        return;
    }

    int written = snprintf(dst + used,
                           dstSize - used,
                           "%s%s",
                           (*first ? "" : "|"),
                           name);
    if (written > 0) {
        *first = false;
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
    if (bits & DIRTY_VAO) mglAppendFlagName(dst, dstSize, "VAO", &first);
    if (bits & DIRTY_STATE) mglAppendFlagName(dst, dstSize, "STATE", &first);
    if (bits & DIRTY_BUFFER) mglAppendFlagName(dst, dstSize, "BUFFER", &first);
    if (bits & DIRTY_TEX) mglAppendFlagName(dst, dstSize, "TEX", &first);
    if (bits & DIRTY_TEX_PARAM) mglAppendFlagName(dst, dstSize, "TEX_PARAM", &first);
    if (bits & DIRTY_TEX_BINDING) mglAppendFlagName(dst, dstSize, "TEX_BINDING", &first);
    if (bits & DIRTY_SAMPLER) mglAppendFlagName(dst, dstSize, "SAMPLER", &first);
    if (bits & DIRTY_SHADER) mglAppendFlagName(dst, dstSize, "SHADER", &first);
    if (bits & DIRTY_PROGRAM) mglAppendFlagName(dst, dstSize, "PROGRAM", &first);
    if (bits & DIRTY_FBO) mglAppendFlagName(dst, dstSize, "FBO", &first);
    if (bits & DIRTY_DRAWABLE) mglAppendFlagName(dst, dstSize, "DRAWABLE", &first);
    if (bits & DIRTY_RENDER_STATE) mglAppendFlagName(dst, dstSize, "RENDER_STATE", &first);
    if (bits & DIRTY_ALPHA_STATE) mglAppendFlagName(dst, dstSize, "ALPHA_STATE", &first);
    if (bits & DIRTY_IMAGE_UNIT_STATE) mglAppendFlagName(dst, dstSize, "IMAGE_UNIT", &first);
    if (bits & DIRTY_BUFFER_BASE_STATE) mglAppendFlagName(dst, dstSize, "BUFFER_BASE", &first);
    if (bits & DIRTY_ALL_BIT) mglAppendFlagName(dst, dstSize, "ALL_BIT", &first);
    if (bits == DIRTY_ALL) mglAppendFlagName(dst, dstSize, "ALL", &first);

    if (first) {
        snprintf(dst, dstSize, "0x%x", bits);
    }
}
