/* SPDX-License-Identifier: LGPL-3.0-only */

/*
 * mgl_texture_swizzle.h -- swizzle channels without the GL vocabulary.
 *
 * The upload bakers ask "which channel does this swizzle select?" and need to
 * treat GL_ONE as the maximum value of whatever format they are writing.
 * Naming the channels here keeps that question out of the Metal backend and
 * keeps this header free of `extern "C"`, so C++ TUs can define their own
 * functions that take these values without linkage surprises.
 */

#ifndef MGL_TEXTURE_SWIZZLE_H
#define MGL_TEXTURE_SWIZZLE_H

#include <stdint.h>

#include "glm_context.h"

typedef enum MGLTextureSwizzleChannel {
    MGL_SWIZZLE_CHANNEL_INVALID = 0,
    MGL_SWIZZLE_CHANNEL_RED,
    MGL_SWIZZLE_CHANNEL_GREEN,
    MGL_SWIZZLE_CHANNEL_BLUE,
    MGL_SWIZZLE_CHANNEL_ALPHA,
    MGL_SWIZZLE_CHANNEL_ZERO,
    MGL_SWIZZLE_CHANNEL_ONE
} MGLTextureSwizzleChannel;

static inline MGLTextureSwizzleChannel
mglTextureSwizzleChannel(uint32_t gl_swizzle)
{
    switch (gl_swizzle) {
        case GL_RED: return MGL_SWIZZLE_CHANNEL_RED;
        case GL_GREEN: return MGL_SWIZZLE_CHANNEL_GREEN;
        case GL_BLUE: return MGL_SWIZZLE_CHANNEL_BLUE;
        case GL_ALPHA: return MGL_SWIZZLE_CHANNEL_ALPHA;
        case GL_ZERO: return MGL_SWIZZLE_CHANNEL_ZERO;
        case GL_ONE: return MGL_SWIZZLE_CHANNEL_ONE;
        default: return MGL_SWIZZLE_CHANNEL_INVALID;
    }
}

#endif /* MGL_TEXTURE_SWIZZLE_H */
