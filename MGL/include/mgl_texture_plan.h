/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_texture_plan.h -- the texture creation plan (pure C, no Metal).
 *
 * A GL texture becomes a Metal texture descriptor in a fixed sequence of
 * decisions: target -> Metal texture type, pixel format substitution, mip
 * count, multisample emulation, 1D backing, storage mode and usage.  The
 * decisions used to live inline in mglTextureCreateFromGLTexture next to the
 * Metal calls and the Texture* reads.  The plan takes the facts as plain
 * values and returns the descriptor plus the flags the upload code needs, so
 * it can be tested without a device (see test_texture_plan.c).
 *
 * Metal enum values appear as the stable ABI numbers of mgl_render_values.h.
 */

#ifndef MGL_TEXTURE_PLAN_H
#define MGL_TEXTURE_PLAN_H

#include <stdint.h>

#include "mgl_render_api_texture.h" /* MGLRenderTextureDescriptorState, TargetPlan */

#ifdef __cplusplus
extern "C" {
#endif

/* Descriptor field values (MTLCPUCacheMode / MTLStorageMode / MTLTextureUsage
 * numbers). */
enum {
    MGL_TEXTURE_CPU_CACHE_DEFAULT = 0u,
    MGL_TEXTURE_CPU_CACHE_WRITE_COMBINED = 1u,
    MGL_TEXTURE_RESOURCE_STORAGE_SHARED = 0u,
    MGL_TEXTURE_STORAGE_PRIVATE = 2u,
    MGL_TEXTURE_USAGE_SHADER_READ = 1u,
    MGL_TEXTURE_USAGE_SHADER_WRITE = 2u,
    MGL_TEXTURE_USAGE_RENDER_TARGET = 4u,
    MGL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW = 16u,
    /* MTLTextureUsageShaderAtomic (macOS 14+ / Metal 3.1). */
    MGL_TEXTURE_USAGE_SHADER_ATOMIC = 0x20u,
};

typedef struct MGLTexturePlanInput {
    /* The target's plan (mglRenderTextureTargetPlan): the Metal texture type
     * before any promotion, face count and the 1D backing flags. */
    MGLRenderTextureTargetPlan target;
    uint64_t width, height, depth; /* GL level 0 size; depth = layers/depth */
    uint32_t access;               /* GL_READ_ONLY / GL_WRITE_ONLY / GL_READ_WRITE */
    int is_render_target;
    /* Texture completeness (it needs the target plan's type and faces, so the
     * caller runs it between the two steps). */
    uint32_t effective_mipmap_levels;
    int effective_mipmapped;
    int tex_mipmapped;      /* the GL texture's mipmapped flag after that check */
    /* Metal pixel format for the GL internal format, after the swizzle storage
     * substitution and before the AGX one. */
    uint32_t pixel_format;
    int format_is_depth_or_stencil; /* classification of pixel_format */
    /* Lowest mip level of any face that holds CPU data worth uploading, or
     * MGL_TEXTURE_PLAN_NO_LEVEL.  Whether that data counts depends on how many
     * levels the plan decides to walk. */
    uint32_t first_uploadable_level;
    int dirty_texture_data;         /* DIRTY_TEXTURE_DATA is set */
    int conservative_cpu_cache;     /* device capability */
} MGLTexturePlanInput;

#define MGL_TEXTURE_PLAN_NO_LEVEL 0xffffffffu

enum {
    MGL_TEXTURE_PLAN_OK = 0,
    MGL_TEXTURE_PLAN_BAD_ACCESS = 1, /* not a GL image access mode */
    MGL_TEXTURE_PLAN_BAD_ARGS = -1
};

typedef struct MGLTexturePlan {
    int status; /* MGL_TEXTURE_PLAN_* */
    MGLRenderTextureDescriptorState desc;
    /* What the upload code works from (the type after every promotion). */
    uint32_t upload_texture_type;
    uint32_t pixel_format; /* after the AGX substitution */
    uint32_t effective_mipmap_levels;
    uint32_t upload_level_count;
    int storage_mipmapped;
    int is_array;
    int texture_1d_backed_by_2d;
    int texture_1d_array_backed_by_2d_array;
    int ms_emulated_as_array;
    int agx_converted;
    int has_uploadable_cpu_data; /* uploadable data in the levels walked */
    int needs_cpu_upload;        /* ... and the GL texture says it is dirty */
    /* A multisample texture has one level: the GL texture's mipmapped flag
     * must be cleared by the caller. */
    int clear_tex_mipmapped;
} MGLTexturePlan;

/* Plan a texture's Metal descriptor.  Returns 0 and fills `out` (including
 * out->status); -1 for a NULL argument.  When status is BAD_ACCESS the
 * descriptor is incomplete, but clear_tex_mipmapped is still meaningful: the
 * caller applies it before giving up, as the inline code did. */
int mglTexturePlanBuild(const MGLTexturePlanInput *in, MGLTexturePlan *out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_PLAN_H */
