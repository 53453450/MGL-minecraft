/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_texture_plan.c -- texture creation plan (pure C, no Metal, no GL object).
 * See mgl_texture_plan.h.  The small decisions the plan is made of live here
 * too; they are also entries of the renderer facade (mgl_render_api_texture.h)
 * for the view, fallback and blit paths that make a descriptor of their own.
 */

#include "mgl_texture_plan.h"

#include <string.h>

#include "glcorearb.h"
#include "mgl_render_values.h" /* MGLTextureType*, MGLTextureUsage* */


int mglRenderTextureTargetPlan(
    uint32_t gl_target,
    uint32_t sample_count,
    MGLRenderTextureTargetPlan* plan_out) {
    if (!plan_out) return -1;
    memset(plan_out, 0, sizeof(*plan_out));
    plan_out->num_faces = 1u;

    switch (gl_target) {
        case GL_TEXTURE_1D:
            plan_out->texture_type = MGLTextureType2D;
            plan_out->texture_1d_backed_by_2d = 1u;
            return 0;
        case GL_RENDERBUFFER:
            plan_out->texture_type = sample_count > 1u
                                         ? MGLTextureType2DMultisample
                                         : MGLTextureType2D;
            return 0;
        case GL_TEXTURE_1D_ARRAY:
            /* AIR lowers sampler1DArray to texture2d_array, and Metal cannot
             * view a Texture1DArray as Texture2DArray. */
            plan_out->texture_type = MGLTextureType2DArray;
            plan_out->is_array = 1u;
            plan_out->texture_1d_array_backed_by_2d_array = 1u;
            return 0;
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            plan_out->texture_type = MGLTextureType2D;
            return 0;
        case GL_TEXTURE_2D_ARRAY:
            plan_out->texture_type = MGLTextureType2DArray;
            plan_out->is_array = 1u;
            return 0;
        case GL_TEXTURE_2D_MULTISAMPLE:
            plan_out->texture_type = MGLTextureType2DMultisample;
            return 0;
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            plan_out->texture_type = MGLTextureTypeCube;
            plan_out->num_faces = 6u;
            return 0;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            plan_out->texture_type = MGLTextureTypeCubeArray;
            /* GL stores cube-array levels in faces[0] with depth=cubes*6.
             * Iterating 6 faces during CPU upload marks the create incomplete
             * (faces 1-5 empty) and leaves DIRTY_TEXTURE_DATA set -- later
             * binds can fight imageStore results. */
            plan_out->num_faces = 1u;
            plan_out->is_array = 1u;
            return 0;
        case GL_TEXTURE_3D:
            plan_out->texture_type = MGLTextureType3D;
            return 0;
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            plan_out->texture_type = MGLTextureType2DMultisampleArray;
            plan_out->is_array = 1u;
            return 0;
        default:
            memset(plan_out, 0, sizeof(*plan_out));
            return -1;
    }
}

uint64_t mglRenderExpectedArrayLayers(uint32_t gl_target, int32_t depth) {
    uint64_t layers = depth > 0 ? (uint64_t)depth : 1u;
    if (gl_target == GL_TEXTURE_CUBE_MAP_ARRAY && layers >= 6u &&
        (layers % 6u) == 0u) {
        layers /= 6u;
    }
    return layers;
}

uint32_t mglRenderAGXCompatiblePixelFormat(uint32_t pixel_format, int *converted) {
    int conv = 0;
    switch (pixel_format) {
    case 40u: /* B5G6R5Unorm */
    case 41u: /* A1BGR5Unorm */
    case 43u: /* BGR5A1Unorm */
    case 160u: /* PVRTC_RGB_2BPP */
    case 162u: /* PVRTC_RGB_4BPP */
    case 164u: /* PVRTC_RGBA_2BPP */
    case 166u: /* PVRTC_RGBA_4BPP */
        conv = 1;
        pixel_format = 70u; /* RGBA8Unorm */
        break;
    default:
        break;
    }
    if (converted) {
        *converted = conv;
    }
    return pixel_format;
}

int mglRenderPromote1DArrayDepthStencil(uint32_t tex_type, uint32_t pixel_format) {
    if (tex_type != MGLTextureType1DArray) {
        return 0;
    }
    switch (pixel_format) {
    case 250u: /* Depth16Unorm */
    case 252u: /* Depth32Float */
    case 253u: /* Stencil8 */
    case 255u: /* Depth24Unorm_Stencil8 */
    case 260u: /* Depth32Float_Stencil8 */
    case 261u: /* X32_Stencil8 */
    case 262u: /* X24_Stencil8 */
        return 1;
    default:
        return 0;
    }
}

int mglRenderPromoteMipmapped1D(uint32_t tex_type) {
    return tex_type == MGLTextureType1D ? 1 : 0;
}

int mglRenderPromoteMipmapped1DArray(uint32_t tex_type) {
    return tex_type == MGLTextureType1DArray ? 1 : 0;
}

uint32_t mglRenderUploadLevelCount(int mipmapped, int tex_mipmapped,
                                   uint32_t effective) {
    return mipmapped && tex_mipmapped ? effective : 1u;
}

int mglRenderEmulateMSAsArray(uint32_t tex_type, uint32_t samples, uint64_t depth,
                              uint32_t *out_type, uint32_t *sample_count,
                              uint64_t *array_len, uint64_t *depth_out) {
    if (tex_type != MGLTextureType2DMultisample &&
        tex_type != MGLTextureType2DMultisampleArray) {
        return 0;
    }
    /* Fixed stride matches AIR texelFetch / textureSize for sampler2DMSArray
     * (flat = layer * 8 + sample).  Non-array MS is one logical layer so the
     * store is always 8 planes — needed for TextureView MS→MS_ARRAY where the
     * view exposes layer count via array_length/8. */
    const uint64_t kMsPlaneStride = 8u;
    uint32_t sc = 1u;
    uint64_t d = 1u;
    uint64_t arr = 1u;
    uint32_t ot = MGLTextureType2DArray;
    (void)samples;
    if (tex_type == MGLTextureType2DMultisample) {
        arr = kMsPlaneStride;
    } else {
        uint64_t layers = depth < 1u ? 1u : depth;
        arr = layers * kMsPlaneStride;
    }
    if (out_type) {
        *out_type = ot;
    }
    if (sample_count) {
        *sample_count = sc;
    }
    if (array_len) {
        *array_len = arr;
    }
    if (depth_out) {
        *depth_out = d;
    }
    return 1;
}

int mglRenderPreferSharedStorage(int needs_cpu, int is_depth_stencil) {
    return needs_cpu || is_depth_stencil ? 1 : 0;
}

void mglRenderApply1DBackingToDesc(int backed_1d, int backed_1d_array,
                                   uint64_t height, uint32_t *type,
                                   uint64_t *array_len, uint32_t *height_out) {
    if (backed_1d) {
        if (type) {
            *type = MGLTextureType2D;
        }
        if (height_out) {
            *height_out = 1u;
        }
    }
    if (backed_1d_array) {
        if (type) {
            *type = MGLTextureType2DArray;
        }
        if (array_len) {
            *array_len = height < 1u ? 1u : height;
        }
        if (height_out) {
            *height_out = 1u;
        }
    }
}

int mglRenderTextureUsageForAccess(uint32_t gl_access, uint32_t *usage_out) {
    switch (gl_access) {
    case GL_READ_ONLY:
    case GL_WRITE_ONLY:
    case GL_READ_WRITE:
        if (usage_out) {
            *usage_out = MGLTextureUsageShaderRead | MGLTextureUsageShaderWrite;
        }
        return 1;
    default:
        return 0;
    }
}

int mglRenderPixelFormatNeedsShaderAtomic(uint32_t pixel_format) {
    return pixel_format == 53u /* R32Uint */ || pixel_format == 54u /* R32Sint */
               ? 1
               : 0;
}

int mglRenderTextureArrayDepthForType(uint32_t tex_type, int is_array,
                                      int ms_emulated, uint64_t width,
                                      uint64_t height, uint64_t depth,
                                      uint64_t *array_out, uint64_t *depth_out) {
    (void)width;
    uint64_t array_len = 1u;
    uint64_t d = 1u;
    if (tex_type == MGLTextureTypeCube) {
        d = 1u;
    } else if (tex_type == MGLTextureTypeCubeArray) {
        array_len = mglRenderExpectedArrayLayers(GL_TEXTURE_CUBE_MAP_ARRAY,
                                                 (int32_t)depth);
        d = 1u;
    } else if (tex_type == MGLTextureType1DArray) {
        array_len = height < 1u ? 1u : height;
        d = 1u;
    } else if (is_array && !ms_emulated) {
        array_len = depth < 1u ? 1u : depth;
        d = 1u;
    } else if (!ms_emulated) {
        array_len = 1u;
        d = depth < 1u ? 1u : depth;
    } else {
        return 0;
    }
    if (array_out) {
        *array_out = array_len;
    }
    if (depth_out) {
        *depth_out = d;
    }
    return 1;
}

uint32_t mglRenderTextureDescHeight(uint32_t tex_type, uint32_t height) {
    return tex_type == MGLTextureType1D || tex_type == MGLTextureType1DArray
               ? 1u
               : height;
}

/* ---- the plan ------------------------------------------------------------ */

static uint64_t plan_max_u64(uint64_t a, uint64_t b) { return a > b ? a : b; }

int mglTexturePlanBuild(const MGLTexturePlanInput *in, MGLTexturePlan *out) {
    if (!in || !out) {
        return -1;
    }
    memset(out, 0, sizeof(*out));

    uint32_t tex_type = in->target.texture_type;
    int is_array = in->target.is_array != 0u;
    int backed_1d = in->target.texture_1d_backed_by_2d != 0u;
    int backed_1d_array = in->target.texture_1d_array_backed_by_2d_array != 0u;
    uint32_t effective_levels = in->effective_mipmap_levels;
    int storage_mipmapped = in->effective_mipmapped ? 1 : 0;
    int tex_mipmapped = in->tex_mipmapped ? 1 : 0;
    const uint64_t width = in->width;
    const uint64_t height = in->height;
    const uint64_t depth = in->depth;

    int agx_converted = 0;
    const uint32_t pixel_format =
        mglRenderAGXCompatiblePixelFormat(in->pixel_format, &agx_converted);

    /* Metal has no depth/stencil 1D-array textures. */
    if (mglRenderPromote1DArrayDepthStencil(tex_type, pixel_format)) {
        tex_type = MGLTextureType2DArray;
        backed_1d_array = 1;
    }

    /* A multisample texture has exactly one level. */
    if (tex_type == MGLTextureType2DMultisample ||
        tex_type == MGLTextureType2DMultisampleArray) {
        storage_mipmapped = 0;
        effective_levels = 1u;
        tex_mipmapped = 0;
        out->clear_tex_mipmapped = 1;
    }

    const int mipmapped = storage_mipmapped;
    /* GL may allocate num_levels>1 for a single-base-level image; only the
     * populated mips are walked unless the texture is mipmapped. */
    const uint32_t upload_level_count = mglRenderUploadLevelCount(
        mipmapped, tex_mipmapped, effective_levels);

    MGLRenderTextureDescriptorState *desc = &out->desc;
    desc->texture_type = tex_type;
    desc->pixel_format = pixel_format;
    desc->width = width;
    desc->height = mglRenderTextureDescHeight(tex_type, (uint32_t)height);

    int ms_emulated = 0;
    {
        uint32_t out_type = tex_type;
        uint32_t sample_count = 1u;
        uint64_t array_len = 1u;
        uint64_t desc_depth = 1u;
        /* The emulation always stores one sample per plane, so the GL sample
         * count does not enter. */
        if (mglRenderEmulateMSAsArray(tex_type, 0u, depth, &out_type,
                                      &sample_count, &array_len,
                                      &desc_depth)) {
            ms_emulated = 1;
            tex_type = out_type;
            desc->texture_type = tex_type;
            desc->sample_count = sample_count;
            desc->array_length = array_len;
            desc->depth = desc_depth;
        }
    }

    desc->cpu_cache_mode = in->conservative_cpu_cache
                               ? MGL_TEXTURE_CPU_CACHE_WRITE_COMBINED
                               : MGL_TEXTURE_CPU_CACHE_DEFAULT;

    /* Textures that take a CPU upload (blit / replaceRegion) and depth/stencil
     * ones live in shared storage; private is for pure GPU render targets. */
    const int has_uploadable_cpu_data =
        in->first_uploadable_level < upload_level_count;
    const int needs_cpu_upload =
        in->dirty_texture_data && has_uploadable_cpu_data;
    desc->storage_mode =
        mglRenderPreferSharedStorage(needs_cpu_upload ? 1 : 0,
                                     in->format_is_depth_or_stencil ? 1 : 0)
            ? MGL_TEXTURE_RESOURCE_STORAGE_SHARED
            : MGL_TEXTURE_STORAGE_PRIVATE;
    desc->sample_count = plan_max_u64(desc->sample_count, 1u);
    desc->mipmap_level_count = plan_max_u64(desc->mipmap_level_count, 1u);
    desc->array_length = plan_max_u64(desc->array_length, 1u);
    desc->depth = plan_max_u64(desc->depth, 1u);

    {
        uint64_t array_len = desc->array_length;
        uint64_t desc_depth = desc->depth;
        if (mglRenderTextureArrayDepthForType(tex_type, is_array, ms_emulated,
                                              width, height, depth, &array_len,
                                              &desc_depth)) {
            desc->array_length = array_len;
            desc->depth = desc_depth;
        }
    }

    if (mipmapped) {
        /* Metal allows no mip chain on a 1D or 1D-array texture: back them
         * with 2D / 2D-array and one-pixel-tall slices. */
        if (mglRenderPromoteMipmapped1D(tex_type)) {
            tex_type = MGLTextureType2D;
            backed_1d = 1;
        }
        if (mglRenderPromoteMipmapped1DArray(tex_type)) {
            tex_type = MGLTextureType2DArray;
            backed_1d_array = 1;
        }
        desc->mipmap_level_count = plan_max_u64(1u, effective_levels);
    }

    if (backed_1d || backed_1d_array) {
        uint32_t backed_type = desc->texture_type;
        uint64_t backed_array = desc->array_length;
        uint32_t backed_height = (uint32_t)desc->height;
        mglRenderApply1DBackingToDesc(backed_1d, backed_1d_array, height,
                                      &backed_type, &backed_array,
                                      &backed_height);
        desc->texture_type = backed_type;
        desc->array_length = backed_array;
        desc->height = backed_height;
    }

    out->upload_texture_type = tex_type;
    out->pixel_format = pixel_format;
    out->effective_mipmap_levels = effective_levels;
    out->upload_level_count = upload_level_count;
    out->storage_mipmapped = storage_mipmapped;
    out->is_array = is_array;
    out->texture_1d_backed_by_2d = backed_1d;
    out->texture_1d_array_backed_by_2d_array = backed_1d_array;
    out->ms_emulated_as_array = ms_emulated;
    out->agx_converted = agx_converted;
    out->has_uploadable_cpu_data = has_uploadable_cpu_data;
    out->needs_cpu_upload = needs_cpu_upload;

    /* Usage.  A GL image access mode only governs the image binding; the
     * texture may still be sampled, and AIR declares storage images as
     * read_write, so every mode needs ShaderRead and ShaderWrite. */
    uint32_t access_usage = 0u;
    if (!mglRenderTextureUsageForAccess(in->access, &access_usage)) {
        out->status = MGL_TEXTURE_PLAN_BAD_ACCESS;
        return 0;
    }
    desc->usage = access_usage;
    if (mglRenderPixelFormatNeedsShaderAtomic(pixel_format)) {
        desc->usage |= MGL_TEXTURE_USAGE_SHADER_ATOMIC;
    }
    if (in->is_render_target) {
        desc->usage |=
            MGL_TEXTURE_USAGE_RENDER_TARGET | MGL_TEXTURE_USAGE_SHADER_READ;
    }
    /* Same-memory format reinterpretation (RGBA8 <-> BGRA8) for the blit and
     * present paths whose attachment and drawable formats differ. */
    desc->usage |= MGL_TEXTURE_USAGE_PIXEL_FORMAT_VIEW;

    out->status = MGL_TEXTURE_PLAN_OK;
    return 0;
}
