/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * mgl_types_texture.h
 * MGL
 *
 * Texture domain type definitions split from glm_context.h.
 */

#ifndef mgl_types_texture_h
#define mgl_types_texture_h

#include "mgl_types_buffer.h"

enum {
    _TEXTURE_BUFFER_TARGET = 0, // duplicate of _TEXTURE_BUFFER
    _TEXTURE_1D,
    _TEXTURE_2D,
    _TEXTURE_3D,
    _TEXTURE_RECTANGLE,
    _TEXTURE_1D_ARRAY,
    _TEXTURE_2D_ARRAY,
    _TEXTURE_CUBE_MAP,
    _TEXTURE_CUBE_MAP_ARRAY,
    _TEXTURE_2D_MULTISAMPLE,
    _TEXTURE_2D_MULTISAMPLE_ARRAY,
    _RENDERBUFFER,
    _MAX_TEXTURE_TYPES
};

#define DIRTY_TEXTURE_LEVEL 0x1
#define DIRTY_TEXTURE_DATA  (DIRTY_TEXTURE_LEVEL << 1)
#define DIRTY_TEXTURE_PARAM (DIRTY_TEXTURE_DATA << 1)
#define DIRTY_TEXTURE_ACCESS (DIRTY_TEXTURE_PARAM << 1)

typedef struct TextureParameter_t {
    GLenum  depth_stencil_mode;
    GLuint  base_level;
    GLfloat border_color[4];
    GLint   border_color_i[4];
    GLuint   border_color_ui[4];
    GLenum  compare_func;
    GLenum  compare_mode;
    GLfloat lod_bias;
    GLenum  min_filter;
    GLenum  mag_filter;
    GLfloat max_anisotropy;
    GLfloat min_lod;
    GLfloat max_lod;
    GLuint  max_level;
    GLboolean swizzled;
    GLenum  swizzle_r;
    GLenum  swizzle_g;
    GLenum  swizzle_b;
    GLenum  swizzle_a;
    GLenum  wrap_s;
    GLenum  wrap_t;
    GLenum  wrap_r;
    /* GL_EXT_texture_sRGB_decode: GL_TEXTURE_SRGB_DECODE_EXT accepts
     * GL_DECODE_EXT (default, sample sRGB-encoded data through the sRGB
     * pipeline) or GL_SKIP_DECODE_EXT (treat sRGB-backed data as linear).
     * Stored here so the texture-creation / pixel-format selection path can
     * downgrade an sRGB Metal pixel format to its linear variant when
     * SKIP_DECODE is requested.  Kept before mtl_data so the param-equality
     * memcmp (which spans up to offsetof(mtl_data)) detects changes to it.
     * Default is GL_DECODE_EXT; texture structs are bzero'd at creation, so 0
     * is treated as "unset → DECODE" by the getter/helper. */
    GLenum  srgb_decode_ext;
    void *mtl_data;
} TextureParameter;

/* GL_EXT_texture_filter_anisotropic: advertised upper bound for
 * GL_MAX_TEXTURE_MAX_ANISOTROPY and the clamp limit for
 * GL_TEXTURE_MAX_ANISOTROPY. THREE-WAY SYNC requirement — same value must
 * appear in:
 *   1. tex_param.c   setTexParmf / setTexParmi (input clamp)
 *   2. get.c         GL_MAX_TEXTURE_MAX_ANISOTROPY query response
 *   3. MGLRenderer+Texture.m  MTLSamplerDescriptor.maxAnisotropy cap
 * Drift between these causes either spec violation (query says 16, set
 * accepts 32) or Metal assertion (descriptor > device limit). Mirrors the
 * kMGLSnapshotBufferBaseTypes three-way sync discipline. */
static const GLfloat kMGLMaxAnisotropyLimit = 16.0f;

/* GL 4.6 §8.14.1 eq 8.8: biasmax is the implementation-defined constant
 * MAX_TEXTURE_LOD_BIAS.  The sum biastexobj + biasshader is clamped to
 * [-biasmax, biasmax] before use.  GL spec does not mandate a minimum, but
 * mainstream implementations advertise 14.0–16.0; 14.0 is a conservative
 * default used when the system OpenGL provider returns 0 (headless / CGL
 * failure).  TWO-WAY SYNC requirement — same value must appear in:
 *   1. glm_params.c  getMacOSDefaults fallback when system query returns 0
 *   2. MGLRenderer+RenderPass.m  clamp of _mglLodBias before setFragmentBytes */
static const GLfloat kMGLMaxTextureLodBias = 14.0f;

typedef enum MGLTexLevelInitSource_t {
    kTexInitNone = 0,
    kTexImageNull,
    kTexImageCopy,
    kTexImagePBO,
    kTexSubImageCPU,
    kTexSubImagePBO,
    kTexRenderTargetWrite,
    kTexMetalFill
} MGLTexLevelInitSource;

typedef struct TextureLevel_t {
    GLboolean complete;
    GLuint width;
    GLuint height;
    GLuint depth;
    size_t pitch;
    GLuint mtl_format;
    size_t  data_size;
    vm_address_t data;
    GLboolean has_initialized_data;
    GLboolean ever_written;
    GLboolean suspicious_zero_upload;
    GLuint last_init_source;
    size_t last_upload_size;
    const void *last_src_ptr;
    uint64_t last_src_hash;
    GLboolean metal_data_authoritative; /* per-level: Metal data is more recent than CPU data (e.g. after blit) */
} TextureLevel;

enum {
    _CUBE_MAP_POSITIVE_X = 0,
    _CUBE_MAP_NEGATIVE_X,
    _CUBE_MAP_POSITIVE_Y,
    _CUBE_MAP_NEGATIVE_Y,
    _CUBE_MAP_POSITIVE_Z,
    _CUBE_MAP_NEGATIVE_Z,
    _CUBE_MAP_MAX_FACE
};

typedef struct TextureFace_t {
    TextureLevel    *levels;
} TextureFace;

#define DIRTY_SAMPLER_PARAM   0x1
typedef struct Sampler_t {
    GLuint dirty_bits;
    GLuint name;
    TextureParameter params;
    void *mtl_data;
} Sampler;

typedef struct Texture_t {
    GLuint dirty_bits;
    GLuint dirty_on_gpu;
    GLboolean is_render_target;
    GLenum access;
    GLboolean immutable_storage;
    GLuint name;
    GLuint target;
    GLuint index;
    GLuint mipmapped;
    GLboolean genmipmaps;
    GLboolean mtl_requires_private_storage; // depth, multi sample
    TextureParameter params;

    // base level params
    GLenum internalformat;
    GLenum compressed_internalformat; // original compressed format if tex was created with compressed internalformat via glTexImage*, 0 otherwise
    GLuint width;
    GLuint height;
    GLuint depth;
    GLboolean is_array;
    GLboolean complete;
    GLuint num_levels;
    GLuint mipmap_levels;
    GLuint samples;
    GLboolean fixed_sample_locations;
    TextureFace faces[6];
    void    *mtl_data;
    void    *mtl_gl_sampled_data;
    GLuint  mtl_gl_sampled_width;
    GLuint  mtl_gl_sampled_height;
    GLuint  mtl_gl_sampled_format;
    GLuint  mtl_gl_sampled_levels;
    GLuint  mtl_gl_sampled_write_version;
    uint32_t mtl_gl_sampled_dirty_mip_mask;
    GLuint  mtl_render_target_write_version;
    /* Y-Flip Authority: packed (mtl_render_target_write_version << 1) | use_original.
     * Set synchronously with mtl_render_target_write_version in
     * mglMarkTextureLevelRenderTargetWritten.  Low bit = 1 means the RT was
     * written in an orientation that should be sampled from the original Metal
     * texture, not from the Y-flipped RT_SAMPLE_COPY.  This covers VS
     * framebuffer-yflip writes; framebuffer-input blit/post passes do not set
     * this bit. */
    GLuint  mtl_render_yflip_authority;
    /* DontCare inference: renderer frame generation at which this
     * texture was last written as a render target. Compared against the
     * renderer's current generation to decide "first render-target use this
     * frame" (a frame's first write can skip loading prior tile contents). */
    GLuint  mtl_rt_frame_generation;
    GLboolean metal_data_authoritative; // set when Metal texture data is more recent than CPU data (e.g. after copyImageSubData blit)
    Buffer  *texture_buffer;
    GLintptr texture_buffer_offset;
    GLsizeiptr texture_buffer_size;
    GLubyte *stencil_shadow;
    GLuint stencil_shadow_width;
    GLuint stencil_shadow_height;
    GLfloat *depth_shadow;
    GLuint depth_shadow_width;
    GLuint depth_shadow_height;
    GLubyte *rgb10a2_shadow;
    GLuint rgb10a2_shadow_width;
    GLuint rgb10a2_shadow_height;
    /* Cached base-level texture view.  newTextureViewWithPixelFormat:
     * is expensive and called per-draw when base_level != 0 (common in
     * Minecraft).  Cache the view keyed on (source texture, base_level,
     * max_level); self-invalidating via comparison in
     * mglSampledTextureViewForBaseLevel.  Released via CFRelease in
     * texture cleanup.  void* for pure-C header compatibility. */
    void      *mtl_base_level_view;          /* id<MTLTexture> retained via CFRetain */
    void      *mtl_base_level_view_source;   /* source id<MTLTexture> (weak ref for identity check) */
    GLuint     mtl_base_level_view_base;
    GLuint     mtl_base_level_view_max;
    GLuint     mtl_base_level_view_swizzle_r;
    GLuint     mtl_base_level_view_swizzle_g;
    GLuint     mtl_base_level_view_swizzle_b;
    GLuint     mtl_base_level_view_swizzle_a;
    char debug_label[128];
} Texture;

typedef struct TextureUnit_t {
    Texture *textures[_MAX_TEXTURE_TYPES];
} TextureUnit;

#define MGL_RECENT_SAMPLED_2D_HISTORY 8

typedef struct ImageUnit_t {
    GLuint unit;
    GLuint texture;
    GLuint level;
    GLboolean layered;
    GLint layer;
    GLenum access;
    GLenum internalformat;
    Texture *tex;
    /* Retained Metal texture view for non-layered / mip binds.  Released on
     * rebind or reset so the GPU can finish using the previous view. */
    void *mtl_image_view;
} ImageUnit;

typedef struct ProxyTextureQueryState_t {
    GLint width;
    GLint height;
    GLint depth;
    GLint internalformat;
} ProxyTextureQueryState;

typedef struct PixelFormat_t {
    GLuint  format;
    GLuint  type;
    GLuint  mtl_pixel_format;
} PixelFormat;

typedef struct PixelStore_t {
    GLboolean   swap_bytes;
    GLboolean   lsb_first;
    GLint row_length;
    GLint image_height;
    GLint skip_rows;
    GLint skip_pixels;
    GLint skip_images;
    GLint alignment;
    /* GL_ARB_compressed_texture_pixel_storage (core since 4.2).
     * Width + size defaulting to 0 leaves compressed pixel storage inactive. */
    GLint compressed_block_width;
    GLint compressed_block_height;
    GLint compressed_block_depth;
    GLint compressed_block_size;
} PixelStore;

#endif /* mgl_types_texture_h */
