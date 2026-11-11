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

#include <stdbool.h>
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

/* GL_MAX_LABEL_LENGTH (GL 4.6 table 23.68 minimum); texture labels are stored
 * inline, so the reported limit must not exceed this. */
#define MGL_MAX_LABEL_LENGTH 256u

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
    /* Which gl*Parameter* last wrote BORDER_COLOR: GL_FLOAT / GL_INT /
     * GL_UNSIGNED_INT.  Metal only has three named float borders; integer
     * borders ride the AIR sampler-wrap pack (CTS texture_border_clamp R32I). */
    GLenum  border_color_type;
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
     * SKIP_DECODE is requested.  The parameter-equality memcmp spans the whole
     * struct, so it detects changes to it.
     * Default is GL_DECODE_EXT; texture structs are bzero'd at creation, so 0
     * is treated as "unset → DECODE" by the getter/helper. */
    GLenum  srgb_decode_ext;
    /* ARB_sparse_texture: the client asked for sparse storage.  Lives here
     * because glTexParameteri only has the parameter block, not the texture.
     * virtual_page_size_index selects the page layout at TexStorage time
     * (only index 0 is advertised).  num_sparse_levels is 0 until storage
     * exists, then the count of levels that can be committed per page. */
    GLboolean sparse;
    GLint virtual_page_size_index;
    GLint num_sparse_levels;
    /* GL_EXT_texture_filter_minmax; zero also means WEIGHTED_AVERAGE. */
    GLenum reduction_mode;
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
    GLuint pixel_format;
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

struct MGLSamplerBackend;
struct MGLTextureBackend;
struct MGLImageUnitBackend;

#define DIRTY_SAMPLER_PARAM   0x1
typedef struct Sampler_t {
    GLuint dirty_bits;
    GLuint name;
    /* Taken from sampler_table.deletion_generation at create; survives
     * name+address reuse after delete (STATE_DATAFLOW T1-2). */
    uint64_t identity_generation;
    TextureParameter params;
    /* Backend state (the Metal sampler).  Opaque here; see
     * mgl_backend_objects.h. */
    struct MGLSamplerBackend *backend;
    char *debug_label;  /* ObjectLabel, GL 4.6 §20.7; NULL = empty */
} Sampler;

/* One committed texel box.  ARB_sparse_texture2 reads outside every box
 * return zero, and writes there are discarded. */
typedef struct MGLSparseCommit_t {
    GLint level;
    GLint x, y, z;
    GLint w, h, d;
} MGLSparseCommit;

typedef struct Texture_t {
    GLuint dirty_bits;
    GLboolean is_render_target;
    /* ARB_sparse_texture: the client asked for sparse storage
     * (GL_TEXTURE_SPARSE_ARB) and, once created, the placement heap the Metal
     * texture was carved out of.  Commitment needs that heap again, so it
     * travels with the texture and is released with it. */
    void *sparse_heap;
    /* Next free placement-heap tile index for Map.  Each Map must start at a
     * unique offset — reusing 0 aliases every mip/region onto the same
     * physical tiles (Clamp LookupColor rgba8 L0 flake). */
    uint64_t sparse_heap_next_tile;
    /* Page size selected at TexStorage (standard table, index 0). */
    GLint sparse_page_x, sparse_page_y, sparse_page_z;
    MGLSparseCommit *sparse_commits;
    GLuint sparse_commit_count;
    GLuint sparse_commit_cap;
    GLenum access;
    GLboolean immutable_storage;
    GLuint name;
    /* Taken from texture_table.deletion_generation at create (T1-2). */
    uint64_t identity_generation;
    GLuint target;
    GLuint index;
    GLuint mipmapped;
    GLboolean genmipmaps;
    TextureParameter params;

    // base level params
    GLenum internalformat;
    /* When BindImageTexture reinterprets a non-atomic Metal format as
     * R32I/UI (same bpp), Metal ShaderAtomic needs a real R32 store — a
     * PixelFormatView cannot carry ShaderAtomic.  Non-zero overrides the
     * Metal pixel format while GL internalformat stays unchanged so
     * GetTexImage still reads the raw cast bytes. */
    GLenum image_metal_format;
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
    GLboolean metal_data_authoritative; // set when Metal texture data is more recent than CPU data (e.g. after copyImageSubData blit)
    /* F22/F24's fallback flag lives in MGLTextureBackend::storage_is_fallback:
     * it describes the Metal storage, not the GL object. */
    Buffer  *texture_buffer;
    GLintptr texture_buffer_offset;
    GLsizeiptr texture_buffer_size;
    GLboolean texture_buffer_whole;
    GLubyte *stencil_shadow;
    GLuint stencil_shadow_width;
    GLuint stencil_shadow_height;
    GLfloat *depth_shadow;
    GLuint depth_shadow_width;
    GLuint depth_shadow_height;
    GLubyte *rgb10a2_shadow;
    GLuint rgb10a2_shadow_width;
    GLuint rgb10a2_shadow_height;
    /* Backend state kept beside the texture's own storage (the cached
     * base-level view).  Opaque here; see mgl_backend_objects.h and
     * mgl_texture_backend.h. */
    struct MGLTextureBackend *backend;
    /* §8.18/§8.19 state, meaningful only while immutable_storage is set.
     * view_min_level/view_min_layer are absolute in the shared store. */
    GLuint immutable_levels;
    GLuint view_min_level;
    GLuint view_num_levels;
    GLuint view_min_layer;
    GLuint view_num_layers;
    /* TextureView family: views point at the texture that owns the store;
     * the owner lists its views and outlives its name while any remain. */
    struct Texture_t *view_root;
    struct Texture_t **views;
    GLuint view_count;
    GLuint view_capacity;
    GLboolean view_root_deleted;
    char debug_label[MGL_MAX_LABEL_LENGTH];
} Texture;

/* Whether the texture is currently a render target (bookkeeping flag). */
static inline bool mglTextureIsRenderTarget(const Texture *tex)
{
    return tex ? tex->is_render_target != 0 : false;
}

typedef struct TextureUnit_t {
    Texture *textures[_MAX_TEXTURE_TYPES];
} TextureUnit;

typedef struct ImageUnit_t {
    GLuint unit;
    GLuint texture;
    GLuint level;
    GLboolean layered;
    GLint layer;
    GLenum access;
    GLenum internalformat;
    Texture *tex;
    /* Backend state (the Metal view used for non-layered / mip access).
     * Opaque here; see mgl_backend_objects.h and mgl_texture_backend.h. */
    struct MGLImageUnitBackend *backend;
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
