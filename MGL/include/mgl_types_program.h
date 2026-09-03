/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

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
 * mgl_types_program.h
 * MGL
 *
 * Shader / program domain type definitions split from glm_context.h.
 */

#ifndef mgl_types_program_h
#define mgl_types_program_h

#include <stddef.h>

#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"

typedef struct GLMContextRec_t *GLMContext;

enum {
    _VERTEX_SHADER = 0,
    _TESS_CONTROL_SHADER,
    _TESS_EVALUATION_SHADER,
    _GEOMETRY_SHADER,
    _FRAGMENT_SHADER,
    _COMPUTE_SHADER,
    _MAX_SHADER_TYPES
};

/* Stable resource-type indices into the program reflection tables. */
enum {
    _UNKNOWN_RES = 0,
    _UNIFORM_BUFFER_RES,
    _UNIFORM_CONSTANT_RES,
    _STORAGE_BUFFER_RES,
    _STAGE_INPUT_RES,
    _STAGE_OUTPUT_RES,
    _SUBPASS_INPUT_RES,
    _STORAGE_IMAGE_RES,
    _SAMPLED_IMAGE_RES,
    _ATOMIC_COUNTER_RES,
    _PUSH_CONSTANT_RES,
    _SEPARATE_IMAGE_RES,
    _SEPARATE_SAMPLERS_RES,
    _ACCEL_STRUCT_RES,
    _RAY_QUERY,
    MGL_MAX_SHADER_RESOURCES
};
/* Compatibility alias for the historical SPIR-V-era enumerator name. */
#define _MAX_SPIRV_RES MGL_MAX_SHADER_RESOURCES

/* Texture dimensionality stored in MGLShaderResource::image_dim.  The
 * numeric values intentionally match the historical SPIR-V Dim encoding so
 * existing serialized/reflected resource data remains compatible without
 * importing SPIR-V headers. */
typedef enum MGLImageDimension {
    MGL_IMAGE_DIM_NONE = 0,
    MGL_IMAGE_DIM_1D = 0,
    MGL_IMAGE_DIM_2D = 1,
    MGL_IMAGE_DIM_3D = 2,
    MGL_IMAGE_DIM_CUBE = 3,
    MGL_IMAGE_DIM_RECT = 4,
    MGL_IMAGE_DIM_BUFFER = 5,
    MGL_IMAGE_DIM_SUBPASS_DATA = 6
} MGLImageDimension;

typedef enum MGLShaderTextureDataKind {
    MGL_SHADER_TEXTURE_DATA_UNKNOWN = 0,
    MGL_SHADER_TEXTURE_DATA_FLOAT = 1,
    MGL_SHADER_TEXTURE_DATA_SINT = 2,
    MGL_SHADER_TEXTURE_DATA_UINT = 3,
    MGL_SHADER_TEXTURE_DATA_DEPTH = 4
} MGLShaderTextureDataKind;

#define SHADER_MASK_BIT(_TYPE_)    (0x1 << _TYPE_)
#define VERTEX_SHADER_MASK_BIT  SHADER_MASK_BIT(_VERTEX_SHADER)
#define FRAGMENT_SHADER_MASK_BIT  SHADER_MASK_BIT(_FRAGMENT_SHADER)
#define GEOMETRY_SHADER_MASK_BIT  SHADER_MASK_BIT(_GEOMETRY_SHADER)
#define TESS_CONTROL_SHADER_MASK_BIT  SHADER_MASK_BIT(_TESS_CONTROL_SHADER)
#define TESS_EVALUATION_SHADER_MASK_BIT  SHADER_MASK_BIT(_TESS_EVALUATION_SHADER)
#define COMPUTE_SHADER_MASK_BIT  SHADER_MASK_BIT(_COMPUTE_SHADER)

/* How a program's geometry shader is executed (or whether it can be at
 * all).  Decided at link time; the value also drives the GL_MAX_GEOMETRY_*
 * capability reporting and the glGetProgramiv geometry reflection. */
typedef enum MGLGSRoute {
    MGL_GS_ROUTE_NONE = 0,          /* no GS attached */
    MGL_GS_ROUTE_COMPUTE,           /* Compute expansion path. */
    MGL_GS_ROUTE_MESH,              /* Mesh-shader path. */
    MGL_GS_ROUTE_UNSUPPORTED        /* GS attached, no execution path yet */
} MGLGSRoute;

typedef struct Shader_t {
    GLuint dirty_bits;
    GLuint name;
    GLuint type;
    GLuint glm_type;
    const char *mtl_shader_type_name;
    size_t src_len;
    const char *src;
    GLboolean compile_success;
    const char *entry_point;
    char *log;
    int refcount;
    GLboolean delete_status;
} Shader;

/* Per-shader backend module state: AIR serialized metallib bytes + the
 * renderer-owned Metal objects built from them. */
typedef struct MGLShaderModule_t {
    GLuint stage;
    char *entry_point;
    /* AIR backend output: serialized metallib (bitcode container). */
    unsigned char *metallib_bytes;
    size_t metallib_size;
    unsigned char *metallib_tess_capture_bytes;
    size_t metallib_tess_capture_size;
    unsigned char *metallib_cull_capture_bytes;
    size_t metallib_cull_capture_size;
    void *mtl_function;
    void *mtl_library;
    void *mtl_tess_capture_function;
    void *mtl_tess_capture_library;
    void *mtl_cull_capture_function;
    void *mtl_cull_capture_library;
    void *mtl_compute_pipeline;
    GLboolean mgl_injected_framebuffer_yflip; /* true if MGL injected a
                                               * texCoord Y-flip for sampled
                                               * framebuffer in this shader */
    GLboolean needs_runtime_array_size_buffer;
} MGLShaderModule;
/* Compatibility aliases for the historical SPIR-V-era names. */
typedef MGLShaderModule Spirv;

typedef struct SpirvUBOMember_t {
    const char *name;        /* e.g. "var" inside Block { bool var; }            */
    char       *query_name;  /* Program-interface name, possibly block scoped.   */
    GLuint      gl_type;     /* GL_BOOL, GL_FLOAT_VEC4, etc.                    */
    GLuint      offset;      /* byte offset within the UBO (GL_UNIFORM_OFFSET)  */
    GLint       array_stride;  /* GL_UNIFORM_ARRAY_STRIDE, -1 if not an array   */
    GLint       matrix_stride; /* GL_UNIFORM_MATRIX_STRIDE, -1 if not a matrix  */
    GLboolean   is_row_major;  /* GL_UNIFORM_IS_ROW_MAJOR                       */
    GLint       size;        /* GL_UNIFORM_SIZE (array element count, 1 for scalar, 0 for runtime array) */
    GLint       location_offset; /* Plain struct leaf location relative to parent */
    GLint       top_level_array_size;  /* GL_TOP_LEVEL_ARRAY_SIZE for buffer variables */
    GLint       top_level_array_stride; /* GL_TOP_LEVEL_ARRAY_STRIDE for buffer variables */
} SpirvUBOMember;

typedef struct MGLShaderResource_t {
    GLuint  _id;
    GLuint  base_type_id;
    GLuint  type_id;
    const char *name;
    /* Metal allocates texture and sampler indices in independent namespaces. */
    GLuint  combined_sampler_binding;
    GLboolean resource_active;
    GLboolean has_combined_sampler;
    GLuint  set;
    /* GL client binding point. For UBOs, glUniformBlockBinding updates this. */
    GLuint  gl_binding;
    GLuint  ubo_array_size;
    GLboolean ubo_is_array;
    GLuint  ubo_array_element;
    GLuint *ubo_array_bindings;
    GLboolean ubo_has_instance_name;
    char   *ubo_instance_name;
    /* Metal argument slot parsed from generated MSL after resource repair. */
    GLuint  binding;
    GLuint  location;
    GLuint  location_index; /* dual-source blending index (SpvDecorationIndex) */
    GLuint  gl_type;
    GLint   gl_array_size;
    GLboolean is_array; /* true if the underlying shader type is an array */
    GLuint  num_array_dims; /* number of array dimensions (0 if not array) */
    GLint   uniform_location;
    GLint   sampler_unit;
    GLboolean sampler_unit_explicit;
    size_t  required_size;
    GLuint  image_dim;
    GLuint  image_arrayed;
    GLuint  image_multisampled;
    GLuint  texture_data_kind;
    /* True for tessellation patch variables (SpvDecorationPatch). */
    GLboolean is_per_patch;
    /* GS output stream index (GL 4.6 §11.1.3.4); 0 for non-GS stages or
     * stream 0 outputs.  Streams > 0 are transform-feedback only and must
     * be excluded from the rasterizing passthrough vertex function. */
    GLint   stream;
    /* True for interface-block member varyings: their array dimension is
     * an ordinary member array, not the per-input-vertex instance array
     * dimension, so geometry-interface validation must not compare it
     * against the input-vertex count. */
    GLboolean block_member;
    /* UBO member uniforms (only valid for _UNIFORM_BUFFER_RES). */
    SpirvUBOMember       *ubo_members;
    GLuint                ubo_member_count;
    /* When this resource is used as a placeholder during active-uniform
     * enumeration of UBO members, this points to the specific member. */
    const SpirvUBOMember *ubo_member;
} MGLShaderResource;
/* Compatibility alias for the historical SPIR-V-era name. */
typedef MGLShaderResource SpirvResource;

typedef struct MGLShaderResourceList_t {
    GLuint  count;
    MGLShaderResource   *list;
} MGLShaderResourceList;
/* Compatibility alias for the historical SPIR-V-era name. */
typedef MGLShaderResourceList SpirvResourceList;

/* Entry in the cached active-uniform list.  Pointers are into the Program's
 * own shader_resources_list and are valid for the program's link lifetime.
 * Defined here (before Program) because Program holds a pointer to it. */
typedef struct MGLActiveUniformEntry_t {
    MGLShaderResource *res;
    int stage;
    int res_type;
    const SpirvUBOMember *ubo_member;  /* NULL for non-member uniforms */
} MGLActiveUniformEntry;

#define MAX_ATTACHED_SHADERS_PER_STAGE 8
/* Forward declaration: full type defined in mgl_buffer_plan.h.  The plan
 * caches static reflection-derived buffer binding data (metal slots, client
 * bindings, struct packing metadata) so per-draw paths can skip repeated name
 * lookups and program resolution.  Built at link end, invalidated on relink
 * and binding mutations, freed at program deletion. */
typedef struct MGLBufferBindingPlan MGLBufferBindingPlan;

#define MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS 4u

/* Link-time transform-feedback scatter plan.  The plan is populated even for
 * the currently unsupported GS SEPARATE_ATTRIBS execution route so that link
 * validation and the eventual capture backend share one authoritative layout. */
typedef struct MGLTransformFeedbackVaryingPlan_t {
    GLuint buffer_index;
    GLuint component_offset;
    GLuint component_count;
    GLint stream;
    GLboolean builtin;
} MGLTransformFeedbackVaryingPlan;

typedef struct Program_t {
    GLuint dirty_bits;
    GLuint name;
    int refcount;
    GLboolean delete_status;
    Shader *shader_slots[_MAX_SHADER_TYPES];
    Shader *attached_shader_slots[_MAX_SHADER_TYPES][MAX_ATTACHED_SHADERS_PER_STAGE];
    GLuint attached_shader_counts[_MAX_SHADER_TYPES];
    GLbitfield attached_shader_mask;
    GLboolean link_success;
    MGLShaderModule modules[_MAX_SHADER_TYPES];
    MGLShaderResourceList shader_resources_list[_MAX_SHADER_TYPES][MGL_MAX_SHADER_RESOURCES];
    struct {
        unsigned x, y, z;
    } local_workgroup_size;
    GLuint tess_control_output_vertices;  /* from TCS layout(vertices=N) out; */
    /* Geometry shader execution route, decided at link time. */
    MGLGSRoute gs_route;
    GLenum geometry_input_type;
    GLenum geometry_output_type;
    GLuint geometry_vertices_out;
    GLboolean geometry_max_vertices_specified;
    GLuint geometry_invocations;
    /* GS multi-stream XFB layout (GL 4.6 §11.1.3.4): stream s captures to
     * transform-feedback buffer s with a compact position+varyings record
     * of geometry_stream_xfb_stride[s] bytes (stream 0 keeps the full
     * stage-out record at runtime). */
    GLuint geometry_stream_count;           /* streams used: 1..4 */
    GLuint geometry_stream_varying_count[4];
    GLuint geometry_stream_xfb_stride[4];
    /* TES execution mode reflection: layout(...) in; */
    GLenum tess_gen_mode;        /* GL_TRIANGLES / GL_QUADS / GL_ISOLINES */
    GLenum tess_gen_spacing;     /* GL_EQUAL / GL_FRACTIONAL_EVEN / GL_FRACTIONAL_ODD */
    GLenum tess_gen_vertex_order;/* GL_CW / GL_CCW */
    GLboolean tess_gen_point_mode;/* GL_TRUE / GL_FALSE */
    /* TES compiled as AIR compute expansion (isolines, point_mode, or
     * forced for XFB).  Draw/bind paths must use the compute ABI. */
    GLboolean tess_eval_compute;
    /* 1 if the TES compilation unit declared an input primitive mode. */
    GLboolean tess_gen_mode_specified;
    GLint sampler_units[TEXTURE_UNITS];
    GLint sampler_units_by_stage[_MAX_SHADER_TYPES][TEXTURE_UNITS];
    GLboolean sampler_units_explicit[TEXTURE_UNITS];
    GLboolean sampler_units_explicit_by_stage[_MAX_SHADER_TYPES][TEXTURE_UNITS];
    /* Cached bitmap of texture units the program actually samples.
     * Lazily built by mglProgramSamplesTextureUnit on first query and
     * invalidated whenever sampler_units* are modified (glUniform1i,
     * relink).  Eliminates the O(stages * resources) scan per texture
     * unit in the 3 hazard-scan call sites. */
    uint32_t sampled_texture_unit_mask[4];  /* 128 bits */
    uint8_t  sampled_texture_unit_mask_valid;
    /* Precomputed table of which Metal sampler slots are shared across
     * multiple sampler-like resources (across all stages and the 5 sampler
     * resource types).  Built at link time so mglMetalSamplerSlotSharedAcross-
     * Resources can answer in O(1) instead of a full stage×resource scan.
     * sampler_binding_shared_valid gates readers; 0 = not yet built (fall
     * back to full scan).  Invalidated at link start, populated at link end. */
    uint8_t  sampler_binding_shared[TEXTURE_UNITS];
    uint8_t  sampler_binding_shared_valid;
    /* Bitmap of uniform locations that are sampler-like resources.
     * Built at link time so mglSetSamplerUniformUnit can O(1) reject
     * non-sampler locations (the common case for glUniform1i uploading
     * plain ints like FogShape).  Covers locations 0–127; locations
     * outside this range fall back to the full scan. */
    uint64_t sampler_location_bitmap[2];  /* 128 bits */
    uint8_t  sampler_location_bitmap_valid;
    GLboolean uses_vertex_id;
    GLboolean uses_primitive_id;
    GLboolean usesFragCoordParams;
    uint32_t vertexAttribUsageMask;
    GLboolean uses_point_size_params;
    GLboolean uses_cull_distance;
    uint32_t cull_distance_count;
    /* TES-stage cull distance usage (the TES-written gl_CullDistance drives
     * post-tess culling of isolines/point-mode expansions; the VS-side
     * fields above drive the pre-tess capture path). */
    GLboolean tess_uses_cull_distance;
    uint32_t tess_cull_distance_count;
    GLboolean uses_lod_bias;
    /* IR-level reflection cache for mglBufferSlotConflictsForProgram.
     * Computed lazily on first call during link-time resource binding, then
     * reused for all subsequent slot checks.  Invalidated at link start. */
    GLboolean ir_cache_valid;
    GLboolean ir_uses_cull_distance;
    GLboolean ir_uses_frag_coord;
    MGLShaderResourceList *validated_resource_lists[_MAX_SHADER_TYPES][MGL_MAX_SHADER_RESOURCES];
    MGLShaderResource *validated_resource_list_storage[_MAX_SHADER_TYPES][MGL_MAX_SHADER_RESOURCES];
    GLuint validated_resource_list_counts[_MAX_SHADER_TYPES][MGL_MAX_SHADER_RESOURCES];
    uint64_t pipeline_cache_instance_id;
    uint64_t pipeline_cache_generation;
    GLboolean program_separable;
    BufferBaseTarget plain_uniform_buffers[MAX_BINDABLE_BUFFERS];
    /* Active-binding bitmap for plain_uniform_buffers: bit i is set iff
     * plain_uniform_buffers[i].buf != NULL.  Maintained at uniform upload
     * (uniforms.c) and link-time reflection (program.c / mgl_program_reflection.c) so
     * mglComputeDrawBufferBindingHashScan and mglTrackPendingBaseBufferReads
     * can skip the ~84-slot linear scan.  84 bits fit in 2 × uint64_t. */
    uint64_t plain_uniform_active_mask[2];
    /* Cached uniform locations for the legacy clip-plane derivation
     * uniforms (_mglClipPlane / _mglClipPlaneEnabled); -1 when the program
     * does not use gl_ClipVertex.  Looked up at link end, refreshed per
     * draw from the GL clip-plane state (glClipPlane/glEnable). */
    GLint legacy_clip_plane_loc;
    GLint legacy_clip_plane_enabled_loc;
    char *attrib_location_names[MAX_ATTRIBS];
    GLboolean attrib_location_name_owned[MAX_ATTRIBS];
    /* Pre-link fragment output bindings set via glBindFragDataLocation(Indexed).
     * Applied to fragment stage outputs after reflection. */
    char *frag_data_location_names[MAX_ATTRIBS];
    GLuint frag_data_color_numbers[MAX_ATTRIBS];
    GLuint frag_data_indices[MAX_ATTRIBS];
    GLuint frag_data_location_count;
    GLsizei transform_feedback_varying_count;
    GLenum transform_feedback_buffer_mode;
    char transform_feedback_varying_names[MAX_ATTRIBS][96];
    MGLTransformFeedbackVaryingPlan transform_feedback_layout[MAX_ATTRIBS];
    GLuint transform_feedback_layout_buffer_count;
    GLuint transform_feedback_layout_component_count;
    GLboolean transform_feedback_layout_valid;
    /* Built-in variables exposed as active PROGRAM_INPUT / PROGRAM_OUTPUT
     * resources (gl_VertexID, gl_InstanceID, gl_FragDepth, gl_SampleMask, etc.).
     * Stored per-stage so that separate (single-stage) programs can expose
     * their own stage's built-ins.  Kept separate from the main
     * STAGE_INPUT/STAGE_OUTPUT lists so the rendering code (vertex descriptor
     * setup, varyings linking) is unaffected. */
    MGLShaderResource builtin_program_inputs[_MAX_SHADER_TYPES][16];
    GLuint builtin_program_input_count[_MAX_SHADER_TYPES];
    MGLShaderResource builtin_program_outputs[_MAX_SHADER_TYPES][16];
    GLuint builtin_program_output_count[_MAX_SHADER_TYPES];
    /* Cached buffer binding plan (per-stage).  NULL until first build.
     * See mgl_buffer_plan.h for the lifecycle and cache contract. */
    MGLBufferBindingPlan *buffer_binding_plan;
    /* Cached deduplicated active-uniform list, built at link time.
     * Eliminates the O(N³) enumeration in mglProgramActiveUniformCount /
     * mglProgramActiveUniformAt / mglProgramActiveUniformIndexByName /
     * mglProgramActiveUniformMaxNameLength.  Invalidated at link start,
     * populated on successful link.  active_uniform_cache_valid gates
     * readers; GL_FALSE = fall back to original O(N³) path. */
    MGLActiveUniformEntry *active_uniform_cache;
    GLuint active_uniform_cache_count;
    GLint active_uniform_cache_max_name_length;
    GLboolean active_uniform_cache_valid;
    void *mtl_data;
} Program;

GLint mglProgramActiveUniformCount(Program *program);
GLint mglProgramActiveUniformMaxNameLength(Program *program);
MGLShaderResource *mglProgramActiveUniformAt(Program *program, GLuint index, int *stage_out, int *res_type_out);
/* Build the deduplicated active-uniform cache.  Called at link end.
 * Frees any previous cache.  After this, mglProgramActiveUniformCount /
 * mglProgramActiveUniformAt / mglProgramActiveUniformIndexByName /
 * mglProgramActiveUniformMaxNameLength run in O(1) / O(1) / O(N) / O(1). */
void mglBuildActiveUniformCache(Program *program);
/* Free the active-uniform cache.  Called at link start and program free. */
void mglFreeActiveUniformCache(Program *program);
GLint mglProgramActiveUniformIndexByName(Program *program, const GLchar *name);
GLint mglProgramActiveUniformGLType(const MGLShaderResource *res, int res_type);
GLint mglProgramActiveUniformSize(const MGLShaderResource *res, int res_type);
GLsizei mglProgramActiveUniformNameLength(const MGLShaderResource *res);
GLint mglProgramActiveUniformBlockIndex(Program *program, const MGLShaderResource *res);
void mglProgramCopyActiveUniformName(const MGLShaderResource *res, GLsizei bufSize, GLsizei *length, GLchar *name);
GLboolean mglProgramPointerUsableForName(GLMContext ctx, Program *program, GLuint expectedName);
void mglRetainProgramReference(GLMContext ctx, Program *program);
void mglReleaseProgramReference(GLMContext ctx, Program *program);

/* plain_uniform_active_mask helper.  Called at uniform upload / link-time
 * reflection to keep the bitmap in sync with which plain_uniform_buffers
 * slots have buf != NULL.  Draw-command hot paths (hash + hazard tracker)
 * use the bitmap to skip the ~84-slot linear scan.  Slots are only ever
 * populated (never cleared) while a program is alive, so this is the only
 * helper the maintenance sites need. */
static inline void mglProgramPlainUniformSetActive(Program *program, GLuint index)
{
    if (!program || index >= MAX_BINDABLE_BUFFERS) return;
    program->plain_uniform_active_mask[index >> 6] |= (uint64_t)1u << (index & 63u);
}

typedef struct ProgramPipeline_t {
    GLuint name;
    GLboolean validated;
    Program *stage_programs[_MAX_SHADER_TYPES];  // Programs attached to each stage
} ProgramPipeline;

typedef struct TransformFeedback_t {
    GLuint name;
    GLenum target;
    GLboolean created;
    GLboolean active;
    GLboolean paused;
    GLenum primitive_mode;
    GLuint64 primitives_generated;
    GLuint64 primitives_written;
    /* Byte cursors for the current Begin/End capture session. Begin resets
     * them; pause/resume preserves them so subsequent draws append. */
    GLuint64 buffer_write_offsets[MAX_BINDABLE_BUFFERS];
    BufferBaseTarget buffers[MAX_BINDABLE_BUFFERS];
} TransformFeedback;

#endif /* mgl_types_program_h */
