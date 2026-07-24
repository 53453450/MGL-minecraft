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
 * Shader / program / SPIRV domain type definitions split from glm_context.h.
 */

#ifndef mgl_types_program_h
#define mgl_types_program_h

#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"

typedef struct GLMContextRec_t *GLMContext;

typedef enum MGLMSLBindingKind {
    MGL_MSL_BINDING_NONE = 0,
    MGL_MSL_BINDING_TEXTURE,
    MGL_MSL_BINDING_BUFFER,
    MGL_MSL_BINDING_SAMPLER
} MGLMSLBindingKind;

enum {
    _VERTEX_SHADER = 0,
    _TESS_CONTROL_SHADER,
    _TESS_EVALUATION_SHADER,
    _GEOMETRY_SHADER,
    _FRAGMENT_SHADER,
    _COMPUTE_SHADER,
    _MAX_SHADER_TYPES
};

enum {
    _UNKNOWN_RES = 0,
    _UNIFORM_BUFFER_RES,
    _UNIFORM_CONSTANT_RES,
    _STORAGE_BUFFER_RES,
    _STAGE_INPUT_RES,
    _STAGE_OUTPUT_RES,
    _STORAGE_OUTPUT_RES,
    _ATOMIC_COUNTER_RES,
    _PUSH_CONSTANT_RES,
    _SEPARATE_IMAGE_RES,
    _SEPARATE_SAMPLERS_RES,
    _ACCEL_STRUCT_RES,
    _RAY_QUERY,
    _MAX_SPIRV_RES
};

#define SHADER_MASK_BIT(_TYPE_)    (0x1 << _TYPE_)
#define VERTEX_SHADER_MASK_BIT  SHADER_MASK_BIT(_VERTEX_SHADER)
#define FRAGMENT_SHADER_MASK_BIT  SHADER_MASK_BIT(_FRAGMENT_SHADER)
#define GEOMETRY_SHADER_MASK_BIT  SHADER_MASK_BIT(_GEOMETRY_SHADER)
#define TESS_CONTROL_SHADER_MASK_BIT  SHADER_MASK_BIT(_TESS_CONTROL_SHADER)
#define TESS_EVALUATION_SHADER_MASK_BIT  SHADER_MASK_BIT(_TESS_EVALUATION_SHADER)
#define COMPUTE_SHADER_MASK_BIT  SHADER_MASK_BIT(_COMPUTE_SHADER)

typedef struct Shader_t {
    GLuint dirty_bits;
    GLuint name;
    GLuint type;
    GLuint glm_type;
    const char *mtl_shader_type_name;
    size_t src_len;
    const char *src;
    glslang_shader_t *compiled_glsl_shader;
    const char *entry_point;
    char *log;
    int refcount;
    GLboolean delete_status;
    struct {
        void *function;
        void *library;
        void *zero_to_one_function;
        void *zero_to_one_library;
        void *upper_left_function;
        void *upper_left_library;
        void *upper_left_zero_to_one_function;
        void *upper_left_zero_to_one_library;
    } mtl_data;
} Shader;

typedef struct Spirv_t {
    GLuint stage;
    size_t size;
    unsigned int *ir;
    char *msl_str;
    char *entry_point;
    void *mtl_function;
    void *mtl_library;
    void *mtl_compute_pipeline;
    void *mtl_zero_to_one_function;
    void *mtl_zero_to_one_library;
    void *mtl_upper_left_function;
    void *mtl_upper_left_library;
    void *mtl_upper_left_zero_to_one_function;
    void *mtl_upper_left_zero_to_one_library;
    GLboolean mgl_injected_framebuffer_yflip; /* true if MGL injected a
                                               * texCoord Y-flip for sampled
                                               * framebuffer in this shader */
    GLboolean needs_buffer_size_buffer; /* true if SPIRV-Cross MSL uses
                                         * spvBufferSizeConstants for
                                         * runtime-sized SSBO arrays */
    char *msl_str_capture; /* MSL variant compiled with SPIRV-Cross output-capture
                            * options for GPU transform feedback. NULL unless
                            * MGL_XFB_GPU_CAPTURE is set and the stage is the
                            * program's feedback stage. The renderer dispatch
                            * that consumes this variant is not yet wired. */
} Spirv;

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

typedef struct SpirvResource_t {
    GLuint  _id;
    GLuint  base_type_id;
    GLuint  type_id;
    const char *name;
    /* Final identifier and argument kind emitted by SPIRV-Cross for active
     * resources. `name` remains the original GL-facing reflection name. */
    char   *msl_name;
    /* Exact final MSL declarators for resources expanded into more than one
     * argument (for example, UBO arrays lowered to block_0, block_1, ...).
     * msl_name remains the first argument name for existing callers. */
    char  **msl_argument_names;
    GLuint  msl_argument_count;
    char   *msl_combined_sampler_name;
    /* Metal allocates texture and sampler indices in independent namespaces. */
    GLuint  msl_combined_sampler_binding;
    GLboolean msl_active;
    GLboolean msl_has_combined_sampler;
    MGLMSLBindingKind msl_binding_kind;
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
    GLboolean is_array; /* true if the underlying SPIR-V type is an array */
    GLuint  num_array_dims; /* number of SPIR-V array dimensions (0 if not array) */
    GLint   uniform_location;
    GLint   sampler_unit;
    GLboolean sampler_unit_explicit;
    size_t  required_size;
    GLuint  image_dim;
    GLuint  image_arrayed;
    GLuint  image_multisampled;
    /* True for tessellation patch variables (SpvDecorationPatch). */
    GLboolean is_per_patch;
    /* UBO member uniforms (only valid for SPVC_RESOURCE_TYPE_UNIFORM_BUFFER). */
    SpirvUBOMember       *ubo_members;
    GLuint                ubo_member_count;
    /* When this resource is used as a placeholder during active-uniform
     * enumeration of UBO members, this points to the specific member. */
    const SpirvUBOMember *ubo_member;
    /* Per-resource MSL texture type / data kind cache.  The MSL string is
     * immutable post-link, so the type for a given resource never changes
     * between relinks.  Stored on the resource itself (rather than an
     * external NSString→NSNumber NSCache) to eliminate per-draw
     * stringWithFormat + NSDictionary lookup cost.
     *
     * cached_msl_texture_type_valid: 0 = uncached, 1 = cached.
     * MTLTextureType1D == 0 on this SDK, so a separate valid flag is needed
     * for the texture type.  MGLTextureDataKindUnknown == 0, but the cached
     * value is always the *resolved* kind (>= 1), so 0 reliably means
     * uncached for the data kind. */
    uint32_t cached_msl_texture_type;
    uint8_t  cached_msl_texture_type_valid;
    uint32_t cached_msl_data_kind;
} SpirvResource;

typedef struct SpirvResourceList_t {
    GLuint  count;
    SpirvResource   *list;
} SpirvResourceList;

#define MAX_ATTACHED_SHADERS_PER_STAGE 8
#define MGL_MSL_NAMED_ARGUMENT_CACHE_CAPACITY 64u

typedef struct {
    const char *name;
    GLuint binding;
    uint8_t stage;
    uint8_t attribute_kind;
    GLboolean result;
} MGLMSLNamedArgumentCacheEntry;

/* Forward declaration: full type defined in mgl_buffer_plan.h.  The plan
 * caches static reflection-derived buffer binding data (metal slots, client
 * bindings, struct packing metadata) so per-draw paths can skip repeated name
 * lookups and program resolution.  Built at link end, invalidated on relink
 * and binding mutations, freed at program deletion. */
typedef struct MGLBufferBindingPlan MGLBufferBindingPlan;

typedef struct Program_t {
    GLuint dirty_bits;
    GLuint name;
    int refcount;
    GLboolean delete_status;
    Shader *shader_slots[_MAX_SHADER_TYPES];
    Shader *attached_shader_slots[_MAX_SHADER_TYPES][MAX_ATTACHED_SHADERS_PER_STAGE];
    GLuint attached_shader_counts[_MAX_SHADER_TYPES];
    GLbitfield attached_shader_mask;
    glslang_program_t *linked_glsl_program;
    Spirv spirv[_MAX_SHADER_TYPES];
    SpirvResourceList spirv_resources_list[_MAX_SHADER_TYPES][_MAX_SPIRV_RES];
    struct {
        unsigned x, y, z;
    } local_workgroup_size;
    GLuint tess_control_output_vertices;  /* from TCS layout(vertices=N) out; */
    /* TES execution mode reflection: layout(...) in; */
    GLenum tess_gen_mode;        /* GL_TRIANGLES / GL_QUADS / GL_ISOLINES */
    GLenum tess_gen_spacing;     /* GL_EQUAL / GL_FRACTIONAL_EVEN / GL_FRACTIONAL_ODD */
    GLenum tess_gen_vertex_order;/* GL_CW / GL_CCW */
    GLboolean tess_gen_point_mode;/* GL_TRUE / GL_FALSE */
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
    /* P1-6: Precomputed table of which Metal sampler slots are shared across
     * multiple sampler-like resources (across all stages and the 5 sampler
     * resource types).  Built at link time so mglMetalSamplerSlotSharedAcross-
     * Resources can answer in O(1) instead of a full stage×resource scan.
     * sampler_binding_shared_valid gates readers; 0 = not yet built (fall
     * back to full scan).  Invalidated at link start, populated at link end. */
    uint8_t  sampler_binding_shared[TEXTURE_UNITS];
    uint8_t  sampler_binding_shared_valid;
    GLboolean uses_vertex_id;
    GLboolean uses_primitive_id;
    /* MSL query result cache (env-gated by MGL_MSL_CACHE, default ON; =0 off).
     * mslCacheValid is GL_FALSE until mglLinkProgram scans the generated MSL
     * once and populates usesFragCoordParams / vertexAttribUsageMask; while
     * GL_FALSE the per-draw paths fall back to their original strstr() scan.
     * Invalidated at the start of every mglLinkProgram and repopulated on
     * successful link, so the cache always reflects the current MSL. */
    GLboolean mslCacheValid;
    GLboolean usesFragCoordParams;   /* FS: gl_FragCoord params present?  */
    uint32_t vertexAttribUsageMask;  /* VS: bit N set => [[attribute(N)]] */
    GLboolean uses_point_size_params; /* VS/TES/GS: _mgl_point_size_params in MSL */
    GLboolean uses_cull_distance;     /* VS: mgl_CullDistance in MSL */
    MGLMSLNamedArgumentCacheEntry
        msl_named_argument_cache[MGL_MSL_NAMED_ARGUMENT_CACHE_CAPACITY];
    uint8_t msl_named_argument_cache_next;
    SpirvResourceList *validated_resource_lists[_MAX_SHADER_TYPES][_MAX_SPIRV_RES];
    SpirvResource *validated_resource_list_storage[_MAX_SHADER_TYPES][_MAX_SPIRV_RES];
    GLuint validated_resource_list_counts[_MAX_SHADER_TYPES][_MAX_SPIRV_RES];
    /* Process-unique lifetime ID and per-link generation used by the
     * renderer's bounded MSL texture type cache. */
    uint64_t msl_texture_cache_instance_id;
    uint64_t msl_texture_cache_generation;
    GLboolean program_separable;
    BufferBaseTarget plain_uniform_buffers[MAX_BINDABLE_BUFFERS];
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
    /* Built-in variables exposed as active PROGRAM_INPUT / PROGRAM_OUTPUT
     * resources (gl_VertexID, gl_InstanceID, gl_FragDepth, gl_SampleMask, etc.).
     * Stored per-stage so that separate (single-stage) programs can expose
     * their own stage's built-ins.  Kept separate from the main
     * STAGE_INPUT/STAGE_OUTPUT lists so the rendering code (vertex descriptor
     * setup, varyings linking) is unaffected. */
    SpirvResource builtin_program_inputs[_MAX_SHADER_TYPES][16];
    GLuint builtin_program_input_count[_MAX_SHADER_TYPES];
    SpirvResource builtin_program_outputs[_MAX_SHADER_TYPES][16];
    GLuint builtin_program_output_count[_MAX_SHADER_TYPES];
    /* Cached buffer binding plan (per-stage).  NULL until first build.
     * See mgl_buffer_plan.h for the lifecycle and cache contract. */
    MGLBufferBindingPlan *buffer_binding_plan;
    void *mtl_data;
} Program;

GLint mglProgramActiveUniformCount(Program *program);
GLint mglProgramActiveUniformMaxNameLength(Program *program);
SpirvResource *mglProgramActiveUniformAt(Program *program, GLuint index, int *stage_out, int *res_type_out);
GLint mglProgramActiveUniformIndexByName(Program *program, const GLchar *name);
GLint mglProgramActiveUniformGLType(const SpirvResource *res, int res_type);
GLint mglProgramActiveUniformSize(const SpirvResource *res, int res_type);
GLsizei mglProgramActiveUniformNameLength(const SpirvResource *res);
GLint mglProgramActiveUniformBlockIndex(Program *program, const SpirvResource *res);
void mglProgramCopyActiveUniformName(const SpirvResource *res, GLsizei bufSize, GLsizei *length, GLchar *name);
GLboolean mglProgramPointerUsableForName(GLMContext ctx, Program *program, GLuint expectedName);
void mglRetainProgramReference(GLMContext ctx, Program *program);
void mglReleaseProgramReference(GLMContext ctx, Program *program);

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
