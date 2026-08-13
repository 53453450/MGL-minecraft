/*
 * mgl_buffer_plan.h
 * MGL
 *
 * Buffer Binding Plan Subsystem: pre-computes and caches the static
 * reflection-derived data needed by mapGLBuffersToMTLBufferMap each draw,
 * eliminating per-draw name lookups, program resolution, and MSL argument
 * scans for resources that do not change between link and relink.
 *
 * The plan is built once at the end of a successful mglLinkProgram and stored
 * on the Program.  It is invalidated (set invalid) at the start of every
 * mglLinkProgram and whenever glUniformBlockBinding / glShaderStorageBlock
 * Binding mutates a resource's gl_binding (which feeds the cached
 * client_binding_base).  The next draw rebuilds lazily if needed; the common
 * case is: link once, draw thousands of times against the cached plan.
 *
 * Cached per (stage, resource_type, resource_index):
 *   - metal_binding_base  (mglMetalResourceSlot, == res->binding)
 *   - client_binding_base  (mglClientBufferBindingForResource, incl. the
 *                           Minecraft plain-uniform name table lookup)
 *   - element_count        (mglStageBufferResourceElementCount)
 *   - required_size         (getProgramBindingRequiredSize)
 *   - skip_resource         (mglShouldSkipStageBufferResource)
 *   - is_struct_packed      (plain uniform struct packing path)
 *   - allow_global_fallback (mglPlainUniformAllowsGlobalFallback)
 *   - ubo_array_uses_bindings_table (res->ubo_array_bindings != NULL)
 *   - struct member metadata (for the struct packing path)
 *
 * NOT cached (dynamic per-draw):
 *   - Buffer * looked up from ctx->active_state->buffer_base[...].buffers
 *   - offset / size from BufferBaseTarget
 *   - packed struct contents (memcpy'd from plain_uniform_buffers per draw)
 *   - ubo_array_bindings[element] when the table is present (read live so
 *     glUniformBlockBinding does not need to invalidate the plan)
 *
 * All functions here are pure C and have no dependency on the renderer
 * instance, command buffer, or encoder.  The lookup helpers are safe to
 * call from any draw-side path.
 */

#ifndef MGL_BUFFER_PLAN_H
#define MGL_BUFFER_PLAN_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Plan entry flags                                                    */
/* ------------------------------------------------------------------ */
#define MGL_BP_FLAG_SKIP              0x01u  /* mglShouldSkipStageBufferResource */
#define MGL_BP_FLAG_STRUCT_PACKED     0x02u  /* plain uniform struct packing path */
#define MGL_BP_FLAG_ALLOW_FALLBACK    0x04u  /* mglPlainUniformAllowsGlobalFallback */
#define MGL_BP_FLAG_UBO_ARRAY_TABLE   0x08u  /* res->ubo_array_bindings != NULL; read live */

/* ------------------------------------------------------------------ */
/* Struct member metadata (only populated when MGL_BP_FLAG_STRUCT_PACKED) */
/* ------------------------------------------------------------------ */
typedef struct MGLBufferPlanStructMember {
    GLuint member_loc_off;           /* member->location_offset (absolute)  */
    GLuint member_offset_in_elem;    /* member->offset - elem_byte_start     */
    GLuint member_size;              /* member->size (array elem count, 1 scalar) */
    GLuint member_array_stride;      /* member->array_stride or derived      */
    GLint  member_loc;              /* base_loc + member_loc_off            */
    GLboolean is_array_member;       /* member->size > 1                     */
} MGLBufferPlanStructMember;

/* ------------------------------------------------------------------ */
/* Per-resource plan entry                                             */
/* ------------------------------------------------------------------ */
typedef struct MGLBufferPlanEntry {
    /* Resource identity */
    uint16_t resource_type;          /* MGL resource type (UBO/UNIFORM_CONSTANT/SSBO/ATOMIC_COUNTER) */
    uint16_t resource_index;          /* index in shader_resources_list[stage][type].list */
    uint16_t element_count;           /* mglStageBufferResourceElementCount */
    uint16_t struct_member_count;     /* 0 unless MGL_BP_FLAG_STRUCT_PACKED */
    uint32_t flags;                   /* MGL_BP_FLAG_* bitmask */
    uint32_t required_size;           /* getProgramBindingRequiredSize */
    uint32_t metal_binding_base;      /* mglMetalResourceSlot == res->binding */
    uint32_t client_binding_base;     /* mglClientBufferBindingForResource (resolved) */
    /* Struct packing metadata (valid when MGL_BP_FLAG_STRUCT_PACKED) */
    int32_t  base_loc;               /* resource->uniform_location or ->location */
    uint32_t loc_step;               /* mglPlainStructLocStep(resource) */
    uint32_t struct_size;            /* == required_size, cached for clarity */
    /* Struct members (allocated if struct_member_count > 0, else NULL) */
    MGLBufferPlanStructMember *struct_members;
} MGLBufferPlanEntry;

/* ------------------------------------------------------------------ */
/* Per-stage plan                                                      */
/* ------------------------------------------------------------------ */
typedef struct MGLStageBufferPlan {
    GLboolean valid;                 /* GL_FALSE until built; checked at draw */
    uint32_t  entry_count;           /* number of entries across all 4 mapped types */
    MGLBufferPlanEntry *entries;     /* allocated array, entry_count entries */
} MGLStageBufferPlan;

/* ------------------------------------------------------------------ */
/* Whole-program plan (one per Program)                                */
/* ------------------------------------------------------------------ */
typedef struct MGLBufferBindingPlan {
    MGLStageBufferPlan stages[_MAX_SHADER_TYPES];
} MGLBufferBindingPlan;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

/* Build the plan for `program` from its current shader_resources_list.
 * Frees any previously built plan first.  No-op if program is NULL.
 * Safe to call at the end of mglLinkProgram. */
void mglBufferBindingPlanBuild(Program *program);

/* Mark all stages invalid without freeing storage (used by
 * glUniformBlockBinding / glShaderStorageBlockBinding — the next draw
 * will rebuild lazily via mglBufferBindingPlanEnsureBuilt).  No-op if
 * program or plan is NULL. */
void mglBufferBindingPlanInvalidate(Program *program);

/* Free all storage owned by the plan and NULL the pointer on the program.
 * Used by mglFreeProgram.  No-op if program is NULL or plan is NULL. */
void mglBufferBindingPlanDestroy(Program *program);

/* If the plan is invalid, rebuild it.  Returns the plan pointer (which may
 * be NULL if program is NULL or has no resources).  Called at draw time. */
const MGLBufferBindingPlan *mglBufferBindingPlanEnsureBuilt(Program *program);

/* ------------------------------------------------------------------ */
/* Lookup helpers (draw-side, read-only)                               */
/* ------------------------------------------------------------------ */

/* Returns the stage plan, or NULL if plan/stage is out of range. */
const MGLStageBufferPlan *mglStageBufferPlan(const MGLBufferBindingPlan *plan,
                                             int stage);

/* Resolve the client buffer binding for `element` of `entry`.
 * For UBO arrays with a bindings table (MGL_BP_FLAG_UBO_ARRAY_TABLE),
 * this reads `resource->ubo_array_bindings[element]` live so that
 * glUniformBlockBinding mutations are visible without plan invalidation.
 * For all other cases, returns client_binding_base + element. */
GLuint mglBufferPlanClientBindingForElement(const MGLBufferPlanEntry *entry,
                                            const MGLShaderResource *resource,
                                            GLuint element);

/* Resolve the Metal argument slot for `element` of `entry`.
 * Always metal_binding_base + element. */
static inline GLuint mglBufferPlanMetalBindingForElement(const MGLBufferPlanEntry *entry,
                                                         GLuint element)
{
    return entry ? (entry->metal_binding_base + element) : 0u;
}

/* ------------------------------------------------------------------ */
/* Static inline helpers (shared between plan build and draw path)     */
/* ------------------------------------------------------------------ */

/* Compute the uniform-location step between array elements of a plain
 * struct uniform resource (mirrors the logic in mapGLBuffersToMTLBufferMap).
 * Returns 1 when the resource has no UBO members. */
static inline GLuint mglPlainStructLocStep(const MGLShaderResource *res)
{
    if (!res || !res->ubo_members || res->ubo_member_count == 0) {
        return 1;
    }
    GLuint max_loc = 0;
    for (GLuint i = 0; i < res->ubo_member_count; i++) {
        GLuint end = (GLuint)res->ubo_members[i].location_offset +
                     (GLuint)res->ubo_members[i].size;
        if (end > max_loc) {
            max_loc = end;
        }
    }
    GLuint array_size = (res->gl_array_size > 1) ? (GLuint)res->gl_array_size : 1;
    if (array_size == 0) array_size = 1;
    GLuint step = max_loc / array_size;
    return step > 0 ? step : 1;
}

/* Byte size of one element of a GL uniform type.  Used as a fallback array
 * stride for plain struct uniform members lacking ArrayStride decorations. */
static inline GLuint mglGLTypeElementByteSize(GLuint gl_type)
{
    switch (gl_type) {
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_BOOL:
            return 4;
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
        case GL_BOOL_VEC2:
            return 8;
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
        case GL_BOOL_VEC3:
            return 12;
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
        case GL_BOOL_VEC4:
            return 16;
        case GL_FLOAT_MAT2:
            return 8;   /* one column = vec2 */
        case GL_FLOAT_MAT3:
            return 12;  /* one column = vec3 */
        case GL_FLOAT_MAT4:
            return 16;  /* one column = vec4 */
        case GL_FLOAT_MAT2x3:
            return 12;
        case GL_FLOAT_MAT2x4:
            return 16;
        case GL_FLOAT_MAT3x2:
            return 8;
        case GL_FLOAT_MAT3x4:
            return 16;
        case GL_FLOAT_MAT4x2:
            return 8;
        case GL_FLOAT_MAT4x3:
            return 12;
        case GL_DOUBLE:
            return 8;
        default:
            return 4;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* MGL_BUFFER_PLAN_H */
