/*
 * mgl_buffer_plan.c
 * MGL
 *
 * Implementation of the Buffer Binding Plan Subsystem.
 * See mgl_buffer_plan.h for the API contract and design rationale.
 */

#include "mgl_buffer_plan.h"

#include "mgl_shader_resource.h"
#include "mgl_sampler_compat.h"
#include "mgl_program_resource.h"

#include <stdlib.h>
#include <string.h>

/* The four shader resource types whose bindings are resolved each draw
 * in mapShaderBufferResourcesToBufferMap.  Matches the mapped_types[]
 * table in that function. */
static const int kMappedSpvcTypes[4] = {
    _UNIFORM_BUFFER_RES,
    _UNIFORM_CONSTANT_RES,
    _STORAGE_BUFFER_RES,
    _ATOMIC_COUNTER_RES
};

/* ------------------------------------------------------------------ */
/* Internal: free a single stage's storage                             */
/* ------------------------------------------------------------------ */
static void mglStageBufferPlanFree(MGLStageBufferPlan *stage_plan)
{
    if (!stage_plan) {
        return;
    }
    if (stage_plan->entries) {
        for (uint32_t i = 0; i < stage_plan->entry_count; i++) {
            if (stage_plan->entries[i].struct_members) {
                free(stage_plan->entries[i].struct_members);
                stage_plan->entries[i].struct_members = NULL;
            }
        }
        free(stage_plan->entries);
        stage_plan->entries = NULL;
    }
    stage_plan->entry_count = 0;
    stage_plan->valid = GL_FALSE;
}

/* ------------------------------------------------------------------ */
/* Internal: count total resources across the 4 mapped types for a stage */
/* ------------------------------------------------------------------ */
static uint32_t mglCountStageEntries(Program *program, int stage)
{
    uint32_t total = 0;
    for (int t = 0; t < 4; t++) {
        int spvc_type = kMappedSpvcTypes[t];
        if (spvc_type < 0 || spvc_type >= MGL_MAX_SHADER_RESOURCES) {
            continue;
        }
        if (stage < 0 || stage >= _MAX_SHADER_TYPES) {
            continue;
        }
        total += program->shader_resources_list[stage][spvc_type].count;
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* Internal: populate one plan entry from a MGLShaderResource              */
/* ------------------------------------------------------------------ */
static void mglBuildPlanEntry(MGLBufferPlanEntry *entry,
                             Program *program,
                             int stage,
                             int spvc_type,
                             GLuint resource_index,
                             const MGLShaderResource *resource)
{
    memset(entry, 0, sizeof(*entry));

    entry->resource_type   = (uint16_t)spvc_type;
    entry->resource_index  = (uint16_t)resource_index;
    entry->element_count   = (uint16_t)mglStageBufferResourceElementCount(spvc_type, resource);
    entry->required_size   = (uint32_t)resource->required_size;
    entry->metal_binding_base = (uint32_t)mglMetalResourceSlot(resource);
    entry->client_binding_base = (uint32_t)mglClientBufferBindingForResource(spvc_type, resource);
    entry->struct_size     = (uint32_t)resource->required_size;

    uint32_t flags = 0u;

    if (mglShouldSkipStageBufferResource(program, stage, spvc_type, resource)) {
        flags |= MGL_BP_FLAG_SKIP;
    }

    /* Detect plain uniform struct packing path (mirrors
     * mapShaderBufferResourcesToBufferMap lines 775-778). */
    GLboolean is_struct_packed = GL_FALSE;
    if (spvc_type == _UNIFORM_CONSTANT_RES &&
        resource->ubo_members && resource->ubo_member_count > 0 &&
        resource->required_size > 0 &&
        !mglRendererResourceLooksSamplerLike(resource, spvc_type)) {
        is_struct_packed = GL_TRUE;
        flags |= MGL_BP_FLAG_STRUCT_PACKED;
    }

    if (spvc_type == _UNIFORM_CONSTANT_RES) {
        if (mglPlainUniformAllowsGlobalFallback(resource)) {
            flags |= MGL_BP_FLAG_ALLOW_FALLBACK;
        }
    }

    /* UBO/SSBO arrays may have a per-element bindings table
     * (ubo_array_bindings).  When present, the draw path reads it live so
     * that glUniformBlockBinding / glShaderStorageBlockBinding mutations
     * are visible without plan invalidation. */
    if (resource->ubo_array_bindings && resource->ubo_array_size > 1u) {
        flags |= MGL_BP_FLAG_UBO_ARRAY_TABLE;
    }

    entry->flags = flags;

    if (is_struct_packed) {
        entry->loc_step = mglPlainStructLocStep(resource);
        GLint base_loc = resource->uniform_location;
        if (base_loc < 0) {
            base_loc = (GLint)resource->location;
        }
        entry->base_loc = base_loc;
        entry->struct_member_count = (uint16_t)resource->ubo_member_count;

        if (resource->ubo_member_count > 0) {
            size_t alloc_sz = (size_t)resource->ubo_member_count *
                              sizeof(MGLBufferPlanStructMember);
            MGLBufferPlanStructMember *members =
                (MGLBufferPlanStructMember *)malloc(alloc_sz);
            if (members) {
                memset(members, 0, alloc_sz);

                for (GLuint m = 0; m < resource->ubo_member_count; m++) {
                    const SpirvUBOMember *src = &resource->ubo_members[m];
                    MGLBufferPlanStructMember *dst = &members[m];

                    dst->member_loc_off = (GLuint)src->location_offset;

                    GLuint member_offset = src->offset;
                    /* For array elements, member->offset is absolute across
                     * the whole array.  The draw path computes the relative
                     * offset per element; store the raw offset and let the
                     * draw path subtract elem_byte_start (matching the
                     * original logic).  For single-element resources
                     * (element_count == 1), elem_byte_start == 0, so the
                     * raw offset is the relative offset. */
                    dst->member_offset_in_elem = member_offset;

                    dst->member_size = (GLuint)src->size;
                    if (src->array_stride > 0) {
                        dst->member_array_stride = (GLuint)src->array_stride;
                    } else {
                        dst->member_array_stride =
                            mglGLTypeElementByteSize(src->gl_type);
                    }
                    dst->member_loc = base_loc + (GLint)src->location_offset;
                    dst->is_array_member = (src->size > 1) ? GL_TRUE : GL_FALSE;
                }
                entry->struct_members = members;
            } else {
                /* Allocation failed — mark as not struct-packed so the
                 * draw path falls back to the original resource-reading
                 * code, which still works (just slower). */
                flags &= ~MGL_BP_FLAG_STRUCT_PACKED;
                entry->flags = flags;
                entry->struct_member_count = 0;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

void mglBufferBindingPlanBuild(Program *program)
{
    if (!program) {
        return;
    }

    /* Free any previously built plan. */
    mglBufferBindingPlanDestroy(program);

    MGLBufferBindingPlan *plan =
        (MGLBufferBindingPlan *)calloc(1, sizeof(MGLBufferBindingPlan));
    if (!plan) {
        return;
    }
    program->buffer_binding_plan = plan;

    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        MGLStageBufferPlan *stage_plan = &plan->stages[stage];

        uint32_t total = mglCountStageEntries(program, stage);
        if (total == 0) {
            stage_plan->valid = GL_TRUE;
            stage_plan->entry_count = 0;
            stage_plan->entries = NULL;
            continue;
        }

        stage_plan->entries = (MGLBufferPlanEntry *)calloc(total,
                                                           sizeof(MGLBufferPlanEntry));
        if (!stage_plan->entries) {
            /* Leave this stage invalid; other stages may still succeed. */
            stage_plan->valid = GL_FALSE;
            stage_plan->entry_count = 0;
            continue;
        }

        uint32_t idx = 0;
        for (int t = 0; t < 4; t++) {
            int spvc_type = kMappedSpvcTypes[t];
            if (spvc_type < 0 || spvc_type >= MGL_MAX_SHADER_RESOURCES) {
                continue;
            }
            MGLShaderResourceList *rl = &program->shader_resources_list[stage][spvc_type];
            for (GLuint i = 0; i < rl->count; i++) {
                if (idx >= total) {
                    break;
                }
                mglBuildPlanEntry(&stage_plan->entries[idx],
                                  program, stage, spvc_type, i,
                                  &rl->list[i]);
                idx++;
            }
        }

        stage_plan->entry_count = idx;
        stage_plan->valid = GL_TRUE;
    }
}

void mglBufferBindingPlanInvalidate(Program *program)
{
    if (!program || !program->buffer_binding_plan) {
        return;
    }
    for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
        program->buffer_binding_plan->stages[s].valid = GL_FALSE;
    }
}

void mglBufferBindingPlanDestroy(Program *program)
{
    if (!program || !program->buffer_binding_plan) {
        return;
    }
    for (int s = 0; s < _MAX_SHADER_TYPES; s++) {
        mglStageBufferPlanFree(&program->buffer_binding_plan->stages[s]);
    }
    free(program->buffer_binding_plan);
    program->buffer_binding_plan = NULL;
}

const MGLBufferBindingPlan *mglBufferBindingPlanEnsureBuilt(Program *program)
{
    if (!program) {
        return NULL;
    }
    if (!program->buffer_binding_plan ||
        !program->buffer_binding_plan->stages[_VERTEX_SHADER].valid) {
        mglBufferBindingPlanBuild(program);
    }
    return program->buffer_binding_plan;
}

const MGLStageBufferPlan *mglStageBufferPlan(const MGLBufferBindingPlan *plan,
                                             int stage)
{
    if (!plan || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return NULL;
    }
    return &plan->stages[stage];
}

GLuint mglBufferPlanClientBindingForElement(const MGLBufferPlanEntry *entry,
                                            const MGLShaderResource *resource,
                                            GLuint element)
{
    if (!entry) {
        return 0u;
    }

    /* For UBO/SSBO arrays with a per-element bindings table, read the
     * live value from the resource so glUniformBlockBinding /
     * glShaderStorageBlockBinding mutations are visible without plan
     * invalidation. */
    if ((entry->flags & MGL_BP_FLAG_UBO_ARRAY_TABLE) &&
        resource && resource->ubo_array_bindings &&
        element < resource->ubo_array_size) {
        return resource->ubo_array_bindings[element];
    }

    return entry->client_binding_base + element;
}
