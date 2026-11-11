/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Reserved buffer slot policy harness (audit P4).
 *
 * mglBufferSlotConflicts() decides from plain facts which Metal buffer slots a
 * stage's internal buffers occupy; mglBufferSlotFactsFromProgram() is the only
 * place that reads the Program.  reference_conflicts() is the policy as it was
 * when it read the Program directly, and the sweep requires the two to agree
 * for every stage, flag combination and slot.  The link check
 * (mglBufferSlotCheckResource) is exercised on top.
 *
 * Build: see `make test-buffer-slots`.
 */

#include "mgl_buffer_slots.h"
#include "glm_context.h"

#include <stdio.h>
#include <string.h>
#include "mgl_program_backend.h"

static int tests_run;
static int tests_passed;

#define CHECK(cond, label)                                       \
    do {                                                         \
        tests_run++;                                             \
        if (cond) {                                              \
            tests_passed++;                                      \
        } else {                                                 \
            printf("  [FAIL] %s\n", (label));                    \
        }                                                        \
    } while (0)

/* The retired mglBufferSlotConflictsForProgram, reading the Program. */
static GLboolean reference_conflicts(const Program *program, GLuint slot,
                                     int stage)
{
    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return GL_FALSE;
    }
    if (slot == mglRuntimeArraySizeBufferIndexForProgram(program, stage) &&
        mglProgramModuleView(program, stage)->needs_runtime_array_size_buffer) {
        return GL_TRUE;
    }
    if (slot == mglProgramModuleView(program, stage)->sparse_residency_buffer_index &&
        mglProgramModuleView(program, stage)->needs_sparse_residency) {
        return GL_TRUE;
    }
    if (stage == _VERTEX_SHADER || stage == _FRAGMENT_SHADER) {
        if (slot == kMGLSamplerWrapBufferIndex) {
            return GL_TRUE;
        }
        if (slot == kMGLLodBiasMaxBufferIndex &&
            mglProgramStageHasSamplers(program, stage)) {
            return GL_TRUE;
        }
    }
    switch (stage) {
    case _VERTEX_SHADER:
        if (program->uses_point_size_params &&
            slot == kMGLPointSizeBufferIndex) {
            return GL_TRUE;
        }
        if (program->uses_shader_draw_params &&
            slot == kMGLShaderDrawParamsBufferIndex) {
            return GL_TRUE;
        }
        if (program->uses_cull_distance &&
            mglBufferSlotIsReservedForCullDistance(slot)) {
            return GL_TRUE;
        }
        if (slot >= kMGLVertexAttribBufferBase &&
            program->shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES]
                    .count > 0u) {
            return GL_TRUE;
        }
        break;
    case _TESS_CONTROL_SHADER:
        if (slot == 24u || (slot >= 26u && slot <= 29u)) {
            return GL_TRUE;
        }
        break;
    case _TESS_EVALUATION_SHADER:
        if (program->tess_eval_render_vertex) {
            if (slot == 23u || (slot >= 26u && slot <= 30u)) {
                return GL_TRUE;
            }
        } else if (program->tess_eval_compute) {
            if (slot == 22u || (slot >= 24u && slot <= 30u)) {
                return GL_TRUE;
            }
        } else if (slot == 27u || slot == 28u || slot == 30u) {
            return GL_TRUE;
        }
        break;
    case _GEOMETRY_SHADER:
        if (program->gs_route == MGL_GS_ROUTE_COMPUTE &&
            mglBufferSlotIsReservedForGeometry(slot)) {
            return GL_TRUE;
        }
        break;
    case _FRAGMENT_SHADER:
        if (program->usesFragCoordParams &&
            mglBufferSlotIsReservedForFragCoordFixup(slot)) {
            return GL_TRUE;
        }
        break;
    default:
        break;
    }
    return GL_FALSE;
}

static Program g_program;

static void configure(unsigned flags, int stage)
{
    memset(&g_program, 0, sizeof(g_program));
    mglProgramModuleBackend(&g_program, stage)->needs_runtime_array_size_buffer = (flags & 1u) != 0;
    mglProgramModuleBackend(&g_program, stage)->needs_sparse_residency = (flags & 1024u) != 0;
    mglProgramModuleBackend(&g_program, stage)->sparse_residency_buffer_index =
        kMGLSparseResidencyDefaultBufferIndex;
    g_program.uses_point_size_params = (flags & 2u) != 0;
    g_program.uses_shader_draw_params = (flags & 4u) != 0;
    g_program.uses_cull_distance = (flags & 8u) != 0;
    g_program.tess_eval_render_vertex = (flags & 16u) != 0;
    g_program.tess_eval_compute = (flags & 32u) != 0;
    g_program.gs_route = (flags & 64u) ? MGL_GS_ROUTE_COMPUTE : MGL_GS_ROUTE_NONE;
    g_program.usesFragCoordParams = (flags & 128u) != 0;
    g_program.shader_resources_list[_VERTEX_SHADER][_STAGE_INPUT_RES].count =
        (flags & 256u) ? 1u : 0u;
    if (flags & 512u) {
        g_program.shader_resources_list[stage][_SEPARATE_IMAGE_RES].count = 1u;
    }
}

static void test_sweep(void)
{
    int agree = 1;
    long cases = 0;
    for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++) {
        for (unsigned flags = 0; flags < 2048u; flags++) {
            configure(flags, stage);
            MGLBufferSlotFacts facts;
            mglBufferSlotFactsFromProgram(&g_program, stage, &facts);
            for (GLuint slot = 0; slot < 40u; slot++) {
                cases++;
                const GLboolean want = reference_conflicts(&g_program, slot, stage);
                if (mglBufferSlotConflicts(&facts, slot) != want ||
                    mglBufferSlotConflictsForProgram(&g_program, slot, stage) !=
                        want) {
                    agree = 0;
                }
            }
        }
    }
    printf("  swept %ld (stage, flags, slot) cases\n", cases);
    CHECK(agree, "facts-based policy matches the retired Program-reading one");
    CHECK(!mglBufferSlotConflictsForProgram(NULL, 24u, _VERTEX_SHADER) &&
              !mglBufferSlotConflictsForProgram(&g_program, 24u, -1) &&
              !mglBufferSlotConflictsForProgram(&g_program, 24u,
                                                _MAX_SHADER_TYPES),
          "no program or an out-of-range stage conflicts with nothing");
}

static void test_check_resource(void)
{
    MGLBufferSlotFacts facts;
    uint32_t bad = 0u;
    configure(256u, _VERTEX_SHADER); /* a vertex stage with inputs */
    mglBufferSlotFactsFromProgram(&g_program, _VERTEX_SHADER, &facts);

    CHECK(mglBufferSlotCheckResource(&facts, 3u, 4u, &bad) == MGL_BUFFER_SLOT_OK,
          "slots 3..6 are free");
    CHECK(mglBufferSlotCheckResource(&facts, 15u, 3u, &bad) ==
                  MGL_BUFFER_SLOT_RESERVED &&
              bad == 16u,
          "an array running into the attribute region reports the first bad slot");
    CHECK(mglBufferSlotCheckResource(&facts, 29u, 3u, &bad) ==
                  MGL_BUFFER_SLOT_RESERVED &&
              bad == 29u,
          "the attribute region reserves 16..30 when the stage has inputs");
    CHECK(mglBufferSlotCheckResource(&facts, 30u, 2u, &bad) ==
                  MGL_BUFFER_SLOT_RESERVED &&
              bad == 30u,
          "a block array starting in the reserved region stops at its first slot");

    configure(0u, _FRAGMENT_SHADER);
    mglBufferSlotFactsFromProgram(&g_program, _FRAGMENT_SHADER, &facts);
    CHECK(mglBufferSlotCheckResource(&facts, 29u, 2u, &bad) ==
                  MGL_BUFFER_SLOT_OK,
          "slots 29..30 are the last user slots");
    CHECK(mglBufferSlotCheckResource(&facts, 30u, 2u, &bad) ==
                  MGL_BUFFER_SLOT_EXCEEDS_LIMIT &&
              bad == 31u,
          "slot 31 is past the user limit");
    CHECK(mglBufferSlotCheckResource(&facts, 0xffffffffu, 2u, &bad) ==
                  MGL_BUFFER_SLOT_EXCEEDS_LIMIT &&
              bad == 0xffffffffu,
          "a binding near uint32 max is a limit failure, not a wrap");
    CHECK(mglBufferSlotCheckResource(&facts, 0u, 0xffffffffu, &bad) ==
                  MGL_BUFFER_SLOT_RESERVED &&
              bad == 23u,
          "a huge element count stops at the first bad slot (the wrap table)");
    CHECK(mglBufferSlotCheckResource(&facts, 24u, 0xffffffffu, &bad) ==
                  MGL_BUFFER_SLOT_EXCEEDS_LIMIT &&
              bad == 31u,
          "...and past the reserved ones, at the first slot past the limit");
    CHECK(mglBufferSlotCheckResource(&facts, 5u, 0u, &bad) == MGL_BUFFER_SLOT_OK &&
              bad == 5u,
          "a zero element count is checked as one element");
    CHECK(mglBufferSlotCheckResource(&facts, 23u, 1u, &bad) ==
              MGL_BUFFER_SLOT_RESERVED,
          "the sampler wrap table slot is reserved in a fragment stage");

    configure(1024u, _COMPUTE_SHADER);
    mglBufferSlotFactsFromProgram(&g_program, _COMPUTE_SHADER, &facts);
    CHECK(mglBufferSlotCheckResource(&facts, kMGLSparseResidencyDefaultBufferIndex,
                                     1u, &bad) == MGL_BUFFER_SLOT_RESERVED,
          "a sparse-aware compute stage reserves the commit-table slot");
    CHECK(mglBufferSlotCheckResource(&facts, 11u, 1u, &bad) == MGL_BUFFER_SLOT_OK,
          "the slot below the commit table stays available");
}

static void test_sparse_relocation(void)
{
    configure(1024u, _COMPUTE_SHADER);
    MGLShaderResource resource;
    memset(&resource, 0, sizeof(resource));
    resource.binding = kMGLSparseResidencyDefaultBufferIndex;
    g_program.shader_resources_list[_COMPUTE_SHADER][_UNIFORM_BUFFER_RES].list =
        &resource;
    g_program.shader_resources_list[_COMPUTE_SHADER][_UNIFORM_BUFFER_RES].count =
        1u;
    GLuint selected = 99u;
    CHECK(mglBufferSlotFindSparseResidency(
              &g_program, _COMPUTE_SHADER,
              kMGLSparseResidencyDefaultBufferIndex, &selected) &&
              selected == 0u,
          "a user buffer at the default slot relocates the sparse table");
    mglProgramModuleBackend(&g_program, _COMPUTE_SHADER)
        ->sparse_residency_buffer_index = (uint8_t)selected;
    MGLBufferSlotFacts facts;
    mglBufferSlotFactsFromProgram(&g_program, _COMPUTE_SHADER, &facts);
    uint32_t bad = 0u;
    CHECK(mglBufferSlotCheckResource(
              &facts, kMGLSparseResidencyDefaultBufferIndex, 1u, &bad) ==
              MGL_BUFFER_SLOT_OK,
          "the vacated default slot remains legal for the user buffer");
    CHECK(mglBufferSlotCheckResource(&facts, selected, 1u, &bad) ==
              MGL_BUFFER_SLOT_RESERVED,
          "the relocated commit-table slot is reserved instead");

    configure(1025u, _COMPUTE_SHADER);
    CHECK(mglBufferSlotFindSparseResidency(
              &g_program, _COMPUTE_SHADER, MGL_RUNTIME_ARRAY_SIZE_BUFFER_INDEX,
              &selected) &&
              selected == 0u,
          "sparse placement also avoids the runtime-array size table");

    configure(0u, _COMPUTE_SHADER);
    MGLShaderResource resources[kMGLMaxMetalUserBufferCount];
    memset(resources, 0, sizeof(resources));
    for (GLuint i = 0; i < kMGLMaxMetalUserBufferCount; i++) {
        resources[i].binding = i;
    }
    g_program.shader_resources_list[_COMPUTE_SHADER][_UNIFORM_BUFFER_RES].list =
        resources;
    g_program.shader_resources_list[_COMPUTE_SHADER][_UNIFORM_BUFFER_RES].count =
        kMGLMaxMetalUserBufferCount;
    CHECK(!mglBufferSlotFindSparseResidency(
              &g_program, _COMPUTE_SHADER,
              kMGLSparseResidencyDefaultBufferIndex, &selected),
          "a fully occupied stage reports that no commit-table slot exists");
}

int main(void)
{
    printf("reserved buffer slot policy harness (audit P4)\n");
    test_sweep();
    test_check_resource();
    test_sparse_relocation();
    printf("test_buffer_slots: %d/%d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
