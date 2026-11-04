/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * clearStageCompileState harness (audit P6).
 *
 * Relinking a program (and discarding a saved executable) clears each stage's
 * compile state through clearStageCompileState.  It used to leave the TES
 * compute kernel behind -- metallib_bytes_tes_compute, its size and
 * mtl_function_compute -- so a relinked program could keep a function built
 * from the previous link (mgl_render_binding.cpp reloads it only when the
 * function is NULL).  Everything a stage owns must be gone afterwards.
 *
 * Build: see `make test-clear-stage-compile-state`.
 */

#include "mgl_program_reflection.h"

#include <CoreFoundation/CoreFoundation.h>
#include <stdio.h>
#include <stdlib.h>
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

static void *retained_object(void)
{
    return (void *)CFStringCreateWithCString(NULL, "metal-object",
                                             kCFStringEncodingUTF8);
}

static unsigned char *bytes_of(size_t n)
{
    unsigned char *b = (unsigned char *)malloc(n);
    memset(b, 0xAB, n);
    return b;
}

int main(void)
{
    printf("clearStageCompileState harness (audit P6)\n");

    static Program program;
    memset(&program, 0, sizeof(program));
    const int stage = _TESS_EVALUATION_SHADER;
    MGLShaderModule *identity = &program.modules[stage];
    struct MGLShaderModuleBackend *m = mglProgramModuleBackend(&program, stage);

    m->metallib_bytes = bytes_of(16);
    m->metallib_size = 16;
    m->metallib_tess_capture_bytes = bytes_of(8);
    m->metallib_tess_capture_size = 8;
    m->metallib_cull_capture_bytes = bytes_of(8);
    m->metallib_cull_capture_size = 8;
    m->metallib_bytes_tes_compute = bytes_of(32);
    m->metallib_size_tes_compute = 32;
    identity->entry_point = strdup("main");
    m->needs_runtime_array_size_buffer = GL_TRUE;
    m->mtl_function = retained_object();
    m->mtl_library = retained_object();
    m->mtl_tess_capture_function = retained_object();
    m->mtl_tess_capture_library = retained_object();
    m->mtl_cull_capture_function = retained_object();
    m->mtl_cull_capture_library = retained_object();
    m->mtl_function_compute = retained_object();

    clearStageCompileState(&program, stage);

    CHECK(m->metallib_bytes == NULL && m->metallib_size == 0,
          "the main metallib is gone");
    CHECK(m->metallib_tess_capture_bytes == NULL &&
              m->metallib_tess_capture_size == 0 &&
              m->metallib_cull_capture_bytes == NULL &&
              m->metallib_cull_capture_size == 0,
          "the capture variants are gone");
    CHECK(m->metallib_bytes_tes_compute == NULL &&
              m->metallib_size_tes_compute == 0,
          "the TES compute metallib and its size are gone");
    CHECK(m->mtl_function_compute == NULL,
          "the TES compute function is released and cleared");
    CHECK(m->mtl_function == NULL && m->mtl_library == NULL &&
              m->mtl_tess_capture_function == NULL &&
              m->mtl_tess_capture_library == NULL &&
              m->mtl_cull_capture_function == NULL &&
              m->mtl_cull_capture_library == NULL,
          "every other Metal handle is cleared");
    CHECK(identity->entry_point == NULL && !m->needs_runtime_array_size_buffer,
          "entry point and the runtime-array flag are reset");

    CHECK(mglProgramModuleView(&program, stage)->metallib_bytes == NULL &&
              mglProgramModuleView(NULL, stage)->metallib_bytes == NULL &&
              mglProgramModuleView(&program, -1)->metallib_bytes == NULL,
          "a view of a cleared, missing or out-of-range stage reads as zero");

    clearStageCompileState(&program, stage);
    CHECK(1, "clearing an already-clear stage is harmless");
    clearStageCompileState(NULL, stage);
    clearStageCompileState(&program, -1);
    clearStageCompileState(&program, _MAX_SHADER_TYPES);
    CHECK(1, "a NULL program or out-of-range stage is ignored");

    m->metallib_bytes = bytes_of(16);
    m->metallib_size = 16;
    m->mtl_function = retained_object();
    program.backend->pipeline_cache_instance_id = 7u;
    mglProgramBackendRelease(&program);
    CHECK(program.backend == NULL &&
              mglProgramBackendView(&program)->pipeline_cache_instance_id == 0u,
          "releasing the backend frees it and its state reads as zero");
    mglProgramBackendRelease(&program);
    mglProgramBackendRelease(NULL);
    CHECK(1, "releasing again, or a NULL program, is harmless");

    printf("test_clear_stage_compile_state: %d/%d passed\n", tests_passed,
           tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
