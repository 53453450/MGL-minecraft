/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

#ifndef mgl_compute_pipeline_cache_h
#define mgl_compute_pipeline_cache_h

#include <stddef.h>

typedef struct Program_t Program;

#ifdef __cplusplus
extern "C" {
#endif

/* Returns an independent +1 Metal compute-pipeline reference. The renderer's
 * C++ cache retains its own reference and owns cache synchronization. */
int mglGetOrCreateProgramComputePipeline(Program *program,
                                         int stage,
                                         void **pipeline_out,
                                         char *err,
                                         size_t errcap);

#ifdef __cplusplus
}
#endif

#endif /* mgl_compute_pipeline_cache_h */
