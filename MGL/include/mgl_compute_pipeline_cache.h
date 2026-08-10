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
