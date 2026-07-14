#ifndef mgl_compute_pipeline_cache_h
#define mgl_compute_pipeline_cache_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef struct Program_t Program;

/* Returns the default compute pipeline owned by a linked Program stage.
 * MGL_COMPUTE_PSO_CACHE=0 bypasses the Program slot and recreates the PSO. */
id<MTLComputePipelineState> mglGetOrCreateProgramComputePipeline(
    id<MTLDevice> device,
    Program *program,
    int stage,
    NSError **error);

#endif /* mgl_compute_pipeline_cache_h */
