#import "mgl_compute_pipeline_cache.h"
#import "glm_context.h"

#include <stdlib.h>
#include <string.h>
#include <strings.h>

static BOOL mglComputePipelineCacheEnabled(void)
{
    const char *value = getenv("MGL_COMPUTE_PSO_CACHE");
    return !value || value[0] == '\0' ||
           !(strcmp(value, "0") == 0 ||
             strcasecmp(value, "false") == 0 ||
             strcasecmp(value, "no") == 0 ||
             strcasecmp(value, "off") == 0);
}

id<MTLComputePipelineState> mglGetOrCreateProgramComputePipeline(
    id<MTLDevice> device,
    Program *program,
    int stage,
    NSError **error)
{
    if (error) {
        *error = nil;
    }
    BOOL validStage = stage == _COMPUTE_SHADER ||
                      stage == _TESS_CONTROL_SHADER ||
                      stage == _TESS_EVALUATION_SHADER;
    if (!device || !program || !validStage || !program->spirv[stage].mtl_function) {
        if (error) {
            *error = [NSError errorWithDomain:@"MGLComputePipelineCache"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey:
                                                    @"Compute pipeline requires a device, Program, and compiled compute stage"}];
        }
        return nil;
    }

    Spirv *spirv = &program->spirv[stage];
    id<MTLFunction> function = (__bridge id<MTLFunction>)spirv->mtl_function;
    BOOL cacheEnabled = mglComputePipelineCacheEnabled();
    if (cacheEnabled && spirv->mtl_compute_pipeline) {
        return (__bridge id<MTLComputePipelineState>)spirv->mtl_compute_pipeline;
    }

    if (!cacheEnabled) {
        return [device newComputePipelineStateWithFunction:function error:error];
    }

    /* Program state is normally protected by the renderer lock. Serializing
     * misses on the device also prevents duplicate synchronous compiles when
     * callers race before entering the renderer. */
    @synchronized (device) {
        if (spirv->mtl_compute_pipeline) {
            return (__bridge id<MTLComputePipelineState>)spirv->mtl_compute_pipeline;
        }

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:error];
        if (!pipeline) {
            return nil;
        }

        spirv->mtl_compute_pipeline = (void *)CFBridgingRetain(pipeline);
        return pipeline;
    }
}
