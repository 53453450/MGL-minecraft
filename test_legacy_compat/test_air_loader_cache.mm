/* SPDX-License-Identifier: LGPL-3.0-only */
/* LRU bound of the AIR loader's render-pipeline cache (mgl_air_loader.cpp). */

#import <Metal/Metal.h>

#include <stdio.h>
#include <string.h>

#include "mgl_air_loader.h"

static const uint64_t kLimit = 1024;

static void *createPipeline(id<MTLDevice> device, id<MTLFunction> vs,
                            id<MTLFunction> fs, uint64_t instance)
{
    MGLRenderPipelineDescriptorState desc;
    memset(&desc, 0, sizeof(desc));
    desc.vertex_program_instance = instance;
    desc.color_count = 1;
    desc.color_format[0] = MTLPixelFormatRGBA8Unorm;
    desc.color_write_mask[0] = MTLColorWriteMaskAll;
    desc.rasterization_enabled = 1;
    desc.raster_sample_count = 1;
    void *pso = NULL;
    char err[256] = {0};
    if (mglAirCreateRenderPipeline((__bridge void *)device, (__bridge void *)vs,
                                   (__bridge void *)fs, &desc, &pso, err,
                                   sizeof(err)) != 0 || !pso) {
        fprintf(stderr, "FAIL: create pipeline %llu: %s\n",
                (unsigned long long)instance, err);
        return NULL;
    }
    return pso;
}

int main(void)
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("SKIP: no Metal device\n");
            return 0;
        }
        NSError *error = nil;
        id<MTLLibrary> library = [device
            newLibraryWithSource:
                @"#include <metal_stdlib>\n"
                 "using namespace metal;\n"
                 "vertex float4 v(uint vid [[vertex_id]]) { return float4(0.0, 0.0, 0.0, 1.0); }\n"
                 "fragment float4 f() { return float4(0.0); }\n"
                         options:nil
                           error:&error];
        id<MTLFunction> vs = [library newFunctionWithName:@"v"];
        id<MTLFunction> fs = [library newFunctionWithName:@"f"];
        if (!vs || !fs) {
            fprintf(stderr, "FAIL: shader library: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }

        mglAirLoaderShutdown();
        /* Keep references to the first two so their addresses stay unique. */
        void *first = createPipeline(device, vs, fs, 0);
        void *second = createPipeline(device, vs, fs, 1);
        if (!first || !second) return 1;
        void *again = createPipeline(device, vs, fs, 0);
        if (again != first) {
            fprintf(stderr, "FAIL: same key missed the cache\n");
            return 1;
        }
        mglAirRelease(again);

        for (uint64_t i = 2; i < kLimit; ++i) {
            void *pso = createPipeline(device, vs, fs, i);
            if (!pso) return 1;
            mglAirRelease(pso);
        }
        again = createPipeline(device, vs, fs, 0);
        if (again != first) {
            fprintf(stderr, "FAIL: full cache lost a resident key\n");
            return 1;
        }
        mglAirRelease(again);

        void *overflow = createPipeline(device, vs, fs, kLimit);
        if (!overflow) return 1;
        mglAirRelease(overflow);

        int result = 0;
        again = createPipeline(device, vs, fs, 0);
        if (again != first) {
            fprintf(stderr, "FAIL: recently used key was evicted\n");
            result = 1;
        }
        mglAirRelease(again);
        again = createPipeline(device, vs, fs, 1);
        if (again == second) {
            fprintf(stderr, "FAIL: least recently used key survived overflow\n");
            result = 1;
        }
        mglAirRelease(again);

        mglAirRelease(first);
        mglAirRelease(second);
        mglAirLoaderShutdown();
        if (result == 0) printf("AIR_LOADER_CACHE_OK\n");
        return result;
    }
}
