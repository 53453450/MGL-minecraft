/* SPDX-License-Identifier: LGPL-3.0-only
 * Self-developed domain generation and AIR consumption of the same stream.
 * Specification goldens are tested separately by test_tess_domain.c. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "mgl_air_tess_abi.h"
#include "mgl_tess_domain.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int checkStream(id<MTLDevice> device, id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> pipeline,
                       const MGLTessFactorInput &in)
{
    const uint32_t count = mglTessDomainVertexCount(&in);
    const uint32_t base = 2;
    const size_t stride = MGL_AIR_PER_VERTEX_STRIDE;
    unsigned char factors[2 * MGL_AIR_TESS_FACTOR_RECORD_BYTES] = {};
    for (unsigned p = 0; p < 2; p++) {
        for (unsigned i = 0; i < 6; i++) {
            const float value = i < 4 ? in.outer[i] : in.inner[i - 4];
            const __fp16 half = (__fp16)value;
            unsigned char *record = factors + p * MGL_AIR_TESS_FACTOR_RECORD_BYTES;
            memcpy(record + i * 2, &half, sizeof half);
            memcpy(record + MGL_AIR_TESS_FACTOR_EXACT_FLOAT_OFFSET + i * 4,
                   &value, sizeof value);
        }
    }
    id<MTLBuffer> factorBuffer = [device newBufferWithBytes:factors
        length:sizeof factors options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:(base + count + 1) * stride
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> xfb = [device newBufferWithLength:output.length
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> scratch = [device newBufferWithLength:1024
        options:MTLResourceStorageModeShared];
    if (!factorBuffer || !output || !xfb || !scratch) return 1;
    memset(output.contents, 0xa5, output.length);
    memset(xfb.contents, 0xa5, xfb.length);
    memset(scratch.contents, 0, scratch.length);
    std::vector<MGLTessCoord> expectedPoints(count);
    if (count && mglTessGenerateDomain(&in, expectedPoints.data(), count) != count) return 1;
    for (uint32_t i = 0; i < count; i++)
        memcpy((char *)output.contents + (base + i) * stride, &expectedPoints[i], sizeof(MGLTessCoord));
    const uint32_t contract[4] = {1, 1, count, base};
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:scratch offset:0 atIndex:MGL_AIR_TESS_SLOT_TCS_STAGE_IN];
    [encoder setBuffer:factorBuffer offset:0 atIndex:MGL_AIR_TESS_SLOT_TESS_FACTOR];
    [encoder setBuffer:scratch offset:0 atIndex:MGL_AIR_TESS_SLOT_PATCH_OUT];
    [encoder setBuffer:output offset:0 atIndex:MGL_AIR_TESS_SLOT_TCS_OUTPUT];
    [encoder setBytes:contract length:sizeof contract atIndex:MGL_AIR_TESS_SLOT_INDIRECT];
    [encoder setBuffer:scratch offset:0 atIndex:MGL_AIR_TESS_SLOT_GATHER_INDEX];
    [encoder setBuffer:scratch offset:0 atIndex:MGL_AIR_TESS_SLOT_GATHER_PARAMS];
    [encoder setBuffer:xfb offset:0 atIndex:MGL_AIR_TESS_SLOT_XFB_OUT];
    // Extra threads exercise the item-count guard, including discarded patches.
    [encoder dispatchThreads:MTLSizeMake(count + 7, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "TES execution failed: %s\n",
                command.error.localizedDescription.UTF8String);
        return 1;
    }
    for (uint32_t i = 0; i < count; i++) {
        const MGLTessCoord expected = expectedPoints[i];
        const float *actual = (const float *)((const char *)output.contents + (base + i) * stride);
        if (fabsf(actual[0] - expected.u) > 1e-5f ||
            fabsf(actual[1] - expected.v) > 1e-5f ||
            fabsf(actual[2] - expected.w) > 1e-5f || actual[3] != 1.f ||
            !std::isfinite(actual[0]) || !std::isfinite(actual[1]) || !std::isfinite(actual[2])) {
            fprintf(stderr, "TES mismatch mode=%u spacing=%u point=%u winding=%u index=%u: "
                    "GPU=(%g,%g,%g) CPU=(%g,%g,%g)\n", in.gen_mode, in.spacing,
                    in.point_mode, in.winding, i, actual[0], actual[1], actual[2],
                    expected.u, expected.v, expected.w);
            return 1;
        }
        if (memcmp(actual, (const char *)xfb.contents + (base + i) * stride, 16)) return 1;
    }
    for (id<MTLBuffer> buffer in @[output, xfb]) {
        const unsigned char *bytes = (const unsigned char *)buffer.contents;
        for (size_t i = 0; i < buffer.length; i++) {
            if ((i < base * stride || i >= (base + count) * stride) && bytes[i] != 0xa5) {
                fprintf(stderr, "TES wrote outside patch span\n");
                return 1;
            }
        }
    }
    return 0;
}

int main(void)
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device\n"); return 2; }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const uint32_t modes[] = {GL_TRIANGLES, GL_QUADS, GL_ISOLINES};
        const char *modeNames[] = {"triangles", "quads", "isolines"};
        const uint32_t spacings[] = {GL_EQUAL, GL_FRACTIONAL_ODD, GL_FRACTIONAL_EVEN};
        const char *spacingNames[] = {"equal_spacing", "fractional_odd_spacing", "fractional_even_spacing"};
        unsigned checks = 0;
        for (unsigned m = 0; m < 3; m++)
        for (unsigned s = 0; s < 3; s++)
        for (unsigned point = 0; point < 2; point++)
        for (unsigned cw = 0; cw < 2; cw++) {
            char source[512], error[2048] = {};
            snprintf(source, sizeof source,
                "#version 460 core\nlayout(%s,%s,%s%s) in;\n"
                "void main(){gl_Position=vec4(gl_TessCoord,1.0);}\n",
                modeNames[m], spacingNames[s], cw ? "cw" : "ccw", point ? ",point_mode" : "");
            unsigned char *bytes = nullptr;
            size_t size = 0;
            if (mglAirCompileGLSLWithReflectInfoEx(source, MGL_STAGE_TESS_EVALUATION,
                nullptr, &bytes, &size, nullptr, nullptr, MGL_AIR_COMPILE_FORCE_TES_COMPUTE,
                nullptr, error, sizeof error, nullptr) != 0) {
                fprintf(stderr, "TES compile failed: %s\n", error); return 1;
            }
            dispatch_data_t data = dispatch_data_create(bytes, size, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            NSError *metalError = nil;
            id<MTLLibrary> library = [device newLibraryWithData:data error:&metalError];
            mglShaderFree(bytes);
            if (!library) { fprintf(stderr, "TES library: %s\n", metalError.localizedDescription.UTF8String); return 1; }
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
                [library newFunctionWithName:@"main"] error:&metalError];
            if (!pipeline) { fprintf(stderr, "TES pipeline: %s\n", metalError.localizedDescription.UTF8String); return 1; }
            for (unsigned vector = 0; vector < 5; vector++) {
                MGLTessFactorInput in = {{1,1,1,1}, {1,1}, modes[m], spacings[s],
                    (uint32_t)(cw ? GL_CW : GL_CCW), point};
                if (vector == 1) {
                    for (float &v : in.outer) v = 2.5f;
                    in.inner[0] = 2.5f; in.inner[1] = 3.5f;
                } else if (vector == 2) {
                    for (float &v : in.outer) v = 4.f;
                    in.inner[0] = -1.f;
                } else if (vector == 3) {
                    in.outer[0] = INFINITY;
                    in.inner[0] = INFINITY;
                } else if (vector == 4) {
                    in.outer[0] = NAN;
                }
                if (checkStream(device, queue, pipeline, in)) return 1;
                checks++;
            }
        }
        printf("test_tess_air: %u CPU/AIR stream comparisons passed\n", checks);
    }
    return 0;
}
