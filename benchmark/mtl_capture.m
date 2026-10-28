/*
 * mtl_capture.m — device-scope Metal GPU trace capture for the harness.
 *
 * Device scope (not command-buffer scope) on purpose: MGL spreads one frame
 * over several command buffers and render passes, and the question this trace
 * answers is what the whole frame looked like to the GPU.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mtl_capture.h"

int mgl_harness_gpu_trace_begin(const char *path)
{
    if (!path || !path[0]) {
        return -1;
    }
    MTLCaptureManager *manager = [MTLCaptureManager sharedCaptureManager];
    if (![manager supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
        fprintf(stderr,
                "harness: GPU trace document destination unsupported "
                "(run with MTL_CAPTURE_ENABLED=1)\n");
        return -1;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "harness: GPU trace needs a Metal device\n");
        return -1;
    }

    MTLCaptureDescriptor *desc = [[MTLCaptureDescriptor alloc] init];
    desc.captureObject = device;
    desc.destination = MTLCaptureDestinationGPUTraceDocument;
    desc.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];

    NSError *error = nil;
    if (![manager startCaptureWithDescriptor:desc error:&error]) {
        fprintf(stderr, "harness: GPU trace capture failed: %s\n",
                error ? [[error localizedDescription] UTF8String] : "(no error)");
        return -1;
    }
    return 0;
}

int mgl_harness_gpu_trace_end(void)
{
    MTLCaptureManager *manager = [MTLCaptureManager sharedCaptureManager];
    if ([manager isCapturing]) {
        [manager stopCapture];
    }
    return 0;
}
