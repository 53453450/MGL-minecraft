/*
 * test_mglair.mm
 * M1 end-to-end gate: GLSL -> mglShaderCompileGLSL (AST->AIR->metallib)
 * -> newLibraryWithData -> MTLRenderPipelineState.
 *
 * Exit 0 = PSO_OK, 1 = any stage failed.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>

#include "mgl_shader_abi.h"

static const char *kVS =
    "#version 460 core\n"
    "uniform mat4 mvp;\n"
    "in vec3 inPos;\n"
    "out vec2 vUV;\n"
    "void main() {\n"
    "    vUV = inPos.xy;\n"
    "    gl_Position = mvp * vec4(inPos, 1.0);\n"
    "}\n";

static const char *kFS =
    "#version 460 core\n"
    "in vec2 vUV;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "    fragColor = vec4(vUV, 0.0, 1.0);\n"
    "}\n";

static id<MTLLibrary> loadLibrary(id<MTLDevice> dev, const unsigned char *bytes,
                                  size_t size, const char *tag) {
    NSError *err = nil;
    dispatch_data_t data = dispatch_data_create(bytes, size, NULL,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> lib = [dev newLibraryWithData:data error:&err];
    if (!lib) {
        fprintf(stderr, "%s: newLibraryWithData FAIL: %s\n", tag,
                err.localizedDescription.UTF8String ?: "?");
        return nil;
    }
    return lib;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }

        unsigned char *vsBytes = NULL, *fsBytes = NULL;
        size_t vsSize = 0, fsSize = 0;
        char err[512];

        if (mglShaderCompileGLSL(kVS, MGL_STAGE_VERTEX, &vsBytes, &vsSize,
                                 err, sizeof err) != 0) {
            fprintf(stderr, "vs compile FAIL: %s\n", err);
            return 1;
        }
        if (mglShaderCompileGLSL(kFS, MGL_STAGE_FRAGMENT, &fsBytes, &fsSize,
                                 err, sizeof err) != 0) {
            fprintf(stderr, "fs compile FAIL: %s\n", err);
            mglShaderFree(vsBytes);
            return 1;
        }
        printf("metallib sizes: vs=%zu fs=%zu\n", vsSize, fsSize);

        id<MTLLibrary> vsLib = loadLibrary(dev, vsBytes, vsSize, "vs");
        id<MTLLibrary> fsLib = loadLibrary(dev, fsBytes, fsSize, "fs");
        mglShaderFree(vsBytes);
        mglShaderFree(fsBytes);
        if (!vsLib || !fsLib) {
            return 1;
        }

        id<MTLFunction> vsFn = [vsLib newFunctionWithName:@"main"];
        id<MTLFunction> fsFn = [fsLib newFunctionWithName:@"main"];
        if (!vsFn || !fsFn) {
            fprintf(stderr, "newFunctionWithName FAIL (vs=%d fs=%d)\n",
                    vsFn != nil, fsFn != nil);
            return 1;
        }

        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = vsFn;
        pd.fragmentFunction = fsFn;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

        MTLVertexDescriptor *vd = [MTLVertexDescriptor new];
        vd.attributes[0].format = MTLVertexFormatFloat3;
        vd.attributes[0].offset = 0;
        vd.attributes[0].bufferIndex = 0;
        vd.layouts[0].stride = 12;
        pd.vertexDescriptor = vd;

        NSError *perr = nil;
        id<MTLRenderPipelineState> pso =
            [dev newRenderPipelineStateWithDescriptor:pd error:&perr];
        if (!pso) {
            fprintf(stderr, "PSO_FAIL: %s\n",
                    perr.localizedDescription.UTF8String ?: "?");
            return 1;
        }
        printf("PSO_OK\n");
        return 0;
    }
}
