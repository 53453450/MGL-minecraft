/*
 * test_mglair.mm
 * M1 end-to-end gate: GLSL -> mglShaderCompileGLSL (AST->AIR->metallib)
 * -> newLibraryWithData -> MTLRenderPipelineState, plus a numeric
 * correctness check: render a full-screen triangle with a rotation MVP
 * and compare the readback texture against a CPU reference.
 *
 * Exit 0 = PSO_OK + values match, 1 = any failure.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mgl_shader_abi.h"

static const char *kVS =
    "#version 460 core\n"
    "uniform mat4 mvp;\n"
    "uniform mat3 rot3;\n"
    "in vec3 inPos;\n"
    "out vec2 vUV;\n"
    "out vec3 vN;\n"
    "void main() {\n"
    "    int p = 2;\n"
    "    float o = p * 0.5;\n"        /* int*float -> 1.0 */
    "    vec2 k = vec2(0.5);\n"       /* single-arg broadcast */
    "    bool b = true;\n"            /* bool literal */
    "    ivec2 io = ivec2(2);\n"      /* int vector broadcast */
    "    uvec2 uo = uvec2(uint(3), uint(3));\n"  /* uint ctor + scalar ctor */
    "    bvec2 bo = bvec2(true, false);\n"
    "    mat2 mi = mat2(vec2(1.0, 0.0), vec2(0.0, 1.0));\n"  /* column ctor */
    "    vec2 t = mi * vec2(float(io.x) + float(uo.y) - 5.0,\n"
    "                       float(bo.x) - 1.0);\n"
    "    vec2 c01 = mat2(1.0, 2.0, 3.0, 4.0)[1];\n"  /* scalar list, col 1 */
    "    t = t + vec2(mi[0].x - 1.0, c01.x - 3.0);\n"  /* column indexing */
    "    if (uo.y == uint(3)) {\n"          /* uint compare, then branch */
    "        t += vec2(0.0, 0.0);\n"   /* compound assign */
    "    } else {\n"
    "        t += vec2(1.0, 1.0);\n"   /* must not execute */
    "    }\n"
    "    if (io.x < 3 && bo.x) {\n"    /* int compare + logical and */
    "        t -= vec2(0.0, 0.0);\n"
    "    }\n"
    "    if (bo.y || io.x == 2) {\n"   /* logical or, short-circuit */
    "        t += vec2(0.0, 0.0);\n"
    "    }\n"
    "    if (io.x == 2 && uo.x == uint(3)) {\n"
    "        t *= vec2(1.0, 1.0);\n"
    "        if (bo.x) {\n"            /* nested if */
    "            t += vec2(0.0, 0.0);\n"
    "        }\n"
    "    }\n"
    "    if (!bo.y && uo.x == uint(4)) {\n"
    "        t += vec2(1.0, 1.0);\n"   /* must not execute */
    "    }\n"
    "    if (io.x > 5) {\n"
    "        t += vec2(1.0, 1.0);\n"   /* must not execute */
    "    }\n"
    "    vec2 uv = vec2(0.5, 0.0);\n"
    "    float dl = length(uv);\n"                 /* 0.5 */
    "    vec2 dn = normalize(uv);\n"               /* (1, 0) */
    "    float dd = distance(uv, vec2(0.5, 1.0));\n"  /* 1 */
    "    float dt = dot(uv, vec2(2.0, 0.0));\n"    /* 1 */
    "    float da = abs(-0.5);\n"                  /* 0.5 */
    "    float dc = clamp(uv.x, 0.25, 0.5);\n"     /* 0.5 */
    "    vec2 dv = mix(uv, uv + vec2(1.0), vec2(0.0));\n"  /* uv */
    "    float dm = mix(0.0, 1.0, 0.5);\n"         /* 0.5 */
    "    t = t + vec2(dl * 2.0 - 1.0 + dd - 1.0 + dt - 1.0 + da - 0.5\n"
    "                + dc - 0.5 + dm - 0.5,\n"
    "                dn.x - 1.0 + dv.x - uv.x + dv.y - uv.y);\n"
    "    vUV = inPos.xy + vec2(o + k.x - 0.5, k.y - 0.5) + t;\n"
    "    vN = rot3 * inPos;\n"        /* mat3 * vec3 */
    "    gl_Position = mvp * vec4(inPos, 1.0);\n"
    "}\n";

static const char *kFS =
    "#version 460 core\n"
    "in vec2 vUV;\n"
    "in vec3 vN;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "    fragColor = vec4(vUV + vN.xy, 0.5, 1.0);\n"
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

static int checkValues(id<MTLDevice> dev, id<MTLRenderPipelineState> pso) {
    const int W = 64, H = 64;

    /* Rz(30 deg), column-major (GL storage: column 0 first).  mvp is
     * mat4 (64 bytes); rot3 is mat3 std140: 3 columns at 16-byte stride. */
    const float t = 30.0f * (float)M_PI / 180.0f;
    const float c = cosf(t), s = sinf(t);
    const float ubo[28] = {
         c,  s, 0, 0,   /* mvp col 0 */
        -s,  c, 0, 0,   /* mvp col 1 */
         0,  0, 1, 0,   /* mvp col 2 */
         0,  0, 0, 1,   /* mvp col 3 */
         c,  s, 0, 0,   /* rot3 col 0 (offset 64) */
        -s,  c, 0, 0,   /* rot3 col 1 (offset 80) */
         0,  0, 1, 0,   /* rot3 col 2 (offset 96) */
    };

    /* Full-screen triangle in NDC. */
    const float verts[9] = {
        -1, -1, 0,
         3, -1, 0,
        -1,  3, 0,
    };
    /* Transformed NDC positions (w = 1): p = Rz(30) * v. */
    float ap[3], bp[3], cp[3];
    for (int i = 0; i < 3; i++) {
        float *dst = i == 0 ? ap : (i == 1 ? bp : cp);
        float x = verts[3 * i + 0], y = verts[3 * i + 1];
        dst[0] = c * x - s * y;
        dst[1] = s * x + c * y;
        dst[2] = 0;
    }

    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:W height:H mipmapped:NO];
    id<MTLTexture> tex = [dev newTextureWithDescriptor:td];

    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];

    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = tex;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:pso];
    [enc setVertexBytes:ubo length:sizeof(ubo) atIndex:0];
    [enc setVertexBytes:verts length:36 atIndex:1];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    uint8_t *px = (uint8_t *)calloc((size_t)W * H * 4, 1);
    [tex getBytes:px bytesPerRow:(NSUInteger)W * 4
        fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];

    /* CPU reference: vUV interpolates the ORIGINAL vertex positions,
     * with barycentric weights computed in the transformed (rasterized)
     * triangle; compare the interior 32x32 block (away from edges). */
    int bad = 0;
    for (int y = 16; y < H - 16; y++) {
        for (int x = 16; x < W - 16; x++) {
            float ndcX = 2.0f * ((float)x + 0.5f) / W - 1.0f;
            float ndcY = 1.0f - 2.0f * ((float)y + 0.5f) / H;
            uint8_t *p = px + ((size_t)y * W + x) * 4;

            float v0x = bp[0] - ap[0], v0y = bp[1] - ap[1];
            float v1x = cp[0] - ap[0], v1y = cp[1] - ap[1];
            float dx = ndcX - ap[0], dy = ndcY - ap[1];
            float den = v0x * v1y - v1x * v0y;
            float lb = (dx * v1y - dy * v1x) / den;
            float lc = (dy * v0x - dx * v0y) / den;
            float la = 1.0f - lb - lc;

            /* Original inPos at the pixel (MVP^{-1} * ndc). */
            float px = la * verts[0] + lb * verts[3] + lc * verts[6];
            float py = la * verts[1] + lb * verts[4] + lc * verts[7];
            /* vN = rot3 * inPos, interpolated. */
            float nx = c * px - s * py;
            float ny = s * px + c * py;
            /* vUV = inPos + (1, 0); fragColor = vUV + vN.xy. */
            float u = px + 1.0f + nx;
            float v = py + ny;
            u = fminf(1.0f, fmaxf(0.0f, u));
            v = fminf(1.0f, fmaxf(0.0f, v));

            float gotU = (float)p[0] / 255.0f;
            float gotV = (float)p[1] / 255.0f;
            if (fabsf(gotU - u) > 0.006f || fabsf(gotV - v) > 0.006f) {
                if (bad < 10)
                    printf("VALUE_MISMATCH (%d,%d): got (%.3f,%.3f) expect (%.3f,%.3f)\n",
                           x, y, gotU, gotV, u, v);
                bad++;
            }
        }
    }
    free(px);
    if (bad) {
        printf("VALUE_CHECK FAIL (%d pixels)\n", bad);
        return 1;
    }
    printf("VALUE_OK\n");
    return 0;
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
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;

        MTLVertexDescriptor *vd = [MTLVertexDescriptor new];
        vd.attributes[0].format = MTLVertexFormatFloat3;
        vd.attributes[0].offset = 0;
        vd.attributes[0].bufferIndex = 1; /* buffer 0 is reserved for uniforms */
        vd.layouts[1].stride = 12;
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
        return checkValues(dev, pso);
    }
}
