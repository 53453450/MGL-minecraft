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
    "in vec3 inPos;\n"
    "out vec2 vUV;\n"
    "void main() {\n"
    "    vec2 t = vec2(0.0, 0.0);\n"
    "    mat2 A = mat2(1.0, 2.0, 3.0, 4.0);\n"
    "    mat2 B = mat2(5.0, 6.0, 7.0, 8.0);\n"
    "    vec2 x = vec2(2.0, -1.0);\n"
    "    t = t + A * x;\n"
    "    t = t + x * B;\n"
    "    t = t + (A * B) * x;\n"
    "    t = t - (A + 1.0) * x;\n"
    "    mat2 C = A;\n"
    "    C *= B;\n"
    "    t = t - C * x;\n"
    "    t = t - (A - A) * x;\n"
    "    t = t - (2.0 * A - A - A) * x;\n"
    "    mat2 I2 = mat2(1.0, 0.0, 0.0, 1.0);\n"
    "    t = t - (A * inverse(A) - I2) * x;\n"
    "    t = t - (transpose(transpose(A)) - A) * x;\n"
    "    t = t - (matrixCompMult(A, B) - matrixCompMult(A, B)) * x;\n"
    "    t = t - (outerProduct(x, x) - outerProduct(x, x)) * x;\n"
    "    t = t + vec2(determinant(A) + 2.0, determinant(I2) - 1.0);\n"
    "    t = t + (inverse(I2) - I2) * x;\n"
    "    mat3 M3 = mat3(1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);\n"
    "    mat3 I3 = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);\n"
    "    t = t + ((M3 * inverse(M3) - I3) * vec3(1.0)).xy;\n"
    "    mat4 M4 = mat4(1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);\n"
    "    mat4 I4 = mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);\n"
    "    t = t + ((M4 * inverse(M4) - I4) * vec4(1.0)).xy;\n"
    "    int sw = 1;\n"
    "    switch (sw) {\n"
    "    case 1: sw = 2; break;\n"
    "    default: sw = 0;\n"
    "    }\n"
    "    t = t - vec2(float(sw - 2));\n"
    "    vUV = inPos.xy + vec2(t.x - 3.0, t.y - 5.0);\n"
    "    gl_Position = mvp * vec4(inPos, 1.0);\n"
    "}\n";

static const char *kFS =
    "#version 460 core\n"
    "in vec2 vUV;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "    float sel = (vUV.x > 10.0) ? 1.0 : 0.0;\n"
    "    float sel2 = (sel == 0.0) ? 2.0 : 0.0;\n"
    "    vec2 vsel = (vUV.y > 10.0) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);\n"
    "    vec2 vsel2 = (vsel.y == 1.0) ? vec2(3.0, 4.0) : vec2(0.0, 0.0);\n"
    "    int q = 0;\n"
    "    switch (q) {\n"
    "    case 0: q = 5; break;\n"
    "    case 1: q = 6; break;\n"
    "    default: q = 9;\n"
    "    }\n"
    "    q = q - 5;\n"
    "    switch (q) {\n"
    "    case 0: q = 1;\n"
    "    default: q = 2; break;\n"
    "    }\n"
    "    switch (q) {\n"
    "    case 1: q = 4; break;\n"
    "    case 2: q = 0; break;\n"
    "    default: q = 9;\n"
    "    }\n"
    "    switch (q) {\n"
    "    case 0: break;\n"
    "    case 1: q = 4; break;\n"
    "    default: q = 6;\n"
    "    }\n"
    "    switch (q) {\n"
    "    case 5: q = 8; break;\n"
    "    default: q = 3;\n"
    "    }\n"
    "    switch (q) {\n"
    "    case 3:\n"
    "        q = 0;\n"
    "        switch (q) { case 0: q = 2; break; default: q = 9; }\n"
    "        q = q - 2;\n"
    "        break;\n"
    "    default: q = 9;\n"
    "    }\n"
    "    int loopCount = 0;\n"
    "    for (int i = 0; i < 3; i++) {\n"
    "        switch (i) {\n"
    "        case 1: continue;\n"
    "        case 2: loopCount = loopCount + 2; break;\n"
    "        default: loopCount = loopCount + 1;\n"
    "        }\n"
    "        loopCount = loopCount + 1;\n"
    "    }\n"
    "    vec2 dv = vec2(1.0, 2.0);\n"
    "    int di = 1;\n"
    "    dv[di] = 4.0;\n"
    "    dv[0] = 3.0;\n"
    "    dv[di] += 1.0;\n"
    "    float dsum = dv[di] + dv[1];\n"
    "    mat2 dm = mat2(1.0, 2.0, 3.0, 4.0);\n"
    "    dm[di] = vec2(7.0, 8.0);\n"
    "    dm[0] += vec2(1.0, 1.0);\n"
    "    vec2 dcol = dm[di];\n"
    "    dm[1][0] = 9.0;\n"
    "    float delem = dm[1][1];\n"
    "    vec2 lv = vec2(0.0, 0.0);\n"
    "    for (int i = 0; i < 2; i++) {\n"
    "        lv[i] = float(i + 1);\n"
    "    }\n"
    "    float corr2 = dsum + dcol.x + dcol.y + delem + lv[0] + lv[1]\n"
    "                  - 36.0;\n"
    "    vec4 sw = vec4(1.0, 2.0, 3.0, 4.0);\n"
    "    sw.xy = vec2(9.0, 8.0);\n"
    "    sw.z += 1.0;\n"
    "    sw.w = sw.x - 5.0;\n"
    "    sw.yx = vec2(7.0, 6.0);\n"
    "    vec2 swa = sw.zw;\n"
    "    float swb = sw.y;\n"
    "    sw.rgb = vec3(1.0, 2.0, 3.0);\n"
    "    vec3 swc = sw.wzyx.xyz;\n"
    "    mat2 sm = mat2(1.0, 2.0, 3.0, 4.0);\n"
    "    sm[0].x = 5.0;\n"
    "    sm[1] = sm[0].yx;\n"
    "    vec2 sc = sm[0] + sm[1].x;\n"
    "    vec4 lp = vec4(0.0);\n"
    "    for (int i = 0; i < 2; i++) {\n"
    "        lp.xy = vec2(float(i), float(i) + 1.0);\n"
    "    }\n"
    "    float cf = 2.0 * 3.0 + 1.0;\n"
    "    if (true) cf += 1.0;\n"
    "    if (false) cf += 100.0;\n"
    "    if (6 / 2 > 2) cf += 1.0;\n"
    "    float dc = float(loopCount);\n"
    "    if (dc == 5.0) cf += 1.0;\n"
    "    if (dc > 100.0) cf += 1000.0;\n"
    "    float swcorr = swa.x + swa.y + swb + swc.x + swc.y + swc.z\n"
    "                   + sc.x + sc.y + lp.x + lp.y + cf - 48.0;\n"
    "    float off = float(q) + float(loopCount - 5);\n"
    "    vec2 corr = vsel2 - vec2(3.0, 4.0);\n"
    "    if (vUV.x > 1000.0) discard;\n"
    "    fragColor = vec4(vUV + corr, 0.5 + (sel2 - 2.0) - off + corr2 + swcorr, 1.0);\n"
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
            /* Matrix ops net to t=(3,5): A*x=(-1,0), x*B=(4,6),
             * (A*B)*x=(15,22), (A+1)*x=(0,1), C*=B then C*x,
             * (A-A) and (2A-A-A) vanish; matrix builtins all cancel:
             * A*inverse(A)-I2, transpose(transpose(A))-A,
             * matrixCompMult self-diff, outerProduct self-diff,
             * det(A)+2, det(I2)-1, inverse(I2)-I2, and
             * (M3*inverse(M3)-I3), (M4*inverse(M4)-I4), so vUV =
             * inPos.xy. */
            float u = px;
            float v = py;
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
