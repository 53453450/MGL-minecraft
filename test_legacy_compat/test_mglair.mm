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

// 独立 AIR 门禁：不链接 mgl_uniform_reflection.c（SPIRV-Cross 依赖树）。
// sampler 合成 GL uniform location 仅 renderer 消费，门禁不校验；返回 -1。
// （产品库中由 mgl_uniform_reflection.c:2149 提供真实实现。头文件在 extern "C" 块内。）
extern "C" GLint mglSyntheticSamplerUniformLocation(int stage, int res_type, GLuint index) {
    (void)stage; (void)res_type; (void)index;
    return -1;
}

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
    "    while (false) { cf += 100.0; }\n"
    "    for (int zz = 0; false; zz++) cf += 100.0;\n"
    "    do { cf += 1.0; } while (false);\n"
    "    int zw = 0;\n"
    "    while (zw < 3) { cf += 1.0; zw++; }\n"
    "    do { cf += 1.0; zw++; } while (zw < 3);\n"
    "    int bi = -8;\n"
    "    int b1 = bi >> 2;\n"
    "    int b2 = bi << 1;\n"
    "    int b3 = bi & 3;\n"
    "    int b4 = bi | 1;\n"
    "    int b5 = bi ^ 7;\n"
    "    int b6 = ~bi;\n"
    "    int b7 = -8 >> 2;\n"
    "    uint bu = 8u;\n"
    "    int b9 = int(bu >> 1);\n"
    "    int b10 = int(0xF0u & 0x30u);\n"
    "    switch (2) {\n"
    "        case 1: cf += 100.0; break;\n"
    "        case 2: cf += 1.0; break;\n"
    "        default: cf += 100.0;\n"
    "    }\n"
    "    switch (9) {\n"
    "        case 1: cf += 100.0; break;\n"
    "        default: cf += 2.0;\n"
    "    }\n"
    "    switch (7) {\n"
    "        case 1: cf += 100.0; break;\n"
    "        case 2: cf += 100.0;\n"
    "    }\n"
    "    switch (1 + 1) {\n"
    "        case 2: cf += 1.0; break;\n"
    "    }\n"
    "    switch (5) {\n"
    "        case 1: cf += 100.0; break;\n"
    "        case 5: cf += 1.0;\n"
    "        case 6: cf += 2.0; break;\n"
    "        default: cf += 100.0;\n"
    "    }\n"
    "    float swcorr = swa.x + swa.y + swb + swc.x + swc.y + swc.z\n"
    "                   + sc.x + sc.y + lp.x + lp.y + cf\n"
    "                   + float(b1 + b2 + b3 + b4 + b5 + b6 + b7 + b9 + b10)\n"
    "                   - 91.0;\n"
    "    float off = float(q) + float(loopCount - 5);\n"
    "    vec2 corr = vsel2 - vec2(3.0, 4.0);\n"
    "    if (vUV.x > 1000.0) discard;\n"
    "    fragColor = vec4(vUV + corr, 0.5 + (sel2 - 2.0) - off + corr2 + swcorr, 1.0);\n"
    "}\n";

static const char *kCS =
    "#version 460 core\n"
    "layout(local_size_x = 1) in;\n"
    "layout(std430) buffer B { float data[4]; } b;\n"
    "layout(std430) buffer A { int counter; } a;\n"
    "uniform sampler2D tex;\n"
    "uniform int uCounter;\n"
    "void main() {\n"
    "    uCounter += 1 + int(gl_GlobalInvocationID.x);\n"
    "    vec3 vc = cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));\n"
    "    float t2 = atan(1.0, 1.0);\n"
    "    vec2 r1 = unpackUnorm2x16(packUnorm2x16(vec2(1.0, 1.0)));\n"
    "    vec2 r2 = unpackSnorm2x16(packSnorm2x16(vec2(1.0, 1.0)));\n"
    "    vec2 r3 = unpackHalf2x16(0x3800u);\n"
    "    b.data[0] = vc.z * 10.0;\n"
    "    b.data[1] = b.data[0] + t2;\n"
    "    b.data[2] = r1.x + r2.y + r3.x;\n"
    "    b.data[3] = b.data[1] + b.data[2];\n"
    "    a.counter += 5;\n"
    "    uCounter += atomicAdd(a.counter, 7);\n"
    "    vec4 tc = texture(tex, vec2(0.5, 0.5));\n"
    "    vec4 tl = textureLod(tex, vec2(0.25, 0.75), 0.0);\n"
    "    uCounter += int(tc.r * 100.0) + int(tl.g * 100.0);\n"
    "    uCounter += int(textureSize(tex, 0).x);\n"
    "    uCounter += int(b.data[3] * 100.0);\n"
    "}\n";

static const char *kCSRuntimeLength =
    "#version 460 core\n"
    "layout(local_size_x = 1) in;\n"
    "layout(std430) buffer RuntimeData { uint count; float values[]; } dataBuffer;\n"
    "int runtimeLength() { return dataBuffer.values.length(); }\n"
    "void main() { dataBuffer.count = uint(runtimeLength()); }\n";

static const char *kTCS =
    "#version 450 core\n"
    "layout(vertices = 3) out;\n"
    "void main() {\n"
    "    gl_out[gl_InvocationID].gl_Position =\n"
    "        gl_in[gl_InvocationID].gl_Position;\n"
    "    gl_TessLevelOuter[0] = 1.0;\n"
    "    gl_TessLevelOuter[1] = 1.0;\n"
    "    gl_TessLevelOuter[2] = 1.0;\n"
    "    gl_TessLevelOuter[3] = float(gl_PatchVerticesIn + gl_PrimitiveID);\n"
    "    gl_TessLevelInner[0] = 1.0;\n"
    "}\n";

static const char *kTES =
    "#version 450 core\n"
    "layout(triangles, equal_spacing, cw) in;\n"
    "layout(location = 1) in vec2 cpUV[];\n"
    "out vec2 vUV;\n"
    "void main() {\n"
    "    gl_Position = gl_in[0].gl_Position +\n"
    "        vec4(gl_TessCoord, float(gl_PrimitiveID));\n"
    "    vUV = cpUV[1] + gl_TessCoord.xy;\n"
    "}\n";

static const char *kTESFS =
    "#version 450 core\n"
    "in vec2 vUV;\n"
    "out vec4 fragColor;\n"
    "void main() { fragColor = vec4(1.0, vUV.x, vUV.y, 1.0); }\n";

static const char *kVSCullDistance =
    "#version 460 core\n"
    "in vec3 inPos;\n"
    "void main() {\n"
    "    gl_Position = vec4(inPos, 1.0);\n"
    "    gl_CullDistance[0] = inPos.x;\n"
    "}\n";

static const char *kFSCullDistance =
    "#version 460 core\n"
    "out vec4 fragColor;\n"
    "void main() { fragColor = vec4(1.0); }\n";

static const char *kGS =
    "#version 450 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices=6) out;\n"
    "void emit_input(int i) {\n"
    "    gl_Position = gl_in[i].gl_Position;\n"
    "    gl_PointSize = gl_in[i].gl_PointSize; EmitVertex();\n"
    "}\n"
    "void main() {\n"
    "    for (int i = 0; i < 3; i++) {\n"
    "        emit_input(i);\n"
    "    }\n"
    "    EndPrimitive();\n"
    "    for (int i = 2; i >= 0; i--) {\n"
    "        emit_input(i);\n"
    "    }\n"
    "    EndPrimitive();\n"
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
    [enc setVertexBytes:verts length:36 atIndex:16];
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

        if (mglShaderInterfaceCheck(kVS, kFS, err, sizeof err) != 0) {
            fprintf(stderr, "interface check FAIL: %s\n", err);
            return 1;
        }
        /* Negative check: a mismatched fragment interface must be
         * rejected before any GPU work happens. */
        static const char *kFSBad =
            "#version 460 core\n"
            "in vec3 vUV;\n"
            "void main() {\n"
            "}\n";
        if (mglShaderInterfaceCheck(kVS, kFSBad, err, sizeof err) == 0) {
            fprintf(stderr, "interface check accepted a mismatch\n");
            return 1;
        }
        printf("interface mismatch rejected: %s\n", err);

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
        vd.attributes[0].bufferIndex = 16; /* attrs live at kMGLVertexAttribBufferBase */
        vd.layouts[16].stride = 12;
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

        /* CullDistance uses two hidden vertex buffers (slots 29/28) and
         * vertex_id metadata.  Building a real PSO catches malformed AIR
         * argument metadata that container-only tests cannot see. */
        unsigned char *cullVsBytes = NULL, *cullFsBytes = NULL;
        size_t cullVsSize = 0, cullFsSize = 0;
        if (mglShaderCompileGLSL(kVSCullDistance, MGL_STAGE_VERTEX,
                                 &cullVsBytes, &cullVsSize,
                                 err, sizeof err) != 0 ||
            mglShaderCompileGLSL(kFSCullDistance, MGL_STAGE_FRAGMENT,
                                 &cullFsBytes, &cullFsSize,
                                 err, sizeof err) != 0) {
            fprintf(stderr, "cull-distance compile FAIL: %s\n", err);
            return 1;
        }
        id<MTLLibrary> cullVsLib = loadLibrary(
            dev, cullVsBytes, cullVsSize, "cull-vs");
        id<MTLLibrary> cullFsLib = loadLibrary(
            dev, cullFsBytes, cullFsSize, "cull-fs");
        mglShaderFree(cullVsBytes);
        mglShaderFree(cullFsBytes);
        id<MTLFunction> cullVsFn = [cullVsLib newFunctionWithName:@"main"];
        id<MTLFunction> cullFsFn = [cullFsLib newFunctionWithName:@"main"];
        MTLRenderPipelineDescriptor *cullPD = [MTLRenderPipelineDescriptor new];
        cullPD.vertexFunction = cullVsFn;
        cullPD.fragmentFunction = cullFsFn;
        cullPD.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        cullPD.vertexDescriptor = vd;
        NSError *cullPErr = nil;
        id<MTLRenderPipelineState> cullPSO =
            [dev newRenderPipelineStateWithDescriptor:cullPD error:&cullPErr];
        if (!cullVsFn || !cullFsFn || !cullPSO) {
            fprintf(stderr, "CULL_DISTANCE_PSO_FAIL: %s\n",
                    cullPErr.localizedDescription.UTF8String ?: "?");
            return 1;
        }
        MTLTextureDescriptor *cullTD = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
            width:8 height:8 mipmapped:NO];
        id<MTLTexture> cullTarget = [dev newTextureWithDescriptor:cullTD];
        id<MTLCommandQueue> cullQueue = [dev newCommandQueue];
        const float cullPositions[9] = {
            -1.0f, -1.0f, 0.0f,
             3.0f, -1.0f, 0.0f,
            -1.0f,  3.0f, 0.0f,
        };
        id<MTLBuffer> cullPositionBuffer = [dev newBufferWithBytes:cullPositions
            length:sizeof(cullPositions) options:MTLResourceStorageModeShared];
        struct CullParams {
            uint32_t primVertexCount, offset, stride, count, firstVertex;
            uint32_t explicitVertexCount, explicitVertices[4];
        } cullParams = {3, 0, 4, 1, 0, 0, {0, 0, 0, 0}};
        auto drawCullCase = ^uint8_t(const float distances[3]) {
            id<MTLBuffer> distanceBuffer = [dev newBufferWithBytes:distances
                length:3 * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> commandBuffer = [cullQueue commandBuffer];
            MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor
                renderPassDescriptor];
            pass.colorAttachments[0].texture = cullTarget;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
            id<MTLRenderCommandEncoder> encoder =
                [commandBuffer renderCommandEncoderWithDescriptor:pass];
            [encoder setRenderPipelineState:cullPSO];
            [encoder setVertexBuffer:cullPositionBuffer offset:0 atIndex:16];
            [encoder setVertexBuffer:distanceBuffer offset:0 atIndex:29];
            [encoder setVertexBytes:&cullParams length:sizeof(cullParams)
                            atIndex:28];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                         vertexStart:0 vertexCount:3];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            uint8_t pixels[8 * 8 * 4] = {0};
            [cullTarget getBytes:pixels bytesPerRow:8 * 4
                      fromRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0];
            uint8_t maxRed = 0;
            for (size_t i = 0; i < sizeof(pixels); i += 4) {
                maxRed = MAX(maxRed, pixels[i]);
            }
            return maxRed;
        };
        const float allNegative[3] = {-1.0f, -1.0f, -1.0f};
        const float onePositive[3] = {-1.0f, 1.0f, -1.0f};
        uint8_t culledPixel = drawCullCase(allNegative);
        uint8_t visiblePixel = drawCullCase(onePositive);
        if (culledPixel != 0 || visiblePixel < 250) {
            fprintf(stderr,
                    "CULL_DISTANCE_VALUE_FAIL: culled=%u visible=%u\n",
                    (unsigned)culledPixel, (unsigned)visiblePixel);
            return 1;
        }
        printf("CULL_DISTANCE_OK\n");

        /* Compute: kernel reads/writes the uniform buffer, two SSBOs
         * (data[] and an atomic counter), a 2D texture, plus
         * gl_GlobalInvocationID and the new builtins; dispatch a single
         * thread and verify every device buffer
         * (41 + 1 + 1328 + 100 + 0 + 4 + 12 -> 1486, data[3] =
         * 13.285..., counter = 5 + 7 = 12). */
        unsigned char *csBytes = NULL;
        size_t csSize = 0;
        if (mglShaderCompileGLSL(kCS, MGL_STAGE_COMPUTE, &csBytes, &csSize,
                                 err, sizeof err) != 0) {
            fprintf(stderr, "cs compile FAIL: %s\n", err);
            return 1;
        }
        id<MTLLibrary> csLib = loadLibrary(dev, csBytes, csSize, "cs");
        mglShaderFree(csBytes);
        if (!csLib) return 1;
        id<MTLFunction> csFn = [csLib newFunctionWithName:@"main"];
        if (!csFn) {
            fprintf(stderr, "newFunctionWithName FAIL (kernel)\n");
            return 1;
        }
        MTLComputePipelineDescriptor *cpd = [MTLComputePipelineDescriptor new];
        cpd.computeFunction = csFn;
        id<MTLComputePipelineState> csPso =
            [dev newComputePipelineStateWithFunction:csFn error:&perr];
        if (!csPso) {
            fprintf(stderr, "CS_PSO_FAIL: %s\n",
                    perr.localizedDescription.UTF8String ?: "?");
            return 1;
        }
        id<MTLBuffer> cbuf = [dev newBufferWithLength:4
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> ssboB = [dev newBufferWithLength:16
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> ssboA = [dev newBufferWithLength:4
                                               options:MTLResourceStorageModeShared];
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                         width:4 height:4 mipmapped:NO];
        id<MTLTexture> tex = [dev newTextureWithDescriptor:td];
        {
            unsigned char px[4 * 4 * 4];
            for (int i = 0; i < 16; i++) {
                px[i * 4 + 0] = 255;  /* red */
                px[i * 4 + 1] = 0;
                px[i * 4 + 2] = 0;
                px[i * 4 + 3] = 255;
            }
            [tex replaceRegion:MTLRegionMake2D(0, 0, 4, 4)
                   mipmapLevel:0 withBytes:px bytesPerRow:16];
        }
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        id<MTLSamplerState> smp = [dev newSamplerStateWithDescriptor:sd];
        ((int *)cbuf.contents)[0] = 41;
        ((int *)ssboA.contents)[0] = 0;
        id<MTLCommandQueue> cq = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [cq commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:csPso];
        [enc setBuffer:cbuf offset:0 atIndex:0];
        [enc setBuffer:ssboB offset:0 atIndex:1];
        [enc setBuffer:ssboA offset:0 atIndex:2];
        [enc setTexture:tex atIndex:0];
        [enc setSamplerState:smp atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        int csGot = ((int *)cbuf.contents)[0];
        if (csGot != 1486) {
            fprintf(stderr, "COMPUTE_VALUE_FAIL: %d\n", csGot);
            return 1;
        }
        float *bf = (float *)ssboB.contents;
        int *ac = (int *)ssboA.contents;
        if (bf[3] < 13.28 || bf[3] > 13.29) {
            fprintf(stderr, "SSBO_VALUE_FAIL: %f\n", bf[3]);
            return 1;
        }
        if (*ac != 12) {
            fprintf(stderr, "ATOMIC_VALUE_FAIL: %d\n", *ac);
            return 1;
        }
        printf("SSBO_OK ATOMIC_OK\n");

        /* Runtime SSBO .length(): the hidden buffer(25) table contains the
         * visible byte size for Metal slot 0.  The four-byte prefix is
         * excluded before dividing by the std430 float stride. */
        {
            unsigned char *lengthBytes = NULL;
            size_t lengthSize = 0;
            if (mglShaderCompileGLSL(kCSRuntimeLength, MGL_STAGE_COMPUTE,
                                     &lengthBytes, &lengthSize,
                                     err, sizeof err) != 0) {
                fprintf(stderr, "runtime length compile FAIL: %s\n", err);
                return 1;
            }
            id<MTLLibrary> lengthLib = loadLibrary(
                dev, lengthBytes, lengthSize, "runtime_length");
            mglShaderFree(lengthBytes);
            if (!lengthLib) return 1;
            id<MTLFunction> lengthFn = [lengthLib newFunctionWithName:@"main"];
            NSError *lengthErr = nil;
            id<MTLComputePipelineState> lengthPSO =
                [dev newComputePipelineStateWithFunction:lengthFn
                                                     error:&lengthErr];
            if (!lengthPSO) {
                fprintf(stderr, "RUNTIME_LENGTH_PSO_FAIL: %s\n",
                        lengthErr.localizedDescription.UTF8String ?: "?");
                return 1;
            }
            const uint32_t visibleSizes[2] = {4u + 7u * 4u, 4u + 4u * 4u};
            for (uint32_t pass = 0; pass < 2; pass++) {
                id<MTLBuffer> runtimeData = [dev newBufferWithLength:visibleSizes[pass]
                    options:MTLResourceStorageModeShared];
                ((uint32_t *)runtimeData.contents)[0] = 1234u;
                uint32_t sizeConstants[31] = {0};
                sizeConstants[0] = visibleSizes[pass];
                id<MTLBuffer> sizeTable = [dev newBufferWithBytes:sizeConstants
                    length:sizeof sizeConstants options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> lengthCB = [cq commandBuffer];
                id<MTLComputeCommandEncoder> lengthEnc =
                    [lengthCB computeCommandEncoder];
                [lengthEnc setComputePipelineState:lengthPSO];
                [lengthEnc setBuffer:runtimeData offset:0 atIndex:0];
                [lengthEnc setBuffer:sizeTable offset:0 atIndex:25];
                [lengthEnc dispatchThreads:MTLSizeMake(1, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [lengthEnc endEncoding];
                [lengthCB commit];
                [lengthCB waitUntilCompleted];
                uint32_t got = ((uint32_t *)runtimeData.contents)[0];
                uint32_t expected = pass == 0 ? 7u : 4u;
                if (got != expected) {
                    fprintf(stderr,
                            "RUNTIME_LENGTH_VALUE_FAIL: pass=%u got=%u expected=%u status=%lu error=%s\n",
                            pass, got, expected, (unsigned long)lengthCB.status,
                            lengthCB.error.localizedDescription.UTF8String ?: "?");
                    return 1;
                }
            }
            printf("RUNTIME_LENGTH_OK\n");
        }

        /* TCS compute kernel: copy two patches of fixed gl_PerVertex records
         * and let invocation zero write six half factors per patch. */
        {
            unsigned char *tcsBytes = NULL;
            size_t tcsSize = 0;
            if (mglShaderCompileGLSL(kTCS, MGL_STAGE_TESS_CONTROL,
                                     &tcsBytes, &tcsSize,
                                     err, sizeof err) != 0) {
                fprintf(stderr, "tcs compile FAIL: %s\n", err);
                return 1;
            }
            id<MTLLibrary> tcsLib = loadLibrary(dev, tcsBytes, tcsSize, "tcs");
            mglShaderFree(tcsBytes);
            if (!tcsLib) return 1;
            id<MTLFunction> tcsFn = [tcsLib newFunctionWithName:@"main"];
            if (!tcsFn) {
                fprintf(stderr, "newFunctionWithName FAIL (tcs)\n");
                return 1;
            }
            NSError *tcsErr = nil;
            id<MTLComputePipelineState> tcsPso =
                [dev newComputePipelineStateWithFunction:tcsFn error:&tcsErr];
            if (!tcsPso) {
                fprintf(stderr, "TCS_PSO_FAIL: %s\n",
                        tcsErr.localizedDescription.UTF8String ?: "?");
                return 1;
            }

            float stageIn[6][MGL_AIR_PER_VERTEX_STRIDE / sizeof(float)] = {
                {0.0f, 0.0f, 0.0f, 1.0f},
                {1.0f, 0.0f, 0.0f, 1.0f},
                {0.0f, 1.0f, 0.0f, 1.0f},
                {2.0f, 0.0f, 0.0f, 1.0f},
                {3.0f, 0.0f, 0.0f, 1.0f},
                {2.0f, 1.0f, 0.0f, 1.0f},
            };
            id<MTLBuffer> tcsIn = [dev newBufferWithBytes:stageIn
                                                   length:sizeof stageIn
                                                  options:MTLResourceStorageModeShared];
            id<MTLBuffer> factors = [dev newBufferWithLength:24
                                                     options:MTLResourceStorageModeShared];
            id<MTLBuffer> patchOut = [dev newBufferWithLength:32
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> stageOut = [dev newBufferWithLength:sizeof stageIn
                                                      options:MTLResourceStorageModeShared];
            const uint32_t indirectParams[2] = {3u, 1u};
            id<MTLBuffer> indirect = [dev newBufferWithBytes:indirectParams
                                                     length:sizeof indirectParams
                                                    options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> tcb = [cq commandBuffer];
            id<MTLComputeCommandEncoder> tenc = [tcb computeCommandEncoder];
            [tenc setComputePipelineState:tcsPso];
            [tenc setBuffer:tcsIn offset:0 atIndex:24];
            [tenc setBuffer:factors offset:0 atIndex:26];
            [tenc setBuffer:patchOut offset:0 atIndex:27];
            [tenc setBuffer:stageOut offset:0 atIndex:28];
            [tenc setBuffer:indirect offset:0 atIndex:29];
            [tenc dispatchThreadgroups:MTLSizeMake(2, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(3, 1, 1)];
            [tenc endEncoding];
            [tcb commit];
            [tcb waitUntilCompleted];
            if (tcb.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "TCS_DISPATCH_FAIL: %s\n",
                        tcb.error.localizedDescription.UTF8String ?: "?");
                return 1;
            }

            float *tcsOut = (float *)stageOut.contents;
            for (int vertex = 0; vertex < 6; vertex++) {
                for (int lane = 0; lane < 4; lane++) {
                    if (fabsf(tcsOut[vertex *
                                      (MGL_AIR_PER_VERTEX_STRIDE / sizeof(float)) + lane] -
                              stageIn[vertex][lane]) > 1e-6f) {
                        fprintf(stderr,
                                "TCS_STAGE_OUT_FAIL: vertex=%d lane=%d got=%f expected=%f\n",
                                vertex, lane,
                                tcsOut[vertex *
                                       (MGL_AIR_PER_VERTEX_STRIDE / sizeof(float)) + lane],
                                stageIn[vertex][lane]);
                        return 1;
                    }
                }
            }
            const uint16_t *halfFactors = (const uint16_t *)factors.contents;
            const uint16_t expectedFactors[12] = {
                0x3c00u, 0x3c00u, 0x3c00u, 0x4200u, 0x3c00u, 0x3c00u,
                0x3c00u, 0x3c00u, 0x3c00u, 0x4400u, 0x3c00u, 0x3c00u,
            };
            for (int i = 0; i < 12; i++) {
                if (halfFactors[i] != expectedFactors[i]) {
                    fprintf(stderr, "TCS_FACTOR_FAIL: factor=%d bits=0x%04x\n",
                            i, halfFactors[i]);
                    return 1;
                }
            }
            printf("TCS_OK\n");
        }

        /* Native post-tessellation AIR vertex: loading the function must
         * expose the triangle patch ABI and its fixed control-point count. */
        {
            unsigned char *tesBytes = NULL;
            size_t tesSize = 0;
            if (mglShaderCompileGLSL(kTES, MGL_STAGE_TESS_EVALUATION,
                                     &tesBytes, &tesSize,
                                     err, sizeof err) != 0) {
                fprintf(stderr, "tes compile FAIL: %s\n", err);
                return 1;
            }
            id<MTLLibrary> tesLib = loadLibrary(dev, tesBytes, tesSize, "tes");
            mglShaderFree(tesBytes);
            if (!tesLib) return 1;
            id<MTLFunction> tesFn = [tesLib newFunctionWithName:@"main"];
            if (!tesFn) {
                fprintf(stderr, "newFunctionWithName FAIL (tes)\n");
                return 1;
            }
            /* The metallib TESS tag carries 4*controlPointCount + patchKind;
             * without a TCS the GL default patch size (3) is encoded. */
            if (tesFn.patchType != MTLPatchTypeTriangle ||
                tesFn.patchControlPointCount != 3u) {
                fprintf(stderr,
                        "TES_ABI_FAIL: patchType=%lu controlPoints=%lu\n",
                        (unsigned long)tesFn.patchType,
                        (unsigned long)tesFn.patchControlPointCount);
                return 1;
            }

            unsigned char *tesFsBytes = NULL;
            size_t tesFsSize = 0;
            if (mglShaderCompileGLSL(kTESFS, MGL_STAGE_FRAGMENT,
                                     &tesFsBytes, &tesFsSize,
                                     err, sizeof err) != 0) {
                fprintf(stderr, "tes fragment compile FAIL: %s\n", err);
                return 1;
            }
            id<MTLLibrary> tesFsLib =
                loadLibrary(dev, tesFsBytes, tesFsSize, "tes-fragment");
            mglShaderFree(tesFsBytes);
            if (!tesFsLib) return 1;
            id<MTLFunction> tesFsFn = [tesFsLib newFunctionWithName:@"main"];

            MTLRenderPipelineDescriptor *tesPd =
                [MTLRenderPipelineDescriptor new];
            tesPd.vertexFunction = tesFn;
            tesPd.fragmentFunction = tesFsFn;
            tesPd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            tesPd.tessellationPartitionMode = MTLTessellationPartitionModeInteger;
            tesPd.maxTessellationFactor = 16;
            tesPd.tessellationFactorScaleEnabled = NO;
            tesPd.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
            tesPd.tessellationControlPointIndexType =
                MTLTessellationControlPointIndexTypeNone;
            tesPd.tessellationFactorStepFunction =
                MTLTessellationFactorStepFunctionConstant;
            tesPd.tessellationOutputWindingOrder = MTLWindingClockwise;
            MTLVertexDescriptor *tesVd = [MTLVertexDescriptor new];
            tesVd.attributes[0].format = MTLVertexFormatFloat4;
            tesVd.attributes[0].offset = 0;
            tesVd.attributes[0].bufferIndex = 0;
            tesVd.attributes[2].format = MTLVertexFormatFloat2;
            tesVd.attributes[2].offset =
                MGL_AIR_PER_VERTEX_STRIDE + 16;
            tesVd.attributes[2].bufferIndex = 0;
            tesVd.layouts[0].stride = MGL_AIR_PER_VERTEX_STRIDE + 32;
            tesVd.layouts[0].stepFunction =
                MTLVertexStepFunctionPerPatchControlPoint;
            tesPd.vertexDescriptor = tesVd;
            NSError *tesPsoErr = nil;
            id<MTLRenderPipelineState> tesPso =
                [dev newRenderPipelineStateWithDescriptor:tesPd
                                                    error:&tesPsoErr];
            if (!tesPso) {
                fprintf(stderr, "TES_PSO_FAIL: %s\n",
                        tesPsoErr.localizedDescription.UTF8String ?: "?");
                return 1;
            }

            float controlPoints[3][24] = {0};
            controlPoints[0][0] = -1.0f;
            controlPoints[0][1] = -1.0f;
            controlPoints[0][3] = 1.0f;
            controlPoints[1][0] = 1.0f;
            controlPoints[1][1] = -1.0f;
            controlPoints[1][3] = 1.0f;
            controlPoints[1][20] = 0.25f;
            controlPoints[1][21] = 0.5f;
            controlPoints[2][0] = -1.0f;
            controlPoints[2][1] = 1.0f;
            controlPoints[2][3] = 1.0f;
            const uint16_t triangleFactors[4] = {
                0x3c00u, 0x3c00u, 0x3c00u, 0x3c00u,
            };
            id<MTLBuffer> cpBuf = [dev newBufferWithBytes:controlPoints
                                                   length:sizeof controlPoints
                                                  options:MTLResourceStorageModeShared];
            id<MTLBuffer> tfBuf = [dev newBufferWithBytes:triangleFactors
                                                   length:sizeof triangleFactors
                                                  options:MTLResourceStorageModeShared];
            MTLTextureDescriptor *tesTd = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                             width:16 height:16 mipmapped:NO];
            tesTd.storageMode = MTLStorageModeShared;
            tesTd.usage = MTLTextureUsageRenderTarget;
            id<MTLTexture> tesTarget = [dev newTextureWithDescriptor:tesTd];
            MTLRenderPassDescriptor *tesRp = [MTLRenderPassDescriptor
                renderPassDescriptor];
            tesRp.colorAttachments[0].texture = tesTarget;
            tesRp.colorAttachments[0].loadAction = MTLLoadActionClear;
            tesRp.colorAttachments[0].storeAction = MTLStoreActionStore;
            tesRp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
            id<MTLCommandBuffer> tesCb = [cq commandBuffer];
            id<MTLRenderCommandEncoder> tesEnc =
                [tesCb renderCommandEncoderWithDescriptor:tesRp];
            [tesEnc setRenderPipelineState:tesPso];
            [tesEnc setVertexBuffer:cpBuf offset:0 atIndex:0];
            [tesEnc setVertexBuffer:cpBuf offset:0 atIndex:30];
            const uint32_t patchInfo[2] = {3u, 3u};
            [tesEnc setVertexBytes:patchInfo length:sizeof(patchInfo) atIndex:28];
            [tesEnc setTessellationFactorBuffer:tfBuf
                                         offset:0
                                 instanceStride:0];
            [tesEnc drawPatches:3
                     patchStart:0
                     patchCount:1
               patchIndexBuffer:nil
         patchIndexBufferOffset:0
                  instanceCount:1
                   baseInstance:0];
            [tesEnc endEncoding];
            [tesCb commit];
            [tesCb waitUntilCompleted];
            if (tesCb.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "TES_DRAW_FAIL: %s\n",
                        tesCb.error.localizedDescription.UTF8String ?: "?");
                return 1;
            }
            uint8_t tesPixels[16 * 16 * 4] = {0};
            [tesTarget getBytes:tesPixels
                    bytesPerRow:16 * 4
                     fromRegion:MTLRegionMake2D(0, 0, 16, 16)
                    mipmapLevel:0];
            bool sawRed = false;
            for (size_t i = 0; i < sizeof tesPixels; i += 4) {
                if (tesPixels[i] != 0) { sawRed = true; break; }
            }
            if (!sawRed) {
                fprintf(stderr, "TES_DRAW_FAIL: render target stayed clear\n");
                return 1;
            }
            printf("TES_OK\n");
        }

        /* Geometry P1 compute expansion: one triangle thread reads three
         * fixed 32-byte per-vertex records and expands two separately-ended
         * strips into a six-vertex triangle list. */
        {
            unsigned char *gsBytes = NULL;
            size_t gsSize = 0;
            if (mglShaderCompileGLSL(kGS, MGL_STAGE_GEOMETRY,
                                     &gsBytes, &gsSize,
                                     err, sizeof err) != 0) {
                fprintf(stderr, "gs compile FAIL: %s\n", err);
                return 1;
            }
            id<MTLLibrary> gsLib = loadLibrary(dev, gsBytes, gsSize, "gs");
            mglShaderFree(gsBytes);
            if (!gsLib) return 1;
            id<MTLFunction> gsFn = [gsLib newFunctionWithName:@"main"];
            if (!gsFn) {
                fprintf(stderr, "newFunctionWithName FAIL (gs)\n");
                return 1;
            }
            NSError *gsErr = nil;
            id<MTLComputePipelineState> gsPso =
                [dev newComputePipelineStateWithFunction:gsFn error:&gsErr];
            if (!gsPso) {
                fprintf(stderr, "GS_PSO_FAIL: %s\n",
                        gsErr.localizedDescription.UTF8String ?: "?");
                return 1;
            }
            float gsIn[3][MGL_AIR_PER_VERTEX_STRIDE / sizeof(float)] = {
                {-1.0f, -1.0f, 0.0f, 1.0f, 2.0f},
                { 1.0f, -1.0f, 0.0f, 1.0f, 4.0f},
                {-1.0f,  1.0f, 0.0f, 1.0f, 6.0f},
            };
            id<MTLBuffer> gsInput = [dev newBufferWithBytes:gsIn
                                                      length:sizeof gsIn
                                                     options:MTLResourceStorageModeShared];
            id<MTLBuffer> gsOutput = [dev newBufferWithLength:
                                                       14 * MGL_AIR_PER_VERTEX_STRIDE
                                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> gsCount = [dev newBufferWithLength:16
                                                      options:MTLResourceStorageModeShared];
            memset(gsOutput.contents, 0, gsOutput.length);
            memset(gsCount.contents, 0, gsCount.length);
            id<MTLCommandBuffer> gcb = [cq commandBuffer];
            id<MTLComputeCommandEncoder> genc = [gcb computeCommandEncoder];
            [genc setComputePipelineState:gsPso];
            [genc setBuffer:gsInput offset:0 atIndex:24];
            [genc setBuffer:gsOutput offset:0 atIndex:28];
            [genc setBuffer:gsCount offset:0 atIndex:29];
            [genc dispatchThreads:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [genc endEncoding];
            [gcb commit];
            [gcb waitUntilCompleted];
            if (gcb.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "GS_DISPATCH_FAIL: %s\n",
                        gcb.error.localizedDescription.UTF8String ?: "?");
                return 1;
            }
            float *gsOut = (float *)gsOutput.contents +
                2 * (MGL_AIR_PER_VERTEX_STRIDE / sizeof(float));
            if (*(uint32_t *)gsCount.contents != 6u ||
                fabsf(gsOut[0] + 1.0f) > 1e-5f ||
                fabsf(gsOut[16] - 1.0f) > 1e-5f ||
                fabsf(gsOut[32] + 1.0f) > 1e-5f ||
                fabsf(gsOut[48] + 1.0f) > 1e-5f ||
                fabsf(gsOut[64] - 1.0f) > 1e-5f ||
                fabsf(gsOut[80] + 1.0f) > 1e-5f ||
                fabsf(gsOut[4] - 2.0f) > 1e-5f ||
                fabsf(gsOut[20] - 4.0f) > 1e-5f ||
                fabsf(gsOut[36] - 6.0f) > 1e-5f ||
                fabsf(gsOut[52] - 6.0f) > 1e-5f ||
                fabsf(gsOut[68] - 4.0f) > 1e-5f ||
                fabsf(gsOut[84] - 2.0f) > 1e-5f) {
                fprintf(stderr, "GS_VALUE_FAIL: count=%u p0=(%f,%f;%f) p1=(%f,%f;%f) p2=(%f,%f;%f) p3=(%f,%f;%f) p4=(%f,%f;%f) p5=(%f,%f;%f)\n",
                        *(uint32_t *)gsCount.contents,
                        gsOut[0], gsOut[1], gsOut[4],
                        gsOut[16], gsOut[17], gsOut[20],
                        gsOut[32], gsOut[33], gsOut[36],
                        gsOut[48], gsOut[49], gsOut[52],
                        gsOut[64], gsOut[65], gsOut[68],
                        gsOut[80], gsOut[81], gsOut[84]);
                return 1;
            }
            printf("GS_OK\n");
        }

        /* XFB capture variant: vertex outputs (position + varyings) go
         * into a device buffer at index 29, indexed by gl_VertexID.
         * Render 3 vertices with an identity MVP and verify the captured
         * record of vertex 1. */
        {
            static const char *kVSX =
                "#version 460 core\n"
                "uniform mat4 mvp;\n"
                "in vec3 inPos;\n"
                "out vec2 vUV;\n"
                "void main() {\n"
                "    gl_Position = mvp * vec4(inPos, 1.0);\n"
                "    gl_Position.y += float(gl_VertexID);\n"
                "    vUV = inPos.xy;\n"
                "}\n";
            unsigned char *xBytes = NULL;
            size_t xSize = 0;
            if (mglShaderCompileGLSLCapture(kVSX, &xBytes, &xSize,
                                            err, sizeof err) != 0) {
                fprintf(stderr, "capture compile FAIL: %s\n", err);
                return 1;
            }
            id<MTLLibrary> xLib = loadLibrary(dev, xBytes, xSize, "xfb");
            mglShaderFree(xBytes);
            if (!xLib) return 1;
            id<MTLFunction> xFn = [xLib newFunctionWithName:@"main"];
            if (!xFn) {
                fprintf(stderr, "newFunctionWithName FAIL (capture)\n");
                return 1;
            }
            MTLRenderPipelineDescriptor *xpd = [MTLRenderPipelineDescriptor new];
            xpd.vertexFunction = xFn;
            xpd.rasterizationEnabled = NO;
            xpd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            MTLVertexDescriptor *xvd = [MTLVertexDescriptor new];
            xvd.attributes[0].format = MTLVertexFormatFloat3;
            xvd.attributes[0].offset = 0;
            xvd.attributes[0].bufferIndex = 16;
            xvd.layouts[16].stride = 12;
            xpd.vertexDescriptor = xvd;
            NSError *xerr = nil;
            id<MTLRenderPipelineState> xpso =
                [dev newRenderPipelineStateWithDescriptor:xpd error:&xerr];
            if (!xpso) {
                fprintf(stderr, "XFB_PSO_FAIL: %s\n",
                        xerr.localizedDescription.UTF8String ?: "?");
                NSDictionary *ui = xerr.userInfo;
                for (NSString *k in ui) {
                    fprintf(stderr, "  XFB userInfo[%s] = %s\n",
                            k.UTF8String,
                            [ui[k] description].UTF8String ?: "?");
                }
                return 1;
            }
            /* Identity MVP. */
            id<MTLBuffer> mvpBuf = [dev newBufferWithLength:64
                                                    options:MTLResourceStorageModeShared];
            float *m = (float *)mvpBuf.contents;
            for (int i = 0; i < 16; i++) m[i] = 0.0f;
            m[0] = m[5] = m[10] = m[15] = 1.0f;
            /* 3 vertices: (0,0,0) (1,0,0) (0,1,0). */
            float verts[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
            id<MTLBuffer> vbuf = [dev newBufferWithBytes:verts length:36
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> capBuf = [dev newBufferWithLength:3 * 32
                                                    options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> xcb = [cq commandBuffer];
            MTLRenderPassDescriptor *xrp =
                [MTLRenderPassDescriptor renderPassDescriptor];
            xrp.colorAttachments[0].texture = tex;
            xrp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
            xrp.colorAttachments[0].storeAction = MTLStoreActionDontCare;
            id<MTLRenderCommandEncoder> xenc =
                [xcb renderCommandEncoderWithDescriptor:xrp];
            [xenc setRenderPipelineState:xpso];
            [xenc setVertexBuffer:capBuf offset:0 atIndex:29];
            [xenc setVertexBuffer:mvpBuf offset:0 atIndex:0];
            [xenc setVertexBuffer:vbuf offset:0 atIndex:16];
            [xenc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [xenc endEncoding];
            [xcb commit];
            [xcb waitUntilCompleted];
            float *cap = (float *)capBuf.contents;
            /* 32-byte records (align 16): vertex 1 starts at cap[8]. */
            if (fabsf(cap[8] - 1.0f) > 1e-4f || fabsf(cap[9] - 1.0f) > 1e-4f ||
                fabsf(cap[10] - 0.5f) > 1e-4f || fabsf(cap[11] - 1.0f) > 1e-4f ||
                fabsf(cap[12] - 1.0f) > 1e-4f || fabsf(cap[13]) > 1e-4f) {
                fprintf(stderr, "XFB_VALUE_FAIL: pos=(%f,%f,%f,%f) uv=(%f,%f)\n",
                        cap[8], cap[9], cap[10], cap[11], cap[12], cap[13]);
                return 1;
            }
            printf("XFB_OK\n");
        }

        return checkValues(dev, pso);
    }
}
