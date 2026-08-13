#include <metal_stdlib>
using namespace metal;

struct MGLScaledBlitComputeParams { uint2 dstSize; uint srcLevel; uint dstLevel; };

kernel void mgl_scaled_blit_cs(uint2 gid [[thread_position_in_grid]],
                               constant MGLScaledBlitComputeParams& p [[buffer(0)]],
                               texture2d<float, access::read> src [[texture(0)]],
                               texture2d<float, access::write> dst [[texture(1)]]) {
    if (gid.x >= p.dstSize.x || gid.y >= p.dstSize.y) return;
    uint2 srcCoord = uint2(gid.x, p.dstSize.y - 1u - gid.y);
    float4 color = src.read(srcCoord, p.srcLevel);
    dst.write(color, gid, p.dstLevel);
}
