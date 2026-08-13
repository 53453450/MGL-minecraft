#include <metal_stdlib>
using namespace metal;

struct MGLMSAAIntegerResolveParams { uint2 srcOrigin; uint2 dstOrigin; uint2 size; uint2 _padding; };

kernel void mgl_msaa_resolve_uint(texture2d_ms<uint, access::read> src [[texture(0)]], texture2d<uint, access::write> dst [[texture(1)]], constant MGLMSAAIntegerResolveParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.size.x || gid.y >= p.size.y) return;
    uint2 srcCoord = p.srcOrigin + gid;
    uint2 dstCoord = p.dstOrigin + gid;
    // GL requires GL_NEAREST for integer/MSAA blits; choose sample 0 deterministically.
    dst.write(src.read(srcCoord, 0), dstCoord);
}

kernel void mgl_msaa_resolve_int(texture2d_ms<int, access::read> src [[texture(0)]], texture2d<int, access::write> dst [[texture(1)]], constant MGLMSAAIntegerResolveParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.size.x || gid.y >= p.size.y) return;
    uint2 srcCoord = p.srcOrigin + gid;
    uint2 dstCoord = p.dstOrigin + gid;
    // GL requires GL_NEAREST for integer/MSAA blits; choose sample 0 deterministically.
    dst.write(src.read(srcCoord, 0), dstCoord);
}
