#include <metal_stdlib>
using namespace metal;

struct MGLMSAAIntegerResolveParams { uint2 srcOrigin; uint2 dstOrigin; uint2 size; uint2 _padding; };

kernel void mgl_msaa_resolve_uint(texture2d_ms<uint, access::read> src [[texture(0)]], texture2d<uint, access::write> dst [[texture(1)]], constant MGLMSAAIntegerResolveParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    /* AUDIT-043: reject zero size and origin+gid that would wrap past src/dst. */
    if (p.size.x == 0u || p.size.y == 0u) return;
    if (gid.x >= p.size.x || gid.y >= p.size.y) return;
    uint2 srcSize = uint2(src.get_width(), src.get_height());
    uint2 dstSize = uint2(dst.get_width(), dst.get_height());
    if (p.srcOrigin.x >= srcSize.x || p.srcOrigin.y >= srcSize.y ||
        p.dstOrigin.x >= dstSize.x || p.dstOrigin.y >= dstSize.y) return;
    if (gid.x >= srcSize.x - p.srcOrigin.x ||
        gid.y >= srcSize.y - p.srcOrigin.y ||
        gid.x >= dstSize.x - p.dstOrigin.x ||
        gid.y >= dstSize.y - p.dstOrigin.y) return;
    uint2 srcCoord = p.srcOrigin + gid;
    uint2 dstCoord = p.dstOrigin + gid;
    // GL requires GL_NEAREST for integer/MSAA blits; choose sample 0 deterministically.
    dst.write(src.read(srcCoord, 0), dstCoord);
}

kernel void mgl_msaa_resolve_int(texture2d_ms<int, access::read> src [[texture(0)]], texture2d<int, access::write> dst [[texture(1)]], constant MGLMSAAIntegerResolveParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (p.size.x == 0u || p.size.y == 0u) return;
    if (gid.x >= p.size.x || gid.y >= p.size.y) return;
    uint2 srcSize = uint2(src.get_width(), src.get_height());
    uint2 dstSize = uint2(dst.get_width(), dst.get_height());
    if (p.srcOrigin.x >= srcSize.x || p.srcOrigin.y >= srcSize.y ||
        p.dstOrigin.x >= dstSize.x || p.dstOrigin.y >= dstSize.y) return;
    if (gid.x >= srcSize.x - p.srcOrigin.x ||
        gid.y >= srcSize.y - p.srcOrigin.y ||
        gid.x >= dstSize.x - p.dstOrigin.x ||
        gid.y >= dstSize.y - p.dstOrigin.y) return;
    uint2 srcCoord = p.srcOrigin + gid;
    uint2 dstCoord = p.dstOrigin + gid;
    // GL requires GL_NEAREST for integer/MSAA blits; choose sample 0 deterministically.
    dst.write(src.read(srcCoord, 0), dstCoord);
}
