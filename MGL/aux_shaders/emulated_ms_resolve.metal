#include <metal_stdlib>
using namespace metal;

/* GL 4.6 §18.3.1: MS→single-sample blit averages floating-point samples.
 * Emulated MS stores planes as texture2d_array slices (sample_count==1). */
struct MGLEmulatedMSResolveParams {
    uint sampleCount;
    uint baseSlice;
    uint2 size;
};

kernel void mgl_emulated_ms_resolve_float(
    texture2d_array<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant MGLEmulatedMSResolveParams &p [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= p.size.x || gid.y >= p.size.y)
        return;
    uint n = p.sampleCount > 0u ? p.sampleCount : 1u;
    float4 sum = float4(0.0);
    for (uint s = 0u; s < n; s++)
        sum += src.read(gid, p.baseSlice + s);
    dst.write(sum / float(n), gid);
}
