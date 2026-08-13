#include <metal_stdlib>
using namespace metal;

struct MGLClearRectParams { float4 color; float depth; float3 _padding; };
struct MGLClearRectVOut { float4 position [[position]]; };

vertex MGLClearRectVOut mgl_clear_rect_vs(uint vid [[vertex_id]], constant MGLClearRectParams& p [[buffer(0)]]) {
    float2 pos[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0) };
    MGLClearRectVOut o;
    o.position = float4(pos[vid & 3u], p.depth, 1.0);
    return o;
}

fragment float4 mgl_clear_rect_fs(MGLClearRectVOut in [[stage_in]], constant MGLClearRectParams& p [[buffer(0)]]) {
    return p.color;
}
