#include <metal_stdlib>
using namespace metal;

struct MGLScaledBlitParams { float4 uvRect; float forceOpaqueAlpha; float3 _padding; };
struct MGLScaledBlitVOut { float4 position [[position]]; float2 uv; };
struct MGLScaledDepthBlitFOut { float depth [[depth(any)]]; };

vertex MGLScaledBlitVOut mgl_scaled_depth_blit_vs(uint vid [[vertex_id]], constant MGLScaledBlitParams& p [[buffer(0)]]) {
    float2 pos[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0) };
    float2 uv[4] = { float2(p.uvRect.x, p.uvRect.w), float2(p.uvRect.z, p.uvRect.w), float2(p.uvRect.x, p.uvRect.y), float2(p.uvRect.z, p.uvRect.y) };
    MGLScaledBlitVOut o;
    o.position = float4(pos[vid], 0.0, 1.0);
    o.uv = uv[vid];
    return o;
}

fragment MGLScaledDepthBlitFOut mgl_scaled_depth_blit_fs(MGLScaledBlitVOut in [[stage_in]], depth2d<float> srcDepth [[texture(0)]], sampler s [[sampler(0)]]) {
    MGLScaledDepthBlitFOut out;
    out.depth = srcDepth.sample(s, in.uv);
    return out;
}
