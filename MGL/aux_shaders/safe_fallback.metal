#include <metal_stdlib>
using namespace metal;

vertex float4 mgl_safe_fallback_vs(uint vid [[vertex_id]]) {
    return float4(0.0, 0.0, 0.0, 1.0);
}

fragment float4 mgl_safe_fallback_fs() {
    return float4(0.0, 0.0, 0.0, 1.0);
}

/* Integer color targets reject a float4 fragment output. */
fragment int4 mgl_safe_fallback_fs_int() {
    return int4(0);
}

fragment uint4 mgl_safe_fallback_fs_uint() {
    return uint4(0u);
}
