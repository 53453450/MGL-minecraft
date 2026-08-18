#!/bin/bash
# P5 hard gate for the single-path Metal-cpp renderer.
set -u
shopt -s nullglob

failures=0
fail() {
    printf 'check-p5-metalcpp FAIL: %s\n' "$1"
    failures=$((failures + 1))
}

gate_hits=$(rg -n 'MGL_USE_METALCPP' MGL/src MGL/include || true)
if [ -n "$gate_hits" ]; then
    fail "production sources still reference MGL_USE_METALCPP"
    printf '%s\n' "$gate_hits" | sed 's/^/    /'
fi

tool_gate_hits=$(rg -n 'MGL_USE_METALCPP|--ab' \
    benchmark/mgl_benchmark.c scripts/record_p3_baseline.sh || true)
if [ -n "$tool_gate_hits" ]; then
    fail "benchmark or baseline tooling still exposes the removed A/B renderer gate"
    printf '%s\n' "$tool_gate_hits" | sed 's/^/    /'
fi

if ! sed -n '/^test-all:/,/^[^[:space:]]/p' Makefile | \
        rg -q 'check-p5-metalcpp' ||
   sed -n '/^test-all:/,/^[^[:space:]]/p' Makefile | \
        rg -q 'check-p4-metalcpp'; then
    fail "test-all is not wired directly to the terminal P5 gate"
fi

smoke_adapter_hits=$(rg -n 'mgl_render_cpp_objc.h' \
    Makefile test_legacy_compat || true)
if [ -n "$smoke_adapter_hits" ]; then
    fail "standalone smoke still depends on the production ObjC transition adapter"
    printf '%s\n' "$smoke_adapter_hits" | sed 's/^/    /'
fi

adapter_path="MGL/src/mgl_render_cpp_""objc.h"
if [ -e "$adapter_path" ]; then
    fail "the deleted Objective-C transition adapter still exists"
    printf '    %s\n' "$adapter_path"
fi

legacy_bridge_hits=$(rg -n \
    'MGLMetal[A-Za-z]+Ref|mgl_metal_bridge|MGLRendererMetalBridge|MGL_USE_METALCPP' \
    MGL/src MGL/include || true)
if [ -n "$legacy_bridge_hits" ]; then
    fail "legacy Metal ref typedefs, bridge modules, or renderer gates remain"
    printf '%s\n' "$legacy_bridge_hits" | sed 's/^/    /'
fi

objc_metal_operation_hits=$(rg -n \
    '\[(self|_renderPassManager|_pipelineCache|_commandQueue|_device|_layer)[[:space:]]+(newCommandQueue|commandBuffer|renderCommandEncoderWithDescriptor|computeCommandEncoder|blitCommandEncoder|addCompletedHandler|presentDrawable):' \
    MGL/src --glob '*.m' --glob '*.mm' || true)
if [ -n "$objc_metal_operation_hits" ]; then
    fail "Objective-C renderer still invokes direct Metal command operations"
    printf '%s\n' "$objc_metal_operation_hits" | sed 's/^/    /'
fi

# CAMetalLayer/drawable access is a platform-shell concern.  Renderer
# categories may request an opaque drawable/texture through the shell facade,
# but must not send layer selectors or inspect drawable properties directly.
non_platform_platform_hits=$(rg -n \
    'id[[:space:]]*<[[:space:]]*CAMetalDrawable[[:space:]]*>|\[_layer[[:space:]]+nextDrawable\]|_layer[[:space:]]*\.[[:space:]]*(device|pixelFormat|drawableSize|frame|opaque|framebufferOnly|allowsNextDrawableTimeout|magnificationFilter|presentsWithTransaction)|(_drawable|drawable)[[:space:]]*\.[[:space:]]*texture|\[_commandQueue[[:space:]]+class\]' \
    MGL/src --glob '*.m' --glob '*.mm' --glob '!MGLPlatformRendererShell.m' || true)
if [ -n "$non_platform_platform_hits" ]; then
    fail "non-platform Objective-C code still owns or inspects Metal platform objects"
    printf '%s\n' "$non_platform_platform_hits" | sed 's/^/    /'
fi

metal_import_hits=$(rg -n '#import[[:space:]]+<Metal/Metal\.h>|#include[[:space:]]+<Metal/Metal\.h>' \
    MGL/src MGL/include --glob '*.{m,mm,h}' \
    --glob '!MGLPlatformRendererShell.m' --glob '!MGLPlatformRendererShell.h' || true)
if [ -n "$metal_import_hits" ]; then
    fail "non-platform Objective-C sources still import the Metal framework"
    printf '%s\n' "$metal_import_hits" | sed 's/^/    /'
fi

# The ObjC-facing C/value-state surfaces must not carry Metal object types or
# descriptor classes.  C++ backend headers under MGL/src are intentionally
# excluded because they are the owner implementation boundary.
objc_type_hits=""
for path in MGL/src/*.m MGL/src/*.mm MGL/include/*.h; do
    case "$path" in
        MGL/src/mgl_metal_cpp.h|MGL/src/mgl_render_cpp.h|MGL/src/mgl_air_loader.h|MGL/src/MGLPlatformRendererShell.m|MGL/include/MGLPlatformRendererShell.h)
            continue ;;
    esac
    hits=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' "$path" |
        rg -n 'id[[:space:]]*<[[:space:]]*MTL|\bMTL[A-Za-z]+Descriptor\b|\bMTL::' || true)
    if [ -n "$hits" ]; then
        objc_type_hits="${objc_type_hits}${path}:\n${hits}\n"
    fi
done
if [ -n "$objc_type_hits" ]; then
    fail "Objective-C-facing source or private headers still expose Metal object/descriptor types"
    printf '%s\n' "$objc_type_hits" | sed 's/^/    /'
fi

# Metal enum values are part of the C value-state ABI, but their framework
# identifiers must not leak back into non-platform Objective-C.  The neutral
# MGL* constants live in mgl_render_values.h / pixel_utils.h and are checked
# against metal-cpp in the implementation TU.
objc_value_enum_hits=""
value_enum_pattern='\bMTL(TextureType|TextureUsage|StorageMode|ResourceStorageMode|LoadAction|StoreAction|CompareFunction|CommandBufferStatus|PrimitiveType|CullMode|Winding|DepthClipMode|ColorWriteMask|PrimitiveTopologyClass|TessellationPartitionMode|TessellationFactorStepFunction|TessellationFactorFormat|TessellationControlPointIndexType|MultisampleDepthResolveFilter|MultisampleStencilResolveFilter|BlendFactor|BlendOperation|VertexFormat|PixelFormat)'
for path in MGL/src/*.m MGL/src/*.mm MGL/include/*.h; do
    case "$path" in
        MGL/src/MGLPlatformRendererShell.m|MGL/include/MGLPlatformRendererShell.h)
            continue ;;
    esac
    hits=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g; s/"(?:\\.|[^"\\])*"//g' "$path" |
        rg -n "$value_enum_pattern" || true)
    if [ -n "$hits" ]; then
        objc_value_enum_hits="${objc_value_enum_hits}${path}:\n${hits}\n"
    fi
done
if [ -n "$objc_value_enum_hits" ]; then
    fail "non-platform Objective-C still names Metal enum values instead of MGL value-state"
    printf '%b' "$objc_value_enum_hits" | sed 's/^/    /'
fi

platform_shell_hits=$(rg -n 'CAMetalLayer|NSView|nextDrawable' \
    MGL/src/MGLPlatformRendererShell.m MGL/include/MGLPlatformRendererShell.h || true)
if [ -z "$platform_shell_hits" ]; then
    fail "MGLPlatformRendererShell is missing the platform Metal lifetime boundary"
fi

fallback_getter_hits=$(rg -n \
    'mglRenderCppRenderEncoderOwnerGetCurrentForFallback' \
    MGL/src MGL/include test_legacy_compat || true)
if [ -n "$fallback_getter_hits" ]; then
    fail "borrowed render-encoder fallback getter still exists"
    printf '%s\n' "$fallback_getter_hits" | sed 's/^/    /'
fi

# Shared binding/cull-distance buffers are backend-owned Metal objects.  Keep
# these checks narrow so ordinary GL semantic fallback helpers remain valid.
backend_fallback_owner_hits=$(rg -n \
    'mglRendererBackendGetFallbackBindingBuffer|mglRendererBackendGetCullDistanceDummyBuffer' \
    MGL/src/mgl_renderer_backend.h MGL/src/mgl_renderer_backend.cpp \
    MGL/src/MGLRenderer+BindingState.m MGL/src/MGLRenderer+DrawSupport.m || true)
if [ -z "$backend_fallback_owner_hits" ] || \
   ! rg -q 'fallback_binding_buffer' MGL/src/mgl_renderer_backend.cpp || \
   ! rg -q 'fallback_binding_buffer_length' MGL/src/mgl_renderer_backend.cpp || \
   ! rg -q 'cull_distance_dummy_buffer' MGL/src/mgl_renderer_backend.cpp; then
    fail "fallback binding and cull-distance dummy buffers are not backend-owned"
fi

legacy_static_fallback_hits=$(rg -n \
    'static[[:space:]]+(id|MGLMetal[A-Za-z]+Ref)[[:space:]]+(fallbackBindingBuffer|cullMtlBuffer|cullDistanceDummyBuffer)' \
    MGL/src/MGLRenderer+BindingState.m MGL/src/MGLRenderer+DrawSupport.m || true)
if [ -n "$legacy_static_fallback_hits" ]; then
    fail "renderer Objective-C layers still statically own migrated fallback buffers"
    printf '%s\n' "$legacy_static_fallback_hits" | sed 's/^/    /'
fi

queue_fallback_hits=$(rg -n '\[_device newCommandQueue' \
    MGL/src/MGLRenderer+Lifecycle.m \
    MGL/src/MGLRenderer+GPURecovery.m || true)
if [ -n "$queue_fallback_hits" ]; then
    fail "Objective-C command-queue fallback still exists"
    printf '%s\n' "$queue_fallback_hits" | sed 's/^/    /'
fi

macro_files=$(rg -l \
    '^#define (NS_PRIVATE_IMPLEMENTATION|CA_PRIVATE_IMPLEMENTATION|MTL_PRIVATE_IMPLEMENTATION)' \
    MGL --glob '*.{c,cc,cpp,cxx,h,m,mm}' | sort || true)
if [ "$macro_files" != "MGL/src/mgl_render_cpp.cpp" ]; then
    fail "Metal-cpp implementation macros are not owned only by mgl_render_cpp.cpp"
    printf '%s\n' "$macro_files" | sed 's/^/    /'
fi

abi_metal_hits=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' \
    MGL/src/mgl_render_cpp.h MGL/src/mgl_renderer_backend.h | \
    rg -n 'id[[:space:]]*<MTL|MTL::' || true)
if [ -n "$abi_metal_hits" ]; then
    fail "public C ABI exposes Objective-C or Metal-cpp types"
    printf '%s\n' "$abi_metal_hits" | sed 's/^/    /'
fi

render_pass_header_hits=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' \
    MGL/include/MGLRenderPassManager.h | \
    rg -n 'id[[:space:]]*<MTL|MTL[A-Za-z]+Descriptor|MTL::' || true)
if [ -n "$render_pass_header_hits" ]; then
    fail "render-pass manager private interface still exposes Metal types"
    printf '%s\n' "$render_pass_header_hits" | sed 's/^/    /'
fi

# Draw/index and manager adapters are GL-semantic Objective-C shells. Their
# implementation boundaries must carry only opaque handles and value enums;
# Metal-cpp descriptor/selector use belongs in the C++ owner or platform shell.
objc_shell_metal_hits=""
for path in \
    MGL/src/mgl_draw_encode.m MGL/src/MGLRenderPassManager.m \
    MGL/src/MGLPipelineCache.m MGL/src/MGLRenderer+GPURecovery.m \
    MGL/src/MGLRenderer+Lifecycle.m MGL/src/MGLRenderer+SwapDiagnostics.m \
    MGL/src/MGLRenderer+Buffer.m MGL/src/MGLRenderer+Binding.m \
    MGL/src/MGLRenderer+VertexLayout.m MGL/src/MGLRenderer+Compute.m \
    MGL/src/MGLRenderer+BatchReplay.m MGL/src/MGLRenderer+Batch.m \
    MGL/src/MGLRenderer+Draw.m MGL/src/MGLRenderer+DrawSupport.m \
    MGL/src/MGLRenderer+Tessellation.m MGL/src/MGLRenderer+BindingState.m \
    MGL/src/MGLRenderer+RenderPass.m MGL/src/MGLRenderer+Blit.m \
    MGL/src/MGLRenderer.m MGL/src/MGLRenderer+Texture.m \
    MGL/include/MGLRenderer+GPURecovery_Private.h \
    MGL/include/MGLRenderer+SwapDiagnostics_Private.h \
    MGL/include/MGLRenderer+Buffer_Private.h \
    MGL/include/MGLRenderer+Draw_Private.h \
    MGL/include/MGLRenderer+Texture_Private.h \
    MGL/include/MGLRenderer+RenderPass_Private.h \
    MGL/include/MGLRenderer+Blit_Private.h; do
    hits=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' "$path" |
        rg -n 'id[[:space:]]*<MTL|\bMTL[A-Za-z]+Descriptor\b|\bMTL::|\bMTL(CommandBufferStatus|PixelFormat|Viewport|ScissorRect|Origin|Size)\b|\[[^]]+\.(registryID|label|name|pixelFormat|width|height|contents)\b' || true)
    if [ -n "$hits" ]; then
        objc_shell_metal_hits="${objc_shell_metal_hits}${path}:\n${hits}\n"
    fi
done
if [ -n "$objc_shell_metal_hits" ]; then
    fail "GL-semantic Objective-C adapters still contain Metal types/selectors"
    printf '%b' "$objc_shell_metal_hits" | sed 's/^/    /'
fi

if [ -e MGL/src/mgl_index_buffer.m ] || ! [ -f MGL/src/mgl_index_buffer.cpp ]; then
    fail "index-buffer ownership is not isolated in the C++ implementation"
fi
if ! rg -q 'MGL_DRAW_INDEX_UINT16 = 0' MGL/include/mgl_draw_encode.h ||
   ! rg -q 'MGL_DRAW_INDEX_UINT32 = 1' MGL/include/mgl_draw_encode.h; then
    fail "opaque draw ABI does not preserve the Metal index-type numeric contract"
fi

pipeline_cache_header_hits=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' \
    MGL/include/MGLPipelineCache.h | \
    rg -n 'id[[:space:]]*<MTL|MTL[A-Za-z]+Descriptor|MTL::|MTL(PixelFormat|Blend|ColorWrite)' || true)
if [ -n "$pipeline_cache_header_hits" ]; then
    fail "pipeline-cache private interface still exposes Metal types"
    printf '%s\n' "$pipeline_cache_header_hits" | sed 's/^/    /'
fi

pipeline_cache_legacy_descriptor_api_hits=$(rg -n \
    'depthStencilStateForDescriptor|createRenderPipelineStateWithDescriptor' \
    MGL/include/MGLPipelineCache.h MGL/src/MGLPipelineCache.m || true)
if [ -n "$pipeline_cache_legacy_descriptor_api_hits" ]; then
    fail "pipeline-cache Objective-C descriptor conversion API still exists"
    printf '%s\n' "$pipeline_cache_legacy_descriptor_api_hits" | sed 's/^/    /'
fi

pipeline_cache_objc_ownership_hits=$(rg -n \
    'id[[:space:]]+(__strong[[:space:]]+)?_Nullable[[:space:]]*(pipelineState|pipelineVertexFunction|pipelineFragmentFunction)|id[[:space:]]+_device|strong[^\n]*id[[:space:]]+device' \
    MGL/include/MGLPipelineCache.h MGL/src/MGLPipelineCache.m || true)
if [ -n "$pipeline_cache_objc_ownership_hits" ]; then
    fail "pipeline cache ObjC layer still owns Metal object mirrors"
    printf '%s\n' "$pipeline_cache_objc_ownership_hits" | sed 's/^/    /'
fi

legacy_platform_import_hits=$(rg -n \
    '#import <MetalKit/MetalKit\.h>|#include <Metal/Metal\.h>' \
    MGL/src/MGLRenderer.m MGL/src/hash_table.m MGL/include/mgl_rt_sync.h || true)
if [ -n "$legacy_platform_import_hits" ]; then
    fail "non-owner modules retain unused platform Metal imports"
    printf '%s\n' "$legacy_platform_import_hits" | sed 's/^/    /'
fi

# Completed value-state islands must stay Metal-free. Keep this audit local to
# modules whose Metal operations have already moved into mgl_render_cpp.cpp.
value_state_metal_hits=""
for path in \
    MGL/src/mgl_capability.m MGL/include/mgl_capability.h \
    MGL/src/mgl_sync.m MGL/include/mgl_sync.h \
    MGL/src/mgl_vertex_format.m MGL/include/mgl_vertex_format.h \
    MGL/src/mgl_texture_compat.m MGL/include/mgl_texture_compat.h; do
    hits=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' "$path" |
        rg -n 'id[[:space:]]*<MTL|MTL::|\bMTL[A-Za-z]+Descriptor\b|\bMTL(Texture|Vertex|Winding|Swizzle|PixelFormat)[[:space:]*]+[A-Za-z_]' || true)
    if [ -n "$hits" ]; then
        value_state_metal_hits="${value_state_metal_hits}${hits}\n"
    fi
done
if [ -n "$value_state_metal_hits" ]; then
    fail "completed value-state islands still expose Metal types"
    printf '%b' "$value_state_metal_hits" | sed 's/^/    /'
fi

if ! rg -q 'mglRenderCppSampledTextureViewForBaseLevel' \
        MGL/src/mgl_render_cpp.h MGL/src/mgl_render_cpp.cpp ||
   ! rg -q 'mglSampledTextureViewForBaseLevel' \
        MGL/src/mgl_texture_compat.m MGL/include/mgl_texture_compat.h; then
    fail "sampled texture view ownership is missing from the backend facade"
fi

if ! rg -q 'mglRendererBackendCreateProactiveTexture' \
        MGL/src/mgl_renderer_backend.h MGL/src/mgl_renderer_backend.cpp \
        MGL/src/MGLRenderer+Lifecycle.m; then
    fail "proactive texture creation is not owned by the C++ backend"
fi

pipeline_owner_hits=$(rg -n \
    'mglRenderCpp(Create|Destroy|Lookup|Store|Activate|Invalidate|SetPipeline|GetPipeline|Reset|Shutdown).*Pipeline' \
    MGL/src/mgl_render_cpp.h MGL/src/mgl_render_cpp.cpp MGL/src/MGLPipelineCache.m || true)
if [ -z "$pipeline_owner_hits" ]; then
    fail "pipeline cache has no visible C++ owner facade lifecycle"
fi

if ! rg -q 'void \*renderer_backend;' MGL/include/glm_context.h ||
   ! rg -q 'void \*platform_renderer_shell;' MGL/include/glm_context.h; then
    fail "GLMContext is missing the P5 backend/platform roots"
fi

core_platform_hits=$(rg -n \
    'NSView[[:space:]]*\*__strong view|CAMetalLayer[[:space:]]*\*__strong layer' \
    MGL/include/MGLRenderer_State.h || true)
if [ -n "$core_platform_hits" ]; then
    fail "renderer core state still directly owns platform view/layer"
    printf '%s\n' "$core_platform_hits" | sed 's/^/    /'
fi

if [ "$failures" -ne 0 ]; then
    printf 'check-p5-metalcpp: %d violation(s)\n' "$failures"
    exit 1
fi

printf 'P5_SINGLE_PATH_GATE_OK\n'
