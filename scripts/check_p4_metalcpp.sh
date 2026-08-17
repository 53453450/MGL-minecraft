#!/bin/bash
# P4/P4.5 hard gate for the Metal-cpp renderer migration.
set -u

failures=0
fail() {
    printf 'check-p4-metalcpp FAIL: %s\n' "$1"
    failures=$((failures + 1))
}

callback_count=$(awk '
    /^#define MGL_MTL_FUNC_LIST\(M\)/ { in_list=1; next }
    in_list && /^#define MGL_MTL_FUNC_STRUCT/ { in_list=0 }
    in_list && /^[[:space:]]*M\(/ { count++ }
    END { print count + 0 }
' MGL/include/mgl_types_metal_funcs.h)
if [ "$callback_count" -ne 53 ]; then
    fail "GLMMetalFuncs callback count is $callback_count, expected 53"
fi

bridge_selector_hits=$(rg -n \
    'mglBridgeTarget|selector-forward|\[[A-Za-z_][A-Za-z0-9_]*[[:space:]]+[A-Za-z_]' \
    MGL/src/mgl_metal_bridge.m || true)
if [ -n "$bridge_selector_hits" ]; then
    fail "mgl_metal_bridge.m contains renderer selector forwarding"
    printf '%s\n' "$bridge_selector_hits" | sed 's/^/    /'
fi

callback_operation_files=$(rg -l \
    '^((void|int)[[:space:]]+)?mglRendererCallback(DispatchCompute|DispatchComputeIndirect|Draw|BindTexture|FlushDrawBuffer|SwapBuffers|ClearBuffer|BlitFramebuffer|Resource)\(' \
    MGL/src --glob '*.m' | sort || true)
expected_callback_operation_files=$(printf '%s\n' \
    MGL/src/MGLRenderer+Batch.m \
    MGL/src/MGLRenderer+Binding.m \
    MGL/src/MGLRenderer+Blit.m \
    MGL/src/MGLRenderer+Compute.m \
    MGL/src/MGLRenderer+Draw.m \
    MGL/src/MGLRenderer+Texture.m \
    MGL/src/MGLRenderer.m | sort)
if [ "$callback_operation_files" != "$expected_callback_operation_files" ]; then
    fail "callback runtime operation definitions escaped the approved GL-semantic adapters"
    printf '%s\n' "$callback_operation_files" | sed 's/^/    /'
fi

callback_operation_bodies=$(awk '
    /mglRendererCallback(DispatchCompute|DispatchComputeIndirect|Draw|BindTexture|FlushDrawBuffer|SwapBuffers|ClearBuffer|BlitFramebuffer|Resource)\(/ {
        capture = 1
        seen_open = 0
    }
    capture {
        print
        line = $0
        opens = gsub(/\{/, "{", line)
        closes = gsub(/\}/, "}", line)
        if (opens > 0) seen_open = 1
        depth += opens - closes
        if (seen_open && depth == 0) {
            capture = 0
            seen_open = 0
        }
    }
' $callback_operation_files)

callback_direct_metal_hits=$(printf '%s\n' "$callback_operation_bodies" | rg -n \
    '\[[^]]+[[:space:]]+(set(Vertex|Fragment|Object|Tile|Threadgroup|Buffer|Bytes|Texture|Sampler|Viewport|Scissor|Cull|Depth|Stencil|Blend|Triangle|FrontFacing|DepthBias|Visibility|RenderPipeline|ComputePipeline)|drawPrimitives|drawIndexedPrimitives|dispatch(Threadgroups|Threads)|endEncoding|commit|waitUntilCompleted|renderCommandEncoder|computeCommandEncoder|blitCommandEncoder)' \
    || true)
if [ -n "$callback_direct_metal_hits" ]; then
    fail "callback GL-semantic adapter directly invokes a Metal encoder/command-buffer selector"
    printf '%s\n' "$callback_direct_metal_hits" | sed 's/^/    /'
fi

callback_adapter_selectors=$(printf '%s\n' "$callback_operation_bodies" | sed -n \
    's/.*\[renderer \([A-Za-z0-9_]*\).*/\1/p' | sort)
expected_callback_adapter_selectors=$(printf '%s\n' \
    bindMTLTexture \
    flushDrawBuffer \
    mtlBlitFramebuffer \
    mtlClearBuffer \
    mtlCopyImageSubData \
    mtlCopyTexSubImage \
    mtlDispatchComputeIndirectLocked \
    mtlDispatchComputeLocked \
    mtlDrawArrays \
    mtlDrawArraysIndirect \
    mtlDrawArraysInstanced \
    mtlDrawArraysInstancedBaseInstance \
    mtlDrawElements \
    mtlDrawElementsBaseVertex \
    mtlDrawElementsIndirect \
    mtlDrawElementsInstanced \
    mtlDrawElementsInstancedBaseInstance \
    mtlDrawElementsInstancedBaseVertex \
    mtlDrawElementsInstancedBaseVertexBaseInstance \
    mtlDrawRangeElements \
    mtlDrawRangeElementsBaseVertex \
    mtlGenerateMipmaps \
    mtlGetTexImage \
    mtlMultiDrawArrays \
    mtlMultiDrawArraysIndirect \
    mtlMultiDrawElements \
    mtlMultiDrawElementsBaseVertex \
    mtlMultiDrawElementsIndirect \
    mtlReadDepthPixels \
    mtlReadDrawable \
    mtlReadIntegerPixels \
    mtlSwapBuffers \
    mtlTexSubImage \
    mtlTexSubImageBytes | sort)
if [ "$callback_adapter_selectors" != "$expected_callback_adapter_selectors" ]; then
    fail "callback pure-adapter selector set differs from the reviewed 34 GL-semantic entries"
    printf '%s\n' "$callback_adapter_selectors" | sed 's/^/    /'
fi

legacy_invoke_objc_files=$(rg -l 'mglRenderCppInvokeLegacyCallback\(' \
    MGL/src --glob '*.m' | sort || true)
if [ "$legacy_invoke_objc_files" != "MGL/src/mgl_metal_bridge.m" ]; then
    fail "legacy callback invoke is reachable outside the gate-off bridge"
    printf '%s\n' "$legacy_invoke_objc_files" | sed 's/^/    /'
fi

abi_metal_hits=$(awk '
    BEGIN { in_block = 0 }
    {
        source = $0
        code = ""
        while (length(source) > 0) {
            if (in_block) {
                close_at = index(source, "*/")
                if (!close_at) {
                    source = ""
                    break
                }
                source = substr(source, close_at + 2)
                in_block = 0
            }
            block_at = index(source, "/*")
            line_at = index(source, "//")
            if (line_at && (!block_at || line_at < block_at)) {
                code = code substr(source, 1, line_at - 1)
                source = ""
            } else if (block_at) {
                code = code substr(source, 1, block_at - 1)
                source = substr(source, block_at + 2)
                in_block = 1
            } else {
                code = code source
                source = ""
            }
        }
        if (code ~ /id[[:space:]]*<MTL|MTL::/) {
            print NR ":" code
        }
    }
' MGL/src/mgl_render_cpp.h)
if [ -n "$abi_metal_hits" ]; then
    fail "public C ABI exposes Objective-C or Metal-cpp types"
    printf '%s\n' "$abi_metal_hits" | sed 's/^/    /'
fi

macro_files=$(rg -l \
    '^#define (NS_PRIVATE_IMPLEMENTATION|CA_PRIVATE_IMPLEMENTATION|MTL_PRIVATE_IMPLEMENTATION)' \
    MGL --glob '*.{c,cc,cpp,cxx,h,m,mm}' | sort || true)
if [ "$macro_files" != "MGL/src/mgl_render_cpp.cpp" ]; then
    fail "Metal-cpp implementation macros are not owned only by mgl_render_cpp.cpp"
    printf '%s\n' "$macro_files" | sed 's/^/    /'
fi

command_getter_hits=$(rg -n 'mglRenderCppCommandBufferOwnerGetCurrent' \
    MGL/src --glob '*.m' || true)
if [ -n "$command_getter_hits" ]; then
    fail "Objective-C code still borrows CommandBufferOwner.current"
    printf '%s\n' "$command_getter_hits" | sed 's/^/    /'
fi

old_encoder_getter_hits=$(rg -n \
    'mglRenderCppRenderEncoderOwnerGetCurrent\(' MGL/src MGL/include || true)
if [ -n "$old_encoder_getter_hits" ]; then
    fail "unrestricted render-encoder getter still exists"
    printf '%s\n' "$old_encoder_getter_hits" | sed 's/^/    /'
fi

fallback_getter_body=$(awk '
    /mglRenderCppRenderEncoderOwnerGetCurrentForFallback\(/ { capture=1 }
    capture { print }
    capture && /^}/ { exit }
' MGL/src/mgl_render_cpp.cpp)
if ! printf '%s\n' "$fallback_getter_body" | rg -q \
    'mgl_env_flag_enabled_default_on\("MGL_USE_METALCPP"\)' ||
   ! printf '%s\n' "$fallback_getter_body" | rg -q 'return nullptr;'; then
    fail "fallback render-encoder getter is not fail-closed under gate-on"
fi

fallback_getter_sites=$(rg -n \
    'mglRenderCppRenderEncoderOwnerGetCurrentForFallback\(' \
    MGL/src --glob '*.m' | wc -l | tr -d ' ')
if [ "$fallback_getter_sites" -ne 9 ]; then
    fail "fallback render-encoder getter site count is $fallback_getter_sites, expected 9"
fi

unexpected_fallback_files=$(rg -l \
    'mglRenderCppRenderEncoderOwnerGetCurrentForFallback\(' \
    MGL/src --glob '*.m' | sort | comm -23 - <(printf '%s\n' \
        MGL/src/MGLRenderer+Batch.m \
        MGL/src/MGLRenderer+BatchReplay.m \
        MGL/src/MGLRenderer+Draw.m \
        MGL/src/MGLRenderer+DrawSupport.m \
        MGL/src/MGLRenderer+QuerySync.m \
        MGL/src/MGLRenderer+RenderPass.m \
        MGL/src/MGLRenderer+Tessellation.m | sort))
if [ -n "$unexpected_fallback_files" ]; then
    fail "fallback render-encoder getter escaped the approved adapter files"
    printf '%s\n' "$unexpected_fallback_files" | sed 's/^/    /'
fi

queue_fallback_hits=$(rg -n -U \
    'if[[:space:]]*\([^)]*!_commandQueue[^)]*\)[[:space:]]*\{[^}]*newCommandQueue' \
    MGL/src/MGLRenderer+Lifecycle.m \
    MGL/src/MGLRenderer+GPURecovery.m || true)
if [ -n "$queue_fallback_hits" ]; then
    fail "gate-on command-queue owner failure can fall through to ObjC queue creation"
    printf '%s\n' "$queue_fallback_hits" | sed 's/^/    /'
fi

queue_gate_files=$(rg -l -U \
    'if[[:space:]]*\(useMetalCppQueue\)[[:space:]]*\{[^}]*mglRenderCppCreateOrResetCommandQueueOwner\([^}]*\}[[:space:]]*else[[:space:]]*\{[^}]*newCommandQueue' \
    MGL/src/MGLRenderer+Lifecycle.m \
    MGL/src/MGLRenderer+GPURecovery.m | sort || true)
expected_queue_gate_files=$(printf '%s\n' \
    MGL/src/MGLRenderer+GPURecovery.m \
    MGL/src/MGLRenderer+Lifecycle.m | sort)
if [ "$queue_gate_files" != "$expected_queue_gate_files" ]; then
    fail "command-queue ownership is not an explicit gate-on owner / gate-off ObjC branch"
    printf '%s\n' "$queue_gate_files" | sed 's/^/    /'
fi

direct_queue_selectors=$(rg -n '\[_device newCommandQueue' \
    MGL/src/MGLRenderer+Lifecycle.m \
    MGL/src/MGLRenderer+GPURecovery.m || true)
direct_queue_selector_count=$(printf '%s\n' "$direct_queue_selectors" | \
    sed '/^$/d' | wc -l | tr -d ' ')
if [ "$direct_queue_selector_count" -ne 2 ]; then
    fail "direct ObjC command-queue selector count is $direct_queue_selector_count, expected 2 gate-off sites"
    printf '%s\n' "$direct_queue_selectors" | sed 's/^/    /'
fi

gpu_commit_body=$(sed -n \
    '/^- (void)commitCommandBufferWithAGXRecovery:/,/^- (BOOL)shouldSkipGPUOperations/p' \
    MGL/src/MGLRenderer+GPURecovery.m)
gpu_commit_policy_hits=$(printf '%s\n' "$gpu_commit_body" | rg -n \
    'recordGPUError|mglRenderCppClassifyCommandBufferCommit|mglRenderCommandBufferState' || true)
if [ -n "$gpu_commit_policy_hits" ]; then
    fail "GPURecovery commit wrapper still owns command classification or recovery counting"
    printf '%s\n' "$gpu_commit_policy_hits" | sed 's/^/    /'
fi
if ! printf '%s\n' "$gpu_commit_body" | rg -q \
    'mglRenderCppCommandRecoveryRecordTransactionFailure'; then
    fail "GPURecovery exception boundary does not report failure through the C++ recovery owner"
fi

direct_draw_hits=$(rg -n 'drawIndexedPrimitives|drawPrimitives' \
    MGL/src --glob '*.m' || true)
if [ -n "$direct_draw_hits" ]; then
    fail "Objective-C source still contains direct render draw selectors"
    printf '%s\n' "$direct_draw_hits" | sed 's/^/    /'
fi

copyback_transaction_hits=$(rg -n 'mglRenderCppFlushStageBindingCopyBacks' \
    MGL/src MGL/include || true)
if [ -n "$copyback_transaction_hits" ]; then
    fail "removed copy-back lifecycle transaction reappeared"
    printf '%s\n' "$copyback_transaction_hits" | sed 's/^/    /'
fi

borrowed_blit_owner_hits=$(rg -n \
    'mglRenderCppCreateBlitEncoderFromCommandBufferOwner\(' \
    MGL/src --glob '*.m' || true)
if [ -n "$borrowed_blit_owner_hits" ]; then
    fail "Objective-C gate path still borrows an owner blit encoder"
    printf '%s\n' "$borrowed_blit_owner_hits" | sed 's/^/    /'
fi

metal_objc_files=$(rg -l 'id<MTL' MGL/src --glob '*.m' | sort || true)
expected_metal_objc_files=$(printf '%s\n' \
    MGL/src/MGLRenderer+Lifecycle.m \
    MGL/src/MGLRenderer.m | sort)
if [ "$metal_objc_files" != "$expected_metal_objc_files" ]; then
    fail "strict id<MTL whitelist differs from the two platform shells"
    printf '%s\n' "$metal_objc_files" | sed 's/^/    /'
fi

if [ "$failures" -ne 0 ]; then
    printf 'check-p4-metalcpp: %d violation(s)\n' "$failures"
    exit 1
fi

printf 'P4_STATIC_CENSUS_OK callbacks=%s metal_objc_files=2 fallback_getter_sites=%s\n' \
    "$callback_count" "$fallback_getter_sites"
