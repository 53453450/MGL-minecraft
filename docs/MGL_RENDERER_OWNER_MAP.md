# MGL Renderer Owner Map

Authoritative Metal resource ownership after M3 single-path migration.
`MGLRendererBackendHandle` (`mgl_renderer_backend.cpp`) is the C++ owner
aggregate; ObjC ivars hold only platform-shell state and embedded C value
state (`MGLCommandState`, `MGLPipelineCacheState`).

## Backend handle (`MGLRendererBackendHandle`)

| Field | Role | Former ObjC ivar / shell |
|-------|------|--------------------------|
| `command_queue_owner` | MTL command queue lifetime | `_commandQueueOwner` (shell) |
| `command_buffer_owner` | Active/detached CB owner mirror | `_renderPassManager.state->currentCommandBufferOwner` |
| `render_encoder_owner` | Active render encoder | `_renderPassManager.state->currentRenderEncoderOwner` |
| `render_pass_state_owner` | Render-pass descriptor state | `_renderPassManager.state->renderPassStateOwner` |
| `binding_owner` | VS/FS buffer/texture/sampler dedup | `_bindingStateOwner` |
| `query_owner` | Occlusion/timer query state | `_queryStateOwner` |
| `recovery_owner` | GPU error recovery transactions | `_gpuRecovery.commandRecoveryOwner` |
| `tess_factor_buffer` … `tcs_output_buffer` | Tessellation scratch | `_tessellation.*` |
| `sampler_snapshots` | Sampler dedup cache | binding-state sampler snapshot |
| `fallback_*` | Null-texture/buffer fallbacks | `_resourceFallback` (partial) |

Runtime wiring: `mglRenderAttachRuntimeOwners` syncs the three transient
owners (command buffer, render encoder, render pass state) from
`MGLCommandState` on every context attach.

## Command state (`MGLCommandState`)

Embedded in `MGLRenderer` as `_commandState` (replaces
`MGLRenderPassManager`). Managed by `mgl_render_pass_coordinator.c`.

| Field | Role |
|-------|------|
| `renderPassIdentityOwner` | FBO/draw-buffer identity cache |
| `renderPassStateOwner` | MTLRenderPassDescriptor equivalent state |
| `currentCommandBufferOwner` | Per-frame command buffer |
| `currentRenderEncoderOwner` | Active render encoder |
| `mdiArgsScratchOwner` | MDI argument scratch buffer |
| `pendingEventOwner` | Shared-event sync wait queue |
| `lastFboMatch*` | FBO fast-path generation cache |

## Pipeline cache (`MGLPipelineCacheState` + owner)

Embedded as `_pipelineCacheState` / `_pipelineCacheOwner` (replaces
`MGLPipelineCache` ObjC object). Facade: `mgl_pipeline_cache_facade.c`.

| Tracked state | Role |
|---------------|------|
| `pipelineState` | Active MTLRenderPipelineState (borrowed id) |
| `pipeline*Format` | Attachment format fingerprint |
| `pipelineProgramName` | GL program name at PSO build |
| `psoDedupEnabled` / `dsCacheEnabled` | Env-gated caches |

## GL authoritative state

`GLMState` + `dirty_bits` remain in the C layer (`glm_context.h`).
Dirty-bit policy:

- **`mglMarkStateDirtyBits`**: sets domain dirty flags *and* invalidates
  hash caches used by batch merge / dirty-key delta.
- **`mglMarkRendererDirtyBits`**: sets `dirty_bits` only; use when hash
  caches must be preserved (e.g. renderer-internal replay bookkeeping).

## Sync entry (Phase 0+)

```
draw_command.c / backend
  → mglRendererProcessGLState (mgl_renderer_backend.cpp)
  → mglRenderProcessGLState (mgl_renderer_sync.cpp)
  → mglRendererObjCProcessGLState (MGLRenderer+RenderPass.m, migrates to C++)
```

## C++ renderer modules (Phase 2–6)

| Module | Header | ObjC bridge prefix | Routes |
|--------|--------|-------------------|--------|
| sync | `mgl_renderer_sync.h` | `mglRendererObjCProcessGLState` | dirty-domain orchestration, FBO pass sync (`mglRenderSyncRenderPassForFbo`) |
| binding | `mgl_renderer_binding.h` | `mglRendererObjCSyncResourceBindings` | resource rebind before draw |
| batch | `mgl_renderer_batch.h` | `mglRendererObjCFlushDrawBuffer` | deferred batch flush |
| draw | `mgl_renderer_draw.h` | `mglRendererObjCDraw*` | draw dispatch (partial) |
| texture | `mgl_renderer_texture.h` | `mglRendererObjC*Texture*` | bind/mipmap/readback/upload |
| platform | `mgl_renderer_platform.h` | `mglRendererObjCSwap/Clear*` | swap/clear |
| blit | `mgl_renderer_blit.h` | `mglRendererObjCBlitFramebuffer` | framebuffer blit |
| compute | `mgl_renderer_compute.h` | `mglRendererObjCDispatchCompute*` | compute dispatch |

All backend `mglRenderer*` public APIs route through `mglRender*` facades;
`mglRendererCompat*` dispatch has been removed (Phase 6).
